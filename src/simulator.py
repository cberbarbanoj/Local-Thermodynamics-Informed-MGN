"""simulator.py"""

import torch
import torch.nn.functional as F
import lightning.pytorch as pl

from src.model import EncoderProcessorDecoder
from src.utils.utils import NodeType
from src.utils.normalization import Normalizer

class CFDSolver(pl.LightningModule):

    def __init__(self, dims, dInfo, save_folder):
        super().__init__()

        self.dim_hidden      = dInfo['model']['dim_hidden']
        self.layers          = dInfo['model']['n_layers']
        self.passes          = dInfo['model']['passes']
        self.dropout         = dInfo['model']['dropout']
        self.batch_size      = dInfo['model']['batch_size']
        self.state_variables = dInfo['dataset']['state_variables']
        self.data_dim        = 2 if dInfo['dataset']['dataset_dim'] == '2D' else 3
        self.dims            = dims
        self.dim_z           = self.dims['z']
        self.dim_q           = self.dims['q']
        self.dim_vel         = 2
        self.dim_press       = 1
        self.output_size     = self.dim_z
        self.node_input_size = self.dim_z + self.dims['n'] - self.dims['q']
        self.edge_input_size = self.dims['q'] + self.dims['q_0'] + 1
        self.shared_mlp      = False
        self.save_folder     = save_folder

        self.model = EncoderProcessorDecoder(message_passing_num=self.passes, hidden_size=self.dim_hidden,
                                             node_input_size=self.node_input_size, edge_input_size=self.edge_input_size,
                                             shared_mlp=self.shared_mlp, output_size=self.output_size, layers=self.layers,
                                             dropout=self.dropout)

        self._output_vel_normalizer   = Normalizer(size=self.dim_vel, name='output_vel_normalizer', device=self.device)
        self._output_press_normalizer = Normalizer(size=self.dim_press, name='output_press_normalizer', device=self.device)
        self._node_vel_normalizer     = Normalizer(size=self.dim_vel, name='node_vel_normalizer', device=self.device)
        self._node_press_normalizer   = Normalizer(size=self.dim_press, name='node_press_normalizer', device=self.device)
        self._edge_normalizer         = Normalizer(size=self.edge_input_size, name='edge_normalizer', device=self.device)

        self.dt                      = dInfo['dataset']['dt']
        self.dataset_type            = dInfo['dataset']['type']
        self.lambda_deg_start        = dInfo['model']['lambda_deg_start']
        self.lambda_deg_end          = dInfo['model']['lambda_deg_end']
        self.lambda_deg_warmup_start = dInfo['model']['lambda_deg_warmup_start']
        self.lambda_deg_warmup_end   = dInfo['model']['lambda_deg_warmup_end']
        self.noise_var               = dInfo['model']['noise_var']
        self.initial_lr              = dInfo['model']['initial_lr']
        self.final_lr                = dInfo['model']['final_lr']
        self.warmup_steps            = dInfo['model']['warmup_steps']
        self.decay_steps             = dInfo['model']['decay_steps']
        
        # Start accumulating statistics
        self._accumulating = True

        # Rollout simulation
        self.rollout_freq = dInfo['model']['rollout_freq']

    def forward(self, graph, passes_flag=False):

        # Add noise to the training data
        if self.training and (self.noise_var > 0):
            graph = self.__add_noise(graph)

        # Get data from graph
        node_type = graph.n
        current   = graph.x
        target    = graph.y

        # Prepare target data
        target_vel_gradient, target_press_gradient = self.__instant_to_gradient(target, current)
        # Normalize velocity and pressure targets
        target_vel_gradient_norm   = self._output_vel_normalizer(target_vel_gradient, accumulate=self._accumulating)
        target_press_gradient_norm = self._output_press_normalizer(target_press_gradient, accumulate=self._accumulating)
        # Concatenate to get normalized targets
        target_gradient_norm = torch.cat((target_vel_gradient_norm, target_press_gradient_norm), dim=-1)

        # Process the node information
        graph.x = self.__process_node_attr(node_type, graph.x)
        # Process the edge attributes
        graph.edge_attr = self.__process_edge_attr(graph.edge_attr)

        # Forward pass through the MGN
        if passes_flag:
            predicted_gradient_norm, energy_deg, entropy_deg, x_passes = self.model(graph, return_passes=True)
        else:
            predicted_gradient_norm, energy_deg, entropy_deg = self.model(graph)
            x_passes = None

        # Split prediction into velocity and pressure gradients
        predicted_vel_gradient_norm   = predicted_gradient_norm[:, :2]
        predicted_press_gradient_norm = predicted_gradient_norm[:, 2:]

        # Denormalize predicted data
        predicted_vel_gradient   = self._output_vel_normalizer.inverse(predicted_vel_gradient_norm)
        predicted_press_gradient = self._output_press_normalizer.inverse(predicted_press_gradient_norm)

        # Get next step data
        predicted_velocity = current[:, :2] + self.dt * predicted_vel_gradient
        predicted_pressure = current[:, 2:] + self.dt * predicted_press_gradient

        predicted_vel_press = torch.cat((predicted_velocity, predicted_pressure), dim=-1)

        target_vel_press = graph.y

        z_passes = []
        if passes_flag:
            pos = graph.pos.detach().cpu()
            for i, x_out in enumerate(x_passes):
                msg_norm = torch.norm(x_out, dim=1).detach().cpu()
                z_passes.append([i, pos.clone(), msg_norm])

        return predicted_gradient_norm, target_gradient_norm, predicted_vel_press, target_vel_press, energy_deg, entropy_deg, z_passes

    def training_step(self, batch, batch_idx):

        # Accumulate steps during the first training epoch
        if self._accumulating and self.trainer.global_step % 10000 == 0:
            node_vel_mean          = self._node_vel_normalizer._mean().cpu().numpy()
            node_vel_std           = self._node_vel_normalizer._std_with_epsilon().cpu().numpy()
            node_press_mean        = self._node_press_normalizer._mean().cpu().numpy()
            node_press_std         = self._node_press_normalizer._std_with_epsilon().cpu().numpy()
            edge_mean              = self._edge_normalizer._mean().cpu().numpy()
            edge_std               = self._edge_normalizer._std_with_epsilon().cpu().numpy()
            target_vel_grad_mean   = self._output_vel_normalizer._mean().cpu().numpy()
            target_vel_grad_std    = self._output_vel_normalizer._std_with_epsilon().cpu().numpy()
            target_press_grad_mean = self._output_press_normalizer._mean().cpu().numpy()
            target_press_grad_std  = self._output_press_normalizer._std_with_epsilon().cpu().numpy()
            print(f"[Step {self.trainer.global_step}] node vel: mean={node_vel_mean}, std={node_vel_std}")
            print(f"[Step {self.trainer.global_step}] node press: mean={node_press_mean}, std={node_press_std}")
            print(f"[Step {self.trainer.global_step}] edge: mean={edge_mean}, std={edge_std}")
            print(f"[Step {self.trainer.global_step}] target vel grad: mean={target_vel_grad_mean}, std={target_vel_grad_std}")
            print(f"[Step {self.trainer.global_step}] target press grad: mean={target_press_grad_mean}, std={target_press_grad_std}")

        # Freeze the normalizers after the first epoch
        if self._accumulating and self.current_epoch == 0 and self.trainer.is_last_batch:
            print("Normalizers frozen")
            self._node_vel_normalizer.freeze()
            self._node_press_normalizer.freeze()
            self._edge_normalizer.freeze()
            self._output_vel_normalizer.freeze()
            self._output_press_normalizer.freeze()
            print("✔ Normalization finished. Switching to LR decay + λ_deg warmup.")
            self._accumulating = False

        graph = batch

        node_type = batch.n.squeeze()
        mask_loss = (node_type == NodeType.NORMAL)

        predicted_gradient_norm, target_gradient_norm, _, _, energy_deg, entropy_deg, _ = self.forward(graph)

        # Calculate loss
        error = torch.sum((target_gradient_norm - predicted_gradient_norm) ** 2, dim=1)
        loss_data = torch.mean(error[mask_loss])

        loss_deg_E = (energy_deg[mask_loss, :, 0] ** 2).mean()
        loss_deg_S = (entropy_deg[mask_loss, :, 0] ** 2).mean()

        lambda_deg = self.configure_lambda_deg()
        loss = loss_data + lambda_deg * (loss_deg_E + loss_deg_S)

        self.log('train_loss', loss.detach().item(), prog_bar=True, on_epoch=True, on_step=False, batch_size=self.batch_size)
        self.log('train_data', loss_data.detach().item(), prog_bar=True, on_epoch=True, on_step=False,batch_size=self.batch_size)
        self.log('train_deg_E', loss_deg_E.detach().item(), prog_bar=True, on_epoch=True, on_step=False, batch_size=self.batch_size)
        self.log('train_deg_S', loss_deg_S.detach().item(), prog_bar=True, on_epoch=True, on_step=False, batch_size=self.batch_size)
        self.log('lambda_deg', lambda_deg, prog_bar=True, on_epoch=True, on_step=False, batch_size=self.batch_size)


        if self.current_epoch == 0:
            return None  # Skip training to accumulate stats

        return loss

    def validation_step(self, batch, batch_idx):

        graph = batch

        node_type = batch.n.squeeze()
        mask_loss = (node_type == NodeType.NORMAL)

        predicted_gradient_norm, target_gradient_norm, predicted_vel_press, _, energy_deg, entropy_deg, _ = self.forward(graph)

        # Calculate loss
        error = torch.sum((target_gradient_norm - predicted_gradient_norm) ** 2, dim=1)
        loss_data = torch.mean(error[mask_loss])

        loss_deg_E = (energy_deg[mask_loss, :, 0] ** 2).mean()
        loss_deg_S = (entropy_deg[mask_loss, :, 0] ** 2).mean()

        lambda_deg = self.configure_lambda_deg()
        loss = loss_data + lambda_deg * (loss_deg_E + loss_deg_S)

        self.log('val_loss', loss.detach().item(), prog_bar=True, on_epoch=True, on_step=False, batch_size=self.batch_size)
        self.log('val_data', loss_data.detach().item(), prog_bar=True, on_epoch=True, on_step=False,batch_size=self.batch_size)
        self.log('val_deg_E', loss_deg_E.detach().item(), prog_bar=True, on_epoch=True, on_step=False, batch_size=self.batch_size)
        self.log('val_deg_S', loss_deg_S.detach().item(), prog_bar=True, on_epoch=True, on_step=False, batch_size=self.batch_size)
        self.log('lambda_deg', lambda_deg, prog_bar=True, on_epoch=True, on_step=False, batch_size=self.batch_size)

        # === ROLLOUT EVALUATION ===
        if (self.current_epoch % self.rollout_freq == 0) and (self.current_epoch > 0):
            # if self.rollout_simulation in batch.idx:
            if len(self.rollouts_z_t1_pred) == 0:
                # Initial state
                self.rollouts_z_t1_pred.append(graph.x)
                self.rollouts_z_t1_gt.append(graph.x)
                self.rollouts_idx.append(self.local_rank)

            mask_normal = (node_type == NodeType.NORMAL)

            # Update predicted state
            z_t1_pred = predicted_vel_press.clone()
            # Apply BC
            z_t1_pred[~mask_normal] = graph.y[~mask_normal]
            # Append variables
            self.rollouts_z_t1_pred.append(z_t1_pred)
            self.rollouts_z_t1_gt.append(batch.y)
            self.rollouts_idx.append(self.local_rank)

    @torch.no_grad()
    def predict_step(self, batch, batch_idx, passes_flag=False):

        graph = batch
        z_t_input = graph.x.clone()
        node_type = graph.n.clone().squeeze()

        _, _, predicted_vel_press, _, _, _, z_passes = self.forward(graph, passes_flag)

        mask_normal = (node_type == NodeType.NORMAL)

        # Update predicted state
        z_t1_pred = predicted_vel_press.clone()
        z_t1_pred[~mask_normal] = graph.y[~mask_normal]

        return z_t1_pred, graph.y, z_passes

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.initial_lr)
            
        def lr_lambda(step):
            if self._accumulating:
                return 1.0
            elif step >= self.decay_steps:
                return self.final_lr / self.initial_lr
            else:
                decay_progress = (step - self.warmup_steps) / (self.decay_steps - self.warmup_steps)
                return (self.final_lr / self.initial_lr) ** decay_progress

        lr_scheduler = {
            'scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda),
            'interval': 'step',
            'monitor': 'val_loss'}

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}

    def configure_lambda_deg(self):
        if self._accumulating:
            return self.lambda_deg_start

        step = self.trainer.global_step
        if step < self.lambda_deg_warmup_start:
            return self.lambda_deg_start
        elif step >= self.lambda_deg_warmup_end:
            return self.lambda_deg_end
        else:
            increment = (step - self.lambda_deg_warmup_start) / (self.lambda_deg_warmup_end - self.lambda_deg_warmup_start)
            return self.lambda_deg_start + increment * (self.lambda_deg_end - self.lambda_deg_start)

    def __add_noise(self, graph):

        current = graph.x
        type = graph.n.squeeze()
        # Sample random noise
        noise = torch.normal(std=self.noise_var, mean=0.0, size=current.shape).to(self.device)
        # Do not apply noise to NON-NORMAL nodes
        mask_not_normal_node = torch.argwhere(type !=NodeType.NORMAL).squeeze()
        noise[mask_not_normal_node] = 0
        noise_mask = noise.to(self.device)
        graph.x = current + noise_mask

        return graph

    def __process_node_attr(self, types, current):

        node_feature = []
        # build one-hot for node type
        node_type = torch.squeeze(types.long())
        one_hot = F.one_hot(node_type, NodeType.SIZE)
        # split current data into pressure and velocity
        vel   = current[:, :2]
        press = current[:, 2:]
        # Normalize inputs
        vel_norm   = self._node_vel_normalizer(vel, accumulate=self._accumulating)
        press_norm = self._node_press_normalizer(press, accumulate=self._accumulating)
        # Concatenate normalized values
        instant_norm = torch.cat((vel_norm, press_norm), dim=-1)
        # append and concatenate node attributes
        node_feature.append(instant_norm)
        node_feature.append(one_hot)

        node_feats = torch.cat(node_feature, dim=1).float()

        return node_feats

    def __process_edge_attr(self, edge_attr):

        edge_attr = self._edge_normalizer(edge_attr,accumulate=self._accumulating)

        return edge_attr

    def __instant_to_gradient(self, target, current):

        vel_gradient   = (target[:, :2] - current[:, :2]) / self.dt
        press_gradient = (target[:, 2:] - current[:, 2:]) / self.dt

        return vel_gradient, press_gradient
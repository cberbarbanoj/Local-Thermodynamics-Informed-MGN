"""callbacks.py"""

import os
import lightning.pytorch as pl
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import imageio
import wandb
import shutil
from src.utils.plots import plot_2D, plotError
from src.evaluate import roll_out, compute_error, print_error, rrmse_inf, rollout_eval
from lightning.pytorch.callbacks import LearningRateFinder


class HistogramPassesCallback(pl.Callback):

    def on_validation_end(self, trainer, pl_module):
        if (pl_module.current_epoch % 50 == 0 and len(pl_module.error_message_pass) > 0):
            table = wandb.Table(data=pl_module.error_message_pass, columns=["pass", "epoch", "error"])
            trainer.logger.experiment.log(
                {f'error_message_pass': table})


class RolloutCallback(pl.Callback):
    def __init__(self, dataloader, **kwargs):
        super().__init__(**kwargs)
        self.dataloader = dataloader

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        """Called when the val epoch begins."""
        pl_module.rollouts_z_t1_pred = []
        pl_module.rollouts_z_t1_gt = []
        pl_module.rollouts_idx = []

        folder_path = Path(trainer.checkpoint_callback.dirpath) / 'videos'
        if folder_path.exists():
            # Remove the folder and its contents
            shutil.rmtree(folder_path)

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch > 0 and (pl_module.current_epoch + 1) % pl_module.rollout_freq == 0:
            try:
                z_net, z_gt, q_0, edge_index, n, elements, mask = roll_out(pl_module, self.dataloader, pl_module.device, pl_module.data_dim)

                for var in range(3):
                    save_dir = os.path.join(pl_module.save_folder, f'epoch_{trainer.current_epoch}_var_{var}.gif')
                    plot_2D(z_net, z_gt, save_dir=save_dir, var=var, q_0=q_0)
                    trainer.logger.experiment.log({"rollout": wandb.Video(save_dir, format='gif')})

            except:
                print()

    def on_train_end(self, trainer, pl_module):
        z_net, z_gt, q_0, edge_index, n, elements, mask = roll_out(pl_module, self.dataloader, pl_module.device, pl_module.data_dim)
        filePath = os.path.join(pl_module.save_folder, 'metrics.txt')
        with open(filePath, 'w') as f:
            error, L2_list = compute_error(z_net, z_gt, pl_module.state_variables)
            lines = print_error(error)
            f.write('\n'.join(lines))
            print("[Test Evaluation Finished]\n")
            f.close()
        plotError(z_gt, z_net, L2_list, pl_module.state_variables, pl_module.data_dim, pl_module.save_folder)
        for var in range(3):
            save_dir_var = os.path.join(pl_module.save_folder, f'epoch_{trainer.current_epoch}_var_{var}.gif')
            plot_2D(z_net, z_gt, save_dir=save_dir_var, var=var, q_0=q_0)
            trainer.logger.experiment.log({"rollout": wandb.Video(save_dir_var, format='gif')})

        shutil.copyfile(os.path.join('src', 'gnn_nodal.py'), os.path.join(pl_module.save_folder, 'gnn_nodal.py'))
        shutil.copyfile(os.path.join('data', 'jsonFiles', 'dataset_cfd.json'),
                        os.path.join(pl_module.save_folder, 'dataset_cfd.json'))


class EvaluateRollout(pl.Callback):
    def __init__(self, dataloader, split, freq=1, wandb=None, steps=50):
        super().__init__()
        self.dataloader = dataloader
        self.split = split
        self.freq = freq
        self.wandb = wandb
        self.steps = steps

    def on_validation_epoch_end(self, trainer, pl_module: pl.LightningModule):

        if (trainer.current_epoch % self.freq == 0):

            predictions, targets = [], []

            print(f'\nEvaluating TIMGN model on validation steps={self.steps}\n')

            prediction_rollout, target_rollout, q_0, edge_index, n_rollout, elements, mask = rollout_eval(pl_module, self.dataloader, pl_module.device,
                                                                         pl_module.data_dim, self.steps)
            
            predictions.append(prediction_rollout[:, mask, :])
            targets.append(target_rollout[:, mask, :])

            predictions = np.concatenate(predictions, axis=0)
            targets = np.concatenate(targets, axis=0)

            rmse_ux = rrmse_inf(targets[..., 0], predictions[..., 0])
            rmse_uy = rrmse_inf(targets[..., 1], predictions[..., 1])
            rmse_p = rrmse_inf(targets[..., 2], predictions[..., 2])

            if wandb is None:
                trainer.logger.experiment.log(
                    {f"{self.split}  {self.steps}-step U_x rrmse_inf": rmse_ux, "epoch": trainer.current_epoch})
                trainer.logger.experiment.log(
                    {f"{self.split}  {self.steps}-step U_y rrmse_inf": rmse_uy, "epoch": trainer.current_epoch})
                trainer.logger.experiment.log(
                    {f"{self.split}  {self.steps}-step P rrmse_inf": rmse_p, "epoch": trainer.current_epoch})
            else:
                wandb.log({f"{self.split} {self.steps}-step U_x rrmse_inf": rmse_ux, "epoch": trainer.current_epoch})
                wandb.log({f"{self.split} {self.steps}-step U_y rrmse_inf": rmse_uy, "epoch": trainer.current_epoch})
                wandb.log({f"{self.split} {self.steps}-step P rrmse_inf": rmse_p, "epoch": trainer.current_epoch})


class FineTuneLearningRateFinder(LearningRateFinder):
    def __init__(self, milestones, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.milestones = milestones

    def on_fit_start(self, *args, **kwargs):
        return

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch in self.milestones or trainer.current_epoch == 0:
            self.lr_find(trainer, pl_module)


class MessagePassing(pl.Callback):
    def __init__(self, dataloader, rollout_variable=None, rollout_freq=50,
                 rollout_simulation=None, **kwargs):
        super().__init__(**kwargs)
        self.rollout_variable = rollout_variable
        self.rollout_freq = rollout_freq
        if rollout_simulation is None:
            self.rollout_simulation = [0]
            self.rollout_gt = {0: []}
        else:
            self.rollout_simulation = rollout_simulation
            self.rollout_gt = {sim: [] for sim in range(len(rollout_simulation))}
        for sample in dataloader:
            if rollout_simulation is None:
                self.rollout_gt[0].append(sample)
            else:
                self.rollout_gt[int(sample.idx)].append(sample)

    def __clean_artifacts(self, trainer):
        folder_path = Path(trainer.checkpoint_callback.dirpath) / 'videos'
        if folder_path.exists():
            # Remove the folder and its contents
            shutil.rmtree(folder_path)

    def on_validation_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.__clean_artifacts(trainer)

    def make_gif_from_z_passes(self, z_passes, save_path, fps=2):
        if not z_passes:
            print("Warning: z_passes is empty, skipping GIF generation.")
            return None

        temp_folder = Path("temp_frames")
        temp_folder.mkdir(exist_ok=True)
        image_files = []

        all_values = [msg_norm.numpy() for _, _, msg_norm in z_passes]
        all_values = np.concatenate(all_values)
        vmin, vmax = np.min(all_values), np.max(all_values)

        for i, pos, msg_norm in z_passes:
            pos_np = pos.numpy()
            msg_np = msg_norm.numpy().squeeze()

            plt.figure(figsize=(12, 6))
            sc = plt.scatter(pos_np[:, 0], pos_np[:, 1], c=msg_np, cmap='viridis', s=50, vmin=vmin, vmax=vmax)
            plt.title(f'Message Norm - Pass {i}', fontsize=16)
            plt.xlabel("X", fontsize=14)
            plt.ylabel("Y", fontsize=14)
            plt.colorbar(sc, label='||Message||')
            plt.tight_layout()

            frame_file = temp_folder / f"frame_{i}.png"
            plt.savefig(frame_file)
            plt.close()
            image_files.append(frame_file)

        images = [imageio.imread(str(img)) for img in image_files]
        imageio.mimsave(save_path, images, fps=fps)

        for img_file in image_files:
            os.remove(img_file)
        temp_folder.rmdir()
        return save_path

    def on_validation_epoch_end(self, trainer, pl_module, index=0):

        if ((pl_module.current_epoch + 1) % self.rollout_freq == 0) and (pl_module.current_epoch > 0):
            for sim, rollout_gt in self.rollout_gt.items():
                iter_step = len(rollout_gt) // 2
                print(f'\nMessage passing Iter={iter_step} Sim={self.rollout_simulation[sim]}')
                # === Obtener sample de validación y mover a device ===
                device = pl_module.device
                sample = rollout_gt[index].to(device)
                # 🔧 FIX: Resetear entrada a estado base esperado
                sample.x = sample.y[:, :3].clone()
                # === Ejecutar modelo ===
                z_pred, z_t1, z_passes = pl_module.predict_step(sample, 1, passes_flag=True)
                # === Crear GIF de pasos de mensaje ===
                video_dir = Path(trainer.checkpoint_callback.dirpath) / 'videos'
                video_dir.mkdir(exist_ok=True)
                gif_path = video_dir / f"message_pass_sim{sim}.gif"

                self.make_gif_from_z_passes(z_passes, str(gif_path), fps=2)

                # === Loguear a wandb ===
                trainer.logger.experiment.log({
                    f"Message Passing Simulation {self.rollout_simulation[sim]}": wandb.Video(str(gif_path), format="gif")
                })

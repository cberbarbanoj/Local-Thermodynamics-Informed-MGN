#--------------------------------------------
# Main training script
#--------------------------------------------

import os
import json
import argparse
import datetime
import torch
import random
import wandb

import lightning.pytorch as pl
from torch_geometric.loader import DataLoader
from pytorch_lightning.loggers import WandbLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from src.dataLoader.dataset import H5GraphDataset
from src.simulator import CFDSolver
from src.callbacks import RolloutCallback, HistogramPassesCallback, MessagePassing, EvaluateRollout
from src.utils.utils import str2bool

if __name__ == '__main__':

    # Command-line argument parsing
    parser = argparse.ArgumentParser(description='Thermodynamics-Informed Mesh Graph Neural Networks')

    # Study Case
    parser.add_argument('--gpu', default=True, type=str2bool, help='GPU acceleration')
    parser.add_argument('--transfer_learning', default=False, type=str2bool, help='GPU acceleration')
    parser.add_argument('--pretrain_weights', default=r'epoch=21-val_loss=0.01.ckpt', type=str, help='name')

    # Dataset Parameters
    parser.add_argument('--dset_dir', default='data', type=str, help='dataset directory')
    parser.add_argument('--dset_name', default=r'dataset_CYLINDER.json', type=str, help='dataset directory')

    # Save options
    parser.add_argument('--output_dir', default='outputs', type=str, help='output directory')
    parser.add_argument('--output_dir_exp', default=r'outputs/', type=str, help='output directory')
    parser.add_argument('--plot_sim', default=True, type=str2bool, help='plot test simulation')
    parser.add_argument('--experiment_name', default='training_01', type=str, help='experiment output name tensorboard')
    args = parser.parse_args()  # Parse command-line arguments

    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')

    f = open(os.path.join(args.dset_dir, 'jsonFiles', args.dset_name))
    dInfo = json.load(f)

    pl.seed_everything(dInfo['model']['seed'], workers=True)

    # Train set
    train_set        = H5GraphDataset(dInfo, os.path.join(args.dset_dir, dInfo['dataset']['datasetPaths']['train']),short=False, portion=0.1)
    train_dataloader = DataLoader(train_set, batch_size=dInfo['model']['batch_size'], num_workers=2, pin_memory=True, prefetch_factor=8, persistent_workers=True, shuffle=True)

    # Validation set
    val_set        = H5GraphDataset(dInfo, os.path.join(args.dset_dir, dInfo['dataset']['datasetPaths']['val']), short=False, portion=0.1)
    val_dataloader = DataLoader(val_set, batch_size=dInfo['model']['batch_size'], num_workers=2, pin_memory=True, prefetch_factor=4, persistent_workers=True, shuffle=True)

    # Test set
    test_set        = H5GraphDataset(dInfo, os.path.join(args.dset_dir, dInfo['dataset']['datasetPaths']['test']))
    test_dataloader = DataLoader(test_set, batch_size=1)

    # Name of the run
    name = f"TIMGN_Hidden_{dInfo['model']['dim_hidden']}_Passes_{dInfo['model']['passes']}_LambdaDeg_{dInfo['model']['lambda_deg_end']}_Noise_{dInfo['model']['noise_var']}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    save_folder = f'outputs/runs/{name}'

    # Log to WandB
    wandb_logger = WandbLogger(
        name    = name, 
        project = dInfo['project_name']
        )
    
    # Define Callbacks
    early_stop = EarlyStopping(
        monitor   = "val_loss", 
        min_delta = 0.00, 
        patience  = 1000, 
        verbose   = True, 
        mode      = "min"
        )
    checkpoint = ModelCheckpoint(
        dirpath      = save_folder,  
        filename     = '{epoch}-{val_loss:.2f}',
          monitor    = 'val_loss', 
          save_top_k = 3
          )
    lr_monitor      = LearningRateMonitor(logging_interval='epoch')
    rollout         = RolloutCallback(test_dataloader)
    eval_rollout    = EvaluateRollout(test_dataloader, 'test', steps=200)
    message_passing = MessagePassing(test_dataloader, rollout_freq=5)

    # Instantiate model
    net = CFDSolver(
        dims        = train_set.dims,
        dInfo       = dInfo,
        save_folder = save_folder
    )
    print(type(net))
    print(net)
    wandb_logger.watch(net)

    # Load weights for transfer learning
    if args.transfer_learning:
        path_checkpoint = os.path.join(args.dset_dir, 'weights', args.pretrain_weights)
        checkpoint_ = torch.load(path_checkpoint, map_location=device)
        net.load_state_dict(checkpoint_['state_dict'], strict=False)

    # Set Trainer
    trainer = pl.Trainer(
        accelerator          = "auto",
        logger               = wandb_logger,
        callbacks            = [checkpoint, lr_monitor, rollout, eval_rollout, early_stop, message_passing],
        profiler             = "simple",
        num_sanity_val_steps = 0,
        max_epochs           = dInfo['model']['max_epoch'],
        gradient_clip_val    = 0.5,
        precision            = 'bf16-mixed'
        )

    # Train model
    trainer.fit(model=net, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

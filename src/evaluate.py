"""evaluate.py"""

import os
import time
import torch
import numpy as np
from src.utils.utils import NodeType
from src.utils.utils import print_error, create_folder
from src.utils.plots import plotError, plot_2D_pyvista_pred, plot_2D_matplotlib_pred


def rrmse_inf(data_ground_truth, data_predicted, mask=None):
    x, y = data_ground_truth, data_predicted

    if isinstance(x, torch.Tensor):
        x = x.numpy()
    if isinstance(y, torch.Tensor):
        y = y.numpy()
    if mask is not None and isinstance(mask, torch.Tensor):
        mask = mask.numpy()
    # Infinity norm calculation
    se_inf = []
    for i in range(x.shape[0]):
        x_snap = x[i]
        y_snap = y[i]
        if mask is not None:
            x_snap = x_snap[mask]
            y_snap = y_snap[mask]
        error = x_snap - y_snap
        l2_norm_se = np.mean(error ** 2)  # En lugar de dividir por x.shape[-1]
        infinite_norm_se = (np.linalg.norm(x_snap, ord=np.inf) ** 2) + 1e-8
        se_inf.append(l2_norm_se / infinite_norm_se)

    # Compute final error
    mse_inf = np.mean(se_inf)
    rmse_inf = np.sqrt(mse_inf)

    return rmse_inf

def rmse(x, y):

    # Calculate squared error
    se = (x - y) ** 2
    # Calculate mean squared error
    mse = np.mean(se)

    return np.sqrt(mse)

def compute_error(z_net, z_gt, state_variables, mask=None, snapshot_range=None):

    e_all = z_net.numpy() - z_gt.numpy()
    gt_all = z_gt.numpy()

    # Subset by snapshot_range
    if snapshot_range is not None:
        start, end = snapshot_range
        e = e_all[start:end]
        gt = gt_all[start:end]
        z_net_subset = z_net.numpy()[start:end]
    else:
        e = e_all
        gt = gt_all
        z_net_subset = z_net.numpy()

    error = {clave: [] for clave in state_variables}
    L2_list = {clave: [] for clave in state_variables}

    for i, sv in enumerate(state_variables):
        e_ing = rrmse_inf(gt[:, :, i:i+1], z_net_subset[:, :, i:i+1], mask=mask)
        if mask is not None:
            L2 = ((e[1:, mask, i] ** 2).sum(1) / (gt[1:, mask, i] ** 2).sum(1)) ** 0.5
        else:
            L2 = ((e[1:, :, i] ** 2).sum(1) / (gt[1:, :, i] ** 2).sum(1)) ** 0.5
        error[sv] = e_ing
        L2_list[sv].extend(L2)

    return error, L2_list

def roll_out(model, dataloader, device, dim_data):
    data = [sample for sample in dataloader]

    dim_z   = data[0].x.shape[1]
    N_nodes = data[0].x.shape[0]

    z_net = torch.zeros(len(data) + 1, N_nodes, dim_z)
    z_gt  = torch.zeros(len(data) + 1, N_nodes, dim_z)

    z_t        = data[0].x.clone()
    edge_index = data[0].edge_index
    elements   = data[0].elements
    n          = data[0].n
    node_type  = data[0].n.clone().squeeze()
    mask = (node_type == NodeType.NORMAL).numpy()

    # Initial conditions
    z_net[0] = z_t
    z_gt[0]  = z_t

    if hasattr(data[0], 'q_0'):
        q_0 = data[0].q_0
    elif hasattr(data[0], 'pos'):
        q_0 = data[0].pos
    else:
        q_0 = None

    # for sample in data:
    try:
        for t, snap in enumerate(data):
            snap.x          = z_t
            snap            = snap.to(device)
    
            z_t1_pred, z_gt_t1, _ = model.predict_step(snap, 1)

            z_t = z_t1_pred.cpu()
            z_net[t + 1] = z_t
            z_gt[t + 1] = z_gt_t1.cpu()

    except:
        print(f'Rollout failed at: {t} snapshot')

    return z_net, z_gt, q_0, edge_index, n, elements, mask


def rollout_1step(model, dataloader, device):
    """1-step rollout for testing purposes"""
    data = [sample for sample in dataloader]

    T = len(data)
    N, D = data[0].x.shape
    z_pred = torch.zeros(T, N, D)
    z_true = torch.zeros(T, N, D)

    node_type  = data[0].n.clone().squeeze()
    mask = (node_type == NodeType.NORMAL).numpy()

    for t in range(T):
        snap   = data[t].to(device)
        snap.x = snap.x.clone().to(device)

        with torch.no_grad():
            z_t1_pred, z_t1_true, _ = model.predict_step(snap, 1)

        z_pred[t] = z_t1_pred.cpu()
        z_true[t] = z_t1_true.cpu()

    return z_pred, z_true, mask


def rollout_eval(model, dataloader, device, dim_data, max_steps):
    data = [sample for sample in dataloader]

    dim_z   = data[0].x.shape[1]
    N_nodes = data[0].x.shape[0]

    z_net = torch.zeros(len(data) + 1, N_nodes, dim_z)
    z_gt  = torch.zeros(len(data) + 1, N_nodes, dim_z)

    z_t        = data[0].x.clone()
    edge_index = data[0].edge_index
    elements   = data[0].elements
    n          = data[0].n
    node_type  = data[0].n.clone().squeeze()
    mask = (node_type == NodeType.NORMAL).numpy()

    # Initial conditions
    z_net[0] = z_t
    z_gt[0]  = z_t

    if hasattr(data[0], 'q_0'):
        q_0 = data[0].q_0
    elif hasattr(data[0], 'pos'):
        q_0 = data[0].pos
    else:
        q_0 = None

    # for sample in data:
    try:
        for t, snap in enumerate(data):
            if t < max_steps - 1:
                snap.x          = z_t
                snap            = snap.to(device)
        
                z_t1_pred, z_gt_t1, _ = model.predict_step(snap, 1)

                z_t = z_t1_pred.cpu()
                z_net[t + 1] = z_t
                z_gt[t + 1] = z_gt_t1.cpu()

    except Exception as e:
        print(f"Error details: {e}")

    return z_net, z_gt, q_0, edge_index, n, elements, mask


def generate_results(ms_mgn, test_dataloader, dInfo, device, output_dir_exp):
    # Generate output folder
    output_dir_exp = create_folder(output_dir_exp)
    dim_data = 2 if dInfo['dataset']['dataset_dim'] == '2D' else 3
    # Make roll out
    start_time = time.time()
    z_net, z_gt, q_0, edge_index, n, elements, mask = roll_out(ms_mgn, test_dataloader, device, dim_data)
    print(f'Time to generate rollout: {time.time() - start_time}')
    filePath = os.path.join(output_dir_exp, 'metrics.txt')
    with open(filePath, 'w') as f:
        state_vars = dInfo['dataset']['state_variables']
        # One-snapshot prediction error
        error_1snap, _ = compute_error(z_net, z_gt, state_vars, snapshot_range=(0, 2))
        f.write("\nOne snapshot prediction error:\n" + '\n'.join(print_error(error_1snap)))

        # 50 snapshots rollout error
        error_50snap, _ = compute_error(z_net, z_gt, state_vars, snapshot_range=(0, 50))
        f.write("\n50 snapshots prediction error:\n" + '\n'.join(print_error(error_50snap)))

        # 200 snapshots rollout error
        error_200snap, _ = compute_error(z_net, z_gt, state_vars, snapshot_range=(0, 200))
        f.write("\n200 snapshots prediction error:\n" + '\n'.join(print_error(error_200snap)))

        # Complete rollout error
        error_full, L2_list = compute_error(z_net, z_gt, state_vars)
        f.write("\nComplete rollout error:\n" + '\n'.join(print_error(error_full)))

        print("[Test Evaluation Finished]\n")
        f.close()

    plotError(z_gt, z_net, L2_list, dInfo['dataset']['state_variables'], dInfo['dataset']['dataset_dim'],
              output_dir_exp)

    print("[Ready to print rollout]\n")

    for var in range(3):
        save_dir_gif = os.path.join(output_dir_exp, f'result_var_{var}.gif')
        print(f"[Printing rollout for variable {var}. Gif will be saved in {save_dir_gif}]\n")
        plot_2D_pyvista_pred(z_net, z_gt, elements, save_dir=save_dir_gif, var=var, q_0=q_0)
        # Uncomment this and comment previous one if you prefer to use matplotlib to generate the gif
        # plot_2D_matplotlib_pred(z_net, z_gt, elements, save_dir=save_dir_gif, var=var, q_0=q_0)


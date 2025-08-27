"""plots.py"""

import matplotlib
matplotlib.use('Agg')
import gc
import os

import numpy as np
try:
    import moviepy.editor as mp
except ImportError:
    import moviepy as mp
import imageio
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.tri as tri
from matplotlib.ticker import FormatStrFormatter
from matplotlib.colors import Normalize
import pyvista as pv

# Uncomment when working on a device not connected to a physical display
# pv.start_xvfb()

def plot_2D_image(z_net, z_gt, step, var=0, q_0=None, output_dir='outputs'):
    figsize = (24, 8)
    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(1, 3, 2)
    ax2 = fig.add_subplot(1, 3, 1)
    ax3 = fig.add_subplot(1, 3, 3)

    # Oculta los bordes de los ejes
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_color((0.8, 0.8, 0.8))
    ax1.spines['left'].set_color((0.8, 0.8, 0.8))
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_color((0.8, 0.8, 0.8))
    ax2.spines['left'].set_color((0.8, 0.8, 0.8))
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['bottom'].set_color((0.8, 0.8, 0.8))
    ax3.spines['left'].set_color((0.8, 0.8, 0.8))

    ax1.set_title('MeshGraphNet')
    ax1.set_xlabel('X'), ax1.set_ylabel('Y')
    ax2.set_title('Ground Truth')
    ax2.set_xlabel('X'), ax2.set_ylabel('Y')
    ax3.set_title('MeshGraphNet error')
    ax3.set_xlabel('X'), ax3.set_ylabel('Y')

    # Asegura una escala igual en ambos ejes
    ax1.set_aspect('equal')
    ax2.set_aspect('equal')
    ax3.set_aspect('equal')

    # Adjust ranges
    if q_0 is not None:
        X, Y = q_0[:, 0].numpy(), q_0[:, 1].numpy()
    else:
        X, Y = z_gt[:, :, 0].numpy(), z_gt[:, :, 1].numpy()
    z_min, z_max = z_gt[step, :, var].min(), z_gt[step, :, var].max()

    max_range_x = np.array([X.max() - X.min()]).max()
    max_range_y = np.array([Y.max() - Y.min()]).max()
    max_range = max(max_range_x, max_range_y)

    figsize_adjusted = (figsize[0] * (max_range_x / max_range), figsize[1] * (max_range_y / max_range))
    fig.set_size_inches(figsize_adjusted)

    Xb = 0.5 * max_range_x * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (X.max() + X.min())
    Yb = 0.5 * max_range_y * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (Y.max() + Y.min())
    # Initial snapshot
    if q_0 is not None:
        q1_net, q3_net = q_0[:, 0], q_0[:, 1]
        q1_gt, q3_gt = q_0[:, 0], q_0[:, 1]
    else:
        q1_net, q3_net = z_net[step, :, 0], z_net[step, :, 1]
        q1_gt, q3_gt = z_gt[step, :, 0], z_gt[step, :, 1]
    var_net, var_gt = z_net[step, :, var], z_gt[step, :, var]
    var_error = var_gt - var_net
    var_error_min, var_error_max = var_error.min(), var_error.max()
    # Bounding box
    for xb, yb in zip(Xb, Yb):
        ax1.plot([xb], [yb], 'w')
        ax2.plot([xb], [yb], 'w')
        ax3.plot([xb], [yb], 'w')
    # Scatter points
    s1 = ax1.scatter(q1_net, q3_net, c=var_net, vmax=z_max, vmin=z_min)
    s2 = ax2.scatter(q1_gt, q3_gt, c=var_gt, vmax=z_max, vmin=z_min)
    s3 = ax3.scatter(q1_net, q3_net, c=var_error, vmax=var_error_max, vmin=var_error_min)
    # Colorbar
    fig.colorbar(s1, ax=ax1, location='bottom', pad=0.08)
    fig.colorbar(s2, ax=ax2, location='bottom', pad=0.08)
    fig.colorbar(s3, ax=ax3, location='bottom', pad=0.08)

    fig.savefig(os.path.join(output_dir, f'flow_{step}.png'))

    # Oculta las marcas de los ejes y las etiquetas
    ax1.tick_params(axis='both', which='both', length=0)
    plt.savefig(os.path.join(output_dir, f'flow.svg'), format="svg")


def plot_2D(z_net, z_gt, save_dir, var=0, q_0=None):
    T = z_net.size(0)
    figsize = (20, 12)
    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(1, 3, 2)
    ax2 = fig.add_subplot(1, 3, 1)
    ax3 = fig.add_subplot(1, 3, 3)
    ax1.set_title('MeshGraphNet'), ax1.grid()
    ax1.set_xlabel('X'), ax1.set_ylabel('Y')
    ax2.set_title('Ground Truth'), ax2.grid()
    ax2.set_xlabel('X'), ax2.set_ylabel('Y')
    ax3.set_title('MeshGraphNet error'), ax3.grid()
    ax3.set_xlabel('X'), ax3.set_ylabel('Y')

    # Adjust ranges
    if q_0 is not None:
        X, Y = q_0[:, 0].numpy(), q_0[:, 1].numpy()
    else:
        X, Y = z_gt[:, :, 0].numpy(), z_gt[:, :, 1].numpy()
    z_min, z_max = z_gt[:, :, var].min(), z_gt[:, :, var].max()

    max_range_x = np.array([X.max() - X.min()]).max()
    max_range_y = np.array([Y.max() - Y.min()]).max()
    max_range = max(max_range_x, max_range_y)

    figsize_adjusted = (figsize[0] * (max_range_x / max_range), figsize[1] * (max_range_y / max_range))
    fig.set_size_inches(figsize_adjusted)

    Xb = 0.5 * max_range_x * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (X.max() + X.min())
    Yb = 0.5 * max_range_y * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (Y.max() + Y.min())

    # Initial snapshot
    if q_0 is not None:
        q1_net, q3_net = q_0[:, 0], q_0[:, 1]
        q1_gt, q3_gt = q_0[:, 0], q_0[:, 1]
    else:
        q1_net, q3_net = z_net[0, :, 0], z_net[0, :, 1]
        q1_gt, q3_gt = z_gt[0, :, 0], z_gt[0, :, 1]
    var_net, var_gt = z_net[-1, :, var], z_gt[-1, :, var]
    var_error = var_gt - var_net
    var_error_min, var_error_max = var_error.min(), var_error.max()
    # Bounding box
    for xb, yb in zip(Xb, Yb):
        ax1.plot([xb], [yb], 'w')
        ax2.plot([xb], [yb], 'w')
        ax3.plot([xb], [yb], 'w')
    # Scatter points
    s1 = ax1.scatter(q1_net, q3_net, c=var_net,
                     vmax=z_max, vmin=z_min)
    s2 = ax2.scatter(q1_gt, q3_gt, c=var_gt, vmax=z_max,
                     vmin=z_min)
    s3 = ax3.scatter(q1_net, q3_net, c=var_error,
                     vmax=var_error_max, vmin=var_error_min)

    # Colorbar
    fig.colorbar(s1, ax=ax1, location='bottom', pad=0.08)
    fig.colorbar(s2, ax=ax2, location='bottom', pad=0.08)
    fig.colorbar(s3, ax=ax3, location='bottom', pad=0.08)

    # Animation
    def animate(snap):
        ax1.cla()
        ax2.cla()
        ax3.cla()
        ax1.set_title(f'MeshGraphNet, f={str(snap)}'), ax1.grid()
        ax1.set_xlabel('X'), ax1.set_ylabel('Y')
        ax2.set_title('Ground Truth'), ax2.grid()
        ax2.set_xlabel('X'), ax2.set_ylabel('Y')
        ax3.set_title('MeshGraphNet error'), ax3.grid()
        ax3.set_xlabel('X'), ax3.set_ylabel('Y')
        # Bounding box
        for xb, yb in zip(Xb, Yb):
            ax1.plot([xb], [yb], 'w')
            ax2.plot([xb], [yb], 'w')
            ax3.plot([xb], [yb], 'w')
        # Scatter points

        if q_0 is not None:
            q1_net, q3_net = q_0[:, 0], q_0[:, 1]
            q1_gt, q3_gt = q_0[:, 0], q_0[:, 1]
        else:
            q1_net, q3_net = z_net[snap, :, 0], z_net[snap, :, 1]
            q1_gt, q3_gt = z_gt[snap, :, 0], z_gt[snap, :, 1]

        var_net, var_gt = z_net[snap, :, var], z_gt[snap, :, var]
        var_error = var_gt - var_net

        ax1.scatter(q1_net, q3_net, c=var_net, vmax=z_max, vmin=z_min)
        ax2.scatter(q1_gt, q3_gt, c=var_gt, vmax=z_max, vmin=z_min)
        ax3.scatter(q1_net, q3_net, c=var_error, vmax=var_error_max, vmin=var_error_min)
        # fig.savefig(os.path.join('images/', f'beam_{snap}.png'))
        return fig,

    anim = animation.FuncAnimation(fig, animate, frames=T, repeat=False)
    writergif = animation.PillowWriter(fps=20)

    # Save as gif
    # save_dir = os.path.join(output_dir, 'beam.mp4')
    anim.save(save_dir, writer=writergif)
    plt.close('all')

# Plot methods using matplotlib

def plot_2D_matplotlib_pred(z_net, z_gt, elements, save_dir, var=0, q_0=None):
    T = z_net.shape[0]
    os.makedirs(save_dir, exist_ok=True)

    pts   = q_0.cpu().numpy()
    elems = elements.cpu().numpy().astype(np.int32)
    triang = tri.Triangulation(pts[:, 0], pts[:, 1], elems)

    all_data = np.concatenate([
        z_gt[..., var].cpu().numpy().ravel(),
        z_net[..., var].cpu().numpy().ravel()
    ])
    z_min, z_max = all_data.min(), all_data.max()
    levels_data = np.linspace(z_min, z_max, 200)
    data_ticks = np.linspace(z_min, z_max, 5)

    for frame in range(T):
        gt_frame = z_gt[frame, :, var].cpu().numpy()
        net_frame = z_net[frame, :, var].cpu().numpy()

        fig, axs = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

        # Ground Truth
        t1 = axs[0].tricontourf(triang, gt_frame, levels=levels_data,
                                cmap='viridis', vmin=z_min, vmax=z_max)
        axs[0].triplot(triang, color='k', lw=0.2, alpha=0.4)
        axs[0].set_aspect('equal'); axs[0].axis('off')
        axs[0].set_title(f'Ground Truth, f={frame}', fontsize=13)
        c1 = fig.colorbar(t1, ax=axs[0], orientation='horizontal',
                          pad=0.08, ticks=data_ticks, extend='both')
        c1.ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))

        # Prediction
        t2 = axs[1].tricontourf(triang, net_frame, levels=levels_data,
                                cmap='viridis', vmin=z_min, vmax=z_max)
        axs[1].triplot(triang, color='k', lw=0.2, alpha=0.4)
        axs[1].set_aspect('equal'); axs[1].axis('off')
        axs[1].set_title(f'Prediction, f={frame}', fontsize=13)
        c2 = fig.colorbar(t2, ax=axs[1], orientation='horizontal',
                          pad=0.08, ticks=data_ticks, extend='both')
        c2.ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))

        # Guardar imagen sin bordes
        frame_path = os.path.join(save_dir, f"frame_{frame:04d}.png")
        plt.savefig(frame_path, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        del fig
        gc.collect()

    # Crear GIF
    gif_path = os.path.join(save_dir, "animation.gif")
    with imageio.get_writer(gif_path, mode='I', duration=0.04) as writer:
        for frame in range(T):
            img_path = os.path.join(save_dir, f"frame_{frame:04d}.png")
            image = imageio.imread(img_path)
            writer.append_data(image)
            os.remove(img_path)  # Borrar PNG tras añadirlo al GIF


def plot_2D_matplotlib_error(z_net, z_gt, elements, save_dir, var=0, q_0=None):
    T = z_net.shape[0]
    os.makedirs(save_dir, exist_ok=True)

    pts   = q_0.cpu().numpy()
    elems = elements.cpu().numpy().astype(np.int32)
    triang = tri.Triangulation(pts[:, 0], pts[:, 1], elems)

    # Global range for data and error
    all_data = np.concatenate([
        z_gt[..., var].cpu().numpy().ravel(),
        z_net[..., var].cpu().numpy().ravel()
    ])
    z_min, z_max = all_data.min(), all_data.max()
    data_ticks = np.linspace(z_min, z_max, 5)
    levels_data = np.linspace(z_min, z_max, 200)

    all_err = (z_gt[..., var] - z_net[..., var]).cpu().numpy().ravel()
    err_abs_max = np.max(np.abs(all_err))
    vmin, vmax = -err_abs_max, err_abs_max
    levels_err = np.linspace(vmin, vmax, 200)
    error_ticks = np.linspace(vmin, vmax, 5)

    for frame in range(T):
        gt_frame  = z_gt[frame, :, var].cpu().numpy()
        net_frame = z_net[frame, :, var].cpu().numpy()
        err_frame = gt_frame - net_frame

        fig, axs = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

        # Ground Truth
        t1 = axs[0].tricontourf(triang, gt_frame, levels=levels_data,
                                cmap='viridis', vmin=z_min, vmax=z_max)
        axs[0].triplot(triang, color='k', lw=0.2, alpha=0.4)
        axs[0].set_aspect('equal'); axs[0].axis('off')
        axs[0].set_title(f'Ground Truth, f={frame}', fontsize=13)
        c1 = fig.colorbar(t1, ax=axs[0], orientation='horizontal',
                          pad=0.08, ticks=data_ticks, extend='both')
        c1.ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))

        # Prediction
        t2 = axs[1].tricontourf(triang, net_frame, levels=levels_data,
                                cmap='viridis', vmin=z_min, vmax=z_max)
        axs[1].triplot(triang, color='k', lw=0.2, alpha=0.4)
        axs[1].set_aspect('equal'); axs[1].axis('off')
        axs[1].set_title(f'Prediction, f={frame}', fontsize=13)
        c2 = fig.colorbar(t2, ax=axs[1], orientation='horizontal',
                          pad=0.08, ticks=data_ticks, extend='both')
        c2.ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))

        # Error
        t3 = axs[2].tricontourf(triang, err_frame, levels=levels_err,
                                cmap='berlin', vmin=vmin, vmax=vmax)
        axs[2].triplot(triang, color='k', lw=0.2, alpha=0.4)
        axs[2].set_aspect('equal'); axs[2].axis('off')
        axs[2].set_title(f'Error, f={frame}', fontsize=13)
        c3 = fig.colorbar(t3, ax=axs[2], orientation='horizontal',
                          pad=0.08, ticks=error_ticks, extend='both')
        c3.ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))

        frame_path = os.path.join(save_dir, f"frame_{frame:04d}.png")
        plt.savefig(frame_path, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        del fig
        gc.collect()

    # Crear GIF
    gif_path = os.path.join(save_dir, "animation.gif")
    with imageio.get_writer(gif_path, mode='I', duration=0.04) as writer:
        for frame in range(T):
            img_path = os.path.join(save_dir, f"frame_{frame:04d}.png")
            image = imageio.imread(img_path)
            writer.append_data(image)
            os.remove(img_path)


# Plot methods using PyVista
def plot_2D_pyvista_pred(z_net, z_gt, elements, save_dir, var=0, q_0=None):
    T = z_net.shape[0]

    # Prepare PyVista plotter with 3 subplots
    plotter = pv.Plotter(shape=(1, 2), window_size=(2400, 800), off_screen=True)
    plotter.open_gif(save_dir)

    # Precompute min/max for color normalization
    z_min, z_max = z_gt[..., var].min().cpu().item(), z_gt[..., var].max().cpu().item()

    # Get node positions
    points_2d = q_0.cpu().numpy()

    # Convert 2D to 3D
    points = np.c_[points_2d, np.zeros(points_2d.shape[0])]

    def create_mesh(points, elements):
        # Create the mesh object and set points
        mesh = pv.PolyData(points)
        # Create cells for the edges
        cells = []
        for element in elements:
            cells.extend([3, element[0], element[1], element[2]])
        cells = np.array(cells)
        # Set the lines in the mesh
        mesh.faces = cells

        return mesh

    mesh_gt = create_mesh(points, elements)
    mesh_net = create_mesh(points, elements)

    mesh_gt['scalars'] = z_gt[0, :, var].cpu().numpy()
    mesh_net['scalars'] = z_net[0, :, var].cpu().numpy()

    # Add mesh - GT Values
    plotter.subplot(0, 0)
    plotter.add_mesh(mesh_gt, scalars='scalars', point_size=0,
                     scalar_bar_args={'title': 'GT', 'vertical': False, 'title_font_size': 30, 'label_font_size': 24,
                                      'position_x': 0.2, 'position_y': 0.02}, clim=[z_min, z_max], show_edges=True)
    plotter.view_xy()
    plotter.camera.zoom(1.2)

    # Add mesh - Prediction Values
    plotter.subplot(0, 1)
    plotter.add_mesh(mesh_net, scalars='scalars', point_size=0,
                     scalar_bar_args={'title': 'Pred', 'vertical': False, 'title_font_size': 30, 'label_font_size': 24,
                                      'position_x': 0.2, 'position_y': 0.02}, clim=[z_min, z_max], show_edges=True)
    plotter.view_xy()
    plotter.camera.zoom(1.2)

    for snapshot in range(T):
        gt_frame = z_gt[snapshot, :, var].cpu().numpy()
        net_frame = z_net[snapshot, :, var].cpu().numpy()

        mesh_gt['scalars'] = gt_frame
        mesh_net['scalars'] = net_frame

        plotter.subplot(0, 0)
        plotter.remove_actor('text_gt')
        plotter.add_text(f'Ground Truth, f={snapshot}', position='upper_edge', font_size=14, name='text_gt')
        plotter.subplot(0, 1)
        plotter.remove_actor('text_pred')
        plotter.add_text(f'Thermodynamics-informed MGN, f={str(snapshot)}', position='upper_edge', font_size=14,
                         name='text_pred')

        plotter.write_frame()

    # Close the plotter
    plotter.close()

def plot_2D_pyvista_error(z_net, z_gt, elements, save_dir, var=0, q_0=None):
    T = z_net.shape[0]

    # Prepare PyVista plotter with 3 subplots
    plotter = pv.Plotter(shape=(1, 3), window_size=(3600, 800), off_screen=True)
    plotter.open_gif(save_dir)

    # Precompute min/max for consistent colormaps
    z_min, z_max = z_gt[..., var].min().cpu().item(), z_gt[..., var].max().cpu().item()
    var_error_total = (z_gt[..., var] - z_net[..., var]).cpu()
    err_min, err_max = var_error_total.min().item(), var_error_total.max().item()

    points_2d = q_0.cpu().numpy()
    points = np.c_[points_2d, np.zeros(points_2d.shape[0])]

    def create_mesh(points, elements):
        mesh = pv.PolyData(points)
        cells = []
        for element in elements:
            cells.extend([3, element[0], element[1], element[2]])
        cells = np.array(cells)
        mesh.faces = cells
        return mesh
    

    mesh_gt = create_mesh(points, elements)
    mesh_net = create_mesh(points, elements)
    mesh_err = create_mesh(points, elements)

    mesh_gt['scalars']  = z_gt[0, :, var].cpu().numpy()
    mesh_net['scalars'] = z_net[0, :, var].cpu().numpy()
    mesh_err['scalars'] = (z_gt[0, :, var] - z_net[0, :, var]).cpu().numpy()

    plotter.subplot(0, 0)
    plotter.add_mesh(mesh_gt, scalars='scalars', point_size=0, scalar_bar_args={'title': 'GT', 'vertical': False, 'title_font_size': 30, 'label_font_size': 24, 'position_x': 0.15, 'position_y': 0.02},
                     clim=[z_min, z_max], show_edges=True)
    plotter.view_xy()
    plotter.camera.zoom(1.2)

    plotter.subplot(0, 1)
    plotter.add_mesh(mesh_net, scalars='scalars', point_size=0, scalar_bar_args={'title': 'Pred', 'vertical': False, 'title_font_size': 30, 'label_font_size': 24, 'position_x': 0.15, 'position_y': 0.02},
                     clim=[z_min, z_max], show_edges=True)
    plotter.view_xy()
    plotter.camera.zoom(1.2)

    plotter.subplot(0, 2)
    plotter.add_mesh(mesh_err, scalars='scalars', point_size=0, scalar_bar_args={'title': 'Abs. Error', 'vertical': False, 'title_font_size': 30, 'label_font_size': 24, 'position_x': 0.15, 'position_y': 0.02},
                     clim=[err_min, err_max], show_edges=True)
    plotter.view_xy()
    plotter.camera.zoom(1.2)

    for snapshot in range(T):
        gt_frame = z_gt[snapshot, :, var].cpu().numpy()
        net_frame = z_net[snapshot, :, var].cpu().numpy()
        err_frame = (gt_frame - net_frame)

        mesh_gt['scalars'] = gt_frame
        mesh_net['scalars'] = net_frame
        mesh_err['scalars'] = err_frame

        # (Opcional) actualizar texto dinámico
        plotter.subplot(0, 0)
        plotter.remove_actor('text_gt')
        plotter.add_text(f'Ground Truth, f={snapshot}', position='upper_edge', font_size=14, name='text_gt')
        plotter.subplot(0, 1)
        plotter.remove_actor('text_pred')
        plotter.add_text(f'Thermodynamics-informed MGN, f={str(snapshot)}', position='upper_edge', font_size=14, name='text_pred')
        plotter.subplot(0, 2)
        plotter.remove_actor('text_err')
        plotter.add_text(f'Absolute Error, f={str(snapshot)}', position='upper_edge', font_size=14, name='text_err')

        plotter.write_frame()

    # Close the plotter
    plotter.close()

def plotError(gt, z_net, L2_list, state_variables, dataset_dim, output_dir_exp):

    fig = plt.figure(figsize=(20, 20))

    for i, name in enumerate(L2_list):
        ax1 = fig.add_subplot(len(state_variables), 2, i * 2 + 1)
        ax1.set_title(name), ax1.grid()
        ax1.plot((gt[:, :, i]).sum((1)), linestyle='dotted', color='blue', label=f'{name} GT')
        ax1.plot((z_net.numpy()[:, :, i]).sum((1)), color='blue', label=f'{name} net')
        ax1.legend()

        ax2 = fig.add_subplot(len(state_variables), 2, i * 2 + 2)
        ax2.set_title('Error L2'), ax2.grid()
        ax2.plot(L2_list[name], color='blue', label=f'{name} GT')
    plt.savefig(os.path.join(output_dir_exp, 'L2_error.png'))



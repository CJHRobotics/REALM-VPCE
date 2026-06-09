"""
analysis_plots.py

Reusable plotting functions for VPCE place cell analysis.

Functions
---------
plot_place_cell_activations(...)
    Heatmap grid on a dark background with maze wall overlay.

plot_place_cell_activations_overlay(...)
    Same heatmaps blended onto a real base figure image of the maze.
"""

import os
import xml.etree.ElementTree as ET

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.interpolate import griddata

from realm_tools.place_cell_lib.place_cells import rbf_activations
from realm_tools.simulation_lib.environment_parser import parse_all_walls
from realm_tools.simulation_lib.start_position_generator import _outer_boundary


# ---------------------------------------------------------------------------
# Physical sizing constants (inches).
# All figure positions are derived from these so the layout scales correctly
# regardless of how many place cells are plotted.
# ---------------------------------------------------------------------------
_CELL_SIZE      = 4.0   # subplot width and height
_HEADER_H       = 5.9   # fixed header height: title + colorbar area
_TITLE_FROM_TOP = 0.85  # title baseline, measured from top of figure
_CBAR_GAP       = 2.5   # space above subplot grid (accommodates label + ticks)
_CBAR_BAR_H     = 0.75  # height of the colorbar bar itself


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _layout(nrows, ncols):
    """Return figure dimensions and normalised positions from physical sizes."""
    fig_w       = ncols * _CELL_SIZE
    fig_h       = nrows * _CELL_SIZE + _HEADER_H
    subplot_top = nrows * _CELL_SIZE / fig_h
    cbar_bot    = subplot_top + _CBAR_GAP   / fig_h
    cbar_h_n    = _CBAR_BAR_H / fig_h
    title_y     = 1.0 - _TITLE_FROM_TOP / fig_h
    return fig_w, fig_h, subplot_top, cbar_bot, cbar_h_n, title_y


def _interp_grid(x_obs, y_obs, margin=0.1, resolution=300):
    """Build the interpolation meshgrid over the observation extent."""
    xi = np.linspace(x_obs.min() - margin, x_obs.max() + margin, resolution)
    yi = np.linspace(y_obs.min() - margin, y_obs.max() + margin, resolution)
    Xi, Yi = np.meshgrid(xi, yi)
    return Xi, Yi, xi, yi


def _draw_maze(ax, xml_path):
    """Draw maze walls onto an existing Axes object."""
    root  = ET.parse(xml_path).getroot()
    walls = parse_all_walls(root)
    for w in walls:
        color = 'black' if w['wall_type'] == 'boundary' else 'dimgrey'
        ax.plot([w['x1'], w['x2']], [w['y1'], w['y2']],
                color=color, lw=2, solid_capstyle='round', zorder=3)


def _add_colorbar(fig, cbar_bot, cbar_h_n):
    """Add a shared horizontal inferno colorbar to fig."""
    cax  = fig.add_axes([0.15, cbar_bot, 0.7, cbar_h_n])
    sm   = ScalarMappable(cmap='inferno', norm=Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')
    cbar.set_label('Normalised Activation', fontsize=45, color='black', labelpad=12)
    cbar.ax.xaxis.set_tick_params(color='black', labelcolor='black', labelsize=39)


def _save_figure(fig, output_dir, filename):
    """Save fig to output_dir/filename, creating the directory if needed."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"Saved  →  {path}")


# rbf_activations is imported from realm_tools.place_cell_lib.place_cells
# and re-exported here for convenience so callers can import from one place.

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_place_cell_activations(
    activations,
    poses,
    maze_xml,
    maze='',
    method='',
    n_pca_components=0,
    save=False,
    output_dir='analysis/figures/place_cell_activity_plots/',
):
    """
    Plot place cell activation heatmaps on a dark background with maze walls.

    Each subplot shows the interpolated RBF activation of one place cell
    projected onto the x,y plane, superimposed on the maze boundary.

    Parameters
    ----------
    activations      : np.ndarray, shape (N, K) — from rbf_activations()
    poses            : np.ndarray, shape (N, 3) — columns: x, y, theta
    maze_xml         : str  — path to the maze XML file
    maze             : str  — maze name used in the figure title
    method           : str  — clustering method used in the figure title
    n_pca_components : int  — PCA components used in the figure title
    save             : bool — if True, save the figure to output_dir
    output_dir       : str  — directory for saved figures
    """
    n_cells = activations.shape[1]
    ncols   = int(np.ceil(np.sqrt(n_cells)))
    nrows   = int(np.ceil(n_cells / ncols))

    fig_w, fig_h, subplot_top, cbar_bot, cbar_h_n, title_y = _layout(nrows, ncols)

    x_obs = poses[:, 0]
    y_obs = poses[:, 1]
    Xi, Yi, xi, yi = _interp_grid(x_obs, y_obs)

    # Inferno with NaN → white so grid points outside the observation hull
    # don't produce black patches on the white background.
    cmap = plt.get_cmap('inferno').copy()
    cmap.set_bad('white')

    with plt.rc_context({'text.color': 'black', 'axes.labelcolor': 'black',
                         'xtick.color': 'black', 'ytick.color': 'black'}):
        fig, axes = plt.subplots(nrows, ncols,
                                  figsize=(fig_w, fig_h),
                                  facecolor='white')
    axes = axes.flatten()

    for i in range(n_cells):
        ax = axes[i]
        ax.set_facecolor('white')   # override any dark-theme axes default
        Zi = griddata((x_obs, y_obs), activations[:, i], (Xi, Yi), method='linear')
        ax.imshow(Zi, extent=[xi.min(), xi.max(), yi.min(), yi.max()],
                  origin='lower', cmap=cmap,
                  vmin=0, vmax=activations[:, i].max(),
                  aspect='equal', interpolation='bilinear')
        _draw_maze(ax, maze_xml)
        ax.set_title(f'Cell {i}', fontsize=39, color='black', pad=4)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')

    for j in range(n_cells, len(axes)):
        axes[j].set_visible(False)

    plt.subplots_adjust(top=subplot_top, bottom=0.01, hspace=0.35, wspace=0.2)

    pca_str = f'  PCA={n_pca_components}' if n_pca_components else ''
    fig.suptitle(
        f'Place Cell Activations — {maze.upper()}  {method.upper()}  K={n_cells}{pca_str}',
        fontsize=54, color='black', y=title_y
    )

    _add_colorbar(fig, cbar_bot, cbar_h_n)

    if save:
        _save_figure(fig, output_dir,
                     f'{maze}_{method}_K{n_cells}_activations.png')


def plot_place_cell_activations_overlay(
    activations,
    poses,
    maze_xml,
    base_img_path,
    maze='',
    method='',
    flip_base_img=True,
    save=False,
    output_dir='analysis/figures/place_cell_activity_plots/',
):
    """
    Plot place cell activation heatmaps blended onto a real base figure image.

    The heatmap uses per-pixel alpha (alpha = normalised activation), so low-
    activation regions are transparent and the base image shows through.

    Parameters
    ----------
    activations    : np.ndarray, shape (N, K) — from rbf_activations()
    poses          : np.ndarray, shape (N, 3) — columns: x, y, theta
    maze_xml       : str  — path to the maze XML file (used for world extent)
    base_img_path  : str  — path to the base figure image
    maze           : str  — maze name used in the figure title
    method         : str  — clustering method used in the figure title
    flip_base_img  : bool — flip image vertically (True for Webots screenshots)
    save           : bool — if True, save the figure to output_dir
    output_dir     : str  — directory for saved figures
    """
    n_cells = activations.shape[1]
    ncols   = int(np.ceil(np.sqrt(n_cells)))
    nrows   = int(np.ceil(n_cells / ncols))

    fig_w, fig_h, subplot_top, cbar_bot, cbar_h_n, title_y = _layout(nrows, ncols)

    # World extent from boundary wall vertices
    walls    = parse_all_walls(ET.parse(maze_xml).getroot())
    boundary = _outer_boundary(walls)
    bminx, bminy, bmaxx, bmaxy = boundary.bounds
    base_extent = [bminx, bmaxx, bminy, bmaxy]

    base_img = plt.imread(base_img_path)
    if flip_base_img:
        base_img = np.flipud(base_img)

    x_obs = poses[:, 0]
    y_obs = poses[:, 1]
    Xi, Yi, xi, yi = _interp_grid(x_obs, y_obs)
    cmap = plt.get_cmap('inferno')

    with plt.rc_context({'text.color': 'black', 'axes.labelcolor': 'black',
                         'xtick.color': 'black', 'ytick.color': 'black'}):
        fig2, axes2 = plt.subplots(nrows, ncols,
                                    figsize=(fig_w, fig_h),
                                    facecolor='white')
    axes2 = axes2.flatten()

    for i in range(n_cells):
        ax = axes2[i]
        ax.set_facecolor('white')
        ax.imshow(base_img, extent=base_extent, origin='lower',
                  aspect='equal', zorder=0)

        Zi       = griddata((x_obs, y_obs), activations[:, i], (Xi, Yi), method='linear')
        Zi_clean = np.nan_to_num(Zi, nan=0.0)
        peak     = Zi_clean.max()
        Zi_norm  = Zi_clean / peak if peak > 0 else Zi_clean
        rgba     = cmap(Zi_norm)
        rgba[:, :, 3] = Zi_norm

        ax.imshow(rgba, extent=[xi.min(), xi.max(), yi.min(), yi.max()],
                  origin='lower', aspect='equal', interpolation='bilinear', zorder=1)
        ax.set_xlim(base_extent[0], base_extent[1])
        ax.set_ylim(base_extent[2], base_extent[3])
        ax.set_title(f'Cell {i}', fontsize=39, color='black', pad=4)
        ax.set_xticks([])
        ax.set_yticks([])

    for j in range(n_cells, len(axes2)):
        axes2[j].set_visible(False)

    plt.subplots_adjust(top=subplot_top, bottom=0.01, hspace=0.35, wspace=0.2)

    fig2.suptitle(
        f'Place Cell Activations (base overlay) — {maze.upper()}  {method.upper()}  K={n_cells}',
        fontsize=54, color='black', y=title_y
    )

    _add_colorbar(fig2, cbar_bot, cbar_h_n)

    if save:
        _save_figure(fig2, output_dir,
                     f'{maze}_{method}_K{n_cells}_activations_overlay.png')

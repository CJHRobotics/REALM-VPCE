"""Compare place-field banks built from three feature subsets:
    - full     : hog + color_hist + spatial + lidar
    - lidar    : lidar only
    - visual   : hog + color_hist + spatial (no lidar)

Each is fed through the same agglomerative pipeline (n_combine = 8,
combine_mode = 'concat' -> full 360 POV) and the resulting fields are
projected onto the environment in three side-by-side subplots.

Usage:
    python compare_feature_sets.py <env_name>

<env_name> defaults to circ_lm8_r0.
"""

import os
import sys
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, '/Users/titonka/REALM-VPCE')
sys.path.insert(0, os.path.dirname(__file__))

from realm_tools.experiment_lib.place_cell_analysis import load_selective_features
from pipeline import run_place_field_pipeline, build_env

# ------------------------------------------------------------------ CLI / paths
ENV = sys.argv[1] if len(sys.argv) > 1 else 'circ_lm8_r0'
REPO       = '/Users/titonka/REALM-VPCE'
DATA_PATH  = f'{REPO}/data/vpce/collect_data/{ENV}.h5'
XML_PATH   = f'{REPO}/simulation/worlds/environments/vpce/{ENV}.xml'
BANK_ROOT  = f'{REPO}/data_cache/feature_selection_bank/{ENV}'
FIG_DIR    = f'{REPO}/analysis/experiment_feature_selection/figures'
PLOT_PATH  = f'{FIG_DIR}/{ENV}__compare.png'
os.makedirs(FIG_DIR, exist_ok=True)

CONFIGS = [
    dict(name='full',
         label='hog + color + spatial + lidar',
         flags=dict(use_hog=True,  use_color_hist=True,  use_spatial=True,  use_lidar=True)),
    dict(name='lidar',
         label='lidar only',
         flags=dict(use_hog=False, use_color_hist=False, use_spatial=False, use_lidar=True)),
    dict(name='visual',
         label='hog + color + spatial',
         flags=dict(use_hog=True,  use_color_hist=True,  use_spatial=True,  use_lidar=False)),
]

# ------------------------------------------------------------------ env geometry
tree_root = ET.parse(XML_PATH).getroot()

# ------------------------------------------------------------------ run each config
banks = {}
for cfg in CONFIGS:
    print(f'\n===== {cfg["name"].upper()}  ({cfg["label"]}) =====')
    features, xy = load_selective_features(
        DATA_PATH, n_combine=8, n_orientations=8, combine_mode='concat',
        **cfg['flags'],
    )
    env = build_env(xy, tree_root)
    bank_df, kept_mu = run_place_field_pipeline(features, xy, env, tag=cfg['name'])

    out_dir = f'{BANK_ROOT}/{cfg["name"]}'
    os.makedirs(out_dir, exist_ok=True)
    bank_df.to_csv(f'{out_dir}/bank.csv', index=False)
    np.save(f'{out_dir}/bank_mu.npy', kept_mu.astype(np.float32))
    banks[cfg['name']] = (bank_df, env)
    print(f'  bank saved: {len(bank_df)} fields -> {out_dir}')

# ------------------------------------------------------------------ side-by-side plot
walls_xml = tree_root.findall('wall')
landmarks_xml = tree_root.findall('landmark')
walls = [(float(w.get('x1')), float(w.get('y1')),
          float(w.get('x2')), float(w.get('y2'))) for w in walls_xml]
landmarks = [dict(x=float(lm.get('x')), y=float(lm.get('y')),
                  theta=float(lm.get('theta', 0)), width=float(lm.get('width', 0.25)),
                  color=(float(lm.get('red')), float(lm.get('green')), float(lm.get('blue'))))
             for lm in landmarks_xml]

first_env = banks[CONFIGS[0]['name']][1]
aspect = (first_env['x_max'] - first_env['x_min']) / max(first_env['y_max'] - first_env['y_min'], 1e-6)
per_w  = 6 if aspect < 2 else 10
per_h  = max(4.5, per_w / aspect)

fig, axes = plt.subplots(1, len(CONFIGS), figsize=(per_w * len(CONFIGS), per_h),
                         squeeze=False)
axes = axes[0]

# shared radius normalisation across the three panels so colours mean the same thing
r_all = np.concatenate([b['radius_env_m'].to_numpy() for b, _ in banks.values()])
r_all = r_all[r_all > 0]
cmap  = plt.get_cmap('plasma')
norm  = plt.Normalize(vmin=r_all.min(), vmax=r_all.max())

for ax, cfg in zip(axes, CONFIGS):
    bank_df, env = banks[cfg['name']]
    for (x1, y1, x2, y2) in walls:
        ax.plot([x1, x2], [y1, y2], color='black', lw=2, solid_capstyle='round', zorder=6)
    if env['is_circular']:
        th = np.linspace(0, 2*np.pi, 200)
        ax.plot(env['env_cx'] + env['env_R']*np.cos(th),
                env['env_cy'] + env['env_R']*np.sin(th),
                color='black', lw=2, zorder=6)
    for lm in landmarks:
        tx, ty = -np.sin(lm['theta']), np.cos(lm['theta']); hw = lm['width'] / 2
        ax.plot([lm['x'] - hw*tx, lm['x'] + hw*tx],
                [lm['y'] - hw*ty, lm['y'] + hw*ty],
                color=lm['color'], lw=4, solid_capstyle='butt', zorder=7)

    for _, row in bank_df.iterrows():
        color = cmap(norm(row['radius_env_m']))
        ax.add_patch(plt.Circle((row['centroid_x'], row['centroid_y']), row['radius_env_m'],
                                facecolor=color, edgecolor=color, alpha=0.18, lw=1.0, zorder=3))
        ax.add_patch(plt.Circle((row['centroid_x'], row['centroid_y']), row['radius_env_m'],
                                facecolor='none', edgecolor=color, alpha=0.75, lw=1.0, zorder=4))
        ax.plot(row['centroid_x'], row['centroid_y'], marker='+', color=color,
                markersize=6, markeredgewidth=1.2, zorder=5)

    ax.set_xlim(env['x_min'] - 0.4, env['x_max'] + 0.4)
    ax.set_ylim(env['y_min'] - 0.4, env['y_max'] + 0.4)
    ax.set_aspect('equal')
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    ax.set_title(f'{cfg["label"]}\n{len(bank_df)} fields', fontsize=12)
    ax.grid(True, alpha=0.15)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
cbar = fig.colorbar(sm, ax=axes, shrink=0.85, pad=0.02, location='right')
cbar.set_label('field radius (m)')

fig.suptitle(f'{ENV}  |  place fields by feature subset', fontsize=14, y=1.02)
plt.savefig(PLOT_PATH, dpi=140, bbox_inches='tight', facecolor='white')
print(f'\nComparison plot -> {PLOT_PATH}')

"""Place-field pipeline with per-feature toggles.

Choose any subset of {hog, color_hist, spatial, lidar} to feed into the
agglomerative clustering. Everything downstream (Ward linkage, sigma
percentile, environment projection, filters) is identical across choices,
so plots are directly comparable.

Usage
-----
    python run_place_fields.py <env_name>
        [--hog]       [--no-hog]
        [--color]     [--no-color]
        [--spatial]   [--no-spatial]
        [--lidar]     [--no-lidar]
        [--tag <label>]

Defaults: all visual blocks ON, lidar OFF. `<env_name>` defaults to
circ_lm8_r0.

Outputs are namespaced by the enabled feature set (and optional --tag)
so multiple runs on the same environment don't overwrite each other.
"""

import argparse
import os
import sys
import time
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

# Derive repo root from this file's location so imports work on any host.
# This file lives at <repo>/analysis/experiment_feature_selection/run_place_fields.py.
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from realm_tools.experiment_lib.place_cell_analysis import load_selective_features

import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# ------------------------------------------------------------------ CLI
p = argparse.ArgumentParser()
p.add_argument('env', nargs='?', default='circ_lm8_r0')
p.add_argument('--hog',     dest='use_hog',        action='store_true',  default=True)
p.add_argument('--no-hog',  dest='use_hog',        action='store_false')
p.add_argument('--color',   dest='use_color_hist', action='store_true',  default=True)
p.add_argument('--no-color',dest='use_color_hist', action='store_false')
p.add_argument('--spatial', dest='use_spatial',    action='store_true',  default=True)
p.add_argument('--no-spatial', dest='use_spatial', action='store_false')
p.add_argument('--lidar',   dest='use_lidar',      action='store_true',  default=False)
p.add_argument('--no-lidar',dest='use_lidar',      action='store_false')
p.add_argument('--tag',     default='')
args = p.parse_args()

ENV = args.env
FEATURE_FLAGS = dict(
    use_hog        = args.use_hog,
    use_color_hist = args.use_color_hist,
    use_spatial    = args.use_spatial,
    use_lidar      = args.use_lidar,
)
enabled = [k.replace('use_', '') for k, v in FEATURE_FLAGS.items() if v]
FEATURE_LABEL = ('+'.join(enabled) if enabled else 'none') + (f'__{args.tag}' if args.tag else '')

DATA_PATH  = f'{REPO}/data/vpce/collect_data/{ENV}.h5'
XML_PATH   = f'{REPO}/simulation/worlds/environments/vpce/{ENV}.xml'
BANK_DIR   = f'{REPO}/data_cache/feature_selection_bank/{ENV}/{FEATURE_LABEL}'
FIG_DIR    = f'{REPO}/analysis/experiment_feature_selection/figures'
PLOT_PATH  = f'{FIG_DIR}/{ENV}__{FEATURE_LABEL}.png'

# ------------------------------------------------------------------ pruning knobs
SIGMA_PCTL       = 90
ACT_THRESH       = 0.5
BIN_M            = 0.05
MIN_MEMBERS      = 8
MIN_RADIUS_BINS  = 2
ARENA_FRAC_CAP   = 0.20
SCALE_GAP_MIN    = 2.0
R_WALL_MAX       = 0.20
R_CENTER_MAX     = None
R_WALL_PREF      = 0.15
R_CENTER_PREF    = None

os.makedirs(BANK_DIR, exist_ok=True)
os.makedirs(FIG_DIR,  exist_ok=True)
print(f'ENV           = {ENV}')
print(f'FEATURE_LABEL = {FEATURE_LABEL}')
print(f'BANK          = {BANK_DIR}')

# ------------------------------------------------------------------ data
print('Loading features ...')
features, xy = load_selective_features(DATA_PATH, **FEATURE_FLAGS)
N, D = features.shape
print(f'  {N} locations, D = {D}   (blocks: {enabled or "none"})')

X_tr, xy_tr = features, xy
X_ev, xy_ev = features, xy

# ------------------------------------------------------------------ env geometry
tree_root  = ET.parse(XML_PATH).getroot()
walls_xml  = tree_root.findall('wall')
cwalls_xml = tree_root.findall('circular_wall')
landmarks_xml = tree_root.findall('landmark')

is_circular = len(cwalls_xml) > 0
if is_circular:
    cw = cwalls_xml[0]
    env_cx = float(cw.get('x', 0.0)); env_cy = float(cw.get('y', 0.0))
    env_R  = float(cw.get('radius'))
    env_area = float(np.pi * env_R ** 2)
    long_dim = 2 * env_R
    x_min, x_max = env_cx - env_R, env_cx + env_R
    y_min, y_max = env_cy - env_R, env_cy + env_R
    print(f'  circular env : R = {env_R:.2f} m   area = {env_area:.2f} m^2')
else:
    x_min, x_max = xy[:, 0].min(), xy[:, 0].max()
    y_min, y_max = xy[:, 1].min(), xy[:, 1].max()
    env_w, env_h = x_max - x_min, y_max - y_min
    env_area = env_w * env_h; long_dim = max(env_w, env_h)
    print(f'  rectangular env : {env_w:.2f} x {env_h:.2f} m   area = {env_area:.2f} m^2')

cap_area = ARENA_FRAC_CAP * env_area
cap_r    = np.sqrt(cap_area / np.pi)
print(f'  cap : area <= {cap_area:.2f} m^2   (r <= {cap_r:.2f} m)')

GRID_BINS_X = max(2, int(np.ceil((x_max - x_min) / BIN_M)))
GRID_BINS_Y = max(2, int(np.ceil((y_max - y_min) / BIN_M)))

# ------------------------------------------------------------------ pairwise distances
print('Computing pairwise distances ...')
t0 = time.time()
Xtr32 = X_tr.astype(np.float32, copy=False)
sq    = np.sum(Xtr32 ** 2, axis=1)
D2    = sq[:, None] + sq[None, :] - 2.0 * (Xtr32 @ Xtr32.T)
np.maximum(D2, 0, out=D2)
print(f'  D2 {D2.shape} float32 ({D2.nbytes/1e6:.0f} MB)   {time.time()-t0:.1f}s')

# ------------------------------------------------------------------ Ward linkage
print(f'Running Ward linkage on {N} leaves ...')
t0 = time.time()
D_condensed = squareform(np.sqrt(D2).astype(np.float64), checks=False)
Z = linkage(D_condensed, method='ward')
del D_condensed
print(f'  linkage done in {time.time()-t0:.1f}s   Z shape {Z.shape}')

n_nodes  = 2 * N - 1
parent   = [None] * n_nodes
children = [None] * n_nodes
depth    = [0] * n_nodes
count    = [1] * n_nodes
for merge_idx in range(N - 1):
    a, b = int(Z[merge_idx, 0]), int(Z[merge_idx, 1])
    new_id = N + merge_idx
    parent[a] = new_id; parent[b] = new_id
    children[new_id] = (a, b)
    depth[new_id]    = 1 + max(depth[a], depth[b])
    count[new_id]    = count[a] + count[b]

candidate_ids = [nid for nid in range(n_nodes) if count[nid] >= MIN_MEMBERS]
print(f'Candidates after member-count floor (>= {MIN_MEMBERS}): {len(candidate_ids)} / {n_nodes}')

def collect_leaves(nid):
    stack = [nid]; leaves = []
    while stack:
        n = stack.pop()
        if n < N: leaves.append(n)
        else:
            a, b = children[n]; stack.append(a); stack.append(b)
    return np.asarray(leaves, dtype=np.int64)

print('Computing mu, sigma for candidates ...')
t0 = time.time()
cand_mu    = np.empty((len(candidate_ids), D), dtype=np.float32)
cand_sigma = np.empty(len(candidate_ids), dtype=np.float64)
for k, nid in enumerate(candidate_ids):
    m = collect_leaves(nid)
    cand_mu[k] = X_tr[m].mean(axis=0)
    sub = D2[np.ix_(m, m)]
    iu  = np.triu_indices(len(m), k=1)
    cand_sigma[k] = float(np.percentile(np.sqrt(sub[iu]), SIGMA_PCTL))
    if (k + 1) % max(1, len(candidate_ids) // 10) == 0:
        print(f'  {k+1}/{len(candidate_ids)}   elapsed {time.time()-t0:.1f}s')

x_edges = np.linspace(x_min, x_max, GRID_BINS_X + 1)
y_edges = np.linspace(y_min, y_max, GRID_BINS_Y + 1)
bin_area = (x_edges[1] - x_edges[0]) * (y_edges[1] - y_edges[0])
ix_ev = np.clip(np.searchsorted(x_edges, xy_ev[:, 0], side='right') - 1, 0, GRID_BINS_X - 1)
iy_ev = np.clip(np.searchsorted(y_edges, xy_ev[:, 1], side='right') - 1, 0, GRID_BINS_Y - 1)
xc = 0.5 * (x_edges[:-1] + x_edges[1:])
yc = 0.5 * (y_edges[:-1] + y_edges[1:])
print(f'Env grid: {GRID_BINS_X} x {GRID_BINS_Y}   bin_area = {bin_area*1e4:.2f} cm^2')

Xe    = X_ev.astype(np.float64, copy=False)
Xe_sq = (Xe ** 2).sum(axis=1)
n_cand = len(candidate_ids)
cand_area   = np.zeros(n_cand)
cand_radius = np.zeros(n_cand)
cand_cx     = np.zeros(n_cand)
cand_cy     = np.zeros(n_cand)

print('Projecting candidates into environment ...')
t0 = time.time()
BATCH = 128
for start in range(0, n_cand, BATCH):
    stop = min(start + BATCH, n_cand)
    M    = cand_mu[start:stop].astype(np.float64)
    S    = cand_sigma[start:stop]
    M_sq = (M ** 2).sum(axis=1)
    d2   = Xe_sq[:, None] - 2.0 * (Xe @ M.T) + M_sq[None, :]
    np.maximum(d2, 0, out=d2)
    S_safe = np.where(S > 0, S, 1.0)
    raw    = np.exp(-d2 / (2.0 * S_safe[None, :] ** 2))

    for j in range(stop - start):
        k    = start + j
        grid = np.zeros((GRID_BINS_X, GRID_BINS_Y))
        np.maximum.at(grid, (ix_ev, iy_ev), raw[:, j])
        supra = grid >= ACT_THRESH
        n_bins = int(supra.sum())
        cand_area[k]   = n_bins * bin_area
        cand_radius[k] = np.sqrt(cand_area[k] / np.pi) if n_bins else 0.0
        pf             = int(np.argmax(grid))
        pi_idx, pj_idx = np.unravel_index(pf, (GRID_BINS_X, GRID_BINS_Y))
        cand_cx[k]     = xc[pi_idx]; cand_cy[k] = yc[pj_idx]
    if ((start // BATCH) + 1) % 20 == 0 or stop == n_cand:
        print(f'  env readout {stop}/{n_cand}   elapsed {time.time()-t0:.1f}s')

# ------------------------------------------------------------------ filters
min_r_env = MIN_RADIUS_BINS * np.sqrt(bin_area / np.pi)
pass_radius = cand_radius >= min_r_env
print(f'  after radius floor : {int(pass_radius.sum())} / {n_cand}')

if is_circular:
    dist_bnd = env_R - np.hypot(cand_cx - env_cx, cand_cy - env_cy); max_dist = env_R
else:
    dist_bnd = np.minimum.reduce([cand_cx - x_min, x_max - cand_cx,
                                  cand_cy - y_min, y_max - cand_cy])
    max_dist = min((x_max - x_min) / 2, (y_max - y_min) / 2)
dist_bnd = np.clip(dist_bnd, 0.0, None)

r_center_max = R_CENTER_MAX if R_CENTER_MAX is not None else cap_r
r_max_loc    = R_WALL_MAX + (r_center_max - R_WALL_MAX) * (dist_bnd / max_dist)
pass_local   = cand_radius <= r_max_loc
print(f'  after local cap    : {int((pass_radius & pass_local).sum())} / {n_cand}')

pass_scale = pass_radius & pass_local & (cand_area <= cap_area)
print(f'  after global cap   : {int(pass_scale.sum())} / {n_cand}')

surviving = np.where(pass_scale)[0]
r_center_pref = R_CENTER_PREF if R_CENTER_PREF is not None else cap_r
r_expected    = R_WALL_PREF + (r_center_pref - R_WALL_PREF) * (dist_bnd / max_dist)
scale_pref    = -np.abs(np.log(cand_radius / np.clip(r_expected, 1e-6, None)))
order         = surviving[np.argsort(-scale_pref[surviving])]

kept   = []
kept_c = np.empty((0, 2)); kept_r = np.empty((0,))
for k in order:
    r        = cand_radius[k]
    cx_, cy_ = cand_cx[k], cand_cy[k]
    if len(kept):
        d_c        = np.hypot(kept_c[:, 0] - cx_, kept_c[:, 1] - cy_)
        contained  = d_c < np.maximum(kept_r, r)
        scale_gap  = np.maximum(kept_r, r) / np.minimum(kept_r, r)
        similar    = scale_gap < SCALE_GAP_MIN
        if np.any(contained & similar):
            continue
    kept.append(k)
    kept_c = np.vstack([kept_c, [cx_, cy_]]); kept_r = np.append(kept_r, r)
print(f'  after dedup        : {len(kept)} / {n_cand}')

# ------------------------------------------------------------------ bank
rows = []
for k in sorted(kept, key=lambda kk: cand_radius[kk]):
    nid = candidate_ids[k]; ch = children[nid]; child_ids = list(ch) if ch is not None else []
    rows.append({
        'node_id': nid, 'depth': depth[nid], 'n_members': int(count[nid]),
        'sigma_feature': float(cand_sigma[k]),
        'area_env_m2': float(cand_area[k]),
        'radius_env_m': float(cand_radius[k]),
        'centroid_x': float(cand_cx[k]), 'centroid_y': float(cand_cy[k]),
        'parent_id': parent[nid] if parent[nid] is not None else -1,
        'child_ids': ','.join(str(c) for c in child_ids),
        'is_leaf': nid < N,
    })
bank_df = pd.DataFrame(rows).sort_values('radius_env_m').reset_index(drop=True)
bank_df.to_csv(os.path.join(BANK_DIR, 'bank.csv'), index=False)
mu_out = np.stack([cand_mu[k] for k in sorted(kept, key=lambda kk: cand_radius[kk])])
np.save(os.path.join(BANK_DIR, 'bank_mu.npy'), mu_out.astype(np.float32))
print(f'Bank saved: {len(bank_df)} place fields -> {BANK_DIR}')

# ------------------------------------------------------------------ plot
walls = [(float(w.get('x1')), float(w.get('y1')),
          float(w.get('x2')), float(w.get('y2'))) for w in walls_xml]
landmarks = [dict(x=float(lm.get('x')), y=float(lm.get('y')),
                  theta=float(lm.get('theta', 0)), width=float(lm.get('width', 0.25)),
                  color=(float(lm.get('red')), float(lm.get('green')), float(lm.get('blue'))))
             for lm in landmarks_xml]

aspect = (x_max - x_min) / max(y_max - y_min, 1e-6)
fig_w  = 12 if aspect < 2 else 16
fig_h  = max(4.5, fig_w / aspect)
fig, ax = plt.subplots(figsize=(fig_w, fig_h))
for (x1, y1, x2, y2) in walls:
    ax.plot([x1, x2], [y1, y2], color='black', lw=2, solid_capstyle='round', zorder=6)
if is_circular:
    th = np.linspace(0, 2*np.pi, 200)
    ax.plot(env_cx + env_R*np.cos(th), env_cy + env_R*np.sin(th), color='black', lw=2, zorder=6)
for lm in landmarks:
    tx, ty = -np.sin(lm['theta']), np.cos(lm['theta']); hw = lm['width'] / 2
    ax.plot([lm['x'] - hw*tx, lm['x'] + hw*tx],
            [lm['y'] - hw*ty, lm['y'] + hw*ty],
            color=lm['color'], lw=4, solid_capstyle='butt', zorder=7)

r_arr = bank_df['radius_env_m'].to_numpy()
cmap  = plt.get_cmap('plasma'); norm = plt.Normalize(vmin=r_arr.min(), vmax=r_arr.max())
for _, row in bank_df.iterrows():
    color = cmap(norm(row['radius_env_m']))
    ax.add_patch(plt.Circle((row['centroid_x'], row['centroid_y']), row['radius_env_m'],
                            facecolor=color, edgecolor=color, alpha=0.18, lw=1.0, zorder=3))
    ax.add_patch(plt.Circle((row['centroid_x'], row['centroid_y']), row['radius_env_m'],
                            facecolor='none', edgecolor=color, alpha=0.75, lw=1.0, zorder=4))
    ax.plot(row['centroid_x'], row['centroid_y'], marker='+', color=color,
            markersize=6, markeredgewidth=1.2, zorder=5)

ax.set_xlim(x_min - 0.4, x_max + 0.4); ax.set_ylim(y_min - 0.4, y_max + 0.4)
ax.set_aspect('equal'); ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
ax.set_title(f'{ENV}  |  place fields  ({FEATURE_LABEL})', fontsize=13)
ax.grid(True, alpha=0.15)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, shrink=0.85, pad=0.02); cbar.set_label('field radius (m)')

plt.tight_layout()
plt.savefig(PLOT_PATH, dpi=140, bbox_inches='tight', facecolor='white')
print(f'Plot -> {PLOT_PATH}')

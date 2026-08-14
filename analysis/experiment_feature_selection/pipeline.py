"""Reusable place-field pipeline used by both single-run and comparison scripts.

Given a feature matrix + xy per location + environment metadata + a config,
run: pairwise distances -> Ward linkage -> per-node mu/sigma -> environment
projection -> viability / local cap / global cap / dedup, and return a
bank DataFrame plus the surviving mu vectors.
"""

import time
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from scipy.sparse import csr_matrix


DEFAULT_CFG = dict(
    SIGMA_PCTL       = 90,
    ACT_THRESH       = 0.5,
    BIN_M            = 0.05,
    MIN_MEMBERS      = 8,
    MIN_RADIUS_BINS  = 2,
    ARENA_FRAC_CAP   = 0.20,
    SCALE_GAP_MIN    = 2.0,
    # Same-scale spacing: two peers of similar scale must have their
    # centres at least SAME_SCALE_SEPARATION * (r_a + r_b) apart. 1.0 =
    # discs don't touch; 0.5 = discs can overlap by up to ~50 % of the
    # smaller. Cross-scale (radii differ by >= SCALE_GAP_MIN) is
    # unaffected — nesting stays allowed.
    SAME_SCALE_SEPARATION = 0.9,
    # Optional wall-based prior: local hard cap (r_max depends on distance
    # to boundary) + preference-based dedup ordering that prefers
    # r_expected(loc). Turned off by default so what emerges is driven
    # purely by the feature geometry.
    USE_WALL_PRIOR   = False,
    R_WALL_MAX       = 0.20,
    R_CENTER_MAX     = None,
    R_WALL_PREF      = 0.15,
    R_CENTER_PREF    = None,
    # Dedup mode.
    #   'containment' — the original containment-with-scale-gap rule
    #                   (SCALE_GAP_MIN + SAME_SCALE_SEPARATION apply).
    #   'coverage'    — weighted set-cover: greedy pick by marginal
    #                   coverage gain. Each env bin may be covered by up
    #                   to COVERAGE_MAX_K fields; picking stops when the
    #                   best marginal gain falls below
    #                   COVERAGE_MIN_GAIN_FRAC * (first pick's gain).
    DEDUP_MODE            = 'containment',
    COVERAGE_MAX_K        = 2,
    COVERAGE_MIN_GAIN_FRAC = 0.01,
)


def build_env(xy, xml_tree_root):
    """Extract environment geometry from an XML root; return a dict."""
    cwalls = xml_tree_root.findall('circular_wall')
    walls  = xml_tree_root.findall('wall')
    landmarks = xml_tree_root.findall('landmark')
    env = {'walls_xml': walls, 'landmarks_xml': landmarks}
    if cwalls:
        cw = cwalls[0]
        env['is_circular'] = True
        env['env_cx'] = float(cw.get('x', 0.0))
        env['env_cy'] = float(cw.get('y', 0.0))
        env['env_R']  = float(cw.get('radius'))
        env['env_area'] = float(np.pi * env['env_R'] ** 2)
        env['long_dim'] = 2 * env['env_R']
        env['x_min']  = env['env_cx'] - env['env_R']
        env['x_max']  = env['env_cx'] + env['env_R']
        env['y_min']  = env['env_cy'] - env['env_R']
        env['y_max']  = env['env_cy'] + env['env_R']
    else:
        env['is_circular'] = False
        env['x_min'], env['x_max'] = float(xy[:, 0].min()), float(xy[:, 0].max())
        env['y_min'], env['y_max'] = float(xy[:, 1].min()), float(xy[:, 1].max())
        env['env_area'] = (env['x_max'] - env['x_min']) * (env['y_max'] - env['y_min'])
        env['long_dim'] = max(env['x_max'] - env['x_min'], env['y_max'] - env['y_min'])
    return env


def run_place_field_pipeline(features, xy, env, cfg=None, tag='', verbose=True):
    """Run the pipeline and return (bank_df, kept_mu, cand_mu, cand_sigma, aux)."""
    C = dict(DEFAULT_CFG); C.update(cfg or {})
    X_tr, X_ev = features, features
    xy_tr, xy_ev = xy, xy
    N, D = X_tr.shape

    cap_area = C['ARENA_FRAC_CAP'] * env['env_area']
    cap_r    = np.sqrt(cap_area / np.pi)
    if verbose:
        print(f'[{tag}] N={N} D={D}  env_area={env["env_area"]:.2f} m^2  cap_r={cap_r:.2f} m')

    GRID_BINS_X = max(2, int(np.ceil((env['x_max'] - env['x_min']) / C['BIN_M'])))
    GRID_BINS_Y = max(2, int(np.ceil((env['y_max'] - env['y_min']) / C['BIN_M'])))

    # --- pairwise distances -------------------------------------------------
    Xtr32 = X_tr.astype(np.float32, copy=False)
    sq    = np.sum(Xtr32 ** 2, axis=1)
    D2    = sq[:, None] + sq[None, :] - 2.0 * (Xtr32 @ Xtr32.T)
    np.maximum(D2, 0, out=D2)

    # --- Ward linkage -------------------------------------------------------
    t0 = time.time()
    D_condensed = squareform(np.sqrt(D2).astype(np.float64), checks=False)
    Z = linkage(D_condensed, method='ward'); del D_condensed
    if verbose:
        print(f'[{tag}] linkage {time.time()-t0:.1f}s')

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

    # --- candidate pre-filter -----------------------------------------------
    candidate_ids = [nid for nid in range(n_nodes) if count[nid] >= C['MIN_MEMBERS']]
    if verbose:
        print(f'[{tag}] candidates (>= {C["MIN_MEMBERS"]} members): '
              f'{len(candidate_ids)} / {n_nodes}')

    def collect_leaves(nid):
        stack, leaves = [nid], []
        while stack:
            n = stack.pop()
            if n < N: leaves.append(n)
            else:    a, b = children[n]; stack.append(a); stack.append(b)
        return np.asarray(leaves, dtype=np.int64)

    # --- per-candidate mu, sigma --------------------------------------------
    cand_mu    = np.empty((len(candidate_ids), D), dtype=np.float32)
    cand_sigma = np.empty(len(candidate_ids), dtype=np.float64)
    for k, nid in enumerate(candidate_ids):
        m = collect_leaves(nid)
        cand_mu[k] = X_tr[m].mean(axis=0)
        sub = D2[np.ix_(m, m)]
        iu  = np.triu_indices(len(m), k=1)
        cand_sigma[k] = float(np.percentile(np.sqrt(sub[iu]), C['SIGMA_PCTL']))

    # --- env readout --------------------------------------------------------
    x_edges = np.linspace(env['x_min'], env['x_max'], GRID_BINS_X + 1)
    y_edges = np.linspace(env['y_min'], env['y_max'], GRID_BINS_Y + 1)
    bin_area = (x_edges[1] - x_edges[0]) * (y_edges[1] - y_edges[0])
    ix_ev = np.clip(np.searchsorted(x_edges, xy_ev[:, 0], side='right') - 1, 0, GRID_BINS_X - 1)
    iy_ev = np.clip(np.searchsorted(y_edges, xy_ev[:, 1], side='right') - 1, 0, GRID_BINS_Y - 1)
    xc = 0.5 * (x_edges[:-1] + x_edges[1:])
    yc = 0.5 * (y_edges[:-1] + y_edges[1:])

    Xe    = X_ev.astype(np.float64, copy=False)
    Xe_sq = (Xe ** 2).sum(axis=1)
    n_cand = len(candidate_ids)
    cand_area   = np.zeros(n_cand)
    cand_radius = np.zeros(n_cand)
    cand_cx     = np.zeros(n_cand); cand_cy = np.zeros(n_cand)
    # Sparse mask storage: flat bin indices where each field's response
    # is above threshold. Needed by the coverage-based dedup mode.
    cand_mask_ix = [None] * n_cand

    BATCH = 128
    for start in range(0, n_cand, BATCH):
        stop = min(start + BATCH, n_cand)
        M = cand_mu[start:stop].astype(np.float64)
        S = cand_sigma[start:stop]
        M_sq = (M ** 2).sum(axis=1)
        d2 = Xe_sq[:, None] - 2.0 * (Xe @ M.T) + M_sq[None, :]
        np.maximum(d2, 0, out=d2)
        S_safe = np.where(S > 0, S, 1.0)
        raw = np.exp(-d2 / (2.0 * S_safe[None, :] ** 2))
        for j in range(stop - start):
            k = start + j
            grid = np.zeros((GRID_BINS_X, GRID_BINS_Y))
            np.maximum.at(grid, (ix_ev, iy_ev), raw[:, j])
            supra = grid >= C['ACT_THRESH']
            n_bins = int(supra.sum())
            cand_area[k]   = n_bins * bin_area
            cand_radius[k] = np.sqrt(cand_area[k] / np.pi) if n_bins else 0.0
            pf = int(np.argmax(grid))
            pi_idx, pj_idx = np.unravel_index(pf, (GRID_BINS_X, GRID_BINS_Y))
            cand_cx[k] = xc[pi_idx]; cand_cy[k] = yc[pj_idx]
            cand_mask_ix[k] = np.flatnonzero(supra.ravel()).astype(np.int32)

    # --- filters ------------------------------------------------------------
    min_r_env = C['MIN_RADIUS_BINS'] * np.sqrt(bin_area / np.pi)
    pass_radius = cand_radius >= min_r_env

    if C['USE_WALL_PRIOR']:
        if env['is_circular']:
            dist_bnd = env['env_R'] - np.hypot(cand_cx - env['env_cx'], cand_cy - env['env_cy'])
            max_dist = env['env_R']
        else:
            dist_bnd = np.minimum.reduce([
                cand_cx - env['x_min'], env['x_max'] - cand_cx,
                cand_cy - env['y_min'], env['y_max'] - cand_cy,
            ])
            max_dist = min((env['x_max'] - env['x_min']) / 2, (env['y_max'] - env['y_min']) / 2)
        dist_bnd = np.clip(dist_bnd, 0.0, None)

        r_center_max = C['R_CENTER_MAX'] if C['R_CENTER_MAX'] is not None else cap_r
        r_max_loc    = C['R_WALL_MAX'] + (r_center_max - C['R_WALL_MAX']) * (dist_bnd / max_dist)
        pass_local   = cand_radius <= r_max_loc
    else:
        pass_local = np.ones_like(pass_radius, dtype=bool)

    pass_scale = pass_radius & pass_local & (cand_area <= cap_area)

    surviving = np.where(pass_scale)[0]

    if C['DEDUP_MODE'] == 'coverage':
        # ---------- weighted set-cover greedy dedup ------------------------
        # Build a sparse (n_surv x n_bins) coverage matrix from each
        # candidate's suprathreshold mask, then greedily pick the field
        # whose mask covers the most currently-under-saturated bins.
        n_bins_total = GRID_BINS_X * GRID_BINS_Y
        K_MAX        = int(C['COVERAGE_MAX_K'])
        MIN_GAIN     = float(C['COVERAGE_MIN_GAIN_FRAC'])

        rows, cols = [], []
        for i, k in enumerate(surviving):
            mk = cand_mask_ix[k]
            rows.append(np.full(len(mk), i, dtype=np.int32))
            cols.append(mk)
        row_idx = np.concatenate(rows) if rows else np.zeros(0, dtype=np.int32)
        col_idx = np.concatenate(cols) if cols else np.zeros(0, dtype=np.int32)
        data    = np.ones(row_idx.shape[0], dtype=np.uint8)
        M_cov   = csr_matrix((data, (row_idx, col_idx)),
                             shape=(len(surviving), n_bins_total))

        saturation = np.zeros(n_bins_total, dtype=np.int32)
        avail      = np.ones (n_bins_total, dtype=np.int32)   # 1 iff sat < K_MAX
        picked     = np.zeros(len(surviving), dtype=bool)

        kept_local  = []
        first_gain  = None
        while True:
            gains = M_cov @ avail                             # (n_surv,)
            gains[picked] = -1
            best      = int(np.argmax(gains))
            best_gain = int(gains[best])
            if best_gain <= 0:
                break
            if first_gain is None:
                first_gain = best_gain
            if best_gain < MIN_GAIN * first_gain:
                break
            kept_local.append(best)
            picked[best] = True
            bin_ids = M_cov.getrow(best).indices
            saturation[bin_ids] += 1
            avail[bin_ids] = (saturation[bin_ids] < K_MAX).astype(np.int32)

        kept = [surviving[i] for i in kept_local]
        kept_c = np.array([[cand_cx[k], cand_cy[k]] for k in kept]) if kept else np.empty((0, 2))
        kept_r = np.array([cand_radius[k] for k in kept]) if kept else np.empty((0,))

    else:
        # ---------- original containment-with-scale-gap dedup ---------------
        if C['USE_WALL_PRIOR']:
            r_center_pref = C['R_CENTER_PREF'] if C['R_CENTER_PREF'] is not None else cap_r
            r_expected    = C['R_WALL_PREF'] + (r_center_pref - C['R_WALL_PREF']) * (dist_bnd / max_dist)
            scale_pref    = -np.abs(np.log(cand_radius / np.clip(r_expected, 1e-6, None)))
            order         = surviving[np.argsort(-scale_pref[surviving])]
        else:
            # Plain largest-first: coarse fields claim their territory before
            # finer ones are considered.
            order = surviving[np.argsort(-cand_radius[surviving])]

        kept, kept_c, kept_r = [], np.empty((0, 2)), np.empty((0,))
        for k in order:
            r = cand_radius[k]
            cx_, cy_ = cand_cx[k], cand_cy[k]
            if len(kept):
                d_c        = np.hypot(kept_c[:, 0] - cx_, kept_c[:, 1] - cy_)
                similar    = (np.maximum(kept_r, r) / np.minimum(kept_r, r)) < C['SCALE_GAP_MIN']
                too_close  = d_c < C['SAME_SCALE_SEPARATION'] * (kept_r + r)
                # Same-scale peers: enforce spacing.
                # Cross-scale (similar=False): allow nesting freely.
                if np.any(similar & too_close):
                    continue
            kept.append(k)
            kept_c = np.vstack([kept_c, [cx_, cy_]]); kept_r = np.append(kept_r, r)

    if verbose:
        print(f'[{tag}] filter funnel: viab {int(pass_radius.sum())} '
              f'-> local {int((pass_radius & pass_local).sum())} '
              f'-> global {int(pass_scale.sum())} -> dedup {len(kept)}')

    # --- bank ---------------------------------------------------------------
    rows = []
    for k in sorted(kept, key=lambda kk: cand_radius[kk]):
        nid = candidate_ids[k]
        ch  = children[nid]; child_ids = list(ch) if ch is not None else []
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
    kept_mu = np.stack([cand_mu[k] for k in sorted(kept, key=lambda kk: cand_radius[kk])]) \
              if kept else np.empty((0, D), dtype=np.float32)

    return bank_df, kept_mu

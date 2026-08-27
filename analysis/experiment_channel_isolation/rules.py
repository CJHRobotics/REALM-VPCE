"""Rule-governed agglomerative place-field construction.

Implements the subset of [[Agglomeration Rules]] selected for this
experiment. Every rule is a separate, named, individually reportable stage
so the funnel can be inspected and no rule is doing hidden work.

    Rule 1  contiguity        a field is one connected patch of floor
    (Rule 2  reliability      DROPPED — still measured, no longer filters)
    Rule 4  spatial weighting merge cost = feature distance + lambda * space
    Rule 7  anisotropy        fields described by two axes + orientation
    Rule 8  size floor        nothing smaller than the smallest measured field
    Rule 9  size ceiling      nothing larger than the largest measured field
    Rule 11 competition       same-scale neighbours compete; nesting allowed
    Rule 12 tiling stop       drop scale bands that can no longer cover the floor

Rules 3, 5, 6 and 10 are deliberately not implemented here. Rule 2 was
dropped after the first full run: it rejected at most 1% of candidates in any
configuration, and the within-session split-half form we used is the one
criterion not tied to a specific source. Split-half agreement is still
computed and written to the bank as a reported property of each field.

Deliberately excluded from the model: distance-to-the-wall never enters any
merge or admission decision. It is computed only as a reported measurement,
because the claim under test is that wall-dependent field size *emerges*
from the features.

Compute notes
-------------
The two heavy steps run on GPU when one is available:

  * the N x N feature Gram matrix (the dominant cost, ~5e16 MACs for the
    full 59632-d configuration at N = 30147), and
  * the environment readout, an (N x D) @ (D x n_cand) product per run.

Ward linkage itself is sequential and stays in scipy on the CPU. The
feature distance matrix is computed once per channel configuration and
reused across every lambda, since Rule 4 only adds a positional term.
"""

import time

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy import ndimage

try:
    import torch
    _HAS_TORCH = True
except ImportError:                                        # pragma: no cover
    _HAS_TORCH = False


# ---------------------------------------------------------------- config

DEFAULT_CFG = dict(
    # --- environment readout ---------------------------------------------
    # 0.25 m bins put ~6 sample locations in every bin at this dataset's
    # density (96 locations/m^2), so <1% of in-arena bins are empty. Rule 1
    # needs that: a mask riddled with empty bins fragments under connected-
    # component analysis and every field would fail contiguity spuriously.
    BIN_M            = 0.25,
    # Field boundary as a fraction of the field's own peak response. The
    # ephys convention; commonly 0.2 of peak, 0.5 is the stricter reading.
    ACT_THRESH       = 0.50,
    # Field width in feature space: percentile of within-node pairwise
    # distance. The max is an outlier statistic (one stray member sets the
    # width), so a high percentile is used instead.
    SIGMA_PCTL       = 90,
    SIGMA_MAX_MEMBERS = 512,       # subsample cap for the percentile

    # How a node's width becomes a field extent.
    #
    #   'pairwise'  sigma = SIGMA_PCTL of within-node pairwise distance, and
    #               the mask is ACT_THRESH of the node's own peak. This is
    #               the rule everything published so far was built with.
    #
    #   'quantile'  the mask is placed directly at EXTENT_PCTL of the node's
    #               own centroid-distance distribution, by solving for the
    #               sigma that puts the ACT_THRESH cut exactly there.
    #
    # 'pairwise' fails in a way that is invisible until you test it against a
    # known answer. The mask sits at d <= sqrt(d_min^2 + 2 sigma^2 ln(1/T)),
    # where d_min is the closest any location gets to the centroid. As
    # dimensionality rises, distances concentrate: d_min grows toward the
    # global mean distance and the spread of d across the arena shrinks, so
    # that cut engulfs the whole floor unless sigma is far *below* d_min.
    # No percentile of a within-node distance distribution can deliver that
    # -- every member is at least d_min from the centroid, so every such
    # percentile is too large by construction. The statistic is not merely
    # mistuned; it cannot reach the right value.
    #
    # Measured by run_field_recovery.py across all six channels: construct an
    # ideal place cell (a disc of floor of known size and position), hand the
    # model its views, and compare what comes back.
    #
    #   given a 1 m ideal place cell   'pairwise'  6.9 - 10.0 m
    #                                  'quantile'  0.96 - 1.37 m
    #   area error, pooled             'pairwise'  x20 to x68
    #                                  'quantile'  x0.82 to x1.52
    #   ideal cells admitted           'pairwise'  0 - 20%
    #                                  'quantile'  75 - 100%
    #
    # EXTENT_PCTL = 65 selected on ideal cells admitted minus non-fields
    # admitted, which saturates between 65 and 80; 65 matches 80 there to
    # within noise while reconstructing the field substantially more
    # accurately (IoU 0.545 against 0.462).
    #
    # Non-fields excluded from that count are `split` and `ring`: a cluster
    # described by one centroid cannot represent a two-lobed response -- the
    # centroid sits between the lobes, so the field fills the gap -- and no
    # EXTENT_PCTL repairs it. That is a limit of the single-centroid
    # description, and the route past it is multi-field place cells.
    SIGMA_MODE       = 'quantile',
    EXTENT_PCTL      = 65,

    # --- Rule 4 -----------------------------------------------------------
    # cost(a,b) = d_feature^2 / med(d_feature^2) + LAMBDA * d_xy^2 / med(d_xy^2)
    # Both terms are normalised to unit median, so LAMBDA is a pure relative
    # weight and means the same thing for every channel. LAMBDA = 0 is the
    # feature-only rule.
    LAMBDA           = 0.0,

    # --- Rule 1 -----------------------------------------------------------
    CC_FRAC_MIN      = 0.80,       # largest connected component / mask area

    # --- Rule 2 (dropped) -------------------------------------------------
    # None = measure split-half agreement and record it, but do not reject on
    # it. Set a float to re-enable the filter.
    SPLIT_HALF_IOU_MIN = None,

    # --- Rules 8 / 9 ------------------------------------------------------
    # Both expressed as a fraction of environment area. The floor is
    # Harland 2021's smallest field (0.023 m^2 in an 18.6 m^2 arena); the
    # ceiling is the largest reported field (Harland 19%, Eliav 16%).
    RULE8_AREA_FRAC  = 0.023 / 18.6,
    RULE9_AREA_FRAC  = 0.20,

    # --- Rule 11 ----------------------------------------------------------
    # Two fields in the same scale band must have their centres at least
    # SAME_SCALE_SEPARATION * (r_a + r_b) apart. Fields in different bands
    # never compete, so a coarse field and the finer field nested inside it
    # both survive.
    #
    # 0.35 means two equal-sized fields must be about 0.7 radii apart —
    # substantial overlap is allowed. Lateral inhibition sparsifies a
    # population; it does not make place fields disjoint, and real fields at
    # one dorsoventral level overlap heavily. Pushing toward 1.0 forces
    # near-disjoint fields, which then cannot tile and are wiped out by
    # Rule 12.
    #
    # 0.35 rather than the earlier 0.50 because the pruning sweep (job
    # 446620) showed the smallest admissible field size saturates there:
    # relaxing 0.50 -> 0.35 drops color's smallest field from 1.48 m to
    # 0.91 m and admits a whole extra scale band, while 0.35 -> 0.25 -> 0.15
    # buys no further size reduction at all, only more redundant fields at
    # the same scales. The saturation point is a measurable stopping place;
    # it is not, however, a biological measurement, and should be reported
    # as a data-driven choice.
    SAME_SCALE_SEPARATION = 0.35,
    # Band edges are geometric with this ratio. Used ONLY to group fields
    # for Rules 11 and 12 — never to admit or reject one, so the ladder
    # spacing is measured rather than imposed (Rule 10 is not in force).
    BAND_RATIO       = 1.6,

    # --- Rule 12 ----------------------------------------------------------
    TILING_FRAC_MIN  = 0.50,       # arena coverage a band must reach to survive

    # --- runtime ----------------------------------------------------------
    READOUT_BATCH    = 256,
    RANDOM_SEED      = 0,
    USE_GPU          = True,
)


def resolve_cfg(cfg=None):
    C = dict(DEFAULT_CFG)
    C.update(cfg or {})
    return C


def pick_device(use_gpu=True, verbose=True):
    if _HAS_TORCH and use_gpu and torch.cuda.is_available():
        # TF32 needs Ampere (sm_80+). On the cluster's 1080Ti (sm_61) and
        # TitanRTX (sm_75) this flag does nothing and matmuls run in true
        # fp32 — correct, just without the Ampere speedup. Request an A40
        # (--gres=gpu:A40:1, case-sensitive) for both TF32 and 48 GB VRAM.
        cap = torch.cuda.get_device_capability(0)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        dev = torch.device('cuda')
        if verbose:
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            tf32 = 'active' if cap[0] >= 8 else 'unavailable (pre-Ampere)'
            print(f'  device: cuda ({name}, {mem:.1f} GB, sm_{cap[0]}{cap[1]}, '
                  f'tf32 {tf32})')
        return dev
    if verbose:
        print('  device: cpu' + ('' if _HAS_TORCH else ' (torch unavailable)'))
    return torch.device('cpu') if _HAS_TORCH else None


# ---------------------------------------------------------------- geometry

def build_env(xy, xml_tree_root=None):
    """Environment geometry from the world XML, falling back to the data extent."""
    env = {}
    cwalls = xml_tree_root.findall('circular_wall') if xml_tree_root is not None else []
    if cwalls:
        cw = cwalls[0]
        env['is_circular'] = True
        env['env_cx'] = float(cw.get('x', 0.0))
        env['env_cy'] = float(cw.get('y', 0.0))
        env['env_R']  = float(cw.get('radius'))
        env['env_area'] = float(np.pi * env['env_R'] ** 2)
        env['x_min'], env['x_max'] = env['env_cx'] - env['env_R'], env['env_cx'] + env['env_R']
        env['y_min'], env['y_max'] = env['env_cy'] - env['env_R'], env['env_cy'] + env['env_R']
        env['long_dim'] = 2 * env['env_R']
    else:
        env['is_circular'] = False
        # Prefer the declared boundary walls over the extent of the sampled
        # positions. The two differ by the collection grid's keep-out margin
        # (0.2 m), which understates a rectangular arena's area by 5-8% and
        # puts every wall_distance 0.2 m short. The circular branch above
        # already takes its radius from the XML rather than from the data, so
        # reading the walls here is what makes the two branches consistent.
        #
        # Assumes an axis-aligned rectangular boundary: for any other wall
        # layout this is the bounding box, which overestimates the area.
        bw = [w for w in (xml_tree_root.findall('wall')
                          if xml_tree_root is not None else [])
              if w.get('type') == 'boundary']
        if bw:
            xs = [float(w.get(k)) for w in bw for k in ('x1', 'x2')]
            ys = [float(w.get(k)) for w in bw for k in ('y1', 'y2')]
            env['x_min'], env['x_max'] = min(xs), max(xs)
            env['y_min'], env['y_max'] = min(ys), max(ys)
            env['from_walls'] = True
        else:
            env['x_min'], env['x_max'] = float(xy[:, 0].min()), float(xy[:, 0].max())
            env['y_min'], env['y_max'] = float(xy[:, 1].min()), float(xy[:, 1].max())
            env['from_walls'] = False
        env['env_area'] = (env['x_max'] - env['x_min']) * (env['y_max'] - env['y_min'])
        env['long_dim'] = max(env['x_max'] - env['x_min'], env['y_max'] - env['y_min'])
    return env


def plant_sites_by_wall(xy, env, wall_dists, n_sites, rng, tol=0.15):
    """Ideal-place-cell centers at chosen distances from the nearest wall.

    Geometry-agnostic, unlike planting on rings of a nominal arena radius:
    it selects from the sampled positions by their actual `wall_distance`,
    so it behaves the same in a disc, a rectangle and a corridor. Sites on
    one contour are chosen by farthest-point sampling, which spreads them
    over whatever shape that contour happens to be -- a circle in the disc,
    a rounded rectangle in the room, two long lines in the corridor.

    Target distances with too few candidate positions are skipped, so an
    environment simply contributes fewer contours rather than silently
    planting cells outside its own floor. A corridor 5.6 m wide has no
    position 4 m from a wall, and should report none.

    Returns [(x, y, target_w, actual_w), ...].
    """
    d = wall_distance(xy[:, 0], xy[:, 1], env)
    sites = []
    for w in wall_dists:
        cand = np.flatnonzero(np.abs(d - w) <= tol)
        if len(cand) < n_sites:
            continue
        P = xy[cand]
        picked = [int(rng.integers(len(cand)))]
        while len(picked) < n_sites:
            far = ((P[:, None, :] - P[picked][None]) ** 2).sum(-1).min(axis=1)
            picked.append(int(np.argmax(far)))
        for i in picked:
            j = cand[i]
            sites.append((float(xy[j, 0]), float(xy[j, 1]), float(w), float(d[j])))
    return sites


def wall_distance(cx, cy, env):
    """Shortest distance from a point to the environment boundary.

    Reported only. Never consulted by any merge or admission rule.
    """
    cx, cy = np.asarray(cx), np.asarray(cy)
    if env['is_circular']:
        return np.clip(env['env_R'] - np.hypot(cx - env['env_cx'], cy - env['env_cy']), 0, None)
    return np.clip(np.minimum.reduce([cx - env['x_min'], env['x_max'] - cx,
                                      cy - env['y_min'], env['y_max'] - cy]), 0, None)


# ---------------------------------------------------------------- distances

def _fits_on_gpu(X, device, headroom=1.35):
    """Will the feature matrix plus working space fit in free VRAM?

    A live question on the smaller cards: the widest configuration is
    7.2 GB against the 11 GB of a 1080 Ti. Checking up front and falling
    back to the CPU beats an OOM part-way through a long run.
    """
    try:
        free, _ = torch.cuda.mem_get_info(device)
    except Exception:
        return True
    return X.nbytes * headroom < free


def _row_sqnorm(X, chunk=2048):
    """Row-wise squared norms, accumulated in float64 a chunk at a time.

    Casting the whole feature matrix to float64 first would double a 7 GB
    array — enough to exhaust GPU memory on the widest configuration — for
    a result that is only N values wide.
    """
    if _HAS_TORCH and torch.is_tensor(X):
        out = torch.empty(X.shape[0], dtype=torch.float64, device=X.device)
        for s in range(0, X.shape[0], chunk):
            e = min(s + chunk, X.shape[0])
            out[s:e] = (X[s:e].double() ** 2).sum(1)
        return out.float()
    out = np.empty(X.shape[0], dtype=np.float64)
    for s in range(0, X.shape[0], chunk):
        e = min(s + chunk, X.shape[0])
        out[s:e] = (X[s:e].astype(np.float64) ** 2).sum(1)
    return out.astype(np.float32)


def feature_sq_distances(X, device=None, chunk=2048, verbose=True):
    """Full N x N squared Euclidean distance matrix, float32 on the CPU.

    Uses the Gram identity ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b, chunked
    over rows so the GPU only ever holds the feature matrix plus one chunk
    of output.
    """
    N = X.shape[0]
    t0 = time.time()

    if device is not None and device.type == 'cuda' and not _fits_on_gpu(X, device):
        if verbose:
            print(f'  feature matrix {X.nbytes/1e9:.1f} GB exceeds free VRAM '
                  f'— computing distances on CPU')
        device = None

    if device is not None and device.type == 'cuda':
        Xg = torch.from_numpy(np.ascontiguousarray(X)).to(device, non_blocking=True)
        sq = _row_sqnorm(Xg, chunk)
        D2 = np.empty((N, N), dtype=np.float32)
        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            g = Xg[s:e] @ Xg.T
            d2 = sq[s:e, None] + sq[None, :] - 2.0 * g
            torch.clamp_(d2, min=0)
            D2[s:e] = d2.cpu().numpy()
            del g, d2
        del Xg, sq
        torch.cuda.empty_cache()
    else:
        Xc = np.ascontiguousarray(X, dtype=np.float32)
        sq = _row_sqnorm(Xc, chunk)
        D2 = np.empty((N, N), dtype=np.float32)
        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            D2[s:e] = sq[s:e, None] + sq[None, :] - 2.0 * (Xc[s:e] @ Xc.T)
        np.maximum(D2, 0, out=D2)

    # Exact zero on the diagonal; floating-point error there breaks linkage.
    np.fill_diagonal(D2, 0.0)
    if verbose:
        print(f'  feature distances: {N}x{N} in {time.time()-t0:.1f}s '
              f'({D2.nbytes/1e9:.2f} GB)')
    return D2


def _median_offdiag(D2, rng, n_sample=2_000_000):
    """Median of the off-diagonal entries, from a random sample of pairs."""
    N = D2.shape[0]
    i = rng.integers(0, N, size=n_sample)
    j = rng.integers(0, N, size=n_sample)
    keep = i != j
    vals = D2[i[keep], j[keep]]
    m = float(np.median(vals))
    return m if m > 0 else 1.0


def ward_linkage(D2_feat, xy, lam, feat_med, xy_med, verbose=True):
    """Rule 4 — Ward linkage on the combined feature + positional metric.

    The condensed distance vector is built row by row so the positional term
    never needs its own N x N array. Ward's update rule is valid here
    because the combined metric is itself Euclidean: it is the ordinary
    distance in the space formed by the features augmented with scaled
    (x, y) coordinates.
    """
    N = D2_feat.shape[0]
    t0 = time.time()
    cond = np.empty(N * (N - 1) // 2, dtype=np.float64)
    xy64 = np.asarray(xy, dtype=np.float64)
    pos = 0
    for i in range(N - 1):
        m = N - i - 1
        d2 = D2_feat[i, i + 1:].astype(np.float64) / feat_med
        if lam:
            dxy = ((xy64[i + 1:] - xy64[i]) ** 2).sum(axis=1) / xy_med
            d2 = d2 + lam * dxy
        np.sqrt(d2, out=d2)
        cond[pos:pos + m] = d2
        pos += m
    Z = linkage(cond, method='ward')
    del cond
    if verbose:
        print(f'  ward linkage (lambda={lam}): {time.time()-t0:.1f}s')
    return Z


# ---------------------------------------------------------------- tree

def tree_from_linkage(Z, N):
    """Parent / children / depth / member-count arrays for all 2N-1 nodes."""
    n_nodes = 2 * N - 1
    parent   = np.full(n_nodes, -1, dtype=np.int64)
    children = np.full((n_nodes, 2), -1, dtype=np.int64)
    depth    = np.zeros(n_nodes, dtype=np.int32)
    count    = np.ones(n_nodes, dtype=np.int64)
    for k in range(N - 1):
        a, b = int(Z[k, 0]), int(Z[k, 1])
        nid = N + k
        parent[a] = parent[b] = nid
        children[nid] = (a, b)
        depth[nid] = 1 + max(depth[a], depth[b])
        count[nid] = count[a] + count[b]
    return parent, children, depth, count


def node_members(nid, children, N):
    """Leaf indices under a node."""
    stack, leaves = [nid], []
    while stack:
        n = stack.pop()
        if n < N:
            leaves.append(n)
        else:
            a, b = children[n]
            stack.append(int(a)); stack.append(int(b))
    return np.asarray(leaves, dtype=np.int64)


# ---------------------------------------------------------------- readout

def _grid_setup(env, C):
    gx = max(2, int(np.ceil((env['x_max'] - env['x_min']) / C['BIN_M'])))
    gy = max(2, int(np.ceil((env['y_max'] - env['y_min']) / C['BIN_M'])))
    x_edges = np.linspace(env['x_min'], env['x_max'], gx + 1)
    y_edges = np.linspace(env['y_min'], env['y_max'], gy + 1)
    xc = 0.5 * (x_edges[:-1] + x_edges[1:])
    yc = 0.5 * (y_edges[:-1] + y_edges[1:])
    bin_area = (x_edges[1] - x_edges[0]) * (y_edges[1] - y_edges[0])
    GX, GY = np.meshgrid(xc, yc, indexing='ij')
    if env['is_circular']:
        in_env = (GX - env['env_cx']) ** 2 + (GY - env['env_cy']) ** 2 <= env['env_R'] ** 2
    else:
        in_env = np.ones((gx, gy), dtype=bool)
    return dict(gx=gx, gy=gy, xc=xc, yc=yc, x_edges=x_edges, y_edges=y_edges,
                bin_area=bin_area, in_env=in_env, GX=GX, GY=GY)


def _bin_indices(xy, G):
    ix = np.clip(np.searchsorted(G['x_edges'], xy[:, 0], side='right') - 1, 0, G['gx'] - 1)
    iy = np.clip(np.searchsorted(G['y_edges'], xy[:, 1], side='right') - 1, 0, G['gy'] - 1)
    return (ix * G['gy'] + iy).astype(np.int64)


def environment_readout(X, xy, cand_mu, cand_sigma, G, C, device, half_id, verbose=True):
    """Per-candidate response map on the floor, plus the two split-half maps.

    Returns three (n_cand, n_bins) float32 arrays of per-bin peak response:
    over all locations, over half 0, and over half 1. The expensive part —
    the (N x D) @ (D x batch) product — is shared between all three.
    """
    n_cand, D = cand_mu.shape
    n_bins = G['gx'] * G['gy']
    flat = _bin_indices(xy, G)

    out_all = np.zeros((n_cand, n_bins), dtype=np.float32)
    out_a   = np.zeros((n_cand, n_bins), dtype=np.float32)
    out_b   = np.zeros((n_cand, n_bins), dtype=np.float32)
    t0 = time.time()

    on_gpu = device is not None and device.type == 'cuda' and _fits_on_gpu(X, device)
    if device is not None and device.type == 'cuda' and not on_gpu and verbose:
        print('  readout: feature matrix exceeds free VRAM — using CPU')
    if on_gpu:
        Xg    = torch.from_numpy(np.ascontiguousarray(X)).to(device)
        Xsq   = _row_sqnorm(Xg)
        flatg = torch.from_numpy(flat).to(device)
        m0    = torch.from_numpy(half_id == 0).to(device)
        m1    = torch.from_numpy(half_id == 1).to(device)

        for s in range(0, n_cand, C['READOUT_BATCH']):
            e = min(s + C['READOUT_BATCH'], n_cand)
            M = torch.from_numpy(np.ascontiguousarray(cand_mu[s:e])).to(device)
            S = torch.from_numpy(cand_sigma[s:e].astype(np.float32)).to(device)
            d2 = Xsq[:, None] - 2.0 * (Xg @ M.T) + _row_sqnorm(M)[None, :]
            torch.clamp_(d2, min=0)
            S = torch.where(S > 0, S, torch.ones_like(S))
            raw = torch.exp(-d2 / (2.0 * S[None, :] ** 2))          # (N, b)
            for tgt, sel in ((out_all, None), (out_a, m0), (out_b, m1)):
                r = raw if sel is None else raw[sel]
                idx = flatg if sel is None else flatg[sel]
                grid = torch.zeros((e - s, n_bins), device=device)
                grid.scatter_reduce_(1, idx[None, :].expand(e - s, -1),
                                     r.T.contiguous(), reduce='amax',
                                     include_self=True)
                tgt[s:e] = grid.cpu().numpy()
                del grid
            del M, S, d2, raw
        del Xg, Xsq, flatg, m0, m1
        torch.cuda.empty_cache()
    else:
        Xc  = np.ascontiguousarray(X, dtype=np.float32)
        Xsq = _row_sqnorm(Xc)
        sel_a, sel_b = half_id == 0, half_id == 1
        for s in range(0, n_cand, C['READOUT_BATCH']):
            e = min(s + C['READOUT_BATCH'], n_cand)
            M = np.ascontiguousarray(cand_mu[s:e])
            S = cand_sigma[s:e].astype(np.float32)
            d2 = Xsq[:, None] - 2.0 * (Xc @ M.T) + _row_sqnorm(M)[None, :]
            np.maximum(d2, 0, out=d2)
            S = np.where(S > 0, S, 1.0).astype(np.float32)
            raw = np.exp(-d2 / (2.0 * S[None, :] ** 2))
            for tgt, sel in ((out_all, None), (out_a, sel_a), (out_b, sel_b)):
                r = raw if sel is None else raw[sel]
                idx = flat if sel is None else flat[sel]
                for j in range(e - s):
                    np.maximum.at(tgt[s + j], idx, r[:, j])
            del d2, raw
    if verbose:
        print(f'  environment readout: {n_cand} candidates in {time.time()-t0:.1f}s')
    return out_all, out_a, out_b


def _fill_empty_bins(grid2d, occupied, in_env):
    """Give unsampled in-arena bins the max of their sampled neighbours.

    Without this, the handful of bins that happen to contain no sample
    locations punch holes in every mask and Rule 1 rejects fields for a
    sampling artifact rather than for genuine fragmentation.
    """
    holes = in_env & ~occupied
    if not holes.any():
        return grid2d
    filled = grid2d.copy()
    filled[holes] = ndimage.maximum_filter(grid2d, size=3)[holes]
    return filled


def mask_from_grid(flat_grid, G, occupied, C):
    """Field mask: bins at or above ACT_THRESH times the field's own peak."""
    g = _fill_empty_bins(flat_grid.reshape(G['gx'], G['gy']), occupied, G['in_env'])
    peak = g.max()
    if not np.isfinite(peak) or peak <= 0:
        return np.zeros_like(g, dtype=bool), 0.0
    return (g >= C['ACT_THRESH'] * peak) & G['in_env'], float(peak)


# ---------------------------------------------------------------- Rule 7

def field_shape(mask, G):
    """Rule 7 — describe a field by two axes and an orientation, not a radius.

    Second moments of the mask give the axis ratio and orientation; the
    absolute scale is then set so that pi * a * b equals the measured area.
    A single equivalent-disc radius understates an elongated field along one
    axis and overstates it along the other, which corrupts every size
    statistic computed from it.

    Returns dict with area, semi-axes a >= b, elongation, orientation (rad),
    equivalent-disc radius, and the response-weighted centre.
    """
    idx = np.flatnonzero(mask.ravel())
    n = len(idx)
    area = n * G['bin_area']
    if n == 0:
        return dict(area=0.0, a=0.0, b=0.0, elongation=1.0, theta=0.0,
                    r_eq=0.0, cx=np.nan, cy=np.nan)

    ii, jj = np.unravel_index(idx, mask.shape)
    px, py = G['xc'][ii], G['yc'][jj]
    cx, cy = px.mean(), py.mean()
    r_eq = np.sqrt(area / np.pi)

    bin_side = np.sqrt(G['bin_area'])
    if n < 3:
        return dict(area=area, a=r_eq, b=r_eq, elongation=1.0, theta=0.0,
                    r_eq=r_eq, cx=float(cx), cy=float(cy))

    cov = np.cov(np.stack([px - cx, py - cy]))
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]
    # Discretisation floor: an axis can never be measured below one bin.
    evals = np.clip(evals, (bin_side ** 2) / 12.0, None)

    # Uniform ellipse: variance along a semi-axis of length a is a^2 / 4.
    a_raw, b_raw = 2.0 * np.sqrt(evals[0]), 2.0 * np.sqrt(evals[1])
    scale = np.sqrt(area / (np.pi * a_raw * b_raw)) if a_raw * b_raw > 0 else 1.0
    a, b = a_raw * scale, b_raw * scale
    theta = float(np.arctan2(evecs[1, 0], evecs[0, 0]))
    return dict(area=float(area), a=float(a), b=float(b),
                elongation=float(a / b) if b > 0 else 1.0,
                theta=theta, r_eq=float(r_eq), cx=float(cx), cy=float(cy))


# ---------------------------------------------------------------- Rules 1, 2

def largest_component_fraction(mask):
    """Rule 1 — share of the mask held by its largest connected component."""
    total = int(mask.sum())
    if total == 0:
        return 0.0, 0
    lab, n = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
    if n <= 1:
        return 1.0, n
    sizes = np.bincount(lab.ravel())[1:]
    return float(sizes.max() / total), int(n)


def mask_iou(m1, m2):
    """Rule 2 — agreement between the two independent split-half masks."""
    inter = int(np.logical_and(m1, m2).sum())
    union = int(np.logical_or(m1, m2).sum())
    return inter / union if union else 0.0


# ---------------------------------------------------------------- Rules 11, 12

def assign_bands(radii, r_min, ratio):
    """Geometric scale bands. Grouping only — never an admission test."""
    radii = np.asarray(radii, dtype=np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        b = np.floor(np.log(np.maximum(radii, 1e-9) / r_min) / np.log(ratio))
    return np.clip(b, 0, None).astype(int)


def rule11_competition(order, cx, cy, r_mean, band, sep):
    """Rule 11 — within a scale band, a field suppresses nearby peers.

    Fields in different bands never compete, so a coarse field and the finer
    field nested inside it both survive: the scale ladder is preserved while
    same-scale redundancy is removed.
    """
    kept = []
    kept_by_band = {}
    for k in order:
        bk = band[k]
        peers = kept_by_band.get(bk)
        if peers:
            p = np.asarray(peers)
            d = np.hypot(cx[p] - cx[k], cy[p] - cy[k])
            if np.any(d < sep * (r_mean[p] + r_mean[k])):
                continue
        kept.append(k)
        kept_by_band.setdefault(bk, []).append(k)
    return kept


def rule12_tiling(kept, band, masks, G, C):
    """Rule 12 — a scale band survives only if it can still cover the floor.

    A band whose fields cover less than TILING_FRAC_MIN of the environment
    is not a population code for space at that resolution, so it is dropped.
    Coverage is evaluated after Rule 11, on the fields that actually survive
    competition.

    Both ends of the ladder are policed by this, for different reasons.
    Coarse bands fail because too few nodes that large exist — the thinning
    the rule was written for. Fine bands fail because a tiling at that
    resolution would need far more fields than the tree produces; claiming a
    population code there would be claiming resolution we do not have.

    The surviving ladder is the contiguous run of qualifying bands around
    the best-covered one, so it never has a hole in the middle.

    Returns (kept_ids, coverage_by_band, kept_band_range).
    """
    env_bins = int(G['in_env'].sum())
    coverage, kept_arr = {}, np.asarray(kept)
    if kept_arr.size == 0:
        return [], coverage, (-1, -1)
    for bk in sorted(set(int(band[k]) for k in kept)):
        members = kept_arr[band[kept_arr] == bk]
        union = np.zeros(G['in_env'].shape, dtype=bool)
        for k in members:
            union |= masks[k]
        coverage[bk] = float(union.sum() / env_bins) if env_bins else 0.0

    thr = C['TILING_FRAC_MIN']
    qualifying = [b for b in sorted(coverage) if coverage[b] >= thr]
    if not qualifying:
        return [], coverage, (-1, -1)

    anchor = max(coverage, key=lambda b: coverage[b])
    lo = hi = anchor
    bands_sorted = sorted(coverage)
    pos = bands_sorted.index(anchor)
    for b in reversed(bands_sorted[:pos]):
        if coverage[b] < thr:
            break
        lo = b
    for b in bands_sorted[pos + 1:]:
        if coverage[b] < thr:
            break
        hi = b
    return [k for k in kept if lo <= band[k] <= hi], coverage, (lo, hi)


# ---------------------------------------------------------------- pipeline

def build_tree(D2_feat, xy, feat_med, xy_med, cfg=None, verbose=True):
    """The Ward tree alone.

    Depends on LAMBDA and on nothing else — no width statistic, no threshold,
    no admission parameter — so one tree serves a whole sweep of those. Split
    out for the same reason prepare_candidates and admit_fields are split: a
    sweep that rebuilds the tree per setting is both slower and, if anything
    in the build is stochastic, not strictly like-for-like.

    Reuse is valid only across settings that share LAMBDA.
    """
    C = resolve_cfg(cfg)
    N = len(xy)
    Z = ward_linkage(D2_feat, xy, C['LAMBDA'], feat_med, xy_med, verbose=verbose)
    parent, children, depth, count = tree_from_linkage(Z, N)
    return dict(Z=Z, parent=parent, children=children, depth=depth,
                count=count, lam=C['LAMBDA'])


def prepare_candidates(X, xy, env, D2_feat, feat_med, xy_med, cfg=None,
                       device=None, tag='', verbose=True, tree=None):
    """Everything up to, but not including, the admission rules.

    Builds the tree, selects candidate nodes, measures each one's centre and
    width, and evaluates its response across the environment. None of this
    depends on the response threshold or on any admission parameter, so the
    result can be reused across a sweep of those — which makes such a sweep
    both cheaper and provably like-for-like, since every setting is then
    scored against an identical tree and identical candidates.

    Returns a context dict for `admit_fields`.
    """
    C = resolve_cfg(cfg)
    rng = np.random.default_rng(C['RANDOM_SEED'])
    N, D = X.shape
    G = _grid_setup(env, C)

    area_min = C['RULE8_AREA_FRAC'] * env['env_area']       # Rule 8
    area_max = C['RULE9_AREA_FRAC'] * env['env_area']       # Rule 9
    r_min, r_max = np.sqrt(area_min / np.pi), np.sqrt(area_max / np.pi)

    occupied = np.zeros(G['gx'] * G['gy'], dtype=bool)
    occupied[_bin_indices(xy, G)] = True
    occupied = occupied.reshape(G['gx'], G['gy'])
    if verbose:
        occ = float((occupied & G['in_env']).sum() / max(G['in_env'].sum(), 1))
        print(f'[{tag}] grid {G["gx"]}x{G["gy"]} @ {C["BIN_M"]} m  '
              f'({100*occ:.1f}% of in-arena bins sampled)')
        print(f'[{tag}] Rule 8 floor r >= {r_min:.3f} m   '
              f'Rule 9 ceiling r <= {r_max:.3f} m')

    # --- tree -------------------------------------------------------------
    if tree is None:
        tree = build_tree(D2_feat, xy, feat_med, xy_med, cfg=C, verbose=verbose)
    elif tree.get('lam') != C['LAMBDA'] and verbose:
        print(f'[{tag}] !! reusing a tree built at lambda={tree.get("lam")} '
              f'under lambda={C["LAMBDA"]}')
    parent, children, depth, count = (tree['parent'], tree['children'],
                                      tree['depth'], tree['count'])

    # Candidate prefilter. A field at the Rule 8 floor holds about
    # density * area_min locations; require at least half of that so the
    # size rules, not the member count, do the actual work.
    density = N / env['env_area']
    min_members = max(8, int(0.5 * density * area_min))
    max_members = int(2.0 * density * area_max)
    cand = np.flatnonzero((count >= min_members) & (count <= max_members))
    if verbose:
        print(f'[{tag}] candidates: {len(cand)} / {2*N-1} nodes '
              f'({min_members} <= members <= {max_members})')

    # --- mu, sigma --------------------------------------------------------
    t0 = time.time()
    cand_mu = np.empty((len(cand), D), dtype=np.float32)
    cand_sigma = np.empty(len(cand), dtype=np.float32)
    members_of = []
    quantile_mode = C['SIGMA_MODE'] == 'quantile'
    log_inv_t = np.log(1.0 / C['ACT_THRESH'])
    for k, nid in enumerate(cand):
        m = node_members(int(nid), children, N)
        members_of.append(m)
        cand_mu[k] = X[m].mean(axis=0)
        s = m if len(m) <= C['SIGMA_MAX_MEMBERS'] else rng.choice(m, C['SIGMA_MAX_MEMBERS'], replace=False)
        sub = D2_feat[np.ix_(s, s)]
        iu = np.triu_indices(len(s), k=1)
        if not quantile_mode:
            cand_sigma[k] = np.sqrt(np.percentile(sub[iu], C['SIGMA_PCTL']))
            continue
        # Squared distance of each member to the node centroid, straight from
        # the pairwise block: ||x_i - mu||^2 = mean_j d2_ij - mean_jk d2_jk / 2.
        dc2 = sub.mean(axis=1) - 0.5 * sub.mean()
        np.maximum(dc2, 0, out=dc2)
        q2 = float(np.percentile(dc2, C['EXTENT_PCTL']))
        d_min2 = float(dc2.min())
        # sigma placing the ACT_THRESH cut at that quantile. Degenerate when
        # the quantile is not above the closest member (identical vectors, as
        # in lidar's out-of-range centre); sigma = 0 there, and the readout
        # already treats that as a point response.
        cand_sigma[k] = np.sqrt((q2 - d_min2) / (2.0 * log_inv_t)) if q2 > d_min2 else 0.0
    if verbose:
        print(f'  centres + widths: {time.time()-t0:.1f}s')

    # --- environment readout ---------------------------------------------
    half_id = rng.integers(0, 2, size=N)
    resp_all, resp_a, resp_b = environment_readout(
        X, xy, cand_mu, cand_sigma, G, C, device, half_id, verbose=verbose)

    return dict(G=G, occupied=occupied, env=env, N=N, D=D, tag=tag,
                feat_med=feat_med, tree=tree,
                cand=cand, cand_mu=cand_mu, cand_sigma=cand_sigma,
                parent=parent, children=children, depth=depth, count=count,
                resp_all=resp_all, resp_a=resp_a, resp_b=resp_b,
                area_min=area_min, area_max=area_max, r_min=r_min, r_max=r_max,
                min_members=min_members, max_members=max_members)


def admit_fields(ctx, cfg=None, verbose=True):
    """Apply the response threshold and the admission rules to a context.

    Returns
    -------
    bank_df : one row per surviving field
    kept_mu : (n_fields, D) feature-space centres, aligned to bank_df
    report  : funnel counts, band coverage, and per-rule diagnostics
    """
    C = resolve_cfg(cfg)
    G, occupied, env = ctx['G'], ctx['occupied'], ctx['env']
    N, D, tag = ctx['N'], ctx['D'], ctx['tag']
    cand, cand_mu, cand_sigma = ctx['cand'], ctx['cand_mu'], ctx['cand_sigma']
    parent, children, depth, count = (ctx['parent'], ctx['children'],
                                      ctx['depth'], ctx['count'])
    resp_all, resp_a, resp_b = ctx['resp_all'], ctx['resp_a'], ctx['resp_b']
    feat_med = ctx['feat_med']
    area_min, area_max = ctx['area_min'], ctx['area_max']
    r_min, r_max = ctx['r_min'], ctx['r_max']
    min_members, max_members = ctx['min_members'], ctx['max_members']

    # --- per-candidate masks, shapes, rule tests --------------------------
    n_cand = len(cand)
    masks = [None] * n_cand
    shape = [None] * n_cand
    cc_frac = np.zeros(n_cand); n_comp = np.zeros(n_cand, dtype=int)
    sh_iou = np.zeros(n_cand)
    for k in range(n_cand):
        m_all, _ = mask_from_grid(resp_all[k], G, occupied, C)
        masks[k] = m_all
        shape[k] = field_shape(m_all, G)
        cc_frac[k], n_comp[k] = largest_component_fraction(m_all)
        ma, _ = mask_from_grid(resp_a[k], G, occupied, C)
        mb, _ = mask_from_grid(resp_b[k], G, occupied, C)
        sh_iou[k] = mask_iou(ma, mb)

    area = np.array([s['area'] for s in shape])
    a_ax = np.array([s['a'] for s in shape])
    b_ax = np.array([s['b'] for s in shape])
    elong = np.array([s['elongation'] for s in shape])
    theta = np.array([s['theta'] for s in shape])
    r_eq = np.array([s['r_eq'] for s in shape])
    cx = np.array([s['cx'] for s in shape])
    cy = np.array([s['cy'] for s in shape])

    # --- funnel -----------------------------------------------------------
    funnel = [('candidates', n_cand)]

    pass_size = (area >= area_min) & (area <= area_max)                # Rules 8, 9
    funnel.append(('rule_8_9_size', int(pass_size.sum())))

    pass_cc = pass_size & (cc_frac >= C['CC_FRAC_MIN'])                # Rule 1
    funnel.append(('rule_1_contiguity', int(pass_cc.sum())))

    if C['SPLIT_HALF_IOU_MIN'] is not None:                            # Rule 2
        pass_rel = pass_cc & (sh_iou >= C['SPLIT_HALF_IOU_MIN'])
        funnel.append(('rule_2_reliability', int(pass_rel.sum())))
    else:
        pass_rel = pass_cc

    surviving = np.flatnonzero(pass_rel)
    r_mean = 0.5 * (a_ax + b_ax)
    band = assign_bands(r_eq, r_min, C['BAND_RATIO'])

    # Rule 11 — larger and more reliable fields claim their territory first.
    order = surviving[np.lexsort((-sh_iou[surviving], -r_eq[surviving]))]
    kept = rule11_competition(order, cx, cy, r_mean, band, C['SAME_SCALE_SEPARATION'])
    funnel.append(('rule_11_competition', len(kept)))

    kept, coverage, band_range = rule12_tiling(kept, band, masks, G, C)  # Rule 12
    funnel.append(('rule_12_tiling', len(kept)))

    if verbose:
        print(f'[{tag}] funnel: ' + ' -> '.join(f'{n}={c}' for n, c in funnel))
        print(f'[{tag}] band coverage: '
              + ', '.join(f'B{b}={coverage[b]*100:.0f}%' for b in sorted(coverage))
              + f'  (kept bands {band_range[0]}..{band_range[1]})')
        if not kept:
            print(f'[{tag}] !! Rule 12 admitted no band — no scale reached '
                  f'{100*C["TILING_FRAC_MIN"]:.0f}% coverage')

    # --- bank -------------------------------------------------------------
    kept = sorted(kept, key=lambda k: r_eq[k])
    d_wall = wall_distance(cx[kept], cy[kept], env) if kept else np.array([])
    rows = []
    for i, k in enumerate(kept):
        nid = int(cand[k])
        rows.append(dict(
            node_id=nid, depth=int(depth[nid]), n_members=int(count[nid]),
            sigma_feature=float(cand_sigma[k]),
            area_env_m2=float(area[k]), radius_env_m=float(r_eq[k]),
            semi_major_m=float(a_ax[k]), semi_minor_m=float(b_ax[k]),
            elongation=float(elong[k]), orientation_rad=float(theta[k]),
            centroid_x=float(cx[k]), centroid_y=float(cy[k]),
            dist_to_wall_m=float(d_wall[i]),
            scale_band=int(band[k]),
            cc_frac=float(cc_frac[k]), n_components=int(n_comp[k]),
            split_half_iou=float(sh_iou[k]),
            parent_id=int(parent[nid]), is_leaf=bool(nid < N),
        ))
    # Explicit schema so a channel that admits nothing still writes a
    # well-formed, readable bank.csv with headers rather than an empty file.
    BANK_COLUMNS = ['node_id', 'depth', 'n_members', 'sigma_feature',
                    'area_env_m2', 'radius_env_m', 'semi_major_m', 'semi_minor_m',
                    'elongation', 'orientation_rad', 'centroid_x', 'centroid_y',
                    'dist_to_wall_m', 'scale_band', 'cc_frac', 'n_components',
                    'split_half_iou', 'parent_id', 'is_leaf']
    bank_df = pd.DataFrame(rows, columns=BANK_COLUMNS)
    kept_mu = (cand_mu[kept] if kept else np.empty((0, D), dtype=np.float32))

    # How tight is a candidate relative to the whole dataset? sigma_ratio
    # near 1 means a node's members are as spread out in feature space as
    # two random locations are — the channel carries no usable locality at
    # that scale, the RBF response is near-flat, and the "field" covers the
    # arena. This is what separates a channel that fails to localise from
    # one that merely produces few fields.
    sigma_ratio = cand_sigma / np.sqrt(max(feat_med, 1e-12))

    report = dict(
        act_thresh=float(C['ACT_THRESH']),
        funnel=funnel, coverage=coverage,
        median_sigma_ratio=float(np.median(sigma_ratio)),
        cand_sigma_ratio=sigma_ratio,
        band_lo=int(band_range[0]), band_hi=int(band_range[1]),
        r_min=float(r_min), r_max=float(r_max),
        area_min=float(area_min), area_max=float(area_max),
        min_members=int(min_members), max_members=int(max_members),
        n_candidates=int(n_cand),
        # Rule 1 and 2 diagnostics over all candidates that passed on size,
        # so the rejection rates are reported and not merely applied.
        frag_rate=float((cc_frac[pass_size] < C['CC_FRAC_MIN']).mean()) if pass_size.any() else 0.0,
        median_cc_frac=float(np.median(cc_frac[pass_size])) if pass_size.any() else 0.0,
        unreliable_rate=(float((sh_iou[pass_cc] < C['SPLIT_HALF_IOU_MIN']).mean())
                         if pass_cc.any() and C['SPLIT_HALF_IOU_MIN'] is not None else 0.0),
        median_split_half_iou=float(np.median(sh_iou[pass_cc])) if pass_cc.any() else 0.0,
        cand_cc_frac=cc_frac, cand_split_half_iou=sh_iou,
        cand_pass_size=pass_size, cand_r_eq=r_eq, cand_elongation=elong,
        grid=dict(gx=G['gx'], gy=G['gy'], bin_area=G['bin_area']),
    )
    return bank_df, kept_mu, report


def build_bank(X, xy, env, D2_feat, feat_med, xy_med, cfg=None,
               device=None, tag='', verbose=True):
    """Prepare candidates and admit fields in one call.

    Thin wrapper over `prepare_candidates` + `admit_fields`, kept because
    most callers run a single configuration and do not need the split.
    """
    ctx = prepare_candidates(X, xy, env, D2_feat, feat_med, xy_med,
                             cfg=cfg, device=device, tag=tag, verbose=verbose)
    return admit_fields(ctx, cfg=cfg, verbose=verbose)

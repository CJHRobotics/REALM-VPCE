"""Generate the model schematic for the channel-isolation pipeline.

Writes an SVG matching the house style of the existing VPCE-Brain pipeline
figure (white ground, slate palette, numbered step panels). Regenerate after
any change to the model so the figure and the methods text cannot drift
apart.

    python make_model_figure.py [output.svg]

Default output is the VPCE-Brain vault figures directory.
"""

import os
import sys
from math import cos, sin, pi, hypot

import numpy as np

DEFAULT_OUT = os.path.expanduser(
    '~/VPCE-Brain/VPCE-Brain/figures/current_model_pipeline.svg')

# ---------------------------------------------------------------- palette
INK      = '#0f172a'
MUTED    = '#64748b'
FAINT    = '#94a3b8'
LINE     = '#475569'
PANEL_BG = '#f8fafc'
PANEL_ED = '#e2e8f0'
BLUE     = '#3b82f6'
RED      = '#ef4444'
GREEN    = '#10b981'
AMBER    = '#f59e0b'
VIOLET   = '#8b5cf6'
RULE_BG  = '#eef2ff'
RULE_FG  = '#4338ca'

W, H = 1620, 1000
PX, PY, PW, PH = 45, 100, 360, 380
GAP = 30
ROW2_Y = 560


def px(i):
    return PX + (PW + GAP) * i


def esc(s):
    return (s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;'))


def text(x, y, s, size=12, fill=INK, weight='400', anchor='start',
         style='', family=None):
    f = f' font-family="{family}"' if family else ''
    st = f' font-style="{style}"' if style else ''
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" '
            f'font-weight="{weight}" text-anchor="{anchor}"{f}{st}>{esc(s)}</text>')


def mono(x, y, s, size=11, fill=INK, anchor='start'):
    return text(x, y, s, size, fill, anchor=anchor,
                family="ui-monospace, SFMono-Regular, Menlo, monospace")


def panel(i, row, num, title, rule=None):
    """Panel frame with numbered badge, title and optional rule pill."""
    x = px(i)
    y = PY if row == 0 else ROW2_Y
    o = [f'<rect x="{x}" y="{y}" width="{PW}" height="{PH}" rx="12" '
         f'fill="{PANEL_BG}" stroke="{PANEL_ED}" stroke-width="1.5"/>',
         f'<circle cx="{x+26}" cy="{y+28}" r="13" fill="{INK}"/>',
         text(x + 26, y + 33, str(num), 13, '#ffffff', '700', 'middle'),
         text(x + 48, y + 33, title, 15, INK, '600')]
    if rule:
        w = 8 + 6.2 * len(rule)
        o += [f'<rect x="{x+PW-w-14}" y="{y+17}" width="{w}" height="22" rx="11" '
              f'fill="{RULE_BG}"/>',
              text(x + PW - 14 - w / 2, y + 32, rule, 11, RULE_FG, '600', 'middle')]
    return o


def caption(i, row, lines):
    x, y = px(i) + 18, (PY if row == 0 else ROW2_Y) + PH - 14 - 14 * (len(lines) - 1)
    return [text(x, y + 14 * k, ln, 11, MUTED) for k, ln in enumerate(lines)]


def arrow(x1, y1, x2, y2, marker='arrow', color=LINE, width=2, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ''
    return (f'<path d="M{x1:.1f},{y1:.1f} L{x2:.1f},{y2:.1f}" stroke="{color}" '
            f'stroke-width="{width}" fill="none"{d} marker-end="url(#{marker})"/>')


# ================================================================ panels

def p1_acquisition(o):
    x, y = px(0), PY
    cx, cy, R = x + 180, y + 200, 108
    o.append(f'<circle cx="{cx}" cy="{cy}" r="{R}" fill="#ffffff" '
             f'stroke="{INK}" stroke-width="2.5"/>')

    # landmark panels on the inner wall face
    lm = ['#ef4444', '#22c55e', '#3b82f6', '#eab308',
          '#06b6d4', '#f97316', '#a855f7', '#14b8a6']
    for k, c in enumerate(lm):
        a = pi / 8 + k * pi / 4
        o.append(f'<line x1="{cx+(R-9)*cos(a):.1f}" y1="{cy+(R-9)*sin(a):.1f}" '
                 f'x2="{cx+R*cos(a):.1f}" y2="{cy+R*sin(a):.1f}" '
                 f'stroke="{c}" stroke-width="6" stroke-linecap="round"/>')

    # sampled locations
    rng = np.random.default_rng(3)
    pts = []
    while len(pts) < 150:
        u, v = rng.uniform(-1, 1, 2)
        if u * u + v * v <= 0.94:
            pts.append((u * R, v * R))
    for dx, dy in pts:
        o.append(f'<circle cx="{cx+dx:.1f}" cy="{cy+dy:.1f}" r="1.7" '
                 f'fill="{FAINT}"/>')

    # one location, expanded: 8 camera headings + a range scan bounded at 5 m
    lx, ly = cx - 26, cy + 14
    r5 = R * 0.5                                   # 5 m at 10 m = R
    o.append(f'<circle cx="{lx}" cy="{ly}" r="{r5}" fill="{BLUE}" '
             f'fill-opacity="0.07" stroke="{BLUE}" stroke-width="1.4" '
             f'stroke-dasharray="4 3"/>')
    for k in range(8):
        a = k * pi / 4
        o.append(f'<line x1="{lx+7*cos(a):.1f}" y1="{ly+7*sin(a):.1f}" '
                 f'x2="{lx+21*cos(a):.1f}" y2="{ly+21*sin(a):.1f}" '
                 f'stroke="{RED}" stroke-width="1.6"/>')
    o.append(f'<circle cx="{lx}" cy="{ly}" r="4" fill="{RED}"/>')
    o.append(text(lx + r5 + 6, ly - 4, '5 m', 10, BLUE, '600'))
    o.append(text(cx, cy - R - 12, 'walkable disc, r = 10 m', 11, MUTED, '400', 'middle'))

    o += caption(0, 0, ['30,147 locations · 96 m⁻² · area 314.2 m²',
                        '8 camera headings + one 360-beam scan per location'])


def p2_channels(o):
    x, y = px(1), PY
    rows = [('hog', 30240, BLUE), ('color', 4096, RED),
            ('spatial', 24576, GREEN), ('lidar', 720, VIOLET)]
    top, bw = y + 64, 118
    for k, (name, dim, c) in enumerate(rows):
        yy = top + k * 30
        w = bw * (dim / 30240) ** 0.45
        o.append(f'<rect x="{x+86}" y="{yy}" width="{max(w,14):.1f}" height="16" '
                 f'rx="3" fill="{c}" fill-opacity="0.75"/>')
        o.append(text(x + 78, yy + 13, name, 12, INK, '600', 'end'))
        o.append(mono(x + 92 + max(w, 14), yy + 13, f'{dim:,}', 10, MUTED))
    o.append(text(x + 18, top + 4 * 30 + 14, 'dimensions after concatenating the 8',
                  10.5, MUTED))
    o.append(text(x + 18, top + 4 * 30 + 28, 'headings (range stored once per location)',
                  10.5, MUTED))
    o.append(f'<line x1="{x+18}" y1="{top+4*30+44}" x2="{x+PW-18}" '
             f'y2="{top+4*30+44}" stroke="{PANEL_ED}" stroke-width="1.2"/>')

    # range limiting, shown as a polar scan
    scx, scy, sr = x + 84, y + 288, 42
    o.append(f'<circle cx="{scx}" cy="{scy}" r="{sr}" fill="#ffffff" '
             f'stroke="{PANEL_ED}" stroke-width="1.2"/>')
    rng = np.random.default_rng(11)
    for k in range(56):
        a = k * 2 * pi / 56
        rr = rng.uniform(0.22, 1.55)
        inr = rr <= 0.62
        L = sr * min(rr, 0.62)
        col = VIOLET if inr else '#cbd5e1'
        dash = '' if inr else ' stroke-dasharray="2 2"'
        o.append(f'<line x1="{scx}" y1="{scy}" '
                 f'x2="{scx+L*cos(a):.1f}" y2="{scy+L*sin(a):.1f}" '
                 f'stroke="{col}" stroke-width="1.6"{dash}/>')
    o.append(f'<circle cx="{scx}" cy="{scy}" r="{sr*0.62:.1f}" fill="none" '
             f'stroke="{VIOLET}" stroke-width="1.3" stroke-dasharray="4 3"/>')
    o.append(text(x + 144, scy - 18, 'range limit 5 m', 11.5, INK, '600'))
    o.append(text(x + 144, scy - 2, 'beams past it → −1,', 10.5, MUTED))
    o.append(text(x + 144, scy + 12, 'plus a per-beam', 10.5, MUTED))
    o.append(text(x + 144, scy + 26, 'in-range flag', 10.5, MUTED))
    o.append(text(x + 144, scy + 44, '62% of beams masked', 10.5, VIOLET, '600'))

    o += caption(1, 0, ['each block ÷ its RMS row norm, so no channel',
                        'dominates the metric by width or units'])


def p3_metric(o):
    x, y = px(2), PY
    o.append(f'<rect x="{x+16}" y="{y+56}" width="{PW-32}" height="52" rx="8" '
             f'fill="#ffffff" stroke="{PANEL_ED}"/>')
    o.append(mono(x + PW / 2, y + 80, 'd² =  ‖Δf‖² / med  +  λ · ‖Δp‖² / med',
                  12.5, INK, 'middle'))
    o.append(text(x + PW / 2, y + 98, 'appearance          position', 10, MUTED,
                  '400', 'middle'))

    # the same two locations under two values of lambda
    for k, (lab, lam_pull, col) in enumerate([('λ = 0', 0.0, MUTED),
                                              ('λ > 0', 1.0, BLUE)]):
        bx = x + 30 + k * 172
        by = y + 130
        o.append(f'<rect x="{bx}" y="{by}" width="140" height="112" rx="8" '
                 f'fill="#ffffff" stroke="{PANEL_ED}"/>')
        o.append(text(bx + 70, by + 20, lab, 12, col, '700', 'middle'))
        a = (bx + 30, by + 76)
        b = (bx + 110, by + 44)
        o.append(f'<circle cx="{a[0]}" cy="{a[1]}" r="7" fill="{RED}" '
                 f'fill-opacity="0.85"/>')
        o.append(f'<circle cx="{b[0]}" cy="{b[1]}" r="7" fill="{RED}" '
                 f'fill-opacity="0.85"/>')
        if lam_pull == 0:
            o.append(f'<path d="M{a[0]+8},{a[1]-3} L{b[0]-8},{b[1]+4}" '
                     f'stroke="{GREEN}" stroke-width="3"/>')
            o.append(text(bx + 70, by + 102, 'merged: they look alike', 9.5,
                          MUTED, '400', 'middle'))
        else:
            o.append(f'<path d="M{a[0]+8},{a[1]-3} L{b[0]-8},{b[1]+4}" '
                     f'stroke="{FAINT}" stroke-width="1.4" stroke-dasharray="3 3"/>')
            o.append(f'<line x1="{a[0]-14}" y1="{a[1]+16}" x2="{b[0]+14}" '
                     f'y2="{b[1]-16}" stroke="{PANEL_ED}" stroke-width="1"/>')
            o.append(text(bx + 70, by + 102, 'suppressed: far apart', 9.5,
                          MUTED, '400', 'middle'))

    o += caption(2, 0, ['both terms normalised to unit median, so λ is a',
                        'pure weight with one meaning across channels',
                        'swept over λ ∈ {0, 0.1, 0.5, 2}; results at λ = 0'])


def p4_hierarchy(o):
    x, y = px(3), PY
    base, top = y + 250, y + 76
    leaves = 16
    step = 300 / (leaves - 1)
    xs = [x + 30 + k * step for k in range(leaves)]
    hs = [base] * leaves
    rng = np.random.default_rng(5)
    nodes = list(range(leaves))
    lvl = 0
    while len(nodes) > 1:
        lvl += 1
        nxt, i = [], 0
        while i < len(nodes) - 1:
            a, b = nodes[i], nodes[i + 1]
            hgt = base - (base - top) * (lvl / 5.0) * rng.uniform(0.82, 1.0)
            o.append(f'<path d="M{xs[a]:.1f},{hs[a]:.1f} L{xs[a]:.1f},{hgt:.1f} '
                     f'L{xs[b]:.1f},{hgt:.1f} L{xs[b]:.1f},{hs[b]:.1f}" '
                     f'stroke="{LINE}" stroke-width="1.4" fill="none"/>')
            mid = (a + b) // 2
            xs[mid] = (xs[a] + xs[b]) / 2
            hs[mid] = hgt
            nxt.append(mid)
            i += 2
        if i == len(nodes) - 1:
            nxt.append(nodes[i])
        nodes = nxt

    o.append(f'<rect x="{x+22}" y="{y+112}" width="{PW-44}" height="104" rx="6" '
             f'fill="{AMBER}" fill-opacity="0.10" stroke="{AMBER}" '
             f'stroke-width="1.2" stroke-dasharray="5 3"/>')
    o.append(text(x + PW - 30, y + 128, 'candidate nodes', 10.5, '#b45309',
                  '600', 'end'))
    o.append(text(x + 30, y + 268, 'locations', 10.5, MUTED))
    o.append(text(x + PW - 30, y + 88, 'whole arena', 10.5, MUTED, '400', 'end'))

    o.append(f'<rect x="{x+16}" y="{y+284}" width="{PW-32}" height="44" rx="8" '
             f'fill="#ffffff" stroke="{PANEL_ED}"/>')
    o.append(mono(x + 28, y + 302, 'μ = mean of member features', 10.5, INK))
    o.append(mono(x + 28, y + 318, 'σ = P90 of within-node distances', 10.5, INK))

    o += caption(3, 0, ["Ward's criterion on the combined metric;",
                        'the combination is Euclidean, so Ward is valid'])


def p5_projection(o):
    x, y = px(0), ROW2_Y
    o.append(mono(x + PW / 2, y + 72, 'r(x) = exp( −‖f(x) − μ‖² / 2σ² )', 12,
                  INK, 'middle'))

    gx, gy, cell, n = x + 34, y + 92, 11, 13
    rng = np.random.default_rng(2)
    cxx, cyy = 6.0, 5.4
    for a in range(n):
        for b in range(n):
            d = hypot(a - cxx, b - cyy) / 4.4
            v = float(np.exp(-d * d)) * rng.uniform(0.92, 1.0)
            o.append(f'<rect x="{gx+a*cell}" y="{gy+b*cell}" width="{cell-1}" '
                     f'height="{cell-1}" fill="{BLUE}" fill-opacity="{v*0.85:.2f}"/>')
            if v >= 0.5:
                o.append(f'<rect x="{gx+a*cell}" y="{gy+b*cell}" width="{cell-1}" '
                         f'height="{cell-1}" fill="none" stroke="{INK}" '
                         f'stroke-width="0.9"/>')
    o.append(text(gx + n * cell / 2, gy + n * cell + 16, '0.25 m bins, per-bin max',
                  10.5, MUTED, '400', 'middle'))
    o.append(text(gx + n * cell / 2, gy + n * cell + 30, 'mask = ≥ 50% of own peak',
                  10.5, INK, '600', 'middle'))

    # split halves
    hx, hy = x + 218, y + 104
    for k, (lab, dx) in enumerate([('half A', 0), ('half B', 0)]):
        yy = hy + k * 74
        o.append(f'<rect x="{hx}" y="{yy}" width="104" height="56" rx="6" '
                 f'fill="#ffffff" stroke="{PANEL_ED}"/>')
        o.append(f'<ellipse cx="{hx+52+ (0 if k==0 else 5)}" cy="{yy+28}" '
                 f'rx="{30 if k==0 else 27}" ry="{17 if k==0 else 19}" '
                 f'fill="{GREEN}" fill-opacity="0.30" stroke="{GREEN}"/>')
        o.append(text(hx + 8, yy + 16, lab, 10.5, MUTED, '600'))
    o.append(f'<path d="M{hx+52},{hy+60} L{hx+52},{hy+72}" stroke="{FAINT}" '
             f'stroke-width="1.2" stroke-dasharray="3 2"/>')
    o.append(text(hx + 52, hy + 168, 'independent halves', 10.5, MUTED, '400', 'middle'))
    o.append(text(hx + 52, hy + 182, '→ split-half IoU', 10.5, INK, '600', 'middle'))

    o += caption(0, 1, ['unsampled bins take the max of their neighbours,',
                        'so sampling gaps cannot fragment a field'])


def p6_geometry(o):
    x, y = px(1), ROW2_Y
    cx, cy = x + 178, y + 168
    a, b, th = 96, 46, -22

    o.append(f'<circle cx="{cx}" cy="{cy}" r="{(a*b)**0.5:.1f}" fill="none" '
             f'stroke="{FAINT}" stroke-width="1.6" stroke-dasharray="5 4"/>')
    o.append(f'<ellipse cx="{cx}" cy="{cy}" rx="{a}" ry="{b}" '
             f'transform="rotate({th} {cx} {cy})" fill="{BLUE}" '
             f'fill-opacity="0.16" stroke="{BLUE}" stroke-width="2"/>')
    ar = th * pi / 180
    o.append(f'<line x1="{cx}" y1="{cy}" x2="{cx+a*cos(ar):.1f}" '
             f'y2="{cy+a*sin(ar):.1f}" stroke="{INK}" stroke-width="1.8"/>')
    o.append(f'<line x1="{cx}" y1="{cy}" x2="{cx-b*sin(ar):.1f}" '
             f'y2="{cy+b*cos(ar):.1f}" stroke="{INK}" stroke-width="1.8"/>')
    o.append(text(cx + a * cos(ar) * 0.62 + 4, cy + a * sin(ar) * 0.62 - 10, 'a', 14,
                  INK, '700', style='italic'))
    o.append(text(cx - b * sin(ar) * 0.62 - 17, cy + b * cos(ar) * 0.62 + 6, 'b', 14,
                  INK, '700', style='italic'))
    o.append(f'<line x1="{cx}" y1="{cy}" x2="{cx+70}" y2="{cy}" '
             f'stroke="{FAINT}" stroke-width="1.2" stroke-dasharray="3 3"/>')
    o.append(f'<path d="M{cx+44},{cy} A 44 44 0 0 0 {cx+44*cos(ar):.1f},'
             f'{cy+44*sin(ar):.1f}" stroke="{RED}" stroke-width="1.6" fill="none"/>')
    o.append(text(cx + 52, cy - 12, 'θ', 13, RED, '700', style='italic'))
    o.append(text(cx + 96, cy + 74, 'equivalent disc', 10.5, FAINT, '400', 'middle'))

    o.append(f'<rect x="{x+16}" y="{y+272}" width="{PW-32}" height="56" rx="8" '
             f'fill="#ffffff" stroke="{PANEL_ED}"/>')
    o.append(mono(x + 28, y + 292, 'semi-axes from mask second moments,', 10.5, INK))
    o.append(mono(x + 28, y + 308, 'rescaled so π a b = measured area', 10.5, INK))
    o.append(mono(x + 28, y + 324, 'elongation = a / b', 10.5, INK))

    o += caption(1, 1, ['a single radius understates one axis and',
                        'overstates the other — a measurement bias'])


def p7_admission(o):
    x, y = px(2), ROW2_Y
    rows_y = y + 56
    dy = 48

    def label(k, s, tag):
        o.append(text(x + 18, rows_y + k * dy + 14, s, 11.5, INK, '600'))
        o.append(text(x + 18, rows_y + k * dy + 28, tag, 10, MUTED))

    # 1 — area window
    label(0, 'area window', 'Rules 8 · 9')
    ax, aw = x + 150, 190
    o.append(f'<line x1="{ax}" y1="{rows_y+20}" x2="{ax+aw}" y2="{rows_y+20}" '
             f'stroke="{PANEL_ED}" stroke-width="5" stroke-linecap="round"/>')
    o.append(f'<line x1="{ax+aw*0.22}" y1="{rows_y+20}" x2="{ax+aw*0.80}" '
             f'y2="{rows_y+20}" stroke="{GREEN}" stroke-width="5" '
             f'stroke-linecap="round"/>')
    for fx, lab in ((0.22, '0.39 m²'), (0.80, '62.8 m²')):
        o.append(f'<line x1="{ax+aw*fx}" y1="{rows_y+13}" x2="{ax+aw*fx}" '
                 f'y2="{rows_y+27}" stroke="{INK}" stroke-width="1.6"/>')
        o.append(text(ax + aw * fx, rows_y + 40, lab, 9.5, MUTED, '400', 'middle'))

    # 2 — contiguity
    label(1, 'contiguity', 'Rule 1')
    bx = x + 150
    o.append(f'<ellipse cx="{bx+18}" cy="{rows_y+dy+18}" rx="15" ry="12" '
             f'fill="{RED}" fill-opacity="0.28" stroke="{RED}"/>')
    o.append(f'<ellipse cx="{bx+48}" cy="{rows_y+dy+18}" rx="9" ry="8" '
             f'fill="{RED}" fill-opacity="0.28" stroke="{RED}"/>')
    o.append(text(bx + 66, rows_y + dy + 23, '✗', 14, RED, '700'))
    o.append(f'<ellipse cx="{bx+126}" cy="{rows_y+dy+18}" rx="20" ry="13" '
             f'fill="{GREEN}" fill-opacity="0.28" stroke="{GREEN}"/>')
    o.append(text(bx + 152, rows_y + dy + 23, '✓', 14, GREEN, '700'))
    o.append(text(bx + 172, rows_y + dy + 22, '≥ 80%', 9.5, MUTED))

    # 3 — reliability
    label(2, 'reliability', 'Rule 2')
    cx0 = x + 168
    o.append(f'<circle cx="{cx0}" cy="{rows_y+2*dy+18}" r="15" fill="{BLUE}" '
             f'fill-opacity="0.28" stroke="{BLUE}"/>')
    o.append(f'<circle cx="{cx0+18}" cy="{rows_y+2*dy+18}" r="15" fill="{GREEN}" '
             f'fill-opacity="0.28" stroke="{GREEN}"/>')
    o.append(text(cx0 + 46, rows_y + 2 * dy + 22, 'IoU ≥ 0.40', 10.5, MUTED))

    # 4 — same-scale competition
    label(3, 'competition', 'Rule 11')
    dx0 = x + 158
    o.append(f'<circle cx="{dx0}" cy="{rows_y+3*dy+18}" r="14" fill="{VIOLET}" '
             f'fill-opacity="0.28" stroke="{VIOLET}"/>')
    o.append(f'<circle cx="{dx0+16}" cy="{rows_y+3*dy+18}" r="14" fill="none" '
             f'stroke="{FAINT}" stroke-dasharray="3 3"/>')
    o.append(text(dx0 + 36, rows_y + 3 * dy + 14, 'same band → one wins', 9.5, MUTED))
    o.append(text(dx0 + 36, rows_y + 3 * dy + 30, 'nested → both kept', 9.5, MUTED))
    ng = x + PW - 36
    o.append(f'<circle cx="{ng}" cy="{rows_y+3*dy+20}" r="14" fill="none" '
             f'stroke="{AMBER}" stroke-width="1.5"/>')
    o.append(f'<circle cx="{ng}" cy="{rows_y+3*dy+20}" r="6.5" fill="{AMBER}" '
             f'fill-opacity="0.35" stroke="{AMBER}"/>')

    # 5 — coverage
    label(4, 'coverage', 'Rule 12')
    ex, bwid = x + 150, 20
    for k, v in enumerate([0.16, 0.42, 0.66, 0.84, 0.52, 0.30]):
        ok = v >= 0.5
        hgt = 40 * v
        o.append(f'<rect x="{ex+k*(bwid+6)}" y="{rows_y+4*dy+40-hgt:.1f}" '
                 f'width="{bwid}" height="{hgt:.1f}" rx="2" '
                 f'fill="{GREEN if ok else "#cbd5e1"}" fill-opacity="0.8"/>')
    o.append(f'<line x1="{ex-4}" y1="{rows_y+4*dy+20}" x2="{ex+6*(bwid+6)}" '
             f'y2="{rows_y+4*dy+20}" stroke="{RED}" stroke-width="1.4" '
             f'stroke-dasharray="4 3"/>')
    o.append(text(ex + 6 * (bwid + 6) + 6, rows_y + 4 * dy + 24, '50%', 9.5, RED, '600'))
    o.append(text(ex, rows_y + 4 * dy + 54, 'scale bands (ratio 1.6)', 9.5, MUTED))

    o += caption(2, 1, ['bands group fields for the last two criteria only —',
                        'never to admit or reject one, so ladder spacing',
                        'is measured rather than imposed'])


def p8_bank(o):
    x, y = px(3), ROW2_Y
    cx, cy, R = x + 178, y + 190, 112
    o.append(f'<circle cx="{cx}" cy="{cy}" r="{R}" fill="#ffffff" '
             f'stroke="{INK}" stroke-width="2.5"/>')
    rng = np.random.default_rng(17)
    bands = [(12, GREEN), (21, BLUE), (34, VIOLET)]
    for rad, col in reversed(bands):
        n = max(3, int(110 / rad))
        Reff = max(R - rad * 1.25, 6)
        for k in range(n):
            # golden-angle spiral: even coverage without a visible lattice
            a = k * 2.39996 + rng.uniform(-0.25, 0.25)
            rr = Reff * ((k + 0.5) / n) ** 0.5
            ex, ey = cx + rr * cos(a), cy + rr * sin(a)
            wall = R - hypot(ex - cx, ey - cy)
            # near-wall fields drawn elongated along the wall: the shape
            # prediction the model is built to test, not to produce
            elong = 1.0 + 1.9 * max(0.0, 1 - wall / (0.42 * R))
            tang = np.degrees(np.arctan2(ey - cy, ex - cx)) + 90
            o.append(f'<ellipse cx="{ex:.1f}" cy="{ey:.1f}" '
                     f'rx="{rad*elong**0.5:.1f}" ry="{rad/elong**0.5:.1f}" '
                     f'transform="rotate({tang:.1f} {ex:.1f} {ey:.1f})" '
                     f'fill="{col}" fill-opacity="0.13" stroke="{col}" '
                     f'stroke-width="1.3"/>')
    for k, (rad, col) in enumerate(bands):
        yy = y + 66 + k * 17
        o.append(f'<ellipse cx="{x+30}" cy="{yy}" rx="9" ry="6" fill="{col}" '
                 f'fill-opacity="0.2" stroke="{col}" stroke-width="1.3"/>')
        o.append(text(x + 46, yy + 4, f'band {k+1}', 10, MUTED))

    o += caption(3, 1, ['one row per field: area, semi-axes, orientation,',
                        'centre, scale band, contiguity, split-half IoU,',
                        'and distance to the wall — recorded, never used'])


# ================================================================ assembly

def build():
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
         f'font-family="-apple-system, \'Segoe UI\', Helvetica, Arial, sans-serif">',
         '<defs>',
         f'<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" '
         f'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
         f'<path d="M0,0 L10,5 L0,10 z" fill="{LINE}"/></marker>',
         '</defs>',
         f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
         text(W / 2, 46, 'Visual place-field model — pipeline', 25, INK, '600', 'middle'),
         text(W / 2, 72,
              'from range-limited sensing to a rule-admitted bank of anisotropic place fields',
              14, MUTED, '400', 'middle')]

    o += panel(0, 0, 1, 'Acquisition')
    o += panel(1, 0, 2, 'Feature channels')
    o += panel(2, 0, 3, 'Merge metric', 'Rule 4')
    o += panel(3, 0, 4, 'Ward hierarchy')
    o += panel(0, 1, 5, 'Projection')
    o += panel(1, 1, 6, 'Field geometry', 'Rule 7')
    o += panel(2, 1, 7, 'Admission')
    o += panel(3, 1, 8, 'Field bank')

    p1_acquisition(o); p2_channels(o); p3_metric(o); p4_hierarchy(o)
    p5_projection(o); p6_geometry(o); p7_admission(o); p8_bank(o)

    # in-row arrows
    for i in range(3):
        o.append(arrow(px(i) + PW + 5, PY + PH / 2, px(i + 1) - 6, PY + PH / 2))
        o.append(arrow(px(i) + PW + 5, ROW2_Y + PH / 2, px(i + 1) - 6, ROW2_Y + PH / 2))
    # wrap from the end of row 1 to the start of row 2
    wy = PY + PH + 40
    o.append(f'<path d="M{px(3)+PW/2},{PY+PH+6} L{px(3)+PW/2},{wy} '
             f'L{px(0)+PW/2},{wy} L{px(0)+PW/2},{ROW2_Y-8}" stroke="{LINE}" '
             f'stroke-width="2" fill="none" marker-end="url(#arrow)"/>')

    o.append(text(W / 2, H - 22,
                  'Distance to the wall enters no merge or admission decision; '
                  'arena geometry reaches the model only through total area.',
                  12, MUTED, '400', 'middle'))
    o.append('</svg>')
    return '\n'.join(o)


if __name__ == '__main__':
    out = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUT
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        f.write(build())
    print(f'wrote {out}')

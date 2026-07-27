#!/usr/bin/env python
"""Generate a true vector SVG of the EFA factor-analysis path diagram (Fig4 panel A).

Source of truth: figs/EFA_fig.html (the "Tuned" interactive canvas+DOM version the
user maintains). This script re-authors the SAME diagram as native SVG so it can be
combined with the other Fig4 components into one vector figure, instead of being
embedded as a raster screenshot.

Outputs:
  figs/EFA_fig.svg     - vector master (native <ellipse>/<line>/<text>)
  figs/EFA_fig.png     - raster preview / embedding (Pillow, no SVG rasterizer needed)

Run:  python 44fig4_efa_svg.py
"""
from __future__ import annotations

import html
import math
import os
import re
import xml.etree.ElementTree as ET
import xml.sax.saxutils as su

# ---------------------------------------------------------------------------
# DATA  (extracted verbatim from figs/EFA_fig.html CONFIG / factors / rawLoadings / correlations)
# ---------------------------------------------------------------------------
FACTORS = {
    "DC":  dict(label=["Decision", "Caution", "(18.1%)"],     c0="#3b6cb3", c1="#1e407c"),
    "NDT": dict(label=["Non-decision", "Time", "(16.2%)"],    c0="#7E6148", c1="#543F2D"),
    "CP":  dict(label=["Processing", "Efficiency", "(28.2%)"], c0="#8491B4", c1="#5F6B8D"),
    "IP":  dict(label=["Inhibitory", "Process", "(11.1%)"],   c0="#0099B4", c1="#006678"),
}
FACTOR_ORDER = ["DC", "NDT", "CP", "IP"]

LOADINGS = [
    ("CP", "$p|SSP$", 0.981),
    ("CP", "$v_{incong}|DDM$", 0.978),
    ("CP", "$v_{cong}|DDM$", 0.874),
    ("CP", "$v_{ss}|DSTP$", 0.841),
    ("CP", "$v_{c}|DMC$", 0.813),
    ("CP", "$r_d|SSP$", 0.683),
    ("CP", "$v_{ta}|DSTP$", 0.674),
    ("CP", "$a_{ss}|DSTP$", -0.467),
    ("DC", "$a|SSP$", 0.983),
    ("DC", "$a|DDM$", 0.932),
    ("DC", "$a|DSTP$", 0.823),
    ("DC", "$a|DMC$", 0.564),
    ("DC", "$a_{ss}|DSTP$", 0.523),
    ("DC", "$v_{p2}|DSTP$", -0.477),
    ("NDT", "$a|DMC$", 0.525),
    ("NDT", "$t|SSP$", 0.820),
    ("NDT", "$t|DSTP$", 0.811),
    ("NDT", "$t|DDM$", 0.796),
    ("NDT", "$t|DMC$", 0.634),
    ("IP", "$v_{fl}|DSTP$", 0.785),
    ("IP", "$sd_a|SSP$", 0.629),
    ("IP", "$\\tau|DMC$", 0.539),
    ("IP", "$\\alpha|DMC$", 0.507),
    ("IP", "$\\eta|DMC$", 0.484),
]

CORRS = [
    ("CP", "DC", -0.226),
    ("CP", "NDT", 0.196),
    ("CP", "IP", 0.090),
    ("DC", "NDT", 0.125),
    ("DC", "IP", 0.146),
    ("NDT", "IP", -0.125),
]

# layout config (from HTML CONFIG, widened for readability)
L = dict(
    factorGapX=220,      # 因子椭圆之间的水平间距（px）
    factorWidth=140,     # 因子椭圆宽
    factorHeight=100,    # 因子椭圆高
    paramStepY=45,       # 参数盒之间的水平间距（px）
    paramHeight=33,      # 参数盒高（旋转前的高）
    paramWidth=90,       # 参数盒宽（旋转前的宽）
    factorRowY=0.38,     # 因子行纵向位置（0-1，越小越靠上）
    paramRowY=0.80,      # 参数行纵向位置（0-1，越大越靠下）
    corrArchHeight=0.25, # 相关弧线高度系数（越大弧越高）
    labelMinDist=34,     # 加载值标签最小间距（dodge 算法）
    labelDodgeStrength=0.9,
    baseWidth=1,         # 连线基础宽度
    widthScale=8,        # 连线宽度随加载值缩放系数
    labelToEnd=0.50,     # 加载值标签在线上的位置（0-1，0.5=中点）
)
COL = dict(pos="#425E77", neg="#D64045", corr="#919191")

# ================================================================
# FONT SIZES — all text sizes in one place for easy tuning
# ================================================================
FONT = dict(
    # --- EFA factor ellipse labels ---
    factor_main=16,        # factor name line 1-2 (e.g., "Decision Caution")
    factor_sub=13,         # factor percentage line 3 (e.g., "(18.1%)")
    # --- Parameter box labels (rotated -90°) ---
    param_main=14,         # parameter symbol base size (pt)
    param_sub=0.7,         # subscript size ratio (em, relative to param_main)
    param_method=0.82,     # model suffix size ratio (em, relative to param_main)
    # --- Loading & correlation value labels ---
    loading_val=11,        # loading value labels on connection lines
    corr_val=12,           # correlation value labels on arcs
)

# ================================================================
# FIG4 COMPOSE — layout knobs for stacking EFA (panel A) above grid
# ================================================================
FIG4 = dict(
    efa_pad_left=40,       # pt — EFA panel left padding (shifts EFA right)
    efa_pad_right=8,       # pt — EFA panel right padding (pulls EFA right edge in)
    gap=0,                 # pt — vertical gap between EFA panel and grid
    label_a_size=14,       # pt — A panel label font size
    label_a_dx=4,          # pt — A label offset right from EFA panel left edge
    label_a_dy=45,         # pt — A label offset down from EFA panel top
)

# Canvas: 1200 makes EFA content fill ~99% of width so its right edge
# roughly aligns with the 3x3 grid's right edge in Fig4_final
SVG_W, SVG_H = 1100, 500

# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def compute_scene():
    cx = SVG_W / 2.0
    fy = SVG_H * L["factorRowY"]
    factorPos = {}
    for i, k in enumerate(FACTOR_ORDER):
        factorPos[k] = dict(x=cx + (i - 1.5) * L["factorGapX"], y=fy)

    # dominant-factor ordering of unique params (mirrors organizeParameters)
    unique = list(dict.fromkeys(p for (_f, p, _v) in LOADINGS))
    def dom_factor(p):
        best = None
        for (f, pp, v) in LOADINGS:
            if pp == p and (best is None or abs(v) > abs(best[1])):
                best = (f, v)
        return best
    params = []
    for p in unique:
        f, v = dom_factor(p)
        params.append((p, f, abs(v)))
    order = {f: i for i, f in enumerate(FACTOR_ORDER)}
    params.sort(key=lambda t: (order[t[1]], -t[2]))
    sorted_params = [p for (p, _f, _v) in params]

    n = len(sorted_params)
    start_px = cx - (n - 1) * L["paramStepY"] / 2.0
    py = SVG_H * L["paramRowY"]
    paramPos = {p: dict(x=start_px + i * L["paramStepY"], y=py)
                for i, p in enumerate(sorted_params)}

    a = L["factorWidth"] / 2.0
    b = L["factorHeight"] / 2.0

    lines = []
    labels = []
    for idx, (f, p, v) in enumerate(LOADINGS):
        fp = factorPos[f]
        pp = paramPos[p]
        targetY = pp["y"] - L["paramWidth"] / 2.0
        dx = pp["x"] - fp["x"]
        dy = targetY - fp["y"]
        ang = math.atan2(dy, dx)
        r = (a * b) / math.sqrt((b * math.cos(ang)) ** 2 + (a * math.sin(ang)) ** 2)
        sx = fp["x"] + r * math.cos(ang)
        sy = fp["y"] + r * math.sin(ang)
        ex, ey = pp["x"], targetY
        lines.append(dict(sx=sx, sy=sy, ex=ex, ey=ey, val=v))
        labels.append(dict(t=L["labelToEnd"], val=v, id=idx))

    # dodge: slide labels ALONG their own lines (never off-line like HTML original)
    md = L["labelMinDist"]
    fs = L["labelDodgeStrength"]
    for lbl, ln in zip(labels, lines):
        ln_len = math.hypot(ln["ex"] - ln["sx"], ln["ey"] - ln["sy"])
        lbl.update(ux=(ln["ex"] - ln["sx"]) / ln_len,
                   uy=(ln["ey"] - ln["sy"]) / ln_len, ln=ln, len=ln_len)

    def _pos(lbl):
        ln = lbl["ln"]
        return (ln["sx"] + (ln["ex"] - ln["sx"]) * lbl["t"],
                ln["sy"] + (ln["ey"] - ln["sy"]) * lbl["t"])

    for _ in range(80):
        moved = False
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                l1, l2 = labels[i], labels[j]
                x1, y1 = _pos(l1); x2, y2 = _pos(l2)
                dx = x1 - x2; dy = y1 - y2
                dist = math.hypot(dx, dy)
                if dist < md:
                    moved = True
                    if dist == 0:
                        dx, dy, dist = 1, 0, 1
                    nx, ny = dx / dist, dy / dist
                    push = (md - dist) * fs * 0.5
                    l1["t"] += (nx * l1["ux"] + ny * l1["uy"]) * push / l1["len"]
                    l2["t"] -= (nx * l2["ux"] + ny * l2["uy"]) * push / l2["len"]
                    l1["t"] = min(0.80, max(0.20, l1["t"]))
                    l2["t"] = min(0.80, max(0.20, l2["t"]))
        if not moved:
            break

    for lbl in labels:
        lbl["x"], lbl["y"] = _pos(lbl)
        lbl["ox"], lbl["oy"] = lbl["x"], lbl["y"]

    # correlation arcs
    arcs = []
    for (f1, f2, v) in CORRS:
        if f1 not in factorPos or f2 not in factorPos:
            continue
        p1, p2 = factorPos[f1], factorPos[f2]
        dist = abs(p2["x"] - p1["x"])
        arch = max(25, dist * L["corrArchHeight"])
        startY = p1["y"] - L["factorHeight"] / 2.0
        midX = (p1["x"] + p2["x"]) / 2.0
        midY = startY - arch
        endY = p2["y"] - L["factorHeight"] / 2.0
        # bezier midpoint (t=0.5)
        valY = 0.25 * startY + 0.5 * midY + 0.25 * endY
        arcs.append(dict(x1=p1["x"], y1=startY, mx=midX, my=midY,
                         x2=p2["x"], y2=endY, val=v, valY=valY))

    # ---- compute tight bounding box of all content ----
    margin = 5  # px padding around content (tight)
    xs, ys = [], []
    for k, fp in factorPos.items():
        xs.extend([fp["x"] - L["factorWidth"]/2, fp["x"] + L["factorWidth"]/2])
        ys.extend([fp["y"] - L["factorHeight"]/2, fp["y"] + L["factorHeight"]/2])
    for p in sorted_params:
        pp = paramPos[p]
        xs.extend([pp["x"] - 17.5, pp["x"] + 17.5])
        ys.extend([pp["y"] - 45, pp["y"] + 45])
    for arc in arcs:
        ys.append(arc["valY"])  # arc label position
        ys.append(arc["my"])    # arc control point (approx peak)
    for lbl in labels:
        xs.append(lbl["x"]); ys.append(lbl["y"])
    bx0 = max(0, min(xs) - margin)
    by0 = max(0, min(ys) - margin)
    bx1 = min(SVG_W, max(xs) + margin)
    by1 = min(SVG_H, max(ys) + margin)

    return dict(factorPos=factorPos, paramPos=paramPos, sorted_params=sorted_params,
                lines=lines, labels=labels, arcs=arcs,
                bbox=(bx0, by0, bx1 - bx0, by1 - by0))


def parse_param(s):
    """Return list of (kind, text): kind in {'m' main, 's' subscript, 'x' method}."""
    s = s.strip().strip("$")
    if "|" in s:
        param, method = s.split("|", 1)
    else:
        param, method = s, ""
    greek = {"tau": "\u03c4", "alpha": "\u03b1", "eta": "\u03b7"}
    for k, v in greek.items():
        param = param.replace("\\" + k, v)
    segs = []
    cur = ""
    idx = 0
    while idx < len(param):
        c = param[idx]
        if c == "_":
            if idx + 1 < len(param) and param[idx + 1] == "{":
                end = param.index("}", idx + 2)
                sub = param[idx + 2:end]
                if cur:
                    segs.append(("m", cur)); cur = ""
                segs.append(("s", sub))
                idx = end + 1
            else:
                sub = param[idx + 1]
                if cur:
                    segs.append(("m", cur)); cur = ""
                segs.append(("s", sub))
                idx += 2
        else:
            cur += c
            idx += 1
    if cur:
        segs.append(("m", cur))
    if method:
        segs.append(("x", method))
    return segs


# ---------------------------------------------------------------------------
# SVG emission
# ---------------------------------------------------------------------------
def emit_svg(scene):
    bx, by, bw, bh = scene["bbox"]
    parts = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{bw:.0f}" height="{bh:.0f}" viewBox="{bx:.0f} {by:.0f} {bw:.0f} {bh:.0f}" '
        f'font-family="Segoe UI, Arial, sans-serif">'
    )
    # defs: radial gradients for factors + param chip + soft shadow
    parts.append("<defs>")
    for k, d in FACTORS.items():
        parts.append(
            f'<radialGradient id="efa_grad_{k}" cx="30%" cy="30%" r="75%">'
            f'<stop offset="0%" stop-color="{d["c0"]}"/>'
            f'<stop offset="100%" stop-color="{d["c1"]}"/>'
            f'</radialGradient>'
        )
    parts.append(
        '<linearGradient id="efa_param_grad" x1="0" y1="0" x2="0" y2="1">'
        '<stop offset="0%" stop-color="#ffffff"/>'
        '<stop offset="100%" stop-color="#f0f2f5"/>'
        '</linearGradient>'
    )
    parts.append(
        '<filter id="efa_shadow" x="-20%" y="-20%" width="140%" height="140%">'
        '<feDropShadow dx="0" dy="3" stdDeviation="3" '
        'flood-color="rgba(0,0,0,0.18)"/>'
        '</filter>'
    )
    parts.append("</defs>")

    # correlation arcs
    for arc in scene["arcs"]:
        w = max(1.0, abs(arc["val"]) * L["widthScale"] * 1.5)
        col = COL["corr"]
        parts.append(
            f'<path d="M {arc["x1"]:.1f} {arc["y1"]:.1f} '
            f'Q {arc["mx"]:.1f} {arc["my"]:.1f} {arc["x2"]:.1f} {arc["y2"]:.1f}" '
            f'fill="none" stroke="{col}" stroke-width="{w:.2f}" '
            f'stroke-opacity="0.8"/>'
        )
        parts.append(
            f'<text x="{arc["mx"]:.1f}" y="{arc["valY"]:.1f}" '
            f'text-anchor="middle" dominant-baseline="middle" '
            f'font-size="{FONT["corr_val"]}" font-weight="bold" fill="{col}">'
            f'{arc["val"]:+.2f}</text>'
        )

    # loadings
    for ln in scene["lines"]:
        neg = ln["val"] < 0
        col = COL["neg"] if neg else COL["pos"]
        w = L["baseWidth"] + abs(ln["val"]) * L["widthScale"]
        if neg:
            w *= 0.6
        dash = ' stroke-dasharray="8 4"' if neg else ""
        alpha = 0.6 if neg else 0.8
        parts.append(
            f'<line x1="{ln["sx"]:.1f}" y1="{ln["sy"]:.1f}" '
            f'x2="{ln["ex"]:.1f}" y2="{ln["ey"]:.1f}" '
            f'stroke="{col}" stroke-width="{w:.2f}"{dash} stroke-opacity="{alpha}"/>'
        )

    # loading value labels (white chip + text)
    for lbl in scene["labels"]:
        neg = lbl["val"] < 0
        col = COL["neg"] if neg else COL["pos"]
        # connector if dodged far
        dm = math.hypot(lbl["x"] - lbl["ox"], lbl["y"] - lbl["oy"])
        if dm > 12:
            parts.append(
                f'<line x1="{lbl["ox"]:.1f}" y1="{lbl["oy"]:.1f}" '
                f'x2="{lbl["x"]:.1f}" y2="{lbl["y"]:.1f}" '
                f'stroke="{col}" stroke-width="0.5" stroke-opacity="0.5"/>'
            )
        tx, ty = lbl["x"], lbl["y"]
        parts.append(
            f'<rect x="{tx-15:.1f}" y="{ty-9:.1f}" width="30" height="18" rx="3" '
            f'fill="#ffffff" fill-opacity="0.95" stroke="rgba(0,0,0,0.05)"/>'
        )
        parts.append(
            f'<text x="{tx:.1f}" y="{ty:.1f}" text-anchor="middle" '
            f'dominant-baseline="middle" font-size="{FONT["loading_val"]}" font-weight="bold" '
            f'fill="{col}">{lbl["val"]:+.2f}</text>'
        )

    # factor ellipses
    for k in FACTOR_ORDER:
        d = FACTORS[k]
        fp = scene["factorPos"][k]
        rx, ry = L["factorWidth"] / 2.0, L["factorHeight"] / 2.0
        parts.append(
            f'<ellipse cx="{fp["x"]:.1f}" cy="{fp["y"]:.1f}" rx="{rx}" ry="{ry}" '
            f'fill="url(#efa_grad_{k})" stroke="rgba(255,255,255,0.2)" '
            f'stroke-width="1" filter="url(#efa_shadow)"/>'
        )
        # text (3 lines)
        ty0 = fp["y"] - 14
        for li, line in enumerate(d["label"]):
            fsize = FONT["factor_main"] if li < 2 else FONT["factor_sub"]
            parts.append(
                f'<text x="{fp["x"]:.1f}" y="{ty0 + li*18:.1f}" '
                f'text-anchor="middle" dominant-baseline="middle" '
                f'font-size="{fsize}" font-weight="bold" fill="#ffffff" '
                f'fill-opacity="0.95">{html.escape(line)}</text>'
            )

    # parameter boxes (rotated -90 text)
    for p in scene["sorted_params"]:
        pp = scene["paramPos"][p]
        bx, by = pp["x"] - 17.5, pp["y"] - 45.0
        parts.append(
            f'<rect x="{bx:.1f}" y="{by:.1f}" width="35" height="90" rx="8" '
            f'fill="url(#efa_param_grad)" stroke="#ced4da" stroke-width="1"/>'
        )
        # label as rotated text
        segs = parse_param(p)
        tspans = []
        for kind, txt in segs:
            if kind == "m":
                tspans.append(f'<tspan fill="#333333">{html.escape(txt)}</tspan>')
            elif kind == "s":
                tspans.append(
                    f'<tspan fill="#333333" font-size="{FONT["param_sub"]}em" '
                    f'baseline-shift="sub">{html.escape(txt)}</tspan>')
            else:  # method
                tspans.append(
                    f'<tspan fill="#666666" font-size="{FONT["param_method"]}em">|{html.escape(txt)}</tspan>')
        parts.append(
            f'<text x="{pp["x"]:.1f}" y="{pp["y"]:.1f}" '
            f'text-anchor="middle" dominant-baseline="middle" '
            f'font-size="{FONT["param_main"]}" font-weight="600" '
            f'transform="rotate(-90 {pp["x"]:.1f} {pp["y"]:.1f})">'
            f'{"".join(tspans)}</text>'
        )

    parts.append("</svg>")
    return "\n".join(parts)


def _hex(h):
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def svg_to_png(svg_path, png_path, width, height, background="#f4f6f8"):
    """Convert SVG to PNG using resvg (faithful, no external deps)."""
    import resvg_py
    png_bytes = resvg_py.svg_to_bytes(
        svg_path=svg_path,
        width=width, height=height,
        background=background,
    )
    with open(png_path, "wb") as f:
        f.write(png_bytes)
    return png_path


def compose_fig4(scene):
    """Stack EFA (panel A) above the 3×3 reliability+factor-space grid.
    Produces both vector SVG and raster PNG (rendered directly from SVG via resvg)."""
    here = os.path.dirname(os.path.abspath(__file__))
    figs = os.path.join(here, "..", "figs")
    grid_svg_path = os.path.join(figs, "44fig4_v8_combined_3x3.svg")

    # ---- read grid dimensions ----
    with open(grid_svg_path, encoding="utf-8") as f:
        grid_svg_text = f.read()
    m = re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', grid_svg_text)
    if m:
        gw_pt, gh_pt = float(m.group(1)), float(m.group(2))
    else:
        raise RuntimeError(f"Cannot parse viewBox from {grid_svg_path}")

    # ---- compose SVG ----
    def _inner(svg_text):
        i = svg_text.index(">", svg_text.index("<svg"))
        j = svg_text.rindex("</svg>")
        return svg_text[i + 1:j].strip()

    efa_inner = _inner(emit_svg(scene))
    grid_inner = _inner(grid_svg_text)

    # EFA bbox-based dimensions
    bx, by, bw, bh = scene["bbox"]
    efa_aspect = bh / bw  # height/width ratio of EFA content

    # ================================================================
    # Fig4 LAYOUT — sourced from FIG4 dict at top of file
    # ================================================================
    efa_pad_left = FIG4["efa_pad_left"]
    efa_pad_right = FIG4["efa_pad_right"]
    gap_pt = FIG4["gap"]
    label_a_dx = FIG4["label_a_dx"]
    label_a_dy = FIG4["label_a_dy"]
    label_a_size = FIG4["label_a_size"]
    # ================================================================

    efa_w = gw_pt - efa_pad_left - efa_pad_right
    efa_scaled_h = efa_w * efa_aspect
    total_h = efa_scaled_h + gap_pt + gh_pt

    # A label overlaps the EFA panel (drawn LAST so it sits on top),
    # positioned in outer-figure pt coords at the EFA panel's top-left.
    label_x = efa_pad_left + label_a_dx
    label_y = label_a_dy

    final_svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{gw_pt:.2f}" height="{total_h:.2f}" '
        f'viewBox="0 0 {gw_pt:.2f} {total_h:.2f}" '
        f'font-family="Segoe UI, Arial, sans-serif">\n'
        f'<!-- Panel A: EFA factor-analysis path diagram -->\n'
        f'<svg x="{efa_pad_left}" y="0" width="{efa_w:.2f}" height="{efa_scaled_h:.2f}" '
        f'viewBox="{bx:.0f} {by:.0f} {bw:.0f} {bh:.0f}">\n'
        f'{efa_inner}\n'
        f'</svg>\n'
        f'<!-- Panels B-J: Reliability + Factor Space (3x3) -->\n'
        f'<svg x="0" y="{efa_scaled_h + gap_pt:.2f}" '
        f'width="{gw_pt:.2f}" height="{gh_pt:.2f}" '
        f'viewBox="0 0 {gw_pt:.2f} {gh_pt:.2f}">\n'
        f'{grid_inner}\n'
        f'</svg>\n'
        f'<!-- Panel A label (on top, overlapping EFA top-left) -->\n'
        f'<text x="{label_x}" y="{label_y}" font-size="{label_a_size}" '
        f'font-weight="bold" fill="#333333">A</text>\n'
        f'</svg>'
    )
    out_svg = os.path.join(figs, "Fig4_final.svg")
    with open(out_svg, "w", encoding="utf-8") as f:
        f.write(final_svg)
    print("wrote", os.path.normpath(out_svg))

    # ---- render final PNG directly from final SVG via resvg ----
    out_png = os.path.join(figs, "Fig4_final.png")
    scale = 300 / 72  # 300 DPI target
    svg_to_png(out_svg, out_png,
               width=int(gw_pt * scale), height=int(total_h * scale),
               background="#ffffff")
    print("wrote", os.path.normpath(out_png))


def main():
    scene = compute_scene()
    here = os.path.dirname(os.path.abspath(__file__))
    figs = os.path.join(here, "..", "figs")

    svg = emit_svg(scene)
    out_svg = os.path.join(figs, "EFA_fig.svg")
    with open(out_svg, "w", encoding="utf-8") as f:
        f.write(svg)
    print("wrote", os.path.normpath(out_svg))

    out_png = os.path.join(figs, "EFA_fig.png")
    bx, by, bw, bh = scene["bbox"]
    svg_to_png(out_svg, out_png, width=int(bw * 2), height=int(bh * 2))
    print("wrote", os.path.normpath(out_png))

    # Compose full Fig4
    compose_fig4(scene)


if __name__ == "__main__":
    main()

"""
mplot3 — colormaps, colorbar and the 3-D axes renderer (surf / mesh / plot3 /
contour3 / scatter3) for SimForge's DEMO figures.

Everything is emitted as SVG fragments appended to a list `p`, exactly like
mdemo's 2-D renderer. This module imports nothing from mdemo so it can be
tested on its own; the few tiny helpers it needs are duplicated here.

Projection: MATLAB's view(az, el). az is measured from the -y axis, counter-
clockwise seen from above; el is the elevation above the x-y plane.
    C = surface -> camera      = ( cos(el) sin(az), -cos(el) cos(az), sin(el) )
    R = screen right           = ( cos(az),          sin(az),          0       )
    U = screen up              = ( -sin(el) sin(az),  sin(el) cos(az), cos(el) )
Default view is (-37.5, 30), like MATLAB.
"""

import math

# ---------------------------------------------------------------------------
# Colormaps (stop lists sampled from MATLAB / matplotlib)
# ---------------------------------------------------------------------------
COLORMAPS = {
    "parula": [(0.0, (0.2422, 0.1504, 0.6603)), (0.125, (0.2810, 0.3228, 0.9579)),
               (0.25, (0.1786, 0.5289, 0.9682)), (0.375, (0.0689, 0.6948, 0.8394)),
               (0.5, (0.2161, 0.7843, 0.5923)), (0.625, (0.6720, 0.7793, 0.2227)),
               (0.75, (0.9970, 0.7659, 0.2199)), (0.875, (0.9892, 0.8136, 0.1885)),
               (1.0, (0.9769, 0.9839, 0.0805))],
    "jet": [(0.0, (0.0, 0.0, 0.5)), (0.125, (0.0, 0.0, 1.0)), (0.375, (0.0, 1.0, 1.0)),
            (0.625, (1.0, 1.0, 0.0)), (0.875, (1.0, 0.0, 0.0)), (1.0, (0.5, 0.0, 0.0))],
    "hot": [(0.0, (0.04, 0.0, 0.0)), (0.375, (1.0, 0.0, 0.0)), (0.75, (1.0, 1.0, 0.0)),
            (1.0, (1.0, 1.0, 1.0))],
    "cool": [(0.0, (0.0, 1.0, 1.0)), (1.0, (1.0, 0.0, 1.0))],
    "gray": [(0.0, (0.0, 0.0, 0.0)), (1.0, (1.0, 1.0, 1.0))],
    "hsv": [(0.0, (1, 0, 0)), (1 / 6, (1, 1, 0)), (2 / 6, (0, 1, 0)), (3 / 6, (0, 1, 1)),
            (4 / 6, (0, 0, 1)), (5 / 6, (1, 0, 1)), (1.0, (1, 0, 0))],
    "turbo": [(0.0, (0.19, 0.07, 0.23)), (0.25, (0.27, 0.63, 0.99)), (0.5, (0.63, 0.98, 0.25)),
              (0.75, (0.99, 0.53, 0.08)), (1.0, (0.48, 0.02, 0.01))],
    "viridis": [(0.0, (0.267, 0.005, 0.329)), (0.25, (0.229, 0.322, 0.546)),
                (0.5, (0.128, 0.567, 0.551)), (0.75, (0.369, 0.789, 0.383)),
                (1.0, (0.993, 0.906, 0.144))],
    "autumn": [(0.0, (1.0, 0.0, 0.0)), (1.0, (1.0, 1.0, 0.0))],
    "winter": [(0.0, (0.0, 0.0, 1.0)), (1.0, (0.0, 1.0, 0.5))],
    "spring": [(0.0, (1.0, 0.0, 1.0)), (1.0, (1.0, 1.0, 0.0))],
    "summer": [(0.0, (0.0, 0.5, 0.4)), (1.0, (1.0, 1.0, 0.4))],
    "bone": [(0.0, (0.0, 0.0, 0.0)), (0.375, (0.32, 0.32, 0.44)), (0.75, (0.65, 0.77, 0.77)),
             (1.0, (1.0, 1.0, 1.0))],
    "copper": [(0.0, (0.0, 0.0, 0.0)), (0.8, (1.0, 0.62, 0.40)), (1.0, (1.0, 0.78, 0.50))],
}
COLORMAPS["default"] = COLORMAPS["parula"]


def cmap_color(name, t):
    stops = COLORMAPS.get((name or "parula").lower(), COLORMAPS["parula"])
    t = 0.0 if not math.isfinite(t) else max(0.0, min(1.0, t))
    for k in range(len(stops) - 1):
        t0, c0 = stops[k]
        t1, c1 = stops[k + 1]
        if t0 <= t <= t1:
            f = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
            return "#%02X%02X%02X" % tuple(
                max(0, min(255, int(round(255 * (c0[m] + f * (c1[m] - c0[m])))))) for m in range(3))
    c = stops[-1][1]
    return "#%02X%02X%02X" % tuple(int(round(255 * v)) for v in c)


# ---------------------------------------------------------------------------
# small helpers (duplicated from mdemo on purpose)
# ---------------------------------------------------------------------------
_AX = "#262626"
_GRID = "#d9d9d9"
_FONT = "Helvetica, Arial, sans-serif"


def _esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def nice_axis(lo, hi, target=5):
    if not (math.isfinite(lo) and math.isfinite(hi)):
        lo, hi = 0.0, 1.0
    if lo == hi:
        lo, hi = lo - 0.5, hi + 0.5
    raw = (hi - lo) / target
    mag = 10 ** math.floor(math.log10(raw)) if raw > 0 else 1.0
    step = 10 * mag
    for mult in (1, 2, 2.5, 5, 10):
        if raw <= mult * mag:
            step = mult * mag
            break
    lo2 = math.floor(lo / step + 1e-9) * step
    hi2 = math.ceil(hi / step - 1e-9) * step
    ticks, v = [], lo2
    while v <= hi2 + step * 1e-6:
        ticks.append(round(v, 10))
        v += step
    return lo2, hi2, ticks, step


def tick_label(v, step):
    if abs(v) < step * 1e-6:
        v = 0.0
    if abs(v) >= 1e5 or (v != 0 and abs(v) < 1e-3):
        return f"{v:.1e}"
    return f"{v:.4g}"


def _finite(v):
    return isinstance(v, (int, float)) and math.isfinite(v)


# ---------------------------------------------------------------------------
# Colorbar (shared by the 2-D heat renderer and the 3-D renderer)
# ---------------------------------------------------------------------------
def draw_colorbar(p, x, y, w, h, cmap, clim):
    lo, hi = clim
    n = 48
    for k in range(n):
        t0 = k / n
        yy = y + h - (k + 1) * h / n
        p.append(f'<rect x="{x:.1f}" y="{yy:.1f}" width="{w}" height="{h / n + 0.6:.2f}" '
                 f'fill="{cmap_color(cmap, t0)}"/>')
    p.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w}" height="{h:.1f}" fill="none" '
             f'stroke="{_AX}" stroke-width="0.8"/>')
    _, _, ticks, step = nice_axis(lo, hi, 5)
    for t in ticks:
        if t < lo - 1e-9 or t > hi + 1e-9:
            continue
        f = (t - lo) / (hi - lo) if hi > lo else 0.5
        yy = y + h - f * h
        p.append(f'<line x1="{x + w}" y1="{yy:.1f}" x2="{x + w + 4}" y2="{yy:.1f}" '
                 f'stroke="{_AX}" stroke-width="0.8"/>')
        p.append(f'<text x="{x + w + 7}" y="{yy + 3.5:.1f}" fill="{_AX}" font-size="10.5">'
                 f'{_esc(tick_label(t, step))}</text>')


# ---------------------------------------------------------------------------
# 3-D axes
# ---------------------------------------------------------------------------
def _frame(az, el):
    A, E = math.radians(az), math.radians(el)
    ca, sa, ce, se = math.cos(A), math.sin(A), math.cos(E), math.sin(E)
    C = (ce * sa, -ce * ca, se)
    R = (ca, sa, 0.0)
    U = (-se * sa, se * ca, ce)
    return C, R, U


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _data_limits(series, key):
    vals = []
    for s in series:
        v = s.get(key)
        if v is None:
            continue
        if isinstance(v, list) and v and isinstance(v[0], list):
            vals.extend(x for rw in v for x in rw if _finite(x))
        else:
            vals.extend(x for x in v if _finite(x))
    return (min(vals), max(vals)) if vals else (0.0, 1.0)


def render_axes3d(p, ax, ox, oy, W, H, uid=0):
    series = ax.get("series", [])
    az, el = ax.get("view") or (-37.5, 30.0)
    has_cb = bool(ax.get("colorbar"))
    ML, MR, MT, MB = 40, (78 if has_cb else 26), 40, 36
    if H < 300:
        MT, MB = 28, 30
    pw, ph = W - ML - MR, H - MT - MB
    AX0, AY0 = ox + ML, oy + MT

    # --- limits & ticks ------------------------------------------------------
    def lim(key, override):
        if override:
            lo, hi = float(override[0]), float(override[1])
            _, _, ticks, step = nice_axis(lo, hi)
            return lo, hi, [t for t in ticks if lo - 1e-9 <= t <= hi + 1e-9], step
        lo, hi = _data_limits(series, key)
        return nice_axis(lo, hi)

    xlo, xhi, xticks, xstep = lim("X", ax.get("xlim"))
    ylo, yhi, yticks, ystep = lim("Y", ax.get("ylim"))
    zlo, zhi, zticks, zstep = lim("Z", ax.get("zlim"))
    clim = ax.get("clim")
    if not clim:
        cl = _data_limits([s for s in series if s.get("kind") in ("surf", "mesh")], "C")
        if cl == (0.0, 1.0) and not any(s.get("C") for s in series):
            cl = (zlo, zhi) if zhi > zlo else (zlo - 0.5, zlo + 0.5)
        clim = cl
    cmap = ax.get("cmap", "parula")

    # --- projection ------------------------------------------------------------
    C, R, U = _frame(az, el)
    BOX = (1.0, 1.0, 0.78)                  # plot-box aspect, like MATLAB's default

    def nrm(x, y, z):
        return ((x - xlo) / (xhi - xlo) * BOX[0] if xhi > xlo else 0.5,
                (y - ylo) / (yhi - ylo) * BOX[1] if yhi > ylo else 0.5,
                (z - zlo) / (zhi - zlo) * BOX[2] if zhi > zlo else 0.5)

    def proj(P3):
        return (_dot(P3, R), _dot(P3, U), _dot(P3, C))

    corners = [(x, y, z) for x in (0.0, BOX[0]) for y in (0.0, BOX[1]) for z in (0.0, BOX[2])]
    pc = [proj(c) for c in corners]
    sx0, sx1 = min(q[0] for q in pc), max(q[0] for q in pc)
    sy0, sy1 = min(q[1] for q in pc), max(q[1] for q in pc)
    sc = min(pw / (sx1 - sx0), ph / (sy1 - sy0)) * 0.92
    cx = AX0 + (pw - (sx1 - sx0) * sc) / 2 - sx0 * sc
    cy = AY0 + (ph - (sy1 - sy0) * sc) / 2 + sy1 * sc

    def scr(P3):
        q = proj(P3)
        return (cx + q[0] * sc, cy - q[1] * sc, q[2])

    def S(x, y, z):
        return scr(nrm(x, y, z))

    # --- back walls with grid --------------------------------------------------
    # a wall is "back" when its outward normal points away from the camera
    walls = [
        ((0.0, None, None), (-1, 0, 0), "x"), ((BOX[0], None, None), (1, 0, 0), "x"),
        ((None, 0.0, None), (0, -1, 0), "y"), ((None, BOX[1], None), (0, 1, 0), "y"),
        ((None, None, 0.0), (0, 0, -1), "z"), ((None, None, BOX[2]), (0, 0, 1), "z"),
    ]
    xt = [nrm(t, ylo, zlo)[0] for t in xticks]
    yt = [nrm(xlo, t, zlo)[1] for t in yticks]
    zt = [nrm(xlo, ylo, t)[2] for t in zticks]
    for fixed, normal, axis in walls:
        if _dot(normal, C) >= 0:
            continue
        fx, fy, fz = fixed

        def P(u, v):
            if axis == "x":
                return (fx, u, v)
            if axis == "y":
                return (u, fy, v)
            return (u, v, fz)
        if axis == "x":
            us, vs, umax, vmax = yt, zt, BOX[1], BOX[2]
        elif axis == "y":
            us, vs, umax, vmax = xt, zt, BOX[0], BOX[2]
        else:
            us, vs, umax, vmax = xt, yt, BOX[0], BOX[1]
        quad = [scr(P(0, 0)), scr(P(umax, 0)), scr(P(umax, vmax)), scr(P(0, vmax))]
        p.append('<polygon points="' + " ".join(f"{q[0]:.1f},{q[1]:.1f}" for q in quad) +
                 f'" fill="#ffffff" stroke="{_AX}" stroke-width="0.8"/>')
        if ax.get("grid", True):
            for u in us:
                a, b = scr(P(u, 0)), scr(P(u, vmax))
                p.append(f'<line x1="{a[0]:.1f}" y1="{a[1]:.1f}" x2="{b[0]:.1f}" y2="{b[1]:.1f}" '
                         f'stroke="{_GRID}" stroke-width="0.8"/>')
            for v in vs:
                a, b = scr(P(0, v)), scr(P(umax, v))
                p.append(f'<line x1="{a[0]:.1f}" y1="{a[1]:.1f}" x2="{b[0]:.1f}" y2="{b[1]:.1f}" '
                         f'stroke="{_GRID}" stroke-width="0.8"/>')

    # --- data ------------------------------------------------------------------
    cid = f"cp3d{uid}"
    p.append(f'<clipPath id="{cid}"><rect x="{ox}" y="{oy}" width="{W}" height="{H}"/></clipPath>')
    p.append(f'<g clip-path="url(#{cid})">')
    items = []   # (depth, svg)
    shading = ax.get("shading", "faceted")
    for s in series:
        kind = s.get("kind")
        if kind in ("surf", "mesh"):
            X, Y, Z = s["X"], s["Y"], s["Z"]
            Cm = s.get("C") or Z
            r, c = len(Z), len(Z[0]) if Z else 0
            for i in range(r - 1):
                for j in range(c - 1):
                    pts = []
                    ok = True
                    zs = []
                    cs = []
                    for (ii, jj) in ((i, j), (i, j + 1), (i + 1, j + 1), (i + 1, j)):
                        x, y, z = X[ii][jj], Y[ii][jj], Z[ii][jj]
                        if not (_finite(x) and _finite(y) and _finite(z)):
                            ok = False
                            break
                        pts.append(S(x, y, z))
                        zs.append(z)
                        cs.append(Cm[ii][jj] if _finite(Cm[ii][jj]) else z)
                    if not ok:
                        continue
                    depth = sum(q[2] for q in pts) / 4
                    cval = sum(cs) / 4
                    t = (cval - clim[0]) / (clim[1] - clim[0]) if clim[1] > clim[0] else 0.5
                    col = cmap_color(cmap, t)
                    poly = " ".join(f"{q[0]:.1f},{q[1]:.1f}" for q in pts)
                    if kind == "mesh":
                        svg = (f'<polygon points="{poly}" fill="#ffffff" stroke="{col}" '
                               f'stroke-width="0.9" stroke-linejoin="round"/>')
                    elif shading == "faceted":
                        svg = (f'<polygon points="{poly}" fill="{col}" stroke="#000000" '
                               f'stroke-width="0.5" stroke-linejoin="round"/>')
                    else:
                        svg = (f'<polygon points="{poly}" fill="{col}" stroke="{col}" '
                               f'stroke-width="0.6" stroke-linejoin="round"/>')
                    items.append((depth, svg))
        elif kind in ("plot3", "contour3", "floor_contour"):
            X, Y, Z = s["X"], s["Y"], s["Z"]
            col = s.get("color", "#0072BD")
            width = s.get("width", 1.4)
            pts = [S(x, y, z) for x, y, z in zip(X, Y, Z)
                   if _finite(x) and _finite(y) and _finite(z)]
            if len(pts) < 2:
                continue
            depth = sum(q[2] for q in pts) / len(pts)
            d = " ".join(f'{"M" if k == 0 else "L"}{q[0]:.1f},{q[1]:.1f}' for k, q in enumerate(pts))
            dash = {"--": ' stroke-dasharray="7,4"', ":": ' stroke-dasharray="2,4"',
                    "-.": ' stroke-dasharray="8,3,2,3"'}.get(s.get("style") or "", "")
            items.append((depth + (0.0 if kind != "floor_contour" else -1e-6),
                          f'<path d="{d}" fill="none" stroke="{col}" stroke-width="{width:.2g}" '
                          f'stroke-linejoin="round"{dash}/>'))
            if s.get("marker"):
                for q in pts:
                    items.append((q[2], f'<circle cx="{q[0]:.1f}" cy="{q[1]:.1f}" r="3" '
                                        f'fill="none" stroke="{col}" stroke-width="1.4"/>'))
        elif kind == "scatter3":
            col = s.get("color", "#0072BD")
            for x, y, z in zip(s["X"], s["Y"], s["Z"]):
                if _finite(x) and _finite(y) and _finite(z):
                    q = S(x, y, z)
                    items.append((q[2], f'<circle cx="{q[0]:.1f}" cy="{q[1]:.1f}" r="3.2" '
                                        f'fill="none" stroke="{col}" stroke-width="1.5"/>'))
        elif kind == "stem3":
            col = s.get("color", "#0072BD")
            for x, y, z in zip(s["X"], s["Y"], s["Z"]):
                if _finite(x) and _finite(y) and _finite(z):
                    a, b = S(x, y, zlo), S(x, y, z)
                    items.append((b[2], f'<line x1="{a[0]:.1f}" y1="{a[1]:.1f}" x2="{b[0]:.1f}" '
                                        f'y2="{b[1]:.1f}" stroke="{col}" stroke-width="1"/>'
                                        f'<circle cx="{b[0]:.1f}" cy="{b[1]:.1f}" r="3" fill="none" '
                                        f'stroke="{col}" stroke-width="1.3"/>'))
    items.sort(key=lambda it: it[0])          # far first (smaller depth = farther)
    p.extend(svg for _, svg in items)
    p.append("</g>")

    # --- front edges + ticks -----------------------------------------------------
    # bottom corners, pick the one nearest the viewer (largest depth) for x/y ticks
    bottom = [(x, y) for x in (0.0, BOX[0]) for y in (0.0, BOX[1])]
    near = max(bottom, key=lambda c: proj((c[0], c[1], 0.0))[2])
    far_x = BOX[0] - near[0]
    far_y = BOX[1] - near[1]
    # x-axis runs along y = near[1] (from x=0 to x=BOX), y-axis along x = near[0]
    # ticks drawn on the edge that borders the near corner
    def line3(a, b, color=_AX, wd=1.0):
        A, B = scr(a), scr(b)
        p.append(f'<line x1="{A[0]:.1f}" y1="{A[1]:.1f}" x2="{B[0]:.1f}" y2="{B[1]:.1f}" '
                 f'stroke="{color}" stroke-width="{wd}"/>')

    # vertical edges of the box (all four, thin) and bottom rectangle
    for (x, y) in bottom:
        line3((x, y, 0.0), (x, y, BOX[2]), "#9a9a9a", 0.7)
    line3((0, 0, 0), (BOX[0], 0, 0)); line3((0, BOX[1], 0), (BOX[0], BOX[1], 0))
    line3((0, 0, 0), (0, BOX[1], 0)); line3((BOX[0], 0, 0), (BOX[0], BOX[1], 0))

    # x ticks along the edge y = near[1]
    outward_y = 1.0 if near[1] > 0 else -1.0
    for t, u in zip(xticks, xt):
        a = scr((u, near[1], 0.0))
        b = scr((u, near[1] + outward_y * 0.03, 0.0))
        p.append(f'<line x1="{a[0]:.1f}" y1="{a[1]:.1f}" x2="{b[0]:.1f}" y2="{b[1]:.1f}" '
                 f'stroke="{_AX}" stroke-width="1"/>')
        lx, ly = b[0] + (b[0] - a[0]) * 1.2, b[1] + (b[1] - a[1]) * 1.2 + 4
        p.append(f'<text x="{lx:.1f}" y="{ly:.1f}" fill="{_AX}" font-size="10.5" '
                 f'text-anchor="middle">{_esc(tick_label(t, xstep))}</text>')
    # y ticks along the edge x = near[0]
    outward_x = 1.0 if near[0] > 0 else -1.0
    for t, v in zip(yticks, yt):
        a = scr((near[0], v, 0.0))
        b = scr((near[0] + outward_x * 0.03, v, 0.0))
        p.append(f'<line x1="{a[0]:.1f}" y1="{a[1]:.1f}" x2="{b[0]:.1f}" y2="{b[1]:.1f}" '
                 f'stroke="{_AX}" stroke-width="1"/>')
        lx, ly = b[0] + (b[0] - a[0]) * 1.2, b[1] + (b[1] - a[1]) * 1.2 + 4
        p.append(f'<text x="{lx:.1f}" y="{ly:.1f}" fill="{_AX}" font-size="10.5" '
                 f'text-anchor="middle">{_esc(tick_label(t, ystep))}</text>')
    # z ticks on the vertical edge at the far-left bottom corner
    zc = min([(0.0, 0.0), (BOX[0], 0.0), (0.0, BOX[1]), (BOX[0], BOX[1])],
             key=lambda c: scr((c[0], c[1], 0.0))[0])
    for t, w in zip(zticks, zt):
        a = scr((zc[0], zc[1], w))
        p.append(f'<line x1="{a[0]:.1f}" y1="{a[1]:.1f}" x2="{a[0] - 5:.1f}" y2="{a[1]:.1f}" '
                 f'stroke="{_AX}" stroke-width="1"/>')
        p.append(f'<text x="{a[0] - 8:.1f}" y="{a[1] + 3.5:.1f}" fill="{_AX}" font-size="10.5" '
                 f'text-anchor="end">{_esc(tick_label(t, zstep))}</text>')

    # axis labels
    if ax.get("xlabel"):
        m = scr((BOX[0] / 2, near[1] + outward_y * 0.22, 0.0))
        p.append(f'<text x="{m[0]:.1f}" y="{m[1] + 8:.1f}" fill="{_AX}" font-size="12" '
                 f'text-anchor="middle">{_esc(ax["xlabel"])}</text>')
    if ax.get("ylabel"):
        m = scr((near[0] + outward_x * 0.22, BOX[1] / 2, 0.0))
        p.append(f'<text x="{m[0]:.1f}" y="{m[1] + 8:.1f}" fill="{_AX}" font-size="12" '
                 f'text-anchor="middle">{_esc(ax["ylabel"])}</text>')
    if ax.get("zlabel"):
        m = scr((zc[0], zc[1], BOX[2] / 2))
        p.append(f'<text x="{m[0] - 34:.1f}" y="{m[1]:.1f}" fill="{_AX}" font-size="12" '
                 f'text-anchor="middle" transform="rotate(-90 {m[0] - 34:.1f} {m[1]:.1f})">'
                 f'{_esc(ax["zlabel"])}</text>')
    if ax.get("title"):
        p.append(f'<text x="{AX0 + pw / 2:.0f}" y="{oy + MT - 16}" fill="#000000" font-size="14" '
                 f'text-anchor="middle">{_esc(ax["title"])}</text>')

    if has_cb:
        draw_colorbar(p, AX0 + pw + 14, AY0 + ph * 0.1, 12, ph * 0.8, cmap, clim)

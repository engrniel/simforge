"""
msim — numerical simulation for SimForge's DEMO interpreter: ODE solvers,
root finding, minimisation, quadrature and the matrix exponential.

Callbacks are plain Python callables (mdemo wraps the user's @-function):
    f(t, y_list) -> list          for ODEs
    g(x_list)    -> list/float    for fsolve / fminsearch
    h(x)         -> float         for fzero / fminbnd / integral

Pure stdlib (+ mmat for the matrix exponential). MATLAB defaults are kept where
they matter: ode45 RelTol 1e-3 / AbsTol 1e-6, fzero bracket expansion, Nelder-
Mead TolX/TolFun 1e-4.
"""

import math

import mmat


class SimError(ValueError):
    pass


def _err(msg):
    raise SimError(msg)


def _vec(v, n=None, what="state"):
    out = [float(x.real) if isinstance(x, complex) else float(x) for x in v]
    if n is not None and len(out) != n:
        _err(f"the {what} must have {n} element(s), got {len(out)}")
    return out


# ===========================================================================
# ODE solvers
# ===========================================================================
# Dormand-Prince 5(4) tableau
_DP_C = [0.0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0, 1.0]
_DP_A = [
    [],
    [1 / 5],
    [3 / 40, 9 / 40],
    [44 / 45, -56 / 15, 32 / 9],
    [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
    [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
    [35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84],
]
_DP_B5 = [35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0.0]
_DP_B4 = [5179 / 57600, 0.0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40]

# Bogacki-Shampine 3(2)
_BS_C = [0.0, 1 / 2, 3 / 4, 1.0]
_BS_A = [[], [1 / 2], [0.0, 3 / 4], [2 / 9, 1 / 3, 4 / 9]]
_BS_B3 = [2 / 9, 1 / 3, 4 / 9, 0.0]
_BS_B2 = [7 / 24, 1 / 4, 1 / 3, 1 / 8]


def _axpy(y, h, ks, coefs):
    n = len(y)
    out = list(y)
    for k, c in zip(ks, coefs):
        if c == 0.0:
            continue
        for i in range(n):
            out[i] += h * c * k[i]
    return out


def _err_norm(y, ynew, diff, rtol, atol):
    m = 0.0
    for i in range(len(y)):
        sc = atol + rtol * max(abs(y[i]), abs(ynew[i]))
        m = max(m, abs(diff[i]) / sc if sc > 0 else abs(diff[i]))
    return m


def _rk_adaptive(f, tspan, y0, C, A, Bhi, Blo, order, rtol, atol, max_step, first_step=None):
    t0, tf = float(tspan[0]), float(tspan[-1])
    direction = 1.0 if tf >= t0 else -1.0
    span = abs(tf - t0)
    if span == 0:
        return [t0], [list(y0)]
    outputs = [float(v) for v in tspan] if len(tspan) > 2 else None
    hmax = max_step if max_step else span / 100.0
    y = list(y0)
    t = t0
    ts, ys = [t], [list(y)]
    stages = len(C)
    # initial step: a fraction of the span, refined by the first error estimate
    h = min(hmax, span / 100.0) if first_step is None else min(hmax, first_step)
    nfev = 0
    next_out = 1
    while (tf - t) * direction > 1e-12 * max(1.0, abs(tf)):
        target = outputs[next_out] if outputs is not None else tf
        h = min(h, abs(target - t))
        hs = h * direction
        ks = []
        for s in range(stages):
            ys_ = _axpy(y, hs, ks, A[s]) if s else list(y)
            ks.append(f(t + C[s] * hs, ys_))
            nfev += 1
            if nfev > 400000:
                _err("ODE solver: too many function evaluations (is the system stiff? try ode23s)")
        ynew = _axpy(y, hs, ks, Bhi)
        ylow = _axpy(y, hs, ks, Blo)
        diff = [ynew[i] - ylow[i] for i in range(len(y))]
        if any(not math.isfinite(v) for v in ynew):
            h *= 0.25
            if h < 1e-14 * max(1.0, abs(t)):
                _err("ODE solver: the solution became non-finite (blow-up)")
            continue
        e = _err_norm(y, ynew, diff, rtol, atol)
        if e <= 1.0:
            t = t + hs
            y = ynew
            if outputs is None:
                ts.append(t)
                ys.append(list(y))
            elif abs(t - outputs[next_out]) <= 1e-9 * max(1.0, abs(t)):
                ts.append(outputs[next_out])
                ys.append(list(y))
                next_out += 1
                if next_out >= len(outputs):
                    break
            fac = 0.9 * (e ** (-1.0 / order)) if e > 0 else 5.0
            h = min(hmax, h * min(5.0, max(0.2, fac)))
        else:
            h = h * max(0.1, 0.9 * (e ** (-1.0 / order)))
            if h < 1e-14 * max(1.0, abs(t)):
                _err("ODE solver: step size underflow (the problem may be stiff — try ode23s)")
    return ts, ys


def ode45(f, tspan, y0, rtol=1e-3, atol=1e-6, max_step=None):
    return _rk_adaptive(f, tspan, y0, _DP_C, _DP_A, _DP_B5, _DP_B4, 5, rtol, atol, max_step)


def ode23(f, tspan, y0, rtol=1e-3, atol=1e-6, max_step=None):
    return _rk_adaptive(f, tspan, y0, _BS_C, _BS_A, _BS_B3, _BS_B2, 3, rtol, atol, max_step)


def _jacobian(f, t, y, f0):
    n = len(y)
    J = [[0.0] * n for _ in range(n)]
    for j in range(n):
        d = 1e-7 * max(1.0, abs(y[j]))
        yp = list(y)
        yp[j] += d
        fp = f(t, yp)
        for i in range(n):
            J[i][j] = (fp[i] - f0[i]) / d
    return J


def _solve_lin(M, b):
    n = len(b)
    A = [list(M[i]) + [b[i]] for i in range(n)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(A[r][c]))
        if abs(A[p][c]) < 1e-300:
            _err("ode23s: singular iteration matrix")
        A[c], A[p] = A[p], A[c]
        pv = A[c][c]
        for k in range(c, n + 1):
            A[c][k] /= pv
        for r in range(n):
            if r != c and A[r][c] != 0.0:
                fct = A[r][c]
                for k in range(c, n + 1):
                    A[r][k] -= fct * A[c][k]
    return [A[i][n] for i in range(n)]


def ode23s(f, tspan, y0, rtol=1e-3, atol=1e-6, max_step=None):
    """Rosenbrock 2(3) (Shampine & Reichelt) — for stiff systems."""
    t0, tf = float(tspan[0]), float(tspan[-1])
    direction = 1.0 if tf >= t0 else -1.0
    span = abs(tf - t0)
    if span == 0:
        return [t0], [list(y0)]
    outputs = [float(v) for v in tspan] if len(tspan) > 2 else None
    hmax = max_step if max_step else span / 100.0
    d = 1.0 / (2.0 + math.sqrt(2.0))
    e32 = 6.0 + math.sqrt(2.0)
    y = list(y0)
    n = len(y)
    t = t0
    ts, ys = [t], [list(y)]
    h = min(hmax, span / 100.0)
    next_out = 1
    nfev = 0
    while (tf - t) * direction > 1e-12 * max(1.0, abs(tf)):
        target = outputs[next_out] if outputs is not None else tf
        h = min(h, abs(target - t))
        hs = h * direction
        F0 = f(t, y)
        J = _jacobian(f, t, y, F0)
        dt = 1e-7 * max(1.0, abs(t))
        Ft = f(t + dt, y)
        T = [(Ft[i] - F0[i]) / dt for i in range(n)]
        nfev += n + 2
        W = [[(1.0 if i == j else 0.0) - hs * d * J[i][j] for j in range(n)] for i in range(n)]
        k1 = _solve_lin(W, [F0[i] + hs * d * T[i] for i in range(n)])
        F1 = f(t + 0.5 * hs, [y[i] + 0.5 * hs * k1[i] for i in range(n)])
        k2 = _solve_lin(W, [F1[i] - k1[i] for i in range(n)])
        k2 = [k2[i] + k1[i] for i in range(n)]
        ynew = [y[i] + hs * k2[i] for i in range(n)]
        F2 = f(t + hs, ynew)
        k3 = _solve_lin(W, [F2[i] - e32 * (k2[i] - F1[i]) - 2.0 * (k1[i] - F0[i]) + hs * d * T[i]
                            for i in range(n)])
        diff = [hs / 6.0 * (k1[i] - 2.0 * k2[i] + k3[i]) for i in range(n)]
        nfev += 2
        if nfev > 400000:
            _err("ode23s: too many function evaluations")
        if any(not math.isfinite(v) for v in ynew):
            h *= 0.25
            continue
        e = _err_norm(y, ynew, diff, rtol, atol)
        if e <= 1.0:
            t += hs
            y = ynew
            if outputs is None:
                ts.append(t)
                ys.append(list(y))
            elif abs(t - outputs[next_out]) <= 1e-9 * max(1.0, abs(t)):
                ts.append(outputs[next_out])
                ys.append(list(y))
                next_out += 1
                if next_out >= len(outputs):
                    break
            fac = 0.9 * (e ** (-1.0 / 3.0)) if e > 0 else 5.0
            h = min(hmax, h * min(5.0, max(0.2, fac)))
        else:
            h = h * max(0.1, 0.9 * (e ** (-1.0 / 3.0)))
            if h < 1e-14 * max(1.0, abs(t)):
                _err("ode23s: step size underflow")
    return ts, ys


ODE_METHODS = {"ode45": ode45, "ode23": ode23, "ode113": ode45, "ode78": ode45, "ode89": ode45,
               "ode23s": ode23s, "ode15s": ode23s, "ode23t": ode23s, "ode23tb": ode23s}


# ===========================================================================
# Root finding
# ===========================================================================
def _brent_root(f, a, b, fa, fb, tol=2e-12, maxit=200):
    if fa == 0:
        return a
    if fb == 0:
        return b
    if fa * fb > 0:
        _err("fzero: the function values at the interval endpoints must differ in sign")
    c, fc = a, fa
    d = e = b - a
    for _ in range(maxit):
        if fb * fc > 0:
            c, fc = a, fa
            d = e = b - a
        if abs(fc) < abs(fb):
            a, b, c = b, c, b
            fa, fb, fc = fb, fc, fb
        tol1 = 2 * 2.2e-16 * abs(b) + 0.5 * tol
        xm = 0.5 * (c - b)
        if abs(xm) <= tol1 or fb == 0:
            return b
        if abs(e) >= tol1 and abs(fa) > abs(fb):
            s = fb / fa
            if a == c:
                p = 2 * xm * s
                q = 1 - s
            else:
                q = fa / fc
                r = fb / fc
                p = s * (2 * xm * q * (q - r) - (b - a) * (r - 1))
                q = (q - 1) * (r - 1) * (s - 1)
            if p > 0:
                q = -q
            p = abs(p)
            if 2 * p < min(3 * xm * q - abs(tol1 * q), abs(e * q)):
                e = d
                d = p / q
            else:
                d = xm
                e = d
        else:
            d = xm
            e = d
        a, fa = b, fb
        b += d if abs(d) > tol1 else (tol1 if xm > 0 else -tol1)
        fb = f(b)
    return b


def fzero(f, x0):
    """fzero(f, x0) with MATLAB-style bracket search, or fzero(f, [a b])."""
    if isinstance(x0, (list, tuple)):
        if len(x0) == 2:
            a, b = float(x0[0]), float(x0[1])
            return _brent_root(f, a, b, f(a), f(b))
        x0 = x0[0]
    x = float(x0)
    fx = f(x)
    if fx == 0:
        return x
    dx = abs(x) / 50.0 if x != 0 else 1.0 / 50.0
    a = b = x
    fa = fb = fx
    for _ in range(80):
        dx *= math.sqrt(2)
        a, b = x - dx, x + dx
        fa, fb = f(a), f(b)
        if fa * fx <= 0:
            return _brent_root(f, a, x, fa, fx)
        if fb * fx <= 0:
            return _brent_root(f, x, b, fx, fb)
        if not (math.isfinite(fa) and math.isfinite(fb)):
            break
    _err("fzero: could not find a sign change near the starting point; try fzero(f, [a b])")


def fsolve(g, x0, tol=1e-10, maxit=400):
    """Newton with finite-difference Jacobian and back-tracking."""
    x = [float(v) for v in x0]
    n = len(x)

    def F(xx):
        r = g(xx)
        r = r if isinstance(r, (list, tuple)) else [r]
        return [float(v.real) if isinstance(v, complex) else float(v) for v in r]

    fx = F(x)
    if len(fx) != n:
        _err(f"fsolve: the function must return {n} value(s) for {n} unknown(s), got {len(fx)}")
    for it in range(maxit):
        nrm = math.sqrt(sum(v * v for v in fx))
        if nrm < tol:
            return x, fx, 1
        J = [[0.0] * n for _ in range(n)]
        for j in range(n):
            d = 1e-7 * max(1.0, abs(x[j]))
            xp = list(x)
            xp[j] += d
            fp = F(xp)
            for i in range(n):
                J[i][j] = (fp[i] - fx[i]) / d
        try:
            step = _solve_lin(J, [-v for v in fx])
        except SimError:
            step = [-v * 1e-3 for v in fx]
        lam = 1.0
        while lam > 1e-4:
            xn = [x[i] + lam * step[i] for i in range(n)]
            fn = F(xn)
            if math.sqrt(sum(v * v for v in fn)) < nrm:
                x, fx = xn, fn
                break
            lam *= 0.5
        else:
            x = [x[i] + 1e-4 * step[i] for i in range(n)]
            fx = F(x)
    return x, fx, 0


# ===========================================================================
# Minimisation
# ===========================================================================
def fminsearch(g, x0, tolx=1e-4, tolf=1e-4, maxit=None):
    """Nelder-Mead simplex, MATLAB's defaults."""
    x0 = [float(v) for v in x0]
    n = len(x0)
    maxit = maxit or 200 * n

    def F(x):
        v = g(x)
        v = v[0] if isinstance(v, (list, tuple)) else v
        return float(v.real) if isinstance(v, complex) else float(v)

    simplex = [list(x0)]
    for i in range(n):
        p = list(x0)
        p[i] = p[i] * 1.05 if p[i] != 0 else 0.00025
        simplex.append(p)
    fs = [F(p) for p in simplex]
    for _ in range(maxit):
        order = sorted(range(n + 1), key=lambda k: fs[k])
        simplex = [simplex[k] for k in order]
        fs = [fs[k] for k in order]
        if max(abs(fs[k] - fs[0]) for k in range(1, n + 1)) <= tolf and \
           max(abs(simplex[k][i] - simplex[0][i]) for k in range(1, n + 1) for i in range(n)) <= tolx:
            break
        cen = [sum(simplex[k][i] for k in range(n)) / n for i in range(n)]
        xr = [cen[i] + (cen[i] - simplex[-1][i]) for i in range(n)]
        fr = F(xr)
        if fr < fs[0]:
            xe = [cen[i] + 2 * (cen[i] - simplex[-1][i]) for i in range(n)]
            fe = F(xe)
            simplex[-1], fs[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < fs[-2]:
            simplex[-1], fs[-1] = xr, fr
        else:
            if fr < fs[-1]:
                xc = [cen[i] + 0.5 * (xr[i] - cen[i]) for i in range(n)]
                fc = F(xc)
                if fc <= fr:
                    simplex[-1], fs[-1] = xc, fc
                    continue
            else:
                xc = [cen[i] + 0.5 * (simplex[-1][i] - cen[i]) for i in range(n)]
                fc = F(xc)
                if fc < fs[-1]:
                    simplex[-1], fs[-1] = xc, fc
                    continue
            for k in range(1, n + 1):
                simplex[k] = [simplex[0][i] + 0.5 * (simplex[k][i] - simplex[0][i]) for i in range(n)]
                fs[k] = F(simplex[k])
    best = min(range(n + 1), key=lambda k: fs[k])
    return simplex[best], fs[best]


def fminbnd(f, a, b, tol=1e-6, maxit=500):
    """Brent's method (golden section + parabolic interpolation) on [a, b]."""
    a, b = float(a), float(b)
    if a > b:
        a, b = b, a
    c = 0.5 * (3.0 - math.sqrt(5.0))
    v = w = x = a + c * (b - a)
    fv = fw = fx = f(x)
    d = e = 0.0
    for _ in range(maxit):
        xm = 0.5 * (a + b)
        tol1 = 1.5e-8 * abs(x) + tol / 3.0
        tol2 = 2 * tol1
        if abs(x - xm) <= tol2 - 0.5 * (b - a):
            break
        if abs(e) > tol1:
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p = (x - v) * q - (x - w) * r
            q = 2 * (q - r)
            if q > 0:
                p = -p
            q = abs(q)
            etemp = e
            e = d
            if abs(p) >= abs(0.5 * q * etemp) or p <= q * (a - x) or p >= q * (b - x):
                e = (a - x) if x >= xm else (b - x)
                d = c * e
            else:
                d = p / q
                u = x + d
                if u - a < tol2 or b - u < tol2:
                    d = tol1 if xm - x >= 0 else -tol1
        else:
            e = (a - x) if x >= xm else (b - x)
            d = c * e
        u = x + (d if abs(d) >= tol1 else (tol1 if d > 0 else -tol1))
        fu = f(u)
        if fu <= fx:
            if u >= x:
                a = x
            else:
                b = x
            v, w, x = w, x, u
            fv, fw, fx = fw, fx, fu
        else:
            if u < x:
                a = u
            else:
                b = u
            if fu <= fw or w == x:
                v, w = w, u
                fv, fw = fw, fu
            elif fu <= fv or v == x or v == w:
                v, fv = u, fu
    return x, fx


# ===========================================================================
# Quadrature — adaptive Gauss-Kronrod (G7, K15)
# ===========================================================================
_GK_X = [0.991455371120813, 0.949107912342759, 0.864864423359769, 0.741531185599394,
         0.586087235467691, 0.405845151377397, 0.207784955007898, 0.0]
_GK_WK = [0.022935322010529, 0.063092092629979, 0.104790010322250, 0.140653259715525,
          0.169004726639267, 0.190350578064785, 0.204432940075298, 0.209482141084728]
_GK_WG = [0.0, 0.129484966168870, 0.0, 0.279705391489277, 0.0, 0.381830050505119, 0.0,
          0.417959183673469]


def _gk15(f, a, b):
    c = 0.5 * (a + b)
    h = 0.5 * (b - a)
    fc = f(c)
    k = _GK_WK[7] * fc
    g = _GK_WG[7] * fc
    for i in range(7):
        x = h * _GK_X[i]
        f1, f2 = f(c - x), f(c + x)
        k += _GK_WK[i] * (f1 + f2)
        g += _GK_WG[i] * (f1 + f2)
    return k * h, abs((k - g) * h)


_INT_BUDGET = [0]


def integral(f, a, b, abstol=1e-10, reltol=1e-6, maxdepth=40, _top=True):
    """Adaptive G7-K15 with a hard budget of ~30k integrand evaluations: an
    integrand that never settles (e.g. sin^2(x)/x^2 out to infinity) returns
    its best estimate instead of running for ever."""
    if _top:
        _INT_BUDGET[0] = 2000          # ~2000 panels * 15 points
    a, b = float(a), float(b)
    if a == b:
        return 0.0
    if math.isinf(a) or math.isinf(b):
        # map (-inf, inf) -> (-1, 1) with x = t / (1 - t^2)
        def g(t):
            x = t / (1 - t * t)
            return f(x) * (1 + t * t) / (1 - t * t) ** 2
        lo = -1.0 + 1e-9 if math.isinf(a) else (a / (1 + abs(a)) if a != 0 else 0.0)
        hi = 1.0 - 1e-9 if math.isinf(b) else (b / (1 + abs(b)) if b != 0 else 0.0)
        # finite endpoint maps via x = t/(1-t^2) inverse: solve for t (Newton), keep simple
        if not math.isinf(a):
            lo = _inv_map(a)
        if not math.isinf(b):
            hi = _inv_map(b)
        return integral(g, lo, hi, abstol, reltol, maxdepth, _top=False)
    total, err = _gk15(f, a, b)
    _INT_BUDGET[0] -= 1
    if err <= max(abstol, reltol * abs(total)) or maxdepth == 0 or _INT_BUDGET[0] <= 0:
        return total
    m = 0.5 * (a + b)
    return integral(f, a, m, abstol / 2, reltol, maxdepth - 1, _top=False) + \
        integral(f, m, b, abstol / 2, reltol, maxdepth - 1, _top=False)


def _inv_map(x):
    """t such that t / (1 - t^2) = x."""
    if x == 0:
        return 0.0
    return (-1 + math.sqrt(1 + 4 * x * x)) / (2 * x)


# ===========================================================================
# Matrix exponential (scaling & squaring with a degree-6 Pade approximant)
# ===========================================================================
def expm(A):
    M = mmat.to_mat(A)
    if M.r != M.c:
        _err("expm: matrix must be square")
    n = M.r
    nrm = max(sum(abs(M.d[i][j]) for i in range(n)) for j in range(n)) if n else 0.0
    s = max(0, int(math.ceil(math.log2(nrm))) + 1) if nrm > 0.5 else 0
    As = mmat.Mat([[M.d[i][j] / (2 ** s) for j in range(n)] for i in range(n)])
    c = [1.0, 0.5, 5 / 44, 1 / 66, 1 / 792, 1 / 15840, 1 / 665280]
    I = mmat.to_mat(mmat.eye(n))
    X = I
    N = mmat.to_mat(mmat.bcast(I, c[0], lambda a, b: a * b))
    D = mmat.to_mat(mmat.bcast(I, c[0], lambda a, b: a * b))
    for k in range(1, 7):
        X = mmat.to_mat(mmat.matmul(As, X))
        cX = mmat.to_mat(mmat.bcast(X, c[k], lambda a, b: a * b))
        N = mmat.to_mat(mmat.bcast(N, cX, lambda a, b: a + b))
        sign = 1.0 if k % 2 == 0 else -1.0
        D = mmat.to_mat(mmat.bcast(D, mmat.bcast(cX, sign, lambda a, b: a * b), lambda a, b: a + b))
    E = mmat.to_mat(mmat.solve(D, N))
    for _ in range(s):
        E = mmat.to_mat(mmat.matmul(E, E))
    return mmat.norm_val(E)


# ===========================================================================
# Small calculus helpers
# ===========================================================================
def cumtrapz(x, y=None):
    if y is None:
        y = x
        x = list(range(len(y)))
    out = [0.0]
    for i in range(1, len(y)):
        out.append(out[-1] + 0.5 * (x[i] - x[i - 1]) * (y[i] + y[i - 1]))
    return out


def gradient(y, h=1.0, x=None):
    n = len(y)
    if n < 2:
        return [0.0] * n
    if x is None:
        x = [i * h for i in range(n)]
    out = [(y[1] - y[0]) / (x[1] - x[0])]
    for i in range(1, n - 1):
        out.append((y[i + 1] - y[i - 1]) / (x[i + 1] - x[i - 1]))
    out.append((y[-1] - y[-2]) / (x[-1] - x[-2]))
    return out


def polyder(p):
    n = len(p) - 1
    return [p[i] * (n - i) for i in range(n)] or [0.0]


def polyint(p, k=0.0):
    n = len(p)
    return [p[i] / (n - i) for i in range(n)] + [k]

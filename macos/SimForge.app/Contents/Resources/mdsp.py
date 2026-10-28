"""
mdsp — pure-stdlib DSP + filter-design library for SimForge's DEMO
interpreter.

Scope: the laboratory activities of ECE08 "Signals, Spectra and Signal
Processing" (Marikina Polytechnic College / CHED PSG BSECE), chapters 1-7:

  Ch1  signal builders (step, ramp, rect, tri, sinc, sign, gauspuls, dirac)
  Ch2  analog frequency response (freqs)
  Ch3  sampling, decimation/interpolation, quantization, mu-law / A-law
  Ch4  z transform helpers (tf2zp, zp2tf, residuez, impz, stepz, lattice,
       bilinear, impinvar, c2d-style mapping)
  Ch5  DFT/FFT tools (cconv, fftfilt, bitrevorder, kaiser, spectrogram/STFT)
  Ch6  filter design (butter, cheby1, cheby2, ellip, besself + *ord,
       fir1, fir2, firls, firpm, kaiserord, firpmord, grpdelay, filtfilt,
       tf2sos/sos2tf/zp2sos/sosfilt)
  Ch7  random signals + correlation (awgn, snr, xcov, levinson, lpc)

Everything operates on plain Python lists of float/complex. No third-party
dependencies, by design — this must run on a bare student machine.
"""

import cmath
import math
import random


class DspError(ValueError):
    """Raised for user-facing argument errors; mdemo turns this into MError."""


def _err(msg):
    raise DspError(msg)


# ===========================================================================
# Small numeric primitives (self-contained on purpose — mdsp has no imports
# from mdemo so it stays independently testable)
# ===========================================================================
def _lst(v):
    if isinstance(v, (list, tuple)):
        return [x for x in v]
    return [v]


def _re(x):
    return x.real if isinstance(x, complex) else float(x)


def _reals(v):
    return [_re(x) for x in _lst(v)]


def _clean(x, tol=1e-9):
    if isinstance(x, complex) and abs(x.imag) <= tol * (1.0 + abs(x)):
        return x.real
    return x


def _scalar(v):
    v = _lst(v)
    if not v:
        _err("expected a value, got an empty vector")
    return v[0]


def _int(v, name="n"):
    x = _re(_scalar(v))
    if abs(x - round(x)) > 1e-9:
        _err(f"{name} must be an integer")
    return int(round(x))


# --- FFT -------------------------------------------------------------------
def _fft_p2(x, sgn):
    n = len(x)
    if n == 1:
        return x[:]
    ev = _fft_p2(x[0::2], sgn)
    od = _fft_p2(x[1::2], sgn)
    out = [0j] * n
    for k in range(n // 2):
        t = cmath.exp(sgn * 2j * math.pi * k / n) * od[k]
        out[k] = ev[k] + t
        out[k + n // 2] = ev[k] - t
    return out


def _bluestein(x, sgn):
    n = len(x)
    m = 1
    while m < 2 * n + 1:
        m *= 2
    w = [cmath.exp(sgn * 1j * math.pi * (k * k % (2 * n)) / n) for k in range(n)]
    a = [x[k] * w[k] for k in range(n)] + [0j] * (m - n)
    b = [0j] * m
    for k in range(n):
        b[k] = w[k].conjugate()
        if k:
            b[m - k] = w[k].conjugate()
    A = _fft_p2(a, -1.0)
    B = _fft_p2(b, -1.0)
    C = _fft_p2([A[k] * B[k] for k in range(m)], 1.0)
    return [C[k] / m * w[k] for k in range(n)]


def fft(x, inverse=False):
    """DFT / inverse DFT of a list (radix-2 when possible, Bluestein else)."""
    n = len(x)
    if n == 0:
        return []
    sgn = 1.0 if inverse else -1.0
    xc = [complex(v) for v in x]
    y = _fft_p2(xc, sgn) if (n & (n - 1)) == 0 else _bluestein(xc, sgn)
    if inverse:
        y = [v / n for v in y]
    return y


def ifft(x):
    return fft(x, inverse=True)


# --- polynomials -----------------------------------------------------------
def _ptrim(c):
    c = list(c)
    while len(c) > 1 and abs(c[0]) < 1e-300:
        c.pop(0)
    return c


def _pmul(a, b):
    out = [0j] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            out[i + j] += ai * bj
    return out


def _padd(a, b):
    n = max(len(a), len(b))
    a = [0j] * (n - len(a)) + list(a)
    b = [0j] * (n - len(b)) + list(b)
    return [x + y for x, y in zip(a, b)]


def _pval(p, x):
    acc = 0j
    for c in p:
        acc = acc * x + c
    return acc


def poly_from_roots(roots):
    c = [1.0 + 0j]
    for r in roots:
        c = _pmul(c, [1.0 + 0j, -complex(r)])
    return c


def roots(coeffs):
    """Polynomial roots via Durand-Kerner (highest power first)."""
    c = _ptrim([complex(x) for x in coeffs])
    origin = 0
    while len(c) > 1 and c[-1] == 0:
        c.pop()
        origin += 1
    n = len(c) - 1
    if n < 1:
        return [0j] * origin
    lead = c[0]
    c = [x / lead for x in c]
    z = [(0.4 + 0.9j) ** k for k in range(1, n + 1)]
    for _ in range(800):
        worst = 0.0
        for k in range(n):
            d = _pval(c, z[k])
            for j in range(n):
                if j != k:
                    dz = z[k] - z[j]
                    d /= dz if dz != 0 else 1e-30
            z[k] -= d
            worst = max(worst, abs(d))
        if worst < 1e-13:
            break
    return z + [0j] * origin


def _lin_solve(A, b):
    n = len(A)
    M = [[complex(A[i][j]) for j in range(n)] + [complex(b[i])] for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < 1e-300:
            _err("singular system (cannot solve)")
        M[col], M[piv] = M[piv], M[col]
        pv = M[col][col]
        for c in range(col, n + 1):
            M[col][c] /= pv
        for r in range(n):
            if r != col and M[r][col] != 0:
                f = M[r][col]
                for c in range(col, n + 1):
                    M[r][c] -= f * M[col][c]
    return [M[i][n] for i in range(n)]


# ===========================================================================
# Chapter 1 — basic signal builders
# ===========================================================================
def heaviside(t):
    return 1.0 if t > 0 else (0.5 if t == 0 else 0.0)


def stepfun(args):
    """stepfun(t, t0) — unit step that turns on at t0 (1 for t >= t0)."""
    if len(args) < 2:
        _err("stepfun(t, t0) needs two arguments")
    t0 = _re(_scalar(args[1]))
    return [1.0 if _re(v) >= t0 else 0.0 for v in _lst(args[0])]


def rectpuls(args):
    """rectpuls(t, w) — unit rectangle of width w centred on 0."""
    w = _re(_scalar(args[1])) if len(args) > 1 else 1.0
    return [1.0 if (-w / 2 <= _re(v) < w / 2) else 0.0 for v in _lst(args[0])]


def tripuls(args):
    """tripuls(t, w) — unit triangle of full width w centred on 0."""
    w = _re(_scalar(args[1])) if len(args) > 1 else 1.0
    out = []
    for v in _lst(args[0]):
        u = abs(2.0 * _re(v) / w)
        out.append(1.0 - u if u < 1.0 else 0.0)
    return out


def gauspuls(args):
    """gauspuls(t, fc, bw) — Gaussian-modulated RF pulse."""
    t = _lst(args[0])
    fc = _re(_scalar(args[1])) if len(args) > 1 else 1000.0
    bw = _re(_scalar(args[2])) if len(args) > 2 else 0.5
    bwr = -6.0
    a = -(math.pi * fc * bw) ** 2 / (4.0 * math.log(10 ** (bwr / 20)))
    return [math.exp(-a * _re(v) ** 2) * math.cos(2 * math.pi * fc * _re(v)) for v in t]


def dirac(t):
    return math.inf if t == 0 else 0.0


def pulstran(args):
    """pulstran(t, d, 'rectpuls'|'tripuls'|'gauspuls', w) — pulse train."""
    if len(args) < 2:
        _err("pulstran(t, d, shape, w) needs at least t and d")
    t = _reals(args[0])
    d = _reals(args[1])
    shape = args[2].lower() if len(args) > 2 and isinstance(args[2], str) else "rectpuls"
    w = _re(_scalar(args[3])) if len(args) > 3 else 1.0
    out = [0.0] * len(t)
    for delay in d:
        shifted = [v - delay for v in t]
        if shape == "tripuls":
            piece = tripuls([shifted, w])
        elif shape == "gauspuls":
            piece = gauspuls([shifted, w])
        else:
            piece = rectpuls([shifted, w])
        out = [out[i] + piece[i] for i in range(len(t))]
    return out


# ===========================================================================
# Chapter 3 — sampling, rate change, quantization, companding
# ===========================================================================
def _fir_lowpass(order, wc, window=None):
    """Windowed-sinc lowpass, wc normalized to Nyquist (0..1)."""
    taps = order + 1
    alpha = order / 2.0
    win = window if window is not None else _hamming(taps)
    h = []
    for k in range(taps):
        x = k - alpha
        h.append(wc * _sinc(wc * x) * win[k])
    s = sum(h)
    return [v / s for v in h] if s else h


def _sinc(x):
    if x == 0:
        return 1.0
    return math.sin(math.pi * x) / (math.pi * x)


def _hamming(n):
    if n <= 1:
        return [1.0] * max(n, 0)
    return [0.54 - 0.46 * math.cos(2 * math.pi * k / (n - 1)) for k in range(n)]


def decimate(args):
    """decimate(x, r) — anti-alias filter (zero-phase FIR) then keep every r-th."""
    x = _reals(args[0])
    r = _int(args[1] if len(args) > 1 else 2, "r")
    if r < 1:
        _err("decimate: r must be >= 1")
    if r == 1:
        return x[:]
    b = _fir_lowpass(30, 0.8 / r)
    y = filtfilt([b, [1.0], x])
    return y[0::r]


def interp_(args):
    """interp(x, r) — upsample by r and interpolate (lowpass, gain r)."""
    x = _reals(args[0])
    r = _int(args[1] if len(args) > 1 else 2, "r")
    if r < 1:
        _err("interp: r must be >= 1")
    if r == 1:
        return x[:]
    up = [0.0] * (len(x) * r)
    for i, v in enumerate(x):
        up[i * r] = v
    b = [r * c for c in _fir_lowpass(8 * r, 1.0 / r)]
    return filtfilt([b, [1.0], up])


def resample(args):
    """resample(x, p, q) — rational rate change by p/q."""
    x = _reals(args[0])
    p = _int(args[1] if len(args) > 1 else 1, "p")
    q = _int(args[2] if len(args) > 2 else 1, "q")
    if p < 1 or q < 1:
        _err("resample: p and q must be >= 1")
    g = math.gcd(p, q)
    p, q = p // g, q // g
    y = interp_([x, p]) if p > 1 else x[:]
    return decimate([y, q]) if q > 1 else y


def quantiz(args):
    """quantiz(sig, partition, codebook) -> (index, quants)."""
    if len(args) < 2:
        _err("quantiz(sig, partition, codebook) needs at least two arguments")
    sig = _reals(args[0])
    part = _reals(args[1])
    book = _reals(args[2]) if len(args) > 2 else None
    idx, qs = [], []
    for v in sig:
        k = 0
        while k < len(part) and v > part[k]:
            k += 1
        idx.append(float(k))
        if book is not None:
            qs.append(book[min(k, len(book) - 1)])
    return idx, (qs if book is not None else [])


def uquantize(args):
    """uquantize(x, nbits, xmax) — uniform mid-rise quantizer (teaching helper).

    Returns the quantized signal; step size is 2*xmax/2^nbits.
    """
    x = _reals(args[0])
    nb = _int(args[1] if len(args) > 1 else 8, "nbits")
    xmax = _re(_scalar(args[2])) if len(args) > 2 else max((abs(v) for v in x), default=1.0)
    if xmax <= 0:
        xmax = 1.0
    levels = 2 ** nb
    step = 2.0 * xmax / levels
    out = []
    for v in x:
        q = math.floor(v / step) * step + step / 2.0
        out.append(max(-xmax + step / 2, min(xmax - step / 2, q)))
    return out


def sqnr(args):
    """sqnr(x, xq) — signal-to-quantization-noise ratio in dB."""
    x = _reals(args[0])
    xq = _reals(args[1])
    if len(x) != len(xq):
        _err("sqnr: both signals must be the same length")
    ps = sum(v * v for v in x)
    pn = sum((x[i] - xq[i]) ** 2 for i in range(len(x)))
    if pn <= 0:
        return math.inf
    return 10 * math.log10(ps / pn)


def compand(args):
    """compand(x, param, v, method) — 'mu/compressor', 'mu/expander',
    'A/compressor', 'A/expander' (MATLAB comms toolbox semantics)."""
    x = _reals(args[0])
    param = _re(_scalar(args[1])) if len(args) > 1 else 255.0
    vmax = _re(_scalar(args[2])) if len(args) > 2 else 1.0
    method = args[3].lower() if len(args) > 3 and isinstance(args[3], str) else "mu/compressor"
    out = []
    if method.startswith("mu"):
        mu = param
        if "expand" in method:
            for v in x:
                s = (v > 0) - (v < 0)
                out.append(s * vmax / mu * ((1 + mu) ** (abs(v) / vmax) - 1))
        else:
            for v in x:
                s = (v > 0) - (v < 0)
                out.append(s * vmax * math.log(1 + mu * abs(v) / vmax) / math.log(1 + mu))
    else:
        A = param
        lnA = 1 + math.log(A)
        if "expand" in method:
            for v in x:
                s = (v > 0) - (v < 0)
                u = abs(v) / vmax
                if u < 1.0 / lnA:
                    out.append(s * vmax * u * lnA / A)
                else:
                    out.append(s * vmax * math.exp(u * lnA - 1) / A)
        else:
            for v in x:
                s = (v > 0) - (v < 0)
                u = abs(v) / vmax
                if u < 1.0 / A:
                    out.append(s * vmax * A * u / lnA)
                else:
                    out.append(s * vmax * (1 + math.log(A * u)) / lnA)
    return out


def lin2mu(args):
    """lin2mu(x) — linear (-1..1) to 8-bit mu-law codes (0..255)."""
    x = _reals(args[0])
    mu = 255.0
    out = []
    for v in x:
        v = max(-1.0, min(1.0, v))
        s = (v > 0) - (v < 0)
        y = s * math.log(1 + mu * abs(v)) / math.log(1 + mu)
        out.append(float(max(0, min(255, int(round((1 - y) * 127.5))))))
    return out


def mu2lin(args):
    """mu2lin(c) — 8-bit mu-law codes back to linear (-1..1)."""
    c = _reals(args[0])
    mu = 255.0
    out = []
    for v in c:
        y = 1.0 - v / 127.5
        s = (y > 0) - (y < 0)
        out.append(s * ((1 + mu) ** abs(y) - 1) / mu)
    return out


# ===========================================================================
# Chapter 4 — z transform helpers
# ===========================================================================
def tf2zp(args):
    """tf2zp(b, a) -> (z, p, k)."""
    if len(args) < 2:
        _err("tf2zp(b, a) needs two arguments")
    b = _ptrim([complex(_re(v)) for v in _lst(args[0])])
    a = _ptrim([complex(_re(v)) for v in _lst(args[1])])
    if not a or a[0] == 0:
        _err("tf2zp: leading denominator coefficient must be non-zero")
    k = (b[0] / a[0]).real if b else 0.0
    z = [_clean(r) for r in roots(b)] if len(b) > 1 else []
    p = [_clean(r) for r in roots(a)] if len(a) > 1 else []
    return z, p, k


def zp2tf(args):
    """zp2tf(z, p, k) -> (b, a)."""
    if len(args) < 3:
        _err("zp2tf(z, p, k) needs three arguments")
    z = _lst(args[0])
    p = _lst(args[1])
    k = _re(_scalar(args[2]))
    b = [_clean(k * c) for c in poly_from_roots(z)]
    a = [_clean(c) for c in poly_from_roots(p)]
    return [_re(v) for v in b], [_re(v) for v in a]


def residuez(args):
    """residuez(b, a) -> (r, p, k) for B(z)/A(z) in powers of z^-1.

    Distinct poles only (repeated poles need the matrix form we do not carry).
    """
    if len(args) < 2:
        _err("residuez(b, a) needs two arguments")
    b = [complex(_re(v)) for v in _lst(args[0])]
    a = [complex(_re(v)) for v in _lst(args[1])]
    while len(a) > 1 and a[-1] == 0:
        a = a[:-1]
    if not a or a[0] == 0:
        _err("residuez: a(1) must be non-zero")
    b = [v / a[0] for v in b]
    a = [v / a[0] for v in a]
    while len(b) > 1 and b[-1] == 0:
        b = b[:-1]
    N = len(a) - 1
    if N < 1:
        return [], [], [_clean(v) for v in b]
    k = []
    R = b
    if len(b) > N:
        k, R = _polydiv(b, a)
    R = list(R)
    # roots of A(x) with x = z^-1  (A given in ascending powers of x)
    xr = roots(list(reversed(a)))
    pol = [1.0 / xi if xi != 0 else complex(math.inf) for xi in xr]
    aN = a[-1]
    r = []
    for i, xi in enumerate(xr):
        den = aN * xi
        for j, xj in enumerate(xr):
            if j != i:
                d = xi - xj
                if abs(d) < 1e-5 * (1 + abs(xi)):
                    _err("residuez: repeated poles are not supported "
                         "(use residue on the equivalent s-domain form)")
                den *= d
        r.append(_clean(-_pval(list(reversed(R)), xi) / den))
    return r, [_clean(v) for v in pol], [_clean(v) for v in k]


def _polydiv(num, den):
    """Long division of polynomials given in ASCENDING powers of x."""
    num = [complex(v) for v in num]
    den = [complex(v) for v in den]
    q = [0j] * (len(num) - len(den) + 1)
    r = list(num)
    for i in range(len(q) - 1, -1, -1):
        c = r[i + len(den) - 1] / den[-1]
        q[i] = c
        for j in range(len(den)):
            r[i + j] -= c * den[j]
    return q, r[:len(den) - 1]


def impz(args):
    """impz(b, a, n) -> (h, t) impulse response of a difference equation."""
    b = _reals(args[0])
    a = _reals(args[1]) if len(args) > 1 and isinstance(args[1], (list, tuple)) else [1.0]
    n = _int(args[2], "n") if len(args) > 2 else _impz_len(b, a)
    x = [0.0] * n
    if n:
        x[0] = 1.0
    h = lfilter(b, a, x)
    return h, [float(k) for k in range(n)]


def stepz(args):
    """stepz(b, a, n) -> (s, t) step response of a difference equation."""
    b = _reals(args[0])
    a = _reals(args[1]) if len(args) > 1 and isinstance(args[1], (list, tuple)) else [1.0]
    n = _int(args[2], "n") if len(args) > 2 else _impz_len(b, a)
    y = lfilter(b, a, [1.0] * n)
    return y, [float(k) for k in range(n)]


def _impz_len(b, a):
    if len(a) <= 1:
        return max(len(b), 8)
    pr = [abs(r) for r in roots(a)]
    rmax = max(pr) if pr else 0.0
    if rmax >= 1.0 or rmax <= 0:
        return 100
    n = int(math.ceil(math.log(1e-4) / math.log(rmax))) + len(b)
    return max(16, min(2000, n))


def lfilter(b, a, x):
    """Direct-form II transposed difference-equation filter."""
    b = _reals(b)
    a = _reals(a) if a else [1.0]
    if not a or a[0] == 0:
        _err("filter: a(1) must be non-zero")
    b = [v / a[0] for v in b]
    a = [v / a[0] for v in a]
    n = max(len(a), len(b))
    b = b + [0.0] * (n - len(b))
    a = a + [0.0] * (n - len(a))
    zi = [0.0] * (n - 1)
    out = []
    for xn in x:
        yn = b[0] * xn + (zi[0] if zi else 0.0)
        for k in range(n - 2):
            zi[k] = b[k + 1] * xn + zi[k + 1] - a[k + 1] * yn
        if n >= 2:
            zi[n - 2] = b[n - 1] * xn - a[n - 1] * yn
        out.append(yn)
    return out


def filtfilt(args):
    """filtfilt(b, a, x) — zero-phase forward-and-reverse filtering."""
    if len(args) < 3:
        _err("filtfilt(b, a, x) needs three arguments")
    b = _reals(args[0])
    a = _reals(args[1])
    x = _reals(args[2])
    if len(x) < 3:
        return x[:]
    ntaps = max(len(a), len(b))
    edge = min(3 * ntaps, len(x) - 1)
    head = [2 * x[0] - v for v in reversed(x[1:edge + 1])]
    tail = [2 * x[-1] - v for v in reversed(x[-edge - 1:-1])]
    ext = head + x + tail
    y = lfilter(b, a, ext)
    y = list(reversed(lfilter(b, a, list(reversed(y)))))
    return y[len(head):len(head) + len(x)]


def tf2latc(args):
    """tf2latc(b) — FIR lattice reflection coefficients from a(z) = b."""
    a = _reals(args[0])
    if not a:
        _err("tf2latc: empty polynomial")
    a = [v / a[0] for v in a]
    N = len(a) - 1
    k = [0.0] * N
    cur = a[:]
    for m in range(N, 0, -1):
        km = cur[m]
        k[m - 1] = km
        if abs(1 - km * km) < 1e-12:
            _err("tf2latc: |k| = 1, the step-down recursion is singular")
        nxt = [1.0] * (m)
        for i in range(1, m):
            nxt[i] = (cur[i] - km * cur[m - i]) / (1 - km * km)
        cur = nxt
    return k


def latc2tf(args):
    """latc2tf(k) — FIR polynomial from lattice reflection coefficients."""
    k = _reals(args[0])
    a = [1.0]
    for km in k:
        prev = a[:] + [0.0]
        rev = [0.0] + list(reversed(a))
        a = [prev[i] + km * rev[i] for i in range(len(prev))]
    return a


# --- analog <-> digital mappings -------------------------------------------
def bilinear_ba(args):
    """bilinear(b, a, fs) — analog (b,a) to digital via s = 2*fs*(z-1)/(z+1)."""
    if len(args) < 3:
        _err("bilinear(b, a, fs) needs three arguments")
    z, p, k = _ba2zpk(_reals(args[0]), _reals(args[1]))
    fs = _re(_scalar(args[2]))
    zd, pd, kd = _bilinear_zpk(z, p, k, fs)
    return _zpk2ba(zd, pd, kd)


def impinvar(args):
    """impinvar(b, a, fs) — impulse-invariance analog-to-digital mapping."""
    if len(args) < 3:
        _err("impinvar(b, a, fs) needs three arguments")
    b = [complex(v) for v in _reals(args[0])]
    a = [complex(v) for v in _reals(args[1])]
    fs = _re(_scalar(args[2]))
    r, p, k = _s_residue(b, a)
    if k:
        _err("impinvar: the analog filter must be strictly proper")
    # H(z) = sum r_i/fs / (1 - exp(p_i/fs) z^-1)
    num = [0j]
    terms = []
    for ri, pi in zip(r, p):
        terms.append((ri / fs, cmath.exp(pi / fs)))
    for i, (ri, ei) in enumerate(terms):
        piece = [ri]
        for j, (_, ej) in enumerate(terms):
            if j != i:
                piece = _pmul(piece, [1.0 + 0j, -ej])
        num = _padd(num, piece)
    den = [1.0 + 0j]
    for _, ei in terms:
        den = _pmul(den, [1.0 + 0j, -ei])
    # num is a polynomial in z^-1 of degree len(terms)-1; pad on the right
    num = list(num) + [0j] * (len(den) - len(num))
    return [_re(_clean(v)) for v in num], [_re(_clean(v)) for v in den]


def _s_residue(b, a):
    """Partial fractions of B(s)/A(s), distinct poles."""
    b = _ptrim(list(b))
    a = _ptrim(list(a))
    k = []
    if len(b) >= len(a):
        k, b = _poly_divmod(b, a)
    p = roots(a)
    aN = a[0]
    r = []
    for i, pi in enumerate(p):
        den = aN
        for j, pj in enumerate(p):
            if j != i:
                d = pi - pj
                if abs(d) < 1e-6 * (1 + abs(pi)):
                    _err("repeated poles are not supported here")
                den *= d
        r.append(_pval(b, pi) / den)
    return r, p, k


def _poly_divmod(num, den):
    """Long division, highest power first."""
    num = [complex(c) for c in num]
    den = [complex(c) for c in den]
    if len(num) < len(den):
        return [0j], num
    q = [0j] * (len(num) - len(den) + 1)
    r = list(num)
    for i in range(len(q)):
        c = r[i] / den[0]
        q[i] = c
        for j in range(len(den)):
            r[i + j] -= c * den[j]
    return q, _ptrim(r[len(q):]) if len(r) > len(q) else [0j]


# ===========================================================================
# Chapter 5 — DFT / FFT tools
# ===========================================================================
def cconv(args):
    """cconv(x, y, N) — circular convolution of length N (default len(x)+len(y)-1)."""
    if len(args) < 2:
        _err("cconv(x, y, N) needs at least two arguments")
    x = _lst(args[0])
    y = _lst(args[1])
    N = _int(args[2], "N") if len(args) > 2 else len(x) + len(y) - 1
    if N < 1:
        _err("cconv: N must be >= 1")
    xp = [complex(v) for v in x] + [0j] * max(0, N - len(x))
    yp = [complex(v) for v in y] + [0j] * max(0, N - len(y))
    xp = _wrap(xp, N)
    yp = _wrap(yp, N)
    X = fft(xp)
    Y = fft(yp)
    z = ifft([X[k] * Y[k] for k in range(N)])
    return [_clean(v, 1e-10) for v in z]


def _wrap(v, N):
    out = [0j] * N
    for i, val in enumerate(v):
        out[i % N] += val
    return out


def fftfilt(args):
    """fftfilt(b, x, N) — FIR filtering by overlap-add."""
    if len(args) < 2:
        _err("fftfilt(b, x, N) needs at least two arguments")
    b = [complex(v) for v in _lst(args[0])]
    x = [complex(v) for v in _lst(args[1])]
    nb = len(b)
    nx = len(x)
    if nb == 0 or nx == 0:
        return []
    L = _int(args[2], "N") if len(args) > 2 else max(nb, 1) * 4
    nfft = 1
    while nfft < L + nb - 1:
        nfft *= 2
    L = nfft - nb + 1
    B = fft(b + [0j] * (nfft - nb))
    y = [0j] * (nx + nb - 1)
    pos = 0
    while pos < nx:
        blk = x[pos:pos + L]
        Xb = fft(blk + [0j] * (nfft - len(blk)))
        yb = ifft([Xb[k] * B[k] for k in range(nfft)])
        for i in range(min(nfft, len(y) - pos)):
            y[pos + i] += yb[i]
        pos += L
    return [_clean(v, 1e-10) for v in y[:nx]]


def bitrevorder(args):
    """bitrevorder(x) — permute a 2^k-length vector into bit-reversed order."""
    x = _lst(args[0])
    n = len(x)
    if n & (n - 1):
        _err("bitrevorder: length must be a power of two")
    bits = n.bit_length() - 1
    out = [None] * n
    for i in range(n):
        r = int(format(i, "0%db" % bits)[::-1], 2) if bits else 0
        out[i] = x[r]
    return out


def dftmtx_row(args):
    """dftmtx_row(N, k) — row k (0-based) of the N-point DFT matrix.

    Full 2-D matrices are not supported by the DEMO interpreter, so the DFT
    matrix is exposed one row (one twiddle-factor set) at a time.
    """
    N = _int(args[0], "N")
    k = _int(args[1], "k") if len(args) > 1 else 0
    return [cmath.exp(-2j * math.pi * k * n / N) for n in range(N)]


def _i0(x):
    """Modified Bessel I0 via series."""
    s = 1.0
    t = 1.0
    for k in range(1, 60):
        t *= (x / (2.0 * k)) ** 2
        s += t
        if t < 1e-16 * s:
            break
    return s


def kaiser(args):
    """kaiser(n, beta) — Kaiser window."""
    n = _int(args[0], "n")
    beta = _re(_scalar(args[1])) if len(args) > 1 else 0.5
    if n <= 0:
        return []
    if n == 1:
        return [1.0]
    d = _i0(beta)
    return [_i0(beta * math.sqrt(max(0.0, 1 - (2.0 * k / (n - 1) - 1) ** 2))) / d
            for k in range(n)]


def flattopwin(args):
    n = _int(args[0], "n")
    if n <= 1:
        return [1.0] * max(n, 0)
    a = [0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368]
    out = []
    for k in range(n):
        u = 2 * math.pi * k / (n - 1)
        out.append(a[0] - a[1] * math.cos(u) + a[2] * math.cos(2 * u)
                   - a[3] * math.cos(3 * u) + a[4] * math.cos(4 * u))
    return out


def _window_by_name(name, n):
    name = (name or "hamming").lower()
    if name in ("rect", "rectwin", "boxcar", "rectangular"):
        return [1.0] * n
    if name in ("hann", "hanning"):
        return [0.5 * (1 - math.cos(2 * math.pi * k / (n - 1))) for k in range(n)] \
            if n > 1 else [1.0] * n
    if name == "blackman":
        return [0.42 - 0.5 * math.cos(2 * math.pi * k / (n - 1))
                + 0.08 * math.cos(4 * math.pi * k / (n - 1)) for k in range(n)] \
            if n > 1 else [1.0] * n
    if name == "bartlett":
        return [1.0 - abs(2.0 * k / (n - 1) - 1) for k in range(n)] if n > 1 else [1.0] * n
    if name == "flattop":
        return flattopwin([n])
    return _hamming(n)


def spectrogram(args):
    """spectrogram(x, win, noverlap, nfft, fs) -> (t, f, P) with P a list of
    columns (one per frame) in dB. Used by mdemo to draw an STFT heat map."""
    x = _reals(args[0])
    n = len(x)
    if n < 8:
        _err("spectrogram: need at least 8 samples")
    win = args[1] if len(args) > 1 else None
    if isinstance(win, (list, tuple)) and len(win) > 1:
        w = _reals(win)
        wl = len(w)
    else:
        wl = _int(win, "window") if win is not None else max(8, n // 8)
        wl = max(4, min(wl, n))
        w = _window_by_name("hamming", wl)
    nov = _int(args[2], "noverlap") if len(args) > 2 and args[2] is not None else wl // 2
    nov = max(0, min(nov, wl - 1))
    nfft = _int(args[3], "nfft") if len(args) > 3 and args[3] is not None else max(wl, 256)
    fs = _re(_scalar(args[4])) if len(args) > 4 else 1.0
    hop = wl - nov
    half = nfft // 2 + 1
    cols, times = [], []
    pos = 0
    while pos + wl <= n:
        seg = [x[pos + i] * w[i] for i in range(wl)]
        seg = seg + [0.0] * (nfft - wl) if nfft > wl else seg[:nfft]
        X = fft(seg)
        col = []
        for k in range(half):
            m = abs(X[k]) ** 2 / (fs * sum(v * v for v in w))
            col.append(10 * math.log10(m) if m > 1e-20 else -200.0)
        cols.append(col)
        times.append((pos + wl / 2.0) / fs)
        pos += hop
    if not cols:
        _err("spectrogram: window is longer than the signal")
    f = [fs * k / nfft for k in range(half)]
    return times, f, cols


# ===========================================================================
# Chapter 6 — filter design
# ===========================================================================
# --- analog prototypes (lowpass, cutoff 1 rad/s), returned as (z, p, k) ----
def buttap(N):
    if N < 1:
        _err("filter order must be >= 1")
    p = [cmath.exp(1j * math.pi * (2 * k + N - 1) / (2 * N)) for k in range(1, N + 1)]
    return [], p, 1.0


def cheb1ap(N, rp):
    if rp <= 0:
        _err("cheby1: passband ripple Rp must be > 0 dB")
    eps = math.sqrt(10 ** (0.1 * rp) - 1)
    mu = math.asinh(1.0 / eps) / N
    p = []
    for k in range(1, N + 1):
        th = math.pi * (2 * k - 1) / (2 * N)
        p.append(complex(-math.sinh(mu) * math.sin(th), math.cosh(mu) * math.cos(th)))
    k = 1.0 + 0j
    for pp in p:
        k *= -pp
    k = k.real
    if N % 2 == 0:
        k /= math.sqrt(1 + eps * eps)
    return [], p, k


def cheb2ap(N, rs):
    if rs <= 0:
        _err("cheby2: stopband attenuation Rs must be > 0 dB")
    de = 1.0 / math.sqrt(10 ** (0.1 * rs) - 1)
    mu = math.asinh(1.0 / de) / N
    if N % 2:
        m = list(range(-N + 1, 0, 2)) + list(range(2, N, 2))
    else:
        m = list(range(-N + 1, N, 2))
    z = [(-1j / math.sin(mi * math.pi / (2 * N))).conjugate() for mi in m]
    p = []
    for mi in range(-N + 1, N, 2):
        e = -cmath.exp(1j * math.pi * mi / (2 * N))
        pp = complex(math.sinh(mu) * e.real, math.cosh(mu) * e.imag)
        p.append(1.0 / pp)
    num = 1.0 + 0j
    for pp in p:
        num *= -pp
    den = 1.0 + 0j
    for zz in z:
        den *= -zz
    return z, p, (num / den).real


def besselap(N):
    a = [math.factorial(2 * N - k) / (2 ** (N - k) * math.factorial(k) * math.factorial(N - k))
         for k in range(N, -1, -1)]
    p = roots(a)
    return [], p, float(a[-1])


# --- elliptic prototype: Landen / Jacobi elliptic functions ----------------
def _landen(k, tol=1e-15, maxit=60):
    v = []
    if k <= 0 or k >= 1:
        return v
    for _ in range(maxit):
        if k <= tol:
            break
        k = (k / (1 + math.sqrt(1 - k * k))) ** 2
        v.append(k)
    return v


def ellipk(k):
    """Complete elliptic integral of the first kind, modulus form."""
    if k >= 1:
        return math.inf
    if k > 0.9999995:
        kp = math.sqrt(1 - k * k)
        L = -math.log(kp / 4)
        return L + (L - 1) * kp * kp / 4
    prod = 1.0
    for v in _landen(k):
        prod *= (1 + v)
    return prod * math.pi / 2


def _cde(u, k):
    w = cmath.cos(complex(u) * math.pi / 2)
    for v in reversed(_landen(k)):
        w = (1 + v) * w / (1 + v * w * w)
    return w


def _sne(u, k):
    w = cmath.sin(complex(u) * math.pi / 2)
    for v in reversed(_landen(k)):
        w = (1 + v) * w / (1 + v * w * w)
    return w


def _asne(w, k):
    v = _landen(k)
    w = complex(w)
    for n, vn in enumerate(v):
        kprev = k if n == 0 else v[n - 1]
        w = w * 2 / ((1 + vn) * (1 + cmath.sqrt(1 - w * w * kprev * kprev)))
    return 2 / math.pi * cmath.asin(w)


def _ellipdeg2(frac, k1):
    """Solve the degree equation through the nome expansion."""
    K1 = ellipk(k1)
    K1p = ellipk(math.sqrt(1 - k1 * k1))
    q1 = math.exp(-math.pi * K1p / K1)
    q = q1 ** frac
    M = 7
    num = sum(q ** (m * (m + 1)) for m in range(M + 1))
    den = 1 + 2 * sum(q ** (m * m) for m in range(1, M + 1))
    return 4 * math.sqrt(q) * (num / den) ** 2


def _ellipdeg(N, k1):
    """Given the ripple modulus k1, return the selectivity modulus k."""
    if k1 < 1e-6:
        return _ellipdeg2(1.0 / N, k1)
    L = N // 2
    kc = math.sqrt(1 - k1 * k1)
    prod = 1.0
    for i in range(1, L + 1):
        prod *= abs(_sne((2 * i - 1) / N, kc)) ** 4
    kp = min(1.0, kc ** N * prod)
    return math.sqrt(max(0.0, 1 - kp * kp))


def ellipap(N, rp, rs):
    if rp <= 0 or rs <= 0:
        _err("ellip: Rp and Rs must both be > 0 dB")
    ep = math.sqrt(10 ** (0.1 * rp) - 1)
    es = math.sqrt(10 ** (0.1 * rs) - 1)
    if es <= ep:
        _err("ellip: Rs must be larger than Rp")
    if N == 1:
        p = [complex(-1.0 / ep, 0.0)]
        return [], p, 1.0 / ep
    k1 = ep / es
    k = _ellipdeg(N, k1)
    L = N // 2
    r = N % 2
    ui = [(2 * i - 1) / N for i in range(1, L + 1)]
    zeros, poles = [], []
    for u in ui:
        zeta = _cde(u, k)
        za = 1j / (k * zeta)
        zeros.extend([za, za.conjugate()])
    v0 = (-1j / N) * _asne(1j / ep, k1)
    for u in ui:
        pa = 1j * _cde(complex(u) - 1j * v0, k)
        if pa.real > 0:
            pa = -pa.conjugate()
        poles.extend([pa, pa.conjugate()])
    if r:
        pa0 = 1j * _sne(1j * v0, k)
        pa0 = complex(-abs(pa0.real), 0.0)
        poles.append(pa0)
    G0 = 1.0 if N % 2 else 1.0 / math.sqrt(1 + ep * ep)
    np_ = 1.0 + 0j
    for pp in poles:
        np_ *= -pp
    nz = 1.0 + 0j
    for zz in zeros:
        nz *= -zz
    kgain = (G0 * np_ / nz).real
    return zeros, poles, kgain


# --- frequency transforms --------------------------------------------------
def _lp2lp(z, p, k, wo):
    deg = len(p) - len(z)
    return [wo * zz for zz in z], [wo * pp for pp in p], k * wo ** deg


def _lp2hp(z, p, k, wo):
    deg = len(p) - len(z)
    zn = [wo / zz for zz in z if zz != 0]
    pn = [wo / pp for pp in p if pp != 0]
    zn += [0j] * deg
    num = 1.0 + 0j
    for zz in z:
        num *= -zz
    den = 1.0 + 0j
    for pp in p:
        den *= -pp
    return zn, pn, (k * (num / den)).real if den != 0 else k


def _lp2bp(z, p, k, wo, bw):
    deg = len(p) - len(z)
    zn, pn = [], []
    for zz in z:
        h = zz * bw / 2
        d = cmath.sqrt(h * h - wo * wo)
        zn.extend([h + d, h - d])
    for pp in p:
        h = pp * bw / 2
        d = cmath.sqrt(h * h - wo * wo)
        pn.extend([h + d, h - d])
    zn += [0j] * deg
    return zn, pn, k * bw ** deg


def _lp2bs(z, p, k, wo, bw):
    deg = len(p) - len(z)
    zn, pn = [], []
    for zz in z:
        h = (bw / 2) / zz
        d = cmath.sqrt(h * h - wo * wo)
        zn.extend([h + d, h - d])
    for pp in p:
        h = (bw / 2) / pp
        d = cmath.sqrt(h * h - wo * wo)
        pn.extend([h + d, h - d])
    for _ in range(deg):
        zn.extend([1j * wo, -1j * wo])
    num = 1.0 + 0j
    for zz in z:
        num *= -zz
    den = 1.0 + 0j
    for pp in p:
        den *= -pp
    return zn, pn, (k * (num / den)).real if den != 0 else k


def _bilinear_zpk(z, p, k, fs):
    deg = len(p) - len(z)
    fs2 = 2.0 * fs
    zd = [(fs2 + zz) / (fs2 - zz) for zz in z]
    pd = [(fs2 + pp) / (fs2 - pp) for pp in p]
    zd += [-1.0 + 0j] * deg
    num = 1.0 + 0j
    for zz in z:
        num *= (fs2 - zz)
    den = 1.0 + 0j
    for pp in p:
        den *= (fs2 - pp)
    return zd, pd, k * (num / den).real


def _zpk2ba(z, p, k):
    b = [_re(_clean(k * c, 1e-8)) for c in poly_from_roots(z)]
    a = [_re(_clean(c, 1e-8)) for c in poly_from_roots(p)]
    return b, a


def _ba2zpk(b, a):
    b = _ptrim([complex(v) for v in b])
    a = _ptrim([complex(v) for v in a])
    if not a or a[0] == 0:
        _err("a(1) must be non-zero")
    k = (b[0] / a[0]).real if b else 0.0
    return roots(b) if len(b) > 1 else [], roots(a) if len(a) > 1 else [], k


def _btype(name):
    n = (name or "low").lower().strip()
    if n in ("low", "lowpass"):
        return "low"
    if n in ("high", "highpass"):
        return "high"
    if n in ("bandpass", "pass", "dc-0"):     # DC-0 / DC-1: kaiserord's multiband ftype
        return "bandpass"
    if n in ("stop", "bandstop", "notch", "dc-1"):
        return "stop"
    _err(f"unknown filter type '{name}' (use 'low', 'high', 'bandpass' or 'stop')")


def iirfilter(N, Wn, rp=None, rs=None, btype="low", ftype="butter", analog=False):
    """Design an IIR filter; returns (b, a). Wn is a scalar or a 2-list.

    Digital: Wn normalized so 1.0 == the Nyquist frequency (MATLAB convention).
    Analog: Wn in rad/s.
    """
    N = int(N)
    if N < 1:
        _err("filter order must be >= 1")
    Wn = [float(_re(v)) for v in _lst(Wn)]
    btype = _btype(btype)
    if btype in ("bandpass", "stop") and len(Wn) != 2:
        _err(f"{btype} design needs a two-element Wn = [w1 w2]")
    if btype in ("low", "high") and len(Wn) != 1:
        # MATLAB infers bandpass when two edges are given
        btype = "bandpass" if btype == "low" else "stop"
    if not analog:
        for w in Wn:
            if not (0 < w < 1):
                _err("digital Wn must be between 0 and 1 (1 = Nyquist)")
    ftype = ftype.lower()
    if ftype == "butter":
        z, p, k = buttap(N)
    elif ftype == "cheby1":
        z, p, k = cheb1ap(N, rp)
    elif ftype == "cheby2":
        z, p, k = cheb2ap(N, rs)
    elif ftype == "ellip":
        z, p, k = ellipap(N, rp, rs)
    elif ftype == "bessel":
        z, p, k = besselap(N)
    else:
        _err(f"unknown design '{ftype}'")

    if analog:
        warped = Wn
        fs = None
    else:
        fs = 2.0
        warped = [2 * fs * math.tan(math.pi * w / fs) for w in Wn]

    if btype == "low":
        z, p, k = _lp2lp(z, p, k, warped[0])
    elif btype == "high":
        z, p, k = _lp2hp(z, p, k, warped[0])
    else:
        bw = warped[1] - warped[0]
        wo = math.sqrt(warped[0] * warped[1])
        if bw <= 0:
            _err("Wn must be increasing: Wn(1) < Wn(2)")
        if btype == "bandpass":
            z, p, k = _lp2bp(z, p, k, wo, bw)
        else:
            z, p, k = _lp2bs(z, p, k, wo, bw)

    if not analog:
        z, p, k = _bilinear_zpk(z, p, k, fs)
    return _zpk2ba(z, p, k)


def _split_iir_args(name, args, needs_rp, needs_rs):
    """Parse MATLAB-style (n, Wn, [Rp,] [Rs,] ['type'], ['s'])."""
    if len(args) < 2:
        _err(f"{name} needs at least an order and a cutoff: {name}(n, Wn, ...)")
    N = _int(args[0], "n")
    Wn = args[1]
    i = 2
    rp = rs = None
    if needs_rp:
        if i >= len(args):
            _err(f"{name}(n, Rp, Wn, ...) — missing ripple")
        rp = _re(_scalar(args[i]))
        i += 1
    if needs_rs:
        if i >= len(args):
            _err(f"{name}(n, Rs, Wn, ...) — missing attenuation")
        rs = _re(_scalar(args[i]))
        i += 1
    btype = "low"
    analog = False
    for a in args[i:]:
        if isinstance(a, str):
            if a.lower() == "s":
                analog = True
            else:
                btype = a
    if isinstance(Wn, (list, tuple)) and len(Wn) == 2 and btype in ("low", "Low"):
        btype = "bandpass"
    return N, Wn, rp, rs, btype, analog


def butter(args):
    N, Wn, _, _, bt, an = _split_iir_args("butter", args, False, False)
    return iirfilter(N, Wn, btype=bt, ftype="butter", analog=an)


def cheby1(args):
    # MATLAB: cheby1(n, Rp, Wn, ...)
    if len(args) < 3:
        _err("cheby1(n, Rp, Wn, ...) needs three arguments")
    N = _int(args[0], "n")
    rp = _re(_scalar(args[1]))
    Wn = args[2]
    bt, an = _tail_opts(args[3:], Wn)
    return iirfilter(N, Wn, rp=rp, btype=bt, ftype="cheby1", analog=an)


def cheby2(args):
    if len(args) < 3:
        _err("cheby2(n, Rs, Wn, ...) needs three arguments")
    N = _int(args[0], "n")
    rs = _re(_scalar(args[1]))
    Wn = args[2]
    bt, an = _tail_opts(args[3:], Wn)
    return iirfilter(N, Wn, rs=rs, btype=bt, ftype="cheby2", analog=an)


def ellip(args):
    if len(args) < 4:
        _err("ellip(n, Rp, Rs, Wn, ...) needs four arguments")
    N = _int(args[0], "n")
    rp = _re(_scalar(args[1]))
    rs = _re(_scalar(args[2]))
    Wn = args[3]
    bt, an = _tail_opts(args[4:], Wn)
    return iirfilter(N, Wn, rp=rp, rs=rs, btype=bt, ftype="ellip", analog=an)


def besself(args):
    if len(args) < 2:
        _err("besself(n, Wo) needs two arguments")
    N = _int(args[0], "n")
    Wn = args[1]
    bt, _ = _tail_opts(args[2:], Wn)
    return iirfilter(N, Wn, btype=bt, ftype="bessel", analog=True)


def _tail_opts(tail, Wn):
    btype = "low"
    analog = False
    for a in tail:
        if isinstance(a, str):
            if a.lower() == "s":
                analog = True
            else:
                btype = a
    if isinstance(Wn, (list, tuple)) and len(Wn) == 2 and btype == "low":
        btype = "bandpass"
    return btype, analog


# --- order selection -------------------------------------------------------
def _ord_setup(wp, ws, analog):
    wp = [float(_re(v)) for v in _lst(wp)]
    ws = [float(_re(v)) for v in _lst(ws)]
    if len(wp) != len(ws):
        _err("Wp and Ws must have the same number of edges")
    if analog:
        return wp, ws
    for w in wp + ws:
        if not (0 < w < 1):
            _err("digital band edges must be between 0 and 1 (1 = Nyquist)")
    return [math.tan(math.pi * w / 2) for w in wp], [math.tan(math.pi * w / 2) for w in ws]


def _nat_ratio(wpw, wsw):
    """Selectivity ratio and band type from the warped edges."""
    if len(wpw) == 1:
        if wpw[0] < wsw[0]:
            return wsw[0] / wpw[0], "low"
        return wpw[0] / wsw[0], "high"
    if wsw[0] > wpw[0]:            # stopband inside passband -> bandstop
        nat = [(w * (wpw[0] - wpw[1])) / (w * w - wpw[0] * wpw[1]) for w in wsw]
        return min(abs(v) for v in nat), "stop"
    nat = [(w * w - wpw[0] * wpw[1]) / (w * (wpw[0] - wpw[1])) for w in wsw]
    return min(abs(v) for v in nat), "bandpass"


def _unwarp(w, analog):
    return w if analog else 2.0 / math.pi * math.atan(w)


def _wn_from_w0(W0, wpw, kind, analog):
    if kind == "low":
        return [_unwarp(W0 * wpw[0], analog)]
    if kind == "high":
        return [_unwarp(wpw[0] / W0, analog)]
    # band: solve for the two edges that keep the geometric centre
    w0 = math.sqrt(wpw[0] * wpw[1])
    bw = (wpw[1] - wpw[0])
    if kind == "bandpass":
        bwn = bw * W0
    else:
        bwn = bw / W0
    lo = (-bwn + math.sqrt(bwn * bwn + 4 * w0 * w0)) / 2
    hi = lo + bwn
    return [_unwarp(lo, analog), _unwarp(hi, analog)]


def _ord_common(args, name):
    if len(args) < 4:
        _err(f"{name}(Wp, Ws, Rp, Rs) needs four arguments")
    analog = any(isinstance(a, str) and a.lower() == "s" for a in args[4:])
    wpw, wsw = _ord_setup(args[0], args[1], analog)
    rp = _re(_scalar(args[2]))
    rs = _re(_scalar(args[3]))
    if rp <= 0 or rs <= 0:
        _err(f"{name}: Rp and Rs must be > 0 dB")
    nat, kind = _nat_ratio(wpw, wsw)
    if nat <= 1:
        _err(f"{name}: the stopband edge must be further out than the passband edge")
    return wpw, wsw, rp, rs, nat, kind, analog


def buttord(args):
    wpw, wsw, rp, rs, nat, kind, analog = _ord_common(args, "buttord")
    GP = 10 ** (0.1 * rp)
    GS = 10 ** (0.1 * rs)
    N = int(math.ceil(math.log10((GS - 1) / (GP - 1)) / (2 * math.log10(nat))))
    N = max(1, N)
    W0 = (GP - 1) ** (-1.0 / (2 * N))
    return float(N), _wn_from_w0(W0, wpw, kind, analog)


def cheb1ord(args):
    wpw, wsw, rp, rs, nat, kind, analog = _ord_common(args, "cheb1ord")
    GP = 10 ** (0.1 * rp)
    GS = 10 ** (0.1 * rs)
    N = int(math.ceil(math.acosh(math.sqrt((GS - 1) / (GP - 1))) / math.acosh(nat)))
    N = max(1, N)
    return float(N), _wn_from_w0(1.0, wpw, kind, analog)


def cheb2ord(args):
    wpw, wsw, rp, rs, nat, kind, analog = _ord_common(args, "cheb2ord")
    GP = 10 ** (0.1 * rp)
    GS = 10 ** (0.1 * rs)
    N = int(math.ceil(math.acosh(math.sqrt((GS - 1) / (GP - 1))) / math.acosh(nat)))
    N = max(1, N)
    # Chebyshev II: Wn is the stopband edge that lands the passband edge exactly
    # on Wp for this order (MATLAB convention): Ws' = Wp * cosh(acosh(sqrt(...))/N).
    # (The old code returned Ws / cosh(...), which put the stopband INSIDE the
    # passband and gave -35 dB at Wp — see the ECE08 Lab 7 regression.)
    C = math.cosh(math.acosh(math.sqrt((GS - 1) / (GP - 1))) / N)
    if kind == "low":
        return float(N), [_unwarp(wpw[0] * C, analog)]
    if kind == "high":
        return float(N), [_unwarp(wpw[0] / C, analog)]
    return float(N), [_unwarp(w, analog) for w in wsw]


def ellipord(args):
    wpw, wsw, rp, rs, nat, kind, analog = _ord_common(args, "ellipord")
    GP = 10 ** (0.1 * rp)
    GS = 10 ** (0.1 * rs)
    arg1 = math.sqrt((GP - 1) / (GS - 1))
    arg0 = 1.0 / nat
    d0 = (ellipk(arg0), ellipk(math.sqrt(max(0.0, 1 - arg0 * arg0))))
    d1 = (ellipk(arg1), ellipk(math.sqrt(max(0.0, 1 - arg1 * arg1))))
    N = int(math.ceil(d0[0] * d1[1] / (d0[1] * d1[0])))
    N = max(1, N)
    return float(N), _wn_from_w0(1.0, wpw, kind, analog)


# --- FIR design ------------------------------------------------------------
def _fir_bands(Wn, ftype):
    """Return the pass bands as (left, right) pairs in units of Nyquist."""
    Wn = [float(_re(v)) for v in _lst(Wn)]
    for w in Wn:
        if not (0 < w < 1):
            _err("fir1: cutoff frequencies must be between 0 and 1 (1 = Nyquist)")
    t = _btype(ftype)
    if len(Wn) == 1:
        if t in ("bandpass", "stop"):
            _err("fir1: bandpass/stop needs two cutoff frequencies")
        return ([(0.0, Wn[0])] if t == "low" else [(Wn[0], 1.0)]), t
    if len(Wn) == 2:
        if t == "low":
            t = "bandpass"
        if t in ("bandpass",):
            return [(Wn[0], Wn[1])], t
        return [(0.0, Wn[0]), (Wn[1], 1.0)], "stop"
    # multiband: alternate, starting in the stopband unless 'DC-1' asked
    bands = []
    edges = [0.0] + Wn + [1.0]
    start = 0 if t in ("low", "stop") else 1
    for i in range(start, len(edges) - 1, 2):
        bands.append((edges[i], edges[i + 1]))
    return bands, t


def fir1(args):
    """fir1(n, Wn, 'type', window) — windowed linear-phase FIR."""
    if len(args) < 2:
        _err("fir1(n, Wn, ...) needs at least two arguments")
    n = _int(args[0], "n")
    Wn = args[1]
    ftype = "low"
    window = None
    for a in args[2:]:
        if isinstance(a, str):
            ftype = a
        elif isinstance(a, (list, tuple)):
            window = _reals(a)
    bands, t = _fir_bands(Wn, ftype)
    if t in ("high", "stop") and n % 2:
        n += 1                       # Type II cannot pass Nyquist
    taps = n + 1
    if window is None:
        window = _hamming(taps)
    if len(window) != taps:
        _err(f"fir1: the window must have n+1 = {taps} points")
    alpha = n / 2.0
    h = [0.0] * taps
    for k in range(taps):
        x = k - alpha
        acc = 0.0
        for lo, hi in bands:
            acc += hi * _sinc(hi * x) - lo * _sinc(lo * x)
        h[k] = acc * window[k]
    # normalize the gain at the middle of the first passband
    lo, hi = bands[0]
    if lo == 0:
        f0 = 0.0
    elif hi == 1.0:
        f0 = 1.0
    else:
        f0 = (lo + hi) / 2
    s = sum(h[k] * math.cos(math.pi * f0 * (k - alpha)) for k in range(taps))
    if abs(s) > 1e-12:
        h = [v / s for v in h]
    return h


def fir2(args):
    """fir2(n, f, m, npt, window) — frequency-sampling FIR design."""
    if len(args) < 3:
        _err("fir2(n, f, m) needs at least three arguments")
    n = _int(args[0], "n")
    f = _reals(args[1])
    m = _reals(args[2])
    if len(f) != len(m):
        _err("fir2: f and m must be the same length")
    if abs(f[0]) > 1e-12 or abs(f[-1] - 1.0) > 1e-12:
        _err("fir2: f must start at 0 and end at 1")
    window = None
    npt = None
    for a in args[3:]:
        if isinstance(a, (list, tuple)):
            window = _reals(a)
        elif a is not None:
            npt = _int(a, "npt")
    taps = n + 1
    if npt is None:
        npt = 512
        while npt < 8 * taps:
            npt *= 2
    if window is None:
        window = _hamming(taps)
    if len(window) != taps:
        _err(f"fir2: the window must have n+1 = {taps} points")
    grid = [k / npt for k in range(npt + 1)]
    mag = _interp_lin(f, m, grid)
    alpha = n / 2.0
    spec = [mag[k] * cmath.exp(-1j * math.pi * alpha * grid[k]) for k in range(npt + 1)]
    full = spec + [spec[npt - k].conjugate() for k in range(1, npt)]
    h = ifft(full)
    return [h[k].real * window[k] for k in range(taps)]


def _interp_lin(xs, ys, xq):
    out = []
    for q in xq:
        if q <= xs[0]:
            out.append(ys[0])
            continue
        if q >= xs[-1]:
            out.append(ys[-1])
            continue
        for i in range(len(xs) - 1):
            if xs[i] <= q <= xs[i + 1]:
                if xs[i + 1] == xs[i]:
                    out.append(ys[i + 1])
                else:
                    t = (q - xs[i]) / (xs[i + 1] - xs[i])
                    out.append(ys[i] + t * (ys[i + 1] - ys[i]))
                break
    return out


def _parse_bands(f, a):
    """f = band edge pairs (0..1 of Nyquist), a = amplitude per edge."""
    if len(f) % 2 or len(f) != len(a):
        _err("band edges f and amplitudes a must pair up: f = [f1 f2 f3 f4], a = [a1 a2 a3 a4]")
    bands = []
    for i in range(0, len(f), 2):
        bands.append((f[i], f[i + 1], a[i], a[i + 1]))
    return bands


def firls(args):
    """firls(n, f, a) — linear-phase FIR by weighted least squares (Type I)."""
    if len(args) < 3:
        _err("firls(n, f, a) needs three arguments")
    n = _int(args[0], "n")
    f = _reals(args[1])
    a = _reals(args[2])
    w = _reals(args[3]) if len(args) > 3 else None
    if n % 2:
        n += 1
    bands = _parse_bands(f, a)
    if w is None:
        w = [1.0] * len(bands)
    if len(w) != len(bands):
        _err("firls: one weight per band")
    L = n // 2
    # Least squares over a dense grid (simple, robust, and fast enough here)
    grid, des, wt = _dense_grid(bands, w, 16 * (L + 1))
    cols = L + 1
    A = [[0.0] * cols for _ in range(cols)]
    rhs = [0.0] * cols
    for gi, om in enumerate(grid):
        basis = [math.cos(k * om) for k in range(cols)]
        ww = wt[gi] ** 2
        for r in range(cols):
            rhs[r] += ww * des[gi] * basis[r]
            for c in range(cols):
                A[r][c] += ww * basis[r] * basis[c]
    coef = [_re(v) for v in _lin_solve(A, rhs)]
    h = [0.0] * (n + 1)
    h[L] = coef[0]
    for k in range(1, L + 1):
        h[L - k] = coef[k] / 2
        h[L + k] = coef[k] / 2
    return h


def _dense_grid(bands, weights, density):
    grid, des, wt = [], [], []
    total = sum(max(1e-9, b[1] - b[0]) for b in bands)
    for bi, (f1, f2, a1, a2) in enumerate(bands):
        npts = max(4, int(round(density * (f2 - f1) / total)))
        for k in range(npts + 1):
            t = k / npts
            fr = f1 + (f2 - f1) * t
            grid.append(math.pi * fr)
            des.append(a1 + (a2 - a1) * t)
            wt.append(weights[bi])
    return grid, des, wt


def firpm(args):
    """firpm(n, f, a, w) — equiripple (Parks-McClellan / Remez) FIR design."""
    if len(args) < 3:
        _err("firpm(n, f, a) needs three arguments")
    n = _int(args[0], "n")
    f = _reals(args[1])
    a = _reals(args[2])
    w = _reals(args[3]) if len(args) > 3 else None
    bands = _parse_bands(f, a)
    if w is None:
        w = [1.0] * len(bands)
    if len(w) != len(bands):
        _err("firpm: one weight per band")

    typeII = (n % 2 == 1)
    r = (n - 1) // 2 if typeII else n // 2      # degree of the cosine polynomial
    if r < 1:
        _err("firpm: order is too low")
    density = max(16 * (r + 1), 200)
    grid, des, wt = _dense_grid(bands, w, density)
    if typeII:
        # H(w) = cos(w/2) * P(w): fold cos(w/2) into the desired/weight functions
        keep = []
        for i, om in enumerate(grid):
            q = math.cos(om / 2)
            if q < 1e-6:
                continue
            keep.append((om, des[i] / q, wt[i] * q))
        if not keep:
            _err("firpm: the band definition leaves nothing to fit")
        grid = [k[0] for k in keep]
        des = [k[1] for k in keep]
        wt = [k[2] for k in keep]

    ext = _remez(r, grid, des, wt)
    # Sample the resulting amplitude response and reconstruct h by IDFT
    x = [math.cos(grid[i]) for i in ext]
    dd = [des[i] for i in ext]
    ww = [wt[i] for i in ext]
    delta, yv = _remez_solve(x, dd, ww)
    taps = n + 1
    amp = []
    for k in range(taps):
        om = 2 * math.pi * k / taps
        p = _bary(math.cos(om), x[:r + 1], yv[:r + 1])
        amp.append(p * math.cos(om / 2) if typeII else p)
    spec = [amp[k] * cmath.exp(-1j * (2 * math.pi * k / taps) * (n / 2.0))
            for k in range(taps)]
    h = ifft(spec)
    return [v.real for v in h]


def _remez_solve(x, des, wt):
    m = len(x)
    b = []
    for i in range(m):
        num = 1.0
        for j in range(m):
            if j != i:
                d = x[i] - x[j]
                num /= d if d != 0 else 1e-30
        b.append(num)
    top = sum(b[i] * des[i] for i in range(m))
    bot = sum(b[i] * ((-1) ** i) / wt[i] for i in range(m))
    delta = top / bot if bot != 0 else 0.0
    y = [des[i] - ((-1) ** i) * delta / wt[i] for i in range(m)]
    return delta, y


def _bary(xq, xs, ys):
    c = []
    for i in range(len(xs)):
        num = 1.0
        for j in range(len(xs)):
            if j != i:
                d = xs[i] - xs[j]
                num /= d if d != 0 else 1e-30
        c.append(num)
    top = bot = 0.0
    for i in range(len(xs)):
        d = xq - xs[i]
        if abs(d) < 1e-14:
            return ys[i]
        top += c[i] * ys[i] / d
        bot += c[i] / d
    return top / bot if bot != 0 else 0.0


def _remez(r, grid, des, wt, maxiter=40):
    """Remez exchange; returns the indices of r+2 extremal grid points."""
    ng = len(grid)
    m = r + 2
    if ng < m:
        _err("firpm: too few grid points — raise the order or widen the bands")
    ext = [int(round(k * (ng - 1) / (m - 1))) for k in range(m)]
    last = None
    for _ in range(maxiter):
        x = [math.cos(grid[i]) for i in ext]
        dd = [des[i] for i in ext]
        ww = [wt[i] for i in ext]
        delta, y = _remez_solve(x, dd, ww)
        xs, ys = x[:r + 1], y[:r + 1]
        err = []
        for i in range(ng):
            p = _bary(math.cos(grid[i]), xs, ys)
            err.append(wt[i] * (des[i] - p))
        cand = []
        for i in range(ng):
            lo = err[i - 1] if i > 0 else err[i]
            hi = err[i + 1] if i < ng - 1 else err[i]
            if (err[i] >= lo and err[i] >= hi and err[i] > 0) or \
               (err[i] <= lo and err[i] <= hi and err[i] < 0):
                if cand and _sgn(err[cand[-1]]) == _sgn(err[i]):
                    if abs(err[i]) > abs(err[cand[-1]]):
                        cand[-1] = i
                else:
                    cand.append(i)
        if len(cand) < m:
            break
        while len(cand) > m:
            if abs(err[cand[0]]) < abs(err[cand[-1]]):
                cand.pop(0)
            else:
                cand.pop()
        if cand == last:
            ext = cand
            break
        last = cand
        ext = cand
    return ext


def _sgn(v):
    return (v > 0) - (v < 0)


def kaiserord(args):
    """kaiserord(f, a, dev, fs) -> (n, Wn, beta, ftype) for a Kaiser-window design."""
    if len(args) < 3:
        _err("kaiserord(f, a, dev, fs) needs at least three arguments")
    f = _reals(args[0])
    a = _reals(args[1])
    dev = _reals(args[2])
    fs = _re(_scalar(args[3])) if len(args) > 3 else 2.0
    if len(f) < 2:
        _err("kaiserord: need at least one transition band")
    d = min(dev)
    A = -20 * math.log10(d)
    if A > 50:
        beta = 0.1102 * (A - 8.7)
    elif A >= 21:
        beta = 0.5842 * (A - 21) ** 0.4 + 0.07886 * (A - 21)
    else:
        beta = 0.0
    dw = 2 * math.pi * min(f[i + 1] - f[i] for i in range(0, len(f) - 1, 2)) / fs
    n = int(math.ceil((A - 8) / (2.285 * dw)))
    n = max(1, n)
    Wn = [(f[i] + f[i + 1]) / (fs / 2) for i in range(0, len(f) - 1, 2)]
    # filter type for fir1, from the band amplitudes (MATLAB's ftype output)
    pas = [1 if v > 0.5 else 0 for v in a]
    if len(pas) == 2:
        ftype = "low" if pas[0] else "high"
    elif len(pas) == 3 and pas == [0, 1, 0]:
        ftype = "bandpass"
    elif len(pas) == 3 and pas == [1, 0, 1]:
        ftype = "stop"
    else:
        ftype = "DC-1" if pas[0] else "DC-0"
    return float(n), [w / 2 for w in Wn], beta, ftype


def firpmord(args):
    """firpmord(f, a, dev, fs) -> (n, fo, ao, w) — Herrmann order estimate."""
    if len(args) < 3:
        _err("firpmord(f, a, dev, fs) needs at least three arguments")
    f = _reals(args[0])
    a = _reals(args[1])
    dev = _reals(args[2])
    fs = _re(_scalar(args[3])) if len(args) > 3 else 2.0
    if len(f) < 2:
        _err("firpmord: need at least one transition band")
    d1 = max(dev)
    d2 = min(dev)
    l1 = math.log10(d1)
    l2 = math.log10(d2)
    Dinf = (0.005309 * l1 * l1 + 0.07114 * l1 - 0.4761) * l2 \
        - (0.00266 * l1 * l1 + 0.5941 * l1 + 0.4278)
    fpen = 11.01217 + 0.51244 * (l1 - l2)
    df = min(f[i + 1] - f[i] for i in range(0, len(f) - 1, 2)) / fs
    n = int(math.ceil((Dinf - fpen * df * df) / df))
    n = max(1, n)
    fo = [0.0] + [v / (fs / 2) for v in f] + [1.0]
    ao = [v for v in a for _ in (0, 1)]        # one amplitude per band EDGE (MATLAB)
    w = [d1 / d for d in dev]
    return float(n), fo, ao, w


# --- responses -------------------------------------------------------------
def freqz(b, a, npts=512, whole=False, ws=None):
    """Digital frequency response. Returns (w, H) with w in rad/sample.
    MATLAB grid: npts points over [0, pi) — or [0, 2*pi) with whole — so
    w(k) = k*pi/npts; `ws` evaluates at given frequencies instead."""
    b = [complex(v) for v in _lst(b)]
    a = [complex(v) for v in _lst(a)] if a else [1.0 + 0j]
    if ws is None:
        top = 2 * math.pi if whole else math.pi
        ws = [top * k / npts for k in range(npts)]
    H = []
    for w in ws:
        z = cmath.exp(-1j * w)
        num = sum(b[k] * z ** k for k in range(len(b)))
        den = sum(a[k] * z ** k for k in range(len(a)))
        H.append(num / den if den != 0 else complex(math.inf))
    return ws, H


def freqs(b, a, w=None, npts=400):
    """Analog frequency response. Returns (w, H)."""
    b = [complex(v) for v in _lst(b)]
    a = [complex(v) for v in _lst(a)]
    if w is None:
        crit = [abs(r) for r in roots(b) + roots(a) if abs(r) > 1e-9]
        lo = math.floor(math.log10(min(crit))) - 1 if crit else -1
        hi = math.ceil(math.log10(max(crit))) + 1 if crit else 2
        if hi - lo < 3:
            hi = lo + 3
        w = [10.0 ** (lo + (hi - lo) * k / (npts - 1)) for k in range(npts)]
    else:
        w = _reals(w)
    H = []
    for wi in w:
        den = _pval(a, 1j * wi)
        H.append(_pval(b, 1j * wi) / den if den != 0 else complex(math.inf))
    return w, H


def grpdelay(b, a, npts=512, whole=False, ws=None):
    """Group delay in samples. Returns (w, gd) on the same grid as freqz."""
    b = [complex(v) for v in _lst(b)]
    a = [complex(v) for v in _lst(a)] if a else [1.0 + 0j]
    if ws is None:
        top = 2 * math.pi if whole else math.pi
        ws = [top * k / npts for k in range(npts)]
    gd = []
    for w in ws:
        z = cmath.exp(-1j * w)
        B = sum(b[k] * z ** k for k in range(len(b)))
        Bp = sum(k * b[k] * z ** k for k in range(len(b)))
        A = sum(a[k] * z ** k for k in range(len(a)))
        Ap = sum(k * a[k] * z ** k for k in range(len(a)))
        tb = (Bp / B).real if abs(B) > 1e-14 else 0.0
        ta = (Ap / A).real if abs(A) > 1e-14 else 0.0
        gd.append(tb - ta)
    return ws, gd


# --- second-order sections (flattened 6*L row vector) ----------------------
def tf2sos(args):
    """tf2sos(b, a) — cascade of biquads as a FLAT row of 6*L numbers
    [b0 b1 b2 a0 a1 a2  b0 b1 b2 a0 a1 a2 ...] (the DEMO interpreter has no
    2-D matrices, so the standard L-by-6 matrix is returned row-major)."""
    if len(args) < 2:
        _err("tf2sos(b, a) needs two arguments")
    z, p, k = _ba2zpk(_reals(args[0]), _reals(args[1]))
    return zp2sos([z, p, k])


def zp2sos(args):
    if len(args) < 3:
        _err("zp2sos(z, p, k) needs three arguments")
    z = [complex(v) for v in _lst(args[0])]
    p = [complex(v) for v in _lst(args[1])]
    k = _re(_scalar(args[2]))
    zs = _pair_roots(z)
    ps = _pair_roots(p)
    L = max(len(zs), len(ps), 1)
    while len(zs) < L:
        zs.append([])
    while len(ps) < L:
        ps.append([])
    out = []
    for i in range(L):
        bn = [_re(_clean(c, 1e-8)) for c in poly_from_roots(zs[i])]
        an = [_re(_clean(c, 1e-8)) for c in poly_from_roots(ps[i])]
        bn = bn + [0.0] * (3 - len(bn))
        an = an + [0.0] * (3 - len(an))
        if i == 0:
            bn = [k * v for v in bn]
        out.extend(bn[:3] + an[:3])
    return out


# Root-finding on a repeated root of multiplicity m carries an error of about
# eps**(1/m) (~1e-4 for a 4-fold root), so section pairing uses a generous
# relative tolerance. Genuine filter roots are far further apart than this.
_PAIR_TOL = 2e-3


def _pair_roots(rs):
    """Group roots into conjugate pairs / singles for biquad sections."""
    # snap near-real roots onto the real axis first
    rest = []
    for r in rs:
        r = complex(r)
        rest.append(complex(r.real, 0.0) if abs(r.imag) <= _PAIR_TOL * (1 + abs(r))
                    else r)
    rest.sort(key=lambda r: (abs(r.imag) < 1e-12, r.real, r.imag))
    out = []
    while rest:
        r = rest.pop(0)
        if r.imag != 0.0:
            tol = _PAIR_TOL * (1 + abs(r))
            best, bestd = None, tol
            for i, q in enumerate(rest):
                d = abs(q - r.conjugate())
                if d < bestd:
                    best, bestd = i, d
            if best is None:
                out.append([r])
            else:
                q = rest.pop(best)
                mid = complex((r.real + q.real) / 2, (abs(r.imag) + abs(q.imag)) / 2)
                out.append([mid, mid.conjugate()])
        else:
            if rest and rest[0].imag == 0.0:
                out.append([r, rest.pop(0)])
            else:
                out.append([r])
    return out


def sos2tf(args):
    """sos2tf(sos) — flat 6*L row back to (b, a)."""
    s = _reals(args[0])
    if len(s) % 6:
        _err("sos2tf: the flattened sos vector must have a multiple of 6 entries")
    b = [1.0 + 0j]
    a = [1.0 + 0j]
    for i in range(0, len(s), 6):
        b = _pmul(b, [complex(v) for v in s[i:i + 3]])
        a = _pmul(a, [complex(v) for v in s[i + 3:i + 6]])
    return [_re(_clean(v, 1e-8)) for v in _ptrim(b)], [_re(_clean(v, 1e-8)) for v in _ptrim(a)]


def sosfilt(args):
    """sosfilt(sos, x) — run a cascade of biquads."""
    if len(args) < 2:
        _err("sosfilt(sos, x) needs two arguments")
    s = _reals(args[0])
    x = _reals(args[1])
    if len(s) % 6:
        _err("sosfilt: the flattened sos vector must have a multiple of 6 entries")
    y = x[:]
    for i in range(0, len(s), 6):
        y = lfilter(s[i:i + 3], s[i + 3:i + 6], y)
    return y


# ===========================================================================
# Chapter 7 — random signals, correlation, LPC
# ===========================================================================
def awgn(args):
    """awgn(x, snr_db) — add white Gaussian noise at the requested SNR."""
    x = _reals(args[0])
    snr_db = _re(_scalar(args[1])) if len(args) > 1 else 10.0
    n = len(x)
    if n == 0:
        return []
    ps = sum(v * v for v in x) / n
    pn = ps / (10 ** (snr_db / 10.0))
    sd = math.sqrt(pn)
    return [x[i] + random.gauss(0.0, sd) for i in range(n)]


def snr_db(args):
    """snr(signal, noise) — power ratio in dB."""
    if len(args) < 2:
        _err("snr(signal, noise) needs two arguments")
    s = _reals(args[0])
    nz = _reals(args[1])
    ps = sum(v * v for v in s)
    pn = sum(v * v for v in nz)
    if pn <= 0:
        return math.inf
    return 10 * math.log10(ps / pn)


def xcov(args):
    """xcov(x, y) — cross-covariance (cross-correlation of mean-removed data)."""
    x = _reals(args[0])
    y = _reals(args[1]) if len(args) > 1 and isinstance(args[1], (list, tuple)) else x
    mx = sum(x) / len(x) if x else 0.0
    my = sum(y) / len(y) if y else 0.0
    return xcorr_raw([v - mx for v in x], [v - my for v in y])


def xcorr_raw(x, y):
    n = max(len(x), len(y))
    x = list(x) + [0.0] * (n - len(x))
    y = list(y) + [0.0] * (n - len(y))
    out = []
    for lag in range(-(n - 1), n):
        s = 0.0
        for i in range(n):
            j = i - lag
            if 0 <= j < n:
                s += x[i] * y[j]
        out.append(s)
    return out


def levinson(args):
    """levinson(r, n) -> (a, e, k) — Levinson-Durbin recursion."""
    r = _reals(args[0])
    n = _int(args[1], "n") if len(args) > 1 else len(r) - 1
    if n < 1 or n >= len(r):
        _err("levinson: order must satisfy 1 <= n < length(r)")
    a = [1.0]
    e = r[0]
    if e == 0:
        _err("levinson: r(1) must be non-zero")
    ks = []
    for m in range(1, n + 1):
        acc = r[m] + sum(a[i] * r[m - i] for i in range(1, len(a)))
        k = -acc / e
        ks.append(k)
        newa = a + [0.0]
        for i in range(1, m + 1):
            newa[i] = a[i] if i < len(a) else 0.0
            newa[i] += k * (a[m - i] if (m - i) < len(a) else 0.0)
        a = newa
        e *= (1 - k * k)
        if e <= 0:
            e = 1e-30
            break
    return a, e, ks


def lpc(args):
    """lpc(x, p) -> (a, e) — linear-prediction coefficients."""
    x = _reals(args[0])
    p = _int(args[1], "p") if len(args) > 1 else 8
    if p < 1 or p >= len(x):
        _err("lpc: order must satisfy 1 <= p < length(x)")
    r = [sum(x[i] * x[i + k] for i in range(len(x) - k)) for k in range(p + 1)]
    if r[0] == 0:
        _err("lpc: the signal is all zeros")
    a, e, _ = levinson([r, p])
    return a, e


def autocorr_norm(args):
    """autocorr(x, maxlag) — normalized autocorrelation, lags 0..maxlag."""
    x = _reals(args[0])
    n = len(x)
    maxlag = _int(args[1], "maxlag") if len(args) > 1 else min(20, n - 1)
    m = sum(x) / n if n else 0.0
    xc = [v - m for v in x]
    r0 = sum(v * v for v in xc)
    if r0 == 0:
        _err("autocorr: the signal has zero variance")
    return [sum(xc[i] * xc[i + k] for i in range(n - k)) / r0 for k in range(maxlag + 1)]

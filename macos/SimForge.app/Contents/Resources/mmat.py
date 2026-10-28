"""
mmat — 2-D matrices and linear algebra for SimForge's DEMO interpreter.

Value model used by mdemo.py:
    float / complex        scalar
    list                   ROW vector (1 x n)          — unchanged, the fast path
    Mat                    r x c matrix; a COLUMN vector is a Mat with c == 1
    str, tf dict, ...      as before

A Mat with a single row is always collapsed back to a plain list and a 1x1 Mat
to a scalar (see `norm_val`), so every pre-existing vector routine keeps
working. Column-major order is used wherever MATLAB flattens (A(:), numel,
sum(A(:)), ...).

Pure stdlib. Root finding is borrowed from mdsp (also pure stdlib).
"""

import cmath
import math
import random

import mdsp


class MatError(ValueError):
    pass


def _err(msg):
    raise MatError(msg)


# ===========================================================================
# The matrix type
# ===========================================================================
class Mat:
    __slots__ = ("r", "c", "d")

    def __init__(self, rows):
        rows = [list(rw) for rw in rows]
        if not rows:
            self.r, self.c, self.d = 0, 0, []
            return
        c = len(rows[0])
        for rw in rows:
            if len(rw) != c:
                _err("matrix rows must all have the same number of columns")
        self.r, self.c, self.d = len(rows), c, rows

    def __repr__(self):
        return f"Mat({self.r}x{self.c})"

    def col(self, j):
        return [self.d[i][j] for i in range(self.r)]

    def T(self):
        return Mat([[self.d[i][j] for i in range(self.r)] for j in range(self.c)])


def is_mat(v):
    return isinstance(v, Mat)


def shape(v):
    """(rows, cols) for scalar / list / Mat / str."""
    if is_mat(v):
        return v.r, v.c
    if isinstance(v, list):
        return 1, len(v)
    if isinstance(v, str):
        return 1, len(v)
    return 1, 1


def to_mat(v):
    if is_mat(v):
        return v
    if isinstance(v, list):
        return Mat([list(v)])
    return Mat([[v]])


def norm_val(v):
    """Collapse 1xn -> list and 1x1 -> scalar so old code paths keep working."""
    if is_mat(v):
        if v.r == 1 and v.c == 1:
            return v.d[0][0]
        if v.r == 1:
            return list(v.d[0])
        if v.r == 0 or v.c == 0:
            return []
    return v


def flat_cm(v):
    """Column-major flatten (MATLAB's A(:) order)."""
    if is_mat(v):
        return [v.d[i][j] for j in range(v.c) for i in range(v.r)]
    if isinstance(v, list):
        return list(v)
    return [v]


def from_flat_cm(flat, r, c):
    if len(flat) != r * c:
        _err(f"cannot reshape {len(flat)} elements into {r}x{c}")
    return norm_val(Mat([[flat[j * r + i] for j in range(c)] for i in range(r)]))


def colvec(lst):
    return norm_val(Mat([[x] for x in lst]))


def numel(v):
    r, c = shape(v)
    return r * c


# ===========================================================================
# Broadcasting (MATLAB implicit expansion) and mapping
# ===========================================================================
def bcast(a, b, f):
    A, B = to_mat(a), to_mat(b)
    ra, ca, rb, cb = A.r, A.c, B.r, B.c
    if not ((ra == rb or ra == 1 or rb == 1) and (ca == cb or ca == 1 or cb == 1)):
        _err(f"array sizes don't match ({ra}x{ca} vs {rb}x{cb})")
    r, c = max(ra, rb), max(ca, cb)
    out = []
    for i in range(r):
        ia = 0 if ra == 1 else i
        ib = 0 if rb == 1 else i
        rowa, rowb = A.d[ia], B.d[ib]
        out.append([f(rowa[0 if ca == 1 else j], rowb[0 if cb == 1 else j])
                    for j in range(c)])
    return norm_val(Mat(out))


def mat_map(v, f):
    if is_mat(v):
        return norm_val(Mat([[f(x) for x in rw] for rw in v.d]))
    if isinstance(v, list):
        return [f(x) for x in v]
    return f(v)


# ===========================================================================
# Core linear algebra
# ===========================================================================
def transpose(v):
    if is_mat(v):
        return norm_val(v.T())
    if isinstance(v, list):
        return colvec(v)
    return v


def ctranspose(v):
    def cj(x):
        return x.conjugate() if isinstance(x, complex) else x
    return transpose(mat_map(v, cj))


def matmul(a, b):
    A, B = to_mat(a), to_mat(b)
    if A.r == 1 and A.c == 1:
        return bcast(a, b, lambda x, y: x * y)
    if B.r == 1 and B.c == 1:
        return bcast(a, b, lambda x, y: x * y)
    if A.c != B.r:
        _err(f"inner matrix dimensions must agree ({A.r}x{A.c} * {B.r}x{B.c})")
    Bt = B.T().d
    out = [[sum(A.d[i][k] * Bt[j][k] for k in range(A.c)) for j in range(B.c)]
           for i in range(A.r)]
    return norm_val(Mat(out))


def eye(n, m=None):
    m = n if m is None else m
    return norm_val(Mat([[1.0 if i == j else 0.0 for j in range(m)] for i in range(n)]))


def fill(r, c, val):
    return norm_val(Mat([[val] * c for _ in range(r)]))


def fill_fn(r, c, fn):
    return norm_val(Mat([[fn() for _ in range(c)] for _ in range(r)]))


def _lu_decomp(A):
    """Partial-pivot LU on a copy; returns (LU, perm, sign) or raises."""
    n = A.r
    M = [list(rw) for rw in A.d]
    perm = list(range(n))
    sign = 1.0
    for k in range(n):
        piv = max(range(k, n), key=lambda i: abs(M[i][k]))
        if abs(M[piv][k]) < 1e-300:
            return M, perm, 0.0
        if piv != k:
            M[k], M[piv] = M[piv], M[k]
            perm[k], perm[piv] = perm[piv], perm[k]
            sign = -sign
        for i in range(k + 1, n):
            f = M[i][k] / M[k][k]
            M[i][k] = f
            for j in range(k + 1, n):
                M[i][j] -= f * M[k][j]
    return M, perm, sign


def det(v):
    A = to_mat(v)
    if A.r != A.c:
        _err("det: matrix must be square")
    if A.r == 0:
        return 1.0
    LU, _, sign = _lu_decomp(A)
    if sign == 0.0:
        return 0.0
    d = sign
    for i in range(A.r):
        d *= LU[i][i]
    return _clean(d)


def solve(a, b):
    """A \\ B by Gauss-Jordan with partial pivoting (complex-capable)."""
    A, B = to_mat(a), to_mat(b)
    if A.r == 1 and A.c == 1:
        return bcast(b, a, lambda x, y: x / y)
    if A.r != A.c:
        return _lstsq(A, B)
    if A.r != B.r:
        _err(f"A\\b: dimensions do not agree ({A.r}x{A.c} vs {B.r}x{B.c})")
    n = A.r
    M = [[complex(A.d[i][j]) for j in range(n)] + [complex(B.d[i][j]) for j in range(B.c)]
         for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < 1e-14:
            _err("matrix is singular to working precision")
        M[col], M[piv] = M[piv], M[col]
        pv = M[col][col]
        for c in range(col, n + B.c):
            M[col][c] /= pv
        for r in range(n):
            if r != col and M[r][col] != 0:
                f = M[r][col]
                for c in range(col, n + B.c):
                    M[r][c] -= f * M[col][c]
    return norm_val(Mat([[_clean(M[i][n + j]) for j in range(B.c)] for i in range(n)]))


def _lstsq(A, B):
    """Least squares for non-square A: solve (A'A) x = A'B."""
    At = A.T()
    return solve(matmul(At, A), matmul(At, B))


def inv(v):
    A = to_mat(v)
    if A.r != A.c:
        _err("inv: matrix must be square")
    return solve(A, eye(A.r))


def mrdivide(a, b):
    """A / B  ==  A * inv(B)  ==  (B' \\ A')'"""
    B = to_mat(b)
    if B.r == 1 and B.c == 1:
        return bcast(a, b, lambda x, y: x / y)
    return transpose(solve(transpose(B), transpose(a)))


def mpower(a, k):
    A = to_mat(a)
    if A.r != A.c:
        _err("matrix power needs a square matrix")
    if abs(k - round(k)) > 1e-9:
        _err("matrix power needs an integer exponent (use .^ for element-wise)")
    k = int(round(k))
    if k < 0:
        A = to_mat(inv(A))
        k = -k
    result = to_mat(eye(A.r))
    base = A
    while k:
        if k & 1:
            result = to_mat(matmul(result, base))
        base = to_mat(matmul(base, base))
        k >>= 1
    return norm_val(result)


def trace(v):
    A = to_mat(v)
    return sum(A.d[i][i] for i in range(min(A.r, A.c)))


def rank(v, tol=None):
    A = to_mat(v)
    M = [[complex(x) for x in rw] for rw in A.d]
    r, c = A.r, A.c
    scale = max((abs(x) for rw in M for x in rw), default=0.0)
    tol = tol if tol is not None else max(r, c) * scale * 1e-12
    rk = 0
    row = 0
    for col in range(c):
        if row >= r:
            break
        piv = max(range(row, r), key=lambda i: abs(M[i][col]))
        if abs(M[piv][col]) <= tol:
            continue
        M[row], M[piv] = M[piv], M[row]
        for i in range(row + 1, r):
            f = M[i][col] / M[row][col]
            for j in range(col, c):
                M[i][j] -= f * M[row][j]
        row += 1
        rk += 1
    return float(rk)


def norm(v, kind=2):
    if isinstance(kind, str):
        kind = kind.lower()
    if not is_mat(v) or v.c == 1 or v.r == 1:
        x = flat_cm(v)
        if kind in (2, "fro"):
            return math.sqrt(sum(abs(e) ** 2 for e in x))
        if kind == 1:
            return sum(abs(e) for e in x)
        if kind in ("inf", math.inf):
            return max(abs(e) for e in x) if x else 0.0
        if kind in ("-inf", -math.inf):
            return min(abs(e) for e in x) if x else 0.0
        return sum(abs(e) ** kind for e in x) ** (1.0 / kind)
    A = v
    if kind == "fro":
        return math.sqrt(sum(abs(e) ** 2 for rw in A.d for e in rw))
    if kind == 1:
        return max(sum(abs(A.d[i][j]) for i in range(A.r)) for j in range(A.c))
    if kind in ("inf", math.inf):
        return max(sum(abs(e) for e in rw) for rw in A.d)
    # 2-norm: sqrt of the largest eigenvalue of A'A (power iteration)
    AtA = to_mat(matmul(ctranspose(A), A))
    x = [1.0] * AtA.r
    lam = 0.0
    for _ in range(300):
        y = [sum(AtA.d[i][k] * x[k] for k in range(AtA.r)) for i in range(AtA.r)]
        nrm = math.sqrt(sum(abs(e) ** 2 for e in y))
        if nrm == 0:
            return 0.0
        x = [e / nrm for e in y]
        if abs(nrm - lam) < 1e-13 * max(1.0, nrm):
            lam = nrm
            break
        lam = nrm
    return math.sqrt(abs(lam))


def _clean(x, tol=1e-10):
    if isinstance(x, complex):
        if abs(x.imag) <= tol * (1.0 + abs(x)):
            return x.real
        return x
    return x


# --- eigenvalues -----------------------------------------------------------
def _is_symmetric(A):
    if A.r != A.c:
        return False
    for i in range(A.r):
        for j in range(i + 1, A.c):
            a, b = A.d[i][j], A.d[j][i]
            if isinstance(a, complex) or isinstance(b, complex):
                return False
            if abs(a - b) > 1e-10 * (1 + abs(a) + abs(b)):
                return False
    return True


def _jacobi_eig(A):
    """Symmetric eigen-decomposition by cyclic Jacobi rotations."""
    n = A.r
    a = [[float(x) for x in rw] for rw in A.d]
    V = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for _sweep in range(100):
        off = sum(a[i][j] ** 2 for i in range(n) for j in range(n) if i != j)
        if off < 1e-22:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                if abs(a[p][q]) < 1e-300:
                    continue
                theta = (a[q][q] - a[p][p]) / (2 * a[p][q])
                t = (1.0 if theta >= 0 else -1.0) / (abs(theta) + math.sqrt(theta * theta + 1))
                c = 1 / math.sqrt(t * t + 1)
                s = t * c
                for k in range(n):
                    akp, akq = a[k][p], a[k][q]
                    a[k][p] = c * akp - s * akq
                    a[k][q] = s * akp + c * akq
                for k in range(n):
                    apk, aqk = a[p][k], a[q][k]
                    a[p][k] = c * apk - s * aqk
                    a[q][k] = s * apk + c * aqk
                for k in range(n):
                    vkp, vkq = V[k][p], V[k][q]
                    V[k][p] = c * vkp - s * vkq
                    V[k][q] = s * vkp + c * vkq
    vals = [a[i][i] for i in range(n)]
    order = sorted(range(n), key=lambda i: vals[i])
    vals = [vals[i] for i in order]
    V = [[V[k][i] for i in order] for k in range(n)]
    return vals, V


def charpoly(A):
    """Coefficients of det(sI - A), highest power first (Faddeev-LeVerrier)."""
    n = A.r
    I = to_mat(eye(n))
    M = to_mat(fill(n, n, 0.0))
    coeffs = [1.0]
    for k in range(1, n + 1):
        # M_k = A M_{k-1} + c_{n-k+1} I
        AM = to_mat(matmul(A, M)) if k > 1 else to_mat(fill(n, n, 0.0))
        M = to_mat(bcast(AM, bcast(I, coeffs[-1], lambda x, y: x * y),
                         lambda x, y: x + y))
        AMk = to_mat(matmul(A, M))
        ck = -trace(AMk) / k
        coeffs.append(ck)
    return coeffs


def eig(v, want_vectors=False):
    A = to_mat(v)
    if A.r != A.c:
        _err("eig: matrix must be square")
    n = A.r
    if n == 0:
        return ([], None) if want_vectors else []
    if _is_symmetric(A):
        vals, V = _jacobi_eig(A)
        if want_vectors:
            return vals, norm_val(Mat(V))
        return vals
    vals = [_clean(r) for r in mdsp.roots(charpoly(A))]
    vals.sort(key=lambda z: (abs(complex(z)), cmath.phase(complex(z))))
    if not want_vectors:
        return vals
    # eigenvectors by inverse iteration on (A - (lam + eps) I)
    V = []
    for lam in vals:
        shift = complex(lam) + 1e-7 * (1 + abs(complex(lam)))
        Ash = Mat([[complex(A.d[i][j]) - (shift if i == j else 0) for j in range(n)]
                   for i in range(n)])
        x = [complex(1.0, 0.0)] * n
        for _ in range(12):
            try:
                y = flat_cm(solve(Ash, colvec(x)))
            except MatError:
                break
            nrm = math.sqrt(sum(abs(e) ** 2 for e in y))
            if nrm == 0:
                break
            x = [e / nrm for e in y]
        # normalise the sign like MATLAB (largest component positive real)
        big = max(range(n), key=lambda i: abs(x[i]))
        ph = cmath.phase(x[big]) if x[big] != 0 else 0.0
        x = [_clean(e * cmath.exp(-1j * ph)) for e in x]
        V.append(x)
    Vm = Mat([[V[j][i] for j in range(n)] for i in range(n)])
    return vals, norm_val(Vm)


# ===========================================================================
# Shape / construction helpers
# ===========================================================================
def reshape(v, r, c):
    flat = flat_cm(v)
    if r is None:
        r = len(flat) // c
    if c is None:
        c = len(flat) // r
    return from_flat_cm(flat, int(r), int(c))


def repmat(v, m, n):
    A = to_mat(v)
    rows = []
    for _ in range(int(m)):
        for rw in A.d:
            rows.append(list(rw) * int(n))
    return norm_val(Mat(rows))


def diag(v, k=0):
    if is_mat(v) and v.r > 1 and v.c > 1:
        A = v
        out = []
        for i in range(A.r):
            j = i + k
            if 0 <= j < A.c:
                out.append(A.d[i][j])
        return colvec(out)
    x = flat_cm(v)
    n = len(x) + abs(k)
    M = [[0.0] * n for _ in range(n)]
    for i, val in enumerate(x):
        r, c = (i, i + k) if k >= 0 else (i - k, i)
        M[r][c] = val
    return norm_val(Mat(M))


def triu(v, k=0):
    A = to_mat(v)
    return norm_val(Mat([[A.d[i][j] if j - i >= k else 0.0 for j in range(A.c)]
                         for i in range(A.r)]))


def tril(v, k=0):
    A = to_mat(v)
    return norm_val(Mat([[A.d[i][j] if j - i <= k else 0.0 for j in range(A.c)]
                         for i in range(A.r)]))


def kron(a, b):
    A, B = to_mat(a), to_mat(b)
    rows = []
    for i in range(A.r):
        for p in range(B.r):
            rows.append([A.d[i][j] * B.d[p][q] for j in range(A.c) for q in range(B.c)])
    return norm_val(Mat(rows))


def cross(a, b):
    x, y = flat_cm(a), flat_cm(b)
    if len(x) != 3 or len(y) != 3:
        _err("cross: both vectors must have 3 elements")
    out = [x[1] * y[2] - x[2] * y[1], x[2] * y[0] - x[0] * y[2], x[0] * y[1] - x[1] * y[0]]
    return colvec(out) if (is_mat(a) and a.c == 1) else out


def magic(n):
    n = int(n)
    if n < 1:
        return []
    if n == 2:
        return norm_val(Mat([[1.0, 3.0], [4.0, 2.0]]))
    M = [[0] * n for _ in range(n)]
    if n % 2 == 1:
        i, j = 0, n // 2
        for k in range(1, n * n + 1):
            M[i][j] = k
            ni, nj = (i - 1) % n, (j + 1) % n
            if M[ni][nj]:
                ni, nj = (i + 1) % n, j
            i, j = ni, nj
    elif n % 4 == 0:
        k = 1
        for i in range(n):
            for j in range(n):
                M[i][j] = k
                k += 1
        for i in range(n):
            for j in range(n):
                if (i % 4 in (0, 3)) == (j % 4 in (0, 3)):
                    M[i][j] = n * n + 1 - M[i][j]
    else:  # singly even: LUX method
        h = n // 2
        sub = to_mat(magic(h)).d
        add = [0, 2, 3, 1]
        for i in range(h):
            for j in range(h):
                v = sub[i][j]
                M[i][j] = v
                M[i + h][j + h] = v + h * h
                M[i][j + h] = v + 2 * h * h
                M[i + h][j] = v + 3 * h * h
        k = (n - 2) // 4
        for i in range(h):
            cols = list(range(k)) + list(range(n - k + 1, n))
            if i == h // 2:
                cols = list(range(1, k + 1)) + list(range(n - k + 1, n))
            for j in cols:
                M[i][j], M[i + h][j] = M[i + h][j], M[i][j]
        _ = add
    return norm_val(Mat([[float(x) for x in rw] for rw in M]))


def meshgrid(x, y=None):
    xs = flat_cm(x)
    ys = flat_cm(y) if y is not None else xs
    X = Mat([list(xs) for _ in ys])
    Y = Mat([[yv] * len(xs) for yv in ys])
    return norm_val(X), norm_val(Y)


def peaks(n=49):
    """MATLAB's peaks: three Gaussians on [-3, 3]^2. Returns (X, Y, Z)."""
    if isinstance(n, (list, Mat)):
        xs = flat_cm(n)
    else:
        n = int(n)
        xs = [-3.0 + 6.0 * k / (n - 1) for k in range(n)] if n > 1 else [0.0]
    X, Y = meshgrid(xs, xs)
    Xm, Ym = to_mat(X), to_mat(Y)
    Z = []
    for i in range(Xm.r):
        row = []
        for j in range(Xm.c):
            x, y = Xm.d[i][j], Ym.d[i][j]
            z = (3 * (1 - x) ** 2 * math.exp(-x * x - (y + 1) ** 2)
                 - 10 * (x / 5 - x ** 3 - y ** 5) * math.exp(-x * x - y * y)
                 - 1.0 / 3 * math.exp(-(x + 1) ** 2 - y * y))
            row.append(z)
        Z.append(row)
    return X, Y, norm_val(Mat(Z))


def sphere(n=20):
    n = int(n)
    th = [-math.pi + 2 * math.pi * k / n for k in range(n + 1)]
    ph = [-math.pi / 2 + math.pi * k / n for k in range(n + 1)]
    X = Mat([[math.cos(p) * math.cos(t) for t in th] for p in ph])
    Y = Mat([[math.cos(p) * math.sin(t) for t in th] for p in ph])
    Z = Mat([[math.sin(p) for _ in th] for p in ph])
    return norm_val(X), norm_val(Y), norm_val(Z)


def cylinder(rv=1.0, n=20):
    n = int(n)
    rs = flat_cm(rv) if isinstance(rv, (list, Mat)) else [float(rv), float(rv)]
    th = [2 * math.pi * k / n for k in range(n + 1)]
    m = len(rs)
    X = Mat([[r * math.cos(t) for t in th] for r in rs])
    Y = Mat([[r * math.sin(t) for t in th] for r in rs])
    Z = Mat([[k / (m - 1) if m > 1 else 0.0] * (n + 1) for k in range(m)])
    return norm_val(X), norm_val(Y), norm_val(Z)


# ===========================================================================
# Column-wise reductions (MATLAB semantics for sum(A), max(A), ...)
# ===========================================================================
def colwise(v, f, dim=None):
    """Apply f(list)->scalar down each column (dim 1) or across rows (dim 2)."""
    A = to_mat(v)
    if dim is None:
        dim = 2 if A.r == 1 else 1
    if dim == 1:
        return norm_val(Mat([[f(A.col(j)) for j in range(A.c)]]))
    return norm_val(Mat([[f(list(rw))] for rw in A.d]))


def colwise_vec(v, f, dim=None):
    """f(list)->list applied down each column (cumsum, sort, ...)."""
    A = to_mat(v)
    if dim is None:
        dim = 2 if A.r == 1 else 1
    if dim == 1:
        cols = [f(A.col(j)) for j in range(A.c)]
        return norm_val(Mat([[cols[j][i] for j in range(A.c)] for i in range(A.r)]))
    return norm_val(Mat([f(list(rw)) for rw in A.d]))


def find_rc(v):
    """[row, col] = find(A) — 1-based, column-major order."""
    A = to_mat(v)
    rows, cols = [], []
    for j in range(A.c):
        for i in range(A.r):
            if A.d[i][j] != 0:
                rows.append(float(i + 1))
                cols.append(float(j + 1))
    return colvec(rows), colvec(cols)


# ===========================================================================
# 2-D indexing
# ===========================================================================
def index2(v, ri, ci):
    """A(ri, ci) where ri/ci are lists of 1-based ints or None for ':'."""
    A = to_mat(v)
    rows = list(range(A.r)) if ri is None else [_chk(k, A.r, "row") for k in ri]
    cols = list(range(A.c)) if ci is None else [_chk(k, A.c, "column") for k in ci]
    return norm_val(Mat([[A.d[i][j] for j in cols] for i in rows]))


def _chk(k, n, what):
    k = int(round(k))
    if k < 1 or k > n:
        _err(f"{what} index {k} out of bounds (1..{n})")
    return k - 1


def assign2(v, ri, ci, val):
    """A(ri, ci) = val with MATLAB-style growth. Returns the new value."""
    A = to_mat(v) if v is not None else Mat([])
    rows = list(range(A.r)) if ri is None else [int(round(k)) - 1 for k in ri]
    cols = list(range(A.c)) if ci is None else [int(round(k)) - 1 for k in ci]
    if ri is None and A.r == 0:
        rows = list(range(shape(val)[0]))
    if ci is None and A.c == 0:
        cols = list(range(shape(val)[1]))
    for k in rows + cols:
        if k < 0:
            _err("index must be a positive integer")
    need_r = max([A.r] + [k + 1 for k in rows])
    need_c = max([A.c] + [k + 1 for k in cols])
    d = [list(rw) + [0.0] * (need_c - A.c) for rw in A.d]
    for _ in range(need_r - A.r):
        d.append([0.0] * need_c)
    V = to_mat(val)
    if V.r == 1 and V.c == 1:
        for i in rows:
            for j in cols:
                d[i][j] = V.d[0][0]
        return norm_val(Mat(d))
    if V.r * V.c != len(rows) * len(cols):
        _err(f"indexed assignment: size mismatch ({len(rows)}x{len(cols)} vs {V.r}x{V.c})")
    # accept a vector for a single row / column slice, else require same shape
    if V.r == len(rows) and V.c == len(cols):
        src = V.d
    else:
        flat = flat_cm(V)
        src = [[flat[jj * len(rows) + ii] for jj in range(len(cols))] for ii in range(len(rows))]
    for ii, i in enumerate(rows):
        for jj, j in enumerate(cols):
            d[i][j] = src[ii][jj]
    return norm_val(Mat(d))


# ===========================================================================
# Display
# ===========================================================================
def fmt_mat(M, fmt_num):
    """MATLAB-style aligned matrix block (rows of right-aligned numbers)."""
    cells = [[fmt_num(x) for x in rw] for rw in M.d]
    if M.c > 12 or M.r > 40:
        shown = [[fmt_num(x) for x in rw[:8]] for rw in M.d[:12]]
        w = max((len(s) for rw in shown for s in rw), default=1)
        lines = ["  " + "  ".join(s.rjust(w) for s in rw)
                 + ("  ..." if M.c > 8 else "") for rw in shown]
        if M.r > 12:
            lines.append("  ...")
        lines.append(f"  ({M.r}x{M.c} matrix)")
        return "\n".join(lines)
    w = max((len(s) for rw in cells for s in rw), default=1)
    return "\n".join("  " + "  ".join(s.rjust(w) for s in rw) for rw in cells)


# ===========================================================================
# Contours (marching squares) — used by contour / contourf / surfc / meshc
# ===========================================================================
def as_grid(v):
    """A matrix from a Mat OR a list-of-rows (to_mat would read a list as one row)."""
    if is_mat(v):
        return v
    if isinstance(v, list) and v and isinstance(v[0], list):
        return Mat(v)
    return to_mat(v)


def contour_lines(X, Y, Z, levels):
    """Return [(level, [[(x, y), ...], ...]), ...] as open/closed segments
    joined into polylines."""
    Xm, Ym, Zm = as_grid(X), as_grid(Y), as_grid(Z)
    r, c = Zm.r, Zm.c
    out = []
    for lev in levels:
        segs = []
        for i in range(r - 1):
            for j in range(c - 1):
                # corners: 0=(i,j) 1=(i,j+1) 2=(i+1,j+1) 3=(i+1,j)
                pts = [(Xm.d[i][j], Ym.d[i][j], Zm.d[i][j]),
                       (Xm.d[i][j + 1], Ym.d[i][j + 1], Zm.d[i][j + 1]),
                       (Xm.d[i + 1][j + 1], Ym.d[i + 1][j + 1], Zm.d[i + 1][j + 1]),
                       (Xm.d[i + 1][j], Ym.d[i + 1][j], Zm.d[i + 1][j])]
                idx = 0
                for k, p in enumerate(pts):
                    if p[2] >= lev:
                        idx |= 1 << k
                if idx in (0, 15):
                    continue
                edges = [(0, 1), (1, 2), (2, 3), (3, 0)]

                def interp(e):
                    a, b = pts[e[0]], pts[e[1]]
                    dz = b[2] - a[2]
                    t = 0.5 if dz == 0 else (lev - a[2]) / dz
                    return (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
                crossing = [interp(e) for e in edges
                            if (pts[e[0]][2] >= lev) != (pts[e[1]][2] >= lev)]
                if len(crossing) == 2:
                    segs.append((crossing[0], crossing[1]))
                elif len(crossing) == 4:
                    # saddle: pair by the centre value
                    zc = sum(p[2] for p in pts) / 4
                    if (zc >= lev) == (pts[0][2] >= lev):
                        segs.append((crossing[0], crossing[3]))
                        segs.append((crossing[1], crossing[2]))
                    else:
                        segs.append((crossing[0], crossing[1]))
                        segs.append((crossing[2], crossing[3]))
        out.append((lev, _join_segments(segs)))
    return out


def _join_segments(segs, tol=1e-9):
    """Chain 2-point segments into polylines."""
    def key(p):
        return (round(p[0] / tol), round(p[1] / tol))
    remaining = list(segs)
    lines = []
    while remaining:
        a, b = remaining.pop()
        line = [a, b]
        changed = True
        while changed and remaining:
            changed = False
            head, tail = key(line[0]), key(line[-1])
            for k, (p, q) in enumerate(remaining):
                kp, kq = key(p), key(q)
                if kp == tail:
                    line.append(q); remaining.pop(k); changed = True; break
                if kq == tail:
                    line.append(p); remaining.pop(k); changed = True; break
                if kq == head:
                    line.insert(0, p); remaining.pop(k); changed = True; break
                if kp == head:
                    line.insert(0, q); remaining.pop(k); changed = True; break
        lines.append(line)
    return lines


def nice_levels(zmin, zmax, n=8):
    if zmax <= zmin:
        return [zmin]
    raw = (zmax - zmin) / n
    mag = 10 ** math.floor(math.log10(raw))
    step = mag * 10
    for m in (1, 2, 2.5, 5, 10):
        if raw <= m * mag:
            step = m * mag
            break
    lo = math.ceil(zmin / step) * step
    levels = []
    v = lo
    while v < zmax - step * 1e-6:
        levels.append(round(v, 12))
        v += step
    return levels or [(zmin + zmax) / 2]


def delete2(v, ri, ci):
    """A(ri, :) = []  or  A(:, ci) = []  — remove whole rows or whole columns."""
    A = to_mat(v) if v is not None else Mat([])
    if ri is not None and ci is not None:
        _err("a null assignment can have only one non-colon index")
    if ri is None and ci is None:
        return []
    if ci is None:
        drop = {int(round(k)) - 1 for k in ri}
        rows = [rw for i, rw in enumerate(A.d) if i not in drop]
        return norm_val(Mat(rows)) if rows else []
    drop = {int(round(k)) - 1 for k in ci}
    rows = [[x for j, x in enumerate(rw) if j not in drop] for rw in A.d]
    if not rows or not rows[0]:
        return []
    return norm_val(Mat(rows))

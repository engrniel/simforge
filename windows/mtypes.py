"""
mtypes — cell arrays, structs and the string/predicate library for the DEMO
interpreter.

Kept separate from mdemo.py the same way mmat/mdsp/mplot3 are: pure stdlib, no
imports from mdemo, independently testable. mdemo imports it and wires the names
into its function tables.

Value model (extends mdemo's scalar | list = row vector | Mat = r x c):
  * ``Cell``   — a 2-D cell array. ``d`` is rows-of-lists of arbitrary values.
                 ``{1, 'a', [1 2 3]}`` is 1x3. Content indexing ``c{i}`` returns
                 the element; paren indexing ``c(i)`` returns a 1x1 Cell.
  * ``Struct`` — a scalar struct. ``f`` is an insertion-ordered field dict.
                 Struct ARRAYS are deliberately not supported; see the gap list
                 in mdemo's module docstring.
"""

import math
import re


class MTypeError(Exception):
    pass


# ---------------------------------------------------------------------------
# Containers
# ---------------------------------------------------------------------------

class Cell:
    __slots__ = ("d",)

    def __init__(self, rows=None):
        self.d = [list(r) for r in (rows or [])]

    @property
    def r(self):
        return len(self.d)

    @property
    def c(self):
        return len(self.d[0]) if self.d else 0

    def size(self):
        return (self.r, self.c)

    def flat(self):
        """Column-major flatten, like c(:)."""
        return [self.d[i][j] for j in range(self.c) for i in range(self.r)]

    def copy(self):
        return Cell([list(r) for r in self.d])

    def __repr__(self):
        return f"Cell({self.r}x{self.c})"


class Struct:
    __slots__ = ("f",)

    def __init__(self, fields=None):
        self.f = dict(fields or {})

    def copy(self):
        return Struct(dict(self.f))

    def __repr__(self):
        return f"Struct({','.join(self.f)})"


def is_cell(v):
    return isinstance(v, Cell)


def is_struct(v):
    return isinstance(v, Struct)


def cell_of(n, m=1):
    """cell(n) / cell(n, m) — every element starts as an empty [] like MATLAB."""
    n, m = int(n), int(m)
    if n < 0 or m < 0:
        raise MTypeError("cell dimensions must be non-negative")
    return Cell([[[] for _ in range(m)] for _ in range(n)])


def cell_get(c, idx):
    """c{k} with a 1-based column-major index."""
    flat = c.flat()
    if idx < 1 or idx > len(flat):
        raise MTypeError(f"index {idx} out of bounds for a {c.r}x{c.c} cell")
    return flat[idx - 1]


def cell_get2(c, i, j):
    if not (1 <= i <= c.r and 1 <= j <= c.c):
        raise MTypeError(f"index ({i},{j}) out of bounds for a {c.r}x{c.c} cell")
    return c.d[i - 1][j - 1]


def cell_grow(c, r, cc):
    """Grow in place to at least r x cc, padding with [] like MATLAB."""
    while len(c.d) < r:
        c.d.append([[] for _ in range(max(c.c, cc))])
    width = max(c.c, cc)
    for row in c.d:
        while len(row) < width:
            row.append([])
    return c


def cell_set(c, idx, val):
    """c{k} = v with a linear column-major index; grows a vector cell."""
    r, cc = c.r, c.c
    n = r * cc
    if idx < 1:
        raise MTypeError("index must be a positive integer")
    if idx > n:
        if r <= 1:                      # a row cell grows along the row
            cell_grow(c, 1, idx)
        elif cc == 1:                   # a column cell grows down
            cell_grow(c, idx, 1)
        else:
            raise MTypeError("cannot grow a 2-D cell by a linear index")
    i = (idx - 1) % max(c.r, 1)
    j = (idx - 1) // max(c.r, 1)
    c.d[i][j] = val
    return c


def cell_set2(c, i, j, val):
    if i < 1 or j < 1:
        raise MTypeError("index must be a positive integer")
    cell_grow(c, i, j)
    c.d[i - 1][j - 1] = val
    return c


def cell_delete(c, idx):
    """c(k) = [] — drop elements from a vector cell."""
    drop = set(idx)
    flat = [v for k, v in enumerate(c.flat(), 1) if k not in drop]
    if c.c == 1 and c.r > 1:
        return Cell([[v] for v in flat])
    return Cell([flat]) if flat else Cell([])


def cell_index(c, idx):
    """c(k) — a Cell of the selected elements (paren indexing keeps the cell)."""
    flat = c.flat()
    out = []
    for k in idx:
        if k < 1 or k > len(flat):
            raise MTypeError(f"index {k} out of bounds for a {c.r}x{c.c} cell")
        out.append(flat[k - 1])
    if c.c == 1 and c.r > 1:
        return Cell([[v] for v in out])
    return Cell([out])


def cellhcat(parts):
    """Build one row of a { } literal.

    Braces do NOT splice: {{1,2},3} is a 1x2 cell whose first element is itself
    a cell. Splicing is what SQUARE brackets do — [c1 c2] — and that lives in
    mdemo's _hcat.
    """
    return list(parts)


def cellcat(pieces):
    """[c1 c2] — horizontal concatenation of cells, which DOES splice."""
    row = []
    for p in pieces:
        if isinstance(p, Cell):
            if p.r > 1:
                raise MTypeError("only row cells can be concatenated horizontally")
            row.extend(p.d[0] if p.d else [])
        else:
            row.append(p)
    return Cell([row]) if row else Cell([])


# ---------------------------------------------------------------------------
# Struct helpers
# ---------------------------------------------------------------------------

def struct_new(args):
    """struct() / struct('a', 1, 'b', 2)."""
    if len(args) % 2:
        raise MTypeError("struct needs name/value pairs")
    s = Struct()
    for k in range(0, len(args), 2):
        name = args[k]
        if not isinstance(name, str):
            raise MTypeError("struct field names must be strings")
        s.f[name] = args[k + 1]
    return s


def fieldnames(s):
    if not isinstance(s, Struct):
        raise MTypeError("fieldnames needs a struct")
    return Cell([[k] for k in s.f]) if s.f else Cell([])


def isfield(s, name):
    if not isinstance(s, Struct):
        return 0.0
    if isinstance(name, Cell):
        return [1.0 if (isinstance(n, str) and n in s.f) else 0.0 for n in name.flat()]
    return 1.0 if (isinstance(name, str) and name in s.f) else 0.0


def rmfield(s, name):
    if not isinstance(s, Struct):
        raise MTypeError("rmfield needs a struct")
    if name not in s.f:
        raise MTypeError(f"a struct has no field '{name}'")
    out = s.copy()
    del out.f[name]
    return out


def setfield(s, name, val):
    out = s.copy() if isinstance(s, Struct) else Struct()
    out.f[name] = val
    return out


def getfield(s, name):
    if not isinstance(s, Struct) or name not in s.f:
        raise MTypeError(f"a struct has no field '{name}'")
    return s.f[name]


def orderfields(s):
    return Struct({k: s.f[k] for k in sorted(s.f)})


# ---------------------------------------------------------------------------
# Strings
# ---------------------------------------------------------------------------

def _s(v):
    if isinstance(v, str):
        return v
    raise MTypeError("expected a character string")


def strsplit(s, sep=None):
    s = _s(s)
    parts = s.split(_s(sep)) if sep is not None else s.split()
    return Cell([parts]) if parts else Cell([[""]])


def strjoin(c, sep=" "):
    if not isinstance(c, Cell):
        raise MTypeError("strjoin needs a cell array of strings")
    return _s(sep).join(str(v) for v in c.flat())


def strtrim(v):
    if isinstance(v, Cell):
        return Cell([[strtrim(x) for x in row] for row in v.d])
    return _s(v).strip()


def strrep(s, a, b):
    if isinstance(s, Cell):
        return Cell([[strrep(x, a, b) for x in row] for row in s.d])
    return _s(s).replace(_s(a), _s(b))


def strfind(s, pat):
    """1-based positions of every (overlapping) occurrence."""
    s, pat = _s(s), _s(pat)
    if not pat:
        return []
    out, start = [], 0
    while True:
        k = s.find(pat, start)
        if k < 0:
            break
        out.append(float(k + 1))
        start = k + 1
    return out


def strcat(args):
    """Horizontal string concatenation; trailing blanks of char args are dropped."""
    if any(isinstance(a, Cell) for a in args):
        width = max(len(a.flat()) for a in args if isinstance(a, Cell))
        rows = []
        for k in range(width):
            piece = []
            for a in args:
                piece.append(str(a.flat()[k]) if isinstance(a, Cell) else str(a))
            rows.append("".join(piece))
        return Cell([rows])
    return "".join(_s(a).rstrip() if isinstance(a, str) else str(a) for a in args)


def _cmp_str(a, b, n=None, fold=False):
    if isinstance(a, Cell) or isinstance(b, Cell):
        cell, other = (a, b) if isinstance(a, Cell) else (b, a)
        if isinstance(other, Cell):
            pairs = list(zip(cell.flat(), other.flat()))
        else:
            pairs = [(x, other) for x in cell.flat()]
        return [_cmp_str(x, y, n, fold) for x, y in pairs]
    if not (isinstance(a, str) and isinstance(b, str)):
        return 0.0
    x, y = (a.lower(), b.lower()) if fold else (a, b)
    if n is not None:
        n = int(n)
        return 1.0 if x[:n] == y[:n] and len(x) >= n and len(y) >= n else 0.0
    return 1.0 if x == y else 0.0


def contains(s, pat):
    if isinstance(s, Cell):
        return [contains(x, pat) for x in s.flat()]
    return 1.0 if _s(pat) in _s(s) else 0.0


def startswith(s, pat):
    if isinstance(s, Cell):
        return [startswith(x, pat) for x in s.flat()]
    return 1.0 if _s(s).startswith(_s(pat)) else 0.0


def endswith(s, pat):
    if isinstance(s, Cell):
        return [endswith(x, pat) for x in s.flat()]
    return 1.0 if _s(s).endswith(_s(pat)) else 0.0


def _re(pat):
    """MATLAB regexp syntax is close enough to Python's for course work."""
    return re.compile(_s(pat))


def regexprep(s, pat, rep):
    rep = re.sub(r"\$(\d)", r"\\\1", _s(rep))
    if isinstance(s, Cell):
        return Cell([[_re(pat).sub(rep, _s(x)) for x in row] for row in s.d])
    return _re(pat).sub(rep, _s(s))


def regexp_match(s, pat, mode="match"):
    rx = _re(pat)
    s = _s(s)
    if mode == "match":
        found = rx.findall(s)
        return Cell([found]) if found else Cell([])
    if mode == "start":
        return [float(m.start() + 1) for m in rx.finditer(s)]
    if mode == "tokens":
        return Cell([[Cell([list(m.groups())]) for m in rx.finditer(s)]])
    if mode == "once":
        m = rx.search(s)
        return m.group(0) if m else ""
    raise MTypeError(f"regexp option '{mode}' is not supported")


def upper(v):
    if isinstance(v, Cell):
        return Cell([[upper(x) for x in row] for row in v.d])
    return _s(v).upper()


def lower(v):
    if isinstance(v, Cell):
        return Cell([[lower(x) for x in row] for row in v.d])
    return _s(v).lower()


def blanks(n):
    return " " * int(n)


def deblank(v):
    if isinstance(v, Cell):
        return Cell([[deblank(x) for x in row] for row in v.d])
    return _s(v).rstrip()


def str2double(v):
    if isinstance(v, Cell):
        return [str2double(x) for x in v.flat()]
    try:
        return float(str(v).strip())
    except (TypeError, ValueError):
        return float("nan")


def ismember_str(a, c):
    """ismember('x', {'x','y'}) — the string form students actually use."""
    if isinstance(c, Cell):
        return 1.0 if any(isinstance(x, str) and x == a for x in c.flat()) else 0.0
    return 1.0 if a == c else 0.0


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def summarize(v, fmt_num):
    """One-line summary of any value, used by cell and struct display."""
    if isinstance(v, str):
        return f"'{v}'"
    if isinstance(v, Cell):
        return f"{{{v.r}x{v.c} cell}}"
    if isinstance(v, Struct):
        return "[1x1 struct]"
    if isinstance(v, list):
        if not v:
            return "[]"
        if len(v) <= 8:
            return "[" + " ".join(fmt_num(x).strip() for x in v) + "]"
        return f"[1x{len(v)} double]"
    if isinstance(v, tuple) and v and v[0] in ("mfunc", "fhandle"):
        return "@function"
    if hasattr(v, "r") and hasattr(v, "c"):
        return f"[{v.r}x{v.c} double]"
    if isinstance(v, (int, float, complex)):
        return fmt_num(v).strip()
    return "[value]"


def format_cell(c, fmt_num, indent=""):
    """MATLAB-ish cell display:  {[1]  'two'  [1x3 double]}."""
    if c.r == 0 or c.c == 0:
        return indent + "  {}(0x0)"
    lines = []
    for i in range(c.r):
        cells = [summarize(v, fmt_num) for v in c.d[i]]
        width = max((len(x) for x in cells), default=1)
        lines.append(indent + "    " + "    ".join(x.ljust(width) for x in cells))
    return "\n".join(lines)


def format_struct(s, fmt_num, indent=""):
    if not s.f:
        return indent + "  struct with no fields"
    width = max(len(k) for k in s.f)
    lines = []
    for k, v in s.f.items():
        lines.append(f"{indent}    {k.rjust(width)}: {summarize(v, fmt_num)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Type predicates
# ---------------------------------------------------------------------------

def class_of(v, is_mat, is_tf, is_func):
    if isinstance(v, str):
        return "char"
    if isinstance(v, Cell):
        return "cell"
    if isinstance(v, Struct):
        return "struct"
    if is_func(v):
        return "function_handle"
    if is_tf(v):
        return "tf"
    if isinstance(v, complex):
        return "complex double"
    return "double"


def numeric_like(v):
    if isinstance(v, (Cell, Struct, str)):
        return False
    return isinstance(v, (int, float, complex, list)) or (
        hasattr(v, "r") and hasattr(v, "d"))


def is_finite_num(x):
    return isinstance(x, (int, float)) and math.isfinite(x)

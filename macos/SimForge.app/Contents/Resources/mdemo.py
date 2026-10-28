"""
mdemo — a small MATLAB-subset interpreter + SVG plotter for DEMO mode.

When no real MATLAB engine is present, this actually *executes* a useful subset
of MATLAB so the IDE behaves like the real thing: variables, vectors, element-
wise math, many built-in functions, ranges/indexing, and plotting that renders
crisp SVG figures.

Supported (subset):
  assignment            x = expr;     (';' suppresses echo)
  scalars / row vectors [1 2 3]  [1,2,3]
  ranges                a:b   a:step:b
  operators             + - * / ^  and element-wise  .* ./ .^   unary -
  indexing              x(3)  x(2:4)  x(end)
  functions             sin cos tan asin acos atan sinh cosh tanh
                        exp log log2 log10 sqrt abs sign floor ceil round fix
                        mod rem atan2 hypot power max min
                        sum mean prod length numel std var rms median sort
                        cumsum cumprod diff dot trapz
                        linspace logspace zeros ones rand randn
  signals & spectra     fft ifft fftshift ifftshift conv deconv filter xcorr
                        unwrap sinc square sawtooth chirp nextpow2 hilbert
                        freqz periodogram pwelch (spectra + digital-filter resp.)
                        interp1 polyfit residue (interp / fitting / partial frac)
                        hamming hann hanning blackman bartlett rectwin
                        real imag conj angle mag2db pow2db db2mag db2pow
                        deg2rad rad2deg flip fliplr downsample upsample
  complex numbers       1i 2.5j  exp(1i*w*t)  abs/angle/real/imag of complex
  constants             pi e i j
  control flow          if/elseif/else, for, while, break, continue, return
  comparisons/logic     == ~= < <= > >=   & | && ||   ~x
  indexed assignment    x(i) = v   x(end+1) = v  (vector grows like MATLAB)
  functions (user)      function y = f(x) ... end   [a,b] = f(x)   recursion
  anonymous functions   f = @(x) x.^2 - 2;   g = @sin;   f(3)
  algorithms helpers    any all find isempty factorial strcmp num2str error
  control systems       tf('s') / tf(num,den) / zpk(z,p,k), tf arithmetic,
                        step impulse lsim bode margin nyquist rlocus pzmap
                        feedback series parallel stepinfo damp bandwidth
                        dcgain pole zero roots poly polyval semilogx
  printing              disp(x)   fprintf('...%g...\\n', a, b)   bare expr -> ans
  plotting              plot(y) plot(x,y) plot(x,y,'r--o') plot(x,y1,x,y2)
                        bar scatter stem stairs area
                        semilogx semilogy loglog | hold on/off | figure
                        subplot(r,c,k) sgtitle xlim ylim axis xline yline text
                        title xlabel ylabel legend grid on/off

ECE08 (Signals, Spectra and Signal Processing) laboratory set — see mdsp.py:
  Ch1 signals           heaviside stepfun rectpuls tripuls gauspuls pulstran dirac
  Ch3 sampling          decimate interp resample uquantize sqnr quantiz
                        compand lin2mu mu2lin
  Ch4 z transform       zplane impz stepz tf2zp zp2tf residuez
                        bilinear impinvar tf2latc latc2tf
  Ch5 DFT / FFT         cconv fftfilt bitrevorder kaiser flattopwin
                        spectrogram dftmtx_row
  Ch6 filter design     butter cheby1 cheby2 ellip besself
                        buttord cheb1ord cheb2ord ellipord
                        fir1 fir2 firls firpm kaiserord firpmord
                        freqs grpdelay filtfilt tf2sos zp2sos sos2tf sosfilt
  Ch7 random signals    awgn snr xcov autocorr levinson lpc

Limitations (documented): 2-D matrices are flattened to row vectors, so
second-order sections come back as a flat 6-per-section row and the DFT matrix
is exposed one row at a time; switch/case and cell arrays are not supported.
"""

import cmath
import json
import math
import os
import re
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mdsp                                     # noqa: E402  (DSP + filter design)
import mmat                                     # noqa: E402  (2-D matrices, linear algebra)
import mplot3                                   # noqa: E402  (colormaps, 3-D axes)
import msim                                     # noqa: E402  (ODE / roots / optimisation)
import mcircuit                                 # noqa: E402  (SPICE-style circuit engine)
import mtypes                                   # noqa: E402  (cells, structs, strings)
from mmat import Mat, is_mat                    # noqa: E402
from mtypes import Cell, Struct, is_cell, is_struct   # noqa: E402


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------
_TOKEN_RE = re.compile(
    r"""
      (?P<NL>[ \t]*\n[ \t]*)
    | (?P<SP>\s+)
    | (?P<DOTOP>\.\*|\./|\.\^|\.')
    | (?P<NUM>(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?[ij]?)
    | (?P<STR>'(?:[^']|'')*'|"(?:[^"]|"")*")
    | (?P<ID>[A-Za-z_]\w*)
    | (?P<OP>==|~=|<=|>=|&&|\|\||[-+*/^\\():,\[\]{};<>&|~@.])
    """,
    re.VERBOSE,
)


def tokenize(s):
    toks = []
    pos = 0
    depth = 0
    while pos < len(s):
        if s[pos] == "'":
            # MATLAB's rule: a quote right after an operand is the transpose
            # operator, otherwise it opens a string.  Inside [ ] a quote that
            # follows whitespace starts a new element, so it is a string:
            # [98 'c'] is two elements, x' is a transpose.
            prev_raw = toks[-1] if toks else (None, None)
            prev = next((t for t in reversed(toks) if t[0] not in ("SP", "NL")), (None, None))
            spaced = prev_raw[0] in ("SP", "NL")
            if (prev[0] in ("ID", "NUM") or prev[1] in (")", "]", "'", ".'")) and not (depth > 0 and spaced):
                toks.append(("OP", "'"))
                pos += 1
                continue
        m = _TOKEN_RE.match(s, pos)
        if not m:
            raise MError(f"unexpected character '{s[pos]}'")
        pos = m.end()
        kind = m.lastgroup
        val = m.group()
        if kind == "STR":
            q = val[0]
            val = val[1:-1].replace(q + q, q)
        elif kind == "OP" and val in ("[", "{"):
            depth += 1
        elif kind == "OP" and val in ("]", "}"):
            depth = max(0, depth - 1)
        toks.append((kind, val))
    return toks


class MError(Exception):
    pass


# ---------------------------------------------------------------------------
# Parser  ->  AST tuples
# ---------------------------------------------------------------------------
_BRACKET_BIN = {"*", "/", "^", ".*", "./", ".^", "\\", "==", "~=", "<", "<=", ">", ">=",
                "&", "|", "&&", "||"}
_BRACKET_SIGN = {"+", "-"}


class Parser:
    def __init__(self, toks):
        self.toks = toks
        self.i = 0

    def _skip_sp(self):
        while self.i < len(self.toks) and self.toks[self.i][0] in ("SP", "NL"):
            self.i += 1

    def _peek_after(self):
        """The significant token after the current one."""
        j = self.i
        while j < len(self.toks) and self.toks[j][0] in ("SP", "NL"):
            j += 1
        j += 1
        while j < len(self.toks) and self.toks[j][0] in ("SP", "NL"):
            j += 1
        return self.toks[j] if j < len(self.toks) else (None, None)

    def _peek_after_dot(self):
        """The token after a '.', when it can start a field access."""
        j = self.i
        while j < len(self.toks) and self.toks[j][0] in ("SP", "NL"):
            j += 1
        j += 1                                    # the '.' itself
        if j < len(self.toks) and self.toks[j][0] == "ID":
            return self.toks[j]
        if j < len(self.toks) and self.toks[j][0] == "OP" and self.toks[j][1] == "(":
            return self.toks[j]
        return None

    def peek(self):
        self._skip_sp()
        return self.toks[self.i] if self.i < len(self.toks) else (None, None)

    def op(self):
        """The current token's text ONLY when it is an operator or punctuation.

        Comparing peek()[1] directly is a trap: the string literal '-' has the
        value "-" too, so `strjoin(p, '-')` parsed the string as a unary minus.
        Every operator test goes through here.
        """
        k, v = self.peek()
        return v if k in ("OP", "DOTOP") else None

    def next(self):
        self._skip_sp()
        t = self.toks[self.i] if self.i < len(self.toks) else (None, None)
        self.i += 1
        return t

    def expect(self, val):
        k, v = self.next()
        if v != val:
            raise MError(f"expected '{val}' but got '{v}'")

    def parse(self):
        node = self.parse_or()
        if self.peek()[0] is not None:
            raise MError(f"unexpected '{self.peek()[1]}'")
        return node

    def _eof(self):
        raise MError("the expression ends too early — check for a missing ')' or operand")

    def parse_or(self):
        node = self.parse_and()
        while self.op() in ("||", "|"):
            op = self.next()[1]
            node = ("bin", op, node, self.parse_and())
        return node

    def parse_and(self):
        node = self.parse_cmp()
        while self.op() in ("&&", "&"):
            op = self.next()[1]
            node = ("bin", op, node, self.parse_cmp())
        return node

    def parse_cmp(self):
        node = self.parse_range()
        while self.op() in ("==", "~=", "<", "<=", ">", ">="):
            op = self.next()[1]
            node = ("bin", op, node, self.parse_range())
        return node

    def parse_range(self):
        first = self.parse_add()
        if self.op() == ":":
            parts = [first]
            while self.op() == ":":
                self.next()
                parts.append(self.parse_add())
            return ("range", parts)
        return first

    def parse_add(self):
        node = self.parse_mul()
        while self.op() in ("+", "-"):
            op = self.next()[1]
            node = ("bin", op, node, self.parse_mul())
        return node

    def parse_mul(self):
        node = self.parse_unary()
        while self.op() in ("*", "/", ".*", "./", "\\"):
            op = self.next()[1]
            node = ("bin", op, node, self.parse_unary())
        return node

    def parse_unary(self):
        if self.op() in ("+", "-", "~"):
            op = self.next()[1]
            node = self.parse_unary()
            if op == "-":
                return ("neg", node)
            if op == "~":
                return ("not", node)
            return node
        return self.parse_pow()

    def parse_pow(self):
        base = self.parse_postfix()
        if self.op() in ("^", ".^"):
            op = self.next()[1]
            return ("bin", op, base, self.parse_unary())  # right-assoc
        return base

    def parse_postfix(self):
        node = self.parse_primary()
        while True:
            k, v = self.peek()
            if k not in ("OP", "DOTOP"):
                break
            if v == "(" and node[0] == "id":
                self.next()  # (
                args = self.parse_args()
                self.expect(")")
                node = ("call", node[1], args)
            elif v == "(" and node[0] in ("field", "dynfield", "cellidx", "index", "call"):
                self.next()
                args = self.parse_args()
                self.expect(")")
                node = ("index", node, args)
            elif v == "{":
                self.next()
                args = self.parse_args()
                self.expect("}")
                node = ("cellidx", node, args)
            elif v == "." and self._peek_after_dot() is not None:
                self.next()          # .
                k2, v2 = self.next()
                if k2 == "ID":
                    node = ("field", node, v2)
                elif v2 == "(":      # dynamic field name:  s.(name)
                    e = self.parse_or()
                    self.expect(")")
                    node = ("dynfield", node, e)
                else:
                    raise MError("expected a field name after '.'")
            elif v == "'":
                self.next()
                node = ("ctranspose", node)
            elif v == ".'":
                self.next()
                node = ("transpose", node)
            else:
                break
        return node

    def _parse_arg(self):
        # a bare ':' means "every index along this dimension"
        after = self._peek_after()
        if self.op() == ":" and after[0] == "OP" and after[1] in (",", ")"):
            self.next()
            return ("colon",)
        return self.parse_or()

    def parse_args(self):
        args = []
        if self.op() in (")", "}"):
            return args
        args.append(self._parse_arg())
        while self.op() == ",":
            self.next()
            args.append(self._parse_arg())
        return args

    def parse_primary(self):
        k, v = self.peek()
        if k == "NUM":
            self.next()
            if v[-1] in "ij":
                return ("num", complex(0.0, float(v[:-1] or "1")))
            return ("num", float(v))
        if k == "STR":
            self.next()
            return ("str", v)
        if k == "ID":
            self.next()
            return ("id", v)
        if k not in ("OP", "DOTOP"):
            raise MError(f"unexpected '{v}'" if v is not None else
                         "the expression ends too early")
        if v == "(":
            self.next()
            e = self.parse_or()
            self.expect(")")
            return e
        if v == "[":
            return self.parse_matrix()
        if v == "{":
            return self.parse_cell()
        if v == "@":
            return self.parse_function_ref()
        if v is None:
            self._eof()
        raise MError(f"unexpected '{v}'")

    def parse_function_ref(self):
        """@(x, y) expr  (anonymous fn)  or  @name  (function handle)."""
        self.next()  # @
        k, v = self.peek()
        if v == "(":
            self.next()
            params = []
            while True:
                k2, v2 = self.next()
                if v2 == ")":
                    break
                if k2 == "ID":
                    params.append(v2)
                elif v2 != ",":
                    raise MError("bad anonymous-function parameter list")
            return ("lambda", params, self.parse_or())
        if k == "ID":
            self.next()
            return ("fh", v)
        raise MError("expected '(' or a function name after '@'")

    def parse_matrix(self):
        """[a b; c d]  ->  ("mat", [[a, b], [c, d]]) — ';' or a newline ends a row."""
        return ("mat", self._parse_rows("[", "]"))

    def parse_cell(self):
        """{1, 'two', [1 2 3]}  ->  ("cellexp", rows) — same layout rules as [ ]."""
        return ("cellexp", self._parse_rows("{", "}"))

    def _parse_rows(self, ob, cb):
        """Shared row/element splitter for [ ] and { }."""
        self.next()  # the opening bracket
        depth = 1
        inner = []
        while self.i < len(self.toks):
            k, v = self.toks[self.i]
            if k == "OP" and v == ob:
                depth += 1
            elif k == "OP" and v == cb:
                depth -= 1
                if depth == 0:
                    self.i += 1
                    break
            inner.append((k, v))
            self.i += 1
        rows, elems, cur, d = [], [], [], 0
        for (k, v) in inner:
            if k == "OP" and v in ("(", "[", "{"):
                d += 1
            elif k == "OP" and v in (")", "]", "}"):
                d -= 1
            if d == 0 and (k == "NL" or (k == "OP" and v == ";")):
                if cur:
                    elems.append(cur)
                    cur = []
                rows.append(elems)
                elems = []
                continue
            if d == 0 and (k == "SP" or (k == "OP" and v == ",")):
                if cur:
                    elems.append(cur)
                    cur = []
                continue
            cur.append((k, v))
        if cur:
            elems.append(cur)
        rows.append(elems)
        rows = [r for r in rows if r]

        def merge(elems):
            # MATLAB's whitespace rules inside brackets:
            #   [a - b] [a * b] [a == b]  ->  one element (operator with spaces on BOTH sides)
            #   [a- b]  [a *b]            ->  one element (operator glued to one side)
            #   [a -b]                    ->  TWO elements ("-b" is a signed operand)
            #   [-b ...]                  ->  a leading sign belongs to the next element
            def is_op(tok, ops):
                return tok[0] in ("OP", "DOTOP") and tok[1] in ops
            merged, i = [], 0
            while i < len(elems):
                e = elems[i]
                if len(e) == 1 and is_op(e[0], _BRACKET_BIN | _BRACKET_SIGN) and i + 1 < len(elems):
                    if merged:
                        merged[-1] = merged[-1] + e + elems[i + 1]
                    else:
                        merged.append(e + elems[i + 1])
                    i += 2
                    continue
                if merged and is_op(e[0], _BRACKET_BIN):          # "*b" continues the previous
                    merged[-1] = merged[-1] + e
                    i += 1
                    continue
                if merged and is_op(merged[-1][-1], _BRACKET_BIN | _BRACKET_SIGN):   # "a-" needs this
                    merged[-1] = merged[-1] + e
                    i += 1
                    continue
                merged.append(e)
                i += 1
            return [Parser(e).parse() for e in merged]
        return [merge(r) for r in rows]


# ---------------------------------------------------------------------------
# Values:  float (scalar)  |  list[float] (row vector)  |  str
# ---------------------------------------------------------------------------
class _CSList(list):
    """A comma-separated list, what c{1:3} expands to inside [ ] or { }."""


class _Colon:
    """The bare ':' index ("everything along this dimension")."""


COLON = _Colon()


def is_vec(v):
    return isinstance(v, list) and not isinstance(v, _CSList)


def _is_empty(v):
    """[] — the value MATLAB uses for "nothing here yet"."""
    return isinstance(v, list) and len(v) == 0


def as_list(v):
    if isinstance(v, list):
        return v
    if is_mat(v):
        return mmat.flat_cm(v)            # column-major, like MATLAB's A(:)
    return [v]


def scalar(v):
    if isinstance(v, list):
        return v[0]
    if is_mat(v):
        return v.d[0][0] if v.r and v.c else 0.0
    return v


def _is_func(v):
    return isinstance(v, tuple) and len(v) > 0 and v[0] in ("mfunc", "fhandle")


def _is_tf(v):
    return isinstance(v, dict) and v.get("__tf__")


def _truthy(v):
    if isinstance(v, str):
        return len(v) > 0
    lst = as_list(v)
    return len(lst) > 0 and all(x != 0 for x in lst)


class BreakEx(Exception):
    pass


class ContinueEx(Exception):
    pass


class ReturnEx(Exception):
    pass


def elementwise(a, b, f):
    if is_mat(a) or is_mat(b):
        try:
            return mmat.bcast(a, b, f)
        except mmat.MatError as ex:
            raise MError(str(ex))
    if is_vec(a) and is_vec(b):
        if len(a) != len(b):
            raise MError(f"array sizes don't match ({len(a)} vs {len(b)})")
        return [f(x, y) for x, y in zip(a, b)]
    if is_vec(a):
        return [f(x, b) for x in a]
    if is_vec(b):
        return [f(a, y) for y in b]
    return f(a, b)


# ---------------------------------------------------------------------------
# Block structure: if/elseif/else, for, while, function ... end
# ---------------------------------------------------------------------------
_KW_RE = re.compile(r"(elseif|else|end|if|for|while|break|continue|return|function"
                    r"|switch|case|otherwise|try|catch)\b")


def _kw_of(text):
    m = _KW_RE.match(text)
    return m.group(1) if m else None


def _parse_func_sig(text):
    """'function [q, r] = divide(a, b)' -> ('divide', ['q','r'], ['a','b'])."""
    rest = text[len("function"):].strip()
    if "=" in rest:
        left, right = rest.split("=", 1)
        left = left.strip()
        if left.startswith("["):
            outs = [o.strip() for o in left.strip("[]").split(",") if o.strip()]
        else:
            outs = [left]
    else:
        outs, right = [], rest
    m = re.fullmatch(r"([A-Za-z_]\w*)\s*(?:\((.*)\))?", right.strip())
    if not m:
        raise MError(f"bad function definition '{text}'")
    params = [p.strip() for p in (m.group(2) or "").split(",") if p.strip()]
    return m.group(1), outs, params


def build_blocks(stmts):
    """Turn the flat statement list into a tree; hoist function definitions."""
    funcs = {}

    def parse_body(i, terms):
        body = []
        while i < len(stmts):
            text, sup = stmts[i]
            k = _kw_of(text)
            if k in terms:
                return body, i
            if k == "if":
                branches = []
                cond = text[2:].strip()
                sub, i = parse_body(i + 1, ("elseif", "else", "end"))
                branches.append((cond, sub))
                while i < len(stmts) and _kw_of(stmts[i][0]) == "elseif":
                    c = stmts[i][0][6:].strip()
                    sub, i = parse_body(i + 1, ("elseif", "else", "end"))
                    branches.append((c, sub))
                els = []
                if i < len(stmts) and _kw_of(stmts[i][0]) == "else":
                    els, i = parse_body(i + 1, ("end",))
                if i >= len(stmts):
                    raise MError("missing 'end' for 'if'")
                body.append(("if", branches, els))
                i += 1
            elif k == "for":
                m = re.match(r"for\s+([A-Za-z_]\w*)\s*=\s*(.+)$", text)
                if not m:
                    raise MError(f"bad for statement '{text}'")
                sub, i = parse_body(i + 1, ("end",))
                if i >= len(stmts):
                    raise MError("missing 'end' for 'for'")
                body.append(("for", m.group(1), m.group(2), sub))
                i += 1
            elif k == "while":
                sub, i = parse_body(i + 1, ("end",))
                if i >= len(stmts):
                    raise MError("missing 'end' for 'while'")
                body.append(("while", text[5:].strip(), sub))
                i += 1
            elif k == "switch":
                subject = text[6:].strip()
                cases, other = [], []
                # skip anything between `switch x` and the first `case`
                _, i = parse_body(i + 1, ("case", "otherwise", "end"))
                while i < len(stmts) and _kw_of(stmts[i][0]) == "case":
                    label = stmts[i][0][4:].strip()
                    sub, i = parse_body(i + 1, ("case", "otherwise", "end"))
                    cases.append((label, sub))
                if i < len(stmts) and _kw_of(stmts[i][0]) == "otherwise":
                    other, i = parse_body(i + 1, ("end",))
                if i >= len(stmts):
                    raise MError("missing 'end' for 'switch'")
                body.append(("switch", subject, cases, other))
                i += 1
            elif k == "try":
                sub, i = parse_body(i + 1, ("catch", "end"))
                errvar, handler = None, []
                if i < len(stmts) and _kw_of(stmts[i][0]) == "catch":
                    rest = stmts[i][0][5:].strip()
                    errvar = rest if re.fullmatch(r"[A-Za-z_]\w*", rest) else None
                    handler, i = parse_body(i + 1, ("end",))
                if i >= len(stmts):
                    raise MError("missing 'end' for 'try'")
                body.append(("try", sub, errvar, handler))
                i += 1
            elif k == "function":
                name, outs, params = _parse_func_sig(text)
                sub, i = parse_body(i + 1, ("end", "function"))
                if i < len(stmts) and _kw_of(stmts[i][0]) == "end":
                    i += 1
                funcs[name] = (outs, params, sub)
            elif k in ("break", "continue", "return"):
                body.append((k,))
                i += 1
            else:
                body.append(("stmt", text, sup))
                i += 1
        return body, i

    tree, _ = parse_body(0, ())
    return tree, funcs


# ---------------------------------------------------------------------------
# Interpreter
# ---------------------------------------------------------------------------
# legend(..., Name, Value) properties that are not legend entries.
_LEGEND_PROPS = {"location", "orientation", "autoupdate", "box", "fontsize", "fontname",
                 "numcolumns", "textcolor", "color", "edgecolor", "interpreter",
                 "fontweight", "title", "visible", "linewidth"}

# MATLAB's default axes ColorOrder (R2014b+).
_PALETTE = ["#0072BD", "#D95319", "#EDB120", "#7E2F8E", "#77AC30", "#4DBEEE", "#A2142F"]
# MATLAB single-letter color shorthands.
_COLORS = {
    "r": "#FF0000", "g": "#00BF00", "b": "#0000FF", "c": "#00BFBF",
    "m": "#BF00BF", "y": "#BFBF00", "k": "#000000", "w": "#FFFFFF",
}


_OP_LIMIT = 400_000     # executed statements/iterations before we assume a hang
_MAX_DEPTH = 120        # user-function recursion depth


class Interp:
    def __init__(self):
        self.vars = {"pi": math.pi, "e": math.e, "ans": None}
        self.funcs = {}
        self.out = []
        self.figures = []      # finished figures
        self.curfig = None     # current figure (may hold several subplot axes)
        self.cur = None        # current axes dict
        self.hold = False
        self._color_idx = 0
        self.ops = 0
        self.depth = 0
        self.ckt = None        # circuit being built by resistor(...) / voltage(...) / ...
        # --- session state (an Interp is reused across runs, like a MATLAB session)
        self.cancel = None     # threading.Event set by the sidecar on Stop
        self.clc = False       # `clc` asks the UI to clear the Command Window
        self.fig_store = {}    # figure id -> figure dict, for interactive re-render
        self._next_fid = 1
        self.cwd = os.path.expanduser("~")   # kept in step with the sidecar's cwd
        self._bare_args = None  # arg list of a statement-level call (nargout = 0)

    # -- public -----------------------------------------------------------
    def reset(self):
        """Wipe the workspace, like `clear` + `close all` on a fresh session."""
        self.vars = {"pi": math.pi, "e": math.e, "ans": None}
        self.funcs = {}
        self.fig_store = {}
        self.ckt = None

    def _begin(self):
        """Reset the per-run scratch state; keep variables and functions."""
        self.out = []
        self.figures = []
        self.curfig = None
        self.cur = None
        self.hold = False
        self._color_idx = 0
        self.ops = 0
        self.depth = 0
        self.clc = False

    def stopped(self):
        return self.cancel is not None and self.cancel.is_set()

    def run(self, code):
        self._begin()
        error = None
        try:
            tree, funcs = build_blocks(split_statements(code))
            self.funcs.update(funcs)   # functions stay defined for the session
            self.exec_block(tree)
        except (BreakEx, ContinueEx):
            error = "'break'/'continue' used outside a loop"
        except ReturnEx:
            pass
        except MError as ex:
            error = str(ex)
        except RecursionError:
            error = "maximum recursion depth exceeded"
        except Exception as ex:  # pragma: no cover - safety net
            error = f"{ex}"
        self._finish_figure()
        figs = []
        for f in self.figures:
            final = render_svg(f)
            fid = self._next_fid
            self._next_fid += 1
            self.fig_store[fid] = f
            item = {"svg": final, "id": fid}
            if f.get("frames"):
                frames = list(f["frames"])
                delays = list(f["delays"])
                if frames[-1] != final:
                    frames.append(final)
                    delays.append(delays[-1] if delays else 40)
                item["frames"] = frames
                item["delays"] = delays
            figs.append(item)
        # keep the store bounded — only recent figures stay re-renderable
        if len(self.fig_store) > 40:
            for k in sorted(self.fig_store)[:-40]:
                del self.fig_store[k]
        return {
            "stdout": "".join(self.out),
            "error": error,
            "figures": figs,
            "workspace": self.workspace(),
            "clc": self.clc,
        }

    # -- interactive figures ------------------------------------------------
    def fit_series(self, fid, ax_i, s_i, kind, deg):
        """Basic Fitting / Data Statistics on one plotted series.

        The fit is added to the STORED figure and the figure re-rendered, the
        same round-trip zoom uses — so the fit line is drawn by the one
        renderer, with correct clipping and a legend entry.
        """
        f = self.fig_store.get(int(fid))
        if f is None:
            raise MError("figure is no longer available (re-run the script)")
        axes = f.get("axes") or []
        if not (0 <= int(ax_i) < len(axes)):
            raise MError("no such axes")
        ax = axes[int(ax_i)]
        real = [sr for sr in ax["series"] if sr["kind"] != "heat" and not sr.get("fit")]
        if not (0 <= int(s_i) < len(real)):
            raise MError("no such data series")
        src = real[int(s_i)]
        xs = [float(_cr(v)) for v in src["x"]]
        ys = [float(_cr(v)) for v in src["y"]]
        pts = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
        if len(pts) < 2:
            raise MError("this series has too few finite points")
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]

        stats = {
            "n": len(ys), "min": min(ys), "max": max(ys),
            "mean": sum(ys) / len(ys),
            "median": sorted(ys)[len(ys) // 2] if len(ys) % 2 else
                      (sorted(ys)[len(ys) // 2 - 1] + sorted(ys)[len(ys) // 2]) / 2,
            "std": _std(ys), "range": max(ys) - min(ys),
            "xmin": min(xs), "xmax": max(xs),
        }
        if kind == "stats":
            return {"ok": True, "stats": stats}
        if kind == "clear":
            ax["series"] = [sr for sr in ax["series"] if not sr.get("fit")]
            if ax.get("legend"):
                ax["legend"] = [n for n in ax["legend"] if not n.startswith("fit ")]
            return {"ok": True, "svg": render_svg(f), "stats": stats}

        deg = max(1, min(int(deg), 8))
        if deg >= len(xs):
            raise MError(f"a degree-{deg} fit needs more than {deg} points")
        coeffs = _polyfit([xs, ys, float(deg)])
        lo, hi = min(xs), max(xs)
        step = (hi - lo) / 200 if hi > lo else 1.0
        fx = [lo + step * k for k in range(201)]
        fy = [_peval(coeffs, x) for x in fx]
        resid = [y - _peval(coeffs, x) for x, y in zip(xs, ys)]
        sse = sum(r * r for r in resid)
        ybar = stats["mean"]
        sst = sum((y - ybar) ** 2 for y in ys)
        stats["rmse"] = math.sqrt(sse / len(ys))
        stats["r2"] = (1 - sse / sst) if sst > 1e-300 else 1.0
        stats["coeffs"] = coeffs

        ax["series"] = [sr for sr in ax["series"] if not sr.get("fit")]
        label = f"fit deg {deg}"
        ax["series"].append({
            "kind": "line", "x": fx, "y": fy, "color": "#FF3B30",
            "style": "--", "marker": None, "width": 1.8, "fit": True,
        })
        names = list(ax.get("legend") or [])
        if names:
            names = [n for n in names if not n.startswith("fit ")]
            names.append(label)
            ax["legend"] = names
        return {"ok": True, "svg": render_svg(f), "stats": stats}


    def set_limits(self, fid, ax_i, xlim, ylim):
        """Re-render one stored figure with new axis limits (zoom / pan / restore)."""
        f = self.fig_store.get(int(fid))
        if f is None:
            raise MError("figure is no longer available (re-run the script)")
        axes = f.get("axes") or []
        if not (0 <= int(ax_i) < len(axes)):
            raise MError("no such axes")
        ax = axes[int(ax_i)]
        # remember what the script itself asked for, so Restore View can go back
        if "_xlim0" not in ax:
            ax["_xlim0"] = ax.get("xlim")
            ax["_ylim0"] = ax.get("ylim")
        for key, lim in (("xlim", xlim), ("ylim", ylim)):
            if lim == "reset":
                base = ax.get("_" + key + "0")
                if base is None:
                    ax.pop(key, None)
                else:
                    ax[key] = base
            elif lim is not None:
                ax[key] = [float(lim[0]), float(lim[1])]
        return render_svg(f)

    # -- block execution ----------------------------------------------------
    def tick(self):
        self.ops += 1
        if self.cancel is not None and (self.ops & 0x3FF) == 0 and self.cancel.is_set():
            raise MError("execution stopped")
        if self.ops > _OP_LIMIT:
            raise MError("execution limit reached (possible infinite loop)")

    def exec_block(self, body):
        for node in body:
            self.tick()
            kind = node[0]
            if kind == "stmt":
                self.exec_stmt(node[1], node[2])
            elif kind == "if":
                taken = False
                for cond, sub in node[1]:
                    if _truthy(self.eval(cond)):
                        self.exec_block(sub)
                        taken = True
                        break
                if not taken:
                    self.exec_block(node[2])
            elif kind == "for":
                ast = Parser(tokenize(node[2])).parse()
                if ast[0] == "range":
                    seq = iter_range([scalar(self.ev(p)) for p in ast[1]])
                else:
                    rng = self.ev(ast)
                    if is_mat(rng) and rng.r > 1:
                        seq = [mmat.colvec(rng.col(j)) for j in range(rng.c)]  # columns, like MATLAB
                    else:
                        seq = as_list(rng)
                for v in seq:
                    self.tick()
                    self.vars[node[1]] = v
                    try:
                        self.exec_block(node[3])
                    except ContinueEx:
                        continue
                    except BreakEx:
                        break
            elif kind == "while":
                while _truthy(self.eval(node[1])):
                    self.tick()
                    try:
                        self.exec_block(node[2])
                    except ContinueEx:
                        continue
                    except BreakEx:
                        break
            elif kind == "switch":
                subject = self.eval(node[1])
                for label, sub in node[2]:
                    if self._case_matches(subject, label):
                        self.exec_block(sub)
                        break
                else:
                    self.exec_block(node[3])
            elif kind == "try":
                try:
                    self.exec_block(node[1])
                except (BreakEx, ContinueEx, ReturnEx):
                    raise
                except MError as ex:
                    if self.stopped():
                        raise                       # Stop is not a catchable error
                    if node[2]:
                        self.vars[node[2]] = Struct({
                            "message": str(ex),
                            "identifier": getattr(ex, "identifier", ""),
                            "stack": Cell([]),
                        })
                    self.exec_block(node[3])
                except RecursionError:
                    raise
                except Exception as ex:             # a Python-level slip is still an error
                    if node[2]:
                        self.vars[node[2]] = Struct({
                            "message": str(ex), "identifier": "", "stack": Cell([])})
                    self.exec_block(node[3])
            elif kind == "break":
                raise BreakEx()
            elif kind == "continue":
                raise ContinueEx()
            elif kind == "return":
                raise ReturnEx()

    def _map_call(self, name, args, nargout=1):
        """cellfun / arrayfun / structfun — MATLAB's map operations."""
        if len(args) < 2:
            raise MError(f"{name} needs a function and a container")
        f = args[0]
        if isinstance(f, str):                       # cellfun('isempty', c) legacy form
            f = ("fhandle", f)
        if not _is_func(f):
            raise MError(f"{name}: the first argument must be a function handle")
        uniform = True
        rest = list(args[1:])
        for k in range(len(rest) - 1):
            if isinstance(rest[k], str) and rest[k].lower() == "uniformoutput":
                uniform = _truthy(rest[k + 1])
                rest = rest[:k]
                break
        if name == "structfun":
            if not is_struct(rest[0]):
                raise MError("structfun needs a struct")
            items = [[v] for v in rest[0].f.values()]
            shape = (len(items), 1)
        elif name == "cellfun":
            cells = [c for c in rest]
            if not all(is_cell(c) for c in cells):
                raise MError("cellfun needs cell arrays")
            flats = [c.flat() for c in cells]
            n = len(flats[0])
            if any(len(fl) != n for fl in flats):
                raise MError("cellfun: inputs must be the same size")
            items = [[fl[k] for fl in flats] for k in range(n)]
            shape = (cells[0].r, cells[0].c)
        else:                                        # arrayfun
            flats = [as_list(a) for a in rest]
            n = len(flats[0])
            items = [[fl[k] for fl in flats] for k in range(n)]
            shape = (1, n)
        out = [self.apply_func(f, item) for item in items]
        if uniform:
            vals = []
            for v in out:
                if isinstance(v, str) or is_cell(v) or is_struct(v):
                    raise MError(f"{name}: non-scalar output — add 'UniformOutput', false")
                vals.append(scalar(v))
            if shape[1] == 1 and shape[0] > 1:
                return mmat.colvec(vals)
            return vals if len(vals) != 1 else vals[0]
        if shape[1] == 1 and shape[0] > 1:
            return Cell([[v] for v in out])
        return Cell([out]) if out else Cell([])

    def _case_matches(self, subject, label):
        """`case 3`, `case 'abc'`, `case {1, 2}` — MATLAB's three forms."""
        val = self.eval(label)
        if is_cell(val):
            return any(self._same_case(subject, x) for x in val.flat())
        return self._same_case(subject, val)

    @staticmethod
    def _same_case(a, b):
        if isinstance(a, str) or isinstance(b, str):
            return isinstance(a, str) and isinstance(b, str) and a == b
        try:
            return abs(_cr(scalar(a)) - _cr(scalar(b))) < 1e-12
        except Exception:
            return False

    def call_user(self, name, args, nargout=1):
        outs, params, body = self.funcs[name]
        if len(args) > len(params):
            raise MError(f"{name}: too many input arguments")
        if nargout > max(1, len(outs)):
            raise MError(f"{name}: too many output arguments")
        self.depth += 1
        if self.depth > _MAX_DEPTH:
            self.depth -= 1
            raise MError("maximum recursion depth exceeded")
        saved = self.vars
        self.vars = {"pi": math.pi, "e": math.e, "ans": None,
                     "nargin": float(len(args)), "nargout": float(nargout)}
        for p, a in zip(params, args):
            self.vars[p] = a
        try:
            try:
                self.exec_block(body)
            except ReturnEx:
                pass
            scope = self.vars
        finally:
            self.vars = saved
            self.depth -= 1
        if not outs:
            return None
        res = []
        for o in outs[:max(1, nargout)]:
            if o not in scope or scope[o] is None:
                raise MError(f"{name}: output argument '{o}' was not assigned")
            res.append(scope[o])
        return res[0] if nargout == 1 else res

    def apply_func(self, f, args):
        """Call an anonymous function or a function handle with evaluated args."""
        if f[0] == "fhandle":
            if f[1] in self.funcs:
                return self.call_user(f[1], args)
            return self._call_evaluated(f[1], args)
        _, params, body, closure = f
        if len(args) != len(params):
            raise MError(f"anonymous function expects {len(params)} argument(s)")
        saved = self.vars
        self.vars = dict(closure)
        for p, a in zip(params, args):
            self.vars[p] = a
        try:
            return self.ev(body)
        finally:
            self.vars = saved

    def workspace(self):
        ws = []
        for name, val in self.vars.items():
            if name in ("pi", "e") or val is None:
                continue
            if isinstance(val, str):
                cls, size = "char", f"1x{len(val)}"
            elif is_cell(val):
                cls, size = "cell", f"{val.r}x{val.c}"
            elif is_struct(val):
                cls, size = "struct", "1x1"
            elif _is_tf(val):
                cls, size = "tf", "1x1"
            elif _is_gh(val):
                cls, size = val.get("kind", "Line"), "1x1"
            elif isinstance(val, dict) and val.get("__odeopts__"):
                cls, size = "odeset options", "1x1"
            elif _is_func(val):
                cls, size = "function_handle", "1x1"
            elif is_vec(val):
                cls = "complex double" if _has_cplx(val) else "double"
                size = f"1x{len(val)}"
            elif is_mat(val):
                cls = "complex double" if _has_cplx(as_list(val)) else "double"
                size = f"{val.r}x{val.c}"
            else:
                cls = "complex double" if isinstance(val, complex) else "double"
                size = "1x1"
            entry = {"name": name, "class": cls, "size": size}
            pv = _preview(val)
            if pv is not None:
                entry["value"] = pv
            ws.append(entry)
        return ws

    # -- statements -------------------------------------------------------
    def exec_stmt(self, text, suppressed):
        text = text.strip()
        if not text:
            return
        if self.try_command(text):
            return
        eq = top_level_assign(text)
        if eq is not None:
            name = text[:eq].strip()
            rhs = text[eq + 1:].strip()
            if name.startswith("["):
                self._multi_assign(name, rhs, suppressed)
                return
            m = re.fullmatch(r"([A-Za-z_]\w*)\s*\((.*)\)", name)
            if m:
                self._indexed_assign(m.group(1), m.group(2), self.eval(rhs))
                if not suppressed:
                    self.display(m.group(1), self.vars[m.group(1)])
                return
            if not re.fullmatch(r"[A-Za-z_]\w*", name):
                # s.a = v, s.a.b = v, c{2} = v, c{1}.x = v, s.a(3) = v
                root = self._path_assign(name, self.eval(rhs))
                if not suppressed:
                    self.display(root, self.vars[root])
                return
            val = self.eval(rhs)
            self.vars[name] = val
            if not suppressed:
                self.display(name, val)
            return
        # bare expression (graphics handles returned by plot/surf don't touch ans)
        ast = Parser(tokenize(text)).parse()
        # A statement that is just f(...) calls f with nargout = 0, so freqz /
        # impz / grpdelay plot; inside an expression or an assignment they
        # return data instead (MATLAB semantics).
        self._bare_args = ast[2] if ast[0] == "call" else None
        try:
            val = self.ev(ast)
        finally:
            self._bare_args = None
        if val is not None and not _is_gh(val):
            self.vars["ans"] = val
            if not suppressed:
                self.display("ans", val)

    # -- workspace files ---------------------------------------------------
    def _who(self, detailed):
        ws = self.workspace()
        if not ws:
            return ""
        if not detailed:
            names = [w["name"] for w in ws]
            lines, row = [], []
            for n in names:
                row.append(n.ljust(14))
                if len(row) == 4:
                    lines.append("  " + "".join(row).rstrip())
                    row = []
            if row:
                lines.append("  " + "".join(row).rstrip())
            return "Your variables are:\n\n" + "\n".join(lines) + "\n\n"
        head = f"  {'Name':<16}{'Size':<12}{'Class':<18}\n"
        body = "".join(f"  {w['name']:<16}{w['size']:<12}{w['class']:<18}\n" for w in ws)
        return head + body + "\n"

    def _mat_path(self, name):
        base = name if name.endswith((".mat", ".json")) else name + ".mat"
        return base if os.path.isabs(base) else os.path.join(self.cwd, base)

    def _save_vars(self, name, only):
        """save x  /  save x a b — a JSON container, not a real .mat file."""
        keep = {}
        for k, v in self.vars.items():
            if k in ("pi", "e") or v is None:
                continue
            if only and k not in only:
                continue
            enc = _encode_value(v)
            if enc is not None:
                keep[k] = enc
        path = self._mat_path(name)
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"simforge_workspace": 1, "vars": keep}, fh)
        except OSError as ex:
            raise MError(f"could not save '{path}': {ex}")
        self.out.append(f"saved {len(keep)} variable(s) to {path}\n")

    def _load_vars(self, name, only):
        path = self._mat_path(name)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                blob = json.load(fh)
        except OSError as ex:
            raise MError(f"could not load '{path}': {ex}")
        except ValueError:
            raise MError(f"'{path}' is not a SimForge workspace file")
        if not isinstance(blob, dict) or "vars" not in blob:
            raise MError(f"'{path}' is not a SimForge workspace file")
        n = 0
        for k, enc in blob["vars"].items():
            if only and k not in only:
                continue
            self.vars[k] = _decode_value(enc)
            n += 1
        self.out.append(f"loaded {n} variable(s) from {path}\n")

    # -- lvalue paths ------------------------------------------------------
    def _path_assign(self, target, value):
        """Assign through a chain like  s.a{2}.b(3)  and return the root name."""
        try:
            node = Parser(tokenize(target)).parse()
        except MError as ex:
            raise MError(f"unsupported assignment target '{target}' ({ex})")
        chain = []
        while node[0] in ("field", "dynfield", "cellidx", "index", "call"):
            if node[0] == "field":
                chain.append(("fld", node[2]))
                node = node[1]
            elif node[0] == "dynfield":
                chain.append(("fld", scalar_str(self.ev(node[2]))))
                node = node[1]
            elif node[0] == "cellidx":
                chain.append(("cell", node[2]))
                node = node[1]
            elif node[0] == "index":
                chain.append(("idx", node[2]))
                node = node[1]
            else:                                   # ("call", name, args) == base(args)
                chain.append(("idx", node[2]))
                node = ("id", node[1])
        if node[0] != "id":
            raise MError(f"unsupported assignment target '{target}'")
        root = node[1]
        chain.reverse()
        self.vars[root] = self._set_path(self.vars.get(root), chain, value)
        return root

    def _set_path(self, cur, chain, value):
        """Recursively rebuild `cur` with `value` stored at the end of `chain`."""
        kind, arg = chain[0]
        rest = chain[1:]
        if kind == "fld":
            base = cur.copy() if is_struct(cur) else Struct()
            if cur is not None and not is_struct(cur) and not _is_empty(cur):
                raise MError("cannot add a field to a non-struct value")
            inner = base.f.get(arg)
            base.f[arg] = value if not rest else self._set_path(inner, rest, value)
            return base
        if kind == "cell":
            base = cur.copy() if is_cell(cur) else Cell([[]] if cur is None or _is_empty(cur) else None)
            if cur is not None and not is_cell(cur) and not _is_empty(cur):
                raise MError("'{}' assignment needs a cell array")
            if not base.d:
                base.d = [[]]
            try:
                if len(arg) == 2:
                    i = int(_cr(scalar(self._eval_index_node(arg[0], base.r))))
                    j = int(_cr(scalar(self._eval_index_node(arg[1], base.c))))
                    inner = base.d[i - 1][j - 1] if (i <= base.r and j <= base.c) else None
                    mtypes.cell_set2(base, i, j,
                                     value if not rest else self._set_path(inner, rest, value))
                else:
                    n = base.r * base.c
                    k = int(_cr(scalar(self._eval_index_node(arg[0], n))))
                    flat = base.flat()
                    inner = flat[k - 1] if 1 <= k <= len(flat) else None
                    mtypes.cell_set(base, k,
                                    value if not rest else self._set_path(inner, rest, value))
            except mtypes.MTypeError as ex:
                raise MError(str(ex))
            return base
        # kind == "idx" — numeric or cell paren-indexing at the end of the chain
        if rest:
            raise MError("indexing in the middle of an assignment target is not supported")
        if is_cell(cur):
            if isinstance(value, list) and not value:
                n = cur.r * cur.c
                idx = self._to_indices(self._eval_index_node(arg[0], n), n)
                return mtypes.cell_delete(cur, idx)
            raise MError("use c{i} = v to set a cell element")
        holder = "__pathtmp__"
        saved = self.vars.get(holder)
        self.vars[holder] = cur
        try:
            self._indexed_assign(holder, None, value, idx_nodes=arg)
            return self.vars[holder]
        finally:
            if saved is None:
                self.vars.pop(holder, None)
            else:
                self.vars[holder] = saved

    def _eval_index_text(self, text, n):
        """Evaluate one index expression with `end` bound to n; ':' -> None."""
        text = text.strip()
        if text == ":":
            return None
        old = self.vars.get("end")
        self.vars["end"] = float(n)
        try:
            return as_list(self.eval(text))
        finally:
            if old is None:
                self.vars.pop("end", None)
            else:
                self.vars["end"] = old

    def _eval_index_node(self, node, n):
        if node[0] == "colon":
            return None
        old = self.vars.get("end")
        self.vars["end"] = float(n)
        try:
            return as_list(self.ev(node))
        finally:
            if old is None:
                self.vars.pop("end", None)
            else:
                self.vars["end"] = old

    @staticmethod
    def _to_indices(idx, n):
        """Numeric indices -> ints; a same-length 0/1 vector is a logical mask."""
        vals = [_cr(k) for k in idx]
        if n and len(vals) == n and all(v in (0.0, 1.0) for v in vals) and any(v == 0.0 for v in vals):
            return [k + 1 for k, v in enumerate(vals) if v == 1.0]
        out = []
        for v in vals:
            k = int(round(v))
            if abs(v - k) > 1e-9:
                raise MError("index must be a positive integer")
            out.append(k)
        return out

    def _eval_idx(self, spec, n):
        """One index of an assignment target: source text, or an already-parsed node."""
        return self._eval_index_node(spec, n) if isinstance(spec, tuple) \
            else self._eval_index_text(spec, n)

    def _indexed_assign(self, base, idx_text, val, idx_nodes=None):
        """x(i) = v, A(i, j) = v, A(:, j) = v, x(i) = [] — grows like MATLAB."""
        cur = self.vars.get(base)
        parts = idx_nodes if idx_nodes is not None else _split_top_commas(idx_text)
        r, c = mmat.shape(cur) if cur is not None else (0, 0)
        deleting = isinstance(val, list) and len(val) == 0
        try:
            if len(parts) == 2:
                ri = self._eval_idx(parts[0], r)
                ci = self._eval_idx(parts[1], c)
                ri = None if ri is None else self._to_indices(ri, r)
                ci = None if ci is None else self._to_indices(ci, c)
                if deleting:
                    self.vars[base] = mmat.delete2(cur, ri, ci)
                else:
                    self.vars[base] = mmat.assign2(cur, ri, ci, val)
                return
            if len(parts) != 1:
                raise MError("only 1-D and 2-D indexing are supported")
            flat = list(mmat.flat_cm(cur)) if cur is not None else []
            idx = self._eval_idx(parts[0], len(flat))
            idx = list(range(1, len(flat) + 1)) if idx is None else self._to_indices(idx, len(flat))
            if deleting:
                drop = set(idx)
                keep = [x for k, x in enumerate(flat, 1) if k not in drop]
                self.vars[base] = mmat.colvec(keep) if (is_mat(cur) and cur.c == 1) else keep
                return
            vals = as_list(val)
            if len(vals) == 1 and len(idx) > 1:
                vals = vals * len(idx)
            if len(vals) != len(idx):
                raise MError("indexed assignment: size mismatch")
            for k in idx:
                if k < 1:
                    raise MError(f"index {k} must be a positive integer")
                if k > len(flat):
                    flat.extend([0.0] * (k - len(flat)))
            for k, v in zip(idx, vals):
                flat[k - 1] = v
            if is_mat(cur) and cur.c == 1:
                self.vars[base] = mmat.colvec(flat)
            elif is_mat(cur) and len(flat) == r * c:
                self.vars[base] = mmat.from_flat_cm(flat, r, c)
            else:
                self.vars[base] = flat
        except mmat.MatError as ex:
            raise MError(str(ex))

    def _multi_assign(self, name, rhs, suppressed):
        """[q, r] = f(...) — multiple outputs from a user-defined function."""
        targets = [t.strip() for t in name.strip("[]").split(",") if t.strip()]
        ast = Parser(tokenize(rhs)).parse()
        if ast[0] == "call" and ast[1] in _SIM_FUNCS and ast[1] not in self.funcs:
            args = [self.ev(a) for a in ast[2]]
            results = self._sim_call(ast[1], args, nargout=len(targets))
        elif ast[0] == "call" and ast[1] in _CKT_FUNCS and ast[1] not in self.funcs:
            args = [self.ev(a) for a in ast[2]]
            results = self._ckt_call(ast[1], args, nargout=len(targets))
        elif ast[0] == "call" and ast[1] in _MULTI_FUNCS and ast[1] not in self.funcs:
            args = [self.ev(a) for a in ast[2]]
            results = _MULTI_FUNCS[ast[1]](args)
        elif ast[0] == "call" and ast[1] in self.funcs:
            args = [self.ev(a) for a in ast[2]]
            results = self.call_user(ast[1], args, nargout=len(targets))
            if len(targets) == 1:
                results = [results]
        else:
            raise MError("multiple outputs need a function call that returns multiple values")
        if len(results) < len(targets):
            raise MError(f"function returns {len(results)} value(s), "
                         f"but {len(targets)} were requested")
        for t, v in zip(targets, results):
            if t == "~":                      # MATLAB's "ignore this output"
                continue
            self.vars[t] = v
            if not suppressed:
                self.display(t, v)

    def try_command(self, text):
        """Handle space-separated commands: hold on, grid off, close all, clc..."""
        m2 = re.fullmatch(r"shading\s+(interp|flat|faceted)", text)
        if m2:
            self._ensure_fig()["shading"] = m2.group(1)
            return True
        m2 = re.fullmatch(r"colormap\s+(\w+)", text)
        if m2:
            self._set_colormap(m2.group(1))
            return True
        if text == "colorbar":
            self._ensure_fig()["colorbar"] = True
            return True
        if text in ("box on", "box off", "rotate3d", "rotate3d on", "rotate3d off",
                    "axis vis3d", "axis square", "axis auto", "hidden on", "hidden off"):
            return True
        m = re.fullmatch(r"(hold|grid|axis)\s+(on|off|equal|tight)", text)
        if m:
            cmd, arg = m.groups()
            if cmd == "hold":
                self.hold = (arg == "on")
            elif cmd == "grid":
                self._ensure_fig()["grid"] = (arg == "on")
            elif cmd == "axis":
                ax = self._ensure_fig()
                if arg == "equal":
                    ax["equal"] = True
                elif arg == "tight":
                    ax["equal"] = False
            return True
        if text in ("hold", "grid"):
            if text == "hold":
                self.hold = not self.hold
            else:
                f = self._ensure_fig()
                f["grid"] = not f.get("grid", False)
            return True
        if text in ("clc", "shg", "refresh", "axis manual", "hold all", "drawnow limitrate",
                    "drawnow update", "drawnow nocallbacks"):
            if text == "clc":
                self.clc = True          # the UI clears the Command Window
            if text == "hold all":
                self.hold = True
            if text.startswith("drawnow"):
                self._snapshot(40)
            return True
        if text in ("who", "whos"):
            self.out.append(self._who(text == "whos"))
            return True
        m = re.fullmatch(r"(save|load)\s+(\S+)((\s+[A-Za-z_]\w*)*)", text)
        if m:
            names = m.group(3).split()
            if m.group(1) == "save":
                self._save_vars(m.group(2), names)
            else:
                self._load_vars(m.group(2), names)
            return True
        # `clear` / `clearvars` — command syntax only (clear(x) is not a thing)
        m = re.fullmatch(r"(clear|clearvars)((\s+[A-Za-z_]\w*)*)", text)
        if m:
            names = m.group(2).split()
            if not names or names == ["all"] or names == ["variables"]:
                keep = {"pi": self.vars["pi"], "e": self.vars["e"], "ans": None}
                self.vars = keep
                if not names or names == ["all"]:
                    self.funcs = {}
            else:
                for n in names:
                    self.vars.pop(n, None)
            return True
        if text == "drawnow":
            self._snapshot(40)
            return True
        if text == "pause":
            self._snapshot(500)
            return True
        if text == "cla":
            if self.cur is not None:
                for k in ("series", "vlines", "hlines", "texts"):
                    self.cur[k] = []
                self._color_idx = 0
            return True
        if text == "clf":
            # keep the figure (and its animation frames), drop its axes
            if self.curfig is not None:
                self.curfig["axes"] = []
                self.curfig["slots"] = {}
            self.cur = None
            self._color_idx = 0
            return True
        if text in ("close", "close all", "close all force"):
            self.cur = None
            return True
        if text == "figure":
            self._finish_figure()
            self.cur = None
            return True
        return False

    # -- expression eval --------------------------------------------------
    def eval(self, text):
        toks = tokenize(text)
        ast = Parser(toks).parse()
        return self.ev(ast)

    def ev(self, node):
        t = node[0]
        if t == "num":
            return node[1]
        if t == "str":
            return node[1]
        if t == "id":
            name = node[1]
            if name in self.vars:
                return self.vars[name]
            if name == "end":
                raise MError("'end' is only valid inside an index")
            if name in self.funcs:             # `show` / `v = getv;` — no parentheses
                return self.call_user(name, [])
            if name in _ZERO_ARG:
                return _ZERO_ARG[name]()
            if name in _BUILDERS:              # `u = rand;`  `rng`  `zeros`
                return _norm_builder(_BUILDERS[name]([]))
            raise MError(f"undefined variable or function '{name}'")
        if t == "neg":
            v = self.ev(node[1])
            if _is_tf(v):
                return make_tf([-c for c in v["num"]], v["den"])
            return elementwise(v, 0, lambda x, _: -x)
        if t == "not":
            v = self.ev(node[1])
            return elementwise(v, 0, lambda x, _: 1.0 if x == 0 else 0.0)
        if t == "lambda":
            return ("mfunc", node[1], node[2], dict(self.vars))
        if t == "fh":
            return ("fhandle", node[1])
        if t == "colon":
            raise MError("':' on its own is only valid inside an index, e.g. A(:, 2)")
        if t == "ctranspose":
            v = self.ev(node[1])
            if _is_tf(v) or _is_func(v) or isinstance(v, str):
                raise MError("transpose needs a numeric array")
            return mmat.ctranspose(v)
        if t == "transpose":
            v = self.ev(node[1])
            if _is_tf(v) or _is_func(v) or isinstance(v, str):
                raise MError("transpose needs a numeric array")
            return mmat.transpose(v)
        if t in ("mat", "vec"):
            rows = node[1] if t == "mat" else [node[1]]
            try:
                return _vcat([_hcat([self.ev(e) for e in row]) for row in rows])
            except mmat.MatError as ex:
                raise MError(str(ex))
        if t == "range":
            parts = [scalar(self.ev(p)) for p in node[1]]
            return make_range(parts)
        if t == "bin":
            return self.ev_bin(node[1], self.ev(node[2]), self.ev(node[3]))
        if t == "call":
            return self.ev_call(node[1], node[2])
        if t == "cellexp":
            rows = []
            for row in node[1]:
                vals = []
                for e in row:
                    v = self.ev(e)
                    vals.extend(v) if isinstance(v, _CSList) else vals.append(v)
                rows.append(mtypes.cellhcat(vals))
            rows = [r for r in rows if r] if len(rows) > 1 else rows
            if not rows or not rows[0]:
                return Cell([])
            width = len(rows[0])
            if any(len(r) != width for r in rows):
                raise MError("cell rows must have the same number of elements")
            return Cell(rows)
        if t == "index":
            return self._index(self.ev(node[1]), node[2])
        if t == "cellidx":
            return self._cell_content(self.ev(node[1]), node[2])
        if t == "field":
            return self._get_field(self.ev(node[1]), node[2])
        if t == "dynfield":
            return self._get_field(self.ev(node[1]), scalar_str(self.ev(node[2])))
        raise MError(f"cannot evaluate {t}")

    # -- cells & structs ---------------------------------------------------
    def _cell_content(self, val, idx_nodes):
        """c{i} / c{i,j} — the contents, not a cell."""
        if not is_cell(val):
            raise MError("'{}' indexing needs a cell array")
        try:
            if len(idx_nodes) == 2:
                i = int(_cr(scalar(self._eval_index_node(idx_nodes[0], val.r))))
                j = int(_cr(scalar(self._eval_index_node(idx_nodes[1], val.c))))
                return mtypes.cell_get2(val, i, j)
            idx = self._to_indices(
                self._eval_index_node(idx_nodes[0], val.r * val.c), val.r * val.c)
            if len(idx) == 1:
                return mtypes.cell_get(val, idx[0])
            # c{1:3} is a comma-separated list; students use it inside [ ] or { }
            return _CSList([mtypes.cell_get(val, k) for k in idx])
        except mtypes.MTypeError as ex:
            raise MError(str(ex))

    @staticmethod
    def _get_field(val, name):
        if not is_struct(val):
            raise MError(f"cannot read field '{name}' of a non-struct value")
        if name not in val.f:
            raise MError(f"a struct has no field '{name}'")
        return val.f[name]

    def ev_bin(self, op, a, b):
        if _is_tf(a) or _is_tf(b):
            return _tf_bin(op, a, b)
        if is_mat(a) or is_mat(b):
            try:
                if op == "*":
                    return mmat.matmul(a, b)
                if op == "/":
                    return mmat.mrdivide(a, b)
                if op == "\\":
                    return mmat.solve(a, b)
                if op == "^":
                    if is_mat(a) and not is_mat(b):
                        return mmat.mpower(a, float(_cr(scalar(b))))
                    raise MError("only matrix^integer is supported (use .^ for element-wise)")
            except mmat.MatError as ex:
                raise MError(str(ex))
        if op == "\\":
            # scalar \ x  ==  x / scalar ;  row-vector systems are not meaningful
            if not is_vec(a):
                return elementwise(b, a, _safe_div)
            raise MError("A\\b needs a matrix A (build it with [a b; c d])")
        if op in _CMP_OPS:
            return elementwise(a, b, _CMP_OPS[op])
        if op in ("&", "&&"):
            return elementwise(a, b, lambda x, y: 1.0 if (x != 0 and y != 0) else 0.0)
        if op in ("|", "||"):
            return elementwise(a, b, lambda x, y: 1.0 if (x != 0 or y != 0) else 0.0)
        if op in ("+",):
            return elementwise(a, b, lambda x, y: x + y)
        if op == "-":
            return elementwise(a, b, lambda x, y: x - y)
        if op in (".*", "*"):
            # plain * between vectors is treated element-wise in demo mode
            return elementwise(a, b, lambda x, y: x * y)
        if op in ("./", "/"):
            return elementwise(a, b, _safe_div)
        if op in (".^", "^"):
            return elementwise(a, b, lambda x, y: _safe_pow(x, y))
        raise MError(f"unknown operator '{op}'")

    def ev_call(self, name, arg_nodes):
        # Plotting / void commands first.
        if name in ("drawnow", "pause", "getframe", "comet", "comet3", "animatedline",
                    "addpoints", "clearpoints", "set", "get", "gca", "gcf", "delete",
                    "movie", "refreshdata", "waitforbuttonpress", "beep"):
            return self._anim_call(name, [self.ev(a) for a in arg_nodes])
        if name in ("cellfun", "arrayfun", "structfun"):
            return self._map_call(name, [self.ev(a) for a in arg_nodes], nargout=1)
        if name in _SIM_FUNCS:
            return self._sim_call(name, [self.ev(a) for a in arg_nodes], nargout=1)
        if name in _CKT_FUNCS:
            return self._ckt_call(name, [self.ev(a) for a in arg_nodes], nargout=1)
        if name in _PLOT3_FUNCS:
            return self._plot3_call(name, arg_nodes)
        if name == "colormap":
            # colormap(jet) / colormap('hot') / colormap(gray(64)) / colormap default
            nm = "parula"
            if arg_nodes:
                a0 = arg_nodes[0]
                if a0[0] in ("id", "call") and a0[1].lower() in mplot3.COLORMAPS:
                    nm = a0[1]
                elif a0[0] == "str":
                    nm = a0[1]
            self._set_colormap(nm)
            return None
        if name in ("histogram", "hist"):
            return self._hist_call(name, arg_nodes)
        if name in _PLOT_FUNCS:
            return self._plot_call(name, arg_nodes)
        if name in ("title", "xlabel", "ylabel"):
            self._ensure_fig()[name] = str(scalar_str(self.ev(arg_nodes[0])))
            return None
        if name == "legend":
            return self._legend([self.ev(a) for a in arg_nodes])
        if name == "subplot":
            return self._subplot([self.ev(a) for a in arg_nodes])
        if name in ("sgtitle", "suptitle"):
            self._ensure_figure()["sgtitle"] = str(scalar_str(self.ev(arg_nodes[0])))
            return None
        if name in ("xlim", "ylim", "zlim", "axis", "xline", "yline", "text", "colorbar",
                    "zlabel", "view", "shading", "caxis", "clim", "daspect", "pbaspect",
                    "camlight", "lighting", "material", "rotate3d", "box"):
            return self._axes_call(name, [self.ev(a) for a in arg_nodes])
        if name == "disp":
            self.out.append(self._fmt_value(self.ev(arg_nodes[0])) + "\n")
            return None
        if name == "fprintf":
            self.out.append(self._do_fprintf(arg_nodes))
            return None
        if name == "sprintf":
            return self._sprintf(arg_nodes)
        if name == "error":
            raise MError(scalar_str(self.ev(arg_nodes[0])) if arg_nodes else "error")

        # A variable holding a callable: anonymous fn or function handle.
        if name in self.vars and _is_func(self.vars[name]):
            return self.apply_func(self.vars[name], [self.ev(a) for a in arg_nodes])
        # Indexing of an existing variable: x(i), A(i, j), A(:, j), x(:)
        if name in self.vars and len(arg_nodes) in (1, 2):
            return self._index(self.vars[name], arg_nodes)

        args = [self.ev(a) for a in arg_nodes]
        # User-defined functions shadow built-ins (like MATLAB local functions).
        if name in self.funcs:
            return self.call_user(name, args)
        # Signal commands that draw (take vectors, not a tf).
        bare = arg_nodes is self._bare_args
        if bare and name == "rng" and args and name not in self.funcs:
            _rng(args)                     # rng(seed) as a statement prints nothing
            return None
        if not bare and (name == "freqz" or name in _DSP_PLOTS) and name in _MULTI_FUNCS:
            _need(name, args, 1)
            return _MULTI_FUNCS[name](args)[0]
        if name in ("freqz", "periodogram", "pwelch"):
            _need(name, args, 1)
            return self._signal_call(name, args)
        if name in _DSP_PLOTS:
            _need(name, args, 1)
            return self._dsp_plot(name, args)
        # Control-systems commands that draw or print need the interpreter.
        if name in ("step", "impulse", "bode", "pzmap", "margin",
                    "rlocus", "nyquist", "lsim", "stepinfo", "damp"):
            _need(name, args, 1)
            return self._control_call(name, args)
        if name in _CONTROL_FUNCS:
            _need(name, args, 1)
            return _CONTROL_FUNCS[name](args)
        return self._call_evaluated(name, args)

    def _call_evaluated(self, name, args):
        """Dispatch to the pure function tables with already-evaluated args."""
        if name in _ELEMENTWISE:
            _need(name, args, 1)
            return elementwise(args[0], 0, lambda x, _: _ELEMENTWISE[name](x))
        if name in _BINARY:
            _need(name, args, 2)
            return elementwise(args[0], args[1], _BINARY[name])
        if name in _TYPE_FUNCS:
            try:
                return _TYPE_FUNCS[name](args)
            except mtypes.MTypeError as ex:
                raise MError(str(ex))
        if any(is_cell(a) or is_struct(a) for a in args) and name in _CONTAINER_FUNCS:
            try:
                return _CONTAINER_FUNCS[name](args)
            except mtypes.MTypeError as ex:
                raise MError(str(ex))
        if name in _MATFUNCS:
            try:
                return _MATFUNCS[name](args)
            except mmat.MatError as ex:
                raise MError(str(ex))
        if name in _REDUCERS:
            _need(name, args, 1)
            r = _reduce_call(name, args)
            if r is not None:
                return r
            return _REDUCERS[name](as_list(args[0]))
        if name in _BUILDERS:
            return _BUILDERS[name](args)
        if name in _VECTOR_FUNCS:
            _need(name, args, 1)
            return _VECTOR_FUNCS[name](args)
        if name in self.funcs:
            return self.call_user(name, args)
        raise MError(f"unknown function '{name}'")

    def _index(self, val, idx_nodes):
        if is_cell(val):
            try:
                n = val.r * val.c
                if len(idx_nodes) == 2:
                    ri = self._to_indices(self._eval_index_node(idx_nodes[0], val.r), val.r)
                    ci = self._to_indices(self._eval_index_node(idx_nodes[1], val.c), val.c)
                    return Cell([[mtypes.cell_get2(val, i, j) for j in ci] for i in ri])
                node = idx_nodes[0]
                idx = list(range(1, n + 1)) if node[0] == "colon" else \
                    self._to_indices(self._eval_index_node(node, n), n)
                return mtypes.cell_index(val, idx)
            except mtypes.MTypeError as ex:
                raise MError(str(ex))
        if is_struct(val):
            idx = self._to_indices(self._eval_index_node(idx_nodes[0], 1), 1)
            if idx == [1]:
                return val                    # s(1) of a scalar struct is itself
            raise MError("struct arrays are not supported")
        if isinstance(val, str):
            chars = list(val)
            node = idx_nodes[0]
            idx = list(range(1, len(chars) + 1)) if node[0] == "colon" else \
                self._to_indices(self._eval_index_node(node, len(chars)), len(chars))
            out = []
            for k in idx:
                if k < 1 or k > len(chars):
                    raise MError(f"index {k} out of bounds (1..{len(chars)})")
                out.append(chars[k - 1])
            return "".join(out)
        try:
            if len(idx_nodes) == 2:
                r, c = mmat.shape(val)
                ri = self._eval_index_node(idx_nodes[0], r)
                ci = self._eval_index_node(idx_nodes[1], c)
                ri = None if ri is None else self._to_indices(ri, r)
                ci = None if ci is None else self._to_indices(ci, c)
                return mmat.index2(val, ri, ci)
            node = idx_nodes[0]
            vec = as_list(val)
            if node[0] == "colon":
                return mmat.colvec(vec)                 # A(:) is always a column
            idx = self._to_indices(self._eval_index_node(node, len(vec)), len(vec))
            out = []
            for ki in idx:
                if ki < 1 or ki > len(vec):
                    raise MError(f"index {ki} out of bounds (1..{len(vec)})")
                out.append(vec[ki - 1])
            if len(out) == 1:
                return out[0]
            # a column vector indexed by a vector stays a column
            if is_mat(val) and val.c == 1:
                return mmat.colvec(out)
            return out
        except mmat.MatError as ex:
            raise MError(str(ex))

    # -- plotting ---------------------------------------------------------
    def _ensure_fig(self):
        """Current AXES (a subplot cell) — created on first use."""
        if self.cur is None:
            if self.curfig is None:
                self.curfig = new_figure()
            ax = new_axes()
            self.curfig["axes"].append(ax)
            self.curfig["slots"][(1,)] = ax
            self.cur = ax
        return self.cur

    def _ensure_figure(self):
        """Current FIGURE (the container that holds the subplot axes)."""
        self._ensure_fig()
        return self.curfig

    def _finish_figure(self):
        if self.curfig is not None and fig_has_content(self.curfig):
            self.figures.append(self.curfig)
        self.curfig = None
        self.cur = None
        self._color_idx = 0

    def _add_fig(self, ax):
        """Publish a one-axes figure built by a command (bode, freqz, ...)."""
        base = new_axes()
        base.update(ax)
        fg = new_figure()
        fg["axes"] = [base]
        self.figures.append(fg)

    def _subplot(self, args):
        if len(args) == 1:
            v = int(scalar(args[0]))
            rows, cols, slots = v // 100, (v // 10) % 10, [v % 10]
        elif len(args) >= 3:
            rows = int(scalar(args[0]))
            cols = int(scalar(args[1]))
            slots = [int(round(float(_cr(k)))) for k in as_list(args[2])]
        else:
            raise MError("subplot(rows, cols, k)")
        if rows < 1 or cols < 1:
            raise MError("subplot: rows and cols must be >= 1")
        for k in slots:
            if k < 1 or k > rows * cols:
                raise MError(f"subplot: index {k} is outside a {rows}x{cols} grid")
        if self.curfig is None:
            self.curfig = new_figure()
        fg = self.curfig
        fg["rows"], fg["cols"] = rows, cols
        key = tuple(sorted(slots))
        ax = fg["slots"].get(key)
        if ax is None:
            ax = new_axes()
            r0 = min((k - 1) // cols for k in slots)
            r1 = max((k - 1) // cols for k in slots)
            c0 = min((k - 1) % cols for k in slots)
            c1 = max((k - 1) % cols for k in slots)
            ax["row"], ax["col"] = r0, c0
            ax["span"], ax["rspan"] = c1 - c0 + 1, r1 - r0 + 1
            fg["axes"].append(ax)
            fg["slots"][key] = ax
        self.cur = ax
        self._color_idx = 0
        self.hold = False
        return None

    def _next_color(self):
        c = _PALETTE[self._color_idx % len(_PALETTE)]
        self._color_idx += 1
        return c

    def _hist_call(self, name, arg_nodes):
        """histogram(x) / histogram(x, nbins) / hist(x, nbins): bin counts as bars."""
        if not arg_nodes:
            raise MError(f"{name} needs data")
        data = [v for v in _plot_reals(as_list(self.ev(arg_nodes[0]))) if not math.isnan(v)]
        if not data:
            raise MError(f"{name}: no data to bin")
        if len(arg_nodes) > 1:
            nb = int(scalar(self.ev(arg_nodes[1])))
            if nb < 1:
                raise MError(f"{name}: the number of bins must be a positive integer")
        else:   # hist defaults to 10 bins; histogram picks a count from the data
            nb = 10 if name == "hist" else max(1, math.ceil(math.sqrt(len(data))))
        lo, hi = min(data), max(data)
        if hi == lo:
            lo, hi = lo - 0.5, hi + 0.5
        w = (hi - lo) / nb
        counts = [0.0] * nb
        for v in data:
            counts[min(nb - 1, int((v - lo) / w))] += 1
        fig = self._ensure_fig()
        if not self.hold:
            fig["series"] = []
            self._color_idx = 0
        _, style, marker = parse_spec("")
        s = {
            "x": [lo + (k + 0.5) * w for k in range(nb)], "y": counts, "kind": "bar",
            "color": self._next_color(), "style": style, "marker": marker,
            "width": 1.3, "msize": 5.0,
        }
        fig["series"].append(s)
        return self._handle(fig, s)

    def _plot_call(self, kind, arg_nodes):
        fig = self._ensure_fig()
        if kind == "semilogx":
            fig["xlog"] = True
            kind = "plot"
        elif kind == "semilogy":
            fig["ylog"] = True
            kind = "plot"
        elif kind == "loglog":
            fig["xlog"] = True
            fig["ylog"] = True
            kind = "plot"
        if not self.hold:
            fig["series"] = []
            self._color_idx = 0
        # Evaluate args; strings are either line specs or property names.
        vals = [("str", a[1]) if a[0] == "str" else ("num", self.ev(a)) for a in arg_nodes]
        # plot(Y) / plot(x, Y) with a matrix Y draws one line per column
        expanded = []
        i = 0
        while i < len(vals):
            k, v = vals[i]
            nxt = vals[i + 1] if i + 1 < len(vals) else (None, None)
            if k == "num" and is_mat(v) and v.r > 1 and v.c > 1 and not (nxt[0] == "num"):
                for j in range(v.c):
                    expanded.append(("num", [float(t + 1) for t in range(v.r)]))
                    expanded.append(("num", v.col(j)))
                i += 1
                continue
            if k == "num" and nxt[0] == "num" and is_mat(nxt[1]) and nxt[1].r > 1 and nxt[1].c > 1:
                xs = as_list(v)
                for j in range(nxt[1].c):
                    expanded.append(("num", xs))
                    expanded.append(("num", nxt[1].col(j)))
                i += 2
                continue
            expanded.append((k, v))
            i += 1
        vals = expanded
        i, n = 0, len(vals)
        while i < n:
            if vals[i][0] == "str":   # stray leading string -> skip
                i += 1
                continue
            first = _plot_reals(as_list(vals[i][1]))
            i += 1
            # a following plain numeric (not a property value) is the y vector
            if i < n and vals[i][0] == "num":
                x, y = first, _plot_reals(as_list(vals[i][1]))
                i += 1
            else:
                y = first
                x = list(range(1, len(first) + 1))
            # optional line spec ('r--o'), but not a property keyword
            spec = ""
            if i < n and vals[i][0] == "str" and vals[i][1].lower() not in _PLOT_PROPS:
                spec = vals[i][1]
                i += 1
            color, style, marker = parse_spec(spec)
            s = {
                "x": x, "y": y, "kind": kind, "color": color or self._next_color(),
                "style": style, "marker": marker, "width": 1.3, "msize": 5.0,
            }
            # name/value property pairs: 'LineWidth', 2, 'Color', 'r', ...
            while i + 1 < n and vals[i][0] == "str" and vals[i][1].lower() in _PLOT_PROPS:
                self._apply_prop(s, vals[i][1].lower(), vals[i + 1][1])
                i += 2
            fig["series"].append(s)
        return self._handle(fig, fig["series"][-1]) if fig["series"] else None

    def _apply_prop(self, s, key, val):
        if key == "linewidth" and not isinstance(val, str):
            s["width"] = float(scalar(val))
        elif key == "markersize" and not isinstance(val, str):
            s["msize"] = float(scalar(val))
        elif key == "color":
            s["color"] = _to_color(val)
        elif key == "linestyle" and isinstance(val, str):
            s["style"] = None if val == "none" else val
        elif key == "marker" and isinstance(val, str):
            s["marker"] = None if val == "none" else val
        elif key in ("markerfacecolor", "markeredgecolor"):
            pass  # accepted, not distinctly rendered
        elif key == "displayname" and isinstance(val, str):
            s["name"] = val

    # -- control-systems commands ------------------------------------------
    def _control_call(self, name, args):
        sys = args[0]
        if not _is_tf(sys):
            raise MError(f"{name} expects a transfer function (make one with tf(num, den))")
        if name in ("step", "impulse"):
            tfin = float(_cr(scalar(args[1]))) if len(args) > 1 else None
            t, y = _tf_time_response(sys, name, tfin)
            fig = self._ensure_fig()
            if not self.hold:
                fig["series"] = []
                self._color_idx = 0
            fig["series"].append({"x": t, "y": y, "kind": "plot",
                                  "color": self._next_color(), "style": None,
                                  "marker": None, "width": 1.5, "msize": 5.0})
            if not fig.get("title"):
                fig["title"] = "Step Response" if name == "step" else "Impulse Response"
            fig["xlabel"] = fig.get("xlabel") or "Time (s)"
            fig["ylabel"] = fig.get("ylabel") or "Amplitude"
            fig["grid"] = True
            return None
        if name == "bode":
            w, mdb, phd = _tf_freq_response(sys)
            self._finish_figure()
            base = {"kind": "plot", "color": _PALETTE[0], "style": None,
                    "marker": None, "width": 1.5, "msize": 5.0}
            self._add_fig({"series": [dict(base, x=w, y=mdb)],
                           "title": "Bode Magnitude",
                           "xlabel": "Frequency (rad/s)", "ylabel": "Magnitude (dB)",
                           "grid": True, "xlog": True})
            self._add_fig({"series": [dict(base, x=w, y=phd)],
                           "title": "Bode Phase",
                           "xlabel": "Frequency (rad/s)", "ylabel": "Phase (deg)",
                           "grid": True, "xlog": True})
            return None
        if name == "pzmap":
            poles = _roots_dk(sys["den"])
            zeros = _roots_dk(sys["num"])
            fig = self._ensure_fig()
            fig["series"] = []
            fig["series"].append({"x": [p.real for p in poles], "y": [p.imag for p in poles],
                                  "kind": "scatter", "color": "#0072BD", "style": None,
                                  "marker": "x", "width": 1.3, "msize": 7.0})
            if zeros:
                fig["series"].append({"x": [z.real for z in zeros], "y": [z.imag for z in zeros],
                                      "kind": "scatter", "color": "#D95319", "style": None,
                                      "marker": "o", "width": 1.3, "msize": 7.0})
            fig["title"] = fig.get("title") or "Pole-Zero Map"
            fig["xlabel"] = "Real axis"
            fig["ylabel"] = "Imaginary axis"
            fig["grid"] = True
            return None
        if name == "margin":
            gm, wg, pm, wp = _tf_margins(sys)
            gtxt = f"{gm:.2f} dB (at {wg:.4g} rad/s)" if gm is not None else "Inf"
            ptxt = f"{pm:.2f} deg (at {wp:.4g} rad/s)" if pm is not None else "Inf"
            self.out.append(f"Gain margin:  {gtxt}\nPhase margin: {ptxt}\n")
            return None
        if name == "lsim":
            _need("lsim", args, 3)
            t, y = _tf_lsim(sys, args[1], args[2])
            fig = self._ensure_fig()
            if not self.hold:
                fig["series"] = []
                self._color_idx = 0
            u = [float(_cr(v)) for v in as_list(args[1])]
            if len(u) == len(t):
                fig["series"].append({"x": t, "y": u, "kind": "plot",
                                      "color": "#7d7d7d", "style": "--", "marker": None,
                                      "width": 1.0, "msize": 5.0, "name": "input"})
            fig["series"].append({"x": t, "y": y, "kind": "plot",
                                  "color": self._next_color(), "style": None,
                                  "marker": None, "width": 1.6, "msize": 5.0,
                                  "name": "output"})
            fig["title"] = fig.get("title") or "Linear Simulation"
            fig["xlabel"] = fig.get("xlabel") or "Time (s)"
            fig["ylabel"] = fig.get("ylabel") or "Amplitude"
            fig["grid"] = True
            return None
        if name == "rlocus":
            self._rlocus(sys)
            return None
        if name == "nyquist":
            self._nyquist(sys)
            return None
        if name == "stepinfo":
            info = _stepinfo(sys)
            self.out.append(
                "        RiseTime: {RiseTime:.4g}\n"
                "     SettlingTime: {SettlingTime:.4g}\n"
                "        Overshoot: {Overshoot:.4g} %\n"
                "             Peak: {Peak:.4g}\n"
                "         PeakTime: {PeakTime:.4g}\n".format(**info))
            return None
        if name == "damp":
            poles = _roots_dk(sys["den"])
            self.out.append("        Pole              Damping        Frequency (rad/s)\n")
            for p in poles:
                wn = abs(p)
                zeta = -p.real / wn if wn > 1e-12 else 1.0
                ps = f"{_clean_c(p):.4g}" if isinstance(_clean_c(p), complex) \
                    else f"{_clean_c(p):.4g}"
                self.out.append(f"  {ps:>16}   {zeta:>10.4f}     {wn:>14.4g}\n")
            return None
        return None

    # -- signal-processing commands ----------------------------------------
    def _signal_call(self, name, args):
        if name == "freqz":
            b, a, ws, fs = _freq_args("freqz", args)
            mag, ph = [], []
            for w in ws:
                z = cmath.exp(-1j * w)
                num = sum(complex(b[k]) * z ** k for k in range(len(b)))
                den = sum(complex(a[k]) * z ** k for k in range(len(a)))
                h = num / den if den != 0 else complex(float("inf"))
                m = abs(h)
                mag.append(20 * math.log10(m) if m > 0 else -120.0)
                ph.append(cmath.phase(h))
            # normalized (×π rad/sample), or Hz when a sample rate was given
            wn = _freq_out(ws, fs) if fs else [w / math.pi for w in ws]
            xl = "Frequency (Hz)" if fs else "Normalized frequency (×π rad/sample)"
            phd = [p * 180 / math.pi for p in _vf_unwrap([ph])]
            self._finish_figure()
            base = {"kind": "plot", "color": _PALETTE[0], "style": None,
                    "marker": None, "width": 1.5, "msize": 5.0}
            self._add_fig({"series": [dict(base, x=wn, y=mag)],
                           "title": "Digital Filter — Magnitude",
                           "xlabel": xl,
                           "ylabel": "Magnitude (dB)", "grid": True})
            self._add_fig({"series": [dict(base, x=wn, y=phd)],
                           "title": "Digital Filter — Phase",
                           "xlabel": xl,
                           "ylabel": "Phase (deg)", "grid": True})
            return None
        if name in ("periodogram", "pwelch"):
            x = [float(_cr(v)) for v in as_list(args[0])]
            fs = float(_cr(scalar(args[1]))) if len(args) > 1 and args[1] is not None \
                and not is_vec(args[1]) else (float(_cr(scalar(args[-1])))
                                              if len(args) > 1 and not is_vec(args[-1]) else 1.0)
            n = len(x)
            if n < 2:
                raise MError(f"{name}: need at least two samples")
            w = [0.5 - 0.5 * math.cos(2 * math.pi * k / (n - 1)) for k in range(n)]  # Hann
            xw = [x[k] * w[k] for k in range(n)]
            U = sum(wi * wi for wi in w) / n
            X = _dft([complex(v) for v in xw], inverse=False)
            half = n // 2 + 1
            pxx = [(abs(X[k]) ** 2) / (fs * n * U) for k in range(half)]
            for k in range(1, half - (1 if n % 2 == 0 else 0)):
                pxx[k] *= 2
            f = [fs * k / n for k in range(half)]
            pdb = [10 * math.log10(p) if p > 0 else -150.0 for p in pxx]
            self._finish_figure()
            self._add_fig({
                "series": [{"x": f, "y": pdb, "kind": "plot", "color": _PALETTE[0],
                            "style": None, "marker": None, "width": 1.4, "msize": 5.0}],
                "title": "Power Spectral Density (Welch/Hann)",
                "xlabel": "Frequency (Hz)", "ylabel": "PSD (dB/Hz)", "grid": True})
            return None
        return None

    def _rlocus(self, sys):
        num = sys["num"]
        den = sys["den"]
        gains = [0.0] + [10.0 ** (-3 + 6 * k / 199) for k in range(200)]
        gains += [g for g in (2, 5, 10, 20, 50, 100, 200, 500, 1000)]
        gains = sorted(set(gains))
        loci = []
        for K in gains:
            poly = _padd(den, [K * c for c in num])
            loci.extend((r.real, r.imag) for r in _roots_dk(poly))
        ol_poles = _roots_dk(den)
        ol_zeros = _roots_dk(num)
        fig = self._ensure_fig()
        fig["series"] = []
        fig["series"].append({"x": [p[0] for p in loci], "y": [p[1] for p in loci],
                              "kind": "scatter", "color": "#4a90d9", "style": None,
                              "marker": ".", "width": 1.0, "msize": 2.0})
        fig["series"].append({"x": [p.real for p in ol_poles], "y": [p.imag for p in ol_poles],
                              "kind": "scatter", "color": "#D95319", "style": None,
                              "marker": "x", "width": 1.3, "msize": 8.0})
        if ol_zeros:
            fig["series"].append({"x": [z.real for z in ol_zeros],
                                  "y": [z.imag for z in ol_zeros], "kind": "scatter",
                                  "color": "#EDB120", "style": None, "marker": "o",
                                  "width": 1.3, "msize": 8.0})
        fig["title"] = fig.get("title") or "Root Locus"
        fig["xlabel"] = "Real axis"
        fig["ylabel"] = "Imaginary axis"
        fig["grid"] = True

    def _nyquist(self, sys):
        num, den = sys["num"], sys["den"]
        crit = [abs(rt) for rt in _roots_dk(num) + _roots_dk(den) if abs(rt) > 1e-9]
        lo = math.floor(math.log10(min(crit))) - 2 if crit else -2
        hi = math.ceil(math.log10(max(crit))) + 2 if crit else 3
        N = 60 * max(3, hi - lo)
        re, im = [], []
        for k in range(N):
            w = 10.0 ** (lo + (hi - lo) * k / (N - 1))
            dv = _peval(den, 1j * w)
            if dv == 0:
                continue
            h = _peval(num, 1j * w) / dv
            re.append(h.real)
            im.append(h.imag)
        fig = self._ensure_fig()
        fig["series"] = []
        fig["series"].append({"x": re, "y": im, "kind": "plot", "color": _PALETTE[0],
                              "style": None, "marker": None, "width": 1.4, "msize": 5.0})
        fig["series"].append({"x": re, "y": [-v for v in im], "kind": "plot",
                              "color": _PALETTE[0], "style": "--", "marker": None,
                              "width": 1.0, "msize": 5.0})
        fig["series"].append({"x": [-1.0], "y": [0.0], "kind": "scatter",
                              "color": "#D95319", "style": None, "marker": "+",
                              "width": 1.3, "msize": 9.0})
        fig["title"] = fig.get("title") or "Nyquist Diagram"
        fig["xlabel"] = "Real axis"
        fig["ylabel"] = "Imaginary axis"
        fig["grid"] = True

    # -- axes decoration ---------------------------------------------------
    def _axes_call(self, name, args):
        ax = self._ensure_fig()
        if name == "colorbar":
            ax["colorbar"] = not (args and isinstance(args[0], str) and args[0].lower() == "off")
            return None
        if name in ("daspect", "pbaspect", "camlight", "lighting", "material", "rotate3d", "box"):
            return None
        if name == "zlabel":
            ax["zlabel"] = scalar_str(args[0]) if args else None
            return None
        if name == "zlim":
            if args:
                v = [float(_cr(x)) for x in as_list(args[0])]
                if len(v) == 2:
                    ax["zlim"] = v
            return None
        if name in ("caxis", "clim"):
            if args:
                v = [float(_cr(x)) for x in as_list(args[0])]
                if len(v) == 2:
                    ax["clim"] = (v[0], v[1])
            return None
        if name == "shading":
            if args and isinstance(args[0], str):
                ax["shading"] = args[0].lower()
            return None
        if name == "view":
            if len(args) >= 2:
                ax["view"] = (float(_cr(scalar(args[0]))), float(_cr(scalar(args[1]))))
            elif args:
                v = [float(_cr(x)) for x in as_list(args[0])]
                if len(v) == 2:
                    ax["view"] = (v[0], v[1])
                elif v and v[0] == 2:
                    ax["view"] = (0.0, 90.0)
                elif v and v[0] == 3:
                    ax["view"] = (-37.5, 30.0)
            return None
        if name == "xlim":
            if args:
                v = [float(_cr(x)) for x in as_list(args[0])]
                if len(v) == 2:
                    ax["xlim"] = v
            return None
        if name == "ylim":
            if args:
                v = [float(_cr(x)) for x in as_list(args[0])]
                if len(v) == 2:
                    ax["ylim"] = v
            return None
        if name == "axis":
            if args and isinstance(args[0], str):
                if args[0].lower() == "equal":
                    ax["equal"] = True
                return None
            if args:
                v = [float(_cr(x)) for x in as_list(args[0])]
                if len(v) >= 4:
                    ax["xlim"] = v[0:2]
                    ax["ylim"] = v[2:4]
                    if len(v) >= 6:
                        ax["zlim"] = v[4:6]
                elif len(v) == 2:
                    ax["xlim"] = v
            return None
        if name in ("xline", "yline"):
            if not args:
                raise MError(f"{name}(value) needs a position")
            colr = "#D95319"
            for a in args[1:]:
                if isinstance(a, str):
                    colr = _to_color(a)
            for v in as_list(args[0]):
                ax["vlines" if name == "xline" else "hlines"].append(
                    (float(_cr(v)), colr))
            return None
        if name == "text":
            if len(args) < 3:
                raise MError("text(x, y, 'label') needs three arguments")
            ax["texts"].append((float(_cr(scalar(args[0]))),
                                float(_cr(scalar(args[1]))),
                                scalar_str(args[2])))
            return None
        return None

    # -- DSP commands that draw --------------------------------------------
    def _series(self, x, y, kind="plot", color=None, style=None, marker=None,
                width=1.4, msize=5.0, name=None):
        s = {"x": list(x), "y": list(y), "kind": kind,
             "color": color or self._next_color(), "style": style,
             "marker": marker, "width": width, "msize": msize}
        if name:
            s["name"] = name
        return s

    def _fresh_axes(self):
        ax = self._ensure_fig()
        if not self.hold:
            ax["series"] = []
            self._color_idx = 0
        return ax

    def _dsp_plot(self, name, args):
        try:
            return self._dsp_plot_inner(name, args)
        except mdsp.DspError as ex:
            raise MError(str(ex))

    def _dsp_plot_inner(self, name, args):
        if name in ("impz", "stepz"):
            h, t = (mdsp.impz(args) if name == "impz" else mdsp.stepz(args))
            ax = self._fresh_axes()
            ax["series"].append(self._series(t, h, "stem"))
            ax["title"] = ax.get("title") or (
                "Impulse Response" if name == "impz" else "Step Response")
            ax["xlabel"] = ax.get("xlabel") or "n (samples)"
            ax["ylabel"] = ax.get("ylabel") or "Amplitude"
            ax["grid"] = True
            return None
        if name == "zplane":
            a0 = args[0]
            if len(args) > 1 and is_vec(args[1]) and len(as_list(args[1])) > 1:
                z, p, _ = mdsp.tf2zp([a0, args[1]])
            elif len(args) > 1:
                z, p = as_list(a0), as_list(args[1])
            else:
                z, p, _ = mdsp.tf2zp([a0, [1.0]])
            ax = self._fresh_axes()
            ax["circle"] = True
            ax["equal"] = True
            if z:
                ax["series"].append(self._series(
                    [complex(v).real for v in z], [complex(v).imag for v in z],
                    "scatter", color="#0072BD", marker="o", msize=8.0))
            if p:
                ax["series"].append(self._series(
                    [complex(v).real for v in p], [complex(v).imag for v in p],
                    "scatter", color="#D95319", marker="x", msize=8.0))
            ax["title"] = ax.get("title") or "Pole-Zero Plot (z-plane)"
            ax["xlabel"] = "Real part"
            ax["ylabel"] = "Imaginary part"
            ax["grid"] = True
            return None
        if name == "freqs":
            b = as_list(args[0])
            a = as_list(args[1]) if len(args) > 1 else [1.0]
            w = as_list(args[2]) if len(args) > 2 else None
            ws, H = mdsp.freqs(b, a, w)
            mag = [20 * math.log10(abs(h)) if abs(h) > 0 else -200.0 for h in H]
            ph = [p * 180 / math.pi for p in _vf_unwrap([[cmath.phase(h) for h in H]])]
            self._finish_figure()
            self._add_fig({"series": [self._series(ws, mag, color=_PALETTE[0])],
                           "title": "Analog Filter — Magnitude",
                           "xlabel": "Frequency (rad/s)", "ylabel": "Magnitude (dB)",
                           "grid": True, "xlog": True})
            self._color_idx = 0
            self._add_fig({"series": [self._series(ws, ph, color=_PALETTE[0])],
                           "title": "Analog Filter — Phase",
                           "xlabel": "Frequency (rad/s)", "ylabel": "Phase (deg)",
                           "grid": True, "xlog": True})
            self._color_idx = 0
            return None
        if name == "grpdelay":
            b, a, ws, fs = _freq_args("grpdelay", args)
            _, gd = mdsp.grpdelay(b, a, ws=ws)
            ax = self._fresh_axes()
            ax["series"].append(self._series(
                _freq_out(ws, fs) if fs else [w / math.pi for w in ws], gd))
            ax["title"] = ax.get("title") or "Group Delay"
            ax["xlabel"] = "Frequency (Hz)" if fs else "Normalized frequency (×π rad/sample)"
            ax["ylabel"] = "Group delay (samples)"
            ax["grid"] = True
            return None
        if name == "spectrogram":
            t, f, P = mdsp.spectrogram(args)
            ax = self._fresh_axes()
            ax["series"] = [{"x": t, "y": f, "z": P, "kind": "heat",
                             "color": "#0072BD", "style": None, "marker": None,
                             "width": 1.0, "msize": 5.0}]
            ax["title"] = ax.get("title") or "Spectrogram (STFT, dB)"
            ax["xlabel"] = ax.get("xlabel") or "Time (s)"
            ax["ylabel"] = ax.get("ylabel") or "Frequency (Hz)"
            return None
        return None

    # -- animation: frames, handles, comet, animatedline -------------------------
    def _snapshot(self, delay_ms=40):
        """Record the current figure as an animation frame (drawnow / pause)."""
        fg = self.curfig
        if fg is None or not fig_has_content(fg):
            return
        fg["tick"] += 1
        if fg["tick"] % fg["stride"]:
            return
        fg["frames"].append(render_svg(fg))
        fg["delays"].append(int(max(10, min(2000, delay_ms))) * fg["stride"])
        if len(fg["frames"]) >= _MAX_FRAMES:
            fg["frames"] = fg["frames"][::2]
            fg["delays"] = [d * 2 for d in fg["delays"][::2]]
            fg["stride"] *= 2

    def _handle(self, ax, series, kind="Line"):
        return {"__gh__": True, "kind": kind, "ax": ax, "s": series}

    def _anim_call(self, name, args):
        if name == "drawnow":
            self._snapshot(40)
            return None
        if name == "pause":
            t = float(_cr(scalar(args[0]))) if args and not isinstance(args[0], str) else 0.5
            self._snapshot(t * 1000.0)
            return None
        if name == "getframe":
            self._snapshot(40)
            return float(len(self.curfig["frames"]) if self.curfig else 0)
        if name in ("movie", "refreshdata", "waitforbuttonpress", "beep"):
            return None
        if name == "gca":
            return {"__gh__": True, "kind": "Axes", "ax": self._ensure_fig(), "s": None}
        if name == "gcf":
            return {"__gh__": True, "kind": "Figure", "ax": None, "s": None,
                    "fig": self._ensure_figure()}
        if name == "delete":
            for h in args:
                if _is_gh(h) and h.get("s") is not None and h["s"] in h["ax"]["series"]:
                    h["ax"]["series"].remove(h["s"])
            return None
        if name == "animatedline":
            ax = self._ensure_fig()
            s = {"x": [], "y": [], "kind": "plot", "color": self._next_color(), "style": None,
                 "marker": None, "width": 1.4, "msize": 5.0}
            i = 0
            while i + 1 < len(args):
                if isinstance(args[i], str) and args[i].lower() in _PLOT_PROPS:
                    self._apply_prop(s, args[i].lower(), args[i + 1])
                    i += 2
                else:
                    i += 1
            if s.get("marker") and not s.get("style") and len(args) and isinstance(args[0], str):
                pass
            ax["series"].append(s)
            return self._handle(ax, s)
        if name == "addpoints":
            _need("addpoints", args, 3)
            h = args[0]
            if not _is_gh(h) or h.get("s") is None:
                raise MError("addpoints: first argument must be an animatedline handle")
            xs = _plot_reals(as_list(args[1]))
            ys = _plot_reals(as_list(args[2]))
            if len(xs) != len(ys):
                raise MError("addpoints: x and y must be the same length")
            h["s"]["x"].extend(xs)
            h["s"]["y"].extend(ys)
            return None
        if name == "clearpoints":
            h = args[0] if args else None
            if _is_gh(h) and h.get("s") is not None:
                h["s"]["x"], h["s"]["y"] = [], []
            return None
        if name in ("comet", "comet3"):
            return self._comet(name, args)
        if name == "set":
            return self._set_props(args)
        if name == "get":
            _need("get", args, 2)
            h, prop = args[0], str(args[1]).lower()
            if not _is_gh(h):
                raise MError("get: first argument must be a graphics handle")
            s_ = h.get("s")
            if s_ is not None:
                if prop == "xdata":
                    return list(s_.get("X", s_.get("x", [])))
                if prop == "ydata":
                    return list(s_.get("Y", s_.get("y", [])))
                if prop == "zdata":
                    return list(s_.get("Z", []))
                if prop == "linewidth":
                    return s_.get("width", 1.0)
            ax = h.get("ax")
            if ax is not None and prop in ("xlim", "ylim", "zlim"):
                return list(ax.get(prop) or [])
            raise MError(f"get: property '{args[1]}' is not available")
        return None

    def _set_props(self, args):
        _need("set", args, 3)
        h = args[0]
        if not _is_gh(h):
            raise MError("set: first argument must be a graphics handle (from plot, surf, gca ...)")
        i = 1
        while i + 1 < len(args):
            prop = str(args[i]).lower()
            val = args[i + 1]
            i += 2
            s_ = h.get("s")
            ax = h.get("ax")
            if s_ is not None:
                if prop in ("xdata", "ydata", "zdata", "cdata"):
                    key3 = {"xdata": "X", "ydata": "Y", "zdata": "Z", "cdata": "C"}[prop]
                    if "X" in s_ and s_.get("kind") in ("surf", "mesh"):
                        M = mmat.to_mat(val)
                        s_[key3] = [[(v.real if isinstance(v, complex) else float(v)) for v in rw]
                                    for rw in M.d]
                    elif "X" in s_:                       # plot3 / scatter3 / stem3
                        s_[key3] = _plot_reals(as_list(val))
                    else:
                        s_[prop[0]] = _plot_reals(as_list(val))
                elif prop == "visible":
                    s_["hidden"] = (str(val).lower() == "off")
                elif prop in _PLOT_PROPS:
                    self._apply_prop(s_, prop, val)
                elif prop in ("string",):
                    pass
                continue
            if ax is not None:
                if prop in ("xlim", "ylim", "zlim"):
                    v = [float(_cr(x)) for x in as_list(val)]
                    if len(v) == 2:
                        ax[prop] = v
                elif prop == "view":
                    v = [float(_cr(x)) for x in as_list(val)]
                    if len(v) == 2:
                        ax["view"] = (v[0], v[1])
                elif prop in ("xgrid", "ygrid", "zgrid"):
                    ax["grid"] = (str(val).lower() == "on")
                elif prop == "colormap":
                    self._set_colormap(val if isinstance(val, str) else "parula")
                # other axes / figure properties are accepted and ignored
        return None

    def _comet(self, name, args):
        if len(args) < 2 or (name == "comet3" and len(args) < 3):
            raise MError(f"{name}(x, y{', z' if name == 'comet3' else ''}) needs the coordinate vectors")
        x = _plot_reals(as_list(args[0]))
        y = _plot_reals(as_list(args[1]))
        z = _plot_reals(as_list(args[2])) if name == "comet3" else None
        n = len(x)
        if len(y) != n or (z is not None and len(z) != n):
            raise MError(f"{name}: coordinate vectors must be the same length")
        ax = self._fresh_axes()
        fin = lambda v: [t for t in v if math.isfinite(t)]  # noqa: E731
        ax["xlim"] = list(_nice_axis(min(fin(x)), max(fin(x)))[:2]) if fin(x) else None
        ax["ylim"] = list(_nice_axis(min(fin(y)), max(fin(y)))[:2]) if fin(y) else None
        if z is not None:
            ax["is3d"] = True
            ax["grid"] = True
            ax["zlim"] = list(_nice_axis(min(fin(z)), max(fin(z)))[:2]) if fin(z) else None
        col = self._next_color()
        steps = max(1, n // 90)
        for k in range(steps, n + 1, steps):
            ax["series"] = [s for s in ax["series"] if not s.get("_comet")]
            if z is None:
                ax["series"].append({"kind": "plot", "x": x[:k], "y": y[:k], "color": col,
                                     "style": None, "marker": None, "width": 1.5, "msize": 5.0,
                                     "_comet": True})
                ax["series"].append({"kind": "scatter", "x": [x[k - 1]], "y": [y[k - 1]],
                                     "color": "#D95319", "style": None, "marker": "o",
                                     "width": 1.5, "msize": 7.0, "_comet": True})
            else:
                ax["series"].append({"kind": "plot3", "X": x[:k], "Y": y[:k], "Z": z[:k],
                                     "color": col, "style": None, "marker": None, "width": 1.5,
                                     "msize": 5.0, "x": [], "y": [], "_comet": True})
                ax["series"].append({"kind": "scatter3", "X": [x[k - 1]], "Y": [y[k - 1]],
                                     "Z": [z[k - 1]], "color": "#D95319", "style": None,
                                     "marker": "o", "width": 1.5, "msize": 7.0, "x": [], "y": [],
                                     "_comet": True})
            self._snapshot(30)
        # final state: the full trajectory without the head
        ax["series"] = [s for s in ax["series"] if not (s.get("_comet") and s["kind"] in ("scatter", "scatter3"))]
        return None

    # -- numerical simulation (msim) -----------------------------------------------
    def _callable(self, f, what):
        if not _is_func(f):
            raise MError(f"{what}: the first argument must be a function handle, e.g. @(t,y) -y")
        interp = self

        def call(*xs):
            r = interp.apply_func(f, list(xs))
            return r
        return call

    def _ode_callable(self, f):
        base = self._callable(f, "ode solver")

        def fun(t, y):
            yv = mmat.colvec(list(y)) if len(y) > 1 else y[0]
            r = base(float(t), yv)
            out = as_list(r)
            if len(out) != len(y):
                raise MError(f"the ODE function must return {len(y)} value(s), got {len(out)}")
            return [float(v.real) if isinstance(v, complex) else float(v) for v in out]
        return fun

    def _scalar_callable(self, f, what):
        base = self._callable(f, what)

        def fun(x):
            r = base(float(x))
            r = scalar(r)
            return float(r.real) if isinstance(r, complex) else float(r)
        return fun

    def _vector_callable(self, f, what):
        base = self._callable(f, what)

        def fun(x):
            r = base(list(x) if len(x) > 1 else x[0])
            out = as_list(r)
            return [float(v.real) if isinstance(v, complex) else float(v) for v in out]
        return fun

    def _sim_call(self, name, args, nargout=1):
        try:
            return self._sim_call_inner(name, args, nargout)
        except msim.SimError as ex:
            raise MError(str(ex))
        except mmat.MatError as ex:
            raise MError(str(ex))

    def _sim_call_inner(self, name, args, nargout):
        def one(v):
            # v is the full output list; a single-output call gets the first value
            return v[0] if nargout == 1 else list(v)

        if name in msim.ODE_METHODS:
            if nargout < 2:
                raise MError(f"{name} returns two outputs — write  [t, y] = {name}(@f, tspan, y0)")
            _need(name, args, 3)
            fun = self._ode_callable(args[0])
            tspan = [float(_cr(v)) for v in as_list(args[1])]
            y0 = [float(_cr(v)) for v in as_list(args[2])]
            opts = args[3] if len(args) > 3 and isinstance(args[3], dict) else {}
            ts, ys = msim.ODE_METHODS[name](
                fun, tspan, y0,
                rtol=float(opts.get("RelTol", 1e-3)), atol=float(opts.get("AbsTol", 1e-6)),
                max_step=(float(opts["MaxStep"]) if "MaxStep" in opts else None))
            T = mmat.colvec(ts)
            Y = mmat.norm_val(Mat(ys)) if len(y0) > 1 else mmat.colvec([r[0] for r in ys])
            return [T, Y]
        if name == "odeset":
            opts = {"__odeopts__": True}
            i = 0
            while i + 1 < len(args):
                if isinstance(args[i], str):
                    key = args[i]
                    val = args[i + 1]
                    opts[{"reltol": "RelTol", "abstol": "AbsTol", "maxstep": "MaxStep"}.get(
                        key.lower(), key)] = float(_cr(scalar(val))) if not isinstance(val, str) else val
                i += 2
            return one([opts])
        if name == "fzero":
            _need("fzero", args, 2)
            f = self._scalar_callable(args[0], "fzero")
            x0 = args[1]
            x0 = [float(_cr(v)) for v in as_list(x0)] if (is_vec(x0) or is_mat(x0)) else float(_cr(x0))
            x = msim.fzero(f, x0)
            return x if nargout == 1 else [x, f(x), 1.0]
        if name == "fsolve":
            _need("fsolve", args, 2)
            g = self._vector_callable(args[0], "fsolve")
            x0 = [float(_cr(v)) for v in as_list(args[1])]
            x, fx, flag = msim.fsolve(g, x0)
            xr = x if len(x) > 1 else x[0]
            if is_mat(args[1]) and args[1].c == 1 and len(x) > 1:
                xr = mmat.colvec(x)
            return xr if nargout == 1 else [xr, (fx if len(fx) > 1 else fx[0]), float(flag)]
        if name in ("fminsearch", "fminunc"):
            _need(name, args, 2)
            g = self._vector_callable(args[0], name)
            x0 = [float(_cr(v)) for v in as_list(args[1])]
            x, fx = msim.fminsearch(lambda v: g(v)[0], x0)
            xr = x if len(x) > 1 else x[0]
            return xr if nargout == 1 else [xr, fx, 1.0]
        if name == "fminbnd":
            _need("fminbnd", args, 3)
            f = self._scalar_callable(args[0], "fminbnd")
            x, fx = msim.fminbnd(f, float(_cr(scalar(args[1]))), float(_cr(scalar(args[2]))))
            return x if nargout == 1 else [x, fx, 1.0]
        if name in ("integral", "quad", "quadgk", "quadl", "quadv"):
            _need(name, args, 3)
            f = self._scalar_callable(args[0], name)
            a = float(_cr(scalar(args[1])))
            b = float(_cr(scalar(args[2])))
            return one([msim.integral(f, a, b)])
        if name == "expm":
            _need("expm", args, 1)
            return one([msim.expm(args[0])])
        if name == "cumtrapz":
            _need("cumtrapz", args, 1)
            if len(args) == 1:
                return one([msim.cumtrapz(_plot_reals(as_list(args[0])))])
            return one([msim.cumtrapz(_plot_reals(as_list(args[0])), _plot_reals(as_list(args[1])))])
        if name == "gradient":
            _need("gradient", args, 1)
            y = _plot_reals(as_list(args[0]))
            if len(args) > 1 and (is_vec(args[1]) or is_mat(args[1])):
                return one([msim.gradient(y, x=_plot_reals(as_list(args[1])))])
            h = float(_cr(scalar(args[1]))) if len(args) > 1 else 1.0
            return one([msim.gradient(y, h)])
        if name == "polyder":
            return one([msim.polyder(_plot_reals(as_list(args[0])))])
        if name == "polyint":
            k = float(_cr(scalar(args[1]))) if len(args) > 1 else 0.0
            return one([msim.polyint(_plot_reals(as_list(args[0])), k)])
        raise MError(f"unknown function '{name}'")

    # -- circuits (mcircuit) ---------------------------------------------------
    def _ckt(self):
        if self.ckt is None:
            self.ckt = mcircuit.Circuit("Circuit")
        return self.ckt

    def _ckt_call(self, name, args, nargout=1):
        try:
            return self._ckt_call_inner(name, args, nargout)
        except mcircuit.CktError as ex:
            raise MError(str(ex))

    def _ckt_call_inner(self, name, args, nargout):
        def S(k):                       # string argument (node name / ref)
            if k >= len(args):
                raise MError(f"{name}: missing argument {k + 1}")
            v = args[k]
            if isinstance(v, str):
                return v
            v = _cr(scalar(v))
            return str(int(v)) if float(v).is_integer() else str(v)

        def F(k, dflt=None):
            if k >= len(args) or args[k] is None:
                if dflt is None:
                    raise MError(f"{name}: missing argument {k + 1}")
                return dflt
            v = args[k]
            if isinstance(v, str):
                return mcircuit.parse_val(v)
            return float(_cr(scalar(v)))

        c = self._ckt()
        if name == "circuit":
            self.ckt = mcircuit.Circuit(args[0] if args and isinstance(args[0], str) else "Circuit")
            return None
        if name == "netlist":
            if not args or not isinstance(args[0], str):
                raise MError("netlist('...SPICE text...')")
            self.ckt = mcircuit.Circuit("netlist")
            self.ckt.load_netlist(args[0])
            return None
        if name in ("resistor", "capacitor", "inductor", "lamp"):
            kind = {"resistor": "R", "capacitor": "C", "inductor": "L", "lamp": "R"}[name]
            c.rlc(kind, S(0), S(1), S(2), F(3), lamp=(name == "lamp"), pwr=(F(4, -1.0) if F(4, -1.0) > 0 else None))
            if F(4, -1.0) > 0:
                c.hidden.append(f"P({S(0)})")        # rated part: watch its dissipation
            return None
        if name in ("voltage", "current"):
            kind = "V" if name == "voltage" else "I"
            spec = args[3] if len(args) > 3 else 0.0
            params = None
            if isinstance(spec, str):
                params = [float(_cr(v)) for v in as_list(args[4])] if len(args) > 4 else []
            else:
                spec = float(_cr(scalar(spec)))
            c.source(kind, S(0), S(1), S(2), spec, params)
            return None
        if name == "diode":                      # diode(ref,a,k) | diode(ref,a,k,'Ge') | diode(ref,a,k,IS,N)
            if len(args) > 3 and isinstance(args[3], str):
                c.diode(S(0), S(1), S(2), material=args[3])
            else:
                c.diode(S(0), S(1), S(2), IS=F(3, mcircuit.IS), N=F(4, 1.0))
            return None
        if name == "zener":
            c.zener(S(0), S(1), S(2), F(3, 5.1), F(4, 5e-3))
            return None
        if name == "led":                        # led(ref,a,k,'blue') | led(ref,a,k,'blue',3.0)
            color = args[3] if len(args) > 3 and isinstance(args[3], str) else "red"
            vf = F(4, -1.0) if len(args) > 4 else (F(3, -1.0) if len(args) > 3 and not isinstance(args[3], str) else -1.0)
            c.led(S(0), S(1), S(2), color, vf if vf > 0 else None)
            return None
        if name == "bjt":
            c.bjt(S(0), S(1), S(2), S(3), args[4] if len(args) > 4 else "NPN", F(5, 100.0), F(6, -1.0) if len(args) > 6 else None)
            return None
        if name == "jfet":
            c.jfet(S(0), S(1), S(2), S(3), args[4] if len(args) > 4 else "NJF", F(5, 3e-3), F(6, -2.0))
            return None
        if name == "comparator":
            c.comparator(S(0), S(1), S(2), S(3), F(4, 5.0), F(5, 0.0))
            return None
        if name in ("voltmeter", "ammeter", "ohmmeter"):
            if name == "voltmeter":
                c.hidden.extend([f"V({S(1)})", f"V({S(2)})"])
            elif name == "ammeter":
                c.source("V", "V" + S(0), S(1), S(2), 0.0, None)
                c.els[-1]["meter"] = True
                c.hidden.append(f"I(V{S(0)})")
            c.instruments.append((name, S(0), None, S(1), S(2)))
            return None
        if name == "mosfet":
            c.mosfet(S(0), S(1), S(2), S(3), args[4] if len(args) > 4 else "NMOS", F(5, 1.0), F(6, 1e-3))
            return None
        if name == "opamp":                      # opamp(ref,out,in+,in-[,gain,rails])
            c.opamp(S(0), S(1), S(2), S(3), F(4, 1e6), F(5, 0.0))
            return None
        if name == "vcvs":
            c.vcvs(S(0), S(1), S(2), S(3), S(4), F(5, 100.0))
            return None
        if name == "vccs":
            c.vccs(S(0), S(1), S(2), S(3), S(4), F(5, 1e-3))
            return None
        if name == "sw":
            c.switch(S(0), S(1), S(2), args[3] if len(args) > 3 else "on")
            return None
        if name == "pot":
            c.pot(S(0), S(1), S(2), S(3), F(4, 1e4), F(5, 0.5))
            return None
        if name == "transformer":
            c.transformer(S(0), S(1), S(2), S(3), S(4), F(5, 1.0))
            return None
        # instruments become probes / helper sources, like in the Circuit Lab
        if name == "multimeter":
            mode = (args[3] if len(args) > 3 and isinstance(args[3], str) else "V").upper()
            if mode == "A":
                c.source("V", "V" + S(0), S(1), S(2), 0.0, None)
                c.els[-1]["meter"] = True
                c.probes.append(f"I(V{S(0)})")
            else:
                c.probes.extend([f"V({S(1)})", f"V({S(2)})"])
            c.instruments.append(("multimeter", S(0), mode, S(1), S(2)))
            return None
        if name == "oscilloscope":
            c.probes.append(f"V({S(1)})")
            if len(args) > 2 and S(2) != "0":
                c.probes.append(f"V({S(2)})")
            return None
        if name == "fgen":
            kind = (args[3] if len(args) > 3 and isinstance(args[3], str) else "SIN").upper()
            c.source("V", "V" + S(0), S(1), S(2), kind, [F(6, 0.0), F(4, 1.0), F(5, 1e3)])
            return None
        if name == "bodeplotter":
            c.analysis.update(type="ac", f0=F(3, 10.0), f1=F(4, 1e6))
            c.probes.extend([f"V({S(2)})", f"V({S(1)})"])
            return None
        if name == "wattmeter":
            c.source("V", "V" + S(0), S(3), S(4), 0.0, None)
            c.els[-1]["meter"] = True
            c.probes.extend([f"V({S(1)})", f"V({S(2)})", f"I(V{S(0)})"])
            return None
        # analyses
        if name == "transient":
            c.analysis.update(type="tran", tstep=F(0), tstop=F(1))
            return None
        if name == "acsweep":
            c.analysis.update(type="ac", f0=F(0), f1=F(1), nppd=int(F(2, 40)))
            return None
        if name == "dcop":
            c.analysis["type"] = "op"
            return None
        if name == "dcsweep":
            c.analysis.update(type="dcsweep", src=S(0), start=F(1), stop=F(2), inc=F(3))
            return None
        if name == "stepparam":
            c.step = {"name": S(0), "start": F(1), "stop": F(2), "n": max(2, min(12, int(F(3, 5))))}
            return None
        if name == "fourier":
            c.four = {"f0": F(0), "node": S(1)}
            return None
        if name == "probe":
            c.probes.extend(S(k) for k in range(len(args)))
            return None
        if name == "simulate":
            return self._ckt_simulate(args, nargout)
        raise MError(f"unknown function '{name}'")

    def _ckt_simulate(self, args, nargout):
        if self.ckt is None or not self.ckt.els:
            raise MError("simulate: build a circuit first (resistor(...), voltage(...), ...)")
        noplot = any(isinstance(a, str) and a.lower() == "noplot" for a in args)
        res = mcircuit.run(self.ckt)
        base = res["runs"][0]["result"] if res["type"] == "step" else res
        runs = [(r["result"], r["label"]) for r in res["runs"]] if res["type"] == "step" else [(res, None)]
        typ = base["type"]
        if typ == "op":
            lines = ["", "DC operating point", ""]
            for n, v in base["nodeV"].items():
                lines.append(f"   V({n}) = {_fmt_rd(v)} V")
            for n, i in base["elI"].items():
                if n.startswith("__"):
                    continue
                lines.append(f"   I({n}) = {_fmt_rd(i)} A    P = {_fmt_rd(base['elP'][n])} W")
            self.out.append("\n".join(lines) + "\n\n")
            self._ckt_readouts(base)
            vcol = mmat.colvec(list(base["nodeV"].values()))
            icol = mmat.colvec(list(base["elI"].values()))
            return vcol if nargout == 1 else [vcol, icol]
        if typ == "ac":
            f = base["freqs"]
            if not noplot:
                self._finish_figure()
                for key, title, ylabel in (("mag", "Bode magnitude", "Magnitude (dB)"), ("phase", "Bode phase", "Phase (deg)")):
                    series = []
                    for r, lab in runs:
                        for n in r["probeNames"]:
                            series.append(self._series(f, r[key][n], name=(n + (" " + lab if lab else ""))))
                    self._add_fig({"series": series, "title": f"{self.ckt.title} — {title}", "xlabel": "Frequency (Hz)",
                                   "ylabel": ylabel, "grid": True, "xlog": True,
                                   "legend": [s_["name"] for s_ in series]})
                    self._color_idx = 0
            names = base["probeNames"]
            mag = mmat.norm_val(Mat([[base["mag"][n][k] for n in names] for k in range(len(f))]))
            ph = mmat.norm_val(Mat([[base["phase"][n][k] for n in names] for k in range(len(f))]))
            return mmat.colvec(f) if nargout == 1 else [mmat.colvec(f), mag, ph][:nargout]
        xs = base["time"] if typ == "tran" else base["sweep"]
        if not noplot:
            ax = self._fresh_axes()
            for r, lab in runs:
                x_ = r["time"] if typ == "tran" else r["sweep"]
                for n in r["probeNames"]:
                    ax["series"].append(self._series(x_, r["traces"][n], name=(n + (" " + lab if lab else ""))))
            ax["legend"] = [s_["name"] for s_ in ax["series"]]
            ax["title"] = ax.get("title") or self.ckt.title
            ax["xlabel"] = "Time (s)" if typ == "tran" else base.get("xlabel", "sweep")
            ax["ylabel"] = "Probe value"
            ax["grid"] = True
        if base.get("fourier"):
            fo = base["fourier"]
            self.out.append(f"\nFourier analysis of {fo['probe']} at {fo['f0']:g} Hz:  THD = {fo['thd'] * 100:.2f} %\n")
            for h in fo["harmonics"]:
                self.out.append(f"   h{h['n']}  {h['f']:>10g} Hz   {_fmt_rd(h['mag'])} V   {h['phase']:8.1f} deg\n")
            self.out.append("\n")
        self._ckt_readouts(base)
        if base.get("unconverged"):
            self.out.append(f"warning: {base['unconverged']} time step(s) did not fully converge\n")
        cols = [r["traces"][n] for r, _lab in runs for n in r["probeNames"]]   # every run, every probe
        V = mmat.norm_val(Mat([[c[k] for c in cols] for k in range(len(xs))]))
        return mmat.colvec(xs) if nargout == 1 else [mmat.colvec(xs), V]

    def _ckt_readouts(self, base):
        """Print what the placed testers / meters read (like the Circuit Lab panels)."""
        inst = self.ckt.instruments if self.ckt else []
        if not inst and not self.ckt.hidden:
            return
        traces = base.get("traces", {})
        nodeV = base.get("nodeV", {})

        def vnode(n):
            return 0.0 if n == "0" else nodeV.get(n, 0.0)

        def vtrace(n):
            return traces.get(f"V({n})") or traces.get(n)
        lines = ["", "Instrument readings"]
        for e in self.ckt.els:                      # overload flags for rated resistors / lamps
            key = f"P({e.get('name')})"
            if key in traces and e.get("pwr"):
                pk = max(abs(v) for v in traces[key])
                if pk > e["pwr"]:
                    lines.append(f"   warning: {e['name']} dissipates {_fmt_rd(pk)} W peak, rated {_fmt_rd(e['pwr'])} W")
        for kind, name, mode, a, b in inst:
            if kind in ("voltmeter", "multimeter") and (mode in (None, "V")):
                ta, tb = vtrace(a), vtrace(b)
                if ta is not None:
                    d = [x - (tb[i] if tb else 0.0) for i, x in enumerate(ta)]
                    m = mcircuit.measure(base.get("time") or base.get("sweep"), d)
                    lines.append(f"   {name}  V({a},{b}) = {_fmt_rd(m['avg'])} V dc   {_fmt_rd(m['acrms'])} V ac rms")
                else:
                    lines.append(f"   {name}  V({a},{b}) = {_fmt_rd(vnode(a) - vnode(b))} V")
            elif kind == "ammeter" or (kind == "multimeter" and mode == "A"):
                key = f"I(V{name})"
                if key in traces:
                    m = mcircuit.measure(base.get("time") or base.get("sweep"), traces[key])
                    lines.append(f"   {name}  I = {_fmt_rd(m['avg'])} A dc   {_fmt_rd(m['acrms'])} A ac rms")
                else:
                    lines.append(f"   {name}  I = {_fmt_rd(base.get('elI', {}).get('V' + name, 0.0))} A")
            elif kind == "ohmmeter" or (kind == "multimeter" and mode == "Ω"):
                try:
                    R = mcircuit.measure_resistance(self.ckt, a, b)
                    lines.append(f"   {name}  R({a},{b}) = {_fmt_rd(R)} ohm")
                except mcircuit.CktError as ex:
                    lines.append(f"   {name}  R({a},{b}) = ? ({ex})")
        self.out.append("\n".join(lines) + "\n\n")

    # -- 3-D graphics --------------------------------------------------------
    def _set_colormap(self, name):
        name = (name or "parula").lower()
        if name not in mplot3.COLORMAPS:
            raise MError(f"unknown colormap '{name}' (try parula, jet, hot, cool, gray, hsv, "
                         f"turbo, viridis, autumn, winter, spring, summer, bone, copper)")
        fg = self._ensure_figure()
        fg["cmap"] = name
        for ax in fg["axes"]:
            ax["cmap"] = name

    def _grid_args(self, nums):
        """(Z) | (X, Y, Z) | (X, Y, Z, C) -> rows-of-lists X, Y, Z, C."""
        if not nums:
            raise MError("surface plots need Z or X, Y, Z")
        if len(nums) == 1 or len(nums) == 2:
            Z = mmat.to_mat(nums[0])
            xs = [float(j + 1) for j in range(Z.c)]
            ys = [float(i + 1) for i in range(Z.r)]
            X, Y = mmat.meshgrid(xs, ys)
            C = nums[1] if len(nums) == 2 else None
        else:
            X, Y, Z = nums[0], nums[1], nums[2]
            C = nums[3] if len(nums) > 3 else None
            Zm = mmat.to_mat(Z)
            if not is_mat(X) or X.r == 1 or (is_mat(X) and X.c == 1):
                xs, ys = as_list(X), as_list(Y)
                if len(xs) != Zm.c or len(ys) != Zm.r:
                    raise MError(f"surf: X ({len(xs)}) and Y ({len(ys)}) must match Z ({Zm.r}x{Zm.c})")
                X, Y = mmat.meshgrid(xs, ys)
        Xm, Ym, Zm = mmat.to_mat(X), mmat.to_mat(Y), mmat.to_mat(Z)
        if not (Xm.r == Ym.r == Zm.r and Xm.c == Ym.c == Zm.c):
            raise MError("X, Y and Z must be the same size")
        Cm = mmat.to_mat(C).d if C is not None else None

        def real_rows(M):
            return [[(v.real if isinstance(v, complex) else float(v)) for v in rw] for rw in M.d]
        return real_rows(Xm), real_rows(Ym), real_rows(Zm), \
            ([[(v.real if isinstance(v, complex) else float(v)) for v in rw] for rw in Cm]
             if Cm is not None else None)

    def _plot3_call(self, name, arg_nodes):
        vals = [("str", a[1]) if a[0] == "str" else ("num", self.ev(a)) for a in arg_nodes]
        nums = [v for k, v in vals if k == "num"]
        strs = [v for k, v in vals if k == "str"]
        ax = self._fresh_axes()
        fg = self._ensure_figure()
        if fg.get("cmap") and not ax.get("cmap"):
            ax["cmap"] = fg["cmap"]
        try:
            if name in ("surf", "surfc", "mesh", "meshc", "surfl", "waterfall"):
                X, Y, Z, C = self._grid_args(nums)
                kind = "mesh" if name in ("mesh", "meshc", "waterfall") else "surf"
                ax["series"].append({"kind": kind, "X": X, "Y": Y, "Z": Z, "C": C,
                                     "color": "#0072BD", "style": None, "marker": None,
                                     "width": 1.0, "msize": 5.0, "x": [], "y": []})
                ax["is3d"] = True
                if name in ("surfc", "meshc"):
                    self._add_floor_contours(ax, X, Y, Z)
                ax["grid"] = True
                return self._handle(ax, ax["series"][-1], "Surface")
            if name in ("plot3", "scatter3", "stem3"):
                if len(nums) < 3:
                    raise MError(f"{name}(x, y, z) needs three vectors")
                x, y, z = (_plot_reals(as_list(v)) for v in nums[:3])
                if not (len(x) == len(y) == len(z)):
                    raise MError(f"{name}: x, y and z must be the same length")
                spec = strs[0] if strs and strs[0].lower() not in _PLOT_PROPS else ""
                color, style, marker = parse_spec(spec)
                kind = {"plot3": "plot3", "scatter3": "scatter3", "stem3": "stem3"}[name]
                ax["series"].append({"kind": kind, "X": x, "Y": y, "Z": z,
                                     "color": color or self._next_color(), "style": style,
                                     "marker": marker, "width": 1.4, "msize": 5.0,
                                     "x": [], "y": []})
                ax["is3d"] = True
                ax["grid"] = True
                return self._handle(ax, ax["series"][-1])
            if name in ("contour", "contourf", "contour3"):
                lv = None
                if len(nums) in (2, 4):
                    lv = nums[-1]
                    nums = nums[:-1]
                X, Y, Z, _ = self._grid_args(nums)
                zf = [v for rw in Z for v in rw if math.isfinite(v)]
                zmin, zmax = (min(zf), max(zf)) if zf else (0.0, 1.0)
                if lv is None:
                    levels = mmat.nice_levels(zmin, zmax, 8)
                elif not is_vec(lv) and not is_mat(lv):
                    levels = mmat.nice_levels(zmin, zmax, max(1, int(_cr(lv))))
                else:
                    levels = [float(_cr(v)) for v in as_list(lv)]
                cmap = ax.get("cmap") or "parula"
                ax["clim"] = ax.get("clim") or (zmin, zmax)
                if name == "contourf":
                    xs = X[0]
                    ys = [rw[0] for rw in Y]
                    ax["series"].append({"kind": "heat", "x": xs, "y": ys,
                                         "z": [[Z[i][j] for i in range(len(ys))] for j in range(len(xs))],
                                         "zmin": zmin, "zmax": zmax, "cmap": cmap,
                                         "color": "#000", "style": None, "marker": None,
                                         "width": 1.0, "msize": 5.0})
                for lev, lines in mmat.contour_lines(X, Y, Z, levels):
                    t = (lev - zmin) / (zmax - zmin) if zmax > zmin else 0.5
                    col = "#000000" if name == "contourf" else mplot3.cmap_color(cmap, t)
                    for line in lines:
                        if name == "contour3":
                            ax["series"].append({"kind": "contour3", "X": [q[0] for q in line],
                                                 "Y": [q[1] for q in line], "Z": [lev] * len(line),
                                                 "color": col, "style": None, "marker": None,
                                                 "width": 1.3, "msize": 5.0, "x": [], "y": []})
                        else:
                            ax["series"].append({"kind": "plot", "x": [q[0] for q in line],
                                                 "y": [q[1] for q in line], "color": col,
                                                 "style": None, "marker": None, "width": 1.3,
                                                 "msize": 5.0, "level": lev})
                if name == "contour3":
                    ax["is3d"] = True
                ax["grid"] = ax.get("grid", False)
                return None
            if name in ("pcolor", "imagesc", "image"):
                X, Y, Z, _ = self._grid_args(nums if name == "pcolor" else nums[-1:])
                xs = X[0]
                ys = [rw[0] for rw in Y]
                zf = [v for rw in Z for v in rw if math.isfinite(v)]
                zmin, zmax = (min(zf), max(zf)) if zf else (0.0, 1.0)
                if len(nums) == 3 and name != "pcolor":         # imagesc(x, y, C)
                    xs = as_list(nums[0]) if len(as_list(nums[0])) == len(xs) else xs
                    ys = as_list(nums[1]) if len(as_list(nums[1])) == len(ys) else ys
                ax["series"].append({"kind": "heat", "x": xs, "y": ys,
                                     "z": [[Z[i][j] for i in range(len(ys))] for j in range(len(xs))],
                                     "zmin": zmin, "zmax": zmax, "cmap": ax.get("cmap") or "parula",
                                     "color": "#000", "style": None, "marker": None,
                                     "width": 1.0, "msize": 5.0})
                ax["clim"] = ax.get("clim") or (zmin, zmax)
                if name in ("imagesc", "image"):
                    ax["yreverse"] = True
                return None
        except mmat.MatError as ex:
            raise MError(str(ex))
        return None

    def _add_floor_contours(self, ax, X, Y, Z):
        zf = [v for rw in Z for v in rw if math.isfinite(v)]
        if not zf:
            return
        zmin, zmax = min(zf), max(zf)
        cmap = ax.get("cmap") or "parula"
        for lev, lines in mmat.contour_lines(X, Y, Z, mmat.nice_levels(zmin, zmax, 8)):
            t = (lev - zmin) / (zmax - zmin) if zmax > zmin else 0.5
            for line in lines:
                ax["series"].append({"kind": "floor_contour", "X": [q[0] for q in line],
                                     "Y": [q[1] for q in line], "Z": [zmin] * len(line),
                                     "color": mplot3.cmap_color(cmap, t), "style": None,
                                     "marker": None, "width": 1.2, "msize": 5.0, "x": [], "y": []})

    # -- printing ---------------------------------------------------------
    def _legend(self, args):
        """legend('a','b'), legend({'a','b'}), legend(..., 'Location', 'best'),
        legend off / show / boxoff."""
        names, i = [], 0
        while i < len(args):
            v = args[i]
            if is_cell(v):
                names.extend(str(scalar_str(x)) for x in v.flat())
            elif isinstance(v, str) and v.lower() in _LEGEND_PROPS:
                i += 1                         # name/value pair: skip the value
            elif isinstance(v, str) and v.lower() in ("off", "hide"):
                self._ensure_fig().pop("legend", None)
                return None
            elif isinstance(v, str) and v.lower() in ("show", "boxon", "boxoff", "toggle"):
                pass
            else:
                names.append(str(scalar_str(v)))
            i += 1
        ax = self._ensure_fig()
        if names:
            ax["legend"] = names
        elif not ax.get("legend"):
            ax["legend"] = [f"data{k + 1}" for k in range(len(ax.get("series", [])))]
        return None

    def display(self, name, val):
        if is_cell(val):
            self.out.append(f"\n{name} =\n\n{mtypes.format_cell(val, _fmt_num)}\n\n")
            return
        if is_struct(val):
            self.out.append(f"\n{name} =\n\n{mtypes.format_struct(val, _fmt_num)}\n\n")
            return
        if _is_tf(val):
            self.out.append(_tf_display(name, val))
            return
        if _is_gh(val):
            self.out.append(f"\n{name} =\n\n  {val.get('kind', 'Line')} with properties\n\n")
            return
        if isinstance(val, dict) and val.get("__odeopts__"):
            body = "\n".join(f"    {k}: {_fmt_num(v)}" for k, v in val.items() if k != "__odeopts__")
            self.out.append(f"\n{name} =\n\n{body or '    (defaults)'}\n\n")
            return
        if is_mat(val):
            self.out.append(f"\n{name} =\n\n{mmat.fmt_mat(val, _arr_fmt(as_list(val)))}\n\n")
            return
        if _is_func(val):
            kind = "@" + val[1] if val[0] == "fhandle" else "@(" + ", ".join(val[1]) + ") …"
            self.out.append(f"\n{name} =\n\n    function_handle: {kind}\n\n")
            return
        if isinstance(val, str):
            self.out.append(f"\n{name} =\n\n    '{val}'\n\n")
        elif is_vec(val):
            self.out.append(f"\n{name} =\n\n{_fmt_vec(val)}\n\n")
        else:
            self.out.append(f"\n{name} =\n\n   {_fmt_num(val)}\n\n")

    def _fmt_value(self, val):
        if is_cell(val):
            return mtypes.format_cell(val, _fmt_num)
        if is_struct(val):
            return mtypes.format_struct(val, _fmt_num)
        if _is_tf(val):
            return _tf_display("", val).replace("\n =\n", "").strip("\n")
        if is_mat(val):
            return mmat.fmt_mat(val, _arr_fmt(as_list(val)))
        if _is_func(val):
            return "    function_handle"
        if isinstance(val, str):
            return val
        if is_vec(val):
            return _fmt_vec(val)
        return f"   {_fmt_num(val)}"

    def _do_fprintf(self, arg_nodes):
        if not arg_nodes:
            return ""
        fmt = scalar_str(self.ev(arg_nodes[0]))
        rest = []
        for a in arg_nodes[1:]:
            rest.extend(as_list(self.ev(a)))
        return _c_format(fmt, rest)

    def _sprintf(self, arg_nodes):
        fmt = scalar_str(self.ev(arg_nodes[0]))
        rest = []
        for a in arg_nodes[1:]:
            rest.extend(as_list(self.ev(a)))
        return _c_format(fmt, rest)


# ---------------------------------------------------------------------------
# Helpers: numbers, formatting, statements
# ---------------------------------------------------------------------------
def _safe_div(x, y):
    if isinstance(x, complex) or isinstance(y, complex):
        try:
            return x / y
        except ZeroDivisionError:
            return complex(math.nan, math.nan)
    if y == 0:
        return math.inf if x > 0 else (-math.inf if x < 0 else math.nan)
    return x / y


def _safe_pow(x, y):
    try:
        if isinstance(x, complex) or isinstance(y, complex):
            return x ** y
        if x < 0 and not float(y).is_integer():
            return complex(x) ** y  # MATLAB: (-8)^(1/3) is complex
        return float(x) ** float(y)
    except (ValueError, OverflowError, ZeroDivisionError):
        return math.nan


# A list of 10M floats is ~320 MB in Python. Past that a typo like 1:1e10
# would exhaust memory and kill the engine, so it is an error instead.
_MAX_RANGE = 10_000_000


def _range_bounds(parts):
    """(start, step, count) of a:c or a:b:c."""
    if len(parts) == 2:
        a, c = parts
        b = 1.0
    elif len(parts) == 3:
        a, b, c = parts
    else:
        raise MError("range needs 2 or 3 parts")
    if b == 0 or any(isinstance(x, float) and math.isnan(x) for x in (a, b, c)):
        return a, b, 0
    if math.isinf(a) or math.isinf(c) or math.isinf(b):
        raise MError("range bounds must be finite")
    return a, b, max(0, int(math.floor((c - a) / b + 1e-9)) + 1)


def make_range(parts):
    a, b, n = _range_bounds(parts)
    if n > _MAX_RANGE:
        raise MError(f"range a:b would have {n:,} elements (limit {_MAX_RANGE:,})")
    return [a + k * b for k in range(n)]


def iter_range(parts):
    """Lazy a:b:c for `for` loops: MATLAB never builds the vector there, so
    `for k = 1:1e9` must start at once (and Stop must reach the first tick)."""
    a, b, n = _range_bounds(parts)
    return (a + k * b for k in range(n))


def _fmt_num(x, small=True):
    if isinstance(x, complex):
        if x.imag == 0:
            return _fmt_num(x.real, small)
        sign = "-" if x.imag < 0 else "+"
        return f"{_fmt_num(x.real, small)} {sign} {_fmt_num(abs(x.imag), small)}i"
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return "NaN" if math.isnan(x) else ("Inf" if x > 0 else "-Inf")
    if small and x != 0 and abs(x) < 1e-3:   # format short: eps -> 2.2204e-16
        return f"{x:.4e}"
    if abs(x - round(x)) < 1e-12 and abs(x) < 1e15:
        return str(int(round(x)))
    if abs(x) >= 1e5:
        return f"{x:.4e}"
    return f"{x:.4f}"


def _num2str(x):
    """num2str(x): MATLAB's '%.<k>g' with k = max(floor(log10|x|), 0) + 5."""
    if isinstance(x, complex) or not math.isfinite(x) or x == round(x):
        return _fmt_num(x)
    k = max(int(math.floor(math.log10(abs(x)))), 0) + 5
    return f"{x:.{k}g}"


def _fmt_rd(x):
    """Instrument readouts stay fixed-point (no exponent form for tiny values)."""
    return _fmt_num(x, small=False)


def _arr_fmt(values):
    """Element formatter for an array. Rounding noise next to ordinary values
    prints as 0 (MATLAB's shared scale); exponent form only when every
    nonzero entry is tiny."""
    mags = [abs(v) for v in values
            if isinstance(v, (int, float, complex)) and v == v and v != 0 and abs(v) != math.inf]
    if mags and max(mags) < 1e-3:
        return _fmt_num
    return lambda x: _fmt_num(x, small=False)


def _fmt_vec(vec):
    f = _arr_fmt(vec)
    if len(vec) > 24:
        head = "   ".join(f(v) for v in vec[:12])
        return f"   {head}   ...  (1x{len(vec)} vector)"
    return "   " + "   ".join(f(v) for v in vec)


def scalar_str(v):
    return v if isinstance(v, str) else _fmt_num(scalar(v))


def _c_format(fmt, args):
    fmt = fmt.replace("\\n", "\n").replace("\\t", "\t")
    specs = list(re.finditer(r"%[-+ 0-9.]*[dieEfgGsc%]", fmt))
    if not specs:
        return fmt
    out = []
    ai = 0

    def one_pass():
        nonlocal ai
        res = []
        last = 0
        for m in specs:
            res.append(fmt[last:m.start()])
            last = m.end()
            sp = m.group()
            if sp == "%%":
                res.append("%")
                continue
            val = args[ai] if ai < len(args) else 0
            ai += 1
            if isinstance(val, complex):
                val = val.real
            try:
                if sp[-1] in "dieEfgG" and isinstance(val, float) and not math.isfinite(val):
                    # MATLAB prints NaN / Inf / -Inf, keeping the field width
                    word = "NaN" if math.isnan(val) else ("Inf" if val > 0 else "-Inf")
                    res.append((re.sub(r"\.\d*", "", sp[:-1]) + "s") % word)
                elif sp[-1] in "dis":
                    if sp[-1] == "d" or sp[-1] == "i":
                        res.append(sp.replace("i", "d") % int(round(val)) if not isinstance(val, str) else str(val))
                    elif sp[-1] == "s":
                        res.append(sp % (val if isinstance(val, str) else _fmt_num(val)))
                else:
                    res.append(sp % float(val))
            except (TypeError, ValueError):
                res.append(str(val))
        res.append(fmt[last:])
        return "".join(res)

    # MATLAB recycles the format string until args are consumed.
    if not args:
        return one_pass()
    while ai < len(args):
        out.append(one_pass())
        if ai == 0:  # format consumed no args; avoid infinite loop
            break
    return "".join(out)


def _iter_chars(text):
    """Yield (index, char, in_string) applying MATLAB's quote rule: a ' that
    directly follows an operand (identifier, number, ')' or ']', or another
    transpose) is the transpose operator, not a string delimiter."""
    quote = None
    prev_sig = ""
    closed_str = False
    depth = 0
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        prev_space = i > 0 and text[i - 1].isspace()
        if quote:
            if ch == quote:
                if i + 1 < n and text[i + 1] == quote:      # doubled quote = escape
                    yield (i, ch, True)
                    yield (i + 1, text[i + 1], True)
                    i += 2
                    continue
                quote = None
                closed_str = True
                yield (i, ch, True)
                i += 1
                continue
            yield (i, ch, True)
            i += 1
            continue
        if ch == '"':
            quote = '"'
            closed_str = False
            yield (i, ch, True)
        elif ch == "'":
            is_t = (not closed_str and (prev_sig.isalnum() or prev_sig in "_)]'.")
                    and not (depth > 0 and prev_space))
            if is_t:
                prev_sig = "'"
                yield (i, ch, False)
            else:
                quote = "'"
                closed_str = False
                yield (i, ch, True)
        else:
            if not ch.isspace():
                prev_sig = ch
                closed_str = False
                if ch == "[":
                    depth += 1
                elif ch == "]":
                    depth = max(0, depth - 1)
            yield (i, ch, False)
        i += 1


def strip_comment(line):
    out = []
    for _, ch, ins in _iter_chars(line):
        if not ins and ch == "%":
            break
        out.append(ch)
    return "".join(out)


def _split_top_commas(text):
    """Split on commas that are not nested in brackets or strings."""
    parts, cur, depth = [], "", 0
    for _, ch, ins in _iter_chars(text):
        if not ins:
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth -= 1
            elif ch == "," and depth == 0:
                parts.append(cur)
                cur = ""
                continue
        cur += ch
    parts.append(cur)
    return parts


def _lf(code):
    """Windows files arrive with CRLF; a stray \r breaks row-separated matrix literals."""
    return code.replace("\r\n", "\n").replace("\r", "\n")


def split_statements(code):
    # Join '...' continuations and strip comments line-by-line.
    code = _lf(code)
    joined = []
    buf = ""
    for raw in code.split("\n"):
        line = strip_comment(raw)
        if line.rstrip().endswith("..."):
            buf += line.rstrip()[:-3]
            continue
        buf += line
        joined.append(buf)
        buf = ""
    if buf:
        joined.append(buf)
    text = "\n".join(joined)

    stmts = []
    cur = ""
    depth = 0
    for _, ch, ins in _iter_chars(text):
        if ins:
            cur += ch
            continue
        if ch in "([{":
            depth += 1
            cur += ch
        elif ch in ")]}":
            depth -= 1
            cur += ch
        elif depth == 0 and ch in ";\n,":
            if cur.strip():
                stmts.append((cur.strip(), ch == ";"))
            cur = ""
        else:
            cur += ch
    if cur.strip():
        stmts.append((cur.strip(), False))
    return stmts


def top_level_assign(text):
    """Index of a top-level '=' that is assignment (not == <= >= ~=)."""
    depth = 0
    for i, ch, ins in _iter_chars(text):
        if ins:
            continue
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == "=" and depth == 0:
            prev = text[i - 1] if i else ""
            nxt = text[i + 1] if i + 1 < len(text) else ""
            if prev in "<>=~!" or nxt == "=":
                return None
            return i
    return None


def _plot_reals(v):
    """Plotting uses the real part of complex data (like MATLAB, sans warning)."""
    return [x.real if isinstance(x, complex) else x for x in v]


def parse_spec(spec):
    """Parse a MATLAB line spec like 'r--o' -> (color, style, marker)."""
    color = style = marker = None
    s = spec
    for st in ("--", "-.", ":", "-"):
        if st in s:
            style = st
            s = s.replace(st, "", 1)
            break
    for ch in s:
        if ch in _COLORS:
            color = _COLORS[ch]
        elif ch in "o+*.xsd^v":
            marker = ch
    return color, style, marker


def _need(name, args, n):
    if len(args) < n:
        raise MError(f"{name} expects {n} argument(s)")


# Function tables ------------------------------------------------------------
def _cw(fr, fc):
    """Real function with a complex fallback (cmath) for complex inputs."""
    def f(x):
        return fc(x) if isinstance(x, complex) else fr(x)
    return f


def _sqrt(x):
    if isinstance(x, complex):
        return cmath.sqrt(x)
    return math.sqrt(x) if x >= 0 else cmath.sqrt(complex(x))


def _log_base(base):
    def f(x):
        if isinstance(x, complex) or x < 0:
            v = cmath.log(complex(x))
            return v if base is None else v / math.log(base)
        if x == 0:
            return -math.inf
        return math.log(x) if base is None else math.log(x, base)
    return f


def _sinc(x):
    if x == 0:
        return 1.0
    return math.sin(math.pi * x) / (math.pi * x)


def _mag2db(x):
    a = abs(x)
    return -math.inf if a == 0 else 20 * math.log10(a)


def _pow2db(x):
    a = abs(x)
    return -math.inf if a == 0 else 10 * math.log10(a)


def _nextpow2(x):
    a = abs(x)
    return 0.0 if a <= 1 else float(math.ceil(math.log2(a)))


_ELEMENTWISE = {
    "sin": _cw(math.sin, cmath.sin), "cos": _cw(math.cos, cmath.cos),
    "tan": _cw(math.tan, cmath.tan),
    "asin": lambda x: math.asin(max(-1, min(1, x))),
    "acos": lambda x: math.acos(max(-1, min(1, x))),
    "atan": math.atan, "sinh": _cw(math.sinh, cmath.sinh),
    "cosh": _cw(math.cosh, cmath.cosh), "tanh": _cw(math.tanh, cmath.tanh),
    "exp": _cw(math.exp, cmath.exp), "sqrt": _sqrt,
    "log": _log_base(None),
    "log2": _log_base(2), "log10": _log_base(10),
    "abs": abs, "sign": lambda x: (x > 0) - (x < 0), "floor": math.floor,
    "ceil": math.ceil, "round": lambda x: float(round(x)), "fix": math.trunc,
    # signals & complex helpers
    "real": lambda x: x.real, "imag": lambda x: x.imag,
    "conj": lambda x: x.conjugate(), "angle": cmath.phase,
    "sinc": _sinc, "deg2rad": math.radians, "rad2deg": math.degrees,
    "mag2db": _mag2db, "pow2db": _pow2db, "db": _mag2db,
    "db2mag": lambda x: 10 ** (x / 20), "db2pow": lambda x: 10 ** (x / 10),
    "nextpow2": _nextpow2,
}
_BINARY = {
    "mod": lambda a, b: a - b * math.floor(a / b) if b else math.nan,
    "rem": lambda a, b: math.fmod(a, b) if b else math.nan,
    "atan2": math.atan2, "hypot": math.hypot, "power": _safe_pow,
}


def _has_cplx(v):
    return any(isinstance(x, complex) for x in v)


def _sort_key(v):
    # MATLAB orders complex values by magnitude, then phase.
    return (lambda x: (abs(x), cmath.phase(x))) if _has_cplx(v) else (lambda x: x)


def _reduce_max(v):
    return max(v, key=_sort_key(v)) if v else math.nan


def _reduce_min(v):
    return min(v, key=_sort_key(v)) if v else math.nan


def _var(v):
    if len(v) < 2:
        return 0.0
    m = sum(v) / len(v)
    return sum(abs(x - m) ** 2 for x in v) / (len(v) - 1)


def _std(v):
    return math.sqrt(_var(v))


def _rms(v):
    if not v:
        return math.nan
    return math.sqrt(sum(abs(x) ** 2 for x in v) / len(v))


def _median(v):
    s = sorted(v, key=_sort_key(v))
    n = len(s)
    if n == 0:
        return math.nan
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def _cumsum(v):
    out, acc = [], 0.0
    for x in v:
        acc += x
        out.append(acc)
    return out


def _cumprod(v):
    out, acc = [], 1.0
    for x in v:
        acc *= x
        out.append(acc)
    return out


_REDUCERS = {
    "sum": lambda v: sum(v), "mean": lambda v: sum(v) / len(v) if v else math.nan,
    "prod": lambda v: math.prod(v) if v else 1.0, "length": lambda v: float(len(v)),
    "numel": lambda v: float(len(v)), "max": _reduce_max, "min": _reduce_min,
    "std": _std, "var": _var, "rms": _rms, "median": _median,
    "sort": lambda v: sorted(v, key=_sort_key(v)), "cumsum": _cumsum,
    "cumprod": _cumprod,
}


def _build_linspace(args):
    a = scalar(args[0])
    b = scalar(args[1])
    n = int(scalar(args[2])) if len(args) > 2 else 100
    if n < 2:
        return [b]
    return [a + (b - a) * k / (n - 1) for k in range(n)]


def _build_logspace(args):
    a = scalar(args[0])
    b = scalar(args[1])
    n = int(scalar(args[2])) if len(args) > 2 else 50
    return [10 ** (a + (b - a) * k / (n - 1)) for k in range(n)]


def _build_fill(val):
    def f(args):
        nums = [int(scalar(a)) for a in args] if args else [1]
        n = nums[-1] if nums else 1
        return [float(val)] * max(0, n)
    return f


def _build_rand(args):
    nums = [int(scalar(a)) for a in args] if args else [1]
    n = nums[-1] if nums else 1
    return [random.random() for _ in range(max(0, n))]


def _build_randn(args):
    nums = [int(scalar(a)) for a in args] if args else [1]
    n = nums[-1] if nums else 1
    return [random.gauss(0.0, 1.0) for _ in range(max(0, n))]


def _build_window(shape):
    """Window builders: hamming(N), hann(N), ..."""
    def f(args):
        _need("window", args, 1)
        n = int(scalar(args[0]))
        if n <= 0:
            return []
        if n == 1:
            return [1.0]
        return [shape(k / (n - 1)) for k in range(n)]
    return f


_WINDOWS = {
    "hamming": lambda u: 0.54 - 0.46 * math.cos(2 * math.pi * u),
    "hann": lambda u: 0.5 * (1 - math.cos(2 * math.pi * u)),
    "blackman": lambda u: 0.42 - 0.5 * math.cos(2 * math.pi * u) + 0.08 * math.cos(4 * math.pi * u),
    "bartlett": lambda u: 1.0 - abs(2 * u - 1),
    "rectwin": lambda u: 1.0,
}

_BUILDERS = {
    "linspace": _build_linspace, "logspace": _build_logspace,
    "zeros": _build_fill(0.0), "ones": _build_fill(1.0),
    "rand": _build_rand, "randn": _build_randn,
    **{name: _build_window(shape) for name, shape in _WINDOWS.items()},
    "hanning": _build_window(_WINDOWS["hann"]),
}


# --- FFT engine (pure stdlib): radix-2 for powers of two, Bluestein else ----
def _fft_pow2(x, sgn):
    n = len(x)
    if n == 1:
        return x[:]
    ev = _fft_pow2(x[0::2], sgn)
    od = _fft_pow2(x[1::2], sgn)
    out = [0j] * n
    for k in range(n // 2):
        t = cmath.exp(sgn * 2j * math.pi * k / n) * od[k]
        out[k] = ev[k] + t
        out[k + n // 2] = ev[k] - t
    return out


def _bluestein(x, sgn):
    """Chirp-z: DFT of arbitrary length via a power-of-two convolution."""
    n = len(x)
    m = 1
    while m < 2 * n + 1:
        m <<= 1
    w = [cmath.exp(sgn * 1j * math.pi * ((k * k) % (2 * n)) / n) for k in range(n)]
    a = [complex(x[k]) * w[k] for k in range(n)] + [0j] * (m - n)
    b = [0j] * m
    b[0] = w[0].conjugate()
    for k in range(1, n):
        b[k] = b[m - k] = w[k].conjugate()
    fa = _fft_pow2(a, -1)
    fb = _fft_pow2(b, -1)
    c = _fft_pow2([fa[k] * fb[k] for k in range(m)], 1)
    return [c[k] / m * w[k] for k in range(n)]


def _dft(x, inverse=False):
    n = len(x)
    if n == 0:
        return []
    sgn = 1 if inverse else -1
    if n & (n - 1) == 0:
        out = _fft_pow2([complex(v) for v in x], sgn)
    else:
        out = _bluestein(x, sgn)
    if inverse:
        out = [v / n for v in out]
    return out


def _fft_args(args, inverse):
    x = as_list(args[0])
    if len(args) > 1:
        n = int(scalar(args[1]))
        x = x[:n] + [0.0] * max(0, n - len(x))
    return _dft(x, inverse)


def _vf_fftshift(args):
    v = as_list(args[0])
    h = (len(v) + 1) // 2
    return v[h:] + v[:h]


def _vf_ifftshift(args):
    v = as_list(args[0])
    h = len(v) // 2
    return v[h:] + v[:h]


def _vf_conv(args):
    _need("conv", args, 2)
    a, b = as_list(args[0]), as_list(args[1])
    if not a or not b:
        return []
    out = [0.0] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            out[i + j] += ai * bj
    return out


def _vf_filter(args):
    _need("filter", args, 3)
    b, a, x = as_list(args[0]), as_list(args[1]), as_list(args[2])
    if not a or a[0] == 0:
        raise MError("filter: a(1) must be nonzero")
    a0 = a[0]
    b = [bi / a0 for bi in b]
    a = [ai / a0 for ai in a]
    y = []
    for n in range(len(x)):
        acc = sum(b[k] * x[n - k] for k in range(len(b)) if 0 <= n - k)
        acc -= sum(a[k] * y[n - k] for k in range(1, len(a)) if 0 <= n - k)
        y.append(acc)
    return y


def _vf_xcorr(args):
    x = as_list(args[0])
    y = as_list(args[1]) if len(args) > 1 and is_vec(args[1]) else x
    n = max(len(x), len(y))
    x = x + [0.0] * (n - len(x))
    y = y + [0.0] * (n - len(y))
    out = []
    for m in range(-(n - 1), n):
        acc = 0.0
        for k in range(n):
            j = k + m
            if 0 <= j < n:
                acc += x[j] * (y[k].conjugate() if isinstance(y[k], complex) else y[k])
        out.append(acc)
    return out


def _vf_unwrap(args):
    v = as_list(args[0])
    if not v:
        return []
    out = [v[0]]
    shift = 0.0
    for k in range(1, len(v)):
        d = v[k] - v[k - 1]
        if d > math.pi:
            shift -= 2 * math.pi
        elif d < -math.pi:
            shift += 2 * math.pi
        out.append(v[k] + shift)
    return out


def _vf_diff(args):
    v = as_list(args[0])
    return [v[k + 1] - v[k] for k in range(len(v) - 1)]


def _vf_trapz(args):
    if len(args) >= 2:
        x, y = as_list(args[0]), as_list(args[1])
        if len(x) != len(y):
            raise MError("trapz: x and y must be the same length")
        return sum((x[k + 1] - x[k]) * (y[k] + y[k + 1]) / 2 for k in range(len(x) - 1))
    y = as_list(args[0])
    return sum((y[k] + y[k + 1]) / 2 for k in range(len(y) - 1))


def _vf_dot(args):
    _need("dot", args, 2)
    a, b = as_list(args[0]), as_list(args[1])
    if len(a) != len(b):
        raise MError("dot: vectors must be the same length")
    return sum((x.conjugate() if isinstance(x, complex) else x) * y for x, y in zip(a, b))


def _vf_downsample(args):
    _need("downsample", args, 2)
    v = as_list(args[0])
    n = max(1, int(scalar(args[1])))
    return v[::n]


def _vf_upsample(args):
    _need("upsample", args, 2)
    v = as_list(args[0])
    n = max(1, int(scalar(args[1])))
    out = []
    for x in v:
        out.append(x)
        out.extend([0.0] * (n - 1))
    return out


def _sq_one(t, duty):
    frac = (t / (2 * math.pi)) % 1.0
    return 1.0 if frac * 100 < duty else -1.0


def _saw_one(t, width):
    frac = (t / (2 * math.pi)) % 1.0
    if width <= 0:
        return 1.0 - 2 * frac
    if frac < width:
        return 2 * frac / width - 1
    if width >= 1:
        return -1.0
    return 1 - 2 * (frac - width) / (1 - width)


def _vf_square(args):
    _need("square", args, 1)
    duty = float(scalar(args[1])) if len(args) > 1 else 50.0
    return elementwise(args[0], 0, lambda t, _: _sq_one(t, duty))


def _vf_sawtooth(args):
    _need("sawtooth", args, 1)
    width = float(scalar(args[1])) if len(args) > 1 else 1.0
    return elementwise(args[0], 0, lambda t, _: _saw_one(t, width))


def _vf_chirp(args):
    _need("chirp", args, 1)
    f0 = float(scalar(args[1])) if len(args) > 1 else 0.0
    t1 = float(scalar(args[2])) if len(args) > 2 else 1.0
    f1 = float(scalar(args[3])) if len(args) > 3 else 100.0
    if t1 == 0:
        raise MError("chirp: t1 must be nonzero")
    k = (f1 - f0) / (2 * t1)
    return elementwise(args[0], 0,
                       lambda t, _: math.cos(2 * math.pi * (f0 * t + k * t * t)))


_VECTOR_FUNCS = {
    "fft": lambda args: _fft_args(args, False),
    "ifft": lambda args: _fft_args(args, True),
    "fftshift": _vf_fftshift, "ifftshift": _vf_ifftshift,
    "conv": _vf_conv, "filter": _vf_filter, "xcorr": _vf_xcorr,
    "unwrap": _vf_unwrap, "diff": _vf_diff, "trapz": _vf_trapz, "dot": _vf_dot,
    "flip": lambda args: list(reversed(as_list(args[0]))),
    "fliplr": lambda args: list(reversed(as_list(args[0]))),
    "downsample": _vf_downsample, "upsample": _vf_upsample,
    "square": _vf_square, "sawtooth": _vf_sawtooth, "chirp": _vf_chirp,
}
_ZERO_ARG = {"pi": lambda: math.pi, "e": lambda: math.e,
             "i": lambda: 1j, "j": lambda: 1j,
             "Inf": lambda: math.inf, "inf": lambda: math.inf,
             "NaN": lambda: math.nan, "nan": lambda: math.nan,
             "eps": lambda: 2.220446049250313e-16,
             "realmax": lambda: 1.7976931348623157e308, "realmin": lambda: 2.2250738585072014e-308,
             "true": lambda: 1.0, "false": lambda: 0.0}


def _cr(x):
    return x.real if isinstance(x, complex) else x


_CMP_OPS = {
    "==": lambda x, y: 1.0 if x == y else 0.0,
    "~=": lambda x, y: 1.0 if x != y else 0.0,
    "<": lambda x, y: 1.0 if _cr(x) < _cr(y) else 0.0,
    "<=": lambda x, y: 1.0 if _cr(x) <= _cr(y) else 0.0,
    ">": lambda x, y: 1.0 if _cr(x) > _cr(y) else 0.0,
    ">=": lambda x, y: 1.0 if _cr(x) >= _cr(y) else 0.0,
}


# ---------------------------------------------------------------------------
# Control systems: polynomials, roots, transfer functions
# ---------------------------------------------------------------------------
def _peval(p, x):
    """Horner evaluation; works for real and complex."""
    acc = 0.0
    for c in p:
        acc = acc * x + c
    return acc


def _trim_poly(c):
    c = list(c)
    while len(c) > 1 and abs(c[0]) < 1e-300:
        c.pop(0)
    return c


def _clean_c(c):
    return c.real if abs(c.imag) < 1e-8 * (1 + abs(c)) else c


def _roots_dk(coeffs):
    """Polynomial roots via Durand-Kerner (pure stdlib)."""
    c = _trim_poly([complex(x) for x in coeffs])
    origin = 0
    while len(c) > 1 and abs(c[-1]) == 0:
        c.pop()
        origin += 1
    n = len(c) - 1
    roots = [0j] * origin
    if n >= 1:
        lead = c[0]
        c = [x / lead for x in c]
        z = [(0.4 + 0.9j) ** k for k in range(1, n + 1)]
        for _ in range(500):
            maxd = 0.0
            for k in range(n):
                d = _peval(c, z[k])
                for j2 in range(n):
                    if j2 != k:
                        dz = z[k] - z[j2]
                        d /= dz if dz != 0 else 1e-30
                z[k] -= d
                maxd = max(maxd, abs(d))
            if maxd < 1e-12:
                break
        roots = z + roots
    return roots


def _vf_roots(args):
    r = [_clean_c(v) for v in _roots_dk(as_list(args[0]))]
    return mmat.colvec(r) if r else []       # MATLAB returns a column


def _vf_poly(args):
    c = [1.0 + 0j]
    for r in as_list(args[0]):
        c = _pmul(c, [1.0 + 0j, -complex(r)])
    return [_clean_c(x) for x in c]


def _vf_polyval(args):
    _need("polyval", args, 2)
    p = as_list(args[0])
    return elementwise(args[1], 0, lambda x, _: _peval(p, x))


def _pmul(a, b):
    out = [0.0] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        for j2, bj in enumerate(b):
            out[i + j2] += ai * bj
    return out


def _padd(a, b):
    n = max(len(a), len(b))
    a = [0.0] * (n - len(a)) + list(a)
    b = [0.0] * (n - len(b)) + list(b)
    return [x + y for x, y in zip(a, b)]


def make_tf(num, den):
    num = _trim_poly([float(_cr(c)) for c in as_list(num)])
    den = _trim_poly([float(_cr(c)) for c in as_list(den)])
    if all(abs(c) < 1e-300 for c in den):
        raise MError("tf: denominator cannot be zero")
    return {"__tf__": True, "num": num, "den": den}


def _as_tf(v):
    if _is_tf(v):
        return v
    if isinstance(v, (int, float)) or (is_vec(v) and len(v) == 1):
        return make_tf([float(_cr(scalar(v)))], [1.0])
    raise MError("cannot combine this value with a transfer function")


def _tf_bin(op, a, b):
    if op in ("^", ".^"):
        if not _is_tf(a):
            raise MError("transfer-function power needs tf ^ integer")
        k = int(round(float(_cr(scalar(b)))))
        num, den = ([1.0], [1.0])
        base = a if k >= 0 else make_tf(a["den"], a["num"])
        for _ in range(abs(k)):
            num, den = _pmul(num, base["num"]), _pmul(den, base["den"])
        return make_tf(num, den)
    A, B = _as_tf(a), _as_tf(b)
    if op == "+":
        return make_tf(_padd(_pmul(A["num"], B["den"]), _pmul(B["num"], A["den"])),
                       _pmul(A["den"], B["den"]))
    if op == "-":
        return make_tf(_padd(_pmul(A["num"], B["den"]), [-c for c in _pmul(B["num"], A["den"])]),
                       _pmul(A["den"], B["den"]))
    if op in ("*", ".*"):
        return make_tf(_pmul(A["num"], B["num"]), _pmul(A["den"], B["den"]))
    if op in ("/", "./"):
        if all(abs(c) < 1e-300 for c in B["num"]):
            raise MError("division by a zero transfer function")
        return make_tf(_pmul(A["num"], B["den"]), _pmul(A["den"], B["num"]))
    raise MError(f"operator '{op}' is not defined for transfer functions")


def _poly_str(c, var="s"):
    """'[1 3 2]' -> 's^2 + 3 s + 2'."""
    n = len(c) - 1
    terms = []
    for k, coef in enumerate(c):
        p = n - k
        if coef == 0 and n > 0:
            continue
        mag = abs(coef)
        sgn = "-" if _cr(coef) < 0 else "+"
        if p == 0:
            body = _fmt_num(mag)
        else:
            sv = var if p == 1 else f"{var}^{p}"
            body = sv if mag == 1 else f"{_fmt_num(mag)} {sv}"
        terms.append((sgn, body))
    if not terms:
        return "0"
    first_sgn, first_body = terms[0]
    out = ("-" if first_sgn == "-" else "") + first_body
    for sgn, body in terms[1:]:
        out += f" {sgn} {body}"
    return out


def _tf_display(name, sys):
    ns = _poly_str(sys["num"])
    ds = _poly_str(sys["den"])
    w = max(len(ns), len(ds)) + 2
    bar = "-" * w
    return (f"\n{name} =\n\n  {ns.center(w)}\n  {bar}\n  {ds.center(w)}\n\n"
            f"  Continuous-time transfer function.\n\n")


def _cf_tf(args):
    if len(args) == 1 and isinstance(args[0], str):
        if args[0] in ("s", "p"):
            return make_tf([1.0, 0.0], [1.0])
        raise MError("tf: unknown variable (use tf('s'))")
    if len(args) == 1 and _is_tf(args[0]):
        return args[0]
    if len(args) == 2:
        return make_tf(args[0], args[1])
    raise MError("tf expects tf('s') or tf(num, den)")


def _cf_feedback(args):
    _need("feedback", args, 1)
    G = _as_tf(args[0])
    H = _as_tf(args[1]) if len(args) > 1 else make_tf([1.0], [1.0])
    sign = float(_cr(scalar(args[2]))) if len(args) > 2 else -1.0
    num = _pmul(G["num"], H["den"])
    loop = _pmul(G["num"], H["num"])
    den = _padd(_pmul(G["den"], H["den"]), [-sign * c for c in loop])
    return make_tf(num, den)


def _cf_dcgain(args):
    sys = _as_tf(args[0])
    n0 = _peval(sys["num"], 0.0)
    d0 = _peval(sys["den"], 0.0)
    return _safe_div(float(_cr(n0)), float(_cr(d0)))


_CONTROL_FUNCS = {
    "tf": _cf_tf,
    "feedback": _cf_feedback,
    "series": lambda args: _tf_bin("*", _as_tf(args[0]), _as_tf(args[1])),
    "parallel": lambda args: _tf_bin("+", _as_tf(args[0]), _as_tf(args[1])),
    "dcgain": _cf_dcgain,
    "pole": lambda args: [_clean_c(r) for r in _roots_dk(_as_tf(args[0])["den"])],
    "zero": lambda args: [_clean_c(r) for r in _roots_dk(_as_tf(args[0])["num"])],
    "zpk": lambda args: _zpk(args),
    "bandwidth": lambda args: _cf_bandwidth(args),
}


def _cf_bandwidth(args):
    """-3 dB bandwidth (rad/s) relative to the DC gain."""
    sys = _as_tf(args[0])
    w, mdb, _ = _tf_freq_response(sys, 200)
    dc = 20 * math.log10(abs(_cf_dcgain([sys]))) if _cf_dcgain([sys]) != 0 else mdb[0]
    target = dc - 3.0103
    for k in range(1, len(w)):
        if (mdb[k - 1] - target) * (mdb[k] - target) <= 0 and mdb[k] != mdb[k - 1]:
            f = (target - mdb[k - 1]) / (mdb[k] - mdb[k - 1])
            return w[k - 1] + f * (w[k] - w[k - 1])
    return float("nan")


def _fact(x):
    k = int(round(_cr(x)))
    if k < 0:
        raise MError("factorial: input must be a non-negative integer")
    return float(math.factorial(k))


_ELEMENTWISE["factorial"] = _fact
_REDUCERS.update({
    "any": lambda v: 1.0 if any(x != 0 for x in v) else 0.0,
    "all": lambda v: 1.0 if all(x != 0 for x in v) else 0.0,
    "isempty": lambda v: 1.0 if len(v) == 0 else 0.0,
    "find": lambda v: [float(k + 1) for k, x in enumerate(v) if x != 0],
})
_VECTOR_FUNCS.update({
    "roots": _vf_roots,
    "poly": _vf_poly,
    "polyval": _vf_polyval,
    "deconv": lambda args: _vf_deconv(args),
    "residue": lambda args: _vf_residue(args),
    "polyfit": lambda args: _polyfit(args),
    "interp1": lambda args: _interp1(args),
    "hilbert": lambda args: _hilbert(args),
    "strcmp": lambda args: 1.0 if (len(args) > 1 and args[0] == args[1]) else 0.0,
    "num2str": lambda args: (args[0] if isinstance(args[0], str)
                             else (_fmt_vec(args[0]).strip() if is_vec(args[0])
                                   else _num2str(scalar(args[0])))),
})


def _tf_time_response(sys, kind, tfinal=None):
    """Step/impulse response via controllable-canonical state space + RK4."""
    num = _trim_poly([float(_cr(c)) for c in sys["num"]])
    den = _trim_poly([float(_cr(c)) for c in sys["den"]])
    n = len(den) - 1
    if len(num) - 1 > n:
        raise MError(f"{kind}: improper transfer function (numerator degree > denominator)")
    a0 = den[0]
    den = [c / a0 for c in den]
    num = [c / a0 for c in num]
    if n == 0:
        k = num[0] if num else 0.0
        y = k if kind == "step" else 0.0
        return [0.0, 1.0], [y, y]
    b = [0.0] * (n + 1 - len(num)) + num
    d = b[0]
    r = [b[k] - d * den[k] for k in range(1, n + 1)]   # strictly proper part
    arow = [-den[n - k] for k in range(n)]
    C = [r[n - k] for k in range(1, n + 1)]
    poles = _roots_dk(den)
    rates = [abs(p.real) for p in poles if abs(p.real) > 1e-9]
    T = tfinal if tfinal else (8.0 / min(rates) if rates else 20.0)
    T = min(max(T, 1e-3), 1e5)
    fast = max([abs(p) for p in poles] + [1.0 / T])
    steps = int(min(30000, max(1500, T * fast * 8)))
    dt = T / steps
    u = 1.0 if kind == "step" else 0.0
    x = [0.0] * n
    if kind == "impulse":
        x[n - 1] = 1.0                                  # x(0) = B

    def deriv(xv):
        xd = [xv[k + 1] for k in range(n - 1)]
        xd.append(sum(arow[k] * xv[k] for k in range(n)) + u)
        return xd

    def out_y(xv):
        return sum(C[k] * xv[k] for k in range(n)) + d * u

    keep = max(1, steps // 1200)
    ts, ys = [0.0], [out_y(x)]
    for s2 in range(1, steps + 1):
        k1 = deriv(x)
        k2 = deriv([x[k] + dt / 2 * k1[k] for k in range(n)])
        k3 = deriv([x[k] + dt / 2 * k2[k] for k in range(n)])
        k4 = deriv([x[k] + dt * k3[k] for k in range(n)])
        x = [x[k] + dt / 6 * (k1[k] + 2 * k2[k] + 2 * k3[k] + k4[k]) for k in range(n)]
        if s2 % keep == 0 or s2 == steps:
            ts.append(s2 * dt)
            ys.append(out_y(x))
    return ts, ys


def _tf_freq_response(sys, npts_per_decade=60):
    num, den = sys["num"], sys["den"]
    crit = [abs(rt) for rt in _roots_dk(num) + _roots_dk(den) if abs(rt) > 1e-9]
    lo = math.floor(math.log10(min(crit))) - 1 if crit else -1
    hi = math.ceil(math.log10(max(crit))) + 1 if crit else 2
    if hi - lo < 3:
        hi = lo + 3
    N = npts_per_decade * (hi - lo)
    ws = [10.0 ** (lo + (hi - lo) * k / (N - 1)) for k in range(N)]
    mdb, ph = [], []
    for w in ws:
        dv = _peval(den, 1j * w)
        if dv == 0:
            mdb.append(math.inf)
            ph.append(ph[-1] if ph else 0.0)
            continue
        h = _peval(num, 1j * w) / dv
        a = abs(h)
        mdb.append(20 * math.log10(a) if a > 0 else -math.inf)
        ph.append(cmath.phase(h))
    ph = _vf_unwrap([ph])
    phd = [p * 180 / math.pi for p in ph]
    return ws, mdb, phd


def _tf_margins(sys):
    w, mdb, ph = _tf_freq_response(sys, 200)
    gm = wg = pm = wp = None
    for k in range(1, len(w)):
        if not (math.isfinite(ph[k - 1]) and math.isfinite(ph[k])):
            continue
        if gm is None and (ph[k - 1] - (-180.0)) * (ph[k] - (-180.0)) <= 0 and ph[k] != ph[k - 1]:
            f = (-180.0 - ph[k - 1]) / (ph[k] - ph[k - 1])
            wg = w[k - 1] + f * (w[k] - w[k - 1])
            gm = -(mdb[k - 1] + f * (mdb[k] - mdb[k - 1]))
        if pm is None and math.isfinite(mdb[k - 1]) and math.isfinite(mdb[k]) \
                and mdb[k - 1] * mdb[k] <= 0 and mdb[k] != mdb[k - 1]:
            f = (0.0 - mdb[k - 1]) / (mdb[k] - mdb[k - 1])
            wp = w[k - 1] + f * (w[k] - w[k - 1])
            pm = 180.0 + (ph[k - 1] + f * (ph[k] - ph[k - 1]))
    return gm, wg, pm, wp


# ===========================================================================
# Advanced signals & systems + control-systems library (all pure stdlib)
# ===========================================================================
def _lin_solve(A, b):
    """Solve A x = b by Gauss-Jordan with partial pivoting. Complex-capable."""
    n = len(A)
    M = [[complex(A[i][j]) for j in range(n)] + [complex(b[i])] for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < 1e-300:
            raise MError("singular system (cannot solve)")
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


def _poly_divmod_c(num, den):
    """Polynomial long division (complex). Returns (quotient, remainder)."""
    num = [complex(c) for c in num]
    den = [complex(c) for c in den]
    while len(den) > 1 and den[0] == 0:
        den = den[1:]
    if all(c == 0 for c in den):
        raise MError("deconv: divisor polynomial is zero")
    if len(num) < len(den):
        return [0j], num
    q = [0j] * (len(num) - len(den) + 1)
    r = list(num)
    for i in range(len(q)):
        coef = r[i] / den[0]
        q[i] = coef
        for j in range(len(den)):
            r[i + j] -= coef * den[j]
    return q, r


def _deflate(c, p):
    """Divide poly c (highest-first) by (s - p) via synthetic division."""
    c = [complex(x) for x in c]
    q = [c[0]]
    for k in range(1, len(c)):
        q.append(c[k] + p * q[-1])
    return q[:-1]                    # drop remainder (root => ~0)


def _vf_deconv(args):
    _need("deconv", args, 2)
    q, _ = _poly_divmod_c(as_list(args[0]), as_list(args[1]))
    return [_clean_c(x) for x in q]


def _mf_deconv(args):
    _need("deconv", args, 2)
    b = [complex(_cr(x)) for x in as_list(args[0])]
    q, r = _poly_divmod_c(b, as_list(args[1]))
    r = [0j] * (len(b) - len(r)) + r            # pad to length of b (MATLAB parity)
    return [[_clean_c(x) for x in q], [_clean_c(x) for x in r]]


def _cluster_roots(roots, tol=1e-4):
    """Group nearly-equal roots -> list of (value, multiplicity)."""
    groups = []
    for r in roots:
        for g in groups:
            if abs(r - g[0]) <= tol * (1 + abs(r)):
                g[1] += 1
                g[2].append(r)
                break
        else:
            groups.append([r, 1, [r]])
    return [(sum(g[2]) / len(g[2]), g[1]) for g in groups]


def _residue(bv, av):
    """Partial-fraction expansion. Returns (r, p, k) lists (MATLAB residue)."""
    b = _trim_poly([complex(_cr(x)) for x in as_list(bv)])
    a = _trim_poly([complex(_cr(x)) for x in as_list(av)])
    if len(a) < 2:
        raise MError("residue: denominator must have degree >= 1")
    k = []
    nrem = b
    if len(b) >= len(a):
        k, nrem = _poly_divmod_c(b, a)
    nrem = _trim_poly(nrem) if nrem else [0j]
    poles = _roots_dk(a)
    groups = _cluster_roots(poles)
    n = len(a) - 1
    cols, order = [], []                          # order: (pole, power)
    for pv, m in groups:
        base = a
        for lvl in range(1, m + 1):
            base = _deflate(base, pv)             # base = a / (s - pv)^lvl
            padded = [0j] * (n - len(base)) + list(base)
            cols.append(padded)
            order.append((pv, lvl))
    rhs = [0j] * (n - len(nrem)) + list(nrem)
    M = [[cols[c][row] for c in range(n)] for row in range(n)]
    coeffs = _lin_solve(M, rhs)
    r = [_clean_c(c) for c in coeffs]
    p = [_clean_c(pv) for pv, _ in order]
    kk = [_clean_c(c) for c in k] if k else []
    return r, p, kk


def _vf_residue(args):
    _need("residue", args, 2)
    return _residue(args[0], args[1])[0]


def _mf_residue(args):
    _need("residue", args, 2)
    r, p, k = _residue(args[0], args[1])
    return [r, p, k if k else []]


def _polyfit(args):
    _need("polyfit", args, 3)
    x = [float(_cr(v)) for v in as_list(args[0])]
    y = [float(_cr(v)) for v in as_list(args[1])]
    deg = int(round(float(_cr(scalar(args[2])))))
    if len(x) != len(y):
        raise MError("polyfit: x and y must be the same length")
    cols = deg + 1
    # Normal equations (A^T A) c = A^T y, A[i][j] = x_i^(deg-j)
    ATA = [[sum(x[i] ** ((deg - a) + (deg - b)) for i in range(len(x)))
            for b in range(cols)] for a in range(cols)]
    ATy = [sum((x[i] ** (deg - a)) * y[i] for i in range(len(x))) for a in range(cols)]
    return [_clean_c(c) for c in _lin_solve(ATA, ATy)]


def _interp1(args):
    _need("interp1", args, 3)
    x = [float(_cr(v)) for v in as_list(args[0])]
    y = as_list(args[1])
    xq = as_list(args[2])
    method = args[3].lower() if len(args) > 3 and isinstance(args[3], str) else "linear"
    pts = sorted(zip(x, y), key=lambda t: t[0])
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]

    def one(q):
        q = float(_cr(q))
        if q < xs[0] or q > xs[-1]:
            return float("nan")
        for i in range(len(xs) - 1):
            if xs[i] <= q <= xs[i + 1]:
                if method == "nearest":
                    return ys[i] if (q - xs[i]) <= (xs[i + 1] - q) else ys[i + 1]
                if xs[i + 1] == xs[i]:
                    return ys[i]
                t = (q - xs[i]) / (xs[i + 1] - xs[i])
                return ys[i] + t * (ys[i + 1] - ys[i])
        return ys[-1]

    out = [one(q) for q in xq]
    return out if is_vec(args[2]) else out[0]


def _hilbert(args):
    """Analytic signal via FFT (returns complex vector; imag = Hilbert xform)."""
    x = as_list(args[0])
    n = len(x)
    if n == 0:
        return []
    X = _dft([complex(v) for v in x], inverse=False)
    h = [0.0] * n
    h[0] = 1.0
    if n % 2 == 0:
        h[n // 2] = 1.0
        for i in range(1, n // 2):
            h[i] = 2.0
    else:
        for i in range(1, (n + 1) // 2):
            h[i] = 2.0
    return _dft([X[i] * h[i] for i in range(n)], inverse=True)


def _zpk(args):
    """Build a transfer function from zeros, poles, gain."""
    _need("zpk", args, 3)
    z = as_list(args[0])
    p = as_list(args[1])
    kgain = float(_cr(scalar(args[2])))
    num = [complex(kgain)]
    for zi in z:
        num = _pmul(num, [1.0 + 0j, -complex(zi)])
    den = [1.0 + 0j]
    for pi in p:
        den = _pmul(den, [1.0 + 0j, -complex(pi)])
    return make_tf([_cr(c) for c in num], [_cr(c) for c in den])


def _tf_lsim(sys, u, t):
    """Simulate an LTI system's response to input u at time samples t (RK4)."""
    num = _trim_poly([float(_cr(c)) for c in sys["num"]])
    den = _trim_poly([float(_cr(c)) for c in sys["den"]])
    n = len(den) - 1
    if len(num) - 1 > n:
        raise MError("lsim: improper transfer function")
    a0 = den[0]
    den = [c / a0 for c in den]
    num = [c / a0 for c in num]
    tv = [float(_cr(v)) for v in as_list(t)]
    uv = [float(_cr(v)) for v in as_list(u)]
    if len(tv) < 2:
        raise MError("lsim: t must have at least two samples")
    if len(uv) == 1:
        uv = uv * len(tv)
    if len(uv) != len(tv):
        raise MError("lsim: u and t must be the same length")
    if n == 0:
        g = num[0] if num else 0.0
        return tv, [g * ui for ui in uv]
    b = [0.0] * (n + 1 - len(num)) + num
    d = b[0]
    r = [b[k] - d * den[k] for k in range(1, n + 1)]
    arow = [-den[n - k] for k in range(n)]
    C = [r[n - k] for k in range(1, n + 1)]
    x = [0.0] * n

    def uat(tt):
        if tt <= tv[0]:
            return uv[0]
        if tt >= tv[-1]:
            return uv[-1]
        lo, hi = 0, len(tv) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if tv[mid] <= tt:
                lo = mid
            else:
                hi = mid
        frac = (tt - tv[lo]) / (tv[hi] - tv[lo])
        return uv[lo] + frac * (uv[hi] - uv[lo])

    def deriv(xv, uu):
        xd = [xv[k + 1] for k in range(n - 1)]
        xd.append(sum(arow[k] * xv[k] for k in range(n)) + uu)
        return xd

    ys = [sum(C[k] * x[k] for k in range(n)) + d * uv[0]]
    for i in range(len(tv) - 1):
        h = tv[i + 1] - tv[i]
        u0, uh, u1 = uat(tv[i]), uat(tv[i] + h / 2), uat(tv[i + 1])
        k1 = deriv(x, u0)
        k2 = deriv([x[k] + h / 2 * k1[k] for k in range(n)], uh)
        k3 = deriv([x[k] + h / 2 * k2[k] for k in range(n)], uh)
        k4 = deriv([x[k] + h * k3[k] for k in range(n)], u1)
        x = [x[k] + h / 6 * (k1[k] + 2 * k2[k] + 2 * k3[k] + k4[k]) for k in range(n)]
        ys.append(sum(C[k] * x[k] for k in range(n)) + d * u1)
    return tv, ys


def _stepinfo(sys):
    """Rise/settle/overshoot/peak metrics from the step response."""
    t, y = _tf_time_response(sys, "step")
    yf = y[-1]
    if abs(yf) < 1e-12:
        raise MError("stepinfo: final value is zero (no steady state)")
    peak = max(y, key=lambda v: abs(v))
    ipeak = y.index(peak)
    tpeak = t[ipeak]
    overshoot = max(0.0, (peak - yf) / yf * 100.0) if yf > 0 else \
        max(0.0, (yf - peak) / yf * 100.0)

    def crossing(frac):
        target = frac * yf
        for i in range(1, len(y)):
            lo, hi = y[i - 1], y[i]
            if (lo - target) * (hi - target) <= 0 and hi != lo:
                f = (target - lo) / (hi - lo)
                return t[i - 1] + f * (t[i] - t[i - 1])
        return float("nan")

    t10, t90 = crossing(0.10), crossing(0.90)
    rise = t90 - t10 if (t90 == t90 and t10 == t10) else float("nan")
    band = 0.02 * abs(yf)
    settle = t[-1]
    for i in range(len(y) - 1, -1, -1):
        if abs(y[i] - yf) > band:
            settle = t[i + 1] if i + 1 < len(t) else t[-1]
            break
    else:
        settle = 0.0
    return {"RiseTime": rise, "SettlingTime": settle, "Overshoot": overshoot,
            "Peak": abs(peak), "PeakTime": tpeak, "FinalValue": yf}


def _mf_size(args):
    return [float(x) for x in mmat.shape(args[0])]


# Built-ins that can return multiple outputs: [a, b, ...] = f(...)
_MULTI_FUNCS = {
    "deconv": _mf_deconv,
    "residue": _mf_residue,
    "size": _mf_size,
}


_PLOT_FUNCS = {"plot", "bar", "scatter", "stem", "plot3", "semilogx"}
_PLOT_PROPS = {"linewidth", "color", "linestyle", "marker", "markersize",
               "markerfacecolor", "markeredgecolor", "displayname"}
_COLOR_NAMES = {
    "red": "#FF0000", "green": "#00BF00", "blue": "#0000FF", "cyan": "#00BFBF",
    "magenta": "#BF00BF", "yellow": "#BFBF00", "black": "#000000", "white": "#FFFFFF",
}


def _to_color(val):
    """Resolve a Color value: 'r', 'red', '#RRGGBB', or [r g b] in 0..1."""
    if isinstance(val, str):
        if val in _COLORS:
            return _COLORS[val]
        if val.lower() in _COLOR_NAMES:
            return _COLOR_NAMES[val.lower()]
        if val.startswith("#"):
            return val
        return "#0072BD"
    rgb = as_list(val)
    if len(rgb) >= 3:
        return "#%02X%02X%02X" % tuple(max(0, min(255, int(round(c * 255)))) for c in rgb[:3])
    return "#0072BD"


# ---------------------------------------------------------------------------
# SVG renderer
# ---------------------------------------------------------------------------
def _esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def _attr(s):
    """Escape for an XML attribute delimited by double quotes."""
    return _esc(s).replace('"', "&quot;")


def _nice_axis(lo, hi, target=6, pad=0.0):
    """MATLAB-style limits: snap to round 'nice' bounds that contain the data."""
    if lo == hi:
        lo, hi = lo - 0.5, hi + 0.5
    span = hi - lo
    if pad:
        lo -= span * pad
        hi += span * pad
        span = hi - lo
    raw = span / target
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


def _tick_label(v, step):
    if abs(v) < step * 1e-6:
        v = 0.0
    if abs(v) >= 1e5 or (v != 0 and abs(v) < 1e-3):
        return f"{v:.1e}"
    return f"{v:.4g}"


# MATLAB-figure palette
_AX = "#262626"        # axes / ticks / labels  (MATLAB [0.15 0.15 0.15])
_GRID = "#d9d9d9"      # gridlines (GridAlpha 0.15 over white)
_FONT = "Helvetica, Arial, sans-serif"


def new_axes():
    """A single set of axes (one subplot cell)."""
    return {"series": [], "title": None, "xlabel": None, "ylabel": None,
            "legend": None, "grid": False, "xlog": False, "ylog": False,
            "xlim": None, "ylim": None, "equal": False, "circle": False,
            "vlines": [], "hlines": [], "texts": [],
            "row": 0, "col": 0, "span": 1, "rspan": 1,
            "is3d": False, "view": None, "cmap": None, "colorbar": False,
            "shading": "faceted", "zlabel": None, "zlim": None, "clim": None,
            "yreverse": False}


def new_figure():
    return {"axes": [], "rows": 1, "cols": 1, "slots": {}, "sgtitle": None,
            "frames": [], "delays": [], "stride": 1, "tick": 0}


_MAX_FRAMES = 200          # frames kept per figure; older ones are decimated 2:1


def _is_gh(v):
    return isinstance(v, dict) and v.get("__gh__")


def _ax_has_content(ax):
    return bool(ax["series"] or ax["vlines"] or ax["hlines"] or ax["texts"]
                or ax["circle"])


def fig_has_content(fig):
    return any(_ax_has_content(a) for a in fig["axes"])


def render_svg(fig):
    """Render a figure (one or more subplot axes) to a standalone SVG."""
    axes = fig.get("axes") or []
    rows = max(1, int(fig.get("rows", 1)))
    cols = max(1, int(fig.get("cols", 1)))
    CW = 620 if cols == 1 else max(300, min(500, 960 // cols))
    CH = 390 if rows == 1 else (300 if rows == 2 else 250)
    top = 28 if fig.get("sgtitle") else 0
    W, H = CW * cols, CH * rows + top
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
         f'width="100%" style="max-width:{W}px;font-family:{_FONT}">']
    p.append(f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>')
    if fig.get("sgtitle"):
        p.append(f'<text x="{W / 2:.0f}" y="20" fill="#000000" font-size="15" '
                 f'font-weight="600" text-anchor="middle">{_esc(fig["sgtitle"])}</text>')
    for idx, ax in enumerate(axes):
        r = ax.get("row", idx // cols)
        c = ax.get("col", idx % cols)
        _render_axes(p, ax, c * CW, top + r * CH,
                     CW * ax.get("span", 1), CH * ax.get("rspan", 1), idx)
    p.append("</svg>")
    return "".join(p)


def _render_axes(p, fig, ox, oy, W, H, uid=0):
    """Draw one set of axes into the rect at (ox, oy) of size W x H."""
    if fig.get("is3d"):
        mplot3.render_axes3d(p, fig, ox, oy, W, H, uid)
        return
    ML, MR, MT, MB = 64, 24, 42, 52
    if fig.get("colorbar"):
        MR += 66
    if H < 300:
        MT, MB = 30, 44
    pw, ph = W - ML - MR, H - MT - MB
    AX0, AY0 = ox + ML, oy + MT

    series = fig["series"]
    xlog = fig.get("xlog", False)
    ylog = fig.get("ylog", False)

    def xv(x):
        return math.log10(x) if xlog else x

    def yv(y):
        return math.log10(y) if ylog else y

    xs = [xv(x) for s in series for x in s["x"]
          if math.isfinite(x) and (not xlog or x > 0)]
    ys = [yv(y) for s in series for y in s["y"]
          if math.isfinite(y) and (not ylog or y > 0)]
    for s in series:
        if s["kind"] == "heat":
            xs = xs + list(s["x"])
            ys = ys + list(s["y"])
    if fig.get("circle"):
        xs = xs + [-1.15, 1.15]
        ys = ys + [-1.15, 1.15]
    if not xs or not ys:
        xs, ys = [0, 1], [0, 1]
    has_bar = any(s["kind"] == "bar" for s in series)

    xlim, ylim = fig.get("xlim"), fig.get("ylim")
    if xlog:
        if xlim:
            xlo, xhi = math.log10(max(xlim[0], 1e-300)), math.log10(max(xlim[1], 1e-300))
        else:
            xlo, xhi = math.floor(min(xs)), math.ceil(max(xs))
        if xlo == xhi:
            xhi += 1
        xticks, xstep = list(range(int(math.floor(xlo)), int(math.ceil(xhi)) + 1)), 1
    elif xlim:
        xlo, xhi = float(xlim[0]), float(xlim[1])
        _, _, xticks, xstep = _nice_axis(xlo, xhi)
        xticks = [t for t in xticks if xlo - 1e-9 <= t <= xhi + 1e-9]
    else:
        xlo, xhi, xticks, xstep = _nice_axis(min(xs), max(xs), pad=0.04 if has_bar else 0.0)

    if ylog:
        if ylim:
            ylo, yhi = math.log10(max(ylim[0], 1e-300)), math.log10(max(ylim[1], 1e-300))
        else:
            ylo, yhi = math.floor(min(ys)), math.ceil(max(ys))
        if ylo == yhi:
            yhi += 1
        yticks, ystep = list(range(int(math.floor(ylo)), int(math.ceil(yhi)) + 1)), 1
    elif ylim:
        ylo, yhi = float(ylim[0]), float(ylim[1])
        _, _, yticks, ystep = _nice_axis(ylo, yhi)
        yticks = [t for t in yticks if ylo - 1e-9 <= t <= yhi + 1e-9]
    else:
        ylo, yhi, yticks, ystep = _nice_axis(min(ys + ([0] if has_bar else [])), max(ys))

    if fig.get("equal") and not (xlog or ylog):
        # match the data-units-per-pixel on both axes (zplane, pzmap)
        sx = (xhi - xlo) / max(pw, 1)
        sy = (yhi - ylo) / max(ph, 1)
        s = max(sx, sy)
        cx, cy = (xlo + xhi) / 2, (ylo + yhi) / 2
        xlo, xhi = cx - s * pw / 2, cx + s * pw / 2
        ylo, yhi = cy - s * ph / 2, cy + s * ph / 2
        _, _, xticks, xstep = _nice_axis(xlo, xhi)
        _, _, yticks, ystep = _nice_axis(ylo, yhi)
        xticks = [t for t in xticks if xlo <= t <= xhi]
        yticks = [t for t in yticks if ylo <= t <= yhi]

    def sxr(v):
        return AX0 + (v - xlo) / (xhi - xlo) * pw

    def sx(x):
        return sxr(xv(x))

    def syr(v):
        f = (v - ylo) / (yhi - ylo)
        return (AY0 + f * ph) if fig.get("yreverse") else (AY0 + ph - f * ph)

    def sy(y):
        if ylog and y <= 0:
            return AY0 + ph + 1e4          # off-axis; the clip path hides it
        return syr(yv(y))

    p.append(f'<rect x="{AX0}" y="{AY0}" width="{pw}" height="{ph}" fill="#ffffff"/>')

    if fig.get("grid", False):
        for xt in xticks:
            X = sxr(xt)
            p.append(f'<line x1="{X:.1f}" y1="{AY0}" x2="{X:.1f}" y2="{AY0 + ph}" '
                     f'stroke="{_GRID}" stroke-width="0.8"/>')
        for yt in yticks:
            Y = syr(yt)
            p.append(f'<line x1="{AX0}" y1="{Y:.1f}" x2="{AX0 + pw}" y2="{Y:.1f}" '
                     f'stroke="{_GRID}" stroke-width="0.8"/>')

    cid = f"cp{uid}"
    p.append(f'<clipPath id="{cid}"><rect x="{AX0}" y="{AY0}" width="{pw}" height="{ph}"/></clipPath>')
    p.append(f'<g clip-path="url(#{cid})">')
    if fig.get("circle"):
        _draw_unit_circle(p, sx, sy)
    for s in series:
        if s.get("hidden"):
            continue
        if s["kind"] == "heat":
            _draw_heat(p, s, sx, sy)
        else:
            _draw_series(p, s, sx, sy, AY0, ph, ylo if not ylog else 10 ** ylo, xlog)
    for v in fig.get("vlines", []):
        X = sx(v[0])
        p.append(f'<line x1="{X:.1f}" y1="{AY0}" x2="{X:.1f}" y2="{AY0 + ph}" '
                 f'stroke="{v[1]}" stroke-width="1.4" stroke-dasharray="6,4"/>')
    for v in fig.get("hlines", []):
        Y = sy(v[0])
        p.append(f'<line x1="{AX0}" y1="{Y:.1f}" x2="{AX0 + pw}" y2="{Y:.1f}" '
                 f'stroke="{v[1]}" stroke-width="1.4" stroke-dasharray="6,4"/>')
    for tx in fig.get("texts", []):
        p.append(f'<text x="{sx(tx[0]):.1f}" y="{sy(tx[1]):.1f}" fill="{_AX}" '
                 f'font-size="11.5">{_esc(tx[2])}</text>')
    p.append("</g>")

    p.append(f'<rect x="{AX0}" y="{AY0}" width="{pw}" height="{ph}" fill="none" '
             f'stroke="{_AX}" stroke-width="1"/>')

    for xt in xticks:
        X = sxr(xt)
        lbl = _decade_label(xt) if xlog else _tick_label(xt, xstep)
        p.append(f'<line x1="{X:.1f}" y1="{AY0 + ph}" x2="{X:.1f}" y2="{AY0 + ph - 6}" '
                 f'stroke="{_AX}" stroke-width="1"/>')
        p.append(f'<text x="{X:.1f}" y="{AY0 + ph + 17}" fill="{_AX}" font-size="11.5" '
                 f'text-anchor="middle">{_esc(lbl)}</text>')
    for yt in yticks:
        Y = syr(yt)
        lbl = _decade_label(yt) if ylog else _tick_label(yt, ystep)
        p.append(f'<line x1="{AX0}" y1="{Y:.1f}" x2="{AX0 + 6}" y2="{Y:.1f}" '
                 f'stroke="{_AX}" stroke-width="1"/>')
        p.append(f'<text x="{AX0 - 8}" y="{Y + 4:.1f}" fill="{_AX}" font-size="11.5" '
                 f'text-anchor="end">{_esc(lbl)}</text>')

    if fig.get("title"):
        p.append(f'<text x="{AX0 + pw / 2:.0f}" y="{oy + MT - 16}" fill="#000000" '
                 f'font-size="14" text-anchor="middle">{_esc(fig["title"])}</text>')
    if fig.get("xlabel"):
        p.append(f'<text x="{AX0 + pw / 2:.0f}" y="{oy + H - 13}" fill="{_AX}" '
                 f'font-size="12.5" text-anchor="middle">{_esc(fig["xlabel"])}</text>')
    if fig.get("ylabel"):
        cy = AY0 + ph / 2
        p.append(f'<text x="{ox + 18}" y="{cy:.0f}" fill="{_AX}" font-size="12.5" '
                 f'text-anchor="middle" transform="rotate(-90 {ox + 18} {cy:.0f})">'
                 f'{_esc(fig["ylabel"])}</text>')

    legend = fig.get("legend")
    if legend:
        drawn = [s for s in series if s["kind"] != "heat"]
        names = [n for n in legend][:len(drawn)]
        bw = 16 + 10 + max((len(n) for n in names), default=4) * 6.6
        bh = 8 + len(names) * 17
        bx, by = AX0 + pw - bw - 8, AY0 + 8
        p.append(f'<rect x="{bx:.0f}" y="{by}" width="{bw:.0f}" height="{bh}" '
                 f'fill="#ffffff" stroke="{_AX}" stroke-width="0.8"/>')
        for i, name in enumerate(names):
            yy = by + 16 + i * 17
            c = drawn[i]["color"]
            p.append(f'<line x1="{bx + 8:.0f}" y1="{yy - 4}" x2="{bx + 26:.0f}" '
                     f'y2="{yy - 4}" stroke="{c}" stroke-width="2"/>')
            p.append(f'<text x="{bx + 32:.0f}" y="{yy}" fill="{_AX}" '
                     f'font-size="11.5">{_esc(name)}</text>')

    if fig.get("colorbar"):
        clim = fig.get("clim")
        if not clim:
            heats = [s for s in series if s["kind"] == "heat"]
            if heats:
                clim = (heats[0].get("zmin", 0.0), heats[0].get("zmax", 1.0))
        if clim:
            mplot3.draw_colorbar(p, AX0 + pw + 14, AY0 + ph * 0.08, 12, ph * 0.84,
                                 fig.get("cmap") or "parula", clim)

    # Interaction layer: a transparent hit rect carrying the pixel<->data mapping
    # and (for small series) the sample points, so the UI can zoom, pan and show
    # data tips without another round trip.  See makeFigureTools() in main.js.
    tips = []
    for i, s in enumerate(series):
        if s["kind"] == "heat" or s.get("hidden"):
            continue
        xsr, ysr = s.get("x") or [], s.get("y") or []
        if not xsr or len(xsr) > 4000:
            continue
        names = fig.get("legend") or []
        tips.append({
            "n": names[len(tips)] if len(tips) < len(names) else "",
            "c": s.get("color", "#000"),
            "x": [round(float(v), 10) for v in xsr],
            "y": [round(float(v), 10) for v in ysr],
        })
        if len(tips) >= 8:
            break
    meta = {
        "ax": uid, "x0": round(AX0, 2), "y0": round(AY0, 2),
        "w": round(pw, 2), "h": round(ph, 2),
        "xlo": xlo, "xhi": xhi, "ylo": ylo, "yhi": yhi,
        "xlog": bool(xlog), "ylog": bool(ylog),
        "yrev": bool(fig.get("yreverse")), "s": tips,
    }
    p.append(f'<rect class="sf-hit" x="{AX0}" y="{AY0}" width="{pw}" height="{ph}" '
             f'fill="none" pointer-events="all" '
             f'data-ax="{_attr(json.dumps(meta, separators=(",", ":")))}"/>')


def _draw_unit_circle(p, sx, sy):
    pts = []
    for k in range(181):
        th = 2 * math.pi * k / 180
        pts.append(f'{sx(math.cos(th)):.1f},{sy(math.sin(th)):.1f}')
    p.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="#9a9a9a" '
             f'stroke-width="1" stroke-dasharray="4,3"/>')
    p.append(f'<line x1="{sx(-1.3):.1f}" y1="{sy(0):.1f}" x2="{sx(1.3):.1f}" '
             f'y2="{sy(0):.1f}" stroke="#c8c8c8" stroke-width="0.8"/>')
    p.append(f'<line x1="{sx(0):.1f}" y1="{sy(-1.3):.1f}" x2="{sx(0):.1f}" '
             f'y2="{sy(1.3):.1f}" stroke="#c8c8c8" stroke-width="0.8"/>')


def _draw_heat(p, s, sx, sy):
    """Spectrogram / STFT: one rect per (time, frequency) cell."""
    xs, ys, z = s["x"], s["y"], s["z"]
    if not xs or not ys or not z:
        return
    lo = s.get("zmin")
    hi = s.get("zmax")
    if lo is None or hi is None:
        flat = [v for col in z for v in col if math.isfinite(v)]
        hi = max(flat) if flat else 1.0
        lo = max(hi - 80.0, min(flat) if flat else 0.0)
    rng = (hi - lo) or 1.0
    dx = (xs[1] - xs[0]) if len(xs) > 1 else 1.0
    dy = (ys[1] - ys[0]) if len(ys) > 1 else 1.0
    for i, xc in enumerate(xs):
        x0 = sx(xc - dx / 2)
        x1 = sx(xc + dx / 2)
        col = z[i]
        for j, yc in enumerate(ys):
            if j >= len(col):
                break
            y0 = sy(yc + dy / 2)
            y1 = sy(yc - dy / 2)
            p.append(f'<rect x="{min(x0, x1):.2f}" y="{min(y0, y1):.2f}" '
                     f'width="{abs(x1 - x0) + 0.6:.2f}" height="{abs(y1 - y0) + 0.6:.2f}" '
                     f'fill="{mplot3.cmap_color(s.get("cmap") or "parula", (col[j] - lo) / rng)}"/>')


def _decade_label(k):
    v = 10.0 ** k
    if -3 <= k <= 4:
        return f"{v:g}"
    return f"1e{int(k):+03d}"


def _draw_series(p, s, sx, sy, MT, ph, ylo, xlog=False):
    color = s["color"]
    width = s.get("width", 1.3)
    msize = s.get("msize", 5.0)
    # Non-finite samples (NaN/Inf, e.g. from mag2db(0)) are skipped like MATLAB.
    pts = [(x, y) for x, y in zip(s["x"], s["y"])
           if math.isfinite(x) and math.isfinite(y) and (not xlog or x > 0)]
    kind = s["kind"]
    if kind == "stairs":
        if not pts:
            return
        d = [f'M{sx(pts[0][0]):.1f},{sy(pts[0][1]):.1f}']
        for i in range(1, len(pts)):
            d.append(f'L{sx(pts[i][0]):.1f},{sy(pts[i - 1][1]):.1f}')
            d.append(f'L{sx(pts[i][0]):.1f},{sy(pts[i][1]):.1f}')
        p.append(f'<path d="{" ".join(d)}" fill="none" stroke="{color}" '
                 f'stroke-width="{width:.2g}"/>')
        return
    if kind == "area":
        if not pts:
            return
        base = sy(max(0.0, ylo))
        d = [f'M{sx(pts[0][0]):.1f},{base:.1f}']
        for x, y in pts:
            d.append(f'L{sx(x):.1f},{sy(y):.1f}')
        d.append(f'L{sx(pts[-1][0]):.1f},{base:.1f}Z')
        p.append(f'<path d="{" ".join(d)}" fill="{color}" fill-opacity="0.35" '
                 f'stroke="{color}" stroke-width="{width:.2g}"/>')
        return
    if kind == "bar":
        if len(pts) > 1:
            bw = (sx(s["x"][1]) - sx(s["x"][0])) * 0.78
        else:
            bw = 24
        base = sy(max(0, ylo))
        for x, y in pts:
            X, Y = sx(x), sy(y)
            top = min(Y, base)
            h = abs(base - Y)
            p.append(f'<rect x="{X-bw/2:.1f}" y="{top:.1f}" width="{bw:.1f}" '
                     f'height="{h:.1f}" fill="{color}" stroke="#262626" stroke-width="0.6"/>')
        return
    if kind == "stem":
        base = sy(max(0, ylo))
        for x, y in pts:
            X, Y = sx(x), sy(y)
            p.append(f'<line x1="{X:.1f}" y1="{base:.1f}" x2="{X:.1f}" y2="{Y:.1f}" stroke="{color}"/>')
            p.append(f'<circle cx="{X:.1f}" cy="{Y:.1f}" r="3" fill="none" stroke="{color}"/>')
        return
    # line / scatter
    if kind != "scatter" and (s["style"] or not s["marker"]):
        dash = {"--": "7,4", ":": "2,4", "-.": "8,3,2,3"}.get(s["style"], "")
        d = " ".join(f'{"M" if i == 0 else "L"}{sx(x):.1f},{sy(y):.1f}' for i, (x, y) in enumerate(pts))
        dattr = f' stroke-dasharray="{dash}"' if dash else ""
        p.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width:.2g}" '
                 f'stroke-linejoin="round"{dattr}/>')
    marker = s["marker"] or ("o" if kind == "scatter" else None)
    if marker:
        for x, y in pts:
            _draw_marker(p, marker, sx(x), sy(y), color, msize)


def _draw_marker(p, m, x, y, color, msize=5.0):
    r = msize * 0.7
    if m == "o":
        p.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="none" stroke="{color}" stroke-width="1.6"/>')
    elif m == ".":
        p.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2" fill="{color}"/>')
    elif m == "s":
        p.append(f'<rect x="{x-r:.1f}" y="{y-r:.1f}" width="{2*r}" height="{2*r}" fill="none" stroke="{color}" stroke-width="1.6"/>')
    elif m in "+x":
        a = r
        rot = ' transform="rotate(45 %.1f %.1f)"' % (x, y) if m == "x" else ""
        p.append(f'<g stroke="{color}" stroke-width="1.6"{rot}>'
                 f'<line x1="{x-a:.1f}" y1="{y:.1f}" x2="{x+a:.1f}" y2="{y:.1f}"/>'
                 f'<line x1="{x:.1f}" y1="{y-a:.1f}" x2="{x:.1f}" y2="{y+a:.1f}"/></g>')
    elif m == "*":
        p.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.5" fill="{color}"/>')
    else:  # d ^ v -> small diamond
        p.append(f'<rect x="{x-r:.1f}" y="{y-r:.1f}" width="{2*r}" height="{2*r}" '
                 f'fill="{color}" transform="rotate(45 {x:.1f} {y:.1f})"/>')



# ===========================================================================
# ECE08 signal-processing library — mdsp wired into the interpreter tables
# (Signals, Spectra and Signal Processing: chapters 1-7 laboratory set)
# ===========================================================================
_PLOT_FUNCS |= {"stairs", "area", "semilogy", "loglog"}

# DSP commands that DRAW when called bare (they also appear in _MULTI_FUNCS
# so that [h, t] = impz(...) returns data instead of plotting).
_DSP_PLOTS = {"impz", "stepz", "zplane", "freqs", "grpdelay", "spectrogram"}


def _norm_out(v):
    """1-element vectors coming back from mdsp display as MATLAB scalars."""
    if isinstance(v, tuple):
        v = list(v)
    if isinstance(v, list) and len(v) == 1 and not isinstance(v[0], (list, tuple)):
        return v[0]
    return v


def _dsp1(fn, pick=None):
    """Single-output wrapper for an mdsp function."""
    def call(args):
        try:
            res = fn(args)
        except mdsp.DspError as ex:
            raise MError(str(ex))
        if pick is not None:
            res = list(res)[pick]
        return _norm_out(res)
    return call


def _dspn(fn):
    """Multi-output wrapper: [a, b, ...] = f(...)"""
    def call(args):
        try:
            res = fn(args)
        except mdsp.DspError as ex:
            raise MError(str(ex))
        return [_norm_out(v) for v in list(res)]
    return call


def _iir_single(name):
    def call(args):
        raise MError(f"{name} returns two values — write  [b, a] = {name}(...)  "
                     f"and then use filter(b, a, x) or freqz(b, a)")
    return call


def _freq_args(name, args):
    """MATLAB's freqz/grpdelay argument forms -> (b, a, w_rad, fs).
    (b,a) (b,a,n) (b,a,n,'whole') (b,a,n,fs) (b,a,n,'whole',fs)
    (b,a,w) (b,a,f,fs) — a vector third argument is the frequency grid
    (rad/sample, or Hz when fs is given). fs is None when not given."""
    _need(name, args, 1)
    b = as_list(args[0])
    a = as_list(args[1]) if len(args) > 1 and not isinstance(args[1], str) \
        and as_list(args[1]) else [1.0]
    whole, nums = False, []
    for v in args[2:]:
        if isinstance(v, str):
            if v.lower() == "whole":
                whole = True
            elif v.lower() != "half":
                raise MError(f"{name}: unknown option '{v}'")
        else:
            nums.append(v)
    if len(nums) > 2:
        raise MError(f"{name}: too many numeric arguments")
    fs = float(_cr(scalar(nums[1]))) if len(nums) > 1 else None
    grid = as_list(nums[0]) if nums else []
    if len(grid) > 1:
        f = [float(_cr(v)) for v in grid]
        ws = [2 * math.pi * v / fs for v in f] if fs else f
        return b, a, ws, fs
    n = int(round(_cr(grid[0]))) if grid else 512
    if n < 1:
        raise MError(f"{name}: the number of points must be positive")
    n = min(n, 65536)
    top = 2 * math.pi if whole else math.pi
    return b, a, [top * k / n for k in range(n)], fs


def _freq_out(ws, fs):
    """rad/sample -> Hz when the caller gave a sample rate."""
    return [w * fs / (2 * math.pi) for w in ws] if fs else list(ws)


def _mf_freqz(args):
    b, a, ws, fs = _freq_args("freqz", args)
    _, H = mdsp.freqz(b, a, ws=ws)
    return [H, _freq_out(ws, fs)]


def _mf_freqs(args):
    b = as_list(args[0])
    a = as_list(args[1]) if len(args) > 1 else [1.0]
    wv = as_list(args[2]) if len(args) > 2 else None
    w, H = mdsp.freqs(b, a, wv)
    return [H, w]


def _mf_grpdelay(args):
    b, a, ws, fs = _freq_args("grpdelay", args)
    _, gd = mdsp.grpdelay(b, a, ws=ws)
    return [gd, _freq_out(ws, fs)]


# --- Ch1: basic signal builders -------------------------------------------
_ELEMENTWISE["heaviside"] = mdsp.heaviside
_ELEMENTWISE["dirac"] = mdsp.dirac
_ELEMENTWISE["isnan"] = lambda x: 1.0 if (isinstance(x, float) and math.isnan(x)) else 0.0
_ELEMENTWISE["isinf"] = lambda x: 1.0 if (isinstance(x, float) and math.isinf(x)) else 0.0
_ELEMENTWISE["isfinite"] = lambda x: 0.0 if (isinstance(x, float) and not math.isfinite(x)) else 1.0
_MATFUNCS_PRED = {
    "isnumeric": lambda args: 0.0 if isinstance(args[0], (str, dict, tuple)) else 1.0,
    "ischar": lambda args: 1.0 if isinstance(args[0], str) else 0.0,
    "isreal": lambda args: 0.0 if any(isinstance(x, complex) and x.imag != 0
                                      for x in as_list(args[0])) else 1.0,
    "islogical": lambda args: 1.0 if all(x in (0.0, 1.0) for x in as_list(args[0])) else 0.0,
    "isa": lambda args: 1.0 if (isinstance(args[1], str) and args[1] in ("double", "numeric", "float")
                              and not isinstance(args[0], (str, dict, tuple))) else 0.0,
}

# --- windows (Ch5) ---------------------------------------------------------
_BUILDERS["kaiser"] = _dsp1(mdsp.kaiser)
_BUILDERS["flattopwin"] = _dsp1(mdsp.flattopwin)

_VECTOR_FUNCS.update({
    # Ch1 — signal builders
    "stepfun": _dsp1(mdsp.stepfun),
    "rectpuls": _dsp1(mdsp.rectpuls),
    "tripuls": _dsp1(mdsp.tripuls),
    "gauspuls": _dsp1(mdsp.gauspuls),
    "pulstran": _dsp1(mdsp.pulstran),
    # Ch3 — sampling, rate change, quantization, companding
    "decimate": _dsp1(mdsp.decimate),
    "interp": _dsp1(mdsp.interp_),
    "resample": _dsp1(mdsp.resample),
    "uquantize": _dsp1(mdsp.uquantize),
    "sqnr": _dsp1(mdsp.sqnr),
    "compand": _dsp1(mdsp.compand),
    "lin2mu": _dsp1(mdsp.lin2mu),
    "mu2lin": _dsp1(mdsp.mu2lin),
    "quantiz": _dsp1(mdsp.quantiz, pick=0),
    # Ch4 — z transform
    "tf2zp": _dsp1(mdsp.tf2zp, pick=0),
    "zp2tf": _dsp1(mdsp.zp2tf, pick=0),
    "residuez": _dsp1(mdsp.residuez, pick=0),
    "tf2latc": _dsp1(mdsp.tf2latc),
    "latc2tf": _dsp1(mdsp.latc2tf),
    "bilinear": _dsp1(mdsp.bilinear_ba, pick=0),
    "impinvar": _dsp1(mdsp.impinvar, pick=0),
    # Ch5 — DFT / FFT tools
    "cconv": _dsp1(mdsp.cconv),
    "fftfilt": _dsp1(mdsp.fftfilt),
    "bitrevorder": _dsp1(mdsp.bitrevorder),
    "dftmtx_row": _dsp1(mdsp.dftmtx_row),
    # Ch6 — filter design
    "fir1": _dsp1(mdsp.fir1),
    "fir2": _dsp1(mdsp.fir2),
    "firls": _dsp1(mdsp.firls),
    "firpm": _dsp1(mdsp.firpm),
    "filtfilt": _dsp1(mdsp.filtfilt),
    "tf2sos": _dsp1(mdsp.tf2sos),
    "zp2sos": _dsp1(mdsp.zp2sos),
    "sos2tf": _dsp1(mdsp.sos2tf, pick=0),
    "sosfilt": _dsp1(mdsp.sosfilt),
    "buttord": _dsp1(mdsp.buttord, pick=0),
    "cheb1ord": _dsp1(mdsp.cheb1ord, pick=0),
    "cheb2ord": _dsp1(mdsp.cheb2ord, pick=0),
    "ellipord": _dsp1(mdsp.ellipord, pick=0),
    "kaiserord": _dsp1(mdsp.kaiserord, pick=0),
    "firpmord": _dsp1(mdsp.firpmord, pick=0),
    "butter": _iir_single("butter"),
    "cheby1": _iir_single("cheby1"),
    "cheby2": _iir_single("cheby2"),
    "ellip": _iir_single("ellip"),
    "besself": _iir_single("besself"),
    # Ch7 — random signals & correlation
    "awgn": _dsp1(mdsp.awgn),
    "snr": _dsp1(mdsp.snr_db),
    "xcov": _dsp1(mdsp.xcov),
    "autocorr": _dsp1(mdsp.autocorr_norm),
    "levinson": _dsp1(mdsp.levinson, pick=0),
    "lpc": _dsp1(mdsp.lpc, pick=0),
})

def _mf_extreme(pick):
    """[m, i] = max(x) / min(x) — value and its 1-based index."""
    def call(args):
        v = as_list(args[0])
        if not v:
            raise MError("max/min: the vector is empty")
        m = _reduce_max(v) if pick == "max" else _reduce_min(v)
        return [m, float(v.index(m) + 1)]
    return call


def _mf_sort(args):
    v = as_list(args[0])
    order = sorted(range(len(v)), key=lambda i: _sort_key(v)(v[i]))
    return [[v[i] for i in order], [float(i + 1) for i in order]]


_MULTI_FUNCS.update({
    "max": _mf_extreme("max"),
    "min": _mf_extreme("min"),
    "sort": _mf_sort,
})

_MULTI_FUNCS.update({
    # [b, a] = design(...)
    "butter": _dspn(mdsp.butter),
    "cheby1": _dspn(mdsp.cheby1),
    "cheby2": _dspn(mdsp.cheby2),
    "ellip": _dspn(mdsp.ellip),
    "besself": _dspn(mdsp.besself),
    "bilinear": _dspn(mdsp.bilinear_ba),
    "impinvar": _dspn(mdsp.impinvar),
    "zp2tf": _dspn(mdsp.zp2tf),
    "sos2tf": _dspn(mdsp.sos2tf),
    # [n, Wn] = order estimate
    "buttord": _dspn(mdsp.buttord),
    "cheb1ord": _dspn(mdsp.cheb1ord),
    "cheb2ord": _dspn(mdsp.cheb2ord),
    "ellipord": _dspn(mdsp.ellipord),
    "kaiserord": _dspn(mdsp.kaiserord),
    "firpmord": _dspn(mdsp.firpmord),
    # [z, p, k] / [r, p, k]
    "tf2zp": _dspn(mdsp.tf2zp),
    "residuez": _dspn(mdsp.residuez),
    # responses as data
    "impz": _dspn(mdsp.impz),
    "stepz": _dspn(mdsp.stepz),
    "freqz": _dspn(_mf_freqz),
    "freqs": _dspn(_mf_freqs),
    "grpdelay": _dspn(_mf_grpdelay),
    "quantiz": _dspn(mdsp.quantiz),
    # Ch7
    "levinson": _dspn(mdsp.levinson),
    "lpc": _dspn(mdsp.lpc),
})



# ===========================================================================
# 2-D matrices: concatenation, previews, function tables, 3-D plot names
# ===========================================================================
def _hcat(pieces):
    """[a b c] — horizontal concatenation of one bracket row."""
    flat = []
    for x in pieces:                      # c{1:3} expands in place
        if isinstance(x, _CSList):
            flat.extend(x)
        else:
            flat.append(x)
    pieces = [x for x in flat if not (isinstance(x, list) and len(x) == 0)]
    if not pieces:
        return []
    if any(is_cell(x) for x in pieces):
        try:
            return mtypes.cellcat(pieces)
        except mtypes.MTypeError as ex:
            raise MError(str(ex))
    if all(isinstance(x, str) for x in pieces):
        return "".join(pieces)
    if any(isinstance(x, str) for x in pieces):
        # ['abc' 10 'def'] — numbers become characters, like MATLAB's char concatenation
        return "".join(x if isinstance(x, str) else
                       "".join(chr(int(round(_cr(v)))) for v in as_list(x)) for x in pieces)
    if not any(is_mat(x) for x in pieces):
        out = []
        for x in pieces:
            out.extend(as_list(x))
        return out
    mats = [mmat.to_mat(x) for x in pieces]
    r = mats[0].r
    for m in mats:
        if m.r != r:
            raise MError("horizontal concatenation: rows must match")
    return mmat.norm_val(Mat([sum((m.d[i] for m in mats), []) for i in range(r)]))


def _vcat(rows):
    """[row1; row2; ...] — vertical concatenation."""
    rows = [x for x in rows if not (isinstance(x, list) and len(x) == 0)]
    if not rows:
        return []
    if len(rows) == 1:
        return rows[0]
    if any(isinstance(x, str) for x in rows):
        raise MError("character matrices are not supported; use one string per variable")
    mats = [mmat.to_mat(x) for x in rows]
    c = mats[0].c
    for m in mats:
        if m.c != c:
            raise MError("vertical concatenation: columns must match")
    return mmat.norm_val(Mat([list(rw) for m in mats for rw in m.d]))


def _json_num(x):
    if isinstance(x, complex):
        return _fmt_num(x)
    if isinstance(x, float) and not math.isfinite(x):
        return _fmt_num(x)
    return x


def _preview(val, limit=600):
    """Small numeric values travel to the UI so the Variable Editor can show them."""
    if isinstance(val, str):
        return val
    if is_cell(val):
        # a grid of one-line summaries: the UI shows it read-only
        return {"cell": [[mtypes.summarize(v, _fmt_num) for v in row] for row in val.d]}
    if is_struct(val):
        return {"struct": [[k, mtypes.summarize(v, _fmt_num)] for k, v in val.f.items()]}
    if _is_tf(val) or _is_func(val) or val is None or isinstance(val, dict):
        return None
    if is_mat(val):
        rows = val.d
    elif is_vec(val):
        rows = [val]
    else:
        rows = [[val]]
    if sum(len(r) for r in rows) > limit:
        return None
    return [[_json_num(x) for x in r] for r in rows]


def _dims_from_args(args, name):
    """zeros(n) -> (n, n); zeros(m, n); zeros([m n]); zeros(size(A))."""
    if not args:
        return 1, 1
    if len(args) == 1:
        v = args[0]
        if is_vec(v) or is_mat(v):
            d = [int(round(_cr(x))) for x in as_list(v)]
            if len(d) >= 2:
                return d[0], d[1]
            return d[0], d[0]
        n = int(round(_cr(scalar(v))))
        return n, n
    return int(round(_cr(scalar(args[0])))), int(round(_cr(scalar(args[1]))))


def _norm_builder(v):
    """A builder called with no arguments makes a 1x1 — show it as a scalar."""
    return mmat.norm_val(v) if is_mat(v) else v


def _mk_fill(val):
    def f(args):
        r, c = _dims_from_args(args, "zeros")
        return mmat.fill(max(r, 0), max(c, 0), val)
    return f


_RNG = {"seed": 0.0, "type": "twister"}


def _seed(n):
    _RNG["seed"] = float(n)
    random.seed(int(n))



def _mk_rand(fn):
    def f(args):
        # legacy seeding: rand('seed', n), randn('state', n), rand('twister', n)
        if args and isinstance(args[0], str) and args[0].lower() in ("seed", "state", "twister"):
            if len(args) < 2:
                return _RNG["seed"]
            _seed(int(round(_cr(scalar(args[1])))))
            return None
        while args and isinstance(args[-1], str):        # rand(3, 'double')
            if args[-1].lower() not in ("double", "single"):
                raise MError(f"rand: unknown option '{args[-1]}'")
            args = args[:-1]
        r, c = _dims_from_args(args, "rand")
        return mmat.fill_fn(max(r, 0), max(c, 0), fn)
    return f


def _rng(args):
    """rng(seed), rng(seed, gen), rng('default'), rng('shuffle'), s = rng, rng(s)."""
    prev = mtypes.Struct({"Type": _RNG["type"], "Seed": _RNG["seed"], "State": _RNG["seed"]})
    if not args:
        return prev
    v = args[0]
    if is_struct(v):
        _seed(int(round(_cr(scalar(v.f.get("Seed", 0.0))))))
    elif isinstance(v, str):
        if v.lower() == "default":
            _RNG["type"] = "twister"
            _seed(0)
        elif v.lower() == "shuffle":
            n = random.SystemRandom().randrange(2 ** 32)
            _seed(n)
        else:
            raise MError(f"rng: unknown option '{v}'")
    else:
        n = int(round(_cr(scalar(v))))
        if n < 0:
            raise MError("rng: the seed must be a non-negative integer")
        _seed(n)
    if len(args) > 1 and isinstance(args[1], str):
        _RNG["type"] = args[1]
    return prev


def _double(args):
    _need("double", args, 1)
    v = args[0]
    if isinstance(v, str):
        codes = [float(ord(ch)) for ch in v]
        return codes[0] if len(codes) == 1 else codes
    if is_cell(v) or is_struct(v) or _is_func(v) or _is_tf(v):
        raise MError(f"Conversion to double from {_class_of([v])} is not possible.")
    return v


_BUILDERS.update({
    "zeros": _mk_fill(0.0), "ones": _mk_fill(1.0),
    "nan": _mk_fill(math.nan), "NaN": _mk_fill(math.nan),
    "inf": _mk_fill(math.inf), "Inf": _mk_fill(math.inf),
    "true": _mk_fill(1.0), "false": _mk_fill(0.0),
    "rand": _mk_rand(random.random), "randn": _mk_rand(lambda: random.gauss(0.0, 1.0)),
    "rng": _rng,
})


def _size_fn(args):
    _need("size", args, 1)
    r, c = mmat.shape(args[0])
    if len(args) > 1:
        d = int(round(_cr(scalar(args[1]))))
        return float(r) if d == 1 else (float(c) if d == 2 else 1.0)
    return [float(r), float(c)]


def _find_fn(args):
    _need("find", args, 1)
    v = args[0]
    if is_mat(v) and v.r > 1:
        rows, cols = mmat.find_rc(v)
        r, c = as_list(rows), as_list(cols)
        return mmat.colvec([(cc - 1) * v.r + rr for rr, cc in zip(r, c)])
    idx = [float(k + 1) for k, x in enumerate(as_list(v)) if x != 0]
    if len(args) > 1:
        idx = idx[:int(_cr(scalar(args[1])))]
    return idx


def _eye_fn(args):
    if not args:
        return 1.0
    r, c = _dims_from_args(args, "eye")
    return mmat.eye(r, c)


def _linspace_col(args):
    return None


_MATFUNCS = {
    **_MATFUNCS_PRED,
    "size": _size_fn,
    "numel": lambda args: float(mmat.numel(args[0])),
    "length": lambda args: float(max(mmat.shape(args[0])) if mmat.numel(args[0]) else 0),
    "ndims": lambda args: 2.0,
    "isempty": lambda args: 1.0 if mmat.numel(args[0]) == 0 else 0.0,
    "isscalar": lambda args: 1.0 if mmat.shape(args[0]) == (1, 1) else 0.0,
    "isvector": lambda args: 1.0 if 1 in mmat.shape(args[0]) else 0.0,
    "isrow": lambda args: 1.0 if mmat.shape(args[0])[0] == 1 else 0.0,
    "iscolumn": lambda args: 1.0 if mmat.shape(args[0])[1] == 1 else 0.0,
    "ismatrix": lambda args: 1.0,
    "issquare": lambda args: 1.0 if mmat.shape(args[0])[0] == mmat.shape(args[0])[1] else 0.0,
    "nnz": lambda args: float(sum(1 for x in as_list(args[0]) if x != 0)),
    "find": _find_fn,
    "eye": _eye_fn,
    "magic": lambda args: mmat.magic(_cr(scalar(args[0]))),
    "inv": lambda args: mmat.inv(args[0]),
    "pinv": lambda args: mmat.inv(args[0]),
    "det": lambda args: mmat.det(args[0]),
    "trace": lambda args: mmat.trace(args[0]),
    "rank": lambda args: mmat.rank(args[0]),
    "transpose": lambda args: mmat.transpose(args[0]),
    "ctranspose": lambda args: mmat.ctranspose(args[0]),
    "norm": lambda args: mmat.norm(args[0], args[1] if len(args) > 1 else 2),
    "reshape": lambda args: mmat.reshape(
        args[0],
        None if (len(args) > 1 and isinstance(args[1], list) and len(args[1]) == 0) else
        (int(_cr(scalar(args[1]))) if len(args) == 3 else int(_cr(as_list(args[1])[0]))),
        None if (len(args) > 2 and isinstance(args[2], list) and len(args[2]) == 0) else
        (int(_cr(scalar(args[2]))) if len(args) == 3 else int(_cr(as_list(args[1])[1])))),
    "repmat": lambda args: mmat.repmat(args[0], _cr(scalar(args[1])),
                                       _cr(scalar(args[2])) if len(args) > 2 else _cr(scalar(args[1]))),
    "diag": lambda args: mmat.diag(args[0], int(_cr(scalar(args[1]))) if len(args) > 1 else 0),
    "triu": lambda args: mmat.triu(args[0], int(_cr(scalar(args[1]))) if len(args) > 1 else 0),
    "tril": lambda args: mmat.tril(args[0], int(_cr(scalar(args[1]))) if len(args) > 1 else 0),
    "kron": lambda args: mmat.kron(args[0], args[1]),
    "cross": lambda args: mmat.cross(args[0], args[1]),
    "mtimes": lambda args: mmat.matmul(args[0], args[1]),
    "mldivide": lambda args: mmat.solve(args[0], args[1]),
    "linsolve": lambda args: mmat.solve(args[0], args[1]),
    "mpower": lambda args: mmat.mpower(args[0], _cr(scalar(args[1]))),
    "eig": lambda args: mmat.colvec(mmat.eig(args[0])),
    "meshgrid": lambda args: mmat.meshgrid(args[0], args[1] if len(args) > 1 else None)[0],
    "peaks": lambda args: mmat.peaks(args[0] if args else 49)[2],
    "sphere": lambda args: mmat.sphere(_cr(scalar(args[0])) if args else 20)[2],
    "cylinder": lambda args: mmat.cylinder(args[0] if args else 1.0,
                                           _cr(scalar(args[1])) if len(args) > 1 else 20)[2],
    "fliplr": lambda args: (mmat.norm_val(Mat([list(reversed(rw)) for rw in args[0].d]))
                            if is_mat(args[0]) else list(reversed(as_list(args[0])))),
    "flipud": lambda args: (mmat.norm_val(Mat(list(reversed(args[0].d))))
                            if is_mat(args[0]) else list(as_list(args[0]))),
    "flip": lambda args: (mmat.norm_val(Mat(list(reversed(args[0].d))))
                          if is_mat(args[0]) and args[0].r > 1
                          else list(reversed(as_list(args[0])))),
    "rot90": lambda args: mmat.norm_val(Mat(list(reversed(mmat.to_mat(args[0]).T().d)))),
    "horzcat": lambda args: _hcat(list(args)),
    "vertcat": lambda args: _vcat(list(args)),
    "cat": lambda args: (_hcat(list(args[1:])) if _cr(scalar(args[0])) == 2 else _vcat(list(args[1:]))),
    "dot": lambda args: sum(x * (y.conjugate() if isinstance(y, complex) else y)
                            for x, y in zip(as_list(args[0]), as_list(args[1]))),
    "colon": lambda args: make_range([_cr(scalar(a)) for a in args]),
}

_COL_VEC = {"cumsum": _cumsum, "cumprod": _cumprod,
            "sort": lambda v: sorted(v, key=_sort_key(v))}


def _reduce_call(name, args):
    """sum/mean/max/... with MATLAB matrix (column-wise) and dim semantics.
    Returns None to fall back to the plain row-vector reducer."""
    a = args[0]
    if name in ("max", "min"):
        if len(args) >= 2 and not (isinstance(args[1], list) and len(args[1]) == 0):
            f = ((lambda x, y: x if _cr(x) >= _cr(y) else y) if name == "max"
                 else (lambda x, y: x if _cr(x) <= _cr(y) else y))
            return elementwise(a, args[1], f)
        dim = int(_cr(scalar(args[2]))) if len(args) >= 3 else None
    elif name in ("std", "var"):
        dim = int(_cr(scalar(args[2]))) if len(args) >= 3 else None
    elif name in ("sort",) and len(args) >= 2 and isinstance(args[1], str):
        v = as_list(a)
        out = sorted(v, key=_sort_key(v), reverse=(args[1].lower() == "descend"))
        return mmat.colvec(out) if (is_mat(a) and a.c == 1) else out
    else:
        dim = int(_cr(scalar(args[1]))) if len(args) >= 2 else None
    if not (is_mat(a) or dim is not None):
        return None
    if name in _COL_VEC:
        return mmat.colwise_vec(a, _COL_VEC[name], dim)
    if name in _REDUCERS:
        return mmat.colwise(a, _REDUCERS[name], dim)
    return None


# multi-output forms
def _mf_meshgrid(args):
    X, Y = mmat.meshgrid(args[0], args[1] if len(args) > 1 else None)
    return [X, Y]


def _mf_peaks(args):
    return list(mmat.peaks(args[0] if args else 49))


def _mf_eig(args):
    vals, V = mmat.eig(args[0], want_vectors=True)
    return [V, mmat.diag(vals)]


def _mf_find(args):
    r, c = mmat.find_rc(args[0])
    return [r, c]


def _mf_extreme_mat(pick):
    base = _mf_extreme(pick)

    def call(args):
        a = args[0]
        if is_mat(a) and a.r > 1 and a.c > 1:
            vals, idx = [], []
            for j in range(a.c):
                m, i = base([a.col(j)])
                vals.append(m)
                idx.append(i)
            return [vals, idx]
        if is_mat(a):
            m, i = base([as_list(a)])
            return [m, i]
        return base(args)
    return call


_MULTI_FUNCS.update({
    "size": lambda args: [float(x) for x in mmat.shape(args[0])],
    "meshgrid": _mf_meshgrid,
    "peaks": _mf_peaks,
    "sphere": lambda args: list(mmat.sphere(_cr(scalar(args[0])) if args else 20)),
    "cylinder": lambda args: list(mmat.cylinder(args[0] if args else 1.0,
                                                _cr(scalar(args[1])) if len(args) > 1 else 20)),
    "eig": _mf_eig,
    "find": _mf_find,
    "max": _mf_extreme_mat("max"),
    "min": _mf_extreme_mat("min"),
})

_PLOT3_FUNCS = {"surf", "surfc", "surfl", "mesh", "meshc", "waterfall", "plot3", "scatter3",
                "stem3", "contour", "contourf", "contour3", "pcolor", "imagesc", "image"}

# numerical simulation — dispatched through Interp._sim_call (needs the interpreter
# to call back into user functions)
_SIM_FUNCS = set(msim.ODE_METHODS) | {"odeset", "fzero", "fsolve", "fminsearch", "fminunc",
                                      "fminbnd", "integral", "quad", "quadgk", "quadl", "quadv",
                                      "expm", "cumtrapz", "gradient", "polyder", "polyint"}

# MATLAB circuit script functions (Circuit Lab language) — Interp._ckt_call
_CKT_FUNCS = {"circuit", "netlist", "resistor", "capacitor", "inductor", "lamp", "voltage", "current",
              "diode", "zener", "led", "bjt", "mosfet", "jfet", "opamp", "comparator", "vcvs", "vccs", "sw", "pot",
              "transformer", "multimeter", "voltmeter", "ammeter", "ohmmeter", "oscilloscope", "fgen",
              "bodeplotter", "wattmeter",
              "transient", "acsweep", "dcop", "dcsweep", "stepparam", "fourier", "probe", "simulate"}


def run_demo_code(code):
    """Entry point: returns the sidecar response dict for DEMO mode."""
    interp = Interp()
    res = interp.run(code)
    figs = []
    for f in res["figures"]:
        item = {"format": "svg", "data": f["svg"]}
        if f.get("frames"):
            item["frames"] = f["frames"]
            item["delays"] = f["delays"]
        figs.append(item)
    return {
        "ok": res["error"] is None,
        "mode": "demo",
        "stdout": res["stdout"],
        "stderr": "",
        "error": res["error"],
        "figures": figs,
        "workspace": res["workspace"],
    }


# ---------------------------------------------------------------------------
# Cells, structs, strings and type predicates — mtypes wired into the tables.
#
# _TYPE_FUNCS always wins (these names mean nothing numeric).
# _CONTAINER_FUNCS only wins when an argument IS a cell or a struct, so
# numel/size/class keep their fast numeric path for ordinary arrays.
# ---------------------------------------------------------------------------

def _need_str(name, args, n):
    _need(name, args, n)


def _cell_size(args):
    v = args[0]
    r, c = (v.r, v.c) if is_cell(v) else (1, 1)
    if len(args) > 1:
        d = int(_cr(scalar(args[1])))
        return float(r if d == 1 else (c if d == 2 else 1))
    return [float(r), float(c)]


def _class_of(args):
    return mtypes.class_of(args[0], is_mat, _is_tf, _is_func)


def _isa(args):
    want = scalar_str(args[1])
    got = _class_of([args[0]])
    if want == "numeric":
        return 1.0 if got in ("double", "complex double", "logical") else 0.0
    if want == "float":
        return 1.0 if got in ("double", "complex double") else 0.0
    return 1.0 if got == want else 0.0


def _num2cell(args):
    vals = as_list(args[0])
    return Cell([vals]) if vals else Cell([])


def _cell2mat(args):
    c = args[0]
    if not is_cell(c):
        raise MError("cell2mat needs a cell array")
    rows = []
    for row in c.d:
        flat = []
        for v in row:
            flat.extend(as_list(v))
        rows.append(flat)
    if len(rows) == 1:
        return rows[0]
    return mmat.Mat(rows)


def _iscellstr(args):
    c = args[0]
    return 1.0 if is_cell(c) and all(isinstance(v, str) for v in c.flat()) else 0.0


def _sort_cellstr(args):
    c = args[0]
    if not (is_cell(c) and all(isinstance(v, str) for v in c.flat())):
        return None
    return Cell([sorted(c.flat())])


_TYPE_FUNCS = {
    # containers
    "cell": lambda a: mtypes.cell_of(scalar(a[0]), scalar(a[1]) if len(a) > 1 else scalar(a[0])),
    "struct": lambda a: mtypes.struct_new(a),
    "fieldnames": lambda a: mtypes.fieldnames(a[0]),
    "isfield": lambda a: mtypes.isfield(a[0], a[1]),
    "rmfield": lambda a: mtypes.rmfield(a[0], scalar_str(a[1])),
    "setfield": lambda a: mtypes.setfield(a[0], scalar_str(a[1]), a[2]),
    "getfield": lambda a: mtypes.getfield(a[0], scalar_str(a[1])),
    "orderfields": lambda a: mtypes.orderfields(a[0]),
    "num2cell": _num2cell,
    "cell2mat": _cell2mat,
    "iscellstr": _iscellstr,
    "iscell": lambda a: 1.0 if is_cell(a[0]) else 0.0,
    "isstruct": lambda a: 1.0 if is_struct(a[0]) else 0.0,
    # strings
    "strsplit": lambda a: mtypes.strsplit(a[0], a[1] if len(a) > 1 else None),
    "strjoin": lambda a: mtypes.strjoin(a[0], scalar_str(a[1]) if len(a) > 1 else " "),
    "strtrim": lambda a: mtypes.strtrim(a[0]),
    "strrep": lambda a: mtypes.strrep(a[0], a[1], a[2]),
    "strfind": lambda a: mtypes.strfind(a[0], a[1]),
    "strcat": mtypes.strcat,
    "strcmp": lambda a: mtypes._cmp_str(a[0], a[1]),
    "strcmpi": lambda a: mtypes._cmp_str(a[0], a[1], fold=True),
    "strncmp": lambda a: mtypes._cmp_str(a[0], a[1], n=scalar(a[2])),
    "strncmpi": lambda a: mtypes._cmp_str(a[0], a[1], n=scalar(a[2]), fold=True),
    "contains": lambda a: mtypes.contains(a[0], a[1]),
    "startsWith": lambda a: mtypes.startswith(a[0], a[1]),
    "endsWith": lambda a: mtypes.endswith(a[0], a[1]),
    "regexprep": lambda a: mtypes.regexprep(a[0], a[1], a[2]),
    "regexp": lambda a: mtypes.regexp_match(
        a[0], a[1], scalar_str(a[2]) if len(a) > 2 else "match"),
    "upper": lambda a: mtypes.upper(a[0]),
    "lower": lambda a: mtypes.lower(a[0]),
    "deblank": lambda a: mtypes.deblank(a[0]),
    "blanks": lambda a: mtypes.blanks(scalar(a[0])),
    "str2double": lambda a: mtypes.str2double(a[0]),
    "ischar": lambda a: 1.0 if isinstance(a[0], str) else 0.0,
    "isstring": lambda a: 1.0 if isinstance(a[0], str) else 0.0,
    "class": _class_of,
    "isa": _isa,
    "double": _double,
}
# `upper`/`lower` on a char matrix would collide with nothing numeric, but
# `sort` on a cellstr must not reach the numeric reducer.
_CONTAINER_FUNCS = {
    "numel": lambda a: float(a[0].r * a[0].c) if is_cell(a[0]) else 1.0,
    "length": lambda a: float(max(a[0].r, a[0].c)) if is_cell(a[0]) else 1.0,
    "size": _cell_size,
    "isempty": lambda a: 1.0 if (is_cell(a[0]) and a[0].r * a[0].c == 0) else 0.0,
    "isscalar": lambda a: 1.0 if is_struct(a[0]) or (is_cell(a[0]) and a[0].r * a[0].c == 1) else 0.0,
    "isvector": lambda a: 1.0 if is_cell(a[0]) and 1 in (a[0].r, a[0].c) else 0.0,
    "ndims": lambda a: 2.0,
    "sort": lambda a: _sort_cellstr(a),
    "transpose": lambda a: Cell([list(col) for col in zip(*a[0].d)]) if is_cell(a[0]) else a[0],
}
_TYPE_FUNCS["isnumeric"] = lambda a: 1.0 if mtypes.numeric_like(a[0]) and not isinstance(a[0], str) else 0.0
_TYPE_FUNCS["islogical"] = lambda a: 0.0
_TYPE_FUNCS["isreal"] = lambda a: 0.0 if _has_cplx(as_list(a[0])) else 1.0
_TYPE_FUNCS["ismember"] = lambda a: (mtypes.ismember_str(a[0], a[1])
                                     if isinstance(a[0], str) or is_cell(a[1])
                                     else _ismember_num(a))
_TYPE_FUNCS["validatestring"] = lambda a: scalar_str(a[0])


def _ismember_num(args):
    hay = set(round(_cr(x), 12) for x in as_list(args[1]))
    vals = [1.0 if round(_cr(x), 12) in hay else 0.0 for x in as_list(args[0])]
    return vals[0] if len(vals) == 1 else vals


# `exist` answers about variables and functions, which only the interpreter knows.
_KNOWN_TABLES = None


def _known_names(interp):
    global _KNOWN_TABLES
    if _KNOWN_TABLES is None:
        _KNOWN_TABLES = set()
        for tbl in (_ELEMENTWISE, _BINARY, _REDUCERS, _BUILDERS, _VECTOR_FUNCS,
                    _MATFUNCS, _MULTI_FUNCS, _TYPE_FUNCS, _ZERO_ARG):
            _KNOWN_TABLES |= set(tbl)
    return _KNOWN_TABLES | set(interp.funcs)


# ---------------------------------------------------------------------------
# Workspace serialisation (save / load) and the code analyzer.
# ---------------------------------------------------------------------------

def _encode_value(v):
    """A JSON-safe form of any value `save` can round-trip. None = skip it."""
    if isinstance(v, str):
        return {"t": "char", "v": v}
    if is_cell(v):
        rows = [[_encode_value(x) for x in row] for row in v.d]
        if any(x is None for row in rows for x in row):
            return None
        return {"t": "cell", "v": rows}
    if is_struct(v):
        fields = {k: _encode_value(x) for k, x in v.f.items()}
        if any(x is None for x in fields.values()):
            return None
        return {"t": "struct", "v": fields}
    if isinstance(v, complex):
        return {"t": "complex", "v": [v.real, v.imag]}
    if is_mat(v):
        if _has_cplx(as_list(v)):
            return {"t": "cmat", "v": [[[x.real, x.imag] for x in _cx(row)] for row in v.d]}
        return {"t": "mat", "v": [[float(x) for x in row] for row in v.d]}
    if isinstance(v, list):
        if _has_cplx(v):
            return {"t": "cvec", "v": [[complex(x).real, complex(x).imag] for x in v]}
        return {"t": "vec", "v": [float(x) for x in v]}
    if isinstance(v, (int, float)):
        return {"t": "num", "v": float(v)}
    return None                       # tf models, handles, graphics: not saved


def _cx(row):
    return [complex(x) for x in row]


def _decode_value(enc):
    t, v = enc.get("t"), enc.get("v")
    if t == "char":
        return v
    if t == "num":
        return float(v)
    if t == "complex":
        return complex(v[0], v[1])
    if t == "vec":
        return [float(x) for x in v]
    if t == "cvec":
        return [complex(x[0], x[1]) for x in v]
    if t == "mat":
        return mmat.Mat([[float(x) for x in row] for row in v])
    if t == "cmat":
        return mmat.Mat([[complex(x[0], x[1]) for x in row] for row in v])
    if t == "cell":
        return Cell([[_decode_value(x) for x in row] for row in v])
    if t == "struct":
        return Struct({k: _decode_value(x) for k, x in v.items()})
    raise MError(f"unknown saved type '{t}'")


# --- code analyzer ---------------------------------------------------------
_LINT_ASSIGN = re.compile(r"^\s*([A-Za-z_]\w*)\s*=[^=]")
# MATLAB command syntax — `grid on`, `hold off`, `format long`, `close all`,
# `save results`. Bare words after an identifier, with no operator anywhere.
_LINT_COMMAND = re.compile(r"[A-Za-z_]\w*(?:\s+[A-Za-z_0-9.\-/\\~]+)+")
_LINT_LOOP_KW = ("for", "while", "if", "elseif", "else", "end", "switch", "case",
                 "otherwise", "try", "catch", "function", "break", "continue", "return")


_LINT_NEW_STMT = re.compile(r"[A-Za-z_]\w*\s*=(?!=)")


def _lint_logical_lines(lines):
    """(first line no, text, unclosed) per statement line. A '...' continuation
    or a bracket left open (a matrix written over several rows) joins the next
    line, so `A = [1 2` / `3 4];` is checked as one statement. A line that
    clearly starts a new statement (a keyword or `name =`) ends the join, so a
    forgotten ']' is reported instead of hiding the rest of the file."""
    out, buf, start, depth = [], "", 0, 0
    joined_by_dots = False
    for n, raw in enumerate(lines, 1):
        text = strip_comment(raw).strip()
        if not text and not buf:
            continue
        if buf and depth > 0 and not joined_by_dots and text and (
                _kw_of(text) or _LINT_NEW_STMT.match(text)):
            out.append((start, buf.strip(), True))
            buf, depth = "", 0
        if not buf:
            start = n
        # inside brackets a line break is a row separator; after '...' a space
        sep = "\n" if depth > 0 and not joined_by_dots else " "
        cont = text.endswith("...")
        if cont:
            text = text[:-3]
        for _, ch, ins in _iter_chars(text):
            if not ins:
                depth += ch in "([{"
                depth -= ch in ")]}"
        buf = buf + sep + text if buf else text
        joined_by_dots = cont
        if cont or depth > 0:
            continue
        out.append((start, buf.strip(), False))
        buf, depth = "", 0
    if buf.strip():
        out.append((start, buf.strip(), depth > 0))
    return out


def lint(code):
    """mlint-lite: real syntax errors first, then the mistakes that eat lab time."""
    issues = []
    lines = _lf(code).split("\n")

    # 1. block structure
    depth, stack = 0, []
    for n, raw in enumerate(lines, 1):
        text = strip_comment(raw).strip()
        if not text:
            continue
        # per statement, so a one-line `if x, y = 0; end` opens and closes
        for stmt, _ in split_statements(text):
            kw = _kw_of(stmt)
            if kw in ("if", "for", "while", "switch", "try", "function"):
                stack.append((kw, n))
            elif kw == "end":
                if not stack:
                    issues.append({"line": n, "severity": "error",
                                   "msg": "'end' with no matching block"})
                else:
                    stack.pop()
    for kw, n in stack:
        if kw == "function":
            continue                     # a file of functions may end without `end`
        issues.append({"line": n, "severity": "error",
                       "msg": f"'{kw}' is never closed by an 'end'"})

    # 2. per-statement parse, so a typo is reported on its own line
    assigned, used = {}, set()
    for n, text, unclosed in _lint_logical_lines(lines):
        if unclosed:
            issues.append({"line": n, "severity": "error",
                           "msg": "a bracket opened on this line is never closed"})
            continue
        kw = _kw_of(text)
        if kw in _LINT_LOOP_KW:
            if kw in ("if", "while"):
                text = text[len(kw):].strip()
            elif kw == "elseif":
                text = text[6:].strip()
            elif kw == "case":
                text = text[4:].strip()
            elif kw == "switch":
                text = text[6:].strip()
            elif kw == "for":
                m = re.match(r"for\s+[A-Za-z_]\w*\s*=\s*(.+)$", text)
                text = m.group(1) if m else ""
            else:
                text = ""
        if not text:
            continue
        for stmt in split_statements(text):
            src = stmt[0].strip()
            if not src or _kw_of(src):
                continue
            if _LINT_COMMAND.fullmatch(src):
                continue          # command syntax: `grid on`, `hold off`, `save data`
            eq = top_level_assign(src)
            target, expr = (None, src)
            if eq is not None:
                target, expr = src[:eq].strip(), src[eq + 1:].strip()
            try:
                if expr:
                    Parser(tokenize(expr)).parse()
            except MError as ex:
                issues.append({"line": n, "severity": "error", "msg": str(ex)})
                continue
            for name in re.findall(r"(?<![.\w])([A-Za-z_]\w*)\s*(?!\()", expr):
                used.add(name)
            for name in re.findall(r"(?<![.\w])([A-Za-z_]\w*)\s*\(", expr):
                used.add(name)
            m = _LINT_ASSIGN.match(src)
            if m and target and re.fullmatch(r"[A-Za-z_]\w*", target):
                assigned.setdefault(target, n)

    # 3. assigned but never used (MATLAB's most common warning)
    for name, n in assigned.items():
        if name not in used and name != "ans":
            issues.append({"line": n, "severity": "warning",
                           "msg": f"the value assigned to '{name}' is never used"})

    # 4. a loop-growing array — the classic slow-lab-script mistake
    for n, raw in enumerate(lines, 1):
        text = strip_comment(raw).strip()
        if re.search(r"\b([A-Za-z_]\w*)\s*\(\s*end\s*\+\s*1\s*\)\s*=", text):
            issues.append({"line": n, "severity": "info",
                           "msg": "array grows inside a loop — preallocate with zeros() "
                                  "for a faster script"})
    issues.sort(key=lambda x: (x["line"], x["severity"]))
    return issues

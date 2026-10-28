"""
mcircuit — the SimForge circuit engine for the MATLAB interpreter (a port of
src/circuit.js so the same circuit behaves identically in the Editor and in
the Circuit Lab).

Modified Nodal Analysis, backward-Euler companions, Newton–Raphson for the
nonlinear devices.  Devices: R C L, V/I (DC SIN PULSE TRI SQUARE), E (VCVS),
G (VCCS), X (ideal op-amp), D, Z (Zener), LED, Q (BJT Ebers–Moll), M (MOSFET
level 1), S (switch), P (pot), T (ideal transformer), LAMP.
Analyses: tran, op, dcsweep, ac (about the DC bias), .step, .four.

The MATLAB-facing functions (resistor(...), voltage(...), simulate() ...) live
in mdemo.py and build a `Circuit` through this module's API.
"""

import cmath
import math

VT = 0.025852
IS = 1e-14
GMIN = 1e-9


class CktError(ValueError):
    pass


def _err(msg):
    raise CktError(msg)


_SUF = {"f": 1e-15, "p": 1e-12, "n": 1e-9, "u": 1e-6, "m": 1e-3, "k": 1e3, "meg": 1e6, "g": 1e9, "t": 1e12}


def parse_val(s):
    if isinstance(s, (int, float)):
        return float(s)
    import re
    m = re.match(r"^\s*([-+0-9.eE]+)\s*(meg|[fpnumkgtF]?)", str(s))
    if not m:
        _err(f"bad value '{s}'")
    x = float(m.group(1))
    k = m.group(2).lower()
    return x * _SUF.get(k, 1.0)


def _lim(v, vmax=0.9, vmin=-80.0):
    return max(vmin, min(v, vmax))


# ===========================================================================
# Circuit container
# ===========================================================================
class Circuit:
    def __init__(self, title="Circuit"):
        self.title = title
        self.els = []
        self.probes = []
        self.hidden = []                          # simulated but not returned/plotted
        self.analysis = {"type": "tran", "tstep": 1e-5, "tstop": 5e-3, "f0": 10.0, "f1": 1e6, "nppd": 40}
        self.step = None
        self.four = None
        self.instruments = []
        self._final = None

    # --- element API (node names are strings) ------------------------------
    def _n(self, x):
        s = str(x)
        if isinstance(x, float) and x == int(x):
            s = str(int(x))
        return s

    def add(self, el):
        self.els.append(el)
        self._final = None
        return el

    def rlc(self, kind, name, a, b, val, lamp=False, pwr=None):
        return self.add({"type": kind, "name": name, "a": self._n(a), "b": self._n(b), "val": float(val), "lamp": lamp,
                         "pwr": float(pwr) if pwr is not None else (1.0 if lamp else 0.25)})

    def source(self, kind, name, p, n, spec, params):
        src = {"kind": "DC", "dc": 0.0}
        if isinstance(spec, (int, float)):
            src = {"kind": "DC", "dc": float(spec)}
        else:
            k = str(spec).upper()
            pr = list(params) if params is not None else []
            if k == "DC":
                src = {"kind": "DC", "dc": float(pr[0]) if pr else 0.0}
            elif k == "PULSE":
                v = pr + [0.0] * (7 - len(pr))
                src = {"kind": "PULSE", "v1": v[0], "v2": v[1], "td": v[2], "tr": v[3] or 1e-6, "tf": v[4] or 1e-6, "pw": v[5], "per": v[6]}
            elif k in ("SIN", "TRI", "SQUARE"):
                v = pr + [0.0] * (4 - len(pr))
                src = {"kind": k, "off": v[0], "amp": v[1], "freq": v[2] or 1e3, "ph": v[3], "duty": 0.5}
            elif k == "CLOCK":                       # digital clock: low high freq duty
                v = pr + [0.0, 5.0, 1e3, 0.5][len(pr):]
                lo, hi, f, duty = v[0], v[1], v[2] or 1e3, v[3] or 0.5
                per = 1.0 / f
                edge = per * 0.01
                src = {"kind": "PULSE", "v1": lo, "v2": hi, "td": 0.0, "tr": edge, "tf": edge,
                       "pw": max(edge, duty * per - edge), "per": per, "clock": True, "freq": f, "duty": duty}
            else:
                _err(f"unknown source waveform '{spec}' (DC, SIN, PULSE, TRI, SQUARE)")
        return self.add({"type": kind, "name": name, "a": self._n(p), "b": self._n(n), "src": src})

    MATERIALS = {"si": (2.5e-9, 1.75), "silicon": (2.5e-9, 1.75), "ge": (1e-7, 1.3), "germanium": (1e-7, 1.3),
                 "schottky": (1e-8, 1.05), "ideal": (1e-14, 1.0), "textbook": (1e-14, 1.0)}

    def diode(self, name, a, k, IS=None, N=None, material=None, led=False):
        if material is not None:
            IS, N = self.MATERIALS.get(str(material).lower(), (IS, N))
        return self.add({"type": "D", "name": name, "a": self._n(a), "b": self._n(k),
                         "IS": float(IS) if IS is not None else globals()["IS"], "N": float(N) if N is not None else 1.0, "led": led})

    def led(self, name, a, k, color="red", vf=None):
        vfs = {"red": 1.8, "yellow": 2.0, "green": 2.2, "blue": 3.0, "white": 3.2, "infrared": 1.2}
        vf = float(vf) if vf is not None else vfs.get(str(color).lower(), 1.8)
        n = 1.9
        return self.add({"type": "D", "name": name, "a": self._n(a), "b": self._n(k), "IS": 0.02 / math.exp(vf / (n * VT)),
                         "N": n, "led": True, "VF": vf, "color": str(color)})

    def zener(self, name, a, k, bv=5.1, izt=5e-3):
        return self.add({"type": "Z", "name": name, "a": self._n(a), "b": self._n(k), "IS": IS, "N": 1.0, "BV": float(bv), "IZT": float(izt)})

    def bjt(self, name, c, b, e, pol="NPN", bf=100.0, is_=None):
        return self.add({"type": "Q", "name": name, "c": self._n(c), "b": self._n(b), "e": self._n(e),
                         "pol": -1 if str(pol).upper().startswith("P") else 1, "BF": float(bf), "BR": 1.0,
                         "IS": float(is_) if is_ is not None else IS})

    def jfet(self, name, d, g, s_, pol="NJF", idss=3e-3, vp=-2.0):
        """Shichman-Hodges JFET: VTO = -|Vp|, KP = 2 IDSS / Vp^2 (gate junction ideal)."""
        vp, idss = float(vp), float(idss)
        return self.add({"type": "M", "name": name, "d": self._n(d), "g": self._n(g), "s": self._n(s_),
                         "pol": -1 if str(pol).upper().startswith("P") else 1, "jfet": True, "IDSS": idss, "VP": vp,
                         "VTO": -abs(vp), "KP": 2 * idss / (vp * vp), "LAMBDA": 0.01})

    def comparator(self, name, out, inp, inn, voh=5.0, vol=0.0, gain=1e5):
        return self.add({"type": "E", "name": name, "a": self._n(out), "b": "0", "cp": self._n(inp), "cn": self._n(inn),
                         "gain": float(gain), "cmp": True, "sat": True, "voh": float(voh), "vol": float(vol)})

    def mosfet(self, name, d, g, s, pol="NMOS", vto=1.0, kp=1e-3, lam=0.01):
        return self.add({"type": "M", "name": name, "d": self._n(d), "g": self._n(g), "s": self._n(s),
                         "pol": -1 if str(pol).upper().startswith("P") else 1, "VTO": float(vto), "KP": float(kp), "LAMBDA": float(lam)})

    def opamp(self, name, out, inp, inn, gain=1e6, rails=0.0):
        rails = float(rails)
        return self.add({"type": "E", "name": name, "a": self._n(out), "b": "0", "cp": self._n(inp), "cn": self._n(inn),
                         "gain": float(gain), "opamp": True, "rails": rails, "sat": rails > 0, "voh": rails, "vol": -rails})

    def vcvs(self, name, op, on, cp, cn, gain):
        return self.add({"type": "E", "name": name, "a": self._n(op), "b": self._n(on), "cp": self._n(cp), "cn": self._n(cn), "gain": float(gain)})

    def vccs(self, name, op, on, cp, cn, gm):
        return self.add({"type": "G", "name": name, "a": self._n(op), "b": self._n(on), "cp": self._n(cp), "cn": self._n(cn), "gm": float(gm)})

    def switch(self, name, a, b, state="on"):
        on = str(state).lower() not in ("off", "0", "open")
        return self.add({"type": "R", "name": name, "a": self._n(a), "b": self._n(b), "val": 1e-3 if on else 1e9, "sw": on})

    def pot(self, name, a, w, b, rtotal=1e4, pos=0.5):
        pos = max(0.001, min(0.999, float(pos)))
        self.add({"type": "R", "name": name + "_a", "a": self._n(a), "b": self._n(w), "val": float(rtotal) * pos, "pot": name})
        return self.add({"type": "R", "name": name + "_b", "a": self._n(w), "b": self._n(b), "val": float(rtotal) * (1 - pos), "pot": name})

    def transformer(self, name, pp, pn, sp, sn, n=1.0):
        return self.add({"type": "T", "name": name, "a": self._n(pp), "b": self._n(pn), "sp": self._n(sp), "sn": self._n(sn), "n": float(n)})

    # --- SPICE netlist import (same grammar as the JS engine) ---------------
    def load_netlist(self, text):
        import re
        for raw in text.split("\n"):
            line = re.sub(r"\*.*", "", raw).strip()
            if not line:
                continue
            t = line.split()
            nm = t[0]
            if nm.startswith("."):
                d = nm.lower()
                if d == ".tran":
                    self.analysis.update(type="tran", tstep=parse_val(t[1]), tstop=parse_val(t[2]))
                elif d == ".op":
                    self.analysis["type"] = "op"
                elif d == ".dc":
                    if len(t) >= 5:
                        self.analysis.update(type="dcsweep", src=t[1], start=parse_val(t[2]), stop=parse_val(t[3]), inc=parse_val(t[4]))
                    else:
                        self.analysis["type"] = "op"
                elif d == ".ac":
                    self.analysis.update(type="ac", f0=parse_val(t[1]), f1=parse_val(t[2]), nppd=int(t[3]) if len(t) > 3 else 40)
                elif d in (".probe", ".plot"):
                    self.probes.extend(t[1:])
                elif d == ".hidden":
                    self.hidden.extend(t[1:])
                elif d == ".step":
                    self.step = {"name": t[1], "start": parse_val(t[2]), "stop": parse_val(t[3]), "n": max(2, min(12, int(t[4]) if len(t) > 4 else 5))}
                elif d == ".four":
                    self.four = {"f0": parse_val(t[1]), "node": t[2]}
                continue
            up = nm.upper()
            kvs = {k.upper(): parse_val(v) for k, v in (x.split("=", 1) for x in t if "=" in x)}
            typ = "LED" if up.startswith("LED") else "LAMP" if up.startswith("LAMP") else "CMP" if up.startswith("CMP") else up[0]
            if typ in ("V", "I"):
                spec = t[3].upper() if len(t) > 3 else "DC"
                if spec in ("DC", "SIN", "PULSE", "TRI", "SQUARE", "CLOCK"):
                    params = [parse_val(x) for x in t[4:] if "=" not in x]
                    self.source(typ, nm, t[1], t[2], spec, params)
                else:
                    self.source(typ, nm, t[1], t[2], "DC", [parse_val(t[3])])
            elif typ in ("R", "C", "L"):
                self.rlc(typ, nm, t[1], t[2], parse_val(t[3]), pwr=kvs.get("PWR"))
            elif typ == "LAMP":
                self.rlc("R", nm, t[1], t[2], parse_val(t[3]), lamp=True, pwr=kvs.get("PWR"))
            elif typ == "D":
                self.diode(nm, t[1], t[2], IS=kvs.get("IS", IS), N=kvs.get("N", 1.0))
            elif typ == "Z":
                self.zener(nm, t[1], t[2], kvs.get("BV", 5.1), kvs.get("IZT", 5e-3))
            elif typ == "LED":
                self.led(nm, t[1], t[2], vf=kvs.get("VF", 1.8))
            elif typ == "Q":
                self.bjt(nm, t[1], t[2], t[3], t[4] if len(t) > 4 and "=" not in t[4] else "NPN", kvs.get("BF", 100), kvs.get("IS"))
            elif typ == "M":
                self.mosfet(nm, t[1], t[2], t[3], t[4] if len(t) > 4 and "=" not in t[4] else "NMOS", kvs.get("VTO", 1), kvs.get("KP", 1e-3), kvs.get("LAMBDA", 0.01))
            elif typ == "J":
                self.jfet(nm, t[1], t[2], t[3], t[4] if len(t) > 4 and "=" not in t[4] else "NJF", kvs.get("IDSS", 3e-3), kvs.get("VP", -2.0))
            elif typ == "CMP":
                self.comparator(nm, t[1], t[2], t[3], kvs.get("VOH", 5.0), kvs.get("VOL", 0.0), kvs.get("A", 1e5))
            elif typ == "S":
                self.switch(nm, t[1], t[2], t[3] if len(t) > 3 else "on")
            elif typ == "P":
                self.pot(nm, t[1], t[2], t[3], parse_val(t[4]) if len(t) > 4 else 1e4, parse_val(t[5]) if len(t) > 5 else 0.5)
            elif typ == "T":
                self.transformer(nm, t[1], t[2], t[3], t[4], parse_val(t[5]) if len(t) > 5 else 1.0)
            elif typ == "E":
                self.vcvs(nm, t[1], t[2], t[3], t[4], parse_val(t[5]))
            elif typ == "G":
                self.vccs(nm, t[1], t[2], t[3], t[4], parse_val(t[5]))
            elif typ == "X":
                self.opamp(nm, t[1], t[2], t[3], kvs.get("A", 1e6), kvs.get("RAILS", 0.0))
            else:
                _err(f"unknown element '{nm}'")
        return self

    # --- node numbering -----------------------------------------------------
    def finalize(self):
        if self._final:
            return self._final
        if not self.els:
            _err("the circuit has no elements")
        nodes = {"0": -1}

        def idx(n):
            if n is None:
                _err("an element is missing a node")
            if n not in nodes:
                nodes[n] = len(nodes) - 1
            return nodes[n]
        for e in self.els:
            if e["type"] == "Q":
                e["nc"], e["nb"], e["ne"] = idx(e["c"]), idx(e["b"]), idx(e["e"])
            elif e["type"] == "M":
                e["nd"], e["ng"], e["ns"] = idx(e["d"]), idx(e["g"]), idx(e["s"])
            else:
                e["na"], e["nb"] = idx(e["a"]), idx(e["b"])
            if "cp" in e:
                e["ncp"], e["ncn"] = idx(e["cp"]), idx(e["cn"])
            if "sp" in e:
                e["nsp"], e["nsn"] = idx(e["sp"]), idx(e["sn"])
        N = len(nodes) - 1
        bi = N
        for e in self.els:
            if e["type"] in ("V", "L", "E", "T"):
                e["bi"] = bi
                bi += 1
        self.nodes = nodes
        self.N = N
        self.M = bi
        self._final = True
        return True


# ===========================================================================
# Solvers
# ===========================================================================
def _solve(A, z):
    n = len(z)
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(A[r][c]))
        if abs(A[p][c]) < 1e-18:
            continue
        A[c], A[p] = A[p], A[c]
        z[c], z[p] = z[p], z[c]
        piv = A[c][c]
        for r in range(n):
            if r == c:
                continue
            f = A[r][c] / piv
            if f == 0:
                continue
            rowc, rowr = A[c], A[r]
            for k in range(c, n):
                rowr[k] -= f * rowc[k]
            z[r] -= f * z[c]
    x = [0.0] * n
    for c in range(n):
        if abs(A[c][c]) > 1e-18:
            x[c] = z[c] / A[c][c]
    return x


def src_value(src, t):
    k = src["kind"]
    if k == "DC":
        return src.get("dc", 0.0)
    if k == "SIN":
        return src["off"] + src["amp"] * math.sin(2 * math.pi * src["freq"] * t + src.get("ph", 0.0) * math.pi / 180)
    if k == "SQUARE":
        ph = (t * src["freq"]) % 1.0
        return src["off"] + (src["amp"] if ph < src.get("duty", 0.5) else -src["amp"])
    if k == "TRI":
        ph = (t * src["freq"]) % 1.0
        return src["off"] + src["amp"] * ((4 * ph - 1) if ph < 0.5 else (3 - 4 * ph))
    if k == "PULSE":
        v1, v2, td, tr, tf, pw, per = (src["v1"], src["v2"], src["td"], src["tr"], src["tf"], src["pw"], src["per"])
        tt = t - td
        if tt < 0:
            return v1
        if per > 0:
            tt = tt % per
        if tt < tr:
            return v1 + (v2 - v1) * (tt / (tr or 1e-12))
        if tt < tr + pw:
            return v2
        if tt < tr + pw + tf:
            return v2 + (v1 - v2) * ((tt - tr - pw) / (tf or 1e-12))
        return v1
    return 0.0


def _sat_stage(e, vd):
    """Saturating gain stage (op-amp with rails / comparator): Vo = clamp(A vd, vol, voh), rail slope 1e-4 A."""
    gr = 1e-3                                  # absolute rail slope (1 mV/V of overdrive)
    lo, hi = e["vol"] / e["gain"], e["voh"] / e["gain"]
    reg = 1 if vd > hi else -1 if vd < lo else 0
    if reg != 0 and e.get("_reg") == -reg:     # never jump rail-to-rail: linearise in between (kills NR chatter)
        vd = hi if reg > 0 else lo
        reg = 0
    e["_reg"] = reg
    if reg > 0:
        return e["voh"] + gr * (vd - hi), gr, vd
    if reg < 0:
        return e["vol"] + gr * (vd - lo), gr, vd
    return e["gain"] * vd, e["gain"], vd


def mna_step(ckt, st, t, h, mode, zero_sources=False):
    els, N, M = ckt.els, ckt.N, ckt.M
    dc = mode == "dc"
    vg = st.get("vGuess") or [0.0] * M
    nonlinear = any(e["type"] in ("D", "Z", "Q", "M") or (e["type"] == "E" and e.get("sat")) for e in els)
    converged = not nonlinear
    for _it in range(120):
        A = [[0.0] * M for _ in range(M)]
        z = [0.0] * M

        def G(a, b, g):
            if a >= 0:
                A[a][a] += g
            if b >= 0:
                A[b][b] += g
            if a >= 0 and b >= 0:
                A[a][b] -= g
                A[b][a] -= g

        def CS(a, b, i):
            if a >= 0:
                z[a] += i
            if b >= 0:
                z[b] -= i

        def VC(r, c, g):
            if r >= 0 and c >= 0:
                A[r][c] += g

        def V(k):
            return vg[k] if k >= 0 else 0.0
        for i in range(N):
            A[i][i] += GMIN
        for e in els:
            ty = e["type"]
            if ty == "R":
                G(e["na"], e["nb"], 1.0 / e["val"])
            elif ty == "C":
                if dc:
                    continue
                g = e["val"] / h
                vp = st["vC"].get(e["name"], 0.0)
                G(e["na"], e["nb"], g)
                CS(e["na"], e["nb"], g * vp)
            elif ty == "L":
                m = e["bi"]
                req = 0.0 if dc else e["val"] / h
                ip = st["iL"].get(e["name"], 0.0)
                if e["na"] >= 0:
                    A[e["na"]][m] += 1
                    A[m][e["na"]] += 1
                if e["nb"] >= 0:
                    A[e["nb"]][m] -= 1
                    A[m][e["nb"]] -= 1
                A[m][m] -= req
                z[m] -= req * ip
            elif ty == "V":
                m = e["bi"]
                v = 0.0 if (zero_sources and not e.get("meter")) else src_value(e["src"], t)
                if e["na"] >= 0:
                    A[e["na"]][m] += 1
                    A[m][e["na"]] += 1
                if e["nb"] >= 0:
                    A[e["nb"]][m] -= 1
                    A[m][e["nb"]] -= 1
                z[m] += v
            elif ty == "I":
                i = 0.0 if (zero_sources and not e.get("meter")) else src_value(e["src"], t)
                CS(e["na"], e["nb"], -i)
            elif ty == "E":
                m = e["bi"]
                if e["na"] >= 0:
                    A[e["na"]][m] += 1
                    A[m][e["na"]] += 1
                if e["nb"] >= 0:
                    A[e["nb"]][m] -= 1
                    A[m][e["nb"]] -= 1
                if e.get("sat"):                     # piecewise-linear rails, linearised
                    vo, g, vd = _sat_stage(e, V(e["ncp"]) - V(e["ncn"]))   # vd = linearisation point
                    VC(m, e["ncp"], -g); VC(m, e["ncn"], g)
                    z[m] += vo - g * vd
                else:
                    VC(m, e["ncp"], -e["gain"])
                    VC(m, e["ncn"], e["gain"])
            elif ty == "G":
                gm = e["gm"]
                VC(e["na"], e["ncp"], gm); VC(e["na"], e["ncn"], -gm)
                VC(e["nb"], e["ncp"], -gm); VC(e["nb"], e["ncn"], gm)
            elif ty == "T":
                m, n = e["bi"], e["n"]
                if e["nsp"] >= 0:
                    A[e["nsp"]][m] += 1
                    A[m][e["nsp"]] += 1
                if e["nsn"] >= 0:
                    A[e["nsn"]][m] -= 1
                    A[m][e["nsn"]] -= 1
                VC(m, e["na"], -n); VC(m, e["nb"], n)
                if e["na"] >= 0:
                    A[e["na"]][m] -= n
                if e["nb"] >= 0:
                    A[e["nb"]][m] += n
            elif ty in ("D", "Z"):
                nvt = e.get("N", 1.0) * VT
                vd = _lim(V(e["na"]) - V(e["nb"]), nvt * math.log(10 / e["IS"]))
                ex = math.exp(vd / nvt)
                idd = e["IS"] * (ex - 1)
                geq = e["IS"] / nvt * ex
                if ty == "Z":                       # Izt (default 5 mA) at -BV, diode-like slope
                    vr = _lim(-(vd + e["BV"]), 0.2)
                    iz = e.get("IZT", 5e-3) * math.exp(vr / VT)
                    idd -= iz
                    geq += iz / VT
                geq += GMIN
                ieq = idd - geq * vd
                G(e["na"], e["nb"], geq)
                CS(e["na"], e["nb"], -ieq)
                e["_i"] = idd
            elif ty == "Q":
                p = e["pol"]
                vbe = _lim(p * (V(e["nb"]) - V(e["ne"])))
                vbc = _lim(p * (V(e["nb"]) - V(e["nc"])))
                ef, er = math.exp(vbe / VT), math.exp(vbc / VT)
                ibe = e["IS"] / e["BF"] * (ef - 1); gbe = e["IS"] / e["BF"] / VT * ef + GMIN
                ibc = e["IS"] / e["BR"] * (er - 1); gbc = e["IS"] / e["BR"] / VT * er + GMIN
                ict = e["IS"] * (ef - er)
                gmf, gmr = e["IS"] / VT * ef, e["IS"] / VT * er
                Ibe0 = ibe - gbe * vbe; Ibc0 = ibc - gbc * vbc; Ict0 = ict - gmf * vbe + gmr * vbc
                b, c, en = e["nb"], e["nc"], e["ne"]
                G(b, en, gbe); CS(b, en, -p * Ibe0)
                G(b, c, gbc); CS(b, c, -p * Ibc0)
                VC(c, b, gmf); VC(c, en, -gmf); VC(c, b, -gmr); VC(c, c, gmr)
                VC(en, b, -gmf); VC(en, en, gmf); VC(en, b, gmr); VC(en, c, -gmr)
                CS(c, en, -p * Ict0)
                e["_ic"] = p * (ict - ibc); e["_ib"] = p * (ibe + ibc)
            elif ty == "M":
                p = e["pol"]
                vgs = p * (V(e["ng"]) - V(e["ns"])); vds = p * (V(e["nd"]) - V(e["ns"]))
                vov, k, lam = vgs - e["VTO"], e["KP"], e.get("LAMBDA", 0.0)
                if vov <= 0:
                    idd, gm, gds = 0.0, 0.0, GMIN
                elif vds < vov:
                    idd = k * (vov * vds - vds * vds / 2) * (1 + lam * vds)
                    gm = k * vds * (1 + lam * vds)
                    gds = k * (vov - vds) * (1 + lam * vds) + k * (vov * vds - vds * vds / 2) * lam + GMIN
                else:
                    idd = k / 2 * vov * vov * (1 + lam * vds)
                    gm = k * vov * (1 + lam * vds)
                    gds = k / 2 * vov * vov * lam + GMIN
                Id0 = idd - gm * vgs - gds * vds
                d, g, s = e["nd"], e["ng"], e["ns"]
                VC(d, g, gm); VC(d, s, -gm); VC(s, g, -gm); VC(s, s, gm)
                G(d, s, gds)
                CS(d, s, -p * Id0)
                e["_id"] = p * idd
        x = _solve([row[:] for row in A], z[:])
        if not nonlinear:
            vg = x
            break
        dmax = 0.0
        nxt = [0.0] * M
        for i in range(M):
            dv = x[i] - vg[i]
            if abs(dv) > 1.5:
                dv = 1.5 if dv > 0 else -1.5
            nxt[i] = vg[i] + dv
            dmax = max(dmax, abs(dv))
        vg = nxt
        if dmax < 1e-6:
            converged = True
            break
    st["vGuess"] = vg
    st["converged"] = converged
    for e in els:
        if e["type"] == "C":
            va = vg[e["na"]] if e["na"] >= 0 else 0.0
            vb = vg[e["nb"]] if e["nb"] >= 0 else 0.0
            e["_i"] = 0.0 if dc else (e["val"] / h) * ((va - vb) - st["vC"].get(e["name"], 0.0))
            st["vC"][e["name"]] = va - vb
        elif e["type"] == "L":
            st["iL"][e["name"]] = vg[e["bi"]]
    return vg


def element_current(e, x):
    def V(k):
        return x[k] if k >= 0 else 0.0
    ty = e["type"]
    if ty == "R":
        return (V(e["na"]) - V(e["nb"])) / e["val"]
    if ty in ("L", "V", "E", "T"):
        return x[e["bi"]]
    if ty == "I":
        return src_value(e["src"], e.get("_t", 0.0))
    if ty == "G":
        return e["gm"] * (V(e["ncp"]) - V(e["ncn"]))
    if ty in ("C", "D", "Z"):
        return e.get("_i", 0.0)
    if ty == "Q":
        return e.get("_ic", 0.0)
    if ty == "M":
        return e.get("_id", 0.0)
    return 0.0


def element_voltage(e, x):
    def V(k):
        return x[k] if k >= 0 else 0.0
    if e["type"] == "Q":
        return V(e["nc"]) - V(e["ne"])
    if e["type"] == "M":
        return V(e["nd"]) - V(e["ns"])
    return V(e["na"]) - V(e["nb"])


def ac_step(ckt, w, op):
    els, N, M = ckt.els, ckt.N, ckt.M
    A = [[0j] * M for _ in range(M)]
    z = [0j] * M

    def G(a, b, g):
        if a >= 0:
            A[a][a] += g
        if b >= 0:
            A[b][b] += g
        if a >= 0 and b >= 0:
            A[a][b] -= g
            A[b][a] -= g

    def VC(r, c, g):
        if r >= 0 and c >= 0:
            A[r][c] += g

    def V(k):
        return op[k] if (op and k >= 0) else 0.0
    for i in range(N):
        A[i][i] += GMIN
    for e in els:
        ty = e["type"]
        if ty == "R":
            G(e["na"], e["nb"], 1.0 / e["val"])
        elif ty == "C":
            G(e["na"], e["nb"], 1j * w * e["val"])
        elif ty == "L":
            m = e["bi"]
            if e["na"] >= 0:
                A[e["na"]][m] += 1; A[m][e["na"]] += 1
            if e["nb"] >= 0:
                A[e["nb"]][m] -= 1; A[m][e["nb"]] -= 1
            A[m][m] -= 1j * w * e["val"]
        elif ty == "V":
            m = e["bi"]
            amp = 0.0 if e["src"]["kind"] == "DC" else e["src"].get("amp", 1.0)
            if e["na"] >= 0:
                A[e["na"]][m] += 1; A[m][e["na"]] += 1
            if e["nb"] >= 0:
                A[e["nb"]][m] -= 1; A[m][e["nb"]] -= 1
            z[m] += amp
        elif ty == "I":
            amp = 0.0 if e["src"]["kind"] == "DC" else e["src"].get("amp", 1.0)
            if e["na"] >= 0:
                z[e["na"]] -= amp
            if e["nb"] >= 0:
                z[e["nb"]] += amp
        elif ty == "E":
            m = e["bi"]
            if e["na"] >= 0:
                A[e["na"]][m] += 1; A[m][e["na"]] += 1
            if e["nb"] >= 0:
                A[e["nb"]][m] -= 1; A[m][e["nb"]] -= 1
            g = e["gain"]
            if e.get("sat"):
                g = _sat_stage(e, V(e["ncp"]) - V(e["ncn"]))[1]
            VC(m, e["ncp"], -g); VC(m, e["ncn"], g)
        elif ty == "G":
            gm = e["gm"]
            VC(e["na"], e["ncp"], gm); VC(e["na"], e["ncn"], -gm); VC(e["nb"], e["ncp"], -gm); VC(e["nb"], e["ncn"], gm)
        elif ty == "T":
            m, n = e["bi"], e["n"]
            if e["nsp"] >= 0:
                A[e["nsp"]][m] += 1; A[m][e["nsp"]] += 1
            if e["nsn"] >= 0:
                A[e["nsn"]][m] -= 1; A[m][e["nsn"]] -= 1
            VC(m, e["na"], -n); VC(m, e["nb"], n)
            if e["na"] >= 0:
                A[e["na"]][m] -= n
            if e["nb"] >= 0:
                A[e["nb"]][m] += n
        elif ty in ("D", "Z"):
            nvt = e.get("N", 1.0) * VT
            vd = _lim(V(e["na"]) - V(e["nb"]), nvt * math.log(10 / e["IS"]))
            g = e["IS"] / nvt * math.exp(vd / nvt) + GMIN
            if ty == "Z":
                g += e.get("IZT", 5e-3) / VT * math.exp(_lim(-(vd + e["BV"]), 0.2) / VT)
            G(e["na"], e["nb"], g)
        elif ty == "Q":
            p = e["pol"]
            vbe = _lim(p * (V(e["nb"]) - V(e["ne"]))); vbc = _lim(p * (V(e["nb"]) - V(e["nc"])))
            ef, er = math.exp(vbe / VT), math.exp(vbc / VT)
            gbe = e["IS"] / e["BF"] / VT * ef + GMIN; gbc = e["IS"] / e["BR"] / VT * er + GMIN
            gmf, gmr = e["IS"] / VT * ef, e["IS"] / VT * er
            G(e["nb"], e["ne"], gbe); G(e["nb"], e["nc"], gbc)
            c, b, en = e["nc"], e["nb"], e["ne"]
            VC(c, b, gmf); VC(c, en, -gmf); VC(c, b, -gmr); VC(c, c, gmr)
            VC(en, b, -gmf); VC(en, en, gmf); VC(en, b, gmr); VC(en, c, -gmr)
        elif ty == "M":
            p = e["pol"]
            vgs = p * (V(e["ng"]) - V(e["ns"])); vds = p * (V(e["nd"]) - V(e["ns"]))
            vov, k, lam = vgs - e["VTO"], e["KP"], e.get("LAMBDA", 0.0)
            gm, gds = 0.0, GMIN
            if vov > 0:
                if vds < vov:
                    gm = k * vds; gds = k * (vov - vds) + GMIN
                else:
                    gm = k * vov * (1 + lam * vds); gds = k / 2 * vov * vov * lam + GMIN
            d, g, s = e["nd"], e["ng"], e["ns"]
            VC(d, g, gm); VC(d, s, -gm); VC(s, g, -gm); VC(s, s, gm)
            G(d, s, gds)
    return _solve(A, z)


# ===========================================================================
# Probes and analyses
# ===========================================================================
def _probe_spec(name, ckt):
    import re
    m = re.match(r"^([VIP])\((.+)\)$", name, re.I)
    if not m:
        return {"kind": "V", "node": name, "label": name}
    k, arg = m.group(1).upper(), m.group(2)
    if k == "V":
        return {"kind": "V", "node": arg, "label": f"V({arg})"}
    el = next((e for e in ckt.els if e["name"] == arg or e["name"] == arg + "_a" or e.get("pot") == arg), None)
    if el is None:
        _err(f"probe {name}: no element '{arg}'")
    return {"kind": k, "el": el, "label": f"{k}({arg})"}


def _probe_value(spec, x, ckt):
    if spec["kind"] == "V":
        k = ckt.nodes.get(spec["node"])
        if k is None:
            _err(f"probe: unknown node '{spec['node']}'")
        return x[k] if k >= 0 else 0.0
    i = element_current(spec["el"], x)
    return i if spec["kind"] == "I" else i * element_voltage(spec["el"], x)


def _default_probes(ckt):
    named = [n for n in ckt.nodes if n != "0" and not (n[0] in "nN" and n[1:].isdigit())]
    auto = [n for n in ckt.nodes if n != "0" and (n[0] in "nN" and n[1:].isdigit())]
    return (named + auto)[:6]


def run_once(ckt):
    ckt.finalize()
    A = ckt.analysis
    names = ckt.probes or _default_probes(ckt)
    all_names = names + [h for h in ckt.hidden if h not in names]
    specs = [_probe_spec(p, ckt) for p in all_names]
    labels = [s["label"] for s in specs]
    vis = labels[:len(names)]                    # hidden probes are computed but not returned

    if A["type"] == "op":
        st = {"vC": {}, "iL": {}, "vGuess": None}
        x = mna_step(ckt, st, 0.0, 1.0, "dc")
        nodeV = {n: x[k] for n, k in ckt.nodes.items() if k >= 0}
        elI = {e["name"]: element_current(e, x) for e in ckt.els}
        elP = {e["name"]: elI[e["name"]] * element_voltage(e, x) for e in ckt.els}
        return {"type": "op", "nodeV": nodeV, "elI": elI, "elP": elP, "probeNames": vis,
                "values": {lab: _probe_value(s, x, ckt) for s, lab in zip(specs, labels)}, "converged": st["converged"]}

    if A["type"] == "ac":
        st = {"vC": {}, "iL": {}, "vGuess": None}
        op = mna_step(ckt, st, 0.0, 1.0, "dc")
        decades = math.log10(A["f1"] / A["f0"])
        npts = max(10, round(decades * A["nppd"]))
        freqs, mag, phase = [], {lab: [] for lab in labels}, {lab: [] for lab in labels}
        for i in range(npts + 1):
            f = A["f0"] * 10 ** ((i / npts) * decades)
            xc = ac_step(ckt, 2 * math.pi * f, op)
            freqs.append(f)
            for s, lab in zip(specs, labels):
                if s["kind"] == "V":
                    k = ckt.nodes.get(s["node"])
                    if k is None:
                        _err(f"probe: unknown node '{s['node']}'")
                    h = xc[k] if k >= 0 else 0j
                else:
                    e = s["el"]
                    if e["type"] == "R":
                        va = xc[e["na"]] if e["na"] >= 0 else 0j
                        vb = xc[e["nb"]] if e["nb"] >= 0 else 0j
                        h = (va - vb) / e["val"]
                    else:
                        h = xc[e["bi"]] if "bi" in e else 0j
                mag[lab].append(20 * math.log10(abs(h) + 1e-15))
                phase[lab].append(cmath.phase(h) * 180 / math.pi)
        return {"type": "ac", "freqs": freqs, "mag": mag, "phase": phase, "probeNames": vis}

    if A["type"] == "dcsweep":
        src = next((e for e in ckt.els if e["name"] == A["src"]), None)
        if src is None or src["type"] not in ("V", "I"):
            _err(f"dcsweep: no source '{A['src']}'")
        st = {"vC": {}, "iL": {}, "vGuess": None}
        sweep, traces = [], {lab: [] for lab in labels}
        saved = dict(src["src"])
        n = min(2000, int((A["stop"] - A["start"]) / A["inc"] + 1e-9) + 1)
        for s_ in range(n):
            v = A["start"] + s_ * A["inc"]
            src["src"] = {"kind": "DC", "dc": v}
            x = mna_step(ckt, st, 0.0, 1.0, "dc")
            sweep.append(v)
            for sp, lab in zip(specs, labels):
                traces[lab].append(_probe_value(sp, x, ckt))
        src["src"] = saved
        return {"type": "dcsweep", "sweep": sweep, "traces": traces, "probeNames": vis, "xlabel": A["src"]}

    st = {"vC": {}, "iL": {}, "vGuess": None}
    try:                                   # initial conditions from the DC operating point
        op = mna_step(ckt, {"vC": {}, "iL": {}, "vGuess": None}, 0.0, 1.0, "dc")
        for e in ckt.els:
            if e["type"] == "C":
                va = op[e["na"]] if e["na"] >= 0 else 0.0
                vb = op[e["nb"]] if e["nb"] >= 0 else 0.0
                st["vC"][e["name"]] = va - vb
            elif e["type"] == "L":
                st["iL"][e["name"]] = op[e["bi"]]
        st["vGuess"] = op
    except Exception:
        st = {"vC": {}, "iL": {}, "vGuess": None}
    h = A["tstep"]
    nsteps = min(20000, int(math.ceil(A["tstop"] / h)))
    time, traces = [], {lab: [] for lab in labels}
    unconv = 0
    led = {}
    for s_ in range(nsteps + 1):
        t = s_ * h
        for e in ckt.els:
            if e["type"] == "I":
                e["_t"] = t
        x = mna_step(ckt, st, t, h, "tran")
        if not st["converged"]:
            unconv += 1
        time.append(t)
        for sp, lab in zip(specs, labels):
            traces[lab].append(_probe_value(sp, x, ckt))
        for e in ckt.els:
            if e.get("led"):
                led[e["name"]] = max(led.get(e["name"], 0.0), e.get("_i", 0.0))
    x = st["vGuess"]
    nodeV = {n: x[k] for n, k in ckt.nodes.items() if k >= 0}
    elI = {e["name"]: element_current(e, x) for e in ckt.els}
    elP = {e["name"]: elI[e["name"]] * element_voltage(e, x) for e in ckt.els}
    out = {"type": "tran", "time": time, "traces": traces, "probeNames": vis, "unconverged": unconv,
           "nodeV": nodeV, "elI": elI, "elP": elP, "led": led}
    if ckt.four:
        out["fourier"] = fourier(out, ckt.four)
    return out


def fourier(res, spec):
    time, traces = res["time"], res["traces"]
    name = next((k for k in traces if k == spec["node"] or k == f"V({spec['node']})"), next(iter(traces)))
    y = traces[name]
    T = 1.0 / spec["f0"]
    t_end = time[-1]
    nper = max(1, int(t_end / T) - 1)
    t0 = t_end - nper * T
    idx = [k for k, tv in enumerate(time) if tv >= t0]
    if len(idx) < 8:
        return None
    dc = sum(y[k] for k in idx) / len(idx)
    harm = []
    for hN in range(1, 10):
        re = im = 0.0
        for k in idx:
            ph = 2 * math.pi * hN * spec["f0"] * time[k]
            re += y[k] * math.cos(ph)
            im -= y[k] * math.sin(ph)
        harm.append({"n": hN, "f": hN * spec["f0"], "mag": 2 * math.hypot(re, im) / len(idx), "phase": math.atan2(im, re) * 180 / math.pi})
    h1 = harm[0]["mag"] or 1e-12
    thd = math.sqrt(sum(h["mag"] ** 2 for h in harm[1:])) / h1
    return {"probe": name, "dc": dc, "harmonics": harm, "thd": thd, "f0": spec["f0"]}


def run(ckt):
    """Run the circuit; a .step expands into overlaid runs."""
    ckt.finalize()
    if not ckt.step:
        return run_once(ckt)
    st = ckt.step
    target = [e for e in ckt.els if e["name"] == st["name"] or e.get("pot") == st["name"] or e["name"] == st["name"] + "_a"]
    if not target:
        _err(f"stepparam: no element '{st['name']}'")
    runs = []
    for i in range(st["n"]):
        v = st["start"] + (st["stop"] - st["start"]) * i / (st["n"] - 1)
        for e in target:
            if "val" in e:
                e["val"] = v
            elif "src" in e:
                e["src"]["dc"] = v
        runs.append({"label": f"{st['name']}={v:g}", "value": v, "result": run_once(ckt)})
    return {"type": "step", "runs": runs, "probeNames": runs[0]["result"]["probeNames"], "base": runs[0]["result"]["type"]}


def measure(time, y):
    if not y or len(y) < 2:
        return None
    mn, mx = min(y), max(y)
    avg = sum(y) / len(y)
    rms = math.sqrt(sum(v * v for v in y) / len(y))
    cross = [time[k] for k in range(1, len(y)) if y[k - 1] < avg <= y[k]]
    freq = (len(cross) - 1) / (cross[-1] - cross[0]) if len(cross) >= 2 and cross[-1] > cross[0] else 0.0
    return {"min": mn, "max": mx, "pp": mx - mn, "avg": avg, "rms": rms,
            "acrms": math.sqrt(max(0.0, rms * rms - avg * avg)), "freq": freq}


def measure_resistance(ckt, node_a, node_b):
    """Ohmmeter: 1 A test current between two nodes with every source zeroed."""
    ckt.finalize()
    ia, ib = ckt.nodes.get(str(node_a)), ckt.nodes.get(str(node_b))
    if ia is None or ib is None:
        return float("nan")
    probe = {"type": "I", "name": "__ohm", "a": str(node_a), "b": str(node_b), "na": ia, "nb": ib,
             "src": {"kind": "DC", "dc": -1e-3}, "meter": True}   # 1 mA test current
    ckt.els.append(probe)
    try:
        x = mna_step(ckt, {"vC": {}, "iL": {}, "vGuess": None}, 0.0, 1.0, "dc", zero_sources=True)
    finally:
        ckt.els.remove(probe)
    va = x[ia] if ia >= 0 else 0.0
    vb = x[ib] if ib >= 0 else 0.0
    return abs(va - vb) / 1e-3

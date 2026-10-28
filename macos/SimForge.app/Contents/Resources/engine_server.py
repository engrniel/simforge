#!/usr/bin/env python3
"""
SimForge — execution sidecar.

Owns a single persistent MATLAB session and runs student code against it,
capturing text output, errors, figures (as PNGs) and the workspace.

Two transports (same engine underneath):
  * stdio  (default)      : newline-delimited JSON in/out. Used by the Tauri app.
  * --http [port]         : tiny HTTP server with POST /run. Used for browser dev.

Both transports are CONCURRENT: a long run executes on a worker thread so a
`stop` request can still be answered while it is in flight. Every stdio response
echoes the request's "id" because responses no longer arrive in request order.

If MATLAB / the MATLAB Engine API for Python is not installed, the sidecar runs
in DEMO mode so the whole app is still usable for development and demos.

Request kinds
  status  -> engine mode
  run     {code}                       -> stdout/error/figures/workspace
  stop                                 -> cancels the run in flight
  reset                                -> clears the base workspace
  figlim  {fig, ax, xlim, ylim}        -> re-renders a stored figure (zoom/pan)
  figfit  {fig, ax, series, fit, deg}  -> Basic Fitting / Data Statistics
  fs      {op, ...}                    -> working directory + file operations
  lint    {code}                       -> code-analyzer issues
  varset  {name, row, col, value}      -> edit one cell from the Variable Editor
"""

import sys
import json
import io
import os
import base64
import tempfile
import threading
import traceback

# ----------------------------------------------------------------------------
# Engine layer
# ----------------------------------------------------------------------------

class Engine:
    """Wraps a MATLAB session, or a DEMO simulator when MATLAB is absent."""

    def __init__(self):
        self.mode = "demo"          # "matlab" once a real session starts
        self.eng = None
        self._ml = None
        self._interp = None         # persistent DEMO session
        self._lock = threading.Lock()
        self.cancel = threading.Event()
        self._future = None         # in-flight MATLAB FutureResult
        self.cwd = os.path.expanduser("~")
        try:
            import matlab.engine  # noqa: F401
            self._ml = matlab.engine
            self.mode = "starting"   # real engine available, start lazily
        except Exception:
            self.mode = "demo"

    # -- lifecycle ----------------------------------------------------------
    def ensure_started(self):
        """Start the MATLAB session on first use (it takes ~10-20s)."""
        if self.mode == "demo":
            return
        if self.eng is not None:
            return
        try:
            self.eng = self._ml.start_matlab()
            self.mode = "matlab"
        except Exception:
            # Engine API present but no license / no MATLAB reachable.
            self.mode = "demo"

    def status(self):
        return {
            "mode": "matlab" if self.mode == "matlab" else "demo",
            "matlab_available": self.mode in ("matlab", "starting"),
            "cwd": self.cwd,
        }

    def interp(self):
        """The persistent DEMO interpreter — one base workspace per session."""
        if self._interp is None:
            from mdemo import Interp
            self._interp = Interp()
            self._interp.cancel = self.cancel
            self._interp.cwd = self.cwd
        return self._interp

    # -- execution ----------------------------------------------------------
    def run(self, code):
        self.cancel.clear()
        if self.mode != "matlab" and self.mode != "starting":
            return self._run_demo(code)
        self.ensure_started()
        if self.mode != "matlab":
            return self._run_demo(code)
        return self._run_matlab(code)

    def stop(self):
        """Ask the run in flight to give up. Safe to call when nothing runs."""
        self.cancel.set()
        fut = self._future
        if fut is not None:
            try:
                fut.cancel()
            except Exception:
                pass
        return {"ok": True, "stopped": True}

    def reset(self):
        """Clear the base workspace (both engines)."""
        if self.mode == "matlab" and self.eng is not None:
            try:
                self.eng.eval("clear all; close all force;", nargout=0)
            except Exception:
                pass
        elif self._interp is not None:
            self._interp.reset()
        return {"ok": True, "workspace": []}

    def figlim(self, msg):
        """Zoom / pan / restore one stored DEMO figure by re-rendering it."""
        if self._interp is None:
            return {"ok": False, "error": "no figures in this session"}
        try:
            svg = self._interp.set_limits(
                msg.get("fig"), msg.get("ax", 0), msg.get("xlim"), msg.get("ylim"))
            return {"ok": True, "svg": svg}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def figfit(self, msg):
        """Add a polynomial fit (or read statistics) for one plotted series."""
        if self._interp is None:
            return {"ok": False, "error": "no figures in this session"}
        try:
            return self._interp.fit_series(
                msg.get("fig"), msg.get("ax", 0), msg.get("series", 0),
                msg.get("fit", "poly"), int(msg.get("deg", 1)))
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def lint(self, code):
        """Code analyzer. Only the DEMO interpreter can do this locally."""
        try:
            from mdemo import lint as do_lint
            return {"ok": True, "issues": do_lint(code)}
        except Exception as e:
            return {"ok": False, "error": str(e), "issues": []}

    def varset(self, msg):
        """Write one value back from the Variable Editor grid."""
        name = msg.get("name") or ""
        if not name.isidentifier():
            return {"ok": False, "error": "bad variable name"}
        row, col = int(msg.get("row", 1)), int(msg.get("col", 1))
        raw = str(msg.get("value", "")).strip()
        if not raw:
            return {"ok": False, "error": "empty value"}
        if msg.get("char"):
            code = f"{name} = {raw};"
        else:
            code = f"{name}({row},{col}) = {raw};"
        res = self.run(code)
        return {"ok": res.get("error") is None, "error": res.get("error"),
                "workspace": res.get("workspace", [])}

    def _run_matlab(self, code):
        out, err = io.StringIO(), io.StringIO()
        ok, error = True, None
        try:
            # background=True gives a FutureResult, which Stop can cancel.
            fut = self.eng.eval(code, nargout=0, stdout=out, stderr=err,
                                background=True)
            self._future = fut
            fut.result()
        except Exception as e:  # MatlabExecutionError, CancelledError, ...
            ok = False
            error = "execution stopped" if self.cancel.is_set() else str(e)
        finally:
            self._future = None

        figures = self._capture_figures()
        workspace = self._capture_workspace()
        return {
            "ok": ok,
            "mode": "matlab",
            "stdout": out.getvalue(),
            "stderr": err.getvalue(),
            "error": error,
            "figures": figures,
            "workspace": workspace,
        }

    def _capture_figures(self):
        figs = []
        try:
            self.eng.eval(
                "set(0,'DefaultFigureVisible','off');"
                "CC__figs = findall(groot,'Type','figure');",
                nargout=0,
            )
            n = int(self.eng.eval("numel(CC__figs)", nargout=1))
            for i in range(1, n + 1):
                tmp = os.path.join(tempfile.gettempdir(), f"cc_fig_{i}.png")
                # exportgraphics (R2020a+); falls back to saveas on older releases.
                try:
                    self.eng.eval(
                        f"exportgraphics(CC__figs({i}), '{tmp}', 'Resolution', 120);",
                        nargout=0,
                    )
                except Exception:
                    self.eng.eval(f"saveas(CC__figs({i}), '{tmp}');", nargout=0)
                with open(tmp, "rb") as fh:
                    b64 = base64.b64encode(fh.read()).decode("ascii")
                    figs.append({"format": "png", "data": b64})
                try:
                    os.remove(tmp)
                except OSError:
                    pass
            self.eng.eval("close all force; clear CC__figs;", nargout=0)
        except Exception:
            pass
        return figs

    def _capture_workspace(self):
        ws = []
        try:
            names = self.eng.eval("who", nargout=1)
            if isinstance(names, str):
                names = [names]
            for name in names:
                if not name or name.startswith("CC__"):
                    continue
                try:
                    cls = self.eng.eval(f"class({name})", nargout=1)
                    size = self.eng.eval(f"mat2str(size({name}))", nargout=1)
                except Exception:
                    cls, size = "?", "?"
                entry = {"name": name, "class": cls, "size": size}
                entry_val = self._preview_matlab(name, cls)
                if entry_val is not None:
                    entry["value"] = entry_val
                ws.append(entry)
        except Exception:
            pass
        return ws

    def _preview_matlab(self, name, cls):
        """Small numeric/char values travel to the UI (Value column + editor)."""
        try:
            if cls == "char":
                return self.eng.eval(f"{name}(:)'", nargout=1)
            if cls not in ("double", "single", "logical", "int8", "int16",
                           "int32", "int64", "uint8", "uint16", "uint32", "uint64"):
                return None
            if int(self.eng.eval(f"numel({name})", nargout=1)) > 600:
                return None
            raw = self.eng.eval(f"double({name})", nargout=1)
            if isinstance(raw, (int, float)):
                return [[float(raw)]]
            return [[float(v) for v in row] for row in raw]
        except Exception:
            return None

    # -- demo fallback ------------------------------------------------------
    def _run_demo(self, code):
        # A small MATLAB-subset interpreter actually runs the code and renders
        # crisp SVG figures (see mdemo.py). The interpreter is persistent, so
        # variables survive between runs exactly like a MATLAB session.
        interp = self.interp()
        res = interp.run(code)
        figs = []
        for f in res["figures"]:
            item = {"format": "svg", "data": f["svg"], "id": f.get("id")}
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
            "clc": res.get("clc", False),
        }

    # -- filesystem ---------------------------------------------------------
    def fs(self, msg):
        """Working-directory model + file IO for the Files panel and Open/Save."""
        op = msg.get("op", "list")
        try:
            if op == "cwd":
                return {"ok": True, "cwd": self.cwd, "entries": self._list(self.cwd)}
            if op == "list":
                path = os.path.abspath(os.path.expanduser(msg.get("path") or self.cwd))
                if not os.path.isdir(path):
                    return {"ok": False, "error": f"not a folder: {path}"}
                return {"ok": True, "cwd": path, "entries": self._list(path)}
            if op == "chdir":
                path = os.path.abspath(os.path.expanduser(msg.get("path") or self.cwd))
                if not os.path.isdir(path):
                    return {"ok": False, "error": f"not a folder: {path}"}
                self.cwd = path
                if self._interp is not None:
                    self._interp.cwd = path
                if self.mode == "matlab" and self.eng is not None:
                    try:
                        self.eng.cd(path, nargout=0)
                    except Exception:
                        pass
                return {"ok": True, "cwd": path, "entries": self._list(path)}
            if op == "read":
                path = self._resolve(msg.get("path"))
                if os.path.getsize(path) > 4 * 1024 * 1024:
                    return {"ok": False, "error": "file is larger than 4 MB"}
                with open(path, "r", encoding="utf-8", errors="replace") as fh:
                    return {"ok": True, "path": path, "text": fh.read()}
            if op == "write":
                path = self._resolve(msg.get("path"))
                os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
                if msg.get("base64") is not None:     # binary, e.g. a PNG figure
                    with open(path, "wb") as fh:
                        fh.write(base64.b64decode(msg["base64"]))
                else:
                    with open(path, "w", encoding="utf-8") as fh:
                        fh.write(msg.get("text") or "")
                return {"ok": True, "path": path, "entries": self._list(self.cwd)}
            if op == "mkdir":
                path = self._resolve(msg.get("path"))
                os.makedirs(path, exist_ok=True)
                return {"ok": True, "path": path, "entries": self._list(self.cwd)}
            if op == "rename":
                src, dst = self._resolve(msg.get("path")), self._resolve(msg.get("to"))
                os.rename(src, dst)
                return {"ok": True, "path": dst, "entries": self._list(self.cwd)}
            if op == "delete":
                path = self._resolve(msg.get("path"))
                if os.path.isdir(path):
                    os.rmdir(path)          # only empty folders, on purpose
                else:
                    os.remove(path)
                return {"ok": True, "entries": self._list(self.cwd)}
            if op == "home":
                return {"ok": True, "home": os.path.expanduser("~")}
            return {"ok": False, "error": f"unknown fs op: {op}"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def _resolve(self, path):
        path = os.path.expanduser(path or "")
        return path if os.path.isabs(path) else os.path.join(self.cwd, path)

    @staticmethod
    def _list(path):
        entries = []
        try:
            for name in sorted(os.listdir(path), key=lambda n: n.lower()):
                if name.startswith("."):
                    continue
                full = os.path.join(path, name)
                try:
                    st = os.stat(full)
                    is_dir = os.path.isdir(full)
                    entries.append({
                        "name": name,
                        "dir": is_dir,
                        "size": 0 if is_dir else st.st_size,
                        "mtime": int(st.st_mtime),
                    })
                except OSError:
                    continue
        except OSError:
            pass
        entries.sort(key=lambda e: (not e["dir"], e["name"].lower()))
        return entries


# ----------------------------------------------------------------------------
# Transports
# ----------------------------------------------------------------------------

def handle(engine, msg):
    """Dispatch one request dict -> response dict."""
    kind = msg.get("kind", "run")
    try:
        if kind == "status":
            return engine.status()
        if kind == "stop":
            return engine.stop()
        if kind == "reset":
            return engine.reset()
        if kind == "figlim":
            return engine.figlim(msg)
        if kind == "figfit":
            return engine.figfit(msg)
        if kind == "lint":
            return engine.lint(msg.get("code", ""))
        if kind == "varset":
            with engine._lock:
                return engine.varset(msg)
        if kind == "fs":
            return engine.fs(msg)
        if kind == "run":
            with engine._lock:          # one run at a time, like one MATLAB session
                return engine.run(msg.get("code", ""))
        return {"ok": False, "error": f"unknown kind: {kind}"}
    except Exception as e:
        return {
            "ok": False, "mode": engine.status()["mode"], "stdout": "",
            "stderr": "", "error": f"sidecar error: {e}\n{traceback.format_exc()}",
            "figures": [], "workspace": [],
        }


def serve_stdio(engine):
    """Concurrent stdio transport: every response echoes the request id."""
    # The app speaks UTF-8. Windows pipes default to the ANSI code page (cp1252),
    # where bytes such as 0x81 (in "ρ") do not decode and kill this loop. The
    # app also passes -X utf8; this covers a PYTHONIOENCODING that overrides it.
    for stream in (sys.stdin, sys.stdout):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    if sys.stderr is None:                 # GUI parent with no console
        sys.stderr = open(os.devnull, "w")
    # Runs execute on worker threads. Windows gives a thread a ~1-2 MB stack,
    # so deep user recursion could overflow the C stack and kill the process
    # outright; a roomy stack lets Python's RecursionError fire first.
    try:
        threading.stack_size(64 * 1024 * 1024)
    except (ValueError, RuntimeError):
        pass
    print(f"[sidecar] ready mode={engine.status()['mode']}", file=sys.stderr, flush=True)
    write_lock = threading.Lock()

    def reply(msg, resp):
        if "id" in msg:
            resp = dict(resp)
            resp["id"] = msg["id"]
        with write_lock:
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()

    workers = []
    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if msg.get("kind") == "run":
            # off the reader thread, so `stop` can still be read and answered
            t = threading.Thread(target=lambda m=msg: reply(m, handle(engine, m)),
                                 daemon=True)
            t.start()
            workers = [w for w in workers if w.is_alive()] + [t]
        else:
            reply(msg, handle(engine, msg))
    # stdin closed: let a run that is still in flight finish and answer, rather
    # than dropping its response on the way out.
    for t in workers:
        t.join(timeout=30)


def serve_http(engine, port):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _cors(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")

        def do_OPTIONS(self):
            self.send_response(204)
            self._cors()
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self):
            if self.path.rstrip("/") in ("/status", "/health"):
                self._json(handle(engine, {"kind": "status"}))
            else:
                self._json({"error": "use POST /run"}, code=404)

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length) if length else b"{}"
            try:
                msg = json.loads(body or b"{}")
            except json.JSONDecodeError:
                msg = {}
            msg.setdefault("kind", "run")
            self._json(handle(engine, msg))

        def _json(self, obj, code=200):
            payload = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self._cors()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass  # quiet

    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    srv.daemon_threads = True
    print(f"[sidecar] HTTP on http://127.0.0.1:{port}  mode={engine.status()['mode']}",
          file=sys.stderr, flush=True)
    srv.serve_forever()


def main():
    engine = Engine()
    if "--http" in sys.argv:
        idx = sys.argv.index("--http")
        port = 8765
        if idx + 1 < len(sys.argv):
            try:
                port = int(sys.argv[idx + 1])
            except ValueError:
                pass
        serve_http(engine, port)
    else:
        serve_stdio(engine)


if __name__ == "__main__":
    main()

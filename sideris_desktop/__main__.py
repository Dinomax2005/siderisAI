"""Entry point for ``python -m sideris_desktop`` and the packaged app.

No arguments opens the desktop window.  Anything else is treated as a
CLI command, so ``Sideris.exe chat`` and ``python -m sideris_desktop
status`` behave the same way.

The frozen Windows/macOS builds are windowed, which means ``sys.stdout``
can be ``None``.  Every piece of output therefore goes through
:func:`_write_stdout`, and ``--selftest`` writes a JSON report to a file
instead of relying on a terminal.  The release workflow uses that to
prove the *packaged* binary really imports Tk, the app and the bundled
inference runtime.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

REQUIRED_SELFTEST_CHECKS = ("version", "assets", "tkinter", "app", "config")


def _write_stdout(text: str) -> None:
    """Print without assuming a terminal exists (windowed builds)."""
    stream = sys.stdout
    if stream is None:
        return
    try:
        stream.write(text + "\n")
        stream.flush()
    except Exception:
        pass


def _write_stderr(text: str) -> None:
    stream = sys.stderr
    if stream is None:
        return
    try:
        stream.write(text + "\n")
        stream.flush()
    except Exception:
        pass


def _run_selftest(target: str) -> int:
    """Exercise the frozen bundle and write a JSON report to *target*."""
    report: dict = {
        "ok": False,
        "frozen": bool(getattr(sys, "frozen", False)),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": sys.platform,
        "checks": {},
        "errors": {},
        "warnings": {},
    }

    def check(name: str, func, required: bool = True) -> None:
        try:
            report["checks"][name] = func()
        except Exception as exc:  # noqa: BLE001 - report, never crash
            detail = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            report["errors"][name] = detail
            if required:
                report["checks"][name] = None
            else:
                report["warnings"][name] = detail

    def version_check() -> str:
        from . import __app_name__, __version__

        return f"{__app_name__} {__version__}"

    def assets_check() -> str:
        from .config import asset_dir

        icon = asset_dir() / "logo-256.png"
        if not icon.is_file():
            raise FileNotFoundError(f"window icon missing: {icon}")
        return str(icon)

    def tkinter_check() -> str:
        import tkinter

        # TkVersion/TclVersion are the attributes Tkinter actually exposes;
        # there is no TkPatchLevel.
        return f"Tk {tkinter.TkVersion} / Tcl {tkinter.TclVersion}"

    def app_check() -> str:
        import sideris_desktop.app as app_module

        if not hasattr(app_module, "SiderisApp"):
            raise AttributeError("sideris_desktop.app has no SiderisApp")
        return "imported"

    def config_check() -> str:
        from .config import paths

        return str(paths().root)

    def llama_check() -> str:
        from .installer import activate_runtime

        activate_runtime()
        import llama_cpp

        return str(getattr(llama_cpp, "__version__", "unknown"))

    def inference_check() -> str:
        from .installer import inference_ready

        if not inference_ready():
            raise RuntimeError("no inference backend importable")
        return "ready"

    check("version", version_check)
    check("assets", assets_check)
    check("tkinter", tkinter_check)
    check("app", app_check)
    check("config", config_check)
    check("llama_cpp", llama_check, required=False)
    check("inference", inference_check, required=False)

    report["ok"] = all(report["checks"].get(name) for name in REQUIRED_SELFTEST_CHECKS)

    output = Path(target).expanduser()
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        _write_stderr(f"sideris: could not write self-test report to {output}: {exc}")
        return 1

    if not report["ok"]:
        _write_stderr(f"sideris: self-test failed, see {output}")
    return 0 if report["ok"] else 1


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    # Make pip-installed extras (llama.cpp / training wheels) importable
    # before anything tries to use them.
    try:
        from .installer import activate_runtime

        activate_runtime()
    except Exception:
        pass

    if len(args) == 2 and args[0] == "--selftest":
        return _run_selftest(args[1])

    if args in (["--version"], ["-V"]):
        from . import __app_name__, __version__

        _write_stdout(f"{__app_name__} {__version__}")
        return 0

    if args == ["--smoke"]:
        from .app import run_gui

        return run_gui(smoke=True)

    if not args:
        from .app import run_gui

        return run_gui()

    from .cli import main as cli_main

    return cli_main(args)


if __name__ == "__main__":
    raise SystemExit(main())

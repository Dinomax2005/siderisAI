"""Optional background service and start-at-login integration.

Every function here is opt-in.  The setup wizard calls them only after the
user ticks the corresponding permission, and the functions re-check the
permission themselves before touching the operating system.

  Windows  Scheduled Task "SiderisBackground", trigger ONLOGON, no admin
  Linux    systemd --user unit ~/.config/systemd/user/sideris.service
  macOS    LaunchAgent ~/Library/LaunchAgents/ai.sideris.desktop.plist

The service itself is ``sideris service run``: it starts the loopback
HTTP server, writes ``state/service-heartbeat.json`` every 15 seconds and
saves a final stopped state on SIGTERM/SIGINT.
"""

from __future__ import annotations

import json
import os
import platform
import plistlib
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from . import __app_name__, __version__
from .config import (
    APP_ID,
    LAUNCH_AGENT_LABEL,
    SERVICE_NAME,
    SERVICE_NAME_LINUX,
    Paths,
    SettingsStore,
    app_executable,
    is_frozen,
    paths,
    read_json,
    utc_now,
    write_json,
)

HEARTBEAT_SECONDS = 15


class ServiceError(RuntimeError):
    pass


def _system() -> str:
    return platform.system()


def _permission_enabled() -> bool:
    return SettingsStore().load().permissions.background_service


def _run(command: list[str], *, check: bool = True, timeout: float = 60) -> subprocess.CompletedProcess:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            stdin=subprocess.DEVNULL,
        )
    except FileNotFoundError as exc:
        raise ServiceError(f"{command[0]} is not available on this system.") from exc
    except subprocess.TimeoutExpired as exc:
        raise ServiceError(f"{command[0]} timed out.") from exc
    if check and result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise ServiceError(f"{command[0]} failed: {detail or 'exit ' + str(result.returncode)}")
    return result


def _gui_command() -> list[str]:
    if is_frozen():
        return [str(Path(sys.executable).resolve())]
    return [sys.executable, "-m", "sideris_desktop"]


def _service_command(port: int | None = None) -> list[str]:
    args = ["service", "run", "--quiet"]
    if port:
        args += ["--port", str(port)]
    if is_frozen():
        return [str(Path(sys.executable).resolve())] + args
    python = sys.executable
    if os.name == "nt":
        candidate = Path(python).with_name("pythonw.exe")
        if candidate.is_file():
            python = str(candidate)
    return [python, "-m", "sideris_desktop"] + args


# --------------------------------------------------------------------------
# status
# --------------------------------------------------------------------------
def heartbeat_info(paths_obj: Paths | None = None) -> dict:
    obj = paths_obj or paths()
    data = read_json(obj.heartbeat, {}) or {}
    if not isinstance(data, dict):
        return {}
    pid = data.get("pid")
    alive = False
    if isinstance(pid, int):
        if os.name == "nt":
            alive = _pid_running(pid)
        else:
            try:
                os.kill(pid, 0)
                alive = True
            except (OSError, ProcessLookupError):
                alive = False
    if data.get("running") and not alive:
        data["stale"] = True
    return data


def _pid_running(pid: int) -> bool:
    import ctypes

    handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))
    if not handle:
        return False
    code = ctypes.c_ulong()
    try:
        ok = ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        return bool(ok) and code.value == 259
    finally:
        ctypes.windll.kernel32.CloseHandle(handle)


def service_status(paths_obj: Paths | None = None) -> dict:
    obj = paths_obj or paths()
    state = read_json(obj.service_state, {}) or {}
    heartbeat = heartbeat_info(obj)
    system = _system()
    supported = system in {"Windows", "Linux", "Darwin"}
    return {
        "installed": bool(state.get("installed")),
        "mode": state.get("mode"),
        "system": system,
        "supported": supported,
        "running": bool(heartbeat.get("running") and not heartbeat.get("stale")),
        "pid": heartbeat.get("pid"),
        "last_heartbeat": heartbeat.get("updated_at"),
        "heartbeat": heartbeat,
        "installed_at": state.get("installed_at"),
        "start_at_login": login_item_status(obj),
        "path": state.get("path"),
    }


# --------------------------------------------------------------------------
# install / uninstall
# --------------------------------------------------------------------------
def install_service(*, permission: bool | None = None) -> dict:
    allowed = _permission_enabled() if permission is None else permission
    if not allowed:
        raise PermissionError(
            "The background service permission is off. Nothing was installed."
        )
    system = _system()
    obj = paths()
    obj.ensure()
    port = SettingsStore(obj.root).load().port
    command = _service_command(port)

    if system == "Windows":
        command_line = subprocess.list2cmdline(command)
        _run(
            [
                "schtasks", "/Create", "/TN", SERVICE_NAME,
                "/TR", command_line, "/SC", "ONLOGON",
                "/RL", "LIMITED", "/F",
            ]
        )
        _run(["schtasks", "/Run", "/TN", SERVICE_NAME], check=False)
        mode = "schtasks"
    elif system == "Linux":
        unit_dir = Path.home() / ".config" / "systemd" / "user"
        unit_dir.mkdir(parents=True, exist_ok=True)
        unit_path = unit_dir / SERVICE_NAME_LINUX
        unit_path.write_text(
            "[Unit]\n"
            f"Description={__app_name__} local AI service\n"
            "After=default.target\n\n"
            "[Service]\n"
            "Type=simple\n"
            f"ExecStart={' '.join(_quote(part) for part in command)}\n"
            "Restart=on-failure\n"
            "RestartSec=5\n"
            f"Environment=SIDERIS_HOME={obj.root}\n\n"
            "[Install]\n"
            "WantedBy=default.target\n",
            encoding="utf-8",
        )
        _run(["systemctl", "--user", "daemon-reload"])
        _run(["systemctl", "--user", "enable", "--now", SERVICE_NAME_LINUX])
        mode = "systemd-user"
    elif system == "Darwin":
        plist_path = _launch_agent_path(LAUNCH_AGENT_LABEL)
        plist_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "Label": LAUNCH_AGENT_LABEL,
            "ProgramArguments": command,
            "RunAtLoad": True,
            "KeepAlive": False,
            "ProcessType": "Background",
            "StandardOutPath": str(obj.logs / "service.log"),
            "StandardErrorPath": str(obj.logs / "service.log"),
            "EnvironmentVariables": {"SIDERIS_HOME": str(obj.root)},
        }
        with open(plist_path, "wb") as fh:
            plistlib.dump(payload, fh)
        _launchctl_load(plist_path)
        mode = "launchagent"
    else:
        raise ServiceError(f"Background service is not supported on {system}.")

    state = {
        "installed": True,
        "mode": mode,
        "command": command,
        "installed_at": utc_now(),
        "version": __version__,
        "path": str(obj.service_state),
    }
    write_json(obj.service_state, state)
    return state


def uninstall_service(*, permission: bool | None = None) -> dict:
    # Uninstalling is always allowed; removing something the user no longer
    # wants must never be blocked by a permission.
    system = _system()
    obj = paths()
    errors: list[str] = []
    try:
        if system == "Windows":
            _run(["schtasks", "/Delete", "/TN", SERVICE_NAME, "/F"], check=False)
        elif system == "Linux":
            _run(["systemctl", "--user", "disable", "--now", SERVICE_NAME_LINUX], check=False)
            unit_path = Path.home() / ".config" / "systemd" / "user" / SERVICE_NAME_LINUX
            if unit_path.exists():
                unit_path.unlink()
            _run(["systemctl", "--user", "daemon-reload"], check=False)
        elif system == "Darwin":
            plist_path = _launch_agent_path(LAUNCH_AGENT_LABEL)
            if plist_path.exists():
                _launchctl_unload(plist_path)
                plist_path.unlink()
    except (ServiceError, OSError) as exc:
        errors.append(str(exc))

    write_json(
        obj.service_state,
        {
            "installed": False,
            "mode": None,
            "uninstalled_at": utc_now(),
            "errors": errors,
        },
    )
    try:
        if obj.heartbeat.exists():
            obj.heartbeat.unlink()
    except OSError:
        pass
    return {"installed": False, "errors": errors}


def _quote(part: str) -> str:
    if not part or any(ch.isspace() for ch in part):
        return '"' + part.replace('"', '\\"') + '"'
    return part


def _launch_agent_path(label: str) -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{label}.plist"


def _launchctl_load(plist_path: Path) -> None:
    domain = f"gui/{os.getuid()}"
    result = _run(["launchctl", "bootstrap", domain, str(plist_path)], check=False)
    if result.returncode != 0:
        _run(["launchctl", "load", "-w", str(plist_path)])


def _launchctl_unload(plist_path: Path) -> None:
    domain = f"gui/{os.getuid()}"
    result = _run(["launchctl", "bootout", domain, str(plist_path)], check=False)
    if result.returncode != 0:
        _run(["launchctl", "unload", "-w", str(plist_path)], check=False)


# --------------------------------------------------------------------------
# start at login (the window, not the service)
# --------------------------------------------------------------------------
def _startup_shortcut_windows() -> Path:
    base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
    return Path(base) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup" / f"{__app_name__}.cmd"


def _linux_autostart_path() -> Path:
    base = Path(os.environ.get("XDG_CONFIG_HOME") or (Path.home() / ".config"))
    return base / "autostart" / "sideris.desktop"


def login_item_status(paths_obj: Paths | None = None) -> bool:
    system = _system()
    if system == "Windows":
        return _startup_shortcut_windows().exists()
    if system == "Linux":
        return _linux_autostart_path().exists()
    if system == "Darwin":
        return _launch_agent_path(f"{APP_ID}.login").exists()
    return False


def install_login_item(*, permission: bool | None = None) -> bool:
    allowed = (
        SettingsStore().load().permissions.start_at_login
        if permission is None
        else permission
    )
    if not allowed:
        raise PermissionError("Start at login permission is off. Nothing was changed.")
    system = _system()
    command = _gui_command()
    if system == "Windows":
        path = _startup_shortcut_windows()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "@echo off\r\n"
            + "start \"\" "
            + subprocess.list2cmdline(command)
            + "\r\n",
            encoding="utf-8",
        )
        return True
    if system == "Linux":
        path = _linux_autostart_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        exec_line = " ".join(_quote(part) for part in command)
        path.write_text(
            "[Desktop Entry]\n"
            "Type=Application\n"
            f"Name={__app_name__}\n"
            "Comment=Private local AI assistant\n"
            f"Exec={exec_line}\n"
            "Terminal=false\n"
            "X-GNOME-Autostart-enabled=true\n",
            encoding="utf-8",
        )
        return True
    if system == "Darwin":
        path = _launch_agent_path(f"{APP_ID}.login")
        path.parent.mkdir(parents=True, exist_ok=True)
        app_path = _app_bundle_path()
        if app_path:
            args = ["/usr/bin/open", "-a", str(app_path)]
        else:
            args = command
        with open(path, "wb") as fh:
            plistlib.dump(
                {
                    "Label": f"{APP_ID}.login",
                    "ProgramArguments": args,
                    "RunAtLoad": True,
                    "KeepAlive": False,
                },
                fh,
            )
        return True
    raise ServiceError(f"Start at login is not supported on {system}.")


def remove_login_item() -> bool:
    system = _system()
    removed = False
    if system == "Windows":
        path = _startup_shortcut_windows()
    elif system == "Linux":
        path = _linux_autostart_path()
    elif system == "Darwin":
        path = _launch_agent_path(f"{APP_ID}.login")
        if path.exists():
            _launchctl_unload(path)
    else:
        return False
    try:
        if path.exists():
            path.unlink()
            removed = True
    except OSError:
        pass
    return removed


def _app_bundle_path() -> Path | None:
    if not is_frozen():
        return None
    exe = Path(sys.executable).resolve()
    for parent in exe.parents:
        if parent.suffix == ".app":
            return parent
    return None


# --------------------------------------------------------------------------
# the service process
# --------------------------------------------------------------------------
def run_service(port: int | None = None, *, quiet: bool = False) -> int:
    """Run the local service with a heartbeat and graceful shutdown."""
    from .server import LocalRuntime, run_server

    obj = paths()
    obj.ensure()
    if not _permission_enabled():
        message = (
            "The background service permission is off. Enable it in Settings, "
            "or run 'sideris serve' for a foreground service."
        )
        print(message, file=sys.stderr)
        return 3

    runtime = LocalRuntime(obj.root)
    settings_port = port or SettingsStore(obj.root).load().port
    started = utc_now()
    stop = threading.Event()

    def heartbeat_loop() -> None:
        while not stop.is_set():
            write_json(
                obj.heartbeat,
                {
                    "running": True,
                    "pid": os.getpid(),
                    "started_at": started,
                    "updated_at": utc_now(),
                    "version": __version__,
                    "port": settings_port,
                },
            )
            stop.wait(HEARTBEAT_SECONDS)

    def graceful(signum=None, frame=None):  # noqa: ARG001
        stop.set()
        write_json(
            obj.heartbeat,
            {
                "running": False,
                "pid": os.getpid(),
                "started_at": started,
                "stopped_at": utc_now(),
                "reason": f"signal {signum}" if signum else "shutdown",
                "version": __version__,
            },
        )

    thread = threading.Thread(target=heartbeat_loop, daemon=True)
    thread.start()
    try:
        return run_server(
            obj.root,
            port=settings_port,
            runtime=runtime,
            announce=not quiet,
            on_signal=graceful,
        )
    finally:
        graceful()
        stop.set()
        thread.join(timeout=2)


__all__ = [
    "ServiceError",
    "install_service",
    "uninstall_service",
    "service_status",
    "install_login_item",
    "remove_login_item",
    "login_item_status",
    "heartbeat_info",
    "run_service",
    "app_executable",
    "json",
]

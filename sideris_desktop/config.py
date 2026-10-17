"""Configuration, paths and permissions for Sideris Desktop.

Everything Sideris stores lives in one per-user data directory:

  Windows   %LOCALAPPDATA%\\Sideris
  macOS     ~/Library/Application Support/Sideris
  Linux     ~/.local/share/sideris  (XDG_DATA_HOME honoured)

``SIDERIS_HOME`` overrides the location.  ``LUNO_HOME`` is still accepted
for people upgrading from the 4.x "Luno" builds.

Permissions are deliberately conservative: every optional capability is
off until the person using the computer ticks the box.  Nothing downloads
a model, installs a background service, trains, or starts at login merely
because the app was opened.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from . import __app_name__, __legacy_name__

APP_NAME = __app_name__
APP_ID = "ai.sideris.desktop"
SERVICE_NAME = "SiderisBackground"
SERVICE_NAME_LINUX = "sideris.service"
LAUNCH_AGENT_LABEL = "ai.sideris.desktop"
DEFAULT_PORT = 8765
SETTINGS_FILENAME = "settings.json"


# --------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------
def is_frozen() -> bool:
    """True when running from a PyInstaller bundle."""
    return bool(getattr(sys, "frozen", False))


def bundle_dir() -> Path:
    """Directory that holds read-only files shipped with the app."""
    if is_frozen():
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parents[1]


def asset_dir() -> Path:
    """Where logo and icon files live (source tree, bundle or install share)."""
    override = os.environ.get("SIDERIS_ASSETS")
    if override:
        return Path(override).expanduser()
    return bundle_dir() / "assets"


def app_executable() -> str:
    """The command that launches the graphical app again.

    In a frozen build that is the Sideris executable itself; from source
    it is ``python -m sideris_desktop``.
    """
    if is_frozen():
        return str(Path(sys.executable).resolve())
    return f"{sys.executable} -m {__name__.split('.')[0]}"


def default_data_dir() -> Path:
    override = os.environ.get("SIDERIS_HOME") or os.environ.get("LUNO_HOME")
    if override:
        return Path(override).expanduser()

    system = platform.system()
    if system == "Windows":
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        if base:
            return Path(base) / APP_NAME
        return Path.home() / "AppData" / "Local" / APP_NAME
    if system == "Darwin":
        return Path.home() / "Library" / "Application Support" / APP_NAME
    xdg = os.environ.get("XDG_DATA_HOME")
    base = Path(xdg) if xdg else Path.home() / ".local" / "share"
    return base / APP_NAME.lower()


@dataclass(frozen=True)
class Paths:
    """All filesystem locations used by the product."""

    root: Path

    # --- top level -------------------------------------------------------
    @property
    def settings(self) -> Path:
        return self.root / SETTINGS_FILENAME

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def chats(self) -> Path:
        return self.root / "chats"

    @property
    def state(self) -> Path:
        return self.root / "state"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def workspace(self) -> Path:
        return self.root / "workspace"

    @property
    def memory(self) -> Path:
        return self.root / "memory.jsonl"

    @property
    def datasets(self) -> Path:
        return self.root / "datasets"

    @property
    def training(self) -> Path:
        return self.root / "training"

    @property
    def runs(self) -> Path:
        return self.training / "runs"

    @property
    def checkpoints(self) -> Path:
        return self.training / "checkpoints"

    @property
    def runtime(self) -> Path:
        return self.root / "runtime"

    @property
    def runtime_src(self) -> Path:
        return self.runtime / "src"

    @property
    def runtime_pkgs(self) -> Path:
        return self.runtime / "pkgs"

    @property
    def training_venv(self) -> Path:
        return self.runtime / "training-venv"

    # --- state files -----------------------------------------------------
    @property
    def setup_state(self) -> Path:
        return self.state / "setup-state.json"

    @property
    def training_state(self) -> Path:
        return self.state / "training-state.json"

    @property
    def service_state(self) -> Path:
        return self.state / "service.json"

    @property
    def heartbeat(self) -> Path:
        return self.state / "service-heartbeat.json"

    @property
    def server_info(self) -> Path:
        return self.state / "server.json"

    def ensure(self) -> None:
        for path in (
            self.root,
            self.models,
            self.chats,
            self.state,
            self.logs,
            self.workspace,
            self.datasets,
            self.training,
            self.runs,
            self.checkpoints,
            self.runtime,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def describe(self) -> dict[str, str]:
        return {
            "root": str(self.root),
            "models": str(self.models),
            "chats": str(self.chats),
            "workspace": str(self.workspace),
            "datasets": str(self.datasets),
            "logs": str(self.logs),
            "runtime": str(self.runtime),
        }


def paths(root: str | os.PathLike[str] | None = None) -> Paths:
    return Paths(Path(root).expanduser() if root else default_data_dir())


# --------------------------------------------------------------------------
# atomic json helpers
# --------------------------------------------------------------------------
def read_json(path: Path, default: Any = None) -> Any:
    """Read JSON, returning ``default`` when missing or corrupt.

    A corrupt file is renamed once to ``<name>.corrupt`` so the user (and
    support) can inspect it instead of losing evidence silently.
    """
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        return default
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        try:
            if path.exists():
                path.replace(path.with_suffix(path.suffix + ".corrupt"))
        except OSError:
            pass
        return default


def write_json(path: Path, payload: Any) -> None:
    """Atomically write JSON (write temp, fsync, replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    finally:
        try:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
        except OSError:
            pass


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# --------------------------------------------------------------------------
# permissions
# --------------------------------------------------------------------------
def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "enabled"}
    return default


@dataclass
class Permissions:
    """Opt-in switches.  Every one of them starts OFF."""

    model_download: bool = False
    background_service: bool = False
    training: bool = False
    start_at_login: bool = False

    def to_dict(self) -> dict[str, bool]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Any) -> "Permissions":
        if not isinstance(data, dict):
            return cls()
        return cls(
            **{
                f.name: _as_bool(data.get(f.name, False))
                for f in fields(cls)
            }
        )

    def enabled(self) -> list[str]:
        return [key for key, value in self.to_dict().items() if value]

    def summary(self) -> str:
        return ", ".join(self.enabled()) or "all off"


# --------------------------------------------------------------------------
# settings
# --------------------------------------------------------------------------
@dataclass
class Settings:
    model: str = "qwen2.5-1.5b"
    thinking_level: str = "medium"
    port: int = DEFAULT_PORT
    context_size: int = 4096
    gpu_layers: int = 0
    permissions: Permissions = field(default_factory=Permissions)
    first_run: bool = True
    onboarding_complete: bool = False
    theme: str = "stone"
    last_chat_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["permissions"] = self.permissions.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: Any) -> "Settings":
        data = data if isinstance(data, dict) else {}
        default = cls()
        try:
            port = int(data.get("port", default.port))
        except (TypeError, ValueError):
            port = default.port
        try:
            context_size = int(data.get("context_size", default.context_size))
        except (TypeError, ValueError):
            context_size = default.context_size
        context_size = max(1024, min(context_size, 32768))
        try:
            gpu_layers = int(data.get("gpu_layers", default.gpu_layers))
        except (TypeError, ValueError):
            gpu_layers = default.gpu_layers
        level = str(data.get("thinking_level", default.thinking_level)).lower()
        if level not in {"low", "medium", "high", "max"}:
            level = default.thinking_level
        return cls(
            model=str(data.get("model") or default.model),
            thinking_level=level,
            port=max(1024, min(port, 65535)),
            context_size=context_size,
            gpu_layers=max(-1, min(gpu_layers, 200)),
            permissions=Permissions.from_dict(data.get("permissions")),
            first_run=_as_bool(data.get("first_run", default.first_run), default.first_run),
            onboarding_complete=_as_bool(
                data.get("onboarding_complete", default.onboarding_complete),
                default.onboarding_complete,
            ),
            theme=str(data.get("theme") or default.theme),
            last_chat_id=data.get("last_chat_id") or None,
        )


class SettingsStore:
    """Tiny JSON-backed settings store with atomic saves."""

    def __init__(self, root: str | os.PathLike[str] | None = None) -> None:
        self.paths = paths(root)
        self._cache: Settings | None = None

    @property
    def root(self) -> Path:
        return self.paths.root

    def load(self, refresh: bool = False) -> Settings:
        if self._cache is None or refresh:
            self._cache = Settings.from_dict(read_json(self.paths.settings, {}))
        return self._cache

    def save(self, settings: Settings) -> Settings:
        self.paths.ensure()
        write_json(self.paths.settings, settings.to_dict())
        self._cache = settings
        return settings

    def update(self, **changes: Any) -> Settings:
        settings = self.load()
        for key, value in changes.items():
            if hasattr(settings, key):
                setattr(settings, key, value)
        return self.save(settings)

    def set_permission(self, name: str, value: bool) -> Settings:
        settings = self.load()
        if name not in {f.name for f in fields(Permissions)}:
            raise KeyError(f"unknown permission: {name}")
        setattr(settings.permissions, name, bool(value))
        return self.save(settings)


# --------------------------------------------------------------------------
# misc platform helpers
# --------------------------------------------------------------------------
def platform_label() -> str:
    system = platform.system()
    return {"Windows": "Windows", "Darwin": "macOS"}.get(system, system or "Unknown")


def open_path(path: Path) -> None:
    """Open a file or folder in the desktop's file manager."""
    import subprocess

    path = Path(path)
    if platform.system() == "Windows":
        os.startfile(str(path))  # type: ignore[attr-defined]
    elif platform.system() == "Darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


def total_ram_gb() -> float:
    """Best-effort physical memory size, in GiB (0.0 when unknown)."""
    system = platform.system()
    try:
        if system == "Linux":
            with open("/proc/meminfo", "r", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("MemTotal:"):
                        return round(int(line.split()[1]) / 1024 / 1024, 1)
        elif system == "Darwin":
            import subprocess

            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
            return round(int(out.strip()) / 1024**3, 1)
        elif system == "Windows":
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return round(stat.ullTotalPhys / 1024**3, 1)
    except Exception:
        pass
    return 0.0

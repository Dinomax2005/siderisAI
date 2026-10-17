"""Resumable, permission-aware installation.

The first-run wizard calls :class:`Setup`.  Every step is recorded in
``state/setup-state.json`` *before and after* it runs, so closing the app
(or losing power) in the middle of a download or a pip install means the
next attempt picks up where it stopped instead of starting over.

Steps
-----
``directories``  create the per-user data folders, copy the training source
``inference``    make sure a local inference runtime is available
``model``        download the selected GGUF (only with model_download)
``training``     install torch/transformers/peft/datasets/accelerate (only
                 with training; into a private venv, never the system)
``service``      install the background service (only with background_service)
``login``        start the window at login (only with start_at_login)

The shipped installers bundle llama-cpp-python, so ``inference`` is
normally already satisfied.  From a source checkout the step installs it.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import __app_name__, __version__
from .config import (
    Paths,
    Settings,
    SettingsStore,
    app_executable,
    bundle_dir,
    is_frozen,
    paths,
    read_json,
    utc_now,
    write_json,
)
from .downloads import Progress, download, human_bytes
from .models import DEFAULT_MODEL, ModelManager, catalogue, get_spec, recommend_key
from . import service as service_module

LLAMA_CPP_VERSION = "0.3.36"
TRAINING_PACKAGES = [
    "transformers>=4.40,<5",
    "peft>=0.10",
    "datasets>=2.18",
    "accelerate>=0.29",
    "safetensors>=0.4",
    "sentencepiece>=0.1.99",
]
SETUP_STATE_VERSION = 1


class SetupError(RuntimeError):
    pass


class StepFailed(SetupError):
    pass


@dataclass
class Step:
    key: str
    title: str
    permission: str | None = None
    description: str = ""


STEPS: tuple[Step, ...] = (
    Step("directories", "Create data folders", None, "Settings, chats, models and logs."),
    Step("inference", "Local inference runtime", "model_download", "llama.cpp for GGUF models."),
    Step("model", "Download the model", "model_download", "Large file, resumable."),
    Step("training", "Fine-tuning runtime", "training", "torch + transformers + peft."),
    Step("service", "Background service", "background_service", "Start with the computer."),
    Step("login", "Open at login", "start_at_login", "Launch the window after sign-in."),
)

STEP_TITLES = {step.key: step.title for step in STEPS}


# --------------------------------------------------------------------------
# runtime helpers
# --------------------------------------------------------------------------
def runtime_package_dir(root: str | os.PathLike[str] | None = None) -> Path:
    return paths(root).runtime_pkgs


def activate_runtime(root: str | os.PathLike[str] | None = None) -> None:
    """Make pip-installed extras importable, including DLL directories."""
    package_dir = runtime_package_dir(root)
    if not package_dir.is_dir():
        return
    path = str(package_dir)
    if path not in sys.path:
        sys.path.insert(0, path)
    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        for dll_dir in (
            package_dir,
            package_dir / "llama_cpp" / "lib",
            package_dir / "torch" / "lib",
        ):
            if dll_dir.is_dir():
                try:
                    os.add_dll_directory(str(dll_dir))  # type: ignore[attr-defined]
                except OSError:
                    pass


def inference_ready(root: str | os.PathLike[str] | None = None) -> bool:
    activate_runtime(root)
    try:
        import llama_cpp  # noqa: F401

        return True
    except Exception:
        pass
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401

        return True
    except Exception:
        return False


def _prebuilt_wheel_url() -> str | None:
    system = platform.system()
    machine = platform.machine().lower()
    base = (
        "https://github.com/abetlen/llama-cpp-python/releases/download/"
        f"v{LLAMA_CPP_VERSION}"
    )
    if system == "Windows" and machine in {"amd64", "x86_64"}:
        name = f"llama_cpp_python-{LLAMA_CPP_VERSION}-py3-none-win_amd64.whl"
    elif system == "Linux" and machine in {"x86_64", "amd64"}:
        name = (
            f"llama_cpp_python-{LLAMA_CPP_VERSION}-py3-none-"
            "manylinux2014_x86_64.manylinux_2_17_x86_64.whl"
        )
    elif system == "Linux" and machine in {"aarch64", "arm64"}:
        name = (
            f"llama_cpp_python-{LLAMA_CPP_VERSION}-py3-none-"
            "manylinux2014_aarch64.manylinux_2_17_aarch64.whl"
        )
    elif system == "Darwin" and machine in {"arm64", "aarch64"}:
        name = f"llama_cpp_python-{LLAMA_CPP_VERSION}-py3-none-macosx_11_0_arm64.whl"
    else:
        return None
    return f"{base}/{name}"


def _pip_env(paths_obj: Paths) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "PIP_NO_INPUT": "1",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_CACHE_DIR": str(paths_obj.runtime / "pip-cache"),
            "PYTHONUNBUFFERED": "1",
            "SIDERIS_HOME": str(paths_obj.root),
        }
    )
    return env


def find_system_python() -> str | None:
    """A Python with pip + venv, used for the optional training runtime."""
    if not is_frozen():
        return sys.executable
    candidates: list[list[str]] = []
    if os.name == "nt":
        candidates += [["py", "-3.11"], ["py", "-3"], ["python"], ["python3"]]
    else:
        candidates += [["python3"], ["python"]]
        for minor in ("13", "12", "11", "10"):
            candidates.insert(0, [f"python3.{minor}"])
    seen: set[str] = set()
    for command in candidates:
        try:
            result = subprocess.run(
                command + ["-c", "import sys, venv, ensurepip; print(sys.executable)"],
                capture_output=True,
                text=True,
                timeout=20,
                stdin=subprocess.DEVNULL,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if result.returncode == 0:
            value = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
            if value and value not in seen:
                seen.add(value)
                version = subprocess.run(
                    command + ["-c", "import sys; print('%d.%d' % sys.version_info[:2])"],
                    capture_output=True,
                    text=True,
                    timeout=20,
                ).stdout.strip()
                if version and tuple(int(x) for x in version.split(".")) >= (3, 10):
                    return value
    return None


# --------------------------------------------------------------------------
# setup
# --------------------------------------------------------------------------
class Setup:
    """Runs (and resumes) every installation step."""

    def __init__(
        self,
        root: str | os.PathLike[str] | None = None,
        settings_store: SettingsStore | None = None,
        on_log: Callable[[str], None] | None = None,
        on_progress: Callable[[float, str], None] | None = None,
    ) -> None:
        self.paths: Paths = paths(root)
        self.settings_store = settings_store or SettingsStore(self.paths.root)
        self.models = ModelManager(self.paths.root)
        self.on_log = on_log or (lambda line: None)
        self.on_progress = on_progress or (lambda percent, message: None)
        self._cancel = threading.Event()
        self._process: subprocess.Popen | None = None
        self._log_path = self.paths.logs / "setup.log"

    # -- state -----------------------------------------------------------
    def state(self) -> dict:
        data = read_json(self.paths.setup_state, {}) or {}
        if not isinstance(data, dict):
            data = {}
        data.setdefault("version", SETUP_STATE_VERSION)
        data.setdefault("steps", {})
        if not isinstance(data["steps"], dict):
            data["steps"] = {}
        return data

    def _save_state(self, data: dict) -> None:
        data["version"] = SETUP_STATE_VERSION
        data["updated_at"] = utc_now()
        self.paths.ensure()
        write_json(self.paths.setup_state, data)

    def _record(
        self,
        key: str,
        status: str,
        detail: str = "",
        **extra,
    ) -> None:
        data = self.state()
        entry = data["steps"].get(key) or {}
        entry.update(
            {
                "status": status,
                "title": STEP_TITLES.get(key, key),
                "detail": detail,
                "updated_at": utc_now(),
            }
        )
        if status in {"done", "failed", "skipped"}:
            entry["finished_at"] = utc_now()
        if extra:
            entry.update(extra)
        data["steps"][key] = entry
        self._save_state(data)

    def log(self, line: str) -> None:
        line = str(line).rstrip()
        if not line:
            return
        self.on_log(line)
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._log_path, "a", encoding="utf-8") as fh:
                fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {line}\n")
        except OSError:
            pass

    def cancel(self) -> None:
        self._cancel.set()
        process = self._process
        if process is not None and process.poll() is None:
            try:
                process.terminate()
            except OSError:
                pass

    # -- validation ------------------------------------------------------
    def step_done(self, key: str, settings: Settings) -> bool:
        state = self.state().get("steps", {}).get(key, {})
        if state.get("status") != "done":
            return False
        if key == "directories":
            return self.paths.root.is_dir() and self.paths.models.is_dir()
        if key == "inference":
            return inference_ready(self.paths.root)
        if key == "model":
            return state.get("model") == settings.model and self.models.is_complete(settings.model)
        if key == "training":
            return (self.paths.state / "training-runtime.json").is_file() and (
                self.paths.training_venv.is_dir()
            )
        if key == "service":
            return bool((read_json(self.paths.service_state, {}) or {}).get("installed"))
        if key == "login":
            return service_module.login_item_status(self.paths)
        return True

    def status(self) -> dict:
        settings = self.settings_store.load()
        steps = []
        for step in STEPS:
            entry = self.state().get("steps", {}).get(step.key, {})
            permitted = (
                True
                if step.permission is None
                else bool(getattr(settings.permissions, step.permission, False))
            )
            steps.append(
                {
                    "key": step.key,
                    "title": step.title,
                    "description": step.description,
                    "permission": step.permission,
                    "permitted": permitted,
                    "status": entry.get("status", "pending"),
                    "done": self.step_done(step.key, settings),
                    "detail": entry.get("detail", ""),
                    "updated_at": entry.get("updated_at"),
                }
            )
        required = ("directories", "inference", "model")
        ready = all(
            item["done"] for item in steps if item["key"] in required
        )
        return {
            "steps": steps,
            "ready": ready,
            "model": settings.model,
            "permissions": settings.permissions.to_dict(),
            "data_dir": str(self.paths.root),
            "log": str(self._log_path),
            "is_frozen": is_frozen(),
            "suggested_model": recommend_key(),
        }

    # -- run -------------------------------------------------------------
    def run(
        self,
        *,
        model: str | None = None,
        settings: Settings | None = None,
    ) -> dict:
        settings = settings or self.settings_store.load()
        if model:
            settings.model = model
        try:
            get_spec(settings.model)
        except KeyError:
            raise SetupError(f"Unknown model {settings.model!r}.") from None
        self.paths.ensure()
        self._cancel.clear()
        data = self.state()
        data.update(
            {
                "started_at": data.get("started_at") or utc_now(),
                "run_started_at": utc_now(),
                "model": settings.model,
                "permissions": settings.permissions.to_dict(),
                "version": __version__,
            }
        )
        self._save_state(data)

        total = len(STEPS)
        results: list[dict] = []
        overall_ok = True
        for index, step in enumerate(STEPS):
            base = index / total * 100

            def progress(percent: float, message: str, _base=base, _step=step) -> None:
                scaled = _base + max(0.0, min(100.0, percent)) / total
                self.on_progress(scaled, message or _step.title)

            if self._cancel.is_set():
                self._record(step.key, "skipped", "Cancelled by the user.")
                results.append({"key": step.key, "status": "skipped", "detail": "cancelled"})
                overall_ok = False
                continue

            if self.step_done(step.key, settings):
                self.log(f"[skip] {step.title} - already done")
                extra = {"model": settings.model} if step.key == "model" else {}
                self._record(step.key, "done", "Already completed - skipped.", **extra)
                results.append({"key": step.key, "status": "skipped-done", "detail": "already done"})
                progress(100, f"{step.title} already done")
                continue

            if step.permission and not getattr(settings.permissions, step.permission, False):
                self.log(f"[skip] {step.title} - permission '{step.permission}' is off")
                self._record(
                    step.key,
                    "skipped",
                    f"Permission '{step.permission}' is off.",
                )
                results.append(
                    {"key": step.key, "status": "skipped", "detail": "permission off"}
                )
                progress(100, f"{step.title} skipped (permission off)")
                continue

            self.log(f"[run ] {step.title}")
            self._record(step.key, "running", "")
            self.on_progress(base, step.title)
            try:
                detail = self._run_step(step.key, settings, progress)
            except StepFailed as exc:
                overall_ok = False
                message = str(exc)
                self.log(f"[fail] {step.title}: {message}")
                self._record(step.key, "failed", message, model=settings.model)
                results.append({"key": step.key, "status": "failed", "detail": message})
                continue
            except Exception as exc:  # never lose the rest of setup to one bug
                overall_ok = False
                message = f"{type(exc).__name__}: {exc}"
                self.log(f"[fail] {step.title}: {message}")
                self._record(step.key, "failed", message, model=settings.model)
                results.append({"key": step.key, "status": "failed", "detail": message})
                continue
            self.log(f"[ ok ] {step.title}: {detail}")
            extra = {"model": settings.model} if step.key == "model" else {}
            self._record(step.key, "done", detail, **extra)
            results.append({"key": step.key, "status": "done", "detail": detail})
            progress(100, f"{step.title} complete")

        data = self.state()
        data["last_run"] = {
            "finished_at": utc_now(),
            "ok": overall_ok,
            "results": results,
        }
        data["permissions"] = settings.permissions.to_dict()
        data["model"] = settings.model
        self._save_state(data)
        self.on_progress(100, "Setup finished" if overall_ok else "Setup finished with problems")
        return {
            "ok": overall_ok,
            "results": results,
            "model": settings.model,
            "permissions": settings.permissions.to_dict(),
            "data_dir": str(self.paths.root),
            "status": self.status(),
        }

    def _run_step(
        self,
        key: str,
        settings: Settings,
        progress: Callable[[float, str], None],
    ) -> str:
        if key == "directories":
            return self._directories()
        if key == "inference":
            return self._inference()
        if key == "model":
            return self._model(settings, progress)
        if key == "training":
            return self._training(progress)
        if key == "service":
            return self._service()
        if key == "login":
            return self._login()
        raise StepFailed(f"Unknown step {key}")

    # -- individual steps ------------------------------------------------
    def _directories(self) -> str:
        self.paths.ensure()
        self.paths.checkpoints.mkdir(parents=True, exist_ok=True)
        self.paths.runs.mkdir(parents=True, exist_ok=True)
        self.paths.datasets.mkdir(parents=True, exist_ok=True)
        copied = self._copy_training_source()
        detail = f"Data directory ready: {self.paths.root}"
        if copied:
            detail += f" (training source copied to {copied})"
        return detail

    def _copy_training_source(self) -> Path | None:
        if not is_frozen():
            return None
        source = bundle_dir() / "sideris_source"
        if not source.is_dir():
            return None
        target = self.paths.runtime_src
        package = target / "sideris_desktop"
        package.mkdir(parents=True, exist_ok=True)
        for item in source.rglob("*"):
            if not item.is_file():
                continue
            relative = item.relative_to(source)
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(item, destination)
            except OSError:
                pass
        return target

    def _inference(self) -> str:
        if inference_ready(self.paths.root):
            try:
                import llama_cpp

                version = getattr(llama_cpp, "__version__", "unknown")
                return f"llama.cpp runtime ready ({version})."
            except Exception:
                return "Inference runtime ready (transformers fallback)."

        self.log("llama.cpp not found; installing it locally.")
        package_dir = self.paths.runtime_pkgs
        package_dir.mkdir(parents=True, exist_ok=True)
        python = find_system_python()
        if not python:
            raise StepFailed(
                "No Python interpreter with pip was found to install the inference "
                "runtime. Reinstall using the official Sideris installer, which "
                "bundles llama.cpp."
            )
        wheel_url = _prebuilt_wheel_url()
        target = wheel_url or f"llama-cpp-python=={LLAMA_CPP_VERSION}"
        if wheel_url:
            wheels = self.paths.runtime / "wheels"
            wheels.mkdir(parents=True, exist_ok=True)
            wheel_path = wheels / wheel_url.rsplit("/", 1)[-1]
            if not wheel_path.is_file():
                self.log(f"Downloading {wheel_url.rsplit('/', 1)[-1]}")
                try:
                    download(
                        wheel_url,
                        wheel_path,
                        progress=lambda p: self.log(
                            f"  wheel {human_bytes(p.downloaded)} / {human_bytes(p.total)}"
                        ),
                    )
                except Exception as exc:
                    raise StepFailed(f"Could not download the runtime wheel: {exc}") from exc
            target = str(wheel_path)

        command = [
            python, "-m", "pip", "install", "--upgrade", "--target", str(package_dir),
            "--disable-pip-version-check", target,
        ]
        try:
            self._run_streaming(command, timeout=1800)
        except StepFailed:
            fallback = f"llama-cpp-python=={LLAMA_CPP_VERSION}"
            if target == fallback:
                raise
            # A prebuilt wheel can be corrupt upstream (the 0.3.36 macOS
            # arm64 asset is).  Fall back to a source build rather than
            # leaving the user with no runtime at all.
            self.log("Prebuilt wheel failed; trying a source build (this can take several minutes).")
            self._run_streaming(
                [
                    python, "-m", "pip", "install", "--upgrade", "--target",
                    str(package_dir), "--disable-pip-version-check", fallback,
                ],
                timeout=3600,
            )
        activate_runtime(self.paths.root)
        if not inference_ready(self.paths.root):
            raise StepFailed(
                "The runtime was installed but could not be imported. "
                f"See {self._log_path}."
            )
        return "Installed llama.cpp into the private runtime folder."

    def _model(self, settings: Settings, progress: Callable[[float, str], None]) -> str:
        spec = get_spec(settings.model)
        if self.models.is_complete(settings.model):
            return f"{spec.display_name} already installed."
        free = self.models.disk_free_gb()
        if free and free < spec.download_gb * 1.15 + 0.3:
            raise StepFailed(
                f"Not enough disk space: {spec.display_name} needs about "
                f"{spec.size_label}, {free} GB free."
            )
        self.log(f"Downloading {spec.display_name} ({spec.size_label}) from Hugging Face.")
        self.log("The download is resumable: if it stops, run setup again.")

        last = [0.0]

        def on_progress(item: Progress) -> None:
            percent = item.percent if item.percent is not None else 0.0
            if item.total:
                message = (
                    f"{spec.display_name}: {human_bytes(item.downloaded)} / "
                    f"{human_bytes(item.total)} ({percent:.0f}%)"
                )
            else:
                message = f"{spec.display_name}: {human_bytes(item.downloaded)}"
            progress(percent, message)
            now = time.time()
            if now - last[0] > 5:
                last[0] = now
                self.log("  " + message)

        try:
            result = self.models.download(
                settings.model, progress=on_progress, permission=True
            )
        except Exception as exc:
            raise StepFailed(str(exc)) from exc
        return (
            f"{spec.display_name} installed ({human_bytes(result.bytes_written)}"
            + (", resumed" if result.resumed else "")
            + ")."
        )

    def _training(self, progress: Callable[[float, str], None]) -> str:
        marker_path = self.paths.state / "training-runtime.json"
        venv = self.paths.training_venv
        python = self._venv_python(venv)
        if python.is_file() and marker_path.is_file():
            return "Training runtime already installed."

        system_python = find_system_python()
        if not system_python:
            raise StepFailed(
                "Fine-tuning needs a system Python 3.10-3.13 with venv support. "
                "Install Python from python.org, then run setup again."
            )
        if not python.is_file():
            self.log(f"Creating private training environment: {venv}")
            self._run_streaming(
                [system_python, "-m", "venv", str(venv)], timeout=600
            )
            python = self._venv_python(venv)
        if not python.is_file():
            raise StepFailed("Could not create the training virtual environment.")

        progress(5, "Installing pip")
        self._run_streaming(
            [str(python), "-m", "pip", "install", "--upgrade", "pip", "wheel",
             "--disable-pip-version-check"],
            timeout=900,
        )
        progress(15, "Installing PyTorch")
        torch_cmd = [
            str(python), "-m", "pip", "install", "--disable-pip-version-check",
            "torch", "--index-url", "https://download.pytorch.org/whl/cpu",
        ]
        try:
            self._run_streaming(torch_cmd, timeout=3600)
        except StepFailed as exc:
            self.log(f"CPU wheel index failed ({exc}); trying the default index.")
            self._run_streaming(
                [
                    str(python), "-m", "pip", "install",
                    "--disable-pip-version-check", "torch",
                ],
                timeout=3600,
            )
        progress(60, "Installing transformers, peft, datasets")
        self._run_streaming(
            [
                str(python), "-m", "pip", "install",
                "--disable-pip-version-check", *TRAINING_PACKAGES,
            ],
            timeout=3600,
        )
        progress(95, "Verifying")
        self._run_streaming(
            [
                str(python), "-c",
                "import torch, transformers, peft, datasets, accelerate; "
                "print('torch', torch.__version__, 'transformers', transformers.__version__)",
            ],
            timeout=300,
        )
        write_json(
            marker_path,
            {
                "ready": True,
                "python": str(python),
                "torch": _package_version(python, "torch"),
                "transformers": _package_version(python, "transformers"),
                "peft": _package_version(python, "peft"),
                "installed_at": utc_now(),
                "packages": TRAINING_PACKAGES,
            },
        )
        return "Training runtime installed into a private virtual environment."

    def _service(self) -> str:
        state = service_module.install_service(permission=True)
        return f"Background service installed ({state.get('mode')})."

    def _login(self) -> str:
        service_module.install_login_item(permission=True)
        return "Sideris will open at login."

    # -- process helpers -------------------------------------------------
    @staticmethod
    def _venv_python(venv: Path) -> Path:
        if os.name == "nt":
            return venv / "Scripts" / "python.exe"
        return venv / "bin" / "python"

    def _run_streaming(
        self,
        command: list[str],
        *,
        timeout: float = 1800,
        env: dict | None = None,
    ) -> None:
        self.log("$ " + " ".join(command))
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                text=True,
                bufsize=1,
                env=env or _pip_env(self.paths),
                cwd=str(self.paths.root),
            )
        except OSError as exc:
            raise StepFailed(f"Could not run {command[0]}: {exc}") from exc
        self._process = process
        started = time.time()
        tail: list[str] = []
        assert process.stdout is not None
        try:
            for line in process.stdout:
                line = line.rstrip()
                if line:
                    tail.append(line)
                    tail = tail[-40:]
                    self.log("  " + line)
                if self._cancel.is_set():
                    process.terminate()
                    raise StepFailed("Cancelled.")
                if time.time() - started > timeout:
                    process.terminate()
                    raise StepFailed("The installer step timed out.")
            code = process.wait(timeout=60)
        finally:
            self._process = None
        if code != 0:
            detail = " | ".join(tail[-4:]) or f"exit code {code}"
            raise StepFailed(f"Command failed ({code}): {detail}")

    def _training_python(self) -> Path:
        return self._venv_python(self.paths.training_venv)


def _package_version(python: Path, package: str) -> str | None:
    try:
        result = subprocess.run(
            [str(python), "-c", f"import {package}; print(getattr({package}, '__version__', '?'))"],
            capture_output=True,
            text=True,
            timeout=60,
            stdin=subprocess.DEVNULL,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None


def quick_status() -> dict:
    setup = Setup()
    status = setup.status()
    status["version"] = __version__
    status["app"] = __app_name__
    status["catalogue"] = [
        {
            "key": spec.key,
            "name": spec.display_name,
            "size": spec.size_label,
            "ram": spec.ram_label,
            "recommended_for": spec.recommended_for,
            "installed": setup.models.is_complete(spec.key),
            "url": spec.url,
        }
        for spec in catalogue()
    ]
    status["installed_models"] = [
        {"key": item.key, "name": item.name, "size": item.size_label}
        for item in setup.models.installed()
    ]
    status["default_model"] = DEFAULT_MODEL
    return status


__all__ = [
    "Setup",
    "SetupError",
    "StepFailed",
    "STEPS",
    "activate_runtime",
    "inference_ready",
    "find_system_python",
    "quick_status",
    "app_executable",
]

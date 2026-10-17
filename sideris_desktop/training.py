"""Supervision of real LoRA fine-tuning runs.

Training is deliberately *not* run inside the app process:

* the trainer may use gigabytes of memory and a lot of CPU/GPU;
* it must survive the window being closed;
* it must be stoppable without killing the UI.

So :class:`TrainingManager` owns a normal OS child process
(``python -m sideris_desktop.train_worker ...``), keeps durable state in
``state/training-state.json``, and the worker writes live progress and
keeps its own checkpoints.  If the machine loses power, or the user
stops a run, the next start finds the highest ``checkpoint-N`` directory
and adds ``--resume-from`` automatically.  Resuming is the default, not
an advanced option.

Training needs the optional runtime (torch, transformers, peft,
datasets, accelerate), which the setup wizard installs into a private
virtual environment only when the training permission is granted.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import __app_name__
from .config import (
    Paths,
    SettingsStore,
    paths,
    read_json,
    utc_now,
    write_json,
)
from .models import DEFAULT_MODEL, get_spec

DEFAULT_SAVE_STEPS = 50
KEEP_CHECKPOINTS = 3
CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")
DEFAULT_MAX_HOURS = 2.0
DEFAULT_EPOCHS = 1.0


class TrainingError(RuntimeError):
    pass


class TrainingPermissionError(TrainingError):
    """Training was requested while the permission is off."""


class TrainingNotReadyError(TrainingError):
    """The optional training runtime is not installed yet."""


def _pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    if os.name == "nt":
        import ctypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid)
        )
        if not handle:
            return False
        code = ctypes.c_ulong()
        try:
            ok = ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
            return bool(ok) and code.value == 259  # STILL_ACTIVE
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def latest_checkpoint(base: str | os.PathLike[str] | None = None) -> Path | None:
    """Return the highest-numbered ``checkpoint-N`` directory."""
    root = Path(base) if base else paths().checkpoints
    best: tuple[int, Path] | None = None
    if not root.is_dir():
        return None
    for candidate in root.rglob("checkpoint-*"):
        if not candidate.is_dir():
            continue
        match = CHECKPOINT_RE.match(candidate.name)
        if not match:
            continue
        step = int(match.group(1))
        if best is None or step > best[0]:
            best = (step, candidate)
    return best[1] if best else None


def prepare_dataset(dataset: str | os.PathLike[str]) -> Path:
    """Validate a .jsonl dataset and explain exactly what is wrong if not."""
    path = Path(dataset).expanduser()
    if not path.is_file():
        raise TrainingError(f"Dataset not found: {path}")
    if path.suffix.lower() != ".jsonl":
        raise TrainingError(
            "Dataset must be a .jsonl file with one JSON object per line: "
            '{"text": "..."} or {"prompt": "...", "response": "..."}.'
        )
    try:
        with open(path, "r", encoding="utf-8") as fh:
            first = fh.readline()
    except OSError as exc:
        raise TrainingError(f"Cannot read dataset: {exc}") from exc
    if not first.strip():
        raise TrainingError("Dataset is empty. Add at least one JSON line.")
    try:
        record = json.loads(first)
    except json.JSONDecodeError as exc:
        raise TrainingError(f"First line is not valid JSON: {exc}") from exc
    if not isinstance(record, dict) or not (
        isinstance(record.get("text"), str)
        or (isinstance(record.get("prompt"), str) and isinstance(record.get("response"), str))
    ):
        raise TrainingError(
            'Each line must contain "text", or both "prompt" and "response".'
        )
    return path.resolve()


def example_dataset() -> str:
    return (
        '{"prompt": "Summarise our refund policy in one sentence.", '
        '"response": "Refunds are available within 30 days of purchase with a receipt."}\n'
        '{"text": "The support team answers billing questions within one business day."}\n'
    )


@dataclass
class TrainingState:
    status: str = "idle"           # idle|starting|running|stopping|stopped|completed|failed|interrupted
    run_id: str | None = None
    pid: int | None = None
    dataset: str | None = None
    base_model: str = get_spec(DEFAULT_MODEL).train_repo
    max_hours: float = DEFAULT_MAX_HOURS
    epochs: float = DEFAULT_EPOCHS
    save_steps: int = DEFAULT_SAVE_STEPS
    max_steps: int | None = None
    step: int = 0
    epoch: float = 0.0
    loss: float | None = None
    started_at: str | None = None
    updated_at: str | None = None
    ended_at: str | None = None
    resumed_from: str | None = None
    checkpoint: str | None = None
    exit_code: int | None = None
    error: str | None = None
    log_file: str | None = None
    command: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data) -> "TrainingState":
        if not isinstance(data, dict):
            return cls()
        default = cls()
        payload = {}
        for name in default.to_dict():
            value = data.get(name, getattr(default, name))
            payload[name] = value
        try:
            payload["max_hours"] = float(payload.get("max_hours") or DEFAULT_MAX_HOURS)
        except (TypeError, ValueError):
            payload["max_hours"] = DEFAULT_MAX_HOURS
        try:
            payload["epochs"] = float(payload.get("epochs") or DEFAULT_EPOCHS)
        except (TypeError, ValueError):
            payload["epochs"] = DEFAULT_EPOCHS
        for name in ("step", "save_steps"):
            try:
                payload[name] = int(payload.get(name) or 0)
            except (TypeError, ValueError):
                payload[name] = 0
        if not isinstance(payload.get("command"), list):
            payload["command"] = []
        return cls(**{k: payload.get(k) for k in default.to_dict()})

    @property
    def active(self) -> bool:
        return self.status in {"starting", "running", "stopping"}


class TrainingManager:
    """Owns the trainer child process and its durable state."""

    def __init__(
        self,
        root: str | os.PathLike[str] | None = None,
        settings_store: SettingsStore | None = None,
    ) -> None:
        self.paths: Paths = paths(root)
        self.settings_store = settings_store or SettingsStore(self.paths.root)
        self._lock = threading.RLock()
        self._process: subprocess.Popen | None = None
        self._monitor: threading.Thread | None = None
        self._monitor_stop = threading.Event()
        self._last_progress_mtime = 0.0

    # -- state -----------------------------------------------------------
    def state(self) -> TrainingState:
        state = TrainingState.from_dict(read_json(self.paths.training_state, {}))
        if state.active and not _pid_alive(state.pid):
            # The app was closed, or the machine went down.  The run did not
            # finish; everything up to the last checkpoint is still valid.
            state.status = "interrupted"
            state.updated_at = utc_now()
            state.ended_at = state.ended_at or state.updated_at
            if not state.error:
                state.error = (
                    "The training process stopped unexpectedly. Start again to "
                    "resume from the latest checkpoint."
                )
            self._save(state)
        return state

    def _save(self, state: TrainingState) -> TrainingState:
        state.updated_at = utc_now()
        self.paths.ensure()
        write_json(self.paths.training_state, state.to_dict())
        return state

    def status(self) -> dict:
        state = self.state()
        worker = self._worker_progress(state)
        checkpoint = state.checkpoint or (
            str(latest_checkpoint(self.paths.checkpoints)) if latest_checkpoint(self.paths.checkpoints) else None
        )
        return {
            "state": state.to_dict(),
            "worker": worker,
            "checkpoint": checkpoint,
            "can_resume": bool(worker.get("resumable") or checkpoint),
            "latest_checkpoint": str(latest_checkpoint(self.paths.checkpoints))
            if latest_checkpoint(self.paths.checkpoints)
            else None,
            "permission": self.settings_store.load().permissions.training,
            "runtime_ready": self.runtime_ready(),
            "python": str(self.training_python() or ""),
            "checkpoints_dir": str(self.paths.checkpoints),
            "logs": str(self.paths.logs),
        }

    def _worker_progress(self, state: TrainingState) -> dict:
        if not state.run_id:
            return {}
        path = self._worker_state_path(state.run_id)
        data = read_json(path, {}) or {}
        if not isinstance(data, dict):
            data = {}
        data.setdefault("status", state.status)
        data["resumable"] = bool(
            data.get("status") in {"stopped", "interrupted", "failed"}
            or state.status in {"stopped", "interrupted", "failed"}
        )
        return data

    def _worker_state_path(self, run_id: str) -> Path:
        return self.paths.runs / run_id / "worker-state.json"

    def _control_path(self, run_id: str) -> Path:
        return self.paths.runs / run_id / "control.json"

    # -- runtime ---------------------------------------------------------
    def training_python(self) -> Path | None:
        venv = self.paths.training_venv
        if os.name == "nt":
            candidate = venv / "Scripts" / "python.exe"
        else:
            candidate = venv / "bin" / "python"
        if candidate.is_file():
            return candidate
        if not getattr(sys, "frozen", False):
            return Path(sys.executable)
        return None

    def runtime_ready(self) -> bool:
        python = self.training_python()
        if not python:
            return False
        marker = read_json(self.paths.state / "training-runtime.json", {}) or {}
        if marker.get("ready") is not True:
            return False
        return python.is_file()

    def _source_paths(self) -> list[str]:
        env_root = self.paths.root
        src = self.paths.runtime_src
        values: list[str] = []
        if src.is_dir():
            values.append(str(src))
        candidates = [
            Path(__file__).resolve().parents[1],
            env_root,
        ]
        for candidate in candidates:
            if (candidate / "sideris_desktop" / "__init__.py").is_file():
                values.append(str(candidate))
        # De-duplicate while keeping order.
        seen: set[str] = set()
        ordered = []
        for value in values:
            if value not in seen:
                seen.add(value)
                ordered.append(value)
        return ordered

    def environment(self) -> dict[str, str]:
        env = dict(os.environ)
        env.update(
            {
                "SIDERIS_HOME": str(self.paths.root),
                "PYTHONUNBUFFERED": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "HF_HOME": str(self.paths.runtime / "hf-cache"),
                "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
                "PYTHONPATH": os.pathsep.join(
                    self._source_paths() + [env.get("PYTHONPATH", "")]
                ).strip(os.pathsep),
            }
        )
        return env

    def build_command(
        self,
        dataset: str,
        *,
        hours: float = DEFAULT_MAX_HOURS,
        epochs: float = DEFAULT_EPOCHS,
        save_steps: int = DEFAULT_SAVE_STEPS,
        resume: bool = True,
        base_model: str | None = None,
        run_id: str | None = None,
    ) -> list[str]:
        python = self.training_python() or Path(sys.executable)
        settings = self.settings_store.load()
        try:
            spec = get_spec(settings.model)
        except KeyError:
            spec = get_spec(DEFAULT_MODEL)
        run_id = run_id or time.strftime("run-%Y%m%d-%H%M%S")
        checkpoint = latest_checkpoint(self.paths.checkpoints) if resume else None
        command = [
            str(python),
            "-m",
            "sideris_desktop.train_worker",
            "--dataset",
            str(dataset),
            "--base-model",
            base_model or spec.train_repo,
            "--max-hours",
            f"{max(float(hours), 0.01):g}",
            "--epochs",
            f"{max(float(epochs), 0.01):g}",
            "--save-steps",
            str(max(int(save_steps), 1)),
            "--keep",
            str(KEEP_CHECKPOINTS),
            "--checkpoints-dir",
            str(self.paths.checkpoints),
            "--run-id",
            run_id,
            "--state-file",
            str(self._worker_state_path(run_id)),
            "--control-file",
            str(self._control_path(run_id)),
        ]
        if checkpoint:
            command += ["--resume-from", str(checkpoint)]
        return command

    def resume_command(self) -> list[str]:
        """Command that would be used to continue the latest run."""
        state = self.state()
        dataset = state.dataset
        if not dataset:
            raise TrainingError("No previous training run found.")
        return self.build_command(
            dataset,
            hours=state.max_hours,
            epochs=state.epochs,
            save_steps=state.save_steps or DEFAULT_SAVE_STEPS,
            base_model=state.base_model,
            run_id=state.run_id or None,
        )

    # -- lifecycle -------------------------------------------------------
    def start(
        self,
        dataset: str,
        *,
        hours: float = DEFAULT_MAX_HOURS,
        epochs: float = DEFAULT_EPOCHS,
        save_steps: int = DEFAULT_SAVE_STEPS,
        permission: bool | None = None,
    ) -> TrainingState:
        with self._lock:
            settings = self.settings_store.load()
            allowed = settings.permissions.training if permission is None else permission
            if not allowed:
                raise TrainingPermissionError(
                    "Fine-tuning is disabled. Enable the training permission first."
                )
            if self.state().active and _pid_alive(self.state().pid):
                raise TrainingError("A training run is already in progress.")
            if not self.runtime_ready():
                raise TrainingNotReadyError(
                    "The training runtime is not installed yet. Run Setup with the "
                    "training permission enabled - it installs torch, transformers, "
                    "peft, datasets and accelerate into a private environment."
                )
            path = prepare_dataset(dataset)
            if not 0.01 <= float(hours) <= 168:
                raise TrainingError("Time budget must be between 0.01 and 168 hours.")
            if not 0.01 <= float(epochs) <= 100:
                raise TrainingError("Epochs must be between 0.01 and 100.")

            run_id = time.strftime("run-%Y%m%d-%H%M%S")
            run_dir = self.paths.runs / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            checkpoint = latest_checkpoint(self.paths.checkpoints)
            command = self.build_command(
                path,
                hours=hours,
                epochs=epochs,
                save_steps=save_steps,
                run_id=run_id,
            )
            log_file = self.paths.logs / f"training-{run_id}.log"
            state = TrainingState(
                status="starting",
                run_id=run_id,
                dataset=str(path),
                max_hours=float(hours),
                epochs=float(epochs),
                save_steps=int(save_steps),
                started_at=utc_now(),
                updated_at=utc_now(),
                resumed_from=str(checkpoint) if checkpoint else None,
                checkpoint=str(checkpoint) if checkpoint else None,
                log_file=str(log_file),
                command=command,
            )
            self._save(state)

            env = self.environment()
            log_handle = open(log_file, "ab", buffering=0)
            try:
                popen_kwargs: dict = {
                    "cwd": str(self.paths.root),
                    "env": env,
                    "stdin": subprocess.DEVNULL,
                    "stdout": log_handle,
                    "stderr": subprocess.STDOUT,
                }
                if os.name == "nt":
                    popen_kwargs["creationflags"] = (
                        subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
                        | subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
                    )
                else:
                    popen_kwargs["start_new_session"] = True
                process = subprocess.Popen(command, **popen_kwargs)
            except OSError as exc:
                state.status = "failed"
                state.error = f"Could not start the trainer: {exc}"
                state.ended_at = utc_now()
                self._save(state)
                raise TrainingError(state.error) from exc
            finally:
                log_handle.close()

            state.pid = process.pid
            state.status = "running"
            self._process = process
            self._save(state)
            write_json(
                self._control_path(run_id),
                {"stop": False, "requested_at": None},
            )
            self._start_monitor()
            return state

    def stop(self, *, wait: float = 0.0) -> TrainingState:
        with self._lock:
            state = self.state()
            if not state.active:
                raise TrainingError("No training run is active.")
            if state.run_id:
                write_json(
                    self._control_path(state.run_id),
                    {"stop": True, "requested_at": utc_now()},
                )
            pid = state.pid
            if pid and _pid_alive(pid):
                self._signal_stop(pid)
            state.status = "stopping"
            state.updated_at = utc_now()
            self._save(state)
            process = self._process
            if wait and process is not None:
                try:
                    process.wait(timeout=wait)
                except subprocess.TimeoutExpired:
                    pass
            return self.state()

    @staticmethod
    def _signal_stop(pid: int) -> None:
        try:
            if os.name == "nt":
                # CTRL_BREAK lets the worker flush a checkpoint; fall back to
                # terminate() if the process has no console.
                try:
                    os.kill(pid, signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
                except (AttributeError, OSError):
                    os.kill(pid, signal.SIGTERM)
            else:
                os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            pass

    def _start_monitor(self) -> None:
        if self._monitor and self._monitor.is_alive():
            return
        self._monitor_stop.clear()
        self._monitor = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor.start()

    def _monitor_loop(self) -> None:
        while not self._monitor_stop.is_set():
            time.sleep(1.5)
            try:
                state = TrainingState.from_dict(read_json(self.paths.training_state, {}))
                if not state.active:
                    break
                worker = read_json(self._worker_state_path(state.run_id or ""), {}) or {}
                if isinstance(worker, dict):
                    changed = False
                    for key in ("step", "max_steps", "epoch", "loss", "checkpoint"):
                        if key in worker and worker[key] not in (None, ""):
                            value = worker[key]
                            if key == "checkpoint":
                                value = str(value)
                            if getattr(state, key) != value:
                                setattr(state, key, value)
                                changed = True
                    if worker.get("status") == "stopping" and state.status != "stopping":
                        state.status = "stopping"
                        changed = True
                    if changed:
                        self._save(state)

                process = self._process
                exited = False
                exit_code = None
                if process is not None:
                    exit_code = process.poll()
                    exited = exit_code is not None
                if not exited and state.pid and not _pid_alive(state.pid):
                    exited = True
                    exit_code = state.exit_code

                if exited:
                    worker = read_json(self._worker_state_path(state.run_id or ""), {}) or {}
                    worker_status = str(worker.get("status") or "")
                    state.step = int(worker.get("step") or state.step or 0)
                    state.max_steps = worker.get("max_steps") or state.max_steps
                    state.epoch = worker.get("epoch") or state.epoch
                    state.loss = worker.get("loss", state.loss)
                    state.checkpoint = worker.get("checkpoint") or state.checkpoint
                    state.exit_code = exit_code
                    state.ended_at = utc_now()
                    latest = latest_checkpoint(self.paths.checkpoints)
                    if latest:
                        state.checkpoint = str(latest)
                    if worker_status == "completed":
                        state.status = "completed"
                    elif worker_status in {"stopped", "interrupted"} or state.status == "stopping":
                        state.status = "interrupted" if worker_status == "interrupted" else "stopped"
                        state.error = state.error or worker.get("error") or (
                            "Run stopped. Start again to resume from the latest checkpoint."
                        )
                    elif exit_code == 0:
                        state.status = "completed"
                    else:
                        state.status = "failed"
                        state.error = (
                            worker.get("error")
                            or f"The trainer exited with code {exit_code}. See {state.log_file}."
                        )
                    self._save(state)
                    self._process = None
                    break
            except Exception:
                # A supervisor must never die because a state file was
                # mid-write; the next tick retries.
                continue

    def wait(self, timeout: float | None = None) -> TrainingState:
        process = self._process
        if process is not None:
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                pass
        # Give the monitor a moment to persist the final state.
        deadline = time.time() + 3
        while time.time() < deadline:
            state = self.state()
            if not state.active:
                return state
            time.sleep(0.2)
        return self.state()

    def logs_tail(self, lines: int = 80) -> str:
        state = self.state()
        if not state.log_file:
            return ""
        path = Path(state.log_file)
        if not path.is_file():
            return ""
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                content = fh.readlines()
            return "".join(content[-lines:])
        except OSError:
            return ""


def default_training_dirs(paths_obj: Paths | None = None) -> None:
    obj = paths_obj or paths()
    obj.checkpoints.mkdir(parents=True, exist_ok=True)
    obj.runs.mkdir(parents=True, exist_ok=True)
    obj.logs.mkdir(parents=True, exist_ok=True)


__all__ = [
    "TrainingManager",
    "TrainingState",
    "TrainingError",
    "TrainingPermissionError",
    "TrainingNotReadyError",
    "latest_checkpoint",
    "prepare_dataset",
    "default_training_dirs",
    "DEFAULT_SAVE_STEPS",
    "KEEP_CHECKPOINTS",
    "example_dataset",
    "__app_name__",
    "shutil",
]

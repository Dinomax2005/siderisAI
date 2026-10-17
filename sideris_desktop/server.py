"""The local HTTP service.

Sideris exposes a small JSON API on ``127.0.0.1:8765`` so the desktop UI
(and scripts) can talk to the local model.  It binds to loopback only:
there is no cloud relay, no public listener, and no remote access.

Endpoints
---------
GET  /health                 liveness, version, model availability
GET  /v1/status              full local status (permissions, model, training)
GET  /v1/models              catalogue + installed weights
POST /v1/chat                {messages, level} -> complete answer
POST /v1/chat/stream         same, Server-Sent Events token stream
GET  /v1/training            training state / progress
POST /v1/training/start      {dataset, hours, epochs}
POST /v1/training/stop       graceful stop (checkpoint is kept)

Training endpoints answer ``403`` while the training permission is off.
Chat answers ``503`` with an honest error when no model is installed -
never a fabricated reply.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

from . import __app_name__, __release_name__, __version__
from .config import (
    DEFAULT_PORT,
    Paths,
    SettingsStore,
    paths,
    read_json,
    utc_now,
    write_json,
)
from .core import Assistant, Reply, backend_report
from .downloads import human_bytes
from .models import ModelManager
from .training import (
    TrainingError,
    TrainingManager,
    TrainingNotReadyError,
    TrainingPermissionError,
)

MAX_BODY_BYTES = 2 * 1024 * 1024
MAX_MESSAGES = 200
MAX_MESSAGE_CHARS = 200_000
SERVER_NAME = f"{__app_name__}/{__version__}"


class LocalRuntime:
    """Shared local state used by the HTTP service and the desktop app."""

    def __init__(self, root: str | os.PathLike[str] | None = None) -> None:
        self.paths: Paths = paths(root)
        self.paths.ensure()
        self.settings_store = SettingsStore(self.paths.root)
        self.models = ModelManager(self.paths.root)
        self.training = TrainingManager(self.paths.root, self.settings_store)
        self.started_at = time.time()
        self._assistant: Assistant | None = None
        self._signature: tuple | None = None
        self._lock = threading.Lock()

    def settings(self):
        return self.settings_store.load(refresh=True)

    def assistant(self) -> Assistant:
        settings = self.settings()
        selected = self.models.select(settings.model)
        signature = (
            settings.model,
            settings.context_size,
            settings.gpu_layers,
            str(selected[1]) if selected else None,
        )
        with self._lock:
            if self._assistant is None or self._signature != signature:
                self._assistant = Assistant(
                    settings, self.models, root=self.paths.root
                )
                self._signature = signature
            return self._assistant

    def model_selected(self):
        settings = self.settings()
        return self.models.select(settings.model)

    def refresh_models(self) -> None:
        """Called after setup installs weights so new models become visible."""
        with self._lock:
            self._assistant = None
            self._signature = None

    def health(self) -> dict:
        settings = self.settings()
        selected = self.models.select(settings.model)
        return {
            "status": "ok",
            "app": __app_name__,
            "release": __release_name__,
            "version": __version__,
            "uptime_seconds": round(time.time() - self.started_at, 1),
            "model_installed": bool(selected),
            "model": selected[0] if selected else None,
            "backend": backend_report()["preferred"],
            "data_dir": str(self.paths.root),
        }

    def status(self) -> dict:
        settings = self.settings()
        selected = self.models.select(settings.model)
        checkpoint = self.training.status().get("latest_checkpoint")
        service = read_json(self.paths.service_state, {}) or {}
        heartbeat = read_json(self.paths.heartbeat, {}) or {}
        return {
            "app": __app_name__,
            "release": __release_name__,
            "version": __version__,
            "status": "ok",
            "uptime_seconds": round(time.time() - self.started_at, 1),
            "settings": settings.to_dict(),
            "permissions": settings.permissions.to_dict(),
            "thinking_level": settings.thinking_level,
            "model": {
                "key": selected[0] if selected else None,
                "path": str(selected[1]) if selected else None,
                "installed": bool(selected),
                "size": human_bytes(selected[1].stat().st_size) if selected else None,
            },
            "motd": self.models.status(),
            "backend": backend_report(),
            "training": {
                "permission": settings.permissions.training,
                "checkpoint": checkpoint,
                "runtime_ready": self.training.runtime_ready(),
            },
            "service": {
                "installed": bool(service.get("installed")),
                "running": bool(heartbeat.get("running")),
                "last_heartbeat": heartbeat.get("updated_at"),
            },
            "paths": self.paths.describe(),
            "platform": sys.platform,
        }


def _json_bytes(payload: Any, status: int = 200) -> tuple[int, bytes]:
    return status, json.dumps(payload, ensure_ascii=False).encode("utf-8")


def _error(status: int, code: str, message: str) -> tuple[int, dict]:
    return status, {"error": {"code": code, "message": message}}


class SiderisServer(ThreadingHTTPServer):
    """Threaded loopback server.  One model, serialised generations."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], runtime: LocalRuntime) -> None:
        self.runtime = runtime
        self.log_path = runtime.paths.logs / "server.log"
        super().__init__(address, SiderisHandler)


class SiderisHandler(BaseHTTPRequestHandler):
    server_version = SERVER_NAME
    sys_version = ""
    protocol_version = "HTTP/1.1"

    # -- plumbing --------------------------------------------------------
    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
        try:
            line = "%s %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), fmt % args)
            log_path = getattr(self.server, "log_path", None)
            if log_path:
                if log_path.exists() and log_path.stat().st_size > 1_000_000:
                    log_path.replace(log_path.with_suffix(".log.old"))
                with open(log_path, "a", encoding="utf-8") as fh:
                    fh.write(line)
        except OSError:
            pass

    def _host_ok(self) -> bool:
        host = self.headers.get("Host", "")
        if not host:
            return False
        hostname = host.strip()
        if hostname.startswith("["):
            hostname = hostname[1 : hostname.find("]")]
        elif ":" in hostname:
            hostname = hostname.rsplit(":", 1)[0]
        return hostname.lower() in {"127.0.0.1", "localhost", "::1"}

    def _origin_ok(self) -> bool:
        origin = self.headers.get("Origin")
        if not origin:
            return True
        parsed = urlparse(origin)
        return (parsed.hostname or "").lower() in {"127.0.0.1", "localhost", "::1"}

    def _send_json(self, status: int, payload: Any) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _send_error_json(self, status: int, code: str, message: str) -> None:
        self._send_json(status, {"error": {"code": code, "message": message}})

    def _read_json(self) -> tuple[dict | None, str | None]:
        length_header = self.headers.get("Content-Length")
        if not length_header:
            return None, "Request body is required."
        try:
            length = int(length_header)
        except ValueError:
            return None, "Invalid Content-Length."
        if length > MAX_BODY_BYTES:
            return None, "Request body is too large."
        content_type = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if content_type and content_type != "application/json":
            return None, "Content-Type must be application/json."
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None, "Request body is not valid JSON."
        if not isinstance(payload, dict):
            return None, "Request body must be a JSON object."
        return payload, None

    def _guard(self) -> bool:
        if not self._host_ok():
            self._send_error_json(403, "forbidden_host", "This service only accepts loopback requests.")
            return False
        if not self._origin_ok():
            self._send_error_json(403, "forbidden_origin", "Cross-origin requests are not allowed.")
            return False
        return True

    # -- GET -------------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802
        if not self._guard():
            return
        route = urlparse(self.path).path.rstrip("/") or "/"
        runtime: LocalRuntime = self.server.runtime
        if route == "/health":
            self._send_json(200, runtime.health())
        elif route == "/":
            self._send_json(
                200,
                {
                    "app": __app_name__,
                    "version": __version__,
                    "local_only": True,
                    "endpoints": [
                        "GET /health",
                        "GET /v1/status",
                        "GET /v1/models",
                        "POST /v1/chat",
                        "POST /v1/chat/stream",
                        "GET /v1/training",
                        "POST /v1/training/start",
                        "POST /v1/training/stop",
                    ],
                },
            )
        elif route == "/v1/status":
            self._send_json(200, runtime.status())
        elif route == "/v1/models":
            self._send_json(200, runtime.models.status())
        elif route == "/v1/training":
            # Reading training state is itself a training endpoint: it can
            # expose dataset paths and run logs.  Keep the permission gate
            # consistent with the start/stop endpoints.
            if not runtime.settings().permissions.training:
                self._send_error_json(
                    403,
                    "permission_denied",
                    "Training permission is off. Enable it in Settings to inspect training.",
                )
            else:
                self._send_json(200, runtime.training.status())
        else:
            self._send_error_json(404, "not_found", f"No route {route}.")

    # -- POST ------------------------------------------------------------
    def do_POST(self) -> None:  # noqa: N802
        if not self._guard():
            return
        route = urlparse(self.path).path.rstrip("/")
        payload, error = self._read_json()
        if error:
            self._send_error_json(400, "bad_request", error)
            return
        assert payload is not None
        runtime: LocalRuntime = self.server.runtime
        try:
            if route == "/v1/chat":
                self._chat(payload, runtime, stream=False)
            elif route == "/v1/chat/stream":
                self._chat(payload, runtime, stream=True)
            elif route == "/v1/training/start":
                self._training_start(payload, runtime)
            elif route == "/v1/training/stop":
                self._training_stop(payload, runtime)
            else:
                self._send_error_json(404, "not_found", f"No route {route}.")
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:  # pragma: no cover - defensive
            self._send_error_json(500, "internal_error", f"{type(exc).__name__}: {exc}")

    # -- chat ------------------------------------------------------------
    def _validate_messages(self, payload: dict) -> tuple[list[dict] | None, str | None]:
        messages = payload.get("messages")
        if messages is None and isinstance(payload.get("prompt"), str):
            messages = [{"role": "user", "content": payload["prompt"]}]
        if not isinstance(messages, list) or not messages:
            return None, "messages must be a non-empty list of {role, content} objects."
        if len(messages) > MAX_MESSAGES:
            return None, f"At most {MAX_MESSAGES} messages are allowed."
        cleaned = []
        for message in messages:
            if not isinstance(message, dict):
                return None, "Each message must be an object."
            role = str(message.get("role") or "").lower()
            content = message.get("content")
            if role not in {"user", "assistant", "system"}:
                return None, "Message role must be user, assistant or system."
            if not isinstance(content, str) or not content.strip():
                return None, "Message content must be a non-empty string."
            if len(content) > MAX_MESSAGE_CHARS:
                return None, "Message content is too long."
            cleaned.append({"role": role, "content": content})
        if not any(m["role"] == "user" for m in cleaned):
            return None, "At least one user message is required."
        return cleaned, None

    def _check_ready(self, runtime: LocalRuntime) -> tuple[dict | None, int, str, str]:
        settings = runtime.settings()
        selected = runtime.models.select(settings.model)
        if not selected:
            return (
                None,
                503,
                "model_missing",
                "No local model is installed yet, so there is nothing to answer with. "
                "Open Setup, choose a model and finish installation. Sideris will not "
                "fake a reply.",
            )
        report = backend_report()
        if not report["available"]:
            return (
                None,
                503,
                "backend_missing",
                "The model is on disk but no inference runtime is installed. Run Setup "
                "to install llama-cpp-python.",
            )
        return {"settings": settings, "selected": selected}, 200, "", ""

    def _chat(self, payload: dict, runtime: LocalRuntime, *, stream: bool) -> None:
        messages, error = self._validate_messages(payload)
        if error:
            self._send_error_json(400, "bad_request", error)
            return
        assert messages is not None
        ready, status, code, message = self._check_ready(runtime)
        if not ready:
            self._send_error_json(status, code, message)
            return
        level = payload.get("level")
        if level is not None and str(level).lower() not in {"low", "medium", "high", "max"}:
            self._send_error_json(400, "bad_request", "level must be low, medium, high or max.")
            return
        assistant = runtime.assistant()

        if not stream:
            try:
                reply = assistant.chat(messages, level=level)
            except Exception as exc:
                self._send_error_json(500, "generation_error", f"{type(exc).__name__}: {exc}")
                return
            if reply.error:
                self._send_error_json(503, "generation_unavailable", reply.error)
                return
            body = reply.to_dict()
            body["reply"] = reply.text
            self._send_json(200, body)
            return

        # Server-Sent Events: status, tokens, then a terminal event.
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        self.close_connection = True
        lock = threading.Lock()

        def send_event(event: dict) -> None:
            data = json.dumps(event, ensure_ascii=False).encode("utf-8")
            with lock:
                try:
                    self.wfile.write(b"data: " + data + b"\n\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    self.close_connection = True

        try:
            reply: Reply = assistant.chat(
                messages,
                level=level,
                on_token=lambda piece: send_event({"type": "token", "text": piece}),
                on_status=lambda text: send_event({"type": "status", "text": text}),
            )
        except Exception as exc:
            send_event({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
            return
        if reply.error:
            send_event({"type": "error", "message": reply.error, "code": "generation_unavailable"})
            return
        body = reply.to_dict()
        body["reply"] = reply.text
        send_event({"type": "done", "reply": body})

    # -- training --------------------------------------------------------
    def _training_permitted(self, runtime: LocalRuntime) -> bool:
        return runtime.settings().permissions.training

    def _training_start(self, payload: dict, runtime: LocalRuntime) -> None:
        if not self._training_permitted(runtime):
            self._send_error_json(
                403,
                "permission_denied",
                "Training permission is off. Enable it in Settings before starting a run.",
            )
            return
        dataset = payload.get("dataset")
        if not isinstance(dataset, str) or not dataset.strip():
            self._send_error_json(400, "bad_request", "dataset must be a path to a .jsonl file.")
            return
        try:
            hours = float(payload.get("hours", 2.0))
            epochs = float(payload.get("epochs", 1.0))
        except (TypeError, ValueError):
            self._send_error_json(400, "bad_request", "hours and epochs must be numbers.")
            return
        try:
            state = runtime.training.start(dataset, hours=hours, epochs=epochs)
        except TrainingPermissionError as exc:
            self._send_error_json(403, "permission_denied", str(exc))
        except TrainingNotReadyError as exc:
            self._send_error_json(503, "training_runtime_missing", str(exc))
        except TrainingError as exc:
            self._send_error_json(400, "training_error", str(exc))
        else:
            self._send_json(202, {"state": state.to_dict(), "status": runtime.training.status()})

    def _training_stop(self, payload: dict, runtime: LocalRuntime) -> None:
        if not self._training_permitted(runtime):
            self._send_error_json(
                403,
                "permission_denied",
                "Training permission is off. Enable it in Settings before controlling a run.",
            )
            return
        try:
            state = runtime.training.stop()
        except TrainingError as exc:
            self._send_error_json(409, "not_running", str(exc))
        else:
            self._send_json(200, {"state": state.to_dict(), "status": runtime.training.status()})


# --------------------------------------------------------------------------
# process entry point
# --------------------------------------------------------------------------
def create_server(
    runtime: LocalRuntime | None = None,
    host: str = "127.0.0.1",
    port: int = 0,
) -> SiderisServer:
    if host not in {"127.0.0.1", "localhost"}:
        raise ValueError("Sideris only binds to loopback (127.0.0.1).")
    return SiderisServer((host, port), runtime or LocalRuntime())


def _write_server_info(runtime: LocalRuntime, host: str, port: int) -> None:
    write_json(
        runtime.paths.server_info,
        {
            "host": host,
            "port": port,
            "pid": os.getpid(),
            "started_at": utc_now(),
            "version": __version__,
            "url": f"http://{host}:{port}",
        },
    )


def _clear_server_info(runtime: LocalRuntime) -> None:
    try:
        info = read_json(runtime.paths.server_info, {}) or {}
        if info.get("pid") == os.getpid() and runtime.paths.server_info.exists():
            runtime.paths.server_info.unlink()
    except OSError:
        pass


def run_server(
    root: str | os.PathLike[str] | None = None,
    host: str = "127.0.0.1",
    port: int = DEFAULT_PORT,
    *,
    runtime: LocalRuntime | None = None,
    ready: threading.Event | None = None,
    announce: bool = True,
    on_signal: Callable[[], None] | None = None,
) -> int:
    runtime = runtime or LocalRuntime(root)
    try:
        server = create_server(runtime, host=host, port=port)
    except OSError as exc:
        raise SystemExit(
            f"Cannot listen on {host}:{port} - {exc}. "
            "Is another Sideris service already running?"
        ) from exc
    actual_port = server.server_address[1]
    _write_server_info(runtime, host, actual_port)
    if announce:
        print(f"{__app_name__} {__version__} local service: http://{host}:{actual_port}")
        print(f"Data directory: {runtime.paths.root}")

    stop = threading.Event()

    def _handle(signum, frame):  # noqa: ARG001
        stop.set()
        if on_signal is not None:
            try:
                on_signal()
            except Exception:
                pass
        threading.Thread(target=server.shutdown, daemon=True).start()

    if threading.current_thread() is threading.main_thread():
        for name in ("SIGTERM", "SIGINT", "SIGBREAK"):
            if hasattr(signal, name):
                try:
                    signal.signal(getattr(signal, name), _handle)
                except (ValueError, OSError):
                    pass

    if ready is not None:
        ready.set()
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        server.server_close()
        _clear_server_info(runtime)
    return 0


def probe_server(port: int = DEFAULT_PORT, timeout: float = 0.6) -> dict | None:
    """Return /health from an already-running local service, if any."""
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/health", timeout=timeout
        ) as response:
            payload = json.loads(response.read().decode("utf-8"))
            if isinstance(payload, dict) and payload.get("app") == __app_name__:
                return payload
    except (urllib.error.URLError, OSError, json.JSONDecodeError, ValueError):
        return None
    return None


def pick_free_port(preferred: int = DEFAULT_PORT) -> int:
    """Preferred port if free, otherwise an ephemeral one."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", preferred))
            return preferred
    except OSError:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])


__all__ = [
    "LocalRuntime",
    "SiderisServer",
    "create_server",
    "run_server",
    "probe_server",
    "pick_free_port",
    "HTTPStatus",
    "Path",
]

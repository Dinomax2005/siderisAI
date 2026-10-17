"""Tests for Sideris Desktop.

Run with:  python -m unittest discover -s tests -v

Everything here uses the standard library.  The suite exercises the
behaviours the product actually promises:

* downloads resume with an HTTP Range request and never restart from zero;
* settings and permissions round-trip, with permissions defaulting off;
* the highest checkpoint is selected and ``--resume-from`` is automatic;
* training state survives a JSON round-trip;
* the local service reports health and refuses to invent an answer when
  no model is installed (HTTP 503);
* training endpoints are forbidden (403) without permission;
* the local tools, memory and workspace behave safely.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import re
import shutil
import struct
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sideris_desktop import __version__  # noqa: E402
from sideris_desktop.cli import main as cli_main  # noqa: E402
from sideris_desktop.config import (  # noqa: E402
    Permissions,
    Settings,
    SettingsStore,
    default_data_dir,
    paths,
    read_json,
    write_json,
)
from sideris_desktop.core import (  # noqa: E402
    MemoryStore,
    Workspace,
    ToolError,
    safe_calculate,
    score_candidate,
    try_arithmetic,
)
from sideris_desktop.downloads import (  # noqa: E402
    DownloadError,
    download,
    part_path,
    sha256_file,
    state_path,
)
from sideris_desktop.models import (  # noqa: E402
    CATALOGUE,
    DEFAULT_MODEL,
    ModelManager,
    get_spec,
    recommend_key,
)
from sideris_desktop.server import LocalRuntime, create_server  # noqa: E402
from sideris_desktop.training import (  # noqa: E402
    TrainingManager,
    TrainingState,
    latest_checkpoint,
)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
class RangeHTTPHandler(BaseHTTPRequestHandler):
    """Serves a fixed payload, optionally honouring Range requests."""

    payload = b""
    support_range = True
    requests: list[dict] = []

    def log_message(self, *args):  # silence the test output
        pass

    def do_GET(self):  # noqa: N802
        RangeHTTPHandler.requests.append(
            {key.lower(): value for key, value in self.headers.items()}
        )
        range_header = self.headers.get("Range")
        if range_header and RangeHTTPHandler.support_range:
            match = re.match(r"bytes=(\d+)-", range_header)
            start = int(match.group(1)) if match else 0
            body = RangeHTTPHandler.payload[start:]
            self.send_response(206)
            self.send_header(
                "Content-Range",
                f"bytes {start}-{len(RangeHTTPHandler.payload) - 1}/{len(RangeHTTPHandler.payload)}",
            )
            self.send_header("Content-Length", str(len(body)))
            self.send_header("ETag", '"test-v1"')
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(RangeHTTPHandler.payload)))
        self.send_header("ETag", '"test-v1"')
        self.end_headers()
        self.wfile.write(RangeHTTPHandler.payload)


class HTTPFixture:
    def __init__(self, payload: bytes, support_range: bool = True):
        RangeHTTPHandler.payload = payload
        RangeHTTPHandler.support_range = support_range
        RangeHTTPHandler.requests = []
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), RangeHTTPHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        host, port = self.server.server_address
        self.url = f"http://{host}:{port}/model.gguf"
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()


class EnvIsolation(unittest.TestCase):
    """Give every test its own SIDERIS_HOME and no proxy interference."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="sideris-test-"))
        self.env = mock.patch.dict(
            os.environ,
            {
                "SIDERIS_HOME": str(self.tmp),
                "NO_PROXY": "127.0.0.1,localhost",
                "no_proxy": "127.0.0.1,localhost",
            },
        )
        self.env.start()

    def tearDown(self):
        self.env.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)


# --------------------------------------------------------------------------
# version / config
# --------------------------------------------------------------------------
class VersionAndConfigTests(EnvIsolation):
    def test_version(self):
        self.assertEqual(__version__, "5.0.0")

    def test_settings_round_trip_permissions_default_off(self):
        store = SettingsStore(self.tmp)
        settings = store.load()
        self.assertEqual(settings.model, DEFAULT_MODEL)
        self.assertEqual(settings.thinking_level, "medium")
        self.assertEqual(
            settings.permissions.to_dict(),
            {
                "model_download": False,
                "background_service": False,
                "training": False,
                "start_at_login": False,
            },
        )
        settings.model = "qwen2.5-3b"
        settings.thinking_level = "high"
        settings.port = 9123
        store.save(settings)

        reloaded = SettingsStore(self.tmp).load()
        self.assertEqual(reloaded.model, "qwen2.5-3b")
        self.assertEqual(reloaded.thinking_level, "high")
        self.assertEqual(reloaded.port, 9123)
        self.assertFalse(reloaded.permissions.model_download)
        self.assertFalse(reloaded.permissions.background_service)
        self.assertFalse(reloaded.permissions.training)
        self.assertFalse(reloaded.permissions.start_at_login)

    def test_permissions_persist_and_unknown_values_default_off(self):
        write_json(
            paths(self.tmp).settings,
            {
                "model": "qwen2.5-1.5b",
                "permissions": {"model_download": True, "training": "yes", "nope": True},
            },
        )
        loaded = SettingsStore(self.tmp).load()
        self.assertTrue(loaded.permissions.model_download)
        self.assertTrue(loaded.permissions.training)
        self.assertFalse(loaded.permissions.background_service)
        self.assertFalse(loaded.permissions.start_at_login)

    def test_permissions_dataclass_defaults(self):
        self.assertEqual(
            Permissions().enabled(),
            [],
        )

    def test_env_override_and_legacy_name(self):
        with mock.patch.dict(os.environ, {"SIDERIS_HOME": str(self.tmp)}):
            self.assertEqual(default_data_dir(), self.tmp)
        env = dict(os.environ)
        env.pop("SIDERIS_HOME", None)
        env["LUNO_HOME"] = str(self.tmp / "legacy")
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(default_data_dir(), self.tmp / "legacy")

    def test_corrupt_settings_are_ignored_not_fatal(self):
        settings_file = paths(self.tmp).settings
        settings_file.parent.mkdir(parents=True, exist_ok=True)
        settings_file.write_text("{not json", encoding="utf-8")
        loaded = SettingsStore(self.tmp).load()
        self.assertEqual(loaded.model, DEFAULT_MODEL)
        self.assertTrue(settings_file.with_suffix(".json.corrupt").exists())


# --------------------------------------------------------------------------
# catalogue
# --------------------------------------------------------------------------
class CatalogueTests(unittest.TestCase):
    def test_required_models_and_urls(self):
        for key, size in (("qwen2.5-0.5b", "0.5"), ("qwen2.5-1.5b", "1.5"), ("qwen2.5-3b", "3")):
            spec = get_spec(key)
            self.assertEqual(
                spec.url,
                "https://huggingface.co/Qwen/Qwen2.5-"
                f"{size}B-Instruct-GGUF/resolve/main/"
                f"qwen2.5-{size}b-instruct-q4_k_m.gguf",
            )
            self.assertEqual(spec.train_repo, f"Qwen/Qwen2.5-{size}B-Instruct")
        self.assertEqual(DEFAULT_MODEL, "qwen2.5-1.5b")
        self.assertLessEqual(get_spec("qwen2.5-1.5b").min_ram_gb, 8)

    def test_recommendation(self):
        self.assertEqual(recommend_key(4), "qwen2.5-0.5b")
        self.assertEqual(recommend_key(8), "qwen2.5-1.5b")
        self.assertEqual(recommend_key(12), "qwen2.5-3b")
        self.assertEqual(recommend_key(0), DEFAULT_MODEL)

    def test_manager_has_no_models_on_empty_home(self):
        tmp = Path(tempfile.mkdtemp(prefix="sideris-models-"))
        try:
            manager = ModelManager(tmp)
            self.assertEqual(manager.installed(), [])
            self.assertIsNone(manager.select())
            self.assertFalse(manager.exists(DEFAULT_MODEL))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------
# resumable downloads
# --------------------------------------------------------------------------
class DownloadTests(EnvIsolation):
    def test_resume_sends_range_and_keeps_bytes(self):
        payload = bytes((index * 7) % 256 for index in range(260_000))
        with HTTPFixture(payload, support_range=True) as fixture:
            dest = self.tmp / "qwen.gguf"
            seed = 91_337
            part_path(dest).parent.mkdir(parents=True, exist_ok=True)
            part_path(dest).write_bytes(payload[:seed])
            write_json(
                state_path(dest),
                {
                    "url": fixture.url,
                    "etag": '"test-v1"',
                    "total_bytes": len(payload),
                    "downloaded_bytes": seed,
                    "status": "in-progress",
                },
            )
            result = download(fixture.url, dest)

        self.assertTrue(result.resumed)
        self.assertEqual(result.bytes_written, len(payload))
        self.assertEqual(dest.read_bytes(), payload)
        self.assertFalse(part_path(dest).exists())
        self.assertEqual(read_json(state_path(dest))["status"], "complete")

        first = RangeHTTPHandler.requests[0]
        self.assertEqual(first.get("range"), f"bytes={seed}-")
        self.assertIn("identity", first.get("accept-encoding", ""))

    def test_server_ignoring_range_restarts_cleanly(self):
        payload = b"S" * 50_000 + b"fresh" * 10_000
        with HTTPFixture(payload, support_range=False) as fixture:
            dest = self.tmp / "restart.gguf"
            part_path(dest).write_bytes(payload[:20_000])
            write_json(
                state_path(dest),
                {"url": fixture.url, "total_bytes": len(payload), "downloaded_bytes": 20_000},
            )
            result = download(fixture.url, dest)

        self.assertFalse(result.resumed)
        self.assertEqual(dest.read_bytes(), payload)
        self.assertEqual(RangeHTTPHandler.requests[0].get("range"), "bytes=20000-")

    def test_checksum_mismatch_discards_file(self):
        payload = b"x" * 10_000
        with HTTPFixture(payload) as fixture:
            dest = self.tmp / "bad.gguf"
            with self.assertRaises(DownloadError):
                download(fixture.url, dest, expected_sha256="0" * 64)
        self.assertFalse(dest.exists())
        self.assertFalse(part_path(dest).exists())

    def test_checksum_success(self):
        payload = b"deterministic payload" * 5000
        expected = __import__("hashlib").sha256(payload).hexdigest()
        with HTTPFixture(payload) as fixture:
            dest = self.tmp / "good.gguf"
            result = download(fixture.url, dest, expected_sha256=expected)
        self.assertEqual(result.sha256, expected)
        self.assertEqual(dest.read_bytes(), payload)

    def test_state_file_names(self):
        dest = Path("/tmp/model.gguf")
        self.assertEqual(part_path(dest).name, "model.gguf.part")
        self.assertEqual(state_path(dest).name, "model.gguf.part.state.json")


# --------------------------------------------------------------------------
# training checkpoints and state
# --------------------------------------------------------------------------
class TrainingStateTests(EnvIsolation):
    def test_latest_checkpoint_is_highest_number(self):
        base = self.tmp / "checkpoints"
        for name in ("checkpoint-10", "checkpoint-150", "checkpoint-90", "not-a-checkpoint"):
            (base / name).mkdir(parents=True)
        (base / "checkpoint-999").write_text("a file, not a directory")
        selected = latest_checkpoint(base)
        self.assertIsNotNone(selected)
        self.assertEqual(selected.name, "checkpoint-150")

    def test_resume_command_contains_resume_from_checkpoint_150(self):
        data_paths = paths(self.tmp)
        data_paths.ensure()
        checkpoint = data_paths.checkpoints / "checkpoint-150"
        checkpoint.mkdir(parents=True)
        dataset = self.tmp / "data.jsonl"
        dataset.write_text('{"text": "hello"}\n', encoding="utf-8")
        manager = TrainingManager(self.tmp)
        state = TrainingState(
            status="interrupted",
            run_id="run-test",
            dataset=str(dataset),
            base_model="Qwen/Qwen2.5-1.5B-Instruct",
            max_hours=2.5,
            epochs=2.0,
        )
        write_json(data_paths.training_state, state.to_dict())

        command = manager.resume_command()
        self.assertIn("--resume-from", command)
        resume_value = command[command.index("--resume-from") + 1]
        self.assertTrue(resume_value.endswith("checkpoint-150"))
        self.assertIn("-m", command)
        self.assertIn("sideris_desktop.train_worker", command)
        self.assertIn("--save-steps", command)
        self.assertEqual(command[command.index("--save-steps") + 1], "50")
        self.assertIn("--max-hours", command)
        self.assertEqual(command[command.index("--max-hours") + 1], "2.5")

    def test_training_state_round_trip(self):
        state = TrainingState(
            status="running",
            run_id="run-1",
            pid=1234,
            dataset="/tmp/d.jsonl",
            base_model="Qwen/Qwen2.5-3B-Instruct",
            max_hours=1.5,
            epochs=3.0,
            save_steps=50,
            max_steps=300,
            step=125,
            epoch=1.25,
            loss=0.8421,
            checkpoint="/tmp/checkpoint-100",
            resumed_from="/tmp/checkpoint-50",
            command=["python", "-m", "sideris_desktop.train_worker"],
        )
        restored = TrainingState.from_dict(state.to_dict())
        self.assertEqual(restored.to_dict(), state.to_dict())
        self.assertTrue(restored.active)

    def test_dataset_validation(self):
        from sideris_desktop.training import TrainingError, prepare_dataset

        good = self.tmp / "good.jsonl"
        good.write_text('{"prompt": "hi", "response": "hello"}\n', encoding="utf-8")
        self.assertEqual(prepare_dataset(good), good.resolve())

        bad = self.tmp / "bad.jsonl"
        bad.write_text("not json\n", encoding="utf-8")
        with self.assertRaises(TrainingError):
            prepare_dataset(bad)
        with self.assertRaises(TrainingError):
            prepare_dataset(self.tmp / "missing.jsonl")

    def test_training_permission_off_blocks_start(self):
        from sideris_desktop.training import TrainingPermissionError

        manager = TrainingManager(self.tmp)
        with self.assertRaises(TrainingPermissionError):
            manager.start(self.tmp / "whatever.jsonl")


# --------------------------------------------------------------------------
# local HTTP service
# --------------------------------------------------------------------------
class ServerTests(EnvIsolation):
    def setUp(self):
        super().setUp()
        self.runtime = LocalRuntime(self.tmp)
        self.server = create_server(self.runtime, port=0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        host, port = self.server.server_address
        self.base = f"http://{host}:{port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        super().tearDown()

    def _get(self, path: str):
        try:
            with urllib.request.urlopen(self.base + path, timeout=10) as response:
                return response.status, json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read().decode("utf-8"))

    def _post(self, path: str, payload: dict, headers: dict | None = None):
        request = urllib.request.Request(
            self.base + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", **(headers or {})},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                return response.status, json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read().decode("utf-8"))

    def test_health_ok(self):
        status, payload = self._get("/health")
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["version"], "5.0.0")
        self.assertFalse(payload["model_installed"])

    def test_status_lists_permissions_all_off(self):
        status, payload = self._get("/v1/status")
        self.assertEqual(status, 200)
        self.assertEqual(
            payload["permissions"],
            {
                "model_download": False,
                "background_service": False,
                "training": False,
                "start_at_login": False,
            },
        )

    def test_chat_without_model_returns_503(self):
        status, payload = self._post(
            "/v1/chat", {"messages": [{"role": "user", "content": "hello"}], "level": "medium"}
        )
        self.assertEqual(status, 503)
        self.assertEqual(payload["error"]["code"], "model_missing")
        self.assertIn("will not fake a reply", payload["error"]["message"])

    def test_stream_without_model_returns_503_json(self):
        status, payload = self._post(
            "/v1/chat/stream", {"messages": [{"role": "user", "content": "hello"}]}
        )
        self.assertEqual(status, 503)
        self.assertEqual(payload["error"]["code"], "model_missing")

    def test_training_endpoints_forbidden_without_permission(self):
        status, payload = self._post(
            "/v1/training/start",
            {"dataset": str(self.tmp / "data.jsonl"), "hours": 1, "epochs": 1},
        )
        self.assertEqual(status, 403)
        self.assertEqual(payload["error"]["code"], "permission_denied")

        status, payload = self._post("/v1/training/stop", {})
        self.assertEqual(status, 403)

        status, payload = self._get("/v1/training")
        self.assertEqual(status, 403)
        self.assertEqual(payload["error"]["code"], "permission_denied")

    def test_training_start_with_permission_but_no_runtime_is_503(self):
        store = SettingsStore(self.tmp)
        settings = store.load()
        settings.permissions.training = True
        store.save(settings)
        dataset = self.tmp / "data.jsonl"
        dataset.write_text('{"text": "hello"}\n', encoding="utf-8")
        status, payload = self._post(
            "/v1/training/start", {"dataset": str(dataset), "hours": 1, "epochs": 1}
        )
        self.assertEqual(status, 503)
        self.assertEqual(payload["error"]["code"], "training_runtime_missing")

    def test_bad_json_and_host_guard(self):
        request = urllib.request.Request(
            self.base + "/v1/chat",
            data=b"not json",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(request, timeout=10)
        self.assertEqual(caught.exception.code, 400)

        request = urllib.request.Request(
            self.base + "/health", headers={"Host": "evil.example"}
        )
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(request, timeout=10)
        self.assertEqual(caught.exception.code, 403)

    def test_streaming_chat_uses_local_assistant(self):
        from sideris_desktop.core import Reply

        class FakeAssistant:
            def chat(self, messages, level=None, on_token=None, on_status=None):
                if on_status:
                    on_status("Thinking")
                if on_token:
                    on_token("Hello ")
                    on_token("from Sideris")
                return Reply(
                    text="Hello from Sideris",
                    level=level or "medium",
                    model="fake",
                    backend="fake",
                    elapsed=0.01,
                )

        fake = FakeAssistant()
        model_file = self.tmp / "model.gguf"
        model_file.write_bytes(b"\0" * 4096)
        with mock.patch.object(self.runtime, "models") as models_mock, \
                mock.patch.object(self.runtime, "assistant", return_value=fake), \
                mock.patch(
                    "sideris_desktop.server.backend_report",
                    return_value={"available": True, "preferred": "fake", "llama_cpp": False, "transformers": True},
                ):
            models_mock.select.return_value = ("qwen2.5-1.5b", model_file)
            status, _ = self._post(
                "/v1/chat",
                {"messages": [{"role": "user", "content": "hi"}], "level": "high"},
            )
            self.assertEqual(status, 200)

            request = urllib.request.Request(
                self.base + "/v1/chat/stream",
                data=json.dumps(
                    {"messages": [{"role": "user", "content": "hi"}], "level": "high"}
                ).encode("utf-8"),
                headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=20) as response:
                self.assertEqual(response.status, 200)
                body = response.read().decode("utf-8")
        self.assertIn('"type": "status"', body)
        self.assertIn('"type": "token"', body)
        self.assertIn('"type": "done"', body)
        self.assertIn("Hello from Sideris", body)


# --------------------------------------------------------------------------
# tools, memory, workspace
# --------------------------------------------------------------------------
class ToolTests(unittest.TestCase):
    def test_calculator(self):
        expression, value = try_arithmetic("what is 12*8+1?")
        self.assertEqual(expression, "12*8+1")
        self.assertEqual(value, 97.0)
        self.assertIsNone(try_arithmetic("tell me a story"))

    def test_safe_calculate_blocks_code(self):
        with self.assertRaises(ToolError):
            safe_calculate("__import__('os').system('echo pwned')")
        with self.assertRaises(ToolError):
            safe_calculate("open('/etc/passwd').read()")
        self.assertEqual(safe_calculate("2**10"), 1024.0)
        self.assertAlmostEqual(safe_calculate("sqrt(81)"), 9.0)

    def test_scoring_prefers_substance_and_penalises_unclosed_fences(self):
        refusal = "I'm sorry, I cannot help with that."
        answer = (
            "A transformer processes the whole sequence at once. Each layer "
            "mixes information across tokens with attention, then refines it "
            "with a small feed-forward network. Stacking those layers lets the "
            "model build increasingly abstract representations, and the output "
            "head predicts the next token.\n\n"
            "- Attention decides which tokens matter to each other.\n"
            "- Positional encoding preserves order.\n"
            "- Residual connections keep training stable."
        )
        broken = answer + "\n\n```python\nprint('never closed')\n"
        self.assertGreater(score_candidate(answer)["score"], score_candidate(refusal)["score"])
        self.assertLess(
            score_candidate(broken)["score"],
            score_candidate(answer)["score"],
        )

    def test_memory_capture_and_relevance(self):
        tmp = Path(tempfile.mkdtemp(prefix="sideris-memory-"))
        try:
            memory = MemoryStore(tmp / "memory.jsonl")
            added = memory.capture("My name is Alex and I live in Vienna.")
            self.assertTrue(added)
            facts = memory.relevant("What is my name?")
            self.assertTrue(any("Alex" in fact for fact in facts))
            self.assertFalse(memory.capture("My name is Alex and I live in Vienna."))
            explicit = memory.capture("Remember that I prefer concise answers.")
            self.assertTrue(explicit)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_workspace_reads_and_refuses_traversal(self):
        tmp = Path(tempfile.mkdtemp(prefix="sideris-workspace-"))
        try:
            (tmp / "notes.md").write_text("The launch date is Friday.\n", encoding="utf-8")
            workspace = Workspace(tmp)
            self.assertIn("notes.md", workspace.list_names())
            self.assertIn("launch date", workspace.read("notes.md"))
            hits = workspace.search("launch date")
            self.assertTrue(hits)
            self.assertEqual(hits[0][0], "notes.md")
            with self.assertRaises(ToolError):
                workspace.read("../secret.txt")
            self.assertIsNone(workspace.find_by_name("../../etc/passwd"))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
class CliTests(EnvIsolation):
    def test_status_json(self):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = cli_main(["status", "--json"])
        self.assertEqual(code, 0)
        payload = json.loads(buffer.getvalue())
        self.assertEqual(payload["version"], "5.0.0")
        self.assertEqual(payload["data_dir"], str(self.tmp))
        self.assertFalse(any(payload["permissions"].values()))
        self.assertFalse(payload["model"]["ready"])

    def test_status_human_is_honest(self):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            code = cli_main(["status"])
        self.assertEqual(code, 0)
        output = buffer.getvalue()
        self.assertIn("Sideris Desktop 5.0.0", output)
        self.assertIn("not installed", output.lower())

    def test_models_list(self):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = cli_main(["models", "list"])
        self.assertEqual(code, 0)
        self.assertIn("qwen2.5-1.5b", buffer.getvalue())


# --------------------------------------------------------------------------
# packaged entry point
# --------------------------------------------------------------------------
class LauncherTests(EnvIsolation):
    """The frozen build runs a script as ``__main__``.

    PyInstaller executes its entry script without a package context.  These
    tests run ``sideris_launcher.py`` exactly that way, so a regression to
    "point PyInstaller at sideris_desktop/__main__.py" fails here instead of
    silently producing installers that crash on launch.
    """

    def _run(self, *args: str, timeout: float = 60.0):
        import subprocess

        return subprocess.run(
            [sys.executable, str(ROOT / "sideris_launcher.py"), *args],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "SIDERIS_HOME": str(self.tmp)},
        )

    def test_launcher_reports_version_as_a_bare_script(self):
        result = self._run("--version")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "Sideris 5.0.0")

    def test_launcher_status_works_as_a_bare_script(self):
        result = self._run("status", "--json")
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["version"], "5.0.0")
        self.assertEqual(payload["data_dir"], str(self.tmp))

    def test_selftest_report(self):
        import subprocess

        target = self.tmp / "reports" / "selftest.json"
        result = self._run("--selftest", str(target))
        self.assertTrue(target.is_file(), result.stderr)
        report = read_json(target)
        self.assertEqual(report["checks"]["version"], "Sideris 5.0.0")
        self.assertIn("assets", report["checks"])
        self.assertEqual(report["frozen"], False)
        # Some Linux Pythons ship a tkinter package that cannot load Tcl, so
        # probe by really importing it rather than by looking for the folder.
        tk = subprocess.run(
            [sys.executable, "-c", "import tkinter"], capture_output=True, text=True
        )
        if tk.returncode == 0:
            self.assertTrue(report["ok"], report)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(report["checks"]["app"], "imported")
        else:
            self.assertIsNone(report["checks"]["tkinter"])
            self.assertEqual(result.returncode, 1)

    def test_print_does_not_crash_without_stdout(self):
        from sideris_desktop.cli import _print

        with mock.patch.object(sys, "stdout", None):
            _print("nobody is listening")  # must not raise



# --------------------------------------------------------------------------
# product invariants
# --------------------------------------------------------------------------
class ProductInvariantTests(unittest.TestCase):
    def test_logo_is_untouched_and_icons_cover_required_sizes(self):
        assets = ROOT / "assets"
        logo = (assets / "logo.png").read_bytes()
        self.assertEqual(logo[:8], b"\x89PNG\r\n\x1a\n")
        self.assertGreater(len(logo), 1000)

        ico = (assets / "sideris.ico").read_bytes()
        reserved, kind, count = struct.unpack("<HHH", ico[:6])
        self.assertEqual((reserved, kind), (0, 1))
        sizes = set()
        for index in range(count):
            entry = 6 + index * 16
            width = ico[entry] or 256
            height = ico[entry + 1] or 256
            sizes.add((width, height))
        self.assertTrue(
            {(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)}
            <= sizes
        )

        icns = (assets / "sideris.icns").read_bytes()
        self.assertEqual(icns[:4], b"icns")
        self.assertEqual(struct.unpack(">I", icns[4:8])[0], len(icns))

    def test_transformers_fallback_is_local_only_by_default(self):
        from sideris_desktop.inference import Engine, TransformersBackend

        backend = TransformersBackend("Qwen/Qwen2.5-1.5B-Instruct")
        self.assertTrue(backend.local_files_only)
        engine = Engine(Path("not-installed.gguf"), allow_hf_download=False)
        self.assertFalse(engine.allow_hf_download)
        self.assertFalse(TransformersBackend("x", local_files_only=False).local_files_only)

    def test_inference_install_is_gated_by_model_download_permission(self):
        from sideris_desktop.installer import STEPS

        permissions = {step.key: step.permission for step in STEPS}
        self.assertEqual(permissions["inference"], "model_download")
        self.assertEqual(permissions["model"], "model_download")
        self.assertEqual(permissions["training"], "training")
        self.assertEqual(permissions["service"], "background_service")
        self.assertEqual(permissions["login"], "start_at_login")

    def test_repository_does_not_ship_the_dead_browser_app(self):
        for name in ("src", "index.html", "vite.config.ts", "package.json", "public"):
            self.assertFalse((ROOT / name).exists(), f"old browser file still present: {name}")


if __name__ == "__main__":
    unittest.main()

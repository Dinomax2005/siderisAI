# Sideris Desktop 5.0.0 - Architecture

Sideris is a standard-library-first Python application. The only third-party
code required for the shipped product is a bundled llama.cpp runtime; everything
else (GUI, HTTP service, downloader, setup state machine, training supervisor) is
stdlib. That choice keeps the installer small, the supply chain short and the app
easy to inspect.

## Goals

1. **Local by construction.** A model, chats, tools and memory live on the
   person's computer. There is no AI cloud and no API key.
2. **Honest behaviour.** No fake replies, no invented capabilities, no
   ChatGPT/Claude claims. Missing prerequisites produce clear errors.
3. **Interruptions are normal.** Model downloads, setup steps and training runs
   resume after a crash, a stop or a power loss.
4. **Consent before optional work.** Downloading, serving in the background,
   training and start-at-login are all off until explicitly enabled.
5. **Native and quiet.** Tkinter, a stone-and-copper palette, no web view, no
   telemetry.

## Runtime topology

```
┌───────────────────────────────┐        loopback HTTP        ┌────────────────────────────┐
│  Tkinter window (app.py)      │  ───────────────────────▶  │  ThreadingHTTPServer       │
│  Chat / Setup / Training /    │   127.0.0.1:8765 only      │  server.py                 │
│  Settings                     │  ◀───────────────────────  │                            │
└──────────────┬────────────────┘        SSE tokens          └─────────────┬──────────────┘
               │                                                          │
               │ local setup + status                     LocalRuntime ────┤
               ▼                                                          │
┌───────────────────────────────┐                          ┌──────────────▼─────────────┐
│  Setup (installer.py)         │                          │ Assistant (core.py)        │
│  model download, venv,        │                          │  prompts, tools, memory,   │
│  service, login item          │                          │  thinking levels, scoring  │
└──────────────┬────────────────┘                          └──────────────┬─────────────┘
               │                                                          │
               ▼                                                          ▼
┌───────────────────────────────┐                          ┌────────────────────────────┐
│  service.py                   │                          │ Engine (inference.py)      │
│  schtasks / systemd / launchd │                          │  llama.cpp or transformers │
│  heartbeat file               │                          └──────────────┬─────────────┘
└───────────────────────────────┘                                         │
                                                          GGUF file in the data directory
┌───────────────────────────────┐
│  TrainingManager (training)   │  spawns a child process
│  state + monitor              │  ──────────────────────▶  python -m sideris_desktop.train_worker
└───────────────────────────────┘                           transformers Trainer + peft LoRA
                                                            checkpoint-N, worker-state.json
```

The window never loads a model in its own process (except when the user runs the
CLI `chat` command). It talks to the local service over HTTP, which is also the
API scripts use. The service and the window share one `LocalRuntime` when the
window starts the server itself.

## Modules

| File | Responsibility |
|---|---|
| `sideris_desktop/config.py` | Per-user paths, atomic JSON, settings, `Permissions` (all false) |
| `sideris_desktop/models.py` | Catalogue, URL construction, installed-model discovery, selection |
| `sideris_desktop/downloads.py` | Resumable HTTP: `.part`, state JSON, Range/206, SHA-256 |
| `sideris_desktop/inference.py` | llama.cpp and transformers backends, lazy `Engine`, token counts |
| `sideris_desktop/core.py` | System prompt, thinking levels, candidate scoring, tools, memory, workspace retrieval |
| `sideris_desktop/server.py` | Loopback HTTP API, validation, SSE, `LocalRuntime` |
| `sideris_desktop/training.py` | `TrainingState`, `TrainingManager`, checkpoint discovery, child supervision |
| `sideris_desktop/train_worker.py` | Real LoRA training, checkpoints, deadline, stop handling |
| `sideris_desktop/installer.py` | Resumable setup steps, permission enforcement, private training venv |
| `sideris_desktop/service.py` | Background service + start-at-login per OS, heartbeat, signals |
| `sideris_desktop/app.py` | Tkinter window, chat store, Markdown renderer, SSE client |
| `sideris_desktop/cli.py` | `setup`, `chat`, `serve`, `service`, `train`, `status`, `models` |
| `sideris_desktop/__main__.py` | No args → GUI; `--smoke` for CI; otherwise CLI |

## Data directory

```
<SIDERIS_HOME>/
  settings.json                 model, level, port, permissions
  memory.jsonl                  one remembered fact per line
  chats/<id>.json               one conversation per file
  models/                       *.gguf, *.part, *.part.state.json
  workspace/                    the only files the assistant may read
  datasets/                     .jsonl training data
  training/
    runs/<run-id>/              worker-state.json, control.json, final/, summary
    checkpoints/checkpoint-N/   adapter + optimizer state (last 3)
  state/
    setup-state.json            status of each setup step
    training-state.json         the manager's durable view
    training-runtime.json       training venv marker
    service.json                service installation record
    service-heartbeat.json      live service heartbeat
    server.json                 running service info (CLI serve)
  runtime/
    pkgs/                       optional pip-installed extras
    src/                        source copied from a frozen build for training
    training-venv/              private torch/transformers/peft environment
    hf-cache/                   Hugging Face cache for training weights
    wheels/, pip-cache/
  logs/                         setup.log, server.log, training-*.log, service.log
```

`SIDERIS_HOME` overrides the root. `LUNO_HOME` is accepted for upgrades from the
4.x Luno builds. On Windows the default is `%LOCALAPPDATA%\Sideris`; on macOS
`~/Library/Application Support/Sideris`; on Linux `$XDG_DATA_HOME/sideris` or
`~/.local/share/sideris`.

## Permissions

`Permissions` is a dataclass with four booleans, every one defaulting to `False`:
`model_download`, `background_service`, `training`, `start_at_login`.

Enforcement is layered, not just cosmetic:

- `Setup.run` skips permission-gated steps and records `skipped` with the reason.
- `ModelManager.download` raises `PermissionError` unless explicitly allowed.
- `TrainingManager.start` raises `TrainingPermissionError`.
- The HTTP service returns **403** on all training endpoints (`GET /v1/training`,
  `POST /v1/training/start`, `POST /v1/training/stop`) when training is off.
- `service.install_service` / `install_login_item` re-read settings and raise
  `PermissionError` without consent.
- The setup `inference` step is gated by `model_download`; the shipped
  installers bundle llama.cpp, so it is already satisfied there.
- The transformers fallback is local-only by default. It never downloads base
  weights unless `model_download` has been granted.
- Uninstalling a service is **not** permission-gated: removing something the user
  no longer wants must always work.

## Setup state machine

`Setup` runs six steps in order: `directories`, `inference`, `model`, `training`,
`service`, `login`. Before each step it writes `status: running`; on completion it
writes `done` with a detail string; failures are recorded with the error and the
remaining steps continue. `step_done()` re-validates the underlying artifact (the
directory exists, the runtime imports, the model file is complete, the venv marker
exists, the service is installed, the login item exists), so a stale "done" can
never hide a deleted file.

The model step uses `ModelManager.download`, which prints progress into the GUI log
and is resumable. The training step creates `runtime/training-venv` with a system
Python (3.10-3.13), upgrades pip, installs CPU PyTorch from the official CPU
index, then transformers/peft/datasets/accelerate/safetensors/sentencepiece, and
verifies the imports before writing a readiness marker. An interrupted pip run
leaves a venv that the next attempt reuses.

## Download protocol

`downloads.download` is deliberately conservative:

1. Read `<dest>.part.state.json`; compare the URL and total size.
2. Use the **actual size of `<dest>.part`** as the offset.
3. Send `Range: bytes=<offset>-` plus `If-Range` when a validator is known.
4. On `206`, verify `Content-Range` starts at the offset and append (`"ab"`).
5. On `200`, discard the partial file and restart (`"wb"`) rather than mixing
   generations of a changed file.
6. On `416`, remove the stale partial and report it.
7. Write chunks, flush, fsync, and update the state file with bytes/speed.
8. On a short read, keep the bytes and report that a resume is possible.
9. Verify SHA-256 when supplied; delete a corrupt file.
10. Atomically `os.replace(part, dest)` and mark the state complete.

Only `identity` content encoding is accepted, so offsets remain byte-accurate.
Disk space is checked before downloading.

## Inference and quality

`Engine` picks llama.cpp when a GGUF and the runtime exist, otherwise a
transformers fallback (which needs the HF weights). Model loading is lazy and
serialized; all generations pass through a lock because a llama.cpp context is
stateful.

`Assistant.chat` builds a system prompt with identity, date, OS, version,
long-term memory, trusted local tool results and a workspace listing. History is
trimmed to a character budget, keeping the newest turns and always the latest
user message.

Tools run before generation:

- arithmetic through an AST evaluator with a strict whitelist (no `eval`);
- local clock/date;
- workspace listing, file read and keyword search, all sandboxed to
  `workspace/` with path traversal rejected;
- a pure arithmetic or clock question is answered directly by the tool, which is
  faster and cannot hallucinate.

Thinking levels:

| Level | Max tokens | Passes | Temperature(s) |
|---|---|---|---|
| low | 180 | 1 | 0.60 |
| medium | 420 | 1 | 0.70 |
| high | 800 | draft + revise | 0.65, then 0.45 |
| max | 1200 | 3 candidates + revise | 0.55, 0.75, 0.92, then 0.40 |

`score_candidate` rewards substance, unique-word ratio, complete code fences and
bullet structure; it penalizes refusals, filler openers, rambling, truncation and
unbalanced fences. The winner is revised with an explicit instruction to return
only the improved answer.

Memory captures explicit "remember ..." statements and a small set of clear
patterns ("my name is", "I prefer", "I live in"). Deduplication and a 120-fact cap
keep the prompt small; relevance is token overlap plus recency.

## Training lifecycle

`TrainingManager.start`:

1. Checks the training permission (again).
2. Checks that the private runtime marker and venv exist.
3. Validates the `.jsonl` dataset.
4. Finds the highest `checkpoint-N` and appends `--resume-from`.
5. Writes `state/training-state.json` and spawns
   `python -m sideris_desktop.train_worker` with its own process group (Windows:
   `CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW`).
6. Writes a control file (`control.json`) and starts a monitor thread.

The worker:

- loads `{"text"}` or `{"prompt","response"}` records, applies the Qwen chat
  template, tokenizes and truncates to 1024 tokens;
- loads the base model in float32 on CPU (bfloat16 on CUDA, float16 on MPS);
- attaches a rank-16 LoRA adapter to attention and MLP projections;
- trains with `Trainer` (batch 1, grad accumulation 4, LR 2e-4, cosine schedule);
- writes `worker-state.json` after every step for the UI;
- saves a checkpoint every `--save-steps` (default 50), keeping the last 3;
- checks the wall-clock deadline and the control file at every step end;
- on stop, sets `control.should_save` so a checkpoint is written before exit;
- writes `final/` (adapter + tokenizer) and `training-summary.json`;
- exits with a status the monitor turns into `completed`, `stopped`, `failed` or
  `interrupted`.

`TrainingManager.stop` writes the control file *and* sends SIGTERM (or
CTRL_BREAK on Windows), then marks the state `stopping`. The manager's monitor
survives app restarts by validating the PID and, when the app was closed, marks a
dead run `interrupted` so the next start resumes. Signal handlers in the worker
only set an event; the training loop performs the actual save at a step boundary.
This avoids corrupting the checkpoint.

Checkpoint selection is a plain numeric maximum over `checkpoint-*` directories,
including when they were produced by an earlier run. That is intentional: the
latest state of *this model's* training is the correct place to continue.

## Background service

- **Windows:** `schtasks /Create /TN SiderisBackground /SC ONLOGON /RL LIMITED`,
  a per-user task with no admin prompt. Uninstall deletes it.
- **Linux:** `~/.config/systemd/user/sideris.service` with
  `systemctl --user enable --now`; uninstall disables, stops and removes it.
- **macOS:** `~/Library/LaunchAgents/ai.sideris.desktop.plist`, loaded with
  `launchctl bootstrap gui/$UID` (falling back to `launchctl load -w`).

The service command is `Sideris service run` (or `python -m sideris_desktop
service run`). It checks the permission, starts the loopback server, writes
`state/service-heartbeat.json` every 15 seconds and, on SIGTERM/SIGINT, writes a
final stopped heartbeat and shuts the server down. Start-at-login is a separate
per-user mechanism (Startup `.cmd`, `~/.config/autostart`, or a second
LaunchAgent) and is independent from the service.

## HTTP service

`ThreadingHTTPServer`, `daemon_threads`, bound to `127.0.0.1` only. Guards:

- `Host` must be `127.0.0.1`, `localhost` or `[::1]` (DNS-rebinding defence).
- A present `Origin` must be loopback.
- JSON content type, 2 MB body cap, 200 message cap, 200k characters per message.
- One generation at a time per engine; concurrent requests wait on the lock.
- No CORS headers, so a browser page cannot read responses.

SSE (`POST /v1/chat/stream`) emits `status`, `token`, `done` or `error` events and
closes the connection. Chat returns **503** `model_missing` when no weights are
installed, `backend_missing` when no runtime is importable, or
`generation_unavailable` with the backend's error. It never fabricates text.
Training endpoints return **403** `permission_denied` when training is off.

## GUI

`SiderisApp` builds the shell once and packs one of four pages. A single `Text`
widget renders the conversation with tags for user turns, assistant captions,
headings, lists, quotes, inline code and fenced code. A small line-based Markdown
renderer handles the common subset without a dependency. Assistant action rows
(Copy, Regenerate) are embedded windows.

Chat is an SSE client: a worker thread reads events into a `queue.Queue`, and a
Tk `after` poller updates the transcript on the UI thread. Run status, setup logs
and training progress use the same queue pattern, so no worker thread touches Tk.
Training polls `TrainingManager.status()` every five seconds.

## Packaging and release

`Luno.spec` builds a onedir `Sideris` plus a console `SiderisCLI`, bundles
`assets/`, the Python source for the training worker, and `llama_cpp` (via
`collect_all`) when present. `torch`, `transformers`, `peft`, `datasets` and other
heavy training packages are excluded: they belong in the opt-in private venv, not
in a 300 MB installer. The Inno Setup script installs per-user with
`PrivilegesRequired=lowest`.

PyInstaller runs its entry script as `__main__`, without a package context, so
the entry point is the tiny top-level `sideris_launcher.py` which imports
`sideris_desktop.__main__`. Pointing PyInstaller at `sideris_desktop/__main__.py`
directly would break every relative import inside it. A unit test runs the
launcher as a bare script to keep that regression from coming back.

Windowed Windows and macOS builds have no `sys.stdout`, so all packaged output
goes through guarded writers, and `--selftest <path>` writes a JSON report
(version, assets, Tk, app import, llama.cpp) instead of printing. It exits
non-zero when a required piece is missing.

`.github/workflows/release.yml` runs 39 stdlib tests on Windows, macOS and Linux,
then builds all three artifacts, runs `--selftest` on every packaged binary,
smoke-tests the packaged GUI on Windows and Linux (Xvfb), and attaches the
installer, macOS zip and Linux tarball to the GitHub Release via
`softprops/action-gh-release@v2`. Builds are split into one workflow step per
phase (tools, PyInstaller, verify, package) so a failure names its phase, and the
bash steps emit the tail of the failing log as a check-run annotation.

## Security posture and limits

- Loopback-only HTTP, no auth token, intentionally no remote access. A malicious
  local process can already read the user's files; the API adds no privilege.
- Arithmetic uses an AST whitelist, never `eval`.
- Workspace reads resolve and verify containment; absolute paths and `..` are
  rejected.
- Tool results from files are explicitly framed as data, not instructions, and the
  system prompt tells the model not to follow instructions found in them.
- The model download URL is fixed by the catalogue. There is no arbitrary URL
  download API.
- Known limitations: the llama.cpp wheel is CPU-only; Intel Macs are not built;
  training needs a system Python; the LoRA adapter is not automatically loaded
  into the built-in chat; a 1.5B model is not a frontier model.

## Testing

`tests/test_desktop.py` uses only `unittest`. It covers:

- settings round-trip, permissions defaulting off, corrupt-file recovery;
- resumable download against a local Range-capable HTTP server, including a
  pre-seeded `.part`, a 200-only server, checksum success and mismatch;
- highest checkpoint selection and a resume command containing
  `--resume-from .../checkpoint-150`;
- training-state JSON round-trip and dataset validation;
- `/health` OK, `/v1/chat` 503 with no model, training endpoints 403, streaming
  SSE with a fake assistant;
- the calculator sandbox, scoring rubric, memory and workspace traversal defence;
- `python -m sideris_desktop status --json`.

Run:

```bash
python -m unittest discover -s tests -v
```

<p align="center">
  <img src="assets/logo.png" width="96" alt="Sideris AI logo">
</p>

<h1 align="center">Sideris AI Desktop 5.0.0</h1>

<p align="center">
  <strong>A private AI assistant that runs on your computer.</strong><br>
  Your model. Your chats. Your machine. No API keys, no cloud, no telemetry.
</p>

<p align="center">
  <a href="https://github.com/Dinomax2005/siderisAI/releases/latest"><strong>Download the latest release</strong></a>
  &nbsp;·&nbsp;
  <a href="docs/ARCHITECTURE.md">Architecture</a>
  &nbsp;·&nbsp;
  <a href="docs/TRAINING.md">Training</a>
</p>

---

## Download

Everything is on the **[Releases page](https://github.com/Dinomax2005/siderisAI/releases/latest)**.

| Platform | File | Notes |
|---|---|---|
| **Windows 10/11 (x64)** | [`Sideris-Setup-5.0.0.exe`](https://github.com/Dinomax2005/siderisAI/releases/latest/download/Sideris-Setup-5.0.0.exe) | Per-user install, **no administrator password**. Installs to `%LOCALAPPDATA%\Programs\Sideris`. |
| **macOS 11+ (Apple silicon)** | [`Sideris-5.0.0-macos-arm64.zip`](https://github.com/Dinomax2005/siderisAI/releases/latest/download/Sideris-5.0.0-macos-arm64.zip) | Unzip, drag `Sideris.app` to Applications. |
| **Linux (x86_64)** | [`Sideris-5.0.0-linux-x86_64.tar.gz`](https://github.com/Dinomax2005/siderisAI/releases/latest/download/Sideris-5.0.0-linux-x86_64.tar.gz) | Extract and run `./Sideris/Sideris`. |

The installers include the **llama.cpp** inference runtime. The model itself is a
separate download (0.4-9 GB) that happens the first time you finish setup, and only
after you grant the download permission.

The builds are not code-signed (code-signing certificates cost money), so:

* **Windows** may show a SmartScreen prompt. Choose **More info → Run anyway**.
* **macOS** may say the app is from an unidentified developer. Right-click
  `Sideris.app` → **Open**, or run
  `xattr -dr com.apple.quarantine /Applications/Sideris.app` once.

Every packaged binary in the release is self-tested before upload: it proves Tk,
the app package, the icons and the bundled llama.cpp runtime all import.

### First run

1. Open Sideris.
2. Go to **Setup**, pick a model and tick only the permissions you want.
3. Press **Install / resume setup**. Watch the log; the download is resumable.
4. Go to **Chat**. Pick a thinking level and start.

The default model is **Qwen2.5-1.5B-Instruct (Q4_K_M)**: about 1 GB on disk and
comfortable on an 8 GB laptop. If you have 16-32 GB of RAM, the 7B or 14B model is
noticeably stronger. Sideris shows the RAM recommendation for each model.

---

## What Sideris is

A native desktop application (Python + Tkinter, no Electron, no embedded browser)
that turns your own PC into the data centre:

- **A local language model** runs on your CPU through llama.cpp. Nothing is sent
  to an AI provider. There is no account and no API key.
- **A real conversation** with persistent chats, Markdown rendering and streaming
  tokens, in a calm stone-and-copper interface.
- **Thinking levels** that spend real compute: low, medium, high (draft then
  revise) and max (three candidates, scored against a deterministic rubric, then
  revised).
- **Local tools** for arithmetic, the system clock, and reading or searching files
  in your workspace. Pure arithmetic and clock questions are answered
  deterministically by the tool, so they cannot be hallucinated.
- **Long-term memory** of facts you explicitly ask it to remember, plus a few
  clearly implied ones, stored as plain JSON lines on your disk and injected into
  the prompt when relevant.
- **Workspace retrieval** that greps your own text files and puts the best passages
  in context.
- **A local HTTP API** on `127.0.0.1:8765` for scripts and the UI.
- **Optional LoRA fine-tuning** on your own `.jsonl` data, in a separate process,
  with automatic checkpoint resume.
- **An optional background service** and start-at-login, installed only with your
  permission.

### Sideris Core

Sideris is not a thin wrapper around an API. The local **Sideris Core**
(`sideris_desktop/core.py`) is the orchestration layer that decides how to use
the model:

- it retrieves relevant long-term facts and earlier conversations;
- it runs verified local tools (arithmetic, clock, workspace search) and gives
  their output to the model as ground truth, so simple maths and file questions
  cannot be hallucinated;
- it supplies a strict system prompt with today's date, the host platform and
  the user's own context;
- it spends real compute on deliberation: one pass at low/medium, a draft plus
  a revision at high, and three independently sampled candidates scored by a
  deterministic rubric and then revised at max;
- it streams the answer from the local runtime and keeps the full conversation
  as the working memory.

That is a real local core, but it is not a frontier model. A 0.5B-14B model is
still a 0.5B-14B model; the core gets more out of it rather than pretending it
has capabilities that it does not.

### The interface

A 1020×700 native window with a stone sidebar (Chat, Setup, Training, Settings),
serif display type, white cards, warm copper accents and a single conversation
column. Chat page: thinking-level selector, streaming answers, Markdown, code
blocks, copy and regenerate. Training page: dataset picker, hours budget, live
status polled every five seconds, and an honest explanation of what fine-tuning
can and cannot do.

---

## Permissions: everything optional is off

Sideris is deliberately conservative. Four permissions default to **off**:

| Permission | What it lets Sideris do | Default |
|---|---|---|
| `model_download` | Fetch model weights from Hugging Face to your data folder | **off** |
| `background_service` | Install a per-user service that keeps the local API running | **off** |
| `training` | Install PyTorch/PEFT and run fine-tuning on your data | **off** |
| `start_at_login` | Open the Sideris window after you sign in | **off** |

The setup wizard records every step in `state/setup-state.json` and skips steps
that are already valid. Turning a permission off later never uninstalls something
silently; the Settings page has explicit install/remove buttons.

---

## Resuming is a core feature, not an afterthought

Large downloads and long training runs get interrupted. Sideris is built for that.

**Model downloads** write to `<model>.part` and keep
`<model>.part.state.json`. On the next run it sends
`Range: bytes=<existing>-`, expects `206 Partial Content` and appends. If a CDN
answers `200` instead, it restarts cleanly rather than corrupting the file. The
byte count on disk is the source of truth, so a power loss costs at most the OS
page cache, never the whole download. SHA-256 is verified when a digest is known.

**Training** checkpoints every 50 steps (keeping the last three) and records state
in `state/training-state.json`. The next start finds the highest
`checkpoint-N` and automatically adds `--resume-from <path>`. A `--max-hours`
budget stops cleanly at the deadline and saves a checkpoint first. Stopping from
the UI or an OS shutdown behaves the same way: restart and it continues.

**Setup** records each step as it completes, so an interrupted pip install or
model download resumes instead of starting over.

---

## Models

All models are Qwen2.5-Instruct in GGUF Q4_K_M format, Apache-2.0 licensed.

| Key | Size on disk | Minimum RAM | Feel |
|---|---|---|---|
| `qwen2.5-0.5b` | ~0.4 GB | 4 GB | Fastest; simple questions and drafts |
| `qwen2.5-1.5b` | ~1.0 GB | 8 GB | **Default**; good everyday balance |
| `qwen2.5-3b` | ~2.0 GB | 12 GB | A clear step up in reasoning and writing |
| `qwen2.5-7b` | ~4.7 GB | 16 GB | Strong for a laptop; slower on CPU |
| `qwen2.5-14b` | ~9.0 GB | 32 GB | Best quality here; needs a big machine |

Download URL pattern:

```
https://huggingface.co/Qwen/Qwen2.5-{size}B-Instruct-GGUF/resolve/main/qwen2.5-{size}b-instruct-q4_k_m.gguf
```

Models live in your data directory and can be removed at any time.

---

## Thinking levels

| Level | Tokens | Passes | What actually happens |
|---|---|---|---|
| Low | 180 | 1 | One short, quick response |
| Medium | 420 | 1 | One balanced response (default) |
| High | 800 | 2 | A full draft, then a revision pass |
| Max | 1200 | 4 | Three candidates at temperatures 0.55 / 0.75 / 0.92, scored for substance, vocabulary, structure, unclosed code fences and refusals; the winner is revised |

Max is genuinely slower, especially on a CPU. It is not a cosmetic label.

---

## Fine-tuning, honestly

Sideris can run **real LoRA fine-tuning** with `transformers` + `peft` on a
`.jsonl` file of your own examples. It runs in a separate worker process, so
closing the window does not kill it, and it resumes from checkpoints after a
reboot.

```jsonl
{"prompt": "Summarise our refund policy.", "response": "Refunds are available within 30 days with a receipt."}
{"text": "Our support hours are 9:00 to 17:00 CET, Monday to Friday."}
```

What it does: adapts a small local model to your formats, vocabulary and tone, and
can measurably improve narrow, repetitive tasks. What it does **not** do: make a
1.5B model into ChatGPT. Frontier systems are pretrained in data centres on
trillions of tokens; that is a different scale of compute, data and engineering.
Sideris says this in the app too.

The result is a standard PEFT LoRA adapter in
`training/runs/<run-id>/final`, with `checkpoint-N` directories kept alongside it.
Version 5.0.0 does not automatically load that adapter into the built-in
llama.cpp chat; merging or converting it to GGUF is a separate step. Training
requires a system Python 3.10-3.13 (used to create a private virtual environment)
and several GB of free disk for PyTorch and the base weights.

More detail: [docs/TRAINING.md](docs/TRAINING.md).

---

## Privacy

- **No telemetry, analytics, accounts or API keys.**
- The HTTP service binds to **127.0.0.1 only** and rejects non-loopback `Host`
  headers and cross-origin requests.
- The only network requests are:
  1. the model download from Hugging Face, after permission;
  2. the optional PyTorch/transformers install from PyPI, after permission;
  3. the base weights download when a training run starts.
- Your chats, memory, files and models stay in one per-user directory:

| OS | Data directory |
|---|---|
| Windows | `%LOCALAPPDATA%\Sideris` |
| macOS | `~/Library/Application Support/Sideris` |
| Linux | `~/.local/share/sideris` (or `$XDG_DATA_HOME`) |

Override with `SIDERIS_HOME`. The old `LUNO_HOME` variable is still honoured for
people upgrading from the 4.x "Luno" builds.

---

## Command line

The packaged app ships both `Sideris` (window) and `SiderisCLI` (terminal). From a
source checkout, use `python -m sideris_desktop`.

```bash
sideris                                  # open the window (no subcommand)
sideris setup --model qwen2.5-1.5b --yes # allow the download and install
sideris setup --model qwen2.5-3b --yes --background --training --start-at-login
sideris chat --level high                # terminal chat, streamed
sideris chat "Explain LoRA in two sentences."
sideris serve                            # local API on 127.0.0.1:8765
sideris service install|uninstall|run|status
sideris train start --dataset data.jsonl --hours 2 --epochs 1
sideris train status
sideris train stop
sideris status --json
sideris models list
sideris example-dataset
```

`--yes` is an explicit grant of the **model download** consent for that run. It
does not enable the service, training or start-at-login; those need their own
flags or the Settings page.

### HTTP API

```bash
curl http://127.0.0.1:8765/health
curl -X POST http://127.0.0.1:8765/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello"}],"level":"medium"}'
```

`GET /health`, `GET /v1/status`, `GET /v1/models`, `POST /v1/chat`,
`POST /v1/chat/stream` (Server-Sent Events), `GET /v1/training`,
`POST /v1/training/start`, `POST /v1/training/stop`.

Without a model, chat returns **503** with an honest error - never a fabricated
answer. Training endpoints return **403** while the training permission is off.

---

## Build from source

```bash
git clone https://github.com/Dinomax2005/siderisAI.git
cd siderisAI
python -m unittest discover -s tests -v     # 43 stdlib tests
python -m sideris_desktop                    # open the window
```

On Linux you need Tk and the shared Python runtime PyInstaller links against:
`sudo apt install python3-tk python3-venv libpython3.10` (use `libpython3.X` for
your interpreter). On Debian/Ubuntu you can install Sideris with
`./installer/install.sh` (per-user, no sudo). The installers are built with
PyInstaller and, on Windows, Inno Setup:

```bash
pip install pyinstaller .
pyinstaller --noconfirm Luno.spec
# prove the packaged app imports Tk, the app and llama.cpp, without a display:
./dist/Sideris/SiderisCLI --selftest selftest.json   # Windows: dist\Sideris\Sideris.exe --selftest
# Windows installer:
ISCC installer/windows/SiderisSetup.iss
```

`--selftest` is what the release workflow runs on every packaged binary before it
publishes anything. It writes a JSON report and exits non-zero if Tk, the app
package, the assets or the bundled inference runtime are missing.

Icons are derived from the original artwork, never redrawn:

```bash
python scripts/make_icons.py   # needs Pillow
```

---

## Honest answers to fair questions

**Is this as smart as ChatGPT or Claude?**
No. It runs a 0.5B-14B open-weight model on your own hardware. The product is
engineered to get the most out of that model - deliberation, tools, memory,
retrieval - and it is genuinely useful for writing, code review, planning and
working with your own files. It is not a frontier model, and no honest local app
of this size can be. The README and the app say so rather than pretending.

**Does it use the internet?**
Only for the permissioned model download and the optional training runtime. The
assistant itself has no browsing tool and will tell you when a question needs
live data.

**Can it see my files?**
Only text files in your Sideris workspace folder, and only when you ask it to
read or search them. Path traversal outside the workspace is rejected.

**What happens if I close the app during training?**
The trainer is a separate process and keeps running. If the machine shuts down,
restart training and it resumes from `checkpoint-N`.

**Why is there still a file called `Luno.spec`?**
The 5.0 line began as "Luno Desktop". The product, window, installers and data
directory are now Sideris; the spec filename is kept so existing build scripts do
not break.

---

## License

Apache-2.0. The model weights are downloaded separately under their own
(Apache-2.0) licence from Qwen.

<p align="center"><sub>Sideris Desktop 5.0.0 · built for computers that belong to the people using them.</sub></p>

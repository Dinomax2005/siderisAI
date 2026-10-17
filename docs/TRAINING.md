# Fine-tuning with Sideris Desktop

Sideris can fine-tune the small local model on your own data with a real LoRA
run. It is not a toy simulation: it uses PyTorch, Hugging Face `transformers`, the
`peft` library and a `Trainer` loop, saves checkpoints, and resumes after a stop
or a power loss.

It is also not frontier-model training, and the app says so. Read the honesty
section at the end before you start.

## What LoRA does

A full fine-tune updates every weight in the model. LoRA instead freezes the base
model and trains two small matrices next to selected layers. For Qwen2.5-1.5B the
trainable part is a few million parameters, not 1.5 billion. The result is a small
adapter file that changes how the model behaves without changing the base weights.

What that buys you:

- your output formats and document structures;
- your product vocabulary and tone;
- narrow, repetitive tasks with consistent answers;
- a private model that does not send your examples to a provider.

What it does not buy you:

- new general knowledge at frontier scale;
- the reasoning of a model trained on trillions of tokens;
- a fix for a task the base model fundamentally cannot represent.

## Requirements

- A working Sideris installation.
- **Training permission** enabled (Setup or Settings). It is off by default.
- The **training runtime**: PyTorch (CPU), transformers, peft, datasets,
  accelerate, safetensors and sentencepiece, installed by Setup into
  `runtime/training-venv`. Expect several hundred MB to a few GB.
- A **system Python 3.10-3.13** with `venv` support. The packaged app needs it to
  create that private environment; the app does not modify your system Python.
- 4 GB+ free disk, plus room for the base model weights (about 3 GB for the 1.5B
  model in bfloat16/float32; more for larger models).
- Time. CPU fine-tuning is slow. Start with a small dataset and a short budget.

## Dataset format

A UTF-8 `.jsonl` file, one JSON object per line. Two accepted shapes:

```jsonl
{"text": "Our refund window is 30 days from delivery. A receipt is required."}
{"prompt": "How long is the refund window?", "response": "30 days from delivery, with a receipt."}
{"system": "You are a support assistant for Acme.", "prompt": "Where is my order?", "response": "Send me the order number and I will check the tracking."}
```

Rules:

- `text` is used as a raw document/completion.
- `prompt` + `response` is formatted as a user/assistant chat turn. `system` is
  optional.
- Blank lines are skipped. Every non-blank line must be a JSON object.
- Keep records short (a few hundred tokens). They are truncated to 1024 tokens by
  default (`--max-seq-len` in the worker).
- Quality beats quantity. A few hundred clean, consistent examples usually beat
  thousands of noisy ones.
- Do not include secrets you would not want in a checkpoint file on your disk.

Create an example and edit it:

```bash
sideris example-dataset
# writes ~/.local/share/sideris/datasets/example.jsonl
```

In the app: **Training → Create example**, or **Browse…** to choose your file.

## Starting a run

### In the window

1. **Training**.
2. Choose the dataset.
3. Set the time budget (hours) and epochs.
4. **Start training**.

The page polls the worker every five seconds and shows status, step/max steps,
loss, the latest checkpoint and the tail of the log.

### From the terminal

```bash
sideris train start --dataset /path/to/data.jsonl --hours 2 --epochs 1
sideris train status
sideris train stop
```

`status --json` prints the complete state for scripts.

### Over HTTP

```bash
curl -X POST http://127.0.0.1:8765/v1/training/start \
  -H 'Content-Type: application/json' \
  -d '{"dataset": "/path/to/data.jsonl", "hours": 2, "epochs": 1}'
```

Returns **403** without the training permission, **503** if the runtime is not
installed, **409** if a run is active.

## What happens, step by step

1. **Supervisor.** `TrainingManager` validates the dataset and finds the highest
   `checkpoint-N` under `training/checkpoints`.
2. **Child process.** It spawns
   `python -m sideris_desktop.train_worker ... --max-hours 2 --save-steps 50
   --keep 3` with `SIDERIS_HOME`, `PYTHONPATH` and `HF_HOME` set, in a new process
   group, and writes `state/training-state.json`.
3. **Data.** The worker reads the JSONL, applies the Qwen chat template, tokenizes
   with truncation, and builds a Hugging Face `Dataset`.
4. **Model.** It loads `Qwen/Qwen2.5-<size>B-Instruct` (from the local HF cache if
   present, otherwise downloaded), sets `use_cache=False`, enables input grads and
   attaches a rank-16 LoRA adapter to `q_proj`, `k_proj`, `v_proj`, `o_proj`,
   `gate_proj`, `up_proj`, `down_proj`.
5. **Training.** `Trainer` with batch size 1, gradient accumulation 4, learning
   rate 2e-4, cosine schedule, warmup, seed 42, and no evaluation pass.
6. **Progress.** After every step the worker atomically writes
   `training/runs/<run-id>/worker-state.json` with step, max steps, epoch, loss and
   the latest checkpoint. The UI reads it.
7. **Checkpoints.** Every 50 steps (`--save-steps`), and immediately before a stop,
   a `checkpoint-N` directory is written with the adapter and optimizer state.
   Only the last three are kept.
8. **Finish.** The final adapter and tokenizer are saved to
   `training/runs/<run-id>/final`, and `training-summary.json` is written.

## Resuming

This is the central promise of training in Sideris:

- **Stop button / `sideris train stop`:** the manager writes `{"stop": true}` to
  `control.json` and sends SIGTERM (CTRL_BREAK on Windows). The worker sets its
  stop flag, lets the current step finish, sets `control.should_save`, writes a
  checkpoint and exits. State becomes `stopped`.
- **Close the window:** the trainer is a separate process and keeps running.
- **Power loss / kill -9:** the last completed checkpoint is intact. On the next
  launch the manager sees a dead PID and marks the run `interrupted`.
- **Time budget:** checked at each step boundary. At the deadline the worker stops
  cleanly, saves, and exits with reason `time budget reached`.
- **Next start:** `latest_checkpoint()` scans every `checkpoint-*` directory,
  chooses the numerically highest, and adds
  `--resume-from <path>` automatically. You do not need a flag.

Checkpoints are the adapter plus optimizer state, not a full model copy, so three
of them are small.

## Output files

```
<data>/training/
  checkpoints/checkpoint-50/          adapter_model.safetensors, optimizer.pt, ...
  checkpoints/checkpoint-100/
  runs/run-YYYYmmdd-HHMMSS/
    worker-state.json
    control.json
    final/                            adapter + tokenizer (standard PEFT format)
    training-summary.json
```

`state/training-state.json` is the manager's durable view; the worker's own
progress is in `worker-state.json`.

## Using the adapter

The result is a standard PEFT LoRA adapter. You can load it in any Python program
that uses `transformers` + `peft`:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base = "Qwen/Qwen2.5-1.5B-Instruct"
adapter = "…/training/runs/<run-id>/final"

tokenizer = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float32)
model = PeftModel.from_pretrained(model, adapter)
model.eval()

messages = [{"role": "user", "content": "Summarise our refund policy."}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt")
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

To use the adapter in the built-in llama.cpp chat, convert or merge it to GGUF
(the `convert_lora_to_gguf.py` script in the llama.cpp repository) or merge the
adapter into the base weights and export a GGUF with `llama.cpp`'s conversion
tooling. **Sideris 5.0.0 does not automate that conversion**, and the chat engine
does not load PEFT adapters directly. The Training page and README state this so
the output is not oversold.

## Troubleshooting

**"The training runtime is not installed"**
Run Setup with the training permission enabled, or
`sideris setup --training --yes`. It creates a private venv; your system Python is
not modified. The app needs Python 3.10-3.13 installed.

**"No Python interpreter with pip was found"**
Install Python from python.org (Windows/macOS) or your package manager (Linux),
including `venv`/`ensurepip`, then run setup again. The packaged app cannot create
a venv without a system Python.

**PyTorch install fails**
The installer tries `https://download.pytorch.org/whl/cpu` first and then the
default PyPI index. A proxy or offline machine will fail both; the log tail in the
Setup page records the reason.

**Out of memory**
Use a smaller base model (0.5B/1.5B), a shorter `--max-seq-len`, fewer records, or
close other applications. The trainer uses float32 on CPU, which is memory-hungry.

**Training is very slow**
Expected on CPU. A 1.5B model with a small dataset can take hours for one epoch.
Set a time budget you are comfortable with; the run stops cleanly and resumes.

**The run says "interrupted"**
The worker process died (reboot, kill, crash). Start again; the manager will
resume from the highest checkpoint. Check
`logs/training-<run-id>.log` for the last error.

**Dataset line errors**
The error names the line. Each line must parse as a JSON object with `text`, or
`prompt` + `response`.

## Honesty

Sideris is a local application for small open-weight models. Fine-tuning helps a
small model follow your patterns. It does not reproduce frontier-model pretraining,
which uses data-centre-scale compute and trillions of tokens. The app says this on
the Training page, in the README and here. That is deliberate: an honest tool is
more useful than an inflated claim.

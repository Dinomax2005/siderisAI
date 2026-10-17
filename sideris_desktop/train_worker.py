"""Standalone LoRA trainer, launched by ``TrainingManager``.

Run as its own process so that a training run:

* survives closing the Sideris window;
* can be stopped and resumed without touching the app;
* checkpoints every ``--save-steps`` steps and keeps the last three;
* stops cleanly when the ``--hours`` budget is exhausted.

Dataset format (JSON Lines):

    {"text": "..."}                              # raw completion / document
    {"prompt": "...", "response": "..."}         # chat pair
    {"system": "...", "prompt": "...", "response": "..."}   # optional system

This is genuine LoRA fine-tuning with transformers + peft.  It adapts a
small open model to your data.  It is not, and cannot be, frontier-model
pretraining: that needs tens of thousands of GPUs and trillions of tokens.
The app says so plainly on the Training page.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import sys
import time
import traceback
from pathlib import Path

from .config import paths, read_json, utc_now, write_json

DEFAULT_MAX_SEQ_LEN = 1024


def _now() -> str:
    return utc_now()


class StateWriter:
    """Atomic progress file read by the desktop app."""

    def __init__(self, path: Path, run_id: str) -> None:
        self.path = Path(path)
        self.run_id = run_id
        self.data: dict = {
            "run_id": run_id,
            "status": "starting",
            "phase": "starting",
            "step": 0,
            "max_steps": None,
            "epoch": 0.0,
            "loss": None,
            "checkpoint": None,
            "reason": None,
            "error": None,
            "pid": os.getpid(),
            "updated_at": _now(),
        }
        self.write()

    def update(self, **values) -> None:
        self.data.update(values)
        self.data["updated_at"] = _now()
        self.write()

    def write(self) -> None:
        try:
            write_json(self.path, self.data)
        except OSError:
            pass


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="sideris_desktop.train_worker",
        description="Sideris local LoRA trainer",
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--hours", "--max-hours", dest="hours", type=float, default=2.0,
                        help="stop cleanly after this many hours")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--keep", type=int, default=3)
    parser.add_argument("--checkpoints-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--state-file", required=True)
    parser.add_argument("--control-file", default=None)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--max-seq-len", type=int, default=DEFAULT_MAX_SEQ_LEN)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    return parser.parse_args(argv)


def load_records(dataset: Path) -> list[dict]:
    records: list[dict] = []
    with open(dataset, "r", encoding="utf-8") as fh:
        for number, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Dataset line {number} is not valid JSON: {exc}")
            if not isinstance(record, dict):
                raise ValueError(f"Dataset line {number} is not a JSON object.")
            if not (
                isinstance(record.get("text"), str)
                or (
                    isinstance(record.get("prompt"), str)
                    and isinstance(record.get("response"), str)
                )
            ):
                raise ValueError(
                    f'Dataset line {number} needs "text" or "prompt"+"response".'
                )
            records.append(record)
    if not records:
        raise ValueError("Dataset contains no usable records.")
    return records


def record_to_text(record: dict, tokenizer) -> str:
    if isinstance(record.get("text"), str):
        return record["text"]
    messages = []
    if record.get("system"):
        messages.append({"role": "system", "content": str(record["system"])})
    messages.append({"role": "user", "content": str(record["prompt"])})
    messages.append({"role": "assistant", "content": str(record["response"])})
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False)
    except Exception:
        return (
            f"<|im_start|>user\n{record['prompt']}<|im_end|>\n"
            f"<|im_start|>assistant\n{record['response']}<|im_end|>\n"
        )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data_paths = paths()
    checkpoints_dir = Path(args.checkpoints_dir)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    run_dir = data_paths.runs / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = data_paths.logs / f"training-{args.run_id}.log"
    state = StateWriter(Path(args.state_file), args.run_id)

    free_gb = shutil.disk_usage(str(data_paths.root)).free / 1024**3
    if free_gb < 4:
        state.update(
            status="failed",
            phase="preflight",
            error=f"Only {free_gb:.1f} GB free. Free at least 4 GB for the base model and checkpoints.",
        )
        return 2

    stop_event = {"stop": False, "reason": None}

    def request_stop(signum, frame):  # noqa: ARG001
        stop_event["stop"] = True
        stop_event["reason"] = stop_event["reason"] or f"signal {signum}"

    for name in ("SIGTERM", "SIGINT", "SIGBREAK"):
        if hasattr(signal, name):
            try:
                signal.signal(getattr(signal, name), request_stop)
            except (ValueError, OSError):
                pass

    control_file = Path(args.control_file) if args.control_file else None
    started_at = time.time()
    deadline = started_at + max(args.hours, 0.01) * 3600

    def control_requests_stop() -> bool:
        if stop_event["stop"]:
            return True
        if control_file is not None:
            payload = read_json(control_file, {}) or {}
            if isinstance(payload, dict) and payload.get("stop"):
                stop_event["stop"] = True
                stop_event["reason"] = stop_event["reason"] or "stop requested in the app"
                return True
        return False

    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainerCallback,
            TrainingArguments,
        )
    except Exception as exc:  # pragma: no cover - depends on optional runtime
        state.update(
            status="failed",
            phase="import",
            error=(
                "The training runtime is incomplete: "
                f"{exc}. Run Setup with the training permission enabled."
            ),
        )
        return 2

    resume_from = Path(args.resume_from) if args.resume_from else None
    if resume_from and not resume_from.is_dir():
        resume_from = None

    try:
        state.update(
            status="running",
            phase="loading dataset",
            resumed_from=str(resume_from) if resume_from else None,
            max_hours=args.hours,
        )
        records = load_records(Path(args.dataset))
        state.update(phase="loading tokenizer")

        tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        encoded = []
        for record in records:
            text = record_to_text(record, tokenizer)
            tokens = tokenizer(
                text,
                truncation=True,
                max_length=max(args.max_seq_len, 64),
                padding=False,
            )
            tokens["labels"] = list(tokens["input_ids"])
            encoded.append(tokens)
        dataset = Dataset.from_list(encoded)
        state.update(phase="loading model", records=len(records))

        use_cuda = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
        use_mps = bool(
            getattr(getattr(torch, "backends", None), "mps", None)
            and torch.backends.mps.is_available()
        )
        if use_cuda:
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        elif use_mps:
            dtype = torch.float16
        else:
            dtype = torch.float32

        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        model.config.use_cache = False
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

        lora = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        )
        model = get_peft_model(model, lora)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        state.update(
            phase="training",
            trainable_params=trainable,
            total_params=total,
            device="cuda" if use_cuda else ("mps" if use_mps else "cpu"),
        )

        class ProgressCallback(TrainerCallback):
            """Persist progress, enforce the deadline and checkpoint on stop."""

            def __init__(self) -> None:
                self.last_loss = None

            def _write(self, **values) -> None:
                latest = None
                try:
                    for candidate in checkpoints_dir.glob("checkpoint-*"):
                        if candidate.is_dir():
                            number = int(candidate.name.split("-")[-1])
                            if latest is None or number > latest[0]:
                                latest = (number, candidate)
                except (ValueError, OSError):
                    pass
                values.setdefault("checkpoint", str(latest[1]) if latest else None)
                state.update(**values)

            def on_train_begin(self, args_, state_, control, **kwargs):
                self._write(status="running", phase="training", max_steps=state_.max_steps)

            def on_log(self, args_, state_, control, logs=None, **kwargs):
                if logs and "loss" in logs:
                    try:
                        self.last_loss = float(logs["loss"])
                    except (TypeError, ValueError):
                        pass
                    self._write(loss=self.last_loss, step=state_.global_step)

            def on_step_end(self, args_, state_, control, **kwargs):
                elapsed = time.time() - started_at
                reason = None
                if control_requests_stop():
                    reason = stop_event["reason"] or "stop requested"
                elif time.time() >= deadline:
                    reason = "time budget reached"
                if reason:
                    control.should_training_stop = True
                    control.should_save = True
                    if not stop_event["stop"]:
                        stop_event["stop"] = True
                        stop_event["reason"] = reason
                    self._write(
                        status="stopping",
                        step=state_.global_step,
                        epoch=state_.epoch,
                        loss=self.last_loss,
                        reason=reason,
                        elapsed_seconds=round(elapsed, 1),
                    )
                elif state_.global_step % 5 == 0:
                    self._write(
                        step=state_.global_step,
                        max_steps=state_.max_steps,
                        epoch=state_.epoch,
                        loss=self.last_loss,
                        elapsed_seconds=round(elapsed, 1),
                    )

            def on_save(self, args_, state_, control, checkpoint=None, **kwargs):
                folder = getattr(checkpoint, "folder", None) if checkpoint else None
                self._write(checkpoint=str(folder) if folder else None, step=state_.global_step)

            def on_train_end(self, args_, state_, control, **kwargs):
                self._write(step=state_.global_step, loss=self.last_loss)

        training_args = TrainingArguments(
            output_dir=str(checkpoints_dir),
            per_device_train_batch_size=max(args.batch_size, 1),
            gradient_accumulation_steps=max(args.grad_accum, 1),
            learning_rate=args.learning_rate,
            num_train_epochs=max(args.epochs, 0.01),
            logging_steps=5,
            save_strategy="steps",
            save_steps=max(args.save_steps, 1),
            save_total_limit=max(args.keep, 1),
            report_to=[],
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            use_cpu=not (use_cuda or use_mps),
            fp16=False,
            bf16=use_cuda and dtype == torch.bfloat16,
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
            seed=42,
            data_seed=42,
            gradient_checkpointing=use_cuda,
            optim="adamw_torch",
            logging_first_step=True,
            disable_tqdm=True,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
            callbacks=[ProgressCallback()],
        )

        state.update(
            status="running",
            phase="training",
            started_at=_now(),
            resumed_from=str(resume_from) if resume_from else None,
        )
        if resume_from:
            trainer.train(resume_from_checkpoint=str(resume_from))
        else:
            trainer.train()

        final_dir = run_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(final_dir))
        tokenizer.save_pretrained(str(final_dir))

        latest = None
        for candidate in checkpoints_dir.glob("checkpoint-*"):
            if candidate.is_dir():
                try:
                    number = int(candidate.name.split("-")[-1])
                except ValueError:
                    continue
                if latest is None or number > latest[0]:
                    latest = (number, candidate)
        summary = {
            "run_id": args.run_id,
            "dataset": args.dataset,
            "base_model": args.base_model,
            "records": len(records),
            "epochs": args.epochs,
            "max_hours": args.hours,
            "resumed_from": str(resume_from) if resume_from else None,
            "final_dir": str(final_dir),
            "checkpoint": str(latest[1]) if latest else None,
            "finished_at": _now(),
            "note": (
                "LoRA adapter trained locally. It adapts style and task "
                "behaviour; it is not frontier-model pretraining."
            ),
        }
        write_json(run_dir / "training-summary.json", summary)

        if stop_event["stop"]:
            state.update(
                status="stopped",
                phase="stopped",
                reason=stop_event["reason"] or "stopped",
                checkpoint=str(latest[1]) if latest else None,
                final_dir=str(final_dir),
                finished_at=_now(),
            )
        else:
            state.update(
                status="completed",
                phase="completed",
                checkpoint=str(latest[1]) if latest else None,
                final_dir=str(final_dir),
                finished_at=_now(),
            )
        return 0

    except KeyboardInterrupt:
        state.update(status="stopped", phase="stopped", reason="interrupted", finished_at=_now())
        return 130
    except Exception as exc:  # pragma: no cover - environment dependent
        details = traceback.format_exc()
        state.update(
            status="failed",
            phase=state.data.get("phase"),
            error=f"{type(exc).__name__}: {exc}",
            traceback=details[-4000:],
            finished_at=_now(),
        )
        try:
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write(details + "\n")
        except OSError:
            pass
        print(details, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

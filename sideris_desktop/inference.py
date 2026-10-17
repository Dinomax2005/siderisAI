"""Local inference backends.

GGUF models run through ``llama-cpp-python`` (bundled with the installers).
If a GGUF file exists but llama.cpp is unavailable, the engine can fall
back to a Hugging Face ``transformers`` pipeline with the same base model.
Neither path touches the network at generation time.

``Engine`` loads the model lazily, serialises generations (llama.cpp
contexts are not thread-safe) and exposes token counting for the UI.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Callable

from .models import get_spec


def approx_tokens(text: str) -> int:
    """Cheap token estimate used when no tokenizer is loaded."""
    return max(1, len(text or "") // 4)


class NoBackendError(RuntimeError):
    pass


class BackendLoadError(RuntimeError):
    pass


def _threads() -> int:
    return max(1, min(os.cpu_count() or 4, 8))


class LlamaCppBackend:
    name = "llama.cpp"

    def __init__(self, model_path: Path, *, context: int = 4096, gpu_layers: int = 0) -> None:
        self.model_path = Path(model_path)
        self.context = context
        self.gpu_layers = gpu_layers
        self._llm = None
        self._lock = threading.Lock()

    @staticmethod
    def available() -> bool:
        try:
            import llama_cpp  # noqa: F401

            return True
        except Exception:
            return False

    def load(self) -> None:
        if self._llm is not None:
            return
        with self._lock:
            if self._llm is not None:
                return
            try:
                from llama_cpp import Llama
            except Exception as exc:  # pragma: no cover - depends on install
                raise NoBackendError(
                    "llama-cpp-python is not installed. Run Sideris setup to "
                    "install the local inference runtime."
                ) from exc
            if not self.model_path.is_file():
                raise BackendLoadError(f"Model file is missing: {self.model_path}")
            try:
                self._llm = Llama(
                    model_path=str(self.model_path),
                    n_ctx=self.context,
                    n_threads=_threads(),
                    n_batch=256,
                    n_gpu_layers=self.gpu_layers,
                    chat_format="chatml",
                    verbose=False,
                )
            except Exception as exc:
                raise BackendLoadError(
                    f"Could not load {self.model_path.name}: {exc}. "
                    "The file may be incomplete - run setup again to verify it."
                ) from exc

    def count_tokens(self, text: str) -> int:
        try:
            self.load()
            return len(self._llm.tokenize(text.encode("utf-8")))  # type: ignore[union-attr]
        except Exception:
            return approx_tokens(text)

    def generate(
        self,
        messages: list[dict],
        *,
        max_tokens: int,
        temperature: float = 0.7,
        top_p: float = 0.9,
        seed: int | None = None,
        on_token: Callable[[str], None] | None = None,
    ) -> str:
        self.load()
        assert self._llm is not None
        params = dict(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=40,
            min_p=0.05,
            repeat_penalty=1.08,
            presence_penalty=0.0,
            stop=["<|im_end|>", "<|endoftext|>"],
            stream=True,
        )
        if seed is not None:
            params["seed"] = seed
        try:
            stream = self._llm.create_chat_completion(**params)
            parts: list[str] = []
            for chunk in stream:
                try:
                    piece = chunk["choices"][0]["delta"].get("content") or ""
                except (KeyError, IndexError, TypeError):
                    continue
                if piece:
                    parts.append(piece)
                    if on_token:
                        on_token(piece)
            return "".join(parts).strip()
        except Exception as exc:
            raise BackendLoadError(f"Generation failed: {exc}") from exc


class TransformersBackend:
    """Fallback for machines that already have torch + transformers.

    This uses the original (non-GGUF) Hugging Face checkpoint.  By default
    it is strictly local: only already-cached weights are loaded, so opening
    a chat never starts a multi-gigabyte download behind the user's back.
    The caller may opt in to a download *only* when the user has granted the
    model-download permission.
    """

    name = "transformers"

    def __init__(
        self,
        repo_or_path: str,
        *,
        context: int = 4096,
        local_files_only: bool = True,
    ) -> None:
        self.repo_or_path = repo_or_path
        self.context = context
        self.local_files_only = local_files_only
        self._model = None
        self._tokenizer = None
        self._lock = threading.Lock()

    @staticmethod
    def available() -> bool:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401

            return True
        except Exception:
            return False

    def load(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                import torch
            except Exception as exc:
                raise NoBackendError(
                    "No inference backend is installed. Install llama-cpp-python "
                    "or torch+transformers, then run setup again."
                ) from exc
            try:
                local_only = self.local_files_only
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.repo_or_path, local_files_only=local_only
                )
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.repo_or_path,
                    torch_dtype="auto",
                    device_map="auto" if torch.cuda.is_available() else None,
                    local_files_only=local_only,
                )
                self._model.eval()
            except Exception as exc:
                raise BackendLoadError(
                    f"Could not load {self.repo_or_path}: {exc}"
                ) from exc

    def count_tokens(self, text: str) -> int:
        if self._tokenizer is None:
            return approx_tokens(text)
        return len(self._tokenizer.encode(text))

    def _prompt(self, messages: list[dict]) -> str:
        self.load()
        assert self._tokenizer is not None
        try:
            return self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            text = "".join(
                f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages
            )
            return text + "<|im_start|>assistant\n"

    @property
    def _torch(self):
        import torch

        return torch

    def generate(
        self,
        messages: list[dict],
        *,
        max_tokens: int,
        temperature: float = 0.7,
        top_p: float = 0.9,
        seed: int | None = None,
        on_token: Callable[[str], None] | None = None,
    ) -> str:
        self.load()
        torch = self._torch
        assert self._model is not None and self._tokenizer is not None
        if seed is not None:
            torch.manual_seed(seed)
        prompt = self._prompt(messages)
        encoded = self._tokenizer(prompt, return_tensors="pt")
        device = getattr(self._model, "device", "cpu")
        encoded = {k: v.to(device) for k, v in encoded.items()}
        kwargs = dict(
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-4),
            top_p=top_p,
            repetition_penalty=1.08,
            pad_token_id=self._tokenizer.eos_token_id,
        )
        input_len = encoded["input_ids"].shape[1]
        if on_token:
            from transformers import TextIteratorStreamer

            streamer = TextIteratorStreamer(
                self._tokenizer, skip_prompt=True, skip_special_tokens=True
            )
            thread = threading.Thread(
                target=self._model.generate,
                kwargs={**encoded, **kwargs, "streamer": streamer},
                daemon=True,
            )
            thread.start()
            parts = []
            for piece in streamer:
                parts.append(piece)
                on_token(piece)
            thread.join(timeout=1)
            return "".join(parts).strip()

        with torch.no_grad():
            output = self._model.generate(**encoded, **kwargs)
        reply = self._tokenizer.decode(output[0][input_len:], skip_special_tokens=True)
        return reply.strip()


def backend_report() -> dict:
    llama = LlamaCppBackend.available()
    transformers = TransformersBackend.available()
    return {
        "llama_cpp": llama,
        "transformers": transformers,
        "available": llama or transformers,
        "preferred": "llama.cpp" if llama else ("transformers" if transformers else None),
    }


class Engine:
    """Lazily loads one model and exposes generation + token counts."""

    def __init__(
        self,
        model_path: Path | None,
        *,
        model_key: str | None = None,
        context: int = 4096,
        gpu_layers: int = 0,
        allow_hf_download: bool = False,
    ) -> None:
        self.model_path = Path(model_path) if model_path else None
        self.model_key = model_key
        self.context = context
        self.gpu_layers = gpu_layers
        # The transformers fallback may fetch weights only when the user has
        # explicitly granted model_download.  Otherwise it is local-only.
        self.allow_hf_download = bool(allow_hf_download)
        self._backend = None
        self._lock = threading.Lock()
        self._gen_lock = threading.Lock()

    def backend(self):
        if self._backend is not None:
            return self._backend
        with self._lock:
            if self._backend is not None:
                return self._backend
            if self.model_path and LlamaCppBackend.available():
                self._backend = LlamaCppBackend(
                    self.model_path, context=self.context, gpu_layers=self.gpu_layers
                )
            elif TransformersBackend.available():
                repo = "Qwen/Qwen2.5-1.5B-Instruct"
                if self.model_key:
                    try:
                        repo = get_spec(self.model_key).train_repo
                    except KeyError:
                        pass
                self._backend = TransformersBackend(
                    repo,
                    context=self.context,
                    local_files_only=not self.allow_hf_download,
                )
            else:
                raise NoBackendError(
                    "No local inference backend is installed. Run setup to install "
                    "llama-cpp-python (recommended), or install torch + transformers."
                )
            return self._backend

    @property
    def backend_name(self) -> str:
        try:
            return self.backend().name
        except NoBackendError:
            return "none"

    def count_tokens(self, text: str) -> int:
        try:
            return self.backend().count_tokens(text)
        except Exception:
            return approx_tokens(text)

    def generate(self, messages, **kwargs) -> str:
        # One generation at a time: llama.cpp contexts are stateful and the
        # UI/service may issue overlapping requests.
        with self._gen_lock:
            return self.backend().generate(messages, **kwargs)



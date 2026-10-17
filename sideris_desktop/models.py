"""The local model catalogue.

Sideris ships no weights inside the installer.  Models are downloaded on
request, only after the model-download permission is granted, and are
plain GGUF files that stay on the person's disk.

The default is Qwen2.5-1.5B-Instruct (Q4_K_M): roughly 1 GB on disk and
comfortable on an 8 GB machine.  Larger entries are offered because
quality scales with size; the app is honest that a 1.5B model is not a
frontier cloud model, and never pretends otherwise.
"""

from __future__ import annotations

import os
import platform
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import __app_name__
from .config import Paths, paths, read_json, total_ram_gb
from .downloads import DownloadResult, Progress, human_bytes, partial_info, download

DEFAULT_MODEL = "qwen2.5-1.5b"

BASE_URL = (
    "https://huggingface.co/Qwen/Qwen2.5-{size}B-Instruct-GGUF/resolve/main/"
    "qwen2.5-{size}b-instruct-q4_k_m.gguf"
)


@dataclass(frozen=True)
class ModelSpec:
    key: str
    size_b: float
    download_gb: float
    min_ram_gb: int
    quality: int
    blurb: str
    context: int = 8192
    license: str = "Apache-2.0"
    filename: str = ""
    url: str = ""
    train_repo: str = ""

    @property
    def display_name(self) -> str:
        return f"Qwen2.5 {self.size_b:g}B Instruct"

    @property
    def size_label(self) -> str:
        return f"~{self.download_gb:g} GB"

    @property
    def ram_label(self) -> str:
        return f"{self.min_ram_gb} GB RAM"

    @property
    def recommended_for(self) -> str:
        if self.min_ram_gb <= 4:
            return "Fastest, lightest"
        if self.min_ram_gb <= 8:
            return "Balanced default"
        if self.min_ram_gb <= 12:
            return "Noticeably smarter"
        if self.min_ram_gb <= 16:
            return "Strong answers"
        return "Best quality, big machine"

    @property
    def label(self) -> str:
        return f"{self.display_name}  ·  {self.size_label}  ·  {self.recommended_for}"

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "name": self.display_name,
            "size_b": self.size_b,
            "download_gb": self.download_gb,
            "min_ram_gb": self.min_ram_gb,
            "quality": self.quality,
            "blurb": self.blurb,
            "context": self.context,
            "license": self.license,
            "filename": self.filename,
            "url": self.url,
            "train_repo": self.train_repo,
            "size_label": self.size_label,
            "recommended_for": self.recommended_for,
        }


def _spec(size: float, download_gb: float, min_ram: int, quality: int, blurb: str,
          context: int = 8192) -> ModelSpec:
    key = f"qwen2.5-{size:g}b"
    return ModelSpec(
        key=key,
        size_b=size,
        download_gb=download_gb,
        min_ram_gb=min_ram,
        quality=quality,
        blurb=blurb,
        context=context,
        filename=f"qwen2.5-{size:g}b-instruct-q4_k_m.gguf",
        url=BASE_URL.format(size=f"{size:g}"),
        train_repo=f"Qwen/Qwen2.5-{size:g}B-Instruct",
    )


CATALOGUE: dict[str, ModelSpec] = {
    spec.key: spec
    for spec in (
        _spec(0.5, 0.4, 4, 2,
              "Runs on almost anything. Good for quick drafts and simple questions."),
        _spec(1.5, 1.0, 8, 3,
              "The default. Fast on a normal laptop and solid for everyday use."),
        _spec(3, 2.0, 12, 4,
              "A clear step up in reasoning and writing, still laptop-friendly."),
        _spec(7, 4.7, 16, 4,
              "Strong local assistant for 16 GB machines. Slower on CPU."),
        _spec(14, 9.0, 32, 5,
              "Best quality in this list. Needs 32 GB of RAM and patience."),
    )
}


def catalogue() -> list[ModelSpec]:
    return list(CATALOGUE.values())


def get_spec(key: str) -> ModelSpec:
    try:
        return CATALOGUE[key]
    except KeyError:
        known = ", ".join(CATALOGUE)
        raise KeyError(f"unknown model {key!r}; available: {known}") from None


def recommend_key(ram_gb: float | None = None) -> str:
    ram = ram_gb if ram_gb is not None else total_ram_gb()
    if not ram:
        return DEFAULT_MODEL
    best = "qwen2.5-0.5b"
    for spec in CATALOGUE.values():
        if spec.min_ram_gb <= ram:
            best = spec.key
    return best


@dataclass
class InstalledModel:
    key: str
    path: Path
    size_bytes: int
    spec: ModelSpec | None

    @property
    def name(self) -> str:
        return self.spec.display_name if self.spec else self.path.name

    @property
    def size_label(self) -> str:
        return human_bytes(self.size_bytes)


class ModelManager:
    """Locate, inspect and download the GGUF weights."""

    def __init__(self, root: str | os.PathLike[str] | None = None) -> None:
        self.paths: Paths = paths(root)
        self.allow_download = False  # set by the installer after consent

    # -- paths -----------------------------------------------------------
    def path_for(self, key: str) -> Path:
        return self.paths.models / get_spec(key).filename

    def exists(self, key: str) -> bool:
        path = self.path_for(key)
        try:
            return path.is_file() and path.stat().st_size > 1024 * 1024
        except OSError:
            return False

    def is_complete(self, key: str) -> bool:
        if not self.exists(key):
            return False
        state = read_json(self.paths.models / (get_spec(key).filename + ".part.state.json"), {})
        if isinstance(state, dict) and state.get("status") == "complete":
            total = state.get("total_bytes")
            try:
                if total and path_size(self.path_for(key)) != int(total):
                    return False
            except (TypeError, ValueError):
                pass
        return True

    def installed(self) -> list[InstalledModel]:
        found: list[InstalledModel] = []
        if not self.paths.models.is_dir():
            return found
        for path in sorted(self.paths.models.glob("*.gguf")):
            key = next(
                (k for k, spec in CATALOGUE.items() if spec.filename == path.name),
                None,
            )
            try:
                size = path.stat().st_size
            except OSError:
                continue
            if size < 1024 * 1024 or path.name.endswith(".part"):
                continue
            found.append(
                InstalledModel(
                    key=key or path.stem,
                    path=path,
                    size_bytes=size,
                    spec=CATALOGUE.get(key) if key else None,
                )
            )
        return found

    def select(self, preferred: str | None = None) -> tuple[str, Path] | None:
        """Return the best available model (preferred, else any installed)."""
        if preferred and self.is_complete(preferred):
            return preferred, self.path_for(preferred)
        for item in self.installed():
            if item.key in CATALOGUE:
                return item.key, item.path
        return None

    def default_path(self) -> Path | None:
        selected = self.select()
        return selected[1] if selected else None

    def partial(self, key: str) -> dict:
        return partial_info(self.path_for(key))

    def disk_free_gb(self) -> float:
        try:
            return round(shutil.disk_usage(str(self.paths.root)).free / 1024**3, 1)
        except OSError:
            return 0.0

    # -- network ---------------------------------------------------------
    def download(
        self,
        key: str,
        progress: Callable[[Progress], None] | None = None,
        *,
        permission: bool | None = None,
    ) -> DownloadResult:
        allowed = self.allow_download if permission is None else permission
        if not allowed:
            raise PermissionError(
                "Model downloads are off. Enable 'Download local models' in "
                f"{__app_name__} Settings first."
            )
        spec = get_spec(key)
        dest = self.path_for(key)
        if self.is_complete(key):
            return DownloadResult(
                path=dest,
                bytes_written=path_size(dest),
                total_bytes=path_size(dest),
                resumed=True,
                sha256=None,
                attempts=0,
            )
        free = self.disk_free_gb()
        if free and free < spec.download_gb * 1.15 + 0.3:
            raise OSError(
                f"Not enough disk space for {spec.display_name}: "
                f"{spec.size_label} needed, {free} GB free."
            )
        return download(spec.url, dest, progress=progress, expected_sha256=None)

    def remove(self, key: str) -> bool:
        path = self.path_for(key)
        removed = False
        for candidate in (path, path.with_name(path.name + ".part"),
                          path.with_name(path.name + ".part.state.json")):
            try:
                if candidate.exists():
                    candidate.unlink()
                    removed = True
            except OSError:
                pass
        return removed

    def status(self) -> dict:
        installed = [
            {
                "key": item.key,
                "name": item.name,
                "path": str(item.path),
                "size": item.size_label,
                "size_bytes": item.size_bytes,
            }
            for item in self.installed()
        ]
        return {
            "catalogue": [spec.to_dict() for spec in catalogue()],
            "installed": installed,
            "default": DEFAULT_MODEL,
            "recommended": recommend_key(),
            "ram_gb": total_ram_gb(),
            "platform": platform.system(),
            "models_dir": str(self.paths.models),
            "disk_free_gb": self.disk_free_gb(),
        }


def path_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0

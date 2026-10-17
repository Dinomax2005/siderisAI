"""Resumable HTTP downloads.

Model weights are large (hundreds of megabytes to tens of gigabytes) and
home connections drop.  A download therefore never goes straight to its
final path:

  <dest>.part              the bytes received so far
  <dest>.part.state.json   url, etag, total size, attempts, timestamp

The byte count on disk - not the JSON - is the source of truth when
resuming, so a hard power loss can lose at most the OS dirty page cache,
never the whole file.  On restart we send ``Range: bytes=<size>-`` and
expect ``206 Partial Content``; if a server or CDN answers ``200`` we
start over cleanly instead of appending garbage.  SHA-256 is checked when
a digest is known, and the finished file is moved into place atomically.
"""

from __future__ import annotations

import hashlib
import http.client
import os
import re
import shutil
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .config import paths, read_json, utc_now, write_json

CHUNK_SIZE = 1024 * 256
PROGRESS_INTERVAL = 0.2
USER_AGENT = "Sideris-Desktop/5.0 (+local model downloader)"
CONTENT_RANGE_RE = re.compile(r"bytes\s+(\d+)-(\d+)/(\d+|\*)")


class DownloadError(RuntimeError):
    """A download failed in a way the user should be told about."""


class DiskSpaceError(DownloadError):
    """There is not enough free disk space to finish the download."""


@dataclass
class DownloadResult:
    path: Path
    bytes_written: int
    total_bytes: int | None
    resumed: bool
    sha256: str | None
    attempts: int = 1
    elapsed: float = 0.0


@dataclass
class Progress:
    path: Path
    downloaded: int
    total: int | None
    resumed: bool = False
    speed: float = 0.0
    message: str = ""

    @property
    def percent(self) -> float | None:
        if not self.total:
            return None
        return max(0.0, min(100.0, self.downloaded * 100.0 / self.total))


ProgressCallback = Callable[[Progress], None]


def part_path(dest: Path) -> Path:
    return dest.with_name(dest.name + ".part")


def state_path(dest: Path) -> Path:
    return dest.with_name(dest.name + ".part.state.json")


def human_bytes(value: float | int | None) -> str:
    if value is None:
        return "?"
    value = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _cleanup_empty(part: Path) -> None:
    try:
        if part.exists():
            part.unlink()
    except OSError:
        pass


def _free_bytes(path: Path) -> int:
    try:
        return shutil.disk_usage(str(path)).free
    except OSError:
        return 0


def _write_state(dest: Path, **payload: object) -> None:
    state = {
        "url": payload.get("url"),
        "etag": payload.get("etag"),
        "last_modified": payload.get("last_modified"),
        "total_bytes": payload.get("total_bytes"),
        "downloaded_bytes": payload.get("downloaded_bytes", 0),
        "updated_at": utc_now(),
        "attempts": payload.get("attempts", 1),
        "status": payload.get("status", "in-progress"),
    }
    try:
        write_json(state_path(dest), state)
    except OSError:
        # Progress bookkeeping is an optimisation; the .part file itself
        # is enough to resume, so never fail a download over it.
        pass


def _check_disk_space(part: Path, remaining: int | None) -> None:
    if remaining is None or remaining <= 0:
        return
    free = _free_bytes(part.parent)
    reserve = 128 * 1024 * 1024
    if free and free < remaining + reserve:
        raise DiskSpaceError(
            "Not enough free disk space: need about "
            f"{human_bytes(remaining + reserve)} more, "
            f"{human_bytes(free)} available."
        )


def download(
    url: str,
    dest: str | os.PathLike[str],
    *,
    expected_sha256: str | None = None,
    progress: ProgressCallback | None = None,
    timeout: float = 60.0,
    stall_timeout: float = 180.0,
    max_attempts: int = 3,
) -> DownloadResult:
    """Download ``url`` to ``dest``, resuming a previous attempt if any.

    Raises :class:`DownloadError` (or :class:`DiskSpaceError`) with a
    message meant to be shown directly to a person.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = part_path(dest)
    started = time.time()

    previous = read_json(state_path(dest), {}) or {}
    existing = _file_size(part)

    # A previous attempt for a different URL must not be mixed in.
    if previous.get("url") and previous.get("url") != url and existing:
        _cleanup_empty(part)
        existing = 0
    if existing and previous.get("total_bytes"):
        try:
            if existing > int(previous["total_bytes"]):
                _cleanup_empty(part)
                existing = 0
        except (TypeError, ValueError):
            pass

    resumed = existing > 0
    total: int | None = None
    attempts = int(previous.get("attempts") or 1)

    if progress:
        progress(
            Progress(
                path=dest,
                downloaded=existing,
                total=previous.get("total_bytes") if isinstance(previous.get("total_bytes"), int) else None,
                resumed=resumed,
                message="Resuming download" if resumed else "Starting download",
            )
        )

    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            total, resumed_final = _attempt(
                url,
                dest,
                part,
                existing=existing,
                expected_sha256=expected_sha256,
                progress=progress,
                timeout=timeout,
                stall_timeout=stall_timeout,
                attempts=attempts + attempt - 1,
                previous=previous,
            )
            resumed = resumed_final
            break
        except (urllib.error.URLError, socket.timeout, TimeoutError,
                ConnectionError, http.client.HTTPException, OSError) as exc:
            last_error = exc
            existing = _file_size(part)
            if attempt >= max_attempts:
                raise DownloadError(
                    f"Download failed after {attempt} attempts: {exc}. "
                    f"{human_bytes(existing)} kept - run setup again to resume."
                ) from exc
            time.sleep(min(2.0 * attempt, 6.0))
        except DownloadError:
            raise

    digest = sha256_file(dest) if expected_sha256 else None
    if expected_sha256 and digest and digest.lower() != expected_sha256.lower():
        raise DownloadError(
            "Checksum mismatch: the downloaded file is corrupt and was discarded."
        )

    if progress:
        final_total = total or _file_size(dest)
        progress(
            Progress(
                path=dest,
                downloaded=_file_size(dest),
                total=final_total,
                resumed=resumed,
                message="Download complete",
            )
        )

    return DownloadResult(
        path=dest,
        bytes_written=_file_size(dest),
        total_bytes=total,
        resumed=resumed,
        sha256=digest,
        attempts=attempts,
        elapsed=round(time.time() - started, 2),
    )


def _attempt(
    url: str,
    dest: Path,
    part: Path,
    *,
    existing: int,
    expected_sha256: str | None,
    progress: ProgressCallback | None,
    timeout: float,
    stall_timeout: float,
    attempts: int,
    previous: dict,
) -> tuple[int | None, bool]:
    """One HTTP attempt.

    Returns ``(total_size, resumed)`` where ``resumed`` says whether the
    final file was appended to an existing ``.part`` (206) or restarted
    cleanly because the server ignored the Range header (200).
    """
    headers = {
        "User-Agent": USER_AGENT,
        "Accept-Encoding": "identity",
    }
    if existing > 0:
        headers["Range"] = f"bytes={existing}-"
        # If the file changed remotely we want a fresh 200, not a mixed file.
        validator = previous.get("etag") or previous.get("last_modified")
        if validator:
            headers["If-Range"] = str(validator)

    request = urllib.request.Request(url, headers=headers, method="GET")
    try:
        response = urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and existing:
            # Range not satisfiable: either already complete or stale.
            _cleanup_empty(part)
            existing = 0
            raise DownloadError(
                "The partial file was rejected by the server and has been reset."
            ) from exc
        raise

    with response:
        status = getattr(response, "status", 200) or 200
        response_headers = response.headers
        etag = response_headers.get("ETag")
        last_modified = response_headers.get("Last-Modified")
        encoding = (response_headers.get("Content-Encoding") or "identity").lower()
        if encoding not in ("", "identity"):
            raise DownloadError(
                f"Server used unsupported content encoding {encoding!r}."
            )

        if existing and status == 206:
            content_range = response_headers.get("Content-Range") or ""
            match = CONTENT_RANGE_RE.match(content_range.strip())
            if not match or int(match.group(1)) != existing:
                _cleanup_empty(part)
                raise DownloadError(
                    "Server returned an unexpected byte range; restarting cleanly."
                )
            total = None if match.group(3) == "*" else int(match.group(3))
            mode = "ab"
            resumed = True
        elif existing and status == 200:
            # Server ignored or rejected the Range request.  Starting over is
            # the only correct move; appending would corrupt the file.
            _cleanup_empty(part)
            existing = 0
            mode = "wb"
            resumed = False
            length = response_headers.get("Content-Length")
            total = int(length) if length and length.isdigit() else None
        else:
            mode = "wb"
            existing = 0
            resumed = False
            content_range = response_headers.get("Content-Range") or ""
            match = CONTENT_RANGE_RE.match(content_range.strip())
            length = response_headers.get("Content-Length")
            if match:
                total = None if match.group(3) == "*" else int(match.group(3))
            elif length and length.isdigit():
                total = int(length)
            else:
                total = None

        _check_disk_space(part, remaining=(total - existing) if total else None)
        _write_state(
            dest,
            url=url,
            etag=etag,
            last_modified=last_modified,
            total_bytes=total,
            downloaded_bytes=existing,
            attempts=attempts,
            status="in-progress",
        )

        downloaded = existing
        last_progress = 0.0
        speed_window = (time.time(), downloaded)

        with open(part, mode) as fh:
            while True:
                try:
                    block = response.read(CHUNK_SIZE)
                except (socket.timeout, TimeoutError) as exc:
                    raise DownloadError(
                        f"Network stalled after {human_bytes(downloaded)}; "
                        "run setup again to resume."
                    ) from exc
                if not block:
                    break
                fh.write(block)
                downloaded += len(block)
                now = time.time()
                if progress and now - last_progress >= PROGRESS_INTERVAL:
                    last_progress = now
                    window_start, window_bytes = speed_window
                    elapsed = max(now - window_start, 1e-6)
                    speed = max((downloaded - window_bytes) / elapsed, 0.0)
                    if now - window_start > 5:
                        speed_window = (now, downloaded)
                    progress(
                        Progress(
                            path=dest,
                            downloaded=downloaded,
                            total=total,
                            resumed=resumed,
                            speed=speed,
                        )
                    )
            fh.flush()
            try:
                os.fsync(fh.fileno())
            except OSError:
                pass

        if total is not None and downloaded != total:
            # Keep whatever arrived; a later run resumes from here.
            _write_state(
                dest,
                url=url,
                etag=etag,
                last_modified=last_modified,
                total_bytes=total,
                downloaded_bytes=downloaded,
                attempts=attempts,
                status="incomplete",
            )
            raise DownloadError(
                f"Connection closed early at {human_bytes(downloaded)} of "
                f"{human_bytes(total)}. Progress was kept - run setup again to resume."
            )

        digest = sha256_file(part) if expected_sha256 else None
        if expected_sha256 and digest and digest.lower() != expected_sha256.lower():
            _cleanup_empty(part)
            raise DownloadError(
                "Checksum mismatch: the download was corrupt and has been removed."
            )

        os.replace(part, dest)
        _write_state(
            dest,
            url=url,
            etag=etag,
            last_modified=last_modified,
            total_bytes=total,
            downloaded_bytes=downloaded,
            attempts=attempts,
            status="complete",
        )
        return total, bool(existing > 0 and resumed)


def partial_info(dest: str | os.PathLike[str]) -> dict:
    """Describe an interrupted download for progress displays."""
    dest = Path(dest)
    state = read_json(state_path(dest), {}) or {}
    existing = _file_size(part_path(dest))
    total = state.get("total_bytes")
    return {
        "partial_bytes": existing,
        "total_bytes": total if isinstance(total, int) else None,
        "resumable": existing > 0 and not dest.exists(),
        "state": state,
        "part": str(part_path(dest)),
    }


# ``paths`` is imported for callers that want the canonical model folder;
# keeping the import documents the intentional dependency direction.
_ = paths

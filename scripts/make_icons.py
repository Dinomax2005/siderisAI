#!/usr/bin/env python3
"""Derive Sideris icon files from the original assets/logo.png.

The original artwork is never redrawn or regenerated: every icon is a
resampled copy of the source image.

Outputs
-------
assets/logo-256.png   256px transparent PNG (window icon / docs)
assets/sideris.ico    16, 24, 32, 48, 64, 128, 256 px Windows icon
assets/sideris.icns   macOS icon (16..1024 px, PNG based)
"""
from __future__ import annotations

import struct
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "assets" / "logo.png"

ICO_SIZES = (16, 24, 32, 48, 64, 128, 256)
ICNS_PNG_SIZES = (16, 32, 64, 128, 256, 512, 1024)


def _square(img: Image.Image, pad_ratio: float = 0.04) -> Image.Image:
    """Center the artwork on a transparent square canvas."""
    bbox = img.getchannel("A").getbbox()
    art = img.crop(bbox) if bbox else img
    side = max(art.size)
    side = int(round(side * (1 + pad_ratio * 2)))
    canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    canvas.paste(art, ((side - art.width) // 2, (side - art.height) // 2), art)
    return canvas


def _resize(img: Image.Image, size: int) -> Image.Image:
    return img.resize((size, size), Image.Resampling.LANCZOS)


def _icns_entry(ostype: bytes, png: bytes) -> bytes:
    return ostype + struct.pack(">I", len(png) + 8) + png


def build() -> None:
    src = Image.open(SRC).convert("RGBA")
    base = _square(src)

    out_dir = ROOT / "assets"
    _resize(base, 256).save(out_dir / "logo-256.png")

    frames = [_resize(base, s) for s in ICO_SIZES]
    frames[-1].save(
        out_dir / "sideris.ico",
        format="ICO",
        sizes=[(s, s) for s in ICO_SIZES],
        append_images=frames[:-1],
    )

    # ICNS is a container of PNG chunks. 'ic10' is 1024px, 'ic09' 512,
    # 'ic08' 256, 'ic07' 128, 'icp6' 64, 'icp5' 32, 'icp4' 16.
    chunks = []
    mapping = {
        16: b"icp4",
        32: b"icp5",
        64: b"icp6",
        128: b"ic07",
        256: b"ic08",
        512: b"ic09",
        1024: b"ic10",
    }
    import io

    for size in ICNS_PNG_SIZES:
        buf = io.BytesIO()
        _resize(base, size).save(buf, format="PNG")
        chunks.append(_icns_entry(mapping[size], buf.getvalue()))
    body = b"".join(chunks)
    (out_dir / "sideris.icns").write_bytes(
        b"icns" + struct.pack(">I", len(body) + 8) + body
    )

    print("wrote", out_dir / "sideris.ico", (out_dir / "sideris.ico").stat().st_size)
    print("wrote", out_dir / "sideris.icns", (out_dir / "sideris.icns").stat().st_size)
    print("wrote", out_dir / "logo-256.png")


if __name__ == "__main__":
    build()

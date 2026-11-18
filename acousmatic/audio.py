"""Decoding. Any format ffmpeg reads, to float32 stereo at a fixed rate."""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SR_HI = 44100   # descriptors, loudness, stereo field
SR = 22050      # stems, rhythm, harmony, pitch


@dataclass
class Audio:
    path: Path
    stereo: np.ndarray      # (2, n) at SR_HI
    sr: int
    sha256: str
    tags: dict

    @property
    def duration(self) -> float:
        return self.stereo.shape[1] / self.sr


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tags(path: Path) -> dict:
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format_tags=title,artist,album,date,track",
                              "-of", "json", str(path)], capture_output=True, text=True, timeout=30).stdout
        t = json.loads(out).get("format", {}).get("tags", {})
        return {k.lower(): v for k, v in t.items()}
    except (subprocess.SubprocessError, ValueError):
        return {}


def decode(path: Path, sr: int = SR_HI) -> np.ndarray:
    proc = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-f", "f32le", "-ac", "2",
                           "-ar", str(sr), "-"], capture_output=True, timeout=600)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg couldn't decode {path}: {proc.stderr.decode(errors='replace')[:400]}")
    y = np.frombuffer(proc.stdout, dtype=np.float32)
    return y.reshape(-1, 2).T.copy()


def load(path) -> Audio:
    path = Path(path)
    return Audio(path=path, stereo=decode(path), sr=SR_HI, sha256=sha256(path), tags=tags(path))

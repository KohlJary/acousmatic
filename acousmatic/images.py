"""Pictures of the sound: spectrograms by section and stem, and the stereo field."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

from .audio import SR

plt.rcParams.update({"font.size": 9})


def _mel_db(y, sr=SR):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128, fmax=sr / 2)
    return librosa.power_to_db(S, ref=np.max)


def overview(mix, sections, bars, path: Path, title: str):
    fig, ax = plt.subplots(figsize=(14, 4), dpi=100)
    librosa.display.specshow(_mel_db(mix), sr=SR, hop_length=512, x_axis="time", y_axis="mel", ax=ax, cmap="magma")
    for s in sections:
        t = bars[s["start_bar"] - 1]["start"]
        ax.axvline(t, color="white", lw=1)
        ax.text(t + 0.5, ax.get_ylim()[1] * 0.8, s["label"], color="white", fontsize=12, weight="bold")
    ax.set_title(title)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def section_stems(stems: dict, t0, t1, path: Path, title: str):
    names = ["lead", "bass", "drums", "other"]
    fig, axes = plt.subplots(len(names), 1, figsize=(12, 9), dpi=90, sharex=True)
    a, b = int(t0 * SR), int(t1 * SR)
    for ax, n in zip(axes, names):
        seg = stems[n][a:b]
        if len(seg) < 2048:
            continue
        librosa.display.specshow(_mel_db(seg), sr=SR, hop_length=512, x_axis="time", y_axis="mel", ax=ax, cmap="magma")
        ax.set_ylabel(n)
        ax.label_outer()
    axes[0].set_title(title)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def stereo_field(field, sections, bars, path: Path, title: str):
    """Pan (left blue, right red) per frequency band over time."""
    edges = np.geomspace(40, 18000, 25)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        r = (field.f >= lo) & (field.f < hi)
        l, rr = field.L[r].sum(axis=0), field.R[r].sum(axis=0)
        rows.append((rr - l) / (rr + l + 1e-12))
    P = np.array(rows)
    k = 8
    P = P[:, : P.shape[1] // k * k].reshape(P.shape[0], -1, k).mean(axis=2)
    fig, ax = plt.subplots(figsize=(14, 4), dpi=100)
    ax.imshow(P, aspect="auto", origin="lower", cmap="coolwarm", vmin=-0.6, vmax=0.6,
              extent=[0, field.times[-1], 0, len(edges) - 1])
    ax.set_yticks(range(0, len(edges), 4), [f"{int(e)}" for e in edges[::4]])
    ax.set_ylabel("Hz"); ax.set_xlabel("time (s)")
    for s in sections:
        t = bars[s["start_bar"] - 1]["start"]
        ax.axvline(t, color="black", lw=1)
    ax.set_title(title + "  (blue = left, red = right, white = centre)")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)

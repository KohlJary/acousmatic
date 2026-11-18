"""Pulling the mix apart with signal processing only.

- The stereo field: mid (L+R)/2 holds what's panned centre (usually lead vocal, bass,
  kick, snare); side (L-R)/2 holds what's spread or panned.
- Harmonic/percussive separation: median filtering of the mid spectrogram along time
  (sustained tones) and along frequency (hits).
- Bass: the harmonic mid below a crossover.
- Lead (vocal or lead line): the harmonic, centre-panned, non-repeating part of the
  mid in the voice band. Non-repeating comes from REPET-SIM: each frame's background is
  the median of the frames most like it, and what stands out from that is the lead.
- Other: what's left of the harmonic content, mid and side.

These are approximations. On a dense mix the stems bleed into one another; the record
says how much (`bleed`).
"""
from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np

from .audio import SR, SR_HI

N_FFT = 2048
HOP = 512
BASS_HZ = 220.0
LEAD_BAND = (150.0, 8000.0)


@dataclass
class Stems:
    sr: int
    mid: np.ndarray
    side: np.ndarray
    lead: np.ndarray
    bass: np.ndarray
    drums: np.ndarray
    other: np.ndarray
    bleed: dict

    def as_dict(self):
        return {"lead": self.lead, "bass": self.bass, "drums": self.drums, "other": self.other}


def _smoothstep(f, lo, hi):
    x = np.clip((f - lo) / max(hi - lo, 1e-9), 0, 1)
    return x * x * (3 - 2 * x)


def separate(stereo_hi: np.ndarray) -> Stems:
    stereo = librosa.resample(stereo_hi, orig_sr=SR_HI, target_sr=SR, res_type="soxr_hq")
    mid = (stereo[0] + stereo[1]) / 2
    side = (stereo[0] - stereo[1]) / 2
    M = librosa.stft(mid, n_fft=N_FFT, hop_length=HOP)
    S = librosa.stft(side, n_fft=N_FFT, hop_length=HOP)
    freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)[:, None]

    Hm, _ = librosa.decompose.hpss(np.abs(M), mask=True, margin=1.0)
    _, Pm = librosa.decompose.hpss(np.abs(M), mask=True, margin=(1.0, 2.5))
    Hs, Ps = librosa.decompose.hpss(np.abs(S), mask=True, margin=(1.0, 2.5))

    # bass: harmonic mid below the crossover
    low = 1.0 - _smoothstep(freqs, BASS_HZ * 0.8, BASS_HZ * 1.25)
    bass_mask = Hm * low

    # lead: harmonic, centre, non-repeating, in the voice band
    Mh = np.abs(M) * Hm
    width = int(librosa.time_to_frames(2.0, sr=SR, hop_length=HOP))
    # a very short piece (a few seconds) can't hold a 2-second window on each side
    width = max(1, min(width, (Mh.shape[-1] - 1) // 2 - 1))
    bg = librosa.decompose.nn_filter(Mh, aggregate=np.median, metric="cosine", width=width)
    bg = np.minimum(Mh, bg)
    fg_mask = librosa.util.softmask(Mh - bg, 2.0 * bg, power=2)
    centre = np.abs(M) ** 2 / (np.abs(M) ** 2 + np.abs(S) ** 2 + 1e-12)
    band = _smoothstep(freqs, LEAD_BAND[0] * 0.8, LEAD_BAND[0]) * (1 - _smoothstep(freqs, LEAD_BAND[1], LEAD_BAND[1] * 1.3))
    lead_mask = Hm * fg_mask * centre * band * (1 - low)

    other_mask = np.clip(Hm - bass_mask - lead_mask, 0, 1)

    n = len(mid)
    inv = lambda X: librosa.istft(X, hop_length=HOP, length=n)
    lead = inv(M * lead_mask)
    bass = inv(M * bass_mask)
    drums = inv(M * Pm) + inv(S * Ps)
    other = inv(M * other_mask) + inv(S * Hs)

    def share(mask, X):
        return float((np.abs(X) ** 2 * mask ** 2).sum())
    total = float((np.abs(M) ** 2).sum() + (np.abs(S) ** 2).sum()) + 1e-12
    # bleed: how much of each stem's energy sits where another stem's mask is also strong
    def overlap(a, b):
        return float(np.minimum(a, b).mean() / (a.mean() + 1e-12))
    bleed = {
        "lead_into_other": round(overlap(lead_mask, other_mask), 3),
        "energy_share": {
            "lead": round(share(lead_mask, M) / total, 3),
            "bass": round(share(bass_mask, M) / total, 3),
            "drums": round((share(Pm, M) + share(Ps, S)) / total, 3),
            "other": round((share(other_mask, M) + share(Hs, S)) / total, 3),
        },
    }
    return Stems(sr=SR, mid=mid, side=side, lead=lead, bass=bass, drums=drums, other=other, bleed=bleed)

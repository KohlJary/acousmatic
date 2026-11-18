"""How it sounds: loudness, spectrum, texture and the stereo field, per section and stem.

All measurements; the plain-language lines at the end are rules over them (a change of
3 dB or more is "louder", a centroid 25% higher is "brighter"), so they say no more
than the numbers do.
"""
from __future__ import annotations

import librosa
import numpy as np
import pyloudnorm

from .audio import SR, SR_HI

BANDS = {"sub": (20, 60), "bass": (60, 250), "low_mid": (250, 500), "mid": (500, 2000),
         "presence": (2000, 6000), "air": (6000, 20000)}
N_FFT_HI = 4096
HOP_HI = 1024


def _db(x):
    return float(10 * np.log10(x + 1e-12))


def loudness(stereo_hi):
    meter = pyloudnorm.Meter(SR_HI)
    try:
        lufs = float(meter.integrated_loudness(stereo_hi.T))
    except ValueError:
        lufs = float("nan")
    peak = float(np.abs(stereo_hi).max())
    rms = float(np.sqrt(np.mean(stereo_hi ** 2)))
    return {"integrated_lufs": round(lufs, 1), "peak_dbfs": round(20 * np.log10(peak + 1e-12), 1),
            "crest_db": round(20 * np.log10((peak + 1e-12) / (rms + 1e-12)), 1)}


class Field:
    """Spectra of the full-rate mix, kept once for per-section reads."""

    def __init__(self, stereo_hi):
        L = np.abs(librosa.stft(stereo_hi[0], n_fft=N_FFT_HI, hop_length=HOP_HI)) ** 2
        R = np.abs(librosa.stft(stereo_hi[1], n_fft=N_FFT_HI, hop_length=HOP_HI)) ** 2
        M = np.abs(librosa.stft((stereo_hi[0] + stereo_hi[1]) / 2, n_fft=N_FFT_HI, hop_length=HOP_HI)) ** 2
        S = np.abs(librosa.stft((stereo_hi[0] - stereo_hi[1]) / 2, n_fft=N_FFT_HI, hop_length=HOP_HI)) ** 2
        self.L, self.R, self.M, self.S = L, R, M, S
        self.f = librosa.fft_frequencies(sr=SR_HI, n_fft=N_FFT_HI)
        self.times = librosa.frames_to_time(np.arange(M.shape[1]), sr=SR_HI, hop_length=HOP_HI)
        self.mean_power = float((M.sum() + S.sum()) / max(M.shape[1], 1))

    def cols(self, t0, t1):
        return (self.times >= t0) & (self.times < t1)

    def section(self, t0, t1):
        c = self.cols(t0, t1)
        if not c.any():
            return None
        M, S, L, R = self.M[:, c], self.S[:, c], self.L[:, c], self.R[:, c]
        total = M.sum() + S.sum() + 1e-12
        out = {"level_db": round(_db((M.sum() + S.sum()) / c.sum()) - _db(self.mean_power), 1), "bands": {}, "width": {}, "pan": {}}
        for name, (lo, hi) in BANDS.items():
            r = (self.f >= lo) & (self.f < hi)
            m, s = M[r].sum(), S[r].sum()
            out["bands"][name] = round(float((m + s) / total), 3)
            out["width"][name] = round(float(s / (m + s + 1e-12)), 3)
            l, rr = L[r].sum(), R[r].sum()
            out["pan"][name] = round(float((rr - l) / (rr + l + 1e-12)), 3)
        spec = (M + S).mean(axis=1)
        out["centroid_hz"] = int(round(float((self.f * spec).sum() / (spec.sum() + 1e-12))))
        out["flatness"] = round(float(librosa.feature.spectral_flatness(S=np.sqrt(M + S)).mean()), 3)
        out["width_overall"] = round(float(S.sum() / total), 3)
        return out


def stem_section(y, t0, t1, ref_rms):
    a, b = int(t0 * SR), int(t1 * SR)
    seg = y[a:b]
    if len(seg) < 2048:
        return None
    rms = float(np.sqrt(np.mean(seg ** 2)))
    S = np.abs(librosa.stft(seg, n_fft=2048, hop_length=512))
    return {"level_db_rel": round(20 * np.log10((rms + 1e-9) / (ref_rms + 1e-9)), 1),
            "centroid_hz": int(round(float(librosa.feature.spectral_centroid(S=S, sr=SR).mean()))),
            "flatness": round(float(librosa.feature.spectral_flatness(S=S).mean()), 3)}


def character(stem, d):
    """A few plain words for a stem's sound in a section, from its numbers."""
    if d is None:
        return ""
    words = []
    if d["level_db_rel"] < -30:
        return "absent or nearly"
    if d["level_db_rel"] < -18:
        words.append("faint")
    if d["flatness"] > 0.25:
        words.append("noisy/distorted")
    elif d["flatness"] < 0.03:
        words.append("clean, tonal")
    c = d["centroid_hz"]
    if stem in ("lead", "other"):
        words.append("bright" if c > 3000 else "dark" if c < 900 else "")
    return ", ".join(w for w in words if w)


def changes(prev, cur, stems_prev, stems_cur):
    """Plain lines about what changed from the previous section to this one."""
    if prev is None:
        return []
    lines = []
    d = cur["level_db"] - prev["level_db"]
    if d >= 3:
        lines.append(f"louder by {d:.0f} dB")
    elif d <= -3:
        lines.append(f"quieter by {-d:.0f} dB")
    r = cur["centroid_hz"] / max(prev["centroid_hz"], 1)
    if r >= 1.25:
        lines.append("brighter")
    elif r <= 0.8:
        lines.append("darker")
    w = cur["width_overall"] - prev["width_overall"]
    if w >= 0.08:
        lines.append("wider in the stereo field")
    elif w <= -0.08:
        lines.append("narrower, more centred")
    for stem in ("lead", "bass", "drums", "other"):
        a, b = (stems_prev or {}).get(stem), (stems_cur or {}).get(stem)
        if not a or not b:
            continue
        if a["level_db_rel"] < -24 <= b["level_db_rel"]:
            lines.append(f"{stem} comes in")
        elif b["level_db_rel"] < -24 <= a["level_db_rel"]:
            lines.append(f"{stem} drops out")
        elif b["level_db_rel"] - a["level_db_rel"] >= 6:
            lines.append(f"{stem} steps forward")
        elif a["level_db_rel"] - b["level_db_rel"] >= 6:
            lines.append(f"{stem} pulls back")
    return lines

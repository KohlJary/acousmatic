"""Which settings made this 808 kick.

The TR-808 bass drum is a bridged-T resonator struck by a pulse. Its front-panel knobs
are Tone and Decay (and Level); its pitch is set inside the machine, near 50 Hz. So
a kick is read as:
- Decay: how fast it dies away, from the slope of its envelope (not where the file
  ends: samples are often trimmed).
- Tone: how much click there is in the first few milliseconds (the Tone knob opens a
  filter on the attack).
- Re-tuning: its pitch against a stock 808's ~51 Hz, in semitones (modern 808s are
  usually re-pitched to the song's key).
- Saturation: how much of the body is in the overtones (a stock 808 is nearly a sine).
- Glide: the pitch at the start against the pitch it settles to.
Knob positions come from calibration against Michael Fischer's TR-808 set (1994),
sampled from a real unit at knob positions 0, 2.5, 5, 7.5 and 10. No model of any
kind is learned: measurements, and a monotone mapping read off those samples.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import soundfile as sf

STOCK_F0 = 51.5
CAL_PATH = Path(__file__).with_name("drum808_calibration.json")


def _mono(path):
    import subprocess
    y, sr = None, None
    try:
        y, sr = sf.read(str(path), always_2d=True)
        y = y.mean(axis=1)
    except Exception:
        from .audio import decode
        st = decode(Path(path), 44100)
        y, sr = st.mean(axis=0), 44100
    return y.astype(np.float64), sr


def _envelope_db(y, sr, hop=64, win=256):
    n = 1 + max(0, len(y) - win) // hop
    env = np.array([np.sqrt(np.mean(y[i * hop:i * hop + win] ** 2)) for i in range(n)])
    return 20 * np.log10(env + 1e-12), hop / sr


def _f0(seg, sr, fmin=25.0, fmax=200.0):
    """Pitch by autocorrelation (a kick's body is near-sinusoidal)."""
    if len(seg) < int(sr / fmin) * 3:
        return None
    seg = seg - seg.mean()
    ac = np.correlate(seg, seg, mode="full")[len(seg) - 1:]
    lo, hi = int(sr / fmax), int(sr / fmin)
    if hi >= len(ac):
        return None
    k = lo + int(np.argmax(ac[lo:hi]))
    if 0 < k < len(ac) - 1:   # parabolic refinement, kept only when it stays within a lag
        a, b, c = ac[k - 1], ac[k], ac[k + 1]
        d = 0.5 * (a - c) / (a - 2 * b + c + 1e-12)
        if abs(d) <= 1:
            k = k + d
    return sr / k if k > 0 and ac[int(round(k))] > 0 else None


def measure(y, sr) -> Dict[str, Optional[float]]:
    y = np.asarray(y, dtype=np.float64)
    y = y / (np.abs(y).max() + 1e-12)
    db, dt = _envelope_db(y, sr)
    pk = int(np.argmax(db))
    rel = db - db[pk]
    # decay: seconds to fall 40 dB, from the slope between -3 and -35 dB (trim-proof)
    after = rel[pk:]
    idx = np.where((after < -3) & (after > -35))[0]
    if len(idx) >= 3:
        t = idx * dt
        slope = np.polyfit(t, after[idx], 1)[0]
        decay40 = float(40.0 / -slope) if slope < 0 else None
    else:
        decay40 = None
    # pitch: settled (from 40 ms) and early (5-30 ms)
    start = pk * 64
    body = y[start + int(0.04 * sr): start + int(0.04 * sr) + int(0.12 * sr)]
    f_body = _f0(body, sr)
    early = y[start + int(0.005 * sr): start + int(0.035 * sr)]
    f_early = _f0(early, sr, fmin=30)
    # click: share of spectral magnitude above 300 Hz in the 20 ms from where the sound
    # starts (magnitude: in power the 51 Hz body swamps it). This reads the Tone knob the
    # same at every Decay on the reference set.
    on = int(np.argmax(np.abs(y) > 0.01))
    n20 = int(0.02 * sr)
    head = y[on:on + n20]
    head = head * np.hanning(len(head))
    S = np.abs(np.fft.rfft(head, n=8192))
    f = np.fft.rfftfreq(8192, 1 / sr)
    # the line scales with the kick's pitch (300 Hz on a stock 808 at ~51 Hz), so re-pitching
    # doesn't read as more or less click
    pitch_ok = bool(f_body) and decay40 is not None and decay40 >= 0.2   # too short to ring: no pitch to trust
    cut = 300.0 * ((f_body / STOCK_F0) if pitch_ok else 1.0)
    click = float(S[f > cut].sum() / (S.sum() + 1e-12))
    # saturation: energy at 2f, 3f against f in the body
    sat = None
    if f_body:
        B = np.abs(np.fft.rfft(body * np.hanning(len(body)), n=1 << 16)) ** 2
        fb = np.fft.rfftfreq(1 << 16, 1 / sr)
        def band(h):
            return B[(fb > h * f_body * 0.94) & (fb < h * f_body * 1.06)].sum()
        sat = float(10 * np.log10((band(2) + band(3) + 1e-12) / (band(1) + 1e-12)))
    return {"decay40_s": decay40, "click": round(click, 4),
            "f0_hz": round(f_body, 2) if f_body and pitch_ok else None,
            "f0_early_hz": round(f_early, 2) if f_early else None,
            "overtones_db": round(sat, 1) if sat is not None and pitch_ok else None,
            "length_s": round(len(y) / sr, 3)}


KNOBS = [0.0, 2.5, 5.0, 7.5, 10.0]


def calibrate(folder: Path, out: Path = CAL_PATH) -> Dict:
    """Read Fischer's BD files (BDttdd.WAV: tone, decay) and keep, per knob position,
    the median of what was measured."""
    code = {"00": 0.0, "25": 2.5, "50": 5.0, "75": 7.5, "10": 10.0}
    rows = []
    for p in sorted(Path(folder).glob("BD*.WAV")):
        y, sr = _mono(p)
        m = measure(y, sr)
        rows.append({"file": p.name, "tone": code[p.stem[2:4]], "decay": code[p.stem[4:6]], **m})
    cal = {"decay": {}, "tone": {}, "f0": float(np.median([r["f0_hz"] for r in rows if r["f0_hz"]])),
           "overtones_db": float(np.median([r["overtones_db"] for r in rows if r["overtones_db"] is not None])),
           "rows": rows}
    for k in KNOBS:
        d = [r["decay40_s"] for r in rows if r["decay"] == k and r["decay40_s"]]
        t = [r["click"] for r in rows if r["tone"] == k]
        cal["decay"][str(k)] = float(np.median(d))
        cal["tone"][str(k)] = float(np.median(t))
    out.write_text(json.dumps(cal, indent=1))
    return cal


def _knob(value, table, log=False):
    xs = np.array([float(table[str(k)]) for k in KNOBS])
    ks = np.array(KNOBS)
    if log:
        xs, value = np.log(xs), np.log(max(value, 1e-6))
    order = np.argsort(xs)
    xs, ks = xs[order], ks[order]
    if value <= xs[0]:
        return float(ks[0]), value < xs[0] - 0.25 * (xs[1] - xs[0])
    if value >= xs[-1]:
        return float(ks[-1]), value > xs[-1] + 0.25 * (xs[-1] - xs[-2])
    return float(np.interp(value, xs, ks)), False


def read_settings(y, sr, cal: Optional[Dict] = None) -> Dict:
    cal = cal or json.loads(CAL_PATH.read_text())
    m = measure(y, sr)
    out: Dict = {"measured": m}
    semis = 12 * np.log2(m["f0_hz"] / cal["f0"]) if m["f0_hz"] else 0.0
    if m["decay40_s"]:
        k, beyond = _knob(m["decay40_s"], cal["decay"], log=True)
        out["decay_as_heard"] = round(k * 2) / 2
        # a sampler pitches by playing faster or slower, which also shortens or stretches
        # the ring by the same ratio: the 808's own Decay, if that's how it was re-pitched
        k0, beyond = _knob(m["decay40_s"] * 2 ** (semis / 12), cal["decay"], log=True)
        out["decay"] = round(k0 * 2) / 2
        if beyond:
            out["decay_note"] = ("rings longer than a stock 808 can" if k == 10 else "shorter than a stock 808 can")
    extra = (m["overtones_db"] - cal["overtones_db"]) if m["overtones_db"] is not None else 0.0
    if extra >= 15:
        out["tone"] = None
        out["tone_note"] = "unreadable: saturation's overtones cover the click the Tone knob sets"
    else:
        k, beyond = _knob(m["click"], cal["tone"])
        out["tone"] = round(k * 2) / 2
        if beyond and k == 10:
            out["tone_note"] = "more click than a stock 808 makes (layered, or processed)"
    if m["f0_hz"]:
        semis = 12 * np.log2(m["f0_hz"] / cal["f0"])
        out["retuned_semitones"] = round(float(semis), 1)
        names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        midi = 69 + 12 * np.log2(m["f0_hz"] / 440.0)
        r = int(round(midi))
        out["pitch"] = f"{names[r % 12]}{r // 12 - 1} ({m['f0_hz']} Hz, {int(round(100 * (midi - r))):+d} cents)"
    if m["f0_early_hz"] and m["f0_hz"]:
        out["glide_semitones"] = round(float(12 * np.log2(m["f0_early_hz"] / m["f0_hz"])), 1)
    if m["overtones_db"] is not None:
        extra = m["overtones_db"] - cal["overtones_db"]
        out["saturation"] = ("none (as stock)" if extra < 6 else "light" if extra < 15 else
                             "heavy" if extra < 25 else "very heavy")
        out["overtones_vs_stock_db"] = round(float(extra), 1)
    return out

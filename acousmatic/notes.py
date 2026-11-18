"""Notes: the lines the lead and the bass play, and the drum hits.

Pitch: pYIN (probabilistic YIN) on each stem gives a pitch and a voicing probability
per frame. Frames are joined into notes where the pitch holds within a semitone, and
notes are snapped to the 16th-note grid from the beat tracker.
Drums: onsets in three frequency bands of the percussive stem (kick below 150 Hz,
snare 150 Hz-3 kHz with a noisy spectrum, cymbals and hats above 6 kHz).
"""
from __future__ import annotations

import librosa
import numpy as np

from .audio import SR

HOP_P = 256


def _snap(t, grid):
    i = int(np.argmin(np.abs(grid - t)))
    return i


def _onset_frames(y, lo=None, hi=None):
    """Note starts in a stem: rises in its linear energy (in a band, if given)."""
    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=HOP_P))
    f = librosa.fft_frequencies(sr=SR, n_fft=1024)
    rows = np.ones(len(f), bool) if lo is None else (f >= lo) & (f < hi)
    env = _flux(S, rows)
    on = librosa.onset.onset_detect(onset_envelope=env, sr=SR, hop_length=HOP_P, units="frames",
                                    delta=0.08, wait=4, normalize=False)
    return set(int(x) for x in on)


def line(y, fmin, fmax, grid, frame_length=2048, min_dur=0.08, vthresh=0.1, onset_band=None):
    """A monophonic line: pYIN pitch, its voiced/unvoiced decision (a distorted bass is
    voiced with low confidence, so the floor is low), lightly smoothed, split into notes
    where the pitch moves a semitone or a new onset starts."""
    f0, voiced, vprob = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=SR, frame_length=frame_length, hop_length=HOP_P)
    times = librosa.times_like(f0, sr=SR, hop_length=HOP_P)
    keep = voiced & (vprob >= vthresh)
    midi = librosa.hz_to_midi(np.where(keep, f0, np.nan))
    # median-smooth voiced runs over 5 frames (about 60 ms)
    sm = midi.copy()
    for i in range(len(midi)):
        w = midi[max(0, i - 2): i + 3]
        w = w[~np.isnan(w)]
        if not np.isnan(midi[i]) and len(w):
            sm[i] = float(np.median(w))
    midi = sm
    onsets = _onset_frames(y, *(onset_band or (None, None)))
    min_frames = int(min_dur * SR / HOP_P)
    notes, cur = [], None
    for i, (t, m, p) in enumerate(zip(times, midi, vprob)):
        if np.isnan(m):
            if cur:
                notes.append(cur); cur = None
            continue
        new_onset = any(o in onsets for o in (i - 1, i, i + 1)) and cur and len(cur["pitches"]) >= min_frames // 2
        if cur and not new_onset and abs(m - np.median(cur["pitches"])) <= 0.7:
            cur["pitches"].append(m); cur["probs"].append(p); cur["end"] = t
        else:
            if cur:
                notes.append(cur)
            cur = {"start": t, "end": t, "pitches": [m], "probs": [p]}
    if cur:
        notes.append(cur)
    hop_t = HOP_P / SR
    out = []
    for n in notes:
        end = n["end"] + hop_t
        if end - n["start"] < min_dur:
            continue
        pitch = int(round(float(np.median(n["pitches"]))))
        a, b = _snap(n["start"], grid), _snap(end, grid)
        if b <= a:
            b = a + 1
        out.append({"start": round(float(n["start"]), 3), "end": round(float(end), 3), "midi": pitch,
                    "name": librosa.midi_to_note(pitch, unicode=False), "g0": a, "g1": b,
                    "confidence": round(float(np.mean(n["probs"])), 3),
                    "cents_off": int(round(100 * (float(np.median(n["pitches"])) - pitch)))})
    out.sort(key=lambda n: (n["g0"], -n["confidence"]))
    clean = []
    for n in out:
        if clean and n["g0"] == clean[-1]["g0"]:
            continue
        if clean and clean[-1]["g1"] > n["g0"]:
            clean[-1]["g1"] = n["g0"]
        clean.append(n)
    voiced_share = float(np.mean(~np.isnan(midi))) if len(midi) else 0.0
    return clean, voiced_share


def lead_line(lead, grid):
    return line(lead, fmin=librosa.note_to_hz("E2"), fmax=librosa.note_to_hz("C6"), grid=grid, vthresh=0.2)


def bass_line(bass, grid):
    return line(bass, fmin=librosa.note_to_hz("A0"), fmax=librosa.note_to_hz("G3"), grid=grid,
                frame_length=4096, min_dur=0.08, vthresh=0.05, onset_band=(30, 400))


BANDS = {"kick": (30, 150), "snare": (1000, 5000), "hat": (6000, 11000)}
DELTA = {"kick": 0.1, "snare": 0.3, "hat": 0.2}


def _flux(S, rows):
    """Rise in the band's linear energy, scaled so the strongest hits sit near 1.
    Linear, not decibels: on a dB scale quiet pulses (a synth bass) look like hits."""
    E = S[rows].sum(axis=0)
    d = np.maximum(0.0, np.diff(E, prepend=E[0]))
    pos = d[d > 0]
    if not len(pos):
        return d
    return np.minimum(d / (np.percentile(pos, 99.5) + 1e-12), 1.0)


def drums(y, grid):
    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=HOP_P))
    f = librosa.fft_frequencies(sr=SR, n_fft=1024)
    hits, strength = {}, {}
    for name, (lo, hi) in BANDS.items():
        env = _flux(S, (f >= lo) & (f < hi))
        on = librosa.onset.onset_detect(onset_envelope=env, sr=SR, hop_length=HOP_P, units="frames",
                                        delta=DELTA[name], wait=8, normalize=False)
        times = librosa.frames_to_time(np.asarray(on, dtype=int), sr=SR, hop_length=HOP_P)
        slots = {}
        for t, o in zip(times, on):
            k = _snap(t, grid)
            slots[k] = max(slots.get(k, 0.0), float(env[max(0, o - 2): o + 3].max()))
        hits[name], strength[name] = sorted(slots), slots
    # a kick's upper harmonics reach the snare band: a snare on a kick has to stand out
    kicks = set(hits["kick"])
    hits["snare"] = [k for k in hits["snare"] if k not in kicks or strength["snare"][k] > 0.6]
    return hits

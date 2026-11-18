"""Time: onsets, tempo, beats, meter and bars.

Beat tracking is dynamic programming over a spectral-flux onset envelope (Ellis 2007,
as in librosa). Meter: for each candidate grouping (3, 4, 5, 6, 7 beats) and phase, how
much stronger the low-frequency accent (kick) is on the would-be downbeats than on the
other beats. Odd meters and changing meters are hard; the record gives the contrast
so a weak call is visible as weak.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import librosa
import numpy as np

from .audio import SR
from .stems import HOP

METERS = (4, 3, 6, 5, 7)


@dataclass
class Rhythm:
    tempo: float
    beats: np.ndarray             # beat times, seconds
    meter: int
    meter_contrast: dict          # meter -> accent contrast of its best phase
    downbeat_index: int           # index into beats of the first downbeat
    bars: list = field(default_factory=list)   # [{"n", "start", "end", "beats": [t...]}]
    local_tempo: list = field(default_factory=list)
    phase_shifted: bool = False


def _band_onset(y, lo, hi):
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=HOP))
    f = librosa.fft_frequencies(sr=SR, n_fft=2048)
    rows = (f >= lo) & (f < hi)
    return librosa.onset.onset_strength(S=librosa.amplitude_to_db(S[rows], ref=np.max), sr=SR, hop_length=HOP)


def analyse(mix: np.ndarray, drums: np.ndarray) -> Rhythm:
    env = librosa.onset.onset_strength(y=mix, sr=SR, hop_length=HOP, aggregate=np.median)
    env_d = librosa.onset.onset_strength(y=drums, sr=SR, hop_length=HOP)
    n = min(len(env), len(env_d))
    env = env[:n] / (env.max() + 1e-9) + env_d[:n] / (env_d.max() + 1e-9)
    tempo, beat_frames = librosa.beat.beat_track(onset_envelope=env, sr=SR, hop_length=HOP, units="frames")
    tempo = float(np.atleast_1d(tempo)[0])
    low = _band_onset(drums, 30, 150)
    low = low / (low.max() + 1e-9)
    mid = _band_onset(drums, 150, 3000)
    mid = mid / (mid.max() + 1e-9)

    # Phase: beat trackers can lock onto the off-beats (hats are the busiest onsets).
    # Kick and snare mark the beat; if they sit on the midpoints rather than the beats,
    # move the beats by half a beat.
    acc_env = low + mid
    mids = ((beat_frames[:-1] + beat_frames[1:]) // 2).astype(int)
    def at(frames):
        f = frames[frames < len(acc_env)]
        return float(np.mean([acc_env[max(0, x - 2): x + 3].max() for x in f])) if len(f) else 0.0
    on_beat, off_beat = at(beat_frames), at(mids)
    phase_shifted = off_beat > 1.15 * on_beat
    if phase_shifted:
        beat_frames = mids
    beats = librosa.frames_to_time(beat_frames, sr=SR, hop_length=HOP)
    win = 3
    acc = np.array([low[max(0, b - win): b + win + 1].max() if b < len(low) else 0.0 for b in beat_frames])

    contrast = {}
    best = (0.0, 4, 0)
    for m in METERS:
        if len(acc) < 4 * m:
            continue
        scores = []
        for ph in range(m):
            on = acc[ph::m]
            off = np.delete(acc, np.arange(ph, len(acc), m))
            scores.append((on.mean() - off.mean()) / (acc.std() + 1e-9))
        ph = int(np.argmax(scores))
        c = float(scores[ph])
        # a longer grouping must beat a shorter one clearly to be preferred
        contrast[m] = round(c, 3)
        penalty = {4: 1.0, 3: 1.05, 6: 1.15, 5: 1.25, 7: 1.3}[m]
        if c / penalty > best[0]:
            best = (c / penalty, m, ph)
    _, meter, first = best

    bars = []
    starts = list(range(first, len(beats), meter))
    # beats before the first downbeat are a pickup, outside bar 1
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else len(beats)
        bt = beats[s:e]
        end = beats[e] if e < len(beats) else float(bt[-1] + (60.0 / tempo))
        bars.append({"n": i + 1, "start": round(float(bt[0]), 3), "end": round(float(end), 3),
                     "beats": [round(float(t), 3) for t in bt]})
    ibi = np.diff(beats)
    local = []
    for b in bars:
        sel = (beats[:-1] >= b["start"]) & (beats[:-1] < b["end"])
        if sel.any():
            local.append(round(float(60.0 / np.median(ibi[sel])), 1))
        else:
            local.append(round(tempo, 1))
    return Rhythm(tempo=round(tempo, 1), beats=beats, meter=meter, meter_contrast=contrast,
                  downbeat_index=first, bars=bars, local_tempo=local,
                  phase_shifted=bool(phase_shifted))


def grid(rhythm: Rhythm, per_beat: int = 4) -> np.ndarray:
    """Subdivision times (16ths by default) across the beats, extended one beat each side."""
    b = rhythm.beats
    if len(b) < 2:
        return b
    ibi = np.median(np.diff(b))
    ext = np.concatenate([[b[0] - ibi], b, [b[-1] + ibi]])
    pts = []
    for i in range(len(ext) - 1):
        pts.extend(np.linspace(ext[i], ext[i + 1], per_beat, endpoint=False))
    pts.append(ext[-1])
    return np.array(pts)

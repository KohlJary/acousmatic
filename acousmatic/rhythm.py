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
    tempo_candidates: list = field(default_factory=list)
    carried: dict = field(default_factory=dict)


def _band_onset(y, lo, hi):
    """Rise in a band's linear energy (on a dB scale, quiet pulses look like hits)."""
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=HOP))
    f = librosa.fft_frequencies(sr=SR, n_fft=2048)
    E = S[(f >= lo) & (f < hi)].sum(axis=0)
    d = np.maximum(0.0, np.diff(E, prepend=E[0]))
    pos = d[d > 0]
    return np.minimum(d / (np.percentile(pos, 99.5) + 1e-12), 1.0) if len(pos) else d


def _tempo_candidates(env):
    """Strongest tempogram peaks and their 2:1, 3:2 and 4:3 relatives, 60-200 BPM."""
    tg = librosa.feature.tempogram(onset_envelope=env, sr=SR, hop_length=HOP).mean(axis=1)
    bpms = librosa.tempo_frequencies(len(tg), sr=SR, hop_length=HOP)
    sel = np.where((bpms > 50) & (bpms < 260))[0]
    peaks = [i for i in sel if tg[i] >= tg[i - 1] and tg[i] >= tg[min(i + 1, len(tg) - 1)]]
    peaks = sorted(peaks, key=lambda i: -tg[i])[:4]
    out = []
    for i in peaks:
        for r in (1, 0.5, 2, 2 / 3, 3 / 2, 3 / 4, 4 / 3):
            t = float(bpms[i] * r)
            if 60 <= t <= 200 and not any(abs(t / x - 1) < 0.03 for x in out):
                out.append(t)
    return out


def _grid_fit(env, beat_times):
    """How well the onsets sit on a 16th grid laid over these beats (0..1, strength-weighted)."""
    on = librosa.onset.onset_detect(onset_envelope=env, sr=SR, hop_length=HOP, units="frames")
    b = np.asarray(beat_times)
    if len(b) < 8 or not len(on):
        return 0.0
    w = env[on]; t = librosa.frames_to_time(on, sr=SR, hop_length=HOP)
    g = np.concatenate([np.linspace(b[i], b[i + 1], 4, endpoint=False) for i in range(len(b) - 1)])
    six = np.median(np.diff(b)) / 4
    sel = (t >= b[0]) & (t <= b[-1])
    t, w = t[sel], w[sel]
    if not len(t):
        return 0.0
    d = np.array([np.abs(g - x).min() for x in t]) / six
    return float((w * np.exp(-(d / 0.2) ** 2)).sum() / (w.sum() + 1e-12))


def _track(env):
    """Choose among candidate tempos by how well the onsets fit a 16th grid at each,
    with a mild preference for tempos near 115 BPM (it only breaks near-ties between
    octaves). The strongest periodicity isn't always the beat: a 3+3+2 kick pattern
    pulses at 4/3 of the tempo."""
    rows = []
    for T in _tempo_candidates(env):
        _, bf = librosa.beat.beat_track(onset_envelope=env, sr=SR, hop_length=HOP, start_bpm=T,
                                        tightness=400, units="frames")
        if len(bf) < 8:
            continue
        bt = librosa.frames_to_time(bf, sr=SR, hop_length=HOP)
        # the mean of the middle half of the beat intervals: a median of frame-quantised
        # intervals only resolves the tempo to a couple of percent
        d = np.diff(bt)
        med = float(np.median(d))
        d = d[(d > 0.75 * med) & (d < 1.33 * med)]      # drop skipped or doubled beats only
        real = 60.0 / float(d.mean())
        prior = float(np.exp(-0.5 * (np.log2(real / 115.0)) ** 2))
        fit = _grid_fit(env, bt)
        rows.append((round(fit * prior, 6), round(fit, 4), round(real, 2), bf))
    if not rows:
        tempo, bf = librosa.beat.beat_track(onset_envelope=env, sr=SR, hop_length=HOP, units="frames")
        return float(np.atleast_1d(tempo)[0]), bf, []
    rows.sort(key=lambda r: (-r[0], r[2]))
    best = rows[0]
    seen, cands = set(), []
    for r in rows:
        k = round(r[2])
        if k not in seen:
            seen.add(k); cands.append({"bpm": r[2], "grid_fit": r[1], "score": r[0]})
    return best[2], best[3], cands


def _carry(beat_frames, mix, floor_db=-45.0):
    """The beat tracker drops beats where the onsets go quiet (a sparse intro or outro).
    Carry the established grid on to where the audio starts and stops sounding, and say
    how many beats were carried rather than tracked."""
    bf = np.asarray(beat_frames, dtype=int)
    if len(bf) < 4:
        return bf, {"before": 0, "after": 0}
    rms = librosa.feature.rms(y=mix, hop_length=HOP)[0]
    db = librosa.amplitude_to_db(rms, ref=np.max)
    sounding = np.where(db > floor_db)[0]
    if not len(sounding):
        return bf, {"before": 0, "after": 0}
    first, last = int(sounding[0]), int(sounding[-1])
    ibi = float(np.mean(np.diff(bf[-min(len(bf), 33):])))
    after = []
    t = bf[-1] + ibi
    while t <= last - ibi / 2:
        after.append(int(round(t))); t += ibi
    ibi0 = float(np.mean(np.diff(bf[:min(len(bf), 33)])))
    before = []
    t = bf[0] - ibi0
    while t >= first - ibi0 / 4 and t >= 0:
        before.insert(0, int(round(t))); t -= ibi0
    return np.concatenate([before, bf, after]).astype(int), {"before": len(before), "after": len(after)}


def _fix_phase(beat_frames, acc_env, window=16, ratio=1.15):
    bf = np.asarray(beat_frames, dtype=int)
    if len(bf) < 4:
        return bf, False
    def at(x):
        x = x[(x >= 0) & (x < len(acc_env))]
        return float(np.mean([acc_env[max(0, i - 2): i + 3].max() for i in x])) if len(x) else 0.0
    ibi = int(np.median(np.diff(bf)))
    out, shifted_any = [], 0
    for w in range(0, len(bf), window):
        seg = bf[w: w + window]
        mids = seg + ibi // 2
        if at(mids) > ratio * at(seg):
            out.extend(mids.tolist()); shifted_any += 1
        else:
            out.extend(seg.tolist())
    out = sorted(out)
    mended = [out[0]]
    for x in out[1:]:
        gap = x - mended[-1]
        if gap < 0.6 * ibi:
            continue
        if gap > 1.5 * ibi:
            mended.append(mended[-1] + gap // 2)
        mended.append(x)
    return np.array(mended, dtype=int), shifted_any > 0


def analyse(mix: np.ndarray, drums: np.ndarray) -> Rhythm:
    env = librosa.onset.onset_strength(y=mix, sr=SR, hop_length=HOP, aggregate=np.median)
    env_d = librosa.onset.onset_strength(y=drums, sr=SR, hop_length=HOP)
    n = min(len(env), len(env_d))
    env = env[:n] / (env.max() + 1e-9) + env_d[:n] / (env_d.max() + 1e-9)
    tempo, beat_frames, tempo_candidates = _track(env)
    low = _band_onset(drums, 30, 150)
    low = low / (low.max() + 1e-9)
    mid = _band_onset(drums, 150, 3000)
    mid = mid / (mid.max() + 1e-9)

    # Phase: beat trackers can lock onto the off-beats (hats are the busiest onsets),
    # for a whole song or for a stretch of it. Kick and snare mark the beat: in each
    # window of beats, if they sit on the midpoints rather than the beats, that window
    # moves by half a beat. Then the joins are mended (no doubled or missing beats).
    beat_frames, phase_shifted = _fix_phase(beat_frames, low + mid)
    beat_frames, carried = _carry(beat_frames, mix)
    beats = librosa.frames_to_time(beat_frames, sr=SR, hop_length=HOP)
    # Downbeat evidence per beat: kick, minus snare (the backbeat marks 2 and 4), plus
    # how much the harmony changes there (chords tend to change on the one).
    win = 3
    def peak(env, b):
        return env[max(0, b - win): b + win + 1].max() if b < len(env) else 0.0
    snare = _band_onset(drums, 1000, 5000)
    kick_b = np.array([peak(low, b) for b in beat_frames])
    snare_b = np.array([peak(snare, b) for b in beat_frames])
    chroma = librosa.feature.chroma_stft(y=mix, sr=SR, hop_length=HOP)
    cb = librosa.util.sync(chroma, [f for f in beat_frames if f < chroma.shape[1]], aggregate=np.median)
    cb = cb / (np.linalg.norm(cb, axis=0, keepdims=True) + 1e-9)
    nov = np.concatenate([[0.0], 1 - (cb[:, 1:] * cb[:, :-1]).sum(axis=0)])
    nov = nov[: len(beat_frames)]
    if len(nov) < len(beat_frames):
        nov = np.pad(nov, (0, len(beat_frames) - len(nov)))
    z = lambda x: (x - x.mean()) / (x.std() + 1e-9)
    acc = z(kick_b) - z(snare_b) + z(nov)

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
                  phase_shifted=bool(phase_shifted), tempo_candidates=tempo_candidates[:5],
                  carried=carried)


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

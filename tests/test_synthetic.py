"""Known answers from synthesized audio, and determinism."""
import json

import numpy as np
import pytest

from acousmatic import harmony, rhythm, stems, notes, sound
from acousmatic.audio import SR, SR_HI


def tone(freqs, dur, sr, amp=0.2):
    t = np.arange(int(dur * sr)) / sr
    y = sum(np.sign(np.sin(2 * np.pi * f * t)) * 0.3 + np.sin(2 * np.pi * f * t) for f in freqs)
    env = np.minimum(1, t / 0.01) * np.exp(-t * 1.5)
    return amp * y * env / len(freqs)


def click_track(bpm, bars, sr):
    spb = 60 / bpm
    n = int(bars * 4 * spb * sr)
    y = np.zeros(n)
    k = np.arange(int(0.03 * sr))
    kick = np.sin(2 * np.pi * 60 * k / sr) * np.exp(-k / (0.01 * sr))
    rng = np.random.default_rng(0)
    snare = rng.standard_normal(len(k)) * np.exp(-k / (0.008 * sr)) * 0.5
    for b in range(bars * 4):
        i = int(b * spb * sr)
        seg = kick if b % 2 == 0 else snare
        y[i:i + len(seg)] += seg[: n - i]
    return y


CHORDS = [("C", [261.63, 329.63, 392.00]), ("Am", [220.00, 261.63, 329.63]),
          ("F", [174.61, 220.00, 261.63]), ("G", [196.00, 246.94, 293.66])]


def song(bpm=120, loops=4, sr=SR_HI):
    spb = 60 / bpm
    bar = 4 * spb
    parts = []
    for _ in range(loops):
        for _, fs in CHORDS:
            parts.append(tone(fs, bar, sr) + tone([fs[0] / 2], bar, sr, amp=0.25))
    harm = np.concatenate(parts)
    drums = click_track(bpm, loops * len(CHORDS), sr)
    n = min(len(harm), len(drums))
    mono = harm[:n] + drums[:n]
    return np.vstack([mono, mono]).astype(np.float32)


@pytest.fixture(scope="module")
def separated():
    return stems.separate(song())


def test_tempo(separated):
    rh = rhythm.analyse(separated.mid, separated.drums)
    assert abs(rh.tempo - 120) < 3 or abs(rh.tempo - 60) < 2 or abs(rh.tempo - 240) < 5


def test_key_and_chords(separated):
    rh = rhythm.analyse(separated.mid, separated.drums)
    import librosa
    bf = librosa.time_to_frames(rh.beats, sr=SR, hop_length=stems.HOP)
    per_beat, Cs = harmony.chords(separated.bass + separated.other, separated.bass, bf)
    k = harmony.key_of(Cs.sum(axis=1))
    assert k["name"] in ("C major", "A minor")
    names = {c["name"] for c in per_beat}
    assert {"C", "F", "G"} <= names or {"C", "Am", "F", "G"} & names


def test_pan():
    t = np.arange(SR_HI * 3) / SR_HI
    y = 0.2 * np.sin(2 * np.pi * 440 * t)
    st = np.vstack([0.2 * y, y]).astype(np.float32)   # mostly right
    f = sound.Field(st)
    sec = f.section(0, 3)
    assert sec["pan"]["mid"] > 0.5


def test_line_on_a_sine():
    import librosa
    sr = SR
    notes_hz = [220.0, 246.94, 261.63, 293.66]
    y = np.concatenate([0.3 * np.sin(2 * np.pi * f * np.arange(int(0.5 * sr)) / sr) for f in notes_hz])
    grid = np.arange(0, 2.2, 0.125)
    ns, _ = notes.line(y, fmin=100, fmax=800, grid=grid)
    assert [n["name"] for n in ns] == ["A3", "B3", "C4", "D4"]


def test_deterministic(tmp_path):
    import soundfile as sf
    from acousmatic.listen import listen
    y = song(loops=2)
    p = tmp_path / "s.wav"
    sf.write(p, y.T, SR_HI)
    a = listen(p, tmp_path / "a", engrave=False, log=lambda m: None)
    b = listen(p, tmp_path / "b", engrave=False, log=lambda m: None)
    for r in (a, b):
        r.pop("elapsed_s", None)
    assert json.dumps(a, sort_keys=True, default=float) == json.dumps(b, sort_keys=True, default=float)

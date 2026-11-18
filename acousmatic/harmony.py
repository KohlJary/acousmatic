"""Harmony: key and chords from chroma.

Chroma is the energy in each of the 12 pitch classes, from a constant-Q transform of
the harmonic stems (bass + other), so drums don't smear it. Key: correlation with the
Krumhansl-Kessler profiles over the 24 major and minor keys. Chords: per beat, cosine
match against chord templates (with the bass stem's lowest pitch class weighting the
root), smoothed with a Viterbi pass so a chord holds unless the evidence moves.
"""
from __future__ import annotations

import librosa
import numpy as np

from .audio import SR
from .stems import HOP

PC = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
KK_MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KK_MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

QUALITIES = {          # intervals above the root
    "": (0, 4, 7),
    "m": (0, 3, 7),
    "5": (0, 7),
    "7": (0, 4, 7, 10),
    "maj7": (0, 4, 7, 11),
    "m7": (0, 3, 7, 10),
    "dim": (0, 3, 6),
    "sus4": (0, 5, 7),
    "sus2": (0, 2, 7),
}


def _templates():
    labels, T = [], []
    for root in range(12):
        for q, iv in QUALITIES.items():
            t = np.zeros(12)
            for k in iv:
                t[(root + k) % 12] = 1.0
            t[root] += 0.5
            labels.append((root, q))
            T.append(t / np.linalg.norm(t))
    return labels, np.array(T)


LABELS, TEMPLATES = _templates()


FLATS = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
SHARPS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
MAJOR_TONIC = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
MINOR_TONIC = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "G#", "A", "Bb", "B"]


def spelling(tonic_pc, mode):
    """Note names that follow the key signature: flats in flat keys, sharps in sharp ones."""
    flat = tonic_pc in ({0, 5, 10, 3, 8, 1} if mode == "major" else {9, 2, 7, 0, 5, 10, 3})
    tonic = (MAJOR_TONIC if mode == "major" else MINOR_TONIC)[tonic_pc]
    return (FLATS if flat else SHARPS), tonic


def note_name(midi, names):
    return f"{names[midi % 12]}{midi // 12 - 1}"


def chord_name(root, q):
    return "N" if root is None else PC[root] + q


def chroma(y):
    C = librosa.feature.chroma_cqt(y=y, sr=SR, hop_length=HOP, bins_per_octave=36)
    return C


def bass_pc(bass):
    C = librosa.feature.chroma_cqt(y=bass, sr=SR, hop_length=HOP, fmin=librosa.note_to_hz("C1"), n_octaves=4)
    return C


def key_of(chroma_vec, bass_weights=None):
    v = chroma_vec / (chroma_vec.sum() + 1e-9)
    scores = []
    for tonic in range(12):
        for mode, prof in (("major", KK_MAJ), ("minor", KK_MIN)):
            scores.append((float(np.corrcoef(v, np.roll(prof, tonic))[0, 1]), tonic, mode))
    scores.sort(reverse=True)
    if bass_weights is not None:
        bw = np.asarray(bass_weights, dtype=float)
        bw = bw / (bw.sum() + 1e-9)
        top = scores[0]
        rel = ((top[1] + 9) % 12, "minor") if top[2] == "major" else ((top[1] + 3) % 12, "major")
        # keys close on the chroma, and always the relative key (same notes, different home)
        near = [x for x in scores if x[0] >= top[0] - 0.1 or (x[1], x[2]) == rel]
        # among keys this close, the one whose tonic the bass dwells on
        near.sort(key=lambda x: -(x[0] + 0.5 * bw[x[1]]))
        rest = [x for x in scores if x not in near]
        scores = near + rest
    best, second = scores[0], scores[1]
    settled = bass_weights is not None and best is not max(scores, key=lambda x: x[0])
    return {"tonic": PC[best[1]], "tonic_pc": best[1], "mode": best[2], "name": f"{PC[best[1]]} {best[2]}",
            "settled_by_bass": bool(settled),
            "strength": round(best[0], 3), "runner_up": f"{PC[second[1]]} {second[2]}",
            "margin": round(best[0] - second[0], 3)}


FIT_CLEAR = 0.8


def chords(harm, bass, beat_frames, self_prob=0.6, bass_weight=0.3, four_note_penalty=0.05, sus_penalty=0.03):
    C = chroma(harm)
    B = bass_pc(bass)
    n = min(C.shape[1], B.shape[1])
    C, B = C[:, :n], B[:, :n]
    idx = [f for f in beat_frames if f < n]
    Cs = librosa.util.sync(C, idx, aggregate=np.median)
    Bs = librosa.util.sync(B, idx, aggregate=np.median)
    energy = librosa.util.sync(librosa.feature.rms(y=harm, hop_length=HOP)[:, :n], idx, aggregate=np.mean)[0]
    X = Cs / (np.linalg.norm(Cs, axis=0, keepdims=True) + 1e-9)
    sim = TEMPLATES @ X                                   # (labels, beats)
    bass_root = np.array([Bs[r[0]] / (Bs.max(axis=0) + 1e-9) for r in LABELS])
    sim = sim + bass_weight * bass_root
    # A power chord only when the third is really absent: two-note templates match
    # any triad on the same root, so they start a little behind.
    sim = sim - 0.06 * np.array([len(QUALITIES[q]) < 3 for _, q in LABELS])[:, None]
    # ... and a four-note chord only when its fourth note is really there: otherwise a
    # 7th template swallows the triads around it (Bmaj7 holds Gb and Bb).
    sim = sim - four_note_penalty * np.array([len(QUALITIES[q]) > 3 for _, q in LABELS])[:, None]
    sim = sim - sus_penalty * np.array([q.startswith("sus") for _, q in LABELS])[:, None]
    flat = 1.0 - (X.max(axis=0) - X.mean(axis=0))         # flat chroma -> no clear chord
    quiet = energy < 0.05 * (energy.max() + 1e-9)
    nochord = np.where(quiet | (flat > 0.85), 1.0, 0.0)
    P = np.vstack([np.exp(8 * sim), np.exp(8 * (0.5 + 0.5 * nochord))[None, :]])
    P = P / P.sum(axis=0, keepdims=True)
    trans = librosa.sequence.transition_loop(P.shape[0], self_prob)
    path = librosa.sequence.viterbi(P, trans)
    # How well the chosen template fits each beat's chroma (cosine, before the bass and
    # penalties): below FIT_CLEAR the chord is marked unclear, so a chromatic line or a
    # dense smear doesn't read as a confident chord.
    raw = TEMPLATES @ X
    out = []
    for i, s in enumerate(path):
        if s == len(LABELS):
            out.append({"root": None, "quality": None, "name": "N", "fit": None,
                        "confidence": round(float(P[s, i]), 3)})
            continue
        r, q = LABELS[s]
        out.append({"root": r, "quality": q, "name": chord_name(r, q), "fit": round(float(raw[s, i]), 3),
                    "confidence": round(float(P[s, i]), 3)})
    return out, Cs


def merge(per_beat, beat_times):
    """Consecutive equal chords -> spans [{name, start, end, beats}]."""
    spans = []
    bt = list(beat_times)
    ibi = float(np.median(np.diff(bt))) if len(bt) > 1 else 0.5
    for i, c in enumerate(per_beat):
        # chroma can run a beat past the last tracked beat: extend the grid for it
        t = float(bt[i]) if i < len(bt) else (bt[-1] + ibi * (i - len(bt) + 1) if bt else 0.0)
        if spans and spans[-1]["name"] == c["name"]:
            spans[-1]["beats"] += 1
            spans[-1]["fits"].append(c.get("fit"))
        else:
            spans.append({"name": c["name"], "root": c["root"], "quality": c["quality"], "start": t, "beats": 1,
                          "fits": [c.get("fit")]})
    for i, s in enumerate(spans):
        nxt = spans[i + 1]["start"] if i + 1 < len(spans) else s["start"] + s["beats"] * ibi
        s["end"] = round(float(nxt), 3)
        s["start"] = round(float(s["start"]), 3)
        fits = [f for f in s.pop("fits", []) if f is not None]
        s["fit"] = round(float(np.mean(fits)), 3) if fits else None
        s["clear"] = s["fit"] is None or s["fit"] >= FIT_CLEAR
    return spans

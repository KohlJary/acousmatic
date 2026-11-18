"""Music theory, read off the chords and notes by rule.

Roman numerals relative to the key, the mode (by how much of the harmonic energy each
mode's scale accounts for), borrowed chords, cadences, the progressions that loop,
harmonic rhythm, the melody's scale degrees, range and contour, and syncopation.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from .harmony import PC

MODES = {
    "ionian (major)": (0, 2, 4, 5, 7, 9, 11),
    "dorian": (0, 2, 3, 5, 7, 9, 10),
    "phrygian": (0, 1, 3, 5, 7, 8, 10),
    "lydian": (0, 2, 4, 6, 7, 9, 11),
    "mixolydian": (0, 2, 4, 5, 7, 9, 10),
    "aeolian (natural minor)": (0, 2, 3, 5, 7, 8, 10),
    "harmonic minor": (0, 2, 3, 5, 7, 8, 11),
    "locrian": (0, 1, 3, 5, 6, 8, 10),
}
NUMERALS = ["I", "II", "III", "IV", "V", "VI", "VII"]
DEGREE_NAMES = {0: "1", 1: "b2", 2: "2", 3: "b3", 4: "3", 5: "4", 6: "#4/b5", 7: "5", 8: "b6", 9: "6", 10: "b7", 11: "7"}
# interval above the tonic -> (degree index, accidental)
_DEG = {0: (0, ""), 1: (1, "b"), 2: (1, ""), 3: (2, "b"), 4: (2, ""), 5: (3, ""), 6: (4, "b"),
        7: (4, ""), 8: (5, "b"), 9: (5, ""), 10: (6, "b"), 11: (6, "")}


def roman(root, quality, tonic_pc, mode):
    if root is None:
        return "N"
    iv = (root - tonic_pc) % 12
    deg, acc = _DEG[iv]
    # in minor, b3 b6 b7 are diatonic: write them without the flat
    diatonic = MODES["aeolian (natural minor)"] if mode == "minor" else MODES["ionian (major)"]
    if mode == "minor" and iv in (3, 8, 10):
        acc = ""
    num = NUMERALS[deg]
    minorish = quality in ("m", "m7", "dim")
    if minorish:
        num = num.lower()
    suffix = {"dim": "°", "7": "7", "maj7": "maj7", "m7": "7", "5": "5", "sus4": "sus4", "sus2": "sus2"}.get(quality, "")
    return acc + num + suffix


def borrowed(root, quality, tonic_pc, mode):
    if root is None or quality is None:
        return False
    scale = MODES["aeolian (natural minor)"] if mode == "minor" else MODES["ionian (major)"]
    scale_pcs = {(tonic_pc + s) % 12 for s in scale}
    from .harmony import QUALITIES
    tones = {(root + i) % 12 for i in QUALITIES[quality]}
    if mode == "minor":
        scale_pcs |= {(tonic_pc + 11) % 12}   # raised 7th of harmonic minor is at home in minor
    return not tones <= scale_pcs


def mode_of(chroma_total, tonic_pc):
    v = np.asarray(chroma_total, dtype=float)
    v = v / (v.sum() + 1e-9)
    fits = []
    for name, sc in MODES.items():
        inside = sum(v[(tonic_pc + s) % 12] for s in sc)
        fits.append((round(float(inside), 3), name))
    fits.sort(reverse=True)
    return {"best": fits[0][1], "fit": fits[0][0], "ranking": [{"mode": n, "fit": f} for f, n in fits[:4]],
            "degree_profile": {DEGREE_NAMES[i]: round(float(v[(tonic_pc + i) % 12]), 3) for i in range(12)}}


def cadences(spans, tonic_pc, mode, bar_of):
    out = []
    names = [roman(s["root"], s["quality"], tonic_pc, mode) for s in spans]
    def base(r):
        return r.rstrip("57°").replace("maj", "").replace("sus4", "").replace("sus2", "")
    for i in range(1, len(spans)):
        a, b = base(names[i - 1]), base(names[i])
        kind = None
        if a.upper() == "V" and b.upper() == "I":
            kind = "authentic (V–I)"
        elif a.upper() == "IV" and b.upper() == "I":
            kind = "plagal (IV–I)"
        elif a.upper() == "V" and b.upper() == "VI":
            kind = "deceptive (V–vi)"
        elif a.upper() == "BVII" and b.upper() == "I":
            kind = "bVII–I (rock / mixolydian)"
        elif a.upper() == "VII" and b.upper() == "I" and mode == "minor":
            kind = "VII–i (aeolian)"
        if kind:
            out.append({"bar": bar_of(spans[i]["start"]), "kind": kind, "from": names[i - 1], "to": names[i]})
    return out


def loops(names, max_len=8, min_reps=2):
    """The chord loops that repeat back to back, longest first."""
    found = []
    i = 0
    while i < len(names):
        best = None
        for L in range(max_len, 1, -1):
            pat = names[i:i + L]
            if len(pat) < L or len(set(pat)) < 2:
                continue
            reps = 1
            while names[i + reps * L: i + (reps + 1) * L] == pat:
                reps += 1
            if reps >= min_reps:
                best = (pat, reps)
                break
        if best:
            found.append({"loop": " – ".join(best[0]), "times": best[1], "from_span": i})
            i += len(best[0]) * best[1]
        else:
            i += 1
    return found


def melody(notes, tonic_pc):
    if not notes:
        return None
    midis = [n["midi"] for n in notes]
    steps = np.diff(midis)
    degs = Counter(DEGREE_NAMES[(m - tonic_pc) % 12] for m in midis)
    import librosa
    return {
        "range": f"{librosa.midi_to_note(min(midis), unicode=False)}–{librosa.midi_to_note(max(midis), unicode=False)}",
        "span_semitones": int(max(midis) - min(midis)),
        "stepwise_share": round(float(np.mean(np.abs(steps) <= 2)), 3) if len(steps) else None,
        "leaps": int(np.sum(np.abs(steps) >= 5)) if len(steps) else 0,
        "most_used_degrees": [d for d, _ in degs.most_common(5)],
        "contour": "".join("/" if s > 0 else "\\" if s < 0 else "-" for s in steps[:64]),
    }


def syncopation(slots, per_beat=4):
    if not slots:
        return None
    off = [s for s in slots if s % per_beat != 0]
    weak_8th = [s for s in off if s % per_beat == per_beat // 2]
    return {"offbeat_share": round(len(off) / len(slots), 3),
            "on_16th_offbeats": round((len(off) - len(weak_8th)) / len(slots), 3)}


def harmonic_rhythm(spans):
    if not spans:
        return None
    beats = [s["beats"] for s in spans if s["name"] != "N"]
    return round(float(np.mean(beats)), 2) if beats else None


def degrees_of(midis, tonic_pc):
    return [DEGREE_NAMES[(m - tonic_pc) % 12] for m in midis]

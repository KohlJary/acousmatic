"""Form: where the song repeats itself.

Laplacian structural segmentation (McFee & Ellis 2014): a beat-synchronous recurrence
matrix from constant-Q spectra (what repeats) joined with a sequence graph from MFCCs
(what's continuous), its spectral embedding clustered into section types. Clustering
uses a fixed seed. Boundaries are snapped to bar lines; very short sections are merged
into their neighbour. Sections are labelled by letter (A, B, C...): which one is a
verse or a hook is for the listener to say.
"""
from __future__ import annotations

import librosa
import numpy as np
import scipy
from sklearn.cluster import KMeans

from .audio import SR
from .stems import HOP


def embed(mix, beat_frames):
    C = librosa.amplitude_to_db(np.abs(librosa.cqt(y=mix, sr=SR, hop_length=HOP, bins_per_octave=36, n_bins=252)),
                                ref=np.max)
    n = C.shape[1]
    idx = [f for f in beat_frames if f < n]
    Csync = librosa.util.sync(C, idx, aggregate=np.median)
    mfcc = librosa.feature.mfcc(y=mix, sr=SR, hop_length=HOP)[:, :n]
    Msync = librosa.util.sync(mfcc, idx)
    if Csync.shape[1] < 16:
        return None
    R = librosa.segment.recurrence_matrix(librosa.feature.stack_memory(Csync, n_steps=4, mode="edge"),
                                          width=3, mode="affinity", sym=True)
    Rf = librosa.segment.path_enhance(R, 15)
    path_distance = np.sum(np.diff(Msync, axis=1) ** 2, axis=0)
    sigma = np.median(path_distance) + 1e-9
    path_sim = np.exp(-path_distance / sigma)
    R_path = np.diag(path_sim, k=1) + np.diag(path_sim, k=-1)
    deg_path = np.sum(R_path, axis=1)
    deg_rec = np.sum(Rf, axis=1)
    mu = deg_path.dot(deg_path + deg_rec) / (np.sum((deg_path + deg_rec) ** 2) + 1e-9)
    A = mu * Rf + (1 - mu) * R_path
    L = scipy.sparse.csgraph.laplacian(A, normed=True)
    evals, evecs = scipy.linalg.eigh(L)
    evecs = scipy.ndimage.median_filter(evecs, size=(9, 1))
    Cnorm = np.cumsum(evecs ** 2, axis=1) ** 0.5
    return evals, evecs, Cnorm


def _bar_labels(emb, k, bars, beat_times):
    evals, evecs, Cnorm = emb
    X = evecs[:, :k] / (Cnorm[:, k - 1: k] + 1e-9)
    beat_labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X)
    out = []
    for b in bars:
        sel = [i for i, t in enumerate(beat_times[:len(beat_labels)]) if b["start"] - 1e-3 <= t < b["end"] - 1e-3]
        labs = beat_labels[sel] if sel else [out[-1] if out else 0]
        out.append(int(np.bincount(np.asarray(labs)).argmax()))
    return out


def _runs(bar_label, min_bars):
    segs = []
    for i, lab in enumerate(bar_label):
        if segs and segs[-1]["c"] == lab:
            segs[-1]["end_bar"] = i + 1
        else:
            segs.append({"c": lab, "start_bar": i + 1, "end_bar": i + 1})
    # fold runs shorter than min_bars into the neighbour before them (or after, at the start)
    i = 0
    while i < len(segs) and len(segs) > 1:
        s = segs[i]
        if s["end_bar"] - s["start_bar"] + 1 < min_bars:
            if i > 0:
                segs[i - 1]["end_bar"] = s["end_bar"]
                segs.pop(i)
                if i < len(segs) and segs[i]["c"] == segs[i - 1]["c"]:
                    segs[i - 1]["end_bar"] = segs[i]["end_bar"]; segs.pop(i)
                continue
            segs[i + 1]["start_bar"] = s["start_bar"]
            segs.pop(i)
            continue
        i += 1
    letters = {}
    for s in segs:
        letters.setdefault(s["c"], chr(ord("A") + len(letters)))
        s["label"] = letters[s["c"]]
        del s["c"]
    return segs, len(letters)


def sections(mix, beat_frames, bars, beat_times, max_k=10, min_bars=2, target_bars=8):
    """Two levels: `large` (the coarsest split the eigen-gap supports) and `sections`
    (finer: the number of section types whose sections run closest to target_bars)."""
    emb = embed(mix, beat_frames)
    if emb is None:
        one = [{"label": "A", "start_bar": 1, "end_bar": len(bars)}]
        return one, {"large": one}
    evals = emb[0]
    gaps = np.diff(evals[: max_k + 1])
    k_large = int(np.clip(np.argmax(gaps[1:]) + 2, 2, max_k))
    large, _ = _runs(_bar_labels(emb, k_large, bars, beat_times), min_bars)
    best = None
    for k in range(k_large, max_k + 1):
        segs, types = _runs(_bar_labels(emb, k, bars, beat_times), min_bars)
        mean_len = len(bars) / max(len(segs), 1)
        score = abs(np.log(mean_len / target_bars))
        if best is None or score < best[0] - 1e-9:
            best = (score, segs, types, k)
    _, segs, types, k = best
    return segs, {"large": large, "section_types": types, "clusters": k,
                  "method": "laplacian segmentation, k-means seed 0"}


# ---------------------------------------------------------------------------
# Phrases and roles: from letters to intro / verse / chorus / bridge / outro
# ---------------------------------------------------------------------------

def snap_phrases(segs, n_bars):
    """Most sections run 4, 8, 12 or 16 bars; the segmentation often cuts a bar early or
    late. Move a boundary by one bar when that makes the section before it a whole phrase
    (a multiple of 4; 8s preferred) and leaves both sides at least 2 bars. Sections that
    can't be made whole that way are kept as heard and marked irregular."""
    segs = [dict(s) for s in segs]
    for i, s in enumerate(segs):
        s["heard_bars"] = s["end_bar"] - s["start_bar"] + 1
    for i in range(len(segs) - 1):
        a, b = segs[i], segs[i + 1]
        L = a["end_bar"] - a["start_bar"] + 1
        best = None
        for d in (0, -1, 1):
            L2 = L + d
            nb = (b["end_bar"] - b["start_bar"] + 1) - d
            if L2 < 2 or nb < 2 or L2 % 4:
                continue
            rank = (0 if L2 % 8 == 0 else 1, abs(d))
            if best is None or rank < best[0]:
                best = (rank, d)
        if best is not None and best[1]:
            a["end_bar"] += best[1]
            b["start_bar"] += best[1]
    for s in segs:
        L = s["end_bar"] - s["start_bar"] + 1
        s["bars"] = L
        s["phrase"] = "whole" if L % 4 == 0 else "irregular"
    return segs


def roles(segs, energy):
    """Each section's role, from how its type repeats and how loud it is (energy: one
    number per section, e.g. RMS dB). Rules, in order:
    chorus: the repeated type with the most energy; verse: the repeated type that most
    often comes just before a chorus; pre-chorus: a short repeated type sitting between
    verse and chorus; intro / outro: a first / last section of a type heard only there,
    or quieter than the song's middle; bridge: a type heard once, after the second chorus
    and before the last; anything else stays '?'. Confidence: high, medium or low."""
    from collections import Counter, defaultdict
    labs = [s["label"] for s in segs]
    count = Counter(labs)
    by = defaultdict(list)
    for s, e in zip(segs, energy):
        by[s["label"]].append(e)
    mean = {k: float(np.mean(v)) for k, v in by.items()}
    med = float(np.median(energy)) if len(energy) else 0.0
    role = {k: "?" for k in count}
    conf = {k: "low" for k in count}
    repeated = [k for k in count if count[k] >= 2]
    chorus = None
    if repeated:
        ranked = sorted(repeated, key=lambda k: (-mean[k], -count[k]))
        chorus = ranked[0]
        role[chorus] = "chorus"
        margin = mean[chorus] - (mean[ranked[1]] if len(ranked) > 1 else med)
        conf[chorus] = "high" if margin >= 2 else "medium" if margin >= 0.75 else "low"
    if chorus:
        before = Counter(labs[i - 1] for i in range(1, len(labs)) if labs[i] == chorus and labs[i - 1] != chorus)
        # a short repeated type right before the chorus, itself preceded by another repeated type: pre-chorus
        for k, n in before.most_common():
            if k in repeated and k != chorus:
                lens = [s["bars"] for s in segs if s["label"] == k]
                prev = Counter(labs[i - 1] for i in range(1, len(labs)) if labs[i] == k)
                feeder = next((p for p, _ in prev.most_common() if p in repeated and p not in (chorus, k)), None)
                if max(lens) <= 4 and feeder:
                    role[k], conf[k] = "pre-chorus", "medium"
                    role[feeder], conf[feeder] = "verse", "medium" if n >= 2 else "low"
                else:
                    role[k], conf[k] = "verse", "high" if n >= 2 else "medium"
                break
    # intro / outro: first and last sections, if their type isn't verse/chorus
    if segs:
        f, l = labs[0], labs[-1]
        if role.get(f) == "?" and (count[f] == 1 or energy[0] < med):
            role[f], conf[f] = "intro", "high" if count[f] == 1 else "medium"
        if len(segs) > 1 and role.get(l) in ("?", "intro") and (count[l] == 1 or energy[-1] < med):
            if role[l] == "intro":   # the intro's type coming back at the end
                out = "outro (intro returns)"
            else:
                out = "outro"
            role[l + "@end"] = out
    # bridge: a type heard once, after the second chorus and before the last
    if chorus:
        ci = [i for i, x in enumerate(labs) if x == chorus]
        if len(ci) >= 2:
            for i in range(ci[1] + 1, ci[-1]):
                k = labs[i]
                if count[k] == 1 and role[k] == "?":
                    role[k], conf[k] = "bridge", "medium"
    last_chorus = max((i for i, x in enumerate(labs) if x == chorus), default=None) if chorus else None
    out = []
    for i, s in enumerate(segs):
        k = s["label"]
        r, c = role[k], conf[k]
        if i == len(segs) - 1 and (k + "@end") in role:
            r, c = role[k + "@end"], "medium"
        elif r == "?" and last_chorus is not None and i > last_chorus:
            r, c = "outro", "low"   # whatever follows the last chorus is the song ending
        out.append({**s, "role": r, "role_confidence": c})
    return out


def form_line(segs):
    """intro 4 · verse 8 · chorus 8 ... with any irregular length marked (~)."""
    return " · ".join(f"{s.get('role', '?')} {s['bars']}{'~' if s.get('phrase') == 'irregular' else ''}"
                      for s in segs)

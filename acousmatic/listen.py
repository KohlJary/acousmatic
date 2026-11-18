"""One listen: a file in, a listening record out (record.json, record.md, images, score)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from . import SCHEMA, __version__, audio, form, harmony, images, notes, rhythm, score, sound, stems, theory
from .audio import SR

METHODS = {
    "stems": "DSP: mid/side, harmonic/percussive median filtering, REPET-SIM, band crossover",
    "rhythm": "DSP: spectral-flux onsets, dynamic-programming beat tracker, low-band accent meter",
    "key": "DSP: CQT chroma, Krumhansl-Kessler profiles",
    "chords": "DSP: beat-synchronous chroma templates, Viterbi smoothing",
    "form": "DSP: Laplacian structural segmentation, k-means seed 0",
    "lead_and_bass_notes": "DSP: pYIN pitch tracking on the stems, snapped to a 16th grid",
    "drums": "DSP: band-limited onset detection on the percussive stem",
    "theory": "rules over the chords and notes",
    "sound": "measurements: loudness (BS.1770), spectrum, flatness, mid/side and L/R energy",
    "words": "published synced lyrics (LRCLIB), slid to the lead stem's energy; no speech recognition",
}


def _spelled_key(kk):
    _, tonic = harmony.spelling(kk["tonic_pc"], kk["mode"])
    return f"{tonic} {kk['mode']}"


def _bar_of(bars):
    starts = np.array([b["start"] for b in bars])
    def f(t):
        return int(max(1, np.searchsorted(starts, t + 1e-3)))
    return f


def listen(path, out_dir, keep_stems=False, engrave=True, log=print, offline=False):
    t_start = time.time()
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "images").mkdir(exist_ok=True)
    a = audio.load(path)
    log(f"decoded {a.duration:.0f}s")
    st = stems.separate(a.stereo)
    log("stems")
    mix = st.mid
    rh = rhythm.analyse(mix, st.drums)
    beat_frames = librosa.time_to_frames(rh.beats, sr=SR, hop_length=stems.HOP)
    grid = rhythm.grid(rh)
    bars = rh.bars
    bar_of = _bar_of(bars)
    log(f"rhythm: {rh.tempo} bpm, {rh.meter}/4, {len(bars)} bars")

    harm_src = st.bass + st.other
    per_beat, Cs = harmony.chords(harm_src, st.bass, beat_frames)
    spans = harmony.merge(per_beat, rh.beats)
    k = harmony.key_of(Cs.sum(axis=1))
    log(f"harmony: {k['name']}, {len(spans)} chord spans")

    secs, form_info = form.sections(mix, beat_frames, bars, rh.beats)
    log(f"form: {''.join(s['label'] for s in secs)}")

    lead_notes, lead_voiced = notes.lead_line(st.lead, grid)
    bass_notes, _ = notes.bass_line(st.bass, grid)
    drum_hits = notes.drums(st.drums, grid)
    log(f"notes: lead {len(lead_notes)}, bass {len(bass_notes)}, drum hits {sum(len(v) for v in drum_hits.values())}")

    from . import words as W
    wd = W.words(a.tags, a.duration, a.sha256, lead=st.lead, sr=SR, offline=offline)
    for l in wd.get("lines", []):
        l["bar"] = bar_of(l["start"])
    log(f"words: {len(wd.get('lines', []))} lines ({wd.get('source')})")

    # spell everything by the key: flats in flat keys, sharps in sharp ones
    # settle the key by where the bass dwells (relative major/minor are near-ties on chroma)
    bw = np.zeros(12)
    for n in bass_notes:
        bw[n["midi"] % 12] += n["end"] - n["start"]
    if bw.sum() > 0:
        k = harmony.key_of(Cs.sum(axis=1), bass_weights=bw)
    names, tonic = harmony.spelling(k["tonic_pc"], k["mode"])
    k["tonic"], k["name"] = tonic, f"{tonic} {k['mode']}"
    for c in spans:
        if c["root"] is not None:
            c["name"] = names[c["root"]] + (c["quality"] or "")
    for n in lead_notes + bass_notes:
        n["name"] = harmony.note_name(n["midi"], names)

    for c in spans:
        c["roman"] = theory.roman(c["root"], c["quality"], k["tonic_pc"], k["mode"])
        c["borrowed"] = theory.borrowed(c["root"], c["quality"], k["tonic_pc"], k["mode"])
        c["bar"] = bar_of(c["start"])
    mode = theory.mode_of(Cs.sum(axis=1), k["tonic_pc"])

    field = sound.Field(a.stereo)
    ref_rms = float(np.sqrt(np.mean(mix ** 2)))
    stem_map = st.as_dict()
    prev = prev_st = None
    for s in secs:
        t0 = bars[s["start_bar"] - 1]["start"]; t1 = bars[s["end_bar"] - 1]["end"]
        s["start"], s["end"] = t0, t1
        s["bars"] = s["end_bar"] - s["start_bar"] + 1
        s["sound"] = field.section(t0, t1)
        s["stems"] = {n: sound.stem_section(y, t0, t1, ref_rms) for n, y in stem_map.items()}
        s["stem_character"] = {n: sound.character(n, d) for n, d in s["stems"].items()}
        s["changes"] = sound.changes(prev, s["sound"], prev_st, s["stems"]) if s["sound"] else []
        prev, prev_st = s["sound"], s["stems"]
        ss = [c for c in spans if t0 - 1e-3 <= c["start"] < t1 - 1e-3]
        names = [c["roman"] for c in ss]
        s["theory"] = {
            "key": _spelled_key(harmony.key_of(Cs[:, [i for i, t in enumerate(rh.beats[:Cs.shape[1]]) if t0 <= t < t1] or [0]].sum(axis=1))),
            "loops": theory.loops(names),
            "harmonic_rhythm_beats_per_chord": theory.harmonic_rhythm(ss),
            "melody": theory.melody([n for n in lead_notes if t0 <= n["start"] < t1], k["tonic_pc"], k["mode"]),
            "syncopation_lead": theory.syncopation([n["g0"] for n in lead_notes if t0 <= n["start"] < t1]),
            "lead_present": bool([n for n in lead_notes if t0 <= n["start"] < t1]),
        }
        img = f"images/section-{s['start_bar']:03d}-{s['label']}.png"
        images.section_stems(stem_map, t0, t1, out / img, f"{s['label']}  bars {s['start_bar']}–{s['end_bar']}")
        s["image"] = img
    log("sound and theory")

    title = a.tags.get("title") or a.path.stem
    images.overview(mix, secs, bars, out / "images/overview.png", f"{title}: spectrogram and sections")
    images.stereo_field(field, secs, bars, out / "images/stereo-field.png", f"{title}: stereo field")

    rec = {
        "schema": SCHEMA, "acousmatic": __version__,
        "source": {"file": a.path.name, "path": str(a.path), "sha256": a.sha256, "duration": round(a.duration, 2),
                   "tags": a.tags},
        "methods": METHODS,
        "stems": st.bleed,
        "loudness": sound.loudness(a.stereo),
        "rhythm": {"tempo": rh.tempo, "meter": rh.meter, "meter_contrast": rh.meter_contrast,
                   "downbeat_index": rh.downbeat_index, "beats": [round(float(t), 3) for t in rh.beats],
                   "bars": bars, "local_tempo": rh.local_tempo, "carried_beats": rh.carried,
                   "tempo_candidates": rh.tempo_candidates},
        "harmony": {"key": k, "mode": mode, "chord_spans": spans,
                    "cadences": theory.cadences(spans, k["tonic_pc"], k["mode"], bar_of)},
        "form": {"sections": secs, **form_info},
        "rhythm_phase_shifted": rh.phase_shifted,
        "notes": {"lead": lead_notes, "bass": bass_notes, "lead_voiced_share": round(lead_voiced, 3)},
        "drums": drum_hits,
        "words": wd,
        "images": {"overview": "images/overview.png", "stereo_field": "images/stereo-field.png"},
    }
    if engrave:
        try:
            rec["score"] = score.build(rec, out)
        except Exception as e:  # the record stands without the score
            rec["score"] = {"error": f"{type(e).__name__}: {e}"}
    if keep_stems:
        (out / "stems").mkdir(exist_ok=True)
        for n, y in stem_map.items():
            sf.write(out / "stems" / f"{n}.flac", y, SR)
    rec["elapsed_s"] = round(time.time() - t_start, 1)
    (out / "record.json").write_text(json.dumps(rec, indent=1, default=float))
    from .text import render
    (out / "record.md").write_text(render(rec))
    log(f"done in {rec['elapsed_s']}s -> {out}")
    return rec

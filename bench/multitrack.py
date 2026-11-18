"""Measure acousmatic against songs released as multitracks.

The mix is the sum of the stems, so it lines up with them exactly. Truth:
- tempo, beats, bar lines and section boundaries from the song's tempo/marker MIDI;
- stems: the isolated lead, bass and drum tracks;
- drum hits and lead/bass notes: the same detectors run on the isolated tracks, so the
  score measures what the mix and the separation cost (not the detectors themselves).

Usage: python bench/multitrack.py SONG.json [--out DIR]
SONG.json: {"name", "dir", "midi", "roles": {"lead": [...], "bass": [...], "drums": [...]}, "beats_per_bar": 4}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import librosa
import mido
import mir_eval
import numpy as np

from acousmatic import audio, form, notes, rhythm, stems
from acousmatic.audio import SR


def truth_from_midi(path, duration):
    m = mido.MidiFile(path)
    tpb = m.ticks_per_beat
    tempo, markers = None, []
    for tr in m.tracks:
        tick = 0
        for msg in tr:
            tick += msg.time
            if msg.is_meta and msg.type == "set_tempo" and tempo is None:
                tempo = mido.tempo2bpm(msg.tempo)
            if msg.is_meta and msg.type == "marker":
                markers.append((tick / tpb, msg.text))
    spb = 60.0 / tempo
    beats = np.arange(0, duration, spb)
    return tempo, beats, [(round(b) * spb, name) for b, name in markers]


def load_stems(d, names):
    ys = []
    for n in names:
        ys.append(audio.decode(Path(d) / n))
    n = min(y.shape[1] for y in ys)
    return sum(y[:, :n] for y in ys)


def snr(ref, est):
    n = min(len(ref), len(est))
    r, e = ref[:n], est[:n]
    return round(float(10 * np.log10((r ** 2).sum() / (((r - e) ** 2).sum() + 1e-12) + 1e-12)), 2)


def notes_f(ref, est):
    if not ref or not est:
        return None
    def arr(ns):
        iv = np.array([[n["start"], max(n["end"], n["start"] + 0.05)] for n in ns])
        return iv, librosa.midi_to_hz(np.array([n["midi"] for n in ns]))
    ri, rp = arr(ref); ei, ep = arr(est)
    p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(ri, rp, ei, ep, offset_ratio=None,
                                                                    onset_tolerance=0.1, pitch_tolerance=50)
    return {"p": round(p, 3), "r": round(r, 3), "f": round(f, 3), "ref_notes": len(ref), "est_notes": len(est)}


def run(cfg):
    d = Path(cfg["dir"])
    files = sorted(p.name for p in d.iterdir() if p.suffix.lower() == ".flac")
    roles = {k: v for k, v in cfg["roles"].items()}
    mix_hi = load_stems(d, files)
    peak = np.abs(mix_hi).max()
    mix_hi = mix_hi / max(peak, 1.0)
    duration = mix_hi.shape[1] / audio.SR_HI
    tempo_true, beats_true, markers = truth_from_midi(cfg["midi"], duration)

    st = stems.separate(mix_hi)
    def ref_mono(names):
        y = load_stems(d, names) / max(peak, 1.0)
        y = librosa.resample(y, orig_sr=audio.SR_HI, target_sr=SR, res_type="soxr_hq")
        return (y[0] + y[1]) / 2
    ref = {k: ref_mono(v) for k, v in roles.items()}

    out = {"song": cfg["name"], "tempo_true": round(tempo_true, 2)}
    out["stems_snr_db"] = {k: snr(ref[k], getattr(st, k)) for k in ("lead", "bass", "drums")}
    # a zero estimate scores 0 dB; above 0 means the stem carries more of the part than of the rest

    rh = rhythm.analyse(st.mid, st.drums)
    t = rh.tempo
    ratio = t / tempo_true
    out["tempo"] = {"est": t, "ratio": round(ratio, 3), "octave_ok": any(abs(ratio - r) < 0.04 for r in (0.5, 1, 2))}
    out["beats_f"] = round(float(mir_eval.beat.f_measure(mir_eval.beat.trim_beats(beats_true),
                                                       mir_eval.beat.trim_beats(rh.beats))), 3)
    bpb = cfg.get("beats_per_bar", 4)
    db_true = beats_true[::bpb]
    db_est = np.array([b["start"] for b in rh.bars])
    out["downbeats_f"] = round(float(mir_eval.onset.f_measure(db_true, db_est, window=0.1)[0]), 3)
    out["meter"] = {"est": rh.meter, "true": bpb, "phase_shifted": rh.phase_shifted}

    bf = librosa.time_to_frames(rh.beats, sr=SR, hop_length=stems.HOP)
    secs, info = form.sections(st.mid, bf, rh.bars, rh.beats)
    b_est = np.array([rh.bars[s["start_bar"] - 1]["start"] for s in secs[1:]])
    b_true = np.array([m[0] for m in markers if 1.0 < m[0] < duration - 1])
    if len(b_est) and len(b_true):
        iv = lambda b: np.column_stack([np.concatenate([[0], b]), np.concatenate([b, [duration]])])
        p, r, f = mir_eval.segment.detection(iv(b_true), iv(b_est), window=3.0, trim=True)
        out["sections"] = {"p": round(p, 3), "r": round(r, 3), "f": round(f, 3), "est": len(secs), "true": len(markers),
                           "large": len(info.get("large", []))}

    grid = rhythm.grid(rh)
    d_est = notes.drums(st.drums, grid)
    d_ref = notes.drums(ref["drums"], grid)
    out["drums_f"] = {}
    for k in d_ref:
        tr = grid[np.array(d_ref[k], dtype=int)] if d_ref[k] else np.array([])
        te = grid[np.array(d_est[k], dtype=int)] if d_est[k] else np.array([])
        out["drums_f"][k] = round(float(mir_eval.onset.f_measure(tr, te, window=0.05)[0]), 3) if len(tr) and len(te) else 0.0
        out["drums_f"][k + "_counts"] = [len(tr), len(te)]
    lead_est, _ = notes.lead_line(st.lead, grid); lead_ref, _ = notes.lead_line(ref["lead"], grid)
    bass_est, _ = notes.bass_line(st.bass, grid); bass_ref, _ = notes.bass_line(ref["bass"], grid)
    out["lead_notes"] = notes_f(lead_ref, lead_est)
    out["bass_notes"] = notes_f(bass_ref, bass_est)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("song", nargs="+")
    a = ap.parse_args()
    for s in a.song:
        print(json.dumps(run(json.load(open(s))), indent=1))
        sys.stdout.flush()

"""A score: the transcribed lead and bass lines with chord symbols, as MusicXML, MIDI
(with the drums) and engraved pages (LilyPond). It is a transcription, not the
composer's score."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from music21 import (clef, harmony, instrument, key as m21key, meter as m21meter, metadata, note, stream,
                     tempo as m21tempo)

DRUM_MIDI = {"kick": 36, "snare": 38, "hat": 42}


def _part(notes, g_db, name, inst, clef_obj=None, per_beat=4):
    p = stream.Part()
    p.partName = name
    p.insert(0, inst)
    if clef_obj:
        p.insert(0, clef_obj)
    for n in notes:
        off = (n["g0"] - g_db) / per_beat
        dur = (n["g1"] - n["g0"]) / per_beat
        if off < 0 or dur <= 0:
            continue
        x = note.Note(n["midi"])
        x.quarterLength = dur
        p.insert(off, x)
    return p


def build(rec, out: Path):
    out = Path(out).resolve()
    g_db = (rec["rhythm"]["downbeat_index"] + 1) * 4
    meter = rec["rhythm"]["meter"]
    s = stream.Score()
    s.insert(0, metadata.Metadata(title=rec["source"]["tags"].get("title", rec["source"]["file"]),
                                  composer=rec["source"]["tags"].get("artist", "")))
    lead = _part(rec["notes"]["lead"], g_db, "Lead (transcribed)", instrument.Vocalist())
    bass = _part(rec["notes"]["bass"], g_db, "Bass (transcribed)", instrument.ElectricBass(), clef.BassClef())
    beat0 = rec["rhythm"]["beats"][rec["rhythm"]["downbeat_index"]]
    beats = rec["rhythm"]["beats"]
    for c in rec["harmony"]["chord_spans"]:
        if c["name"] == "N":
            continue
        try:
            bi = beats.index(c["start"])
        except ValueError:
            continue
        off = bi - rec["rhythm"]["downbeat_index"]
        if off < 0:
            continue
        try:
            import re as _re
            cs = harmony.ChordSymbol(_re.sub(r"^([A-G])b", r"\1-", c["name"]))
            cs.writeAsChord = False
            lead.insert(float(off), cs)
        except Exception:
            pass
    for p in (lead, bass):
        p.insert(0, m21meter.TimeSignature(f"{meter}/4"))
        k = rec["harmony"]["key"]
        p.insert(0, m21key.Key(k["tonic"][0] + k["tonic"][1:].replace("b", "-"),
                               "minor" if k["mode"] == "minor" else "major"))
    lead.insert(0, m21tempo.MetronomeMark(number=round(rec["rhythm"]["tempo"])))
    for p in (lead, bass):
        p.makeRests(fillGaps=True, inPlace=True)
        s.insert(0, p)
    s = s.makeNotation()
    xml = out / "score.musicxml"
    s.write("musicxml", fp=str(xml))

    # MIDI, with drums on channel 10, for listening back to the transcription
    ms = stream.Score()
    ms.insert(0, m21tempo.MetronomeMark(number=round(rec["rhythm"]["tempo"])))
    ms.insert(0, _part(rec["notes"]["lead"], g_db, "lead", instrument.Piano()))
    ms.insert(0, _part(rec["notes"]["bass"], g_db, "bass", instrument.ElectricBass()))
    d = stream.Part(); di = instrument.Percussion(); di.midiChannel = 9; d.insert(0, di)
    for name, slots in rec["drums"].items():
        for g in slots:
            off = (g - g_db) / 4
            if off < 0:
                continue
            x = note.Note(DRUM_MIDI[name]); x.quarterLength = 0.25
            d.insert(off, x)
    ms.insert(0, d)
    ms.write("midi", fp=str(out / "transcription.mid"))

    pages = []
    if shutil.which("musicxml2ly") and shutil.which("lilypond"):
        ly = out / "score.ly"
        subprocess.run(["musicxml2ly", "--no-beaming", "-o", str(ly), str(xml)], capture_output=True, timeout=300)
        if ly.exists():
            subprocess.run(["lilypond", "--png", "-dresolution=100", "-o", str(out / "score"), str(ly)],
                           capture_output=True, timeout=600, cwd=out)
            pages = sorted(p.name for p in out.glob("score*.png"))
    return {"musicxml": xml.name, "midi": "transcription.mid", "pages": pages}

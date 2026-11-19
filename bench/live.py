"""Live ears against the full listen, on the same file.

    python bench/live.py song.flac OUT_DIR

Runs the full listen (unless OUT_DIR/full/record.json is already there) and the
live ears on the file as fast as it decodes (same code as real time, just not
waiting), then compares what the live ears heard with what the full listen
heard: key, tempo, and, second by second, the chord: same root, and the exact
same chord. The full listen isn't ground truth (it has no chord charts either);
this measures how much the live ears give up by hearing the song once, in order.
Prints one JSON line.
"""
import json
import re
import sys
from pathlib import Path

PC = {"C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5, "F#": 6, "Gb": 6,
      "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11}


def root(name):
    m = re.match(r"([A-G][#b]?)", name or "")
    return PC.get(m.group(1)) if m else None


def full_chord_at(spans, t):
    for s in spans:
        if s["start"] <= t < s["end"]:
            return s["name"]
    return None


def live_chord_at(timeline, t):
    cur = None
    for c in timeline:
        if c["t"] > t:
            break
        cur = c["chord"]
    return cur


def main(path, out):
    out = Path(out)
    full_rec = out / "full" / "record.json"
    if not full_rec.exists():
        from acousmatic.listen import listen
        listen(path, out / "full", engrave=False, offline=True, log=lambda *a: None)
    from acousmatic.live import listen_live
    listen_live("file", out / "live", path=path, realtime=False, log=lambda *a: None)
    full = json.loads(full_rec.read_text())
    live = json.loads((out / "live" / "live.json").read_text())

    spans = full["harmony"]["chord_spans"]
    end = min(spans[-1]["end"], live["duration_s"])
    secs = [t + 0.5 for t in range(int(end))]
    pairs = [(full_chord_at(spans, t), live_chord_at(live["chords"], t)) for t in secs]
    pairs = [(f, l) for f, l in pairs if f and l]
    same_root = sum(root(f) == root(l) for f, l in pairs) / len(pairs)
    same_chord = sum(f == l for f, l in pairs) / len(pairs)
    print(json.dumps({
        "file": Path(path).name,
        "key": {"full": full["harmony"]["key"]["name"], "live": live["key"]["name"]},
        "tempo": {"full": full["rhythm"]["tempo"], "live": live["tempo"]},
        "seconds_compared": len(pairs),
        "chord_root_agrees": round(same_root, 3),
        "exact_chord_agrees": round(same_chord, 3),
    }))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

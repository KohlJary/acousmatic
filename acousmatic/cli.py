"""acousmatic listen <file> [-o DIR] [--stems] [--no-score]"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser(prog="acousmatic", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    li = sub.add_parser("listen", help="make a listening record of a song")
    li.add_argument("file")
    li.add_argument("-o", "--out", help="output directory (default: records/<file stem>)")
    li.add_argument("--stems", action="store_true", help="also write the separated stems as FLAC")
    li.add_argument("--no-score", action="store_true", help="skip MusicXML/MIDI/engraving")
    li.add_argument("--offline", action="store_true", help="don't look up published lyrics")
    lv = sub.add_parser("live", help="listen as it plays: the computer's output, or a file at real speed")
    lv.add_argument("--file", help="play this file at real speed (else: what the computer is playing)")
    lv.add_argument("-o", "--out", required=True)
    lv.add_argument("--full", action="store_true", help="also a full record at the end (audio held in memory only)")
    lv.add_argument("--max", type=float, help="stop after this many seconds")
    lv.add_argument("--fast", action="store_true", help="a file as fast as it decodes, not real speed (testing)")
    lv.add_argument("--title"); lv.add_argument("--artist")
    lv.add_argument("--events", action="store_true", help="print each second's listening as a JSON line")
    a = ap.parse_args(argv)
    if a.cmd == "live":
        import json
        from .live import listen_live
        meta = {k: v for k, v in (("title", a.title), ("artist", a.artist)) if v}
        emit = (lambda e: print(json.dumps(e), flush=True)) if a.events else None
        listen_live("file" if a.file else "monitor", Path(a.out), path=a.file, meta=meta, full=a.full,
                    max_s=a.max, realtime=not a.fast, on_event=emit, log=lambda m: print(m, file=sys.stderr))
        return
    if a.cmd == "listen":
        from .listen import listen
        out = Path(a.out) if a.out else Path("records") / Path(a.file).stem
        listen(a.file, out, keep_stems=a.stems, engrave=not a.no_score, offline=a.offline,
               log=lambda m: print(m, file=sys.stderr))


if __name__ == "__main__":
    main()

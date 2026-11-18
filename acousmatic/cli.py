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
    a = ap.parse_args(argv)
    if a.cmd == "listen":
        from .listen import listen
        out = Path(a.out) if a.out else Path("records") / Path(a.file).stem
        listen(a.file, out, keep_stems=a.stems, engrave=not a.no_score, log=lambda m: print(m, file=sys.stderr))


if __name__ == "__main__":
    main()

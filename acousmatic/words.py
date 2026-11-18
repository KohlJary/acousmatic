"""Words: the lyrics, placed in time.

Published synced lyrics from LRCLIB (lrclib.net, an open lyrics database), looked up
by artist, title and length. A sync is timed to one release of the song; this file
may start a little earlier or later, so the lines are slid (within two seconds) to
where the separated lead's energy best matches when lines are being sung. No speech
recognition: if nothing is published, the record says there are no words for it.

Lyrics are copyright their writers. They are for listening, not for republishing.
"""
from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np

API = "https://lrclib.net/api"
UA = "acousmatic (https://github.com/KohlJary/acousmatic)"
CACHE = Path.home() / ".cache" / "acousmatic" / "lyrics"
LRC = re.compile(r"\[(\d+):(\d+(?:\.\d+)?)\]\s*(.*)")


def _get(path, params):
    url = f"{API}/{path}?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read().decode())
    except Exception:
        return None


def lookup(artist: str, title: str, duration: float) -> Optional[dict]:
    if not artist or not title:
        return None
    got = _get("get", {"artist_name": artist, "track_name": title, "duration": int(round(duration))})
    if got and (got.get("syncedLyrics") or got.get("instrumental")):
        return got
    found = _get("search", {"artist_name": artist, "track_name": title}) or []
    synced = [x for x in found if x.get("syncedLyrics") and abs((x.get("duration") or 0) - duration) <= 4]
    synced.sort(key=lambda x: abs((x.get("duration") or 0) - duration))
    return synced[0] if synced else (got or None)


def parse(lrc: str):
    lines = []
    for raw in (lrc or "").splitlines():
        m = LRC.match(raw.strip())
        if m:
            t = int(m.group(1)) * 60 + float(m.group(2))
            lines.append({"start": round(t, 2), "text": m.group(3).strip()})
    lines = [l for l in lines if l["text"]] if lines else []
    for i, l in enumerate(lines):
        l["end"] = lines[i + 1]["start"] if i + 1 < len(lines) else round(l["start"] + 4.0, 2)
    return lines


def align(lines, lead, sr, max_shift=2.0, step=0.05):
    """The offset (seconds) that best lines up 'a line is being sung' with the lead's energy."""
    if not lines or lead is None or not len(lead):
        return 0.0
    hop = int(sr * step)
    env = np.sqrt(np.convolve(lead ** 2, np.ones(hop) / hop, mode="same")[::hop])
    env = (env - env.mean()) / (env.std() + 1e-9)
    n = len(env)
    best, best_score = 0.0, -1e9
    for k in range(int(-max_shift / step), int(max_shift / step) + 1):
        ind = np.zeros(n)
        for l in lines:
            a, b = int((l["start"] + k * step) / step), int((min(l["end"], l["start"] + 3.5) + k * step) / step)
            ind[max(0, a):max(0, min(n, b))] = 1.0
        score = float((ind * env).sum())
        if score > best_score:
            best, best_score = k * step, score
    return round(best, 2)


def words(tags: dict, duration: float, sha: str, lead=None, sr=None, offline=False) -> dict:
    CACHE.mkdir(parents=True, exist_ok=True)
    cached = CACHE / f"{sha}.json"
    got = None
    if cached.exists():
        got = json.loads(cached.read_text())
    elif not offline:
        got = lookup(tags.get("artist") or tags.get("album_artist") or "", tags.get("title") or "", duration)
        if got is not None:
            cached.write_text(json.dumps(got))
    if not got:
        return {"source": None, "lines": [], "note": "no published lyrics found" if not offline else "offline"}
    if got.get("instrumental"):
        return {"source": "lrclib", "instrumental": True, "lines": []}
    lines = parse(got.get("syncedLyrics") or "")
    if not lines:
        return {"source": "lrclib (unsynced)", "lines": [], "plain": got.get("plainLyrics") or ""}
    off = align(lines, lead, sr) if lead is not None else 0.0
    for l in lines:
        l["start"] = round(l["start"] + off, 2)
        l["end"] = round(l["end"] + off, 2)
    return {"source": "lrclib (published, synced)", "offset_s": off, "lines": lines,
            "release": {"artist": got.get("artistName"), "album": got.get("albumName"), "duration": got.get("duration")}}

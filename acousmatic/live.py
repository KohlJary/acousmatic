"""Live ears: listening as the song plays, the way a person does.

Audio comes in as it plays (a stream from the system's output, or a file played at real
speed). Every second the ears say what's happening now: the chord, the key so far, the
tempo, how loud and how bright, and when the music turns a corner into a new section.
When the song ends, a light record is written: form (with roles), key, tempo, the chord
timeline, the sound over time. No audio is written anywhere. It's held in memory only
as long as the listening needs it, then gone.

With full=True the song is kept in memory until it ends, the complete listen (stems,
lines, drums, score) runs on it, and then the audio is dropped. Same rule: a record, never
a copy.
"""
from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path
from typing import Callable, Dict, List, Optional

import librosa
import numpy as np

from . import form, harmony
from .audio import SR, SR_HI

HOP = 512
STEP_S = 1.0          # how often the ears report
WIN_S = 8.0           # what they look back over for tempo and turns
TURN_GAP_S = 8.0      # sections shorter than this aren't called


class LiveEars:
    def __init__(self, emit: Optional[Callable[[Dict], None]] = None, full: bool = False):
        self.emit = emit or (lambda e: None)
        self.full = full
        self.ring = deque()          # recent mono audio at SR, at most WIN_S
        self.ring_n = 0
        self.pending = np.zeros(0)
        self.keep: List[np.ndarray] = []   # whole song in memory only when full
        self.t = 0.0
        self.chroma_sum = np.zeros(12)
        self.events: List[Dict] = []
        self.feats: List[np.ndarray] = []  # per-second timbre+harmony vectors, for turns
        self.turns: List[float] = [0.0]
        self.tempos: List[float] = []
        self.last_chord: Optional[int] = None
        self.bass_sum = np.zeros(12)

    # -- input ---------------------------------------------------------------
    def feed(self, stereo_hi: np.ndarray) -> None:
        """A block of stereo audio at SR_HI, (2, n), as it arrives."""
        if self.full:
            self.keep.append(stereo_hi.astype(np.float32).copy())
        mono = librosa.resample(stereo_hi.mean(axis=0).astype(np.float32), orig_sr=SR_HI, target_sr=SR)
        self.pending = np.concatenate([self.pending, mono])
        step = int(STEP_S * SR)
        while len(self.pending) >= step:
            chunk, self.pending = self.pending[:step], self.pending[step:]
            self.ring.append(chunk)
            self.ring_n += len(chunk)
            while self.ring_n > WIN_S * SR and len(self.ring) > 1:
                self.ring_n -= len(self.ring.popleft())
            self.t += STEP_S
            self._second(chunk)

    # -- one second of listening ------------------------------------------------
    def _second(self, chunk: np.ndarray) -> None:
        win = np.concatenate(list(self.ring))
        rms = float(np.sqrt(np.mean(chunk ** 2)) + 1e-12)
        level_db = 20 * np.log10(rms)
        ev = {"t": round(self.t, 1), "level_db": round(level_db, 1)}
        if level_db < -55:
            ev["silent"] = True
            self.events.append(ev)
            self.emit(ev)
            return
        # as the full listen does it: the pitched part for the chord (drums smear the pitch
        # picture), the bass for the root and the key
        seg = win[-int(2.0 * SR):]
        harm = librosa.effects.harmonic(seg, margin=2.0)
        C = harmony.chroma(harm)
        c_now = C[:, -int(1.0 * SR / HOP):].mean(axis=1)
        from scipy.signal import butter, sosfilt
        low = sosfilt(butter(4, 250 / (SR / 2), output="sos"), harm)
        B = harmony.bass_pc(low)
        b_now = B[:, -int(1.0 * SR / HOP):].mean(axis=1)
        self.chroma_sum += c_now
        self.bass_sum += b_now
        x = c_now / (np.linalg.norm(c_now) + 1e-9)
        raw = harmony.TEMPLATES @ x
        Q = harmony.QUALITIES
        score = (raw + 0.3 * np.array([b_now[r] / (b_now.max() + 1e-9) for r, _ in harmony.LABELS])
                 - 0.06 * np.array([len(Q[q]) < 3 for _, q in harmony.LABELS])
                 - 0.05 * np.array([len(Q[q]) > 3 for _, q in harmony.LABELS])
                 - 0.03 * np.array([q.startswith("sus") for _, q in harmony.LABELS]))
        i = int(np.argmax(score))
        # hysteresis: the chord changes only when the new one clearly wins (live smoothing)
        if self.last_chord is not None and score[i] - score[self.last_chord] < 0.04:
            i = self.last_chord
        self.last_chord = i
        root, q = harmony.LABELS[i]
        key = harmony.key_of(self.chroma_sum, bass_weights=self.bass_sum)
        ev["chord"] = harmony.chord_name(root, q)
        ev["clear"] = bool(raw[i] >= harmony.FIT_CLEAR)   # as the full listen marks it
        ev["key_so_far"] = key["name"]
        cen = librosa.feature.spectral_centroid(y=chunk, sr=SR)
        ev["brightness_hz"] = int(np.median(cen))
        if len(win) >= 4 * SR:
            tempo = float(np.atleast_1d(librosa.beat.beat_track(y=win, sr=SR, hop_length=HOP)[0])[0])
            if tempo > 0:
                self.tempos.append(tempo)
                ev["tempo"] = round(float(np.median(self.tempos[-12:])), 1)
        # what this second sounds like, for noticing when the music turns
        mf = librosa.feature.mfcc(y=chunk, sr=SR, n_mfcc=13).mean(axis=1)
        self.feats.append(np.concatenate([mf / (np.linalg.norm(mf) + 1e-9), c_now / (np.linalg.norm(c_now) + 1e-9),
                                          [level_db / 60.0]]))
        n = len(self.feats)
        if n >= 8 and self.t - self.turns[-1] >= TURN_GAP_S:
            before = np.mean(self.feats[-8:-4], axis=0)
            after = np.mean(self.feats[-4:], axis=0)
            d = float(np.linalg.norm(after - before))
            hist = [float(np.linalg.norm(np.mean(self.feats[j - 4:j], axis=0) - np.mean(self.feats[j - 8:j - 4], axis=0)))
                    for j in range(8, n)]
            thr = (np.median(hist) + 2.0 * np.std(hist)) if len(hist) > 6 else 0.35
            if d > max(thr, 0.2):
                at = round(self.t - 4.0, 1)
                self.turns.append(at)
                ev["turn"] = at
        self.events.append(ev)
        self.emit(ev)

    # -- the end of the song -------------------------------------------------------
    def finish(self, out_dir: Path, meta: Optional[Dict] = None, log=print) -> Dict:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        meta = meta or {}
        sounding = [e for e in self.events if not e.get("silent")]
        dur = self.t
        tempo = round(float(np.median(self.tempos)), 1) if self.tempos else None
        if not sounding:
            secs_note = "nothing was playing"
        key = harmony.key_of(self.chroma_sum, bass_weights=self.bass_sum) if self.chroma_sum.any() else None
        # sections between the turns, in bars when the tempo is known
        bounds = self.turns + [dur]
        bar_s = (4 * 60.0 / tempo) if tempo else None
        secs = []
        for a, b in zip(bounds[:-1], bounds[1:]):
            if b - a < 2:
                continue
            evs = [e for e in sounding if a <= e["t"] < b]
            lvl = float(np.mean([e["level_db"] for e in evs])) if evs else -60.0
            bars = max(1, int(round((b - a) / bar_s))) if bar_s else None
            secs.append({"start": round(a, 1), "end": round(b, 1), "bars": bars or 0, "level_db": round(lvl, 1),
                         "label": "?"})
        # letters by how alike the sections sound, then roles by the same rules as a full listen
        if secs:
            vecs = []
            for s in secs:
                i0, i1 = int(s["start"]), max(int(s["start"]) + 1, int(s["end"]))
                fs = self.feats[i0:i1] or self.feats[-1:]
                vecs.append(np.mean(fs, axis=0))
            letters, protos = [], []
            for v in vecs:
                d = [float(np.linalg.norm(v - p)) for p in protos]
                if d and min(d) < 0.35:
                    letters.append(chr(65 + int(np.argmin(d))))
                else:
                    protos.append(v)
                    letters.append(chr(65 + len(protos) - 1))
            for s, l in zip(secs, letters):
                s["label"] = l
                s["phrase"] = "whole" if s["bars"] and s["bars"] % 4 == 0 else "irregular"
            secs = form.roles(secs, [s["level_db"] for s in secs])
        # the chord timeline, merged
        chords = []
        for e in sounding:
            c = e.get("chord")
            if c and (not chords or chords[-1]["chord"] != c):
                chords.append({"t": e["t"] - STEP_S, "chord": c})
        rec = {"schema": "acousmatic-live-1", "heard": "live, as it played; no audio kept", "meta": meta,
               "duration_s": round(dur, 1), "tempo": tempo, "key": key, "sections": secs, "chords": chords,
               "sound": [{"t": e["t"], "level_db": e["level_db"], "brightness_hz": e.get("brightness_hz")}
                         for e in self.events],
               "turns": self.turns[1:]}
        (out_dir / "live.json").write_text(json.dumps(rec, indent=1))
        (out_dir / "live.md").write_text(render_light(rec))
        if self.full and self.keep:
            log("full record: listening again to the song held in memory")
            from .listen import listen_audio
            stereo = np.concatenate(self.keep, axis=1)
            self.keep = []   # the song leaves memory as soon as the full listen has it
            listen_audio(stereo, out_dir / "full", meta=meta, log=log)
            del stereo
        return rec


def render_light(rec: Dict) -> str:
    m = rec.get("meta") or {}
    k = rec.get("key") or {}
    title = m.get("title") or "what was playing"
    L = [f"# {title}", "", f"{m.get('artist', '')}".strip(), "",
         "Heard live, as it played. No audio was kept; this is what I remember.", "",
         "## At a glance"]
    L.append(f"- {int(rec['duration_s'] // 60)}:{int(rec['duration_s'] % 60):02d} long"
             + (f", about {rec['tempo']} BPM" if rec.get("tempo") else ""))
    if k:
        L.append(f"- Key: {k.get('name')} (strength {k.get('strength')})")
    if rec["sections"]:
        L.append("- Form: " + " · ".join(f"{s['role']} {s['bars']}" for s in rec["sections"])
                 + " (bars, estimated from the tempo; turns heard as the music changed)")
    L += ["", "## Sections"]
    for s in rec["sections"]:
        ch = [c["chord"] for c in rec["chords"] if s["start"] <= c["t"] < s["end"] and c["chord"] != "?"]
        common = [c for c, _ in sorted(((c, ch.count(c)) for c in set(ch)), key=lambda x: -x[1])[:4]]
        L.append(f"- {s['label']} ({s['role']}) {_mmss(s['start'])}-{_mmss(s['end'])}, level {s['level_db']} dB"
                 + (f"; chords mostly {', '.join(common)}" if common else ""))
    return "\n".join(L) + "\n"


def _mmss(t):
    return f"{int(t // 60)}:{int(t % 60):02d}"


# ---------------------------------------------------------------------------
# sources: the system's output, or a file at real speed
# ---------------------------------------------------------------------------

def stream(source: str, path: Optional[str] = None, realtime: bool = True, block_s: float = 0.25):
    """Yield (2, n) float blocks at SR_HI. source: 'file' (played at real speed with
    ffmpeg -re) or 'monitor' (what the computer is playing, via PipeWire/PulseAudio)."""
    import subprocess
    if source == "file":
        cmd = ["ffmpeg", "-loglevel", "error"] + (["-re"] if realtime else []) + ["-i", str(path), "-f", "s16le",
                                                                                   "-ac", "2", "-ar", str(SR_HI), "-"]
    else:
        sink = subprocess.run(["pactl", "get-default-sink"], capture_output=True, text=True).stdout.strip()
        cmd = ["parec", "-d", f"{sink}.monitor", "--format=s16le", "--channels=2", f"--rate={SR_HI}", "--raw"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    n = int(block_s * SR_HI) * 4
    try:
        while True:
            b = p.stdout.read(n)
            if not b:
                break
            x = np.frombuffer(b, dtype=np.int16).astype(np.float32) / 32768.0
            yield x.reshape(-1, 2).T
    finally:
        p.kill()


def listen_live(source: str, out_dir: Path, path: Optional[str] = None, meta: Optional[Dict] = None,
                full: bool = False, max_s: Optional[float] = None, quiet_end_s: float = 8.0,
                realtime: bool = True, on_event=None, log=print) -> Dict:
    ears = LiveEars(emit=on_event, full=full)
    quiet = 0.0
    t0 = time.time()
    for block in stream(source, path, realtime=realtime):
        ears.feed(block)
        lvl = 20 * np.log10(np.sqrt(np.mean(block ** 2)) + 1e-12)
        quiet = quiet + block.shape[1] / SR_HI if lvl < -55 else 0.0
        if source == "monitor" and ears.t > 10 and quiet >= quiet_end_s:
            break   # the song has ended (a stretch of silence)
        if max_s and ears.t >= max_s:
            break
    log(f"listened {ears.t:.0f}s in {time.time() - t0:.0f}s")
    return ears.finish(out_dir, meta=meta, log=log)

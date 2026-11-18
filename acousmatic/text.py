"""The listening record as text, in the order a song is heard.

Written to be read straight through: what's established at the top, then each section
and, inside it, bar by bar: the chord (with its function), the lead line, the bass line
and the drums. Bars that repeat the bar before are folded together.
Positions use counting syllables: 1 e & a.
"""
from __future__ import annotations

SYL = ["", "e", "&", "a"]


def _pos(g, bar_g0):
    k = g - bar_g0
    return f"{k // 4 + 1}{SYL[k % 4]}"


def _mmss(t):
    return f"{int(t // 60)}:{int(t % 60):02d}"


def _bar_rows(rec):
    rh = rec["rhythm"]
    m = rh["meter"]
    spans = rec["harmony"]["chord_spans"]
    lead, bass, drums = rec["notes"]["lead"], rec["notes"]["bass"], rec["drums"]
    rows = []
    for b in rh["bars"]:
        bi = rh["downbeat_index"] + (b["n"] - 1) * m
        g0, g1 = (bi + 1) * 4, (bi + 1 + len(b["beats"])) * 4
        ch = [c for c in spans if b["start"] - 1e-3 <= c["start"] < b["end"] - 1e-3]
        if not ch:
            ch = [c for c in spans if c["start"] <= b["start"]][-1:]
        chords = " ".join(f"{c['name']} ({c['roman']}{', borrowed' if c.get('borrowed') else ''})" for c in ch)
        ln = " ".join(f"{n['name']}@{_pos(n['g0'], g0)}" for n in lead if g0 <= n["g0"] < g1)
        bs = " ".join(f"{n['name']}@{_pos(n['g0'], g0)}" for n in bass if g0 <= n["g0"] < g1)
        dr = []
        for name, mark in (("kick", "K"), ("snare", "S"), ("hat", "h")):
            hits = {g - g0 for g in drums.get(name, []) if g0 <= g < g1}
            if hits:
                dr.append(f"{mark} " + "".join(mark if i in hits else "." for i in range(g1 - g0)))
        wl = " / ".join(l["text"] for l in rec.get("words", {}).get("lines", []) if b["start"] - 1e-3 <= l["start"] < b["end"] - 1e-3)
        rows.append({"n": b["n"], "start": b["start"], "chords": chords, "lead": ln, "bass": bs,
                     "drums": " | ".join(dr), "words": wl})
    return rows


def render(rec):
    src, rh, h = rec["source"], rec["rhythm"], rec["harmony"]
    t = src["tags"]
    L = []
    L.append(f"# {t.get('title', src['file'])}")
    who = " · ".join(x for x in (t.get("artist"), t.get("album"), t.get("date")) if x)
    if who:
        L.append(who)
    L.append("")
    L.append(f"Listening record (acousmatic {rec['acousmatic']}, schema {rec['schema']}). Everything below was measured "
             "by signal processing; notes and chords are transcriptions and can be wrong. Methods are listed at the end.")
    L.append("")
    k, mode = h["key"], h["mode"]
    lo = rec["loudness"]
    L.append("## At a glance")
    L.append(f"- Length {_mmss(src['duration'])}; {len(rh['bars'])} bars of {rh['meter']}/4 at about {rh['tempo']:.0f} BPM "
             f"(meter call: contrast {rh['meter_contrast'].get(rh['meter'])}; others {rh['meter_contrast']})")
    L.append(f"- Key {k['name']} (strength {k['strength']}, runner-up {k['runner_up']}, margin {k['margin']}); "
             f"best-fitting mode {mode['best']} (fit {mode['fit']})")
    L.append(f"- Loudness {lo['integrated_lufs']} LUFS, peak {lo['peak_dbfs']} dBFS, crest {lo['crest_db']} dB")
    L.append(f"- Form: {' '.join(s['label'] for s in rec['form']['sections'])}")
    lg = rec["form"].get("large") or []
    if lg:
        L.append("- Large-scale form: " + ", ".join(f"{s['label']} bars {s['start_bar']}–{s['end_bar']}" for s in lg))
    es = rec["stems"]["energy_share"]
    L.append(f"- Energy by stem: lead {es['lead']}, bass {es['bass']}, drums {es['drums']}, other {es['other']}")
    wd = rec.get("words") or {}
    if wd.get("lines"):
        L.append(f"- Words: {len(wd['lines'])} lines, from {wd['source']} (slid {wd.get('offset_s', 0):+} s to fit this recording); "
                 "they appear in the bars where they're sung")
    elif wd.get("instrumental"):
        L.append("- Words: none (instrumental)")
    else:
        L.append(f"- Words: none available ({wd.get('note') or wd.get('source')})")
    if h["cadences"]:
        L.append("- Cadences: " + "; ".join(f"bar {c['bar']} {c['kind']}" for c in h["cadences"][:12]))
    L.append(f"- Pictures: {rec['images']['overview']}, {rec['images']['stereo_field']}"
             + (f", score pages {', '.join(rec['score'].get('pages', []))}" if rec.get("score", {}).get("pages") else ""))
    L.append("")

    rows = {r["n"]: r for r in _bar_rows(rec)}
    for s in rec["form"]["sections"]:
        snd = s["sound"] or {}
        L.append(f"## {s['label']} · bars {s['start_bar']}–{s['end_bar']} · {_mmss(s['start'])}–{_mmss(s['end'])}")
        th = s["theory"]
        bits = [f"key here {th['key']}"]
        if th["harmonic_rhythm_beats_per_chord"]:
            bits.append(f"a chord every {th['harmonic_rhythm_beats_per_chord']} beats")
        for lp in th["loops"][:2]:
            bits.append(f"loop {lp['loop']} ×{lp['times']}")
        L.append("- Harmony: " + "; ".join(bits))
        if snd:
            bands = snd["bands"]
            top = sorted(bands, key=bands.get, reverse=True)[:3]
            L.append(f"- Sound: level {snd['level_db']:+} dB against the song's average, centroid {snd['centroid_hz']} Hz, flatness {snd['flatness']}, "
                     f"width {snd['width_overall']}; most energy in {', '.join(top)}")
        ch = "; ".join(f"{n}: {c}" for n, c in s["stem_character"].items() if c)
        if ch:
            L.append(f"- Parts: {ch}")
        if s["changes"]:
            L.append(f"- Changed from before: {', '.join(s['changes'])}")
        if th["melody"]:
            m = th["melody"]
            L.append(f"- Lead line: range {m['range']}, {int(100 * (m['stepwise_share'] or 0))}% stepwise, {m['leaps']} leaps, "
                     f"degrees mostly {' '.join(m['most_used_degrees'])}")
        L.append(f"- Picture: {s['image']}")
        L.append("")
        prev_sig, run_start = None, None
        def flush(until):
            if run_start is not None and until > run_start:
                L.append(f"  bars {run_start + 1}–{until}: same as bar {run_start}")
        for n in range(s["start_bar"], s["end_bar"] + 1):
            r = rows[n]
            sig = (r["chords"], r["lead"], r["bass"], r["drums"], r["words"])
            if sig == prev_sig:
                if run_start is None:
                    run_start = n - 1
                continue
            flush(n - 1)
            run_start = None
            prev_sig = sig
            L.append(f"- bar {n} ({_mmss(r['start'])}) {r['chords']}")
            if r["words"]:
                L.append(f"  words: \"{r['words']}\"")
            if r["lead"]:
                L.append(f"  lead: {r['lead']}")
            if r["bass"]:
                L.append(f"  bass: {r['bass']}")
            if r["drums"]:
                L.append(f"  drums: {r['drums']}")
        flush(s["end_bar"])
        L.append("")
    L.append("## Methods")
    for k_, v in rec["methods"].items():
        L.append(f"- {k_}: {v}")
    return "\n".join(L) + "\n"

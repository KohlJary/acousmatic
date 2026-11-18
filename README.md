# Acousmatic

Deterministic listening. Give it a recorded song; it gives back a listening record: a
time-ordered account of what is heard (form, tempo and bars, key, chords and their
function, the lead and bass lines, the drums, how each part sounds and where it sits in
the stereo field), with spectrogram images and an engraved transcription.

No language model takes part. Everything is signal processing with fixed parameters,
so the same file always gives the same record, and each part names the method behind it.
The name is Pythagoras's: the *akousmatikoi* heard the teacher from behind a veil, and
knew him by listening alone.

    acousmatic listen song.flac -o records/song [--stems] [--no-score]

Output: `record.md` (read it straight through), `record.json` (everything), `images/`,
`score.musicxml`, `transcription.mid`, engraved `score-page*.png`.

## How good is it

Measured with `bench/multitrack.py` against songs released as multitracks (Nine Inch
Nails, *The Slip*, CC BY-NC-SA 3.0): the stems summed into a mix, scored against the
isolated tracks and the songs' tempo/marker files. Numbers live in `bench/RESULTS.md`.
Transcriptions are transcriptions; read them as a sketch.

# Results

Measured by `bench/multitrack.py` on Nine Inch Nails, *The Slip* (multitracks released under CC BY-NC-SA 3.0; audio not in this repository). The mix is the sum of the stems. Tempo, beats, bars and sections are judged against each song's tempo/marker file; stems against the isolated tracks; drum hits and notes against the same detectors run on the isolated tracks (so they measure what the mix and separation cost). F is F-measure (0-1). Beats are judged at the octave chosen (double time counts).

| | NIN - Discipline | NIN - Letting You |
|---|---|---|
| tempo (true) | 122.1 (122.0) | 182.1 (91.0) |
| beats F | 0.988 | 0.975 |
| downbeats F | 0.984 | 0.667 |
| section boundaries F (±3 s) | 0.462 | 0.611 |
| drums kick F (±50 ms) | 0.716 | 0.688 |
| drums snare F (±50 ms) | 0.572 | 0.403 |
| drums hat F (±50 ms) | 0.861 | 0.671 |
| lead notes F (onset ±100 ms, ±50 cents) | 0.264 | 0.082 |
| bass notes F | 0.173 | 0.255 |
| stem SNR dB lead / bass / drums | 0.27 / -2.07 / 0.93 | -0.31 / -3.38 / 1.65 |

Letting You is tracked at double time (182.1 against 91), so its bars are half bars and downbeat F is capped near 0.67.

Chords have no ground truth here (the multitracks carry no chord charts). Each chord in a record carries how well it fits its template, and the record says what share of the song's harmony read clearly; on dense, distorted rock that share can be low.

Where it stands: time (tempo, beats, bars) and drums are usable; form is fair; bass lines are rough; the lead (vocal) line and the stems themselves are the weak part, as expected for DSP separation on a dense mix. 0 dB SNR means a stem carries as much of the rest of the mix as of its own part.

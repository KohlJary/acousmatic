# Results

Measured by `bench/multitrack.py` on Nine Inch Nails, *The Slip* (multitracks released under CC BY-NC-SA 3.0; audio not in this repository). The mix is the sum of the stems. Tempo, beats, bars and sections are judged against each song's tempo/marker file; stems against the isolated tracks; drum hits and notes against the same detectors run on the isolated tracks (so they measure what the mix and separation cost). F is F-measure (0-1). Beats are judged at the octave chosen (double time counts).

| | NIN - Discipline | NIN - Letting You |
|---|---|---|
| tempo (true) | 123.0 (122.0) | 184.6 (91.0) |
| beats F | 0.978 | 0.973 |
| downbeats F | 0.974 | 0.662 |
| section boundaries F (±3 s) | 0.538 | 0.629 |
| drums kick F (±50 ms) | 0.715 | 0.688 |
| drums snare F (±50 ms) | 0.578 | 0.403 |
| drums hat F (±50 ms) | 0.863 | 0.671 |
| lead notes F (onset ±100 ms, ±50 cents) | 0.265 | 0.082 |
| bass notes F | 0.176 | 0.255 |
| stem SNR dB lead / bass / drums | 0.27 / -2.07 / 0.93 | -0.31 / -3.38 / 1.65 |

Letting You is tracked at double time (184.6 against 91), so its bars are half bars and downbeat F is capped near 0.67.

Where it stands: time (tempo, beats, bars) and drums are usable; form is fair; bass lines are rough; the lead (vocal) line and the stems themselves are the weak part, as expected for DSP separation on a dense mix. 0 dB SNR means a stem carries as much of the rest of the mix as of its own part.

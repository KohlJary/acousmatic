# Reading an 808 kick's settings

`acousmatic/drum808.py`, calibrated on Michael Fischer's TR-808 set (1994): 25 bass drums
sampled from a real TR-808 at Tone and Decay positions 0, 2.5, 5, 7.5 and 10 (the set
isn't in this repository).

Stock samples, leave-one-out (each read with a calibration that never saw it):
- Decay: mean error 0.08 knob units; Tone: 0.38; 2 of 25 off by more than one step.

525 processed variants (each stock kick re-pitched -7 to +7 semitones the way a sampler
does it, and saturated at three drives):
- Decay (correcting for sampler re-pitching): within one knob step 96%, mean error 0.25.
- Re-tuning, on kicks long enough to ring: mean error 0.31 semitones, within 0.5 st 78%.
- Tone: within one step only 35% (39% without saturation). The click measure doesn't yet
  survive re-pitching; next on the list. Under heavy saturation Tone is withheld, not guessed.

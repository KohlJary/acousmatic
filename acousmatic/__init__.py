"""Acousmatic: deterministic listening.

Give it a recorded song; it returns a listening record, a time-ordered account of
what is heard in it (form, time, harmony, the lines each part plays, how it sounds,
where it sits in the stereo field), with spectrogram images and an engraved score.

No language model takes part. Everything is signal processing with fixed parameters:
the same file gives the same record. Each part of the record names the method that
produced it, and transcriptions say they are transcriptions.
"""
__version__ = "0.1.1"
SCHEMA = 1

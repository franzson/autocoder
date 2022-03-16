"""librosa.load minimum version"""
import pathlib
import warnings
import soundfile as sf
import numpy as np
import scipy.signal




# -- CORE ROUTINES --#
# Load should never be cached, since we cannot verify that the contents of
# 'path' are unchanged across calls.
def load(
    path,
    *,
    sr=44100,
    mono=True,
    offset=0.0,
    duration=None,
    dtype=np.float32,
    res_type="kaiser_best",
):


    try:
        if isinstance(path, sf.SoundFile):
            # If the user passed an existing soundfile object,
            # we can use it directly
            context = path
        else:
            # Otherwise, create the soundfile object
            context = sf.SoundFile(path)

        with context as sf_desc:
            sr_native = sf_desc.samplerate
            print("  ####################################")
            print("  #    ", sr_native)
            if offset:
                # Seek to the start of the target read
                sf_desc.seek(int(offset * sr_native))
            if duration is not None:
                frame_duration = int(duration * sr_native)
            else:
                frame_duration = -1

            # Load the target number of frames, and transpose to match librosa form
            y = sf_desc.read(frames=frame_duration, dtype=dtype, always_2d=False).T

    except RuntimeError as exc:
            raise (exc)

    # Final cleanup for dtype and contiguity
    if mono:
        y = to_mono(y)

    sr = sr_native

    return y, sr


def to_mono(y):

    if y.ndim > 1:
        y = np.mean(y, axis=tuple(range(y.ndim - 1)))

    return y

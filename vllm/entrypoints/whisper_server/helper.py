# SPDX-License-Identifier: Apache-2.0
# Adapted from https://github.com/openai/whisper/blob/fc5ded7d9045c693692f13853857c3f8baea3a7b/whisper/audio.py
# MIT License

import subprocess
from subprocess import CalledProcessError

import numpy as np


def exact_div(x, y):
    assert x % y == 0
    return x // y


# hard-coded audio hyperparameters
SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000 samples in a 30-second chunk
N_FRAMES = exact_div(N_SAMPLES,
                     HOP_LENGTH)  # 3000 frames in a mel spectrogram input

N_SAMPLES_PER_TOKEN = HOP_LENGTH * 2  # the initial convolutions has stride 2
FRAMES_PER_SECOND = exact_div(SAMPLE_RATE, HOP_LENGTH)  # 10ms per audio frame
TOKENS_PER_SECOND = exact_div(SAMPLE_RATE,
                              N_SAMPLES_PER_TOKEN)  # 20ms per audio token


def load_audio_from_bytes(audio_bytes: str, sample_rate: int = SAMPLE_RATE):
    """
    Read bytes from audio file as mono waveform, resampling as necessary

    Parameters
    ----------
    audio_bytes: bytes
    sample_rate: int

    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """

    # This launches a subprocess to decode audio while down-mixing
    # and resampling as necessary.  Requires the ffmpeg CLI in PATH.
    cmd = [
        "ffmpeg", "-nostdin", "-threads", "0", "-i", "-", "-f", "s16le", "-ac",
        "1", "-acodec", "pcm_s16le", "-ar",
        str(sample_rate), "-"
    ]
    try:
        process = subprocess.Popen(cmd,
                                   stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
        stdout, _ = process.communicate(input=audio_bytes)
    except CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(stdout, np.int16).flatten().astype(
        np.float32) / 32768.0


def validate_length(array, max_length: int = N_SAMPLES, axis: int = -1):
    audio_length = array.shape[axis]
    if audio_length > max_length:
        raise ValueError(
            f"Length of audio {audio_length} is bigger than the maximum "
            f"length of {max_length} = MAX_LENGTH * SAMPLING_RATE")

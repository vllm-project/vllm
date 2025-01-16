import numpy as np
from PIL import Image


def random_image(rng: np.random.RandomState, min_wh: int, max_wh: int):
    w, h = rng.randint(min_wh, max_wh, size=(2, ))
    arr = rng.randint(0, 255, size=(w, h, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def random_video(
    rng: np.random.RandomState,
    min_frames: int,
    max_frames: int,
    min_wh: int,
    max_wh: int,
):
    # Temporary workaround for https://github.com/huggingface/transformers/issues/35412
    num_frames = rng.randint(min_frames, max_frames)
    num_frames = (num_frames // 2) * 2

    w, h = rng.randint(min_wh, max_wh, size=(2, ))
    return rng.randint(0, 255, size=(num_frames, w, h, 3), dtype=np.uint8)


def random_audio(
    rng: np.random.RandomState,
    min_len: int,
    max_len: int,
    sr: int,
):
    audio_len = rng.randint(min_len, max_len)
    return rng.rand(audio_len), sr

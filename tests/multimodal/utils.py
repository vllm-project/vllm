# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import cv2
import numpy as np
import numpy.typing as npt
from PIL import Image


def random_image(rng: np.random.RandomState, min_wh: int, max_wh: int):
    w, h = rng.randint(min_wh, max_wh, size=(2,))
    arr = rng.randint(0, 255, size=(w, h, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def random_video(
    rng: np.random.RandomState,
    min_frames: int,
    max_frames: int,
    min_wh: int,
    max_wh: int,
):
    num_frames = rng.randint(min_frames, max_frames)
    w, h = rng.randint(min_wh, max_wh, size=(2,))
    return rng.randint(0, 255, size=(num_frames, w, h, 3), dtype=np.uint8)


def random_audio(
    rng: np.random.RandomState,
    min_len: int,
    max_len: int,
    sr: int,
):
    audio_len = rng.randint(min_len, max_len)
    return rng.rand(audio_len), sr


def create_video_from_image(
    image_path: str,
    video_path: str,
    num_frames: int = 10,
    fps: float = 1.0,
    is_color: bool = True,
    fourcc: str = "mp4v",
):
    image = cv2.imread(image_path)
    if not is_color:
        # Convert to grayscale if is_color is False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = image.shape
    else:
        height, width, _ = image.shape

    video_writer = cv2.VideoWriter(
        video_path,
        cv2.VideoWriter_fourcc(*fourcc),
        fps,
        (width, height),
        isColor=is_color,
    )

    for _ in range(num_frames):
        video_writer.write(image)

    video_writer.release()
    return video_path


def create_long_gop_video(
    num_frames: int = 50,
    fps: int = 30,
    width: int = 64,
    height: int = 64,
) -> bytes:
    """Encode an H.264 clip with one keyframe and green-channel = frame index.

    The marker lets a test recover which frame the decoder actually returned,
    independent of any metadata label.
    """
    import io

    import av

    buf = io.BytesIO()
    with av.open(buf, mode="w", format="mp4") as container:
        stream = container.add_stream("h264", rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.codec_context.gop_size = num_frames
        stream.codec_context.max_b_frames = 0
        stream.codec_context.options = {
            "x264-params": (f"scenecut=0:keyint={num_frames}:min-keyint={num_frames}")
        }
        for i in range(num_frames):
            img = np.zeros((height, width, 3), dtype=np.uint8)
            img[:, :, 1] = i % 256
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return buf.getvalue()


def create_edit_list_trimmed_video(
    num_frames: int = 90,
    trim_start_frame: int = 60,
    fps: int = 30,
) -> tuple[bytes, int]:
    """Stream-copy-cut a long-GOP clip so an mp4 edit list hides the lead-in.

    Remuxes the tail of a single-keyframe clip without re-encoding, rebasing
    timestamps at ``trim_start_frame``: the lead-in packets are still needed
    to decode, so they stay in the file at negative pts and the muxer records
    an edit list. The header sample count stays ``num_frames`` while only the
    trailing frames are presentable — the shape produced by lossless trims
    (e.g. ``ffmpeg -ss ... -c copy``). Returns (video_bytes, visible_frames).
    """
    import io

    import av

    src = create_long_gop_video(num_frames=num_frames, fps=fps)
    buf = io.BytesIO()
    with (
        av.open(io.BytesIO(src)) as source,
        av.open(buf, mode="w", format="mp4") as out,
    ):
        in_stream = source.streams.video[0]
        out_stream = out.add_stream_from_template(in_stream)
        start_pts = round(trim_start_frame / fps / in_stream.time_base)
        for packet in source.demux(in_stream):
            if packet.pts is None or packet.dts is None:
                continue
            packet.pts -= start_pts
            packet.dts -= start_pts
            packet.stream = out_stream
            out.mux(packet)
    return buf.getvalue(), num_frames - trim_start_frame


def cosine_similarity(A: npt.NDArray, B: npt.NDArray, axis: int = -1) -> npt.NDArray:
    """Compute cosine similarity between two vectors."""
    return np.sum(A * B, axis=axis) / (
        np.linalg.norm(A, axis=axis) * np.linalg.norm(B, axis=axis)
    )


def normalize_image(image: npt.NDArray) -> npt.NDArray:
    """Normalize image to [0, 1] range."""
    return image.astype(np.float32) / 255.0

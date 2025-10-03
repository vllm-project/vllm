# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import base64
import io

import decord
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer
from transformers.video_utils import VideoMetadata

from vllm import LLM, SamplingParams


def main():
    model_path = "/home/ekhvedchenia/vlm-hf-code/nano_vl_v2"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    processor = AutoProcessor.from_pretrained(model_path,
                                              trust_remote_code=True)

    sampling_params = SamplingParams(temperature=0, max_tokens=1024)

    video_path = "AdobeStock_726045803.mov"
    video_fps = 1
    video_nframe = 32
    video_nframe_max = -1

    # Get frames and metadata
    image_urls, metadata = sample_video_frames_to_data_urls(
        video_path,
        fps=max(0, int(video_fps)),
        nframe=max(0, int(video_nframe)),
        nframe_max=int(video_nframe_max),
    )
    frames = [pil_image_from_base64(image_url) for image_url in image_urls]

    print(f"Metadata: {metadata}")

    messages = [{
        "role": "system",
        "content": "/no_think"
    }, {
        "role":
        "user",
        "content": [
            {
                "type": "video",
                "video": f"file://{video_path}",
            },
            {
                "type": "text",
                "text": "\nDescribe what you see.",
            },
        ],
    }]
    prompt = tokenizer.apply_chat_template(messages,
                                           tokenize=False,
                                           add_generation_prompt=True)

    # Process with FPS metadata
    if metadata:
        inputs = processor(
            text=[prompt],
            videos=frames,
            videos_kwargs={'video_metadata': metadata},
            return_tensors="pt",
        )
    else:
        inputs = processor(
            text=[prompt],
            videos=frames,
            return_tensors="pt",
        )

    #video_pixels = inputs.data["pixel_values_video"]

    video_pixels = np.random.randint(0,
                                     255, (20, 640, 1024, 3),
                                     dtype=np.uint8)
    metadata = None

    llm = LLM(
        model_path,
        trust_remote_code=True,
        enforce_eager=True,
        video_pruning_rate=0.75,
    )

    llm_inputs = {
        "prompt": prompt,
        "multi_modal_data": {
            "video": (video_pixels, metadata)
        },
    }

    outputs = llm.generate([llm_inputs], sampling_params=sampling_params)
    generated_text = outputs[0].outputs[0].text
    print(generated_text)


def sample_video_frames_to_data_urls(video_path_local,
                                     fps=1,
                                     nframe=0,
                                     nframe_max=-1):
    """
    Sample frames from a video and return base64-encoded data URLs along with metadata.

    Args:
        video_path_local: Path to the video file
        fps: Target frames per second for sampling (if > 0, uses fps-based sampling)
        nframe: Number of frames to sample (used if fps <= 0)
        nframe_max: Maximum number of frames to sample

    Returns:
        tuple: (frame_data_urls, metadata)
        - frame_data_urls: List of base64-encoded frame images
        - metadata: VideoMetadata dataclass containing info about the sampled frames:
            - total_num_frames: Number of sampled frames
            - fps: Effective frame rate of the sampled frames
            - duration: Duration covered by the sampled frames (in seconds)
            - video_backend: Backend used for video processing ('decord')
    """
    import numpy as np
    from PIL import Image

    vid = decord.VideoReader(video_path_local)
    total_frames = len(vid)
    video_fps = vid.get_avg_fps()
    total_duration = total_frames / max(1e-6, video_fps)

    if fps > 0:
        required_frames = int(total_duration * fps)
        desired_frames = max(1, required_frames)
        if nframe_max > 0 and desired_frames > nframe_max:
            desired_frames = nframe_max
        if desired_frames >= total_frames:
            indices = list(range(total_frames))
        elif desired_frames == 1:
            indices = [0]  # Always use first frame for single frame sampling
        else:
            # Generate evenly spaced indices and ensure uniqueness
            raw_indices = np.linspace(0, total_frames - 1, desired_frames)
            indices = list(np.unique(np.round(raw_indices).astype(int)))
    else:
        desired_frames = max(1, int(nframe) if nframe and nframe > 0 else 8)
        if nframe_max > 0 and desired_frames > nframe_max:
            desired_frames = nframe_max
        if desired_frames >= total_frames:
            indices = list(range(total_frames))
        elif desired_frames == 1:
            indices = [0]  # Always use first frame for single frame sampling
        else:
            # Generate evenly spaced indices and ensure uniqueness
            raw_indices = np.linspace(0, total_frames - 1, desired_frames)
            indices = list(np.unique(np.round(raw_indices).astype(int)))

    images = [Image.fromarray(vid[i].asnumpy()) for i in indices]
    frame_urls = [encode_pil_to_jpeg_data_url(im) for im in images]

    # Calculate timestamps for each sampled frame
    timestamps = [float(idx) / video_fps for idx in indices]

    # Calculate metadata for the sampled frames
    sampled_num_frames = len(indices)

    # Duration is the time span from first to last frame
    if len(timestamps) > 1:
        sampled_duration = timestamps[-1] - timestamps[0]
        sampled_fps = (sampled_num_frames -
                       1) / sampled_duration if sampled_duration > 0 else 1.0
    else:
        # Single frame case
        sampled_duration = None
        sampled_fps = None

    metadata = VideoMetadata(
        total_num_frames=sampled_num_frames,
        fps=sampled_fps,
        duration=sampled_duration,
        video_backend=None,
    )

    return frame_urls, metadata


def encode_pil_to_jpeg_data_url(pil_image):
    from io import BytesIO
    buf = BytesIO()
    pil_image.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def pil_image_from_base64(b64_str: str) -> Image.Image:
    # Handle data URLs like "data:image/png;base64,...."
    if b64_str.startswith('data:'):
        b64_str = b64_str.split(',', 1)[1]
    img_bytes = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(img_bytes))


if __name__ == "__main__":
    main()

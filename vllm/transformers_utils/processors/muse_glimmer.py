# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HuggingFace processor for the MuseGlimmer multimodal model (text + image + video).

This packages MuseGlimmer's vision preprocessing and token-span expansion the standard
HF way, so you can do:

    processor = AutoProcessor.from_pretrained(hf_dir, trust_remote_code=True)
    messages = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": "Describe this image."},
    ]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=text, images=[pil_image], return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=256)

It reproduces the exact token layout the model expects:

  image -> <|image_start|> + <|patch|> * N + <|image_end|>
  video -> <|vid_start|>
           ( "Time: X.Xs" + <|video|> * P [+ <|vid_frame_separator|>] )*
           + <|vid_end|>

where N = floor(H/(patch*ds)) * floor(W/(patch*ds)) (<= 4096) for images and
P = the same grid count per temporal frame-group (<= 144) for video, with the
target (H, W) chosen by MuseGlimmerImageProcessor.compute_image_size /
MuseGlimmerVideoProcessor.compute_video_frame_size (the same grid logic as
modeling_muse_glimmer.MuseGlimmerVisionEncoder).

The chat template emits one sentinel per media item -- ``<|image|>`` for images,
``<|video|>`` for videos -- which ``MuseGlimmerProcessor.__call__`` expands into the
spans above using each media item's computed token count. Vision features are
returned in ``pixel_values`` (a list of per-image / per-frame-group tensors) in
the order the sentinels appear, matching how ``MuseGlimmerModel`` merges them at the
``<|patch|>`` / ``<|video|>`` positions.
"""

from __future__ import annotations

import itertools
import math
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms as T
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_processing_utils import BaseImageProcessor
from transformers.processing_utils import ProcessorMixin
from transformers.video_processing_utils import BaseVideoProcessor

# Single-token sentinels emitted by the chat template, one per media item.
IMAGE_SENTINEL = "<|image|>"  # id 200090; absent from the image/video spans
VIDEO_SENTINEL = "<|video|>"  # id 200091; also the per-frame token (see __call__)

# Jinja chat template (multimodal superset of convert_muse_glimmer_to_hf.CHAT_TEMPLATE).
# Message content may be a plain string OR a list of parts
# ({"type": "text"|"image"|"video", ...}); image/video parts render as a single
# sentinel that __call__ expands.
MUSE_GLIMMER_MM_CHAT_TEMPLATE = (
    "{{- bos_token -}}"
    "{%- macro render_parts(content) -%}"
    "{%- if content is string -%}{{- content -}}"
    "{%- else -%}"
    "{%- for part in content -%}"
    "{%- if part['type'] == 'image' -%}{{- '<|image|>' -}}"
    "{%- elif part['type'] == 'video' -%}{{- '<|video|>' -}}"
    "{%- elif part['type'] == 'text' -%}{{- part['text'] -}}"
    "{%- endif -%}"
    "{%- endfor -%}"
    "{%- endif -%}"
    "{%- endmacro -%}"
    # At inference (add_generation_prompt), inject a default system message when
    # the caller supplied none. Full-transcript rendering
    # (add_generation_prompt=False) is left unchanged.
    "{%- set ns = namespace(has_system=false) -%}"
    "{%- for m in messages -%}"
    "{%- if m['role'] == 'system' -%}{%- set ns.has_system = true -%}{%- endif -%}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt and not ns.has_system -%}"
    "{{- '<|start|>system<|message|>You are a helpful assistant.<|eot|>' -}}"
    "{%- endif -%}"
    "{%- for message in messages -%}"
    "{%- set role = message['role'] -%}"
    "{%- if role == 'assistant' -%}"
    "{%- set recipient = message.get('recipient') -%}"
    "{%- set end_turn = message.get('end_turn') -%}"
    "{%- if end_turn is none -%}"
    "{%- set end_turn = not (recipient and recipient != 'user') -%}"
    "{%- endif -%}"
    "{{- '<|start|>assistant' -}}"
    "{%- if recipient -%}{{- ' to=' + recipient -}}{%- endif -%}"
    "{{- '<|message|>' -}}{{- render_parts(message['content']) -}}"
    "{{- ('<|eot|>' if end_turn else '<|eom|>') -}}"
    "{%- elif role == 'tool' -%}"
    "{%- set name = message.get('name', '') -%}"
    # Tool content is emitted as-is (string body or interleaved image/text parts).
    # Any <tool_output ...> wrapper is baked into the SFT data content; the
    # tokenizer does not add it, so the template must not either.
    "{{- '<|start|>tool ' + name + '<|message|>' -}}"
    "{{- render_parts(message['content']) -}}"
    "{{- '<|eot|>' -}}"
    "{%- else -%}"
    "{%- set header = role -%}"
    "{%- if message.get('name') -%}"
    "{%- set header = role + ' ' + message['name'] -%}"
    "{%- endif -%}"
    "{{- '<|start|>' + header + '<|message|>' -}}"
    "{{- render_parts(message['content']) -}}"
    "{{- '<|eot|>' -}}"
    "{%- endif -%}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}{{- '<|start|>assistant' -}}{%- endif -%}"
)


def _grid_size(
    img_w: int, img_h: int, patch_hw: int, max_tokens: int
) -> tuple[int, int, int]:
    """Pick the integer (H, W) grid closest to the aspect ratio under the token cap.

    Replicates MuseGlimmerVisionEncoder._compute_grid_size
    (modeling_muse_glimmer.py) so the processor needs no torch model import.
    Returns (target_h, target_w, n_tokens).
    """
    i_nph = img_h / patch_hw
    i_npw = img_w / patch_hw
    ratio = i_npw / i_nph if i_nph > 0 else 1.0
    if i_nph * i_npw > max_tokens:
        i_nph = (max_tokens / ratio) ** 0.5
        i_npw = i_nph * ratio
    candidates = list(
        set(
            itertools.product(
                [math.floor(i_nph), math.ceil(i_nph)],
                [math.floor(i_npw), math.ceil(i_npw)],
            )
        )
    )
    candidates = [
        (nph, npw)
        for nph, npw in candidates
        if nph >= 1 and npw >= 1 and nph * npw <= max_tokens
    ]
    if not candidates:
        candidates = [(max(1, round(i_nph)), max(1, round(i_npw)))]
    nph, npw = min(candidates, key=lambda c: abs(c[0] / c[1] - img_h / img_w))
    return nph * patch_hw, npw * patch_hw, nph * npw


class MuseGlimmerImageProcessor(BaseImageProcessor):
    """Resize + normalize MuseGlimmer images and compute patch-token counts.

    Variable-resolution: each image is resized to the grid that best matches its
    aspect ratio under the per-image token cap, then normalized with mean/std 0.5.
    Returns per-image tensors (not stacked) because MuseGlimmer consumes a list of
    variable-size images. Video frames are handled by MuseGlimmerVideoProcessor.
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        patch_size: int = 14,
        downsample_factor: int = 2,
        max_image_tokens: int = 4096,
        image_mean: float = 0.5,
        image_std: float = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.downsample_factor = downsample_factor
        self.max_image_tokens = max_image_tokens
        self.image_mean = image_mean
        self.image_std = image_std

    def _to_norm_tensor(self, image: Image.Image) -> torch.Tensor:
        # Functional normalize -- no stored Normalize object, so the processor
        # stays JSON-serializable for to_dict() / save_pretrained.
        return T.functional.normalize(
            T.functional.to_tensor(image),
            [self.image_mean] * 3,
            [self.image_std] * 3,
        )

    # -- size computation (mirrors modeling_muse_glimmer.MuseGlimmerVisionEncoder) --
    def compute_image_size(self, img_w: int, img_h: int) -> tuple[int, int, int]:
        ph = self.patch_size * self.downsample_factor
        return _grid_size(img_w, img_h, ph, self.max_image_tokens)

    # -- preprocessing ---------------------------------------------------------
    def preprocess_image(self, image: Image.Image) -> tuple[torch.Tensor, int]:
        """Return (pixel tensor [3, H, W], n_patch_tokens) for one image."""
        image = image.convert("RGB")
        target_h, target_w, n_tokens = self.compute_image_size(
            image.width, image.height
        )
        image = image.resize((target_w, target_h), Image.LANCZOS)
        return self._to_norm_tensor(image), n_tokens


# torchcodec is the training-faithful video decoder; other decoders (torchvision /
# PyAV) seek + color-convert differently and diverge from training. Optional at
# import time so text/image paths load without it; decode_video raises if used.
try:
    import torchcodec
except Exception:
    torchcodec = None


class MuseGlimmerVideoProcessor(BaseVideoProcessor):
    """MuseGlimmer video preprocessing behind the standard HF
    ``AutoVideoProcessor`` API.

    Exposes MuseGlimmer's training-faithful video handling as a first-class video
    processor so you can call ``processor(videos="clip.mp4")`` and downstream
    users can discover it via the standard HF ``AutoVideoProcessor`` API plus a
    ``video_preprocessor_config.json``. It wraps:

      * torchcodec decode (matches training; other decoders diverge),
      * uniform frame sampling to a whole multiple of ``patch_temporal``,
      * real per-group PTS timestamps (rendered as ``Time: X.Xs`` by
        MuseGlimmerProcessor),
      * ``patch_temporal`` frame-grouping (frames cat on the channel axis ->
        ``[patch_temporal * 3, H, W]``; the encoder detects video by channel count).

    It deliberately overrides ``preprocess`` instead of the BaseVideoProcessor fast
    pipeline (group_videos_by_shape / smart_resize / single stacked tensor): MuseGlimmer
    consumes a LIST of variable-size group tensors and needs the real per-group PTS,
    neither of which the stacked fast path models.
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        patch_size: int = 14,
        downsample_factor: int = 2,
        patch_temporal: int = 2,
        max_video_frame_tokens: int = 144,
        image_mean: float = 0.5,
        image_std: float = 0.5,
        video_num_frames: int = 96,
        video_sampling_fps: float = 2.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.downsample_factor = downsample_factor
        self.patch_temporal = patch_temporal
        self.max_video_frame_tokens = max_video_frame_tokens
        self.image_mean = image_mean
        self.image_std = image_std
        self.video_num_frames = video_num_frames
        self.video_sampling_fps = video_sampling_fps

    def _to_norm_tensor(self, image: Image.Image) -> torch.Tensor:
        return T.functional.normalize(
            T.functional.to_tensor(image),
            [self.image_mean] * 3,
            [self.image_std] * 3,
        )

    def decode_video(self, video_path: str) -> tuple[list[Image.Image], list[float]]:
        """Sample frames + per-group PTS with torchcodec (the training decode path).

        ``timestamps[g]`` is the ACTUAL decoded PTS of the first frame in temporal
        group ``g``; ``len(frames)`` is a whole multiple of ``patch_temporal``.
        """
        if torchcodec is None:
            raise RuntimeError(
                "torchcodec is required for video decoding (it matches the training "
                "decode path). See the Environment Setup steps in hf/README.md."
            )
        pt = self.patch_temporal
        reader = torchcodec.decoders.VideoDecoder(video_path)
        total = len(reader)
        assert reader.metadata.average_fps is not None, (
            f"Video has no FPS metadata: {video_path}"
        )
        fps = reader.metadata.average_fps
        assert self.video_sampling_fps and self.video_sampling_fps > 0, (
            f"video_sampling_fps must be positive, got {self.video_sampling_fps}"
        )
        n = min(
            int(total * self.video_sampling_fps / fps), self.video_num_frames, total
        )
        n = max(pt, (n // pt) * pt)
        n = min(n, total)
        if n < pt:
            raise ValueError(
                f"Video has only {total} decodable frame(s) but needs at least "
                f"{pt} (one temporal patch): {video_path}"
            )
        indices = torch.linspace(0, total - 1, n).long().tolist()
        frames: list[Image.Image] = []
        timestamps: list[float] = []
        for j, i in enumerate(indices):
            fr = reader[i]
            frames.append(
                Image.fromarray(fr.data.permute(1, 2, 0).numpy()).convert("RGB")
            )
            if j % pt == 0:
                pts = getattr(fr, "pts_seconds", None)
                timestamps.append(float(pts) if pts is not None else i / fps)
        return frames, timestamps

    def compute_video_frame_size(self, img_w: int, img_h: int) -> tuple[int, int, int]:
        """Pick (H, W, tokens_per_frame) for a video frame under the per-frame cap.

        Mirrors MuseGlimmerImageProcessor.compute_image_size but uses the (smaller)
        per-frame video token budget; shares the same ``_grid_size`` grid logic as
        modeling_muse_glimmer.MuseGlimmerVisionEncoder.
        """
        ph = self.patch_size * self.downsample_factor
        return _grid_size(img_w, img_h, ph, self.max_video_frame_tokens)

    def _group_frames(
        self, frames: list[Image.Image]
    ) -> tuple[list[torch.Tensor], int, int]:
        """Resize/normalize frames and cat ``patch_temporal`` frames per group.

        Returns (group_tensors, n_groups, tokens_per_group).
        """
        pt = self.patch_temporal
        if pt <= 0:
            raise ValueError(f"patch_temporal must be positive, got {pt}")
        if not frames:
            raise ValueError("video must contain at least one frame")
        if padding := (-len(frames)) % pt:
            frames = [*frames, *([frames[-1]] * padding)]
        first = frames[0].convert("RGB")
        target_h, target_w, n_tokens = self.compute_video_frame_size(
            first.width, first.height
        )
        groups: list[torch.Tensor] = []
        for i in range(0, len(frames), pt):
            grp = [
                self._to_norm_tensor(
                    frames[i + j]
                    .convert("RGB")
                    .resize((target_w, target_h), Image.LANCZOS)
                )
                for j in range(pt)
            ]
            groups.append(torch.cat(grp, dim=0))
        return groups, len(groups), n_tokens

    @staticmethod
    def _normalize_videos(videos) -> list:
        """Normalize the ``videos`` arg to a list of per-video items.

        A video item is a path (str/Path) or a list of PIL frames. Accepts a
        single video (one path, or one list of frames) or a list of those.
        """
        if isinstance(videos, (str, Path)):
            return [videos]
        if videos and not isinstance(videos[0], (list, tuple, str, Path)):
            return [videos]  # a single list of PIL frames
        return list(videos or [])

    def preprocess_one(
        self,
        video: str | Path | list[Image.Image],
        timestamps: list[float] | None = None,
    ) -> tuple[list[torch.Tensor], int, int, list[float]]:
        """Decode (if a path) + group ONE video.

        Returns (group_tensors, n_groups, tokens_per_group, group_timestamps). This
        is the single per-video implementation shared by ``preprocess`` (the HF
        AutoVideoProcessor batch API) and ``MuseGlimmerProcessor.__call__`` (prompt
        building), so the two paths cannot drift. For a path, frames + real
        per-group PTS come from ``decode_video``; for pre-decoded frames the given
        ``timestamps`` are used verbatim (empty if not supplied).
        """
        if isinstance(video, (str, Path)):
            frames, ts = self.decode_video(str(video))
        else:
            frames = [f.convert("RGB") for f in video]
            ts = list(timestamps) if timestamps is not None else []
        groups, n_groups, tokens_per_group = self._group_frames(frames)
        return groups, n_groups, tokens_per_group, ts

    def preprocess(
        self,
        videos,
        video_timestamps: list[list[float]] | None = None,
        return_tensors: str | None = "pt",
        **kwargs,
    ) -> BatchFeature:
        """Preprocess video file path(s) or pre-decoded frame list(s).

        Accepts a single video (path str, or list of PIL frames) or a list of
        those. Returns a BatchFeature carrying, in video order:
          * ``pixel_values``           -- flat list of [pt*3, H, W] group tensors,
          * ``video_num_groups``       -- groups per video,
          * ``video_tokens_per_group`` -- patch-token count per group per video,
          * ``video_timestamps``       -- per-group PTS per video,
        i.e. exactly what MuseGlimmerProcessor needs to build the
        ``<|vid_start|> ( Time: X.Xs <|video|>*P [sep] )* <|vid_end|>`` block.
        """
        videos = self._normalize_videos(videos)
        pixel_values: list[torch.Tensor] = []
        num_groups: list[int] = []
        tokens_per_group: list[int] = []
        out_ts: list[list[float]] = []
        for idx, v in enumerate(videos):
            ts_in = video_timestamps[idx] if video_timestamps else None
            groups, ng, tpg, ts = self.preprocess_one(v, ts_in)
            pixel_values += groups
            num_groups.append(ng)
            tokens_per_group.append(tpg)
            out_ts.append(ts)
        batch = BatchFeature(
            data={
                "video_num_groups": num_groups,
                "video_tokens_per_group": tokens_per_group,
                "video_timestamps": out_ts,
            },
            tensor_type=None,
        )
        # Variable-size list -- BatchFeature would fail to stack into one tensor.
        batch["pixel_values"] = pixel_values
        return batch


class MuseGlimmerProcessor(ProcessorMixin):
    """Bundle MuseGlimmerImageProcessor + MuseGlimmerVideoProcessor + tokenizer;
    expand media sentinels into spans.

    Images go through ``image_processor`` (MuseGlimmerImageProcessor) and videos through
    ``video_processor`` (MuseGlimmerVideoProcessor) -- one preprocessing implementation
    each, no overlap.
    """

    attributes = ["image_processor", "video_processor", "tokenizer"]
    image_processor_class = "AutoImageProcessor"
    video_processor_class = "AutoVideoProcessor"
    tokenizer_class = "PreTrainedTokenizerFast"

    def __init__(
        self,
        image_processor=None,
        video_processor=None,
        tokenizer=None,
        chat_template=None,
        **kwargs,
    ):
        if image_processor is None:
            image_processor = MuseGlimmerImageProcessor()
        if video_processor is None:
            video_processor = MuseGlimmerVideoProcessor()
        super().__init__(
            image_processor,
            video_processor,
            tokenizer,
            chat_template=chat_template or MUSE_GLIMMER_MM_CHAT_TEMPLATE,
            **kwargs,
        )

    def _sid(self, token: str) -> int:
        return self.tokenizer.convert_tokens_to_ids(token)

    def _image_block(self, n_tokens: int) -> list[int]:
        return (
            [self._sid("<|image_start|>")]
            + [self._sid("<|patch|>")] * n_tokens
            + [self._sid("<|image_end|>")]
        )

    def _video_block(
        self,
        n_groups: int,
        tokens_per_group: int,
        timestamps: list[float] | None = None,
    ) -> list[int]:
        """Per-group ``Time: X.Xs`` + <|video|>*P, separated/terminated.

        ``timestamps`` (one per temporal group, in seconds) should be the ACTUAL
        decoded frame times -- training renders the real factored frame PTS, not a
        uniform grid. Falls back to ``g*patch_temporal/fps`` only when timestamps
        are not supplied (e.g. callers that don't track frame times); that
        approximation can differ from training by the per-frame jitter.
        """
        vid = self._sid("<|video|>")
        sep = self._sid("<|vid_frame_separator|>")
        pt = self.video_processor.patch_temporal
        fps = self.video_processor.video_sampling_fps
        block = [self._sid("<|vid_start|>")]
        for g in range(n_groups):
            ts = timestamps[g] if timestamps is not None else g * pt / fps
            block += self.tokenizer.encode(f"Time: {ts:.1f}s", add_special_tokens=False)
            block += [vid] * tokens_per_group
            block.append(sep if g < n_groups - 1 else self._sid("<|vid_end|>"))
        return block

    def __call__(
        self,
        text: str | list[str] | None = None,
        images: list[Image.Image] | None = None,
        videos: list[list[Image.Image] | str] | None = None,
        video_timestamps: list[list[float]] | None = None,
        return_tensors: str | None = "pt",
        **kwargs,
    ) -> BatchFeature:
        if text is None:
            raise ValueError(
                "`text` is required (use apply_chat_template to build it)."
            )
        if isinstance(text, (list, tuple)):
            if len(text) != 1:
                raise ValueError(
                    "MuseGlimmerProcessor supports a single text sample per call."
                )
            text = text[0]

        images = list(images or [])
        videos = list(videos or [])
        image_sentinel = self._sid(IMAGE_SENTINEL)
        video_sentinel = self._sid(VIDEO_SENTINEL)

        # Preprocess media up front so we can expand sentinels in document order.
        # Images via image_processor; videos via video_processor (each video item
        # is a path -- decoded here -- or a pre-decoded list of PIL frames).
        prepped_images = [self.image_processor.preprocess_image(im) for im in images]
        prepped_videos = [
            self.video_processor.preprocess_one(
                v, video_timestamps[i] if video_timestamps else None
            )
            for i, v in enumerate(videos)
        ]

        ids = self.tokenizer.encode(text, add_special_tokens=False)
        n_img = sum(1 for t in ids if t == image_sentinel)
        n_vid = sum(1 for t in ids if t == video_sentinel)
        if n_img != len(prepped_images):
            raise ValueError(
                f"{n_img} image sentinel(s) in text but "
                f"{len(prepped_images)} image(s) given."
            )
        if n_vid != len(prepped_videos):
            raise ValueError(
                f"{n_vid} video sentinel(s) in text but "
                f"{len(prepped_videos)} video(s) given."
            )

        out_ids: list[int] = []
        pixel_values: list[torch.Tensor] = []
        img_i = vid_i = 0
        for tid in ids:
            if tid == image_sentinel:
                tensor, n_tokens = prepped_images[img_i]
                img_i += 1
                out_ids += self._image_block(n_tokens)
                pixel_values.append(tensor)
            elif tid == video_sentinel:
                groups, n_groups, tokens_per_group, ts = prepped_videos[vid_i]
                vid_i += 1
                out_ids += self._video_block(n_groups, tokens_per_group, ts or None)
                pixel_values += groups
            else:
                out_ids.append(tid)

        data: dict = {
            "input_ids": [out_ids],
            "attention_mask": [[1] * len(out_ids)],
        }
        batch = BatchFeature(data=data, tensor_type=return_tensors)
        # Keep pixel_values as a list of variable-size tensors (MuseGlimmerModel
        # consumes a list); BatchFeature would fail to stack them into one tensor.
        if pixel_values:
            batch["pixel_values"] = pixel_values
        return batch


__all__ = [
    "MuseGlimmerImageProcessor",
    "MuseGlimmerVideoProcessor",
    "MuseGlimmerProcessor",
    "MUSE_GLIMMER_MM_CHAT_TEMPLATE",
]

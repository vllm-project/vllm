# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only MiniCPM-V 4.7 model.

Same vision/LLM stack as MiniCPM-V 4.6, plus canvas 3D M-RoPE.
"""

import math
from collections.abc import Mapping
from typing import Any, NamedTuple

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFeatureSpec, MultiModalFieldConfig

from .minicpmv import MiniCPMVDummyInputsBuilder
from .minicpmv4_6 import (
    MiniCPMV4_6ForConditionalGeneration,
    MiniCPMV4_6MultiModalProcessor,
    MiniCPMV4_6ProcessingInfo,
    _scoped_mm_kwarg,
)

logger = init_logger(__name__)


class MiniCPMV4_7ProcessingInfo(MiniCPMV4_6ProcessingInfo):
    def _mm_max_slice_nums(self, **kwargs: object) -> int | None:
        # No schema resolution here: it would call back into `get_hf_processor`
        # below, which resolves this same value.
        merged = self.ctx.get_merged_mm_kwargs(kwargs)
        max_slice = _scoped_mm_kwarg(merged, "image", "max_slice_nums")
        if max_slice is None:
            max_slice = merged.get("max_slice_nums")
        if max_slice is None:
            return None
        return int(max_slice)

    def _configured_max_slice_nums(self) -> int | None:
        # Configured (server-level) value only: per-request values are passed
        # explicitly to the image processor call, so they must not be written
        # back to the shared processor instance.
        return self._mm_max_slice_nums()

    def get_hf_processor(self, **kwargs: object):
        hf_processor = super().get_hf_processor(**kwargs)
        image_processor = getattr(hf_processor, "image_processor", None)
        configured = self._configured_max_slice_nums()
        if image_processor is not None and configured is not None:
            image_processor.max_slice_nums = configured
        return hf_processor

    def get_image_max_slice_num(self) -> int:
        max_slice = self._configured_max_slice_nums()
        if max_slice is not None:
            return max_slice
        return super().get_image_max_slice_num()


class MiniCPMV4_7MultiModalProcessor(MiniCPMV4_6MultiModalProcessor):
    def _get_mm_fields_config(
        self,
        hf_inputs,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        fields = dict(super()._get_mm_fields_config(hf_inputs, hf_processor_mm_kwargs))
        # Canvas M-RoPE places slices from `tgt_sizes`. Stripping the payload of
        # a prefix-cache-covered item only keeps CPU-side metadata, so without
        # this a cached image loses its target sizes, `image_bounds` and
        # `target_sizes` disagree, and the whole request silently falls back to
        # sequential positions.
        for key, modality in (("tgt_sizes", "image"), ("video_tgt_sizes", "video")):
            if key in fields:
                fields[key] = MultiModalFieldConfig.batched(modality, keep_on_cpu=True)
        return fields

    def _video_local_id_prefix(self, video_idx: int) -> str:
        # MiniCPMV4_7Processor emits no local `<image_id>` in front of video
        # placeholders: a video is one temporal sequence and 4.7 was trained
        # without them.
        return ""

    def get_image_prompt_texts(
        self,
        image_size,
        image_idx: int = 0,
        downsample_mode: str | None = None,
        max_slice_nums: int | None = None,
    ) -> str:
        info = self.info
        assert isinstance(info, MiniCPMV4_6ProcessingInfo)
        if max_slice_nums is None:
            max_slice_nums = info.get_image_max_slice_num()
        return info.get_slice_image_placeholder(
            image_size,
            image_idx=image_idx,
            max_slice_nums=max_slice_nums,
            downsample_mode=downsample_mode,
        )


def _as_hw_pairs(tensor: torch.Tensor) -> torch.Tensor:
    data = tensor.to(torch.long)
    if data.numel() == 0:
        return data.reshape(0, 2)
    if data.shape[-1] != 2:
        raise ValueError(f"expected last dim 2, got {tuple(data.shape)}")
    return data.reshape(-1, 2)


def _sequential_positions(seq_len: int) -> torch.Tensor:
    """Positions for the fallback where canvas M-RoPE cannot be computed."""
    return torch.arange(seq_len).unsqueeze(0).expand(3, -1)


@MULTIMODAL_REGISTRY.register_processor(
    MiniCPMV4_7MultiModalProcessor,
    info=MiniCPMV4_7ProcessingInfo,
    dummy_inputs=MiniCPMVDummyInputsBuilder,
)
class MiniCPMV4_7ForConditionalGeneration(MiniCPMV4_6ForConditionalGeneration):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self._init_canvas_mrope(vllm_config)

    def _drop_vision_last_layer(self) -> bool:
        # MiniCPMV4_7Config no longer carries this field (transformers marks it
        # removed), and 4.7 keeps every vision encoder layer.
        return False

    def _init_canvas_mrope(self, vllm_config: VllmConfig) -> None:
        config = self.config
        uses_canvas = bool(getattr(config, "uses_mrope_canvas", False))
        if not uses_canvas:
            mrope_mode = getattr(config, "mrope_mode", None)
            uses_canvas = mrope_mode is not None and "canvas" in str(mrope_mode).lower()

        self._uses_canvas_mrope = uses_canvas
        self._canvas_special_ids: dict[str, int] | None = None
        if not uses_canvas:
            return

        # MiniCPMV4_7Config pins the structural ids in config.json; the
        # conversion script resolves them from the tokenizer. Prefer them so the
        # canvas matches training, and fall back to the tokenizer for checkpoints
        # whose config predates these fields.
        configured_ids: dict[str, int | None] = {
            "im_start_id": getattr(config, "image_start_id", None),
            "im_end_id": getattr(config, "image_end_id", None),
            "slice_start_id": getattr(config, "slice_start_id", None),
            "slice_end_id": getattr(config, "slice_end_id", None),
            "newline_id": getattr(config, "newline_id", None),
        }
        if all(value is not None for value in configured_ids.values()):
            self._canvas_special_ids = {
                name: int(value)
                for name, value in configured_ids.items()
                if value is not None
            }
            return

        from vllm.tokenizers.registry import get_tokenizer

        model_config = vllm_config.model_config
        tokenizer = get_tokenizer(
            model_config.tokenizer,
            revision=model_config.tokenizer_revision,
            trust_remote_code=model_config.trust_remote_code,
        )
        self._canvas_special_ids = {
            "im_start_id": tokenizer.convert_tokens_to_ids(
                getattr(tokenizer, "image_start_token", "<image>")
            ),
            "im_end_id": tokenizer.convert_tokens_to_ids(
                getattr(tokenizer, "image_end_token", "</image>")
            ),
            "slice_start_id": tokenizer.convert_tokens_to_ids(
                getattr(tokenizer, "slice_start_token", "<slice>")
            ),
            "slice_end_id": tokenizer.convert_tokens_to_ids(
                getattr(tokenizer, "slice_end_token", "</slice>")
            ),
            "newline_id": tokenizer.encode("\n", add_special_tokens=False)[0],
        }

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list[MultiModalFeatureSpec],
    ) -> tuple[torch.Tensor, int]:
        seq_len = len(input_tokens)
        if (
            not getattr(self, "_uses_canvas_mrope", False)
            or self._canvas_special_ids is None
            or not mm_features
        ):
            return _sequential_positions(seq_len), 0

        input_ids = torch.tensor(input_tokens, dtype=torch.long)
        image_bounds = build_image_bounds(input_ids, self._canvas_special_ids)

        target_sizes_list: list[torch.Tensor] = []
        for mm_feature in sorted(mm_features, key=lambda f: f.mm_position.offset):
            if mm_feature.data is None:
                continue
            for key in ("tgt_sizes", "video_tgt_sizes"):
                tgt = mm_feature.data.get(key)
                if tgt is None or not isinstance(tgt.data, torch.Tensor):
                    continue
                try:
                    target_sizes_list.append(_as_hw_pairs(tgt.data))
                except ValueError:
                    # Not an (N, 2) size table. The bounds/sizes match below
                    # turns this into a sequential fallback.
                    continue

        if not target_sizes_list or image_bounds.numel() == 0:
            logger.warning(
                "MiniCPM-V 4.7 canvas M-RoPE missing inputs: "
                "tgt_chunks=%s image_bounds=%s; using sequential positions.",
                len(target_sizes_list),
                tuple(image_bounds.shape) if image_bounds is not None else None,
            )
            return _sequential_positions(seq_len), 0

        target_sizes = torch.cat(target_sizes_list, dim=0)
        if target_sizes.ndim != 2 or target_sizes.shape[-1] != 2:
            logger.warning(
                "MiniCPM-V 4.7 canvas M-RoPE unexpected tgt_sizes shape %s; "
                "using sequential positions.",
                tuple(target_sizes.shape),
            )
            return _sequential_positions(seq_len), 0
        if image_bounds.shape[0] != target_sizes.shape[0]:
            logger.warning(
                "MiniCPM-V 4.7 canvas M-RoPE size mismatch: "
                "image_bounds=%s target_sizes=%s; using sequential positions.",
                tuple(image_bounds.shape),
                tuple(target_sizes.shape),
            )
            return _sequential_positions(seq_len), 0

        try:
            pos3d = _compute_canvas_single(
                input_ids,
                torch.arange(seq_len, dtype=torch.long),
                image_bounds,
                target_sizes,
                self._canvas_special_ids,
            )
        except Exception:
            logger.warning(
                "MiniCPM-V 4.7 canvas M-RoPE failed; "
                "falling back to sequential positions.",
                exc_info=True,
            )
            return _sequential_positions(seq_len), 0
        return pos3d, canvas_rope_delta(pos3d, seq_len)


# Canvas 3D M-RoPE: position ids for the slice canvas.
def compute_llm_grid(
    n_tokens: int, vision_grid_height: int, vision_grid_width: int
) -> tuple[int, int]:
    if n_tokens <= 0 or vision_grid_height <= 0 or vision_grid_width <= 0:
        return 0, 0
    vit_total = vision_grid_height * vision_grid_width
    if vit_total == n_tokens:
        return vision_grid_height, vision_grid_width
    if vit_total < n_tokens:
        return 0, 0
    factor = int(round(math.sqrt(vit_total / n_tokens)))
    if (
        factor > 0
        and vision_grid_height % factor == 0
        and vision_grid_width % factor == 0
    ):
        height, width = vision_grid_height // factor, vision_grid_width // factor
        if height * width == n_tokens:
            return height, width
    ratio = vision_grid_height / vision_grid_width
    width = max(1, int(round(math.sqrt(n_tokens / ratio))))
    height = n_tokens // width
    if height * width == n_tokens and height > 0:
        return height, width
    return 0, 0


def build_image_bounds(
    input_ids: torch.LongTensor, special_token_ids: dict
) -> torch.LongTensor:
    start_ids = [
        x
        for x in (
            special_token_ids.get("im_start_id"),
            special_token_ids.get("slice_start_id"),
        )
        if x is not None
    ]
    end_ids = [
        x
        for x in (
            special_token_ids.get("im_end_id"),
            special_token_ids.get("slice_end_id"),
        )
        if x is not None
    ]
    if not start_ids or not end_ids:
        return torch.zeros(0, 2, dtype=torch.long, device=input_ids.device)
    start_cond = torch.zeros_like(input_ids, dtype=torch.bool)
    end_cond = torch.zeros_like(input_ids, dtype=torch.bool)
    for sid in start_ids:
        start_cond |= input_ids == sid
    for eid in end_ids:
        end_cond |= input_ids == eid
    starts = torch.where(start_cond)[0] + 1
    ends = torch.where(end_cond)[0]
    n = min(len(starts), len(ends))
    if n == 0:
        return torch.zeros(0, 2, dtype=torch.long, device=input_ids.device)
    return torch.stack([starts[:n], ends[:n]], dim=-1)


def _compute_canvas_single(
    input_ids, position_ids_2d, image_bound, target_sizes, special_token_ids
):
    # Inference always handles one unpadded sequence. The packed (cu_seqlens)
    # form below is kept only to mirror the transformers reference
    # implementation of its canvas position computation.
    seq_len = input_ids.shape[0]
    cu_seqlens = torch.tensor([0, seq_len], device=input_ids.device, dtype=torch.long)
    return _compute_canvas_packed(
        position_ids_2d.unsqueeze(0),
        cu_seqlens,
        image_bound,
        target_sizes,
        input_ids.unsqueeze(0),
        special_token_ids,
    )[:, 0, :]


def _target_hw(target_sizes: torch.Tensor, index: int) -> tuple[int, int]:
    """Patch-grid height/width recorded for one image or slice."""
    return int(target_sizes[index, 0]), int(target_sizes[index, 1])


class _CanvasLayout(NamedTuple):
    """Grid of one canvas measured in LLM tokens.

    A canvas is a thumbnail plus the slices cut from it, tiled row-major.
    """

    thumb_h: int
    thumb_w: int
    """Thumbnail grid."""
    slice_h: int
    slice_w: int
    """Per-slice grid. Zero when the canvas has no slices."""
    cols: int
    """Slices sharing a canvas row. Zero when the canvas has no slices."""
    height: int
    width: int
    """Assembled canvas."""


def _canvas_layout(
    thumbnail: tuple[int, int, int],
    slices: list[tuple[int, int, int]],
    target_sizes: torch.Tensor,
) -> _CanvasLayout:
    """Measure the canvas for one ``(thumbnail, slices)`` group.

    ``thumbnail`` and each slice are ``(start, end, target_index)`` token spans.
    """
    t_bs, t_be, t_gi = thumbnail
    thumb_h, thumb_w = compute_llm_grid(t_be - t_bs, *_target_hw(target_sizes, t_gi))
    if not slices:
        return _CanvasLayout(
            thumb_h, thumb_w, 0, 0, 0, max(thumb_h, 1), max(thumb_w, 1)
        )

    cols = len(slices)
    for k in range(len(slices) - 1):
        if slices[k + 1][0] - slices[k][1] > 2:
            cols = k + 1
            break
    rows = len(slices) // cols if cols > 0 else 1
    if rows * cols != len(slices):
        rows, cols = 1, len(slices)

    s0_bs, s0_be, s0_gi = slices[0]
    slice_h, slice_w = compute_llm_grid(s0_be - s0_bs, *_target_hw(target_sizes, s0_gi))
    return _CanvasLayout(
        thumb_h, thumb_w, slice_h, slice_w, cols, rows * slice_h, cols * slice_w
    )


def _assign_canvas_span(
    pos3d: torch.Tensor,
    *,
    thumbnail: tuple[int, int, int],
    span_end: int,
    slices: list[tuple[int, int, int]],
    base: int,
    target_sizes: torch.Tensor,
) -> int:
    """Place one canvas at ``base``, writing its 3-D positions into ``pos3d``.

    Shared by the video-frame and single-image paths; they differ only in how
    the per-canvas base advances, not in the geometry placed here.

    Mutates ``pos3d`` in place, as the packed canvas computation has always
    done. Returns the distance to the next canvas base, so each caller keeps
    only its own base advancement.
    """
    t_bs, t_be, _ = thumbnail
    device = pos3d.device
    span_start = t_bs - 1  # the <im_start> token
    n_thumb = t_be - t_bs
    layout = _canvas_layout(thumbnail, slices, target_sizes)
    cols, canvas_H, canvas_W = layout.cols, layout.height, layout.width

    # base coat
    pos3d[0, 0, span_start:span_end] = base
    pos3d[1, 0, span_start:span_end] = base
    pos3d[2, 0, span_start:span_end] = base

    # <im_start> -> halo
    halo_lo = max(base - 1, 0)
    pos3d[1, 0, span_start] = halo_lo
    pos3d[2, 0, span_start] = halo_lo

    # <im_end> -> halo (right after thumbnail)
    if t_be < span_end:
        pos3d[1, 0, t_be] = base + canvas_H
        pos3d[2, 0, t_be] = base + canvas_W

    # thumbnail visual tokens
    llm_th, llm_tw = layout.thumb_h, layout.thumb_w
    if llm_th > 0 and llm_tw > 0 and llm_th * llm_tw == n_thumb:
        if slices and canvas_H > 0 and canvas_W > 0:
            h_c = torch.linspace(0, canvas_H - 1, llm_th, device=device).round().long()
            w_c = torch.linspace(0, canvas_W - 1, llm_tw, device=device).round().long()
        else:
            h_c = torch.arange(llm_th, device=device)
            w_c = torch.arange(llm_tw, device=device)
        h_idx = h_c.view(-1, 1).expand(-1, llm_tw).reshape(-1)
        w_idx = w_c.view(1, -1).expand(llm_th, -1).reshape(-1)
        pos3d[0, 0, t_bs:t_be] = base
        pos3d[1, 0, t_bs:t_be] = h_idx + base
        pos3d[2, 0, t_bs:t_be] = w_idx + base
    else:
        fallback = torch.arange(n_thumb, device=device, dtype=torch.long) + base
        for d in range(3):
            pos3d[d, 0, t_bs:t_be] = fallback

    # slice visual tokens + slice specials
    for k, (s_bs, s_be, s_gi) in enumerate(slices):
        n_s = s_be - s_bs
        sh_k, sw_k = compute_llm_grid(n_s, *_target_hw(target_sizes, s_gi))
        h_off = (k // cols) * layout.slice_h
        w_off = (k % cols) * layout.slice_w

        slice_start_pos = s_bs - 1
        if slice_start_pos >= span_start:
            pos3d[1, 0, slice_start_pos] = base + h_off
            pos3d[2, 0, slice_start_pos] = base + w_off

        slice_end_pos = s_be
        if sh_k > 0 and sw_k > 0:
            end_h = h_off + sh_k - 1
            end_w = w_off + sw_k - 1
        else:
            end_h = h_off
            end_w = w_off
        if slice_end_pos < span_end:
            pos3d[1, 0, slice_end_pos] = base + end_h
            pos3d[2, 0, slice_end_pos] = base + end_w

        if sh_k > 0 and sw_k > 0 and sh_k * sw_k == n_s:
            h_idx = (
                torch.arange(sh_k, device=device)
                .view(-1, 1)
                .expand(-1, sw_k)
                .reshape(-1)
            )
            w_idx = (
                torch.arange(sw_k, device=device)
                .view(1, -1)
                .expand(sh_k, -1)
                .reshape(-1)
            )
            pos3d[0, 0, s_bs:s_be] = base
            pos3d[1, 0, s_bs:s_be] = h_idx + h_off + base
            pos3d[2, 0, s_bs:s_be] = w_idx + w_off + base
        else:
            fallback = torch.arange(n_s, device=device, dtype=torch.long) + base
            for d in range(3):
                pos3d[d, 0, s_bs:s_be] = fallback

    # \n between slice rows -> W = right edge + 1.
    # H stays within the ended row (does NOT cross to next row): the \n sits at
    # the right-outside of the row that just ended, at H = that row's last row
    # and W = one column past the rightmost slice.
    for k in range(len(slices) - 1):
        gap_s = slices[k][1]
        gap_e = slices[k + 1][0]
        if gap_e - gap_s <= 2:
            continue
        boundary_h = ((k // cols) + 1) * layout.slice_h - 1
        right_edge_w = cols * layout.slice_w
        for nl_pos in range(gap_s + 1, gap_e - 1):
            pos3d[1, 0, nl_pos] = base + boundary_h
            pos3d[2, 0, nl_pos] = base + right_edge_w

    return max(canvas_H, canvas_W) + 1


def _compute_canvas_packed(
    position_ids_2d,
    cu_seqlens,
    image_bound,
    target_sizes,
    input_ids,
    special_token_ids,
):
    if position_ids_2d.ndim == 1:
        position_ids_2d = position_ids_2d.unsqueeze(0)
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)

    has_images = (
        isinstance(image_bound, torch.Tensor)
        and image_bound.numel() > 0
        and isinstance(target_sizes, torch.Tensor)
        and target_sizes.numel() > 0
    )
    if not has_images:
        return position_ids_2d.unsqueeze(0).expand(3, -1, -1).contiguous()

    device = position_ids_2d.device
    pos3d = position_ids_2d.unsqueeze(0).expand(3, -1, -1).clone()
    ids_flat = input_ids.view(-1)

    im_start_id = special_token_ids.get("im_start_id")
    im_end_id = special_token_ids.get("im_end_id")
    slice_start_id = special_token_ids.get("slice_start_id")
    slice_end_id = special_token_ids.get("slice_end_id")

    structural_ids = {
        v
        for v in (im_start_id, im_end_id, slice_start_id, slice_end_id)
        if v is not None
    }

    num_seqs = len(cu_seqlens) - 1
    img_ptr = 0
    num_imgs = len(image_bound)

    newline_id = special_token_ids.get("newline_id")
    gap_ok_ids = structural_ids | ({newline_id} if newline_id is not None else set())

    def _group_end_pos(g, s_end):
        """First position after a group's trailing structural tokens.

        Matches the transformers reference, which closes a span over every
        marker in ``markers`` -- newlines included. The chat template joins
        content parts with ``\\n``, so an image followed by text always ends with
        one; leaving it out moves that newline out of the canvas and pushes
        every later position one step forward.
        """
        le = g["slices"][-1][1] if g["slices"] else g["thumbnail"][1]
        ge = le
        while ge < s_end and ids_flat[ge].item() in gap_ok_ids:
            ge += 1
        return min(ge, s_end)

    def _is_tight_gap(g_prev, g_next, s_end):
        """Check if only structural/nl tokens sit between two groups."""
        ge = _group_end_pos(g_prev, s_end)
        ns = g_next["thumbnail"][0] - 1  # im_start of next group
        return all(ids_flat[p].item() in gap_ok_ids for p in range(ge, ns))

    for si in range(num_seqs):
        s_start = cu_seqlens[si].item()
        s_end = cu_seqlens[si + 1].item()
        if s_start >= s_end:
            continue

        seq_imgs = []
        while img_ptr < num_imgs:
            bs_ = image_bound[img_ptr, 0].item()
            be_ = image_bound[img_ptr, 1].item()
            if bs_ >= s_start and be_ <= s_end:
                seq_imgs.append((bs_, be_, img_ptr))
                img_ptr += 1
            else:
                break
        if not seq_imgs:
            continue

        # ---- group images: thumbnail + following slices ----
        # merge consecutive no-slice frames into one video group
        raw_groups = []
        cur_group: dict[str, Any] | None = None
        for bs_, be_, gi in seq_imgs:
            marker_pos = bs_ - 1
            is_slice = (
                slice_start_id is not None
                and marker_pos >= s_start
                and ids_flat[marker_pos].item() == slice_start_id
            )
            if is_slice and cur_group is not None:
                cur_group["slices"].append((bs_, be_, gi))
            else:
                if cur_group is not None:
                    raw_groups.append(cur_group)
                cur_group = {"thumbnail": (bs_, be_, gi), "slices": []}
        if cur_group is not None:
            raw_groups.append(cur_group)

        # merge consecutive tightly-adjacent groups into video groups.
        # Two groups are "tightly adjacent" if the gap between them
        # contains only structural tokens and \n (no real text).
        # This handles both no-slice frames and frames-with-slices.
        groups = []
        i_g = 0
        while i_g < len(raw_groups):
            video_frames = [raw_groups[i_g]]
            j_g = i_g + 1
            while j_g < len(raw_groups) and _is_tight_gap(
                raw_groups[j_g - 1], raw_groups[j_g], s_end
            ):
                video_frames.append(raw_groups[j_g])
                j_g += 1
            if len(video_frames) == 1:
                groups.append(raw_groups[i_g])
            else:
                groups.append({"video_frames": video_frames})
            i_g = j_g

        # ---- assign 3-D positions ----
        pos = 0
        cursor = s_start

        for group in groups:
            is_video_group = "video_frames" in group

            if is_video_group:
                # === Video group: multiple frames merged ===
                # Each frame gets its own base_f so different frames have
                # distinct spatial positions.  Inter-frame gap tokens (\n
                # etc.) are pure 1D text.  After each gap, pos advances
                # by 1 extra so halo_lo of the next frame does not
                # collide with the last gap token.
                frames = group["video_frames"]
                first_frame = frames[0]
                last_frame = frames[-1]

                # group_start = <im_start> of first frame
                group_start = first_frame["thumbnail"][0] - 1
                # group_end = after last structural token of last frame
                group_end = _group_end_pos(last_frame, s_end)

                # text before video group
                text_len = group_start - cursor
                if text_len > 0:
                    t_pos = (
                        torch.arange(text_len, device=device, dtype=torch.long) + pos
                    )
                    for d in range(3):
                        pos3d[d, 0, cursor:group_start] = t_pos
                    pos += text_len

                # process each frame with its own per-frame base
                frame_cursor = group_start
                for vf in frames:
                    t_bs = vf["thumbnail"][0]
                    vf_slices = vf["slices"]
                    im_start_pos = t_bs - 1

                    # --- inter-frame gap: pure 1D text T=H=W=pos ---
                    gap_len = im_start_pos - frame_cursor
                    if gap_len > 0:
                        t_pos = (
                            torch.arange(gap_len, device=device, dtype=torch.long) + pos
                        )
                        for d in range(3):
                            pos3d[d, 0, frame_cursor:im_start_pos] = t_pos
                        pos += gap_len
                        pos += 1  # buffer: prevents halo_lo collision

                    # --- per-frame base ---
                    base_f = pos
                    frame_end = _group_end_pos(vf, group_end)
                    pos = base_f + _assign_canvas_span(
                        pos3d,
                        thumbnail=vf["thumbnail"],
                        span_end=frame_end,
                        slices=vf_slices,
                        base=base_f,
                        target_sizes=target_sizes,
                    )
                    frame_cursor = frame_end

                # trailing tokens after last frame (if any)
                if frame_cursor < group_end:
                    trail_len = group_end - frame_cursor
                    t_pos = (
                        torch.arange(trail_len, device=device, dtype=torch.long) + pos
                    )
                    for d in range(3):
                        pos3d[d, 0, frame_cursor:group_end] = t_pos
                    pos += trail_len

                cursor = group_end

            else:
                # === Single image group (thumbnail + optional slices) ===
                slices = group["slices"]

                group_start = group["thumbnail"][0] - 1
                group_end = _group_end_pos(group, s_end)

                # text before group
                text_len = group_start - cursor
                if text_len > 0:
                    t_pos = (
                        torch.arange(text_len, device=device, dtype=torch.long) + pos
                    )
                    for d in range(3):
                        pos3d[d, 0, cursor:group_start] = t_pos
                    pos += text_len

                base = pos
                pos = base + _assign_canvas_span(
                    pos3d,
                    thumbnail=group["thumbnail"],
                    span_end=group_end,
                    slices=slices,
                    base=base,
                    target_sizes=target_sizes,
                )
                cursor = group_end

        # remaining text
        rem = s_end - cursor
        if rem > 0:
            t_pos = torch.arange(rem, device=device, dtype=torch.long) + pos
            for d in range(3):
                pos3d[d, 0, cursor:s_end] = t_pos

    return pos3d


def canvas_rope_delta(position_ids: torch.Tensor, seq_len: int) -> int:
    """Offset applied to the 1-D positions of tokens decoded after the prompt.

    The first decoded token sits one past the last prefill position.
    """
    return int(position_ids.amax().item()) + 1 - seq_len

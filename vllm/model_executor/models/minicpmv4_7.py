# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only MiniCPM-V 4.7 model.

Same vision/LLM stack as MiniCPM-V 4.6, plus canvas 3D M-RoPE.
"""

import logging
from collections.abc import Iterable

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFeatureSpec

from .minicpmv import MiniCPMVDummyInputsBuilder
from .minicpmv4_6 import (
    MiniCPMV4_6ForConditionalGeneration,
    MiniCPMV4_6MultiModalProcessor,
    MiniCPMV4_6ProcessingInfo,
    _stack_vit_merger_qkv,
)

logger = init_logger(__name__)


class MiniCPMV4_7ProcessingInfo(MiniCPMV4_6ProcessingInfo):
    def _mm_max_slice_nums(self, **kwargs: object) -> int | None:
        merged = self.ctx.get_merged_mm_kwargs(kwargs, modality="image")
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


@MULTIMODAL_REGISTRY.register_processor(
    MiniCPMV4_7MultiModalProcessor,
    info=MiniCPMV4_7ProcessingInfo,
    dummy_inputs=MiniCPMVDummyInputsBuilder,
)
class MiniCPMV4_7ForConditionalGeneration(MiniCPMV4_6ForConditionalGeneration):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self._init_canvas_mrope(vllm_config)

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
        configured_ids = {
            "im_start_id": getattr(config, "image_start_id", None),
            "im_end_id": getattr(config, "image_end_id", None),
            "slice_start_id": getattr(config, "slice_start_id", None),
            "slice_end_id": getattr(config, "slice_end_id", None),
            "newline_id": getattr(config, "newline_id", None),
        }
        if all(value is not None for value in configured_ids.values()):
            self._canvas_special_ids = {
                name: int(value) for name, value in configured_ids.items()
            }
            return

        try:
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
        except Exception:
            logger.warning(
                "Failed to resolve MiniCPM-V 4.7 canvas M-RoPE token IDs; "
                "falling back to sequential positions.",
                exc_info=True,
            )
            self._uses_canvas_mrope = False

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
            positions = torch.arange(seq_len).unsqueeze(0).expand(3, -1)
            return positions, 0

        from .mrope_minicpmv4_7 import (
            _compute_canvas_single,
            build_image_bounds,
            canvas_rope_delta,
        )

        input_ids = torch.tensor(input_tokens, dtype=torch.long)
        image_bounds = build_image_bounds(input_ids, self._canvas_special_ids)

        target_sizes_list: list[torch.Tensor] = []
        debug_shapes = logger.isEnabledFor(logging.DEBUG)
        raw_tgt_shapes: list[list[int]] = []
        for mm_feature in sorted(mm_features, key=lambda f: f.mm_position.offset):
            if mm_feature.data is None:
                continue
            for key in ("tgt_sizes", "video_tgt_sizes"):
                tgt = mm_feature.data.get(key)
                if tgt is None or tgt.data is None:
                    continue
                if debug_shapes:
                    raw_tgt_shapes.append(list(tgt.data.shape))
                try:
                    target_sizes_list.append(_as_hw_pairs(tgt.data))
                except ValueError as exc:
                    logger.warning(
                        "MiniCPM-V 4.7 canvas M-RoPE bad %s shape %s: %s",
                        key,
                        tuple(tgt.data.shape),
                        exc,
                    )

        if not target_sizes_list or image_bounds.numel() == 0:
            logger.warning(
                "MiniCPM-V 4.7 canvas M-RoPE missing inputs: "
                "tgt_chunks=%s image_bounds=%s; using sequential positions.",
                len(target_sizes_list),
                tuple(image_bounds.shape) if image_bounds is not None else None,
            )
            positions = torch.arange(seq_len).unsqueeze(0).expand(3, -1)
            return positions, 0

        target_sizes = torch.cat(target_sizes_list, dim=0)
        if target_sizes.ndim != 2 or target_sizes.shape[-1] != 2:
            logger.warning(
                "MiniCPM-V 4.7 canvas M-RoPE unexpected tgt_sizes shape %s; "
                "using sequential positions.",
                tuple(target_sizes.shape),
            )
            positions = torch.arange(seq_len).unsqueeze(0).expand(3, -1)
            return positions, 0
        if image_bounds.shape[0] != target_sizes.shape[0]:
            logger.warning(
                "MiniCPM-V 4.7 canvas M-RoPE size mismatch: "
                "image_bounds=%s target_sizes=%s; using sequential positions.",
                tuple(image_bounds.shape),
                tuple(target_sizes.shape),
            )
            positions = torch.arange(seq_len).unsqueeze(0).expand(3, -1)
            return positions, 0

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
            positions = torch.arange(seq_len).unsqueeze(0).expand(3, -1)
            return positions, 0
        if debug_shapes:
            logger.debug(
                "MiniCPM-V 4.7 canvas M-RoPE: seq=%s bounds=%s tgt=%s raw_tgt=%s "
                "pos_max=%s",
                seq_len,
                tuple(image_bounds.shape),
                tuple(target_sizes.shape),
                raw_tgt_shapes,
                [int(pos3d[d].max()) for d in range(3)],
            )
        return pos3d, canvas_rope_delta(pos3d, seq_len)

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(
            _stack_vit_merger_qkv(weights),
            mapper=self.hf_to_vllm_mapper,
        )

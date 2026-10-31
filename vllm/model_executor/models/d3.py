# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decision 3.0 (vllm-sr/d3): a Qwen3.5 backbone with an answer-code readout.

The checkpoint is a bare `Qwen3_5Model` plus `readout.safetensors`, one row per
answer code, and `decision_config.json`. The readout scores the codes at the
last prompt token and is served as the `score` head of a classification model.
`D3ForSequenceClassificationConfig` applies the checkpoint's attention mode and
label space.
"""

from typing import TYPE_CHECKING

from transformers.models.qwen3_vl import Qwen3VLProcessor

from vllm.config import VllmConfig
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.multimodal import MULTIMODAL_REGISTRY

from .adapters import as_seq_cls_model
from .interfaces_base import default_pooling_type
from .qwen3_5 import Qwen3_5ForConditionalGeneration, Qwen3_5ProcessingInfo
from .qwen3_vl import Qwen3VLDummyInputsBuilder, Qwen3VLMultiModalProcessor
from .utils import WeightsMapper

READOUT_WEIGHTS = "readout.safetensors"

# The media budgets of the checkpoint's runtime (`d3_runtime.py`), which differ
# from its processor config: images at 65,536 to 1,638,400 pixels, video frames
# at up to 200,704 pixels and at most 32 frames per video.
IMAGE_SIZE = {"shortest_edge": 65_536, "longest_edge": 1_638_400}
VIDEO_FRAME_MAX_PIXELS = 200_704
VIDEO_FRAME_MIN_PIXELS = 4_096
VIDEO_MAX_FRAMES = 32


class D3ProcessingInfo(Qwen3_5ProcessingInfo):
    def get_hf_processor(self, **kwargs: object) -> Qwen3VLProcessor:
        processor = super().get_hf_processor(**kwargs)
        processor.image_processor.size = dict(IMAGE_SIZE)
        video = processor.video_processor
        factor = video.patch_size * video.merge_size
        video.size = {
            "shortest_edge": VIDEO_FRAME_MIN_PIXELS,
            "longest_edge": VIDEO_FRAME_MAX_PIXELS * VIDEO_MAX_FRAMES,
        }
        video.cap_pixels_per_frame = True
        video.max_video_tokens = VIDEO_FRAME_MAX_PIXELS // factor**2
        video.do_sample_frames = False
        return processor


if TYPE_CHECKING:
    _D3ForSequenceClassificationBase = Qwen3_5ForConditionalGeneration
else:
    _D3ForSequenceClassificationBase = as_seq_cls_model(Qwen3_5ForConditionalGeneration)


@default_pooling_type(seq_pooling_type="LAST")
@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=D3ProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class D3ForSequenceClassification(_D3ForSequenceClassificationBase):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "language_model.": "language_model.model.",
            "mtp.": None,
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        model_config = vllm_config.model_config
        self.secondary_weights = [
            DefaultModelLoader.Source(
                model_or_path=model_config.model,
                revision=model_config.revision,
                prefix="score.",
                allow_patterns_overrides=[READOUT_WEIGHTS],
            )
        ]

    def is_unused_checkpoint_weight(self, name: str) -> bool:
        # The readout's only tensor, loaded through `secondary_weights`. A
        # checkpoint without a weight index lists its file with the backbone's.
        return name == "weight"

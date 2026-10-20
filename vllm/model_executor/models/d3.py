# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decision 3.0 (vllm-sr/d3): a Qwen3.5 backbone with an answer-code readout.

The checkpoint is a bare `Qwen3_5Model` plus `readout.safetensors`, one row per
answer code, and `decision_config.json`. The readout scores the codes at the
last prompt token and is served as the `score` head of a classification model.
`D3ForSequenceClassificationConfig` applies the checkpoint's attention mode and
label space.

Under the `prefix_bidir` attention mode a prompt is a state block (the system
turn, the media, `State:` and the state) followed by a question block. The full
attention layers read each block in both directions and the question block also
reads the state block, so the state never sees a question; the linear-attention
layers stay causal. Each request carries its state block's token count in
`PoolingParams.extra_kwargs[STATE_TOKENS]`, and `D3PrefixBidirModelState` makes
the two blocks the PrefixLM ranges of its attention.
"""

from typing import TYPE_CHECKING

from transformers.models.qwen3_vl import Qwen3VLProcessor

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.platforms import current_platform
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.model_states.default import DefaultModelState

from .adapters import as_seq_cls_model
from .interfaces_base import default_pooling_type
from .qwen3_5 import Qwen3_5ForConditionalGeneration, Qwen3_5ProcessingInfo
from .qwen3_vl import Qwen3VLDummyInputsBuilder, Qwen3VLMultiModalProcessor
from .utils import WeightsMapper

logger = init_logger(__name__)

READOUT_WEIGHTS = "readout.safetensors"
# The `PoolingParams.extra_kwargs` key of a `prefix_bidir` request's state block
# token count, set by the System One strategy.
STATE_TOKENS = "d3_state_tokens"

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


class D3PrefixBidirModelState(DefaultModelState):
    """Model Runner V2 state of a `prefix_bidir` checkpoint: a request's state
    block and question block are the PrefixLM ranges of its attention, so the
    attention is causal across them and bidirectional within each."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.blocks: dict[str, list[tuple[int, int]]] = {}

    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None:
        super().add_request(req_index, new_req_data)
        params = new_req_data.pooling_params
        state = (params.extra_kwargs or {}).get(STATE_TOKENS) if params else None
        length = len(new_req_data.prompt_token_ids or ())
        if isinstance(state, int) and 0 < state < length:
            blocks = [(0, state - 1), (state, length - 1)]
        else:
            # Without a state block size (warm-up runs, or a request that did
            # not go through /v1/systemone) the whole prompt is one block.
            logger.debug_once("A prefix_bidir request has no state block size")
            blocks = [(0, length - 1)]
        self.blocks[new_req_data.req_id] = blocks

    def remove_request(self, req_id: str) -> None:
        super().remove_request(req_id)
        self.blocks.pop(req_id, None)

    def get_mm_prefix_ranges(
        self, input_batch: InputBatch
    ) -> dict[int, list[tuple[int, int]]] | None:
        ranges = {
            i: self.blocks[req_id]
            for i, req_id in enumerate(input_batch.req_ids)
            if req_id in self.blocks
        }
        return ranges or None


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
        if model_config.hf_config.decision_config.get("attention_mode") == (
            "prefix_bidir"
        ):
            # The blocks reach the attention only through Model Runner V2 and
            # backends that read PrefixLM ranges, which multimodal checkpoints
            # select while their multimodal inputs are enabled.
            if not (
                current_platform.is_cuda_alike()
                and vllm_config.use_v2_model_runner
                and model_config.is_mm_prefix_lm
            ):
                raise ValueError(
                    "Decision 3.0 prefix_bidir checkpoints need a CUDA or ROCm "
                    "GPU, Model Runner V2 and multimodal inputs enabled"
                )
            # The other attention modes keep the runner's default state.
            self.get_model_state_cls = lambda: D3PrefixBidirModelState

    def is_unused_checkpoint_weight(self, name: str) -> bool:
        # The readout's only tensor, loaded through `secondary_weights`. A
        # checkpoint without a weight index lists its file with the backbone's.
        return name == "weight"

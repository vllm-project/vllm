# SPDX-License-Identifier: Apache-2.0
"""Attention layer with BOK."""
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import torch

from vllm.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionType,
)
from vllm.attention.layer import Attention
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.logger import init_logger
from vllm.utils import cdiv
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from py_bok import (
    create_op,
    forward_inplace,
    AttentionLayerDimensions,
    DeviceDataType,
    RotaryEmbedding,
    RotaryScalingType,
    RotaryPositionalEmbeddingType,
)


if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.worker.gpu_input_batch import InputBatch
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

logger = init_logger(__name__)


class BokAttentionBackend(AttentionBackend):

    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [64, 128, 256]

    @staticmethod
    def get_name() -> str:
        return "BOK_VLLM_V1"

    @staticmethod
    def get_impl_cls() -> type["BokAttentionImpl"]:
        return BokAttentionImpl

    @staticmethod
    def get_metadata_cls() -> type["BokAttentionMetadata"]:
        return BokAttentionMetadata

    @staticmethod
    def get_builder_cls() -> type["BokAttentionMetadataBuilder"]:
        return BokAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> tuple[int, ...]:
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        return (num_blocks, 2, block_size, num_kv_heads, head_size)


@dataclass
class BokAttentionMetadata:
    stream = torch.cuda.Stream()

    # Create basic parameters for a smoke test

    # Model dimensions.
    num_layers = 1

    # Attention layer dimensions.
    num_q_heads = 96
    num_kv_heads = 8
    head_dim = 128
    dims = AttentionLayerDimensions(num_q_heads, num_kv_heads, head_dim)
    max_attention_window_size = 2048

    # Rotary embedding dimensions.
    rotary_embedding_dim = 128
    rotary_embedding_base = 10000
    rotary_embedding_max_positions = 2048
    rotary_embedding_scale = 0
    rotary_embedding = RotaryEmbedding(
        type=RotaryPositionalEmbeddingType.GPT_NEOX,
        rotaryEmbeddingDim=rotary_embedding_dim,
        rotaryEmbeddingBase=rotary_embedding_base,
        rotaryEmbeddingScale=rotary_embedding_scale,
        rotaryEmbeddingMaxPositions=rotary_embedding_max_positions,
        rotaryScalingType=RotaryScalingType.NONE,
    )
    # KV-cache dimensions.
    num_tokens_per_block = 32
    max_blocks_per_sequence = 32

    # Create the attention operation
    fp8_output_scaling = torch.tensor(
        [1.0], device=torch.device("cuda"), dtype=torch.float32
    )
    kv_scale_orig_quant = torch.tensor(
        [1.0], device=torch.device("cuda"), dtype=torch.float32
    )
    kv_scale_quant_orig = torch.tensor(
        [1.0], device=torch.device("cuda"), dtype=torch.float32
    )
    op = create_op(
        inputDataType=DeviceDataType.BF16,
        outputDataType=DeviceDataType.BF16,
        attentionLayerDimensions=dims,
        rotaryEmbedding=rotary_embedding,
        numTokensPerBlock=num_tokens_per_block,
        qScaling=1.0,
        maxAttentionWindowSize=max_attention_window_size,
        cyclicAttentionWindowSize=max_attention_window_size,
        maxNumBlocksPerSequence=max_blocks_per_sequence,
        fp8OutputScaling=fp8_output_scaling,
        kvScaleOrigQuant=kv_scale_orig_quant,
        kvScaleQuantOrig=kv_scale_quant_orig,
    )



class BokAttentionMetadataBuilder:

    def __init__(self, runner: "GPUModelRunner"):
        model_config = runner.model_config

        self.runner = runner
        self.num_heads_q = model_config.get_num_attention_heads(runner.parallel_config)
        self.num_heads_kv = model_config.get_num_kv_heads(runner.parallel_config)
        self.headdim = model_config.get_head_size()
        self.page_size = self.runner.block_size

        # Sliding window size to be used with the AOT scheduler will be
        # populated on first build() call.
        self.aot_schedule = False
        self.aot_sliding_window: Optional[tuple[int, int]] = None

    def use_cascade_attention(self, *args, **kwargs) -> bool:
        return False

    def reorder_batch(self, input_batch, scheduler_output) -> bool:
        return False

    def build(
        self,
        num_reqs: int,
        num_actual_tokens: int,
        max_query_len: int,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
    ):
        max_seq_len = self.runner.seq_lens_np[:num_reqs].max()
        query_start_loc = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens
        block_table = self.runner.input_batch.block_table.get_device_tensor()[:num_reqs]
        slot_mapping = (
            self.runner.slot_mapping_cpu[:num_actual_tokens]
            .to(self.runner.device, non_blocking=True)
            .long()
        )

        if self.aot_sliding_window is None:
            self.aot_sliding_window = (-1, -1)
            # For the AOT scheduler we need the sliding window value to be
            # constant for all layers to. We have to populate this on the first
            # build() call so the layers are constructed (cannot populate)
            # in __init__.
            if self.aot_schedule:
                sliding_window_configs = _get_sliding_window_configs(
                    self.runner.vllm_config
                )
                if len(sliding_window_configs) == 1:
                    sliding_window_config = sliding_window_configs.pop()
                    if sliding_window_config is not None:
                        self.aot_sliding_window = sliding_window_config
                elif len(sliding_window_configs) > 1:
                    self.aot_schedule = False

        attn_metadata = BokAttentionMetadata(
            num_actual_tokens=num_actual_tokens,
            max_query_len=max_query_len,
            query_start_loc=query_start_loc,
            max_seq_len=max_seq_len,
            seq_lens=seq_lens,
            block_table=block_table,
            slot_mapping=slot_mapping,
        )
        return attn_metadata


class BokAttentionImpl(AttentionImpl):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[list[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        blocksparse_params: Optional[dict[str, Any]] = None,
        logits_soft_cap: Optional[float] = None,
        attn_type: AttentionType = AttentionType.DECODER,
        use_irope: bool = False,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        if sliding_window is None:
            self.sliding_window = (-1, -1)
        else:
            self.sliding_window = (sliding_window - 1, 0)
        self.kv_cache_dtype = kv_cache_dtype
        self.logits_soft_cap = 0

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        support_head_sizes = BokAttentionBackend.get_supported_head_sizes()
        if head_size not in support_head_sizes:
            raise ValueError(
                f"Head size {head_size} is not supported by BOK. "
                f"Supported head sizes are: {support_head_sizes}. "
                "Set VLLM_USE_V1=0 to use another attention backend."
            )

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "Encoder self-attention and "
                "encoder/decoder cross-attention "
                "are not implemented for "
                "BOK"
            )
        self.use_irope = use_irope

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: Optional[BokAttentionMetadata],
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with BOK attention kernels.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache = [2, num_blocks, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        NOTE: FP8 quantization, flash-attn expect the size of
              {q,k,v}_descale to be (num_sequences, num_kv_heads).
              We use torch's .expand() to avoid duplicating values
        """
        assert output is not None, "Output tensor must be provided."

        stream = torch.cuda.Stream()

        # Create basic parameters for a smoke test

        # Model dimensions.
        num_layers = 1

        # Attention layer dimensions.
        num_q_heads = 96
        num_kv_heads = 8
        head_dim = 128
        dims = AttentionLayerDimensions(num_q_heads, num_kv_heads, head_dim)
        max_attention_window_size = 2048

        # Rotary embedding dimensions.
        rotary_embedding_dim = 128
        rotary_embedding_base = 10000
        rotary_embedding_max_positions = 2048
        rotary_embedding_scale = 0
        rotary_embedding = RotaryEmbedding(
            type=RotaryPositionalEmbeddingType.GPT_NEOX,
            rotaryEmbeddingDim=rotary_embedding_dim,
            rotaryEmbeddingBase=rotary_embedding_base,
            rotaryEmbeddingScale=rotary_embedding_scale,
            rotaryEmbeddingMaxPositions=rotary_embedding_max_positions,
            rotaryScalingType=RotaryScalingType.NONE,
        )
        # KV-cache dimensions.
        num_tokens_per_block = 32
        max_blocks_per_sequence = 32

        # Create the attention operation
        fp8_output_scaling = torch.tensor(
            [1.0], device=torch.device("cuda"), dtype=torch.float32
        )
        kv_scale_orig_quant = torch.tensor(
            [1.0], device=torch.device("cuda"), dtype=torch.float32
        )
        kv_scale_quant_orig = torch.tensor(
            [1.0], device=torch.device("cuda"), dtype=torch.float32
        )
        op = create_op(
            inputDataType=DeviceDataType.BF16,
            outputDataType=DeviceDataType.BF16,
            attentionLayerDimensions=dims,
            rotaryEmbedding=rotary_embedding,
            numTokensPerBlock=num_tokens_per_block,
            qScaling=1.0,
            maxAttentionWindowSize=max_attention_window_size,
            cyclicAttentionWindowSize=max_attention_window_size,
            maxNumBlocksPerSequence=max_blocks_per_sequence,
            fp8OutputScaling=fp8_output_scaling,
            kvScaleOrigQuant=kv_scale_orig_quant,
            kvScaleQuantOrig=kv_scale_quant_orig,
        )

        # Parameters of the particular op call.
        batch_size = 1
        context_length = 32
        sequence_length = context_length + 1
        context_lengths_host = torch.tensor([context_length], dtype=torch.uint32)
        context_lengths_device = context_lengths_host.cuda()
        sequence_lengths = torch.tensor(
            [sequence_length], device=torch.device("cuda"), dtype=torch.uint32
        )
        past_kv_lengths_host = torch.tensor([context_length], dtype=torch.uint32)
        kv_cache_block_offsets = torch.zeros(
            batch_size,
            max_blocks_per_sequence,
            2,
            device=torch.device("cuda"),
            dtype=torch.uint32,
        )

        q_size = num_q_heads * head_dim
        kv_size = num_kv_heads * head_dim
        qkv_size = q_size + 2 * kv_size
        input_tensor = torch.randn(batch_size, qkv_size, device=torch.device("cuda"))

        actual_kv_cache_pools = [
            torch.zeros(
                num_tokens_per_block * 2 * num_kv_heads * head_dim,
                device=torch.device("cuda"),
            )
            for _ in range(num_layers)
        ]
        kv_cache_pool_mapping_host = torch.zeros(num_layers, dtype=torch.uint32)
        kv_cache_pool_pointers_host = torch.tensor(
            [t.data_ptr() for t in actual_kv_cache_pools], dtype=torch.int64
        )
        host_is_generation_flags = torch.tensor([True], dtype=torch.bool)
        rotary_cos_sin = torch.zeros(
            rotary_embedding_max_positions,
            rotary_embedding_dim,
            2,
            device=torch.device("cuda"),
            dtype=torch.float32,
        )
        output_scaling = torch.tensor(
            [1.0], device=torch.device("cuda"), dtype=torch.float32
        )
        output = torch.zeros(batch_size, q_size, device=torch.device("cuda"))
        workspace = torch.zeros(1 << 30, device=torch.device("cuda"))
        layer_index = 0
        forward_inplace(
            op,
            input_tensor,  # 2d device tensor of floats with dimensions (*, *)
            context_lengths_host,  # 1d host tensor of uint32_t's
            context_lengths_device,  # 1d device tensor of uint32_t's
            sequence_lengths,  # 1d device tensor of uint32_t's
            past_kv_lengths_host,  # 1d host tensor of uint32_t's
            kv_cache_block_offsets,  # 3d device tensor of offsets in the pool of uint32_t's with dimensions (*, *, 2)
            kv_cache_pool_pointers_host,  # 1d host tensor of pointers as int64_t's
            kv_cache_pool_mapping_host,  # 1d host tensor of uint32_t's
            host_is_generation_flags,  # 1d device tensor of bools
            output_scaling,  # 1d device tensor of floats
            rotary_cos_sin,  # Device tensor of floats with dimensions (*, * , 2)
            layer_index,  # int
            output,  # Any 2d device tensor
            workspace,  # Any device tensor,
            stream.cuda_stream,  # The stream to run the kernels on.
        )

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention layer with FlashInfer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, List

import torch
import numpy as np
import math

from vllm.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionType,
)
from vllm.logger import init_logger
from vllm.v1.attention.backends.utils import (
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.v1.worker.block_table import BlockTable

from vllm.utils import current_stream

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.worker.gpu_input_batch import InputBatch
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from py_tke import (
    create_op,
    run_generation_inplace,
    run_context_inplace,
    calculate_workspace_size,
    AttentionLayerDimensions,
    DeviceDataType,
    RotaryEmbedding,
    RotaryScalingType,
    RotaryPositionalEmbeddingType,
    PrefixCacheConfiguration,
    AttentionOp,
    BlockOffsetLayout,
)

logger = init_logger(__name__)

rope_scaling_type_mapping = {
    "none": RotaryScalingType.NONE,
    "linear": RotaryScalingType.LINEAR,
    "dynamic": RotaryScalingType.DYNAMIC,
    "longrope": RotaryScalingType.LONG,
    "llama3": RotaryScalingType.LLAMA3,
}


class TkeAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [64, 128, 256]

    @staticmethod
    def get_name() -> str:
        return "TKE"

    @staticmethod
    def get_impl_cls() -> type[TkeImpl]:
        return TkeImpl

    @staticmethod
    def get_metadata_cls() -> type[TkeMetadata]:
        return TkeMetadata

    @staticmethod
    def get_builder_cls() -> type[TkeMetadataBuilder]:
        return TkeMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> tuple[int, ...]:
        return (num_blocks, 2, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order() -> tuple[int, ...]:
        # `stride_order` indicates the permutation that gets us from
        # `get_kv_cache_shape` to the actual memory layout we want.
        # NOTE: 3 and 2 are swapped because the TRTLLM layout within blocks is [num_heads, num_tokens, dimension]
        return (0, 1, 3, 2, 4)


@dataclass
class TkeMetadata:

    # The shared attention metadata.
    common_attn_metadata: CommonAttentionMetadata

    # The fixed state of the attention operation.
    op: AttentionOp

    # The dimensions of the attention operation.
    attention_layer_dimensions: AttentionLayerDimensions

    # Sequence lengths denotes the length of the full sequence, including already processed tokens.
    # Dimensions: [num_sequences], dtype: uint32
    sequence_lengths_host: torch.Tensor

    # Some space on device that the caller should allocate and pass to the forward calls.
    # The dimension is obtained by calling calculate_workspace_size, in bytes.
    workspace: torch.Tensor

    # The rotary cos/sin cache is a precomputed cache of the rotary cos/sin rotation coefficients.
    # Dimensions: [num_positions, embedding_dim / 2, 2], which usually gets 'shortened' to [num_positions, embedding_dim], dtype: float32
    rotary_cos_sin_cache: torch.Tensor

    # The table of kv-cache offsets. This is provided by vLLM and should be passed to the forward calls.
    # It informs the attention kernels about where in the cache the data is located for each page / block of each sequence.
    block_table: BlockTable

    # The number of sequences in context phase.
    num_context_sequences: int

    # The number of context tokens in the batch.
    num_context_tokens: int

    # The number of generation sequences in the batch.
    num_generation_sequences: int

    # The number of generation tokens in the batch.
    num_generation_tokens: int

    # A buffer to store the fp8 attention outputs before converting them to bf16 and returning them.
    fp8_output_buffer: torch.Tensor


class TkeMetadataBuilder(AttentionMetadataBuilder[TkeMetadata]):

    def __init__(
        self,
        runner: GPUModelRunner,
        kv_cache_spec: AttentionSpec,
        block_table: BlockTable,
    ):
        # These things appear to be implicitly required by vLLM, i.e. if you don't set them, vLLM will try to access them and throw.
        self.runner = runner
        self.kv_cache_spec = kv_cache_spec

        model_config = self.runner.model_config

        self.num_heads_q = model_config.get_num_attention_heads(
            runner.parallel_config)
        self.num_heads_kv = model_config.get_num_kv_heads(
            runner.parallel_config)
        self.head_dimension = model_config.get_head_size()

        # Extract the dimensions of the attention layer.
        self.attention_layer_dimensions = AttentionLayerDimensions()
        self.attention_layer_dimensions.numQHeads = self.num_heads_q
        self.attention_layer_dimensions.numKVHeads = self.num_heads_kv
        self.attention_layer_dimensions.headSize = self.head_dimension

        # Extract the configuration of the KV-cache.
        prefix_cache_configuration = PrefixCacheConfiguration()

        # TODO: instead of hardcoding it, get it from configuration, check that it is FP8,
        # and throw otherwise as long as we only support FP8.
        prefix_cache_configuration.dataType = DeviceDataType.FP8_E4M3

        prefix_cache_configuration.numTokensPerBlock = kv_cache_spec.block_size
        prefix_cache_configuration.maxNumBlocksPerSequence = (
            block_table.max_num_blocks_per_req)
        prefix_cache_configuration.blockOffsetLayout = BlockOffsetLayout.VLLM
        self.block_table = block_table

        # Internal quantity used to size the kv-cache TMA descriptor.
        # TODO: calculate this value from the size of the kv-cache. It needs to be large enough that the TMA descriptor can fit the whole kv-cache tensor.
        # I couldn't find a way to access the actual size of the kv-cache at this time. The 'num_blocks' on the kv-cache config is not set at this point.
        prefix_cache_configuration.maxNumSequences = 4096

        # Store block size for debugging
        self.block_size = kv_cache_spec.block_size

        # Extract information about the rotary positional embedding so that we can calculate the rotation coefficient cache.
        rotary_positional_embedding = RotaryEmbedding()
        rope_scaling = getattr(model_config.hf_config, "rope_scaling", None)
        if rope_scaling is not None:
            scaling_factor = rope_scaling.get("factor", 1.0)
            rotary_positional_embedding.rotaryEmbeddingScale = scaling_factor

            rotary_positional_embedding.rotaryScalingType = rope_scaling_type_mapping[
                rope_scaling.get("rope_type", "none")]
        else:
            rotary_positional_embedding.rotaryEmbeddingScale = 1.0

        max_position_embeddings = getattr(model_config.hf_config,
                                          "max_position_embeddings", 8192)
        rotary_positional_embedding.rotaryEmbeddingMaxPositions = (
            max_position_embeddings)

        rope_theta = getattr(model_config.hf_config, "rope_theta", 10000)
        rotary_positional_embedding.rotaryEmbeddingBase = rope_theta

        rotary_positional_embedding.rotaryEmbeddingDim = (
            runner.vllm_config.model_config.get_head_size())
        rotary_positional_embedding.type = (
            RotaryPositionalEmbeddingType.GPT_NEOX)

        max_attention_window_size = (
            runner.vllm_config.model_config.max_model_len)
        cyclic_attention_window_size = (
            runner.vllm_config.model_config.max_model_len)

        # An internal buffer used by the XQA kernel. Needs to be initialized to 0.
        self.multi_block_semaphores = torch.zeros(
            runner.num_query_heads * runner.scheduler_config.max_num_seqs,
            device=torch.device("cuda"),
            dtype=torch.int32,
            requires_grad=False,
        ).contiguous()

        # TODO: needs to be set correctly on each batch. Move to forward. Seems to be always 1.0 though.
        self.output_scaling_factor = torch.tensor(
            [1.0],
            dtype=torch.float32,
            device=torch.device("cuda"),
            requires_grad=False,
        )

        # NOTE: According to modelopt team, 1.0 is almost always the optimal value.
        # TODO: There should also be the equivalent dequantization factor. Add support for that.
        self.kv_cache_dequantization_factor = torch.tensor(
            [1.0],
            dtype=torch.float32,
            device=torch.device("cuda"),
            requires_grad=False,
        )

        # Create a representation of the fixed parameters of the attention operation.
        self.op = create_op(
            inputDataType=DeviceDataType.BF16,
            outputDataType=DeviceDataType.FP8_E4M3,
            attentionLayerDimensions=self.attention_layer_dimensions,
            rotaryEmbedding=rotary_positional_embedding,
            prefixCacheConfiguration=prefix_cache_configuration,
            qScaling=
            1.0,  # TODO: seems to be 1.0 most of the time, still, set correctly ultimately.
            maxAttentionWindowSize=max_attention_window_size,
            cyclicAttentionWindowSize=cyclic_attention_window_size,
            outputScalingFactor=self.output_scaling_factor,
            kvCacheDequantizationFactor=self.kv_cache_dequantization_factor,
            multiBlockSemaphores=self.multi_block_semaphores,
            enableSpeculativeDecoding=False,
            enablePDL=True,  # TODO: remove and default to true internally.
        )

        # The size in bytes of the workspace needed by FMHA and XQA.
        workspace_size = calculate_workspace_size(
            self.op,
            self.runner.max_num_tokens,
            self.runner.scheduler_config.max_num_seqs,
        )

        # Allocate the workspace.
        self.workspace = torch.zeros(
            workspace_size,
            device=torch.device("cuda"),
            dtype=torch.int8,
            requires_grad=False,
        ).contiguous()

        _, self.rotary_cos_sin_ndarray = (
            create_sinusoidal_positions_for_attention_plugin(
                rotary_positional_embedding.rotaryEmbeddingMaxPositions,
                rotary_positional_embedding.rotaryEmbeddingDim,
                rotary_positional_embedding.rotaryEmbeddingBase,
                rotary_positional_embedding.rotaryEmbeddingScale,
                rotary_positional_embedding.rotaryScalingType,
                rope_scaling_config=rope_scaling,
            ))
        self.rotary_cos_sin = torch.tensor(
            self.rotary_cos_sin_ndarray,
            dtype=torch.float32,
            device="cuda",
            requires_grad=False,
        ).contiguous()

        # The kernel produces FP8 outputs, but the backend is expected to return BF16 data.
        self.fp8_output_buffer = torch.zeros(
            self.runner.max_num_tokens,
            self.runner.num_query_heads *
            self.runner.vllm_config.model_config.get_head_size(),
            device=torch.device("cuda"),
            dtype=torch.float8_e4m3fn,
            requires_grad=False,
        ).contiguous()

        # Buffer to store the sequence lengths on the host. Required by the ops.
        self.sequence_lengths_host = torch.zeros(
            runner.scheduler_config.max_num_seqs,
            device=torch.device("cpu"),
            dtype=torch.int32,
            requires_grad=False,
        ).contiguous()

    def use_cascade_attention(self, *args, **kwargs) -> bool:
        return False  # TODO: implement this

    # TODO: this could use a unit test. A lot depends on this being correct.
    def reorder_batch(self, input_batch: InputBatch,
                      scheduler_output: SchedulerOutput) -> bool:
        """
        Reorders the sequences in the batch so that the "decode" sequences are at the back of the batch.
        We identify "decode" sequences as those with a single scheduled token, but it shouldn't matter: we can in theory also use our generation kernels for 1-token long context requests.
        """

        decode_indexes: List[int] = []
        prefill_indexes: List[int] = []
        num_prefill_tokens: int = 0
        num_decode_tokens: int = 0

        for index_in_batch, request_id in enumerate(input_batch.req_ids):
            num_tokens = scheduler_output.num_scheduled_tokens[request_id]
            if num_tokens == 1:
                decode_indexes.append(index_in_batch)
                num_decode_tokens += num_tokens
            else:
                prefill_indexes.append(index_in_batch)
                num_prefill_tokens += num_tokens

        num_decodes = len(decode_indexes)
        num_prefills = len(prefill_indexes)
        modified_batch = False

        for i in range(1, min(num_decodes, num_prefills) + 1):
            prefill_idx = prefill_indexes[num_prefills - i]
            decode_idx = decode_indexes[i - 1]
            # If prefill sequence is already positioned before decode sequence, we're done
            if prefill_idx <= decode_idx:
                break
            input_batch.swap_states(prefill_idx, decode_idx)
            modified_batch = True

        # This is an 'arbitrary' split from vLLM's perspective, so we need to calculate these and store them ourselves.
        self._num_context_sequences = num_prefills
        self._num_context_tokens = num_prefill_tokens
        self._num_generation_sequences = num_decodes
        self._num_generation_tokens = num_decode_tokens

        return modified_batch

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
    ):
        return TkeMetadata(
            common_attn_metadata=common_attn_metadata,
            op=self.op,
            attention_layer_dimensions=self.attention_layer_dimensions,
            sequence_lengths_host=self.runner.seq_lens_cpu,
            workspace=self.workspace,
            rotary_cos_sin_cache=self.rotary_cos_sin,
            block_table=self.block_table,
            num_context_sequences=self._num_context_sequences,
            num_context_tokens=self._num_context_tokens,
            num_generation_sequences=self._num_generation_sequences,
            num_generation_tokens=self._num_generation_tokens,
            fp8_output_buffer=self.fp8_output_buffer,
        )


class TkeImpl(AttentionImpl):

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
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: Optional[int] = None,
        use_irope: bool = False,
    ) -> None:
        self.scale = scale
        self.num_heads = num_heads
        self.head_size = head_size
        self.num_kv_heads = num_kv_heads
        self.scale = float(scale)
        self.alibi_slopes = None
        if sliding_window is None:
            self.sliding_window = (-1, -1)
        else:
            self.sliding_window = (sliding_window - 1, 0)
        self.kv_cache_dtype = kv_cache_dtype
        self.logits_soft_cap = logits_soft_cap
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "Encoder self-attention is not implemented for TKE.")

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TkeMetadata,
        output: Optional[torch.Tensor] = None,
        output_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with TensorRT-LLM Kernel Export.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: should be None for this backend, unused, qkv is passed as a single tensor as query.
            value: should be None for this backend, unused, qkv is passed as a single tensor as query.
            kv_cache = [num_blocks, 2, num_kv_heads, block_size, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        assert output is not None, "Output tensor must be provided."

        if attn_metadata is None:
            return output

        input_sequence_lengths_device = attn_metadata.common_attn_metadata.query_start_loc.diff(
        )
        cuda_stream = current_stream()
        if attn_metadata.num_context_sequences > 0:
            max_sequence_length = attn_metadata.sequence_lengths_host[:
                                                                      attn_metadata
                                                                      .
                                                                      num_context_sequences].max(
                                                                      ).item()
            run_context_inplace(
                op=attn_metadata.op,
                numContextSequences=attn_metadata.num_context_sequences,
                numContextTokens=attn_metadata.num_context_tokens,
                maxSequenceLength=int(max_sequence_length),
                qkv=query,
                sequenceLengthsDevice=attn_metadata.common_attn_metadata.
                seq_lens,
                inputSequenceLengthsDevice=input_sequence_lengths_device,
                kvCacheBlockOffsets=attn_metadata.block_table.block_table,
                kvCachePoolPtr=kv_cache.view(torch.int8),
                rotaryCosSin=attn_metadata.rotary_cos_sin_cache,
                output=attn_metadata.fp8_output_buffer.view(torch.int8),
                workspace=attn_metadata.workspace,
                stream=cuda_stream.cuda_stream,
            )
        if attn_metadata.num_generation_sequences > 0:
            max_sequence_length = attn_metadata.sequence_lengths_host[
                attn_metadata.num_context_sequences:].max().item()
            run_generation_inplace(
                op=attn_metadata.op,
                numGenerationSequences=attn_metadata.num_generation_sequences,
                numGenerationTokens=attn_metadata.num_generation_tokens,
                maxSequenceLength=int(max_sequence_length),
                qkv=query[attn_metadata.num_context_tokens:],
                sequenceLengthsDevice=attn_metadata.common_attn_metadata.
                seq_lens[attn_metadata.num_context_sequences:],
                inputSequenceLengthsDevice=input_sequence_lengths_device[
                    attn_metadata.num_context_sequences:],
                kvCacheBlockOffsets=attn_metadata.block_table.
                block_table[attn_metadata.num_context_sequences:],
                kvCachePoolPtr=kv_cache.view(torch.int8),
                rotaryCosSin=attn_metadata.rotary_cos_sin_cache,
                output=attn_metadata.fp8_output_buffer.view(torch.int8),
                workspace=attn_metadata.workspace,
                stream=cuda_stream.cuda_stream,
            )

        num_tokens = attn_metadata.common_attn_metadata.num_actual_tokens
        output[:num_tokens].copy_(attn_metadata.fp8_output_buffer[:num_tokens])
        return output


def apply_llama3_scaling(inv_freqs: np.ndarray, rope_scaling_config: dict):
    scale_factor = rope_scaling_config.get("factor", 8.0)
    low_freq_factor = rope_scaling_config.get("low_freq_factor", 1.0)
    high_freq_factor = rope_scaling_config.get("high_freq_factor", 4.0)
    old_context_len = rope_scaling_config.get(
        "original_max_position_embeddings", 8192)

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    new_inv_freqs = []
    for inv_freq in inv_freqs:
        wavelen = 2 * math.pi / inv_freq
        if wavelen < high_freq_wavelen:
            new_inv_freqs.append(inv_freq)
        elif wavelen > low_freq_wavelen:
            new_inv_freqs.append(inv_freq / scale_factor)
        else:
            assert low_freq_wavelen != high_freq_wavelen
            smooth = (old_context_len / wavelen -
                      low_freq_factor) / (high_freq_factor - low_freq_factor)
            new_inv_freqs.append((1 - smooth) * inv_freq / scale_factor +
                                 smooth * inv_freq)
    return np.array(new_inv_freqs, dtype=inv_freqs.dtype)


def create_sinusoidal_positions_for_attention_plugin(
    num_pos: int,
    dim: int,
    theta: float,
    scale: float,
    scale_type: RotaryScalingType,
    # Other scaling configs that only used by certain scaling types.
    rope_scaling_config: Optional[dict] = None,
    dtype=np.float32,
) -> tuple[np.ndarray, np.ndarray]:
    if scale_type == RotaryScalingType.LINEAR:
        scale = 1.0 / scale
    if scale_type == RotaryScalingType.LLAMA3:
        assert rope_scaling_config is not None, (
            "rotary_scaling config must be provided.")
        inv_freq = 1.0 / (theta**(np.arange(0, dim, 2) / dim)).astype(dtype)
        inv_freq = apply_llama3_scaling(inv_freq, rope_scaling_config)
    else:
        inv_freq = scale / (theta**(np.arange(0, dim, 2) / dim)).astype(dtype)
    sinusoid_inp = np.expand_dims(
        np.einsum(
            "i , j -> i j",
            np.arange(num_pos, dtype=dtype),
            inv_freq,
            dtype=dtype,
        ),
        axis=-1,
    )
    # fuse cos/sin into float2 (cos, sin).
    concat = np.concatenate(
        (np.cos(sinusoid_inp), np.sin(sinusoid_inp)),
        axis=-1)  # np.cos(sinusoid_inp).shape = (32768, 64, 1)

    return inv_freq, concat.astype(dtype).reshape(num_pos, dim)

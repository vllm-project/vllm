# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU-resident Qwen4Exp position-learning enhancement layers."""

import math
from collections.abc import Iterable, Sequence

import torch
import torch.nn.functional as F
from torch import nn

import vllm.envs as envs
from vllm.config import CacheConfig, ModelConfig, VllmConfig, get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
    is_conv_state_dim_first,
)
from vllm.model_executor.layers.ple_offload_layer import (
    PleOffloadLayer,
    is_offload_process,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4Config
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    create_fp8_scale_parameter,
    create_fp8_weight_parameter,
    is_fp8,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped,
)
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.model_executor.parameter import PerTensorScaleParameter
from vllm.model_executor.utils import set_weight_attrs
from vllm.transformers_utils.configs.qwen4_exp import (
    Qwen4ExpTextConfig,
)
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.attention.backends.short_conv_attn import (
    PleShortConvAttentionBackend,
    PleShortConvAttentionMetadata,
)
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

from ..common.ple import PLEVocabParallelEmbedding, copy_ple_embedding_shard_

logger = init_logger(__name__)


class Qwen4ExpPLEGroupedNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float,
        group_size: int | None,
        dtype: torch.dtype | None,
    ) -> None:
        super().__init__()
        if group_size is not None and hidden_size % group_size:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by "
                f"group_size ({group_size})"
            )
        self.eps = eps
        self.group_size = group_size
        self.weight = nn.Parameter(torch.zeros(hidden_size, dtype=dtype))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        if self.group_size is None:
            variance = hidden_states.square().mean(dim=-1, keepdim=True)
            normalized = hidden_states * torch.rsqrt(variance + self.eps)
        else:
            grouped = hidden_states.unflatten(
                -1, (hidden_states.shape[-1] // self.group_size, self.group_size)
            )
            variance = grouped.square().mean(dim=-1, keepdim=True)
            normalized = (grouped * torch.rsqrt(variance + self.eps)).flatten(-2)
        return (normalized * (1.0 + self.weight.float())).to(input_dtype)


class Qwen4ExpPLEFp8EmbeddingMethod(QuantizeMethodBase):
    """FP8 PLE embedding with one global checkpoint scale."""

    def create_weights(
        self,
        layer: nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del input_size, output_size, params_dtype
        weight_loader = extra_weight_attrs.get("weight_loader")
        weight = create_fp8_weight_parameter(
            sum(output_partition_sizes), input_size_per_partition, weight_loader
        )
        layer.register_parameter("weight", weight)

        weight_scale = create_fp8_scale_parameter(
            PerTensorScaleParameter,
            output_partition_sizes,
            input_size_per_partition,
            None,
            weight_loader,
            scale_dtype=torch.bfloat16,
        )
        layer.register_parameter("weight_scale", weight_scale)

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError("PLE FP8 weights only support embedding lookup")

    def embedding(self, layer: nn.Module, input_: torch.Tensor) -> torch.Tensor:
        return F.embedding(input_, layer.weight)


_NVFP4_BLOCK_SIZE = 16


class Qwen4ExpPLENVFp4EmbeddingMethod(QuantizeMethodBase):
    """NVFP4 PLE embedding kept packed at runtime.

    The table stays as uint8 codes with FP8 block scales (16 values per
    block) and one FP32 global scale. A lookup only gathers the packed rows;
    the PLE layer dequantizes them after lookup.
    """

    def create_weights(
        self,
        layer: nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del input_size, output_size, extra_weight_attrs
        if input_size_per_partition % _NVFP4_BLOCK_SIZE:
            raise ValueError(
                "NVFP4 PLE embedding requires the embedding dim to be a "
                f"multiple of {_NVFP4_BLOCK_SIZE}, got {input_size_per_partition}"
            )
        self.params_dtype = params_dtype
        self.head_dim = input_size_per_partition
        self.packed_row_width = (
            input_size_per_partition // 2
            + input_size_per_partition // _NVFP4_BLOCK_SIZE
        )
        rows = sum(output_partition_sizes)
        weight = nn.Parameter(
            torch.empty(rows, input_size_per_partition // 2, dtype=torch.uint8),
            requires_grad=False,
        )
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight", weight)

        weight_scale = nn.Parameter(
            torch.empty(
                rows,
                input_size_per_partition // _NVFP4_BLOCK_SIZE,
                dtype=torch.float8_e4m3fn,
            ),
            requires_grad=False,
        )
        set_weight_attrs(weight_scale, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight_scale", weight_scale)

        weight_scale_2 = nn.Parameter(
            torch.empty((), dtype=torch.float32), requires_grad=False
        )
        layer.register_parameter("weight_scale_2", weight_scale_2)
        self._lut: torch.Tensor | None = None

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        # Build the lookup table before CUDA graph capture.
        self._lut = torch.tensor(
            _FP4_VALUES, dtype=torch.float32, device=layer.weight.device
        )

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError("PLE NVFP4 weights only support embedding lookup")

    def embedding(self, layer: nn.Module, input_: torch.Tensor) -> torch.Tensor:
        """Gather packed rows for deferred dequantization."""
        codes = F.embedding(input_, layer.weight)
        scales = F.embedding(input_, layer.weight_scale)
        return torch.cat((codes, scales.view(torch.uint8)), dim=-1)

    def dequantize(
        self,
        packed_rows: torch.Tensor,
        scale_2: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Dequantize rows containing NVFP4 codes and block scales."""
        lut = self._lut
        if lut is None or lut.device != packed_rows.device:
            lut = torch.tensor(
                _FP4_VALUES, dtype=torch.float32, device=packed_rows.device
            )
            self._lut = lut
        return _dequant_nvfp4_rows(
            packed_rows, self.head_dim, scale_2, output_dtype, lut
        )


def _ple_dtype_is_fp8(ple_embedding_dtype: object) -> bool:
    """Return whether the model config declares FP8 PLE checkpoint weights."""

    if ple_embedding_dtype is None:
        return False
    if isinstance(ple_embedding_dtype, torch.dtype):
        return ple_embedding_dtype == torch.float8_e4m3fn
    return str(ple_embedding_dtype).rsplit(".", 1)[-1] == "float8_e4m3fn"


def _get_ple_embedding_quant_method(
    quant_config: QuantizationConfig | None,
    prefix: str,
    ple_embedding_dtype: object = None,
) -> QuantizeMethodBase | None:
    """Select a packed PLE embedding method for quantized checkpoint shards."""

    if isinstance(quant_config, Fp8Config):
        if not quant_config.is_checkpoint_fp8_serialized:
            return None
        ignored_layers = quant_config.ignored_layers
        if is_layer_skipped(
            prefix,
            ignored_layers,
            quant_config.packed_modules_mapping,
            match_mode=quant_config.ignored_layers_match_mode,
        ):
            return None
        # PLE checkpoint shards form one runtime embedding parameter.
        shard_prefix = f"{prefix}.shard_"
        if any(name.startswith(shard_prefix) for name in ignored_layers):
            return None
        return Qwen4ExpPLEFp8EmbeddingMethod()

    if isinstance(quant_config, ModelOptNvFp4Config):
        if not quant_config.is_checkpoint_nvfp4_serialized:
            return None
        if not quant_config.is_layer_excluded(prefix):
            logger.info_once("PLE embedding %s uses the runtime NVFP4 method", prefix)
            return Qwen4ExpPLENVFp4EmbeddingMethod()
        if _ple_dtype_is_fp8(ple_embedding_dtype):
            logger.info_once(
                "Excluded ModelOpt PLE embedding %s uses the runtime FP8 method",
                prefix,
            )
            return Qwen4ExpPLEFp8EmbeddingMethod()

    return None


_FP4_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _dequant_nvfp4_codes(
    packed: torch.Tensor,
    scale: torch.Tensor,
    scale_2: torch.Tensor,
    lut: torch.Tensor | None = None,
) -> torch.Tensor:
    """Unpack NVFP4 codes with FP8 block scales and an FP32 global scale."""
    half = packed.shape[-1]
    codes = torch.stack([packed & 0xF, (packed >> 4) & 0xF], dim=-1).reshape(
        *packed.shape[:-1], half * 2
    )
    if lut is None:
        lut = torch.tensor(_FP4_VALUES, dtype=torch.float32, device=packed.device)
    magnitude = lut[(codes & 0x7).long()]
    sign = ((codes >> 3) & 1).to(torch.float32)
    fp4 = (magnitude * (1 - 2 * sign)).reshape(
        *packed.shape[:-1], -1, _NVFP4_BLOCK_SIZE
    )
    output = fp4 * scale.float().unsqueeze(-1) * scale_2.float()
    return output.reshape(*packed.shape[:-1], half * 2)


def _dequant_nvfp4_rows(
    packed_rows: torch.Tensor,
    head_dim: int,
    scale_2: torch.Tensor,
    output_dtype: torch.dtype,
    lut: torch.Tensor,
) -> torch.Tensor:
    """Dequantize rows containing NVFP4 codes followed by block scales."""
    half = head_dim // 2
    codes = packed_rows[..., :half]
    scales = packed_rows[..., half:].contiguous().view(torch.float8_e4m3fn)
    return _dequant_nvfp4_codes(codes, scales, scale_2, lut).to(output_dtype)


def _get_shared_nvfp4_outer_scale(
    outer_scales: dict[int, torch.Tensor],
) -> torch.Tensor:
    """Validate and return the global scale shared by NVFP4 PLE shards."""
    first_index, reference = next(iter(outer_scales.items()))
    reference = reference.reshape(())
    for shard_index, outer_scale in outer_scales.items():
        if not torch.equal(outer_scale.reshape(()), reference):
            raise ValueError(
                "NVFP4 PLE shards must share the same global scale, but "
                f"shards {first_index} and {shard_index} differ"
            )
    return reference


class Qwen4ExpNGramEmbedding(PleOffloadLayer):
    _offload_quant_method: QuantizeMethodBase | None = None
    _MASK64 = (1 << 64) - 1
    _SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
    _SPLITMIX_M1 = 0xBF58476D1CE4E5B9
    _SPLITMIX_M2 = 0x94D049BB133111EB
    _PLE_LAYER_PRIME = 10007

    @classmethod
    def _splitmix64(cls, value: int) -> int:
        """Mix an integer into a deterministic unsigned 64-bit value."""
        value = (value + cls._SPLITMIX_GAMMA) & cls._MASK64
        value = ((value ^ (value >> 30)) * cls._SPLITMIX_M1) & cls._MASK64
        value = ((value ^ (value >> 27)) * cls._SPLITMIX_M2) & cls._MASK64
        return (value ^ (value >> 31)) & cls._MASK64

    @staticmethod
    def _is_prime_64(value: int) -> bool:
        """Return whether a 64-bit integer is prime."""
        if value < 2:
            return False
        for prime in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
            if value % prime == 0:
                return value == prime
        exponent = value - 1
        shifts = 0
        while exponent % 2 == 0:
            exponent //= 2
            shifts += 1
        for base in (2, 325, 9375, 28178, 450775, 9780504, 1795265022):
            if base % value == 0:
                continue
            witness = pow(base, exponent, value)
            if witness in (1, value - 1):
                continue
            for _ in range(shifts - 1):
                witness = pow(witness, 2, value)
                if witness == value - 1:
                    break
            else:
                return False
        return True

    @classmethod
    def _nth_prime_after(cls, start: int, count: int) -> int:
        """Return the ``count``-th prime strictly greater than ``start``."""
        prime = int(start)
        for _ in range(count):
            candidate = prime + 1
            if candidate <= 2:
                prime = 2
                continue
            if candidate % 2 == 0:
                candidate += 1
            while not cls._is_prime_64(candidate):
                candidate += 2
            prime = candidate
        return prime

    @classmethod
    def _make_layer_multipliers(
        cls,
        *,
        ngram_size: int,
        unigram_vocab_size: int,
        seed: int,
        ple_dense_layer_id: int,
    ) -> list[int]:
        """Build deterministic hash multipliers for one PLE layer."""
        max_multiplier = ((1 << 63) - 1) // unigram_vocab_size
        half_bound = max(1, max_multiplier // 2)
        base_seed = seed + cls._PLE_LAYER_PRIME * ple_dense_layer_id
        multipliers = []
        for index in range(ngram_size):
            value = base_seed + cls._SPLITMIX_GAMMA * (index + 1)
            multipliers.append(2 * (cls._splitmix64(value) % half_bound) + 1)
        return multipliers

    @classmethod
    def _make_vocab_layout(
        cls,
        *,
        ngram_vocab_size_base: int,
        ngram_heads: int,
        ple_dense_layer_id: int,
    ) -> tuple[list[int], list[int], int]:
        """Build per-head vocabulary sizes, offsets, and total row count."""
        sizes: list[int] = []
        offsets: list[int] = []
        offset = 0
        for local_head in range(ngram_heads):
            global_head = ple_dense_layer_id * ngram_heads + local_head
            size = cls._nth_prime_after(ngram_vocab_size_base - 1, global_head + 1)
            sizes.append(size)
            offsets.append(offset)
            offset += size
        return sizes, offsets, offset

    def __init__(
        self,
        config: Qwen4ExpTextConfig,
        embedding_dim: int,
        ple_dense_layer_id: int,
        max_total_tokens: int,
        max_num_reqs: int,
        prefix: str,
        layer_name: str,
        quant_config: QuantizationConfig | None = None,
        params_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.layer_name = layer_name
        self.embedding_dim = embedding_dim
        self.ngram_size = int(config.ngram_size)
        self.heads_per_ngram = int(config.heads_per_ngram)
        self.ngram_heads = (self.ngram_size - 1) * self.heads_per_ngram
        if self.ngram_size < 2:
            raise ValueError(f"ngram_size must be >= 2, got {self.ngram_size}")
        if self.heads_per_ngram <= 0:
            raise ValueError(f"heads_per_ngram must be > 0, got {self.heads_per_ngram}")
        if embedding_dim % self.ngram_heads:
            raise ValueError(
                "ple_embed_dim must be divisible by total ngram heads: "
                f"{embedding_dim} % {self.ngram_heads} != 0"
            )
        self.head_dim = embedding_dim // self.ngram_heads
        self.eos_token_id = int(config.eos_token_id)
        self.unigram_vocab_size = int(config.vocab_size)
        self.split_ngram_parts = int(getattr(config, "split_ngram_parts", 512))
        if self.split_ngram_parts <= 0:
            raise ValueError("split_ngram_parts must be positive")

        multipliers = self._make_layer_multipliers(
            ngram_size=self.ngram_size,
            unigram_vocab_size=self.unigram_vocab_size,
            seed=int(getattr(config, "seed", 1234)),
            ple_dense_layer_id=ple_dense_layer_id,
        )
        self.register_buffer(
            "layer_multipliers",
            torch.tensor(multipliers, dtype=torch.long),
            persistent=True,
        )

        sizes, offsets, total_vocab_size = self._make_vocab_layout(
            ngram_vocab_size_base=int(config.ngram_vocab_size_base),
            ngram_heads=self.ngram_heads,
            ple_dense_layer_id=ple_dense_layer_id,
        )
        self.register_buffer(
            "ngram_heads_vocab_sizes",
            torch.tensor(sizes, dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "ngram_heads_offsets",
            torch.tensor(offsets, dtype=torch.long),
            persistent=True,
        )
        divisor = int(config.make_ngram_vocab_size_divisible_by)
        padded_vocab_size = ((total_vocab_size + divisor - 1) // divisor) * divisor
        self.ngram_embedding = PLEVocabParallelEmbedding(
            padded_vocab_size,
            self.head_dim,
            params_dtype=params_dtype,
            padding_size=divisor,
            prefix=f"{prefix}.ngram_embedding",
            quant_method=_get_ple_embedding_quant_method(
                quant_config,
                f"{prefix}.ngram_embedding",
                getattr(config, "ple_embedding_dtype", None),
            ),
        )
        self.register_buffer(
            "positions_buffer",
            torch.arange(max_total_tokens, dtype=torch.int64),
            persistent=False,
        )
        self.register_buffer(
            "padded_buffer",
            torch.full(
                (max_num_reqs, max_total_tokens),
                self.eos_token_id,
                dtype=torch.int64,
            ),
            persistent=False,
        )

    @staticmethod
    def _shift_precompute(
        tokens: torch.Tensor, eos_token_id: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if tokens.dim() != 2:
            raise ValueError("tokens must be a 2D tensor")
        batch_size, seq_len = tokens.shape
        positions = torch.arange(seq_len, device=tokens.device, dtype=torch.int64)
        eos_positions = torch.where(tokens == eos_token_id, positions, -1)
        previous_eos_inclusive = torch.cummax(eos_positions, dim=1).values
        previous_eos = torch.cat(
            [
                eos_positions.new_full((batch_size, 1), -1),
                previous_eos_inclusive[:, :-1],
            ],
            dim=1,
        )
        return positions, positions.unsqueeze(0) - previous_eos - 1

    @staticmethod
    def _shift_apply(
        tokens: torch.Tensor,
        positions: torch.Tensor,
        position_in_segment: torch.Tensor,
        shift: int,
        eos_token_id: int,
    ) -> torch.Tensor:
        if shift == 0:
            return tokens
        source = positions - shift
        gather_indices = source.clamp_min(0).unsqueeze(0).expand(tokens.shape[0], -1)
        shifted = tokens.gather(1, gather_indices)
        valid = (source.unsqueeze(0) >= 0) & (position_in_segment >= shift)
        return torch.where(valid, shifted, tokens.new_full((), eos_token_id))

    def compute_ngram_ids(
        self,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
    ) -> torch.Tensor:
        """Compute n-gram embedding indices for the current request layout."""
        input_ids = input_ids.reshape(-1).long()
        query_start_loc = query_start_loc.long()
        num_reqs = query_start_loc.numel() - 1
        num_tokens = input_ids.shape[0]
        if num_tokens > self.positions_buffer.numel():
            raise ValueError(
                f"PLE received {num_tokens} tokens, but its workspace supports "
                f"at most {self.positions_buffer.numel()}"
            )
        if num_reqs > self.padded_buffer.shape[0]:
            raise ValueError(
                f"PLE received {num_reqs} requests, but its workspace supports "
                f"at most {self.padded_buffer.shape[0]}"
            )

        # The CPU-offload subprocess is never captured by a CUDA Graph, so its
        # pack workspace can narrow to the actual maximum sequence length. The
        # regular GPU path retains the static maximum-width buffer for capture.
        if is_offload_process():
            if num_reqs <= 0:
                raise ValueError("PLE CPU offload requires at least one request")
            max_seq_len = max(
                1,
                int((query_start_loc[1:] - query_start_loc[:-1]).max().item()),
            )
            # The model runner sends the CUDA-graph padded token count together
            # with an unpadded query_start_loc. Stale padding must not enter the
            # scatter: its clamped indices would overwrite the last real token.
            num_valid_tokens = min(int(query_start_loc[-1].item()), num_tokens)
        else:
            max_seq_len = self.padded_buffer.shape[1]
            num_valid_tokens = num_tokens

        positions = self.positions_buffer[:num_tokens]
        packed = self.padded_buffer[:num_reqs, :max_seq_len]
        packed.fill_(self.eos_token_id)
        request_indices = torch.searchsorted(query_start_loc, positions, right=True) - 1
        request_indices.clamp_(max=num_reqs - 1)
        columns = (positions - query_start_loc[request_indices]).clamp(
            0, packed.shape[1] - 1
        )
        packed[request_indices[:num_valid_tokens], columns[:num_valid_tokens]] = (
            input_ids[:num_valid_tokens]
        )
        ngram_context = ngram_context[:num_reqs].to(
            device=input_ids.device, dtype=torch.long
        )

        context = torch.cat([ngram_context, packed], dim=-1)
        positions_2d, position_in_segment = self._shift_precompute(
            context, self.eos_token_id
        )
        shifted = [context]
        for shift in range(1, self.ngram_size):
            shifted.append(
                self._shift_apply(
                    context,
                    positions_2d,
                    position_in_segment,
                    shift,
                    self.eos_token_id,
                )
            )
        adjusted_columns = columns + self.ngram_size - 1
        id_blocks = []
        for ngram in range(2, self.ngram_size + 1):
            start = (ngram - 2) * self.heads_per_ngram
            end = start + self.heads_per_ngram
            mixed = shifted[0] * self.layer_multipliers[0]
            for index in range(1, ngram):
                mixed = torch.bitwise_xor(
                    mixed, shifted[index] * self.layer_multipliers[index]
                )
            sizes = self.ngram_heads_vocab_sizes[start:end]
            offsets = self.ngram_heads_offsets[start:end]
            ids = torch.remainder(mixed.unsqueeze(-1), sizes) + offsets
            id_blocks.append(ids[request_indices, adjusted_columns])
        return torch.cat(id_blocks, dim=-1)

    def forward_impl(  # type: ignore[override]
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
        output_buffer: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del hidden_states
        ngram_ids = self.compute_ngram_ids(
            input_ids,
            query_start_loc,
            ngram_context,
        )
        num_tokens = input_ids.reshape(-1).shape[0]
        if output_buffer is not None:
            quant_method = getattr(self.ngram_embedding, "quant_method", None)
            if isinstance(quant_method, Qwen4ExpPLENVFp4EmbeddingMethod):
                output_dim = self.get_offload_output_dim(self.embedding_dim)
                output = output_buffer[:num_tokens, :output_dim]
                output.copy_(self.ngram_embedding(ngram_ids).flatten(-2))
                return output
            output = output_buffer[:num_tokens, : self.embedding_dim]
            torch.index_select(
                self.ngram_embedding.weight,
                0,
                ngram_ids.reshape(-1),
                out=output.reshape(-1, self.head_dim),
            )
            return output
        return self.ngram_embedding(ngram_ids).flatten(-2)

    def get_offload_output_dtype(self, default_dtype: torch.dtype) -> torch.dtype:
        """Keep quantized lookup results in their embedding storage dtype."""
        embedding = getattr(self, "ngram_embedding", None)
        weight = getattr(embedding, "weight", None)
        if weight is not None:
            return weight.dtype
        if isinstance(self._offload_quant_method, Qwen4ExpPLENVFp4EmbeddingMethod):
            return torch.uint8
        if isinstance(self._offload_quant_method, Qwen4ExpPLEFp8EmbeddingMethod):
            return torch.float8_e4m3fn
        return default_dtype

    def get_offload_output_dim(self, default_dim: int) -> int:
        """Keep NVFP4 lookup rows packed while transferring them to the GPU."""
        quant_method = getattr(
            getattr(self, "ngram_embedding", None), "quant_method", None
        )
        if quant_method is None:
            quant_method = self._offload_quant_method
        if isinstance(quant_method, Qwen4ExpPLENVFp4EmbeddingMethod):
            if default_dim % _NVFP4_BLOCK_SIZE:
                raise ValueError(
                    "NVFP4 PLE output dim must be a multiple of "
                    f"{_NVFP4_BLOCK_SIZE}, got {default_dim}"
                )
            return default_dim // 2 + default_dim // _NVFP4_BLOCK_SIZE
        return default_dim

    def initialize_dummy_offload_metadata(self, device: torch.device) -> None:
        """Initialize quantization metadata skipped by the dummy loader."""
        quant_method = self._offload_quant_method
        if isinstance(quant_method, Qwen4ExpPLEFp8EmbeddingMethod):
            self.register_buffer(
                "_offload_weight_scale",
                torch.ones((), dtype=torch.float32, device=device),
                persistent=False,
            )
        elif isinstance(quant_method, Qwen4ExpPLENVFp4EmbeddingMethod):
            self.register_buffer(
                "_offload_weight_scale_2",
                torch.ones((), dtype=torch.float32, device=device),
                persistent=False,
            )
            self.register_buffer(
                "_offload_nvfp4_lut",
                torch.tensor(_FP4_VALUES, dtype=torch.float32, device=device),
                persistent=False,
            )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load hash buffers and checkpoint-split embedding rows."""

        # Prefix grouping may invoke this method repeatedly for one module.
        # Treat the global FP8 scale as incrementally loaded state.
        # GPU workers retain only dequantization metadata. The CPU process owns
        # the embedding table and transfers quantized lookup rows unchanged.
        if envs.VLLM_PLE_CPU_OFFLOAD and not is_offload_process():
            retained: set[str] = set()
            quant_method = self._offload_quant_method
            if isinstance(quant_method, Qwen4ExpPLEFp8EmbeddingMethod):
                for name, loaded_weight in weights:
                    if name != "ngram_embedding.weight_scale":
                        continue
                    self.register_buffer(
                        "_offload_weight_scale",
                        loaded_weight.to(
                            device=torch.accelerator.current_accelerator()
                        ),
                        persistent=False,
                    )
                    retained.add(name)
            elif isinstance(quant_method, Qwen4ExpPLENVFp4EmbeddingMethod):
                outer_scales: dict[int, torch.Tensor] = {}
                for name, loaded_weight in weights:
                    if not name.startswith(
                        "ngram_embedding.shard_"
                    ) or not name.endswith(".weight_scale_2"):
                        continue
                    shard_text = name[
                        len("ngram_embedding.shard_") : -len(".weight_scale_2")
                    ]
                    shard_index = int(shard_text)
                    outer_scales[shard_index] = loaded_weight
                    retained.add(name)
                if not retained:
                    raise ValueError(
                        "NVFP4 PLE offload checkpoint is missing its global scale"
                    )
                scale_2 = _get_shared_nvfp4_outer_scale(outer_scales).to(
                    device=torch.accelerator.current_accelerator()
                )
                self.register_buffer(
                    "_offload_weight_scale_2", scale_2, persistent=False
                )
                self.register_buffer(
                    "_offload_nvfp4_lut",
                    torch.tensor(
                        _FP4_VALUES, dtype=torch.float32, device=scale_2.device
                    ),
                    persistent=False,
                )
            else:
                for _ in weights:
                    pass
            return retained

        persistent_buffers = {
            "layer_multipliers": self.layer_multipliers,
            "ngram_heads_offsets": self.ngram_heads_offsets,
            "ngram_heads_vocab_sizes": self.ngram_heads_vocab_sizes,
        }
        loaded: set[str] = set()
        regular_weights: list[tuple[str, torch.Tensor]] = []
        shard_prefix = "ngram_embedding.shard_"
        quant_method = getattr(self.ngram_embedding, "quant_method", None)
        nvfp4_runtime = isinstance(quant_method, Qwen4ExpPLENVFp4EmbeddingMethod)
        packed_codes: dict[int, torch.Tensor] = {}
        packed_scales: dict[int, torch.Tensor] = {}
        packed_outer_scales: dict[int, torch.Tensor] = {}

        for name, loaded_weight in weights:
            leaf_name = name.rsplit(".", 1)[-1]
            if leaf_name.startswith("hashstats_") or leaf_name == "token_lookup":
                continue
            if name in persistent_buffers:
                buffer = persistent_buffers[name]
                if buffer.shape != loaded_weight.shape:
                    raise ValueError(
                        f"Shape mismatch for {name}: expected "
                        f"{tuple(buffer.shape)}, got {tuple(loaded_weight.shape)}"
                    )
                buffer.copy_(loaded_weight.to(device=buffer.device, dtype=buffer.dtype))
                loaded.add(name)
                continue

            if nvfp4_runtime and name.startswith(shard_prefix):
                suffix = name[len(shard_prefix) :]
                for ending, sink in (
                    (".weight_scale_2", packed_outer_scales),
                    (".weight_scale", packed_scales),
                    (".weight", packed_codes),
                ):
                    if not suffix.endswith(ending):
                        continue
                    shard_text = suffix[: -len(ending)]
                    shard_index = int(shard_text)
                    if shard_index >= self.split_ngram_parts:
                        raise ValueError(
                            f"PLE embedding shard index {shard_index} exceeds "
                            f"split_ngram_parts={self.split_ngram_parts}"
                        )
                    sink[shard_index] = loaded_weight
                    break
                else:
                    regular_weights.append((name, loaded_weight))
                continue

            if (
                not nvfp4_runtime
                and name.startswith(shard_prefix)
                and name.endswith(".weight")
            ):
                shard_text = name[len(shard_prefix) : -len(".weight")]
                if not shard_text.isdigit():
                    regular_weights.append((name, loaded_weight))
                    continue
                shard_index = int(shard_text)
                if shard_index >= self.split_ngram_parts:
                    raise ValueError(
                        f"PLE embedding shard index {shard_index} exceeds "
                        f"split_ngram_parts={self.split_ngram_parts}"
                    )
                embedding = self.ngram_embedding
                shard_size = (
                    embedding.org_vocab_size + self.split_ngram_parts - 1
                ) // self.split_ngram_parts
                checkpoint_start = shard_index * shard_size
                expected_rows = max(
                    0,
                    min(shard_size, embedding.org_vocab_size - checkpoint_start),
                )
                expected_shape = (expected_rows, embedding.embedding_dim)
                if tuple(loaded_weight.shape) != expected_shape:
                    raise ValueError(
                        f"Shape mismatch for PLE embedding shard {shard_index}: "
                        f"expected {expected_shape}, got "
                        f"{tuple(loaded_weight.shape)}"
                    )
                embedding.weight.weight_loader(
                    embedding.weight,
                    loaded_weight,
                    checkpoint_start=checkpoint_start,
                )
                loaded.add("ngram_embedding.weight")
                continue
            regular_weights.append((name, loaded_weight))

        if nvfp4_runtime:
            if not packed_codes:
                raise ValueError("NVFP4 PLE checkpoint contains no packed shards")
            for shard_index in packed_codes:
                if (
                    shard_index not in packed_scales
                    or shard_index not in packed_outer_scales
                ):
                    raise ValueError(
                        f"NVFP4 PLE shard {shard_index} is missing its scale tensors"
                    )
            shared_outer_scale = _get_shared_nvfp4_outer_scale(
                {
                    shard_index: packed_outer_scales[shard_index]
                    for shard_index in packed_codes
                }
            )

        for shard_index, codes in packed_codes.items():
            embedding = self.ngram_embedding
            shard_size = (
                embedding.org_vocab_size + self.split_ngram_parts - 1
            ) // self.split_ngram_parts
            checkpoint_start = shard_index * shard_size
            expected_rows = max(
                0, min(shard_size, embedding.org_vocab_size - checkpoint_start)
            )
            expected_codes_shape = (expected_rows, embedding.embedding_dim // 2)
            if tuple(codes.shape) != expected_codes_shape:
                raise ValueError(
                    f"Shape mismatch for packed PLE shard {shard_index}: "
                    f"expected {expected_codes_shape}, got {tuple(codes.shape)}"
                )
            tp_bounds = {
                "checkpoint_start": checkpoint_start,
                "tp_start": embedding.shard_indices.org_vocab_start_index,
                "tp_end": embedding.shard_indices.org_vocab_end_index,
            }
            scales = packed_scales[shard_index]
            expected_scales_shape = (
                expected_rows,
                embedding.embedding_dim // _NVFP4_BLOCK_SIZE,
            )
            if tuple(scales.shape) != expected_scales_shape:
                raise ValueError(
                    f"Scale shape mismatch for packed PLE shard {shard_index}: "
                    f"expected {expected_scales_shape}, got {tuple(scales.shape)}"
                )
            copy_ple_embedding_shard_(embedding.weight.data, codes, **tp_bounds)
            copy_ple_embedding_shard_(embedding.weight_scale.data, scales, **tp_bounds)

        if nvfp4_runtime:
            self.ngram_embedding.weight_scale_2.data.copy_(shared_outer_scale)
            loaded.update(
                {
                    "ngram_embedding.weight",
                    "ngram_embedding.weight_scale",
                    "ngram_embedding.weight_scale_2",
                }
            )

        if regular_weights:
            loaded.update(AutoWeightsLoader(self).load_weights(regular_weights))
        return loaded


class Qwen4ExpPLELayer(nn.Module, MambaBase):
    def __init__(
        self,
        config: Qwen4ExpTextConfig,
        vllm_config: VllmConfig,
        layer_idx: int = 0,
        ple_dense_layer_id: int | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.model_config: ModelConfig = model_config
        self.cache_config: CacheConfig = cache_config
        self.layer_idx = layer_idx
        self.ple_dense_layer_id = (
            int(ple_dense_layer_id)
            if ple_dense_layer_id is not None
            else int(layer_idx)
        )
        self.prefix = prefix
        self.hidden_size = int(config.hidden_size)
        self.ple_embedding_dim = int(config.ple_embed_dim)
        self.ple_ngram_heads = (int(config.ngram_size) - 1) * int(
            config.heads_per_ngram
        )
        self.ple_head_dim = self.ple_embedding_dim // self.ple_ngram_heads
        self.hc_count = config.hc_count
        self.hc_hidden_size = self.hidden_size * self.hc_count
        self.conv_kernel_size = int(config.ple_conv_kernel_size)
        self.short_conv_dilation = int(config.ngram_size)
        self.conv_state_len = (self.conv_kernel_size - 1) * self.short_conv_dilation
        self.num_spec_tokens = vllm_config.num_speculative_tokens
        self.activation = "silu"
        # The offload process builds the surrounding model on meta while
        # this subtree must own real CPU storage. GPU workers skip the
        # subclass constructor and retain only an empty IPC placeholder.
        with torch.device(PleOffloadLayer.get_target_device()):
            ple_embedding = Qwen4ExpNGramEmbedding(
                config,
                int(config.ple_embed_dim),
                self.ple_dense_layer_id,
                vllm_config.scheduler_config.max_num_batched_tokens,
                vllm_config.scheduler_config.max_num_seqs,
                prefix=f"{prefix}.ple_embedding",
                layer_name=prefix,
                quant_config=quant_config,
                params_dtype=model_config.dtype,
            )
        if envs.VLLM_PLE_CPU_OFFLOAD and not is_offload_process():
            ple_embedding._offload_quant_method = _get_ple_embedding_quant_method(
                quant_config,
                f"{prefix}.ple_embedding.ngram_embedding",
                getattr(config, "ple_embedding_dtype", None),
            )
        self.ple_embedding: nn.Module = ple_embedding
        self.key_proj = ReplicatedLinear(
            int(config.ple_embed_dim),
            self.hc_hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.key_proj",
        )
        self.value_proj = ReplicatedLinear(
            int(config.ple_embed_dim),
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.value_proj",
        )
        norm_args = (
            self.hc_hidden_size,
            config.rms_norm_eps,
            self.hidden_size,
            model_config.dtype,
        )
        self.norm_key = Qwen4ExpPLEGroupedNorm(*norm_args)
        self.norm_query = Qwen4ExpPLEGroupedNorm(*norm_args)
        self.norm_conv = Qwen4ExpPLEGroupedNorm(*norm_args)
        self.conv1d = nn.Conv1d(
            self.hc_hidden_size,
            self.hc_hidden_size,
            self.conv_kernel_size,
            groups=self.hc_hidden_size,
            padding=self.conv_state_len,
            dilation=self.short_conv_dilation,
            bias=False,
            dtype=model_config.dtype,
        )
        nn.init.zeros_(self.conv1d.weight)
        self.conv1d.weight._no_reinit = True
        self.kv_cache = (torch.tensor([]),)
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def _get_embedding_weight_scale(self) -> torch.Tensor | None:
        embedding = getattr(self.ple_embedding, "ngram_embedding", None)
        weight_scale = getattr(embedding, "weight_scale", None)
        if weight_scale is not None:
            return weight_scale
        return getattr(self.ple_embedding, "_offload_weight_scale", None)

    def _dequantize_embeddings(
        self,
        embeddings: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Dequantize PLE lookup output."""

        quant_method = getattr(
            getattr(self.ple_embedding, "ngram_embedding", None),
            "quant_method",
            None,
        )
        offload_quant_method = getattr(
            self.ple_embedding, "_offload_quant_method", None
        )
        if isinstance(offload_quant_method, Qwen4ExpPLENVFp4EmbeddingMethod):
            offload_scale_2 = getattr(
                self.ple_embedding, "_offload_weight_scale_2", None
            )
            if offload_scale_2 is None:
                raise RuntimeError("NVFP4 PLE offload is missing its global scale")
            packed_row_width = (
                self.ple_head_dim // 2 + self.ple_head_dim // _NVFP4_BLOCK_SIZE
            )
            packed_rows = embeddings.unflatten(
                -1, (self.ple_ngram_heads, packed_row_width)
            )
            dequantized = _dequant_nvfp4_rows(
                packed_rows,
                self.ple_head_dim,
                offload_scale_2,
                output_dtype,
                self.ple_embedding._offload_nvfp4_lut,
            )
            return dequantized.flatten(-2)
        if isinstance(quant_method, Qwen4ExpPLENVFp4EmbeddingMethod):
            packed_rows = embeddings.unflatten(
                -1,
                (self.ple_embedding.ngram_heads, quant_method.packed_row_width),
            )
            dequantized = quant_method.dequantize(
                packed_rows,
                self.ple_embedding.ngram_embedding.weight_scale_2,
                output_dtype,
            )
            return dequantized.flatten(-2)
        if not is_fp8(embeddings):
            return embeddings
        weight_scale = self._get_embedding_weight_scale()
        if weight_scale is None:
            raise RuntimeError("FP8 PLE embedding is missing its global scale")
        if weight_scale.device != embeddings.device:
            raise RuntimeError("FP8 PLE embedding scale must be on the output device")
        return embeddings.to(output_dtype) * weight_scale.to(output_dtype)

    @property
    def mamba_type(self) -> MambaAttentionBackendEnum:
        return MambaAttentionBackendEnum.SHORT_CONV

    @property
    def is_kv_cache_tp_replicated(self) -> bool:
        return True

    def get_attn_backend(self) -> type[PleShortConvAttentionBackend]:
        return PleShortConvAttentionBackend

    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        return MambaStateDtypeCalculator.short_conv_state_dtype(
            self.model_config.dtype, self.cache_config.mamba_cache_dtype
        )

    def get_state_shape(self) -> Sequence[tuple[int, ...]]:
        return MambaStateShapeCalculator.short_conv_state_shape(
            tp_world_size=1,
            intermediate_size=self.hc_hidden_size,
            conv_kernel=self.conv_state_len + 1,
            num_spec=self.num_spec_tokens,
        )

    def _apply_norm(
        self, norm: Qwen4ExpPLEGroupedNorm, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        shape = hidden_states.shape
        return norm(hidden_states.flatten(-2)).reshape(shape)

    def _short_conv_fallback(self, inputs: torch.Tensor) -> torch.Tensor:
        # Profiling / CUDA graph capture only; conv state is not updated.
        inputs_t = inputs.transpose(0, 1).unsqueeze(0)
        output = self.conv1d(inputs_t)[..., : inputs_t.size(-1)]
        return F.silu(output).squeeze(0).transpose(0, 1)

    def _short_conv_dilated_decode_batched(
        self,
        x_d: torch.Tensor,
        conv_state: torch.Tensor,
        conv_weights: torch.Tensor,
        state_indices_tensor_d: torch.Tensor,
        has_initial_states_d: torch.Tensor | None,
    ) -> torch.Tensor:
        state_indices = state_indices_tensor_d.to(
            device=conv_state.device, dtype=torch.int64
        )
        # FULL cudagraph padded decode rows use NULL_BLOCK_ID. Remap them to
        # slot 0 for a safe gather, then zero output and skip write-back.
        valid_state = state_indices != NULL_BLOCK_ID
        state_indices = torch.where(
            valid_state, state_indices, torch.zeros_like(state_indices)
        )
        if has_initial_states_d is None:
            has_initial_state = valid_state
        else:
            if has_initial_states_d.numel() < state_indices_tensor_d.numel():
                raise ValueError(
                    "has_initial_states_d size mismatch: "
                    f"got {has_initial_states_d.numel()}, "
                    f"need >= {state_indices_tensor_d.numel()}."
                )
            has_initial_state = has_initial_states_d[
                : state_indices_tensor_d.numel()
            ].to(device=conv_state.device, dtype=torch.bool)
            has_initial_state = has_initial_state & valid_state

        cached_state = conv_state.index_select(0, state_indices)
        state = cached_state[..., : self.conv_state_len].to(x_d.dtype)
        if self.conv_state_len > 0:
            initial_state = torch.where(
                has_initial_state.view(-1, 1, 1),
                state,
                torch.zeros_like(state),
            )
            history = torch.cat((initial_state, x_d.unsqueeze(-1)), dim=-1)
        else:
            history = x_d.unsqueeze(-1)

        conv_output = F.conv1d(
            history,
            conv_weights.unsqueeze(1).contiguous(),
            groups=history.size(1),
            dilation=self.short_conv_dilation,
        ).squeeze(-1)
        output = F.silu(conv_output)
        output = output * valid_state.view(-1, 1).to(output.dtype)

        if self.conv_state_len > 0:
            next_state = history[..., -self.conv_state_len :]
            # Padded rows are remapped to the reserved null slot. Preserve its
            # existing value while writing the new states for valid rows.
            existing_base_state = cached_state[..., : self.conv_state_len]
            safe_next_state = torch.where(
                valid_state.view(-1, 1, 1),
                next_state.to(conv_state.dtype),
                existing_base_state,
            )
            cached_state[..., : self.conv_state_len] = safe_next_state
            conv_state.index_copy_(0, state_indices, cached_state)

        return output

    def _short_conv_dilated_prefill_batched(
        self,
        x_p: torch.Tensor,
        metadata: PleShortConvAttentionMetadata,
        conv_state: torch.Tensor,
        conv_weights: torch.Tensor,
        state_indices_tensor_p: torch.Tensor,
        num_prefills: int,
        num_decode_tokens: int,
        num_prefill_tokens: int,
    ) -> torch.Tensor:
        # ``non_spec_query_start_loc`` covers the non-spec (decode + prefill)
        # requests and equals ``query_start_loc`` when spec-decode is inactive.
        non_spec_query_start_loc = metadata.non_spec_query_start_loc
        if non_spec_query_start_loc is None:
            raise ValueError("query_start_loc is required for prefill short-conv")
        query_start_loc_p = (
            non_spec_query_start_loc[-num_prefills - 1 :] - num_decode_tokens
        )
        # The metadata builder guarantees that the prefill query offsets start
        # at 0 and end at num_prefill_tokens. Avoid reading those values here,
        # since doing so would force a device-to-host synchronization.
        has_initial_states_p = metadata.has_initial_states_p
        if has_initial_states_p is None:
            raise ValueError("has_initial_states_p is required for prefill short-conv")

        output = torch.empty_like(x_p)
        q_starts = query_start_loc_p.to(torch.int64)
        if state_indices_tensor_p.numel() < num_prefills:
            raise ValueError(
                "state_indices_tensor_p size mismatch: "
                f"got {state_indices_tensor_p.numel()}, "
                f"need >= {num_prefills}."
            )
        if has_initial_states_p.numel() < num_prefills:
            raise ValueError(
                "has_initial_states_p size mismatch: "
                f"got {has_initial_states_p.numel()}, "
                f"need >= {num_prefills}."
            )
        if num_prefills == 0 or x_p.numel() == 0:
            return output
        lengths = q_starts[1:] - q_starts[:-1]
        # Use the CPU-computed packing width from the metadata builder instead
        # of synchronizing on lengths.max().
        max_len = metadata.max_prefill_query_len
        if max_len <= 0:
            return output

        hidden_size = x_p.shape[1]
        positions = torch.arange(
            num_prefill_tokens, device=x_p.device, dtype=torch.int64
        )
        req_indices = torch.searchsorted(q_starts[1:], positions, right=True)
        col_indices = positions - q_starts[req_indices]

        packed_tokens = x_p.new_zeros((num_prefills, max_len, hidden_size))
        packed_tokens[req_indices, col_indices] = x_p
        packed_tokens = packed_tokens.transpose(1, 2).contiguous()

        state_indices = state_indices_tensor_p[:num_prefills].to(
            device=conv_state.device, dtype=torch.int64
        )
        valid_state = state_indices != NULL_BLOCK_ID
        state_indices = torch.where(
            valid_state, state_indices, torch.zeros_like(state_indices)
        )
        has_initial = has_initial_states_p[:num_prefills].to(
            device=conv_state.device, dtype=torch.bool
        )
        if self.conv_state_len > 0:
            if conv_state.shape[0] == 0:
                state = conv_state.new_zeros(
                    (num_prefills, hidden_size, self.conv_state_len),
                    dtype=x_p.dtype,
                )
            else:
                state = conv_state.index_select(0, state_indices)[
                    ..., : self.conv_state_len
                ].to(x_p.dtype)
            use_initial_mask = (valid_state & has_initial).view(num_prefills, 1, 1)
            initial_state = torch.where(
                use_initial_mask,
                state,
                torch.zeros_like(state),
            )
            history = torch.cat((initial_state, packed_tokens), dim=-1)
        else:
            history = packed_tokens

        conv_output = F.conv1d(
            history,
            conv_weights.unsqueeze(1).contiguous(),
            groups=history.size(1),
            dilation=self.short_conv_dilation,
        )
        conv_output = F.silu(conv_output).transpose(1, 2).contiguous()

        token_positions = torch.arange(max_len, device=x_p.device, dtype=torch.int64)
        valid_tokens = token_positions.view(1, max_len) < lengths.view(num_prefills, 1)
        valid_output_mask = valid_tokens & valid_state.to(device=x_p.device).view(
            num_prefills, 1
        )
        conv_output.masked_fill_(~valid_output_mask.unsqueeze(-1), 0)
        output.copy_(conv_output[req_indices, col_indices])

        if self.conv_state_len > 0 and conv_state.shape[0] > 0:
            state_starts = lengths.to(device=history.device, dtype=torch.int64).view(
                num_prefills, 1, 1
            )
            state_offsets = torch.arange(
                self.conv_state_len, device=history.device, dtype=torch.int64
            ).view(1, 1, self.conv_state_len)
            next_state = history.gather(
                dim=2,
                index=(state_starts + state_offsets).expand(-1, history.size(1), -1),
            )
            # Write back without a host synchronization. Valid, non-empty rows
            # receive their new state; padding and zero-length rows keep the
            # current cache value.
            existing_state = conv_state.index_select(0, state_indices)
            existing_base_state = existing_state[..., : self.conv_state_len]
            update_mask = valid_state & (lengths.to(device=conv_state.device) > 0)
            safe_next_state = torch.where(
                update_mask.view(num_prefills, 1, 1),
                next_state.to(conv_state.dtype),
                existing_base_state,
            )
            existing_state[..., : self.conv_state_len] = safe_next_state
            conv_state.index_copy_(0, state_indices, existing_state)
        return output

    def _short_conv_dilated_spec_batched(
        self,
        x_spec: torch.Tensor,
        conv_state: torch.Tensor,
        conv_weights: torch.Tensor,
        spec_state_indices_tensor: torch.Tensor,
        spec_query_start_loc: torch.Tensor,
        num_accepted_tokens: torch.Tensor,
        spec_query_len: int,
    ) -> torch.Tensor:
        """Dilated short-conv for speculative-decode (MTP) requests.

        Each spec request feeds multiple (draft + 1) query tokens. The conv
        outputs are computed causally after rolling back the previous draft
        state by ``num_accepted_tokens - 1``. The current candidate inputs stay
        in the extended cache for the next forward, matching
        ``causal_conv1d_update``.

        ``spec_query_len`` (== num_speculative_tokens + 1) is the maximum query
        length and is a Python int, so no host synchronization is needed; this
        keeps the path safe for full CUDA-graph capture/replay where the buffers
        are padded at the request level.
        """
        num_reqs = spec_state_indices_tensor.numel()
        hidden_size = x_spec.size(-1)
        # Use a fixed packing width instead of synchronizing on lengths.max().
        max_len = spec_query_len
        # Full CUDA graphs can pad these buffers. Only the first num_reqs
        # accepted-token counts belong to actual speculative requests.
        num_accepted_tokens = num_accepted_tokens[:num_reqs]
        q_starts = spec_query_start_loc[: num_reqs + 1].to(torch.int64)
        # Keep the number of real speculative tokens on the device.
        total_real_tokens = q_starts[num_reqs]

        state_indices = spec_state_indices_tensor.to(
            device=conv_state.device, dtype=torch.int64
        )
        valid_state = state_indices != NULL_BLOCK_ID
        state_indices = torch.where(
            valid_state, state_indices, torch.zeros_like(state_indices)
        )
        positions = torch.arange(
            x_spec.size(0), device=x_spec.device, dtype=torch.int64
        )
        # Route graph-padded token rows to the discarded dummy request so that
        # they cannot overwrite real packed data.
        req_indices = torch.searchsorted(q_starts[1:], positions, right=True)
        valid_tokens = (positions < total_real_tokens) & (req_indices < num_reqs)
        clamped_req_indices = req_indices.clamp_max(max(num_reqs - 1, 0))
        col_indices = (positions - q_starts[clamped_req_indices]).clamp_(0, max_len - 1)
        pack_req_indices = torch.where(
            valid_tokens,
            clamped_req_indices,
            torch.full_like(req_indices, num_reqs),
        )
        pack_col_indices = torch.where(
            valid_tokens, col_indices, torch.zeros_like(col_indices)
        )

        # The last request row is the dummy sink for graph padding.
        packed = x_spec.new_zeros((num_reqs + 1, max_len, hidden_size))
        packed[pack_req_indices, pack_col_indices] = x_spec
        packed = packed.transpose(1, 2).contiguous()

        if self.conv_state_len > 0:
            cached_state = conv_state.index_select(0, state_indices)
            rollback_offsets = num_accepted_tokens.to(
                device=conv_state.device, dtype=torch.int64
            ).sub(1)
            rollback_offsets = torch.where(
                valid_state,
                rollback_offsets.clamp_(0, max_len - 1),
                torch.zeros_like(rollback_offsets),
            )
            state_offsets = torch.arange(
                self.conv_state_len, device=conv_state.device, dtype=torch.int64
            ).view(1, 1, self.conv_state_len)
            rollback_indices = rollback_offsets.view(-1, 1, 1) + state_offsets
            state = cached_state.gather(
                2, rollback_indices.expand(-1, hidden_size, -1)
            ).to(x_spec.dtype)
            state = torch.where(
                valid_state.view(num_reqs, 1, 1),
                state,
                torch.zeros_like(state),
            )
            # Append a zeroed dummy-row state to match the [num_reqs + 1] pack.
            dummy_state = state.new_zeros((1, hidden_size, self.conv_state_len))
            state_full = torch.cat((state, dummy_state), dim=0)
            history = torch.cat((state_full, packed), dim=-1)
        else:
            history = packed

        conv_output = F.conv1d(
            history,
            conv_weights.unsqueeze(1).contiguous(),
            groups=history.size(1),
            dilation=self.short_conv_dilation,
        )
        conv_output = F.silu(conv_output).transpose(1, 2).contiguous()

        output = conv_output[pack_req_indices, pack_col_indices]
        output = output * valid_tokens.view(-1, 1).to(output.dtype)

        # Keep all current candidate inputs in the extended state. On the next
        # target forward, ``num_accepted_tokens - 1`` selects the rollback
        # window before processing the newly scheduled tokens.
        if self.conv_state_len > 0:
            state_capacity = self.conv_state_len + max_len - 1
            if conv_state.size(-1) < state_capacity:
                raise RuntimeError(
                    "PLE short-conv cache cannot retain speculative tokens: "
                    f"got {conv_state.size(-1)}, need {state_capacity}."
                )
            candidate_state = history[:num_reqs, :, 1 : state_capacity + 1]
            query_lengths = q_starts[1:] - q_starts[:-1]
            state_positions = torch.arange(
                state_capacity, device=history.device, dtype=torch.int64
            ).view(1, 1, state_capacity)
            update_lengths = (self.conv_state_len + query_lengths - 1).view(
                num_reqs, 1, 1
            )
            update_mask = valid_state.view(num_reqs, 1, 1) & (
                state_positions < update_lengths
            )
            existing_state = cached_state[..., :state_capacity]
            next_state = torch.where(
                update_mask,
                candidate_state.to(conv_state.dtype),
                existing_state,
            )
            cached_state[..., :state_capacity] = next_state
            conv_state.index_copy_(0, state_indices, cached_state)

        return output

    def _short_conv_dilated_dispatch(
        self,
        inputs: torch.Tensor,
        metadata: PleShortConvAttentionMetadata,
        conv_state: torch.Tensor,
        conv_weights: torch.Tensor,
    ) -> torch.Tensor:
        num_prefills = metadata.num_prefills
        num_decodes = metadata.num_decodes
        num_decode_tokens = metadata.num_decode_tokens
        num_prefill_tokens = metadata.num_prefill_tokens
        has_prefill = num_prefills > 0
        has_decode = num_decodes > 0
        has_spec = metadata.spec_sequence_masks is not None
        x = inputs[: metadata.num_actual_tokens]

        # Split spec / non-spec tokens.
        if has_spec:
            if has_prefill or has_decode:
                assert metadata.spec_token_indx is not None
                assert metadata.non_spec_token_indx is not None
                x_spec = x.index_select(0, metadata.spec_token_indx.long())
                x_non_spec = x.index_select(0, metadata.non_spec_token_indx.long())
            else:
                x_spec = x
                x_non_spec = None
        else:
            x_spec = None
            x_non_spec = x

        spec_output = None
        # 1. Run the multi-query speculative-decode part.
        if has_spec:
            assert metadata.spec_state_indices_tensor is not None
            assert metadata.spec_query_start_loc is not None
            assert metadata.num_accepted_tokens is not None
            spec_output = self._short_conv_dilated_spec_batched(
                x_spec=x_spec,
                conv_state=conv_state,
                conv_weights=conv_weights,
                spec_state_indices_tensor=metadata.spec_state_indices_tensor[
                    : metadata.num_spec_decodes
                ],
                spec_query_start_loc=metadata.spec_query_start_loc,
                num_accepted_tokens=metadata.num_accepted_tokens,
                spec_query_len=metadata.spec_query_len,
            )

        # 2. Run regular decode and prefill requests.
        conv_out_non_spec = None
        state_indices_tensor = metadata.state_indices_tensor
        if x_non_spec is not None:
            assert state_indices_tensor is not None
            if has_prefill:
                state_indices_tensor_d, state_indices_tensor_p = torch.split(
                    state_indices_tensor,
                    [num_decodes, num_prefills],
                    dim=0,
                )
                x_d, x_p = torch.split(
                    x_non_spec,
                    [num_decode_tokens, num_prefill_tokens],
                    dim=0,
                )
                non_spec_parts: list[torch.Tensor] = []
                if has_decode:
                    non_spec_parts.append(
                        self._short_conv_dilated_decode_batched(
                            x_d=x_d,
                            conv_state=conv_state,
                            conv_weights=conv_weights,
                            state_indices_tensor_d=state_indices_tensor_d,
                            has_initial_states_d=metadata.has_initial_states_d,
                        )
                    )
                non_spec_parts.append(
                    self._short_conv_dilated_prefill_batched(
                        x_p=x_p,
                        metadata=metadata,
                        conv_state=conv_state,
                        conv_weights=conv_weights,
                        state_indices_tensor_p=state_indices_tensor_p,
                        num_prefills=num_prefills,
                        num_decode_tokens=num_decode_tokens,
                        num_prefill_tokens=num_prefill_tokens,
                    )
                )
                conv_out_non_spec = torch.vstack(non_spec_parts)
            else:
                conv_out_non_spec = self._short_conv_dilated_decode_batched(
                    x_d=x_non_spec,
                    conv_state=conv_state,
                    conv_weights=conv_weights,
                    state_indices_tensor_d=state_indices_tensor[: x_non_spec.size(0)],
                    has_initial_states_d=metadata.has_initial_states_d,
                )

        # 3. Merge both parts back into the original token order.
        if has_spec and conv_out_non_spec is not None:
            assert metadata.spec_token_indx is not None
            assert metadata.non_spec_token_indx is not None
            assert spec_output is not None
            output = x.new_empty((metadata.num_actual_tokens, x.size(-1)))
            output.index_copy_(0, metadata.spec_token_indx, spec_output)
            output.index_copy_(0, metadata.non_spec_token_indx, conv_out_non_spec)
            return output
        elif has_spec:
            assert spec_output is not None
            return spec_output
        if conv_out_non_spec is None:
            return x
        return conv_out_non_spec

    def _short_conv(self, inputs: torch.Tensor) -> torch.Tensor:
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            return self._short_conv_fallback(inputs)

        if not isinstance(attn_metadata, dict):
            raise RuntimeError(
                "PLE short-conv expects per-layer attention metadata dict "
                f"during inference, got {type(attn_metadata).__name__}."
            )

        layer_attn_metadata = attn_metadata.get(self.prefix)
        if layer_attn_metadata is None:
            # MRV2 omits Mamba-family metadata during profile warmup.
            return self._short_conv_fallback(inputs)
        if not isinstance(layer_attn_metadata, PleShortConvAttentionMetadata):
            raise TypeError(
                "Expected PleShortConvAttentionMetadata for layer "
                f"'{self.prefix}', got "
                f"{type(layer_attn_metadata).__name__}."
            )

        conv_state = self.kv_cache[0]
        if not is_conv_state_dim_first():
            conv_state = conv_state.transpose(-1, -2)
        conv_weights = self.conv1d.weight.squeeze(1)

        state_capacity = self.conv_state_len + self.num_spec_tokens
        if state_capacity > 0:
            if conv_state.size(-1) < state_capacity:
                raise RuntimeError(
                    "PLE short-conv cache is smaller than expected for "
                    f"dilated convolution: got {conv_state.size(-1)}, "
                    f"expect at least {state_capacity}."
                )
            conv_state = conv_state[..., -state_capacity:]
        return self._short_conv_dilated_dispatch(
            inputs,
            layer_attn_metadata,
            conv_state,
            conv_weights.to(dtype=inputs.dtype),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
    ) -> torch.Tensor:
        input_ids = input_ids.reshape(-1)
        if input_ids.shape[0] != hidden_states.shape[0]:
            raise ValueError(
                "PLE expects input_ids and hidden_states to have the same "
                f"token length, got {input_ids.shape[0]} and "
                f"{hidden_states.shape[0]}"
            )
        embeddings = torch.ops.vllm.qwen4_exp_ple_embed(
            hidden_states,
            input_ids,
            query_start_loc,
            ngram_context,
            self.prefix,
        )
        embeddings = self._dequantize_embeddings(embeddings, hidden_states.dtype)
        key, _ = self.key_proj(embeddings)
        value, _ = self.value_proj(embeddings)
        token_count = hidden_states.shape[0]
        key = key.reshape(token_count, self.hc_count, self.hidden_size)
        query = hidden_states.reshape(token_count, self.hc_count, self.hidden_size)
        key = self._apply_norm(self.norm_key, key)
        query = self._apply_norm(self.norm_query, query)
        gate = (key * query).sum(dim=-1, keepdim=True) / math.sqrt(self.hidden_size)
        gate = torch.sigmoid(gate.sign() * gate.abs().clamp_min(1e-6).sqrt())
        gated_value = gate * value.unsqueeze(-2)
        normalized = self._apply_norm(self.norm_conv, gated_value).flatten(-2)
        conv_output = torch.zeros_like(normalized)
        torch.ops.vllm.qwen4_exp_ple_short_conv(
            normalized,
            conv_output,
            self.prefix,
        )
        return gated_value.flatten(-2) + conv_output


def qwen4_exp_compute_ple_ngram_ids(
    input_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    ngram_context: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    """Compute request-dependent PLE n-gram IDs outside piecewise graphs."""
    layer = get_forward_context().no_compile_layers[layer_name]
    ngram_ids = layer.ple_embedding.compute_ngram_ids(
        input_ids,
        query_start_loc,
        ngram_context,
    )
    output.copy_(ngram_ids)


def qwen4_exp_compute_ple_ngram_ids_fake(
    input_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    ngram_context: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    return


def qwen4_exp_ple_short_conv(
    inputs: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    layer = get_forward_context().no_compile_layers[layer_name]
    result = layer._short_conv(inputs)
    output[: result.shape[0]].copy_(result)


def qwen4_exp_ple_short_conv_fake(
    inputs: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    return


direct_register_custom_op(
    op_name="qwen4_exp_compute_ple_ngram_ids",
    op_func=qwen4_exp_compute_ple_ngram_ids,
    mutates_args=["output"],
    fake_impl=qwen4_exp_compute_ple_ngram_ids_fake,
)


direct_register_custom_op(
    op_name="qwen4_exp_ple_short_conv",
    op_func=qwen4_exp_ple_short_conv,
    mutates_args=["output"],
    fake_impl=qwen4_exp_ple_short_conv_fake,
)


# Keep data-dependent n-gram hashing and the large table gather out of the
# compiled model graph, where Inductor can miscompile the embedding indices.
def qwen4_exp_ple_embed(
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    ngram_context: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    """Run the PLE embedding lookup eagerly."""
    layer = get_forward_context().no_compile_layers[layer_name]
    return layer.ple_embedding(hidden_states, input_ids, query_start_loc, ngram_context)


def qwen4_exp_ple_embed_fake(
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    ngram_context: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    """Return PLE lookup metadata while tracing the custom op."""
    del query_start_loc, ngram_context
    layer = get_forward_context().no_compile_layers[layer_name]
    embedding = layer.ple_embedding
    if embedding._is_cpu_offloaded:
        output_buffer = embedding._gpu_output_buffer
        return torch.empty(
            (input_ids.shape[0], output_buffer.shape[-1]),
            device=hidden_states.device,
            dtype=output_buffer.dtype,
        )
    quant_method = embedding.ngram_embedding.quant_method
    if isinstance(quant_method, Qwen4ExpPLENVFp4EmbeddingMethod):
        return torch.empty(
            (
                input_ids.shape[0],
                embedding.ngram_heads * quant_method.packed_row_width,
            ),
            device=hidden_states.device,
            dtype=torch.uint8,
        )
    return torch.empty(
        (input_ids.shape[0], embedding.embedding_dim),
        device=hidden_states.device,
        dtype=embedding.ngram_embedding.weight.dtype,
    )


direct_register_custom_op(
    op_name="qwen4_exp_ple_embed",
    op_func=qwen4_exp_ple_embed,
    mutates_args=[],
    fake_impl=qwen4_exp_ple_embed_fake,
)


__all__ = [
    "Qwen4ExpNGramEmbedding",
    "Qwen4ExpPLEGroupedNorm",
    "Qwen4ExpPLENVFp4EmbeddingMethod",
    "Qwen4ExpPLELayer",
]

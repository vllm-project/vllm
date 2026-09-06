# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU-resident Qwen4Exp position-learning enhancement layers."""

from collections.abc import Iterable, Sequence

import torch
import torch.nn.functional as F
from torch import nn

import vllm.envs as envs
from vllm.config import CacheConfig, ModelConfig, VllmConfig, get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import MergedColumnParallelLinear
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
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptMixedPrecisionConfig,
    ModelOptNvFp4Config,
)
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

from ..common.ple import PLEVocabParallelEmbedding, copy_ple_embedding_shard_
from .ops.ple import ple_conv, ple_gate, ple_ngram_ids

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
            scale_dtype=torch.float32,
        )
        layer.register_parameter("weight_scale", weight_scale)

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        """Reject FP8 PLE checkpoints without a global scale."""
        sentinel = torch.finfo(torch.float32).min
        if torch.any(layer.weight_scale == sentinel):
            raise ValueError("FP8 PLE checkpoint is missing its global scale")

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

    if isinstance(quant_config, ModelOptMixedPrecisionConfig):
        if quant_config._resolve_quant_algo(prefix) == "FP8":
            return Qwen4ExpPLEFp8EmbeddingMethod()
        return None

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
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute n-gram embedding indices for the current request layout."""
        input_ids = input_ids.reshape(-1)
        num_reqs = query_start_loc.numel() - 1
        num_tokens = input_ids.shape[0]

        if input_ids.is_cuda:
            return ple_ngram_ids(
                input_ids=input_ids,
                query_start_loc=query_start_loc,
                ngram_context=ngram_context,
                layer_multipliers=self.layer_multipliers,
                ngram_heads_vocab_sizes=self.ngram_heads_vocab_sizes,
                ngram_heads_offsets=self.ngram_heads_offsets,
                eos_token_id=self.eos_token_id,
                heads_per_ngram=self.heads_per_ngram,
                output=output,
            )
        input_ids = input_ids.long()
        query_start_loc = query_start_loc.long()
        if num_reqs <= 0:
            raise ValueError("PLE requires at least one request")

        if is_offload_process():
            max_seq_len = max(
                1,
                int((query_start_loc[1:] - query_start_loc[:-1]).max().item()),
            )
            # CUDA graphs can pad input_ids beyond the valid request layout.
            # Keep those rows from overwriting the final real token.
            num_valid_tokens = min(int(query_start_loc[-1].item()), num_tokens)
        else:
            max_seq_len = num_tokens
            num_valid_tokens = num_tokens

        positions = torch.arange(num_tokens, device=input_ids.device, dtype=torch.int64)
        packed = torch.full(
            (num_reqs, max_seq_len),
            self.eos_token_id,
            device=input_ids.device,
            dtype=torch.int64,
        )
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
        if input_ids.is_cuda:
            # Keep request-dependent ID generation outside piecewise graphs.
            ngram_ids = input_ids.new_empty(
                (input_ids.numel(), self.ngram_heads), dtype=torch.long
            )
            torch.ops.vllm.qwen4_exp_compute_ple_ngram_ids(
                input_ids,
                query_start_loc,
                ngram_context,
                ngram_ids,
                self.layer_name,
            )
        else:
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
        # The PLE cache is TP-replicated, so this merged projection is too.
        self.kv_proj = MergedColumnParallelLinear(
            int(config.ple_embed_dim),
            [self.hc_hidden_size, self.hidden_size],
            bias=False,
            params_dtype=model_config.dtype,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_proj",
            disable_tp=True,
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

    def _short_conv_dilated_dispatch(
        self,
        inputs: torch.Tensor,
        residual: torch.Tensor,
        metadata: PleShortConvAttentionMetadata,
        conv_state: torch.Tensor,
        conv_weights: torch.Tensor,
    ) -> None:
        num_prefills = metadata.num_prefills
        num_decodes = metadata.num_decodes
        num_decode_tokens = metadata.num_decode_tokens
        num_prefill_tokens = metadata.num_prefill_tokens
        has_prefill = num_prefills > 0
        has_decode = num_decodes > 0
        has_spec = metadata.spec_sequence_masks is not None
        has_non_spec = has_prefill or has_decode
        inputs = inputs[: metadata.num_actual_tokens]
        residual = residual[: metadata.num_actual_tokens]

        spec_token_indices = None
        non_spec_token_indices = None
        if has_spec and has_non_spec:
            assert metadata.spec_token_indx is not None
            assert metadata.non_spec_token_indx is not None
            spec_token_indices = metadata.spec_token_indx
            non_spec_token_indices = metadata.non_spec_token_indx

        if has_spec:
            assert metadata.spec_state_indices_tensor is not None
            query_start_loc = metadata.spec_query_start_loc
            num_accepted_tokens = metadata.num_accepted_tokens
            assert query_start_loc is not None
            assert num_accepted_tokens is not None
            spec_state_indices = metadata.spec_state_indices_tensor[
                : metadata.num_spec_decodes
            ]
            # Mixed batches stay in their original row order; the kernels map
            # logical spec/non-spec rows instead of materializing both groups.
            ple_conv(
                inputs=inputs,
                residual=residual,
                conv_state=conv_state,
                conv_weights=conv_weights,
                state_indices=spec_state_indices,
                mode="spec",
                dilation=self.short_conv_dilation,
                query_start_loc=query_start_loc,
                num_accepted_tokens=num_accepted_tokens,
                spec_query_len=metadata.spec_query_len,
                token_indices=spec_token_indices,
            )

        if not has_non_spec:
            return

        state_indices = metadata.state_indices_tensor
        assert state_indices is not None
        if has_prefill:
            state_indices_d, state_indices_p = torch.split(
                state_indices, [num_decodes, num_prefills], dim=0
            )
            if non_spec_token_indices is None:
                inputs_d, inputs_p = torch.split(
                    inputs, [num_decode_tokens, num_prefill_tokens], dim=0
                )
                residual_d, residual_p = torch.split(
                    residual, [num_decode_tokens, num_prefill_tokens], dim=0
                )
                token_indices_d = None
                token_indices_p = None
            else:
                inputs_d = inputs_p = inputs
                residual_d = residual_p = residual
                token_indices_d, token_indices_p = torch.split(
                    non_spec_token_indices,
                    [num_decode_tokens, num_prefill_tokens],
                    dim=0,
                )

            if has_decode:
                ple_conv(
                    inputs=inputs_d,
                    residual=residual_d,
                    conv_state=conv_state,
                    conv_weights=conv_weights,
                    state_indices=state_indices_d,
                    mode="decode",
                    dilation=self.short_conv_dilation,
                    has_initial_states=metadata.has_initial_states_d,
                    token_indices=token_indices_d,
                )

            query_start_loc = metadata.non_spec_query_start_loc
            if query_start_loc is None:
                raise ValueError("query_start_loc is required for prefill short-conv")
            query_start_loc = query_start_loc[-num_prefills - 1 :] - num_decode_tokens
            has_initial_states = metadata.has_initial_states_p
            if has_initial_states is None:
                raise ValueError(
                    "has_initial_states_p is required for prefill short-conv"
                )
            ple_conv(
                inputs=inputs_p,
                residual=residual_p,
                conv_state=conv_state,
                conv_weights=conv_weights,
                state_indices=state_indices_p,
                mode="prefill",
                dilation=self.short_conv_dilation,
                query_start_loc=query_start_loc,
                has_initial_states=has_initial_states,
                token_indices=token_indices_p,
            )
        else:
            num_decode_rows = (
                non_spec_token_indices.numel()
                if non_spec_token_indices is not None
                else inputs.size(0)
            )
            ple_conv(
                inputs=inputs,
                residual=residual,
                conv_state=conv_state,
                conv_weights=conv_weights,
                state_indices=state_indices[:num_decode_rows],
                mode="decode",
                dilation=self.short_conv_dilation,
                has_initial_states=metadata.has_initial_states_d,
                token_indices=non_spec_token_indices,
            )

    def _short_conv(self, inputs: torch.Tensor, residual: torch.Tensor) -> None:
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        # Profiling omits all metadata or this Mamba entry. The residual
        # already contains the gated output, so short convolution is a no-op.
        if attn_metadata is None:
            return

        if not isinstance(attn_metadata, dict):
            raise RuntimeError(
                "PLE short-conv expects per-layer attention metadata dict "
                f"during inference, got {type(attn_metadata).__name__}."
            )

        layer_attn_metadata = attn_metadata.get(self.prefix)
        if layer_attn_metadata is None:
            return
        if not isinstance(layer_attn_metadata, PleShortConvAttentionMetadata):
            raise TypeError(
                "Expected PleShortConvAttentionMetadata for layer "
                f"'{self.prefix}', got "
                f"{type(layer_attn_metadata).__name__}."
            )

        conv_state = self.kv_cache[0]
        # Canonicalize both backend cache layouts to [slot, channel, window].
        if not is_conv_state_dim_first():
            conv_state = conv_state.transpose(-1, -2)
        conv_weights = self.conv1d.weight.squeeze(1)

        state_capacity = self.conv_state_len + self.num_spec_tokens
        if state_capacity > 0:
            state_size = conv_state.size(-1)
            if state_size < state_capacity:
                raise RuntimeError(
                    "PLE short-conv cache is smaller than expected for "
                    f"dilated convolution: got {state_size}, "
                    f"expect at least {state_capacity}."
                )
            conv_state = conv_state[..., -state_capacity:]
        self._short_conv_dilated_dispatch(
            inputs=inputs,
            residual=residual,
            metadata=layer_attn_metadata,
            conv_state=conv_state,
            conv_weights=conv_weights.to(dtype=inputs.dtype),
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
        kv, _ = self.kv_proj(embeddings)
        key, value = kv.split(self.kv_proj.output_sizes, dim=-1)
        gated_output, conv_input = ple_gate(
            key,
            value,
            hidden_states,
            self.norm_key.weight,
            self.norm_query.weight,
            self.norm_conv.weight,
            self.norm_key.eps,
        )
        # State routing depends on runtime request metadata and remains outside
        # the piecewise graph; short convolution accumulates into gated_output.
        torch.ops.vllm.qwen4_exp_ple_short_conv(conv_input, gated_output, self.prefix)
        return gated_output


def qwen4_exp_compute_ple_ngram_ids(
    input_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    ngram_context: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    """Compute request-dependent PLE n-gram IDs outside piecewise graphs."""
    layer = get_forward_context().no_compile_layers[layer_name]
    layer.ple_embedding.compute_ngram_ids(
        input_ids,
        query_start_loc,
        ngram_context,
        output,
    )


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
    residual_output: torch.Tensor,
    layer_name: str,
) -> None:
    layer = get_forward_context().no_compile_layers[layer_name]
    layer._short_conv(inputs, residual_output)


def qwen4_exp_ple_short_conv_fake(
    inputs: torch.Tensor,
    residual_output: torch.Tensor,
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
    mutates_args=["residual_output"],
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

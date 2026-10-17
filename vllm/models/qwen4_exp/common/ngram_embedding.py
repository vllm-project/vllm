# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared Qwen4Exp n-gram embedding storage with device and pinned-host backends.

Both the NVIDIA and AMD Qwen4Exp implementations use these classes so the (large)
n-gram embedding table can be kept in pinned host memory and looked up through
Unified Virtual Addressing on any CUDA-alike platform.
"""

import mmap
import threading
import weakref
from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.distributed import get_dp_group, get_etp_group, get_tp_group
from vllm.distributed.device_communicators.cuda_wrapper import CudaRTLibrary
from vllm.forward_context import DPMetadata, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
    CompressedTensorsConfig,
)
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.inc import INCConfig
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptMixedPrecisionConfig,
    ModelOptQuantConfigBase,
)
from vllm.model_executor.layers.quantization.quark.quark import QuarkConfig
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    create_fp8_scale_parameter,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    _e2m1_inline,
    _e4m3_inline,
    dequantize_to_dtype,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped,
)
from vllm.model_executor.parameter import (
    ModelWeightParameter,
    PerTensorScaleParameter,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.platform_utils import is_uva_available
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

from .ple import PLEVocabParallelEmbedding

logger = init_logger(__name__)


class Qwen4ExpPLEEmbedding(PLEVocabParallelEmbedding, ABC):
    """ETP-sharded PLE table shared by device and pinned-host backends."""

    supports_prefetch: ClassVar[bool] = False

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        params_dtype: torch.dtype,
        padding_size: int,
        prefix: str,
        embedding_method: "Qwen4ExpPLEEmbeddingMethod",
        num_ngram_heads: int = 1,
        max_total_tokens: int = 0,
        data_parallel_rank: int = 0,
        reduce_results: bool = True,
    ) -> None:
        del num_ngram_heads, max_total_tokens
        super().__init__(
            num_embeddings,
            embedding_dim,
            params_dtype=params_dtype,
            padding_size=padding_size,
            prefix=prefix,
            quant_method=embedding_method,
            parallel_group=get_etp_group(),
            reduce_results=reduce_results,
        )
        self.embedding_method = embedding_method
        self.data_parallel_rank = data_parallel_rank
        tp_size = get_tp_group().world_size
        if self.tp_size % tp_size:
            raise ValueError(
                "ETP size must be divisible by TP size, but got "
                f"ETP={self.tp_size} and TP={tp_size}"
            )
        self.etp_data_parallel_size = self.tp_size // tp_size

    @abstractmethod
    def allocate_embedding_weight(
        self,
        num_embeddings: int,
        embedding_dim: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Allocate storage for the complete embedding weight."""
        raise NotImplementedError

    def dequantize(
        self,
        embeddings: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Delegate storage-format conversion to the embedding method."""
        return self.embedding_method.dequantize(self, embeddings, output_dtype)

    def _get_dp_gather_slot(self, local_num_tokens: int) -> tuple[int, int]:
        """Return the per-DP slot size and this rank's slot offset."""
        if self.etp_data_parallel_size == 1:
            return local_num_tokens, 0
        dp_metadata: DPMetadata | None = get_forward_context().dp_metadata
        if dp_metadata is None:
            raise RuntimeError("ETP spanning DP requires DP token metadata")
        group_start = (self.data_parallel_rank // self.etp_data_parallel_size) * (
            self.etp_data_parallel_size
        )
        group_end = group_start + self.etp_data_parallel_size
        token_counts = dp_metadata.num_tokens_across_dp_cpu.tolist()
        group_counts = token_counts[group_start:group_end]
        slot_size = max(group_counts)
        dp_rank = get_dp_group().rank_in_group
        return slot_size, dp_rank * slot_size

    def _gather_dp_ids(
        self,
        ngram_ids: torch.Tensor,
        slot_size: int,
    ) -> torch.Tensor:
        """Gather DP-local IDs that share one ETP-sharded PLE table."""
        if self.etp_data_parallel_size == 1:
            return ngram_ids
        if ngram_ids.shape[0] < slot_size:
            padding = ngram_ids.new_zeros(
                slot_size - ngram_ids.shape[0], ngram_ids.shape[1]
            )
            ngram_ids = torch.cat((ngram_ids, padding), dim=0)
        return get_dp_group().all_gather(ngram_ids, dim=0)

    def _select_embeddings(
        self,
        embeddings: torch.Tensor,
        local_num_tokens: int,
        slot_offset: int,
    ) -> torch.Tensor:
        """Select this DP rank's rows from the ETP-reduced embeddings."""
        if self.etp_data_parallel_size == 1:
            return embeddings
        return embeddings.narrow(0, slot_offset, local_num_tokens)

    @abstractmethod
    def start_prefetch(self, ngram_ids: torch.Tensor) -> None:
        """Start an asynchronous lookup when supported."""
        raise NotImplementedError

    @abstractmethod
    def finalize_prefetch(self, total_tokens: int) -> torch.Tensor:
        """Return prefetched embeddings for the full input token set."""
        raise NotImplementedError


class Qwen4ExpPLEEmbeddingMethod(QuantizeMethodBase):
    """Quantization interface shared by resident and pinned PLE tables."""

    # PLE post-load processing only validates scales in their current storage.
    requires_device_loading: bool = False

    @staticmethod
    def from_quant_config(
        quant_config: QuantizationConfig | None,
        prefix: str,
        embedding_dtype: str | None = None,
    ) -> "Qwen4ExpPLEEmbeddingMethod":
        """Select the concrete PLE embedding format for a layer."""
        if embedding_dtype == "nvfp4":
            return Qwen4ExpPLENvFp4EmbeddingMethod()
        if embedding_dtype == "float8_e4m3fn":
            return Qwen4ExpPLEFp8EmbeddingMethod()
        if embedding_dtype is not None:
            raise ValueError(
                "ple_embedding_dtype must be 'nvfp4' or 'float8_e4m3fn', "
                f"got {embedding_dtype!r}"
            )
        if quant_config is None:
            return Qwen4ExpPLEUnquantizedEmbeddingMethod()
        if isinstance(quant_config, ModelOptMixedPrecisionConfig):
            quant_algo = quant_config._resolve_quant_algo(prefix)
            if quant_algo == "FP8":
                return Qwen4ExpPLEFp8EmbeddingMethod()
            if quant_algo == "NVFP4":
                return Qwen4ExpPLENvFp4EmbeddingMethod()
            if quant_algo is not None:
                raise ValueError(
                    f"Unsupported quant_algo {quant_algo!r} for the PLE embedding "
                    f"{prefix!r}; expected FP8 or NVFP4"
                )
            return Qwen4ExpPLEUnquantizedEmbeddingMethod()
        if isinstance(
            quant_config, ModelOptQuantConfigBase
        ) and quant_config.is_layer_excluded(prefix):
            return Qwen4ExpPLEUnquantizedEmbeddingMethod()
        if (
            isinstance(quant_config, CompressedTensorsConfig)
            and quant_config.get_scheme_dict(None, layer_name=prefix) is None
        ):
            return Qwen4ExpPLEUnquantizedEmbeddingMethod()
        if (
            isinstance(quant_config, INCConfig)
            and not quant_config.config_parser.resolve(None, prefix).quantized
        ):
            return Qwen4ExpPLEUnquantizedEmbeddingMethod()
        # Quark quantizes only Linear and MoE layers; PLE tables stay BF16.
        if isinstance(quant_config, QuarkConfig):
            return Qwen4ExpPLEUnquantizedEmbeddingMethod()
        if not isinstance(quant_config, Fp8Config):
            raise NotImplementedError(
                "Qwen4Exp PLE embedding does not support quantization config "
                f"{type(quant_config).__name__}"
            )

        ignored_layers = quant_config.ignored_layers
        if is_layer_skipped(
            prefix,
            ignored_layers,
            quant_config.packed_modules_mapping,
            match_mode=quant_config.ignored_layers_match_mode,
        ):
            return Qwen4ExpPLEUnquantizedEmbeddingMethod()
        # PLE checkpoint shards form one runtime embedding parameter.
        shard_prefix = f"{prefix}.shard_"
        if any(name.startswith(shard_prefix) for name in ignored_layers):
            return Qwen4ExpPLEUnquantizedEmbeddingMethod()
        if not quant_config.is_checkpoint_fp8_serialized:
            raise NotImplementedError(
                "Qwen4Exp PLE embedding only supports serialized FP8 checkpoints"
            )
        return Qwen4ExpPLEFp8EmbeddingMethod()

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError("PLE weights only support embedding lookup")

    def embedding(self, layer: nn.Module, input_: torch.Tensor) -> torch.Tensor:
        return F.embedding(input_, layer.weight)

    def lookup_dtype(self, layer: nn.Module) -> torch.dtype:
        """Return the dtype of rows that a pinned-host lookup writes."""
        return layer.weight.dtype

    def lookup_from_pinned(
        self,
        layer: "Qwen4ExpPLEPinnedHostEmbedding",
        ids: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        """Copy the ETP-owned rows for ``ids`` from pinned host memory."""
        row_bytes = layer.embedding_dim * layer.weight.element_size()
        _lookup_ple_embedding_from_pinned_kernel[(ids.numel(),)](
            layer._uva_weight,
            ids,
            output,
            row_bytes,
            layer.shard_indices.org_vocab_start_index,
            layer.shard_indices.org_vocab_end_index,
            BLOCK_D=triton.next_power_of_2(row_bytes),
        )

    @abstractmethod
    def dequantize(
        self,
        layer: nn.Module,
        embeddings: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Convert looked-up PLE rows to the activation dtype."""
        raise NotImplementedError


class Qwen4ExpPLEUnquantizedEmbeddingMethod(Qwen4ExpPLEEmbeddingMethod):
    """Unquantized PLE embedding storage and lookup semantics."""

    def create_weights(
        self,
        layer: Qwen4ExpPLEEmbedding,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del input_size, output_size
        weight = nn.Parameter(
            layer.allocate_embedding_weight(
                sum(output_partition_sizes),
                input_size_per_partition,
                params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        set_weight_attrs(weight, extra_weight_attrs)
        layer.register_parameter("weight", weight)

    def dequantize(
        self,
        layer: nn.Module,
        embeddings: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        del layer, output_dtype
        return embeddings


class Qwen4ExpPLEFp8EmbeddingMethod(Qwen4ExpPLEEmbeddingMethod):
    """FP8 PLE embedding with one global checkpoint scale."""

    def create_weights(
        self,
        layer: Qwen4ExpPLEEmbedding,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del input_size, output_size, params_dtype
        weight_loader = extra_weight_attrs.get("weight_loader")
        weight = ModelWeightParameter(
            data=layer.allocate_embedding_weight(
                sum(output_partition_sizes),
                input_size_per_partition,
                torch.float8_e4m3fn,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
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

    def dequantize(
        self,
        layer: nn.Module,
        embeddings: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        weight_scale = getattr(layer, "weight_scale", None)
        if weight_scale is None:
            raise RuntimeError("FP8 PLE embedding is missing its global scale")
        if weight_scale.device != embeddings.device:
            raise RuntimeError("FP8 PLE embedding scale must be on the output device")
        return embeddings.to(output_dtype) * weight_scale.to(output_dtype)


# NVFP4 packs two E2M1 values per byte and shares one E4M3 scale per block.
_NVFP4_VALUES_PER_BYTE = 2
_NVFP4_BLOCK_SIZE = 16


class Qwen4ExpPLENvFp4EmbeddingMethod(Qwen4ExpPLEEmbeddingMethod):
    """NVFP4 PLE embedding: packed E2M1 rows, block scales, one global scale.

    Rows stay packed in storage. A lookup decodes only the selected rows to
    the activation dtype, before the ETP reduction.
    """

    def create_weights(
        self,
        layer: Qwen4ExpPLEEmbedding,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del input_size, output_size, params_dtype
        if input_size_per_partition % _NVFP4_BLOCK_SIZE:
            raise ValueError(
                "NVFP4 PLE embedding dimension must be divisible by "
                f"{_NVFP4_BLOCK_SIZE}, got {input_size_per_partition}"
            )
        weight_loader = extra_weight_attrs.get("weight_loader")
        num_rows = sum(output_partition_sizes)
        for name, width, dtype in (
            (
                "weight",
                input_size_per_partition // _NVFP4_VALUES_PER_BYTE,
                torch.uint8,
            ),
            (
                "weight_scale",
                input_size_per_partition // _NVFP4_BLOCK_SIZE,
                torch.float8_e4m3fn,
            ),
        ):
            parameter = ModelWeightParameter(
                data=layer.allocate_embedding_weight(num_rows, width, dtype),
                input_dim=1,
                output_dim=0,
                weight_loader=weight_loader,
            )
            layer.register_parameter(name, parameter)
        weight_scale_2 = create_fp8_scale_parameter(
            PerTensorScaleParameter,
            output_partition_sizes,
            input_size_per_partition,
            None,
            weight_loader,
            scale_dtype=torch.float32,
        )
        layer.register_parameter("weight_scale_2", weight_scale_2)
        if layer.weight_scale.is_pinned():
            self._uva_weight_scale = get_accelerator_view_from_cpu_tensor(
                layer.weight_scale
            )

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        """Reject NVFP4 PLE checkpoints without a global scale."""
        sentinel = torch.finfo(torch.float32).min
        if torch.any(layer.weight_scale_2 == sentinel):
            raise ValueError("NVFP4 PLE checkpoint is missing its global scale")

    def lookup_dtype(self, layer: nn.Module) -> torch.dtype:
        return layer.params_dtype

    def embedding(self, layer: nn.Module, input_: torch.Tensor) -> torch.Tensor:
        ids = input_.reshape(-1)
        if not input_.is_cuda:
            rows = dequantize_to_dtype(
                F.embedding(ids, layer.weight),
                F.embedding(ids, layer.weight_scale),
                layer.weight_scale_2,
                layer.params_dtype,
                swizzle=False,
            )
            return rows.reshape(*input_.shape, layer.embedding_dim)
        output = torch.empty(
            (*input_.shape, layer.embedding_dim),
            dtype=layer.params_dtype,
            device=input_.device,
        )
        if ids.numel():
            # Device rows are already local, so every ID in range is owned.
            _lookup_nvfp4_ple_embedding_kernel[(ids.numel(),)](
                layer.weight,
                layer.weight_scale.view(torch.uint8),
                layer.weight_scale_2,
                ids,
                output,
                layer.embedding_dim,
                0,
                layer.weight.shape[0],
                BLOCK_D=triton.next_power_of_2(layer.embedding_dim),
            )
        return output

    def lookup_from_pinned(
        self,
        layer: "Qwen4ExpPLEPinnedHostEmbedding",
        ids: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        _lookup_nvfp4_ple_embedding_kernel[(ids.numel(),)](
            layer._uva_weight,
            self._uva_weight_scale.view(torch.uint8),
            layer.weight_scale_2,
            ids,
            output,
            layer.embedding_dim,
            layer.shard_indices.org_vocab_start_index,
            layer.shard_indices.org_vocab_end_index,
            BLOCK_D=triton.next_power_of_2(layer.embedding_dim),
        )

    def dequantize(
        self,
        layer: nn.Module,
        embeddings: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        del layer
        # The lookup already decoded the rows to the activation dtype.
        return embeddings.to(output_dtype)


@triton.jit
def _lookup_nvfp4_ple_embedding_kernel(
    weight_ptr,
    scale_ptr,
    global_scale_ptr,
    ids_ptr,
    output_ptr,
    embedding_dim,
    vocab_start,
    vocab_end,
    BLOCK_D: tl.constexpr,
):
    """Decode owned NVFP4 PLE rows; write zeros for rows another rank owns."""
    row = tl.program_id(0).to(tl.int64)
    idx = tl.load(ids_ptr + row).to(tl.int64)
    owned = (idx >= vocab_start) & (idx < vocab_end)
    local_idx = tl.where(owned, idx - vocab_start, 0)
    offsets = tl.arange(0, BLOCK_D)
    in_row = offsets < embedding_dim
    mask = owned & in_row
    packed = tl.load(
        weight_ptr + local_idx * (embedding_dim // 2) + offsets // 2,
        mask=mask,
        other=0,
    )
    # The low nibble holds the even element and the high nibble the odd one.
    codes = (packed >> ((offsets % 2) * 4)) & 0xF
    scales = _e4m3_inline(
        tl.load(
            scale_ptr + local_idx * (embedding_dim // 16) + offsets // 16,
            mask=mask,
            other=0,
        )
    )
    global_scale = tl.load(global_scale_ptr).to(tl.float32)
    values = _e2m1_inline(codes) * scales * global_scale
    tl.store(output_ptr + row * embedding_dim + offsets, values, mask=in_row)


class Qwen4ExpPLEDeviceEmbedding(Qwen4ExpPLEEmbedding):
    """PLE table allocated on the active model device."""

    def allocate_embedding_weight(
        self,
        num_embeddings: int,
        embedding_dim: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Allocate the complete PLE weight on the active device."""
        return torch.empty(num_embeddings, embedding_dim, dtype=dtype)

    def start_prefetch(self, ngram_ids: torch.Tensor) -> None:
        """Resident embedding prefetch is a no-op."""
        return None

    def finalize_prefetch(self, total_tokens: int) -> torch.Tensor:
        """Reject prefetch finalization for resident embeddings."""
        raise RuntimeError("Device PLE embedding does not support prefetch")

    def forward(self, ngram_ids: torch.Tensor) -> torch.Tensor:
        """Gather ETP IDs, optionally reduce lookup results, and select local rows."""
        slot_size, slot_offset = self._get_dp_gather_slot(ngram_ids.shape[0])
        gathered_ids = self._gather_dp_ids(ngram_ids, slot_size)
        embeddings = super().forward(gathered_ids)
        return self._select_embeddings(
            embeddings,
            ngram_ids.shape[0],
            slot_offset,
        )


@triton.jit
def _lookup_ple_embedding_from_pinned_kernel(
    weight_ptr: tl.pointer_type(tl.uint8),  # type: ignore[valid-type]
    ids_ptr,
    output_ptr: tl.pointer_type(tl.uint8),  # type: ignore[valid-type]
    row_bytes,
    tp_vocab_start,
    tp_vocab_end,
    BLOCK_D: tl.constexpr,
):
    """Copy TP-owned PLE rows as raw bytes from a CUDA view of pinned host memory.

    Byte pointers keep the storage dtype out of the kernel signature, so
    dtypes Triton cannot lower on the GPU (FP8 E4M3FN before SM89) still work.
    """
    row_id = tl.program_id(0).to(tl.int64)
    global_idx = tl.load(ids_ptr + row_id)
    in_range = (global_idx >= tp_vocab_start) & (global_idx < tp_vocab_end)
    local_idx = tl.where(in_range, global_idx - tp_vocab_start, 0)
    offsets = tl.arange(0, BLOCK_D)
    store_mask = offsets < row_bytes
    load_mask = store_mask & in_range
    values = tl.load(
        weight_ptr + local_idx * row_bytes + offsets,
        mask=load_mask,
        other=0,
    )
    tl.store(
        output_ptr + row_id * row_bytes + offsets,
        values,
        mask=store_mask,
    )


def _unregister_host_table(
    cudart: CudaRTLibrary, mapping: mmap.mmap, pointer: int
) -> None:
    # Holds the mapping until CUDA has released the registration.
    if cudart.cudaHostUnregister(pointer) != 0:
        cudart.cudaGetLastError()
        logger.warning("cudaHostUnregister of the PLE table failed")


def _registered_host_table(
    num_embeddings: int, embedding_dim: int, dtype: torch.dtype
) -> torch.Tensor | None:
    """Exact-size registered host table, or None to fall back to pin_memory=True.

    The mapping is unregistered once the table and every view of it, including
    the UVA view, are gone.
    """
    num_bytes = num_embeddings * embedding_dim * dtype.itemsize
    if num_bytes == 0 or not current_platform.is_cuda():
        return None
    try:
        cudart = CudaRTLibrary()
        mapping = mmap.mmap(-1, num_bytes, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
        # A fork must not copy-on-write the pinned pages into the child.
        mapping.madvise(mmap.MADV_DONTFORK)
    except (AssertionError, AttributeError, OSError) as exc:
        logger.warning_once("PLE table mapping failed (%s); using pin_memory.", exc)
        return None
    owner = np.frombuffer(mapping, dtype=np.uint8)
    pointer = owner.ctypes.data
    # Register, drain and unregister through one runtime handle.
    if (result := cudart.cudaHostRegister(pointer, num_bytes)) != 0:
        cudart.cudaGetLastError()
        logger.warning_once(
            "cudaHostRegister of the PLE table failed (error %d); using pin_memory.",
            result,
        )
        return None
    # Tensor views retain owner; its finalizer retains mapping until unregister.
    finalizer = weakref.finalize(
        owner, _unregister_host_table, cudart, mapping, pointer
    )
    finalizer.atexit = False  # type: ignore[misc]
    table = torch.from_numpy(owner)
    # The UVA view would otherwise silently use a private pinned copy.
    if not table.is_pinned():
        logger.warning_once("CUDA did not pin the PLE table; using pin_memory.")
        return None
    return table.view(dtype).view(num_embeddings, embedding_dim)


class Qwen4ExpPLEPinnedHostEmbedding(Qwen4ExpPLEEmbedding):
    """PLE table loaded into pinned CPU memory and looked up through UVA."""

    supports_prefetch: ClassVar[bool] = True

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        params_dtype: torch.dtype,
        padding_size: int,
        prefix: str,
        embedding_method: Qwen4ExpPLEEmbeddingMethod,
        num_ngram_heads: int = 1,
        max_total_tokens: int = 0,
        data_parallel_rank: int = 0,
        reduce_results: bool = True,
    ) -> None:
        if not is_uva_available():
            raise RuntimeError("Engram CPU offload requires UVA support")
        super().__init__(
            num_embeddings,
            embedding_dim,
            params_dtype=params_dtype,
            padding_size=padding_size,
            prefix=prefix,
            embedding_method=embedding_method,
            num_ngram_heads=num_ngram_heads,
            max_total_tokens=max_total_tokens,
            data_parallel_rank=data_parallel_rank,
            reduce_results=reduce_results,
        )
        self._uva_weight = get_accelerator_view_from_cpu_tensor(self.weight)
        self._prefetch_stream: torch.cuda.Stream | None = None
        self._prefetch_buffer: torch.Tensor | None = None
        self._prefetch_alloc_lock = threading.Lock()
        self._prefetch_rows = max_total_tokens * self.etp_data_parallel_size
        self._num_ngram_heads = num_ngram_heads
        self._output_dim = num_ngram_heads * self.embedding_dim

    def allocate_embedding_weight(
        self,
        num_embeddings: int,
        embedding_dim: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Allocate the complete PLE weight directly in pinned CPU memory."""
        table = _registered_host_table(num_embeddings, embedding_dim, dtype)
        if table is not None:
            return table
        return torch.empty(
            num_embeddings,
            embedding_dim,
            dtype=dtype,
            device="cpu",
            pin_memory=True,
        )

    def _lookup(
        self,
        input_ids: torch.Tensor,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Look up local ETP rows in the embedding method's lookup dtype."""
        expected_shape = (*input_ids.shape, self.embedding_dim)
        lookup_dtype = self.embedding_method.lookup_dtype(self)
        if output is None:
            output = torch.empty(
                expected_shape,
                dtype=lookup_dtype,
                device=input_ids.device,
            )
        elif (
            tuple(output.shape) != expected_shape
            or output.dtype != lookup_dtype
            or output.device != input_ids.device
        ):
            raise ValueError(
                "PLE prefetch output must match the input shape, lookup dtype, "
                "and input device"
            )

        flat_ids = input_ids.reshape(-1).long()
        if flat_ids.numel():
            self.embedding_method.lookup_from_pinned(self, flat_ids, output)
        return output

    def sync_lookup(self, ngram_ids: torch.Tensor) -> torch.Tensor:
        """Synchronous UVA lookup for platforms without prefetch wiring."""
        slot_size, slot_offset = self._get_dp_gather_slot(ngram_ids.shape[0])
        gathered_ids = self._gather_dp_ids(ngram_ids, slot_size)
        embeddings = self._lookup(gathered_ids)
        if self.reduce_results:
            embeddings = self._reduce_etp_embeddings(embeddings)
        else:
            # The outer TP reduce-scatter requires ETP=TP.
            assert self.etp_data_parallel_size == 1
        return self._select_embeddings(
            embeddings,
            ngram_ids.shape[0],
            slot_offset,
        )

    def _reduce_etp_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Combine pinned lookup results owned by different ETP ranks."""
        if self.tp_size == 1:
            return embeddings
        assert self.parallel_group is not None
        if embeddings.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            # Each vocabulary row has one owner, so reduce the raw FP8 bytes.
            reduced = self.parallel_group.all_reduce(embeddings.view(torch.int8))
            return reduced.view(embeddings.dtype)
        return self.parallel_group.all_reduce(embeddings)

    @eager_break_during_capture
    def start_prefetch(self, ngram_ids: torch.Tensor) -> None:
        """Gather ETP IDs and launch their UVA lookup on the side stream."""
        buffer = self._prefetch_buffer
        if buffer is None:
            # First use allocates. The eager profile run always precedes
            # cudagraph capture, so allocation never happens mid-capture;
            # the lock keeps concurrent first callers from tearing the
            # stream/buffer pair.
            with self._prefetch_alloc_lock:
                buffer = self._prefetch_buffer
                if buffer is None:
                    if torch.cuda.is_current_stream_capturing():
                        raise RuntimeError(
                            "pinned PLE prefetch buffer must be allocated "
                            "eagerly, before cudagraph capture"
                        )
                    self._prefetch_stream = torch.cuda.Stream(
                        device=self._uva_weight.device
                    )
                    buffer = torch.empty(
                        self._prefetch_rows,
                        self._num_ngram_heads,
                        self.embedding_dim,
                        dtype=self.embedding_method.lookup_dtype(self),
                        device=self._uva_weight.device,
                    )
                    self._prefetch_buffer = buffer
        prefetch_stream = self._prefetch_stream
        if prefetch_stream is None:
            raise RuntimeError("pinned PLE prefetch stream was not allocated")
        slot_size, _ = self._get_dp_gather_slot(ngram_ids.shape[0])
        gathered_ids = self._gather_dp_ids(ngram_ids, slot_size)
        if gathered_ids.shape[0] > buffer.shape[0]:
            raise ValueError(
                f"pinned PLE prefetch buffer holds {buffer.shape[0]} rows, "
                f"but the batch needs {gathered_ids.shape[0]}"
            )
        active_output = buffer[: gathered_ids.shape[0]]
        prefetch_stream.wait_stream(torch.cuda.current_stream())
        gathered_ids.record_stream(prefetch_stream)
        with torch.cuda.stream(prefetch_stream):
            self._lookup(gathered_ids, output=active_output)

    @eager_break_during_capture
    def _finalize_prefetch(
        self,
        prefetch_output: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        """Join the side stream, optionally reduce ETP shards, and select rows."""
        prefetch_stream = self._prefetch_stream
        if prefetch_stream is None:
            raise RuntimeError("pinned PLE finalize requires a prior start_prefetch")
        torch.cuda.current_stream().wait_stream(prefetch_stream)
        slot_size, slot_offset = self._get_dp_gather_slot(output.shape[0])
        embeddings = prefetch_output[: slot_size * self.etp_data_parallel_size]
        if self.reduce_results:
            embeddings = self._reduce_etp_embeddings(embeddings)
        else:
            # The outer TP reduce-scatter requires ETP=TP.
            assert self.etp_data_parallel_size == 1
        embeddings = self._select_embeddings(
            embeddings,
            output.shape[0],
            slot_offset,
        )
        output.copy_(embeddings.flatten(-2))

    def finalize_prefetch(self, total_tokens: int) -> torch.Tensor:
        """Finalize the full-token pinned lookup with optional ETP reduction."""
        buffer = self._prefetch_buffer
        if buffer is None:
            raise RuntimeError("pinned PLE lookup requires a prior start_prefetch")
        # total_tokens counts this DP rank's tokens before SP sharding.
        output = buffer.new_empty((total_tokens, self._output_dim))
        self._finalize_prefetch(buffer, output)
        return output

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        raise RuntimeError(
            "Pinned-host PLE embedding requires "
            "start_prefetch() and finalize_prefetch()."
        )

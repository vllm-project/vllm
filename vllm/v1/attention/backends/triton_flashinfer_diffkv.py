# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DiffKV attention routing prefill to FlashInfer and the rest to Triton.

Prefill steps of plain causal full-attention layers run FlashInfer's paged
prefill (``head_dim_vo != head_dim_qk``) directly on the Triton DiffKV packed
cache; decode, spec-decode verification and CUDA-graph batches stay on
``TRITON_ATTN_DIFFKV``.
"""

from dataclasses import dataclass
from typing import Any, ClassVar, cast

import numpy as np
import torch

from vllm import envs
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.utils.flashinfer import has_flashinfer
from vllm.utils.gpu_sync_debug import gpu_sync_allowed
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import PIN_MEMORY, is_quantized_kv_cache
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    max_decode_query_len,
)
from vllm.v1.attention.backends.composite import (
    create_composite_attention_backend,
    requires_mm_prefix,
)
from vllm.v1.attention.backends.triton_attn_diffkv import (
    TritonAttentionDiffKVBackend,
    TritonAttentionDiffKVImpl,
)
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheSpec
from vllm.v1.utils import CpuGpuBuffer

logger = init_logger(__name__)

# Short extends stay on Triton; FlashInfer's plan() costs more than it saves.
MIN_FLASHINFER_QUERY_LEN = 16

_WORKSPACE_BYTES_PER_ELEM = 16

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)
_SUPPORTED_KV_CACHE_DTYPES = ("auto", "float16", "bfloat16")
# Covered by tests/kernels/attention/test_flashinfer_diffkv_prefill.py.
_VALIDATED_HEAD_SIZES = ((192, 128),)


def plan_paged_prefill(
    wrapper: Any,
    *,
    qo_indptr_cpu: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    block_table: torch.Tensor,
    page_size: int,
    paged_kv_indptr: CpuGpuBuffer,
    paged_kv_indices: torch.Tensor,
    paged_kv_last_page_len: CpuGpuBuffer,
    num_qo_heads: int,
    num_kv_heads: int,
    head_size_qk: int,
    head_size_v: int,
    sm_scale: float,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
) -> None:
    """Plan a causal paged prefill; free of builder state so tests can call it."""
    from vllm.v1.attention.backends.flashinfer import _copy_page_indices_kernel

    seq_lens_np = seq_lens_cpu.numpy()
    num_reqs = seq_lens_np.shape[0]
    num_pages = (seq_lens_np + page_size - 1) // page_size
    np.cumsum(num_pages, dtype=np.int32, out=paged_kv_indptr.np[1 : num_reqs + 1])
    indptr_gpu = paged_kv_indptr.copy_to_gpu(num_reqs + 1)
    indices = paged_kv_indices[: int(paged_kv_indptr.np[num_reqs])]
    _copy_page_indices_kernel[(num_reqs,)](
        indices,
        block_table,
        block_table.stride(0),
        indptr_gpu,
        BLOCK_SIZE=1024,
    )
    last = seq_lens_np % page_size
    paged_kv_last_page_len.np[:num_reqs] = np.where(
        (last == 0) & (seq_lens_np != 0), page_size, last
    )
    kv_indptr_cpu = paged_kv_indptr.cpu[: num_reqs + 1]
    last_page_len_cpu = paged_kv_last_page_len.cpu[:num_reqs]
    seq_lens_cpu = seq_lens_cpu.to(torch.int32)
    # plan() copies these to the GPU with non_blocking=True; stage them in
    # pinned memory so the copies stay async, as FlashInferMetadataBuilder does.
    if PIN_MEMORY:
        qo_indptr_cpu = qo_indptr_cpu.pin_memory()
        kv_indptr_cpu = kv_indptr_cpu.pin_memory()
        last_page_len_cpu = last_page_len_cpu.pin_memory()
        seq_lens_cpu = seq_lens_cpu.pin_memory()
    wrapper.plan(
        qo_indptr_cpu,
        kv_indptr_cpu,
        indices,
        last_page_len_cpu,
        num_qo_heads,
        num_kv_heads,
        head_size_qk,
        page_size,
        head_dim_vo=head_size_v,
        causal=True,
        sm_scale=sm_scale,
        q_data_type=q_dtype,
        kv_data_type=kv_dtype,
        seq_lens=seq_lens_cpu,
    )


def host_seq_lens(
    common_attn_metadata: CommonAttentionMetadata, max_decode_query_len: int
) -> torch.Tensor:
    """Exact per-request KV lengths on the host.

    ``seq_lens_cpu_upper_bound`` is exact for prefill rows and, without
    speculative decoding, for every row. Only batches that may contain
    spec-decode rows (query length <= ``max_decode_query_len``) read the exact
    lengths back from the GPU.
    """
    cm = common_attn_metadata
    num_reqs = cm.num_reqs
    upper = cm.seq_lens_cpu_upper_bound
    if upper is not None:
        query_lens = (
            cm.query_start_loc_cpu[1 : num_reqs + 1]
            - (cm.query_start_loc_cpu[:num_reqs])
        )
        if max_decode_query_len == 1 or bool((query_lens > max_decode_query_len).all()):
            return upper[:num_reqs]
    with gpu_sync_allowed():
        return cm.seq_lens[:num_reqs].cpu()


def supports_flashinfer_prefill(impl: Any) -> bool:
    return (
        impl.sliding_window in (None, (-1, -1))
        and impl.sinks is None
        and impl.alibi_slopes is None
        and not impl.logits_soft_cap
        and not is_quantized_kv_cache(impl.kv_cache_dtype)
    )


def run_paged_prefill(
    wrapper: Any,
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    output: torch.Tensor,
    head_size_qk: int,
    head_size_v: int,
) -> None:
    """Run on the logical (B, H, N, hqk + hv) cache; K/V are strided views."""
    wrapper.run(
        query,
        (
            kv_cache[..., :head_size_qk],
            kv_cache[..., head_size_qk : head_size_qk + head_size_v],
        ),
        out=output,
    )


@dataclass
class FlashInferDiffKVMetadata:
    num_actual_tokens: int
    prefill_wrapper: Any


class FlashInferDiffKVMetadataBuilder(
    AttentionMetadataBuilder[FlashInferDiffKVMetadata]
):
    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        from flashinfer import BatchPrefillWithPagedKVCacheWrapper

        self._wrapper_cls = BatchPrefillWithPagedKVCacheWrapper
        self._wrapper = None
        self._workspace_buffer: torch.Tensor | None = None

        impls = [
            cast(Any, getattr(layer, "impl", layer))
            for layer in get_layers_from_vllm_config(
                vllm_config,
                AttentionLayerBase,  # type: ignore[type-abstract]
                layer_names,
            ).values()
        ]
        shapes = {(i.num_heads, i.num_kv_heads, i.head_size, i.scale) for i in impls}
        assert len(shapes) == 1, f"non-uniform attention layers in group: {shapes}"
        self.num_qo_heads, self.num_kv_heads, self.head_size_qk, self.scale = (
            shapes.pop()
        )
        self.head_size_v = TritonAttentionDiffKVBackend.head_size_v
        self.q_dtype = vllm_config.model_config.dtype
        self.kv_dtype = kv_cache_spec.dtype
        self.kernel_block_size = kv_cache_spec.block_size
        self.max_decode_query_len = max_decode_query_len(vllm_config)

        scheduler_config = vllm_config.scheduler_config
        self.max_num_batched_tokens = scheduler_config.max_num_batched_tokens
        max_num_reqs = scheduler_config.max_num_seqs
        max_num_pages = max_num_reqs * cdiv(
            vllm_config.model_config.max_model_len, self.kernel_block_size
        )
        self.paged_kv_indptr = CpuGpuBuffer(
            max_num_reqs + 1, dtype=torch.int32, device=device, pin_memory=False
        )
        self.paged_kv_indices = torch.zeros(
            max_num_pages, dtype=torch.int32, device=device
        )
        self.paged_kv_last_page_len = CpuGpuBuffer(
            max_num_reqs, dtype=torch.int32, device=device, pin_memory=False
        )

    @classmethod
    def get_cudagraph_support(
        cls, vllm_config: VllmConfig, kv_cache_spec: KVCacheSpec
    ) -> AttentionCGSupport:
        # Never captured; DiffKVPrefillRouting bounds the composite's support.
        return AttentionCGSupport.ALWAYS

    def _get_workspace_buffer(self) -> torch.Tensor:
        if self._workspace_buffer is None:
            size = max(
                envs.VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE,
                self.max_num_batched_tokens
                * self.num_qo_heads
                * max(self.head_size_qk, self.head_size_v)
                * _WORKSPACE_BYTES_PER_ELEM,
            )
            self._workspace_buffer = torch.zeros(
                size, dtype=torch.uint8, device=self.device
            )
        return self._workspace_buffer

    def set_workspace_buffer(self, workspace_buffer: torch.Tensor) -> None:
        self._workspace_buffer = workspace_buffer
        self._wrapper = None

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> FlashInferDiffKVMetadata:
        cm = common_attn_metadata
        num_reqs = cm.num_reqs
        if self._wrapper is None:
            self._wrapper = self._wrapper_cls(self._get_workspace_buffer(), "HND")
        qo_indptr_cpu = cm.query_start_loc_cpu[: num_reqs + 1]
        plan_paged_prefill(
            self._wrapper,
            qo_indptr_cpu=qo_indptr_cpu,
            seq_lens_cpu=host_seq_lens(cm, self.max_decode_query_len),
            block_table=cm.block_table_tensor,
            page_size=self.kernel_block_size,
            paged_kv_indptr=self.paged_kv_indptr,
            paged_kv_indices=self.paged_kv_indices,
            paged_kv_last_page_len=self.paged_kv_last_page_len,
            num_qo_heads=self.num_qo_heads,
            num_kv_heads=self.num_kv_heads,
            head_size_qk=self.head_size_qk,
            head_size_v=self.head_size_v,
            sm_scale=self.scale,
            q_dtype=self.q_dtype,
            kv_dtype=self.kv_dtype,
        )
        return FlashInferDiffKVMetadata(
            num_actual_tokens=int(qo_indptr_cpu[num_reqs]),
            prefill_wrapper=self._wrapper,
        )


class FlashInferDiffKVImpl(TritonAttentionDiffKVImpl):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.flashinfer_supported = supports_flashinfer_prefill(self)

    def forward(  # type: ignore[override]
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashInferDiffKVMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not supported for FlashInferDiffKVImpl"
            )
        if attn_metadata is None:
            return output.fill_(0)
        assert self.flashinfer_supported
        n = attn_metadata.num_actual_tokens
        run_paged_prefill(
            attn_metadata.prefill_wrapper,
            query[:n],
            kv_cache,
            output[:n],
            self.head_size,
            TritonAttentionDiffKVBackend.head_size_v,
        )
        return output


class FlashInferDiffKVBackend(TritonAttentionDiffKVBackend):
    """Prefill child of ``TRITON_FLASHINFER_DIFFKV``; shares the Triton cache.

    Per-layer eligibility is enforced by ``DiffKVPrefillRouting``.
    """

    @staticmethod
    def get_name() -> str:
        return "FLASHINFER_DIFFKV_PREFILL"

    @staticmethod
    def get_impl_cls() -> type[FlashInferDiffKVImpl]:
        return FlashInferDiffKVImpl

    @staticmethod
    def get_builder_cls() -> type[FlashInferDiffKVMetadataBuilder]:  # type: ignore[override]
        return FlashInferDiffKVMetadataBuilder

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major == 12

    @classmethod
    def supports_mm_prefix(cls) -> bool:
        return False


class DiffKVPrefillRouting:
    """FlashInfer (1) for whole batches containing a prefill, Triton (0) otherwise.

    Decode-only and verification-only batches stay on Triton; decode rows in a
    mixed batch run on FlashInfer with it.
    """

    capture_variant: ClassVar[int] = 0

    def __init__(self, layer_names: list[str], vllm_config: VllmConfig) -> None:
        layers = get_layers_from_vllm_config(
            vllm_config,
            AttentionLayerBase,  # type: ignore[type-abstract]
            layer_names,
        )
        self.enabled = bool(layers) and all(
            getattr(
                getattr(layer.impl, "causal_impl", None), "flashinfer_supported", False
            )
            for layer in layers.values()
        )
        if not self.enabled:
            logger.info_once(
                "TRITON_FLASHINFER_DIFFKV: a layer in %s needs a feature the "
                "FlashInfer prefill path does not cover; using Triton for it.",
                layer_names[0] if layer_names else "<empty group>",
            )
        self.min_query_len = self.flashinfer_min_query_len(vllm_config)

    @staticmethod
    def flashinfer_min_query_len(vllm_config: VllmConfig) -> int:
        # Uniform (graph-capturable) batches are never wider than this.
        return max(MIN_FLASHINFER_QUERY_LEN, max_decode_query_len(vllm_config))

    def select(self, metadata: CommonAttentionMetadata) -> int:
        return int(
            self.enabled
            and metadata.max_query_len > self.min_query_len
            and not requires_mm_prefix(metadata)
        )

    @staticmethod
    def get_cudagraph_support(
        vllm_config: VllmConfig, kv_cache_spec: KVCacheSpec
    ) -> AttentionCGSupport:
        return AttentionCGSupport.UNIFORM_BATCH

    @staticmethod
    def variant_uses_mm_prefix(variant: int) -> bool:
        return variant == 0


_TritonFlashInferDiffKV = create_composite_attention_backend(
    TritonAttentionDiffKVBackend,
    FlashInferDiffKVBackend,
    name="_TritonFlashInferDiffKV",
    backend_name="TRITON_FLASHINFER_DIFFKV",
    module=__name__,
    routing_policy=DiffKVPrefillRouting,
    device_major=12,
)


class TritonFlashInferDiffKVBackend(_TritonFlashInferDiffKV):  # type: ignore[valid-type,misc]
    @classmethod
    def set_head_size_v(cls, head_size_v: int) -> None:
        TritonAttentionDiffKVBackend.set_head_size_v(head_size_v)

    @classmethod
    def is_supported_on_current_device(
        cls,
        head_size: int,
        head_size_v: int,
        has_sinks: bool,
        sliding_window: int | None,
        dtype: torch.dtype,
        kv_cache_dtype: str,
    ) -> bool:
        """Auto-selection check; sm_90/sm_100 use FA DiffKV instead."""
        if not envs.VLLM_DIFFKV_FLASHINFER_PREFILL:
            return False
        if not current_platform.is_cuda() or not has_flashinfer():
            return False
        capability = current_platform.get_device_capability()
        if capability is None or not cls.supports_compute_capability(capability):
            return False
        return (
            not has_sinks
            and sliding_window is None
            and (head_size, head_size_v) in _VALIDATED_HEAD_SIZES
            and dtype in _SUPPORTED_DTYPES
            and kv_cache_dtype in _SUPPORTED_KV_CACHE_DTYPES
        )

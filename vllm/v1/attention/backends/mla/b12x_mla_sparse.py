# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B12x sparse MLA attention backend."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

import torch

from vllm import _custom_ops as ops
from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.distributed import get_dcp_group
from vllm.model_executor.layers.attention.mla_attention import MLACommonPrefillMetadata
from vllm.model_executor.layers.attention.sparse_mla_attention import (
    SparseMLACommonImpl,
    SparseMLACommonMetadataBuilder,
)
from vllm.platforms.interface import DeviceCapability
from vllm.utils.b12x import get_b12x_sparse_mla
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionLayer,
    AttentionMetadata,
    AttentionType,
    CommonAttentionMetadata,
    MLAAttentionImpl,
    MultipleOf,
)
from vllm.v1.attention.backends.utils import get_dcp_local_seq_lens
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.v1.kv_cache_layout import KVCacheLayout
from vllm.v1.worker.workspace import current_workspace_manager

if TYPE_CHECKING:
    from vllm.model_executor.models.deepseek_v2 import Indexer


class B12xMLASparseBackend(AttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "fp8",
        "fp8_e4m3",
        "fp8_ds_mla",
    ]

    @staticmethod
    def get_name() -> str:
        return "B12X"

    @staticmethod
    def get_impl_cls() -> type[MLAAttentionImpl]:
        return B12xMLASparseImpl

    @staticmethod
    def get_builder_cls() -> type["B12xMLASparseMetadataBuilder"]:
        return B12xMLASparseMetadataBuilder

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [576]

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...]:
        return (KVCacheLayout.BLHNC,)

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [64]

    @classmethod
    def is_mla(cls) -> bool:
        return True

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @classmethod
    def supports_device_cpu_query_lens_mismatch(cls) -> bool:
        return False

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return (capability.major, capability.minor) in ((12, 0), (12, 1))

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: CacheDType | None,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        use_mm_prefix: bool,
        device_capability: DeviceCapability,
    ) -> str | None:
        from vllm.config import get_current_vllm_config

        module = get_b12x_sparse_mla()
        if module is None:
            return "B12X sparse MLA requires the optional b12x package"
        if not module.is_supported():
            return "B12X sparse MLA is not supported on the current device"
        vllm_config = get_current_vllm_config()
        if vllm_config.model_config is not None:
            hf_config = vllm_config.model_config.hf_text_config
            if getattr(hf_config, "index_topk", None) is None:
                return "B12X sparse MLA requires a model with index_topk"
            if int(getattr(hf_config, "kv_lora_rank", 0)) != 512:
                return "B12X sparse MLA requires kv_lora_rank=512"
            if int(getattr(hf_config, "qk_rope_head_dim", 0)) != 64:
                return "B12X sparse MLA requires qk_rope_head_dim=64"
        return None


@dataclass
class B12xMLASparseMetadata(AttentionMetadata):
    num_reqs: int
    max_query_len: int
    max_seq_len: int
    num_actual_tokens: int
    query_start_loc: torch.Tensor
    slot_mapping: torch.Tensor
    block_table: torch.Tensor
    req_id_per_token: torch.Tensor
    seq_lens: torch.Tensor
    cache_seq_lens_per_token: torch.Tensor = field(init=False)
    num_decodes: int
    num_prefills: int
    num_decode_tokens: int
    prefill_max_seq_len: int = 0
    prefill: MLACommonPrefillMetadata | None = None
    block_size: int = 64
    topk_tokens: int = 2048
    cp_kv_cache_interleave_size: int = 1


def _write_rank_local_selected_lengths(
    global_lengths: torch.Tensor,
    output: torch.Tensor,
    *,
    dcp_size: int,
    dcp_rank: int,
    interleave_size: int,
    topk: int,
) -> None:
    lengths = global_lengths
    if dcp_size > 1:
        lengths = get_dcp_local_seq_lens(
            global_lengths,
            dcp_size,
            dcp_rank,
            interleave_size,
        )
    torch.clamp(lengths, min=0, max=topk, out=output)


class B12xMLASparseMetadataBuilder(
    SparseMLACommonMetadataBuilder[B12xMLASparseMetadata]
):
    metadata_cls = B12xMLASparseMetadata
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.dcp_rank = get_dcp_group().rank_in_group if self.dcp_world_size > 1 else 0
        num_q_heads = vllm_config.model_config.get_num_attention_heads(
            vllm_config.parallel_config
        )
        threshold = {8: 128, 16: 128, 32: 128, 64: 256, 128: 1024}.get(
            num_q_heads, 1024
        )
        self._init_reorder_batch_threshold(
            threshold,
            supports_spec_as_decode=True,
            supports_dcp_with_varlen=True,
        )
        self.cache_seq_lens_per_token_buffer = torch.empty(
            (vllm_config.scheduler_config.max_num_batched_tokens,),
            dtype=torch.int32,
            device=device,
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> B12xMLASparseMetadata:
        metadata = super().build(
            common_prefix_len,
            common_attn_metadata,
            fast_build,
        )
        num_tokens = metadata.num_actual_tokens
        positions = common_attn_metadata.positions
        if positions is None:
            raise RuntimeError("B12X sparse MLA metadata requires token positions.")
        positions = positions[:num_tokens]
        cache_seq_lens_per_token = self.cache_seq_lens_per_token_buffer[:num_tokens]
        cache_seq_lens_per_token.copy_(positions, non_blocking=True)
        cache_seq_lens_per_token.add_(1)
        _write_rank_local_selected_lengths(
            cache_seq_lens_per_token,
            cache_seq_lens_per_token,
            dcp_size=self.dcp_world_size,
            dcp_rank=self.dcp_rank,
            interleave_size=self.cp_kv_cache_interleave_size,
            topk=self.topk_tokens,
        )
        metadata.cache_seq_lens_per_token = cache_seq_lens_per_token
        return metadata


class B12xMLASparseImpl(SparseMLACommonImpl[B12xMLASparseMetadata]):
    can_return_lse_for_decode = True
    lse_base_on_e = True
    supports_dense_mha_prefill = False
    supports_pcp = False

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        topk_indices_buffer: torch.Tensor | None = None,
        indexer: "Indexer | None" = None,
        **mla_args,
    ) -> None:
        if any((alibi_slopes, sliding_window, logits_soft_cap)):
            raise NotImplementedError(
                "B12X sparse MLA does not support ALiBi, sliding window, or "
                "logit soft caps."
            )
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "B12X sparse MLA supports decoder self-attention only."
            )

        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            indexer=indexer,
            topk_indices_buffer=topk_indices_buffer,
            **mla_args,
        )
        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
        if self.kv_lora_rank != 512 or self.qk_rope_head_dim != 64:
            raise ValueError(
                "B12X sparse MLA requires kv_lora_rank=512 and qk_rope_head_dim=64."
            )
        if head_size != 576:
            raise ValueError("B12X sparse MLA requires head_size=576.")
        if self.topk_indices_buffer is None:
            raise ValueError("B12X sparse MLA requires a top-k index buffer.")
        if kv_cache_dtype != "fp8_ds_mla":
            raise ValueError(
                "B12X sparse MLA requires the packed fp8_ds_mla KV cache; "
                f"got kv_cache_dtype={kv_cache_dtype!r}."
            )

        module = get_b12x_sparse_mla()
        if module is None:
            raise RuntimeError("B12X sparse MLA requires `pip install vllm[b12x]`.")
        if not module.is_supported():
            raise RuntimeError("B12X sparse MLA is not supported on this device.")
        for name in ("Caps", "bind", "plan", "run"):
            getattr(module, name)
        self._bind = module.bind
        self._run = module.run
        self._module = module

        scheduler_config = vllm_config.scheduler_config
        max_tokens = int(scheduler_config.max_num_batched_tokens)
        max_seqs = int(scheduler_config.max_num_seqs)
        self._input_num_heads = self.num_heads * self.dcp_world_size
        self._q_head_dim = self.kv_lora_rank + self.qk_rope_head_dim
        self._topk_tokens = int(self.topk_indices_buffer.shape[-1])
        self._max_tokens = max_tokens
        self._max_seqs = max_seqs
        self._kv_dtype = torch.uint8
        self._kernel_page_size = 0
        self.supports_quant_query_input = False

    def _set_kernel_page_size(self, kernel_page_size: int) -> None:
        if kernel_page_size <= 0 or kernel_page_size % 64:
            raise ValueError(
                "B12X sparse MLA kernel page size must be a positive multiple "
                f"of 64, got {kernel_page_size}."
            )
        if kernel_page_size == self._kernel_page_size:
            return

        def make_plan(mode: str, max_q_rows: int, max_batch: int):
            return self._module.plan(
                self._module.Caps(
                    device=torch.device(
                        "cuda", torch.accelerator.current_device_index()
                    ),
                    num_q_heads=self._input_num_heads,
                    max_q_rows=max_q_rows,
                    max_width=self._topk_tokens,
                    softmax_scale=self.scale,
                    dtype=torch.bfloat16,
                    kv_dtype=self._kv_dtype,
                    head_dim=self._q_head_dim,
                    v_head_dim=self.kv_lora_rank,
                    mode=mode,
                    max_batch=max_batch,
                    max_chunks_per_row=max(1, (self._topk_tokens + 63) // 64),
                    page_size=kernel_page_size,
                    return_lse=self.need_to_return_lse_for_decode,
                    lse_scale="natural",
                )
            )

        self._decode_plan = make_plan("decode", self._max_seqs, self._max_seqs)
        self._extend_plan = make_plan("extend", self._max_tokens, self._max_seqs)
        self._scratch_nbytes = max(
            int(self._decode_plan.layout.nbytes),
            int(self._extend_plan.layout.nbytes),
        )
        self._q_spec = (
            (self._max_tokens, self._input_num_heads, self._q_head_dim),
            torch.bfloat16,
        )
        self._scratch_spec = ((self._scratch_nbytes,), torch.uint8)
        self._device = self._decode_plan.caps.device
        self._kernel_page_size = kernel_page_size

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        if kv_cache.numel() == 0:
            return
        if kv_cache.ndim < 2:
            raise ValueError(
                "B12X sparse MLA cache must expose its page dimension, got "
                f"shape={tuple(kv_cache.shape)}."
            )
        self._set_kernel_page_size(int(kv_cache.shape[1]))

    def _borrow_workspaces(self) -> tuple[torch.Tensor, torch.Tensor]:
        q_buffer, scratch = current_workspace_manager().get_simultaneous(
            self._q_spec, self._scratch_spec
        )
        return q_buffer, scratch

    def supports_fused_mla_query_output(
        self,
        num_heads: int,
        output_dtype: torch.dtype,
    ) -> bool:
        return bool(
            self.dcp_world_size == 1
            and output_dtype == torch.bfloat16
            and num_heads == self._input_num_heads
            and self._q_head_dim == 576
        )

    def get_fused_mla_query_output(
        self,
        num_tokens: int,
        num_heads: int,
        output_dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if (
            not self.supports_fused_mla_query_output(num_heads, output_dtype)
            or num_tokens <= 0
            or num_tokens > self._max_tokens
        ):
            return None
        q_buffer, _ = self._borrow_workspaces()
        output = q_buffer[:num_tokens, :num_heads]
        if not output.is_contiguous():
            raise RuntimeError("B12X fused MLA query output must be contiguous.")
        return output

    def b12x_warmup_key(self) -> tuple[object, ...]:
        return (
            type(self),
            self._device,
            self._input_num_heads,
            self._q_head_dim,
            self._topk_tokens,
            self._max_tokens,
            self._decode_plan.caps.max_q_rows,
            self.need_to_return_lse_for_decode,
            self._kernel_page_size,
        )

    def warmup(self, token_counts: tuple[int, ...]) -> None:
        decode_capacity = int(self._decode_plan.caps.max_q_rows)
        decode_rows = {
            int(rows) for rows in token_counts if 0 < int(rows) <= decode_capacity
        }
        decode_rows.add(1)
        extend_rows = {1, 2, 4, self._max_tokens}
        kv_cache = torch.zeros(
            (1, self._kernel_page_size, 656),
            dtype=torch.uint8,
            device=self._device,
        )
        q_buffer, scratch = self._borrow_workspaces()

        for plan, rows_to_warm in (
            (self._decode_plan, sorted(decode_rows)),
            (self._extend_plan, sorted(extend_rows)),
        ):
            for rows in rows_to_warm:
                if rows > int(plan.caps.max_q_rows):
                    continue
                q = q_buffer[:rows]
                q.zero_()
                selected_indices = torch.zeros(
                    (rows, self._topk_tokens),
                    dtype=torch.int32,
                    device=self._device,
                )
                cache_lengths = torch.full(
                    (rows if plan is self._decode_plan else 1,),
                    self._kernel_page_size,
                    dtype=torch.int32,
                    device=self._device,
                )
                selected_lengths = torch.ones(
                    (rows,), dtype=torch.int32, device=self._device
                )
                binding = self._bind(
                    plan,
                    scratch=scratch,
                    q=q,
                    kv_cache=kv_cache,
                    selected_indices=selected_indices,
                    cache_lengths=cache_lengths,
                    selected_lengths=selected_lengths,
                )
                self._run(binding)

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: B12xMLASparseMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        del layer
        cache_page_size = int(kv_c_and_k_pe_cache.shape[1])
        metadata_page_size = int(attn_metadata.block_size)
        if (
            cache_page_size != self._kernel_page_size
            or metadata_page_size != self._kernel_page_size
        ):
            raise RuntimeError(
                "B12X sparse MLA page geometry does not match the bound plan: "
                f"cache={cache_page_size}, metadata={metadata_page_size}, "
                f"plan={self._kernel_page_size}."
            )
        plan = (
            self._decode_plan if attn_metadata.max_query_len <= 1 else self._extend_plan
        )
        q_buffer, scratch = self._borrow_workspaces()

        if isinstance(q, tuple):
            q_nope, q_pe = q
            num_tokens = int(q_nope.shape[0])
            q_all = q_buffer[:num_tokens]
            if int(q_pe.shape[-1]) == 0:
                q_all.copy_(q_nope)
            else:
                ops.concat_mla_q(q_nope, q_pe, q_all)
        else:
            num_tokens = int(q.shape[0])
            q_all = q_buffer[:num_tokens]
            exact_workspace_alias = (
                tuple(q.shape) == tuple(q_all.shape)
                and tuple(q.stride()) == tuple(q_all.stride())
                and q.dtype == q_all.dtype
                and q.device == q_all.device
                and q.untyped_storage().data_ptr() == q_all.untyped_storage().data_ptr()
                and q.storage_offset() == q_all.storage_offset()
            )
            if not exact_workspace_alias:
                q_all.copy_(q)

        if int(q_all.shape[1]) != self._input_num_heads:
            raise ValueError(
                "B12X sparse MLA query heads do not match the planned head "
                f"count: {q_all.shape[1]} != {self._input_num_heads}."
            )

        assert self.topk_indices_buffer is not None
        selected_indices = self.topk_indices_buffer[:num_tokens]
        active_counts = attn_metadata.cache_seq_lens_per_token[:num_tokens]

        # Cache lengths are request-sized; active counts are query-row-sized.
        cache_seq_lens = attn_metadata.seq_lens[: attn_metadata.num_reqs].contiguous()
        binding = self._bind(
            plan,
            scratch=scratch,
            q=q_all,
            kv_cache=kv_c_and_k_pe_cache,
            selected_indices=selected_indices,
            cache_lengths=cache_seq_lens,
            selected_lengths=active_counts,
        )
        result = self._run(binding)
        if self.need_to_return_lse_for_decode:
            output, lse = result
            return output, lse
        assert isinstance(result, torch.Tensor)
        return result, None

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 attention: relative-bias GQA with short-conv K/V, L2-clipped
q/k/r, gated output and per-token attention sinks."""

from dataclasses import dataclass, field
from functools import partial
from typing import cast

import numpy as np
import torch
from torch import nn

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.models.inkling.nvidia.ops.fa4_rel_attention import bucket_max_seqlen_q
from vllm.models.inkling.nvidia.sconv_swa_attn import InklingConvState
from vllm.platforms import current_platform
from vllm.utils.torch_utils import (
    canonicalize_singleton_dim_strides,
    kv_cache_dtype_str_to_dtype,
)
from vllm.v1.attention.backend import (
    AttentionBackend,
    CommonAttentionMetadata,
    max_decode_query_len,
)
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionMetadata,
    FlashAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.core.single_type_kv_cache_manager import (
    FullAttentionManager,
    SlidingWindowManager,
)
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheLayerConfig

from ..configs import Inkling2ModelConfig
from .lora import add_linear_lora
from .ops import use_sm100_kernels
from .ops.attention import (
    _INKLING2_FA4_REL_ATTENTION_KERNEL,
    gate_sink,
    plan_attention,
    plan_decode_attention,
    rel_decode_attention,
)
from .ops.attention_sm90 import MODE_OUT_LSE, rel_attention
from .ops.linear import linear
from .ops.sconv import attention_prologue, shift_rel_proj
from .sconv import K_STREAM, V_STREAM, Inkling2ShortConv, conv_metadata


@dataclass
class Inkling2AttentionMetadata(FlashAttentionMetadata):
    # Host copies for planning FA4 splits (exact for prefill requests).
    query_lens_cpu: np.ndarray = field(default_factory=lambda: np.empty(0, np.int64))
    seq_lens_cpu: np.ndarray = field(default_factory=lambda: np.empty(0, np.int64))


class Inkling2AttentionMetadataBuilder(FlashAttentionMetadataBuilder):
    """Orders decode/verify requests first and records the split, so that a
    mixed batch runs its decode part on `rel_decode_attention`."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=True)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> Inkling2AttentionMetadata:
        cm = common_attn_metadata
        assert cm.seq_lens_cpu_upper_bound is not None
        md = Inkling2AttentionMetadata(
            **vars(super().build(common_prefix_len, cm, fast_build)),
            query_lens_cpu=cm.query_start_loc_cpu.diff().numpy(),
            seq_lens_cpu=cm.seq_lens_cpu_upper_bound.numpy(),
        )
        assert self.reorder_batch_threshold is not None
        (
            md.num_decode_reqs,
            md.num_prefill_reqs,
            md.num_decode_tokens,
            md.num_prefill_tokens,
        ) = split_decodes_and_prefills(
            common_attn_metadata, decode_threshold=self.reorder_batch_threshold
        )
        return md


class Inkling2AttentionBackend(FlashAttentionBackend):
    @staticmethod
    def get_builder_cls() -> type[FlashAttentionMetadataBuilder]:
        return Inkling2AttentionMetadataBuilder


def _kv_cache_layer_config(
    vllm_config: VllmConfig,
    num_kv_heads: int,
    head_dim: int,
    sliding_window: int | None,
) -> KVCacheLayerConfig:
    spec = AttentionSpec(
        block_size=vllm_config.cache_config.block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_dim,
        dtype=kv_cache_dtype_str_to_dtype(
            vllm_config.cache_config.cache_dtype, vllm_config.model_config
        ),
    )
    if sliding_window is not None:
        return SlidingWindowManager.layer_config(spec, sliding_window=sliding_window)
    return FullAttentionManager.layer_config(spec)


class Inkling2AttentionPassCache(nn.Module, AttentionLayerBase):
    """The KV cache of one extra loop pass of a looped `Inkling2Attention`
    (each pass of the loop core attends over its own keys and values)."""

    def __init__(
        self,
        prefix: str,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        sliding_window: int | None,
    ) -> None:
        super().__init__()
        self.prefix = prefix
        # Read by the metadata builders of the layer's attention group.
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.sliding_window = sliding_window
        self.kv_cache = torch.tensor([])
        forward_context = get_current_vllm_config().compilation_config
        if prefix in forward_context.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        forward_context.static_forward_context[prefix] = self

    def get_attn_backend(self) -> type[AttentionBackend]:
        return Inkling2AttentionBackend

    def get_kv_cache_config(self, vllm_config: VllmConfig) -> KVCacheLayerConfig:
        return _kv_cache_layer_config(
            vllm_config, self.num_kv_heads, self.head_dim, self.sliding_window
        )


class Inkling2Attention(nn.Module, AttentionLayerBase):
    """Inkling2 attention for one decoder layer.

    Local layers project all of q/k/v from one input; full layers project the
    query side (q, r, gate, sink) from `x_q`, k from `x_k` and v from
    `x_v`, and use dynamic K/V convolutions and log-scaled q/r. Local layers
    given separate `x_k` / `x_v` (loop passes after the first) project them
    as full layers do.

    A loop core layer runs once per loop pass, each pass with its own KV and
    conv-state caches: `conv_states[p]` and pass `p`'s KV cache (this layer's
    for pass 0, `pass_caches[p - 1]` after). With `loop_share_cache`, every
    pass reads and writes this layer's KV cache (later tokens see the last
    pass's K/V).
    """

    def __init__(
        self,
        config: Inkling2ModelConfig,
        is_local: bool,
        conv_state: InklingConvState,
        prefix: str,
        pass_conv_states: tuple[InklingConvState, ...] = (),
    ) -> None:
        super().__init__()
        self.prefix = prefix
        self.is_local = is_local
        num_heads = (
            config.swa_num_attention_heads if is_local else config.num_attention_heads
        )
        num_kv_heads = (
            config.swa_num_key_value_heads if is_local else config.num_key_value_heads
        )
        self.head_dim = config.swa_head_dim if is_local else config.head_dim
        self.d_rel = config.d_rel
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        assert num_heads % tp_size == 0 and num_kv_heads % tp_size == 0
        self.num_heads = num_heads // tp_size
        self.num_kv_heads = num_kv_heads // tp_size
        self.scaling = self.head_dim**-0.5
        self.sliding_window = config.sliding_window_size
        self.rel_extent = config.sliding_window_size if is_local else config.rel_extent
        self.window_size = (config.sliding_window_size - 1, 0) if is_local else (-1, -1)
        self.qk_clip = config.qk_l2_clip_threshold
        self.rel_clip = config.rel_l2_clip_threshold
        self.eps = config.rms_norm_eps
        assert config.use_value_norm_no_gamma

        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        # Query-side outputs first so full layers can project them in one GEMM.
        self.split_sizes = [
            q_size,
            self.num_heads * self.d_rel,
            q_size,
            self.num_heads,
            kv_size,
            kv_size,
        ]
        # Full layers project the query side alone; rounding its N up to 8
        # keeps the output rows 16-byte aligned for cuBLAS (the extra columns,
        # from the K rows, are discarded).
        q_side = sum(self.split_sizes[:4])
        self.q_side_splits = [*self.split_sizes[:4], -q_side % 8]
        pad_size = -sum(self.split_sizes) % 8
        if pad_size:
            self.split_sizes.append(pad_size)
        self.qkvr = MergedColumnParallelLinear(
            config.hidden_size,
            [size * tp_size for size in self.split_sizes],
            bias=False,
            prefix=f"{prefix}.qkvr",
        )
        self.wo_ud = RowParallelLinear(
            q_size * tp_size,
            config.hidden_size,
            bias=False,
            reduce_results=False,
            prefix=f"{prefix}.wo_ud",
        )
        self.rel_logits_proj = nn.Parameter(
            torch.empty(self.d_rel, self.rel_extent), requires_grad=False
        )
        self.q_scale = nn.Parameter(
            torch.empty(self.num_heads, self.head_dim), requires_grad=False
        )
        self.r_scale = nn.Parameter(
            torch.empty(self.num_heads, self.d_rel), requires_grad=False
        )
        for param in (self.q_scale, self.r_scale):
            set_weight_attrs(
                param,
                {
                    "weight_loader": lambda p, w: p.data.copy_(
                        w.narrow(0, tp_rank * p.shape[0], p.shape[0])
                    )
                },
            )

        self.conv_state = conv_state
        self.conv_states = (conv_state, *pass_conv_states)
        self.pass_caches = nn.ModuleList(
            Inkling2AttentionPassCache(
                f"{prefix}.pass{p}",
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                self.sliding_window if is_local else None,
            )
            for p in range(1, 1 if config.loop_share_cache else len(self.conv_states))
        )
        self.k_sconv, self.v_sconv = (
            Inkling2ShortConv(
                kv_size,
                config.sconv_kernel_size,
                conv_state,
                stream,
                dynamic_input_size=None if is_local else config.hidden_size,
                num_heads=self.num_kv_heads,
            )
            for stream in (K_STREAM, V_STREAM)
        )

        vllm_config = get_current_vllm_config()
        self.max_decode_query_len = max_decode_query_len(vllm_config)
        cache_dtype = vllm_config.cache_config.cache_dtype
        supported: tuple[str, ...] = ("auto", "bfloat16")
        if current_platform.is_device_capability_family(120):
            supported += ("float16", "fp8", "fp8_e4m3")
        if cache_dtype not in supported:
            raise ValueError(f"Inkling2 KV cache dtype must be one of {supported}")
        self.fp8_kv = cache_dtype in ("fp8", "fp8_e4m3")
        if cache_dtype == "float16" and vllm_config.speculative_config is not None:
            raise ValueError("Inkling2 FP16 KV requires speculative decoding disabled")
        self.kv_cache = torch.tensor([])
        forward_context = vllm_config.compilation_config.static_forward_context
        if prefix in forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        forward_context[prefix] = self
        if vllm_config.kernel_config.enable_jit_warmup and use_sm100_kernels():
            _INKLING2_FA4_REL_ATTENTION_KERNEL.register_warmup()

    def process_weights_after_loading(self) -> None:
        self.rel_proj_shifted = shift_rel_proj(self.rel_logits_proj, self.is_local)

    def get_attn_backend(self) -> type[AttentionBackend]:
        return Inkling2AttentionBackend

    def get_kv_cache_config(self, vllm_config: VllmConfig) -> KVCacheLayerConfig:
        return _kv_cache_layer_config(
            vllm_config,
            self.num_kv_heads,
            self.head_dim,
            self.sliding_window if self.is_local else None,
        )

    def _pass_cache(self, pass_idx: int) -> tuple[str, torch.Tensor]:
        """The attention metadata key and KV cache of loop pass `pass_idx`."""
        if pass_idx == 0 or not self.pass_caches:
            return self.prefix, self.kv_cache
        cache = self.pass_caches[pass_idx - 1]
        return cache.prefix, cache.kv_cache

    def _project(
        self, x_q: torch.Tensor, x_k: torch.Tensor, x_v: torch.Tensor
    ) -> list[torch.Tensor]:
        # ``linear`` on the weight bypasses the layer (and a LoRA wrapper's
        # forward), so the LoRA deltas are added here (no-op without LoRA).
        weight = self.qkvr.weight
        if x_k is x_q and x_v is x_q:
            projected = linear(x_q, weight)
            add_linear_lora(self.qkvr, projected, x_q, range(6))
            return list(projected.split(self.split_sizes, dim=-1)[:6])
        q_side, kv_size = sum(self.split_sizes[:4]), self.split_sizes[4]
        k_end = q_side + kv_size
        projected = linear(x_q, weight[: sum(self.q_side_splits)])
        k = linear(x_k, weight[q_side:k_end])
        v = linear(x_v, weight[k_end : k_end + kv_size])
        add_linear_lora(self.qkvr, projected, x_q, range(4))
        add_linear_lora(self.qkvr, k, x_k, [4])
        add_linear_lora(self.qkvr, v, x_v, [5])
        return [*projected.split(self.q_side_splits, dim=-1)[:4], k, v]

    def _split_kv_cache(
        self, kv_cache: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.fp8_kv:
            # Every loop pass owns byte storage containing unit-scale E4M3 KV.
            kv_cache = kv_cache.view(torch.float8_e4m3fn)
        key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_dim, dim=-1)
        return (
            canonicalize_singleton_dim_strides(key_cache),
            canonicalize_singleton_dim_strides(value_cache),
        )

    def forward(
        self,
        positions: torch.Tensor,
        x_q: torch.Tensor,
        x_k: torch.Tensor | None = None,
        x_v: torch.Tensor | None = None,
        tau: torch.Tensor | None = None,
        out: torch.Tensor | None = None,
        pass_idx: int = 0,
    ) -> torch.Tensor:
        """Return this rank's bf16 `(T, hidden)` partial output (to be
        all-reduced over TP), written to `out` if given.

        Local layers read q/k/v from `x_q` (or `x_k` / `x_v` if given); full
        layers take `x_k`, `x_v` and the fp32 log-scaling `tau` for q/r.
        `pass_idx` picks the loop pass's caches.
        """
        if self.is_local and x_k is None:
            x_k = x_v = x_q
        assert x_k is not None and x_v is not None
        q, r, gate, sinks, k, v = self._project(x_q, x_k, x_v)
        output = torch.empty(gate.shape, dtype=q.dtype, device=q.device)
        self._attention(
            positions,
            q,
            r,
            gate,
            sinks,
            k,
            v,
            self.k_sconv.dynamic_taps(x_k),
            self.v_sconv.dynamic_taps(x_v),
            tau,
            output,
            pass_idx,
        )
        out = linear(output, self.wo_ud.weight, out)
        add_linear_lora(self.wo_ud, out, output)
        return out

    @eager_break_during_capture
    def _attention(
        self,
        positions: torch.Tensor,
        q: torch.Tensor,
        r: torch.Tensor,
        gate: torch.Tensor,
        sinks: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        k_dynamic: torch.Tensor | None,
        v_dynamic: torch.Tensor | None,
        tau: torch.Tensor | None,
        output: torch.Tensor,
        pass_idx: int = 0,
    ) -> None:
        """Write the gated attention output to `output`.

        The kernel plan and the prologue's layout follow the batch's
        decode/prefill split, so this runs eagerly in piecewise CUDA graphs,
        which are captured on a dummy batch and replayed for any batch of the
        same token count.
        """
        num_tokens = q.shape[0]
        q = q.view(num_tokens, self.num_heads, self.head_dim)
        attn_metadata = get_forward_context().attn_metadata
        layer_name, kv_cache = self._pass_cache(pass_idx)
        conv_state = self.conv_states[pass_idx]
        md = None
        if isinstance(attn_metadata, dict) and kv_cache.numel() > 0:
            md = cast(Inkling2AttentionMetadata, attn_metadata[layer_name])
        num_decodes, num_decode_tokens, decode_splits = 0, 0, 0
        if md is not None:
            num_decodes, num_decode_tokens, decode_splits = self._plan_decode(
                md, num_tokens
            )
        clipped_q = q.new_empty(q.shape)
        # Sheared for tml_fa4 (see attention_prologue); the decode kernels
        # read the same layout.
        rel_logits = q.new_empty(
            (num_tokens + 128, self.num_heads, self.rel_extent + 256)
        )
        if md is not None:
            conv_md = conv_metadata(conv_state)
            assert conv_md is not None
            key_cache, value_cache = self._split_kv_cache(kv_cache)
            attention_prologue(
                q,
                r.view(num_tokens, self.num_heads, self.d_rel),
                k,
                v,
                self.q_scale,
                self.r_scale,
                tau,
                self.rel_proj_shifted,
                self.k_sconv.conv_weight,
                self.v_sconv.conv_weight,
                k_dynamic,
                v_dynamic,
                conv_state.kv_cache,
                positions,
                conv_md.block_table,
                conv_md.seq_idx,
                conv_md.slot_mapping,
                conv_md.query_start,
                key_cache,
                value_cache,
                md.slot_mapping,
                clipped_q,
                rel_logits,
                md.query_start_loc,
                md.seq_lens,
                clip_threshold=self.qk_clip,
                rel_threshold=self.rel_clip,
                norm_eps=self.eps,
            )
        if decode_splits:
            assert md is not None
            self._decode_attention(
                md,
                kv_cache,
                clipped_q[:num_decode_tokens],
                rel_logits,
                gate[:num_decode_tokens],
                sinks[:num_decode_tokens],
                output[:num_decode_tokens],
                decode_splits,
                num_decodes,
            )
        if num_decode_tokens < num_tokens:
            out = q.new_empty(
                (num_tokens - num_decode_tokens, self.num_heads, self.head_dim),
                dtype=torch.float32,
            )
            lse = q.new_empty((self.num_heads, out.shape[0]), dtype=torch.float32)
            if md is not None:
                num_splits, use_scheduler = plan_attention(
                    self.is_local,
                    md.query_lens_cpu[num_decodes:],
                    md.seq_lens_cpu[num_decodes:],
                    self.num_heads,
                    self.num_kv_heads,
                )
                self._fa4_attention(
                    md,
                    kv_cache,
                    clipped_q[num_decode_tokens:],
                    rel_logits[num_decode_tokens:],
                    out,
                    lse,
                    num_splits,
                    use_scheduler,
                    num_decodes,
                )
            else:
                out.zero_()
                lse.zero_()
            gate_sink(
                out,
                gate[num_decode_tokens:],
                lse.t(),
                sinks[num_decode_tokens:],
                output[num_decode_tokens:],
            )

    def _plan_decode(
        self, md: FlashAttentionMetadata, num_tokens: int
    ) -> tuple[int, int, int]:
        """Return `(num_decodes, num_decode_tokens, num_splits)` for
        `rel_decode_attention`: the whole batch, or in a mixed batch the
        leading decode/verify requests."""
        num_reqs = md.seq_lens.shape[0]
        plan = partial(
            plan_decode_attention,
            self.is_local,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            rel_extent=self.rel_extent,
        )
        num_splits = plan(num_reqs=num_reqs, max_query_len=md.max_query_len)
        if num_splits:
            return num_reqs, num_tokens, num_splits
        num_decodes, num_decode_tokens = md.num_decode_reqs, md.num_decode_tokens
        if not 0 < num_decodes < num_reqs:
            return 0, 0, 0
        max_query_len = min(
            self.max_decode_query_len, num_decode_tokens - num_decodes + 1
        )
        num_splits = plan(num_reqs=num_decodes, max_query_len=max_query_len)
        if not num_splits:
            return 0, 0, 0
        return num_decodes, num_decode_tokens, num_splits

    def _decode_attention(
        self,
        md: FlashAttentionMetadata,
        kv_cache: torch.Tensor,
        q: torch.Tensor,
        rel_logits: torch.Tensor,
        gate: torch.Tensor,
        sinks: torch.Tensor,
        output: torch.Tensor,
        num_splits: int,
        num_reqs: int,
    ) -> None:
        key_cache, value_cache = self._split_kv_cache(kv_cache)
        mixed = num_reqs < md.seq_lens.shape[0]
        num_tokens = md.num_decode_tokens if mixed else md.num_actual_tokens
        max_query_len = md.max_query_len
        if mixed:
            max_query_len = min(self.max_decode_query_len, num_tokens - num_reqs + 1)
        rel_decode_attention(
            q,
            key_cache,
            value_cache,
            rel_logits,
            gate,
            sinks,
            output,
            block_table=md.block_table[:num_reqs],
            cache_seqlens=md.seq_lens[:num_reqs],
            cu_seqlens_q=md.query_start_loc[: num_reqs + 1],
            max_query_len=max_query_len,
            num_actual_tokens=num_tokens,
            softmax_scale=self.scaling,
            is_local=self.is_local,
            num_splits=num_splits,
        )

    def _fa4_attention(
        self,
        md: FlashAttentionMetadata,
        kv_cache: torch.Tensor,
        q: torch.Tensor,
        rel_logits: torch.Tensor,
        out: torch.Tensor,
        lse: torch.Tensor,
        num_splits: int,
        use_scheduler: bool,
        num_decodes: int,
    ) -> None:
        """FA4 for the requests after the first `num_decodes`."""
        num_decode_tokens = md.num_decode_tokens if num_decodes else 0
        nt = md.num_actual_tokens - num_decode_tokens
        cu_seqlens_q = md.query_start_loc[num_decodes:]
        if num_decodes:
            cu_seqlens_q = cu_seqlens_q - num_decode_tokens
        key_cache, value_cache = self._split_kv_cache(kv_cache)
        if not use_sm100_kernels():
            rel_attention(
                q[:nt],
                key_cache,
                value_cache,
                out,
                lse,
                rel_logits,
                block_table=md.block_table[num_decodes:],
                cache_seqlens=md.seq_lens[num_decodes:],
                cu_seqlens_q=cu_seqlens_q,
                max_query_len=md.max_query_len,
                softmax_scale=self.scaling,
                window=self.sliding_window if self.is_local else 0,
                num_splits=1,
                mode=MODE_OUT_LSE,
            )
            out[nt:].zero_()
            lse[:, nt:].zero_()
            return
        _INKLING2_FA4_REL_ATTENTION_KERNEL(
            q[:nt],
            key_cache,
            value_cache,
            block_table=md.block_table[num_decodes:],
            cache_seqlens=md.seq_lens[num_decodes:],
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=bucket_max_seqlen_q(md.max_query_len),
            softmax_scale=self.scaling,
            window_size=self.window_size,
            sheared_bias=rel_logits[: nt + 128],
            num_splits=num_splits,
            use_scheduler=use_scheduler,
            out=out[:nt],
            lse=lse[:, :nt],
        )
        out[nt:].zero_()
        lse[:, nt:].zero_()

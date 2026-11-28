# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Gated delta product (GDP) linear attention.

A gated delta product layer applies `m = linear_num_householder` delta-rule updates
per token: token `t` carries `m` key/value pairs and `m` write strengths `beta`;
the state decays once (on the first update) and is read once, by the query, after the
last update. This is exactly a gated delta *rule* over a virtual sequence with `m`
steps per token, which is how it runs here: on vLLM's gated delta rule kernels, with
zero decay on steps `1..m-1` and the query placed on step `m-1`.

A layer can own several independent recurrent states (`cache_prefixes`), for models
that run the same weights more than once per token (looped models): one
`GatedDeltaProductCache` per extra state registers it with the cache manager, and
`forward` selects the state by index.
"""

import math
import weakref

import torch
from torch import nn

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.layernorm import RMSNormGated
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.mamba.abstract import (
    MambaBase,
    get_mamba_prefill_checkpoint_alignment,
    get_mamba_prefill_checkpoint_grid,
)
from vllm.model_executor.layers.mamba.checkpoint import (
    MambaPrefillCheckpointExporter,
)
from vllm.model_executor.layers.mamba.gdn.base import GatedDeltaNetAttention
from vllm.model_executor.layers.mamba.mamba_mixer2 import mamba_v2_sharded_weight_loader
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateShapeCalculator,
    is_conv_state_dim_first,
)
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from vllm.model_executor.layers.mamba.ops.causal_conv1d_prefill import (
    causal_conv1d_prefill,
)
from vllm.model_executor.layers.mamba.ops.gated_delta_product import (
    fused_recurrent_gated_delta_product,
)
from vllm.model_executor.layers.mamba.ops.gdp_chunk_prefill import (
    gdp_chunk_prefill,
    gdp_chunk_prefill_supported,
)
from vllm.model_executor.model_loader.weight_utils import sharded_weight_loader
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.third_party.flash_linear_attention.ops import chunk_gated_delta_rule
from vllm.third_party.flash_linear_attention.ops.index import (
    prepare_chunk_indices,
    prepare_chunk_offsets,
)
from vllm.third_party.flash_linear_attention.ops.l2norm import l2norm_fwd
from vllm.third_party.flash_linear_attention.ops.utils import FLA_CHUNK_SIZE
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    async_tensor_h2d,
    direct_register_custom_op,
)
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum

logger = init_logger(__name__)


def _set_weight_loader(weight: torch.Tensor, loader) -> None:
    if hasattr(weight, "weight_loader"):
        delattr(weight, "weight_loader")
    set_weight_attrs(weight, {"weight_loader": loader})


# CUDA limits grid axes 1 and 2 to this many blocks.
_GRID_AXIS_LIMIT = 65535


# Virtual-sequence chunk metadata of the latest prefill batches, shared by every layer
# of a forward (each layer has its own metadata object, built from the same query
# starts): [(query starts on the host, m, device, (cu_seqlens, chunk_indices,
# chunk_offsets))], keyed by value.
_virtual_chunk_metadata_cache: list[tuple] = []
_VIRTUAL_CHUNK_METADATA_CACHE_SIZE = 4


def virtual_chunk_metadata(
    query_start_cpu: torch.Tensor | None, device: torch.device, m: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """(cu_seqlens, chunk_indices, chunk_offsets) of a prefill batch's virtual
    sequences, built once per batch from the host copy of the query starts (no
    device-to-host sync) and reused by every layer. None without the host copy.
    """
    if query_start_cpu is None:
        return None
    for cached_query_start, cached_m, cached_device, cached in reversed(
        _virtual_chunk_metadata_cache
    ):
        if (
            cached_m == m
            and cached_device == device
            and torch.equal(cached_query_start, query_start_cpu)
        ):
            return cached
    cu_seqlens_cpu = (query_start_cpu * m).to(torch.int32)
    result = (
        async_tensor_h2d(cu_seqlens_cpu, device=device),
        async_tensor_h2d(
            prepare_chunk_indices(cu_seqlens_cpu, FLA_CHUNK_SIZE), device=device
        ),
        async_tensor_h2d(
            prepare_chunk_offsets(cu_seqlens_cpu, FLA_CHUNK_SIZE), device=device
        ),
    )
    _virtual_chunk_metadata_cache.append((query_start_cpu.clone(), m, device, result))
    del _virtual_chunk_metadata_cache[:-_VIRTUAL_CHUNK_METADATA_CACHE_SIZE]
    return result


class GatedDeltaProductCache(nn.Module, MambaBase):
    """An additional recurrent state of a `GatedDeltaProductAttention` layer.

    Registers itself under its own layer name so the cache manager allocates a state for
    it, and runs the owning layer's core on that state. Holds no weights.
    """

    def __init__(self, owner: "GatedDeltaProductAttention", prefix: str) -> None:
        super().__init__()
        object.__setattr__(self, "_owner_ref", weakref.ref(owner))
        self.prefix = prefix
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    @property
    def owner(self) -> "GatedDeltaProductAttention":
        owner = self._owner_ref()
        if owner is None:
            raise RuntimeError(f"{self.prefix}: the owning layer was released")
        return owner

    @property
    def mamba_type(self) -> MambaAttentionBackendEnum:
        return self.owner.mamba_type

    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        return self.owner.get_state_dtype()

    def get_state_shape(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return self.owner.get_state_shape()

    def get_prefill_checkpoint_alignment(self) -> int | None:
        return self.owner.get_prefill_checkpoint_alignment()

    def get_kv_cache_checkpoint_alignment(self, vllm_config: VllmConfig) -> int | None:
        return self.owner.get_kv_cache_checkpoint_alignment(vllm_config)

    def _forward_core(self, **kwargs) -> None:
        self.owner._forward_core_on(self, **kwargs)


@PluggableLayer.register("gated_delta_product_attention")
class GatedDeltaProductAttention(GatedDeltaNetAttention):
    """Gated delta product linear attention on vLLM's gated delta rule kernels.

    Projections (one sharded GEMM): `z` (output gate), `m` value copies, `m` key
    copies, the query, `m` write strengths `b` and the decay input `a`. `v/k/q`
    pass a causal convolution; keys and query are L2-normalized; `beta = sigmoid(b)`,
    `g = -exp(A_log) * softplus(a + dt_bias)`; the output passes a gated RMSNorm
    (grouped by key group) and the output projection.

    """

    def __init__(
        self,
        config,
        vllm_config: VllmConfig,
        prefix: str = "",
        cache_prefixes: list[str] | None = None,
    ) -> None:
        cache_prefixes = list(cache_prefixes) if cache_prefixes else [prefix]
        super().__init__(config, vllm_config, prefix=cache_prefixes[0])
        if self.quant_config is not None:
            raise NotImplementedError(
                "Quantized gated delta product layers are not supported."
            )
        self.activation = "silu"  # convolution activation
        self.cache_prefixes = cache_prefixes
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_v_dim = config.linear_value_head_dim
        self.head_k_dim = config.linear_key_head_dim
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.num_householder = m = config.linear_num_householder
        if self.num_v_heads % self.tp_size or self.num_k_heads % self.tp_size:
            raise ValueError(
                f"linear_num_value_heads={self.num_v_heads} and "
                f"linear_num_key_heads={self.num_k_heads} must be divisible by "
                f"tensor parallel size {self.tp_size}"
            )
        self.value_dim = self.num_v_heads * self.head_v_dim
        self.key_dim = self.num_k_heads * self.head_k_dim
        self.conv_dim = m * self.value_dim + (m + 1) * self.key_dim
        self.local_v_heads = self.num_v_heads // self.tp_size
        self.local_k_heads = self.num_k_heads // self.tp_size
        self.local_value_dim = self.value_dim // self.tp_size
        self.local_key_dim = self.key_dim // self.tp_size

        # Output sections: z, v_0..v_{m-1}, k_0..k_{m-1}, q, b_0..b_{m-1}, a.
        self.in_proj = MergedColumnParallelLinear(
            self.hidden_size,
            [self.value_dim] * (m + 1)
            + [self.key_dim] * (m + 1)
            + [self.num_v_heads] * (m + 1),
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.in_proj",
        )
        self.conv1d = ColumnParallelLinear(
            input_size=self.conv_kernel_size,
            output_size=self.conv_dim,
            bias=False,
            prefix=f"{prefix}.conv1d",
        )
        self.conv1d.weight.data = self.conv1d.weight.data.unsqueeze(1)
        # Channels v_0..v_{m-1}, k_0..k_{m-1}, q, each sharded by heads / key groups.
        _set_weight_loader(
            self.conv1d.weight,
            mamba_v2_sharded_weight_loader(
                [(self.value_dim, 0, False)] * m + [(self.key_dim, 0, False)] * (m + 1),
                self.tp_size,
                self.tp_rank,
            ),
        )
        self.dt_bias = nn.Parameter(torch.ones(self.local_v_heads))
        self.A_log = nn.Parameter(torch.empty(self.local_v_heads, dtype=torch.float32))
        set_weight_attrs(self.dt_bias, {"weight_loader": sharded_weight_loader(0)})
        set_weight_attrs(self.A_log, {"weight_loader": sharded_weight_loader(0)})
        self.norm = RMSNormGated(
            self.local_value_dim,
            eps=config.rms_norm_eps,
            group_size=self.value_dim // self.num_k_heads,
            norm_before_gate=False,
            activation="silu",
            device=current_platform.current_device(),
        )
        _set_weight_loader(self.norm.weight, sharded_weight_loader(0))
        self.out_proj = RowParallelLinear(
            self.value_dim,
            self.hidden_size,
            bias=False,
            input_is_parallel=True,
            quant_config=None,
            prefix=f"{prefix}.out_proj",
        )

        compilation_config = get_current_vllm_config().compilation_config
        if self.prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {self.prefix}")
        compilation_config.static_forward_context[self.prefix] = self
        self.extra_caches = nn.ModuleList(
            [GatedDeltaProductCache(self, p) for p in self.cache_prefixes[1:]]
        )
        self.fast_prefill = gdp_chunk_prefill_supported(
            m, self.head_k_dim, self.head_v_dim
        )
        self._prefill_kernels_warmed_up = False
        self._checkpoint_alignment = None
        if vllm_config.cache_config.enable_prefix_caching:
            self._checkpoint_alignment = FLA_CHUNK_SIZE // math.gcd(FLA_CHUNK_SIZE, m)
        self._checkpoint_exporter = MambaPrefillCheckpointExporter(
            state_len=self.conv_kernel_size - 1
        )

    def get_prefill_checkpoint_alignment(self) -> int | None:
        return self._checkpoint_alignment

    def get_kv_cache_checkpoint_alignment(self, vllm_config: VllmConfig) -> int | None:
        if (
            self._checkpoint_alignment is None
            or get_mamba_prefill_checkpoint_alignment(vllm_config) is None
        ):
            return None
        return get_mamba_prefill_checkpoint_grid(vllm_config)

    def get_state_shape(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return MambaStateShapeCalculator.gated_delta_product_state_shape(
            self.tp_size,
            self.num_k_heads,
            self.num_v_heads,
            self.head_k_dim,
            self.head_v_dim,
            self.num_householder,
            self.conv_kernel_size,
        )

    def load_in_proj_weight(self, role: str, loaded_weight: torch.Tensor) -> None:
        """Load one per-role projection (`z`, `v`, `k`, `q`, `b` or `a`)
        into the merged input projection. `v`, `k` and `b` hold their `m` copies
        stacked.
        """
        m = self.num_householder
        first_shard = {
            "z": 0,
            "v": 1,
            "k": m + 1,
            "q": 2 * m + 1,
            "b": 2 * m + 2,
            "a": 3 * m + 2,
        }
        pieces = loaded_weight.chunk(m) if role in ("v", "k", "b") else (loaded_weight,)
        loader = self.in_proj.weight.weight_loader
        for i, piece in enumerate(pieces):
            loader(self.in_proj.weight, piece, first_shard[role] + i)

    def forward(
        self, hidden_states: torch.Tensor, cache_index: int = 0
    ) -> torch.Tensor:
        """Run the layer on its recurrent state `cache_index`."""
        num_tokens = hidden_states.shape[0]
        m = self.num_householder
        projected, _ = self.in_proj(hidden_states)
        conv_end = (
            self.local_value_dim
            + m * self.local_value_dim
            + (m + 1) * self.local_key_dim
        )
        z = projected[:, : self.local_value_dim]
        mixed_qkv = projected[:, self.local_value_dim : conv_end]
        b = projected[:, conv_end : conv_end + m * self.local_v_heads]
        a = projected[:, conv_end + m * self.local_v_heads :]
        core_attn_out = torch.zeros(
            num_tokens,
            self.local_v_heads,
            self.head_v_dim,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        torch.ops.vllm.gated_delta_product_core(
            mixed_qkv,
            b,
            a,
            core_attn_out,
            _encode_layer_name(self.cache_prefixes[cache_index]),
        )
        normed = self.norm(core_attn_out.view(num_tokens, self.local_value_dim), z)
        output, _ = self.out_proj(normed)
        return output

    # ------------------------------------------------------------------ core

    def _forward_core(self, **kwargs) -> None:
        self._forward_core_on(self, **kwargs)

    def _forward_core_on(
        self,
        cache_layer: "GatedDeltaProductAttention | GatedDeltaProductCache",
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        core_attn_out: torch.Tensor,
    ) -> None:
        """Convolution and gated delta product on the recurrent state of
        `cache_layer`."""
        attn_metadata = get_forward_context().attn_metadata
        if attn_metadata is None:
            # Profile run: tune the prefill kernels while memory is still free.
            self._warmup_prefill_kernels(mixed_qkv.dtype, mixed_qkv.device)
            return
        assert isinstance(attn_metadata, dict)
        metadata = attn_metadata.get(cache_layer.prefix)
        if metadata is None:
            # Dummy runs skip mamba-family metadata.
            return
        assert isinstance(metadata, GDNAttentionMetadata)
        num_tokens = metadata.num_actual_tokens
        if num_tokens == 0:
            return
        mixed_qkv, b, a = mixed_qkv[:num_tokens], b[:num_tokens], a[:num_tokens]
        conv_state = (
            cache_layer.kv_cache[0]
            if is_conv_state_dim_first()
            else cache_layer.kv_cache[0].transpose(-1, -2)
        )
        ssm_state = cache_layer.kv_cache[1]
        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )

        state_indices = metadata.non_spec_state_indices_tensor
        assert state_indices is not None
        if metadata.num_prefills > 0:
            self._prefill(
                metadata,
                mixed_qkv,
                b,
                a,
                conv_state,
                ssm_state,
                conv_weights,
                core_attn_out[:num_tokens],
            )
        else:
            conv_out = causal_conv1d_update(
                mixed_qkv,
                conv_state,
                conv_weights,
                None,
                self.activation,
                conv_state_indices=state_indices[:num_tokens],
                validate_data=False,
            )
            fused_recurrent_gated_delta_product(
                conv_out,
                b,
                a,
                self.A_log,
                self.dt_bias,
                ssm_state,
                core_attn_out[:num_tokens],
                num_householder=self.num_householder,
                num_groups=self.local_k_heads,
                num_heads=self.local_v_heads,
                state_dim=self.head_k_dim,
                head_dim=self.head_v_dim,
                ssm_state_indices=state_indices[:num_tokens],
            )

    def _prefill(
        self, metadata, mixed_qkv, b, a, conv_state, ssm_state, conv_weights, out
    ):
        query_start = metadata.non_spec_query_start_loc
        state_indices = metadata.non_spec_state_indices_tensor
        has_initial = metadata.has_initial_state
        assert (
            query_start is not None
            and state_indices is not None
            and has_initial is not None
        )
        checkpoint = metadata.checkpoint
        fast_conv = self.fast_prefill and self.conv_kernel_size == 4
        if fast_conv:
            conv_out = causal_conv1d_prefill(
                mixed_qkv,
                conv_weights,
                conv_state,
                query_start,
                state_indices,
                has_initial,
                checkpoint_offsets=None
                if checkpoint is None
                else checkpoint.checkpoint_offsets,
                checkpoint_indices=None
                if checkpoint is None
                else checkpoint.state_indices,
            )
        else:
            conv_out = causal_conv1d_fn(
                mixed_qkv.contiguous().transpose(0, 1),
                conv_weights,
                None,
                activation=self.activation,
                conv_states=conv_state,
                has_initial_state=has_initial,
                cache_indices=state_indices,
                query_start_loc=query_start,
                metadata=metadata,
            ).transpose(0, 1)
        chunks = virtual_chunk_metadata(
            metadata.non_spec_query_start_loc_cpu,
            query_start.device,
            self.num_householder,
        )
        if self.fast_prefill:
            gdp_chunk_prefill(
                conv_out,
                b,
                a,
                self.A_log,
                self.dt_bias,
                ssm_state,
                query_start,
                state_indices,
                has_initial,
                num_householder=self.num_householder,
                num_heads=self.local_v_heads,
                num_groups=self.local_k_heads,
                head_k_dim=self.head_k_dim,
                head_v_dim=self.head_v_dim,
                chunk_metadata=chunks,
                checkpoint_offsets=None
                if checkpoint is None
                else checkpoint.checkpoint_offsets,
                checkpoint_indices=None
                if checkpoint is None
                else checkpoint.state_indices,
                out=out,
            )
            if checkpoint is not None and not fast_conv:
                self._checkpoint_exporter.export(
                    checkpoint,
                    raw_qkv=mixed_qkv,
                    conv_state=conv_state,
                    cu_seqlens=query_start,
                )
        else:
            q, k, v, g, beta = self._virtual_inputs(conv_out, b, a)
            initial = ssm_state[state_indices].contiguous()
            initial.masked_fill_(~has_initial.view(-1, 1, 1, 1), 0)
            checkpoint_state = (
                torch.empty_like(initial) if checkpoint is not None else None
            )
            virtual_out, final = self._chunked_prefill(
                q,
                k,
                v,
                g,
                beta,
                initial,
                query_start,
                checkpoint_state,
                None if checkpoint is None else checkpoint.checkpoint_offsets,
                chunks,
            )
            ssm_state[state_indices] = final.to(ssm_state.dtype)
            out.copy_(
                virtual_out.reshape(
                    -1, self.num_householder, self.local_v_heads, self.head_v_dim
                )[:, -1]
            )
            if checkpoint is not None:
                self._checkpoint_exporter.export(
                    checkpoint,
                    raw_qkv=mixed_qkv,
                    conv_state=conv_state,
                    cu_seqlens=query_start,
                    recurrent_checkpoint=checkpoint_state,
                    recurrent_state=ssm_state,
                )

    def _virtual_inputs(
        self, conv_out: torch.Tensor, b: torch.Tensor, a: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        """(q, k, v, g, beta) of the virtual gated delta rule sequence, `m` steps per
        token."""
        num_tokens = conv_out.shape[0]
        m = self.num_householder
        heads, groups = self.local_v_heads, self.local_k_heads
        pieces = conv_out.split(
            [self.local_value_dim] * m + [self.local_key_dim] * (m + 1), dim=-1
        )
        value = torch.stack(
            [p.view(num_tokens, heads, self.head_v_dim) for p in pieces[:m]], dim=1
        ).reshape(num_tokens * m, heads, self.head_v_dim)
        key = torch.stack(
            [p.view(num_tokens, groups, self.head_k_dim) for p in pieces[m : 2 * m]],
            dim=1,
        ).reshape(num_tokens * m, groups, self.head_k_dim)
        query = pieces[-1].reshape(num_tokens, groups, self.head_k_dim)
        if heads != groups:
            key = key.repeat_interleave(heads // groups, dim=1)
            query = query.repeat_interleave(heads // groups, dim=1)
        key = l2norm_fwd(key.contiguous())
        query = l2norm_fwd(query.contiguous())
        # The query reads the state after the token's last step; the decay acts on its
        # first.
        virtual_query = query.new_zeros(num_tokens, m, heads, self.head_k_dim)
        virtual_query[:, -1] = query
        g = -self.A_log.float().exp() * nn.functional.softplus(
            a.float() + self.dt_bias.float()
        )
        virtual_g = g.new_zeros(num_tokens, m, heads)
        virtual_g[:, 0] = g
        beta = torch.sigmoid(b.view(num_tokens, m, heads))
        return (
            virtual_query.reshape(1, num_tokens * m, heads, self.head_k_dim),
            key.unsqueeze(0).contiguous(),
            value.unsqueeze(0).contiguous(),
            virtual_g.reshape(1, num_tokens * m, heads),
            beta.reshape(1, num_tokens * m, heads).contiguous(),
        )

    def _chunked_prefill(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor,
        query_start: torch.Tensor,
        checkpoint_state: torch.Tensor | None = None,
        checkpoint_offsets: torch.Tensor | None = None,
        chunk_metadata: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The chunked gated delta rule over the virtual sequences of a prefill batch.

        The chunked kernels launch one block per (sequence, value head) on a grid axis
        limited to 65535 blocks, so a batch with more sequences than
        `65535 // value_heads` runs in groups of sequences (each group is a contiguous
        slice of the virtual tokens; inputs are `[1, virtual tokens, heads, ...]`).
        """
        m = self.num_householder
        num_seqs = query_start.shape[0] - 1
        max_seqs = _GRID_AXIS_LIMIT // q.shape[-2]

        def run(q, k, v, g, beta, initial_state, real_cu_seqlens, states, offsets):
            cu_seqlens = (real_cu_seqlens * m).to(torch.int32)
            if chunk_metadata is not None and num_seqs <= max_seqs:
                cu_seqlens, indices, offsets_by_chunk = chunk_metadata
            else:
                indices = prepare_chunk_indices(cu_seqlens, FLA_CHUNK_SIZE)
                offsets_by_chunk = prepare_chunk_offsets(cu_seqlens, FLA_CHUNK_SIZE)
            return chunk_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=initial_state,
                output_final_state=True,
                checkpoint_state=states,
                checkpoint_offsets=None if offsets is None else offsets * m,
                cu_seqlens=cu_seqlens,
                chunk_indices=indices,
                chunk_offsets=offsets_by_chunk,
            )

        if num_seqs <= max_seqs:
            return run(
                q,
                k,
                v,
                g,
                beta,
                initial_state,
                query_start,
                checkpoint_state,
                checkpoint_offsets,
            )
        bounds = query_start.tolist()
        outs, finals = [], []
        for s0 in range(0, num_seqs, max_seqs):
            s1 = min(s0 + max_seqs, num_seqs)
            t0, t1 = bounds[s0] * m, bounds[s1] * m
            out, final = run(
                q[:, t0:t1],
                k[:, t0:t1],
                v[:, t0:t1],
                g[:, t0:t1],
                beta[:, t0:t1],
                initial_state[s0:s1],
                query_start[s0 : s1 + 1] - bounds[s0],
                None if checkpoint_state is None else checkpoint_state[s0:s1],
                None if checkpoint_offsets is None else checkpoint_offsets[s0:s1],
            )
            outs.append(out)
            finals.append(final)
        return torch.cat(outs, dim=1), torch.cat(finals)

    def _warmup_fast_prefill(
        self, dtype: torch.dtype, state_dtype: torch.dtype, device: torch.device
    ) -> None:
        """Compile the fast path's kernels (fixed configurations, no autotuning) on a
        one-chunk sequence, with and without a checkpoint."""
        m, heads = self.num_householder, self.local_v_heads
        tokens = FLA_CHUNK_SIZE
        conv_dim = m * self.local_value_dim + (m + 1) * self.local_key_dim
        cu_seqlens = torch.tensor([0, tokens], dtype=torch.int32, device=device)
        slot = torch.ones(1, dtype=torch.int32, device=device)
        has_state = torch.zeros(1, dtype=torch.bool, device=device)
        virtual_cu = (cu_seqlens * m).to(torch.int32)
        chunk_metadata = (
            virtual_cu,
            prepare_chunk_indices(virtual_cu, FLA_CHUNK_SIZE),
            prepare_chunk_offsets(virtual_cu, FLA_CHUNK_SIZE),
        )
        try:
            for checkpoints in ((None, None), (torch.zeros_like(slot), slot + 1)):
                conv_out = torch.randn(tokens, conv_dim, dtype=dtype, device=device)
                if self.conv_kernel_size == 4:
                    shape = (
                        (3, conv_dim, 3)
                        if is_conv_state_dim_first()
                        else (3, 3, conv_dim)
                    )
                    conv_state = torch.zeros(shape, dtype=dtype, device=device)
                    if not is_conv_state_dim_first():
                        conv_state = conv_state.transpose(-1, -2)
                    conv_out = causal_conv1d_prefill(
                        conv_out,
                        torch.zeros(conv_dim, 4, dtype=dtype, device=device),
                        conv_state,
                        cu_seqlens,
                        slot,
                        has_state,
                        checkpoint_offsets=checkpoints[0],
                        checkpoint_indices=checkpoints[1],
                    )
                gdp_chunk_prefill(
                    conv_out,
                    torch.zeros(tokens, m * heads, dtype=dtype, device=device),
                    torch.zeros(tokens, heads, dtype=dtype, device=device),
                    torch.zeros(heads, dtype=torch.float32, device=device),
                    torch.zeros(heads, dtype=self.dt_bias.dtype, device=device),
                    torch.zeros(
                        3,
                        heads,
                        self.head_v_dim,
                        self.head_k_dim,
                        dtype=state_dtype,
                        device=device,
                    ),
                    cu_seqlens,
                    slot,
                    has_state,
                    num_householder=m,
                    num_heads=heads,
                    num_groups=self.local_k_heads,
                    head_k_dim=self.head_k_dim,
                    head_v_dim=self.head_v_dim,
                    chunk_metadata=chunk_metadata,
                    checkpoint_offsets=checkpoints[0],
                    checkpoint_indices=checkpoints[1],
                )
        except Exception:
            logger.warning(
                "Gated delta product prefill kernel warmup failed for %s; the first "
                "prefill will compile its kernels.",
                self.prefix,
                exc_info=True,
            )

    def _warmup_prefill_kernels(self, dtype: torch.dtype, device: torch.device) -> None:
        """Autotune the chunked kernels on a one-chunk sequence during the profile run,
        so the first real prefill does not need autotuning memory after the caches are
        allocated.
        """
        if self._prefill_kernels_warmed_up:
            return
        self._prefill_kernels_warmed_up = True
        heads, tokens = self.local_v_heads, FLA_CHUNK_SIZE * self.num_householder
        _, state_dtype = self.get_state_dtype()
        if self.fast_prefill:
            self._warmup_fast_prefill(dtype, state_dtype, device)
            return
        q = torch.randn(1, tokens, heads, self.head_k_dim, dtype=dtype, device=device)
        k = l2norm_fwd(torch.randn_like(q))
        v = torch.randn(1, tokens, heads, self.head_v_dim, dtype=dtype, device=device)
        g = torch.zeros(1, tokens, heads, dtype=torch.float32, device=device)
        beta = torch.sigmoid(torch.randn(1, tokens, heads, dtype=dtype, device=device))
        state = torch.zeros(
            1, heads, self.head_v_dim, self.head_k_dim, dtype=state_dtype, device=device
        )
        cu_seqlens = torch.tensor([0, tokens], dtype=torch.int32, device=device)
        try:
            chunk_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=state,
                output_final_state=True,
                cu_seqlens=cu_seqlens,
                chunk_indices=prepare_chunk_indices(cu_seqlens, FLA_CHUNK_SIZE),
                chunk_offsets=prepare_chunk_offsets(cu_seqlens, FLA_CHUNK_SIZE),
            )
        except Exception:
            logger.warning(
                "Gated delta product prefill kernel warmup failed for %s; the first "
                "prefill may need additional autotuning memory.",
                self.prefix,
                exc_info=True,
            )


@eager_break_during_capture
def gated_delta_product_core(
    mixed_qkv: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    core_attn_out: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    """Convolution and gated delta product of one layer's recurrent state
    (`layer_name`)."""
    forward_context: ForwardContext = get_forward_context()
    layer = forward_context.no_compile_layers[_resolve_layer_name(layer_name)]
    layer._forward_core(mixed_qkv=mixed_qkv, b=b, a=a, core_attn_out=core_attn_out)


direct_register_custom_op(
    op_name="gated_delta_product_core",
    op_func=gated_delta_product_core,
    mutates_args=["core_attn_out"],
)

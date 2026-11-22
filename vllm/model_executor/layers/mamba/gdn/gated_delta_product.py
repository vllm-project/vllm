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
from vllm.model_executor.layers.mamba.abstract import MambaBase
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
from vllm.model_executor.layers.mamba.ops.gated_delta_product import (
    fused_recurrent_gated_delta_product,
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
        self._prefill_kernels_warmed_up = False

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
            query_start = metadata.non_spec_query_start_loc
            assert query_start is not None and metadata.has_initial_state is not None
            conv_out = causal_conv1d_fn(
                mixed_qkv.contiguous().transpose(0, 1),
                conv_weights,
                None,
                activation=self.activation,
                conv_states=conv_state,
                has_initial_state=metadata.has_initial_state,
                cache_indices=state_indices,
                query_start_loc=query_start,
                metadata=metadata,
            ).transpose(0, 1)
            q, k, v, g, beta = self._virtual_inputs(conv_out, b, a)
            initial_state = ssm_state[state_indices].contiguous()
            initial_state[~metadata.has_initial_state] = 0
            out, final_state = self._chunked_prefill(
                q, k, v, g, beta, initial_state, query_start
            )
            ssm_state[state_indices] = final_state.to(ssm_state.dtype)
            core_attn_out[:num_tokens] = out.reshape(
                num_tokens, self.num_householder, self.local_v_heads, self.head_v_dim
            )[:, -1]
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

        def run(q, k, v, g, beta, initial_state, real_cu_seqlens):
            cu_seqlens = (real_cu_seqlens * m).to(torch.int32)
            return chunk_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=initial_state,
                output_final_state=True,
                cu_seqlens=cu_seqlens,
                chunk_indices=prepare_chunk_indices(cu_seqlens, FLA_CHUNK_SIZE),
                chunk_offsets=prepare_chunk_offsets(cu_seqlens, FLA_CHUNK_SIZE),
            )

        if num_seqs <= max_seqs:
            return run(q, k, v, g, beta, initial_state, query_start)
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
            )
            outs.append(out)
            finals.append(final)
        return torch.cat(outs, dim=1), torch.cat(finals)

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

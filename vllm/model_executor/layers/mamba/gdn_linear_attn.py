# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Qwen3-Next/Qwen3.5 model."""

import torch
from einops import rearrange
from torch import nn
from transformers.activations import ACT2FN

from vllm import envs
from vllm.config import (
    VllmConfig,
    get_current_vllm_config,
)
from vllm.distributed import (
    divide,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp, PluggableLayer
from vllm.model_executor.layers.fla.ops import (
    chunk_gated_delta_rule as fla_chunk_gated_delta_rule,
)
from vllm.model_executor.layers.fla.ops import (
    fused_recurrent_gated_delta_rule,
    fused_recurrent_gated_delta_rule_packed_decode,
    fused_sigmoid_gating_delta_rule_update,
)
from vllm.model_executor.layers.fla.ops.chunk import l2norm_fwd
from vllm.model_executor.layers.layernorm import RMSNormGated
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_mixer2 import mamba_v2_sharded_weight_loader
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.model_loader.weight_utils import (
    sharded_weight_loader,
)
from vllm.model_executor.models.utils import extract_layer_index
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.transformers_utils.configs.qwen3_next import Qwen3NextConfig
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

logger = init_logger(__name__)


def _copy_state_to_dest_from_pool(
    dest: torch.Tensor,
    pool: torch.Tensor,
    pool_indices: torch.Tensor,
    dest_offset: int = 0,
) -> None:
    """Copy pool[pool_indices[valid]] into dest[valid + dest_offset] for valid where
    pool_indices >= 0. Used to fill a destination buffer from a state pool using
    slot indices (e.g. initial_state from ssm_state for decode/prefill).
    """
    valid = torch.nonzero(pool_indices >= 0, as_tuple=False).squeeze(-1)
    if valid.numel() == 0:
        return
    pool_idx = pool_indices.index_select(0, valid).to(
        device=pool.device, dtype=torch.long
    )
    data = pool.index_select(0, pool_idx)
    dest.index_copy_(0, valid + dest_offset, data)


def _copy_state_to_pool_from_src(
    pool: torch.Tensor,
    pool_slot_indices: torch.Tensor,
    src: torch.Tensor,
    src_offset: int = 0,
) -> None:
    """Copy src[valid + src_offset] into pool[pool_slot_indices[valid]] for valid
    where pool_slot_indices >= 0. Used to write state back into the pool at
    given slot indices (e.g. last recurrent state for prefill -> ssm_state).
    """
    valid = torch.nonzero(pool_slot_indices >= 0, as_tuple=False).squeeze(-1)
    if valid.numel() == 0:
        return
    slots = pool_slot_indices.index_select(0, valid).to(
        device=pool.device, dtype=torch.long
    )
    data = src.index_select(0, valid + src_offset).to(pool.dtype)
    pool.index_copy_(0, slots, data)


def fi_chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = True,
):
    from flashinfer.gdn_prefill import (
        chunk_gated_delta_rule as chunk_gated_delta_rule_fi,
    )

    if use_qk_l2norm_in_kernel:
        q = l2norm_fwd(q)
        k = l2norm_fwd(k)

    # use flashinfer implementation
    q = q.squeeze(0).contiguous()
    k = k.squeeze(0).contiguous()
    v = v.squeeze(0).contiguous()

    g = g.squeeze(0).contiguous()
    beta = beta.squeeze(0).contiguous()
    fi_state = initial_state.to(torch.float32)
    fi_g = g.to(torch.float32)
    fi_beta = beta.to(torch.float32)
    result = chunk_gated_delta_rule_fi(
        q=q,
        k=k,
        v=v,
        g=torch.exp(fi_g),
        beta=fi_beta,
        initial_state=fi_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
    )
    # FlashInfer returns (output, state) when output_final_state=True,
    # or just output when output_final_state=False.
    # Unsqueeze back to 4D (1, L, H, D) to match fla output format
    if output_final_state:
        output, final_state = result
        return output.unsqueeze(0), final_state
    else:
        return result.unsqueeze(0), None


@CustomOp.register("chunk_gated_delta_rule")
class ChunkGatedDeltaRule(CustomOp):
    def __init__(self) -> None:
        super().__init__()
        vllm_config = get_current_vllm_config()
        backend_cfg = vllm_config.additional_config.get("gdn_prefill_backend", "auto")
        backend = str(backend_cfg).strip().lower()

        supports_flashinfer = (
            current_platform.is_cuda() and current_platform.is_device_capability(90)
        )

        if backend == "flashinfer":
            use_flashinfer = supports_flashinfer
            if not use_flashinfer:
                logger.warning_once(
                    "GDN prefill backend 'flashinfer' is selected but "
                    "cannot use this kernel on the current platform. "
                    "Falling back to Triton/FLA."
                )
        elif backend == "triton":
            use_flashinfer = False
        else:
            use_flashinfer = supports_flashinfer

        # APC ("all" mode) requires returning intermediate chunk states, which
        # FlashInfer does not support. Fall back to Triton/FLA in that case.
        apc_enabled = vllm_config.cache_config.mamba_cache_mode == "all"
        if use_flashinfer and apc_enabled:
            logger.warning_once(
                "GDN prefill FlashInfer backend does not support APC "
                "(`mamba_cache_mode='all'`). Falling back to Triton/FLA."
            )
            use_flashinfer = False

        if use_flashinfer:
            logger.info_once("Using FlashInfer GDN prefill kernel", scope="local")
            logger.info_once(
                "FlashInfer GDN prefill kernel is JIT-compiled; first run may "
                "take a while to compile. Set `--gdn-prefill-backend triton` to "
                "avoid JIT compile time.",
                scope="local",
            )
        else:
            logger.info_once("Using Triton/FLA GDN prefill kernel", scope="local")

        self._forward_method = (
            self.forward_cuda if use_flashinfer else self.forward_native
        )

    def forward_cuda(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor,
        output_final_state: bool,
        cu_seqlens: torch.Tensor | None = None,
        use_qk_l2norm_in_kernel: bool = True,
    ):
        return fi_chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )

    def forward_native(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor,
        output_final_state: bool,
        cu_seqlens: torch.Tensor | None = None,
        use_qk_l2norm_in_kernel: bool = True,
        return_intermediate_states: bool = False,
        state_dtype: torch.dtype | None = None,
    ):
        return fla_chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            return_intermediate_states=return_intermediate_states,
            state_dtype=state_dtype,
        )


@PluggableLayer.register("gated_delta_net_attention")
class GatedDeltaNetAttention(PluggableLayer, MambaBase):
    @property
    def mamba_type(self) -> str:
        return "gdn_attention"

    def get_state_dtype(self) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            self.model_config.dtype,
            self.cache_config.mamba_cache_dtype,
            self.cache_config.mamba_ssm_cache_dtype,
        )

    def get_state_shape(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            self.tp_size,
            self.num_k_heads,
            self.num_v_heads,
            self.head_k_dim,
            self.head_v_dim,
            self.conv_kernel_size,
            self.num_spec,
        )

    def __init__(
        self,
        config: Qwen3NextConfig,
        vllm_config: VllmConfig,
        prefix: str = "",
        create_in_proj_qkvz: bool = True,
        gqa_interleaved_layout=False,
    ) -> None:
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads

        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_idx = extract_layer_index(prefix)
        self.activation = config.hidden_act
        self.act = ACT2FN[config.hidden_act]
        self.layer_norm_epsilon = config.rms_norm_eps
        self.prefix = prefix
        self.config = config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.speculative_config = vllm_config.speculative_config
        self.num_spec = (
            self.speculative_config.num_speculative_tokens
            if self.speculative_config
            else 0
        )
        self.gqa_interleaved_layout = gqa_interleaved_layout

        # QKV
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = ColumnParallelLinear(
            input_size=self.conv_kernel_size,
            output_size=self.conv_dim,
            bias=False,
            prefix=f"{prefix}.conv1d",
        )
        self.conv1d.weight.data = self.conv1d.weight.data.unsqueeze(1)

        # projection of the input hidden states
        # Qwen3-Next and Qwen3.5 has a different qkv_proj layout,
        # we need to create qkvz_proj adaptively here.
        # When create_in_proj_qkvz is False (e.g. LoRA enabled in Qwen3.5),
        # in_proj_qkv and in_proj_z are created separately instead.
        if create_in_proj_qkvz:
            self.in_proj_qkvz = self.create_qkvz_proj(
                hidden_size=self.hidden_size,
                key_dim=self.key_dim,
                value_dim=self.value_dim,
                quant_config=quant_config,
                prefix=f"{prefix}.in_proj_qkvz",
            )
        else:
            # LoRA case (Qwen3.5 only): keep q/k/v and z as separate modules
            # so that LoRA adapters can be applied independently.
            self.in_proj_qkv = MergedColumnParallelLinear(
                input_size=self.hidden_size,
                output_sizes=[self.key_dim, self.key_dim, self.value_dim],
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.in_proj_qkv",
            )
            self.in_proj_z = ColumnParallelLinear(
                input_size=self.hidden_size,
                output_size=self.value_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.in_proj_z",
            )
        # ba_proj doesn't support blockwise fp8 quantization.
        # Qwen3-Next and Qwen3.5 have different in_proj_ba checkpoint
        # layouts, so we use a factory method to create the projection.
        self.in_proj_ba = self.create_ba_proj(
            hidden_size=self.hidden_size,
            num_v_heads=self.num_v_heads,
            quant_config=quant_config,
            prefix=f"{prefix}.in_proj_ba",
        )

        query_key_settings = (self.key_dim, 0, False)
        value_settings = (self.value_dim, 0, False)

        self.conv1d.weight.weight_loader = mamba_v2_sharded_weight_loader(
            [
                query_key_settings,
                query_key_settings,
                value_settings,
            ],
            self.tp_size,
            self.tp_rank,
        )

        # selective projection used to make dt, B and C input dependent

        # time step projection (discretization)
        # instantiate once and copy inv_dt in init_weights of PretrainedModel
        self.dt_bias = nn.Parameter(
            torch.ones(self.num_v_heads // self.tp_size),
        )
        self.A_log = nn.Parameter(
            torch.empty(
                divide(self.num_v_heads, self.tp_size),
                dtype=torch.float32,
            )
        )

        set_weight_attrs(self.A_log, {"weight_loader": sharded_weight_loader(0)})
        set_weight_attrs(self.dt_bias, {"weight_loader": sharded_weight_loader(0)})

        self.norm = RMSNormGated(
            self.head_v_dim,
            eps=self.layer_norm_epsilon,
            group_size=None,
            norm_before_gate=True,
            device=current_platform.current_device(),
        )

        self.out_proj = RowParallelLinear(
            self.value_dim,
            self.hidden_size,
            bias=False,
            input_is_parallel=True,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
        )

        self.chunk_gated_delta_rule = ChunkGatedDeltaRule()
        self.enable_packed_recurrent_decode = (
            envs.VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE
        )

        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def create_qkvz_proj(
        self,
        hidden_size: int,
        key_dim: int,
        value_dim: int,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> MergedColumnParallelLinear:
        # When gqa_interleaved_layout=True (Qwen3-Next), qkvz weights are
        # stored as a single fused tensor with interleaved GQA layout, so we
        # use one output shard to preserve the interleaving across TP ranks.
        # When gqa_interleaved_layout=False (Qwen3.5), the checkpoint has
        # separate q, k, v, z weights, so we use 4 independent output sizes.
        output_sizes = (
            [sum((key_dim, key_dim, value_dim, value_dim))]
            if self.gqa_interleaved_layout
            else [key_dim, key_dim, value_dim, value_dim]
        )
        return MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=output_sizes,
            bias=False,
            quant_config=quant_config,
            prefix=prefix,
        )

    def create_ba_proj(
        self,
        hidden_size: int,
        num_v_heads: int,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> MergedColumnParallelLinear:
        # When gqa_interleaved_layout=True (Qwen3-Next), in_proj_ba is stored
        # as a single fused weight [b_g0, a_g0, b_g1, a_g1, ...] interleaved
        # by key-head group; a single output shard preserves this across TP.
        # When gqa_interleaved_layout=False (Qwen3.5), in_proj_b and in_proj_a
        # are separate checkpoint weights, so we use 2 independent output sizes.
        output_sizes = (
            [num_v_heads * 2] if self.gqa_interleaved_layout else [num_v_heads] * 2
        )
        return MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=output_sizes,
            bias=False,
            quant_config=quant_config,
            prefix=prefix,
        )

    def fix_query_key_value_ordering(
        self,
        mixed_qkvz: torch.Tensor,
        mixed_ba: torch.Tensor,
    ):
        """
        Derives `query`, `key` and `value` tensors from `mixed_qkvzba`.
        """
        new_tensor_shape_qkvz = mixed_qkvz.size()[:-1] + (
            self.num_k_heads // self.tp_size,
            (
                self.head_k_dim
                + self.head_k_dim
                + (self.head_v_dim + self.head_v_dim)
                * self.num_v_heads
                // self.num_k_heads
            ),
        )
        new_tensor_shape_ba = mixed_ba.size()[:-1] + (
            self.num_k_heads // self.tp_size,
            2 * self.num_v_heads // self.num_k_heads,
        )

        mixed_qkvz = mixed_qkvz.view(*new_tensor_shape_qkvz)
        mixed_ba = mixed_ba.view(*new_tensor_shape_ba)

        split_arg_list_qkvz = [
            self.head_k_dim,
            self.head_k_dim,
            (self.num_v_heads // self.num_k_heads * self.head_v_dim),
            (self.num_v_heads // self.num_k_heads * self.head_v_dim),
        ]
        split_arg_list_ba = [
            self.num_v_heads // self.num_k_heads,
            self.num_v_heads // self.num_k_heads,
        ]

        # [b, sq, ng, (hn + hn + np/ng * hn + np/ng + np/ng)]
        # --> [b, sq, ng, hn], [b, sq, ng, hn], [b, sq, ng, np/ng * hn],
        #  [b, sq, ng, np/ng * hn], [b, sq, ng, np/ng], [b, sq, ng, np/ng]
        (query, key, value, z) = torch.split(mixed_qkvz, split_arg_list_qkvz, dim=2)
        (b, a) = torch.split(mixed_ba, split_arg_list_ba, dim=2)

        # [b, sq, ng, np/ng * hn] -> [b, sq, np, hn]
        value = value.reshape(value.size(0), -1, self.head_v_dim)
        z = z.reshape(z.size(0), -1, self.head_v_dim)
        b = b.reshape(b.size(0), self.num_v_heads // self.tp_size)
        a = a.reshape(a.size(0), self.num_v_heads // self.tp_size)

        return query, key, value, z, b, a

    def rearrange_mixed_qkv(self, mixed_qkv):
        if mixed_qkv is None:
            return None, None, None
        query, key, value = torch.split(
            mixed_qkv,
            [
                self.key_dim // self.tp_size,
                self.key_dim // self.tp_size,
                self.value_dim // self.tp_size,
            ],
            dim=-1,
        )
        query, key = map(
            lambda x: rearrange(x, "l (h d) -> 1 l h d", d=self.head_k_dim),
            (query, key),
        )
        value = rearrange(value, "l (h d) -> 1 l h d", d=self.head_v_dim)
        return query.contiguous(), key.contiguous(), value.contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ):
        """
        Forward pass with three parts:
        1. Input projection
        2. Core attention (custom op)
        3. Output projection
        """
        num_tokens = hidden_states.size(0)
        # ============================================================
        # Part 1: Input Projection
        # ============================================================
        if hasattr(self, "in_proj_qkv"):
            # LoRA path (Qwen3.5 only): separate in_proj_qkv and in_proj_z
            mixed_qkv, _ = self.in_proj_qkv(hidden_states)
            ba, _ = self.in_proj_ba(hidden_states)
            z, _ = self.in_proj_z(hidden_states)
            z = z.reshape(z.size(0), -1, self.head_v_dim)
            b, a = ba.chunk(2, dim=-1)
            b = b.contiguous()
            a = a.contiguous()
        else:
            mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
            ba, _ = self.in_proj_ba(hidden_states)

            if self.gqa_interleaved_layout:
                # Qwen3-Next: unpack the interleaved GQA layout
                query, key, value, z, b, a = self.fix_query_key_value_ordering(
                    mixed_qkvz, ba
                )
                query, key, value = map(
                    lambda x: rearrange(x, "l p d -> l (p d)"), (query, key, value)
                )
                mixed_qkv = torch.cat((query, key, value), dim=-1)
            else:
                # Qwen3.5: weights are already in [q, k, v, z] and [b, a] order
                qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size
                z_size = self.value_dim // self.tp_size
                mixed_qkv, z = mixed_qkvz.split([qkv_size, z_size], dim=-1)
                z = z.reshape(z.size(0), -1, self.head_v_dim)
                b, a = ba.chunk(2, dim=-1)
                b = b.contiguous()
                a = a.contiguous()

        # ============================================================
        # Part 2: Core Attention (Custom Op)
        # ============================================================
        # Note: we should not use torch.empty here like other attention backends,
        # see discussions in https://github.com/vllm-project/vllm/pull/28182
        core_attn_out = torch.zeros(
            (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        torch.ops.vllm.gdn_attention_core(
            mixed_qkv,
            b,
            a,
            core_attn_out,
            self.prefix,
        )

        # ============================================================
        # Part 3: Output Projection
        # ============================================================
        z_shape_og = z.shape
        # Reshape input data into 2D tensor
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = rearrange(core_attn_out, "... h d -> ... (h d)")
        output[:num_tokens], _ = self.out_proj(core_attn_out)

    def _warmup_prefill_kernels(self, mixed_qkv: torch.Tensor) -> None:
        """Warm up GDN prefill kernels during V1 profiling.

        During V1 profile runs, ``_forward_core`` returns early because
        ``attn_metadata`` is ``None``, so the autotuned kernels used by
        ``chunk_gated_delta_rule`` (e.g. ``solve_tril``,
        ``chunk_scaled_dot_kkt``) are never invoked.  After profiling,
        vLLM allocates KV cache using most of the remaining GPU memory.
        When the first real inference triggers the autotuner it OOMs
        because there is not enough memory left for benchmarking.

        This method runs minimal forward passes through
        ``chunk_gated_delta_rule`` with small dummy tensors to force
        autotuning while GPU memory is still plentiful.  The autotuner
        results are cached globally, so only the first layer incurs
        actual benchmarking cost.

        Most kernels use a fixed ``BT = chunk_size`` (64), but
        ``chunk_fwd_kernel_o`` recomputes ``BT`` from the sequence
        length: ``min(64, max(16, next_power_of_2(T)))``.  Since ``BT``
        is part of its autotune key, we run warmup passes with T = 16,
        32, and 64 to cover all possible ``BT`` values.

        The decode path uses ``fused_sigmoid_gating_delta_rule_update``
        which has fixed kernel parameters (no autotuning), so only the
        prefill (chunked) path needs warming up.
        """
        if hasattr(self, "_prefill_kernels_warmed_up"):
            return
        self._prefill_kernels_warmed_up = True

        device = mixed_qkv.device
        dtype = mixed_qkv.dtype
        num_k_heads = self.num_k_heads // self.tp_size
        num_v_heads = self.num_v_heads // self.tp_size
        _, state_dtype = self.get_state_dtype()

        # Run warmup for each possible BT value of chunk_fwd_kernel_o:
        #   T=16 → BT=16, T=32 → BT=32, T=64 → BT=64.
        # Other kernels always use BT=chunk_size(64), so their autotune
        # cache is populated on the first pass and reused thereafter.
        for T in (16, 32, 64):
            q = torch.randn(
                1, T, num_k_heads, self.head_k_dim, device=device, dtype=dtype
            )
            k = torch.randn(
                1, T, num_k_heads, self.head_k_dim, device=device, dtype=dtype
            )
            v = torch.randn(
                1, T, num_v_heads, self.head_v_dim, device=device, dtype=dtype
            )
            # NOTE: g and beta must have the same dtypes as during
            # inference, so we construct them with the same function
            # (fused_gdn_gating). dummy_a and dummy_b are throwaway
            # inputs required by that function.
            dummy_a = torch.randn(T, num_v_heads, device=device, dtype=dtype)
            dummy_b = torch.randn(T, num_v_heads, device=device, dtype=dtype)
            g, beta = fused_gdn_gating(self.A_log, dummy_a, dummy_b, self.dt_bias)
            state = torch.zeros(
                1,
                num_v_heads,
                self.head_v_dim,
                self.head_k_dim,
                device=device,
                dtype=state_dtype,
            )
            cu_seqlens = torch.tensor([0, T], device=device, dtype=torch.int32)

            try:
                self.chunk_gated_delta_rule(
                    q=q,
                    k=k,
                    v=v,
                    g=g,
                    beta=beta,
                    initial_state=state,
                    output_final_state=True,
                    cu_seqlens=cu_seqlens,
                    use_qk_l2norm_in_kernel=True,
                )
            except Exception:
                logger.warning(
                    "GDN prefill kernel warmup (T=%d) failed for "
                    "layer %s. First inference may OOM due to "
                    "autotuner.",
                    T,
                    self.prefix,
                    exc_info=True,
                )
            else:
                logger.debug(
                    "GDN prefill kernel warmup (T=%d) completed for layer %s",
                    T,
                    self.prefix,
                )
            finally:
                del q, k, v, dummy_a, dummy_b, g, beta, state, cu_seqlens

        torch.accelerator.empty_cache()

    def _forward_core(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        core_attn_out: torch.Tensor,
    ):
        forward_context = get_forward_context()
        attn_metadata: AttentionMetadata = forward_context.attn_metadata

        if attn_metadata is None:
            # V1 profile run — warm up prefill kernels so that
            # autotuning completes before KV cache allocation.
            self._warmup_prefill_kernels(mixed_qkv)
            return

        assert isinstance(attn_metadata, dict)
        attn_metadata = attn_metadata[self.prefix]
        assert isinstance(attn_metadata, GDNAttentionMetadata)

        # Full automatic prefix caching (APC) needs special metadata to be tracked.
        # Check this before the fast path so we can exclude APC decode from it.
        state_indices_tensor = attn_metadata.state_indices_tensor
        block_idx_last_scheduled_token = attn_metadata.block_idx_last_scheduled_token
        block_idx_last_computed_token = attn_metadata.block_idx_last_computed_token
        prefix_caching_enabled = bool(
            state_indices_tensor is not None
            and block_idx_last_scheduled_token is not None
            and block_idx_last_computed_token is not None
        )

        if (
            not prefix_caching_enabled
            and self.enable_packed_recurrent_decode
            and attn_metadata.spec_sequence_masks is None
            and attn_metadata.num_prefills == 0
            and attn_metadata.num_decodes > 0
        ):
            return self._forward_core_decode_non_spec(
                mixed_qkv=mixed_qkv,
                b=b,
                a=a,
                core_attn_out=core_attn_out,
                attn_metadata=attn_metadata,
            )

        has_initial_state = attn_metadata.has_initial_state
        spec_query_start_loc = attn_metadata.spec_query_start_loc
        non_spec_query_start_loc = attn_metadata.non_spec_query_start_loc
        spec_sequence_masks = attn_metadata.spec_sequence_masks
        spec_token_indx = attn_metadata.spec_token_indx
        non_spec_token_indx = attn_metadata.non_spec_token_indx
        spec_state_indices_tensor = attn_metadata.spec_state_indices_tensor  # noqa: E501
        non_spec_state_indices_tensor = attn_metadata.non_spec_state_indices_tensor  # noqa: E501
        self_kv_cache = self.kv_cache
        conv_state = self_kv_cache[0].transpose(-1, -2)
        ssm_state = self_kv_cache[1]
        num_actual_tokens = attn_metadata.num_actual_tokens
        num_accepted_tokens = attn_metadata.num_accepted_tokens

        if prefix_caching_enabled:
            # Set up APC-specific state
            block_idx_first_scheduled_token_p = (
                attn_metadata.block_idx_first_scheduled_token_p
            )
            num_computed_tokens_p = attn_metadata.num_computed_tokens_p
            block_size = attn_metadata.block_size
            chunk_size = attn_metadata.chunk_size
            num_decodes = attn_metadata.num_decodes
            num_prefills = attn_metadata.num_prefills
            start_non_spec_prefill = num_decodes
            end_non_spec_prefill = start_non_spec_prefill + num_prefills

            state_indices_tensor_d, state_indices_tensor_p = torch.split(
                state_indices_tensor,
                [num_decodes, num_prefills],
                dim=0,
            )
            (
                block_idx_last_computed_token_d,
                block_idx_last_computed_token_p,
            ) = torch.split(
                block_idx_last_computed_token,
                [num_decodes, num_prefills],
                dim=0,
            )
            (
                block_idx_last_scheduled_token_d,
                block_idx_last_scheduled_token_p,
            ) = torch.split(
                block_idx_last_scheduled_token,
                [num_decodes, num_prefills],
                dim=0,
            )

            state_indices_decode = None
            ssm_state_indices_decode = None
            state_indices_prefill = None

            if num_decodes > 0:
                state_indices_decode = state_indices_tensor_d.gather(
                    1, block_idx_last_scheduled_token_d.unsqueeze(1)
                ).squeeze(1)
                ssm_state_indices_decode = state_indices_tensor_d.gather(
                    1, block_idx_last_computed_token_d.unsqueeze(1)
                ).squeeze(1)

            if num_prefills > 0:
                state_indices_prefill = state_indices_tensor_p.gather(
                    1, block_idx_last_scheduled_token_p.unsqueeze(1)
                ).squeeze(1)

        mixed_qkv = mixed_qkv[:num_actual_tokens]
        b = b[:num_actual_tokens]
        a = a[:num_actual_tokens]

        # 1. Convolution sequence transformation
        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )

        if spec_sequence_masks is not None:
            if attn_metadata.num_prefills == 0 and attn_metadata.num_decodes == 0:
                mixed_qkv_spec = mixed_qkv
                mixed_qkv_non_spec = None
            else:
                mixed_qkv_spec = mixed_qkv.index_select(0, spec_token_indx)
                mixed_qkv_non_spec = mixed_qkv.index_select(0, non_spec_token_indx)
        else:
            mixed_qkv_spec = None
            mixed_qkv_non_spec = mixed_qkv

        # 1.1: Process the multi-query part
        if spec_sequence_masks is not None:
            assert spec_state_indices_tensor is not None
            mixed_qkv_spec = causal_conv1d_update(
                mixed_qkv_spec,
                conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=spec_state_indices_tensor[:, 0][
                    : attn_metadata.num_spec_decodes
                ],
                num_accepted_tokens=num_accepted_tokens,
                query_start_loc=spec_query_start_loc,
                max_query_len=spec_state_indices_tensor.size(-1),
                validate_data=False,
            )

        # 1.2: Process the remaining part
        if prefix_caching_enabled:
            # When APC is enabled, decode and prefill tokens must be processed
            # with separate kernels then concatenated back together.
            mixed_qkv_decode = None
            mixed_qkv_prefill = None

            if attn_metadata.num_decodes > 0:
                assert mixed_qkv_non_spec is not None
                assert state_indices_tensor_d is not None
                assert block_idx_last_scheduled_token_d is not None
                assert block_idx_last_computed_token_d is not None
                mixed_qkv_decode = mixed_qkv_non_spec[: attn_metadata.num_decode_tokens]
                mixed_qkv_decode = causal_conv1d_update(
                    mixed_qkv_decode,
                    conv_state,
                    conv_weights,
                    self.conv1d.bias,
                    self.activation,
                    conv_state_indices=state_indices_tensor_d,
                    block_idx_last_scheduled_token=block_idx_last_scheduled_token_d,
                    initial_state_idx=block_idx_last_computed_token_d,
                    validate_data=True,
                )

            if attn_metadata.num_prefills > 0:
                assert mixed_qkv_non_spec is not None
                assert has_initial_state is not None
                assert non_spec_query_start_loc is not None
                assert state_indices_tensor_p is not None
                assert block_size is not None

                mixed_qkv_prefill = mixed_qkv_non_spec[
                    attn_metadata.num_decode_tokens :
                ]
                mixed_qkv_prefill_T = mixed_qkv_prefill.transpose(0, 1)

                # For mixed batches, slice tensors to only include prefills
                has_initial_state_prefill = has_initial_state[
                    num_decodes : num_decodes + num_prefills
                ]
                query_start_loc_prefill = non_spec_query_start_loc[
                    num_decodes : num_decodes + num_prefills + 1
                ]
                # Adjust to start at 0 since we sliced mixed_qkv for prefill-only
                if query_start_loc_prefill[0] != 0:
                    query_start_loc_prefill = (
                        query_start_loc_prefill - query_start_loc_prefill[0]
                    )

                mixed_qkv_prefill = causal_conv1d_fn(
                    mixed_qkv_prefill_T,
                    conv_weights,
                    self.conv1d.bias,
                    activation=self.activation,
                    conv_states=conv_state,
                    has_initial_state=has_initial_state_prefill,
                    cache_indices=state_indices_tensor_p,
                    query_start_loc=query_start_loc_prefill,
                    block_idx_first_scheduled_token=block_idx_first_scheduled_token_p,
                    block_idx_last_scheduled_token=block_idx_last_scheduled_token_p,
                    initial_state_idx=block_idx_last_computed_token_p,
                    block_size_to_align=block_size,
                    num_computed_tokens=num_computed_tokens_p,
                    metadata=None,
                ).transpose(0, 1)

            # Concatenate decode and prefill conv results back together
            if mixed_qkv_decode is not None and mixed_qkv_prefill is not None:
                mixed_qkv_non_spec = torch.cat(
                    [mixed_qkv_decode, mixed_qkv_prefill], dim=0
                )
            elif mixed_qkv_decode is not None:
                mixed_qkv_non_spec = mixed_qkv_decode
            elif mixed_qkv_prefill is not None:
                mixed_qkv_non_spec = mixed_qkv_prefill
            else:
                mixed_qkv_non_spec = None
        # Otherwise process using standard non-full-mode APC
        elif attn_metadata.num_prefills > 0:
            assert mixed_qkv_non_spec is not None
            mixed_qkv_non_spec_T = mixed_qkv_non_spec.transpose(0, 1)
            # - "cache_indices" updates the conv_state cache in positions
            #   pointed to by "state_indices_tensor"
            mixed_qkv_non_spec = causal_conv1d_fn(
                mixed_qkv_non_spec_T,
                conv_weights,
                self.conv1d.bias,
                activation=self.activation,
                conv_states=conv_state,
                has_initial_state=has_initial_state,
                cache_indices=non_spec_state_indices_tensor,
                query_start_loc=non_spec_query_start_loc,
                metadata=attn_metadata,
            ).transpose(0, 1)
        elif attn_metadata.num_decodes > 0:
            assert mixed_qkv_non_spec is not None
            mixed_qkv_non_spec = causal_conv1d_update(
                mixed_qkv_non_spec,
                conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=non_spec_state_indices_tensor[
                    : attn_metadata.num_actual_tokens
                ],
                validate_data=True,
            )
        else:
            mixed_qkv_non_spec = None

        query_spec, key_spec, value_spec = self.rearrange_mixed_qkv(mixed_qkv_spec)
        query_non_spec, key_non_spec, value_non_spec = self.rearrange_mixed_qkv(
            mixed_qkv_non_spec
        )

        if attn_metadata.num_prefills > 0:
            g, beta = fused_gdn_gating(self.A_log, a, b, self.dt_bias)
            if spec_sequence_masks is not None:
                g_non_spec = g.index_select(1, non_spec_token_indx)
                beta_non_spec = beta.index_select(1, non_spec_token_indx)
            else:
                g_non_spec = g
                beta_non_spec = beta
        else:
            g_non_spec = None
            beta_non_spec = None

        # 2. Recurrent attention

        # 2.1: Process the multi-query part
        if spec_sequence_masks is not None:
            core_attn_out_spec, last_recurrent_state = (
                fused_sigmoid_gating_delta_rule_update(
                    A_log=self.A_log,
                    a=a,
                    b=b,
                    dt_bias=self.dt_bias,
                    q=query_spec,
                    k=key_spec,
                    v=value_spec,
                    initial_state=ssm_state,
                    inplace_final_state=True,
                    cu_seqlens=spec_query_start_loc[
                        : attn_metadata.num_spec_decodes + 1
                    ],
                    ssm_state_indices=spec_state_indices_tensor,
                    num_accepted_tokens=num_accepted_tokens,
                    use_qk_l2norm_in_kernel=True,
                )
            )
        else:
            core_attn_out_spec, last_recurrent_state = None, None

        # 2.2: Process the remaining part
        if prefix_caching_enabled:
            if num_prefills > 0:
                # Handle mixed batches (decode + prefill) and prefill-only.
                # We always use the FLA kernel for prefill during all-mode prefix
                # caching because:
                # 1) only the FLA kernel outputs intermediate states (FI does not)
                # 2) the FLA kernel lets us set ssm_state dtype to avoid numerical
                #    discrepancies vs the non-APC path.
                assert state_indices_tensor_p is not None
                assert block_idx_last_computed_token_p is not None
                block_state_indices = state_indices_tensor_p.gather(
                    1, block_idx_last_computed_token_p.unsqueeze(1)
                ).squeeze(1)

                assert query_non_spec is not None
                assert key_non_spec is not None
                assert value_non_spec is not None
                assert g_non_spec is not None
                assert beta_non_spec is not None
                assert non_spec_query_start_loc is not None
                cu_seqlens = non_spec_query_start_loc[: end_non_spec_prefill + 1]
                assert num_decodes + num_prefills == len(cu_seqlens) - 1

                # Build initial state for all sequences (decode + prefill)
                initial_state = ssm_state.new_zeros(
                    (num_decodes + num_prefills, *ssm_state.shape[1:])
                )

                # For decode sequences (if any), copy their existing state
                if num_decodes > 0:
                    assert ssm_state_indices_decode is not None
                    _copy_state_to_dest_from_pool(
                        dest=initial_state,
                        pool=ssm_state,
                        pool_indices=ssm_state_indices_decode,
                    )

                # For prefill sequences, copy cached state from block indices
                if block_state_indices.numel() > 0:
                    _copy_state_to_dest_from_pool(
                        dest=initial_state,
                        pool=ssm_state,
                        pool_indices=block_state_indices,
                        dest_offset=num_decodes,
                    )

                    if has_initial_state is not None:
                        req_has_initial_state = has_initial_state[
                            start_non_spec_prefill:end_non_spec_prefill
                        ]
                        prefill_indices = torch.arange(
                            num_decodes,
                            num_decodes + num_prefills,
                            device=initial_state.device,
                        )
                        initial_state[prefill_indices[~req_has_initial_state], ...] = 0

                (
                    core_attn_out_non_spec,
                    last_recurrent_state,
                    chunk_state_history,
                ) = self.chunk_gated_delta_rule(
                    q=query_non_spec,
                    k=key_non_spec,
                    v=value_non_spec,
                    g=g_non_spec,
                    beta=beta_non_spec,
                    initial_state=initial_state,
                    output_final_state=True,
                    cu_seqlens=cu_seqlens,
                    use_qk_l2norm_in_kernel=True,
                    return_intermediate_states=True,
                    state_dtype=ssm_state.dtype,
                )

                # Write the last recurrent state for prefill requests
                assert state_indices_prefill is not None
                _copy_state_to_pool_from_src(
                    pool=ssm_state,
                    pool_slot_indices=state_indices_prefill,
                    src=last_recurrent_state,
                    src_offset=num_decodes,
                )

                # Write the last recurrent state for decode requests (if any)
                if num_decodes > 0:
                    assert state_indices_decode is not None
                    decode_states = last_recurrent_state[:num_decodes]
                    valid_decode_slots = state_indices_decode >= 0
                    dest_slots = state_indices_decode.clamp(min=0).to(
                        device=ssm_state.device, dtype=torch.long
                    )
                    valid_decode_slots_broadcast = valid_decode_slots.view(
                        -1,
                        *([1] * (last_recurrent_state.dim() - 1)),
                    )
                    prior_state = ssm_state.index_select(0, dest_slots).to(
                        last_recurrent_state.dtype
                    )
                    decode_states = torch.where(
                        valid_decode_slots_broadcast,
                        decode_states,
                        prior_state,
                    )
                    ssm_state.index_copy_(
                        0,
                        dest_slots,
                        decode_states.to(ssm_state.dtype),
                    )

                # Write intermediate states (one per block) to the ssm state pool
                assert chunk_state_history is not None
                assert chunk_state_history.numel() > 0
                assert block_idx_first_scheduled_token_p is not None
                assert block_idx_last_scheduled_token_p is not None
                assert state_indices_tensor_p is not None
                assert attn_metadata.last_chunk_indices_p is not None
                assert num_computed_tokens_p is not None
                assert chunk_size is not None
                assert block_size is not None

                chunk_history = chunk_state_history.to(ssm_state.dtype)
                total_chunks = chunk_history.shape[0]
                last_chunk_indices = attn_metadata.last_chunk_indices_p
                prefill_chunk_count = (
                    int(last_chunk_indices[-1].item()) + 1
                    if (
                        last_chunk_indices is not None
                        and last_chunk_indices.numel() > 0
                    )
                    else 0
                )
                decode_chunk_count = max(total_chunks - prefill_chunk_count, 0)
                chunk_history_prefill = chunk_history[decode_chunk_count:]
                if chunk_history_prefill.shape[0] > 0:
                    chunks_per_block = block_size // chunk_size
                    for seq_idx in range(num_prefills):
                        block_first = int(
                            block_idx_first_scheduled_token_p[seq_idx].item()
                        )
                        block_last = int(
                            block_idx_last_scheduled_token_p[seq_idx].item()
                        )
                        n_blocks_to_fill = block_last - block_first
                        if n_blocks_to_fill <= 0:
                            continue

                        cache_blocks = state_indices_tensor_p[
                            seq_idx, block_first:block_last
                        ].to(torch.long)

                        first_chunk = (
                            0
                            if seq_idx == 0
                            else int(last_chunk_indices[seq_idx - 1].item()) + 1
                        )
                        # h[i] = state BEFORE chunk i = state AFTER chunks 0..i-1
                        # State after a full block of chunks_per_block chunks
                        # = h[first_chunk + chunks_per_block]
                        first_aligned_chunk = first_chunk + chunks_per_block
                        num_unaligned_tokens = int(
                            num_computed_tokens_p[seq_idx].item() % block_size
                        )
                        if num_unaligned_tokens > 0:
                            first_aligned_chunk -= num_unaligned_tokens // chunk_size
                        chunk_stop = (
                            first_aligned_chunk + n_blocks_to_fill * chunks_per_block
                        )
                        cached_states = chunk_history_prefill[
                            first_aligned_chunk:chunk_stop:chunks_per_block
                        ]
                        ssm_state[cache_blocks] = cached_states

            elif num_decodes > 0:
                # APC decode-only path
                assert non_spec_query_start_loc is not None
                assert ssm_state_indices_decode is not None
                valid_src_decode_slots = ssm_state_indices_decode >= 0
                ssm_state_indices_decode_clamped = ssm_state_indices_decode.clamp(min=0)
                initial_state_decode = ssm_state.index_select(
                    0,
                    ssm_state_indices_decode_clamped.to(torch.long),
                )
                initial_state_decode[~valid_src_decode_slots] = 0

                # Compute g/beta for the decode tokens
                g_decode, beta_decode = fused_gdn_gating(self.A_log, a, b, self.dt_bias)
                if spec_sequence_masks is not None:
                    g_decode = g_decode.index_select(1, non_spec_token_indx)
                    beta_decode = beta_decode.index_select(1, non_spec_token_indx)

                core_attn_out_non_spec, last_recurrent_state = (
                    fused_recurrent_gated_delta_rule(
                        q=query_non_spec,
                        k=key_non_spec,
                        v=value_non_spec,
                        g=g_decode,
                        beta=beta_decode,
                        initial_state=initial_state_decode,
                        inplace_final_state=False,
                        cu_seqlens=non_spec_query_start_loc[: num_decodes + 1],
                        ssm_state_indices=None,
                        use_qk_l2norm_in_kernel=True,
                    )
                )

                # Store the final recurrent state manually
                assert state_indices_decode is not None
                valid_decode_slots = state_indices_decode >= 0
                dest_slots = state_indices_decode.clamp(min=0).to(
                    device=ssm_state.device, dtype=torch.long
                )
                valid_decode_slots_broadcast = valid_decode_slots.view(
                    -1,
                    *([1] * (last_recurrent_state.dim() - 1)),
                )
                prior_state = ssm_state.index_select(0, dest_slots).to(
                    last_recurrent_state.dtype
                )
                last_recurrent_state = torch.where(
                    valid_decode_slots_broadcast,
                    last_recurrent_state,
                    prior_state,
                )
                ssm_state.index_copy_(
                    0,
                    dest_slots,
                    last_recurrent_state.to(ssm_state.dtype),
                )
            else:
                core_attn_out_non_spec, last_recurrent_state = None, None

        # Otherwise process using standard non-full-mode APC
        elif attn_metadata.num_prefills > 0:
            initial_state = ssm_state[non_spec_state_indices_tensor].contiguous()
            initial_state[~has_initial_state, ...] = 0
            (
                core_attn_out_non_spec,
                last_recurrent_state,
            ) = self.chunk_gated_delta_rule(
                q=query_non_spec,
                k=key_non_spec,
                v=value_non_spec,
                g=g_non_spec,
                beta=beta_non_spec,
                initial_state=initial_state,
                output_final_state=True,
                cu_seqlens=non_spec_query_start_loc,
                use_qk_l2norm_in_kernel=True,
                state_dtype=ssm_state.dtype,
            )
            # Init cache
            ssm_state[non_spec_state_indices_tensor] = last_recurrent_state.to(
                ssm_state.dtype
            )
        elif attn_metadata.num_decodes > 0:
            core_attn_out_non_spec, last_recurrent_state = (
                fused_sigmoid_gating_delta_rule_update(
                    A_log=self.A_log,
                    a=a,
                    b=b,
                    dt_bias=self.dt_bias,
                    q=query_non_spec,
                    k=key_non_spec,
                    v=value_non_spec,
                    initial_state=ssm_state,
                    inplace_final_state=True,
                    cu_seqlens=non_spec_query_start_loc[
                        : attn_metadata.num_decodes + 1
                    ],
                    ssm_state_indices=non_spec_state_indices_tensor,
                    use_qk_l2norm_in_kernel=True,
                )
            )
        else:
            core_attn_out_non_spec, last_recurrent_state = None, None

        # 3. Merge core attention output
        if spec_sequence_masks is not None and core_attn_out_non_spec is not None:
            merged_out = torch.empty(
                (1, num_actual_tokens, *core_attn_out_spec.shape[2:]),
                dtype=core_attn_out_non_spec.dtype,
                device=core_attn_out_non_spec.device,
            )
            merged_out.index_copy_(1, spec_token_indx, core_attn_out_spec)
            merged_out.index_copy_(1, non_spec_token_indx, core_attn_out_non_spec)
            core_attn_out[:num_actual_tokens] = merged_out.squeeze(0)
        elif spec_sequence_masks is not None:
            core_attn_out[:num_actual_tokens] = core_attn_out_spec.squeeze(0)
        else:
            core_attn_out[:num_actual_tokens] = core_attn_out_non_spec.squeeze(0)

    def _forward_core_decode_non_spec(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        core_attn_out: torch.Tensor,
        attn_metadata: GDNAttentionMetadata,
    ):
        """
        Core attention computation with a packed non-spec decode fast path.
        """
        non_spec_state_indices_tensor = attn_metadata.non_spec_state_indices_tensor  # noqa: E501
        self_kv_cache = self.kv_cache
        conv_state = self_kv_cache[0].transpose(-1, -2)
        ssm_state = self_kv_cache[1]
        num_actual_tokens = attn_metadata.num_actual_tokens

        mixed_qkv = mixed_qkv[:num_actual_tokens]
        b = b[:num_actual_tokens]
        a = a[:num_actual_tokens]

        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )
        mixed_qkv_non_spec = causal_conv1d_update(
            mixed_qkv,
            conv_state,
            conv_weights,
            self.conv1d.bias,
            self.activation,
            conv_state_indices=non_spec_state_indices_tensor[:num_actual_tokens],
            validate_data=False,
        )
        out_buf = core_attn_out[:num_actual_tokens].unsqueeze(1)
        fused_recurrent_gated_delta_rule_packed_decode(
            mixed_qkv=mixed_qkv_non_spec,
            a=a,
            b=b,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            scale=self.head_k_dim**-0.5,
            initial_state=ssm_state,
            out=out_buf,
            ssm_state_indices=non_spec_state_indices_tensor[:num_actual_tokens],
            use_qk_l2norm_in_kernel=True,
        )
        return


def gdn_attention_core(
    mixed_qkv: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    core_attn_out: torch.Tensor,
    layer_name: str,
) -> None:
    """
    Custom op for the core attention computation.
    Only handles the convolution + recurrent attention part.
    Input/output projections are handled outside this op.
    """
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    self._forward_core(
        mixed_qkv=mixed_qkv,
        b=b,
        a=a,
        core_attn_out=core_attn_out,
    )


def gdn_attention_core_fake(
    mixed_qkv: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    core_attn_out: torch.Tensor,
    layer_name: str,
) -> None:
    """Fake implementation for torch.compile."""
    return


direct_register_custom_op(
    op_name="gdn_attention_core",
    op_func=gdn_attention_core,
    mutates_args=["core_attn_out"],
    fake_impl=gdn_attention_core_fake,
)


@triton.jit
def fused_gdn_gating_kernel(
    g,
    beta_output,
    A_log,
    a,
    b,
    dt_bias,
    seq_len,
    NUM_HEADS: tl.constexpr,
    beta: tl.constexpr,
    threshold: tl.constexpr,
    BLK_HEADS: tl.constexpr,
):
    i_b, i_s, i_d = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    head_off = i_d * BLK_HEADS + tl.arange(0, BLK_HEADS)
    off = i_b * seq_len * NUM_HEADS + i_s * NUM_HEADS + head_off
    mask = head_off < NUM_HEADS
    blk_A_log = tl.load(A_log + head_off, mask=mask)
    blk_a = tl.load(a + off, mask=mask)
    blk_b = tl.load(b + off, mask=mask)
    blk_bias = tl.load(dt_bias + head_off, mask=mask)
    # If the model is loaded in fp16, without the .float() here, A might be -inf
    x = blk_a.to(tl.float32) + blk_bias.to(tl.float32)
    softplus_x = tl.where(
        beta * x <= threshold, (1 / beta) * tl.log(1 + tl.exp(beta * x)), x
    )
    blk_g = -tl.exp(blk_A_log.to(tl.float32)) * softplus_x
    tl.store(g + off, blk_g.to(g.dtype.element_ty), mask=mask)
    # compute beta_output = sigmoid(b)
    blk_beta_output = tl.sigmoid(blk_b.to(tl.float32))
    tl.store(
        beta_output + off, blk_beta_output.to(beta_output.dtype.element_ty), mask=mask
    )


def fused_gdn_gating(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fused computation of g and beta for Gated Delta Net.
    g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
    beta_output = b.sigmoid()
    TODO maybe use torch.compile to replace this triton kernel
    """
    batch, num_heads = a.shape
    seq_len = 1
    grid = (batch, seq_len, triton.cdiv(num_heads, 8))
    g = torch.empty(1, batch, num_heads, dtype=torch.float32, device=a.device)
    beta_output = torch.empty(1, batch, num_heads, dtype=b.dtype, device=b.device)
    fused_gdn_gating_kernel[grid](
        g,
        beta_output,
        A_log,
        a,
        b,
        dt_bias,
        seq_len,
        num_heads,
        beta,
        threshold,
        8,
        num_warps=1,
    )
    return g, beta_output

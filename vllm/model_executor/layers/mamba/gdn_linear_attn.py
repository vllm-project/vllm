# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Qwen3-Next/Qwen3.5 model."""

import torch
from einops import rearrange
from torch import nn
from transformers.activations import ACT2FN

from vllm import envs
from vllm._aiter_ops import rocm_aiter_ops
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
    fused_post_conv_prep,
    fused_recurrent_gated_delta_rule_packed_decode,
    fused_sigmoid_gating_delta_rule_update,
)
from vllm.model_executor.layers.fla.ops.chunk import l2norm_fwd
from vllm.model_executor.layers.fla.ops.utils import FLA_CHUNK_SIZE
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
    is_conv_state_dim_first,
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
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
)
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum

# Optional ROCm AITER Triton kernels for the GDN decode fast-path.
# Availability is checked centrally via rocm_aiter_ops; the actual function
# references are imported here so that they can be called without per-call
# import overhead.
GDN_AITER_TRITON_AVAILABLE = rocm_aiter_ops.are_gdn_triton_kernels_available()

if GDN_AITER_TRITON_AVAILABLE:
    from aiter.ops.triton.causal_conv1d_update_single_token import (
        fused_reshape_causal_conv1d_update_single_token as gdn_aiter_fused_reshape_causal_conv1d_update_single_token,  # noqa: E501
    )
    from aiter.ops.triton.gated_delta_net.fused_rearrange_sigmoid_gdr import (
        fused_rearrange_sigmoid_gated_delta_rule as gdn_aiter_fused_rearrange_sigmoid_gated_delta_rule,  # noqa: E501
    )

logger = init_logger(__name__)


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
        additional_config = get_current_vllm_config().additional_config
        assert isinstance(additional_config, dict)
        backend_cfg = additional_config.get("gdn_prefill_backend", "auto")
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

        if use_flashinfer:
            logger.info_once("Using FlashInfer GDN prefill kernel")
            logger.info_once(
                "FlashInfer GDN prefill kernel is JIT-compiled; first run may "
                "take a while to compile. Set `--gdn-prefill-backend triton` to "
                "avoid JIT compile time.",
            )
        else:
            logger.info_once("Using Triton/FLA GDN prefill kernel")

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
        chunk_indices: torch.Tensor | None = None,
        chunk_offsets: torch.Tensor | None = None,
        use_qk_l2norm_in_kernel: bool = True,
        core_attn_out: torch.Tensor | None = None,
    ):
        o, final_state = fi_chunk_gated_delta_rule(
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
        if core_attn_out is not None:
            o_flat = o.squeeze(0).reshape(-1)
            co_flat = core_attn_out.reshape(-1)
            co_flat[: o_flat.numel()].copy_(o_flat)
        return o, final_state

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
        chunk_indices: torch.Tensor | None = None,
        chunk_offsets: torch.Tensor | None = None,
        use_qk_l2norm_in_kernel: bool = True,
        core_attn_out: torch.Tensor | None = None,
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
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            core_attn_out=core_attn_out,
        )


@PluggableLayer.register("gated_delta_net_attention")
class GatedDeltaNetAttention(PluggableLayer, MambaBase):
    @property
    def mamba_type(self) -> MambaAttentionBackendEnum:
        return MambaAttentionBackendEnum.GDN_ATTN

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
        if current_platform.is_xpu():
            self._forward_method = self.forward_xpu
        elif current_platform.is_cpu():
            from vllm.model_executor.layers.mamba.ops.cpu.gdn_attention import (
                register_cpu_gdn_attention_ops,
            )

            register_cpu_gdn_attention_ops()
            self._forward_method = self.forward_cpu
        elif current_platform.is_rocm():
            self._forward_method = self.forward_hip
        else:
            self._forward_method = self.forward_cuda

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
        self.in_proj_qkvz = self.create_qkvz_proj(
            hidden_size=self.hidden_size,
            key_dim=self.key_dim,
            value_dim=self.value_dim,
            quant_config=quant_config,
            prefix=f"{prefix}.in_proj_qkvz",
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

        output_gate_type = getattr(config, "output_gate_type", "silu")
        if output_gate_type == "swish":
            output_gate_type = "silu"
        assert output_gate_type in ["silu", "swish", "sigmoid"], (
            f"unsupported {output_gate_type=}"
        )

        self.norm = RMSNormGated(
            self.head_v_dim,
            eps=self.layer_norm_epsilon,
            group_size=None,
            norm_before_gate=True,
            activation=output_gate_type,
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
        self._prefill_kernels_warmed_up = False
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

    @torch.compile(fullgraph=True)
    def prepare_gdn_attention_core_inputs(
        self,
        mixed_qkvz: torch.Tensor,
        mixed_ba: torch.Tensor,
        num_tokens: int,
    ):
        """
        Derives mixed_qkv, z, b, a from projected qkvz/ba for the GDN custom op.

        For gqa_interleaved_layout (Qwen3-Next): unpack the interleaved
        [ng, (hk + hk + np/ng*hv + np/ng*hv)] layout into contiguous qkv.
        For non-interleaved layout (Qwen3.5): simple split along last dim.
        """
        if not self.gqa_interleaved_layout:
            # Qwen3.5: weights are in [q, k, v, z] order
            assert num_tokens == mixed_qkvz.shape[0]
            qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size
            z_size = self.value_dim // self.tp_size
            mixed_qkv, z_flat = mixed_qkvz.split([qkv_size, z_size], dim=-1)
            n = mixed_qkvz.shape[0]
            z_out = z_flat.reshape(n, -1, self.head_v_dim)
            b, a = mixed_ba.chunk(2, dim=-1)
            return mixed_qkv, z_out, b, a

        # Qwen3-Next: interleaved GQA layout
        base_shape_qkvz = mixed_qkvz.size()[:-1]
        base_shape_ba = mixed_ba.size()[:-1]
        ng = self.num_k_heads // self.tp_size

        new_tensor_shape_qkvz = base_shape_qkvz + (
            ng,
            (
                self.head_k_dim
                + self.head_k_dim
                + (self.head_v_dim + self.head_v_dim)
                * self.num_v_heads
                // self.num_k_heads
            ),
        )
        new_tensor_shape_ba = base_shape_ba + (
            ng,
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

        (query, key, value, z) = torch.split(mixed_qkvz, split_arg_list_qkvz, dim=-1)
        (b, a) = torch.split(mixed_ba, split_arg_list_ba, dim=-1)

        mixed_qkv_logical = torch.cat(
            [
                query.reshape(num_tokens, -1),
                key.reshape(num_tokens, -1),
                value.reshape(num_tokens, -1),
            ],
            dim=-1,
        )

        # The split above produces non-contiguous views into the interleaved
        # buffer.  Concatenating everything into a single flat tensor forces a
        # contiguous copy, then slicing back out gives contiguous q/k/v/z/b/a
        # tensors that downstream kernels require.  Doing this in one cat+slice
        # keeps torch.compile in a single Triton graph instead of emitting
        # separate copy kernels per tensor.  The original code used
        # rearrange(...).contiguous() on each tensor individually.
        fused = torch.cat(
            [
                mixed_qkv_logical.reshape(-1),
                z.reshape(-1),
                b.reshape(-1),
                a.reshape(-1),
            ],
            dim=0,
        )

        curr = 0
        qkv_numel = mixed_qkv_logical.numel()
        z_numel = z.numel()
        b_numel = b.numel()
        a_numel = a.numel()

        mixed_qkv_out = fused[curr : curr + qkv_numel].view(num_tokens, -1)
        curr += qkv_numel

        z_out = fused[curr : curr + z_numel].view(
            num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim
        )
        curr += z_numel

        b_out = fused[curr : curr + b_numel].view(
            num_tokens, self.num_v_heads // self.tp_size
        )
        curr += b_numel

        a_out = fused[curr : curr + a_numel].view(
            num_tokens, self.num_v_heads // self.tp_size
        )

        return mixed_qkv_out, z_out, b_out, a_out

    def rearrange_mixed_qkv(self, mixed_qkv):
        """Split packed qkv into contiguous (1, seq, heads, dim) tensors.

        The original code used ``rearrange(x, "l (h d) -> 1 l h d", d=...)``
        followed by ``.contiguous()`` on each tensor.  This version flattens
        all three splits into a single buffer via ``torch.cat`` so that
        torch.compile emits one Triton copy kernel instead of three separate
        contiguous() calls.
        """
        if mixed_qkv is None:
            return None, None, None

        seq_len = mixed_qkv.shape[0]
        q_dim = self.key_dim // self.tp_size
        k_dim = self.key_dim // self.tp_size
        v_dim = self.value_dim // self.tp_size

        query, key, value = torch.split(mixed_qkv, [q_dim, k_dim, v_dim], dim=-1)

        fused = torch.cat(
            [query.reshape(-1), key.reshape(-1), value.reshape(-1)], dim=0
        )

        q_size = seq_len * q_dim
        k_size = seq_len * k_dim

        q_contig = fused[0:q_size]
        k_contig = fused[q_size : q_size + k_size]
        v_contig = fused[q_size + k_size :]

        query = q_contig.view(1, seq_len, -1, self.head_k_dim)
        key = k_contig.view(1, seq_len, -1, self.head_k_dim)
        value = v_contig.view(1, seq_len, -1, self.head_v_dim)

        return query, key, value

    def forward(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ):
        self._forward_method(hidden_states, output)

    def _output_projection(
        self,
        core_attn_out: torch.Tensor,
        z: torch.Tensor,
        output: torch.Tensor,
        num_tokens: int,
    ):
        """Part 3: RMSNormGated + output linear projection.

        The RMSNormGated + quant sequence is eligible for fusion
        by the compilation pass when fuse_norm_quant is enabled.
        """
        z_shape_og = z.shape
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = rearrange(core_attn_out, "... h d -> ... (h d)")
        output[:num_tokens], _ = self.out_proj(core_attn_out)

    def forward_hip(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ):
        """ROCm forward using AITER Triton fused projection+attention when
        available, otherwise falling back to the generic CUDA path."""
        if GDN_AITER_TRITON_AVAILABLE:
            num_tokens = hidden_states.size(0)
            projected_states_qkvz, _ = self.in_proj_qkvz(hidden_states)
            projected_states_ba, _ = self.in_proj_ba(hidden_states)
            projected_states_qkvz = projected_states_qkvz.view(num_tokens, -1)
            projected_states_ba = projected_states_ba.view(num_tokens, -1)
            core_attn_out = torch.empty(
                (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            z = torch.empty(
                (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
                dtype=projected_states_qkvz.dtype,
                device=projected_states_qkvz.device,
            )

            torch.ops.vllm.gdn_attention_core(
                projected_states_qkvz,
                projected_states_ba,
                z,
                core_attn_out,
                fast_kernel=True,
                layer_name=_encode_layer_name(self.prefix),
            )

            self._output_projection(core_attn_out, z, output, num_tokens)
        else:
            self.forward_cuda(hidden_states, output)

    def forward_cuda(
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
            fast_kernel=False,
            layer_name=_encode_layer_name(self.prefix),
        )

        # ============================================================
        # Part 3: Output Projection
        # ============================================================
        self._output_projection(core_attn_out, z, output, num_tokens)

    def forward_xpu(
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
        projected_states_qkvz, _ = self.in_proj_qkvz(hidden_states)
        projected_states_ba, _ = self.in_proj_ba(hidden_states)

        # ============================================================
        # Part 2: Core Attention
        # ============================================================
        core_attn_out = torch.zeros(
            (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        z = torch.empty_like(core_attn_out)

        torch.ops.vllm.gdn_attention_core_xpu(
            core_attn_out,
            z,
            projected_states_qkvz,
            projected_states_ba,
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

    def forward_cpu(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ):
        assert not hasattr(self, "in_proj_qkv"), "lora isn't supported on CPU."

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

        num_tokens = hidden_states.size(0)
        core_attn_out = torch.zeros(
            (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        torch.ops.vllm.cpu_gdn_attention_core(
            mixed_qkv,
            b,
            a,
            core_attn_out,
            _encode_layer_name(self.prefix),
        )

        z_shape_og = z.shape
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = rearrange(core_attn_out, "... h d -> ... (h d)")
        output[:num_tokens], _ = self.out_proj(core_attn_out)

    def _warmup_prefill_kernels(self, qkv_or_qkvz: torch.Tensor, v_dim: int) -> None:
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

        All kernels including ``chunk_fwd_kernel_o`` now use a fixed
        ``BT = chunk_size`` (64).  A single warmup pass with T = 64
        is sufficient to populate the autotuner cache.

        The decode path uses ``gdn_aiter_fused_rearrange_sigmoid_gated_delta_rule``
        which has fixed kernel parameters (no autotuning), so only the
        prefill (chunked) path needs warming up.
        """
        if self._prefill_kernels_warmed_up:
            return
        self._prefill_kernels_warmed_up = True

        device = qkv_or_qkvz.device
        dtype = qkv_or_qkvz.dtype
        num_k_heads = self.num_k_heads // self.tp_size
        num_v_heads = self.num_v_heads // self.tp_size
        _, state_dtype = self.get_state_dtype()

        # All kernels use BT = chunk_size, so a single pass with T = chunk_size
        # is sufficient to populate every autotuner cache. Mirror the real
        # prefill path here: build q/k/v/g/beta via fused_post_conv_prep and
        # then run chunk_gated_delta_rule with in-kernel L2 norm disabled.
        T = FLA_CHUNK_SIZE
        dummy_mixed_qkv = torch.randn(
            T, qkv_or_qkvz.shape[-1] - v_dim, device=device, dtype=dtype
        )
        dummy_a = torch.randn(T, num_v_heads, device=device, dtype=dtype)
        dummy_b = torch.randn(T, num_v_heads, device=device, dtype=dtype)
        q, k, v, g, beta = fused_post_conv_prep(
            conv_output=dummy_mixed_qkv,
            a=dummy_a,
            b=dummy_b,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            num_k_heads=num_k_heads,
            head_k_dim=self.head_k_dim,
            head_v_dim=self.head_v_dim,
            apply_l2norm=True,
            output_g_exp=False,
        )
        q = q.unsqueeze(0)
        k = k.unsqueeze(0)
        v = v.unsqueeze(0)
        g = g.unsqueeze(0)
        beta = beta.unsqueeze(0)
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
                use_qk_l2norm_in_kernel=False,
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
            del dummy_mixed_qkv, q, k, v, dummy_a, dummy_b, g, beta, state, cu_seqlens

        torch.accelerator.empty_cache()

    def _forward_core_rocm(
        self,
        qkvz: torch.Tensor,
        ba: torch.Tensor,
        z_out: torch.Tensor,
        core_attn_out: torch.Tensor,
    ):
        """ROCm AITER fast path: conv1d + recurrent attention from packed
        qkvz/ba layout.

        For decode-only (no spec, no prefill), dispatches directly to
        ``_forward_core_decode_fast``.  Otherwise unpacks the packed
        layout and falls through to ``_forward_core``.

        Args:
            qkvz: packed [q, k, v, z] projection (num_tokens, qkvz_dim)
            ba:   packed [b, a] gating vectors    (num_tokens, 2*num_heads)
            z_out: **output** buffer for z        (num_tokens, num_heads,
                   head_dim); mutated in-place.
            core_attn_out: Pre-allocated output buffer for attention results.
        """
        forward_context = get_forward_context()
        attn_metadata_raw = forward_context.attn_metadata

        if attn_metadata_raw is None:
            v_dim = core_attn_out.shape[-1] * core_attn_out.shape[-2]
            self._warmup_prefill_kernels(qkvz, v_dim)
            return

        assert isinstance(attn_metadata_raw, dict)
        attn_metadata = attn_metadata_raw[self.prefix]  # type: ignore[index]
        assert isinstance(attn_metadata, GDNAttentionMetadata)

        if (
            attn_metadata.spec_sequence_masks is None
            and attn_metadata.num_prefills == 0
            and attn_metadata.num_decodes > 0
        ):
            return self._forward_core_decode_fast(
                qkvz=qkvz,
                ba=ba,
                z_out=z_out,
                core_attn_out=core_attn_out,
                attn_metadata=attn_metadata,
            )

        core_attn_out.zero_()
        z_out.zero_()
        num_tokens_all = qkvz.shape[0]
        mixed_qkv, z, b, a = self.prepare_gdn_attention_core_inputs(
            qkvz, ba, num_tokens_all
        )
        z_out[:] = z
        self._forward_core(
            mixed_qkv=mixed_qkv,
            b=b,
            a=a,
            core_attn_out=core_attn_out,
        )

    def _forward_core(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        core_attn_out: torch.Tensor,
    ):
        """Core conv1d + recurrent attention (standard path).

        Args:
            mixed_qkv: packed [q, k, v] projection (num_tokens, qkv_dim)
            b: beta gating vector                   (num_tokens, num_heads)
            a: alpha gating vector                  (num_tokens, num_heads)
            core_attn_out: Pre-allocated output buffer for attention results.
        """
        forward_context = get_forward_context()
        attn_metadata_raw = forward_context.attn_metadata

        if attn_metadata_raw is None:
            self._warmup_prefill_kernels(mixed_qkv, 0)
            return

        assert isinstance(attn_metadata_raw, dict)
        attn_metadata = attn_metadata_raw[self.prefix]  # type: ignore[index]
        assert isinstance(attn_metadata, GDNAttentionMetadata)

        if (
            self.enable_packed_recurrent_decode
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
        # conv_state must be (..., dim, width-1) for the conv kernels.
        # DS layout stores it that way directly; SD layout needs a transpose.
        conv_state = (
            self_kv_cache[0]
            if is_conv_state_dim_first()
            else self_kv_cache[0].transpose(-1, -2)
        )
        ssm_state = self_kv_cache[1]
        num_actual_tokens = attn_metadata.num_actual_tokens
        num_accepted_tokens = attn_metadata.num_accepted_tokens

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
            # spec_state_indices_tensor is always set when spec_sequence_masks is set
            assert spec_state_indices_tensor is not None
            mixed_qkv_spec = causal_conv1d_update(
                mixed_qkv_spec,
                conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=spec_state_indices_tensor[:, 0][  # type: ignore[index]
                    : attn_metadata.num_spec_decodes  # type: ignore[attr-defined]
                ],
                num_accepted_tokens=num_accepted_tokens,
                query_start_loc=spec_query_start_loc,
                max_query_len=spec_state_indices_tensor.size(-1),
                validate_data=False,
            )

        # 1.2: Process the remaining part
        if attn_metadata.num_prefills > 0:
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
                conv_state_indices=non_spec_state_indices_tensor[  # type: ignore[index]
                    : attn_metadata.num_actual_tokens  # type: ignore[attr-defined]
                ],
                validate_data=True,
            )
        else:
            mixed_qkv_non_spec = None

        query_spec, key_spec, value_spec = self.rearrange_mixed_qkv(mixed_qkv_spec)
        if attn_metadata.num_prefills > 0:
            assert mixed_qkv_non_spec is not None, (
                "mixed_qkv_non_spec must be provided for prefill path"
            )
            if spec_sequence_masks is not None:
                a_non_spec = a.index_select(0, non_spec_token_indx)
                b_non_spec = b.index_select(0, non_spec_token_indx)
            else:
                a_non_spec = a
                b_non_spec = b

            (
                query_non_spec,
                key_non_spec,
                value_non_spec,
                g_non_spec,
                beta_non_spec,
            ) = fused_post_conv_prep(
                conv_output=mixed_qkv_non_spec,
                a=a_non_spec,
                b=b_non_spec,
                A_log=self.A_log,
                dt_bias=self.dt_bias,
                num_k_heads=self.num_k_heads // self.tp_size,
                head_k_dim=self.head_k_dim,
                head_v_dim=self.head_v_dim,
                apply_l2norm=True,
                output_g_exp=False,
            )
            query_non_spec = query_non_spec.unsqueeze(0)
            key_non_spec = key_non_spec.unsqueeze(0)
            value_non_spec = value_non_spec.unsqueeze(0)
            g_non_spec = g_non_spec.unsqueeze(0)
            beta_non_spec = beta_non_spec.unsqueeze(0)
        else:
            query_non_spec, key_non_spec, value_non_spec = self.rearrange_mixed_qkv(
                mixed_qkv_non_spec
            )
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
                    cu_seqlens=spec_query_start_loc[  # type: ignore[index]
                        : attn_metadata.num_spec_decodes
                        + 1  # type: ignore[attr-defined]
                    ],
                    ssm_state_indices=spec_state_indices_tensor,
                    num_accepted_tokens=num_accepted_tokens,
                    use_qk_l2norm_in_kernel=True,
                )
            )
        else:
            core_attn_out_spec, last_recurrent_state = None, None

        # 2.2: Process the remaining part
        if attn_metadata.num_prefills > 0:
            assert non_spec_state_indices_tensor is not None
            initial_state = ssm_state[non_spec_state_indices_tensor].contiguous()  # type: ignore[index]
            assert has_initial_state is not None
            initial_state[~has_initial_state, ...] = 0  # type: ignore[operator]
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
                chunk_indices=attn_metadata.chunk_indices,
                chunk_offsets=attn_metadata.chunk_offsets,
                use_qk_l2norm_in_kernel=False,
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
                    cu_seqlens=non_spec_query_start_loc[  # type: ignore[index]
                        : attn_metadata.num_decodes
                        + 1  # type: ignore[attr-defined]
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

    def _forward_core_decode_fast(
        self,
        qkvz: torch.Tensor,
        ba: torch.Tensor,
        z_out: torch.Tensor,
        core_attn_out: torch.Tensor,
        attn_metadata: GDNAttentionMetadata,
    ):
        non_spec_query_start_loc = attn_metadata.non_spec_query_start_loc
        non_spec_state_indices_tensor = attn_metadata.non_spec_state_indices_tensor  # noqa: E501
        self_kv_cache = self.kv_cache
        # conv_state must be (..., dim, width-1) for the conv kernels.
        # DS layout stores it that way directly; SD layout needs a transpose.
        conv_state = (
            self_kv_cache[0]
            if is_conv_state_dim_first()
            else self_kv_cache[0].transpose(-1, -2)
        )
        ssm_state = self_kv_cache[1]

        # 1. Convolution sequence transformation
        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )

        mixed_qkv_non_spec, b, a = (
            gdn_aiter_fused_reshape_causal_conv1d_update_single_token(
                qkvz,
                attn_metadata.num_actual_tokens,
                self.num_k_heads // self.tp_size,
                self.num_v_heads // self.tp_size,
                self.head_k_dim,
                self.head_v_dim,
                ba,
                z_out,
                core_attn_out,
                conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=non_spec_state_indices_tensor[  # type: ignore[index]
                    : attn_metadata.num_actual_tokens
                ],
                validate_data=True,
            )
        )

        # 2. Recurrent attention
        gdn_aiter_fused_rearrange_sigmoid_gated_delta_rule(
            A_log=self.A_log,
            a=a,
            b=b,
            dt_bias=self.dt_bias,
            qkv=mixed_qkv_non_spec,
            key_dim=self.key_dim // self.tp_size,
            value_dim=self.value_dim // self.tp_size,
            head_k_dim=self.head_k_dim,
            head_v_dim=self.head_v_dim,
            initial_state=ssm_state,
            inplace_final_state=True,
            cu_seqlens=non_spec_query_start_loc[: attn_metadata.num_decodes + 1],  # type: ignore[index]
            ssm_state_indices=non_spec_state_indices_tensor,
            use_qk_l2norm_in_kernel=True,
            core_attn_out=core_attn_out.reshape(-1),
        )

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
        # conv_state must be (..., dim, width-1) for the conv kernels.
        # DS layout stores it that way directly; SD layout needs a transpose.
        conv_state = (
            self_kv_cache[0]
            if is_conv_state_dim_first()
            else self_kv_cache[0].transpose(-1, -2)
        )
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
            conv_state_indices=non_spec_state_indices_tensor[:num_actual_tokens],  # type: ignore[index]
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
            ssm_state_indices=non_spec_state_indices_tensor[:num_actual_tokens],  # type: ignore[index]
            use_qk_l2norm_in_kernel=True,
        )
        return


def gdn_attention_core(
    qkv_or_qkvz: torch.Tensor,
    b_or_ba: torch.Tensor,
    a_or_z_out: torch.Tensor,
    core_attn_out: torch.Tensor,
    fast_kernel: bool,
    layer_name: LayerNameType,
) -> None:
    """Custom op dispatching to _forward_core or _forward_core_rocm.

    Handles conv1d + recurrent attention only; input/output projections
    are performed by the caller.

    When ``fast_kernel=False`` (standard path):
        qkv_or_qkvz is [q, k, v], b_or_ba is b, a_or_z_out is a (read-only).
    When ``fast_kernel=True`` (AITER Triton fast path, ROCm only):
        qkv_or_qkvz is [q, k, v, z], b_or_ba is [b, a], a_or_z_out is the
        z output buffer (mutated in-place).

    ``core_attn_out`` is always mutated in-place.
    """
    layer_name = _resolve_layer_name(layer_name)
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    if fast_kernel:
        self._forward_core_rocm(
            qkvz=qkv_or_qkvz,
            ba=b_or_ba,
            z_out=a_or_z_out,
            core_attn_out=core_attn_out,
        )
    else:
        self._forward_core(
            mixed_qkv=qkv_or_qkvz,
            b=b_or_ba,
            a=a_or_z_out,
            core_attn_out=core_attn_out,
        )


def gdn_attention_core_fake(
    qkv_or_qkvz: torch.Tensor,
    b_or_ba: torch.Tensor,
    a_or_z_out: torch.Tensor,
    core_attn_out: torch.Tensor,
    fast_kernel: bool,
    layer_name: LayerNameType,
) -> None:
    """Fake implementation for torch.compile."""
    return


direct_register_custom_op(
    op_name="gdn_attention_core",
    op_func=gdn_attention_core,
    mutates_args=["a_or_z_out", "core_attn_out"],
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

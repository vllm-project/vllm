# SPDX-License-Identifier: Apache-2.0
"""Inference-only Telechat4 model.

Telechat4 reuses the DeepSeek-V2/V3 backbone (MLA attention, MoE block,
optional DSA indexer) and replaces the standard residual connection with
Manifold-constrained Hyper-Connections (mHC): the residual stream is expanded
into ``num_residual_streams`` parallel streams that are mixed by
input-dependent, doubly-stochastic matrices produced by a Sinkhorn-Knopp
projection.

The mHC math is identical to DeepSeek-V4's, so instead of private kernels this
implementation reuses the shared, platform-dispatched ops from
``vllm.model_executor.layers.mhc`` (``mhc_pre`` / ``mhc_post``).
A local custom op (``telechat4_transpose_contiguous``) ensures the transposed
``comb_mix`` is C-contiguous for the tilelang ``mhc_post`` kernel in a way that
``torch.compile`` cannot elide.
Checkpoint-to-op parameter mapping:

    fn       <- {attn,ffn}_hc.mapping_weight  (fp32, (n^2 + 2n, n*C))
    hc_scale <- [alpha_pre, alpha_post, alpha_res]  (fp32, (3,))
    hc_base  <- {attn,ffn}_hc.bias            (fp32, (n^2 + 2n,))

Both DSA (``model_type=deepseek_v32`` with ``index_topk`` and
``index_n_heads``) and non-DSA (``model_type=deepseek_v3``) variants are
supported. DSA detection requires both ``index_topk`` and ``index_n_heads``
to be present in the config.
"""

from collections.abc import Iterable
from itertools import islice
from typing import Optional
import os

import torch
from torch import nn
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.deepseek_v2 import (
    DeepseekV2Attention,
    DeepseekV2ForCausalLM,
    DeepseekV2MLAAttention,
    DeepseekV2MLP,
    DeepseekV2Model,
    DeepseekV2MoE,
)
from vllm.model_executor.models.utils import (
    PPMissingLayer,
    make_empty_intermediate_tensors_factory,
    make_layers,
)
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors

from vllm.model_executor.layers.mhc import MHCPostOp, MHCPreOp
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.logger import init_logger
logger = init_logger(__name__)


if os.getenv("USE_FLAGOS") == "1":
    import flag_gems
    FLAG_GEMS_CONFIG = [
        "arange_start",
        "pow_scalar",
        "rand_like",
        "argmax",
        "softmax",
        "softmax_out",
        "cumsum_out",
        "moe_align_block_size",
        "invoke_fused_moe_triton_kernel",
        "topk_softmax",
        "grouped_topk",
        "moe_sum",
    ]
    flag_gems.only_enable(record=False, include=FLAG_GEMS_CONFIG)
    logger.info("telechat4: Flaggems is enabled.")

# comb_mix must be transposed and C-contiguous for the mhc_post tilelang
# kernel (requires strides[-1] == 1).  Wrapping transpose+contiguous in a
# custom op prevents torch.compile from eliding the .contiguous() call,
# which it would otherwise do since the downstream MHCPostOp is opaque.

def _telechat4_transpose_contiguous(x: torch.Tensor) -> torch.Tensor:
    """Transpose last two dims and enforce C-contiguity."""
    return x.transpose(-1, -2).contiguous()


def _telechat4_transpose_contiguous_fake(x: torch.Tensor) -> torch.Tensor:
    """Abstract impl for torch.compile / meta tracing."""
    shape = list(x.shape)
    shape[-1], shape[-2] = shape[-2], shape[-1]
    return torch.empty(shape, dtype=x.dtype, device=x.device)


direct_register_custom_op(
    op_name="telechat4_transpose_contiguous",
    op_func=_telechat4_transpose_contiguous,
    mutates_args=[],
    fake_impl=_telechat4_transpose_contiguous_fake,
)


def _telechat4_ensure_contiguous(x: torch.Tensor) -> torch.Tensor:
    """Dispatch to the registered custom op (opaque to torch.compile)."""
    return torch.ops.vllm.telechat4_transpose_contiguous(x)


# ============================== mHC adapter ==============================


class Telechat4MHC(nn.Module):
    """mHC (Manifold-constrained Hyper-Connections) wrapper.

    Delegates computation to MHCPreOp and MHCPostOp, which dispatch to
    optimized TileLang/Triton/PyTorch kernels.

    API:
      - forward(hidden_states) -> (aggregated, h_res, h_post)
      - fused_h_res_h_post_bda_inference(...) -> updated residual
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.n = config.num_residual_streams
        self.hidden_size = config.hidden_size
        self.sinkhorn_iterations = getattr(
            config, "mhc_sinkhorn_iterations", 20)
        self.norm_eps = getattr(config, "mhc_norm_eps", 1e-6)
        self.pre_eps = getattr(config, "mhc_pre_eps", 1e-6)
        self.post_mult_value = getattr(config, "mhc_post_mult_value", 2.0)
        self.h_res_clamp_min = getattr(
            config, "mhc_h_res_clamp_min", None)
        self.h_res_clamp_max = getattr(
            config, "mhc_h_res_clamp_max", None)

        hc_dim = self.n * self.hidden_size
        mix_hc = self.n * self.n + 2 * self.n

        # Weights stored as float32 to preserve full precision from
        # checkpoint.  If these were created as the default dtype
        # (bfloat16), vLLM would truncate float32 checkpoint weights to
        # bfloat16 during loading, causing precision loss.
        self.hc_fn = nn.Parameter(
            torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_scale = nn.Parameter(
            torch.full((3,),
                       getattr(config, "mhc_init_gating_factor", 0.01),
                       dtype=torch.float32))
        self.hc_base = nn.Parameter(
            torch.zeros(mix_hc, dtype=torch.float32))

        # MHC ops (stateless -- weights are passed as tensor args)
        self.mhc_pre = MHCPreOp()
        self.mhc_post = MHCPostOp()

    def forward(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute mHC pre-mixing: aggregate n-stream -> 1-stream.

        Args:
            hidden_states: [nt, n*C] - n-stream hidden states (flattened)
        Returns:
            aggregated: [nt, C] - single-stream input for the sub-layer
            h_res: [nt, n, n] - residual mixing matrix (comb_mix)
            h_post: [nt, n, 1] - stream expansion weights (post_mix)
        """
        n = self.n
        C = self.hidden_size

        # Keep flattened: [nt, n*C] -> [nt, n, C]
        residual = hidden_states.reshape(-1, n, C)

        # MHCPreOp: compute mix weights and aggregated input.
        post_mix, comb_mix, layer_input = self.mhc_pre(
            residual=residual,
            fn=self.hc_fn,
            hc_scale=self.hc_scale,
            hc_base=self.hc_base,
            rms_eps=self.norm_eps,
            hc_pre_eps=self.pre_eps,
            hc_sinkhorn_eps=self.norm_eps,
            hc_post_mult_value=self.post_mult_value,
            sinkhorn_repeat=self.sinkhorn_iterations,
            # clamp_min=self.h_res_clamp_min,
            # clamp_max=self.h_res_clamp_max,
        )
        # Transpose and ensure C-contiguity via custom op (see comment above).
        comb_mix = _telechat4_ensure_contiguous(comb_mix)
        # layer_input: [nt, C], comb_mix: [nt, n, n], post_mix: [nt, n, 1]
        return layer_input, comb_mix, post_mix

    def fused_h_res_h_post_bda_inference(
        self,
        h_res: torch.Tensor,
        original_residual: torch.Tensor,
        h_post: torch.Tensor,
        layer_output_with_bias: tuple[torch.Tensor, torch.Tensor | None],
    ) -> torch.Tensor:
        """
        Fused residual mixing + post expansion + bias-dropout-add.

        Args:
            h_res: [nt, n, n] - comb_mix from forward()
            original_residual: [nt, n*C] - the n-stream input before aggregation
            h_post: [nt, n, 1] - post_mix from forward()
            layer_output_with_bias: (x [nt,C], bias None)
        Returns:
            output: [nt, n*C] - updated n-stream residual for next layer
        """
        x, _ = layer_output_with_bias

        n = self.n
        C = self.hidden_size

        # All inputs are already flattened: reshape to [nt, n, C] for mhc_post
        residual = original_residual.reshape(-1, n, C)

        # h_res (comb_mix) is already C-contiguous from forward().
        new_residual = self.mhc_post(x, residual, h_post, h_res)
        # new_residual: [nt, n, C], flatten back to [nt, n*C]
        return new_residual.reshape(-1, n * C)


# ============================== Helpers ==============================


def input_expand(x: torch.Tensor, n: int) -> torch.Tensor:
    """(T, C) -> (T, n*C): replicate the single stream into n streams."""
    T, C = x.shape
    return x.unsqueeze(1).expand(T, n, C).reshape(T, n * C)


def output_contract(x: torch.Tensor, n: int) -> torch.Tensor:
    """(T, n*C) -> (T, C): average the n streams back into one."""
    T, nC = x.shape
    return x.view(T, n, nC // n).mean(dim=1)


def _get_llama_4_scaling(
    original_max_position_embeddings: int,
    scaling_beta: float,
    positions: torch.Tensor,
) -> torch.Tensor:
    scaling = 1 + scaling_beta * torch.log(
        1 + torch.floor(positions / original_max_position_embeddings))
    return scaling[..., None, None]


# ============================ Decoder Layer ============================


class Telechat4DecoderLayer(nn.Module):

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        config=None,
        topk_indices_buffer: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        if config is None:
            config = vllm_config.model_config.hf_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config

        self.hidden_size = config.hidden_size
        max_position_embeddings = getattr(config, "max_position_embeddings",
                                          8192)
        moe_layer_freq = getattr(config, "moe_layer_freq", 1)
        layer_idx = int(prefix.split(sep=".")[-1])
        self.layer_idx = layer_idx

        qk_nope_head_dim = getattr(config, "qk_nope_head_dim", 0)
        qk_rope_head_dim = getattr(config, "qk_rope_head_dim", 0)
        v_head_dim = getattr(config, "v_head_dim", 0)
        kv_lora_rank = getattr(config, "kv_lora_rank", 0)
        use_mha = config.model_type == "deepseek" or all(
            dim == 0 for dim in (qk_nope_head_dim, qk_rope_head_dim))
        self.use_mha = use_mha

        if use_mha:
            attn_cls = DeepseekV2Attention
        elif model_config.use_mla:
            attn_cls = DeepseekV2MLAAttention
        else:
            attn_cls = DeepseekV2Attention

        self.self_attn = attn_cls(
            vllm_config=vllm_config,
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            q_lora_rank=config.q_lora_rank if hasattr(config, "q_lora_rank")
            else None,
            kv_lora_rank=kv_lora_rank,
            max_position_embeddings=max_position_embeddings,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
            topk_indices_buffer=topk_indices_buffer,
        )

        if (config.n_routed_experts is not None
                and layer_idx >= config.first_k_dense_replace
                and layer_idx % moe_layer_freq == 0):
            self.mlp = DeepseekV2MoE(
                config=config,
                parallel_config=parallel_config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = DeepseekV2MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )

        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor",
                                             1.0)

        self.n = getattr(config, "num_residual_streams", 1)
        self.enable_mhc = self.n > 1
        if self.enable_mhc:
            self.attn_hc = Telechat4MHC(config)
            self.ffn_hc = Telechat4MHC(config)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        llama_4_scaling: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.enable_mhc:
            # mHC mode: hidden_states carries the flattened streams
            # (T, n*C); the `residual` argument is unused because the
            # streams themselves play the residual role.
            origin_hidden_states = hidden_states

            # --- attention sub-block ---
            aggregated, attn_h_res, attn_h_post = self.attn_hc(
                origin_hidden_states)
            attn_kwargs = {
                "positions": positions,
                "hidden_states": self.input_layernorm(aggregated),
            }
            if not self.use_mha:
                attn_kwargs["llama_4_scaling"] = llama_4_scaling
            attn_out = self.self_attn(**attn_kwargs)
            if (not isinstance(self.self_attn, DeepseekV2Attention)
                    and attn_out.dtype == torch.float16):
                attn_out = attn_out * (1.0 / self.routed_scaling_factor)

            # Fused residual mixing + post expansion for attention.
            hidden_states = self.attn_hc.fused_h_res_h_post_bda_inference(
                h_res=attn_h_res,
                original_residual=origin_hidden_states,
                h_post=attn_h_post,
                layer_output_with_bias=(attn_out, None),
            )

            # --- MLP sub-block ---
            aggregated, mlp_h_res, mlp_h_post = self.ffn_hc(
                hidden_states)
            mlp_out = self.mlp(self.post_attention_layernorm(aggregated))
            if (isinstance(self.mlp, DeepseekV2MLP)
                    and mlp_out.dtype == torch.float16):
                mlp_out = mlp_out * (1.0 / self.routed_scaling_factor)

            # Fused residual mixing + post expansion for MLP.
            hidden_states = self.ffn_hc.fused_h_res_h_post_bda_inference(
                h_res=mlp_h_res,
                original_residual=hidden_states,
                h_post=mlp_h_post,
                layer_output_with_bias=(mlp_out, None),
            )

            return hidden_states, residual

        # --- standard (non-mHC) path, same as DeepseekV2 ---
        if residual is None:
            residual = hidden_states.clone()
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)

        attn_kwargs = {
            "positions": positions,
            "hidden_states": hidden_states,
        }
        if not self.use_mha:
            attn_kwargs["llama_4_scaling"] = llama_4_scaling
        hidden_states = self.self_attn(**attn_kwargs)

        if (not isinstance(self.self_attn, DeepseekV2Attention)
                and hidden_states.dtype == torch.float16):
            hidden_states *= 1.0 / self.routed_scaling_factor
            if self.layer_idx == 0:
                residual *= 1.0 / self.routed_scaling_factor

        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states)

        if (isinstance(self.mlp, DeepseekV2MLP)
                and hidden_states.dtype == torch.float16):
            hidden_states *= 1.0 / self.routed_scaling_factor

        return hidden_states, residual


# ============================== Model ==============================


@support_torch_compile
class Telechat4Model(nn.Module):
    fall_back_to_pt_during_load = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.device = current_platform.device_type

        # Backward compatibility: older Telechat4 configs describe RoPE with
        # `rope_scaling` while vLLM's DeepSeek attention expects
        # `rope_parameters`. TODO: drop this once the released config.json
        # natively provides `rope_parameters`.
        if not hasattr(config, "rope_parameters") and hasattr(config,
                                                              "rope_scaling"):
            rs = config.rope_scaling or {}
            rope_type = rs.get("type", "default")
            if rope_type == "rope":
                rope_type = "deepseek_yarn"
            config.rope_parameters = {
                "rope_type": rope_type,
                "factor": rs.get("factor", 1.0),
                "original_max_position_embeddings": rs.get(
                    "original_max_position_embeddings", 4096),
                "beta_fast": rs.get("beta_fast", 32),
                "beta_slow": rs.get("beta_slow", 1),
                "mscale": rs.get("mscale", 1.0),
                "mscale_all_dim": rs.get("mscale_all_dim", 0),
                "apply_yarn_scaling": rope_type in ("yarn", "deepseek_yarn"),
            }

        self.vocab_size = config.vocab_size
        self.is_v32 = (
            getattr(config, "index_topk", None) is not None
            and getattr(config, "index_n_heads", None) is not None
        )
        if self.is_v32:
            topk_tokens = config.index_topk
            topk_indices_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                topk_tokens,
                dtype=torch.int32,
                device=self.device,
            )
        else:
            topk_indices_buffer = None
            # Non-DSA checkpoints may carry DSA fields set to ``null`` in
            # config.json (e.g. ``index_topk: null``).  ``hasattr`` returns
            # True for attributes whose value is None, so the base class
            # DeepseekV2MLAAttention — which checks
            # ``hasattr(config, "index_topk")`` — would falsely detect a DSA
            # model and crash inside Indexer with NoneType * NoneType.
            # Delete all stale DSA attributes whose value is None so that
            # downstream classes see a clean non-DSA config.
            for _attr in ("index_topk", "index_n_heads", "index_head_dim"):
                if hasattr(config, _attr) and getattr(config, _attr) is None:
                    delattr(config, _attr)
                    logger.info(
                        "telechat4: non-DSA checkpoint has stale '%s' "
                        "set to None; deleted to prevent false DSA "
                        "detection.", _attr)

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: Telechat4DecoderLayer(
                vllm_config,
                prefix,
                topk_indices_buffer=topk_indices_buffer,
            ),
            prefix=f"{prefix}.layers",
        )

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))

        self.aux_hidden_state_layers: tuple[int, ...] = ()

        qk_nope_head_dim = getattr(config, "qk_nope_head_dim", 0)
        qk_rope_head_dim = getattr(config, "qk_rope_head_dim", 0)
        self.use_mha = config.model_type == "deepseek" or all(
            dim == 0 for dim in (qk_nope_head_dim, qk_rope_head_dim))
        self.num_redundant_experts = (
            vllm_config.parallel_config.eplb_config.num_redundant_experts)
        self.num_residual_streams = getattr(config, "num_residual_streams", 1)

        if self.num_residual_streams > 1:
            # The shared mHC ops assert bfloat16 inputs (they were built for
            # DeepSeek-V4, which is bf16-only).
            model_dtype = vllm_config.model_config.dtype
            if model_dtype != torch.bfloat16:
                raise ValueError(
                    "Telechat4 mHC (num_residual_streams > 1) requires "
                    f"bfloat16, got {model_dtype}. The shared mHC ops in "
                    "vllm.model_executor.layers.mhc are bf16-only.")
            # In mHC mode the inter-layer tensor is n*hidden_size wide and
            # carries no separate `residual`, which the current PP
            # intermediate-tensor plumbing does not support.
            if vllm_config.parallel_config.pipeline_parallel_size > 1:
                raise NotImplementedError(
                    "Telechat4 mHC (num_residual_streams > 1) does not "
                    "support pipeline parallelism yet.")

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors],
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                if input_ids is None:
                    raise ValueError(
                        "Either input_ids or inputs_embeds must be provided "
                        "to Telechat4Model.forward")
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        llama_4_scaling_config = getattr(self.config, "llama_4_scaling", None)
        llama_4_scaling: Optional[torch.Tensor]
        if llama_4_scaling_config is not None:
            llama_4_scaling = _get_llama_4_scaling(
                original_max_position_embeddings=llama_4_scaling_config[
                    "original_max_position_embeddings"],
                scaling_beta=llama_4_scaling_config["beta"],
                positions=positions,
            )
        else:
            llama_4_scaling = None

        n_streams = self.num_residual_streams
        if n_streams > 1:
            hidden_states = input_expand(hidden_states, n_streams)

        aux_hidden_states = []
        for idx, layer in enumerate(
                islice(self.layers, self.start_layer, self.end_layer),
                start=self.start_layer,
        ):
            if idx in self.aux_hidden_state_layers:
                aux_hidden_states.append(hidden_states + residual)
            hidden_states, residual = layer(positions, hidden_states, residual,
                                            llama_4_scaling)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual})

        if n_streams > 1:
            hidden_states = output_contract(hidden_states, n_streams)
            # mHC mode never populated `residual`; final RMSNorm is single-arg.
            hidden_states = self.norm(hidden_states)
        else:
            hidden_states = self.norm(hidden_states, residual)

        if len(aux_hidden_states) > 0:
            return hidden_states, aux_hidden_states
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        """Load Telechat4 weights.

        1. Pre-process: merge alpha_{pre,post,res} into hc_scale (float32).
        2. Name remapping: mapping_weight -> hc_fn, bias -> hc_base.
        3. mHC weights ({attn,ffn}_hc.*) are loaded manually here so that
           DeepseekV2Model.load_weights' stacked_params_mapping never sees
           them.
        4. Everything else is delegated to DeepseekV2Model.load_weights.
        """
        weights = list(weights)
        params_dict = dict(self.named_parameters())

        # 1. Pre-process: merge alpha_{pre,post,res} into hc_scale.
        # Old checkpoints have separate alpha_pre/post/res (1,) per mHC
        # module.  Merge them into a single hc_scale [3] tensor in float32
        # to preserve full precision.
        alpha_groups: dict[str, dict[int, torch.Tensor]] = {}
        filtered_weights = []
        for name, tensor in weights:
            if name.endswith("alpha_pre"):
                key = name[:-len("alpha_pre")]
                alpha_groups.setdefault(key, {})[0] = tensor
            elif name.endswith("alpha_post"):
                key = name[:-len("alpha_post")]
                alpha_groups.setdefault(key, {})[1] = tensor
            elif name.endswith("alpha_res"):
                key = name[:-len("alpha_res")]
                alpha_groups.setdefault(key, {})[2] = tensor
            else:
                filtered_weights.append((name, tensor))
        for prefix, alphas in alpha_groups.items():
            hc_scale_key = f"{prefix}hc_scale"
            if hc_scale_key not in params_dict:
                continue
            if 0 in alphas and 1 in alphas and 2 in alphas:
                hc_scale = torch.stack(
                    [alphas[0].view(()), alphas[1].view(()),
                     alphas[2].view(())]).to(dtype=torch.float32)
                filtered_weights.append((hc_scale_key, hc_scale))
        weights = filtered_weights

        # 2. split mHC weights from the rest, apply name remapping
        processed_weights = []
        hc_weights = []

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            # name remapping
            if "indexer.k_norm_bias" in name:
                name = name.replace("indexer.k_norm_bias",
                                    "indexer.k_norm.bias")

            # Legacy checkpoints may carry split pre/post/res biases; only
            # the merged bias (remapped to hc_base) is used.
            skip_patterns = (
                "attn_hc.bias_pre", "attn_hc.bias_post", "attn_hc.bias_res",
                "ffn_hc.bias_pre", "ffn_hc.bias_post", "ffn_hc.bias_res",
            )
            if any(p in name for p in skip_patterns):
                continue

            # Remap old parameter names to new float32 parameters.
            # These are mutually exclusive: a name ending in mapping_weight
            # is remapped to hc_fn; a name ending in _hc.bias is remapped
            # to hc_base. Using elif avoids accidental double-remapping.
            if name.endswith("mapping_weight"):
                name = name.replace("mapping_weight", "hc_fn")
            if name.endswith("_hc.bias"):
                name = name.replace(".bias", ".hc_base")

            # split mHC weights from the rest
            if "attn_hc" in name or "ffn_hc" in name:
                hc_weights.append((name, loaded_weight))
            else:
                processed_weights.append((name, loaded_weight))

        # 3. manual mHC load
        loaded_params: set[str] = set()

        for name, loaded_weight in hc_weights:
            if name not in params_dict:
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)

        # 4. everything else via DeepseekV2Model
        other_loaded = DeepseekV2Model.load_weights(self, processed_weights)
        loaded_params.update(other_loaded)

        return loaded_params


# ============================ Causal LM ============================


class TeleChat4ForCausalLM(DeepseekV2ForCausalLM):
    model_cls = Telechat4Model
    packed_modules_mapping = {
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def set_moe_parameters(self):
        """Override for Telechat4DecoderLayer (not a DeepseekV2DecoderLayer
        instance)."""
        self.expert_weights = []
        self.num_expert_groups = getattr(self.config, "n_group", 1)
        self.moe_layers = []
        self.moe_mlp_layers = []
        example_moe = None
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue
            if hasattr(layer, "mlp") and isinstance(layer.mlp, DeepseekV2MoE):
                example_moe = layer.mlp
                self.moe_mlp_layers.append(layer.mlp)
                self.moe_layers.append(layer.mlp.experts)
        if example_moe is not None:
            self.extract_moe_parameters(example_moe)

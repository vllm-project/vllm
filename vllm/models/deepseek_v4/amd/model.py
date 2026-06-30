# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import typing
from collections.abc import Callable, Iterable
from itertools import islice

import regex as re
import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul, SiluAndMulWithClamp
from vllm.model_executor.layers.fused_moe import (
    FusedMoE,
    GateLinear,
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mhc import (
    HCHeadOp,
    MHCFusedPostPreOp,
    MHCPostOp,
    MHCPreOp,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    extract_layer_index,
    is_pp_missing_parameter,
    make_layers,
    maybe_prefix,
)
from vllm.models.deepseek_v4.amd.rocm import DeepseekV4ROCMAiterMLAAttention
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.utils.import_utils import has_tilelang

logger = init_logger(__name__)

_DSV4_AMD_FOCUS_DIMS = (3753, 2404, 5308, 1040, 532, 4265, 5414, 2949)


def _dsv4_amd_debug_phase(
    input_ids: torch.Tensor | None = None,
    positions: torch.Tensor | None = None,
    hidden_states: torch.Tensor | None = None,
) -> str:
    if positions is not None:
        num_tokens = positions.shape[0]
    elif input_ids is not None:
        num_tokens = input_ids.shape[0]
    elif hidden_states is not None:
        num_tokens = hidden_states.shape[0]
    else:
        num_tokens = -1
    if num_tokens == 1:
        return "RUNTIME_DECODE_N1"
    if num_tokens > 1:
        return f"RUNTIME_PREFILL_N{num_tokens}"
    return "RUNTIME_UNKNOWN"


def _dsv4_amd_debug_tensor(
    counts: dict[str, int],
    label: str,
    tensor: torch.Tensor | None,
    prefix: str,
    input_ids: torch.Tensor | None = None,
    positions: torch.Tensor | None = None,
    max_logs: int = 8,
) -> None:
    if tensor is None:
        return
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        return
    phase = _dsv4_amd_debug_phase(input_ids, positions, tensor)
    label_with_phase = f"{label}_{phase}"
    count = counts.get(label_with_phase, 0)
    if count >= max_logs:
        return
    counts[label_with_phase] = count + 1
    with torch.no_grad():
        tf = tensor.detach().float()
        if tf.numel() == 0:
            logger.warning(
                "[%s] count=%s prefix=%s shape=%s dtype=%s empty_tensor=True",
                label_with_phase,
                count,
                prefix,
                tuple(tensor.shape),
                tensor.dtype,
            )
            return
        pos_info = ""
        if positions is not None and positions.numel() > 0:
            pos = positions.detach()
            pos_info = (
                f" pos_first={pos.flatten()[0].item()}"
                f" pos_last={pos.flatten()[-1].item()}"
            )
        input_info = ""
        if input_ids is not None and input_ids.numel() > 0:
            ids = input_ids.detach()
            input_info = (
                f" input_first={ids.flatten()[0].item()}"
                f" input_last={ids.flatten()[-1].item()}"
            )
        logger.warning(
            "[%s] count=%s prefix=%s shape=%s dtype=%s finite=%s "
            "mean=%.6g std=%.6g max=%.6g nonzero=%.6g%s%s",
            label_with_phase,
            count,
            prefix,
            tuple(tensor.shape),
            tensor.dtype,
            torch.isfinite(tf).all().item(),
            tf.mean().item(),
            tf.std().item(),
            tf.abs().max().item(),
            (tf != 0).float().mean().item(),
            pos_info,
            input_info,
        )


def _dsv4_amd_debug_focus_dims(
    counts: dict[str, int],
    label: str,
    tensor: torch.Tensor | None,
    prefix: str,
    input_ids: torch.Tensor | None = None,
    positions: torch.Tensor | None = None,
    max_logs: int = 8,
) -> None:
    if tensor is None or tensor.ndim < 2:
        return
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        return
    phase = _dsv4_amd_debug_phase(input_ids, positions, tensor)
    label_with_phase = f"{label}_FOCUS_DIMS_{phase}"
    count = counts.get(label_with_phase, 0)
    if count >= max_logs:
        return
    counts[label_with_phase] = count + 1
    with torch.no_grad():
        tf = tensor.detach().float().reshape(-1, tensor.shape[-1])
        dims = [d for d in _DSV4_AMD_FOCUS_DIMS if d < tf.shape[-1]]
        if not dims:
            return
        vals = tf[:, dims]
        mean = vals.mean(dim=0)
        std = vals.std(dim=0, unbiased=False)
        min_vals = vals.min(dim=0).values
        max_vals = vals.max(dim=0).values
        first = vals[0]
        last = vals[-1]
        all_dim_mean = tf.mean(dim=0)
        top_mean_vals, top_mean_ids = torch.topk(
            all_dim_mean.abs(), k=min(12, all_dim_mean.numel())
        )
        logger.warning(
            "[%s] count=%s prefix=%s shape=%s dtype=%s dims=%s "
            "mean=%s std=%s min=%s max=%s first=%s last=%s "
            "top_abs_mean_dims=%s",
            label_with_phase,
            count,
            prefix,
            tuple(tensor.shape),
            tensor.dtype,
            dims,
            [round(v, 6) for v in mean.detach().cpu().tolist()],
            [round(v, 6) for v in std.detach().cpu().tolist()],
            [round(v, 6) for v in min_vals.detach().cpu().tolist()],
            [round(v, 6) for v in max_vals.detach().cpu().tolist()],
            [round(v, 6) for v in first.detach().cpu().tolist()],
            [round(v, 6) for v in last.detach().cpu().tolist()],
            [
                (int(dim.item()), round(float(val.item()), 6))
                for dim, val in zip(top_mean_ids, top_mean_vals)
            ],
        )


def _dsv4_amd_debug_logits_topk(
    counts: dict[str, int],
    label: str,
    logits: torch.Tensor | None,
    prefix: str,
    k: int = 10,
    max_logs: int = 16,
) -> None:
    if logits is None:
        return
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        return
    phase = _dsv4_amd_debug_phase(hidden_states=logits)
    label_with_phase = f"{label}_{phase}"
    count = counts.get(label_with_phase, 0)
    if count >= max_logs:
        return
    counts[label_with_phase] = count + 1
    with torch.no_grad():
        lf = logits.detach().float()
        if lf.numel() == 0 or lf.ndim == 0:
            logger.warning(
                "[%s] count=%s prefix=%s shape=%s dtype=%s empty_or_scalar=True",
                label_with_phase,
                count,
                prefix,
                tuple(logits.shape),
                logits.dtype,
            )
            return
        row = lf.reshape(-1, lf.shape[-1])[-1]
        topk = min(k, row.numel())
        vals, ids = torch.topk(row, k=topk)
        logger.warning(
            "[%s] count=%s prefix=%s shape=%s dtype=%s row=last "
            "top_ids=%s top_vals=%s",
            label_with_phase,
            count,
            prefix,
            tuple(logits.shape),
            logits.dtype,
            ids.detach().cpu().tolist(),
            [round(v, 6) for v in vals.detach().cpu().tolist()],
        )


def _dsv4_amd_local_vocab_id(module: nn.Module, local_idx: int) -> int:
    shard_indices = getattr(module, "shard_indices", None)
    if shard_indices is None:
        return local_idx
    num_org_padded = shard_indices.num_org_elements_padded
    num_org = shard_indices.num_org_elements
    if local_idx < num_org:
        return shard_indices.org_vocab_start_index + local_idx
    if local_idx < num_org_padded:
        return -1
    added_idx = local_idx - num_org_padded
    if added_idx < shard_indices.num_added_elements:
        return shard_indices.added_vocab_start_index + added_idx
    return -1


class DeepseekV4MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        swiglu_limit: float | None = None,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        is_sequence_parallel: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()

        # If is_sequence_parallel, the input and output tensors are sharded
        # across the ranks within the tp_group. In this case the weights are
        # replicated and no collective ops are needed.
        # Otherwise we use standard TP with an allreduce at the end.
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            disable_tp=is_sequence_parallel,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            disable_tp=is_sequence_parallel,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )
        if swiglu_limit is not None:
            self.act_fn = SiluAndMulWithClamp(swiglu_limit)
        else:
            self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class DeepseekV4MoE(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()

        self.tp_size = get_tensor_model_parallel_world_size()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.prefix = prefix
        self._dsv4_amd_debug_counts: dict[str, int] = {}

        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)
        self.hidden_size = config.hidden_size

        self.n_routed_experts = config.n_routed_experts
        self.n_activated_experts = config.num_experts_per_tok
        self.moe_intermediate_size = config.moe_intermediate_size
        self.swiglu_limit = config.swiglu_limit
        self.renormalize = config.norm_topk_prob
        self.scoring_func = getattr(config, "scoring_func", "sqrtsoftplus")

        self.gate = GateLinear(
            input_size=config.hidden_size,
            output_size=config.n_routed_experts,
            bias=False,
            out_dtype=torch.float32,
            prefix=f"{prefix}.gate",
        )

        self.gate.e_score_correction_bias = None
        self.gate.tid2eid = None
        is_hash_moe = extract_layer_index(prefix) < config.num_hash_layers
        self.hash_indices_dtype = torch.int32
        if is_hash_moe:
            # hash MoE doesn't use e_score_correction_bias
            # Use randint instead of empty to avoid garbage values causing
            # invalid memory access in dummy mode (--load-format="dummy")
            self.gate.tid2eid = nn.Parameter(
                torch.randint(
                    0,
                    config.n_routed_experts,
                    (config.vocab_size, config.num_experts_per_tok),
                    dtype=self.hash_indices_dtype,
                ),
                requires_grad=False,
            )
        elif getattr(config, "topk_method", None) == "noaux_tc":
            self.gate.e_score_correction_bias = nn.Parameter(
                torch.empty(config.n_routed_experts, dtype=torch.float32),
                requires_grad=False,
            )

        if config.n_shared_experts is None:
            self.shared_experts = None
        else:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts

            self.shared_experts = DeepseekV4MLP(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=config.hidden_act,
                swiglu_limit=self.swiglu_limit,
                quant_config=quant_config,
                reduce_results=False,
                prefix=f"{prefix}.shared_experts",
            )

        self.tp_rank = get_tensor_model_parallel_rank()
        assert config.n_routed_experts % self.tp_size == 0

        self.n_local_experts = config.n_routed_experts // self.tp_size
        self.experts_start_idx = self.tp_rank * self.n_local_experts
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts

        self.experts = FusedMoE(
            shared_experts=self.shared_experts,
            gate=self.gate,
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            scoring_func=self.scoring_func,
            routed_scaling_factor=self.routed_scaling_factor,
            e_score_correction_bias=self.gate.e_score_correction_bias,
            hash_indices_table=self.gate.tid2eid,
            swiglu_limit=self.swiglu_limit,
            router_logits_dtype=torch.float32,
        )

    def _debug_layer60_moe_tensor(
        self,
        label: str,
        tensor: torch.Tensor | None,
    ) -> None:
        if not self.prefix.endswith("layers.60.ffn"):
            return
        _dsv4_amd_debug_tensor(
            self._dsv4_amd_debug_counts,
            f"DSV4_AMD_LAYER60_MOE_{label}",
            tensor,
            self.prefix,
            max_logs=8,
        )
        _dsv4_amd_debug_focus_dims(
            self._dsv4_amd_debug_counts,
            f"DSV4_AMD_LAYER60_MOE_{label}",
            tensor,
            self.prefix,
            max_logs=8,
        )

    def _debug_layer60_router_topk(self, router_logits: torch.Tensor | None) -> None:
        if not self.prefix.endswith("layers.60.ffn"):
            return
        _dsv4_amd_debug_logits_topk(
            self._dsv4_amd_debug_counts,
            "DSV4_AMD_LAYER60_MOE_ROUTER_TOPK",
            router_logits,
            self.prefix,
            k=self.n_activated_experts,
            max_logs=8,
        )

    def forward(
        self, hidden_states: torch.Tensor, input_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.gate.tid2eid is not None and input_ids is None:
            raise ValueError("DeepSeek V4 hash MoE routing requires input_ids.")

        org_shape = hidden_states.shape
        self._debug_layer60_moe_tensor("INPUT", hidden_states)
        if self.experts.is_internal_router:
            # In this case, the gate/router runs inside the FusedMoE class
            self._debug_layer60_moe_tensor("INTERNAL_ROUTER_INPUT", hidden_states)
            final_hidden_states = self.experts(
                hidden_states=hidden_states,
                router_logits=hidden_states,
                input_ids=input_ids,
            )
        else:
            router_logits, _ = self.gate(hidden_states)
            self._debug_layer60_moe_tensor("ROUTER_LOGITS", router_logits)
            self._debug_layer60_router_topk(router_logits)
            final_hidden_states = self.experts(
                hidden_states=hidden_states,
                router_logits=router_logits,
                input_ids=input_ids,
            )
        self._debug_layer60_moe_tensor("OUTPUT_FLAT", final_hidden_states)

        return final_hidden_states.view(org_shape)


class DeepseekV4DecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config,
        prefix,
        topk_indices_buffer: torch.Tensor | None = None,
        aux_stream_list: list[torch.cuda.Stream] | None = None,
    ):
        super().__init__()

        # Lazy import to avoid top-level tilelang dependency.
        # Registers both torch.ops.vllm.mhc_pre and mhc_post
        import vllm.model_executor.layers.mhc  # noqa: F401

        config = vllm_config.model_config.hf_config
        self.hidden_size = config.hidden_size
        self.prefix = prefix
        self.layer_idx = extract_layer_index(prefix)
        self._dsv4_amd_debug_printed = False
        self._dsv4_amd_debug_counts: dict[str, int] = {}
        if prefix.endswith("layers.0"):
            print(
                f"[DSV4_AMD_LAYER_INIT] prefix={prefix} hidden_size={self.hidden_size}",
                flush=True,
            )

        self.rms_norm_eps = config.rms_norm_eps
        self.attn = DeepseekV4ROCMAiterMLAAttention(
            vllm_config,
            prefix=f"{prefix}.attn",
            topk_indices_buffer=topk_indices_buffer,
            aux_stream_list=aux_stream_list,
        )
        self.ffn = DeepseekV4MoE(vllm_config, prefix=f"{prefix}.ffn")

        self.attn_norm = RMSNorm(self.hidden_size, self.rms_norm_eps)
        self.ffn_norm = RMSNorm(self.hidden_size, self.rms_norm_eps)
        self.hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.hc_post_alpha = 2.0
        mix_hc = (2 + self.hc_mult) * self.hc_mult
        hc_dim = self.hc_mult * self.hidden_size
        self.hc_attn_fn = nn.Parameter(
            torch.empty(
                (mix_hc, hc_dim),
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.hc_ffn_fn = nn.Parameter(
            torch.empty(
                (mix_hc, hc_dim),
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.hc_attn_base = nn.Parameter(
            torch.empty(
                mix_hc,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.hc_ffn_base = nn.Parameter(
            torch.empty(
                mix_hc,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.hc_attn_scale = nn.Parameter(
            torch.empty(
                3,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.hc_ffn_scale = nn.Parameter(
            torch.empty(
                3,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.mhc_pre = MHCPreOp()
        self.mhc_post = MHCPostOp()
        self.mhc_fused_post_pre = MHCFusedPostPreOp()
        self.has_tilelang = has_tilelang()

    def _debug_layer0_tensor(
        self,
        label: str,
        tensor: torch.Tensor | None,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
    ) -> None:
        if not self.prefix.endswith("layers.0"):
            return
        _dsv4_amd_debug_tensor(
            self._dsv4_amd_debug_counts,
            label,
            tensor,
            self.prefix,
            input_ids=input_ids,
            positions=positions,
        )

    def _debug_selected_layer_tensor(
        self,
        label: str,
        tensor: torch.Tensor | None,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
    ) -> None:
        if self.layer_idx not in {0, 15, 30, 45, 60}:
            return
        _dsv4_amd_debug_tensor(
            self._dsv4_amd_debug_counts,
            f"DSV4_AMD_LAYER{self.layer_idx}_{label}",
            tensor,
            self.prefix,
            input_ids=input_ids,
            positions=positions,
            max_logs=4,
        )
        _dsv4_amd_debug_focus_dims(
            self._dsv4_amd_debug_counts,
            f"DSV4_AMD_LAYER{self.layer_idx}_{label}",
            tensor,
            self.prefix,
            input_ids=input_ids,
            positions=positions,
            max_logs=4,
        )

    def hc_pre(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
    ):
        post_mix, res_mix, layer_input = self.mhc_pre(
            residual=x,
            fn=hc_fn,
            hc_scale=hc_scale,
            hc_base=hc_base,
            rms_eps=self.rms_norm_eps,
            hc_pre_eps=self.hc_eps,
            hc_sinkhorn_eps=self.hc_eps,
            hc_post_mult_value=self.hc_post_alpha,
            sinkhorn_repeat=self.hc_sinkhorn_iters,
        )
        return layer_input, post_mix, res_mix

    def hc_post(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ):
        return self.mhc_post(x, residual, post, comb)

    def _attention_input(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        x = self.attn_norm(x)
        if self.prefix.endswith("layers.0") and not self._dsv4_amd_debug_printed:
            self._dsv4_amd_debug_printed = True
            print(
                "[DSV4_AMD_ATTN_INPUT] "
                f"prefix={self.prefix} x={tuple(x.shape)} "
                f"residual={tuple(residual.shape)}",
                flush=True,
            )
        return x

    def _forward_fused_post_pre(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None,
        post_mix: torch.Tensor | None = None,
        res_mix: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if residual is None:
            # Run standalone hc_pre on first layer
            residual = x
            x, post_mix, res_mix = self.hc_pre(
                x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
            )
            self._debug_layer0_tensor(
                "DSV4_AMD_LAYER0_HC_ATTN_PRE_OUT", x, input_ids, positions
            )
        else:
            residual, post_mix, res_mix, x = self.mhc_fused_post_pre(
                x,
                residual,
                post_mix,
                res_mix,
                self.hc_attn_fn,
                self.hc_attn_scale,
                self.hc_attn_base,
                self.rms_norm_eps,
                self.hc_eps,
                self.hc_eps,
                self.hc_post_alpha,
                self.hc_sinkhorn_iters,
            )
            self._debug_layer0_tensor(
                "DSV4_AMD_LAYER0_ATTN_FUSED_POST_PRE_OUT", x, input_ids, positions
            )

        x = self._attention_input(x, residual)
        self._debug_layer0_tensor(
            "DSV4_AMD_LAYER0_ATTN_NORM_OUT", x, input_ids, positions
        )
        x = self.attn(positions, x, None)
        self._debug_layer0_tensor("DSV4_AMD_LAYER0_ATTN_OUT", x, input_ids, positions)
        self._debug_selected_layer_tensor("ATTN_OUT", x, input_ids, positions)

        residual, post_mix, res_mix, x = self.mhc_fused_post_pre(
            x,
            residual,
            post_mix,
            res_mix,
            self.hc_ffn_fn,
            self.hc_ffn_scale,
            self.hc_ffn_base,
            self.rms_norm_eps,
            self.hc_eps,
            self.hc_eps,
            self.hc_post_alpha,
            self.hc_sinkhorn_iters,
        )
        self._debug_layer0_tensor(
            "DSV4_AMD_LAYER0_FFN_FUSED_POST_PRE_OUT", x, input_ids, positions
        )
        x = self.ffn_norm(x)
        self._debug_layer0_tensor(
            "DSV4_AMD_LAYER0_FFN_NORM_OUT", x, input_ids, positions
        )
        x = self.ffn(x, input_ids)
        self._debug_layer0_tensor("DSV4_AMD_LAYER0_FFN_OUT", x, input_ids, positions)
        self._debug_selected_layer_tensor("FFN_OUT", x, input_ids, positions)
        return x, residual, post_mix, res_mix

    def _forward_unfused_post_pre(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None,
        post_mix: torch.Tensor | None = None,
        res_mix: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None
    ]:
        residual = x
        x, post, comb = self.hc_pre(
            x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
        )
        x = self._attention_input(x, residual)
        self._debug_layer0_tensor(
            "DSV4_AMD_LAYER0_UNFUSED_ATTN_NORM_OUT", x, input_ids, positions
        )
        x = self.attn(positions, x, None)
        self._debug_layer0_tensor(
            "DSV4_AMD_LAYER0_UNFUSED_ATTN_OUT", x, input_ids, positions
        )
        self._debug_selected_layer_tensor("UNFUSED_ATTN_OUT", x, input_ids, positions)
        x = self.hc_post(x, residual, post, comb)
        self._debug_layer0_tensor(
            "DSV4_AMD_LAYER0_UNFUSED_ATTN_HC_POST_OUT", x, input_ids, positions
        )

        residual = x
        x, post, comb = self.hc_pre(
            x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base
        )
        x = self.ffn_norm(x)
        self._debug_layer0_tensor(
            "DSV4_AMD_LAYER0_UNFUSED_FFN_NORM_OUT", x, input_ids, positions
        )
        x = self.ffn(x, input_ids)
        self._debug_layer0_tensor(
            "DSV4_AMD_LAYER0_UNFUSED_FFN_OUT", x, input_ids, positions
        )
        self._debug_selected_layer_tensor("UNFUSED_FFN_OUT", x, input_ids, positions)
        x = self.hc_post(x, residual, post, comb)
        self._debug_layer0_tensor(
            "DSV4_AMD_LAYER0_UNFUSED_FFN_HC_POST_OUT", x, input_ids, positions
        )
        return x, None, None, None

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None,
        post_mix: torch.Tensor | None = None,
        res_mix: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None
    ]:
        if not self.has_tilelang:
            return self._forward_unfused_post_pre(
                x, positions, input_ids, post_mix, res_mix, residual
            )
        return self._forward_fused_post_pre(
            x, positions, input_ids, post_mix, res_mix, residual
        )


class DeepseekV4Model(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.vocab_size = config.vocab_size
        self.hc_eps = config.hc_eps
        self.hc_mult = config.hc_mult
        self.hc_dim = self.hc_mult * config.hidden_size
        self.rms_norm_eps = config.rms_norm_eps
        self._dsv4_amd_debug_counts: dict[str, int] = {}

        # Three aux streams: one per non-default input GEMM in
        # DeepseekV4Attention.attn_gemm_parallel_execute
        # (compressor kv_score, indexer.weights_proj, indexer.compressor
        # kv_score). fused_wqa_wkv stays on the default stream.
        # Disable them on ROCm because of hang issues.
        aux_stream_list = (
            None
            if current_platform.is_rocm()
            else [torch.cuda.Stream() for _ in range(3)]
        )

        self.device = current_platform.device_type
        # Reserved topk indices buffer for all Indexer layers to reuse.
        self.topk_indices_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            config.index_topk,
            dtype=torch.int32,
            device=self.device,
        )

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
            lambda prefix: DeepseekV4DecoderLayer(
                vllm_config,
                prefix=prefix,
                topk_indices_buffer=self.topk_indices_buffer,
                aux_stream_list=aux_stream_list,
            ),
            prefix=f"{prefix}.layers",
        )

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, self.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.hc_head_fn = nn.Parameter(
            torch.empty(
                self.hc_mult,
                self.hc_dim,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.hc_head_base = nn.Parameter(
            torch.empty(
                self.hc_mult,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.hc_head_scale = nn.Parameter(
            torch.empty(1, dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_head_op = HCHeadOp()
        self.has_tilelang = has_tilelang()
        # Pre-hc_head residual stream buffer for the MTP draft. Stable
        # address (outside the cudagraph pool) so the copy_ in forward()
        # refreshes it correctly across captured shapes.
        # refreshes it correctly across captured shapes. Only allocated on
        # the last PP rank — that's where MTP target hidden states are
        # produced.
        if get_pp_group().is_last_rank:
            self._mtp_hidden_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                self.hc_dim,
                dtype=vllm_config.model_config.dtype,
                device=self.device,
            )
        else:
            self._mtp_hidden_buffer = None

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def _debug_model_tensor(
        self,
        label: str,
        tensor: torch.Tensor | None,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
    ) -> None:
        _dsv4_amd_debug_tensor(
            self._dsv4_amd_debug_counts,
            label,
            tensor,
            "model",
            input_ids=input_ids,
            positions=positions,
        )

    def _debug_hc_head_params(self) -> None:
        label = "DSV4_AMD_MODEL_HC_HEAD_PARAMS"
        if self._dsv4_amd_debug_counts.get(label, 0) > 0:
            return
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            return
        self._dsv4_amd_debug_counts[label] = 1
        with torch.no_grad():
            for name, param in (
                ("fn", self.hc_head_fn),
                ("base", self.hc_head_base),
                ("scale", self.hc_head_scale),
            ):
                pf = param.detach().float()
                logger.warning(
                    "[%s] name=%s shape=%s dtype=%s finite=%s mean=%.6g "
                    "std=%.6g max=%.6g nonzero=%.6g",
                    label,
                    name,
                    tuple(param.shape),
                    param.dtype,
                    torch.isfinite(pf).all().item(),
                    pf.mean().item(),
                    pf.std().item() if pf.numel() > 1 else 0.0,
                    pf.abs().max().item() if pf.numel() > 0 else 0.0,
                    (pf != 0).float().mean().item() if pf.numel() > 0 else 0.0,
                )

    def make_empty_intermediate_tensors(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> IntermediateTensors:
        # PP intermediate tensors carry the multi-stream hidden_states
        # of shape (num_tokens, hc_mult, hidden_size) — V4 expands the
        # token embedding to hc_mult streams before the first decoder
        # layer and keeps that shape until hc_head() collapses it.
        return IntermediateTensors(
            {
                "hidden_states": torch.zeros(
                    (batch_size, self.hc_mult, self.config.hidden_size),
                    dtype=dtype,
                    device=device,
                ),
            }
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            hidden_states = hidden_states.unsqueeze(-2).repeat(1, self.hc_mult, 1)
            self._debug_model_tensor(
                "DSV4_AMD_MODEL_EMBED_EXPANDED", hidden_states, input_ids, positions
            )
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            self._debug_model_tensor(
                "DSV4_AMD_MODEL_PP_INPUT", hidden_states, input_ids, positions
            )

        residual, post_mix, res_mix = None, None, None
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual, post_mix, res_mix = layer(
                hidden_states,
                positions,
                input_ids,
                post_mix,
                res_mix,
                residual,
            )
        self._debug_model_tensor(
            "DSV4_AMD_MODEL_POST_LAYERS", hidden_states, input_ids, positions
        )
        if layer is not None and self.has_tilelang:
            hidden_states = layer.hc_post(hidden_states, residual, post_mix, res_mix)
            self._debug_model_tensor(
                "DSV4_AMD_MODEL_FINAL_HC_POST_OUT",
                hidden_states,
                input_ids,
                positions,
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})

        # Stash pre-hc_head residual for the MTP draft (captured copy_).
        num_tokens = hidden_states.shape[0]
        self._mtp_hidden_buffer[:num_tokens].copy_(hidden_states.flatten(1))

        self._debug_hc_head_params()
        hidden_states = self.hc_head_op(
            hidden_states,
            self.hc_head_fn,
            self.hc_head_scale,
            self.hc_head_base,
            self.rms_norm_eps,
            self.hc_eps,
        )
        self._debug_model_tensor(
            "DSV4_AMD_MODEL_HC_HEAD_OUT", hidden_states, input_ids, positions
        )
        hidden_states = self.norm(hidden_states)
        self._debug_model_tensor(
            "DSV4_AMD_MODEL_FINAL_NORM_OUT", hidden_states, input_ids, positions
        )
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("gate_up_proj", "w1", 0),
            ("gate_up_proj", "w3", 1),
            ("attn.fused_wqa_wkv", "attn.wq_a", 0),
            ("attn.fused_wqa_wkv", "attn.wkv", 1),
            ("compressor.fused_wkv_wgate", "compressor.wkv", 0),
            ("compressor.fused_wkv_wgate", "compressor.wgate", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        def resolve_input_scale_name(name: str) -> str:
            if name in params_dict:
                return name
            if name.endswith(".input_scale"):
                input_scale_2_name = name[: -len(".input_scale")] + ".input_scale_2"
                if input_scale_2_name in params_dict:
                    return input_scale_2_name
            if name.endswith("_input_scale"):
                input_scale_2_name = name + "_2"
                if input_scale_2_name in params_dict:
                    return input_scale_2_name
            return name

        # TP for attention
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        n_head = self.config.num_attention_heads
        n_local_head = n_head // tp_size
        head_rank_start = n_local_head * tp_rank
        head_rank_end = n_local_head * (tp_rank + 1)

        # Pre-compute expert mapping ONCE.
        expert_mapping = self.get_expert_mapping()

        for name, loaded_weight in weights:
            for param_name, weight_name, shard_id in stacked_params_mapping:
                # Skip non-stacked layers and experts (experts handled below).
                if ".experts." in name:
                    continue
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)

                if is_pp_missing_parameter(name, self):
                    break
                name = resolve_input_scale_name(name)
                if name not in params_dict and name.endswith(".weight_scale"):
                    weight_scale_inv_name = (
                        name[: -len(".weight_scale")] + ".weight_scale_inv"
                    )
                    if weight_scale_inv_name in params_dict:
                        name = weight_scale_inv_name
                if name not in params_dict and name.endswith(".weight_scale_inv"):
                    if param_name == "attn.fused_wqa_wkv":
                        # DeepSeek-V4 checkpoints store attn.wkv scales as
                        # FP8 block scales, while the fused projection registers
                        # NVFP4-style group scales under weight_scale. They are
                        # not layout-compatible, so do not fall back and load
                        # a (num_blocks_n, num_blocks_k) tensor into a
                        # per-group scale matrix.
                        break
                    name = name[: -len(".weight_scale_inv")] + ".weight_scale"
                param = params_dict[name]
                weight_loader = param.weight_loader
                try:
                    weight_loader(param, loaded_weight, shard_id)
                except Exception as exc:
                    print(
                        "[DeepSeek-V4 weight load shape mismatch] "
                        f"name={name} shard_id={shard_id} "
                        f"param_type={type(param).__name__} "
                        f"param_shape={tuple(param.shape)} "
                        f"loaded_shape={tuple(loaded_weight.shape)} "
                        f"loaded_dtype={loaded_weight.dtype} "
                        f"inner_error={exc}",
                        flush=True,
                    )
                    logger.error(
                        "DeepSeek-V4 weight load shape mismatch: name=%s "
                        "shard_id=%s param_type=%s param_shape=%s "
                        "loaded_shape=%s loaded_dtype=%s inner_error=%s",
                        name,
                        shard_id,
                        type(param).__name__,
                        tuple(param.shape),
                        tuple(loaded_weight.shape),
                        loaded_weight.dtype,
                        exc,
                    )
                    raise AssertionError(
                        "DeepSeek-V4 weight load shape mismatch: "
                        f"name={name}, shard_id={shard_id}, "
                        f"param_type={type(param).__name__}, "
                        f"param_shape={tuple(param.shape)}, "
                        f"loaded_shape={tuple(loaded_weight.shape)}, "
                        f"loaded_dtype={loaded_weight.dtype}; "
                        f"inner_error={exc}"
                    ) from exc
                loaded_params.add(name)
                break
            else:
                if ".experts." in name:
                    # E8M0 scales are stored as float8_e8m0fnu in
                    # checkpoints but the MoE param is uint8. copy_()
                    # would do a numeric conversion (e.g. 2^-7 → 0),
                    # destroying the raw exponent bytes.
                    if (
                        "weight_scale" in name
                        and loaded_weight.dtype == torch.float8_e8m0fnu
                    ):
                        loaded_weight = loaded_weight.view(torch.uint8)
                    for mapping in expert_mapping:
                        param_name, weight_name, expert_id, expert_shard_id = mapping
                        if weight_name not in name:
                            continue
                        name_mapped = name.replace(weight_name, param_name)
                        if is_pp_missing_parameter(name_mapped, self):
                            continue
                        name_mapped = resolve_input_scale_name(name_mapped)
                        param = params_dict[name_mapped]
                        # We should ask the weight loader to return success or not
                        # here since otherwise we may skip experts with other
                        # available replicas.
                        weight_loader = typing.cast(
                            Callable[..., bool], param.weight_loader
                        )
                        success = weight_loader(
                            param,
                            loaded_weight,
                            name_mapped,
                            shard_id=expert_shard_id,
                            expert_id=expert_id,
                            return_success=True,
                        )
                        if success:
                            name = name_mapped
                            break
                    loaded_params.add(name_mapped)
                    continue
                elif "attn_sink" in name:
                    if is_pp_missing_parameter(name, self):
                        continue
                    narrow_weight = loaded_weight[head_rank_start:head_rank_end]
                    n = narrow_weight.shape[0]
                    params_dict[name][:n].copy_(narrow_weight)
                    loaded_params.add(name)
                    continue
                else:
                    if is_pp_missing_parameter(name, self):
                        continue
                    name = resolve_input_scale_name(name)
                    if name not in params_dict and name.endswith(".weight_scale"):
                        weight_scale_inv_name = (
                            name[: -len(".weight_scale")] + ".weight_scale_inv"
                        )
                        if weight_scale_inv_name in params_dict:
                            name = weight_scale_inv_name
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                    loaded_params.add(name)
                    continue

        debug_layers = {0, 59, 60}
        for module_name, module in self.named_modules():
            if not module_name.endswith("attn.fused_wqa_wkv"):
                continue
            match = re.search(r"layers\.(\d+)\.attn\.fused_wqa_wkv$", module_name)
            if match is None or int(match.group(1)) not in debug_layers:
                continue
            weight = getattr(module, "weight", None)
            if weight is not None:
                wf = weight.detach().float()
                print(
                    "[DSV4_LOAD_FUSED_WQA_WKV_WEIGHT] "
                    f"name={module_name} shape={tuple(weight.shape)} "
                    f"dtype={weight.dtype} min={wf.min().item():.6g} "
                    f"max={wf.max().item():.6g} mean={wf.mean().item():.6g} "
                    f"nonzero={(wf != 0).float().mean().item():.6g}",
                    flush=True,
                )
            scale_infos = []
            for param_name, param in module.named_parameters(recurse=True):
                if "scale" not in param_name:
                    continue
                pf = param.detach().float()
                scale_infos.append(
                    f"{param_name}:shape={tuple(param.shape)},dtype={param.dtype},"
                    f"min={pf.min().item():.6g},max={pf.max().item():.6g},"
                    f"mean={pf.mean().item():.6g},"
                    f"nonzero={(pf != 0).float().mean().item():.6g}"
                )
            for buffer_name, buffer in module.named_buffers(recurse=True):
                if "scale" not in buffer_name:
                    continue
                bf = buffer.detach().float()
                scale_infos.append(
                    f"buffer.{buffer_name}:shape={tuple(buffer.shape)},"
                    f"dtype={buffer.dtype},min={bf.min().item():.6g},"
                    f"max={bf.max().item():.6g},mean={bf.mean().item():.6g},"
                    f"nonzero={(bf != 0).float().mean().item():.6g}"
                )
            print(
                "[DSV4_LOAD_FUSED_WQA_WKV_SCALES] "
                f"name={module_name} "
                f"{' | '.join(scale_infos) if scale_infos else '<none>'}",
                flush=True,
            )

        return loaded_params

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        return fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="w1",
            ckpt_down_proj_name="w2",
            ckpt_up_proj_name="w3",
            num_experts=self.config.n_routed_experts,
        )


def _make_deepseek_v4_weights_mapper(expert_dtype: str) -> WeightsMapper:
    if expert_dtype == "fp4":
        # MXFP4 experts use Mxfp4MoEMethod, which registers scales as
        # ``w{1,2,3}_weight_scale`` (no _inv suffix). FP8 linear and
        # shared experts use Fp8LinearMethod's block scales, which
        # register as ``weight_scale_inv``.
        scale_regex = {
            re.compile(r"(\.experts\.\d+\.w[123])\.scale$"): r"\1.weight_scale",
            re.compile(r"\.scale$"): ".weight_scale_inv",
        }
    else:
        # FP8 experts use Fp8MoEMethod (block_quant=True), which registers
        # scales as ``w{13,2}_weight_scale_inv``. Map all ``.scale`` keys
        # there.
        scale_regex = {
            re.compile(r"\.scale$"): ".weight_scale_inv",
        }
    return WeightsMapper(
        orig_to_new_prefix={
            "layers.": "model.layers.",
            "embed.": "model.embed.",
            "norm.": "model.norm.",
            "hc_head": "model.hc_head",
            "mtp.": "model.mtp.",
        },
        orig_to_new_regex=scale_regex,
        orig_to_new_suffix={
            "head.weight": "lm_head.weight",
            "embed.weight": "embed_tokens.weight",
            ".ffn.gate.bias": ".ffn.gate.e_score_correction_bias",
        },
        orig_to_new_substr={
            ".shared_experts.w2": ".shared_experts.down_proj",
        },
    )


class DeepseekV4ForCausalLM(nn.Module, SupportsPP):
    model_cls = DeepseekV4Model
    packed_modules_mapping = {
        "gate_up_proj": ["w1", "w3"],
        "fused_wqa_wkv": ["wq_a", "wkv"],
    }

    # Default mapper assumes the original FP4-expert checkpoint layout.
    # Overridden per-instance in __init__ when expert_dtype != "fp4".
    hf_to_vllm_mapper = _make_deepseek_v4_weights_mapper("fp4")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        self.config = config
        self._dsv4_amd_debug_counts: dict[str, int] = {}
        expert_dtype = getattr(config, "expert_dtype", "fp4")
        if expert_dtype != "fp4":
            self.hf_to_vllm_mapper = _make_deepseek_v4_weights_mapper(expert_dtype)

        self.model = self.model_cls(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (  # type: ignore[method-assign]
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def _debug_lm_head_info(self) -> None:
        label = "DSV4_AMD_LM_HEAD_INFO"
        if self._dsv4_amd_debug_counts.get(label, 0) > 0:
            return
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            return
        self._dsv4_amd_debug_counts[label] = 1
        weight = getattr(self.lm_head, "weight", None)
        embed_weight = getattr(getattr(self.model, "embed_tokens", None), "weight", None)
        shard_indices = getattr(self.lm_head, "shard_indices", None)
        with torch.no_grad():
            if weight is None:
                logger.warning("[%s] rank=%s lm_head_weight=None", label, get_tensor_model_parallel_rank())
                return
            wf = weight.detach().float()
            logger.warning(
                "[%s] rank=%s config_vocab=%s lm_head_num_embeddings=%s "
                "lm_head_org_vocab=%s lm_head_padded=%s per_partition=%s "
                "weight_shape=%s weight_dtype=%s finite=%s mean=%.6g std=%.6g "
                "max=%.6g nonzero=%.6g shard=%s",
                label,
                get_tensor_model_parallel_rank(),
                self.config.vocab_size,
                getattr(self.lm_head, "num_embeddings", None),
                getattr(self.lm_head, "org_vocab_size", None),
                getattr(self.lm_head, "num_embeddings_padded", None),
                getattr(self.lm_head, "num_embeddings_per_partition", None),
                tuple(weight.shape),
                weight.dtype,
                torch.isfinite(wf).all().item(),
                wf.mean().item(),
                wf.std().item(),
                wf.abs().max().item(),
                (wf != 0).float().mean().item(),
                shard_indices,
            )
            if embed_weight is not None:
                ewf = embed_weight.detach().float()
                logger.warning(
                    "[%s_EMBED] rank=%s embed_shape=%s embed_dtype=%s "
                    "finite=%s mean=%.6g std=%.6g max=%.6g nonzero=%.6g",
                    label,
                    get_tensor_model_parallel_rank(),
                    tuple(embed_weight.shape),
                    embed_weight.dtype,
                    torch.isfinite(ewf).all().item(),
                    ewf.mean().item(),
                    ewf.std().item(),
                    ewf.abs().max().item(),
                    (ewf != 0).float().mean().item(),
                )
            interesting_ids = [1, 14, 16, 19, 28, 67, 223, 271, 294, 305, 6328, 13318]
            for token_id in interesting_ids:
                local_idx = None
                if shard_indices is None:
                    if token_id < weight.shape[0]:
                        local_idx = token_id
                elif (
                    shard_indices.org_vocab_start_index
                    <= token_id
                    < shard_indices.org_vocab_end_index
                ):
                    local_idx = token_id - shard_indices.org_vocab_start_index
                elif (
                    shard_indices.added_vocab_start_index
                    <= token_id
                    < shard_indices.added_vocab_end_index
                ):
                    local_idx = (
                        shard_indices.num_org_elements_padded
                        + token_id
                        - shard_indices.added_vocab_start_index
                    )
                if local_idx is None or local_idx >= weight.shape[0]:
                    continue
                row = wf[local_idx]
                row_msg = (
                    "[%s_ROW] rank=%s token_id=%s local_idx=%s mean=%.6g "
                    "std=%.6g max=%.6g norm=%.6g nonzero=%.6g"
                )
                row_args = [
                    label,
                    get_tensor_model_parallel_rank(),
                    token_id,
                    local_idx,
                    row.mean().item(),
                    row.std().item(),
                    row.abs().max().item(),
                    torch.linalg.vector_norm(row).item(),
                    (row != 0).float().mean().item(),
                ]
                if embed_weight is not None and local_idx < embed_weight.shape[0]:
                    embed_row = ewf[local_idx]
                    diff = (row - embed_row).abs()
                    row_msg += " embed_diff_max=%.6g embed_diff_mean=%.6g"
                    row_args.extend([diff.max().item(), diff.mean().item()])
                logger.warning(row_msg, *row_args)

    def _debug_manual_lm_head_logits(self, hidden_states: torch.Tensor) -> None:
        weight = getattr(self.lm_head, "weight", None)
        if weight is None or hidden_states is None:
            return
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            return
        phase = _dsv4_amd_debug_phase(hidden_states=hidden_states)
        label = f"DSV4_AMD_MANUAL_LM_HEAD_LOCAL_TOPK_{phase}"
        count = self._dsv4_amd_debug_counts.get(label, 0)
        if count >= 16:
            return
        self._dsv4_amd_debug_counts[label] = count + 1
        with torch.no_grad():
            h = hidden_states.detach().reshape(-1, hidden_states.shape[-1])[-1].float()
            w = weight.detach().float()
            local_logits = torch.matmul(w, h)
            vals, local_ids = torch.topk(local_logits, k=min(10, local_logits.numel()))
            global_ids = [
                _dsv4_amd_local_vocab_id(self.lm_head, int(idx))
                for idx in local_ids.detach().cpu().tolist()
            ]
            logger.warning(
                "[%s] count=%s rank=%s hidden_shape=%s weight_shape=%s "
                "local_ids=%s global_ids=%s vals=%s hidden_norm=%.6g",
                label,
                count,
                get_tensor_model_parallel_rank(),
                tuple(hidden_states.shape),
                tuple(weight.shape),
                local_ids.detach().cpu().tolist(),
                global_ids,
                [round(v, 6) for v in vals.detach().cpu().tolist()],
                torch.linalg.vector_norm(h).item(),
            )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        self._debug_lm_head_info()
        self._debug_manual_lm_head_logits(hidden_states)
        _dsv4_amd_debug_tensor(
            self._dsv4_amd_debug_counts,
            "DSV4_AMD_CAUSAL_LM_LOGITS_INPUT",
            hidden_states,
            "causal_lm",
        )
        logits = self.logits_processor(self.lm_head, hidden_states)
        _dsv4_amd_debug_tensor(
            self._dsv4_amd_debug_counts,
            "DSV4_AMD_CAUSAL_LM_LOGITS_OUT",
            logits,
            "causal_lm",
        )
        _dsv4_amd_debug_logits_topk(
            self._dsv4_amd_debug_counts,
            "DSV4_AMD_CAUSAL_LM_LOGITS_TOPK",
            logits,
            "causal_lm",
        )
        return logits

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        _dsv4_amd_debug_tensor(
            self._dsv4_amd_debug_counts,
            "DSV4_AMD_CAUSAL_LM_FORWARD_OUT",
            hidden_states if isinstance(hidden_states, torch.Tensor) else None,
            "causal_lm",
            input_ids=input_ids,
            positions=positions,
        )
        return hidden_states

    def get_mtp_target_hidden_states(self) -> torch.Tensor | None:
        """Pre-hc_head residual stream buffer (max_num_batched_tokens,
        hc_mult * hidden_size) for the MTP draft model. Populated by
        forward(); valid after each target step."""
        return getattr(self.model, "_mtp_hidden_buffer", None)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self, skip_substrs=["mtp."])
        loaded_params = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
        return loaded_params

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()

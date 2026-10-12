# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Kimi-K3 MI355X mono decode's spec, by the fields of RFC #60904's
``MonoSpec``: the kernel layers, the widths, the supported features and the
refusals at config time (``refuse_config``), after load (``refuse_model``) and
at every cache bind (``refuse_caches``). A refusal lists every reason."""

from typing import TYPE_CHECKING

import torch

from vllm.config import CUDAGraphMode
from vllm.models.kimi_k3.amd.mono.common.plan import BLOCKS

if TYPE_CHECKING:
    from vllm.config import VllmConfig

NAME = "kimi_k3_mi355x"
TP = 8
MAX_TOKENS = 8  # a launch's x rows fit one CTA's LDS


class MonoUnsupported(Exception):
    """The configuration or the loaded model is outside what the kernels serve."""


# Whole layer: K1 (KDA) + K2. FFN layer: vLLM's MLA up to o_proj's input, then
# K2 (o_proj, its all-reduce, the MLP AttnRes and the MoE).
WHOLE, FFN = "whole", "ffn"

# Features beyond plain TP8 decode that the kernels serve.
SUPPORTS = frozenset({"speculative_decoding", "dcp", "eagle_aux_layers"})


def widths(num_spec: int) -> tuple[int, ...]:
    """Rows a launch serves: whole spec-verify requests of ``num_spec + 1``
    tokens, at most MAX_TOKENS rows."""
    q = num_spec + 1
    return tuple(range(q, MAX_TOKENS + 1, q))


def layers(config) -> dict[int, str]:
    """Decoder layer -> kind, for every layer a kernel serves: the MoE layers
    with AttnRes and at least one block before them, minus the MLA block-write
    layers (none in Kimi-K3)."""
    block = config.attn_res_block_size
    if block is None or config.num_experts is None:
        return {}
    out = {}
    for i in range(1, config.num_hidden_layers):
        if i < config.first_k_dense_replace or i % config.moe_layer_freq:
            continue
        if config.is_kda_layer(i):
            out[i] = WHOLE
        elif i % block:
            out[i] = FFN
    return out


def refuse_platform() -> list[str]:
    from vllm.platforms import current_platform

    if not current_platform.is_rocm():
        return ["needs ROCm"]
    from vllm.platforms.rocm import on_gfx950

    if not on_gfx950():
        return ["needs gfx950 (MI355X)"]
    if torch.cuda.get_device_properties().multi_processor_count != BLOCKS:
        # every CTA of a launch must be resident: a partitioned GPU deadlocks
        return [f"needs {BLOCKS} CUs (a partitioned GPU has fewer)"]
    return []


def refuse_config(vllm_config: "VllmConfig") -> list[str]:
    reasons = refuse_platform()
    pc = vllm_config.parallel_config
    if pc.tensor_parallel_size != TP:
        reasons.append(f"needs TP{TP}, got TP{pc.tensor_parallel_size}")
    if pc.pipeline_parallel_size > 1:
        reasons.append("pipeline parallelism")
    if pc.data_parallel_size > 1:
        reasons.append("data parallelism")
    if pc.enable_expert_parallel:
        reasons.append("expert parallelism")
    if pc.enable_eplb:
        reasons.append("EPLB")
    if pc.prefill_context_parallel_size > 1:
        reasons.append("prefill context parallelism")
    if vllm_config.lora_config is not None:
        reasons.append("LoRA")
    if vllm_config.kv_transfer_config is not None:
        # a connector's copy kernels on another stream can hold the CUs the
        # resident grid spin-waits on
        reasons.append("KV connectors")
    if vllm_config.aux_output_config.enable_return_routed_experts:
        reasons.append("routed-expert capture")
    if vllm_config.model_config.enable_sleep_mode:
        reasons.append("sleep mode")
    oc = vllm_config.offload_config
    if oc.uva.cpu_offload_gb > 0 or oc.prefetch.offload_group_size > 0:
        reasons.append("CPU weight offload")
    if vllm_config.weight_transfer_config is not None:
        reasons.append("weight updates")
    if not vllm_config.use_v2_model_runner:
        reasons.append("needs Model Runner V2")

    spec = vllm_config.speculative_config
    if spec is None:
        reasons.append("needs speculative decoding (spec-verify steps)")
    elif spec.num_speculative_tokens + 1 > MAX_TOKENS:
        reasons.append(
            f"{spec.num_speculative_tokens} speculative tokens: a request's "
            f"{spec.num_speculative_tokens + 1} rows exceed {MAX_TOKENS}"
        )
    else:
        cc = vllm_config.compilation_config
        mode = cc.cudagraph_mode
        if mode != CUDAGraphMode.NONE and mode.decode_mode() != CUDAGraphMode.FULL:
            reasons.append(f"CUDA graph mode {mode.name}: decode needs FULL graphs")
        elif mode != CUDAGraphMode.NONE:
            # a width captured as is, so no replay pads a request's rows
            sizes = set(cc.cudagraph_capture_sizes or ())
            missing = [w for w in widths(spec.num_speculative_tokens) if w not in sizes]
            if missing:
                reasons.append(f"CUDA graph capture sizes miss widths {missing}")
    return reasons


def refuse_model(model) -> list[str]:
    """``model``: the loaded ``KimiLinearModel``; every kernel layer's weights
    as the kernels read them, and the MoE backend that lays them out."""
    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import Mxfp4MoeBackend
    from vllm.models.kimi_k3.amd.mono.attention import back, kda
    from vllm.models.kimi_k3.amd.mono.stages import moe

    reasons: list[str] = []
    bf, f32 = torch.bfloat16, torch.float32

    def need(name, t, shape, dtype):
        if t is None:
            reasons.append(f"{name}: missing")
        elif tuple(t.shape) != shape or t.dtype != dtype or not t.is_contiguous():
            reasons.append(
                f"{name}: {tuple(t.shape)} {t.dtype}"
                f"{'' if t.is_contiguous() else ' strided'},"
                f" kernel reads contiguous {shape} {dtype}"
            )

    kinds = layers(model.config)
    if not kinds:
        return ["no layer has an AttnRes latent MoE"]
    for i, kind in kinds.items():
        layer = model.layers[i]
        p = f"layer {i}"
        m = layer.mlp
        if not getattr(m, "use_latent_moe", False) or m.shared_experts is None:
            reasons.append(f"{p}: needs a latent MoE with shared experts")
            continue
        if m.routed_scaling_factor != 1.0:
            reasons.append(f"{p}: routed_scaling_factor {m.routed_scaling_factor}")
        ex = getattr(m.experts, "routed_experts", m.experts)
        qm = getattr(ex, "quant_method", None)
        if getattr(
            qm, "mxfp4_backend", None
        ) != Mxfp4MoeBackend.AITER_MXFP4_BF16 or not getattr(
            qm, "is_k3_situ_aiter", False
        ):
            reasons.append(f"{p}: MoE backend must be AITER_MXFP4_BF16 (K3 SiTU)")
            continue
        H, LAT, E = moe.HIDDEN, moe.LAT, moe.E
        need(f"{p} gate", m.gate.weight, (E, H), bf)
        need(f"{p} gate bias", m.gate.e_score_correction_bias, (E,), f32)
        need(f"{p} latent down", m.routed_expert_down_proj.weight, (LAT, H), bf)
        need(f"{p} latent norm", m.routed_expert_norm.weight, (LAT,), bf)
        need(f"{p} shared gate_up", m.shared_experts.gate_up_proj.weight,
             (2 * moe.SI, H), bf)  # fmt: skip
        need(f"{p} shared down", m.shared_experts.down_proj.weight, (H, moe.SI), bf)
        sharded = getattr(m.routed_output_transform, "row_sharded", False)
        need(f"{p} up_proj", m.routed_expert_up_proj.weight,
             (moe.UP_N if sharded else H, LAT), bf)  # fmt: skip
        for name in ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale"):
            if getattr(ex, name, None) is None:
                reasons.append(f"{p}: experts have no {name}")
        need(f"{p} mlp_res_norm", layer.mlp_res_norm.weight, (H,), bf)
        need(f"{p} mlp_res_proj", layer.mlp_res_proj.weight, (1, H), bf)
        need(f"{p} post_attention_layernorm", layer.post_attention_layernorm.weight,
             (H,), bf)  # fmt: skip
        need(f"{p} o_proj", layer.self_attn.o_proj.weight, (H, back.OK), bf)
        if kind != WHOLE:
            continue
        a = layer.self_attn
        if a.local_num_heads != kda.NH or a.head_dim != kda.HD or a.conv_size != 4:
            reasons.append(f"{p}: KDA heads/head_dim/conv_size")
            continue
        need(f"{p} in_proj", a.in_proj_qkvgfab.weight, (kda.NPROJ, H), bf)
        need(f"{p} f_b_proj", a.f_b_proj.weight, (kda.PROJ, kda.HD), bf)
        need(f"{p} o_norm", a.o_norm.weight, (kda.HD,), bf)
        need(f"{p} conv1d", a.conv1d.weight.view(-1, 4), (kda.QKV, 4), f32)
        need(f"{p} A_log", a.A_log.view(-1), (kda.NH,), f32)
        need(f"{p} dt_bias", a.dt_bias.view(-1), (kda.PROJ,), f32)
        need(f"{p} input_layernorm", layer.input_layernorm.weight, (H,), bf)
        need(f"{p} self_attention_res_norm", layer.self_attention_res_norm.weight,
             (H,), bf)  # fmt: skip
        need(f"{p} self_attention_res_proj", layer.self_attention_res_proj.weight,
             (1, H), bf)  # fmt: skip
    return reasons


def refuse_caches(i: int, conv_state: torch.Tensor, rstate: torch.Tensor) -> list[str]:
    """A KDA layer's conv / recurrent state as K1 reads them."""
    from vllm.models.kimi_k3.amd.mono.attention import kda

    reasons = []
    if conv_state.dtype != torch.bfloat16:
        reasons.append(f"layer {i}: conv state {conv_state.dtype}, needs bf16")
    if rstate.dtype != torch.float32:
        reasons.append(f"layer {i}: recurrent state {rstate.dtype}, needs fp32")
    elif rstate[0].numel() != kda.NH * kda.HD * kda.HD or not rstate[0].is_contiguous():
        reasons.append(f"layer {i}: a slot's recurrent state must be dense")
    return reasons

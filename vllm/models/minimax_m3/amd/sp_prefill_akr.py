# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The amd_kernels `sp_prefill` route (TP4 or TP2): the sequence-parallel prefill
from the package's `MIN_TOKENS`. Each rank keeps a shard of the residual stream; per
layer the two all-reduces become the hand-offs of
`amd_kernels.collectives.seq_parallel_handoffs_h6144_q2048_ranks4` (TP4) or
`seq_parallel_handoffs_h6144_q4096_ranks2` (TP2) (pull collectives for a single
batch, SDMA pushes per DBO microbatch) around the layer's own `attend` and the FFN of
every token, with the whole o_proj on every rank. With the full shared expert on
every rank (TP2, `akr_moe.replicate_shared`) the MoE input gather carries only the
routed FP4 payload and each rank adds its own rows' shared expert in the
reduce-scatter."""

import functools

import torch

from vllm.distributed import get_tensor_model_parallel_world_size, get_tp_group
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.kernels.linear.mxfp8 import rocm_akr
from vllm.models.minimax_m3.amd import akr_attn, akr_config, akr_moe
from vllm.v1.worker.ubatching import dbo_current_ubatch_id, dbo_enabled, dbo_yield

logger = init_logger(__name__)


# TP size -> hand-offs package
_LIBS = {
    4: "seq_parallel_handoffs_h6144_q2048_ranks4",
    2: "seq_parallel_handoffs_h6144_q4096_ranks2",
}


@functools.cache
def _lib():
    """The hand-offs package of this TP size; another TP size gets the four-rank one,
    whose `WORLD` turns the route off."""
    name = _LIBS.get(get_tensor_model_parallel_world_size(), _LIBS[4])
    return akr_config.package(f"collectives.{name}", f"collectives.{_LIBS[4]}")


# from the serving config (make_o_proj_full): batch-token limit, microbatches per step
_CFG = {"max_tokens": 0, "ubatches": 0}
# the single-batch hand-offs and one SDMA hand-off per DBO microbatch, built at the
# first SP forward (collective)
_HANDOFFS = None
_SDMA: list = []


def make_o_proj_full(layer, quant_config, prefix: str):
    """At decoder-layer init: the whole o_proj on amd_kernels' MXFP8 GEMMs (+3.1 GB
    per rank), or None where the sequence-parallel prefill cannot run."""
    from vllm.config import get_current_vllm_config
    from vllm.distributed import get_pp_group
    from vllm.model_executor.layers.linear import ReplicatedLinear

    if not akr_config.on("sp_prefill"):
        return None
    lib = _lib()
    if (
        get_tensor_model_parallel_world_size() != lib.WORLD
        or get_pp_group().world_size > 1
        or not rocm_akr.is_akr_layer(layer.self_attn.qkv_proj)
    ):
        return None
    o_proj = layer.self_attn.o_proj
    full = ReplicatedLinear(
        o_proj.input_size,
        o_proj.output_size,
        bias=False,
        quant_config=quant_config,
        prefix=f"{prefix}.self_attn.o_proj",
    )
    if not rocm_akr.mark_akr_linear(full):
        return None
    cfg = get_current_vllm_config()
    _CFG["max_tokens"] = int(cfg.scheduler_config.max_num_batched_tokens)
    if cfg.parallel_config.use_ubatching:
        _CFG["ubatches"] = cfg.parallel_config.num_ubatches
    return full


def full_o_proj_layers(model) -> frozenset[int]:
    """The decoder layers of `model` (a `MiniMaxM3Model`) with an o_proj_full."""
    return frozenset(
        i
        for i in range(model.start_layer, model.end_layer)
        if getattr(model.layers[i].self_attn, "o_proj_full", None) is not None
    )


def load_o_proj_full(weights, layers: frozenset[int]):
    """A `WeightsMapper` weights transform: each checkpoint o_proj tensor of a layer
    in `layers` once more, for its o_proj_full."""
    from vllm.model_executor.models.utils import extract_layer_index

    for name, weight in weights:
        yield name, weight
        if ".self_attn.o_proj." in name and extract_layer_index(name) in layers:
            yield name.replace(".self_attn.o_proj.", ".self_attn.o_proj_full."), weight


def _unsupported(layer) -> str | None:
    """None when `layer` can run the amd_kernels SP layer, else why not."""
    lib = _lib()
    attn = layer.self_attn
    if not hasattr(attn, "indexer"):
        return "dense attention"
    if attn.num_heads * attn.head_dim != lib.Q:
        return f"attention output per rank {attn.num_heads * attn.head_dim} != {lib.Q}"
    if not rocm_akr.is_akr_layer(attn.qkv_proj):
        return "qkv_proj not on amd_kernels"
    if attn.o_proj_full is None or not rocm_akr.is_akr_layer(attn.o_proj_full):
        return "no amd_kernels o_proj_full"
    for norm in (layer.input_layernorm, layer.post_attention_layernorm):
        if norm.weight.dtype != torch.bfloat16 or norm.weight.shape != (lib.H,):
            return f"norm weights not bf16 [{lib.H}]"
    if layer.is_moe_layer:
        moe = layer.block_sparse_moe
        if not getattr(moe, "use_akr_moe", False) or moe._akr_weights is None:
            return "MoE block not on amd_kernels"
        shared = moe._akr_weights.shared_weights
        if shared is None:
            return "MoE block without the shared expert"
        if getattr(shared, "full", False) and not getattr(lib, "OWNER_SHARED", False):
            return "the full shared expert needs the hand-offs' owner form"
    else:
        mlp = layer.mlp
        if not (
            rocm_akr.is_akr_layer(mlp.gate_up_proj)
            and rocm_akr.is_akr_layer(mlp.down_proj)
        ):
            return "dense MLP not on amd_kernels"
    return None


def plan_model(model, hidden_states: torch.Tensor) -> bool:
    """Whether this forward runs the amd_kernels SP layers."""
    num_tokens = hidden_states.shape[0]
    min_tokens = _lib().MIN_TOKENS
    if (
        torch.cuda.is_current_stream_capturing()
        or num_tokens < min_tokens
        or not _check(model, hidden_states)
    ):
        return False
    if dbo_enabled():
        # one path for every microbatch of the step: their DBO yields must pair up
        slices = get_forward_context().ubatch_slices
        sizes = [s.num_tokens for s in slices] if slices else [num_tokens]
        return len(sizes) == len(_SDMA) and all(
            n >= min_tokens and sdma.supports(n) for n, sdma in zip(sizes, _SDMA)
        )
    assert _HANDOFFS is not None
    return _HANDOFFS.supports(num_tokens)


def _check(model, hidden_states: torch.Tensor) -> bool:
    """At the first SP-sized forward (the profile run): check the layers once and
    build the communicators."""
    if hasattr(model, "_akr_sp_supported"):
        return model._akr_sp_supported
    layers = model.layers[model.start_layer : model.end_layer]
    reason = None
    for layer in layers:
        why = _unsupported(layer)
        if why is not None:
            reason = f"layer {layer.layer_id}: {why}"
            break
    model._akr_sp_supported = reason is None
    if reason is None:
        _init(hidden_states.device, layers)
    else:
        logger.info("amd_kernels sp_prefill: off (%s)", reason)
    return model._akr_sp_supported


def _init(device: torch.device, layers) -> None:
    """Build and warm up the communicators (collective: every TP rank)."""
    global _HANDOFFS
    if _HANDOFFS is not None:
        return
    lib = _lib()
    tp = get_tp_group()
    ph = lib.SeqParallelHandoffs(
        tp.rank_in_group, tp.cpu_group, device, _CFG["max_tokens"]
    )
    full = layers[0].self_attn.o_proj_full
    moe = next(lay.block_sparse_moe for lay in layers if lay.is_moe_layer)
    ph.warm_up(full.weight, full.weight_scale, moe._akr_weights)
    _HANDOFFS = ph
    # microbatching cuts a step's tokens at multiples of tokens // n: the last
    # microbatch takes the remainder, up to n - 1 tokens more
    n = _CFG["ubatches"]
    ub_tokens = ph.max_tokens // max(n, 1) + max(n - 1, 0)
    # a shard copy of the shared expert gathers the larger compact MoE payload
    kw = {}
    if getattr(lib, "OWNER_SHARED", False):
        kw["compact"] = not getattr(moe._akr_weights.shared_weights, "full", False)
    if n > 1 and ub_tokens >= lib.MIN_TOKENS:
        for _ in range(n):
            _SDMA.append(
                lib.SdmaSeqParallelHandoffs(
                    tp.rank_in_group, tp.cpu_group, device, ub_tokens, **kw
                )
            )
    logger.info(
        "amd_kernels sp_prefill: on from %d tokens, hand-offs for %d tokens per call "
        "(%d rows per rank); SDMA hand-offs for %d microbatches up to %d tokens",
        lib.MIN_TOKENS,
        ph.padded,
        ph.rows,
        len(_SDMA),
        ub_tokens if _SDMA else 0,
    )


def _gemm(layer, codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """A switched MXFP8 linear on row-major codes with preshuffled (AITER) scales."""
    from amd_kernels.gemm.mxfp8_common.gemm import mxfp8_gemm

    return mxfp8_gemm(
        codes,
        scales,
        layer.weight,
        layer.weight_scale,
        torch.bfloat16,
        activation_scale_preshuffled=True,
    )


def _dense_mlp(mlp, codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """The dense MLP of every token -> its TP partial `[padded, H]` (no reduce)."""
    from .ops import swiglu_oai_split

    x = swiglu_oai_split(
        _gemm(mlp.gate_up_proj, codes, scales),
        alpha=mlp.swiglu_alpha,
        beta=mlp.swiglu_beta,
        limit=mlp.swiglu_limit,
    )
    # rocm_akr's bf16-input path (mxfp8_linear quantizes in its own launch)
    return mlp.down_proj.quant_method.apply(mlp.down_proj, x)


def _attend(attn, positions: torch.Tensor, qkv: torch.Tensor):
    """The layer's attention of every token: at TP2 with the layer's `qkv` route on,
    the norms, RoPE and NVFP4 inserts on amd_kernels, else the layer's own `attend`."""
    slots = get_forward_context().slot_mapping
    if not (
        attn.akr_qkv_insert
        and get_tensor_model_parallel_world_size() == 2
        and isinstance(slots, dict)
        and attn.layer_name in slots
    ):
        return attn.attend(positions, qkv)
    q, index_q = akr_attn.qkv_norm_rope_nvfp4_insert(
        attn,
        qkv,
        positions,
        slots[attn.layer_name],
        slots[attn.indexer.index_cache.prefix],
        projected=True,
    )
    return attn._attend_nvfp4(q, index_q)


def forward(
    model,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    aux_hidden_states: list[torch.Tensor],
) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
    """The SP forward of `model` from the embeddings to the final norm."""
    if dbo_enabled():
        # this microbatch's SDMA hand-offs, DBO's yield between a phase's pushes
        # and waits
        ph, kw = _SDMA[dbo_current_ubatch_id()], {"between": dbo_yield}
    else:
        ph, kw = _HANDOFFS, {}
    layers = model.layers[model.start_layer : model.end_layer]
    t = hidden_states.shape[0]
    residual = _lib().shard(hidden_states, get_tp_group().rank_in_group)
    codes, scales = ph.input_gather(layers[0].input_layernorm(residual), **kw)
    aux: list[torch.Tensor] = []
    for idx, layer in enumerate(layers):
        attn, post = layer.self_attn, layer.post_attention_layernorm
        # the GEMM covers the padding rows too (their codes are zero): row-major
        # codes and AITER scales of whole 32-row groups
        qkv = _gemm(attn.qkv_proj, codes, scales)[:t]
        attn_output = _attend(attn, positions, qkv)
        o_w = (attn.o_proj_full.weight, attn.o_proj_full.weight_scale)
        norm = (post.weight, post.variance_epsilon)
        if layer.is_moe_layer:
            w = layer.block_sparse_moe._akr_weights
            mxfp4_moe = akr_moe.package(layer.block_sparse_moe).mxfp4_moe
            ffn_in, residual, *shared = ph.attention_moe(
                attn_output, *o_w, residual, *norm, w, **kw
            )
            del attn_output
            if shared and shared[0] is not None:
                # the full shared expert of this rank's rows (a SharedPending on the
                # hand-offs' side stream; ffn_norm_gather / ffn_reduce_scatter join it)
                slots, _ = mxfp4_moe(ffn_in, w, return_partials=True)
                ffn_out = (slots, shared[0])
                del slots
            else:
                # a shard copy: the compact payload and the [2 rows, H] shared partial
                ffn_out = mxfp4_moe(ffn_in, w, return_partials=True)
            del shared
        else:
            *ffn_in, residual = ph.attention_dense(
                attn_output, *o_w, residual, *norm, **kw
            )
            del attn_output
            ffn_out = _dense_mlp(layer.mlp, *ffn_in)
        del ffn_in
        if idx + 1 < len(layers):
            nxt = layers[idx + 1].input_layernorm
            codes, scales, residual = ph.ffn_norm_gather(
                ffn_out, residual, nxt.weight, nxt.variance_epsilon, **kw
            )
        else:
            normed, residual = model.norm(
                ph.ffn_reduce_scatter(ffn_out, **kw), residual
            )
            codes = ph.gather(normed, **kw)
        if model.start_layer + idx + 1 in model.aux_hidden_state_layers:
            aux.append(ph.gather(residual, **kw))
        del ffn_out
    aux_hidden_states += [a[:t] for a in aux]
    if aux_hidden_states:
        return codes[:t], aux_hidden_states
    return codes[:t]

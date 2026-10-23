# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The GLM-5.3-Flash kernel layers: weight packing, cache binding, step eligibility
and the launch of each width.

Every kernel layer is one launch for the whole decoder layer: the attention-side
mHC, the attention (KDA, or sparse MLA with its kpool indexer and cache writes),
o_proj with its TP reduce, the FFN mHC seam, the RMSNorm, the router and the MoE
with its TP reduce. It takes and returns what ``Glm5NextDecoderLayer`` does."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import torch

from vllm.models.glm5next.amd.specialized_decode.mono.config import (
    GLOBAL_INTER,
    HC_MULT,
    HIDDEN,
    INDEX_DIM,
    KDA_DIM,
    KPOOL,
    KV_LORA,
    FlashShard,
)

from . import spec as flash_spec
from .context import DecodeContext as MonoContext
from .context import DecodeStep as MonoStep
from .weights import dsa_weights, kda_weights, layer_weights

if TYPE_CHECKING:
    from torch import nn

# kpool pools a sparse-attention task scores
POOLS_PER_TASK = 256

# the caches each kernel layer reads: (path on the vLLM layer, dtypes, layer filter)
CACHES = {
    "conv": ("self_attn.kv_cache[0]", (torch.bfloat16,), flash_spec.is_kda),
    "rec": ("self_attn.kv_cache[1]", (torch.float32,), flash_spec.is_kda),
    "mla": (
        "self_attn.mla_attn.mla_attn.kv_cache",
        (torch.bfloat16, torch.uint8, torch.float8_e4m3fn),
        flash_spec.is_dsa,
    ),
    "icache": ("self_attn.indexer.k_cache.kv_cache", (torch.uint8,), flash_spec.is_dsa),
    "tail": (
        "self_attn.indexer.tail_cache.kv_cache",
        (torch.bfloat16,),
        flash_spec.is_dsa,
    ),
}


def _kernel_layers(model: nn.Module) -> list[int]:
    return flash_spec.kernel_layers(model.config)


def pack(model: nn.Module, ctx: MonoContext) -> dict[int, Any]:
    """Per kernel layer (FFN weights, the attention weights, launch kind)."""
    inter = GLOBAL_INTER // ctx.tp_size
    packed: dict[int, Any] = {}
    reasons: list[str] = []
    for idx in _kernel_layers(model):
        layer = model.layers[idx]
        attn: Any
        try:
            if flash_spec.is_kda(layer):
                attn, kind = kda_weights(layer, ctx.tp_size), "kda"
            else:
                attn, kind = dsa_weights(layer, ctx.tp_size), "dsa"
            packed[idx] = (layer_weights(layer, inter), attn, kind)
        except ValueError as error:
            reasons.append(f"layer {idx}: {error}")
    if reasons:
        raise ValueError("--monokernel cannot run GLM-5.3-Flash: " + "; ".join(reasons))
    kda = [_kda_name(model.layers[i]) for i in sorted(packed) if packed[i][2] == "kda"]
    ctx.state["kda_names"] = kda
    ctx.state["dsa_names"] = {
        i: _dsa_names(model.layers[i]) for i in sorted(packed) if packed[i][2] == "dsa"
    }
    formats = {packed[i][1].fp8 for i in packed if packed[i][2] == "dsa"}
    if len(formats) > 1:
        raise ValueError(
            "--monokernel cannot run GLM-5.3-Flash: sparse-attention "
            f"projections differ between layers {sorted(formats)}"
        )
    ctx.state["dsa_fp8"] = formats.pop() if formats else (True, True, True)
    # an e4m3 MLA cache and each layer's per-tensor scale (vLLM's _k_scale)
    ctx.state["kv_fp8"] = ctx.vllm_config.cache_config.cache_dtype.startswith("fp8")
    ctx.state["k_scales"] = {
        i: model.layers[i].self_attn.mla_attn.mla_attn._k_scale
        for i in sorted(packed)
        if packed[i][2] == "dsa"
    }
    return packed


def _bind_dsa(
    caches: Mapping[str, torch.Tensor], kv_fp8: bool, k_scales: Mapping[int, Any]
) -> tuple[dict, list[str]]:
    """Each sparse-attention layer's MLA latent cache [blocks, tokens, 512] (bf16,
    or e4m3 bytes with ``kv_fp8``), index cache [blocks, 16 n pools, 132] (16-pool
    tiles) and kpool tail ring [blocks, 2, 4, 128]; the kernel addresses every
    block with a 64-bit base."""
    from vllm.models.glm5next.amd.specialized_decode.mono.ffn.op import FlashDsaCaches

    reasons, out = [], {}
    for key, kv in caches.items():
        idx, name = key.split(".")
        if name != "mla":
            continue
        ic, tail = caches[f"{idx}.icache"], caches[f"{idx}.tail"]
        if kv.dim() == 4 and kv.shape[1] == 1:
            kv = kv.squeeze(1)
        if kv.dim() != 3 or kv.shape[2] != KV_LORA or kv.stride()[1:] != (KV_LORA, 1):
            reasons.append(f"layer {idx}: MLA cache {tuple(kv.shape)} {kv.stride()}")
        if (kv.element_size() == 1) != kv_fp8:
            reasons.append(f"layer {idx}: {kv.dtype} MLA cache, kv_fp8={kv_fp8}")
        if (
            ic.dim() != 3
            or ic.shape[2] != INDEX_DIM + 4
            or ic.stride()[1:] != (INDEX_DIM + 4, 1)
            or ic.stride(0) != ic.shape[1] * (INDEX_DIM + 4)
            or ic.shape[1] % 16
        ):
            reasons.append(f"layer {idx}: index cache {tuple(ic.shape)} {ic.stride()}")
        if (
            tail.dim() != 4
            or tuple(tail.shape[1:]) != (2, KPOOL, INDEX_DIM)
            or tail.stride()[2:] != (INDEX_DIM, 1)
        ):
            reasons.append(f"layer {idx}: kpool tail {tuple(tail.shape)}")
        k_scale = k_scales.get(int(idx)) if kv_fp8 else None
        if kv_fp8 and k_scale is None:
            reasons.append(f"layer {idx}: fp8 MLA cache without its k_scale")
        out[int(idx)] = FlashDsaCaches(kv, ic, tail, k_scale)
    return out, reasons


def bind(ctx: MonoContext, caches: Mapping[str, torch.Tensor]) -> list[str]:
    """Each KDA layer's conv state ([slots, 3P, 3] dim first, else [slots, 3, 3P])
    and recurrent state [slots, H, 128, 128], and each sparse-attention layer's
    caches, kept for the launches."""
    from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
    from vllm.models.glm5next.amd.specialized_decode.mono.ffn.op import FlashKdaState

    shard = FlashShard(ctx.tp_size)
    P, H, D = shard.kda_proj, shard.kda_heads, KDA_DIM
    reasons, states = [], {}
    conv_shape, conv_strides = (3 * P, 3), (3, 1)
    if not is_conv_state_dim_first():
        conv_shape, conv_strides = (3, 3 * P), (3 * P, 1)
    for key, conv in caches.items():
        idx, name = key.split(".")
        if name != "conv":
            continue
        rec = caches[f"{idx}.rec"]
        if tuple(conv.shape[1:]) != conv_shape or conv.stride()[1:] != conv_strides:
            reasons.append(f"layer {idx}: conv state {tuple(conv.shape)}")
        if tuple(rec.shape[1:]) != (H, D, D) or rec.stride()[1:] != (D * D, D, 1):
            reasons.append(f"layer {idx}: recurrent state {tuple(rec.shape)}")
        if conv.shape[0] != rec.shape[0]:
            reasons.append(f"layer {idx}: {conv.shape[0]} conv, {rec.shape[0]} slots")
        states[int(idx)] = FlashKdaState(conv, rec)
    dsa, dsa_reasons = _bind_dsa(
        caches, ctx.state.get("kv_fp8", False), ctx.state.get("k_scales", {})
    )
    reasons += dsa_reasons
    if not reasons:
        ctx.state["kda_states"] = states
        ctx.state["dsa_caches"] = dsa
    return reasons


def eligible(ctx: MonoContext, step: MonoStep) -> str | None:
    cap = flash_spec.TP_MAX_WIDTH.get(ctx.tp_size)
    if cap is not None and step.width > cap:
        return f"more than {cap} rows at TP{ctx.tp_size}"
    for name in ctx.state.get("kda_names", ()):
        meta = step.attn_metadata[name]
        if getattr(meta, "num_spec_decodes", 0):
            return "speculative decode rows"
        indices = getattr(meta, "non_spec_state_indices_tensor", None)
        if indices is None or indices.shape[0] < step.tokens:
            return "no KDA state index a row"
    for _, icache, _ in ctx.state.get("dsa_names", {}).values():
        # one index-cache group for every sparse-attention layer
        if step.attn_metadata[icache].decode is None:
            return "no indexer decode metadata"
        break
    return None


def metadata_names(layers: Mapping[int, nn.Module]) -> list[str]:
    """The kernel layers' attention entries of the forward context (KDA layers;
    MLA, index cache and kpool tail of sparse-attention ones); every decode step
    carries their metadata."""
    from vllm.config import get_current_vllm_config

    ctx = get_current_vllm_config().compilation_config.static_forward_context
    mods = set()
    for layer in layers.values():
        attn = layer.self_attn
        if flash_spec.is_kda(layer):
            mods.add(id(attn))
        else:
            idx = attn.indexer
            mods |= {id(attn.mla_attn.mla_attn), id(idx.k_cache), id(idx.tail_cache)}
    return sorted(name for name, module in ctx.items() if id(module) in mods)


def _kda_name(layer: nn.Module) -> str:
    return layer.self_attn.prefix


def _dsa_names(layer: nn.Module) -> tuple[str, str, str]:
    """(MLA layer, index cache, kpool tail) forward-context names."""
    attn = layer.self_attn
    idx = attn.indexer
    return attn.mla_attn.mla_attn.layer_name, idx.k_cache.prefix, idx.tail_cache.prefix


def max_pools(ctx: MonoContext) -> int:
    """Pools a sparse-attention launch scores at most (kpool): the model length's,
    whole score tasks."""
    n = -(-ctx.vllm_config.model_config.max_model_len // KPOOL)
    return -(-n // POOLS_PER_TASK) * POOLS_PER_TASK


class _Width:
    """Per-width state of one launch kind: the runtime, padding inputs and, for
    KDA, padded copies of the step's state indices; for sparse attention, the
    step's row table."""

    def __init__(self, runtime, width: int, device: torch.device) -> None:
        from vllm.models.glm5next.amd.specialized_decode.mono.ffn.layout import (
            ROW_BT,
            ROW_WORDS,
        )

        self.rt = runtime
        self.pad = (
            torch.zeros(width, HIDDEN, dtype=torch.bfloat16, device=device),
            torch.zeros(width, HC_MULT, HIDDEN, dtype=torch.bfloat16, device=device),
            torch.zeros(width, HC_MULT, 1, dtype=torch.float32, device=device),
            torch.zeros(width, HC_MULT, HC_MULT, dtype=torch.float32, device=device),
        )
        self.width, self.device = width, device
        self.idx_pool: list[torch.Tensor] = []
        self.step: MonoStep | None = None
        self.idx_of: dict[int, torch.Tensor] = {}
        # decode rows are one token a request: row i reads block-table row i
        self.rows = torch.full((width, ROW_WORDS), -1, dtype=torch.int32, device=device)
        self.rows[:, ROW_BT] = torch.arange(width, dtype=torch.int32, device=device)
        self.rows_step: MonoStep | None = None

    def state_idx(self, step: MonoStep, src: torch.Tensor) -> torch.Tensor:
        """KDA layers of one cache group share ``src``; vLLM pads graph rows with
        the null block 0. A step narrower than the width gets one padded copy per
        group (padding -1), in the same order every step."""
        n = step.tokens
        if n == self.width and src.dtype == torch.int32 and src.is_contiguous():
            return src[:n]
        if self.step is not step:
            self.step, self.idx_of = step, {}
        key = src.data_ptr()
        if key not in self.idx_of:
            k = len(self.idx_of)
            if k == len(self.idx_pool):
                self.idx_pool.append(
                    torch.full((self.width,), -1, dtype=torch.int32, device=self.device)
                )
            buf = self.idx_pool[k]
            buf[:n].copy_(src[:n])
            buf[n:].fill_(-1)
            self.idx_of[key] = buf
        return self.idx_of[key]

    def dsa_rows(self, step: MonoStep, names: tuple[str, str, str]) -> torch.Tensor:
        """The step's (position, MLA slot, index-cache pool slot, tail slot) rows;
        every sparse-attention layer shares them (one cache group each)."""
        from vllm.models.glm5next.amd.specialized_decode.mono.ffn.layout import (
            ROW_POOL_SLOT,
            ROW_POS,
            ROW_SLOT,
            ROW_TAIL_SLOT,
        )

        if self.rows_step is step:
            return self.rows
        mla, icache, tail = names
        n, r = step.tokens, self.rows
        r[:n, ROW_POS].copy_(step.positions[:n])
        r[:n, ROW_SLOT].copy_(step.attn_metadata[mla].slot_mapping[:n])
        r[:n, ROW_POOL_SLOT].copy_(step.attn_metadata[icache].slot_mapping[:n])
        r[:n, ROW_TAIL_SLOT].copy_(step.attn_metadata[tail].slot_mapping[:n])
        if n < self.width:
            r[n:, ROW_POS : ROW_TAIL_SLOT + 1].fill_(-1)
        self.rows_step = step
        return r


def _runtime(ctx: MonoContext, width: int, launches: int, kind: str):
    from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
    from vllm.models.glm5next.amd.specialized_decode.mono.ffn.op import FlashFfnRuntime

    widths = ctx.state.setdefault("widths", {})
    if (width, kind) not in widths:
        first = next(iter(widths.values()), None)
        rt = FlashFfnRuntime(
            width,
            ctx.tp_rank,
            ctx.tp_size,
            ctx.cpu_group,
            GLOBAL_INTER // ctx.tp_size,
            launches,
            ctx.state["poll_limit"],
            # after one expired wait the step's later waits return at once, so a
            # stalled peer costs one poll limit a launch kind, well inside vLLM's
            # execute timeout, and the health check stops the engine
            poll_early_out=True,
            flag_addr=None if first is None else first.rt.flag_addr,
            kda=kind == "kda",
            conv_dim_first=is_conv_state_dim_first(),
            dsa=kind == "dsa",
            max_pools=max_pools(ctx) if kind == "dsa" else 0,
            dsa_fp8=ctx.state.get("dsa_fp8", (True, True, True)),
            kv_fp8=kind == "dsa" and ctx.state.get("kv_fp8", False),
        )
        widths[(width, kind)] = _Width(rt, width, ctx.device)
        if first is None and rt.flags is not None:
            ctx.state["flags"] = rt.flags
    return widths[(width, kind)]


def _pad(w: _Width, step: MonoStep, ins):
    n, S = step.tokens, step.width
    if n < S:
        for dst, src in zip(w.pad, ins):
            dst[:n].copy_(src)
        return w.pad
    return tuple(t.contiguous() for t in ins)


class FlashLayer:
    """One whole decoder layer's kernel at one width; ``ends`` the runtimes the
    step's last kernel layer advances."""

    def __init__(self, ctx: MonoContext, layer: nn.Module, op, w: _Width, ends, idx):
        self.ctx, self.layer, self.op, self.w, self.ends = ctx, layer, op, w, ends
        self.idx = idx

    def health(self) -> torch.Tensor:
        return self.w.rt.words

    def finish(self, step: MonoStep, out):
        for rt in self.ends:
            rt.advance_step()
        return tuple(t[: step.tokens] for t in out)

    def inputs(self, step: MonoStep, hidden_states, residual, post, comb):
        if post is None:
            raise RuntimeError("a Flash kernel layer needs the previous mHC state")
        n = step.tokens
        return _pad(
            self.w, step, (hidden_states, residual, post.view(n, HC_MULT, 1), comb)
        )


class FlashKdaLayer(FlashLayer):
    """One whole KDA decoder layer, on its own cache group's state indices (rows
    whose index is not positive are skipped)."""

    def launch(self, step: MonoStep, hidden_states, residual, post, comb):
        ins = self.inputs(step, hidden_states, residual, post, comb)
        name = _kda_name(self.layer)
        idx = self.w.state_idx(
            step, step.attn_metadata[name].non_spec_state_indices_tensor
        )
        state = self.ctx.state["kda_states"][self.idx]
        return self.finish(step, self.op.launch(*ins, state, idx))


class FlashDsaLayer(FlashLayer):
    """One whole sparse-attention decoder layer: it writes the step's MLA cache
    row, kpool tail and completed pool itself."""

    def launch(self, step: MonoStep, hidden_states, residual, post, comb):
        ins = self.inputs(step, hidden_states, residual, post, comb)
        names = self.ctx.state["dsa_names"][self.idx]
        mla, icache, _ = names
        return self.finish(
            step,
            self.op.launch(
                *ins,
                caches=self.ctx.state["dsa_caches"][self.idx],
                rows=self.w.dsa_rows(step, names),
                block_table=step.attn_metadata[mla].block_table,
                idx_block_table=step.attn_metadata[icache].decode.block_table,
            ),
        )


class _Unbuilt:
    """A width a launch kind cannot run at this TP size (never eligible)."""

    def __init__(self, words: torch.Tensor) -> None:
        self.words = words

    def health(self) -> torch.Tensor:
        return self.words

    def launch(self, step: MonoStep, *args):
        raise RuntimeError(f"no Flash kernel for {step.width} rows at this TP size")


def build(ctx: MonoContext, packed: dict, layer: nn.Module, idx: int, width: int):
    from vllm.models.glm5next.amd.specialized_decode.mono.ffn.op import FlashFfnOp

    ffn, attn, kind = packed[idx]
    cap = flash_spec.TP_MAX_WIDTH.get(ctx.tp_size)
    if kind == "dsa" and cap is not None and width > cap:
        some = next(iter(ctx.state.get("widths", {}).values()))
        return _Unbuilt(some.rt.words)
    order = sorted(i for i in packed if packed[i][2] == kind)
    w = _runtime(ctx, width, len(order), kind)
    op = FlashFfnOp(
        w.rt,
        ffn,
        order.index(idx),
        attn if kind == "kda" else None,
        attn if kind == "dsa" else None,
    )
    ends = []
    if idx == max(packed):
        for k in sorted({packed[i][2] for i in packed}):
            if k == "dsa" and cap is not None and width > cap:
                continue
            n = sum(packed[i][2] == k for i in packed)
            ends.append(_runtime(ctx, width, n, k).rt)
    if kind == "kda":
        return FlashKdaLayer(ctx, layer, op, w, ends, idx)
    return FlashDsaLayer(ctx, layer, op, w, ends, idx)

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# DSpine draft head adapted from tencent-dspine (Apache-2.0).
"""DFlash backbone with the DSpine code bridge and serial last-write refinement.

The block's first row is the last committed token (the anchor). Code-bridge
writes condition each subsequent row on its predecessor; ``score_transitions``
rebuilds the last write's fp32 [B, T-1, K, K] transition table so the draft can
redo the final write for every (sender, candidate) pair.
"""

import math
from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.spec_decode.dspine.config import DSpineSpeculatorConfig

from .qwen3_dflash import (
    DFlashQwen3DecoderLayer,
    DFlashQwen3ForCausalLM,
    DFlashQwen3Model,
)

logger = init_logger(__name__)

BRIDGE_PREFIX = "selected_prefix_bridge."


# ---------------------------------------------------------------------------
# Code-bridge Triton kernels (ported from sglang_dspine/kernels.py)
# ---------------------------------------------------------------------------


@triton.jit
def _prepare_read_kernel(
    X,
    Residual,
    Raw,
    Normalized,
    RMS,
    H: tl.constexpr,
    EPS: tl.constexpr,
    ADD: tl.constexpr,
    WIDTH: tl.constexpr,
):
    """Save the residual stream and its per-row RMS before a code-bridge write.

    Raw is bf16(X + Residual) when ADD is set, otherwise X. RMS is
    sqrt(max(mean(Raw**2), fp32 tiny)); Normalized is Raw / (RMS + EPS).
    """
    row = tl.program_id(0)
    h = tl.arange(0, WIDTH)
    x = tl.load(X + row * H + h, h < H, 0).to(tl.float32)
    if ADD:
        residual = tl.load(Residual + row * H + h, h < H, 0).to(tl.float32)
        # The original layer first rounds residual+MLP to hidden dtype.
        x = (x + residual).to(X.dtype.element_ty).to(tl.float32)
    rms = tl.sqrt(tl.maximum(tl.sum(x * x, 0) / H, 1.1754943508222875e-38))
    tl.store(Raw + row * H + h, x, h < H)
    tl.store(Normalized + row * H + h, x / (rms + EPS), h < H)
    tl.store(RMS + row, rms)


@triton.jit
def _normalize_codes_kernel(
    Projected,
    Codebook,
    Anchors,
    Codes,
    C: tl.constexpr,
    T: tl.constexpr,
    ANCHOR_ROW: tl.constexpr,
    EPS: tl.constexpr,
    WIDTH: tl.constexpr,
):
    """Scale codes to sqrt(C), using the token codebook on anchor rows.

    When ANCHOR_ROW is set, anchors use sqrt(C) * Codebook[token]. Other rows
    use sqrt(C) * Projected / (norm(Projected) + EPS), from the layer's read
    projection.
    """
    row = tl.program_id(0)
    k = tl.arange(0, WIDTH)
    if ANCHOR_ROW and row % T == 0:
        token = tl.load(Anchors + row // T)
        z = tl.load(Codebook + token * C + k, k < C, 0).to(tl.float32)
        z = z * tl.sqrt(float(C))
    else:
        z = tl.load(Projected + row * C + k, k < C, 0).to(tl.float32)
        norm = tl.sqrt(tl.sum(z * z, 0))
        z = (tl.sqrt(float(C)) * z) / (norm + EPS)
    tl.store(Codes + row * C + k, z, k < C)


@triton.jit
def _feature_message_kernel(
    Packed, Bias, Message, R: tl.constexpr, T: tl.constexpr, WIDTH: tl.constexpr
):
    """Compute sigmoid(Q[r] + S[r-1] + Bias) * V[r-1], zero on anchor rows.

    Packed contains the code projections [Q | S | V], each R columns wide.
    """
    row = tl.program_id(0)
    k = tl.arange(0, WIDTH)
    query = tl.load(Packed + row * (3 * R) + k, k < R, 0).to(tl.float32)
    prev = tl.maximum(row - 1, 0)
    sender = tl.load(Packed + prev * (3 * R) + R + k, k < R, 0).to(tl.float32)
    value = tl.load(Packed + prev * (3 * R) + 2 * R + k, k < R, 0).to(tl.float32)
    bias = tl.load(Bias + k, k < R, 0).to(tl.float32)
    gate = tl.sigmoid((query + sender) + bias)
    message = tl.where(row % T == 0, 0.0, gate * value)
    tl.store(Message + row * R + k, message, k < R)


@triton.jit
def _feature_finish_kernel(
    Raw,
    Delta,
    RMS,
    Valid,
    Weight,
    Output,
    Normed,
    H: tl.constexpr,
    T: tl.constexpr,
    WRITE: tl.constexpr,
    NORM: tl.constexpr,
    NORM_EPS: tl.constexpr,
    WIDTH: tl.constexpr,
):
    """Write bf16(Raw + RMS * Delta) on valid non-anchor edges when WRITE is set.

    Delta is the projected message. Other rows retain Raw. NORM additionally
    applies the supplied RMSNorm, preserving Qwen3's cast-before-scale order.
    """
    row = tl.program_id(0)
    k = tl.arange(0, WIDTH)
    raw = tl.load(Raw + row * H + k, k < H, 0).to(tl.float32)
    delta = tl.load(Delta + row * H + k, k < H, 0).to(tl.float32)
    rms = tl.load(RMS + row)
    previous = tl.maximum(row - 1, 0)
    edge = (row % T != 0) & tl.load(Valid + row) & tl.load(Valid + previous)
    update = (rms * delta) * tl.cast(edge & WRITE, tl.float32)
    result = (raw + update).to(Output.dtype.element_ty)
    tl.store(Output + row * H + k, result, k < H)
    if NORM:
        x = result.to(tl.float32)
        variance = tl.sum(x * x, 0) / H
        y = (x * tl.rsqrt(variance + NORM_EPS)).to(Output.dtype.element_ty)
        weight = tl.load(Weight + k, k < H, 0)
        # Qwen3RMSNorm casts the normalized activation before affine multiply.
        y = (y.to(tl.float32) * weight.to(tl.float32)).to(Output.dtype.element_ty)
        tl.store(Normed + row * H + k, y, k < H)


# ---------------------------------------------------------------------------
# Refinement width and serial tables
# ---------------------------------------------------------------------------


def resolve_dspine_head_topk(config, num_layers: int | None = None) -> int:
    """Resolve the width K for serial last-write refinement.

    Read ``code_bridge_second_pass`` (mode ``serial_table``, ``first_exact`` is
    the last layer) and/or ``dflash_config.selected_prefix_last_write_refine``
    (kind ``exact_sender``), falling back to ``dflash_config.dspine_top_k`` /
    ``dflash_config.transition_top_k``. Every source found must agree.
    """
    if num_layers is None:
        num_layers = int(config.num_hidden_layers)
    num_layers = int(num_layers)
    dflash_cfg = getattr(config, "dflash_config", None) or {}
    found = {}
    second = getattr(config, "code_bridge_second_pass", None)
    if second is not None:
        second = dict(second)
        unknown = set(second) - {"mode", "first_exact", "head_topk"}
        if unknown:
            raise ValueError(f"Unknown code_bridge_second_pass keys: {sorted(unknown)}")
        if second.get("mode") != "serial_table":
            raise ValueError(
                "DSPINE serves the serial last-write refinement only; the "
                "checkpoint config says "
                f"code_bridge_second_pass.mode={second.get('mode')!r}"
            )
        if int(second.get("first_exact", -1)) != num_layers - 1:
            raise ValueError(
                "serial_table redoes the LAST write only: "
                "code_bridge_second_pass.first_exact must be "
                f"{num_layers - 1}, got {second.get('first_exact')!r}"
            )
        found["code_bridge_second_pass.head_topk"] = int(second["head_topk"])
    refine = dflash_cfg.get("selected_prefix_last_write_refine")
    if refine is not None:
        if refine.get("kind") != "exact_sender":
            raise ValueError(
                "DSPINE expects "
                "selected_prefix_last_write_refine.kind == 'exact_sender', "
                f"got {refine.get('kind')!r}"
            )
        found["dflash_config.selected_prefix_last_write_refine.head_topk"] = int(
            refine["head_topk"]
        )
    for key in ("dspine_top_k", "transition_top_k"):
        if dflash_cfg.get(key) is not None:
            found[f"dflash_config.{key}"] = int(dflash_cfg[key])
    if not found:
        raise ValueError(
            "DSPINE checkpoint config names no refinement width: add "
            "code_bridge_second_pass {mode: serial_table, first_exact, "
            "head_topk}, dflash_config.selected_prefix_last_write_refine."
            "head_topk, or a dflash_config top-k fallback"
        )
    values = set(found.values())
    if len(values) != 1:
        raise ValueError(f"DSPINE refinement widths disagree: {found}")
    k = values.pop()
    if k & (k - 1) or not 2 <= k <= 64:
        raise ValueError(f"DSPINE head_topk must be a power of two in [2, 64], got {k}")
    return k


def build_serial_tables(
    lm_head_w: torch.Tensor,
    norm_w: torch.Tensor,
    write_w: torch.Tensor,
    pv_w: torch.Tensor,
    codebook: torch.Tensor,
    chunk: int = 8192,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(TG [V, R], TSV [V, 2R]) serial-refinement vocabulary tables.

    lm_head_w [V, H], norm_w [H], write_w [H, R] (the last layer's write),
    pv_w [2R, C] = [G_p; A], codebook [V, C]:
      TG[v]  = cast(lm_head_w[v] * norm_w) @ write_w
      TSV[v] = [G_p; A] @ cast(sqrt(C) * codebook[v])
    Tables keep lm_head_w's dtype: bf16 on GPU, fp32 for CPU reference work.
    """
    v, h = lm_head_w.shape
    r = write_w.shape[1]
    c = codebook.shape[1]
    if (
        codebook.shape[0] != v
        or tuple(write_w.shape) != (h, r)
        or tuple(pv_w.shape) != (2 * r, c)
        or norm_w.shape != (h,)
    ):
        raise ValueError("build_serial_tables: shape contract violated")
    table_dtype = torch.bfloat16 if lm_head_w.dtype == torch.bfloat16 else torch.float32
    tg = torch.empty((v, r), device=lm_head_w.device, dtype=table_dtype)
    tsv = torch.empty((v, 2 * r), device=lm_head_w.device, dtype=table_dtype)
    nwf = norm_w.float()
    wb = write_w.to(table_dtype)
    pvb = pv_w.to(table_dtype)
    scale = float(c) ** 0.5
    for s in range(0, v, chunk):
        e = min(s + chunk, v)
        tg[s:e] = torch.matmul((lm_head_w[s:e].float() * nwf).to(table_dtype), wb)
        tsv[s:e] = F.linear((scale * codebook[s:e].float()).to(table_dtype), pvb)
    return tg, tsv


# ---------------------------------------------------------------------------
# Code bridge
# ---------------------------------------------------------------------------


class DSpineCodeBridge(nn.Module):
    """Post-layer writes with DFlash anchor-row semantics.

    ``dflash_config.selected_prefix_code_bridge`` specifies ``code_size``,
    ``rank``, and ``eps``. ``selected_prefix_feature_layers`` is 1-based and
    includes the last layer. Each write fuses the next layer's input RMSNorm
    (or the final norm). ``dflash_config.dspine_bridge_impl`` selects the
    ``fused`` Triton path (CUDA + BF16) or the ``eager`` torch path.
    """

    def __init__(self, config, params_dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__()
        cfg = getattr(config, "dflash_config", None) or {}
        if cfg.get("selected_prefix_terminal", True):
            raise ValueError("DSPINE supports terminal-disabled code bridges only")
        if cfg.get("selected_prefix_feature_write_mode") != "independent":
            raise ValueError("DSPINE requires independent learned writes")
        if cfg.get("selected_prefix_feature_writes") is not True:
            raise ValueError("DSPINE requires selected_prefix_feature_writes = true")
        if not cfg.get("draft_block_causal"):
            raise ValueError(
                "DSPINE requires causal draft attention (draft_block_causal)"
            )
        if cfg.get("selected_prefix_feature_reliability") is not None:
            raise ValueError("Reliability probes are not part of the DSPINE path")
        if cfg.get("selected_prefix_feature_next_token_rows", False):
            raise ValueError(
                "DSPINE's serial refinement is defined for anchor-row "
                "(DFlash) semantics; next-token rows are a DSpark layout"
            )
        bridge = cfg.get("selected_prefix_code_bridge")
        if not bridge:
            raise ValueError(
                "DSPINE draft config lacks dflash_config.selected_prefix_code_bridge"
            )
        self.hidden_size = int(config.hidden_size)
        self.code_size = int(bridge["code_size"])
        self.rank = int(bridge["rank"])
        self.num_layers = int(config.num_hidden_layers)
        self.block_size = int(config.block_size)
        self.eps = float(bridge["eps"])
        feature_layers = cfg.get("selected_prefix_feature_layers")
        if feature_layers is None:
            raise ValueError(
                "DSPINE requires dflash_config.selected_prefix_feature_layers"
            )
        self.active_layers = frozenset(int(i) - 1 for i in feature_layers)
        if not self.active_layers <= set(range(self.num_layers)):
            raise ValueError(
                "selected_prefix_feature_layers out of range: "
                f"{sorted(self.active_layers)}"
            )
        if self.num_layers - 1 not in self.active_layers:
            raise ValueError(
                "DSPINE refines the last layer's write, so the last layer "
                "must be active"
            )
        implementation = cfg.get("dspine_bridge_impl", "eager")
        if implementation not in ("fused", "eager"):
            raise ValueError(f"Unknown DSPINE bridge implementation {implementation!r}")
        self.implementation = implementation
        self.prev = nn.Linear(self.code_size, self.rank, bias=False, dtype=params_dtype)
        self.gate = nn.Linear(self.code_size, self.rank, bias=True, dtype=params_dtype)
        self.predecessor_gate = nn.Linear(
            self.code_size, self.rank, bias=False, dtype=params_dtype
        )
        self.out = nn.Linear(self.rank, self.code_size, bias=False, dtype=params_dtype)
        self.register_buffer(
            "codebook",
            torch.empty(config.vocab_size, self.code_size, dtype=params_dtype),
        )
        self.register_buffer(
            "read_weight",
            torch.empty(self.code_size, self.hidden_size, dtype=params_dtype),
        )
        self.register_buffer(
            "early_read_weights",
            torch.empty(
                self.num_layers - 1,
                self.code_size,
                self.hidden_size,
                dtype=params_dtype,
            ),
        )
        self.feature_write_weights = nn.Parameter(
            torch.empty(
                self.num_layers,
                self.hidden_size,
                self.rank,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        for name in ("independent_write_version", "terminal_disabled_version"):
            self.register_buffer(name, torch.zeros((), dtype=torch.int64))
        self.register_buffer("packed_weight", None, persistent=False)
        self._loaded = set()
        self._ready = False

    def begin_load(self) -> None:
        self._loaded.clear()
        self._ready = False

    def load_weight(self, name: str, tensor: torch.Tensor) -> None:
        destinations = dict(self.named_parameters()) | dict(self.named_buffers())
        if name not in destinations or name == "packed_weight":
            raise ValueError(f"Unexpected DSPINE code-bridge weight: {name}")
        dest = destinations[name]
        if tuple(dest.shape) != tuple(tensor.shape):
            raise ValueError(
                f"DSPINE code-bridge shape mismatch: {name}: "
                f"{tuple(tensor.shape)} vs {tuple(dest.shape)}"
            )
        dest.data.copy_(tensor)
        self._loaded.add(name)

    def prepare(self) -> None:
        expected = (
            set(dict(self.named_parameters())) | set(dict(self.named_buffers()))
        ) - {"packed_weight"}
        missing = expected - self._loaded
        if missing:
            raise ValueError(f"Missing DSPINE code-bridge tensors: {sorted(missing)}")
        if self.out.weight.count_nonzero().item():
            raise ValueError(
                "Terminal-disabled checkpoint contains a nonzero terminal projection"
            )
        for name in ("independent_write_version", "terminal_disabled_version"):
            if getattr(self, name).item() != 1:
                raise ValueError(f"Unsupported DSPINE {name}")
        packed = torch.cat(
            (self.gate.weight, self.predecessor_gate.weight, self.prev.weight),
            dim=0,
        ).contiguous()
        # Preserve graph-captured storage during same-shape weight reloads.
        if self.packed_weight is None:
            self.packed_weight = packed
        else:
            self.packed_weight.copy_(packed)
        self._ready = True

    def layer_read_weight(self, layer: int) -> torch.Tensor:
        return (
            self.read_weight
            if layer == self.num_layers - 1
            else self.early_read_weights[layer]
        )

    @torch.no_grad()
    def forward(
        self,
        hidden: torch.Tensor,
        residual: torch.Tensor | None,
        input_ids: torch.Tensor,
        layer: int,
        next_norm: nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """One write after ``layer``.

        Returns (normed, out, raw, rms, packed): ``normed`` [m, H] is the
        next layer's pre-normed input, ``out`` [m, H] the updated residual
        stream, ``raw`` [m, H] the pre-write residual stream, ``rms`` [m] the
        fp32 per-row RMS, and ``packed`` [m, 3R] the code projections
        [Q | S | V].
        """
        if not self._ready:
            raise RuntimeError("DSPINE code-bridge weights were not prepared")
        if hidden.shape[0] % self.block_size:
            raise ValueError(
                "DSPINE code-bridge input must contain whole request blocks"
            )
        if (
            self.implementation == "fused"
            and hidden.is_cuda
            and hidden.dtype == torch.bfloat16
        ):
            return self._forward_fused(hidden, residual, input_ids, layer, next_norm)
        return self._forward_eager(hidden, residual, input_ids, layer, next_norm)

    def _read_codes(
        self,
        hidden: torch.Tensor,
        residual: torch.Tensor | None,
        input_ids: torch.Tensor,
        layer: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fused read: (raw [m,H], rms [m] fp32, codes [m,C])."""
        m, h = hidden.shape
        c, t = self.code_size, self.block_size
        raw = torch.empty_like(hidden)
        normalized = torch.empty_like(hidden)
        rms = torch.empty(m, device=hidden.device, dtype=torch.float32)
        _prepare_read_kernel[(m,)](
            hidden,
            hidden if residual is None else residual,
            raw,
            normalized,
            rms,
            h,
            self.eps,
            residual is not None,
            triton.next_power_of_2(h),
            enable_fp_fusion=False,
        )
        projected = F.linear(normalized, self.layer_read_weight(layer))
        codes = torch.empty_like(projected)
        anchors = input_ids.reshape(-1, t)[:, 0].contiguous()
        _normalize_codes_kernel[(m,)](
            projected,
            self.codebook,
            anchors,
            codes,
            c,
            t,
            True,
            self.eps,
            triton.next_power_of_2(c),
            enable_fp_fusion=False,
        )
        return raw, rms, codes

    def _forward_fused(
        self,
        hidden: torch.Tensor,
        residual: torch.Tensor | None,
        input_ids: torch.Tensor,
        layer: int,
        next_norm: nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        m = hidden.shape[0]
        r, t, h = self.rank, self.block_size, self.hidden_size
        raw, rms, codes = self._read_codes(hidden, residual, input_ids, layer)
        packed = F.linear(codes, self.packed_weight)
        message = torch.empty((m, r), device=hidden.device, dtype=hidden.dtype)
        _feature_message_kernel[(m,)](
            packed,
            self.gate.bias,
            message,
            r,
            t,
            triton.next_power_of_2(r),
            enable_fp_fusion=False,
        )
        delta = F.linear(message, self.feature_write_weights[layer])
        valid = torch.ones(m, device=raw.device, dtype=torch.bool)
        output = torch.empty_like(raw)
        normed = torch.empty_like(raw)
        _feature_finish_kernel[(m,)](
            raw,
            delta,
            rms,
            valid,
            next_norm.weight,
            output,
            normed,
            h,
            t,
            True,
            True,
            next_norm.variance_epsilon,
            triton.next_power_of_2(h),
            enable_fp_fusion=False,
        )
        return normed, output, raw, rms, packed

    def _forward_eager(
        self,
        hidden: torch.Tensor,
        residual: torch.Tensor | None,
        input_ids: torch.Tensor,
        layer: int,
        next_norm: nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Plain-torch path with the same rounding points as the fused kernels.

        Runs in hidden.dtype: BF16 on GPU, fp32 on CPU for reference checks.
        """
        t, r = self.block_size, self.rank
        raw = hidden if residual is None else hidden + residual
        f = raw.float()
        rms = f.square().mean(-1).clamp_min(torch.finfo(torch.float32).tiny).sqrt()
        normalized = (f / (rms[:, None] + self.eps)).to(raw.dtype)
        projected = F.linear(
            normalized,
            self.layer_read_weight(layer).to(normalized.dtype),
        ).float()
        codes = (
            math.sqrt(self.code_size)
            * projected
            / (projected.norm(dim=-1, keepdim=True) + self.eps)
        )
        anchors = input_ids.reshape(-1, t)[:, 0]
        anchor_codes = (
            math.sqrt(self.code_size) * F.embedding(anchors, self.codebook).float()
        )
        codes = codes.view(-1, t, self.code_size)
        codes = torch.cat((anchor_codes[:, None, :], codes[:, 1:]), dim=1).view(
            -1, self.code_size
        )
        packed = F.linear(codes.to(raw.dtype), self.packed_weight.to(raw.dtype))
        q, s, v = packed.float().view(-1, t, 3 * r).split(r, dim=-1)
        gate = torch.sigmoid(q[:, 1:] + s[:, :-1] + self.gate.bias.float())
        message = torch.cat((torch.zeros_like(q[:, :1]), gate * v[:, :-1]), dim=1).to(
            raw.dtype
        )
        delta = F.linear(
            message.view(-1, r),
            self.feature_write_weights[layer].to(raw.dtype),
        ).float()
        not_anchor = (torch.arange(raw.shape[0], device=raw.device) % t != 0).float()[
            :, None
        ]
        out = (f + (rms[:, None] * delta) * not_anchor).to(raw.dtype)
        x = out.float()
        y = (
            x
            * torch.rsqrt(
                x.square().mean(-1, keepdim=True) + next_norm.variance_epsilon
            )
        ).to(raw.dtype)
        normed = (y.float() * next_norm.weight.float()).to(raw.dtype)
        return normed, out, raw, rms, packed


# ---------------------------------------------------------------------------
# Draft model
# ---------------------------------------------------------------------------


class DSpineQwen3DecoderLayer(DFlashQwen3DecoderLayer):
    """DFlash layer whose input RMSNorm may already be fused into a code write."""

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        input_normed: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if input_normed:
            if residual is None:
                raise RuntimeError("A pre-normed DSPINE layer input needs its residual")
        elif residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class DSpineQwen3Model(DFlashQwen3Model):
    """DFlash draft with a code bridge between every active layer pair."""

    decoder_layer_cls = DSpineQwen3DecoderLayer

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        start_layer_id: int = 0,
        prefix: str = "",
    ) -> None:
        speculative_config = vllm_config.speculative_config
        assert speculative_config is not None
        config = speculative_config.draft_model_config.hf_config
        dflash_config = getattr(config, "dflash_config", None) or {}
        # Layers read config.is_causal during construction; infer it from
        # draft_block_causal when the checkpoint leaves it unset. An explicit
        # False is preserved so the causal check below rejects it.
        if (
            dflash_config.get("draft_block_causal")
            and getattr(config, "is_causal", None) is None
        ):
            config.is_causal = True
        super().__init__(
            vllm_config=vllm_config,
            start_layer_id=start_layer_id,
            prefix=prefix,
        )
        block_size = int(self.config.block_size)
        if 1 + speculative_config.num_speculative_tokens != block_size:
            raise ValueError(
                "DSPINE needs 1 + num_speculative_tokens == draft block "
                f"size ({block_size}), got "
                f"{1 + speculative_config.num_speculative_tokens}"
            )
        for layer in self.layers:
            if layer.self_attn.causal is not True:
                raise ValueError(
                    "DSPINE requires causal draft attention in every layer"
                )
            if layer.self_attn.sliding_window is not None:
                raise ValueError("DSPINE drafts use full attention layers")
        self.selected_prefix_bridge = DSpineCodeBridge(
            self.config, params_dtype=vllm_config.model_config.dtype
        )
        self.head_topk = resolve_dspine_head_topk(self.config, len(self.layers))

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        input_embeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the draft layers and active bridge writes.

        Returns (hidden_states, last_raw, last_rms, last_query): the
        final-normed block states plus the last write's pre-write ``raw``
        [bs*T, H], ``rms`` [bs*T] fp32 and receiver query
        ``packed[:, :R]`` [bs*T, R] for the serial refinement tail.
        """
        if input_embeds is None:
            input_embeds = self.embed_input_ids(input_ids)
        bridge = self.selected_prefix_bridge
        last = len(self.layers) - 1
        hidden_states = input_embeds
        residual: torch.Tensor | None = None
        input_normed = False
        last_raw = last_rms = last_packed = None
        for index, layer in enumerate(self.layers):
            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
                input_normed=input_normed,
            )
            input_normed = False
            if index in bridge.active_layers and hidden_states.numel():
                next_norm = (
                    self.layers[index + 1].input_layernorm
                    if index < last
                    else self.norm
                )
                normed, out, raw, rms, packed = bridge(
                    hidden_states, residual, input_ids, index, next_norm
                )
                hidden_states, residual = normed, out
                input_normed = True
                if index == last:
                    last_raw, last_rms, last_packed = raw, rms, packed
        if hidden_states.numel() and not input_normed:
            raise RuntimeError(
                "DSPINE's last layer must end with a (norm-fused) code write"
            )
        last_query = last_packed[:, : bridge.rank] if last_packed is not None else None
        return hidden_states, last_raw, last_rms, last_query


# ---------------------------------------------------------------------------
# Draft LM wrapper
# ---------------------------------------------------------------------------


class DSpineQwen3ForCausalLM(DFlashQwen3ForCausalLM):
    # The speculator this is a draft model of (see SpeculativeDraftModel).
    speculator_config_class = DSpineSpeculatorConfig

    model_cls = DSpineQwen3Model

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.head_topk = self.model.head_topk
        self.candidate_logits_processor = LogitsProcessor(
            vllm_config.model_config.get_vocab_size()
        )

    def compute_candidates(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Top-K draft candidates with ids sorted ascending by token id."""
        candidate_ids, unary_logits = self.candidate_logits_processor.get_top_k_tokens(
            self.lm_head, hidden_states, self.head_topk
        )
        candidate_ids, order = candidate_ids.sort(dim=-1)
        unary_logits = unary_logits.gather(-1, order)
        return candidate_ids, unary_logits

    @torch.no_grad()
    def materialize_inference_buffers(self, lm_head_weight: torch.Tensor) -> None:
        """Build serial tables and fp32 refinement weights after loading."""
        bridge = self.model.selected_prefix_bridge
        last = len(self.model.layers) - 1
        expected = (int(bridge.codebook.shape[0]), int(self.config.hidden_size))
        if tuple(lm_head_weight.shape) != expected:
            raise ValueError(
                f"DSPINE lm_head_weight {tuple(lm_head_weight.shape)} != {expected}"
            )
        r = int(bridge.rank)
        self.tg, self.tsv = build_serial_tables(
            lm_head_weight,
            self.model.norm.weight.detach(),
            bridge.feature_write_weights[last].detach(),
            bridge.packed_weight[r:].detach(),
            bridge.codebook,
        )
        f32 = torch.float32
        self.pred_gate_f32 = (
            bridge.predecessor_gate.weight.detach().to(f32).contiguous()
        )
        self.prev_f32 = bridge.prev.weight.detach().to(f32).contiguous()
        self.gate_b_f32 = bridge.gate.bias.detach().to(f32).contiguous()
        self.write_f32 = (
            bridge.feature_write_weights[last].detach().to(f32).contiguous()
        )
        self.norm_f32 = self.model.norm.weight.detach().to(f32).contiguous()
        self.norm_eps = float(self.model.norm.variance_epsilon)
        self.lm_head_weight = lm_head_weight
        self.write_gram_f32 = self.write_f32.T @ self.write_f32

    @torch.no_grad()
    def score_transitions(
        self,
        candidate_ids: torch.Tensor,
        raw: torch.Tensor,
        rms: torch.Tensor,
        query: torch.Tensor,
        anchor_token_ids: torch.Tensor,
        *,
        normalize: bool = True,
    ) -> torch.Tensor:
        """fp32 [B, T-1, K, K] transition logits of the redone last write.

        For sender s (the anchor on the first proposal row, the previous
        row's candidate lane otherwise), e = sqrt(C) * codebook[s] and the
        refined state is
        x = raw + rms * write_w @ (sigmoid(query + pred_gate_w @ e + gate_b)
                                   * (prev_w @ e)).
        Row [b, i, j, c] is the lm_head logit of candidate c under sender
        lane j. With ``normalize`` the final RMSNorm denominator
        1/sqrt(mean(x**2) + eps) is applied (matching ref_refine_table /
        serial_rs_table); without it the scores match the serial_table
        greedy table, whose positive scale cannot change the argmax.
        """
        bridge = self.model.selected_prefix_bridge
        b, s, k = candidate_ids.shape
        senders = torch.cat(
            (
                anchor_token_ids[:, None, None].expand(b, 1, k),
                candidate_ids[:, :-1],
            ),
            dim=1,
        )
        sqrt_c = math.sqrt(bridge.codebook.shape[1])
        e = bridge.codebook[senders].float() * sqrt_c  # [B, S, K, C]
        gate = torch.sigmoid(
            query.float()[:, :, None, :] + e @ self.pred_gate_f32.T + self.gate_b_f32
        )
        msg = gate * (e @ self.prev_f32.T)  # [B, S, K, R]
        x = raw.float()
        scale = rms.float()
        # R-space algebra: keep the [B, S, K, H] refined states implicit.
        lm_c = self.lm_head_weight[candidate_ids].float()  # [B, S, K, H]
        nw = self.norm_f32
        base = torch.einsum("bsh,bsch->bsc", x * nw, lm_c)
        tg_c = torch.matmul(lm_c * nw, self.write_f32)  # [B, S, K, R]
        trans = torch.einsum("bsjr,bscr->bsjc", msg, tg_c)
        logits = base[:, :, None, :] + scale[:, :, None, None] * trans
        if normalize:
            wtx = torch.matmul(x, self.write_f32)  # [B, S, R]
            sq = (
                x.square().sum(-1)[:, :, None]
                + 2 * scale[:, :, None] * torch.einsum("bsr,bskr->bsk", wtx, msg)
                + scale[:, :, None].square()
                * torch.einsum(
                    "bskr,bskr->bsk",
                    msg,
                    torch.matmul(msg, self.write_gram_f32),
                )
            )
            denom = torch.sqrt(sq / x.shape[-1] + self.norm_eps)
            logits = logits / denom[:, :, :, None]
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        """Load DFlash weights plus ``selected_prefix_bridge.*`` tensors.

        The optional ``draft_block_causal_version`` and
        ``last_write_refine_version`` tensors must be version 1. Every
        non-bridge floating-point parameter must come from the checkpoint:
        they are poisoned with NaN first and demanded to be overwritten,
        except embed_tokens / lm_head / mask_embedding / draft_id_to_target_id
        when the checkpoint omits them.
        """
        weights = list(weights)
        bridge = self.model.selected_prefix_bridge
        bridge.begin_load()
        float_params = [
            (name, p)
            for name, p in self.named_parameters()
            if not name.startswith("model." + BRIDGE_PREFIX)
            and p.dtype.is_floating_point
        ]
        with torch.no_grad():
            for _, p in float_params:
                p.fill_(float("nan"))

        def dflash_weights():
            for name, tensor in weights:
                short = name.removeprefix("model.")
                if short.startswith(BRIDGE_PREFIX):
                    bridge.load_weight(short[len(BRIDGE_PREFIX) :], tensor)
                    continue
                if short in (
                    "draft_block_causal_version",
                    "last_write_refine_version",
                ):
                    if int(tensor.item()) != 1:
                        raise ValueError(
                            f"Unsupported DSPINE checkpoint version: {short}"
                        )
                    continue
                yield name, tensor

        super().load_weights(dflash_weights())
        checkpoint_names = [name for name, _ in weights]
        optional = (
            "embed_tokens",
            "lm_head",
            "mask_embedding",
            "draft_id_to_target_id",
        )
        unloaded = []
        for name, p in float_params:
            if not torch.isnan(p).any().item():
                continue
            tokens = [tok for tok in optional if tok in name]
            if tokens and not any(
                tok in ckpt for tok in tokens for ckpt in checkpoint_names
            ):
                with torch.no_grad():
                    p.zero_()
                continue
            unloaded.append(name)
        if unloaded:
            raise ValueError(f"DSPINE checkpoint left parameters unloaded: {unloaded}")
        bridge.prepare()
        logger.info(
            "Loaded DSPINE draft: layers=%d bridge=%s active_layers=%s "
            "code_size=%d rank=%d head_topk=%d",
            len(self.model.layers),
            bridge.implementation,
            sorted(i + 1 for i in bridge.active_layers),
            bridge.code_size,
            bridge.rank,
            self.head_topk,
        )


EntryClass = DSpineQwen3ForCausalLM

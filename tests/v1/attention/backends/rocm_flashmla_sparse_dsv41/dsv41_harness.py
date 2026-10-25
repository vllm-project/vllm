# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Driver for the ROCM_FLASHMLA_SPARSE_DSV41 suite.

DeepSeek V4.1's sparse attention is a model-owned layer
(``DeepseekV41ROCMAiterMLAAttention``), not a generic ``AttentionImpl``: its
``forward_mqa`` reads the sliding-window (SWA) cache, the compressed cache of
its KV source and the top-k indices its index source published. Those three
only exist together inside a running model, so the dispatch and numerics
suites run a five-layer stand-in of DeepSeek-V4.1-Flash through the real
engine and check every ``forward_mqa`` call there, from a worker extension
(``Dsv41AttnProbe``):

- which kernels the call ran (spied on the module attributes ``rocm.py``
  resolves at call time), and
- its output against a float32 reference computed from the logical K/V: the
  cache records decoded here in PyTorch (not by vLLM's gather kernel), the
  SWA window and top-k selection rebuilt from positions and block tables,
  the attention sink, and the inverse GPT-J RoPE.

The five layers keep one of each attention kind of the real model
(``compress_ratios`` 0/2/2/1/1 like layers 0, 2, 3, 20 and 24 of the
checkpoint): a sliding-window-only layer, a ratio-2 KV and index source, a
ratio-2 consumer, a ratio-1 KV, index and candidate source, and a ratio-1
re-indexer over those candidates.
"""

from __future__ import annotations

import contextlib
import functools
import importlib
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import torch

# Registers the checkpoint-range random weight loader in every process that
# imports this module (the worker extension does).
from tests.models.test_dsv41_mono_layers import LOAD_FORMAT  # noqa: F401

MODEL = "deepseek-ai/DeepSeek-V4.1-Flash"
ROCM_MODULE = "vllm.models.deepseek_v41.amd.rocm"
AITER_ENV = "VLLM_ROCM_USE_AITER_TRITON_SPARSE_MLA"

# Layer id -> attention kind of the five-layer stand-in.
LAYER_KINDS = {
    0: "swa",
    1: "c2_source",
    2: "c2_consumer",
    3: "c1_source",
    4: "c1_reindex",
}
COMPRESS_RATIO = {0: 0, 1: 2, 2: 2, 3: 1, 4: 1}

TINY = {
    "num_hidden_layers": 5,
    # One entry a layer, the three MTP layers' included.
    "compress_ratios": [0, 2, 2, 1, 1, 0, 0, 0],
    "kv_source_layer_ids": [1, 3],
    "index_source_layer_ids": [1, 3, 4],
    "candidate_source_layer_id": 3,
    "engram_layer_ids": [],
    "dspark_target_layer_ids": [2, 3, 4],
    # The suite tests attention: 16 experts keep the stand-in small.
    "n_routed_experts": 16,
}

# DeepSeek-V4.1 attention geometry (config.json): 64 heads of 512 = 448 NoPE
# + 64 RoPE, one latent KV head, a 128-token window, top-k 512.
HEAD_DIM, ROPE_DIM, NOPE_DIM = 512, 64, 448
WINDOW = 128
INDEX_TOPK = 512


@dataclass(frozen=True)
class Dsv41Heads:
    """Query heads one rank holds. The latent cache has one head at any TP,
    so a TP layout only changes the local head and output-group counts."""

    tp: int
    num_heads: int
    o_groups: int

    @property
    def id(self) -> str:
        return f"tp{self.tp}-h{self.num_heads}"


# Per-rank heads of the deployed layouts (tests/v1/attention/model_shapes.py
# PARALLEL_LAYOUTS: TP1/DP 64, TP4 16, TP8 8; o_groups 8 -> 1 group a rank).
DSV41_HEADS = [Dsv41Heads(1, 64, 8), Dsv41Heads(4, 16, 2), Dsv41Heads(8, 8, 1)]


def hf_overrides(heads: Dsv41Heads = DSV41_HEADS[0], **extra) -> dict:
    """A TP1 stand-in with ``heads``' per-rank query heads."""
    return {
        **TINY,
        "num_attention_heads": heads.num_heads,
        "o_groups": heads.o_groups,
        **extra,
    }


def _apply_overrides(hf_config, overrides: dict):
    text = hf_config.get_text_config()
    for key, value in overrides.items():
        setattr(text, key, value)
    return hf_config


def callable_overrides(overrides: dict):
    """``overrides`` as a callable ``hf_overrides``: unlike a dict, a callable
    is composed onto the draft model's config too (``SpeculativeConfig``), so
    a DSpark draft taps the stand-in's layers, not the checkpoint's 37-39."""
    return functools.partial(_apply_overrides, overrides=overrides)


def engine_kwargs(**overrides) -> dict:
    """``LLM`` arguments for the stand-in on CDNA4: the MXFP4 indexer cache
    with 128-token blocks, as the checkpoint is served on MI350/MI355."""
    kwargs: dict[str, Any] = dict(
        load_format=LOAD_FORMAT,
        # Importing the probe's module registers LOAD_FORMAT in the workers.
        worker_extension_cls=f"{__name__}.Dsv41AttnProbe",
        hf_overrides=hf_overrides(),
        language_model_only=True,
        tokenizer_mode="deepseek_v41",
        attention_config={
            "indexer_kv_dtype": "mxfp4",
            "indexer_sparse_logits": True,
        },
        block_size=128,
        moe_backend="aiter",
        enforce_eager=True,
        # vllm_runner turns chunked prefill off unless told otherwise.
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        max_model_len=8192,
        max_num_seqs=16,
        max_num_batched_tokens=8192,
        gpu_memory_utilization=0.3,
        seed=0,
    )
    kwargs.update(overrides)
    return kwargs


def random_prompts(lens: list[int], seed: int = 0) -> list[list[int]]:
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(1000, 60000, (n,), generator=g).tolist() for n in lens]


# --------------------------------------------------------------------------
# Cache records, decoded in PyTorch.
# --------------------------------------------------------------------------

# V4 record (every arch but SM100): per token 448 fp8 e4m3 NoPE + 64 bf16
# RoPE (576 B) in the block's data region, then 8 scale bytes in the block's
# scale region (7 UE8M0 scales of 64 dims and a pad byte).
_V4_DATA, _V4_SCALES = 576, 8


def _blocks(cache: torch.Tensor) -> torch.Tensor:
    """[num_blocks, bytes] view of one layer's paged uint8 cache."""
    assert cache.dtype == torch.uint8
    return cache.reshape(cache.shape[0], -1)


def decode_v4_records(
    cache: torch.Tensor, slots: torch.Tensor, entries_per_block: int, fnuz: bool
) -> torch.Tensor:
    """Float32 [len(slots), 512] latent rows of a V4-record cache."""
    blocks = _blocks(cache)
    block, off = slots // entries_per_block, slots % entries_per_block
    rows = blocks[block]
    data_idx = (off * _V4_DATA)[:, None] + torch.arange(_V4_DATA, device=slots.device)
    data = torch.gather(rows, 1, data_idx)
    scale_base = entries_per_block * _V4_DATA
    scale_idx = (scale_base + off * _V4_SCALES)[:, None] + torch.arange(
        7, device=slots.device
    )
    exponents = torch.gather(rows, 1, scale_idx).float() - 127.0
    fp8 = torch.float8_e4m3fnuz if fnuz else torch.float8_e4m3fn
    nope = data[:, :NOPE_DIM].contiguous().view(fp8).float()
    nope = nope.view(-1, 7, 64) * torch.exp2(exponents)[:, :, None]
    rope = data[:, NOPE_DIM:].contiguous().view(torch.bfloat16).float()
    return torch.cat([nope.reshape(-1, NOPE_DIM), rope], dim=-1)


def dequant_mxfp8_rows(data: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Float32 rows of an MXFP8 activation: e4m3 values, one UE8M0 per 32."""
    values = data.float().view(data.shape[0], -1, 32)
    exponents = scale.view(torch.uint8)[:, : values.shape[1]].float() - 127.0
    return (values * torch.exp2(exponents)[:, :, None]).view(data.shape[0], -1)


def rope_gptj(
    x: torch.Tensor, positions: torch.Tensor, cos_sin_cache: torch.Tensor
) -> torch.Tensor:
    """Forward GPT-J RoPE of the last ``ROPE_DIM`` dims of [T, ..., 512]."""
    cs = cos_sin_cache[positions.long()].float()
    shape = (cs.shape[0],) + (1,) * (x.dim() - 2) + (ROPE_DIM // 2,)
    cos = cs[:, : ROPE_DIM // 2].reshape(shape)
    sin = cs[:, ROPE_DIM // 2 :].reshape(shape)
    out = x.float().clone()
    even, odd = out[..., NOPE_DIM::2].clone(), out[..., NOPE_DIM + 1 :: 2].clone()
    out[..., NOPE_DIM::2] = even * cos - odd * sin
    out[..., NOPE_DIM + 1 :: 2] = even * sin + odd * cos
    return out


def inverse_rope_gptj(
    x: torch.Tensor, positions: torch.Tensor, cos_sin_cache: torch.Tensor
) -> torch.Tensor:
    """Rotate the last ``ROPE_DIM`` dims of [T, H, 512] by -theta(position),
    GPT-J style (adjacent pairs); cos_sin_cache rows are cos | sin."""
    cs = cos_sin_cache[positions.long()].float()
    cos, sin = cs[:, None, : ROPE_DIM // 2], cs[:, None, ROPE_DIM // 2 :]
    out = x.clone()
    even, odd = x[..., NOPE_DIM::2], x[..., NOPE_DIM + 1 :: 2]
    out[..., NOPE_DIM::2] = even * cos + odd * sin
    out[..., NOPE_DIM + 1 :: 2] = odd * cos - even * sin
    return out


# --------------------------------------------------------------------------
# Reference sparse attention.
# --------------------------------------------------------------------------


@dataclass
class LayerInputs:
    """What one ``forward_mqa`` call read, captured before it ran."""

    layer_id: int
    compress_ratio: int
    q: torch.Tensor  # [T, H, 512] bf16 (post norm and RoPE)
    positions: torch.Tensor  # [T]
    swa_cache: torch.Tensor
    swa_block_table: torch.Tensor
    swa_block_size: int
    token_to_req: torch.Tensor
    is_valid: torch.Tensor
    replay_start: torch.Tensor | None
    compressed_cache: torch.Tensor | None
    compressed_block_table: torch.Tensor | None
    compressed_block_size: int
    topk: torch.Tensor | None  # [T, top_k] local compressed indices
    attn_sink: torch.Tensor
    scale: float
    cos_sin_cache: torch.Tensor
    fnuz: bool


def reference_attention(
    inp: LayerInputs,
    *,
    window: int = WINDOW,
    drop_compressed: bool = False,
    row_chunk: int = 256,
) -> tuple[torch.Tensor, dict[str, int]]:
    """Float32 [T, H, 512] output, inverse-RoPE'd, and selection statistics.

    Each valid token at position p attends to the SWA rows at positions
    max(p - window + 1, replay_start) .. p and to the compressed rows its
    top-k names, plus one sink column per head. The latent is both K and V.
    ``drop_compressed`` is the negative control: a reference that forgets
    the compressed rows must not match a layer that has them.
    """
    q = inp.q.float()
    num_tokens, num_heads, _ = q.shape
    out = torch.zeros_like(q)
    stats = {"rows": 0, "topk_selected": 0, "topk_rows_full": 0, "noncausal": 0}
    device = q.device
    cmp_entries = (
        inp.compressed_block_size // inp.compress_ratio if inp.compress_ratio else 0
    )
    sink = inp.attn_sink[:num_heads].float()
    for start in range(0, num_tokens, row_chunk):
        rows = torch.arange(start, min(start + row_chunk, num_tokens), device=device)
        rows = rows[inp.is_valid[rows].bool()]
        if rows.numel() == 0:
            continue
        reqs = inp.token_to_req[rows].long()
        pos = inp.positions[rows].long()
        lo = (pos - window + 1).clamp(min=0)
        if inp.replay_start is not None:
            lo = torch.maximum(lo, inp.replay_start[reqs].long())
        offs = torch.arange(window, device=device)
        swa_pos = lo[:, None] + offs[None]
        swa_ok = swa_pos <= pos[:, None]
        swa_pos = torch.where(swa_ok, swa_pos, lo[:, None])
        bt = inp.swa_block_table[reqs].long()
        swa_slots = (
            torch.gather(bt, 1, swa_pos // inp.swa_block_size) * inp.swa_block_size
            + swa_pos % inp.swa_block_size
        )
        k_swa = decode_v4_records(
            inp.swa_cache, swa_slots.reshape(-1), inp.swa_block_size, inp.fnuz
        ).view(rows.numel(), window, HEAD_DIM)
        keys, valid = [k_swa], [swa_ok]
        if inp.compress_ratio and not drop_compressed:
            assert inp.topk is not None and inp.compressed_cache is not None
            assert inp.compressed_block_table is not None
            idx = inp.topk[rows].long()
            sel = idx >= 0
            num_compressed = (pos + 1) // inp.compress_ratio
            stats["noncausal"] += int((sel & (idx >= num_compressed[:, None])).sum())
            stats["topk_selected"] += int(sel.sum())
            stats["topk_rows_full"] += int((sel.sum(1) == idx.shape[1]).sum())
            idx = torch.where(sel, idx, 0)
            cbt = inp.compressed_block_table[reqs].long()
            cmp_slots = (
                torch.gather(cbt, 1, idx // cmp_entries) * cmp_entries
                + idx % cmp_entries
            )
            k_cmp = decode_v4_records(
                inp.compressed_cache, cmp_slots.reshape(-1), cmp_entries, False
            ).view(rows.numel(), idx.shape[1], HEAD_DIM)
            keys.append(k_cmp)
            valid.append(sel)
        k = torch.cat(keys, dim=1)
        ok = torch.cat(valid, dim=1)
        logits = torch.einsum("thd,tkd->thk", q[rows], k) * inp.scale
        logits = logits.masked_fill(~ok[:, None, :], float("-inf"))
        logits = torch.cat(
            [logits, sink[None, :, None].expand(rows.numel(), -1, 1)], dim=-1
        )
        probs = torch.softmax(logits, dim=-1)[..., :-1]
        out[rows] = torch.einsum("thk,tkd->thd", probs, k)
        stats["rows"] += rows.numel()
    out = inverse_rope_gptj(out, inp.positions, inp.cos_sin_cache)
    return out, stats


def rel_err(got: torch.Tensor, want: torch.Tensor) -> float:
    if want.numel() == 0:
        return 0.0
    return (
        (got.float() - want.float()).norm() / want.float().norm().clamp(min=1e-30)
    ).item()


# --------------------------------------------------------------------------
# Worker extension.
# --------------------------------------------------------------------------

# Kernels forward_mqa can reach, by the name rocm.py resolves at call time.
SPIED_KERNELS = (
    "rocm_sparse_attn_decode",
    "rocm_sparse_attn_prefill",
    "dequantize_and_gather_k_cache",
    "combine_topk_swa_indices",
    "compute_global_topk_ragged_indices_and_indptr",
    "build_prefill_topk_ragged_indices",
    "rocm_inverse_rope_rows_",
    "rocm_inverse_rope_mxfp8_rows",
)


class _Recorder:
    def __init__(self):
        self.active: list[str] | None = None

    def wrap(self, name, fn):
        def wrapper(*args, **kwargs):
            if self.active is not None:
                self.active.append(name)
            return fn(*args, **kwargs)

        return wrapper


def _phase(swa_md) -> str:
    if swa_md.num_prefills == 0:
        return "decode"
    if swa_md.num_decodes == 0:
        return "prefill"
    return "mixed"


class Dsv41AttnProbe:
    """Worker extension (``worker_extension_cls``) that checks every
    ``forward_mqa`` of the stand-in. Install with ``probe_install``, read
    the aggregate with ``probe_report``."""

    def _attn_layers(self) -> dict[int, Any]:
        rocm = importlib.import_module(ROCM_MODULE)
        model = self.model_runner.model  # type: ignore[attr-defined]
        layers = {
            m.layer_id: m
            for m in model.modules()
            if isinstance(m, rocm.DeepseekV41ROCMAiterMLAAttention)
        }
        # Draft models are separate modules; the target's ids are unique.
        assert len(layers) == len(
            [
                m
                for m in model.modules()
                if isinstance(m, rocm.DeepseekV41ROCMAiterMLAAttention)
            ]
        )
        return layers

    def probe_install(
        self,
        check: bool = True,
        bf16_rerun: bool = False,
        max_checked_calls: int = 64,
        kv_check: bool = False,
    ) -> dict:
        """Wrap the kernels and every layer's forward_mqa.

        ``check`` compares each call (up to ``max_checked_calls`` a layer)
        with the reference; ``bf16_rerun`` also runs the call again into a
        plain bf16 output, the buffer non-gfx950 GPUs and other callers use.
        Returns each layer's static configuration.
        """
        rocm = importlib.import_module(ROCM_MODULE)
        from vllm._aiter_ops import rocm_aiter_ops

        self._rec = _Recorder()
        self._calls: list[dict] = []
        self._checked: dict[int, int] = defaultdict(int)
        self._kv_records: list[dict] = []
        for name in SPIED_KERNELS:
            setattr(rocm, name, self._rec.wrap(name, getattr(rocm, name)))
        fwd = rocm_aiter_ops.triton_sparse_mla_fwd
        rocm_aiter_ops.triton_sparse_mla_fwd = self._rec.wrap(  # type: ignore[method-assign]
            "triton_sparse_mla_fwd", fwd
        )
        layers = self._attn_layers()
        config = {}
        for layer_id, layer in layers.items():
            layer.forward_mqa = self._wrap_forward_mqa(
                layer, layer.forward_mqa, check, bf16_rerun, max_checked_calls
            )
            if kv_check:
                layer._fused_qnorm_rope_kv_insert = self._wrap_kv_insert(
                    layer, layer._fused_qnorm_rope_kv_insert
                )
            config[layer_id] = {
                "compress_ratio": layer.compress_ratio,
                "use_aiter_sparse_mla": layer._use_aiter_sparse_mla,
                "has_indexer": layer.indexer is not None,
                "indexer_cls": type(layer.indexer).__name__
                if layer.indexer is not None
                else None,
                "n_local_heads": layer.n_local_heads,
                "window_size": layer.window_size,
                "kv_cache_dtype": layer.kv_cache_dtype,
            }
        return config

    def _capture(self, layer, q, positions) -> LayerInputs | None:
        from vllm.forward_context import get_forward_context
        from vllm.platforms import current_platform

        md = get_forward_context().attn_metadata
        if not isinstance(md, dict):
            return None
        swa: Any = md[layer.swa_cache_layer.prefix]
        cmp_md: Any = (
            md.get(layer.compressed_cache_prefix)
            if layer.compressed_cache_prefix is not None
            else None
        )
        num_tokens = swa.num_decode_tokens + swa.num_prefill_tokens
        topk = None
        if layer.compress_ratio:
            topk = layer.topk_indices_buffer[:num_tokens].clone()
        return LayerInputs(
            layer_id=layer.layer_id,
            compress_ratio=layer.compress_ratio,
            q=q[:num_tokens].clone(),
            positions=positions[:num_tokens].clone(),
            swa_cache=layer.swa_cache_layer.kv_cache,
            swa_block_table=swa.block_table,
            swa_block_size=swa.block_size,
            token_to_req=swa.token_to_req_indices[:num_tokens],
            is_valid=swa.is_valid_token[:num_tokens],
            replay_start=swa.replay_start,
            compressed_cache=(
                layer._compressed_kv_cache() if layer.compress_ratio else None
            ),
            compressed_block_table=None if cmp_md is None else cmp_md.block_table,
            compressed_block_size=0 if cmp_md is None else cmp_md.block_size,
            topk=topk,
            attn_sink=layer.attn_sink,
            scale=layer.scale,
            cos_sin_cache=layer.rotary_emb.cos_sin_cache,
            fnuz=current_platform.is_fp8_fnuz(),
        )

    def _wrap_kv_insert(self, layer, original):
        """D1/D4: the fused q-RoPE + KV RoPE/quant/insert against RoPE in
        PyTorch, with the written SWA rows decoded by ``decode_v4_records``."""
        from vllm.platforms import current_platform

        def insert(q, kv, positions, attn_metadata):
            if not isinstance(attn_metadata, dict):
                return original(q, kv, positions, attn_metadata)
            swa = attn_metadata[layer.swa_cache_layer.prefix]
            n = kv.shape[0]
            q_in, kv_in, pos = q.clone(), kv.clone(), positions[:n].clone()
            slots = swa.slot_mapping[:n].clone()
            q_out = original(q, kv, positions, attn_metadata)
            cos_sin = layer.rotary_emb.cos_sin_cache
            ok = slots >= 0
            want_kv = rope_gptj(kv_in[ok], pos[ok], cos_sin)
            got_kv = decode_v4_records(
                layer.swa_cache_layer.kv_cache,
                slots[ok].long(),
                swa.block_size,
                current_platform.is_fp8_fnuz(),
            )
            want_q = rope_gptj(q_in, pos, cos_sin)
            self._kv_records.append(
                {
                    "layer": layer.layer_id,
                    "rows": int(ok.sum()),
                    "nope": rel_err(got_kv[:, :NOPE_DIM], want_kv[:, :NOPE_DIM]),
                    "rope": rel_err(got_kv[:, NOPE_DIM:], want_kv[:, NOPE_DIM:]),
                    "q": rel_err(q_out[:n, : layer.n_local_heads], want_q),
                    "q_pad_zero": bool((q_out[:n, layer.n_local_heads :] == 0).all()),
                }
            )
            return q_out

        return insert

    def probe_kv_report(self) -> list[dict]:
        records, self._kv_records = self._kv_records, []
        return records

    def _wrap_forward_mqa(self, layer, original, check, bf16_rerun, max_checked):
        from vllm.forward_context import get_forward_context
        from vllm.model_executor.layers.fusion.quant_activation import (
            QuantizedActivation,
        )

        def forward_mqa(q, kv, positions, output):
            md = get_forward_context().attn_metadata
            record: dict[str, Any] = {"layer": layer.layer_id}
            if not isinstance(md, dict):
                record["phase"] = "profile"
            else:
                swa: Any = md[layer.swa_cache_layer.prefix]
                record.update(
                    phase=_phase(swa),
                    decode_tokens=swa.num_decode_tokens,
                    prefill_tokens=swa.num_prefill_tokens,
                    max_decode_query_len=swa.max_decode_query_len,
                    extend=bool(
                        swa.num_prefills
                        and swa.prefill_seq_lens_cpu is not None
                        and swa.prefill_query_lens_cpu is not None
                        and bool(
                            (
                                swa.prefill_seq_lens_cpu > swa.prefill_query_lens_cpu
                            ).any()
                        )
                    ),
                    max_seq_len=int(swa.seq_lens.max())
                    if swa.seq_lens is not None
                    else 0,
                )
            record["output"] = (
                "mxfp8" if isinstance(output, QuantizedActivation) else "bf16"
            )
            do_check = (
                check
                and isinstance(md, dict)
                and self._checked[layer.layer_id] < max_checked
            )
            inp = self._capture(layer, q, positions) if do_check else None
            kernels: list[str] = []
            self._rec.active = kernels
            try:
                original(q, kv, positions, output)
            finally:
                self._rec.active = None
            record["kernels"] = kernels
            if inp is not None:
                self._checked[layer.layer_id] += 1
                ref, stats = reference_attention(inp)
                record["stats"] = stats
                n = inp.q.shape[0]
                got = _layer_output(output, n, layer.n_local_heads)
                record.update(_errors(got, ref, inp, md, layer))
                neg, _ = reference_attention(inp, drop_compressed=True)
                if layer.compress_ratio and stats["topk_selected"]:
                    record["neg_err"] = rel_err(got, neg)
                record.update(_builder_checks(inp, md, layer))
                if bf16_rerun:
                    plain = torch.empty_like(q)
                    rerun: list[str] = []
                    self._rec.active = rerun
                    try:
                        original(q, kv, positions, plain)
                    finally:
                        self._rec.active = None
                    record["bf16_kernels"] = rerun
                    bf16 = plain[:n, : layer.n_local_heads].float()
                    record["bf16_err"] = rel_err(bf16, ref)
            self._calls.append(record)

        return forward_mqa

    def probe_profile_call(self, num_tokens: int = 16) -> dict:
        """Call every layer's ``forward_mqa`` the way the profiling run does
        (``attn_metadata is None``) on a poisoned output: it must zero the
        output, launch no kernel and grow no workspace."""
        from vllm.forward_context import set_forward_context
        from vllm.v1.worker.workspace import current_workspace_manager

        vllm_config = self.model_runner.vllm_config  # type: ignore[attr-defined]
        manager = current_workspace_manager()

        def workspace_bytes() -> int:
            return sum(
                manager._workspace_size_bytes(w) for w in manager._current_workspaces
            )

        out: dict[int, dict] = {}
        for layer_id, layer in self._attn_layers().items():
            device = layer.attn_sink.device
            q = torch.randn(
                num_tokens,
                layer.n_local_heads,
                HEAD_DIM,
                dtype=torch.bfloat16,
                device=device,
            )
            output = layer._alloc_attn_out(num_tokens, q)
            kind = type(output).__name__
            buffers = (
                [output.data.view(torch.uint8), output.scale.view(torch.uint8)]
                if kind == "QuantizedActivation"
                else [output.view(torch.uint8)]
            )
            for b in buffers:
                b.fill_(0x5A)
            positions = torch.arange(num_tokens, device=device)
            self._calls = []
            before = workspace_bytes()
            with set_forward_context(None, vllm_config):
                layer.forward_mqa(q, q, positions, output)
            torch.accelerator.synchronize()
            call = self._calls[-1]
            out[layer_id] = {
                "output": kind,
                "zeroed": all(bool((b == 0).all()) for b in buffers),
                "kernels": call["kernels"],
                "workspace_grew": workspace_bytes() > before,
            }
        self._calls = []
        return out

    def probe_report(self) -> list[dict]:
        calls, self._calls = self._calls, []
        return calls


def _layer_output(output, num_tokens: int, num_heads: int) -> torch.Tensor:
    from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation

    if isinstance(output, QuantizedActivation):
        rows = dequant_mxfp8_rows(output.data[:num_tokens], output.scale[:num_tokens])
        return rows.view(num_tokens, num_heads, HEAD_DIM)
    return output[:num_tokens, :num_heads].float()


def _errors(got, ref, inp: LayerInputs, md, layer) -> dict:
    swa = md[layer.swa_cache_layer.prefix]
    d = swa.num_decode_tokens
    valid = inp.is_valid.bool()
    out = {"finite": bool(torch.isfinite(got[valid]).all())}
    dec = valid.clone()
    dec[d:] = False
    pre = valid.clone()
    pre[:d] = False
    if dec.any():
        out["decode_err"] = rel_err(got[dec], ref[dec])
    if pre.any():
        out["prefill_err"] = rel_err(got[pre], ref[pre])
    return out


def window_slots(
    positions: torch.Tensor,
    reqs: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    replay_start: torch.Tensor | None,
    window: int = WINDOW,
) -> list[torch.Tensor]:
    """Each row's SWA cache slots, oldest first, rebuilt from positions."""
    rows = []
    for p, r in zip(positions.tolist(), reqs.tolist()):
        lo = max(p - window + 1, 0)
        if replay_start is not None:
            lo = max(lo, int(replay_start[r]))
        pos = torch.arange(lo, p + 1, device=block_table.device)
        rows.append(
            block_table[r, pos // block_size].long() * block_size + pos % block_size
        )
    return rows


def _ragged_matches(indices, indptr, want: list[torch.Tensor]) -> bool:
    if indices is None or indptr is None:
        return False
    bounds = indptr[: len(want) + 1].tolist()
    return all(
        torch.equal(indices[bounds[i] : bounds[i + 1]].long(), w)
        for i, w in enumerate(want)
    )


def _builder_checks(inp: LayerInputs, md, layer) -> dict:
    """The metadata the builders produced, against values rebuilt here."""
    swa = md[layer.swa_cache_layer.prefix]
    d, p = swa.num_decode_tokens, swa.num_prefill_tokens
    out: dict[str, Any] = {}
    valid = inp.is_valid.bool()
    if d and bool(valid[:d].all()):
        want = window_slots(
            inp.positions[:d],
            inp.token_to_req[:d],
            inp.swa_block_table,
            inp.swa_block_size,
            inp.replay_start,
        )
        out["decode_swa_ragged_ok"] = _ragged_matches(
            swa.decode_swa_ragged_indices, swa.decode_swa_ragged_indptr, want
        )
        out["decode_swa_ragged_storage"] = (
            swa.decode_swa_ragged_indices.untyped_storage().data_ptr()
        )
    if p:
        built = swa.prefill_swa_ragged_indices is not None
        out["prefill_swa_ragged_built"] = built
        if built and bool(valid[d : d + p].all()):
            want = window_slots(
                inp.positions[d : d + p],
                inp.token_to_req[d : d + p],
                inp.swa_block_table,
                inp.swa_block_size,
                inp.replay_start,
            )
            out["prefill_swa_ragged_ok"] = _ragged_matches(
                swa.prefill_swa_ragged_indices, swa.prefill_swa_ragged_indptr, want
            )
    if layer.compress_ratio and getattr(layer, "is_kv_source", False):
        cmp_md = md[layer.compressed_cache_prefix]
        ratio = layer.compress_ratio
        entries = cmp_md.block_size // ratio
        pos = inp.positions.long()
        reqs = inp.token_to_req.long()
        state = pos // ratio
        assert inp.compressed_block_table is not None
        bt = inp.compressed_block_table.long()
        want = bt[reqs, state // entries] * entries + state % entries
        last = ((pos + 1) % ratio == 0) & valid & (swa.slot_mapping[: len(pos)] >= 0)
        want = torch.where(last, want, torch.full_like(want, -1))
        got = cmp_md.slot_mapping[: len(pos)].long()
        out["compressed_slot_mapping_ok"] = bool(torch.equal(got, want))
    return out


# --------------------------------------------------------------------------
# Aggregation helpers for the tests.
# --------------------------------------------------------------------------


def summarize(calls: list[dict]) -> dict[tuple[int, str], dict]:
    """(layer, phase) -> {"count", "kernels" (union), "max errors"}."""
    out: dict[tuple[int, str], dict] = {}
    for c in calls:
        key = (c["layer"], c["phase"])
        s = out.setdefault(
            key,
            {"count": 0, "kernels": set(), "outputs": set(), "extend": False},
        )
        s["count"] += 1
        s["kernels"].update(c["kernels"])
        s["outputs"].add(c["output"])
        s["extend"] |= bool(c.get("extend"))
        for k in ("decode_err", "prefill_err", "bf16_err"):
            if k in c:
                s[k] = max(s.get(k, 0.0), c[k])
        if "neg_err" in c:
            s["neg_err"] = min(s.get("neg_err", math.inf), c["neg_err"])
        if "stats" in c:
            for k, v in c["stats"].items():
                s.setdefault("stats", defaultdict(int))[k] += v
        if c.get("finite") is False:
            s["nonfinite"] = s.get("nonfinite", 0) + 1
    return out


@contextlib.contextmanager
def aiter_sparse_mla(monkeypatch, enabled: bool):
    """Set the env gate in this process; spawned engine workers inherit it."""
    monkeypatch.setenv(AITER_ENV, "1" if enabled else "0")
    from vllm._aiter_ops import rocm_aiter_ops

    rocm_aiter_ops.refresh_env_variables()
    try:
        yield
    finally:
        monkeypatch.delenv(AITER_ENV, raising=False)
        rocm_aiter_ops.refresh_env_variables()


__all__ = [
    "AITER_ENV",
    "COMPRESS_RATIO",
    "DSV41_HEADS",
    "Dsv41AttnProbe",
    "Dsv41Heads",
    "LAYER_KINDS",
    "LayerInputs",
    "MODEL",
    "decode_v4_records",
    "dequant_mxfp8_rows",
    "engine_kwargs",
    "hf_overrides",
    "inverse_rope_gptj",
    "rope_gptj",
    "random_prompts",
    "reference_attention",
    "rel_err",
    "summarize",
]

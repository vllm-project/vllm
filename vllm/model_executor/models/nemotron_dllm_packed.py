# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared-context packed reads of Nemotron Labs Diffusion in bidirectional
read mode (`ar_mode=True`, `is_causal=False`).

One engine request carries several questions on one state, laid out as the
shared-context trainer packs them:

    [S][Q1 ... MASK][Q2 ... MASK] ... [Qn ... MASK]

- S <-> S bidirectional; S never attends to any question;
- Qk attends to S and to its own segment only (never to Qj);
- question positions restart at |S| (token j of Qk has position |S| + j);
- every question is read at its own mask slot.

A request opts in with `SamplingParams.extra_args["sc_layout"]`:
`{"s": |S|, "q": [len(Qk) incl. its mask], "c": [[candidate ids of Qk]]}`.
Attention is FlashAttention varlen with one query segment per S / Qk and
per-segment keys gathered as keys(S) + keys(Qk). Requests without a layout in
the same batch keep one segment over their own tokens.

Readout: the runner computes logits only at a request's last position, so a
packed request's last row is rewritten (fp32) to carry every question's
candidate log-probabilities at "virtual" token ids 0..N-1 (question order,
candidates in order), each shifted by -log(num_questions), plus one filler
entry at id N, so that the row's log-sum-exp is 0: the returned logprobs are
the per-slot full-vocabulary logprobs minus log(num_questions).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

SC_LAYOUT_KEY = "sc_layout"
# Values one packed read may return: 1,024 candidates plus the filler.
MAX_VIRTUAL_IDS = 1025


def parse_sc_layout(raw: Any, prompt_len: int, max_virtual_ids: int = MAX_VIRTUAL_IDS):
    """(state_len, [question lengths], [[candidate ids]]) or None; ValueError
    for a bad layout."""
    if raw is None:
        return None
    spec = json.loads(raw) if isinstance(raw, str) else raw
    s, lens, cands = int(spec["s"]), [int(x) for x in spec["q"]], spec["c"]
    if s < 0 or not lens or min(lens) < 1 or len(cands) != len(lens):
        raise ValueError("sc_layout: bad state/question lengths")
    if s + sum(lens) != prompt_len:
        raise ValueError(
            f"sc_layout: |S| + sum(|Qk|) = {s + sum(lens)} != prompt length "
            f"{prompt_len}"
        )
    cands = [[int(t) for t in c] for c in cands]
    if min(len(c) for c in cands) < 1:
        raise ValueError("sc_layout: every question needs candidates")
    if sum(len(c) for c in cands) + 1 > max_virtual_ids:
        raise ValueError("sc_layout: too many candidates in one packed read")
    return s, lens, cands


@dataclass
class PackedReadPlan:
    """Per-batch tensors for packed reads (on the model's device)."""

    num_reqs: int
    cu_q: torch.Tensor  # int32 [num_segments + 1]
    cu_k: torch.Tensor  # int32 [num_segments + 1]
    kv_idx: torch.Tensor  # int64 [cu_k[-1]]
    max_q: int
    max_k: int
    pos_idx: torch.Tensor  # int64: question token rows
    pos_val: torch.Tensor  # int64: their positions (restart at |S|)
    slot_rows: torch.Tensor  # int64: every packed mask slot
    # Virtual readout: row[dst_row, dst_col] = slot_logprob[src_slot, src_col] - log(Q)
    dst_row: torch.Tensor
    dst_col: torch.Tensor
    src_slot: torch.Tensor
    src_col: torch.Tensor
    slot_req: torch.Tensor  # int64 [num_slots]: the packed request of each slot
    packed_rows: torch.Tensor  # int64: logits rows of packed requests
    filler_col: torch.Tensor  # int64: per packed row, its filler virtual id
    num_questions: torch.Tensor  # float32: per packed row


def build_packed_plan(query_start_loc: np.ndarray, layouts: list) -> dict:
    """CPU plan of one batch. `layouts[i]` is None for a plain read or
    (state_len, question lengths, candidate ids) for request i, whose tokens
    are query_start_loc[i]:query_start_loc[i+1] (a whole prompt). Allowed
    attention is exactly the trainer's mask:
    same request & (key in S | key in the query's own segment)."""
    qb, kb, keys = [0], [0], []
    pos_idx, pos_val, slots, slot_req = [], [], [], []
    dst_row, dst_col, src_slot, src_col = [], [], [], []
    packed_rows, filler_col, num_q = [], [], []

    def seg(q_end, ranges):
        qb.append(q_end)
        n = 0
        for x, y in ranges:
            keys.append(np.arange(x, y, dtype=np.int64))
            n += y - x
        kb.append(kb[-1] + n)

    for i, layout in enumerate(layouts):
        a, b = int(query_start_loc[i]), int(query_start_loc[i + 1])
        if layout is None:
            seg(b, [(a, b)])
            continue
        s, lens, cands = layout
        if s + sum(lens) != b - a:
            raise RuntimeError(
                "packed read scheduled partially (bidirectional prompts must run "
                "in one step)"
            )
        if s:
            seg(a + s, [(a, a + s)])
        at, virtual = a + s, 0
        for n, cand in zip(lens, cands):
            seg(at + n, [(a, a + s), (at, at + n)])
            pos_idx.append(np.arange(at, at + n, dtype=np.int64))
            pos_val.append(np.arange(s, s + n, dtype=np.int64))
            j = len(slots)
            slots.append(at + n - 1)
            slot_req.append(len(packed_rows))
            m = len(cand)
            dst_row.append(np.full(m, i, dtype=np.int64))
            dst_col.append(np.arange(virtual, virtual + m, dtype=np.int64))
            src_slot.append(np.full(m, j, dtype=np.int64))
            src_col.append(np.asarray(cand, dtype=np.int64))
            virtual += m
            at += n
        packed_rows.append(i)
        filler_col.append(virtual)
        num_q.append(float(len(lens)))

    def cat(parts):
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)

    cu_q = np.asarray(qb, dtype=np.int32)
    cu_k = np.asarray(kb, dtype=np.int32)
    return dict(
        num_reqs=len(layouts),
        cu_q=cu_q,
        cu_k=cu_k,
        kv_idx=cat(keys),
        max_q=int((cu_q[1:] - cu_q[:-1]).max()),
        max_k=int((cu_k[1:] - cu_k[:-1]).max()),
        pos_idx=cat(pos_idx),
        pos_val=cat(pos_val),
        slot_rows=np.asarray(slots, dtype=np.int64),
        dst_row=cat(dst_row),
        dst_col=cat(dst_col),
        src_slot=cat(src_slot),
        src_col=cat(src_col),
        slot_req=np.asarray(slot_req, dtype=np.int64),
        packed_rows=np.asarray(packed_rows, dtype=np.int64),
        filler_col=np.asarray(filler_col, dtype=np.int64),
        num_questions=np.asarray(num_q, dtype=np.float32),
    )


def packed_attention_mask(query_start_loc: np.ndarray, layouts: list) -> np.ndarray:
    """Dense boolean [T, T] mask of the same layout (reference for tests)."""
    T = int(query_start_loc[len(layouts)])
    allowed = np.zeros((T, T), dtype=bool)
    for i, layout in enumerate(layouts):
        a, b = int(query_start_loc[i]), int(query_start_loc[i + 1])
        if layout is None:
            allowed[a:b, a:b] = True
            continue
        s, lens, _ = layout
        allowed[a : a + s, a : a + s] = True
        at = a + s
        for n in lens:
            allowed[at : at + n, a : a + s] = True
            allowed[at : at + n, at : at + n] = True
            at += n
    return allowed


def write_packed_readout(
    logits: torch.Tensor, plan: PackedReadPlan, slot_logits: torch.Tensor
) -> torch.Tensor:
    """Returns fp32 logits whose packed rows carry the virtual readout (see the
    module docstring). Plain rows are unchanged (bf16 -> fp32 is exact)."""
    if logits.shape[0] != plan.num_reqs:
        raise RuntimeError("packed reads: logits rows do not match the packed batch")
    out = logits.float()
    slot_lp = torch.log_softmax(slot_logits.float(), dim=-1)
    log_q = torch.log(plan.num_questions)  # per packed row
    out.index_fill_(0, plan.packed_rows, float("-inf"))
    shift = log_q.index_select(0, plan.slot_req).index_select(0, plan.src_slot)
    vals = slot_lp[plan.src_slot, plan.src_col] - shift
    out[plan.dst_row, plan.dst_col] = vals
    # Filler: 1 - (sum of the row's values' probabilities), so the row sums to 1.
    # A plain row reduction (deterministic), not index_add_ (CUDA atomics: the
    # filler, and so the label mass, could change in the last bits run to run).
    mass = out.index_select(0, plan.packed_rows).exp().sum(dim=-1)
    filler = torch.log1p(-mass.clamp(max=1.0))
    out[plan.packed_rows, plan.filler_col] = filler
    return out


def segment_attention(impl, layer, query, key, value, output, cu_q, cu_k, max_q, max_k):
    """Bidirectional varlen attention over segments with the FLASH_ATTN
    backend's own kernel and settings."""
    from vllm.v1.attention.backends import flash_attn as fa_backend

    num_segments = cu_q.shape[0] - 1
    descale_shape = (num_segments, impl.num_kv_heads)
    window = list(impl.sliding_window) if impl.sliding_window is not None else None
    fa_backend.flash_attn_varlen_func(
        q=query,
        k=key,
        v=value,
        out=output,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max_q,
        max_seqlen_k=max_k,
        softmax_scale=impl.scale,
        causal=False,
        alibi_slopes=impl.alibi_slopes,
        window_size=window,
        softcap=impl.logits_soft_cap,
        fa_version=impl.vllm_flash_attn_version,
        q_descale=layer._q_scale.expand(descale_shape)
        if impl.supports_quant_query_input
        else None,
        k_descale=layer._k_scale.expand(descale_shape),
        v_descale=layer._v_scale.expand(descale_shape),
        num_splits=1
        if impl.batch_invariant_enabled or getattr(impl, "fa4_hd256", False)
        else 0,
        s_aux=impl.sinks,
    )
    return output


def wrap_packed_encoder_attention(attn_layer: nn.Module) -> bool:
    """Instance-level wrapper of the FlashAttention backend's encoder path: when
    the batch metadata carries a packed plan (`nemotron_sc`), run varlen
    attention over the plan's segments with gathered keys/values. False if the
    layer's backend has no encoder path to wrap (not FLASH_ATTN)."""
    impl = getattr(attn_layer, "impl", None)
    if getattr(impl, "_nemotron_sc_wrapped", False):
        return True
    original = getattr(impl, "_forward_encoder_attention", None)
    if original is None or type(impl).__name__ != "FlashAttentionImpl":
        return False

    def forward_encoder_attention(query, key, value, output, attn_metadata, layer):
        plan = getattr(attn_metadata, "nemotron_sc", None)
        if plan is None:
            return original(query, key, value, output, attn_metadata, layer)
        kg = key.index_select(0, plan.kv_idx)
        vg = value.index_select(0, plan.kv_idx)
        return segment_attention(
            impl,
            layer,
            query,
            kg,
            vg,
            output,
            plan.cu_q,
            plan.cu_k,
            plan.max_q,
            plan.max_k,
        )

    impl._forward_encoder_attention = forward_encoder_attention
    impl._nemotron_sc_wrapped = True
    return True


_PACKED_STATE_CLS = None


def packed_encoder_only_state_cls():
    """EncoderOnlyModelState + packed reads (built lazily: GPU-worker imports)."""
    global _PACKED_STATE_CLS
    if _PACKED_STATE_CLS is not None:
        return _PACKED_STATE_CLS
    from vllm.config.compilation import CUDAGraphMode
    from vllm.utils.torch_utils import async_tensor_h2d
    from vllm.v1.worker.gpu.model_states.encoder_only import EncoderOnlyModelState

    class NemotronPackedReadModelState(EncoderOnlyModelState):
        """Encoder-only reads; requests with `sc_layout` run packed. Batches
        without packed requests take the unchanged encoder-only path."""

        def __init__(self, vllm_config, model, encoder_cache, device):
            super().__init__(vllm_config, model, encoder_cache, device)
            self._sc_layouts: dict[str, tuple] = {}
            self._plan: PackedReadPlan | None = None

        def add_request(self, req_index: int, new_req_data: Any) -> None:
            super().add_request(req_index, new_req_data)
            params = new_req_data.sampling_params
            extra = (params.extra_args or {}) if params is not None else {}
            raw = extra.get(SC_LAYOUT_KEY)
            if raw is None:
                return
            if not getattr(self.model, "_sc_supported", False):
                raise ValueError(
                    "packed Nemotron reads need the FLASH_ATTN attention backend"
                )
            prompt = new_req_data.prompt_token_ids or []
            self._sc_layouts[new_req_data.req_id] = parse_sc_layout(raw, len(prompt))

        def remove_request(self, req_id: str) -> None:
            super().remove_request(req_id)
            self._sc_layouts.pop(req_id, None)

        def prepare_attn(
            self,
            input_batch,
            cudagraph_mode,
            block_tables,
            slot_mappings,
            attn_groups,
            kv_cache_config,
            for_capture=False,
            ubatch_idx: int = 0,
            model_specific_attn_metadata=None,
        ) -> dict[str, Any]:
            metadata = super().prepare_attn(
                input_batch,
                cudagraph_mode,
                block_tables,
                slot_mappings,
                attn_groups,
                kv_cache_config,
                for_capture,
                ubatch_idx,
                model_specific_attn_metadata=model_specific_attn_metadata,
            )
            plan = None
            if self._sc_layouts and not for_capture:
                layouts = [
                    self._sc_layouts.get(r)
                    for r in input_batch.req_ids[: input_batch.num_reqs]
                ]
                if any(layout is not None for layout in layouts):
                    if cudagraph_mode == CUDAGraphMode.FULL:
                        raise RuntimeError(
                            "packed reads run attention eagerly; use PIECEWISE or no "
                            "CUDA graphs (not FULL)"
                        )
                    plan = self._device_plan(
                        build_packed_plan(
                            input_batch.query_start_loc_np[: input_batch.num_reqs + 1],
                            layouts,
                        )
                    )
            for group in self.encoder_attn_groups:
                for name in group.layer_names:
                    metadata[name].nemotron_sc = plan
            self._plan = plan
            self.model._sc_batch = plan
            return metadata

        def _device_plan(self, p: dict) -> PackedReadPlan:
            def dev(x):
                return async_tensor_h2d(x, device=self.device)

            return PackedReadPlan(
                num_reqs=p["num_reqs"],
                cu_q=dev(p["cu_q"]),
                cu_k=dev(p["cu_k"]),
                kv_idx=dev(p["kv_idx"]),
                max_q=p["max_q"],
                max_k=p["max_k"],
                pos_idx=dev(p["pos_idx"]),
                pos_val=dev(p["pos_val"]),
                slot_rows=dev(p["slot_rows"]),
                dst_row=dev(p["dst_row"]),
                dst_col=dev(p["dst_col"]),
                src_slot=dev(p["src_slot"]),
                src_col=dev(p["src_col"]),
                slot_req=dev(p["slot_req"]),
                packed_rows=dev(p["packed_rows"]),
                filler_col=dev(p["filler_col"]),
                num_questions=dev(p["num_questions"]),
            )

        def prepare_inputs(self, input_batch, req_states):
            plan = self._plan
            if plan is not None:
                # Question positions restart at |S|, as in training. In place:
                # the piecewise graphs read the runner's positions buffer.
                input_batch.positions.index_copy_(
                    0, plan.pos_idx, plan.pos_val.to(input_batch.positions.dtype)
                )
                self._plan = None
            return super().prepare_inputs(input_batch, req_states)

        def prepare_dummy_inputs(self, num_reqs: int, num_tokens: int):
            self._plan = None
            self.model._sc_batch = None
            return super().prepare_dummy_inputs(num_reqs, num_tokens)

    _PACKED_STATE_CLS = NemotronPackedReadModelState
    return _PACKED_STATE_CLS

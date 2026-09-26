# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Laya: an RL-agent decision model served as a vLLM pooling (``classify``) model.

Laya is *not* a generative LLM. It is a bidirectional ModernBERT encoder
(28 layers, hidden 1024) followed by a small transformer decision head. Every
answer option is represented by a ``[MASK]`` marker token in the prompt; the
head scores each marker and the softmax over those marker scores is the answer
distribution. A second head predicts whether the agent should escalate to a
human.

One request carries exactly one question, and the returned ``probs`` vector is
``[p_0, ..., p_{K-1}, act_escalate, act_not_escalate]`` where ``K`` is the
number of options. The final two entries are the two outputs of the act head.

The prompt layout (``[CLS] <type> question: <instructions> [SEP]`` followed by
``[MASK] <option text>`` per option, then ``[SEP] <state> [SEP]``) is built by
the client. The question type is recovered from the token right after ``[CLS]``.
"""

import math
import os
from collections.abc import Callable, Iterable, Set
from itertools import chain

import numpy as np
import torch
import torch.nn as nn

try:  # the pooling head needs the fused Ascend attention; CPU checkouts do not
    import torch_npu
except ImportError:  # pragma: no cover - unit tests import this module on CPU
    torch_npu = None


def _fused_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    head_num: int,
    scale: float,
    seq_lens: list[int],
) -> torch.Tensor:
    """Per-sequence attention over a packed token batch, as one fused op.

    TND layout means the operator reads the batch as consecutive sequences
    (``seq_lens`` is the cumulative length of each) and never attends across a
    sequence boundary, which is exactly the semantics of the key-padding mask
    it replaces. Indirected through a module-level function so the CPU
    equivalence test can swap in a reference implementation.
    """
    return torch_npu.npu_fusion_attention(
        query=query,
        key=key,
        value=value,
        head_num=head_num,
        input_layout="TND",
        scale=scale,
        actual_seq_qlen=seq_lens,
        actual_seq_kvlen=seq_lens,
    )[0]

from vllm.config import VllmConfig
from vllm.model_executor.layers.pooler import (
    DispatchPooler,
    Pooler,
    PoolingParamsUpdate,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.tasks import PoolingTask
from vllm.v1.outputs import PoolerOutput
from vllm.v1.pool.metadata import PoolingMetadata

from .interfaces_base import attn_type, default_pooling_type
from .modernbert import ModernBertModel
from .utils import maybe_prefix

QTYPE_NAMES = ("choice", "score", "noul")

# Upper bound on the cached per-step bookkeeping plans. Serving traffic draws
# from a handful of distinct batch compositions, so a small cache removes nearly
# all of the rebuilding while keeping the memory bounded.
# Plan entries are tiny (a few hundred bytes of device index tensors each), so
# the cache is sized for the *key* space, not for memory: a served batch draws
# its rows from a handful of prompt shapes, and every distinct batch composition
# is its own key. Measured on A2 with a mail-style English probe (~90 tokens)
# cycling four lengths, a 64-entry cache missed 40-70% of steps at c16/c32, at
# ~0.5 ms per miss.
_PLAN_CACHE_LIMIT = 512


def _exclusive_prefix(counts: np.ndarray) -> np.ndarray:
    """``[0, c0, c0 + c1, ...]``: the flat start offset of every row."""
    out = np.zeros(counts.size, dtype=np.int64)
    np.cumsum(counts[:-1], out=out[1:])
    return out


def _flatten_rows(rows: list[list[int]], total: int) -> np.ndarray:
    """All rows of a ragged list-of-lists as one contiguous int64 vector."""
    if not total:
        return np.empty(0, dtype=np.int64)
    return np.fromiter(chain.from_iterable(rows), dtype=np.int64, count=total)


def _new_pinned(numel: int, dtype: torch.dtype) -> torch.Tensor:
    """A CPU buffer that is pinned when the platform has a pinned allocator."""
    try:
        return torch.empty(numel, dtype=dtype, pin_memory=True)
    except Exception:
        # No accelerator (CPU-only unit tests) or no pinned allocator.
        return torch.empty(numel, dtype=dtype)


def _stream_event():
    """Event recorded on the current stream, or None where events are missing."""
    try:
        event = torch.npu.Event()
        event.record()
        return event
    except Exception:
        return None


# Fitted temperatures below 1 sharpen the logits instead of softening them. The shipped
# `choice:11+` bucket is 0.1006, which multiplies them ~10x: a 0.24 top probability is
# published as 0.99, so a caller gating on confidence is told a coin flip is a certainty.
# `laya.common.clamp_temperature` (SDK >= 0.3.5) refuses to apply one that sharpens this
# hard; mirror that policy so served probabilities match the official runtime.
TEMP_MIN = 0.5
TEMP_MAX = 5.0
# The upstream research harness (`research/scripts/bench_local.py`) predates the clamp and
# applies the raw fitted value, so `LAYA_TEMPERATURE_CLAMP=0` reproduces the published
# `t4_colab_benchmark.json` probability columns for the `choice:11+` bucket.
TEMPERATURE_CLAMP = os.environ.get("LAYA_TEMPERATURE_CLAMP", "1") != "0"


def _clamp_temperature(value) -> float:
    """Mirror of ``laya.common.clamp_temperature``: confine to [TEMP_MIN, TEMP_MAX]."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 1.0
    if value != value or value in (float("inf"), float("-inf")):
        return 1.0
    return min(TEMP_MAX, max(TEMP_MIN, value))


def _temperature_bucket(qtype_name: str, num_options: int) -> str:
    """Mirror of ``rl_common.temp_bucket``."""
    if num_options <= 2:
        size = "2"
    elif num_options <= 5:
        size = "3-5"
    elif num_options <= 10:
        size = "6-10"
    else:
        size = "11+"
    return f"{qtype_name}:{size}"


class LayaPooler(Pooler):
    """Scores the ``[MASK]`` markers of every request and returns Laya's outputs."""

    def __init__(
        self,
        score_fn: Callable[
            [torch.Tensor, list[int], list[torch.Tensor], list[bool]],
            list[torch.Tensor],
        ],
    ) -> None:
        super().__init__()
        # `score_fn` is a bound method of `LayaForDecision`, so all Laya-specific
        # configuration is read from the model itself.
        self.score_fn = score_fn

    def get_supported_tasks(self) -> Set[PoolingTask]:
        return {"classify"}

    def get_pooling_updates(self, task: PoolingTask) -> PoolingParamsUpdate:
        return PoolingParamsUpdate(requires_token_ids=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> PoolerOutput:
        cursor = pooling_metadata.get_pooling_cursor()
        # Derive the per-request token ranges from the CPU view so the pooler
        # never forces a device->host synchronization.
        offsets = [0]
        for num in cursor.num_scheduled_tokens_cpu.tolist():
            offsets.append(offsets[-1] + int(num))

        prompt_token_ids = pooling_metadata.get_prompt_token_ids_cpu()
        use_activation = [
            params.use_activation is not False
            for params in pooling_metadata.pooling_params
        ]
        return self.score_fn(hidden_states, offsets, prompt_token_ids, use_activation)


@attn_type("encoder_only")
@default_pooling_type(seq_pooling_type="CLS")
class LayaForDecision(nn.Module):
    """Laya decision model, served with the ``classify`` pooling task."""

    is_pooling_model = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config

        self.model = ModernBertModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "modernbert")
        )

        hidden_size = config.hidden_size
        # The decision head is numerically sensitive: it is always evaluated in
        # float32 regardless of the encoder dtype.
        head_dtype = vllm_config.model_config.head_dtype or torch.float32
        self.head_dtype = head_dtype

        self.head_layers = int(getattr(config, "head_layers", 2))
        num_act = int(getattr(config, "n_act", 2))

        num_heads = max(1, hidden_size // 64)
        encoder_layer = nn.TransformerEncoderLayer(
            hidden_size,
            num_heads,
            4 * hidden_size,
            0.1,
            batch_first=True,
            norm_first=True,
        )
        self.head = nn.TransformerEncoder(
            encoder_layer, self.head_layers, enable_nested_tensor=False
        ).to(head_dtype)

        self.type_emb = nn.Embedding(3, hidden_size, dtype=head_dtype)
        self.scorer = nn.Sequential(
            nn.LayerNorm(hidden_size, dtype=head_dtype),
            nn.Linear(hidden_size, hidden_size, dtype=head_dtype),
            nn.GELU(),
            nn.Linear(hidden_size, 1, dtype=head_dtype),
        )
        self.act_head = nn.Sequential(
            nn.Linear(hidden_size + 4, 256, dtype=head_dtype),
            nn.GELU(),
            nn.Linear(256, num_act, dtype=head_dtype),
        )

        # The checkpoint carries a `temperature` buffer that is all ones and is
        # unused at inference; the fitted values live in `temperature_by_options`.
        name_to_qtype = {name: idx for idx, name in enumerate(QTYPE_NAMES)}
        self.type_token_ids = {
            int(token_id): name_to_qtype[name]
            for name, token_id in (getattr(config, "type_token_ids", None) or {}).items()
            if name in name_to_qtype
        }
        self.temperature = list(getattr(config, "temperature", None) or [1.0] * 3)
        self.temperature_by_options = dict(
            getattr(config, "temperature_by_options", None) or {}
        )
        self.mask_token_id = int(getattr(config, "mask_token_id", 50284))

        # Cached per-step bookkeeping for the batched head; see `_step_plan`.
        self._plan_cache: dict[tuple, dict] = {}
        # Reusable pinned staging buffers for the plan build; see `_ship_pinned`.
        self._plan_host: dict[str, list] = {}
        self._plan_host_rr: dict[str, int] = {}

        self.pooler = DispatchPooler({"classify": LayaPooler(self.score_requests)})

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            positions=positions,
        )

    def _answer_temperature(self, qtype: int, num_options: int) -> float:
        key = _temperature_bucket(QTYPE_NAMES[qtype], num_options)
        value = self.temperature_by_options.get(key, self.temperature[qtype])
        return _clamp_temperature(value) if TEMPERATURE_CLAMP else float(value)

    def _head_forward(self, x: torch.Tensor, seq_lens: list[int]) -> torch.Tensor:
        """Run the decision head's transformer stack on a packed token batch.

        ``nn.TransformerEncoder`` needs a rectangular batch, and padding a step
        of unequal prompts costs twice: the pad rows are real compute (a c=16
        step pads 16 prompts up to the longest one), and the resulting
        key-padding mask pushes ``F.scaled_dot_product_attention`` onto the math
        backend, whose host-side decomposition costs ~0.8 ms per step more than
        the fused path. The head only ever sees a batch that is already packed
        -- ``hidden_states`` is the concatenation of the step's prompts -- so
        the same two layers run on it directly with ``npu_fusion_attention`` in
        TND layout: per-prompt attention, no mask, no pad rows. Verified equal
        to the padded path to fp32 rounding (micro_head.py).
        """
        for layer in self.head.layers:
            attn = layer.self_attn
            tokens = x.shape[0]
            heads, head_dim = attn.num_heads, attn.head_dim
            h = layer.norm1(x)
            qkv = nn.functional.linear(h, attn.in_proj_weight, attn.in_proj_bias)
            query, key, value = qkv.chunk(3, dim=-1)
            fused = _fused_attention(
                query=query.reshape(tokens, heads, head_dim),
                key=key.reshape(tokens, heads, head_dim),
                value=value.reshape(tokens, heads, head_dim),
                head_num=heads,
                scale=1.0 / math.sqrt(head_dim),
                seq_lens=seq_lens,
            )
            x = x + layer.dropout1(
                nn.functional.linear(
                    fused.reshape(tokens, -1), attn.out_proj.weight, attn.out_proj.bias
                )
            )
            h = layer.norm2(x)
            x = x + layer.dropout2(
                layer.linear2(layer.dropout(layer.activation(layer.linear1(h))))
            )
        return x

    def _marker_positions(self, ids: torch.Tensor, num_tokens: int) -> list[int]:
        return (ids[:num_tokens] == self.mask_token_id).nonzero(as_tuple=True)[0].tolist()

    def _ship_pinned(
        self, name: str, values, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        """Copy host values to the device through a reusable pinned buffer.

        `values` is either a numpy array (written straight into the pinned
        buffer) or a small CPU tensor (memcpy'd into it). Buffers are recycled
        per `name`, but only once the event of the copy that last used them has
        completed: a plan miss can be followed by another one a millisecond
        later, while the copy queued behind a busy encoder has not run yet, so
        writing the buffer blind would ship the *next* step's indices.
        """
        n = int(values.size) if isinstance(values, np.ndarray) else int(values.numel())
        rings = getattr(self, "_plan_host", None)
        if rings is None:
            rings = self._plan_host = {}
        ring = rings.setdefault(name, [])
        slot = None
        for i, cand in enumerate(ring):
            if cand[0].numel() >= n and (cand[1] is None or cand[1].query()):
                slot = ring.pop(i)
                ring.append(slot)
                break
        if slot is None:
            numel = 4096
            while numel < n:
                numel *= 2
            slot = [_new_pinned(numel, dtype), None]
            ring.append(slot)
        cpu = slot[0]
        if isinstance(values, np.ndarray):
            view = cpu.numpy()
            view[:n] = values
        else:
            cpu[:n].copy_(values)
        out = cpu[:n].to(device, non_blocking=True)
        slot[1] = _stream_event()
        return out

    def _step_plan(
        self,
        lengths: list[int],
        markers: list[list[int]],
        qtypes: list[int],
        use_activation: list[bool],
        device: torch.device,
    ) -> dict:
        """Index/mask tensors the batched head needs, cached by batch shape.

        Every entry depends only on the token counts, the marker positions, the
        question types and the activation flags -- all of which repeat in
        serving traffic (a benchmark replays one prompt, and a real batch draws
        from a handful of shapes). Building them per step costs one mask, three
        host->device copies and the gather indices; the cache turns that into a
        dict lookup.
        """
        num_markers = [len(row) for row in markers]
        key = (
            str(device),
            tuple(lengths),
            tuple(num_markers),
            tuple(pos for row in markers for pos in row),
            tuple(qtypes),
            tuple(use_activation),
        )
        plan = self._plan_cache.get(key)
        if plan is not None:
            # LRU: a hit refreshes the entry so the eviction victim is the plan
            # whose batch shape has gone longest without recurring.
            self._plan_cache.pop(key)
            self._plan_cache[key] = plan
            return plan
        batch = len(lengths)
        kmax = max(num_markers)
        # Everything below is assembled with numpy and shipped in three
        # host->device copies. Building the same values as Python lists and
        # handing them to `torch.tensor` costs ~150 ns per element -- ~3.5 ms
        # for the 23k-element index vector a c16 step needs, plus ~3 ms for the
        # uint8 mask -- and a pageable copy blocks behind the encoder still
        # running on the device. Measured per miss at c16/c32: 8-14 ms, i.e.
        # more than the whole 9 ms encoder step it belongs to. The numpy build
        # is ~0.2 ms and the copies go out of pinned memory, which does not wait
        # for the queue behind it (6.55 ms -> 0.66 ms measured with a 10 ms
        # backlog; see micro_h2d.py).
        lengths_np = np.asarray(lengths, dtype=np.int64)
        counts_np = np.asarray(num_markers, dtype=np.int64)
        rows_cnt = np.arange(batch, dtype=np.int64) * kmax
        # The encoder output arrives packed, request after request: `starts` is
        # where each request begins, which is both the base its `[MASK]`
        # positions are relative to and the row the act head reads.
        starts = _exclusive_prefix(lengths_np)
        offs_cnt = _exclusive_prefix(counts_np)
        seq_lens = np.cumsum(lengths_np).tolist()
        # `pick` takes the `[MASK]` rows back out of the packed batch: request
        # `i` owns the `counts[i]` slots that start at `i * kmax`, and the rest
        # of its `kmax` slots repeat its first row, where `valid` masks them.
        # The answer half of `gather` is exactly those same slot positions.
        total_cnt = int(counts_np.sum())
        pick_n = batch * kmax
        slots = np.arange(total_cnt, dtype=np.int64) + np.repeat(
            rows_cnt - offs_cnt, counts_np
        )
        pick = np.empty(pick_n, dtype=np.int64)
        pick[slots] = _flatten_rows(markers, total_cnt) + np.repeat(starts, counts_np)
        pad_slots = np.ones(pick_n, dtype=bool)
        pad_slots[slots] = False
        pick[pad_slots] = np.repeat(starts, kmax - counts_np)
        # Answers stay padded to `kmax` on the device; `gather` picks exactly the
        # `num_markers[i]` answers of request `i` followed by its two act
        # outputs, so a whole step is handed out as one gather plus one split
        # instead of two kernels and a concat per request.
        # ... and per request it is the answers followed by the two act outputs,
        # so the gather index interleaves request by request.
        gather = np.empty(total_cnt + 2 * batch, dtype=np.int64)
        rows_i = np.arange(batch, dtype=np.int64)
        gather[np.arange(total_cnt, dtype=np.int64) + np.repeat(2 * rows_i, counts_np)] = (
            slots
        )
        act_slot = offs_cnt + 2 * rows_i + counts_np
        gather[np.stack((act_slot, act_slot + 1), axis=1).reshape(-1)] = (
            batch * kmax + np.arange(2 * batch, dtype=np.int64)
        )
        # The type embedding is per token, not per request: one `index_select`
        # of the step's tokens replaces the broadcast add the padded layout used.
        token_types = np.repeat(np.asarray(qtypes, dtype=np.int64), lengths_np)
        idx_np = np.concatenate((pick, starts, token_types, gather, counts_np))
        # Layout of the packed int64 buffer:
        # pick | starts | token_types | gather | counts
        starts_off = pick_n
        types_off = starts_off + batch
        gather_off = types_off + int(token_types.size)
        counts_off = gather_off + int(gather.size)
        # Layout: valid (1 = real answer) | use_act
        valid = np.zeros(pick_n, dtype=np.uint8)
        valid[slots] = 1
        mask_t = self._ship_pinned(
            "mask",
            np.concatenate((valid, np.asarray(use_activation, dtype=np.uint8))),
            torch.uint8,
            device,
        ).view(torch.bool)
        idx = self._ship_pinned("idx", idx_np, torch.long, device)
        floats = [float(num) / 255.0 for num in num_markers]
        floats.extend(math.log(max(num, 2)) for num in num_markers)
        floats.extend(
            self._answer_temperature(qtypes[i], num_markers[i])
            if use_activation[i]
            else 1.0
            for i in range(batch)
        )
        float_t = self._ship_pinned(
            "floats",
            torch.tensor(floats, dtype=self.head_dtype),
            self.head_dtype,
            device,
        )
        valid_n = batch * kmax
        idx_view = idx.narrow
        mask_view = mask_t.narrow
        f_view = float_t.narrow
        plan = {
            "batch": batch,
            "kmax": kmax,
            "num_markers": num_markers,
            "pick": idx_view(0, 0, pick_n),
            "starts": idx_view(0, starts_off, batch),
            "token_types": idx_view(0, types_off, int(token_types.size)),
            "counts": idx_view(0, counts_off, batch),
            "counts_scaled": f_view(0, 0, batch),
            "log_counts": f_view(0, batch, batch),
            "valid": mask_view(0, 0, valid_n).view(batch, kmax),
            "sizes": [num + 2 for num in num_markers],
            "gather": idx_view(0, gather_off, int(gather.size)),
            "use_act": mask_view(0, valid_n, batch),
            "temperature": f_view(0, 2 * batch, batch),
            # Reused by the head: `torch.stack` of four `[batch]` columns costs
            # ~3.3 ms of aclnn tiling on this CANN, a preallocated buffer costs
            # four copies into it.
            "feats": torch.empty((batch, 4), dtype=self.head_dtype, device=device),
            "seq_lens": seq_lens,
        }
        while len(self._plan_cache) >= _PLAN_CACHE_LIMIT:
            self._plan_cache.pop(next(iter(self._plan_cache)))
        self._plan_cache[key] = plan
        return plan

    @torch.no_grad()
    def score_requests(
        self,
        hidden_states: torch.Tensor,
        offsets: list[int],
        prompt_token_ids: list[torch.Tensor],
        use_activation: list[bool],
    ) -> list[torch.Tensor]:
        """Score one engine step's requests with the batched head."""
        return self._score_requests_batched(
            hidden_states, offsets, prompt_token_ids, use_activation
        )

    @torch.no_grad()
    def _score_requests_batched(
        self,
        hidden_states: torch.Tensor,
        offsets: list[int],
        prompt_token_ids: list[torch.Tensor],
        use_activation: list[bool],
    ) -> list[torch.Tensor]:
        """Score every request of one engine step with a single head batch.

        The obvious implementation calls the decision head once per request. On
        Ascend that is pure launch overhead: a 96-token request costs ~40 kernel
        launches for ~0.2 GFLOP, and a step of eight of them was measured at
        ~350 launches inside the pooler alone, with the device idle ~60% of the
        step. Padding the requests of a step into one ``(n, maxlen, d)`` batch
        and masking the padding is numerically identical -- attention is masked
        out and every other head op is per position -- and collapses those
        launches by ~8x.
        """
        if not prompt_token_ids:
            return []

        lengths: list[int] = []
        markers: list[list[int]] = []
        qtypes: list[int] = []
        for i, ids in enumerate(prompt_token_ids):
            num_tokens = min(offsets[i + 1] - offsets[i], ids.numel())
            lengths.append(num_tokens)
            qtypes.append(
                self.type_token_ids.get(int(ids[1]), 0) if ids.numel() > 1 else 0
            )
            markers.append(self._marker_positions(ids, num_tokens))
        has_markers = all(markers)
        # The packed head reads every request straight out of
        # `hidden_states` at the offsets the runner scheduled, so a request
        # that scheduled more tokens than its own prompt holds (never
        # observed) cannot be read that way and keeps the scalar path.
        packed_ok = sum(lengths) == offsets[len(lengths)]
        if not (has_markers and packed_ok):
            # A prompt without a single `[MASK]` marker is malformed for Laya;
            # keep it on the scalar path so a poisoned request cannot abort the
            # whole engine (aclnnTopk 161002 used to kill the EngineCore).
            fallback = self._score_requests_scalar(
                hidden_states, offsets, prompt_token_ids, use_activation
            )
            return fallback

        device = hidden_states.device
        plan = self._step_plan(lengths, markers, qtypes, use_activation, device)
        batch = plan["batch"]
        kmax = plan["kmax"]
        num_markers = plan["num_markers"]

        # The step's prompts already sit one after the other in
        # `hidden_states`, so the type embedding is the only per-request
        # work left before the head runs.
        hidden = hidden_states[: offsets[batch]].to(self.head_dtype)
        hidden = hidden + self.type_emb.weight.index_select(0, plan["token_types"])
        encoded = self._head_forward(hidden, plan["seq_lens"])

        picked = encoded.index_select(0, plan["pick"])
        logits = self.scorer(picked.view(batch, kmax, -1)).squeeze(-1)

        # Padding becomes exactly -inf, so its softmax weight is 0 and it drops
        # out of both the entropy sum and the top-2 margin. A single option then
        # yields margin == top1, matching the scalar implementation.
        masked = logits.masked_fill(~plan["valid"], float("-inf"))
        probs = torch.softmax(masked, -1)
        entropy = -(
            probs * torch.log(probs.clamp_min(1e-9))
        ).sum(-1) / plan["log_counts"]
        # A step whose requests all carry a single `[MASK]` has kmax == 1,
        # where `topk(2)` indexes out of range on Ascend (aclnnTopk 161002)
        # and kills the EngineCore. The scalar path answers that case with
        # margin == top1; mirror it so the two paths agree at kmax == 1.
        kk = min(2, kmax)
        topk_vals = probs.topk(kk, dim=-1).values
        top1 = topk_vals[:, 0]
        margin = top1 - topk_vals[:, 1] if kk >= 2 else top1.clone()
        # Four `[batch]` columns written into a buffer carried by the plan:
        # `torch.stack` of them costs ~3.3 ms of aclnn tiling
        # (aclnnStackGetWorkspaceSize) for a tensor this size.
        feats = plan["feats"]
        feats[:, 0] = top1
        feats[:, 1] = margin
        feats[:, 2] = entropy
        feats[:, 3] = plan["counts_scaled"]
        act = torch.softmax(
            self.act_head(
                torch.cat([encoded.index_select(0, plan["starts"]).float(), feats], dim=-1)
            ),
            -1,
        )
        activated = torch.softmax(masked / plan["temperature"].unsqueeze(1), -1)

        # Answer and act head are concatenated for the whole batch at once
        # and handed out as contiguous 1-D views, replacing two kernels and
        # a host-side concat per request.
        chosen = torch.where(plan["use_act"].unsqueeze(1), activated, masked)
        flat = (
            torch.cat([chosen.reshape(-1), act.reshape(-1)])
            .index_select(0, plan["gather"])
            .float()
        )
        outputs = list(flat.split(plan["sizes"]))
        return outputs

    @torch.no_grad()
    def _score_requests_scalar(
        self,
        hidden_states: torch.Tensor,
        offsets: list[int],
        prompt_token_ids: list[torch.Tensor],
        use_activation: list[bool],
    ) -> list[torch.Tensor]:
        """Reference path, one request at a time, for malformed prompts."""
        outputs: list[torch.Tensor] = []
        for i, ids in enumerate(prompt_token_ids):
            start, end = offsets[i], offsets[i + 1]
            num_tokens = min(end - start, ids.numel())
            h = hidden_states[start : start + num_tokens].to(self.head_dtype)

            qtype = self.type_token_ids.get(int(ids[1]), 0) if ids.numel() > 1 else 0
            h = h + self.type_emb.weight[qtype]

            h = self.head(h.unsqueeze(0)).squeeze(0)

            markers = (ids[:num_tokens] == self.mask_token_id).nonzero(as_tuple=True)[0]
            logits = self.scorer(h[markers]).squeeze(-1).float()

            # Act head inputs: the pooled CLS position plus a detached summary
            # of the answer distribution (top-1, top1-top2 margin, entropy,
            # option count).
            probs = torch.softmax(logits.detach(), -1)
            num_options = max(int(logits.numel()), 2)
            entropy = -(probs * torch.log(probs.clamp_min(1e-9))).sum() / math.log(
                num_options
            )
            if probs.numel() >= 2:
                top2 = probs.topk(2).values
                top1, margin = top2[0], top2[0] - top2[1]
            else:
                top1 = probs.max() if probs.numel() == 1 else logits.new_zeros(())
                margin = top1
            feats = torch.stack(
                [top1, margin, entropy, logits.new_tensor(logits.numel() / 255.0)]
            )
            act_logits = self.act_head(torch.cat([h[0].float(), feats]))
            act = torch.softmax(act_logits, -1)

            if use_activation[i]:
                answer = torch.softmax(
                    logits / self._answer_temperature(qtype, logits.numel()), -1
                )
            else:
                answer = logits

            outputs.append(torch.cat([answer, act]).float())
        return outputs

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        encoder_weights: list[tuple[str, torch.Tensor]] = []

        for name, loaded_weight in weights:
            if name.startswith("encoder."):
                # ModernBERT encoder; handed over without its `encoder.` prefix.
                encoder_weights.append((name[len("encoder.") :], loaded_weight))
                continue
            if name == "temperature":
                continue
            param = params_dict.get(name)
            if param is None:
                continue
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)

        # `track_weights_loading` compares against `self.named_parameters()`, so
        # the names returned by the encoder must be re-prefixed with `model.`
        # (the attribute this module stores the encoder under).
        loaded_params.update(
            f"model.{name}" for name in self.model.load_weights(encoder_weights)
        )
        return loaded_params

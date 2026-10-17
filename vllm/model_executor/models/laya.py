# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Laya: an RL-agent decision model served as a vLLM pooling (``classify``) model.

Laya is not a generative LLM. It is a bidirectional ModernBERT encoder followed
by a small transformer decision head. Every answer option is represented by a
``[MASK]`` marker token in the prompt; the head scores each marker and the
softmax over those marker scores is the answer distribution. A second head
predicts whether the agent should escalate to a human. Nothing in the model
generates text, so there is no KV cache, no sampling and no decode phase.

One request carries exactly one question, and the returned ``probs`` vector is
``[p_0, ..., p_{K-1}, act_escalate, act_not_escalate]`` where ``K`` is the
number of options. The final two entries are the two outputs of the act head.

The prompt layout (``[CLS] <type> question: <instructions> [SEP]`` followed by
``[MASK] <option text>`` per option, then ``[SEP] <state> [SEP]``) is built by
the client. The question type is recovered from the token right after ``[CLS]``.

Every numeric and structural constant below mirrors the reference
implementation that ships with the checkpoint (``rl_common.DecisionModel`` /
``rl_agent_api.RLAgent``). Where a value could plausibly drift between
checkpoints it is read from the model config rather than hard-coded here.
"""

import math
from collections import OrderedDict
from collections.abc import Callable, Iterable, Set
from itertools import chain

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.pooler import (
    DispatchPooler,
    Pooler,
    PoolingParamsUpdate,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.tasks import PoolingTask
from vllm.tokenizers.registry import cached_tokenizer_from_config
from vllm.v1.outputs import PoolerOutput
from vllm.v1.pool.metadata import PoolingMetadata

from .interfaces_base import attn_type, default_pooling_type
from .modernbert import ModernBertModel
from .utils import maybe_prefix

logger = init_logger(__name__)

# The three decision primitives and their ``qtype`` index, mirroring
# ``rl_common.QTYPES``. The index selects both the type embedding row and the
# per-question-type temperature, so the order is part of the checkpoint format.
QTYPE_NAMES = ("choice", "score", "noul")

# Decision-head architecture, mirroring ``rl_common.DecisionModel.__init__``.
# ``HEAD_HEAD_DIM`` is the divisor that turns the encoder width into the head's
# attention head count (``nhead = max(1, d // 64)``), not the encoder's own
# ``head_dim``.
HEAD_HEAD_DIM = 64
HEAD_FFN_MULTIPLIER = 4
HEAD_DROPOUT = 0.1
HEAD_TYPE_EMBEDDINGS = 3
ACT_HIDDEN_SIZE = 256
ACT_FEATURE_DIM = 4
# The act head emits one escalate/act pair per request, so every request owns
# two extra outputs after its `K` answer probabilities.
ACT_OUTPUTS_PER_REQUEST = 2

# Numeric constants of the answer distribution, mirroring
# ``rl_common.DecisionModel.forward``.
# Padding markers are filled with a finite sentinel rather than ``-inf``: the
# upstream implementation uses ``-1e4`` and a finite sentinel keeps the value
# JSON-serialisable when the caller asks for raw logits
# (``use_activation=False``).
ANSWER_MASK_LOGIT = -1e4
ENTROPY_EPSILON = 1e-9
OPTION_COUNT_SCALE = 255.0
MIN_OPTIONS_FOR_ENTROPY = 2

# Deployment knobs (overridable from the model config, see ``LayaForDecision``).
# A served batch draws its rows from a handful of prompt shapes, so a small
# cache of the per-step bookkeeping removes nearly all of the rebuilding while
# keeping memory bounded. Plan entries are tiny (a few hundred bytes of device
# index tensors each), so the limit is sized for the *key* space, not memory.
DEFAULT_MAX_CACHED_PLANS = 512
# Smallest pinned staging buffer; buffers double until they fit a step.
DEFAULT_PINNED_MIN_ELEMENTS = 4096
# The token prefix that turns a question type name into its leading token.
TYPE_PROBE_SUFFIX = " question: "


def _resolve_fused_attention():
    """The packed (TND) fused attention op, or None off Ascend.

    Resolved once per model instance so that importing this module stays
    platform-agnostic: the reference fallback in ``_packed_attention`` is used
    wherever the fused operator is unavailable.
    """
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        return None
    return getattr(torch.ops, "npu", None) and torch.ops.npu.npu_fusion_attention


def _packed_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    head_num: int,
    scale: float,
    seq_lens: list[int],
) -> torch.Tensor:
    """Per-sequence attention over a packed token batch, portable reference.

    ``query``/``key``/``value`` are ``[num_tokens, head_num, head_dim]`` and
    ``seq_lens`` holds the cumulative length of each request, so the batch is
    the concatenation of the step's prompts. A block-diagonal mask reproduces
    the TND semantics of the fused operator: tokens only attend inside their own
    request, which is the key-padding mask it replaces.
    """
    num_tokens = query.shape[0]
    segment = torch.zeros(num_tokens, dtype=torch.long, device=query.device)
    for i, end in enumerate(seq_lens):
        start = seq_lens[i - 1] if i > 0 else 0
        segment[start:end] = i
    mask = segment[:, None] == segment[None, :]
    q = query.transpose(0, 1).unsqueeze(0)
    k = key.transpose(0, 1).unsqueeze(0)
    v = value.transpose(0, 1).unsqueeze(0)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)
    return out.squeeze(0).transpose(0, 1).reshape(num_tokens, head_num, -1)


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


def _temperature_bucket(qtype_name: str, num_options: int) -> str:
    """Mirror of ``rl_common.temp_bucket``: the per-cardinality temperature key."""
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

    # Defaults used when the served config does not carry the Laya head metadata
    # (i.e. when a raw checkpoint directory is used instead of a prepared one).
    default_head_layers = 2
    default_num_act = 2

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        model_config = vllm_config.model_config
        config = model_config.hf_config
        self.config = config

        self.model = ModernBertModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "modernbert")
        )

        hidden_size = config.hidden_size
        # The decision head is numerically sensitive: it is always evaluated in
        # float32 regardless of the encoder dtype.
        head_dtype = model_config.head_dtype or torch.float32
        self.head_dtype = head_dtype

        self.head_layers = int(getattr(config, "head_layers", self.default_head_layers))
        num_act = int(getattr(config, "n_act", self.default_num_act))

        encoder_layer = nn.TransformerEncoderLayer(
            hidden_size,
            max(1, hidden_size // HEAD_HEAD_DIM),
            HEAD_FFN_MULTIPLIER * hidden_size,
            HEAD_DROPOUT,
            batch_first=True,
            norm_first=True,
        )
        self.head = nn.TransformerEncoder(
            encoder_layer, self.head_layers, enable_nested_tensor=False
        ).to(head_dtype)

        self.type_emb = nn.Embedding(len(QTYPE_NAMES), hidden_size, dtype=head_dtype)
        self.scorer = nn.Sequential(
            nn.LayerNorm(hidden_size, dtype=head_dtype),
            nn.Linear(hidden_size, hidden_size, dtype=head_dtype),
            nn.GELU(),
            nn.Linear(hidden_size, 1, dtype=head_dtype),
        )
        self.act_head = nn.Sequential(
            nn.Linear(hidden_size + ACT_FEATURE_DIM, ACT_HIDDEN_SIZE, dtype=head_dtype),
            nn.GELU(),
            nn.Linear(ACT_HIDDEN_SIZE, num_act, dtype=head_dtype),
        )

        self._resolve_token_metadata(model_config, config)

        # The checkpoint carries a `temperature` buffer that is all ones and is
        # unused at inference; the fitted values live in
        # `temperature_by_options` and `temperature` (per question type), and are
        # applied exactly as `rl_agent_api.RLAgent.system_one` applies them.
        self.temperature = [
            float(value) for value in getattr(config, "temperature", None) or []
        ]
        self.temperature_by_options = {
            str(key): float(value)
            for key, value in (
                getattr(config, "temperature_by_options", None) or {}
            ).items()
        }

        # Cached per-step bookkeeping for the batched head; see `_step_plan`.
        self._plan_cache: OrderedDict[tuple, dict] = OrderedDict()
        self._plan_cache_limit = int(
            getattr(config, "plan_cache_limit", DEFAULT_MAX_CACHED_PLANS)
        )
        # Reusable pinned staging buffers for the plan build; see `_ship_pinned`.
        self._plan_host: dict[str, list] = {}
        self._pinned_min_elements = int(
            getattr(config, "pinned_buffer_min_elements", DEFAULT_PINNED_MIN_ELEMENTS)
        )
        self._fused_attention = _resolve_fused_attention()

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

    def _resolve_token_metadata(self, model_config, config) -> None:
        """Resolve the marker token and the question-type token map.

        Both are part of the prompt convention rather than of the encoder
        weights, so a prepared serving directory carries them in ``config.json``.
        When they are absent they are derived from the tokenizer exactly the way
        the prompt builder derives them: ``[MASK]`` marks an option, and the
        first token of ``"<type> question: "`` is the type token that follows
        ``[CLS]``.
        """
        mask_token_id = getattr(config, "mask_token_id", None)
        type_token_ids = getattr(config, "type_token_ids", None)
        if mask_token_id is not None and type_token_ids:
            self.mask_token_id = int(mask_token_id)
            self.type_token_ids = self._qtype_map(type_token_ids)
            return

        tokenizer = None
        try:
            tokenizer = cached_tokenizer_from_config(model_config)
        except Exception:
            logger.warning_once(
                "Could not load the tokenizer while resolving Laya marker "
                "metadata; falling back to the values in the model config."
            )

        if mask_token_id is None:
            mask_token_id = getattr(tokenizer, "mask_token_id", None)
        if mask_token_id is None or int(mask_token_id) < 0:
            raise ValueError(
                "Laya needs the id of its [MASK] marker token, but it is neither "
                "present in the model config as `mask_token_id` nor derivable "
                "from the tokenizer."
            )
        self.mask_token_id = int(mask_token_id)

        if not type_token_ids and tokenizer is not None:
            type_token_ids = {
                name: tokenizer(name + TYPE_PROBE_SUFFIX, add_special_tokens=False)[
                    "input_ids"
                ][0]
                for name in QTYPE_NAMES
            }
        self.type_token_ids = self._qtype_map(type_token_ids or {})

    @staticmethod
    def _qtype_map(type_token_ids) -> dict[int, int]:
        name_to_qtype = {name: i for i, name in enumerate(QTYPE_NAMES)}
        return {
            int(token_id): name_to_qtype[name]
            for name, token_id in type_token_ids.items()
            if name in name_to_qtype
        }

    def _qtype_of(self, ids: torch.Tensor) -> int:
        """Question type of a request, from the token that follows ``[CLS]``."""
        if ids.numel() < 2:
            return 0
        return self.type_token_ids.get(int(ids[1]), 0)

    def _answer_temperature(self, qtype: int, num_options: int) -> float:
        """Fitted temperature for this question, as the upstream API picks it."""
        bucket = _temperature_bucket(QTYPE_NAMES[qtype], num_options)
        if bucket in self.temperature_by_options:
            return self.temperature_by_options[bucket]
        if qtype < len(self.temperature):
            return self.temperature[qtype]
        return 1.0

    def _head_forward(self, x: torch.Tensor, seq_lens: list[int]) -> torch.Tensor:
        """Run the decision head's transformer stack on a packed token batch.

        ``nn.TransformerEncoder`` needs a rectangular batch, and padding a step
        of unequal prompts costs twice: the pad rows are real compute (a c=16
        step pads 16 prompts up to the longest one), and the resulting
        key-padding mask drops ``F.scaled_dot_product_attention`` off the fused
        path. The head only ever sees a batch that is already packed --
        ``hidden_states`` is the concatenation of the step's prompts -- so the
        same layers run on it directly with per-sequence attention: no mask, no
        pad rows. Verified equal to the padded path to fp32 rounding.
        """
        for layer in self.head.layers:
            attn = layer.self_attn
            tokens = x.shape[0]
            heads, head_dim = attn.num_heads, attn.head_dim
            h = layer.norm1(x)
            qkv = F.linear(h, attn.in_proj_weight, attn.in_proj_bias)
            query, key, value = qkv.chunk(3, dim=-1)
            query = query.reshape(tokens, heads, head_dim)
            key = key.reshape(tokens, heads, head_dim)
            value = value.reshape(tokens, heads, head_dim)
            if self._fused_attention is not None:
                fused = self._fused_attention(
                    query=query,
                    key=key,
                    value=value,
                    head_num=heads,
                    input_layout="TND",
                    scale=1.0 / math.sqrt(head_dim),
                    actual_seq_qlen=seq_lens,
                    actual_seq_kvlen=seq_lens,
                )[0]
            else:
                fused = _packed_attention(
                    query, key, value, heads, 1.0 / math.sqrt(head_dim), seq_lens
                )
            x = x + layer.dropout1(
                F.linear(
                    fused.reshape(tokens, -1), attn.out_proj.weight, attn.out_proj.bias
                )
            )
            h = layer.norm2(x)
            x = x + layer.dropout2(
                layer.linear2(layer.dropout(layer.activation(layer.linear1(h))))
            )
        return x

    def _marker_positions(self, ids: torch.Tensor, num_tokens: int) -> list[int]:
        return (
            (ids[:num_tokens] == self.mask_token_id).nonzero(as_tuple=True)[0].tolist()
        )

    def _ship_pinned(
        self, name: str, values, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        """Copy host values to the device through a reusable pinned buffer.

        `values` is either a numpy array (written straight into the pinned
        buffer) or a small CPU tensor (memcpy'd into it). Buffers are recycled
        per `name`, but only once the copy that last used them has completed: a
        plan miss can be followed by another one a millisecond later, while the
        copy queued behind a busy encoder has not run yet, so writing the buffer
        blind would ship the *next* step's indices.
        """
        n = int(values.size) if isinstance(values, np.ndarray) else int(values.numel())
        ring = self._plan_host.setdefault(name, [])
        slot = None
        for i, cand in enumerate(ring):
            ready = cand[1] is None or getattr(cand[1], "query", lambda: True)()
            if cand[0].numel() >= n and ready:
                slot = ring.pop(i)
                ring.append(slot)
                break
        if slot is None:
            numel = self._pinned_min_elements
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
        slot[1] = self._record_stream_event()
        return out

    def _record_stream_event(self):
        """Event on the current stream, or None where events are unavailable."""
        try:
            event = torch.npu.Event()
            event.record()
            return event
        except Exception:
            return None

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
            self._plan_cache.move_to_end(key)
            return plan
        batch = len(lengths)
        kmax = max(num_markers)
        # Everything below is assembled with numpy and shipped in three
        # host->device copies. Building the same values as Python lists and
        # handing them to `torch.tensor` costs ~150 ns per element (~3.5 ms for
        # the 23k-element index vector a c16 step needs) and a pageable copy
        # blocks behind the encoder still running on the device.
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
        gather = np.empty(total_cnt + 2 * batch, dtype=np.int64)
        rows_i = np.arange(batch, dtype=np.int64)
        gather[
            np.arange(total_cnt, dtype=np.int64) + np.repeat(2 * rows_i, counts_np)
        ] = slots
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
        # The act head summarises the answer distribution with four numbers, all
        # of them computed by the upstream model from the clamped option count.
        clamped = [max(num, MIN_OPTIONS_FOR_ENTROPY) for num in num_markers]
        floats = [num / OPTION_COUNT_SCALE for num in clamped]
        floats.extend(math.log(num) for num in clamped)
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
            "sizes": [num + ACT_OUTPUTS_PER_REQUEST for num in num_markers],
            "gather": idx_view(0, gather_off, int(gather.size)),
            "use_act": mask_view(0, valid_n, batch),
            "temperature": f_view(0, 2 * batch, batch),
            # Reused by the head: `torch.stack` of four `[batch]` columns costs
            # ~3.3 ms of aclnn tiling on this CANN, a preallocated buffer costs
            # four copies into it.
            "feats": torch.empty(
                (batch, ACT_FEATURE_DIM), dtype=self.head_dtype, device=device
            ),
            "seq_lens": seq_lens,
        }
        while len(self._plan_cache) >= self._plan_cache_limit:
            self._plan_cache.popitem(last=False)
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
            qtypes.append(self._qtype_of(ids))
            markers.append(self._marker_positions(ids, num_tokens))
        has_markers = all(markers)
        # The packed head reads every request straight out of `hidden_states` at
        # the offsets the runner scheduled, so a request that scheduled more
        # tokens than its own prompt holds cannot be read that way and keeps the
        # scalar path.
        packed_ok = sum(lengths) == offsets[len(lengths)]
        if not (has_markers and packed_ok):
            # A prompt without a single `[MASK]` marker is malformed for Laya;
            # keep it on the scalar path so a poisoned request cannot abort the
            # whole engine.
            return self._score_requests_scalar(
                hidden_states, offsets, prompt_token_ids, use_activation
            )

        device = hidden_states.device
        plan = self._step_plan(lengths, markers, qtypes, use_activation, device)
        batch = plan["batch"]
        kmax = plan["kmax"]

        # The step's prompts already sit one after the other in `hidden_states`,
        # so the type embedding is the only per-request work left before the head
        # runs.
        hidden = hidden_states[: offsets[batch]].to(self.head_dtype)
        hidden = hidden + self.type_emb.weight.index_select(0, plan["token_types"])
        encoded = self._head_forward(hidden, plan["seq_lens"])

        picked = encoded.index_select(0, plan["pick"])
        logits = self.scorer(picked.view(batch, kmax, -1)).squeeze(-1)

        # Padding is filled with the upstream sentinel, so its softmax weight is
        # 0 and it drops out of both the entropy sum and the top-2 margin.
        masked = logits.masked_fill(~plan["valid"], ANSWER_MASK_LOGIT)
        probs = torch.softmax(masked, -1)
        entropy = (
            -(probs * torch.log(probs.clamp_min(ENTROPY_EPSILON))).sum(-1)
            / plan["log_counts"]
        )
        # A step whose requests all carry a single `[MASK]` has kmax == 1, where
        # `topk(2)` indexes out of range on Ascend and kills the EngineCore. The
        # scalar path answers that case with margin == top1; mirror it so the two
        # paths agree at kmax == 1.
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
                torch.cat(
                    [encoded.index_select(0, plan["starts"]).float(), feats], dim=-1
                )
            ),
            -1,
        )
        activated = torch.softmax(masked / plan["temperature"].unsqueeze(1), -1)

        # Answer and act head are concatenated for the whole batch at once and
        # handed out as contiguous 1-D views, replacing two kernels and a
        # host-side concat per request.
        chosen = torch.where(plan["use_act"].unsqueeze(1), activated, masked)
        flat = (
            torch.cat([chosen.reshape(-1), act.reshape(-1)])
            .index_select(0, plan["gather"])
            .float()
        )
        return list(flat.split(plan["sizes"]))

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

            qtype = self._qtype_of(ids)
            h = h + self.type_emb.weight[qtype]
            h = self.head(h.unsqueeze(0)).squeeze(0)

            markers = (ids[:num_tokens] == self.mask_token_id).nonzero(as_tuple=True)[0]
            logits = self.scorer(h[markers]).squeeze(-1).float()

            # Act head inputs: the pooled first position plus a detached summary
            # of the answer distribution (top-1, top1-top2 margin, entropy,
            # option count).
            probs = torch.softmax(logits.detach(), -1)
            num_options = max(int(logits.numel()), MIN_OPTIONS_FOR_ENTROPY)
            entropy = -(
                probs * torch.log(probs.clamp_min(ENTROPY_EPSILON))
            ).sum() / math.log(num_options)
            if probs.numel() >= 2:
                top2 = probs.topk(2).values
                top1, margin = top2[0], top2[0] - top2[1]
            else:
                top1 = probs.max() if probs.numel() == 1 else logits.new_zeros(())
                margin = top1
            feats = torch.stack(
                [
                    top1,
                    margin,
                    entropy,
                    logits.new_tensor(num_options / OPTION_COUNT_SCALE),
                ]
            )
            act = torch.softmax(self.act_head(torch.cat([h[0].float(), feats])), -1)

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
                # An all-ones buffer that inference never reads: the fitted
                # temperatures live in the config.
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

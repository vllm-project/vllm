# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Post-hoc Attention Capture for vLLM

Captures and analyzes attention patterns after request completion
with minimal overhead. Attention scores (Q*K) are computed
on the GPU at request-free time with query buffers and delivered via shared memory.

Module-level functions handle stateless operations (encoding, slot
math, attention computation). The AttentionCapture class manages
only per-worker mutable state (Q buffer, capture slots).
"""

import base64
import contextlib
import gzip
import logging
import pickle
import struct
import time
from bisect import bisect_left
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Any, Optional

import regex as re
import torch

logger = logging.getLogger(__name__)

_LAYER_PATTERNS = [
    re.compile(r"(?:^|\.)(?:layers)\.(\d+)(?:\.|$)"),
    re.compile(r"(?:^|\.)(?:h)\.(\d+)(?:\.|$)"),
    re.compile(r"(?:^|\.)(?:blocks)\.(\d+)(?:\.|$)"),
    re.compile(r"(?:^|\.)(?:decoder\.layers)\.(\d+)(?:\.|$)"),
    re.compile(r"(?:^|\.)(?:model\.layers)\.(\d+)(?:\.|$)"),
    re.compile(r"(?:^|\.)(?:transformer\.h)\.(\d+)(?:\.|$)"),
]

_attn_capture: Optional["AttentionCapture"] = None


def set_attn_capture(instance: Optional["AttentionCapture"]) -> None:
    """Set the global AttentionCapture instance."""
    global _attn_capture
    _attn_capture = instance


def get_attn_capture() -> Optional["AttentionCapture"]:
    """Get the global AttentionCapture instance."""
    return _attn_capture


def load_attn_snapshot(req_id: str) -> list[dict[str, Any]] | None:
    """Load attention snapshot(s) via shared memory (cross-process)."""
    return _shm_read(req_id, timeout=30.0)


@dataclass
class CaptureConfig:
    """Configuration for attention capture."""

    enabled: bool = False
    layers: set[int] | None = None


class AttentionCapture:
    """Per-worker attention capture state.

    Instantiated once per ModelRunner. Buffers Q vectors during inference
    and computes attention scores at request completion time.
    """

    def __init__(self, config: CaptureConfig, model_config=None):
        self.config = config
        self.model_config = model_config
        # Ephemeral: (layer_idx, slot_id) -> Q tensor, cleared after capture().
        self.q_buffer: dict[tuple[int, int], torch.Tensor] = {}
        # Persistent across requests while KV slot lives.
        # Allows full [T,H,T] capture with prefix caching (cache hits reuse initial Q).
        # FIFO-capped by _q_cache_max to limit memory.
        self.q_cache: dict[tuple[int, int], torch.Tensor] = {}
        self._q_cache_max: int = 32768  # ~128MB at float16 128d 16h
        self._layer_idx_cache: dict[str, int] = {}
        self.runtime_enabled_this_step = False
        self.capture_slots: set[int] | None = None

    def buffer_query(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        attn_metadata,
        layer_name: str,
    ) -> None:
        """
        Buffer Q tokens at attention-computation time.
        K is NOT buffered — it is read from KV cache at capture time
        (request completion).
        """
        if not self.config.enabled or attn_metadata is None:
            return

        layer_idx = self.extract_layer_idx(layer_name)
        if layer_idx < 0:
            return

        # Extract slot_ids using attn_metadata from the gpu_model_runner process
        slot_ids = attn_metadata.slot_mapping
        if query.shape[0] != slot_ids.shape[0]:
            return

        # Detaching the query tensor for buffering. Severing CUDA computation trace
        # Query tensor stays in the dict for all requests
        try:
            query_cpu = query.detach().cpu().clone()
        except Exception:
            return  # This error may occur due to gpu memory access issues

        capture_slots = self.capture_slots
        for i in range(query.shape[0]):
            slot_id = slot_ids[i].item()

            # Buffer queries for only set requests at _update_attn_capture_slots()
            if slot_id < 0 or (capture_slots and slot_id not in capture_slots):
                continue

            # cast to float16 for uniform buffer dtype; no-op if already float16
            self.q_buffer[(layer_idx, slot_id)] = query_cpu[i].to(torch.float16)

    def capture(
        self,
        req_state,
        block_size: int,
        kv_caches: list,
        prefix: str | None = None,
    ) -> None:
        """Capture attention scores for a completed request.

        Called after request completion. Computes Q*K attention on GPU
        and writes results to shared memory.
        """
        req_id = req_state.req_id
        layers = resolve_target_layers(req_state, self.config.layers)
        snapshots: list[dict] = []

        for layer_idx in layers:
            # Collect slots that have Q in either the ephemeral buffer or
            # the persistent q_cache (for prefix-cached tokens).
            buf_slots = [sid for (li, sid) in self.q_buffer if li == layer_idx]
            cached_slots = [sid for (li, sid) in self.q_cache if li == layer_idx]
            q_slots = set(buf_slots) | set(cached_slots)
            if not q_slots or not req_state.block_ids:
                continue
            # Find which block group contains buffered slots for this layer
            grp_idx = None
            for gi, block_list in enumerate(req_state.block_ids):
                if not block_list:
                    continue
                if q_slots & slots_from_blocks(block_list, block_size):
                    grp_idx = gi
                    break
            if grp_idx is None:
                continue

            # From the current request block group,
            # order the slot_ids for alignment with QK^T and deduplication.
            slots = ordered_slots_for_group(
                req_state.block_ids[grp_idx], req_state.num_tokens, block_size
            )
            if not slots:
                continue
            # Collect Q tensors from buffer (falling back to q_cache for
            # prefix-cached tokens that were not re-computed this request).
            q_list, q_sids, tok_idx = self._collect_query(layer_idx, slots)
            if not q_list:
                continue

            # Persist Q vectors so future requests can reuse them on cache hit.
            # Evict oldest entries if cap is reached (simple FIFO approximation).
            for sid, q in zip(q_sids, q_list):
                if len(self.q_cache) >= self._q_cache_max:
                    self.q_cache.pop(next(iter(self.q_cache)))
                self.q_cache[(layer_idx, sid)] = q

            # NOTE(jehyun): This class receives the KV Cache from the worker.
            # Reads the KV Cache right after the request is finished
            # before the KV cache is freed.
            # Extract K for ALL slots (not just q_sids) so the key axis covers
            # the full sequence, including prefix-cached tokens.
            kv_idx = layer_idx if layer_idx < len(kv_caches) else grp_idx
            if not kv_caches or kv_idx is None or kv_idx >= len(kv_caches):
                continue
            k_raw = extract_k_from_kv_cache(kv_caches[kv_idx], slots)
            k_list = [k_raw[i] for i in range(k_raw.shape[0])]
            if not k_list:
                continue

            # Build tok_idx covering all slots; q_list entries map to q_sids
            # positions within the full slot list for the QK^T alignment.
            sid_to_pos = {sid: pos for pos, sid in enumerate(slots)}
            q_tok_idx = [sid_to_pos[sid] for sid in q_sids]

            # Checking for alignment before QK Calculation for safety issues
            q_tok_idx, q_list, _ = self._filter_compatible_qk(
                q_tok_idx, q_list, [k_list[i] for i in q_tok_idx]
            )
            if not q_list:
                continue
            tok_idx = q_tok_idx

            # Trim K to the same token positions as Q so shapes always match.
            k_list = [k_list[i] for i in q_tok_idx]
            # Compute Q*K^T: Q=[T, H, d], K=[T, H, d] → [T, H, T]
            q_t, k_t = torch.stack(q_list), torch.stack(k_list)
            if k_t.is_cuda and not q_t.is_cuda:
                q_t = q_t.to(k_t.device)
            scale = 1.0 / (q_t.shape[2] ** 0.5)
            attn = compute_qk_attention(q_t, k_t, scale)
            if attn is None:
                continue

            # Apply prefix slice
            if prefix:
                parts = prefix.split(":")
                q_start = int(parts[0]) if parts[0] else 0
                q_end = int(parts[1]) if len(parts) > 1 and parts[1] else None
                attn = attn[q_start:q_end, :, :]
                tok_idx = tok_idx[q_start:q_end]

            # Create meta_data for tokens, later used by clients for 1:1 matching
            # between attn_scores and token_idx
            tmeta = build_token_meta(req_state, tok_idx, ordered_slots_len=len(slots))
            snapshots.append(encode_snapshot(attn, layer_idx, tmeta))

            # Remove processed slots from q_buffer to prevent
            # cross-request contamination.
            for sid in set(slots):
                self.q_buffer.pop((layer_idx, sid), None)

        # NOTE(jehyun): Write snapshots to shared memory for cross-process delivery.
        # The output_processor (main process) reads them via _shm_read().
        if snapshots:
            _shm_write(req_id, snapshots)

    def cleanup_request_buffers(
        self,
        block_ids: list[list[int]],
        block_size: int,
    ) -> None:
        """
        Remove buffered Q vectors for a finished request.
        Called for ALL requests to prevent stale data leaking into future requests.
        q_cache is evicted only when the underlying KV block is reclaimed.
        """
        if not block_ids:
            return
        remove: set[int] = set()
        for block_list in block_ids:
            remove |= slots_from_blocks(block_list, block_size)
        if self.q_buffer:
            for k in [k for k in self.q_buffer if k[1] in remove]:
                del self.q_buffer[k]

    def extract_layer_idx(self, layer_name: str) -> int:
        """Parse layer index from attention layer name, with caching."""
        v = self._layer_idx_cache.get(layer_name)
        if v is not None:
            return v

        for pat in _LAYER_PATTERNS:
            m = pat.search(layer_name)
            if not m:
                continue
            idx = int(m.group(1))
            self._layer_idx_cache[layer_name] = idx
            return idx

        self._layer_idx_cache[layer_name] = -1
        return -1

    def _collect_query(
        self,
        layer_idx: int,
        slots: list[int],
    ) -> tuple[list[torch.Tensor], list[int], list[int]]:
        """
        Collect Q tensors from buffer in deterministic slot order.
        Falls back to q_cache for prefix-cached tokens whose Q was not
        re-computed this request (prefix caching ON).
        Returns (q_list, q_slot_ids, tok_idx).
        """
        q_list, q_sids, tok_idx = [], [], []
        for si, sid in enumerate(slots):
            key = (layer_idx, sid)
            q = self.q_buffer.get(key)
            if q is None:
                q = self.q_cache.get(key)
            if q is None:
                continue
            q_list.append(q)
            q_sids.append(sid)
            tok_idx.append(si)
        return q_list, q_sids, tok_idx

    @staticmethod
    def _filter_compatible_qk(
        tok_idx: list[int],
        q_list: list[torch.Tensor],
        k_list: list[torch.Tensor],
    ) -> tuple[list[int], list[torch.Tensor], list[torch.Tensor]]:
        """Filter Q/K pairs to those with matching shape and device."""
        if not q_list or not k_list:
            return [], [], []
        q0, k0 = q_list[0], k_list[0]
        triples = [
            (idx, q, k)
            for idx, q, k in zip(tok_idx, q_list, k_list)
            if _sanity_check(q, q0) and _sanity_check(k, k0)
        ]
        if not triples:
            return [], [], []
        return (
            [t[0] for t in triples],
            [t[1] for t in triples],
            [t[2] for t in triples],
        )


def slots_from_blocks(block_list: list[int], block_size: int) -> set[int]:
    """Return the set of all slot IDs covered by the given blocks."""
    slots: set[int] = set()
    for bid in block_list:
        base = bid * block_size
        for off in range(block_size):
            slots.add(base + off)
    return slots


def ordered_slots_for_group(
    block_list: list[int],
    num_tokens: int,
    block_size: int,
) -> list[int]:
    """Build deduplicated, token-order slot list from a block group."""
    ordered: list[int] = []
    seen: set[int] = set()
    cursor = 0
    for block_id in block_list:
        if cursor >= num_tokens:
            break
        base = block_id * block_size
        count = min(block_size, num_tokens - cursor)
        for off in range(count):
            slot_id = base + off
            if slot_id not in seen:
                ordered.append(slot_id)
                seen.add(slot_id)
        cursor += count
    return ordered


def resolve_target_layers(
    req_state,
    default_layers: set[int] | None,
) -> set[int]:
    """Determine target layers: per-request override or server default."""
    target = default_layers or set()
    if req_state.sampling_params and req_state.sampling_params.extra_args:
        layers_str = req_state.sampling_params.extra_args.get("attn_capture_layers")
        if layers_str and layers_str.strip().lower() != "all":
            target = {int(x.strip()) for x in layers_str.split(",")}
    return target


def extract_k_from_kv_cache(
    kv_cache: torch.Tensor,
    slot_ids: list[int],
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """
    Extract K(key) vectors from paged KV cache at given slot positions.

    Backend-agnostic: auto-detects layout from tensor shape.
      - FlashInfer:    [num_blocks, 2, page_size, num_kv_heads, head_dim]
      - FlashAttention:[2, num_blocks, page_size, num_kv_heads, head_dim]
    Returns: Tensor of shape [len(slot_ids), num_kv_heads, head_dim]
    """
    shape = kv_cache.shape
    slot_tensor = torch.tensor(slot_ids, dtype=torch.long, device=kv_cache.device)

    if kv_cache.ndim == 5 and shape[0] == 2:
        page_size, num_slots = shape[2], shape[1] * shape[2]
        k_flat = kv_cache[0].reshape(num_slots, -1)
        k = k_flat[slot_tensor].view(len(slot_ids), shape[3], shape[4])
    elif kv_cache.ndim == 5 and shape[1] == 2:
        page_size = shape[2]
        page_indices = slot_tensor // page_size
        page_offsets = slot_tensor % page_size
        k = kv_cache[page_indices, 0, page_offsets]
    elif kv_cache.ndim == 3 and shape[0] == 2:
        k = kv_cache[0, slot_tensor]
    else:
        raise ValueError(
            f"Unsupported KV cache layout: ndim={kv_cache.ndim} shape={list(shape)}"
        )

    return k.to(dtype)  # NOTE(jehyun): Uniform dtype for downstream float16 computation


def compute_qk_attention(
    q_tensor: torch.Tensor,
    k_tensor: torch.Tensor,
    scale: float,
) -> torch.Tensor | None:
    """
    Compute scaled dot-product attention probabilities.
    Handles GQA by expanding K heads to match Q heads.
    Args:
        q_tensor: [T, num_q_heads, head_dim]
        k_tensor: [T, num_kv_heads, head_dim]
        scale: 1/sqrt(head_dim)
    Returns: [T, num_q_heads, T] attention probabilities, or None on mismatch.
    """
    q = q_tensor.transpose(0, 1)  # [hq, T, d]
    k = k_tensor.transpose(0, 1)  # [hk, T, d]
    hq, Tq, d = q.shape
    hk, Tk, dk = k.shape
    if d != dk:
        return None

    if hk == hq:
        k_m = k
    else:
        # GQA/MQA: repeat each KV head to cover the corresponding Q heads
        k_m = (
            k.repeat_interleave(hq // hk, dim=0)
            if hq % hk == 0
            else k.index_select(
                0,
                torch.clamp(
                    torch.floor(torch.arange(hq, device=k.device) * (hk / hq)).long(),
                    0,
                    hk - 1,
                ),
            )
        )

    scores = torch.bmm(q, k_m.transpose(-2, -1)) * scale  # [hq, T, T]
    causal_mask = torch.ones(Tq, Tk, device=scores.device, dtype=torch.bool).tril()
    scores = scores.masked_fill(~causal_mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return probs.transpose(0, 1)  # [T, hq, T]


def build_token_meta(
    req_state,
    token_idx: list[int],
    *,
    ordered_slots_len: int | None = None,
) -> dict[str, Any]:
    """Build token mapping metadata for post-hoc client-side alignment."""
    prompt_len = int(getattr(req_state, "num_prompt_tokens", 0) or 0)
    # num_tokens already includes the token sampled in the *current* step
    # (appended before _update_states runs), so subtract 1 to get the actual
    # sequence length seen by the attention kernel.
    total_len = int(getattr(req_state, "num_tokens", prompt_len) or prompt_len) - 1
    if prompt_len > total_len:
        prompt_len = total_len

    # Extract multi-modal offsets with (s1,e1),(s2,e2),(s3,e3)...
    raw_vision: list[dict[str, int]] = []
    mm_features = getattr(req_state, "mm_features", None) or []
    for feature in mm_features:
        pos = getattr(feature, "mm_position", None)
        if pos is None:
            continue
        start = int(getattr(pos, "offset", 0) or 0)
        length = int(getattr(pos, "length", 0) or 0)
        end = start + max(length, 0)
        if end <= 0 or start >= prompt_len:
            continue
        start, end = max(0, start), min(prompt_len, end)
        if start < end:
            raw_vision.append({"start": start, "end": end})

    # Sort and merge overlapping vision ranges
    raw_vision.sort(key=lambda r: (r["start"], r["end"]))
    vision_ranges: list[dict[str, int]] = []
    for r in raw_vision:
        if not vision_ranges or r["start"] > vision_ranges[-1]["end"]:
            vision_ranges.append(dict(r))
        else:
            vision_ranges[-1]["end"] = max(vision_ranges[-1]["end"], r["end"])

    # Complement: language ranges fill the gaps between vision ranges.
    # prompt: [-- lang0 --][-- vis0 --][-- lang1 --][-- vis1 --] ...
    #                      ^           ^
    #             l.end == v.start   v.end == l.start
    lang_ranges: list[dict[str, int]] = []
    cursor = 0
    for r in vision_ranges:
        if cursor < r["start"]:
            lang_ranges.append({"start": cursor, "end": r["start"]})
        cursor = max(cursor, r["end"])
    if cursor < prompt_len:
        lang_ranges.append({"start": cursor, "end": prompt_len})

    ordered_len = int(
        ordered_slots_len if ordered_slots_len is not None else len(token_idx)
    )
    window_offset = int(total_len - ordered_len)

    # Calculate prompt boundaries and prompt offsets.
    pb_local = bisect_left(token_idx, prompt_len) if token_idx else None
    idx_shifted = [int(i) + window_offset for i in token_idx]
    pb_offset = bisect_left(idx_shifted, prompt_len) if idx_shifted else None

    return {
        "token_idx": [int(i) for i in token_idx],
        "prompt_len": prompt_len,
        "total_len": total_len,
        "vision_ranges": vision_ranges,
        "lang_ranges": lang_ranges,
        "token_idx_basis": "window_local",
        "win_offset": window_offset,
        "pb_local": pb_local,
        "pb_offset": pb_offset,
    }


def encode_snapshot(
    attn: torch.Tensor,
    layer_idx: int,
    token_meta: dict[str, Any],
) -> dict[str, Any]:
    """Encode attention tensor to gzip+base64 wire format."""
    attn = attn.cpu()
    compressed = gzip.compress(attn.numpy().tobytes())
    return {
        "data": base64.b64encode(compressed).decode("utf-8"),
        "shape": list(attn.shape),
        "dtype": str(attn.dtype),
        "layer_idx": layer_idx,
        "token_meta": token_meta,
    }


def _shm_name(req_id: str) -> str:
    """Deterministic shared-memory segment name from request ID."""
    return "/vkv_" + req_id.replace("-", "")[:40]


def _shm_write(req_id: str, snapshots: list[dict]) -> None:
    """
    Write snapshot list to a named shared-memory segment.
    Protocol: size header is written LAST so readers treat size==0
    as "write in progress" and keep polling.
    """
    payload = pickle.dumps(snapshots)
    size, name = len(payload), _shm_name(req_id)
    with contextlib.suppress(FileNotFoundError):
        stale = shared_memory.SharedMemory(name=name, create=False)
        stale.close()
        stale.unlink()
    mem = shared_memory.SharedMemory(name=name, create=True, size=8 + size)
    mem.buf[8 : 8 + size] = payload
    struct.pack_into("Q", mem.buf, 0, size)  # size LAST (ready signal)
    mem.close()


def _sanity_check(t: torch.Tensor, ref: torch.Tensor) -> bool:
    return (
        isinstance(t, torch.Tensor) and t.shape == ref.shape and t.device == ref.device
    )


def _shm_read(req_id: str, timeout: float = 5.0) -> list[dict] | None:
    """Read snapshot list from shared-memory, polling until available."""
    name, deadline = _shm_name(req_id), time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            mem = shared_memory.SharedMemory(name=name, create=False)
        except FileNotFoundError:
            time.sleep(0.05)
            continue
        size = struct.unpack_from("Q", mem.buf, 0)[0]
        if size == 0:
            mem.close()
            time.sleep(0.01)
            continue
        try:
            data = pickle.loads(bytes(mem.buf[8 : 8 + size]))
        except Exception:
            mem.close()
            time.sleep(0.01)
            continue
        mem.close()
        mem.unlink()
        return data
    with contextlib.suppress(FileNotFoundError):
        mem = shared_memory.SharedMemory(name=name, create=False)
        mem.close()
        mem.unlink()
    return None

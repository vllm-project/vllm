# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Shared utilities for Confident Decoding (entropy-trough selection).

Provides:
- ``read_trough_config``: read trough parameters from vllm_config.
- ``compute_trough_layer_range``: compute candidate layer range for a model.
- ``TroughStateMixin``: base mixin adding trough-related state and methods.
- ``vectorized_entropy_select``: memory-efficient entropy-trough layer selection.
"""

import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = __import__("vllm.logger", fromlist=["logger"]).logger


def read_trough_config(vllm_config: "VllmConfig") -> dict:
    """Read trough decoding config from vllm_config.additional_config
    or vllm_config.model_config.hf_overrides."""
    additional_config = getattr(vllm_config, "additional_config", {}) or {}
    hf_overrides = getattr(vllm_config.model_config, "hf_overrides", {}) or {}

    def _cfg(key: str, default):
        if key in additional_config:
            return additional_config[key]
        if isinstance(hf_overrides, dict) and key in hf_overrides:
            return hf_overrides[key]
        return default

    return {
        "enable_trough_decoding": bool(_cfg("enable_multi_layer_entropy_selection", False)),
        "trough_max_backtrack_layers": int(_cfg("trough_max_backtrack_layers", 0)),
        "trough_backtrack_ratio": float(_cfg("trough_backtrack_ratio", 0.0)),
        "trough_select_method": str(_cfg("select_method", "trough")),
        "trough_p": float(_cfg("p", 1.0)),
        "trough_log_interval": int(_cfg("trough_log_interval", 0)),
    }


def compute_trough_layer_range(num_layers: int, config: dict) -> tuple[int, int]:
    """Compute trough start layer and candidate layer count.

    Returns (trough_start_layer, candidate_layers).
    """
    max_backtrack = config["trough_max_backtrack_layers"]
    backtrack_ratio = config["trough_backtrack_ratio"]

    if max_backtrack > 0:
        candidate_layers = min(num_layers, max_backtrack)
    elif backtrack_ratio > 0:
        candidate_layers = max(1, int(math.ceil(num_layers * backtrack_ratio)))
    else:
        candidate_layers = num_layers

    start_layer = num_layers - candidate_layers
    return start_layer, candidate_layers


class TroughStateMixin:
    """Adds trough decoding state and logic to a CausalLM wrapper.

    Subclasses must provide:
    - ``model`` attribute (inner model, may be TroughModel variant).
    - ``lm_head`` attribute.
    - ``logits_processor`` attribute.
    - ``_trough_start_layer`` on the inner model (set by TroughModel).
    - ``compute_logits_override()`` class method that returns a reference
      implementation of logits computation with trough selection.
    """

    def init_trough_state(self, config: dict) -> None:
        self.trough_max_backtrack_layers = config["trough_max_backtrack_layers"]
        self.trough_backtrack_ratio = config["trough_backtrack_ratio"]
        self.trough_select_method = config["trough_select_method"]
        self.trough_p = config["trough_p"]
        self.trough_log_interval = config["trough_log_interval"]
        self._trough_call_count = 0
        self._trough_buffers: dict[int, torch.Tensor] = {}
        self._last_eager_buf: torch.Tensor | None = None
        self._last_seq_len: int = 0
        self._last_logits_indices: torch.Tensor | None = None

    def init_trough_buffer(
        self,
        output: torch.Tensor,
        trough_states: list[torch.Tensor],
    ) -> None:
        """Apply final norm to collected trough states and store in buffer."""
        if not trough_states:
            return
        normed_layers = [self.model.norm(hs, None) for hs in trough_states]
        self._last_eager_buf = torch.stack(normed_layers)
        self._trough_buffers[output.shape[0]] = self._last_eager_buf

    def compute_trough_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        """Compute logits using entropy-trough layer selection.

        Returns ``None`` if trough is disabled or no buffer is available.
        """
        raise NotImplementedError


def clear_trough_step_state(model) -> None:
    """Reset per-step trough metadata after compute_logits."""
    model._last_logits_indices = None
    model._last_seq_len = 0


def prepare_trough_layer_states(
    *,
    trough_buffers: dict[int, torch.Tensor],
    last_seq_len: int,
    last_logits_indices: torch.Tensor | None,
    sample_batch_size: int,
) -> tuple[torch.Tensor | None, str | None]:
    """Resolve ``[L, B, H]`` layer states aligned with the sample batch.

    Returns ``(layer_states, skip_reason)``. When ``skip_reason`` is set,
    callers must fall back to final-layer logits instead of trough selection.
    """
    if last_seq_len <= 0:
        return None, "invalid_last_seq_len"

    layer_states = trough_buffers.get(last_seq_len)
    if layer_states is None:
        return None, "buffer_miss"

    _, S_buf, _ = layer_states.shape
    B = sample_batch_size

    if last_logits_indices is not None:
        if last_logits_indices.numel() != B:
            return None, "indices_count_mismatch"
        if last_logits_indices.numel() == 0:
            return None, "empty_indices"
        max_idx = int(last_logits_indices.max().item())
        min_idx = int(last_logits_indices.min().item())
        if min_idx < 0 or max_idx >= S_buf:
            return None, "indices_out_of_range"
        return layer_states[:, last_logits_indices], None

    if B == S_buf:
        return layer_states, None

    # Do not guess with ``[:, -B:]`` — that misaligns prompt-logprobs and
    # padded batches when sample rows are not the trailing S_buf positions.
    return None, "missing_logits_indices"


def finalize_trough_step_buffers(model) -> None:
    """Drop eager trough buffers and clear step metadata."""
    seq_len = model._last_seq_len
    captured = model._trough_captured_shapes
    if seq_len not in captured and seq_len in model._trough_buffers:
        model._trough_buffers.pop(seq_len, None)
    clear_trough_step_state(model)


def _resolve_lm_head(model):
    lm_head = getattr(model, "lm_head", None)
    if lm_head is not None:
        return lm_head
    return model.model.embed_tokens


def compute_confident_decoding_logits(
    model,
    hidden_states: torch.Tensor,
) -> torch.Tensor | None:
    """Shared Confident Decoding logits path for CausalLM wrappers."""
    lm_head = _resolve_lm_head(model)
    model._trough_call_count += 1
    B = hidden_states.shape[0]

    layer_states, fallback_reason = prepare_trough_layer_states(
        trough_buffers=model._trough_buffers,
        last_seq_len=model._last_seq_len,
        last_logits_indices=getattr(model, "_last_logits_indices", None),
        sample_batch_size=B,
    )
    if layer_states is None:
        if fallback_reason not in (None, "buffer_miss"):
            logger.debug(
                "Confident Decoding fallback (%s): using final-layer logits",
                fallback_reason,
            )
        clear_trough_step_state(model)
        return model.logits_processor(lm_head, hidden_states)

    inner = model.model
    selected_logits, _, _, _ = vectorized_entropy_select(
        layer_states=layer_states,
        fallback_hidden_states=hidden_states,
        logits_processor=model.logits_processor,
        lm_head=lm_head,
        select_method=model.trough_select_method,
        trough_p=model.trough_p,
        trough_max_backtrack_layers=model.trough_max_backtrack_layers,
        trough_backtrack_ratio=model.trough_backtrack_ratio,
        trough_start_layer=inner._trough_start_layer,
        total_model_layers=len(inner.layers),
        trough_log_interval=model.trough_log_interval,
        trough_call_count=model._trough_call_count,
    )
    finalize_trough_step_buffers(model)
    return selected_logits


def vectorized_entropy_select(
    layer_states: torch.Tensor,
    fallback_hidden_states: torch.Tensor,
    logits_processor,
    lm_head,
    select_method: str,
    trough_p: float,
    trough_max_backtrack_layers: int,
    trough_backtrack_ratio: float,
    trough_start_layer: int,
    total_model_layers: int,
    trough_log_interval: int,
    trough_call_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Compute entropy trough layer selection.

    Memory strategy (no extra ``[L, B, V]`` softmax buffer):

    1. Stable softmax **in-place** on ``all_logits`` (only ``row_max`` / ``Z``
       scalars per row are kept).
    2. Entropy per layer via ``torch.special.entr`` on ``[B, V]`` slices so
       peak temp is one layer, not ``L × B × V``.
    3. Rebuild raw logits in-place with ``log(p) + log(Z) + max`` before
       ``gather`` — never ``exp(probs)`` and never ``(p * p.log_())``.

    Args:
        layer_states: ``[L, B, H]`` normed hidden states per candidate layer.
        fallback_hidden_states: ``[B, H]`` last-layer hidden states fallback.
        logits_processor: LogitsProcessor instance.
        lm_head: language model head (ParallelLMHead or equivalent).
        select_method: selection strategy (``trough``, ``last-m1``, etc.).
        trough_p: stochastic fallback probability to final layer.
        trough_max_backtrack_layers: explicit max backtrack (0=use ratio).
        trough_backtrack_ratio: ratio-based backtrack window.
        trough_start_layer: global model layer index of first candidate.
        total_model_layers: total number of layers in the model.
        trough_log_interval: log period (0=disabled).
        trough_call_count: current step count for logging.

    Returns:
        Tuple of (selected_logits ``[B, V]``, entropy ``[L, B]``,
        layer_states ``[L, B, V]``, L).
    """
    L, B, H = layer_states.shape
    device = layer_states.device

    flat = layer_states.reshape(L * B, H)
    flat_logits = logits_processor(lm_head, flat)
    if flat_logits is None:
        flat_logits = logits_processor(lm_head, fallback_hidden_states)
        return flat_logits, torch.zeros(L, B, device=device), torch.zeros(
            L, B, V, device=device
        ) if (V := flat_logits.shape[-1]) else torch.zeros(L, B, 1, device=device), L
    V = flat_logits.shape[-1]
    all_logits = flat_logits.reshape(L, B, V)

    method = select_method

    if method.startswith("last-"):
        try:
            offset = int(method.split("-m")[-1])
        except (IndexError, ValueError):
            offset = 0
        target_model_layer = max(0, total_model_layers - 1 - offset)
        cand_idx = target_model_layer - trough_start_layer
        cand_idx = max(0, min(L - 1, cand_idx))
        selected_layer_idx = torch.full(
            (B,), cand_idx, device=device, dtype=torch.long
        )
        entropy = torch.zeros(L, B, device=device)
    else:
        # In-place stable softmax; keep row_max and log(Z) for logits rebuild.
        row_max = all_logits.max(dim=-1, keepdim=True).values
        all_logits.sub_(row_max)
        all_logits.exp_()
        Z = all_logits.sum(dim=-1, keepdim=True)
        all_logits.div_(Z)
        log_Z = Z.log()

        # Layer-wise entropy: peak temp is [B, V], not [L, B, V].
        entropy = torch.empty(L, B, device=device, dtype=all_logits.dtype)
        for l_idx in range(L):
            entropy[l_idx] = torch.special.entr(all_logits[l_idx]).sum(dim=-1)

        explicit = int(trough_max_backtrack_layers)
        if explicit > 0:
            max_backtrack = explicit
        elif explicit < 0:
            max_backtrack = L
        else:
            max_backtrack = int(L * trough_backtrack_ratio)
        # Cap backtrack to L-1 so the loop never uses negative layer indices
        # (e.g. max_backtrack=10, L=10 previously yielded min_layer=-1).
        max_backtrack = min(max_backtrack, max(L - 1, 0))
        min_layer = max(0, L - 1 - max_backtrack)

        selected_layer_idx = torch.full(
            (B,), L - 1, device=device, dtype=torch.long
        )
        frozen = torch.zeros(B, dtype=torch.bool, device=device)
        prev_entropy = entropy[L - 1]

        for l_idx in range(L - 2, min_layer - 1, -1):
            cur_entropy = entropy[l_idx]
            improves = cur_entropy < prev_entropy
            update_mask = improves & (~frozen)
            selected_layer_idx = torch.where(
                update_mask,
                torch.full_like(selected_layer_idx, l_idx),
                selected_layer_idx,
            )
            frozen = frozen | (~improves)
            prev_entropy = cur_entropy

        # Rebuild logits: log(p) + log(Z) + max = raw logits (probs in buffer).
        all_logits.log_()
        all_logits.add_(log_Z)
        all_logits.add_(row_max)

        if method == "trough-m2":
            selected_layer_idx = torch.clamp(selected_layer_idx - 2, 0, L - 1)
        elif method == "trough-m1":
            selected_layer_idx = torch.clamp(selected_layer_idx - 1, 0, L - 1)
        elif method == "trough-p1":
            selected_layer_idx = torch.clamp(selected_layer_idx + 1, 0, L - 1)
        elif method == "trough-p2":
            selected_layer_idx = torch.clamp(selected_layer_idx + 2, 0, L - 1)

    p = float(trough_p)
    if p < 1.0:
        rng = torch.rand(B, device=device)
        use_final = rng > p
        selected_layer_idx = torch.where(
            use_final,
            torch.full((B,), L - 1, device=device, dtype=torch.long),
            selected_layer_idx,
        )

    gather_idx = selected_layer_idx.unsqueeze(0).unsqueeze(-1).expand(1, B, V)
    selected_logits = all_logits.gather(0, gather_idx).squeeze(0)

    if B > 0 and trough_log_interval > 0 and trough_call_count % trough_log_interval == 0:
        with torch.no_grad():
            sel = selected_layer_idx
            backtrack_depth = (L - 1) - sel
            num_at_final = (sel == (L - 1)).sum().item()
            preview = min(B, 4)
            if method.startswith("trough"):
                final_entropy = entropy[L - 1]
                sample_pairs = [
                    (int(sel[i].item()), float(final_entropy[i].item()))
                    for i in range(preview)
                ]
            else:
                sample_pairs = [(int(sel[i].item()), 0.0) for i in range(preview)]
            logger.info(
                "[trough-decoding] step=%d tokens=%d layers=%d "
                "select_method=%s p=%.2f "
                "avg_selected_layer=%.2f min_selected_layer=%d "
                "avg_backtrack_depth=%.2f max_backtrack_depth=%d "
                "tokens_kept_at_final=%d/%d sample=%s",
                trough_call_count,
                B,
                L,
                method,
                p,
                sel.float().mean().item(),
                int(sel.min().item()),
                backtrack_depth.float().mean().item(),
                int(backtrack_depth.max().item()),
                num_at_final,
                B,
                sample_pairs,
            )

    return selected_logits, entropy, all_logits, L

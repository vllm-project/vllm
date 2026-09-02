# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
from typing import TYPE_CHECKING, TypedDict

import torch
import torch.nn as nn

import vllm.envs as envs

if TYPE_CHECKING:
    from vllm.config import VllmConfig

from .heads import ProbeHead, SingProbeAttnModel
from .loader import load_probe_head
from .paged_attention import MAX_SPLITS, probe_paged_attention, store_probe_kv
from .probe_kernels import tap_into


class TokenProbeForwardContext(TypedDict):
    capture: bool
    score: bool
    block_table: torch.Tensor | None
    slot_mapping: torch.Tensor | None
    query_start_loc: torch.Tensor | None
    max_query_len: int


class TokenProbe(nn.Module):
    def __init__(
        self,
        *,
        ckpt_path: str | None,
        hidden_size: int,
        dtype: torch.dtype,
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__()
        self.logger = logger or logging.getLogger(__name__)
        self.hidden_size = hidden_size
        self.overlap = envs.VLLM_ENABLE_TOKEN_PROBE_OVERLAP
        self.probe_head: ProbeHead | None = None
        self.tap_slots: dict[int, int] = {}
        self._side_stream: torch.cuda.Stream | None = None
        self._pending_event: torch.cuda.Event | None = None
        self._pending_scores: torch.Tensor | None = None
        self._output_copy_event: torch.cuda.Event | None = None
        self._features: torch.Tensor | None = None
        self._captured = 0
        self._capture_enabled = True
        self._score_enabled = True
        self._positions: torch.Tensor | None = None
        self._block_table: torch.Tensor | None = None
        self._slot_mapping: torch.Tensor | None = None
        self._query_start_loc: torch.Tensor | None = None
        self._max_query_len = 1
        self._launched = False
        self.kv_cache: torch.Tensor | None
        self.register_buffer("kv_cache", None, persistent=False)
        self.kv_block_size = 0

        if ckpt_path is None:
            return
        head = load_probe_head(ckpt_path, dtype=dtype)
        assert isinstance(head, ProbeHead)
        if head.hidden_size is not None and head.hidden_size != hidden_size:
            raise ValueError(
                f"token probe hidden size {head.hidden_size} does not match "
                f"model hidden size {hidden_size}"
            )
        if not head.state_indices:
            raise ValueError("token probe must tap at least one model layer")
        self.probe_head = head
        self.tap_slots = {
            layer_id: slot for slot, layer_id in enumerate(head.state_indices)
        }
        self.logger.info(
            "Loaded vLLM token probe (%s) from %s",
            type(head).__name__,
            ckpt_path,
        )

    @classmethod
    def from_config(cls, vllm_config: "VllmConfig") -> "TokenProbe":
        model_config = vllm_config.model_config
        return cls(
            ckpt_path=model_config.probe_ckpt,
            hidden_size=model_config.get_hidden_size(),
            dtype=model_config.dtype,
        )

    @property
    def enabled(self) -> bool:
        return self.probe_head is not None

    @property
    def uses_kv_cache(self) -> bool:
        return isinstance(self.probe_head, SingProbeAttnModel)

    @property
    def label_names(self) -> tuple[str, ...]:
        if self.probe_head is None:
            return ()
        return self.probe_head.label_names

    @property
    def output_copy_event(self) -> torch.cuda.Event | None:
        return self._output_copy_event

    def initialize_kv_cache(self, num_blocks: int, block_size: int) -> None:
        if not self.uses_kv_cache:
            return
        assert isinstance(self.probe_head, SingProbeAttnModel)
        num_slots = num_blocks * block_size
        expected_shape = (num_slots, 2 * self.probe_head.kv_dim)
        if self.kv_cache is not None and self.kv_cache.shape == expected_shape:
            return
        device = self.probe_head.proj_qkv.weight.device
        dtype = self.probe_head.proj_qkv.weight.dtype
        self.kv_cache = torch.zeros(expected_shape, device=device, dtype=dtype)
        self.kv_block_size = block_size
        self.logger.info(
            "Token probe KV cache: %d slots x %d values (%.2f GiB)",
            num_slots,
            expected_shape[1],
            self.kv_cache.numel() * self.kv_cache.element_size() / 2**30,
        )
        self._warmup_attention()

    def _warmup_attention(self) -> None:
        assert isinstance(self.probe_head, SingProbeAttnModel)
        assert self.kv_cache is not None
        device = self.kv_cache.device
        block_table = torch.zeros(1, 1, dtype=torch.int32, device=device)

        split_counts = [1]
        while split_counts[-1] < MAX_SPLITS:
            split_counts.append(split_counts[-1] * 2)
        for max_query_len in (1, 64):
            query = torch.zeros(
                max_query_len,
                self.probe_head.q_dim,
                dtype=self.kv_cache.dtype,
                device=device,
            )
            positions = torch.zeros(
                max_query_len,
                dtype=torch.int64,
                device=device,
            )
            query_start_loc = torch.tensor(
                [0, max_query_len],
                dtype=torch.int32,
                device=device,
            )
            for splits in split_counts:
                probe_paged_attention(
                    query=query,
                    kv_cache=self.kv_cache,
                    block_table=block_table,
                    positions=positions,
                    query_start_loc=query_start_loc,
                    max_query_len=max_query_len,
                    num_heads=self.probe_head.num_attention_heads,
                    head_dim=self.probe_head.head_dim,
                    block_size=self.kv_block_size,
                    window=self.probe_head.sliding_window,
                    force_splits=splits,
                )

    def initialize_runtime(self, async_output: bool) -> None:
        if not self.enabled or not torch.cuda.is_available():
            return
        device = (
            self.kv_cache.device
            if self.kv_cache is not None
            else torch.device("cuda", torch.accelerator.current_device_index())
        )
        if async_output and self._output_copy_event is None:
            self._output_copy_event = torch.cuda.Event(external=True)
            self._output_copy_event.record(torch.cuda.current_stream(device))
        if self.overlap and self._side_stream is None:
            self._side_stream = torch.cuda.Stream(device=device)

    def begin_forward(
        self,
        *,
        positions: torch.Tensor,
        context: TokenProbeForwardContext | None,
    ) -> None:
        self._features = None
        self._captured = 0
        self._capture_enabled = context is None or context["capture"]
        self._score_enabled = context is None or context["score"]
        self._positions = positions
        self._block_table = None if context is None else context["block_table"]
        self._slot_mapping = None if context is None else context["slot_mapping"]
        self._query_start_loc = None if context is None else context["query_start_loc"]
        self._max_query_len = 1 if context is None else context["max_query_len"]
        self._launched = False

    def capture(
        self,
        *,
        layer_id: int,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> None:
        slot = self.tap_slots.get(layer_id)
        if not self._capture_enabled or slot is None:
            return
        if self._features is None:
            self._features = torch.empty(
                hidden_states.shape[0],
                len(self.tap_slots) * hidden_states.shape[-1],
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
        tap_into(self._features, slot, hidden_states, residual)
        self._captured += 1
        if self._captured == len(self.tap_slots):
            self._launched = self._launch_async()

    def _compute_captured(self) -> torch.Tensor | None:
        if self._features is None:
            return None
        if self._captured != len(self.tap_slots):
            raise RuntimeError(
                f"token probe captured {self._captured} hidden states, expected "
                f"{len(self.tap_slots)}"
            )
        assert self._positions is not None
        if self._score_enabled and self._output_copy_event is not None:
            torch.cuda.current_stream().wait_event(self._output_copy_event)
        return self._compute(
            self._features,
            self._positions,
            self._block_table,
            self._slot_mapping,
            self._query_start_loc,
            self._max_query_len,
        )

    def _compute(
        self,
        features: torch.Tensor,
        positions: torch.Tensor,
        block_table: torch.Tensor | None,
        slot_mapping: torch.Tensor | None,
        query_start_loc: torch.Tensor | None,
        max_query_len: int,
    ) -> torch.Tensor | None:
        if self.probe_head is None:
            return None
        if not isinstance(self.probe_head, SingProbeAttnModel):
            if not self._score_enabled:
                return None
            return self.probe_head.forward_features(features).detach()

        query, key_value = self.probe_head.project(features)
        metadata = (block_table, slot_mapping, query_start_loc)
        if all(item is None for item in metadata):
            # The initial activation-memory profile runs before vLLM has a KV
            # cache configuration. Exercise the trainable tail here; later
            # CUDA-graph profiles and every serving forward use the paged path.
            return self.probe_head.classify(torch.zeros_like(query), query).detach()
        if any(item is None for item in metadata):
            raise RuntimeError("attention token probe metadata is missing")
        if self.kv_cache is None:
            raise RuntimeError("attention token probe KV cache is not initialized")
        assert block_table is not None
        assert slot_mapping is not None
        assert query_start_loc is not None
        store_probe_kv(key_value, self.kv_cache, slot_mapping)
        if not self._score_enabled:
            return None
        attention_output = probe_paged_attention(
            query=query,
            kv_cache=self.kv_cache,
            block_table=block_table,
            positions=positions,
            query_start_loc=query_start_loc,
            max_query_len=max_query_len,
            num_heads=self.probe_head.num_attention_heads,
            head_dim=self.probe_head.head_dim,
            block_size=self.kv_block_size,
            window=self.probe_head.sliding_window,
        )
        return self.probe_head.classify(attention_output, query).detach()

    def _launch_async(self) -> bool:
        if self._side_stream is None:
            return False
        main_stream = torch.cuda.current_stream()
        self._side_stream.wait_stream(main_stream)
        with torch.cuda.stream(self._side_stream):
            self._pending_scores = self._compute_captured()
        self._pending_event = torch.cuda.Event()
        self._pending_event.record(self._side_stream)
        return True

    def _finish_async(self) -> torch.Tensor | None:
        if self._pending_event is None:
            return None
        torch.cuda.current_stream().wait_event(self._pending_event)
        scores = self._pending_scores
        self._pending_event = None
        self._pending_scores = None
        return scores

    def finish_forward(
        self, hidden_states: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        if not self.enabled:
            return hidden_states
        if not self._capture_enabled:
            return hidden_states, None
        scores = self._finish_async() if self._launched else self._compute_captured()
        self._features = None
        return hidden_states, scores

    def parameter_names(self, prefix: str) -> set[str]:
        return {name for name, _ in self.named_parameters(prefix=prefix)}

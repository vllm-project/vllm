# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A deterministic persistent-state model for end-to-end tests.

The model keeps one counter per state slot. Each scheduled step of a request
advances its counter by one and emits a token that is a function of the
counter and the request's last input token; every third step emits EOS, so a
resumable request parks after two tokens and its next input continues from
the counter the slot kept. A slot that is reused without its release fence or
without the initializer, or that is cleared or re-created mid-stream, changes
the output, which is what the regression, pause and cache-clear tests rely on.

`hf_overrides` field `stub_page_mib` sets the state page size in MiB
(default 1).

`EVENTS` records, in order, the hook calls and forward passes of every stub
built in this process, so an in-process engine test can tell how far startup
got even when it fails.
"""

from typing import Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.model_executor.layers.state import PersistentStateLayerBase
from vllm.sequence import IntermediateTensors
from vllm.v1.kv_cache_interface import (
    PersistentStateDescriptor,
    PersistentStateSpec,
)
from vllm.v1.worker.gpu.model_states.interface import ModelState

STATE_LAYER = "state"
TOKENS_PER_TURN = 2
MiB = 1 << 20

# ("initialize_state_slots",), ("prepare_step", dummy_run, is_profile,
# slots_bound) and ("forward", dummy_run) records.
EVENTS: list[tuple[Any, ...]] = []


def expected_tokens(
    counter: int, last_input_token: int, vocab_size: int, eos_token_id: int
) -> list[int]:
    """The tokens one turn emits from `counter` (steps already taken) after an
    input ending in `last_input_token`; the EOS that parks the turn is last."""
    tokens = []
    last = last_input_token
    for _ in range(TOKENS_PER_TURN + 1):
        counter += 1
        last = next_token(counter, last, vocab_size, eos_token_id)
        tokens.append(last)
    return tokens


def next_token(counter: int, last_token: int, vocab_size: int, eos_token_id: int):
    if counter % (TOKENS_PER_TURN + 1) == 0:
        return eos_token_id
    return 3 + (counter * 7919 + last_token) % (vocab_size - 3)


class StubPersistentStateModelState(ModelState):
    """Records the hook calls and binds the step's slot rows for the model."""

    def __init__(self, vllm_config, model, encoder_cache, device):
        super().__init__(vllm_config, model, encoder_cache, device)
        self._handles: dict[str, Any] = {}

    def initialize_state_slots(self, bindings) -> None:
        super().initialize_state_slots(bindings)
        EVENTS.append(("initialize_state_slots",))

    def prepare_step(self, scheduler_output, *, dummy_run: bool, is_profile: bool):
        EVENTS.append(
            ("prepare_step", dummy_run, is_profile, self.state_slots is not None)
        )
        self._handles = {} if dummy_run else (scheduler_output.state_slot_handles or {})

    def prepare_inputs(self, input_batch, req_states) -> dict[str, Any]:
        if self.state_slots is None or not self._handles:
            return {}
        num_reqs = input_batch.num_reqs
        request_ids = tuple(input_batch.req_ids[:num_reqs])
        handles = tuple(self._handles[rid][STATE_LAYER] for rid in request_ids)
        rows = self.state_slots.bind_rows(STATE_LAYER, request_ids, handles)
        return {
            # Pinned and non-blocking, so that the copy does not wait on the
            # device.
            "slot_indices": torch.tensor(rows.slot_indices_cpu)
            .pin_memory()
            .to(self.device, non_blocking=True),
            "count_pool": rows.component_pools["count"],
            "query_start_loc": input_batch.query_start_loc,
            "num_reqs": num_reqs,
        }

    def prepare_dummy_inputs(self, num_reqs: int, num_tokens: int) -> dict[str, Any]:
        return {"dummy_run": True}

    def prepare_inputs_embeds(self, scheduled_encoder_inputs, input_batch, req_states):
        return None

    def prepare_attn(self, *args, **kwargs) -> dict[str, Any]:
        return {}


class StubPersistentStateForCausalLM(nn.Module):
    """One counter per slot; see the module docstring."""

    supports_persistent_state = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        model_config = vllm_config.model_config
        hf = model_config.hf_config
        self.vocab_size = model_config.get_vocab_size()
        self.eos_token_id = int(hf.eos_token_id)
        page_bytes = int(getattr(hf, "stub_page_mib", 1) * MiB)
        spec = PersistentStateSpec(
            descriptors=(
                PersistentStateDescriptor(
                    "count", (1,), torch.int64, 0, 8, torch.zeros
                ),
                PersistentStateDescriptor(
                    "filler", (page_bytes - 8,), torch.uint8, 8, 8, torch.zeros
                ),
            ),
            page_size_bytes=page_bytes,
        )
        self.state = PersistentStateLayerBase(
            spec, prefix=f"{prefix}{STATE_LAYER}", vllm_config=vllm_config
        )
        self.embed = nn.Embedding(self.vocab_size, 8)

    @classmethod
    def get_model_state_cls(cls) -> type[ModelState]:
        return StubPersistentStateModelState

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        slot_indices: torch.Tensor | None = None,
        count_pool: torch.Tensor | None = None,
        query_start_loc: torch.Tensor | None = None,
        num_reqs: int = 0,
        dummy_run: bool = False,
    ) -> torch.Tensor:
        assert input_ids is not None
        EVENTS.append(("forward", dummy_run))
        hidden = torch.zeros(
            input_ids.shape[0], 8, dtype=torch.float32, device=input_ids.device
        )
        if slot_indices is None or count_pool is None or query_start_loc is None:
            return hidden
        last = query_start_loc[1 : num_reqs + 1] - 1
        counters = count_pool[slot_indices, 0] + 1
        count_pool[slot_indices, 0] = counters
        last_tokens = input_ids[last].long()
        tokens = torch.where(
            counters % (TOKENS_PER_TURN + 1) == 0,
            torch.full_like(counters, self.eos_token_id),
            3 + (counters * 7919 + last_tokens) % (self.vocab_size - 3),
        )
        hidden[last, 0] = tokens.to(torch.float32)
        return hidden

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros(
            hidden_states.shape[0],
            self.vocab_size,
            dtype=torch.float32,
            device=hidden_states.device,
        )
        logits.scatter_(1, hidden_states[:, :1].long(), 1.0)
        return logits

    def load_weights(self, weights) -> set[str]:
        loaded = set()
        params = dict(self.named_parameters())
        for name, weight in weights:
            if name in params:
                params[name].data.copy_(weight)
                loaded.add(name)
        return loaded


class StubUndeclaredPersistentStateForCausalLM(StubPersistentStateForCausalLM):
    """The same model without the persistent-state declaration."""

    supports_persistent_state = False

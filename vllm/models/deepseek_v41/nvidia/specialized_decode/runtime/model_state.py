# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The model state of the specialized decode.

`DeepseekV41SpecializedDecodeModelState` is `DeepseekV41ModelState` plus:

- the decode executors of the target and of the DSpark draft model, built and
  warmed up once the weights are loaded (the model runner creates the model
  state after loading the target and draft models, and before memory
  profiling);
- for a step that replays its recorded attention-metadata build, the engram
  lookback windows gathered on a side stream beside it, so the metadata graph
  hands off to the forward's graph with no eager kernel between them;
- no replay-start upload for decode steps whose starts are known to be zero.
"""

import numpy as np
import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.utils.torch_utils import current_stream
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.states import RequestState

from ...model_state import DeepseekV41ModelState
from .. import take_drafts
from .draft import check_low_batch_graph
from .warmup import warm_up


class DeepseekV41SpecializedDecodeModelState(DeepseekV41ModelState):
    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ):
        super().__init__(vllm_config, model, encoder_cache, device)
        # decode_backend=latency_specialized requires FULL decode graphs.
        assert self.attn_metadata_graphs is not None
        language_model = (
            model.get_language_model()
            if hasattr(model, "get_language_model")
            else model
        )
        self.decode = language_model.build_specialized_decode()
        # The DSpark draft model, loaded before the model state, runs on the
        # target's executor.
        drafts = [
            draft.build_specialized_decode(self.decode) for draft in take_drafts()
        ]
        if drafts:
            check_low_batch_graph(vllm_config)
        warm_up(self.decode, vllm_config, drafts)
        # Leading rows of _replay_start known to hold zeros.
        self._zero_replay_starts = 0
        # The runner's request state, from the first prepare_inputs on; the
        # lookback gather that runs beside a metadata graph reads it.
        self._req_states: RequestState | None = None
        # The gather's side stream and the event recorded after it.
        self._lookback_side: tuple[torch.cuda.Stream, torch.cuda.Event] | None = None
        # The batch whose lookback gather was issued beside its metadata graph,
        # and the event that ends it.
        self._staged_lookback: tuple[InputBatch, torch.cuda.Event] | None = None

    def _upload_replay_start(self, values: np.ndarray, num_reqs: int) -> torch.Tensor:
        # Decode steps upload only zeros; the capture's zeroing is the only
        # other writer, and it writes zeros.
        has_replay = bool(values.any())
        if has_replay or num_reqs > self._zero_replay_starts:
            super()._upload_replay_start(values, num_reqs)
            self._zero_replay_starts = 0 if has_replay else num_reqs
        return self._replay_start[:num_reqs]

    def prepare_inputs(
        self, input_batch: InputBatch, req_states: RequestState
    ) -> dict[str, torch.Tensor | None]:
        self._req_states = req_states
        return super().prepare_inputs(input_batch, req_states)

    def _before_metadata_replay(self, input_batch: InputBatch) -> None:
        """Issue the batch's lookback gather on a side stream, beside the
        metadata graph about to replay. It reads the batch's request indices
        and the token history, which the runner has written by now."""
        if self.lookback_token_ids is None or self._req_states is None:
            return
        if self._lookback_side is None:
            self._lookback_side = (torch.cuda.Stream(), torch.cuda.Event())
        self._join_staged_lookback()
        stream, event = self._lookback_side
        stream.wait_stream(current_stream())
        with torch.cuda.stream(stream):
            super()._fill_lookback(input_batch, self._req_states)
        event.record(stream)
        self._staged_lookback = (input_batch, event)

    def _join_staged_lookback(self) -> InputBatch | None:
        """Make the current stream wait for a staged lookback gather, if any;
        returns the batch it gathered."""
        staged, self._staged_lookback = self._staged_lookback, None
        if staged is None:
            return None
        current_stream().wait_event(staged[1])
        return staged[0]

    def _fill_lookback(self, input_batch: InputBatch, req_states: RequestState) -> None:
        # Even a stale gather (a batch whose inputs were never prepared) is
        # joined, so that no two gathers write the window at once.
        if self._join_staged_lookback() is not input_batch:
            super()._fill_lookback(input_batch, req_states)

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V2 model-runner state for Parakeet TDT.

Parakeet's TDT decoder is non-autoregressive: the full greedy transcript is
computed from the encoder output at prefill, and vLLM's decode loop merely
replays the precomputed tokens. The reference PR stored that transcript on the
model instance, which forced ``max_num_seqs = 1``. Here we hold one transcript
per request id and build a per-token forced-token tensor for the whole batch,
so concurrent transcriptions no longer corrupt each other's state.
"""

from typing import Any

import torch

from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.gpu.states import RequestState


class ParakeetTDTModelState(DefaultModelState):
    """Per-request forced-token replay state for Parakeet TDT under V2."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # req_id -> precomputed greedy TDT transcript (incl. trailing EOS).
        self.transcripts: dict[str, list[int]] = {}
        self.eos_token_id = int(self.model_config.hf_config.eos_token_id)

    def get_supported_generation_tasks(self):
        return ("transcription",)

    def get_mm_embeddings(
        self,
        scheduled_encoder_inputs: dict[str, list[int]],
        input_batch: InputBatch,
    ) -> None:
        # Only prefilling requests carry encoder inputs. Run the FastConformer
        # encoder + greedy TDT decode once per request and cache the transcript.
        if not scheduled_encoder_inputs:
            return None

        # Reconstruct the req_id order that execute_mm_encoder will return its
        # outputs in (mirrors EncoderRunner.prepare_mm_inputs iteration + skip).
        ordered_req_ids: list[str] = []
        for req_id, encoder_input_ids in scheduled_encoder_inputs.items():
            mm_features = self.encoder_cache.mm_features[req_id]
            for mm_input_id in encoder_input_ids:
                if mm_features[mm_input_id].data is None:
                    continue
                ordered_req_ids.append(req_id)

        _, mm_kwargs = self.encoder_runner.prepare_mm_inputs(scheduled_encoder_inputs)
        if not mm_kwargs:
            return None

        encoder_outputs = self.encoder_runner.execute_mm_encoder(mm_kwargs)
        for req_id, encoder_output in zip(ordered_req_ids, encoder_outputs):
            self.transcripts[req_id] = self.model.model.greedy_decode(encoder_output)
        return None

    def prepare_inputs(
        self, input_batch: InputBatch, req_states: RequestState
    ) -> dict[str, Any]:
        model_inputs = super().prepare_inputs(input_batch, req_states)

        num_padded = input_batch.num_tokens_after_padding
        positions = input_batch.positions[:num_padded].tolist()
        query_start_loc = input_batch.query_start_loc_np

        # Default padding (and any request without a transcript yet) to EOS.
        forced: list[int] = [self.eos_token_id] * num_padded
        for i, req_id in enumerate(input_batch.req_ids):
            sequence = self.transcripts.get(req_id)
            if not sequence:
                continue
            start = int(query_start_loc[i])
            end = int(query_start_loc[i + 1])
            for token_idx in range(start, end):
                position = positions[token_idx]
                if 0 <= position < len(sequence):
                    forced[token_idx] = sequence[position]

        model_inputs["forced_token_ids"] = torch.tensor(
            forced, dtype=torch.long, device=self.device
        )

        # Drop transcripts for requests no longer in the batch (best-effort;
        # finished requests never reappear). Keeps the cache from growing.
        live = set(input_batch.req_ids)
        for stale in [r for r in self.transcripts if r not in live]:
            del self.transcripts[stale]

        return model_inputs

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch
from transformers.models.nemotron3_5_asr.generation_nemotron3_5_asr import (
    Nemotron3_5AsrRNNTDecoderCache,
)

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.model_executor.models.nemotron3_5_asr import Nemotron3_5AsrDecodeState
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.encoder_budget import MultiModalBudget
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.states import RequestState
from vllm.v1.worker.utils import AttentionGroup


class Nemotron3_5AsrModelState(ModelState):
    """Keep RNNT acoustic and LSTM state isolated per request."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: torch.nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ) -> None:
        assert encoder_cache is not None
        super().__init__(vllm_config, model, encoder_cache, device)
        budget = MultiModalBudget(vllm_config, MULTIMODAL_REGISTRY, enable_cache=False)
        # The shared encoder cache is released after prefill. RNNT needs the
        # acoustic frames until completion, so reserve accounted per-request slots.
        self.acoustic_states = torch.empty(
            (
                self.max_num_reqs,
                budget.mm_max_toks_per_item["audio"],
                self.model.config.decoder_hidden_size,
            ),
            dtype=self.dtype,
            device=device,
        )
        self.audio_hashes: dict[str, str] = {}
        self.request_indices: dict[str, int] = {}
        self.decode_states: dict[str, Nemotron3_5AsrDecodeState] = {}
        self.num_tokens_to_replay: dict[str, int] = {}

    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None:
        audio_features = [
            feature
            for feature in new_req_data.mm_features
            if feature.modality == "audio"
        ]
        if not audio_features and new_req_data.req_id.startswith("_warmup_"):
            return
        if len(audio_features) != 1:
            raise ValueError("Nemotron 3.5 ASR expects one audio per request.")
        self.audio_hashes[new_req_data.req_id] = audio_features[0].identifier
        self.request_indices[new_req_data.req_id] = req_index
        assert new_req_data.prefill_token_ids is not None
        self.num_tokens_to_replay[new_req_data.req_id] = (
            len(new_req_data.prefill_token_ids) - new_req_data.prompt_len
        )

    def remove_request(self, req_id: str) -> None:
        self.audio_hashes.pop(req_id, None)
        self.request_indices.pop(req_id, None)
        self.decode_states.pop(req_id, None)
        self.num_tokens_to_replay.pop(req_id, None)

    def prepare_inputs_embeds(
        self,
        scheduled_encoder_inputs: dict[str, list[int]],
        input_batch: InputBatch,
        req_states: RequestState,
    ) -> None:
        self.execute_mm_encoder(scheduled_encoder_inputs)
        return None

    def prepare_inputs(
        self, input_batch: InputBatch, req_states: RequestState
    ) -> dict[str, Any]:
        # Eager dummy runs use synthetic request IDs and mark every token as
        # padding. Profile the RNNT path without touching live request state.
        if not any(
            req_id in self.request_indices for req_id in input_batch.req_ids
        ) and bool(input_batch.is_padding[0].item()):
            dummy_inputs = self.prepare_dummy_inputs(
                input_batch.num_reqs, input_batch.num_tokens
            )
            dummy_inputs["query_end_positions"] = input_batch.query_start_loc_np[
                1 : input_batch.num_reqs + 1
            ].tolist()
            return dummy_inputs

        states: list[Nemotron3_5AsrDecodeState | None] = []
        for req_id in input_batch.req_ids:
            state = self.decode_states.get(req_id)
            if state is None and req_id in self.audio_hashes:
                audio_hash = self.audio_hashes[req_id]
                encoder_frames = self.encoder_cache.encoder_outputs[audio_hash]
                frames = self.acoustic_states[
                    self.request_indices[req_id], : encoder_frames.shape[0]
                ]
                frames.copy_(encoder_frames)
                state = Nemotron3_5AsrDecodeState(
                    encoder_frames=frames,
                    decoder_cache=Nemotron3_5AsrRNNTDecoderCache(self.model.config),
                    last_token_id=self.model.config.blank_token_id,
                    num_tokens_to_replay=self.num_tokens_to_replay.pop(req_id),
                )
                self.decode_states[req_id] = state
            states.append(state)

        ready = (
            input_batch.num_computed_tokens_np + input_batch.num_scheduled_tokens
            >= input_batch.prefill_len_np
        )
        return {
            "decode_states": states,
            "query_end_positions": input_batch.query_start_loc_np[
                1 : input_batch.num_reqs + 1
            ].tolist(),
            "decode_ready": ready.tolist(),
        }

    def prepare_dummy_inputs(self, num_reqs: int, num_tokens: int) -> dict[str, Any]:
        frames = torch.zeros(
            (1, self.model.config.decoder_hidden_size),
            dtype=self.dtype,
            device=self.device,
        )
        return {
            "decode_states": [
                Nemotron3_5AsrDecodeState(
                    encoder_frames=frames,
                    decoder_cache=Nemotron3_5AsrRNNTDecoderCache(self.model.config),
                    last_token_id=self.model.config.blank_token_id,
                )
                for _ in range(num_reqs)
            ],
            "query_end_positions": [
                (i + 1) * num_tokens // num_reqs for i in range(num_reqs)
            ],
            "decode_ready": [True] * num_reqs,
        }

    def prepare_attn(
        self,
        input_batch: InputBatch,
        cudagraph_mode: CUDAGraphMode,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        for_capture: bool = False,
        ubatch_idx: int = 0,
    ) -> dict[str, Any]:
        return {}

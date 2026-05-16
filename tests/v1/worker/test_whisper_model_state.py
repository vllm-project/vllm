# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.multimodal.inputs import (
    MultiModalFeatureSpec,
    MultiModalKwargsItem,
    PlaceholderRange,
)
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.model_states.whisper import WhisperModelState

pytestmark = pytest.mark.skip_global_cleanup


class _DummyModel:
    def embed_multimodal(self, **kwargs):
        return [torch.tensor([[1.0, 2.0]], dtype=torch.float32)]


def _make_whisper_state() -> WhisperModelState:
    vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=4, max_num_batched_tokens=16),
        model_config=SimpleNamespace(
            max_model_len=16,
            dtype=torch.float32,
            hf_config=SimpleNamespace(max_source_positions=16),
            get_inputs_embeds_size=lambda: 2,
        ),
    )
    return WhisperModelState(
        vllm_config=vllm_config,
        model=_DummyModel(),
        encoder_cache=EncoderCache(),
        device=torch.device("cpu"),
    )


@pytest.mark.cpu_test
@pytest.mark.skip_global_cleanup
def test_whisper_model_state_reuses_cached_encoder_outputs_for_prefill_requests():
    state = _make_whisper_state()
    mm_feature = MultiModalFeatureSpec(
        data=MultiModalKwargsItem.dummy(),
        modality="audio",
        identifier="shared_audio",
        mm_position=PlaceholderRange(offset=0, length=4),
    )
    state.encoder_cache.add_request("req0", [mm_feature])

    input_batch = SimpleNamespace(
        req_ids=["req0"],
        idx_mapping_np=np.array([0], dtype=np.int32),
    )
    req_states = SimpleNamespace(
        req_id_to_index={"req0": 0},
        num_computed_tokens=SimpleNamespace(np=np.array([0], dtype=np.int32)),
        prefill_len=SimpleNamespace(np=np.array([4], dtype=np.int32)),
        num_computed_prefill_tokens=np.array([0], dtype=np.int32),
    )

    state.get_mm_embeddings({"req0": [0]}, input_batch, req_states)
    assert len(state.encoder_outputs) == 1
    first_output = state.encoder_outputs[0]
    assert state.encoder_cache.encoder_outputs["shared_audio"] is first_output

    state.encoder_outputs = []
    state.get_mm_embeddings({}, input_batch, req_states)

    assert state.encoder_outputs == [first_output]
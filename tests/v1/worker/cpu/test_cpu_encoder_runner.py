# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest
import torch
from transformers import LlamaConfig

from vllm.config import ModelConfig, SchedulerConfig, VllmConfig
from vllm.entrypoints.pooling.scoring.utils import compress_token_type_ids
from vllm.multimodal.inputs import (
    MultiModalBatchedField,
    MultiModalFeatureSpec,
    MultiModalFieldElem,
    MultiModalKwargsItem,
    PlaceholderRange,
)
from vllm.platforms import CpuArchEnum, current_platform
from vllm.pooling_params import PoolingParams
from vllm.triton_utils import HAS_TRITON
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.worker.cpu.encoder_cache import EncoderCache
from vllm.v1.worker.cpu.encoder_runner import EncoderRunner
from vllm.v1.worker.cpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.cpu.model_states.default import DefaultModelState
from vllm.v1.worker.cpu.model_states.encoder_only import EncoderOnlyModelState
from vllm.v1.worker.cpu.states import CPURequestState

pytestmark = pytest.mark.skipif(
    not current_platform.is_cpu()
    or HAS_TRITON
    or current_platform.get_cpu_architecture() == CpuArchEnum.X86,
    reason="requires a non-x86 CPU platform without Triton",
)

HIDDEN_SIZE = 4


class _EncoderModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.batches: list[list[int]] = []

    def embed_multimodal(self, values: torch.Tensor) -> list[torch.Tensor]:
        self.batches.append(values.flatten().tolist())
        return [
            value.reshape(1, 1).expand(1, HIDDEN_SIZE).to(torch.float32)
            for value in values
        ]


def _item(value: int) -> MultiModalKwargsItem:
    return MultiModalKwargsItem(
        {
            "values": MultiModalFieldElem(
                data=torch.tensor([value]),
                field=MultiModalBatchedField(),
            )
        }
    )


def _feature(
    identifier: str,
    offset: int,
    length: int,
    modality: str = "image",
    data: MultiModalKwargsItem | None = None,
) -> MultiModalFeatureSpec:
    return MultiModalFeatureSpec(
        data=data,
        modality=modality,
        identifier=identifier,
        mm_position=PlaceholderRange(offset=offset, length=length),
    )


def _make_runner(
    features: list[MultiModalFeatureSpec],
    cached: list[MultiModalFeatureSpec],
) -> EncoderRunner:
    cache = EncoderCache()
    cache.mm_features["req0"] = features
    for feature in cached:
        length = feature.mm_position.length
        cache.encoder_outputs[feature.identifier] = torch.arange(
            length * HIDDEN_SIZE, dtype=torch.float32
        ).reshape(length, HIDDEN_SIZE)
    return EncoderRunner(
        model=None,
        max_num_tokens=64,
        hidden_size=HIDDEN_SIZE,
        encoder_cache=cache,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )


def _gather(
    runner: EncoderRunner, num_scheduled_tokens: int
) -> tuple[list[torch.Tensor], torch.Tensor]:
    return runner.gather_mm_embeddings(
        req_ids=["req0"],
        total_num_scheduled_tokens=num_scheduled_tokens,
        num_scheduled_tokens=np.array([num_scheduled_tokens]),
        query_start_loc=np.array([0]),
        prefill_lens=np.array([1000]),
        num_computed_tokens=np.array([0]),
    )


def test_target_path_raises_on_encoder_cache_miss() -> None:
    runner = _make_runner([_feature("missing", 0, 8)], cached=[])

    with pytest.raises(RuntimeError, match="Encoder cache miss"):
        _gather(runner, num_scheduled_tokens=8)


def test_multi_request_batch_gathers_per_request() -> None:
    first = _feature("first", 0, 8)
    second = _feature("second", 0, 8)
    cache = EncoderCache()
    cache.mm_features = {"req0": [first], "req1": [second]}
    for feature in (first, second):
        cache.encoder_outputs[feature.identifier] = torch.arange(
            feature.mm_position.length * HIDDEN_SIZE, dtype=torch.float32
        ).reshape(feature.mm_position.length, HIDDEN_SIZE)
    runner = EncoderRunner(
        model=None,
        max_num_tokens=64,
        hidden_size=HIDDEN_SIZE,
        encoder_cache=cache,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    mm_embeds, is_mm_embed = runner.gather_mm_embeddings(
        req_ids=["req0", "req1"],
        total_num_scheduled_tokens=16,
        num_scheduled_tokens=np.array([8, 8]),
        query_start_loc=np.array([0, 8]),
        prefill_lens=np.array([1000, 1000]),
        num_computed_tokens=np.array([0, 0]),
    )

    assert len(mm_embeds) == 2
    assert [embedding.modality for embedding in mm_embeds] == ["image", "image"]
    assert is_mm_embed.all()


def test_gather_preserves_mixed_modalities() -> None:
    video = _feature("video", 0, 4, modality="video")
    audio = _feature("audio", 4, 4, modality="audio")
    runner = _make_runner([video, audio], cached=[video, audio])

    mm_embeds, is_mm_embed = _gather(runner, num_scheduled_tokens=8)

    assert [embedding.modality for embedding in mm_embeds] == ["video", "audio"]
    assert is_mm_embed.all()


def test_execute_mm_encoder_batches_by_modality_and_restores_order() -> None:
    model = _EncoderModel()
    runner = EncoderRunner(
        model=model,
        max_num_tokens=64,
        hidden_size=HIDDEN_SIZE,
        encoder_cache=EncoderCache(),
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    items = [
        ("video", _item(0)),
        ("image", _item(1)),
        ("video", _item(2)),
        ("image", _item(3)),
    ]

    outputs = runner.execute_mm_encoder(items)

    assert model.batches == [[1, 3], [0, 2]]
    assert [int(output[0, 0]) for output in outputs] == [0, 1, 2, 3]


@pytest.fixture
def vllm_config(tmp_path) -> VllmConfig:
    hf_config = LlamaConfig(
        hidden_size=HIDDEN_SIZE,
        intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        max_position_embeddings=64,
        vocab_size=32,
        architectures=["LlamaForCausalLM"],
    )
    hf_config.save_pretrained(tmp_path)
    model_config = ModelConfig(
        model=str(tmp_path),
        dtype="float32",
        max_model_len=64,
        skip_tokenizer_init=True,
    )
    scheduler_config = SchedulerConfig(
        max_model_len=64,
        is_encoder_decoder=False,
        max_num_batched_tokens=64,
        max_num_seqs=2,
        is_multimodal_model=True,
    )
    return VllmConfig(
        model_config=model_config,
        scheduler_config=scheduler_config,
    )


def test_model_state_caches_encoder_output_without_gathering(
    vllm_config: VllmConfig,
) -> None:
    model = _EncoderModel()
    cache = EncoderCache()
    cache.add_request(
        "req0",
        [
            _feature("encoded", 0, 1, data=_item(7)),
            _feature("not-encoded", 1, 1),
        ],
    )
    state = DefaultModelState(
        vllm_config,
        model,
        cache,
        torch.device("cpu"),
    )

    state.execute_mm_encoder({"req0": [0, 1]})

    assert model.batches == [[7]]
    assert set(cache.encoder_outputs) == {"encoded"}
    torch.testing.assert_close(
        cache.encoder_outputs["encoded"],
        torch.full((1, HIDDEN_SIZE), 7.0),
    )


def test_model_state_empty_encoder_schedule_is_a_noop(
    vllm_config: VllmConfig,
) -> None:
    model = _EncoderModel()
    cache = EncoderCache()
    cached = torch.ones(1, HIDDEN_SIZE)
    cache.encoder_outputs["existing"] = cached
    state = DefaultModelState(
        vllm_config,
        model,
        cache,
        torch.device("cpu"),
    )

    state.execute_mm_encoder({})

    assert model.batches == []
    assert set(cache.encoder_outputs) == {"existing"}
    assert cache.encoder_outputs["existing"] is cached


def test_encoder_only_state_slices_compressed_token_types_for_chunk(
    vllm_config: VllmConfig,
) -> None:
    tokens = [10, 11, 12, 13, 14, 15]
    request = NewRequestData(
        req_id="req0",
        prompt_token_ids=tokens,
        mm_features=[],
        sampling_params=None,
        pooling_params=PoolingParams(
            extra_kwargs={
                "compressed_token_type_ids": compress_token_type_ids([0, 0, 1, 1, 1, 0])
            }
        ),
        block_ids=(),
        num_computed_tokens=1,
        lora_request=None,
        prefill_token_ids=tokens,
    )
    req_states = CPURequestState(
        max_num_reqs=2,
        max_model_len=64,
        max_num_batched_tokens=64,
        vocab_size=32,
        device=torch.device("cpu"),
    )
    req_states.add_request(
        req_id=request.req_id,
        prompt_len=len(tokens),
        all_token_ids=tokens,
        num_computed_tokens=1,
        max_tokens=1,
    )
    req_index = req_states.req_id_to_index[request.req_id]
    state = EncoderOnlyModelState(
        vllm_config, torch.nn.Module(), None, torch.device("cpu")
    )
    state.add_request(req_index, request)

    input_batch = InputBatch.make_dummy(
        num_reqs=1,
        num_tokens=5,
        input_buffers=InputBuffers(2, 64, torch.device("cpu")),
        is_padding=False,
    )
    input_batch.req_ids[0] = request.req_id
    input_batch.idx_mapping.fill_(req_index)
    input_batch.idx_mapping_np.fill(req_index)
    input_batch.expanded_idx_mapping.fill_(req_index)
    input_batch.num_computed_tokens_np[0] = 1

    model_inputs = state.prepare_inputs(input_batch, req_states)

    token_type_ids = model_inputs["token_type_ids"]
    assert token_type_ids is not None
    assert token_type_ids.tolist() == [0, 1, 1, 1, 0]


def test_encoder_timing_stats_registry() -> None:
    runner = _make_runner([], [])
    runner.enable_timing = True

    with runner.timed_encoder_operation({"req0"}):
        pass
    with runner.timed_encoder_operation({"req0"}):
        pass

    stats = runner.get_encoder_timing_stats()
    assert stats["req0"]["num_encoder_calls"] == 2
    assert stats["req0"]["encoder_forward_secs"] >= 0
    assert runner.get_encoder_timing_stats() == {}

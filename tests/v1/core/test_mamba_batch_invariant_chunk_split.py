# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vllm.config.mamba import MambaBackendEnum
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.core.sched.scheduler import (
    Scheduler,
    _has_mamba2_layers,
    _validate_mamba2_batch_invariant_config,
)
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, MambaSpec
from vllm.v1.outputs import ModelRunnerOutput

from .utils import create_requests, create_scheduler

pytestmark = pytest.mark.cpu_test


def _kv_cache_config(mamba_type: MambaAttentionBackendEnum) -> KVCacheConfig:
    return KVCacheConfig(
        num_blocks=1,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["recurrent_layer"],
                MambaSpec(
                    block_size=16,
                    shapes=((1,),),
                    dtypes=(torch.float32,),
                    mamba_type=mamba_type,
                ),
            )
        ],
    )


@pytest.mark.parametrize(
    ("mamba_type", "expected"),
    [
        (MambaAttentionBackendEnum.MAMBA2, True),
        (MambaAttentionBackendEnum.MAMBA1, False),
        (MambaAttentionBackendEnum.GDN_ATTN, False),
    ],
)
def test_chunk_invariant_split_is_scoped_to_mamba2(
    mamba_type: MambaAttentionBackendEnum,
    expected: bool,
) -> None:
    assert _has_mamba2_layers(_kv_cache_config(mamba_type)) is expected


def _mamba_vllm_config(
    *,
    backend: MambaBackendEnum = MambaBackendEnum.TRITON,
    stochastic_rounding: bool = False,
    use_replayssm: bool = False,
    enable_prefix_caching: bool = False,
    speculative_decoding: bool = False,
    multimodal: bool = False,
    kv_connector: bool = False,
):
    return SimpleNamespace(
        mamba_config=SimpleNamespace(
            backend=backend,
            enable_stochastic_rounding=stochastic_rounding,
        ),
        cache_config=SimpleNamespace(
            use_replayssm=use_replayssm,
            enable_prefix_caching=enable_prefix_caching,
        ),
        speculative_config=(SimpleNamespace() if speculative_decoding else None),
        model_config=SimpleNamespace(is_multimodal_model=multimodal),
        kv_transfer_config=(
            SimpleNamespace(kv_connector="NixlConnector") if kv_connector else None
        ),
    )


def test_default_triton_configuration_is_allowed() -> None:
    _validate_mamba2_batch_invariant_config(_mamba_vllm_config())


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (
            _mamba_vllm_config(backend=MambaBackendEnum.FLASHINFER),
            "only the Triton SSU backend",
        ),
        (
            _mamba_vllm_config(stochastic_rounding=True),
            "incompatible with stochastic rounding",
        ),
        (
            _mamba_vllm_config(use_replayssm=True),
            "not been validated with ReplaySSM",
        ),
        (
            _mamba_vllm_config(enable_prefix_caching=True),
            "does not yet support prefix caching",
        ),
        (
            _mamba_vllm_config(kv_connector=True),
            "not been validated with KV connectors",
        ),
        (
            _mamba_vllm_config(speculative_decoding=True),
            "does not yet support speculative decoding",
        ),
        (
            _mamba_vllm_config(multimodal=True),
            "not been validated for multimodal models",
        ),
    ],
)
def test_unvalidated_mamba_subconfigurations_are_rejected(
    config,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _validate_mamba2_batch_invariant_config(config)


def _split(
    *,
    prompt_tokens: int,
    request_tokens: int,
    computed_tokens: int,
    scheduled_tokens: int,
    chunk_size: int = 256,
) -> int:
    scheduler = SimpleNamespace(mamba_chunk_size=chunk_size)
    request = SimpleNamespace(
        num_prompt_tokens=prompt_tokens,
        num_tokens=request_tokens,
    )
    return Scheduler._mamba_chunk_invariant_split(
        scheduler,
        request,
        scheduled_tokens,
        computed_tokens,
    )


@pytest.mark.parametrize(
    ("computed_tokens", "scheduled_tokens", "expected"),
    [
        (0, 1, 0),
        (0, 255, 0),
        (0, 256, 256),
        (0, 700, 256),
        (256, 300, 256),
        (512, 255, 0),
    ],
)
def test_nonfinal_prefill_stops_on_scan_chunk_boundaries(
    computed_tokens: int,
    scheduled_tokens: int,
    expected: int,
) -> None:
    assert (
        _split(
            prompt_tokens=768,
            request_tokens=768,
            computed_tokens=computed_tokens,
            scheduled_tokens=scheduled_tokens,
        )
        == expected
    )


def test_long_final_prefill_uses_one_chunk_then_tail() -> None:
    assert (
        _split(
            prompt_tokens=257,
            request_tokens=257,
            computed_tokens=0,
            scheduled_tokens=257,
        )
        == 256
    )
    assert (
        _split(
            prompt_tokens=257,
            request_tokens=257,
            computed_tokens=256,
            scheduled_tokens=1,
        )
        == 1
    )


def test_prefill_cannot_resume_inside_scan_chunk() -> None:
    with pytest.raises(RuntimeError, match="cannot resume inside a scan chunk"):
        _split(
            prompt_tokens=257,
            request_tokens=257,
            computed_tokens=17,
            scheduled_tokens=240,
        )


def test_decode_is_not_aligned_as_prefill() -> None:
    assert (
        _split(
            prompt_tokens=768,
            request_tokens=769,
            computed_tokens=768,
            scheduled_tokens=1,
        )
        == 1
    )


def _model_output(scheduler: Scheduler) -> ModelRunnerOutput:
    return ModelRunnerOutput(
        req_ids=[request.request_id for request in scheduler.running],
        req_id_to_index={
            request.request_id: index for index, request in enumerate(scheduler.running)
        },
        sampled_token_ids=[[1000]] * len(scheduler.running),
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


def _enable_mamba_chunk_invariant_prototype(scheduler: Scheduler) -> None:
    scheduler.need_mamba_chunk_invariant_split = True
    scheduler.mamba_chunk_size = 256
    scheduler._mamba_chunk_reservation_pending = False


def _offline_opt_model(tmp_path: Path) -> str:
    model_path = tmp_path / "dummy-opt"
    model_path.mkdir()
    config = {
        "architectures": ["OPTForCausalLM"],
        "model_type": "opt",
        "activation_function": "relu",
        "attention_dropout": 0.0,
        "bos_token_id": 2,
        "do_layer_norm_before": True,
        "dropout": 0.0,
        "enable_bias": True,
        "eos_token_id": 2,
        "ffn_dim": 128,
        "hidden_size": 64,
        "init_std": 0.02,
        "layerdrop": 0.0,
        "max_position_embeddings": 2048,
        "num_attention_heads": 4,
        "num_hidden_layers": 2,
        "pad_token_id": 1,
        "vocab_size": 256,
        "word_embed_proj_dim": 64,
    }
    (model_path / "config.json").write_text(json.dumps(config))
    return str(model_path)


def test_large_budget_does_not_merge_physical_chunks(tmp_path: Path) -> None:
    scheduler = create_scheduler(
        model=_offline_opt_model(tmp_path),
        max_num_seqs=1,
        max_num_batched_tokens=1024,
        max_model_len=1024,
        skip_tokenizer_init=True,
    )
    _enable_mamba_chunk_invariant_prototype(scheduler)

    (prefill_request,) = create_requests(
        num_requests=1,
        num_tokens=513,
        req_ids=["prefill"],
    )
    scheduler.add_request(prefill_request)

    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {"prefill": 256}

    for expected_tokens in (256, 1):
        scheduler.update_from_output(
            output,
            ModelRunnerOutput(
                req_ids=["prefill"],
                req_id_to_index={"prefill": 0},
                sampled_token_ids=[[]],
                logprobs=None,
                prompt_logprobs_dict={},
                pooler_output=[],
            ),
        )
        output = scheduler.schedule()
        assert output.num_scheduled_tokens == {"prefill": expected_tokens}


def test_next_step_reservation_guarantees_prefill_progress(
    tmp_path: Path,
) -> None:
    scheduler = create_scheduler(
        model=_offline_opt_model(tmp_path),
        max_num_seqs=2,
        max_num_batched_tokens=256,
        max_model_len=1024,
        skip_tokenizer_init=True,
    )
    _enable_mamba_chunk_invariant_prototype(scheduler)

    (decode_request,) = create_requests(
        num_requests=1,
        num_tokens=1,
        req_ids=["decode"],
    )
    scheduler.add_request(decode_request)
    first = scheduler.schedule()
    assert first.num_scheduled_tokens == {"decode": 1}
    scheduler.update_from_output(first, _model_output(scheduler))

    (prefill_request,) = create_requests(
        num_requests=1,
        num_tokens=512,
        req_ids=["prefill"],
    )
    scheduler.add_request(prefill_request)

    blocked = scheduler.schedule()
    assert blocked.num_scheduled_tokens == {"decode": 1}
    assert scheduler._mamba_chunk_reservation_pending
    scheduler.update_from_output(blocked, _model_output(scheduler))

    progress = scheduler.schedule()
    assert progress.num_scheduled_tokens == {"prefill": 256}
    assert not scheduler._mamba_chunk_reservation_pending
    scheduler.update_from_output(progress, _model_output(scheduler))
    decode_progress = scheduler.schedule()
    assert decode_progress.num_scheduled_tokens == {"decode": 1}
    assert scheduler._mamba_chunk_reservation_pending

    scheduler.update_from_output(decode_progress, _model_output(scheduler))
    second_prefill_chunk = scheduler.schedule()
    assert second_prefill_chunk.num_scheduled_tokens == {"prefill": 256}
    assert not scheduler._mamba_chunk_reservation_pending


def test_reservation_packs_multiple_prefills_and_then_resumes_decode(
    tmp_path: Path,
) -> None:
    scheduler = create_scheduler(
        model=_offline_opt_model(tmp_path),
        max_num_seqs=3,
        max_num_batched_tokens=512,
        max_model_len=1024,
        skip_tokenizer_init=True,
    )
    _enable_mamba_chunk_invariant_prototype(scheduler)

    (decode_request,) = create_requests(
        num_requests=1,
        num_tokens=1,
        req_ids=["decode"],
    )
    scheduler.add_request(decode_request)
    first = scheduler.schedule()
    assert first.num_scheduled_tokens == {"decode": 1}
    scheduler.update_from_output(first, _model_output(scheduler))

    prefill_requests = create_requests(
        num_requests=2,
        num_tokens=512,
        req_ids=["prefill-0", "prefill-1"],
    )
    for request in prefill_requests:
        scheduler.add_request(request)

    # The first attempt cannot fit the second complete chunk after decode.
    # It arms a one-shot reservation for the following scheduler step.
    blocked = scheduler.schedule()
    assert blocked.num_scheduled_tokens == {
        "decode": 1,
        "prefill-0": 256,
    }
    assert scheduler._mamba_chunk_reservation_pending
    scheduler.update_from_output(blocked, _model_output(scheduler))

    # The reserved step skips decode and packs both eligible prefills. Each
    # request still executes exactly one physical scan chunk.
    packed = scheduler.schedule()
    assert packed.num_scheduled_tokens == {
        "prefill-0": 256,
        "prefill-1": 256,
    }
    assert not scheduler._mamba_chunk_reservation_pending
    scheduler.update_from_output(packed, _model_output(scheduler))

    # The reservation is one-shot: decode resumes on the next attempt.
    resumed = scheduler.schedule()
    assert resumed.num_scheduled_tokens["decode"] == 1


def test_reservation_is_one_shot_without_an_eligible_prefill(
    tmp_path: Path,
) -> None:
    scheduler = create_scheduler(
        model=_offline_opt_model(tmp_path),
        max_num_seqs=2,
        max_num_batched_tokens=256,
        max_model_len=1024,
        skip_tokenizer_init=True,
    )
    _enable_mamba_chunk_invariant_prototype(scheduler)

    (decode_request,) = create_requests(
        num_requests=1,
        num_tokens=1,
        req_ids=["decode"],
    )
    scheduler.add_request(decode_request)
    first = scheduler.schedule()
    assert first.num_scheduled_tokens == {"decode": 1}
    scheduler.update_from_output(first, _model_output(scheduler))

    scheduler._mamba_chunk_reservation_pending = True
    reserved = scheduler.schedule()
    assert reserved.num_scheduled_tokens == {}
    assert not scheduler._mamba_chunk_reservation_pending

    resumed = scheduler.schedule()
    assert resumed.num_scheduled_tokens == {"decode": 1}


def test_reservation_is_one_shot_for_waiting_decode(tmp_path: Path) -> None:
    scheduler = create_scheduler(
        model=_offline_opt_model(tmp_path),
        max_num_seqs=1,
        max_num_batched_tokens=256,
        max_model_len=1024,
        skip_tokenizer_init=True,
    )
    _enable_mamba_chunk_invariant_prototype(scheduler)

    (decode_request,) = create_requests(
        num_requests=1,
        num_tokens=1,
        req_ids=["waiting-decode"],
    )
    decode_request.append_output_token_ids([1000])
    decode_request.num_computed_tokens = 1
    scheduler.add_request(decode_request)

    scheduler._mamba_chunk_reservation_pending = True
    reserved = scheduler.schedule()
    assert reserved.num_scheduled_tokens == {}
    assert not scheduler._mamba_chunk_reservation_pending

    resumed = scheduler.schedule()
    assert resumed.num_scheduled_tokens == {"waiting-decode": 1}

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NIXL P/D for Mamba hybrids in align mode with ``ssm_boundary_state_xfer``:
the decode pod caches the prompt's last block-boundary state, so a later turn
sent to it alone gets a prefix hit. Drives real P and D schedulers."""

from unittest.mock import patch

import pytest

from vllm.platforms import current_platform
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher
from vllm.v1.kv_cache_interface import KVCacheGroupSpec
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus

from .utils import (
    EOS_TOKEN_ID,
    create_model_runner_output,
    create_request,
    create_scheduler,
    create_vllm_config,
    make_kv_cache_config,
)

pytestmark = pytest.mark.cpu_test

BLOCK_SIZE = 16
FA_GROUP, MAMBA_GROUP = 0, 1


@pytest.fixture(autouse=True)
def _cpu_device_type():
    with patch.object(current_platform, "device_type", "cpu"):
        yield


def _make_scheduler(
    kv_role: str, num_mamba_groups: int = 1, ssm_boundary_state_xfer: bool = True
):
    vllm_config = create_vllm_config(
        block_size=BLOCK_SIZE,
        max_num_batched_tokens=256,
        kv_role=kv_role,
        kv_connector_extra_config={"ssm_boundary_state_xfer": ssm_boundary_state_xfer},
    )
    vllm_config.scheduler_config.disable_hybrid_kv_cache_manager = False
    vllm_config.scheduler_config.async_scheduling = False
    vllm_config.cache_config.mamba_cache_mode = "align"
    kv_cache_config = make_kv_cache_config(
        block_size=BLOCK_SIZE,
        mamba_enabled=True,
        mamba_cache_mode="align",
        num_blocks=200,
    )
    mamba_group = kv_cache_config.kv_cache_groups[MAMBA_GROUP]
    for i in range(1, num_mamba_groups):
        kv_cache_config.kv_cache_groups.append(
            KVCacheGroupSpec([f"mamba_extra{i}"], mamba_group.kv_cache_spec)
        )
    return create_scheduler(
        vllm_config, kv_cache_config=kv_cache_config, hash_block_size=BLOCK_SIZE
    )


def _make_request(request_id: str, prompt: list[int], max_tokens: int) -> Request:
    # Ensures the NONE hash is initialized, as create_request does.
    create_request(num_tokens=1, block_size=BLOCK_SIZE)
    sampling_params = SamplingParams(max_tokens=max_tokens)
    sampling_params.update_from_generation_config({}, EOS_TOKEN_ID)
    return Request(
        request_id=request_id,
        prompt_token_ids=list(prompt),
        sampling_params=sampling_params,
        pooling_params=None,
        mm_features=None,
        block_hasher=get_request_block_hasher(BLOCK_SIZE, sha256),
    )


def _run_prefill(
    prefiller, prompt: list[int], request_id: str = "p-1"
) -> tuple[Request, dict]:
    """Prefill ``prompt`` on the producer until it finishes; return the
    request and the kv_transfer_params handed to the decoder."""
    request = _make_request(request_id, prompt, max_tokens=1)
    request.kv_transfer_params = dict(do_remote_prefill=False, do_remote_decode=True)
    prefiller.add_request(request)
    for _ in range(8):
        scheduler_output = prefiller.schedule()
        if request.num_computed_tokens >= request.num_prompt_tokens:
            model_output = create_model_runner_output([request])
        else:
            model_output = ModelRunnerOutput(
                req_ids=[request.request_id],
                req_id_to_index={request.request_id: 0},
                sampled_token_ids=[[]],
            )
        engine_outputs = prefiller.update_from_output(scheduler_output, model_output)
        if engine_outputs and engine_outputs[0].outputs:
            return request, engine_outputs[0].outputs[0].kv_transfer_params
    raise AssertionError("prefill did not finish")


def _run_decode(
    decoder,
    prompt: list[int],
    params: dict,
    num_output: int,
    request_id: str = "d-1",
):
    """Receive ``prompt`` from the prefiller, decode ``num_output`` tokens and
    finish. Returns the local SSM block ids the decoder asked to receive into
    and the generated tokens."""
    request = _make_request(request_id, prompt, max_tokens=num_output)
    request.kv_transfer_params = dict(
        params, do_remote_prefill=True, do_remote_decode=False
    )
    decoder.add_request(request)

    scheduler_output = decoder.schedule()
    recv = scheduler_output.kv_connector_metadata.reqs_to_recv[request.request_id]
    local_ssm_block_ids = list(recv.local_block_ids[MAMBA_GROUP])
    decoder.update_from_output(
        scheduler_output,
        create_model_runner_output([], finished_recving={request.request_id}),
    )

    output_tokens: list[int] = []
    for i in range(num_output):
        scheduler_output = decoder.schedule()
        token = 1000 + i
        last = i == num_output - 1
        decoder.update_from_output(
            scheduler_output,
            create_model_runner_output([request], use_eos=last, token_id=token),
        )
        output_tokens.append(EOS_TOKEN_ID if last else token)
    decoder.schedule()
    assert request.is_finished()
    return local_ssm_block_ids, output_tokens


def _decode_pod_hit(decoder, prompt: list[int], output_tokens: list[int]) -> int:
    turn2 = prompt + output_tokens + [500 + i for i in range(10)]
    request = _make_request("d-next", turn2, max_tokens=1)
    _, num_hit_tokens, _ = decoder.kv_cache_manager.get_computed_blocks(request)
    return num_hit_tokens


def _turn1_prompt(num_tokens: int) -> list[int]:
    return [7 + i for i in range(num_tokens)]


def test_decode_pod_hits_prompt_boundary_after_nixl_receive():
    """58-token prompt, block 16: P computes 57 tokens and also sends its state
    at 48, so the decode pod resumes the next turn from 48."""
    prefiller = _make_scheduler("kv_producer")
    decoder = _make_scheduler("kv_consumer")
    prompt = _turn1_prompt(58)

    _, params = _run_prefill(prefiller, prompt)
    assert params["remote_ssm_boundary_tokens"] == 48
    assert len(params["remote_block_ids"][MAMBA_GROUP]) == 2

    local_ssm_block_ids, output_tokens = _run_decode(
        decoder, prompt, params, num_output=3
    )
    assert len(local_ssm_block_ids) == 2
    assert _decode_pod_hit(decoder, prompt, output_tokens) == 48


@pytest.mark.parametrize(
    "p_xfer,d_xfer", [(False, False), (False, True), (True, False)]
)
def test_boundary_state_needs_flag_on_both_sides(p_xfer, d_xfer):
    """Off by default; with the flag on only one side, the decode pod receives
    only the running state as today and caches no boundary state."""
    prefiller = _make_scheduler("kv_producer", ssm_boundary_state_xfer=p_xfer)
    decoder = _make_scheduler("kv_consumer", ssm_boundary_state_xfer=d_xfer)
    prompt = _turn1_prompt(58)

    _, params = _run_prefill(prefiller, prompt)
    local_ssm_block_ids, output_tokens = _run_decode(
        decoder, prompt, params, num_output=2
    )
    assert len(local_ssm_block_ids) == 1
    assert _decode_pod_hit(decoder, prompt, output_tokens) == 0


def test_prompt_ending_on_boundary_sends_running_state_only():
    """49 tokens: P computes exactly 48, so its running state is the boundary
    state and one slot is enough."""
    prefiller = _make_scheduler("kv_producer")
    decoder = _make_scheduler("kv_consumer")
    prompt = _turn1_prompt(49)

    _, params = _run_prefill(prefiller, prompt)
    assert "remote_ssm_boundary_tokens" not in params
    assert len(params["remote_block_ids"][MAMBA_GROUP]) == 1

    _, output_tokens = _run_decode(decoder, prompt, params, num_output=2)
    assert _decode_pod_hit(decoder, prompt, output_tokens) == 48


def test_prefiller_holds_boundary_state_until_sent():
    """The boundary block must not be recycled while D may still read it."""
    prefiller = _make_scheduler("kv_producer")
    request, params = _run_prefill(prefiller, _turn1_prompt(58))
    boundary = prefiller.kv_cache_manager.block_pool.blocks[
        params["remote_block_ids"][MAMBA_GROUP][0]
    ]
    assert boundary.ref_cnt > 0

    scheduler_output = prefiller.schedule()
    prefiller.update_from_output(
        scheduler_output,
        create_model_runner_output([], finished_sending={request.request_id}),
    )
    prefiller.schedule()
    assert boundary.ref_cnt == 0
    assert boundary.block_hash is not None


def test_aborted_prefill_releases_boundary_pin():
    prefiller = _make_scheduler("kv_producer")
    pool = prefiller.kv_cache_manager.block_pool
    num_free = pool.get_num_free_blocks()
    request = _make_request("p-1", _turn1_prompt(58), max_tokens=1)
    request.kv_transfer_params = dict(do_remote_prefill=False, do_remote_decode=True)
    prefiller.add_request(request)
    scheduler_output = prefiller.schedule()
    prefiller.update_from_output(
        scheduler_output,
        ModelRunnerOutput(
            req_ids=[request.request_id],
            req_id_to_index={request.request_id: 0},
            sampled_token_ids=[[]],
        ),
    )
    prefiller.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED)
    assert pool.get_num_free_blocks() == num_free


def test_decoder_local_hit_past_boundary_reserves_nothing():
    """A decode pod that already caches the prompt has the boundary state."""
    prefiller = _make_scheduler("kv_producer")
    decoder = _make_scheduler("kv_consumer")
    prompt = _turn1_prompt(58)

    _, params = _run_prefill(prefiller, prompt)
    _run_decode(decoder, prompt, params, num_output=2)

    _, params = _run_prefill(prefiller, prompt, request_id="p-2")
    local_ssm_block_ids, _ = _run_decode(
        decoder, prompt, params, num_output=2, request_id="d-2"
    )
    assert len(local_ssm_block_ids) == 1


def test_every_mamba_group_sends_and_caches_its_boundary_state():
    """Hybrids such as Qwen3.5 have several GDN groups; the hit needs each."""
    prefiller = _make_scheduler("kv_producer", num_mamba_groups=3)
    decoder = _make_scheduler("kv_consumer", num_mamba_groups=3)
    prompt = _turn1_prompt(58)

    _, params = _run_prefill(prefiller, prompt)
    assert [len(g) for g in params["remote_block_ids"]] == [4, 2, 2, 2]

    _, output_tokens = _run_decode(decoder, prompt, params, num_output=3)
    assert _decode_pod_hit(decoder, prompt, output_tokens) == 48

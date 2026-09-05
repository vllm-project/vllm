# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pickle

from vllm.sampling_params import SamplingParams
from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy
from vllm.v1.core.sched.output import (
    CachedRequestData,
    KVConnectorBlockState,
    NewRequestData,
    ScheduledEncoderInputStats,
    SchedulerOutput,
    pack_scheduler_output_for_execute_model_fast_path,
    unpack_scheduler_output_from_execute_model_fast_path,
)


def _make_scheduler_output() -> SchedulerOutput:
    return SchedulerOutput(
        scheduled_new_reqs=[
            NewRequestData(
                req_id="req-0",
                prompt_token_ids=[1, 2, 3],
                mm_features=[],
                sampling_params=SamplingParams(max_tokens=16),
                pooling_params=None,
                block_ids=([10, 11], [20]),
                num_computed_tokens=0,
                lora_request=None,
                prompt_is_token_ids=[True, True, True],
                prefill_token_ids=[1, 2, 3],
            )
        ],
        scheduled_cached_reqs=CachedRequestData(
            req_ids=["req-1"],
            resumed_req_ids={"req-1"},
            new_token_ids=[[42]],
            all_token_ids={"req-1": [7, 42]},
            new_block_ids=[([30],)],
            num_computed_tokens=[128],
            num_output_tokens=[1],
        ),
        num_scheduled_tokens={"req-0": 3, "req-1": 1},
        total_num_scheduled_tokens=4,
        scheduled_spec_decode_tokens={"req-1": [99]},
        scheduled_encoder_inputs={"req-0": [0]},
        num_common_prefix_blocks=[2, 0],
        finished_req_ids={"req-done"},
        free_encoder_mm_hashes=["mm-hash"],
        scheduled_encoder_input_stats=ScheduledEncoderInputStats(
            num_inputs=1,
            output_tokens=512,
        ),
        preempted_req_ids={"req-preempt"},
        has_structured_output_requests=True,
        pending_structured_output_tokens=False,
        num_invalid_spec_tokens={"req-1": 0},
        new_block_ids_to_zero=[100, 101],
        kv_cache_block_copies=[KVCacheBlockCopy(src_block_id=1, dst_block_id=2)],
        kv_connector_block_state=KVConnectorBlockState(
            block_ids={"req-1": ([10, 11], [20])},
            boundary_state_offloads={"req-1": [(0, 5, 16)]},
        ),
        has_sync_kv_loads=True,
        num_spec_tokens_to_schedule=2,
    )


def test_scheduler_output_rpc_pack_round_trip() -> None:
    original = _make_scheduler_output()
    payload = pack_scheduler_output_for_execute_model_fast_path(original)
    restored = unpack_scheduler_output_from_execute_model_fast_path(payload)
    assert restored == original


def test_scheduler_output_rpc_pack_pickle_compatible() -> None:
    original = _make_scheduler_output()
    payload = pack_scheduler_output_for_execute_model_fast_path(original)
    round_trip = pickle.loads(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
    restored = unpack_scheduler_output_from_execute_model_fast_path(round_trip)
    assert restored == original

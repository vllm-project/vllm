# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import unittest
from unittest.mock import MagicMock

import torch

from vllm.config import DeviceConfig, VllmConfig
from vllm.multimodal.inputs import (
    MultiModalFeatureSpec,
    MultiModalKwargsItem,
    PlaceholderRange,
)
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import FinishReason
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus, StreamingUpdate
from vllm.v1.structured_output import StructuredOutputManager

STOP_TOKEN = 128001


class DummyRequest(Request):
    def __init__(
        self,
        request_id,
        resumable=True,
        prompt_token_ids=None,
        mm_features: list[MultiModalFeatureSpec] | None = None,
        max_tokens: int | None = 16,
    ):
        super().__init__(
            request_id=request_id,
            prompt_token_ids=prompt_token_ids if prompt_token_ids is not None else [],
            sampling_params=SamplingParams(
                stop_token_ids=[STOP_TOKEN], max_tokens=max_tokens
            ),
            pooling_params=None,
            mm_features=mm_features,
            resumable=resumable,
        )


def create_scheduler() -> Scheduler:
    vllm_config = VllmConfig(device_config=DeviceConfig("cpu"))
    vllm_config.model_config = MagicMock()
    vllm_config.model_config.skip_tokenizer_init = True
    vllm_config.model_config.is_multimodal_model = False
    vllm_config.model_config.is_encoder_decoder = False
    vllm_config.model_config.is_diffusion = False
    vllm_config.model_config.max_model_len = 1024
    vllm_config.model_config.enable_return_routed_experts = False
    vllm_config.cache_config = MagicMock()
    vllm_config.cache_config.num_gpu_blocks = 1000
    vllm_config.cache_config.enable_prefix_caching = False
    kv_cache_config = KVCacheConfig(
        num_blocks=1000,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(
                    block_size=16, num_kv_heads=1, head_size=1, dtype=torch.float32
                ),
            )
        ],
    )
    return Scheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(vllm_config),
        block_size=16,
        hash_block_size=16,
    )


class TestStreamingScheduler(unittest.TestCase):
    def test_add_request(self):
        scheduler = create_scheduler()

        request = DummyRequest(
            request_id="test_request",
            resumable=True,
        )

        scheduler.add_request(request)

        assert "test_request" in scheduler.requests
        assert request.status == RequestStatus.WAITING
        assert len(scheduler.waiting) == 1

        next_request = DummyRequest(
            request_id="test_request",
            resumable=True,
        )
        scheduler.add_request(next_request)

        assert next_request.status == RequestStatus.WAITING
        assert len(scheduler.requests["test_request"].streaming_queue) == 1

    def test_update_request_as_session_max_token(self):
        scheduler = create_scheduler()

        session = DummyRequest(
            request_id="session",
            prompt_token_ids=[1, 2, 3],
        )
        session.num_computed_tokens = len(session.prompt_token_ids)
        session.max_tokens = 10  # Initial max_tokens
        session._output_token_ids = [1] * 10  # reach max_tokens

        new_request = DummyRequest(
            request_id="session",
            prompt_token_ids=[4, 5, 6],
        )
        new_request.sampling_params = SamplingParams(max_tokens=10)
        new_request.max_tokens = 10  # Additional max_tokens from new request

        update = StreamingUpdate.from_request(new_request)
        scheduler._update_request_as_session(session, update)

        assert session.sampling_params.max_tokens == 10
        # _update_request_as_session clears output tokens first, so
        # max_tokens = num_output_tokens (0) + update.max_tokens (10) = 10
        assert session.max_tokens == 10

        session.num_computed_tokens = len(session.prompt_token_ids)

        # Simulate generating 5 more output tokens
        session._output_token_ids = [1] * 5
        new_request2 = DummyRequest(
            request_id="session",
            prompt_token_ids=[7, 8, 9],
        )
        new_request2.sampling_params = SamplingParams(max_tokens=10)
        new_request2.max_tokens = 10
        update2 = StreamingUpdate.from_request(new_request2)
        scheduler._update_request_as_session(session, update2)

        assert session.sampling_params.max_tokens == 10
        # Again, output tokens are cleared first, so max_tokens = 0 + 10 = 10
        assert session.max_tokens == 10

    def test_update_request_as_session(self):
        scheduler = create_scheduler()

        session = DummyRequest(
            request_id="session",
            prompt_token_ids=[1, 2, 3],
        )
        session.num_computed_tokens = len(session.prompt_token_ids)

        new_request = DummyRequest(
            request_id="session",
            prompt_token_ids=[4, 5, 6],
        )
        new_request.sampling_params = SamplingParams(max_tokens=10)

        update = StreamingUpdate.from_request(new_request)
        scheduler._update_request_as_session(session, update)

        assert session.prompt_token_ids == [1, 2, 3, 4, 5, 6]
        assert session._all_token_ids == [1, 2, 3, 4, 5, 6]
        assert session.sampling_params.max_tokens == 10
        assert session.status == RequestStatus.WAITING

    def test_update_request_as_session_with_multimodal(self):
        scheduler = create_scheduler()

        mm_feature = MultiModalFeatureSpec(
            data=MultiModalKwargsItem.dummy(),
            modality="audio",
            identifier="",
            mm_position=PlaceholderRange(offset=1, length=1),
        )
        session = DummyRequest(
            request_id="session",
            prompt_token_ids=[1, 2, 3],
            mm_features=[mm_feature],
        )
        session.num_computed_tokens = len(session.prompt_token_ids)

        mm_feature = MultiModalFeatureSpec(
            data=MultiModalKwargsItem.dummy(),
            modality="audio",
            identifier="",
            mm_position=PlaceholderRange(offset=2, length=1),
        )
        new_request = DummyRequest(
            request_id="session",
            prompt_token_ids=[4, 5, 6, 7],
            mm_features=[mm_feature],
        )
        update = StreamingUpdate.from_request(new_request)
        scheduler._update_request_as_session(session, update)

        assert len(session.mm_features) == 2
        assert session.mm_features[0].mm_position.offset == 1
        # 2 + len([1, 2, 3])
        assert session.mm_features[1].mm_position.offset == 5

    def test_process_streaming_requests_with_finish_session(self):
        """Test that a non-resumable request signals stream completion.

        With the new streaming API, completion is signaled by closing/finishing
        the input generator. When a non-resumable request is added to a session
        in WAITING_FOR_STREAMING_REQ state, the session is finished immediately
        with FINISHED_ABORTED status.
        """
        scheduler = create_scheduler()

        session = DummyRequest(
            request_id="session",
            prompt_token_ids=[1, 2, 3],
            resumable=True,
        )
        scheduler.add_request(session)
        session.status = RequestStatus.WAITING_FOR_STREAMING_REQ
        session.num_computed_tokens = len(session.prompt_token_ids)

        # A non-resumable request signals stream completion
        close_request = DummyRequest(
            request_id="session",
            prompt_token_ids=[0],
            resumable=False,
            max_tokens=1,
        )
        scheduler.add_request(close_request)

        # The session should be immediately finished (stream completed)
        assert session.status == RequestStatus.FINISHED_ABORTED
        # The session should be removed from the scheduler
        assert session.request_id not in scheduler.requests

    def test_streaming_request_session_update(self):
        """Test that a resumable request updates a waiting session directly.

        When a session is in WAITING_FOR_STREAMING_REQ state and a new resumable
        request arrives, the update is applied directly via _update_request_as_session,
        not queued.
        """
        scheduler = create_scheduler()

        session = DummyRequest(
            request_id="session",
            prompt_token_ids=[1, 2, 3],
            resumable=True,
        )
        scheduler.add_request(session)
        session.status = RequestStatus.WAITING_FOR_STREAMING_REQ
        session.num_computed_tokens = len(session.prompt_token_ids)

        next_request = DummyRequest(
            request_id="session",
            prompt_token_ids=[4, 5],
            resumable=True,
        )

        scheduler.add_request(next_request)

        # With the new behavior, when session is in WAITING_FOR_STREAMING_REQ,
        # the update is applied directly (not queued), and session status
        # becomes WAITING
        assert session.status == RequestStatus.WAITING
        assert session.prompt_token_ids == [1, 2, 3, 4, 5]

        _ = scheduler.schedule()

        assert session.status == RequestStatus.RUNNING

    def test_update_request_as_session_with_output_tokens(self):
        scheduler = create_scheduler()

        session = DummyRequest(
            request_id="session",
            prompt_token_ids=[1, 2, 3],  # 3 prompt tokens
        )
        session.append_output_token_ids([10, 11])
        """
        The last output token (11) hasn't been "scheduled" yet, so `num_computed_tokens`
        only includes: 3 prompt + 1 output (the 10) = 4
        """
        session.num_computed_tokens = 4

        new_request = DummyRequest(
            request_id="session",
            prompt_token_ids=[4, 5],
        )

        update = StreamingUpdate.from_request(new_request)
        scheduler._update_request_as_session(session, update)

        # _update_request_as_session keeps computed output tokens (they become
        # part of the prompt) and only discards the final uncomputed sampled
        # token. Computed output token 10 is kept, uncomputed token 11 is
        # discarded.
        assert session._all_token_ids == [1, 2, 3, 10, 4, 5]
        assert session.prompt_token_ids == [1, 2, 3, 10, 4, 5]
        # Output tokens list is cleared
        assert session._output_token_ids == []
        # num_computed_tokens is unchanged (KV cache still valid for computed
        # tokens)
        assert session.num_computed_tokens == 4
        # Verify that the next schedule will only process the new prompt tokens
        # num_new_tokens = num_tokens - num_computed_tokens = 6 - 4 = 2
        num_new_tokens = session.num_tokens - session.num_computed_tokens
        assert num_new_tokens == 2

    def test_streaming_e2e_lifecycle(self):
        """
        Comprehensive integration test covering complete streaming request lifecycle
        including scheduler state management and aliasing bug prevention.

        FULL LIFECYCLE:
        ================
        CYCLE 1 (Initial Decode):
        1. Add streaming request (seq_id=0) with prompt tokens [1,2,3]
        2. Schedule() creates NewRequestData with prompt_token_ids
        3. Model runner caches this prompt_token_ids reference (simulated)
        4. Model executes and generates output token 10
        5. update_from_output() appends token 10 to request._all_token_ids
        6. Request transitions to RUNNING state

        CYCLE 2 (Continue Decode):
        7. Schedule() again - request is now in scheduled_cached_reqs (not new)
        8. Model runner uses CACHED state to calculate num_tokens
        9. Model generates output token (STOP_TOKEN)
        10. update_from_output() appends STOP_TOKEN to request._all_token_ids
        11. Request transitions to WAITING_FOR_STREAMING_REQ

        CYCLE 3 (New Streaming Request):
        12. Add new streaming request (seq_id=1) with prompt tokens [4,5]
        13. Scheduler merges into session, creates NewRequestData again
        14. Model runner caches new prompt_token_ids reference
        15. Verify cached state from Cycle 1 wasn't corrupted by mutations

        CRITICAL BUG PREVENTION:
        ========================
        Without .copy() in _create_new_request_data():
        - Cycle 1 Step 3: cached_state["prompt_token_ids"] aliases
            request._all_token_ids
        - Cycle 1 Step 5: When appending token 10, cached state mutates:
            [1,2,3] -> [1,2,3,10]
        - Cycle 2 Step 8: num_tokens = len([1,2,3,10]) + len([10])
            = 5 (WRONG! Should be 4)
        - Cycle 2: Discard logic would see seq_lens=4 < num_tokens=5
            -> INCORRECTLY DISCARDS

        With .copy() in _create_new_request_data():
        - Cycle 1 Step 3: cached_state["prompt_token_ids"] is independent copy
        - Cycle 1 Step 5: Only request._all_token_ids mutates, cached stays [1,2,3]
        - Cycle 2 Step 8: num_tokens = len([1,2,3]) + len([10]) = 4 (CORRECT)
        - Cycle 2: Discard logic works correctly
        """
        scheduler = create_scheduler()

        # ═══════════════════════════════════════════════════════════════════
        # CYCLE 1: Initial Request Scheduling and First Decode
        # ═══════════════════════════════════════════════════════════════════

        session = DummyRequest(
            request_id="session",
            prompt_token_ids=[1, 2, 3],
        )
        scheduler.add_request(session)

        # Step 2: Schedule creates NewRequestData
        scheduler_output_cycle1 = scheduler.schedule()

        # Verify request is in scheduled_new_reqs (first time scheduling)
        assert len(scheduler_output_cycle1.scheduled_new_reqs) == 1
        new_req_data_cycle1 = scheduler_output_cycle1.scheduled_new_reqs[0]
        assert new_req_data_cycle1.prompt_token_ids == [1, 2, 3]
        assert (
            scheduler_output_cycle1.num_scheduled_tokens[session.request_id] == 3
        )  # [1, 2, 3]
        assert (
            session.request_id
            not in scheduler_output_cycle1.scheduled_cached_reqs.req_ids
        )

        # Step 3: Simulate model runner caching the prompt_token_ids
        # This simulates gpu_model_runner.py:706-720 CachedRequestState creation
        # The model runner makes a copy of prompt_token_ids when creating
        # CachedRequestState
        cached_state_cycle1 = {
            "req_id": session.request_id,
            "prompt_token_ids": list(
                new_req_data_cycle1.prompt_token_ids
            ),  # Explicit copy
            "output_token_ids": [],
            "num_computed_tokens": 0,
        }

        # Store original for verification
        original_cached_prompt_cycle1 = cached_state_cycle1["prompt_token_ids"].copy()

        # Step 4-5: Model execution generates token, scheduler updates request
        output_token_1 = 10
        cached_state_cycle1["output_token_ids"].append(output_token_1)

        mro_cycle1 = ModelRunnerOutput(
            req_ids=[session.request_id],
            req_id_to_index={session.request_id: 0},
            sampled_token_ids=[[output_token_1]],
            logprobs=None,
            prompt_logprobs_dict={session.request_id: None},
            pooler_output=[],
        )
        session.num_computed_tokens = len(session.prompt_token_ids)
        eco_dict_cycle1 = scheduler.update_from_output(
            scheduler_output_cycle1, mro_cycle1
        )

        # Step 6: Verify request state after Cycle 1
        eco_cycle1 = eco_dict_cycle1[session.client_index].outputs[0]
        assert eco_cycle1.finish_reason is None  # Not stopped yet
        assert session.status == RequestStatus.RUNNING
        assert session in scheduler.running
        assert session._all_token_ids == [1, 2, 3, 10]  # Mutation happened here

        # CRITICAL ASSERTION: Cached prompt_token_ids must NOT have changed
        assert (
            cached_state_cycle1["prompt_token_ids"] == original_cached_prompt_cycle1
        ), (
            f"ALIASING BUG DETECTED in Cycle 1! "
            f"cached_state['prompt_token_ids'] was mutated from "
            f"{original_cached_prompt_cycle1} to "
            f"{cached_state_cycle1['prompt_token_ids']}. "
            f"This means _create_new_request_data() didn't call .copy()!"
        )
        assert cached_state_cycle1["prompt_token_ids"] is not session._all_token_ids, (
            "ALIASING BUG! cached_state['prompt_token_ids'] is the same object as "
            "session._all_token_ids. They must be independent copies."
        )

        # ═══════════════════════════════════════════════════════════════════
        # CYCLE 2: Continue Decoding (Using Cached State)
        # ═══════════════════════════════════════════════════════════════════

        # Step 7: Schedule again - now request uses cached state
        scheduler_output_cycle2 = scheduler.schedule()

        # Verify request is NOT in scheduled_new_reqs (already cached)
        assert not scheduler_output_cycle2.scheduled_new_reqs
        assert (
            session.request_id in scheduler_output_cycle2.scheduled_cached_reqs.req_ids
        )
        assert (
            scheduler_output_cycle2.num_scheduled_tokens[session.request_id] == 1
        )  # Only the output token [10]

        # Step 8: Calculate num_tokens like gpu_model_runner.py:1284 does
        # This is where the bug would manifest!
        num_tokens_cycle2 = len(cached_state_cycle1["prompt_token_ids"]) + len(
            cached_state_cycle1["output_token_ids"]
        )

        # CRITICAL ASSERTION: num_tokens must be correct (3 prompt + 1 output = 4)
        # Without .copy(), cached_state["prompt_token_ids"] would be [1,2,3,10]
        # and num_tokens would incorrectly be 5, causing the discard bug
        expected_num_tokens_cycle2 = 4
        assert num_tokens_cycle2 == expected_num_tokens_cycle2, (
            f"DISCARD BUG WOULD TRIGGER! num_tokens calculation is wrong. "
            f"Expected {expected_num_tokens_cycle2}, got {num_tokens_cycle2}. "
            f"cached_state['prompt_token_ids'] = "
            f"{cached_state_cycle1['prompt_token_ids']} (should be [1,2,3], not [1,2,3,"
            f"10]). Without .copy(), this would be 5 = len([1,2,3,10]) + len([10]). "
            f"Discard logic would see: seq_lens={session.num_computed_tokens} "
            f"< num_tokens={num_tokens_cycle2}, triggering incorrect discard!"
        )

        # Step 9-10: Model generates STOP_TOKEN, scheduler updates
        output_token_2 = STOP_TOKEN
        cached_state_cycle1["output_token_ids"].append(output_token_2)

        mro_cycle2 = ModelRunnerOutput(
            req_ids=[session.request_id],
            req_id_to_index={session.request_id: 0},
            sampled_token_ids=[[output_token_2]],
            logprobs=None,
            prompt_logprobs_dict={session.request_id: None},
            pooler_output=[],
        )
        eco_dict_cycle2 = scheduler.update_from_output(
            scheduler_output_cycle2, mro_cycle2
        )

        # Step 11: Verify request transitioned to WAITING_FOR_STREAMING_REQ
        eco_cycle2 = eco_dict_cycle2[session.client_index].outputs[0]
        assert eco_cycle2.finish_reason == FinishReason.STOP
        assert session.status == RequestStatus.WAITING_FOR_STREAMING_REQ
        # Sessions paused for streaming input are blocked-waiting, so they
        # live in the skipped_waiting queue rather than the main waiting queue.
        assert session in scheduler.skipped_waiting
        assert session._all_token_ids == [1, 2, 3, 10, STOP_TOKEN]

        # CRITICAL ASSERTION: Cached prompt_token_ids STILL must not have changed
        assert cached_state_cycle1["prompt_token_ids"] == [1, 2, 3], (
            f"ALIASING BUG DETECTED in Cycle 2! "
            f"cached_state['prompt_token_ids'] = "
            f"{cached_state_cycle1['prompt_token_ids']} (should still be [1,2,3]). "
            f"Mutations from update_from_output() leaked through!"
        )

        # ═══════════════════════════════════════════════════════════════════
        # CYCLE 3: New Streaming Request (Session Continuation)
        # ═══════════════════════════════════════════════════════════════════

        # Step 12: Add new streaming request with seq_id=1
        new_request = DummyRequest(
            request_id="session",
            prompt_token_ids=[4, 5],
        )
        scheduler.add_request(new_request)

        # With the new streaming API, when session is in WAITING_FOR_STREAMING_REQ,
        # the update is applied directly via _update_request_as_session (not queued).
        # The session status becomes WAITING after the update is applied.
        assert session.status == RequestStatus.WAITING

        # Step 13: Scheduler schedules the updated session
        scheduler_output_cycle3 = scheduler.schedule()

        # Verify scheduler created NewRequestData with merged prompt_token_ids
        assert len(scheduler_output_cycle3.scheduled_new_reqs) == 1
        assert (
            scheduler_output_cycle3.scheduled_new_reqs[0].prompt_token_ids
            == session.prompt_token_ids
        )
        assert (
            scheduler_output_cycle3.num_scheduled_tokens[session.request_id] == 2
        )  # Only new tokens [4, 5]
        # Computed output tokens are kept (become part of prompt), only the
        # final uncomputed sampled token (STOP_TOKEN) is discarded
        assert session._all_token_ids == [1, 2, 3, 10, 4, 5]
        assert session.prompt_token_ids == [1, 2, 3, 10, 4, 5]  # Includes kept output
        assert session._output_token_ids == []  # Output tokens are cleared

        # Step 14: Model runner caches NEW prompt_token_ids reference
        # The model runner makes a copy of prompt_token_ids when creating
        # CachedRequestState
        new_req_data_cycle3 = scheduler_output_cycle3.scheduled_new_reqs[0]
        cached_state_cycle3 = {
            "req_id": session.request_id,
            "prompt_token_ids": list(
                new_req_data_cycle3.prompt_token_ids
            ),  # Explicit copy
            "output_token_ids": [],
            "num_computed_tokens": session.num_computed_tokens,
        }

        # Step 15: FINAL CRITICAL VERIFICATION
        # The old cached state from Cycle 1 must still be unchanged
        assert cached_state_cycle1["prompt_token_ids"] == [1, 2, 3], (
            f"PERSISTENT ALIASING BUG! Even after new scheduling cycle, "
            f"old cached_state was mutated to "
            f"{cached_state_cycle1['prompt_token_ids']}. This proves the aliasing bug "
            f"exists!"
        )

        # The new cached state must be independent
        assert cached_state_cycle3["prompt_token_ids"] is not session._all_token_ids, (
            "ALIASING BUG in Cycle 3! Cached state is aliased to _all_token_ids."
        )

        # Both cached states must be independent of each other
        assert (
            cached_state_cycle1["prompt_token_ids"]
            is not cached_state_cycle3["prompt_token_ids"]
        ), "Cached states from different cycles should be independent objects."

    def test_running_session_on_waiting_queue_does_not_kill_engine(self):
        """Regression for issue #52693.

        A resumable streaming session can occupy both `self.running` and a
        waiting queue in one `schedule()` call (pipelined continuation while
        chunked prefill is still in flight). The waiting-loop status gate
        then raises `RuntimeError: Invalid request status: RUNNING`, which
        EngineCore treats as fatal.

        This test plants that illegal dual membership directly and requires
        `schedule()` to isolate the session instead of crashing.
        """
        # Issue #52693 is a scheduler-state bug, not a model-forward bug.
        # A CPU scheduler with dummy KV config is enough to reach the gate.
        # Reuse the file-local helper so this test matches neighboring cases.
        scheduler = create_scheduler()

        # The live crash used a continuation still mid-prefill: computed
        # tokens strictly less than prompt tokens, and zero sampled output.
        # A 32-token prompt is long enough to split that way without being
        # so long that the running loop exhausts the token budget.
        prompt_token_ids = list(range(32))
        # One request object is required: dual membership is the same
        # Request sitting in two queues, not two Requests sharing an id.
        # DummyRequest is resumable by default; pass True so the intent is
        # obvious if that default ever changes.
        session = DummyRequest(
            # Stable id so num_scheduled_tokens lookups are readable.
            # This is not a second request: add_request later reuses it.
            request_id="sess-52693",
            # Full continuation prompt; we will mark only half computed.
            # Distinct ids make a prompt-vs-output mix-up obvious in diffs.
            prompt_token_ids=prompt_token_ids,
            # resumable=True is the streaming-session path in add_request.
            # Non-resumable requests never enter the update-and-requeue loop.
            resumable=True,
        )

        # add_request registers the session in scheduler.requests and
        # enqueues it as WAITING so the first schedule() can allocate KV
        # blocks. Without that allocation, the running loop on the second
        # schedule() would preempt or fail before we reach the status gate.
        scheduler.add_request(session)
        # First schedule() pops the session from waiting, appends it to
        # running, and sets status=RUNNING. We do not call
        # update_from_output, so num_computed_tokens stays at the pre-step
        # value and the session is still prefilling from the scheduler's
        # point of view: the same shape as the instrumented crash.
        first_output = scheduler.schedule()
        # Confirm the setup actually entered the running/prefill state.
        # If this fails, later assertions would be testing the wrong path.
        # status is the field the waiting-loop gate inspects.
        assert session.status == RequestStatus.RUNNING
        # Membership in the running list is what the first loop iterates.
        # The bug is dual membership, so this list must already hold it.
        assert session in scheduler.running
        # First step must have committed tokens under this id. Otherwise
        # the second step is not a continuation of an in-flight prefill.
        assert session.request_id in first_output.num_scheduled_tokens
        # After the first schedule the waiting queue must be empty, otherwise
        # we would be stacking a second legal waiter rather than planting
        # the illegal duplicate-running membership the issue reports.
        assert len(scheduler.waiting) == 0

        # Mirror the crash snapshot: 16 of 32 prompt tokens computed, no
        # decode tokens yet. The running loop will therefore still have
        # leftover prefill work and will put this id in num_scheduled_tokens
        # before the waiting loop runs.
        session.num_computed_tokens = 16
        # Dual membership: the same RUNNING object is also on waiting.
        # This is what the pipelined continuation race produces in production.
        # FCFSRequestQueue.append does not check status, so this is legal at
        # the queue API and fatal at the waiting-loop status gate.
        scheduler.waiting.add_request(session)

        # On unfixed main this raises RuntimeError and kills EngineCore.
        # After the fix it must return a SchedulerOutput instead.
        # Do not wrap in assertRaises: survival of this call is the test.
        scheduler_output = scheduler.schedule()

        # The session may still be running (prefill not finished), but the
        # engine must have survived. A raise would have failed the line
        # above; this asserts the output is a real schedule, not None.
        assert scheduler_output is not None
        # num_scheduled_tokens is a dict keyed by request_id, so a double
        # commit cannot appear as two entries. We still require this id to
        # be scheduled this step: the running loop owns it.
        scheduled_for_session = scheduler_output.num_scheduled_tokens.get(
            session.request_id
        )
        # None would mean the running loop dropped the in-flight prefill.
        # The waiting loop must skip, not cancel, the already-scheduled id.
        assert scheduled_for_session is not None
        # Token count can be the leftover prefill (16) from the running
        # loop. The important property is a single dict entry, not a
        # second waiting-loop allocation on top of it.
        assert session.request_id in scheduler_output.num_scheduled_tokens
        # Count how many times this request object sits in running. The
        # waiting loop currently appends before it raises, so a naive
        # "append then raise" path would duplicate the list entry if the
        # raise were ever swallowed. After a correct skip, it stays once.
        running_occurrences = sum(1 for req in scheduler.running if req is session)
        # Exactly one running-list slot: skip the waiting duplicate, do
        # not finish_requests (that path KeyErrors in _update_after_schedule).
        assert running_occurrences == 1

    def test_continuation_stays_queued_while_running(self):
        """Continuations buffer in streaming_queue until the session leaves RUNNING.

        Root-cause contract for #52693: add_request must not merge a pipelined
        chunk into an in-flight session, and must not put that session on a
        waiting queue while it still occupies self.running.
        """
        # CPU scheduler is enough: this is queue-membership and prompt
        # identity, not a forward pass. Dummy KV config matches neighbors.
        scheduler = create_scheduler()
        # Three prompt tokens so a later merge of [4, 5] is obvious if it
        # happens too early. Distinct values detect "applied while running".
        # resumable=True is the session path that owns streaming_queue.
        session = DummyRequest(
            # Same id as the continuation below: this is one session.
            # A different id would be a new request, not a pipelined chunk.
            request_id="sess-52693-buffer",
            # In-flight prompt. Must remain exactly this list until we
            # leave RUNNING. [4, 5] must not appear here after add_request.
            prompt_token_ids=[1, 2, 3],
            # Non-resumable ids never get a streaming_queue; the buffer
            # rule would not apply and this test would be meaningless.
            resumable=True,
        )
        # First add_request is the session start: WAITING, empty queue.
        # Without this, schedule() has nothing to move into RUNNING.
        scheduler.add_request(session)
        # schedule() pops WAITING, appends running, sets RUNNING, and
        # allocates KV. The current chunk is now in flight.
        scheduler.schedule()
        # The buffer rule is keyed on RUNNING / self.running, not WAITING.
        # If this is not RUNNING, add_request would take the paused path.
        assert session.status == RequestStatus.RUNNING
        # Membership in the list is the second in-flight check in
        # add_request (status can theoretically lag the list).
        assert session in scheduler.running

        # Pipelined next turn: same request_id, new tokens only.
        # This is the live audio+vision arrival while prefill is unfinished.
        continuation = DummyRequest(
            # Must match session.request_id so add_request finds existing.
            # A new id would enqueue a second request, not a continuation.
            request_id="sess-52693-buffer",
            # Delta tokens. If they leak into session.prompt_token_ids
            # before we leave RUNNING, add_request merged too early.
            prompt_token_ids=[4, 5],
            # Must be resumable so StreamingUpdate is queued, not treated
            # as a duplicate-id abort of a one-shot request.
            resumable=True,
        )
        # This is the call that used to apply the delta while RUNNING.
        # After the fix it must only append to streaming_queue.
        scheduler.add_request(continuation)

        # Prompt must still be the in-flight chunk. Merging here is the
        # bug: it grows num_prompt_tokens while status stays RUNNING.
        assert session.prompt_token_ids == [1, 2, 3]
        # The delta lives on the queue until we leave RUNNING.
        # Length 1: exactly the continuation we just added, nothing else.
        assert len(session.streaming_queue) == 1
        # Must not appear on the main waiting queue: that is one half of
        # dual membership and would hit the RUNNING status gate.
        assert session not in scheduler.waiting
        # Must not appear on skipped_waiting either: _handle_stopped_request
        # has not run yet, so a pause enqueue would be premature.
        assert session not in scheduler.skipped_waiting
        # Still owned by the running loop for this chunk. Losing this
        # list slot would mean we finished or preempted too early.
        assert session in scheduler.running
        # Status must remain RUNNING; add_request must not pause us.
        assert session.status == RequestStatus.RUNNING

    def test_queued_continuation_applies_only_after_leaving_running(self):
        """After stop, a queued continuation is applied by promotion, not by stop.

        _handle_stopped_request must pause into WAITING_FOR_STREAMING_REQ
        without calling _update_request_as_session. The next schedule()
        promotes, merges the queue, and only then schedules the new tokens.
        """
        # Same dummy scheduler as the buffer test. We drive one decode
        # token then STOP so stop handling runs with a non-empty queue.
        scheduler = create_scheduler()
        # Distinct request_id from the buffer test so failures are easy
        # to attribute. Prompt [1, 2, 3] matches the e2e lifecycle numbers.
        session = DummyRequest(
            # Unique id: this test both queues and later merges.
            request_id="sess-52693-apply",
            # In-flight prompt before the pipelined [4, 5] continuation.
            prompt_token_ids=[1, 2, 3],
            # Session path: stop must pause, not free the request.
            resumable=True,
        )
        # Session start. Needed before schedule() can allocate KV blocks.
        scheduler.add_request(session)
        # Prefill step. We keep the SchedulerOutput for update_from_output.
        scheduler_output = scheduler.schedule()
        # Match the e2e lifecycle: computed tokens cover the prompt before
        # the first sampled token is applied in update_from_output.
        # Otherwise stop/discard math in _update_request_as_session drifts.
        session.num_computed_tokens = len(session.prompt_token_ids)
        # Non-stop first token so the turn stays RUNNING after this step.
        # STOP on the first sample would pause before we queue the chunk.
        first_token = 10
        # Deliver the sampled token the way EngineCore does after execute.
        scheduler.update_from_output(
            scheduler_output,
            ModelRunnerOutput(
                req_ids=[session.request_id],
                req_id_to_index={session.request_id: 0},
                sampled_token_ids=[[first_token]],
                logprobs=None,
                prompt_logprobs_dict={session.request_id: None},
                pooler_output=[],
            ),
        )
        # Session is still RUNNING (turn not stopped). Queue the next chunk
        # now, while in flight, which is the case _handle_stopped_request
        # used to apply too early.
        assert session.status == RequestStatus.RUNNING
        # Same id as session: this is a continuation, not a new request.
        scheduler.add_request(
            DummyRequest(
                request_id="sess-52693-apply",
                # Tokens that must not appear on the prompt until after
                # we leave RUNNING and promotion merges the queue.
                prompt_token_ids=[4, 5],
                resumable=True,
            )
        )
        # Queue holds the continuation; stop handling must not pop it.
        assert len(session.streaming_queue) == 1
        # Prompt still the original chunk. [4, 5] would mean apply-on-add.
        assert session.prompt_token_ids == [1, 2, 3]

        # Second step samples STOP. update_from_output must pause, not merge.
        scheduler_output = scheduler.schedule()
        # STOP_TOKEN is DummyRequest's stop id. This is a real turn end,
        # not a planted status, so _handle_stopped_request actually runs.
        scheduler.update_from_output(
            scheduler_output,
            ModelRunnerOutput(
                req_ids=[session.request_id],
                req_id_to_index={session.request_id: 0},
                sampled_token_ids=[[STOP_TOKEN]],
                logprobs=None,
                prompt_logprobs_dict={session.request_id: None},
                pooler_output=[],
            ),
        )
        # Left RUNNING via the normal stop path. Still paused: the queued
        # continuation is not applied inside _handle_stopped_request.
        assert session.status == RequestStatus.WAITING_FOR_STREAMING_REQ
        # Caller removes stopped_running_reqs after handle returns False.
        # Still being in running here would be the dual-membership setup.
        assert session not in scheduler.running
        # Paused sessions live in skipped_waiting, not the main queue.
        assert session in scheduler.skipped_waiting
        # Queue still has the continuation. Empty would mean stop applied
        # it (the old _handle_stopped_request behavior).
        assert len(session.streaming_queue) == 1
        # Prompt still lacks [4, 5]; those apply on the next schedule().
        assert 4 not in session.prompt_token_ids
        # Check 5 separately so a partial merge is visible in the failure.
        assert 5 not in session.prompt_token_ids

        # Promotion sees the non-empty queue, merges, status becomes WAITING,
        # then this same schedule() admits the new tokens.
        scheduler_output = scheduler.schedule()
        # After promote+schedule the session is in flight on the new chunk.
        assert session.status == RequestStatus.RUNNING
        # Computed output 10 is kept; STOP is discarded; [4, 5] append.
        # This is the same merge e2e used to do immediately on add_request
        # while paused; we only delayed it until after leaving RUNNING.
        assert session.prompt_token_ids == [1, 2, 3, 10, 4, 5]
        # Queue object still exists (resumable sessions keep the deque).
        assert session.streaming_queue is not None
        # Continuation was consumed by promotion. A leftover item would
        # mean we merged from somewhere else and left the queue stale.
        assert len(session.streaming_queue) == 0
        # Only the new prompt tokens need compute this step (2).
        # 0 would mean we did not schedule the merged delta; >2 would mean
        # we recomputed kept prefix tokens that KV should already hold.
        assert scheduler_output.num_scheduled_tokens[session.request_id] == 2

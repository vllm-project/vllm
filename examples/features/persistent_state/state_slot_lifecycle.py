# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Walk one stream through the persistent-state scheduler on the CPU.

A model with fixed-size per-session state declares a `PersistentStateSpec`.
This example uses a synthetic spec and no model weights. It plays the worker's
part by hand to show what the scheduler does with a state slot: admission
before any input, a slot handle per step, parking between inputs, and release
only after the worker acknowledges cleanup.

See docs/design/persistent_state.md for the contracts.

Run with:

    python examples/features/persistent_state/state_slot_lifecycle.py
"""

from copy import deepcopy

import torch

from vllm.config import CacheConfig, ModelConfig, SchedulerConfig, VllmConfig
from vllm.sampling_params import SamplingParams
from vllm.v1.core.kv_cache_planning import (
    get_global_kv_cache_groups,
    get_kv_cache_configs,
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.state_scheduler import StateScheduler
from vllm.v1.engine import EngineCoreRequest, StreamAdmissionContext
from vllm.v1.kv_cache_interface import (
    KVCacheSpec,
    PersistentStateDescriptor,
    PersistentStateSpec,
    StateCompletion,
)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

# Models with persistent state stop each turn on EOS and park for more input.
PARK_TOKEN = 7

SPEC = PersistentStateSpec(
    state_name="toy_rnn",
    descriptors=(
        # name, shape, dtype, byte offset, alignment, initializer
        PersistentStateDescriptor("hidden", (4,), torch.float32, 0, 16, torch.zeros),
        PersistentStateDescriptor("frames", (1,), torch.int64, 16, 16, torch.zeros),
    ),
    page_size_bytes=32,
)


def build_scheduler(num_pages: int) -> StateScheduler:
    # Only the model's config is fetched; no weights are loaded.
    model_config = ModelConfig(
        model="facebook/opt-125m", max_model_len=32, skip_tokenizer_init=True
    )
    cache_config = CacheConfig(block_size=1, enable_prefix_caching=False)
    # The engine core resolves the layout at startup; standalone use sets it.
    cache_config.kv_cache_layout = "LBNHC"
    vllm_config = VllmConfig(
        model_config=model_config,
        scheduler_config=SchedulerConfig(
            max_num_seqs=2,
            max_num_batched_tokens=32,
            max_model_len=32,
            is_encoder_decoder=False,
            async_scheduling=False,
        ),
        cache_config=cache_config,
    )
    specs: list[dict[str, KVCacheSpec]] = [{"state_layer": SPEC}]
    (kv_cache_config,) = get_kv_cache_configs(
        vllm_config,
        specs,
        [num_pages * SPEC.page_size_bytes],
        get_global_kv_cache_groups(vllm_config, specs),
    )
    # The engine core records the planned block (slot) count before scheduling.
    vllm_config.cache_config.num_gpu_blocks = kv_cache_config.num_blocks
    return StateScheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        block_size=1,
        structured_output_manager=StructuredOutputManager(vllm_config),
        log_stats=False,
    )


def stream_request(request_id: str, prompt_token_ids: list[int]) -> EngineCoreRequest:
    params = SamplingParams(max_tokens=4)
    params.update_from_generation_config({}, PARK_TOKEN)
    return EngineCoreRequest(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        mm_features=None,
        sampling_params=params,
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        resumable=True,
    )


def run_worker(step: SchedulerOutput, token: int) -> ModelRunnerOutput:
    """Stand in for the model runner: emit one token per request and
    acknowledge the step's slot handles and cleanups."""
    req_ids = list(step.num_scheduled_tokens)
    assert step.state_slot_handles is not None
    assert step.state_cleanup_handles is not None
    return ModelRunnerOutput(
        req_ids=req_ids,
        sampled_token_ids=[[token] for _ in req_ids],
        state_completion=StateCompletion(
            step.state_step_seq,
            deepcopy(step.state_slot_handles),
            deepcopy(step.state_cleanup_handles),
        ),
    )


def main() -> None:
    # Three state pages: one is the reserved null block, two are session slots.
    scheduler = build_scheduler(num_pages=3)
    admissions = scheduler.stream_admissions
    assert admissions is not None
    slots = admissions.manager
    print(f"free slots: {slots.num_free_slots}")

    # 1. Reserve a slot before any audio arrives (EngineCore.admit_stream).
    result = admissions.admit(stream_request("s1", []))
    print(f"admitted: {result.accepted}, status: {result.status}")

    # 2. Send input for the admitted stream. The scheduler hands the worker the
    # slot handle (engine epoch, slot, generation) to read and write.
    chunk = stream_request("s1", [1, 2, 3])
    chunk.stream_admission = StreamAdmissionContext()
    scheduler.add_request(Request.from_engine_core_request(chunk, None))
    step = scheduler.schedule()
    assert step.state_slot_handles is not None
    handle = step.state_slot_handles["s1"]["state_layer"]
    print(f"scheduled with slot {handle.slot}, generation {handle.generation}")

    # 3. The turn ends on the park token: the session keeps its slot and waits.
    scheduler.update_from_output(step, run_worker(step, PARK_TOKEN))
    print(f"occupied after park: {admissions.occupied()}")

    # 4. End the session. The slot is retired but stays occupied until the worker
    # acknowledges the cleanup handle sent in the next step.
    scheduler.finish_requests("s1", RequestStatus.FINISHED_ABORTED)
    print(f"free slots before cleanup ack: {slots.num_free_slots}")
    cleanup = scheduler.schedule()
    assert cleanup.state_cleanup_handles is not None
    print(f"cleanup handles: {list(cleanup.state_cleanup_handles)}")
    scheduler.update_from_output(cleanup, run_worker(cleanup, PARK_TOKEN))
    print(f"free slots after cleanup ack: {slots.num_free_slots}")

    # 5. A reused slot gets a new generation, so the old handle cannot reach it.
    reused = slots.allocate("s2")
    assert reused is not None
    print(f"next allocation: slot {reused.slot}, generation {reused.generation}")


if __name__ == "__main__":
    main()

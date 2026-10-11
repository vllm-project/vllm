# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time

import prometheus_client
import pytest
import torch

from tests.conftest import VllmRunner
from tests.utils import fork_new_process_for_each_test
from vllm.config import AttentionConfig, HiSparseConfig, KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.connector import (
    HiSparseConnector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.worker import (
    HiSparseConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
    MultiConnector,
)
from vllm.platforms import current_platform
from vllm.sampling_params import SamplingParams
from vllm.utils.gpu_sync_debug import gpu_sync_allowed
from vllm.v1.hisparse.coordinator import get_hisparse_coordinator

MODEL = "deepseek-ai/DeepSeek-V3.2"

HF_OVERRIDES = {
    "num_hidden_layers": 8,
    "hidden_size": 256,
    "intermediate_size": 512,
    "num_attention_heads": 8,
    "num_key_value_heads": 1,
    "n_routed_experts": 8,
    "num_experts_per_tok": 2,
    "index_topk": 128,
}


def _shrink_config(config):
    overrides = HF_OVERRIDES.copy()
    if any("MTP" in architecture for architecture in config.architectures):
        overrides.pop("num_hidden_layers")
    config.update(overrides)
    return config


def _num_hisparse_spills(runner: VllmRunner) -> int:
    client = runner.llm.llm_engine.engine_core
    coordinator = get_hisparse_coordinator(
        client.engine_core.scheduler.kv_cache_manager
    )
    return coordinator.next_spill_id


def _offload_load_bytes() -> float:
    return sum(
        sample.value
        for metric in prometheus_client.REGISTRY.collect()
        if metric.name == "vllm:kv_offload_load_bytes"
        for sample in metric.samples
        if sample.name == "vllm:kv_offload_load_bytes_total"
    )


def _get_hisparse_worker(runner: VllmRunner) -> HiSparseConnectorWorker:
    engine_core = runner.llm.llm_engine.engine_core.engine_core
    model_runner = engine_core.model_executor.driver_worker.worker.model_runner
    connector = model_runner.kv_connector.kv_connector
    if isinstance(connector, MultiConnector):
        connector = next(
            child
            for child in connector.sub_connectors
            if isinstance(child, HiSparseConnector)
        )
    assert isinstance(connector, HiSparseConnector)
    assert connector.connector_worker is not None
    return connector.connector_worker


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="HiSparse requires NVIDIA CUDA"
)
@pytest.mark.parametrize(
    "with_offloading", [False, True], ids=["standalone", "offload"]
)
@fork_new_process_for_each_test
def test_hisparse_spill_and_prefix_restore(
    monkeypatch: pytest.MonkeyPatch,
    vllm_runner: type[VllmRunner],
    with_offloading: bool,
):
    """Spilled prefixes restore and FULL-graph decode writes reach host KV.

    A regression omitted attention metadata before FULL graph replay, so decode
    completed normally while its newly written KV rows were never copied to host.
    MTP layers write after the target forward; their rows used to be mirrored
    before the drafter wrote them, so the drafter later read stale host rows.
    """
    capability = current_platform.get_device_capability()
    if capability is None or capability.major < 9:
        pytest.skip("Sparse MLA requires Hopper or newer")

    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    monkeypatch.setenv("VLLM_DEEP_GEMM_WARMUP", "skip")

    target = [1000 + i % 64 for i in range(257)]
    pressure = [
        [2000 + request_idx * 128 + i % 64 for i in range(257)]
        for request_idx in range(4)
    ]
    hisparse_connector = {
        "kv_connector": "HiSparseConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {"host_pool_gib": 1},
    }
    kv_transfer_config = KVTransferConfig(**hisparse_connector)
    if with_offloading:
        kv_transfer_config = KVTransferConfig(
            kv_connector="MultiConnector",
            kv_role="kv_both",
            kv_connector_extra_config={
                "connectors": [
                    hisparse_connector,
                    {
                        "kv_connector": "OffloadingConnector",
                        "kv_role": "kv_both",
                        "kv_connector_extra_config": {"cpu_bytes_to_use": 1 << 30},
                    },
                ]
            },
        )

    with vllm_runner(
        MODEL,
        load_format="dummy",
        hf_overrides=_shrink_config,
        attention_config=AttentionConfig(
            hisparse_config=HiSparseConfig(
                device_buffer_size=512,
            )
        ),
        kv_transfer_config=kv_transfer_config,
        block_size=64,
        max_model_len=320,
        max_num_batched_tokens=1024,
        max_num_seqs=4,
        num_gpu_blocks_override=128,
        disable_log_stats=False,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        speculative_config={"method": "mtp", "num_speculative_tokens": 3},
        compilation_config={
            "cudagraph_mode": "FULL_AND_PIECEWISE",
            "cudagraph_capture_sizes": [1, 4],
        },
    ) as runner:
        worker = _get_hisparse_worker(runner)
        engine_core = runner.llm.llm_engine.engine_core.engine_core
        model_runner = engine_core.model_executor.driver_worker.worker.model_runner
        full_graph_calls = 0
        full_replay_pending = False
        original_run_fullgraph = model_runner.cudagraph_manager.run_fullgraph

        def record_fullgraph(batch_desc):
            nonlocal full_graph_calls, full_replay_pending
            full_graph_calls += 1
            full_replay_pending = True
            return original_run_fullgraph(batch_desc)

        monkeypatch.setattr(
            model_runner.cudagraph_manager, "run_fullgraph", record_fullgraph
        )

        # The MTP layer writes its rows after the target forward, so it is
        # mirrored at the start of the next step instead of in finish_forward.
        draft_layers = set(worker._draft_layers)
        assert draft_layers
        original_finish_forward = worker.finish_forward
        original_finish_previous_step = worker._finish_previous_step
        verified_rows = {"target": 0, "draft": 0}
        copied_nonzero_kv = False
        draft_check_pending = False

        def check_host_rows(layer_indices: set[int], kind: str) -> None:
            nonlocal copied_nonzero_kv
            # Synchronize only the test's inspection, not the worker itself.
            with gpu_sync_allowed():
                torch.accelerator.synchronize()
            assert worker._row_mirrors
            for layer_index in layer_indices:
                cache = worker.cache_handles[layer_index]
                source_index = cache.runtime.resident_source_index
                for mirror in worker._row_mirrors:
                    source_row = mirror.source_starts[source_index]
                    destination_row = mirror.destination_start
                    num_rows = mirror.num_rows
                    source_block, source_offset = divmod(
                        source_row, worker.kernel_block_size
                    )
                    with gpu_sync_allowed():
                        gpu_rows = worker.resident_caches[layer_index][
                            source_block, source_offset : source_offset + num_rows
                        ].cpu()
                    host_rows = worker.host_caches[layer_index][
                        destination_row : destination_row + num_rows
                    ]
                    assert torch.equal(host_rows, gpu_rows)
                    copied_nonzero_kv |= bool(torch.count_nonzero(gpu_rows).item())
                    verified_rows[kind] += num_rows

        def finish_forward():
            nonlocal full_replay_pending, draft_check_pending
            try:
                original_finish_forward()
                if not full_replay_pending:
                    return
                target_layers = set(range(len(worker.cache_handles))) - draft_layers
                check_host_rows(target_layers, "target")
                draft_check_pending = True
            finally:
                full_replay_pending = False

        def finish_previous_step():
            nonlocal draft_check_pending
            original_finish_previous_step()
            if draft_check_pending:
                draft_check_pending = False
                check_host_rows(draft_layers, "draft")

        monkeypatch.setattr(worker, "finish_forward", finish_forward)
        monkeypatch.setattr(worker, "_finish_previous_step", finish_previous_step)

        expected = runner.generate_greedy([target], max_tokens=8)

        assert full_graph_calls > 0
        assert verified_rows["target"] > 0
        assert verified_rows["draft"] > 0
        assert copied_nonzero_kv

        runner.generate_greedy(pressure, max_tokens=8)

        assert _num_hisparse_spills(runner) > 0
        load_bytes = _offload_load_bytes()
        actual = runner.generate_greedy([target], max_tokens=8)
        runner.generate_greedy([[42]], max_tokens=1)

        assert actual == expected
        if with_offloading:
            assert _offload_load_bytes() > load_bytes


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="HiSparse requires NVIDIA CUDA"
)
@fork_new_process_for_each_test
def test_hisparse_host_exhaustion_defers_requests(
    monkeypatch: pytest.MonkeyPatch,
    vllm_runner: type[VllmRunner],
):
    """A full host pool defers requests instead of leaving pages GPU-only."""
    capability = current_platform.get_device_capability()
    if capability is None or capability.major < 9:
        pytest.skip("Sparse MLA requires Hopper or newer")
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    monkeypatch.setenv("VLLM_DEEP_GEMM_WARMUP", "skip")
    prompts = [
        [1000 + request_idx * 64 + i % 64 for i in range(257)]
        for request_idx in range(3)
    ]
    with vllm_runner(
        MODEL,
        load_format="dummy",
        hf_overrides=_shrink_config,
        attention_config=AttentionConfig(
            hisparse_config=HiSparseConfig(device_buffer_size=512)
        ),
        kv_transfer_config=KVTransferConfig(
            kv_connector="HiSparseConnector",
            kv_role="kv_both",
            kv_connector_extra_config={"host_pool_gib": 1},
        ),
        block_size=64,
        max_model_len=320,
        max_num_batched_tokens=512,
        max_num_seqs=3,
        num_gpu_blocks_override=128,
        enable_prefix_caching=False,
        enable_chunked_prefill=True,
        enforce_eager=True,
    ) as runner:
        expected = [runner.generate_greedy([p], max_tokens=16)[0] for p in prompts]
        engine = runner.llm.llm_engine.engine_core.engine_core
        coordinator = get_hisparse_coordinator(engine.scheduler.kv_cache_manager)
        host = coordinator.host_manager
        assert host is not None
        # Leave room for one full request plus part of another.
        spare_blocks = 8
        num_pressure = host.block_pool.get_num_free_blocks() - spare_blocks
        host.allocate_new_blocks(
            "host-pressure",
            num_pressure * host.block_size,
            num_pressure * host.block_size,
        )
        original_plan = coordinator.plan_prefix_materialization
        original_count = host.get_num_blocks_to_allocate
        checked_steps = 0
        refusals = 0

        def plan(request_id, num_computed_tokens):
            nonlocal checked_steps
            assert not any(b.is_null for b in host.req_to_blocks[request_id])
            checked_steps += 1
            original_plan(request_id, num_computed_tokens)

        def count(*args, **kwargs):
            nonlocal refusals
            required = original_count(*args, **kwargs)
            refusals += int(required > 0)
            return required

        monkeypatch.setattr(coordinator, "plan_prefix_materialization", plan)
        monkeypatch.setattr(host, "get_num_blocks_to_allocate", count)
        actual = runner.generate_greedy(prompts, max_tokens=16)

        assert checked_steps > 0
        assert refusals > 0
        assert actual == expected


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="HiSparse requires NVIDIA CUDA"
)
@fork_new_process_for_each_test
def test_hisparse_terminal_prefix_reuse(
    monkeypatch: pytest.MonkeyPatch,
    vllm_runner: type[VllmRunner],
):
    """Finished requests publish mirrored host KV after a late acknowledgement."""
    capability = current_platform.get_device_capability()
    if capability is None or capability.major < 9:
        pytest.skip("Sparse MLA requires Hopper or newer")
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    monkeypatch.setenv("VLLM_DEEP_GEMM_WARMUP", "skip")
    target = [1000 + i % 64 for i in range(257)]
    sampling = SamplingParams(temperature=0, max_tokens=1, ignore_eos=True)
    with vllm_runner(
        MODEL,
        load_format="dummy",
        hf_overrides=_shrink_config,
        attention_config=AttentionConfig(
            hisparse_config=HiSparseConfig(
                device_buffer_size=512, eager_host_mirror=True
            )
        ),
        kv_transfer_config=KVTransferConfig(
            kv_connector="HiSparseConnector",
            kv_role="kv_both",
            kv_connector_extra_config={"host_pool_gib": 0.01},
        ),
        block_size=64,
        max_model_len=320,
        max_num_batched_tokens=512,
        max_num_seqs=2,
        num_gpu_blocks_override=128,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        enforce_eager=True,
    ) as runner:
        engine = runner.llm.llm_engine.engine_core.engine_core
        coordinator = get_hisparse_coordinator(engine.scheduler.kv_cache_manager)
        worker = _get_hisparse_worker(runner)
        original_updates = worker.take_transfer_updates
        original_free = coordinator.free
        held_completions: list[int] = []
        update_calls = 0
        terminal_pending_pages = 0

        def take_updates():
            nonlocal update_calls
            with gpu_sync_allowed():
                torch.accelerator.synchronize()
            enqueued, completed = original_updates()
            held_completions.extend(completed)
            update_calls += 1
            if update_calls <= 2:
                return enqueued, []
            completed = held_completions.copy()
            held_completions.clear()
            return enqueued, completed

        def free(request_id):
            nonlocal terminal_pending_pages
            state = coordinator.request_states.get(request_id)
            if state is not None:
                terminal_pending_pages += len(state.pending_pages)
            original_free(request_id)

        def drain_pending_work():
            deadline = time.monotonic() + 10
            while coordinator.has_pending_work():
                assert time.monotonic() < deadline, "HiSparse transfers did not drain"
                runner.llm.llm_engine.step()

        monkeypatch.setattr(worker, "take_transfer_updates", take_updates)
        monkeypatch.setattr(coordinator, "free", free)
        [first] = runner.llm.generate(
            [{"prompt_token_ids": target}], sampling, use_tqdm=False
        )
        assert terminal_pending_pages == 4
        assert first.num_cached_tokens == 0
        assert coordinator.has_pending_work()
        drain_pending_work()
        assert not coordinator.has_pending_work()
        assert not held_completions
        assert update_calls > 2

        [repeated] = runner.llm.generate(
            [{"prompt_token_ids": target}], sampling, use_tqdm=False
        )
        assert repeated.num_cached_tokens == 256
        assert repeated.outputs[0].token_ids == first.outputs[0].token_ids
        drain_pending_work()
        assert not coordinator.has_pending_work()

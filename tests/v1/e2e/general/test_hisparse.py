# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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

        original_finish_forward = worker.finish_forward
        verified_full_replay_rows = 0
        copied_nonzero_kv = False

        def finish_forward():
            nonlocal full_replay_pending, verified_full_replay_rows
            nonlocal copied_nonzero_kv
            try:
                original_finish_forward()
                if not full_replay_pending:
                    return
                # Synchronize only the test's inspection, not finish_forward itself.
                with gpu_sync_allowed():
                    torch.accelerator.synchronize()
                assert worker._row_mirrors
                for layer_index, cache in enumerate(worker.cache_handles):
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
                        verified_full_replay_rows += num_rows
            finally:
                full_replay_pending = False

        monkeypatch.setattr(worker, "finish_forward", finish_forward)

        expected = runner.generate_greedy([target], max_tokens=8)

        assert full_graph_calls > 0
        assert verified_full_replay_rows > 0
        assert copied_nonzero_kv

        runner.generate_greedy(pressure, max_tokens=8)

        assert _num_hisparse_spills(runner) > 0
        load_bytes = _offload_load_bytes()
        actual = runner.generate_greedy([target], max_tokens=8)
        runner.generate_greedy([[42]], max_tokens=1)

        assert actual == expected
        if with_offloading:
            assert _offload_load_bytes() > load_bytes

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PeerWorker IPC loads through a two-rank MoE engine."""

import asyncio
import os
from collections import Counter

import pytest

from tests.utils import multi_gpu_test
from vllm import SamplingParams
from vllm.config import KVTransferConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.metrics.loggers import StatLoggerBase


@multi_gpu_test(num_gpus=2)
@pytest.mark.distributed
def test_peer_worker_restores_prefix_on_another_dp_rank():
    prompt = "Explain how a computer executes a program. " * 32

    async def run(peer):
        counters: Counter[tuple[int, str]] = Counter()
        restored = asyncio.Event()

        class Recorder(StatLoggerBase):
            def __init__(self, vllm_config, engine_index=0):
                pass

            def log_engine_initialized(self):
                pass

            def record(
                self,
                scheduler_stats,
                iteration_stats,
                mm_cache_stats=None,
                engine_idx=0,
            ):
                if scheduler_stats and scheduler_stats.kv_connector_stats:
                    for name, value in scheduler_stats.kv_connector_stats.get(
                        "counters", {}
                    ).items():
                        counters[engine_idx, name] += value
                    if (
                        counters[1, "reader_peer_tokens"] > 0
                        and counters[1, "worker_copy_bytes"] > 0
                    ):
                        restored.set()

        engine = AsyncLLM.from_engine_args(
            AsyncEngineArgs(
                model=os.getenv("MODEL_NAME", "ibm-research/PowerMoE-3b"),
                data_parallel_size=2,
                enable_expert_parallel=True,
                enable_prefix_caching=True,
                enforce_eager=True,
                async_scheduling=False,
                max_model_len=512,
                max_num_batched_tokens=512,
                max_num_seqs=2,
                kv_cache_memory_bytes=128 << 20,
                kv_transfer_config=KVTransferConfig(
                    kv_connector="PeerL1Connector", kv_role="kv_both"
                )
                if peer
                else None,
            ),
            stat_loggers=[Recorder],
        )
        try:
            outputs = []
            for rank in (0, 1):
                async for output in engine.generate(
                    prompt,
                    SamplingParams(temperature=0, max_tokens=8, ignore_eos=True),
                    request_id=f"prefix-{rank}",
                    data_parallel_rank=rank,
                ):
                    final = output
                outputs.append(final.outputs[0].token_ids)
            if peer:
                await asyncio.wait_for(restored.wait(), timeout=30)
            return outputs
        finally:
            engine.shutdown()

    async def compare():
        expected = await run(False)
        assert await run(True) == expected

    asyncio.run(compare())

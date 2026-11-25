# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PeerWorker IPC loads through a two-rank MoE engine."""

import asyncio
import os
from collections import Counter

import pytest

from tests.models.utils import check_logprobs_close
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
                        counters[1, "reader_tokens,ipc,full"] > 0
                        and counters[1, "worker_copy_bytes,ipc,full"] > 0
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
                # The KV budget is explicit; do not require a nearly empty GPU.
                gpu_memory_utilization=0.5,
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
                    SamplingParams(
                        temperature=0, max_tokens=8, ignore_eos=True, logprobs=5
                    ),
                    request_id=f"prefix-{rank}",
                    data_parallel_rank=rank,
                ):
                    final = output
                completion = final.outputs[0]
                outputs.append(
                    (completion.token_ids, completion.text, completion.logprobs)
                )
            if peer:
                await asyncio.wait_for(restored.wait(), timeout=30)
            return outputs
        finally:
            engine.shutdown()

    async def compare():
        expected = await run(False)
        check_logprobs_close(
            outputs_0_lst=expected,
            outputs_1_lst=await run(True),
            name_0="recompute",
            name_1="peer restore",
        )

    asyncio.run(compare())

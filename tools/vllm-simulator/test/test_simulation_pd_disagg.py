"""Test the PD-disaggregated benchmark runner.

One prefill and one decode VLLMWorker; the decode worker uses the native
DecodeBenchConnector as the fake P->D transfer.  Checks that:
- prefill runs only to the first token;
- decode treats all but the last prompt token as transferred KV;
- decode arrives when the prefill finishes, and the merged first-token
  latency spans both instances.
"""

import os
import random

import numpy as np
from transformers import AutoTokenizer

os.environ["VLLM_SIMULATOR_CONFIG_PATH"] = (
    os.path.dirname(__file__) + "/assets/config_vllm.json"
)

from vllm_simulator.simulation.benchmark import (  # noqa: E402
    BenchmarkConfig,
    PDDisaggBenchmarkRunner,
    from_sample_requests,
)
from vllm_simulator.simulation.vllm.vllm_worker import (  # noqa: E402
    EngineArgs,
    VLLMWorker,
)

# After vllm_worker: its import installs the hooks vLLM must see first.
from vllm.benchmarks.datasets import RandomDataset  # noqa: E402
from vllm.config import KVTransferConfig  # noqa: E402

MODEL_PATH = "Qwen/Qwen3-8B"
NUM_PROMPTS = 8
OUTPUT_LEN = 8


def _engine_args(**kwargs) -> EngineArgs:
    return EngineArgs(
        model=MODEL_PATH,
        block_size=16,
        max_model_len=2048,
        num_gpu_blocks_override=200,
        **kwargs,
    )


def test_pd_disagg_benchmark():
    random.seed(0)
    np.random.seed(0)

    prefill_worker = VLLMWorker(_engine_args(), name="prefill0")
    decode_worker = VLLMWorker(
        _engine_args(
            kv_transfer_config=KVTransferConfig(
                kv_connector="DecodeBenchConnector", kv_role="kv_both"
            ),
        ),
        name="decode0",
    )
    runner = PDDisaggBenchmarkRunner(
        prefill_workers=[prefill_worker], decode_workers=[decode_worker]
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    requests = from_sample_requests(RandomDataset().sample(
        tokenizer=tokenizer,
        num_requests=NUM_PROMPTS,
        input_len=384,
        output_len=OUTPUT_LEN,
        range_ratio={"input": 1 / 3, "output": 0.0},
    ))

    metrics = runner.benchmark(
        BenchmarkConfig(request_rate=10, ignore_request_timestamp=True),
        requests=requests,
    )
    assert metrics["completed"] == NUM_PROMPTS

    prefill_stats = {s["rid"]: s for s in prefill_worker.get_request_stats()}
    decode_stats = {s["rid"]: s for s in decode_worker.get_request_stats()}
    merged = runner.get_request_stats()
    assert len(merged) == NUM_PROMPTS

    for stat in merged:
        p = prefill_stats[stat["rid"]]
        d = decode_stats[stat["rid"]]
        assert p["output_length"] == 1
        assert d["ext_kv_hit_len"] == d["input_length"] - 1
        assert d["created_time"] == p["last_event_time"]
        assert len(stat["gen_token_latencies"]) == d["output_length"]
        assert stat["gen_token_latencies"][0] > sum(p["gen_token_latencies"])
        assert stat["last_event_time"] == d["last_event_time"]

    runner.shutdown()


if __name__ == "__main__":
    test_pd_disagg_benchmark()

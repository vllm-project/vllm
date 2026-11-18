# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU offloading with an fs secondary tier on an engine whose workers span
two nodes.

Run by its own 2-node CI step: TP over this node's GPUs and PP over the two nodes,
on Ray, so the workers of the second pipeline stage are on the other node. The
scheduler and its secondary tiers read the CPU region on this node, which the
workers on the other node fill and read through the region mirror.
"""

import json
import os
import random
import time

import requests
from prometheus_client.parser import text_string_to_metric_families

from vllm.utils.math_utils import cdiv

from ..utils import RemoteOpenAIServer

MODEL = "facebook/opt-125m"
BLOCK_SIZE = 16
NUM_GPU_BLOCKS = 160
CHUNK_TOKENS = 256
# Four whole chunks, plus the token computed on a hit; the generated tokens
# fill no further chunk.
PROMPT_CHUNKS = 4
CACHED_TOKENS = PROMPT_CHUNKS * CHUNK_TOKENS
MAX_TOKENS = 8
# Enough prompts to take every GPU block, evicting an earlier prompt.
GPU_FILLERS = cdiv(NUM_GPU_BLOCKS * BLOCK_SIZE, CACHED_TOKENS + 1)
FS_TIER = "1:fs"


def _prompt(seed: int) -> list[int]:
    rng = random.Random(seed)
    return [rng.randrange(1000, 40000) for _ in range(CACHED_TOKENS + 1)]


def _metric(server: RemoteOpenAIServer, name: str, **labels: str) -> float:
    text = requests.get(server.url_for("metrics")).text
    return sum(
        sample.value
        for family in text_string_to_metric_families(text)
        for sample in family.samples
        if sample.name in (name, f"{name}_total")
        and all(sample.labels.get(k) == v for k, v in labels.items())
    )


def _wait_for(condition, what: str, timeout_s: float = 120) -> None:
    deadline = time.monotonic() + timeout_s
    while not condition():
        assert time.monotonic() < deadline, f"timed out waiting for {what}"
        time.sleep(0.1)


def _cpu_capacity_chunks(server: RemoteOpenAIServer) -> int:
    """Chunks of the CPU tier, as the offloading connector publishes them."""
    capacity: list[int] = []

    def published() -> bool:
        text = requests.get(server.url_for("metrics")).text
        capacity.extend(
            int(sample.labels["cpu_kv_cache_size_tokens"]) // CHUNK_TOKENS
            for family in text_string_to_metric_families(text)
            if family.name == "vllm:kv_offload_config_info"
            for sample in family.samples
            if sample.labels.get("tier") == "0:primary"
        )
        return bool(capacity)

    _wait_for(published, "the CPU tier's capacity")
    return capacity[0]


def test_tiers_restore_across_nodes(tmp_path, num_gpus_available):
    """A prompt evicted from the GPU is restored from the CPU tier, and once
    evicted from the CPU tier too, from the fs tier, through the workers on
    the other node. Each restore loads the bytes the GPU computed, so it must
    match a GPU prefix cache hit exactly."""
    fs_root = tmp_path / "fs"
    kv_transfer_config = {
        "kv_connector": "OffloadingConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
            "spec_name": "TieringOffloadingSpec",
            "cpu_bytes_to_use": 256 << 20,
            "block_size": CHUNK_TOKENS,
            "secondary_tiers": [{"type": "fs", "root_dir": str(fs_root)}],
        },
    }
    args = [
        "--tensor-parallel-size",
        str(num_gpus_available),
        "--pipeline-parallel-size",
        "2",
        "--distributed-executor-backend",
        "ray",
        "--max-model-len",
        "2048",
        # The model is small; a fixed share keeps the server from sizing its
        # KV cache to memory that other processes on the node may hold.
        "--gpu-memory-utilization",
        "0.2",
        "--enforce-eager",
        "--block-size",
        str(BLOCK_SIZE),
        "--num-gpu-blocks-override",
        str(NUM_GPU_BLOCKS),
        "--enable-prompt-tokens-details",
        "--kv-transfer-config",
        json.dumps(kv_transfer_config),
    ]
    # The CI nodes are containers joined over TCP: keep the mirror's UCX
    # transfers on TCP, as between hosts without RDMA.
    env = {"UCX_TLS": os.environ.get("UCX_TLS", "tcp,cuda_copy")}

    with RemoteOpenAIServer(MODEL, args, env_dict=env, max_wait_seconds=900) as server:
        client = server.get_client()

        def complete(seed: int) -> tuple[tuple, int]:
            out = client.completions.create(
                model=MODEL,
                prompt=_prompt(seed),
                max_tokens=MAX_TOKENS,
                temperature=0,
                logprobs=1,
            )
            choice = out.choices[0]
            assert choice.finish_reason == "length", choice
            assert out.usage.prompt_tokens_details is not None
            completion = (
                choice.text,
                choice.logprobs.tokens,
                choice.logprobs.token_logprobs,
            )
            return completion, out.usage.prompt_tokens_details.cached_tokens

        def wait_for_fs_tier(num_prompts: int) -> None:
            # Every chunk stored reaches the fs tier, so once all have, every
            # store issued, and every eviction it made in the CPU tier, is done.
            _wait_for(
                lambda: (
                    sum(1 for _ in fs_root.rglob("*.bin"))
                    == num_prompts * PROMPT_CHUNKS
                ),
                "the stores to reach the fs tier",
            )

        def restore() -> tuple[tuple, float]:
            """Request prompt 0 again; return its completion and the bytes read
            from the fs tier meanwhile."""
            gpu_hits = _metric(server, "vllm:prefix_cache_hits")
            external_hits = _metric(server, "vllm:external_prefix_cache_hits")
            fs_reads = _metric(
                server, "vllm:kv_offload_tiering_read_bytes", tier=FS_TIER
            )
            completion, cached = complete(0)
            assert cached == CACHED_TOKENS
            _wait_for(
                lambda: (
                    _metric(server, "vllm:external_prefix_cache_hits")
                    >= external_hits + CACHED_TOKENS
                ),
                "the restore to be counted",
            )
            # Evicted from the GPU: every cached token came from the connector.
            assert _metric(server, "vllm:prefix_cache_hits") == gpu_hits
            fs_read = (
                _metric(server, "vllm:kv_offload_tiering_read_bytes", tier=FS_TIER)
                - fs_reads
            )
            return completion, fs_read

        _, cached = complete(0)
        assert cached == 0
        reference, cached = complete(0)
        assert cached == CACHED_TOKENS
        wait_for_fs_tier(1)
        cpu_chunks = _cpu_capacity_chunks(server)
        # The CPU tier keeps prompt 0 while the GPU fillers are stored.
        assert cpu_chunks >= (1 + GPU_FILLERS) * PROMPT_CHUNKS, cpu_chunks

        for seed in range(1, 1 + GPU_FILLERS):
            complete(seed)
        wait_for_fs_tier(1 + GPU_FILLERS)
        completion, fs_read = restore()
        assert completion == reference
        assert fs_read == 0

        # Prompts newer than prompt 0 that fill the CPU tier evict it.
        num_prompts = 1 + GPU_FILLERS + cdiv(cpu_chunks, PROMPT_CHUNKS)
        for seed in range(1 + GPU_FILLERS, num_prompts):
            complete(seed)
        wait_for_fs_tier(num_prompts)
        completion, fs_read = restore()
        assert completion == reference
        assert fs_read > 0

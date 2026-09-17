# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU KV offload round trip on the layer-pruned GLM-5.3-Flash.

Guards the kpool-tail scratch group handling of the offload connectors: the
tail group has block_size = index_kpool (4) and is not prefix-cacheable, so a
connector that treats every KV group alike asserts on the first store
(``resolve_block_hashes``: 4 % hash_block_size) or at config time
(``tokens_per_block=4 not divisible by tokens_per_hash``). Passes once the
connector skips non-prefix-cacheable groups (vllm-project/vllm#56810 for
SimpleCPUOffloadConnector).
"""

import os

import pytest
import torch

from vllm import LLM, SamplingParams, TokensPrompt
from vllm.config import KVTransferConfig
from vllm.platforms import current_platform

PRUNED_MODEL = os.getenv("GLM5_PRUNED_MODEL", "JaredforReal/GLM-5.3-Flash-4L")
MAX_MODEL_LEN = 16384
# ~2 full auto blocks (4352 tokens each at TP1) plus a partial one.
PROMPT_LEN = 9000
# Per block: 4352 * 1 KiB MLA latent + 4352 / 4 * 132 B indexer entries.
BYTES_PER_BLOCK = 4352 * 1024 + 1088 * 132
NUM_GPU_BLOCKS = 48

pytestmark = pytest.mark.skipif(
    not (
        current_platform.is_cuda()
        and torch.cuda.is_available()
        and torch.cuda.get_device_capability()[0] in (9, 10)
    ),
    reason="GLM-5.3-Flash sparse indexer needs SM90/SM100",
)


def _connector_config(name: str) -> KVTransferConfig:
    if name == "SimpleCPUOffloadConnector":
        extra = {"cpu_bytes_to_use": 4 << 30}
    else:
        extra = {"spec_name": "CPUOffloadingSpec", "cpu_bytes_to_use": 4 << 30}
    return KVTransferConfig(
        kv_connector=name, kv_role="kv_both", kv_connector_extra_config=extra
    )


def _metric(llm: LLM, name: str) -> float:
    return sum(m.value for m in llm.get_metrics() if m.name == name)


@pytest.mark.parametrize(
    "connector", ["SimpleCPUOffloadConnector", "OffloadingConnector"]
)
def test_cpu_offload_round_trip(connector: str):
    llm = LLM(
        model=PRUNED_MODEL,
        max_model_len=MAX_MODEL_LEN,
        enforce_eager=True,
        # Text-only: skips vision-tower profiling (slow JIT) and its encoder cache.
        limit_mm_per_prompt={"image": 0, "video": 0},
        kv_cache_memory_bytes=NUM_GPU_BLOCKS * BYTES_PER_BLOCK,
        max_num_batched_tokens=MAX_MODEL_LEN,
        max_num_seqs=4,
        enable_prefix_caching=True,
        disable_log_stats=False,  # get_metrics() needs stat logging
        kv_transfer_config=_connector_config(connector),
    )
    g = torch.Generator().manual_seed(0)
    prompt = TokensPrompt(
        prompt_token_ids=torch.randint(
            1000, 100000, (PROMPT_LEN,), generator=g
        ).tolist()
    )
    sp = SamplingParams(temperature=0, max_tokens=8)

    cold = llm.generate([prompt], sp, use_tqdm=False)[0].outputs[0].token_ids
    assert llm.reset_prefix_cache()
    warm = llm.generate([prompt], sp, use_tqdm=False)[0].outputs[0].token_ids

    hits = _metric(llm, "vllm:external_prefix_cache_hits")
    assert hits > 0, "second prefill did not load any KV from CPU"
    assert list(cold) == list(warm)

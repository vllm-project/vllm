# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E2E test: routed-experts capture stays correct under KV CPU offload.

The routing data of offloaded blocks must survive GPU block reuse and be
restored when blocks are loaded back from CPU (the scheduler mirrors KV
transfer jobs into the RoutedExpertsManager's offload mirror). CUDA graphs
stay enabled (no enforce_eager) to cover the captured-graph decode path.
"""

import time

import numpy as np
import pytest

from vllm import LLM, SamplingParams, TokensPrompt
from vllm.config import KVTransferConfig
from vllm.engine.arg_utils import EngineArgs

MODEL_NAME = "TitanML/tiny-mixtral"

# tiny-mixtral: 8 experts, top-2 routing, 2 hidden layers. The published
# config has sliding_window=4096; routed-experts + offload requires a
# single FullAttentionSpec group, so override sliding_window=null.
HF_OVERRIDES = {"sliding_window": None}
NUM_HIDDEN_LAYERS = 2
NUM_EXPERTS_PER_TOK = 2

BLOCK_SIZE = 16
# Small GPU block pool so filler prompts deterministically recycle the
# blocks freed by the original request.
NUM_GPU_BLOCKS = 32
NUM_PROMPT_BLOCKS = 4
PROMPT_LEN = NUM_PROMPT_BLOCKS * BLOCK_SIZE
MAX_TOKENS = 8

_RESET_CACHE_TIMEOUT = 10


def _kv_transfer_config() -> KVTransferConfig:
    return KVTransferConfig(
        kv_connector="OffloadingConnector",
        kv_role="kv_both",
        kv_connector_extra_config={"cpu_bytes_to_use": 1 << 28},
    )


@pytest.fixture(scope="module")
def llm() -> LLM:
    return LLM(
        model=MODEL_NAME,
        max_model_len=256,
        block_size=BLOCK_SIZE,
        num_gpu_blocks_override=NUM_GPU_BLOCKS,
        hf_overrides=HF_OVERRIDES,
        enable_return_routed_experts=True,
        kv_transfer_config=_kv_transfer_config(),
    )


def _wait_for_prefix_cache_reset(llm: LLM) -> None:
    """Wait for async offload transfers to finish, then reset the GPU
    prefix cache (CPU-side offloaded blocks are kept)."""
    dummy_params = SamplingParams(max_tokens=1)
    deadline = time.monotonic() + _RESET_CACHE_TIMEOUT
    while not llm.reset_prefix_cache(reset_connector=False):
        if time.monotonic() > deadline:
            raise TimeoutError(
                "reset_prefix_cache did not succeed within "
                f"{_RESET_CACHE_TIMEOUT}s - async offload may be stuck"
            )
        # Force an engine step so the scheduler polls get_finished()
        # and releases GPU blocks held by in-flight async stores.
        llm.generate([TokensPrompt(prompt_token_ids=[0])], dummy_params, use_tqdm=False)


def test_routed_experts_survive_offload_reload(llm: LLM):
    sampling_params = SamplingParams(temperature=0, max_tokens=MAX_TOKENS)
    prompt = TokensPrompt(prompt_token_ids=list(range(1, PROMPT_LEN + 1)))

    # Step A: cold run; routing is captured and prompt blocks get
    # offloaded to CPU asynchronously.
    cold = llm.generate([prompt], sampling_params, use_tqdm=False)[0]
    cold_experts = cold.outputs[0].routed_experts
    assert cold_experts is not None
    assert cold_experts.shape == (
        PROMPT_LEN + MAX_TOKENS - 1,
        NUM_HIDDEN_LAYERS,
        NUM_EXPERTS_PER_TOK,
    )

    # Step B: evict all GPU blocks; the CPU offload cache is kept, so the
    # replay below must reload from CPU.
    _wait_for_prefix_cache_reset(llm)

    # Step C: recycle the freed GPU blocks with different prompts so
    # their slot-buffer entries are overwritten. Without the offload
    # mirror, the original routing data is now destroyed. The fillers
    # cover more blocks than the whole GPU pool.
    filler_blocks = 2 * NUM_GPU_BLOCKS
    fillers = [
        TokensPrompt(
            prompt_token_ids=[1000 + i * PROMPT_LEN + j for j in range(PROMPT_LEN)]
        )
        for i in range(filler_blocks // NUM_PROMPT_BLOCKS)
    ]
    llm.generate(fillers, sampling_params, use_tqdm=False)

    # Step D: replay the original prompt. The full prompt blocks (except
    # the last, which is recomputed because hits are capped at
    # num_tokens - 1) load from CPU; their routing must come back from
    # the mirror byte-for-byte.
    replay = llm.generate([prompt], sampling_params, use_tqdm=False)[0]
    replay_experts = replay.outputs[0].routed_experts
    assert replay_experts is not None
    assert replay.outputs[0].token_ids == cold.outputs[0].token_ids

    loaded = PROMPT_LEN - BLOCK_SIZE
    assert np.array_equal(replay_experts[:loaded], cold_experts[:loaded])
    # Guard against trivially passing on zeroed buffers.
    assert replay_experts[:loaded].any()


def _engine_args(**kwargs) -> EngineArgs:
    return EngineArgs(
        model=MODEL_NAME,
        max_model_len=256,
        hf_overrides=HF_OVERRIDES,
        enable_return_routed_experts=True,
        **kwargs,
    )


def test_rejects_non_offloading_connector():
    args = _engine_args(
        kv_transfer_config=KVTransferConfig(
            kv_connector="SharedStorageConnector",
            kv_role="kv_both",
        )
    )
    with pytest.raises(ValueError, match="only supports the CPU KV offload"):
        args.create_engine_config()


def test_rejects_mismatched_offloaded_block_size():
    args = _engine_args(
        block_size=16,
        kv_transfer_config=KVTransferConfig(
            kv_connector="OffloadingConnector",
            kv_role="kv_both",
            kv_connector_extra_config={
                "cpu_bytes_to_use": 1 << 28,
                "block_size": 48,
            },
        ),
    )
    with pytest.raises(ValueError, match="to equal the GPU"):
        args.create_engine_config()


def test_rejects_non_cpu_offloading_spec():
    args = _engine_args(
        kv_transfer_config=KVTransferConfig(
            kv_connector="OffloadingConnector",
            kv_role="kv_both",
            kv_connector_extra_config={
                "cpu_bytes_to_use": 1 << 28,
                "spec_name": "TieringOffloadingSpec",
            },
        ),
    )
    with pytest.raises(ValueError, match="only supports the CPU KV offload"):
        args.create_engine_config()

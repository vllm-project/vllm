# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E2E test: routed-experts capture stays correct under KV CPU offload.

The routing data of offloaded blocks must survive GPU block reuse and be
restored when blocks are loaded back from CPU (the scheduler replays KV
transfer jobs into the RoutedExpertsManager's offload buffer). CUDA graphs
stay enabled (no enforce_eager) to cover the captured-graph decode path.
"""

import os
import time

import numpy as np
import pytest

from vllm import LLM, SamplingParams, TokensPrompt
from vllm.config import KVTransferConfig
from vllm.engine.arg_utils import EngineArgs

# A tiny full-attention Qwen3-MoE checkpoint checked into the test tree
# (real weights, ~0.7 MB). Used for the GPU end-to-end offload/reload tests
# so they run fully offline (no network / HF download). CUDA graphs stay
# enabled to cover the captured-graph decode path.
MODEL_NAME = os.path.join(os.path.dirname(__file__), "_tiny_moe_ckpt")

# A tiny full-attention MoE config checked into the test tree, used for the
# offline config-validation tests (no network / weights needed; only the HF
# config is read by create_engine_config()).
CONFIG_MODEL = os.path.join(os.path.dirname(__file__), "_tiny_moe")

# tiny Qwen3-MoE: 8 experts, top-2 routing, 2 hidden layers, full attention
# (sliding_window=null in the checked-in config). routed-experts + offload
# requires a single FullAttentionSpec group.
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


def _kv_transfer_config(
    block_size_factor: int = 1,
    disk_root: str | None = None,
) -> KVTransferConfig:
    """Build an OffloadingConnector config.

    Args:
        block_size_factor: offloaded block size = factor x GPU block size.
        disk_root: when set, use the multi-tier ``TieringOffloadingSpec``
            with a filesystem secondary tier rooted here (exercises the
            GPU<->CPU<->disk routed-experts lifecycle).
    """
    extra: dict = {"cpu_bytes_to_use": 1 << 28}
    if block_size_factor != 1:
        extra["block_size"] = BLOCK_SIZE * block_size_factor
    if disk_root is not None:
        extra["spec_name"] = "TieringOffloadingSpec"
        extra["secondary_tiers"] = [{"type": "fs", "root_dir": disk_root}]
    return KVTransferConfig(
        kv_connector="OffloadingConnector",
        kv_role="kv_both",
        kv_connector_extra_config=extra,
    )


def _build_llm(
    block_size_factor: int = 1,
    disk_root: str | None = None,
) -> LLM:
    return LLM(
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        skip_tokenizer_init=True,
        max_model_len=256,
        block_size=BLOCK_SIZE,
        num_gpu_blocks_override=NUM_GPU_BLOCKS,
        gpu_memory_utilization=0.3,
        hf_overrides=HF_OVERRIDES,
        enable_return_routed_experts=True,
        kv_transfer_config=_kv_transfer_config(block_size_factor, disk_root),
    )


@pytest.fixture(scope="module")
def llm() -> LLM:
    return _build_llm()


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


def _run_offload_reload_cycle(llm: LLM) -> None:
    """Cold-run, evict GPU blocks, recycle them, replay; assert routing
    reloads byte-for-byte from the offload buffer (and disk, if tiered).

    Shared by the factor=1 module-fixture test and the factor>1 / disk
    variants, which each build their own engine.
    """
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
    # offload buffer, the original routing data is now destroyed. The fillers
    # cover more blocks than the whole GPU pool. Token ids stay inside
    # the tiny checkpoint's vocab (1024) while differing from the
    # original prompt so blocks are recomputed, not prefix-cache hits.
    filler_blocks = 2 * NUM_GPU_BLOCKS
    num_fillers = filler_blocks // NUM_PROMPT_BLOCKS
    fillers = [
        TokensPrompt(
            prompt_token_ids=[
                100 + ((i * PROMPT_LEN + j) % 900) for j in range(PROMPT_LEN)
            ]
        )
        for i in range(num_fillers)
    ]
    llm.generate(fillers, sampling_params, use_tqdm=False)

    # Step D: replay the original prompt. The full prompt blocks (except
    # the last, which is recomputed because hits are capped at
    # num_tokens - 1) load from CPU; their routing must come back from
    # the offload buffer byte-for-byte.
    replay = llm.generate([prompt], sampling_params, use_tqdm=False)[0]
    replay_experts = replay.outputs[0].routed_experts
    assert replay_experts is not None
    assert replay.outputs[0].token_ids == cold.outputs[0].token_ids

    loaded = PROMPT_LEN - BLOCK_SIZE
    assert np.array_equal(replay_experts[:loaded], cold_experts[:loaded])
    # Guard against trivially passing on zeroed buffers.
    assert replay_experts[:loaded].any()


def test_routed_experts_survive_offload_reload(llm: LLM):
    """factor=1 (offloaded block size == GPU block size), CPU offload."""
    _run_offload_reload_cycle(llm)


@pytest.mark.parametrize("block_size_factor", [2, 3])
def test_routed_experts_survive_offload_reload_factor_gt1(block_size_factor: int):
    """factor>1: one offloaded block packs ``factor`` GPU blocks.

    The routed-experts offload buffer replays the worker's per-block sub-block
    mapping, so routing must still reload byte-for-byte. Each parametrization
    builds its own engine because the factor is fixed at construction.
    """
    llm = _build_llm(block_size_factor=block_size_factor)
    try:
        _run_offload_reload_cycle(llm)
    finally:
        del llm


def test_routed_experts_survive_disk_tiering(tmp_path):
    """GPU<->CPU<->disk: routing follows KV blocks through the secondary
    (filesystem) tier via the cascade/promote sidecar observer."""
    llm = _build_llm(disk_root=str(tmp_path / "kv_tier"))
    try:
        _run_offload_reload_cycle(llm)
    finally:
        del llm


def test_routed_experts_survive_disk_tiering_factor_gt1(tmp_path):
    """Disk tiering with factor>1: combine the sub-block offload mapping
    with the CPU<->disk sidecar lifecycle."""
    llm = _build_llm(block_size_factor=3, disk_root=str(tmp_path / "kv_tier"))
    try:
        _run_offload_reload_cycle(llm)
    finally:
        del llm


def _engine_args(**kwargs) -> EngineArgs:
    return EngineArgs(
        model=CONFIG_MODEL,
        tokenizer=CONFIG_MODEL,
        skip_tokenizer_init=True,
        max_model_len=128,
        hf_overrides=HF_OVERRIDES,
        enable_return_routed_experts=True,
        **kwargs,
    )


def test_rejects_non_offloading_connector():
    # A real, registered connector that is NOT the CPU offload connector
    # (NixlConnector is a PD-disaggregation transport) must be rejected by
    # _verify_return_routed_experts_kv_compat.
    args = _engine_args(
        kv_transfer_config=KVTransferConfig(
            kv_connector="NixlConnector",
            kv_role="kv_both",
        )
    )
    with pytest.raises(ValueError, match="only supports the CPU KV offload"):
        args.create_engine_config()


def test_accepts_factor_gt1_offloaded_block_size():
    """factor>1 (offloaded block size = N x GPU block size) is now allowed.

    The routed-experts offload buffer replays the same per-block sub-block mapping
    the worker KV copy uses, so a larger offloaded block size no longer
    needs to be rejected at config time.
    """
    args = _engine_args(
        block_size=16,
        kv_transfer_config=KVTransferConfig(
            kv_connector="OffloadingConnector",
            kv_role="kv_both",
            kv_connector_extra_config={
                "cpu_bytes_to_use": 1 << 28,
                "block_size": 48,  # factor = 3
            },
        ),
    )
    # Should not raise at config time (authoritative checks run later at
    # scheduler init with the resolved spec).
    args.create_engine_config()


def test_accepts_tiering_offloading_spec():
    """TieringOffloadingSpec (CPU primary + disk/object secondary) is allowed."""
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
    args.create_engine_config()

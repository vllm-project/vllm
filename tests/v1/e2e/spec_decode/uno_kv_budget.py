# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV-cache budgets for the Uno e2e tests, importable without a GPU.

The survivor e2e test sizes ``kv_cache_memory_bytes`` by hand, so it must sit
above the floor vLLM's admission rule enforces and below what the mixed phase's
concurrent requests need. Both e2e budgets are derived here from the pinned
model geometry, and ``tests/v1/spec_decode/test_uno_mrv2.py`` checks them
against the engine floor on CPU, so a model-pin, block-size or ``max_model_len``
change fails without a GB10.

The geometry is pinned as literals so the CPU suite stays hub-free; the e2e
module, which already requires the model, cross-checks the literals against
``AutoConfig`` for ``MODEL_REVISION``.
"""

from collections.abc import Sequence

from vllm.utils.math_utils import cdiv

MODEL_ID = "Qwen/Qwen3-8B"
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
BLOCK_SIZE = 16
DTYPE_BYTES = 2  # bfloat16

# Qwen/Qwen3-8B at MODEL_REVISION: 36 hidden layers, 8 KV heads, 128 head dim.
QWEN3_NUM_LAYERS = 36
QWEN3_NUM_KV_HEADS = 8
QWEN3_HEAD_DIM = 128

# The survivor engine runs at a shorter context so its admission floor (65
# blocks) sits well below the mixed phase's unique footprint (~165 blocks). At
# max_model_len=2048 the floor (129 blocks) nearly equals that footprint.
SURVIVOR_MAX_MODEL_LEN = 1024
SURVIVOR_HEADROOM = (5, 4)

# The finish peers must keep generating until the resident set -- the warmed
# shared prefix stored once, plus the survivor and the two finish peers -- grows
# past the 81-block budget. Counting the prefix once, the set crosses 81 blocks
# at roughly 420 peer decode tokens, so 512 leaves margin before either peer's
# natural finish. The abort peer is retired after two tokens and holds only its
# unique prompt blocks while it runs. An earlier 192-token cap never reached the
# crossing: with the shared prefix counted once the resident set peaked near 54
# blocks, so the scheduler never preempted (GB10 contract12/contract13,
# preemptions=0).
SURVIVOR_FINISH_MAX_TOKENS = 512

# The K-matrix engine keeps upstream's 4096 context on a 2 GiB budget, far above
# its 257-block floor.
MATRIX_MAX_MODEL_LEN = 4096
MATRIX_KV_BUDGET_BYTES = 2 * 1024**3


def qwen3_geometry() -> tuple[int, int, int]:
    """(layers, KV heads, head dim) of the pinned Qwen3-8B as literals."""
    return QWEN3_NUM_LAYERS, QWEN3_NUM_KV_HEADS, QWEN3_HEAD_DIM


def kv_bytes_per_block(dtype_bytes: int = DTYPE_BYTES) -> int:
    """KV bytes held by one ``BLOCK_SIZE``-token block of the pinned model."""
    num_layers, num_kv_heads, head_dim = qwen3_geometry()
    return num_layers * num_kv_heads * head_dim * 2 * dtype_bytes * BLOCK_SIZE


def engine_minimum_kv_bytes(max_model_len: int) -> int:
    """Smallest ``kv_cache_memory_bytes`` that admits one max-len request.

    ``get_kv_cache_configs`` reserves one pool block (the null block) before
    ``_check_enough_kv_cache_memory`` compares the rest with
    ``_max_memory_usage_bytes_from_groups``. For a dense full-attention model
    that need is ``cdiv(max_model_len, block_size)`` blocks of
    ``FullAttentionSpec.max_memory_usage_bytes``, so the budget must cover one
    more block than the request itself. Speculative lookahead does not raise the
    floor: ``FullAttentionSpec`` sizes a full ``max_model_len`` and Uno clamps
    positions and sequence lengths to ``max_model_len``.
    """
    return (cdiv(max_model_len, BLOCK_SIZE) + 1) * kv_bytes_per_block()


def survivor_kv_budget() -> int:
    """Survivor e2e budget: the admission floor plus 25% headroom.

    ``SURVIVOR_MAX_MODEL_LEN=1024`` gives a floor of 65 blocks and a budget of
    81.25 blocks (81 after rounding); the mixed phase is sized so that all four
    prompts are admitted together but their generations grow past the budget,
    which is what forces preemption/recompute.
    """
    numerator, denominator = SURVIVOR_HEADROOM
    return engine_minimum_kv_bytes(SURVIVOR_MAX_MODEL_LEN) * numerator // denominator


def blocks_for_tokens(tokens: int) -> int:
    """Blocks the allocator rounds a request's ``tokens`` up to."""
    return cdiv(tokens, BLOCK_SIZE)


def mixed_admission_blocks(
    prompt_tokens: Sequence[int], shared_prefix_tokens: int
) -> int:
    """Blocks resident when every mixed-phase prompt is admitted at once.

    The warmed shared prefix is stored once; each request then adds only its
    unique suffix blocks plus one block to start decoding. This is the pressure
    the scheduler sees when the peers join, so it must stay below the budget or
    the prompts would wait in the queue instead of triggering preemption.
    """
    prefix_blocks = blocks_for_tokens(shared_prefix_tokens)
    unique = sum(
        max(0, blocks_for_tokens(tokens) - prefix_blocks) for tokens in prompt_tokens
    )
    return prefix_blocks + unique + len(prompt_tokens)


def mixed_growth_blocks(
    prompt_tokens: Sequence[int],
    max_tokens: Sequence[int],
    shared_prefix_tokens: int,
) -> int:
    """Blocks resident if every request reaches its generation cap.

    The warmed shared prefix is stored once, so it is counted once here; each
    request then contributes only its unique suffix blocks. Summing
    ``blocks_for_tokens(prompt + cap)`` per request counts the shared prefix once
    per request and over-estimates the resident set, which is why an earlier
    version of this budget let the survivor's preemption gate pass its
    inequality while the engine still read ``preemptions=0`` (contract12/13).
    This is still an upper bound while the abort peer is alive, and the sum
    above the budget is what makes a running request's ``allocate_slots`` fail
    and preempt the last running peer.
    """
    prefix_blocks = blocks_for_tokens(shared_prefix_tokens)
    return prefix_blocks + sum(
        max(0, blocks_for_tokens(prompt + cap) - prefix_blocks)
        for prompt, cap in zip(prompt_tokens, max_tokens)
    )


def mixed_crossing_tokens(
    peer_prompt_tokens: int,
    shared_prefix_tokens: int,
    budget_blocks: int,
) -> int:
    """Generated tokens at which the two long peers alone outgrow the budget.

    By the time the pair has grown this far the short seed has finished and the
    abort peer is retired, so the resident set is the warmed prefix stored once
    plus each long peer's unique growth. The survivor e2e test asserts this is
    below ``SURVIVOR_FINISH_MAX_TOKENS``: if the crossing were at or past the
    cap, a peer could finish naturally before the scheduler had to preempt one,
    and the token-equality gate would pass without exercising the resume path.
    """
    prefix_blocks = blocks_for_tokens(shared_prefix_tokens)
    tokens = 0
    while True:
        unique = max(0, blocks_for_tokens(peer_prompt_tokens + tokens) - prefix_blocks)
        if prefix_blocks + 2 * unique > budget_blocks:
            return tokens
        tokens += 1

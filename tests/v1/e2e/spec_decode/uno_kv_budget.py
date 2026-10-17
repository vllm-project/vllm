# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV-cache budgets for the Uno e2e tests, importable without a GPU.

The survivor e2e test pins ``num_gpu_blocks_override`` above the floor vLLM's
admission rule enforces and below what the mixed phase's concurrent requests
need. Both e2e budgets are derived here from the pinned model geometry, and
``tests/v1/spec_decode/test_uno_mrv2.py`` checks them against the engine floor
on CPU, so a model-pin, block-size or ``max_model_len`` change fails without a
GPU.

The geometry is pinned as literals so the CPU suite stays hub-free; the e2e
module, which already requires the model, cross-checks the literals against
``AutoConfig`` for ``MODEL_REVISION``.
"""

import math
from collections.abc import Iterable, Mapping, Sequence
from itertools import combinations
from typing import Any, NamedTuple

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
# blocks) sits below the mixed phase's no-prefix-sharing footprint (153 blocks
# with the K lookahead slots the allocator reserves; it read 151 before the
# arithmetic became K-aware). At max_model_len=2048 the floor (129 blocks)
# nearly equals it.
SURVIVOR_MAX_MODEL_LEN = 1024

# Blocks granted above the engine's single-request admission floor. Four blocks
# put the budget at 69 (68 allocatable, see `allocatable_blocks`), which admits
# the four mixed-phase prompts (63 blocks) and still forces preemption for every
# interleaving of the two long peers (see `worst_case_crossing_tokens`). A wider
# margin re-opens the desynchronisation hole this replaced: with the former
# 25% headroom (81 blocks) a peer that lagged its twin by ~285 generated tokens
# let the leader reach its cap inside the pool, so `allocate_slots` never failed
# and the gate never fired.
SURVIVOR_KV_HEADROOM_BLOCKS = 4

# K for the survivor engine. This is the ONE source: the e2e passes it to
# `speculative_config`, the CPU twin builds its scheduler with it, and the
# resident-block arithmetic below reserves it. Uno's `num_lookahead_tokens` is
# exactly K (`VllmConfig.num_lookahead_tokens`), so a running request holds
# slots for its committed tokens plus one sampled token plus K.
SURVIVOR_NUM_SPECULATIVE_TOKENS = 8

# Prefix caching is disabled in the survivor test. The finish peers therefore
# keep generating until their resident sets grow past the 68 allocatable
# blocks. Growing together they cross at 279 generated tokens; with one peer
# stalled at its admission footprint the leader crosses at 551. A 640-token cap
# leaves five block-groups of margin in the worst case, so no peer can finish
# before the scheduler has to preempt one; both crossings are pinned by
# test_uno_mrv2 and printed by each run's geometry probe. The abort peer is
# retired after two tokens and holds only its own prompt blocks while it runs.
SURVIVOR_FINISH_MAX_TOKENS = 640

# Prompt lengths the survivor e2e tokenises at MODEL_REVISION. The e2e measures
# them at runtime and prints them; these literals exist so the CPU suite can
# check the geometry without the hub, and they are only allowed to be what a
# GPU receipt shows. The finish peers are 257 -- eight tokens more than the
# abort peer, which is the per-repeat cost of the two-word peer tag against the
# one-word abort tag, measured on a Blackwell (sm_121) and an Ampere (sm_86)
# GPU run, which both print `worst_case_crossing_tokens=544` and
# `growth_blocks=151`.
SURVIVOR_PROMPT_TOKENS = (171, 257, 257, 249)

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
    """Survivor e2e budget: the admission floor plus a fixed block margin.

    ``SURVIVOR_MAX_MODEL_LEN=1024`` gives a floor of 65 blocks and a budget of
    ``65 + SURVIVOR_KV_HEADROOM_BLOCKS`` = 69 blocks; the mixed phase is sized so
    that all four prompts are admitted together but the two long peers cannot
    both reach their cap inside the pool no matter how far apart their
    generation rates drift, which is what forces preemption/recompute.
    """
    floor_blocks = engine_minimum_kv_bytes(SURVIVOR_MAX_MODEL_LEN) // (
        kv_bytes_per_block()
    )
    return (floor_blocks + SURVIVOR_KV_HEADROOM_BLOCKS) * kv_bytes_per_block()


def allocatable_blocks(budget_blocks: int) -> int:
    """Blocks a request can actually be given out of a ``budget_blocks`` pool.

    ``BlockPool`` pops one block off the free queue as the null block, and
    ``get_usage`` divides by ``num_gpu_blocks - 1``. A pool pinned with
    ``num_gpu_blocks_override=69`` therefore hands out 68 blocks, so one free
    block reads as ``1 - 1/68`` = 98.529% and a full pool reads 100.000%. (On
    the historical 81-block geometry the same one-free-block state read
    98.750%, which is 79 of 80; that number belongs to that geometry only.)
    Every survivor inequality is stated against this count so the receipt and
    the arithmetic share one denominator.
    """
    return budget_blocks - 1


def usage_with_free_blocks(budget_blocks: int, free_blocks: int) -> float:
    """The occupancy ``get_kv_cache_usage`` reports for ``free_blocks`` free.

    Lets a receipt say what a printed percentage means in blocks for the pool
    actually pinned, instead of quoting a percentage from another geometry.
    """
    pool = allocatable_blocks(budget_blocks)
    return 1.0 - (free_blocks / pool)


def blocks_for_tokens(tokens: int) -> int:
    """Blocks the allocator rounds a request's ``tokens`` up to."""
    return cdiv(tokens, BLOCK_SIZE)


def resident_blocks(
    prompt_tokens: int,
    generated_tokens: int,
    num_speculative_tokens: int = SURVIVOR_NUM_SPECULATIVE_TOKENS,
) -> int:
    """Blocks a running request holds after ``generated_tokens`` tokens.

    ``allocate_slots`` reserves ``num_computed + num_new + num_lookahead``
    slots, and for Uno ``num_lookahead_tokens`` is K exactly, so the request
    holds its committed tokens plus the sampled token plus K, rounded up to
    blocks. Deriving it from K rather than a bare ``+1`` is what keeps the
    pre-gate honest when K changes: at K=1 a 257-token prompt holds 17 blocks,
    at K=8 it holds 17, and at K=16 it holds 18.
    """
    return blocks_for_tokens(
        prompt_tokens + generated_tokens + 1 + num_speculative_tokens
    )


def min_resident_blocks(
    prompt_tokens: int,
    num_speculative_tokens: int = SURVIVOR_NUM_SPECULATIVE_TOKENS,
) -> int:
    """Blocks an admitted request holds before it has generated anything.

    A request that is merely lagging, rather than preempted or finished, never
    holds less than this while it is running, which is what the worst-case
    crossing charges the slower peer.
    """
    return resident_blocks(prompt_tokens, 0, num_speculative_tokens)


def prompt_token_ids_are_pairwise_content_distinct(
    prompt_token_ids: Sequence[Sequence[int]],
) -> bool:
    """Return whether every prompt pair differs at a shared token position.

    A length-only difference is not enough: when one prompt is a prefix of the
    other, the shared token positions are identical and this returns ``False``.
    """
    return all(
        any(left != right for left, right in zip(first, second))
        for first, second in combinations(prompt_token_ids, 2)
    )


def mixed_admission_blocks(
    prompt_tokens: Sequence[int],
    shared_prefix_tokens: int,
    *,
    prefix_cache_enabled: bool,
) -> int:
    """Blocks resident when every mixed-phase prompt is admitted at once.

    With prefix caching, the warmed shared prefix is stored once; each request
    then adds only its unique suffix blocks plus one block to start decoding.
    Without prefix caching, every prompt owns its rounded prompt blocks plus
    one block to start decoding. This is the pressure the scheduler sees when
    the peers join, so it must stay below the budget or the prompts would wait
    in the queue instead of triggering preemption.
    """
    if not prefix_cache_enabled:
        return sum(min_resident_blocks(tokens) for tokens in prompt_tokens)

    prefix_blocks = blocks_for_tokens(shared_prefix_tokens)
    unique = sum(
        max(0, blocks_for_tokens(tokens) - prefix_blocks) for tokens in prompt_tokens
    )
    return prefix_blocks + unique + len(prompt_tokens)


def mixed_growth_blocks(
    prompt_tokens: Sequence[int],
    max_tokens: Sequence[int],
    shared_prefix_tokens: int,
    *,
    prefix_cache_enabled: bool,
) -> int:
    """Blocks resident if every request reaches its generation cap.

    Without prefix caching, each request contributes its complete rounded
    prompt-plus-generation footprint. With prefix caching, the warmed shared
    prefix is stored once and each request contributes only its unique growth.
    The result is an upper bound while the abort peer is alive, and the sum
    above the budget is what makes a running request's ``allocate_slots`` fail
    and preempt the last running peer.
    """
    if not prefix_cache_enabled:
        return sum(
            resident_blocks(prompt, cap)
            for prompt, cap in zip(prompt_tokens, max_tokens)
        )

    prefix_blocks = blocks_for_tokens(shared_prefix_tokens)
    return prefix_blocks + sum(
        max(0, blocks_for_tokens(prompt + cap) - prefix_blocks)
        for prompt, cap in zip(prompt_tokens, max_tokens)
    )


TEST_FA_VERSION_ENV = "VLLM_UNO_TEST_FA_VERSION"
TEST_ATTENTION_BACKEND_ENV = "VLLM_UNO_TEST_ATTENTION_BACKEND"
# The attention backends Uno's e2e tests run on: FLASH_ATTN (the tests' choice
# on CUDA) and TRITON_ATTN (their choice on ROCm, and selectable on CUDA).
TEST_ATTENTION_BACKENDS = ("FLASH_ATTN", "TRITON_ATTN")
# The attention implementation class each of those backends builds per layer.
TEST_ATTENTION_IMPLS = {
    "FLASH_ATTN": "FlashAttentionImpl",
    "TRITON_ATTN": "TritonAttentionImpl",
}


def resolve_attention_config(
    raw: str | None, rocm: bool = False, backend: str | None = None
) -> dict[str, object]:
    """The e2e attention config, with the FlashAttention version left open.

    ``backend`` (``VLLM_UNO_TEST_ATTENTION_BACKEND``) picks the backend:
    FLASH_ATTN by default on CUDA, TRITON_ATTN by default and only on ROCm,
    which has no FlashAttention decoder backend. Any other name is refused, as
    is a FlashAttention version pin on TRITON_ATTN: a pin there would name a
    kernel the run never used.

    Both e2e engines used to pin ``flash_attn_version=2``, which meant the
    module could never exercise FlashAttention 3 on any card: the platform
    prefers FA3 on SM90 only when the version is unset, and an explicit 2 wins
    over that preference. Uno's Hopper support was therefore never covered by
    its own tests. The backend stays pinned -- a test that silently ran on
    another backend would be measuring something else -- while the
    FlashAttention version follows the platform: FA3 on Hopper, FA4 on
    SM100, FA2 elsewhere, so Ampere and sm_121 keep the behaviour they were
    validated with.

    ``VLLM_UNO_TEST_FA_VERSION`` pins a version back for a reproducibility
    study. Only 2 and 3 are accepted: 4 is a different kernel family with its
    own downgrade rules, and a value this module cannot honour must fail
    loudly rather than be ignored into the platform default.
    """
    if backend is None:
        backend = "TRITON_ATTN" if rocm else "FLASH_ATTN"
    if backend not in TEST_ATTENTION_BACKENDS:
        raise ValueError(
            f"{TEST_ATTENTION_BACKEND_ENV} must be one of "
            f"{', '.join(TEST_ATTENTION_BACKENDS)} (got {backend!r})"
        )
    if rocm and backend != "TRITON_ATTN":
        raise ValueError(
            f"{TEST_ATTENTION_BACKEND_ENV}={backend} is not available on ROCm; "
            "Uno runs TRITON_ATTN there"
        )
    if backend == "TRITON_ATTN":
        if raw is not None:
            raise ValueError(
                f"{TEST_FA_VERSION_ENV} pins a FlashAttention version, and this "
                f"run uses TRITON_ATTN (got {raw!r}); leave it unset"
            )
        return {"backend": "TRITON_ATTN"}
    config: dict[str, object] = {"backend": "FLASH_ATTN"}
    if raw is None:
        return config
    text = raw.strip()
    if text not in ("2", "3"):
        raise ValueError(
            f"{TEST_FA_VERSION_ENV} must be '2' or '3' (got {raw!r}); leave it "
            "unset to use the platform default"
        )
    config["flash_attn_version"] = int(text)
    return config


def format_attention_receipt(state: Mapping[str, Any], requested: int | None) -> str:
    """One line naming the FlashAttention version the engine actually built.

    Read from the built engine rather than from the request, because the
    request is usually silent: with the version unset the platform chooses,
    and the choice is what the receipt has to record. A run that resolved no
    version, or more than one across layers, says so instead of reporting the
    first.
    """
    versions = sorted(int(v) for v in state.get("flash_attn_versions", []))
    backends = sorted(str(b) for b in state.get("backends", []))
    if len(versions) == 1:
        resolved = f"FlashAttention version {versions[0]}"
    elif not versions and backends and not any("FlashAttention" in b for b in backends):
        resolved = "no FlashAttention (non-FlashAttention backend)"
    elif not versions:
        resolved = "FlashAttention version unreported"
    else:
        resolved = f"FlashAttention versions {versions} (layers disagree)"
    asked = "platform default" if requested is None else f"pinned {requested}"
    return f"{resolved} (requested: {asked}), impls={backends or ['unreported']}"


def token_agreement(
    reference: Sequence[Sequence[int]],
    candidate: Sequence[Sequence[int]],
) -> tuple[int, list[tuple[int, int]]]:
    """Per-prompt exact-token agreement, plus where each disagreement starts.

    Returns ``(matched, divergences)`` where ``matched`` counts the prompts
    whose token ids are identical and each divergence is
    ``(prompt_index, first differing token index)`` -- the length of the shorter
    output when one is a prefix of the other. ``format_divergences`` renders
    them as ``p<i>/t<j>``, the coordinate the GPU tests print, so a test
    failure and a printed matrix line can be compared by eye.
    """
    assert len(reference) == len(candidate), (
        f"output counts differ: {len(reference)} vs {len(candidate)}"
    )
    matched = 0
    divergences: list[tuple[int, int]] = []
    for index, (left, right) in enumerate(zip(reference, candidate)):
        left_ids = list(left)
        right_ids = list(right)
        if left_ids == right_ids:
            matched += 1
            continue
        position = min(len(left_ids), len(right_ids))
        for token_index, (left_id, right_id) in enumerate(zip(left_ids, right_ids)):
            if left_id != right_id:
                position = token_index
                break
        divergences.append((index, position))
    return matched, divergences


def format_divergences(divergences: Sequence[tuple[int, int]]) -> list[str]:
    """Render divergences as the ``p<prompt>/t<token>`` coordinates."""
    return [f"p{prompt}/t{token}" for prompt, token in divergences]


def exact_token_verdict(
    control_divergences: Sequence[tuple[int, int]],
    candidate_divergences: Sequence[tuple[int, int]],
    total: int,
) -> tuple[bool, str]:
    """Judge an exact-token comparison against the instrument's own noise floor.

    Greedy exact-token equality is only an instrument where the plain engine
    reproduces itself. On an RTX 3090 (sm_86) in graph mode it does not: two
    fresh plain engines on the same config and prompts agreed on 3 of 4 prompts,
    diverging at prompt 2 token 31, and the Uno arms diverged at the same prompt
    and token. A test that demands 4 of 4 from Uno there cannot tell an Uno
    defect from the baseline's own spread.

    The judgment is per prompt, not per count: every prompt the candidate
    diverges at must be a prompt the control also diverged at. Comparing counts
    would pass a candidate that broke a prompt the control reproduced exactly
    whenever the control happened to diverge somewhere else -- control {p2},
    candidate {p0}, three matches each. Containment also covers the
    deterministic regime without a special case: an empty control admits only an
    empty candidate.
    """
    extra = uncontained_prompts(
        [prompt for prompt, _ in control_divergences],
        [prompt for prompt, _ in candidate_divergences],
    )
    control_text = format_divergences(sorted(control_divergences)) or "none"
    candidate_text = format_divergences(sorted(candidate_divergences)) or "none"
    return _containment_reason(extra, control_text, candidate_text, total, "tokens")


def uncontained_prompts(
    control_prompts: Iterable[int],
    candidate_prompts: Iterable[int],
) -> list[int]:
    """Prompts the candidate broke that the control reproduced exactly.

    The one containment rule both the token and the text comparison read, so
    they cannot drift apart. Passing a count rather than prompt indices raises
    here, which is what keeps a call site from regressing to the threshold
    arithmetic this replaced.
    """
    return sorted(set(candidate_prompts) - set(control_prompts))


def _containment_reason(
    extra: Sequence[int],
    control_text,
    candidate_text,
    total: int,
    kind: str,
) -> tuple[bool, str]:
    if not extra:
        return True, (
            f"candidate {kind} divergences {candidate_text} are contained in "
            f"the plain control's {control_text} over {total} prompts"
        )
    return False, (
        f"candidate {kind} diverge at prompts {list(extra)} that the plain "
        f"control reproduced exactly: candidate {candidate_text} against "
        f"control {control_text} over {total} prompts"
    )


def text_agreement(
    reference: Sequence[str],
    candidate: Sequence[str],
) -> tuple[int, list[int]]:
    """Per-prompt exact-text agreement and the prompts that differ.

    Detokenised text is a second view of the same comparison, so it is judged
    by the same containment rule rather than by a match threshold: subtracting
    a divergence count from the prompt count over-credits a control that
    diverged twice within one prompt, and can fall below zero once several
    control arms contribute.
    """
    assert len(reference) == len(candidate), (
        f"output counts differ: {len(reference)} vs {len(candidate)}"
    )
    divergent = [
        index
        for index, (left, right) in enumerate(zip(reference, candidate))
        if left != right
    ]
    return len(reference) - len(divergent), divergent


def text_verdict(
    control_prompts: Sequence[int],
    candidate_prompts: Sequence[int],
    total: int,
) -> tuple[bool, str]:
    """Judge text divergences by the same containment rule as the tokens."""
    extra = uncontained_prompts(control_prompts, candidate_prompts)
    control_text = [f"p{prompt}" for prompt in sorted(set(control_prompts))] or "none"
    candidate_text = [
        f"p{prompt}" for prompt in sorted(set(candidate_prompts))
    ] or "none"
    return _containment_reason(extra, control_text, candidate_text, total, "text")


def bf16_step(value: float) -> float:
    """The bfloat16 spacing at ``value``: one unit in the last place.

    bfloat16 keeps 7 explicit significand bits, so representable numbers in
    ``[2**e, 2**(e + 1))`` are ``2**(e - 7)`` apart: 0.25 for a logit of 44.75.
    The model's logits are bfloat16, so two logits one step apart are adjacent
    values, and a change in reduction order (a different query length,
    attention split or process) can move either one by that much.
    """
    magnitude = abs(float(value))
    if magnitude == 0.0 or not math.isfinite(magnitude):
        return 0.0
    _mantissa, exponent = math.frexp(magnitude)
    return math.ldexp(1.0, exponent - 1 - 7)


class TieRow(NamedTuple):
    """One divergence classified as the reference's own bfloat16 tie."""

    prompt: int
    token: int
    reference_id: int
    candidate_id: int
    margin: float
    step: float


class MatrixArm(NamedTuple):
    """One candidate arm of the greedy matrix, as the verdict needs it.

    Carrying the token ids beside the divergence coordinates is what lets the
    single verdict helper classify ties; an arm that passed coordinates alone
    could not be judged by the same rule as its neighbours.
    """

    divergences: Sequence[tuple[int, int]]
    token_ids: Sequence[Sequence[int]]
    text_prompts: Sequence[int]


class ArmVerdict(NamedTuple):
    """Every judgment one arm receives, from one call."""

    ok: bool
    reason: str
    text_ok: bool
    text_reason: str
    ties: list[TieRow]
    detokenisation_only: list[int]


def format_ties(ties: Sequence[TieRow]) -> list[str]:
    """Render ties with their measured margins, for the receipt."""
    return [
        f"p{tie.prompt}/t{tie.token} ref={tie.reference_id} "
        f"candidate={tie.candidate_id} logit_gap={tie.margin:.6f} "
        f"bf16_step={tie.step:g}"
        for tie in ties
    ]


def classify_ties(
    divergences: Sequence[tuple[int, int]],
    reference_ids: Sequence[Sequence[int]],
    candidate_ids: Sequence[Sequence[int]],
    reference_ranked: Sequence[Sequence[Sequence[tuple[int, float]]]] | None,
) -> tuple[list[tuple[int, int]], list[TieRow]]:
    """Split divergences into real ones and the reference's own bfloat16 ties.

    A divergence is a TIE only when all three hold at that position: the
    reference's top-ranked token is the token the reference emitted (otherwise
    the ranked list does not describe this run and nothing is excused), the
    candidate emitted the reference's runner-up, and the reference's top two
    RAW LOGITS differ by at most one bfloat16 step at the runner-up's logit
    (``bf16_step``). The evidence comes from the reference run only, never from
    the candidate. A candidate token outside the reference's top two is never a
    tie, and a runner-up more than one step behind is a real divergence.
    ``matrix_verdicts`` excuses at most ONE tied prompt per batch.

    Each divergence is the first differing position of its prompt
    (``token_agreement``), so an excused tie ends the comparison for that
    prompt: past it the two continuations legitimately differ. A divergence at
    a non-tie position earlier in a prompt is the one reported, and it fails.

    ``reference_ranked`` is per prompt, per position, ranked
    ``(token_id, raw_logit)`` pairs, from an engine run with
    ``logprobs_mode="raw_logits"``. Missing or short ranked data classifies
    nothing.
    """
    if isinstance(divergences, int) or isinstance(reference_ids, int):
        raise TypeError(
            "classify_ties takes divergence coordinates and token ids, not counts"
        )
    real: list[tuple[int, int]] = []
    ties: list[TieRow] = []
    for prompt, token in divergences:
        tie = _tie_row(prompt, token, reference_ids, candidate_ids, reference_ranked)
        if tie is None:
            real.append((prompt, token))
        else:
            ties.append(tie)
    return real, ties


def _tie_row(
    prompt: int,
    token: int,
    reference_ids: Sequence[Sequence[int]],
    candidate_ids: Sequence[Sequence[int]],
    reference_ranked: Sequence[Sequence[Sequence[tuple[int, float]]]] | None,
) -> TieRow | None:
    if reference_ranked is None or prompt >= len(reference_ranked):
        return None
    positions = reference_ranked[prompt]
    if token >= len(positions):
        return None
    ranked = list(positions[token])
    if len(ranked) < 2:
        return None
    if prompt >= len(reference_ids) or token >= len(reference_ids[prompt]):
        return None
    if prompt >= len(candidate_ids) or token >= len(candidate_ids[prompt]):
        return None
    (top_id, top_logit), (second_id, second_logit) = ranked[0], ranked[1]
    if top_id != reference_ids[prompt][token]:
        # The ranked list does not describe the tokens this run emitted, so it
        # cannot license an exception. Excusing a divergence on mismatched
        # diagnostics is exactly the failure this rule is meant to prevent.
        return None
    margin = float(top_logit) - float(second_logit)
    # The runner-up's spacing: at a power of two the spacing doubles above it,
    # so the top logit's step would excuse 32.0 over 31.75, two steps of 0.125.
    step = bf16_step(second_logit)
    if candidate_ids[prompt][token] != second_id or margin > step:
        return None
    return TieRow(prompt, token, top_id, second_id, margin, step)


def matrix_verdicts(
    control_divergences: Sequence[tuple[int, int]],
    candidates: Mapping[str, MatrixArm],
    total: int,
    reference_ids: Sequence[Sequence[int]] | None = None,
    reference_ranked: Sequence[Sequence[Sequence[tuple[int, float]]]] | None = None,
) -> dict[str, ArmVerdict]:
    """Judge every candidate for one batch against ONE completed floor.

    The adapter-disabled arm used to be judged inside the Uno engine's context,
    before the separate-engine control had contributed to the floor, so it was
    held to a weaker floor than the Uno comparison while the comment promised
    the same one. Taking every candidate for a batch through a single call is
    the structural form of that promise.

    Both views and the tie rule live here for the same reason. The token
    comparison, the detokenised-text comparison and the near-tie exception are
    one judgment about one arm; splitting them across call sites is how the
    adapter-disabled arm drifted onto its own floor once already. A caller
    that hands this a bare sequence instead of a `MatrixArm` raises, because
    such a caller could not be carrying the token ids the tie rule reads.

    Ties are removed from the token view and from the text view together. A
    tie is a real difference in the emitted tokens -- it just is not evidence
    of a different computation -- so the prompt's text differs too, and
    excusing one view while failing the other would report the same rounding
    twice under two names.
    """
    verdicts: dict[str, ArmVerdict] = {}
    control_prompts = [prompt for prompt, _ in control_divergences]
    for name, arm in candidates.items():
        if not isinstance(arm, MatrixArm):
            raise TypeError(
                f"{name}: matrix_verdicts takes MatrixArm values carrying token "
                "ids, not bare divergence lists"
            )
        real, ties = classify_ties(
            arm.divergences,
            reference_ids if reference_ids is not None else [],
            arm.token_ids,
            reference_ranked,
        )
        # At most one excused prompt per batch. A real error whose first
        # effect in every prompt were a one-step runner-up would otherwise pass
        # every prompt; past the first tie, further ties count as divergences.
        ties = sorted(ties, key=lambda tie: (tie.prompt, tie.token))
        real = sorted(real + [(tie.prompt, tie.token) for tie in ties[1:]])
        ties = ties[:1]
        ok, reason = exact_token_verdict(control_divergences, real, total)
        tie_prompts = {tie.prompt for tie in ties}
        real_prompts = {prompt for prompt, _ in real}
        text_prompts = [
            prompt
            for prompt in arm.text_prompts
            if prompt not in tie_prompts or prompt in real_prompts
        ]
        text_ok, text_reason = text_verdict(control_prompts, text_prompts, total)
        if ties:
            rendered = ", ".join(format_ties(ties))
            reason = f"{reason}; ties excused: {rendered}"
            text_reason = f"{text_reason}; ties excused: {rendered}"
        # Identical tokens with different text is not a sampling difference; it
        # is the detokeniser. Judged against the arm's UNFILTERED divergences:
        # a tie changes the tokens, so its text is expected to differ too.
        diverged_prompts = {prompt for prompt, _ in arm.divergences}
        detokenisation_only = sorted(set(arm.text_prompts) - diverged_prompts)
        verdicts[name] = ArmVerdict(
            ok, reason, text_ok, text_reason, ties, detokenisation_only
        )
    return verdicts


def resolve_internal_request_ids(
    scheduler_keys: Iterable[str],
    external_ids: Sequence[str],
) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Map external request ids to the ids the scheduler is keyed by.

    ``InputProcessor.assign_request_id`` replaces the id a caller passes to
    ``add_request`` with ``f"{external}-{random_uuid():.8}"`` unless
    ``VLLM_DISABLE_REQUEST_ID_RANDOMIZATION`` is set, and ``Request`` keeps only
    that internal id. Looking a request up by the id the test supplied
    therefore returns ``None`` on every step, leaving the per-request receipt
    empty even while the scheduler's own pool reports 100% occupancy.

    Returns ``(resolved, problems)``: ``resolved`` maps each external id whose
    internal id was unambiguous, ``problems`` maps the rest to the candidates
    found, so a caller can fail with the ids the scheduler actually holds.
    """
    keys = list(scheduler_keys)
    resolved: dict[str, str] = {}
    problems: dict[str, list[str]] = {}
    for external in external_ids:
        candidates = [key for key in keys if _is_internal_id(key, external)]
        if len(candidates) == 1:
            resolved[external] = candidates[0]
        else:
            problems[external] = candidates
    return resolved, problems


def _is_internal_id(key: str, external: str) -> bool:
    """Whether ``key`` is the scheduler's id for the request ``external``."""
    if key == external:
        # VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1.
        return True
    if not key.startswith(f"{external}-"):
        return False
    suffix = key[len(external) + 1 :]
    # `random_uuid():.8` is eight characters of a uuid4 string.
    return len(suffix) == 8 and suffix.isalnum()


def mid_generation_preemption_counts(
    events: Sequence[tuple[str, int]],
) -> dict[str, int]:
    """Count only the preemptions that interrupted a request mid-generation.

    ``events`` are ``(request_id, generated_tokens_at_preemption)`` pairs. A
    request preempted while its prompt is still being chunked has produced no
    tokens, and its recompute re-runs a prefill the gate is not about: only a
    preemption at one or more generated tokens exercises the resume path the
    survivor claim covers.
    """
    counts: dict[str, int] = {}
    for request_id, generated_tokens in events:
        if generated_tokens > 0:
            counts[request_id] = counts.get(request_id, 0) + 1
    return counts


def worst_case_crossing_tokens(
    peer_prompt_tokens: int,
    other_peer_prompt_tokens: Sequence[int],
    allocatable: int,
    num_speculative_tokens: int = SURVIVOR_NUM_SPECULATIVE_TOKENS,
) -> int:
    """Generated tokens at which ONE peer alone must outgrow the pool.

    ``mixed_crossing_tokens`` assumes the long peers grow together. They do not:
    Uno's acceptance is prompt-dependent, so one peer can run several times
    faster than its twin, and a pair that only crosses *together* never crosses
    at all when the leader reaches its cap while the laggard still sits near its
    admission footprint. This is the counted version of that worst case -- the
    leader's own footprint plus the minimum resident footprint of every other
    live long peer -- and the survivor e2e asserts it is below the cap, so the
    scheduler must preempt for every interleaving rather than for the lucky
    ones.
    """
    held = sum(
        min_resident_blocks(tokens, num_speculative_tokens)
        for tokens in other_peer_prompt_tokens
    )
    tokens = 0
    while True:
        leader = resident_blocks(peer_prompt_tokens, tokens, num_speculative_tokens)
        if leader + held > allocatable:
            return tokens
        tokens += 1


def mixed_crossing_tokens(
    peer_prompt_tokens: int,
    shared_prefix_tokens: int,
    capacity_blocks: int,
    *,
    prefix_cache_enabled: bool,
) -> int:
    """Generated tokens at which the two long peers TOGETHER outgrow the pool.

    ``capacity_blocks`` is the allocatable count, not the pinned override (see
    ``allocatable_blocks``). By the time the pair has grown this far the short
    seed has finished and the abort peer is retired. Without prefix caching,
    each peer owns its complete prompt-plus-generation footprint; with prefix
    caching, the warmed prefix is stored once and each peer contributes only its
    unique growth. This is the *best* case for the gate and is reported for
    continuity; the assertion that makes the gate fire is
    ``worst_case_crossing_tokens``, which does not assume equal rates.
    """
    prefix_blocks = blocks_for_tokens(shared_prefix_tokens)
    tokens = 0
    while True:
        if prefix_cache_enabled:
            unique = max(
                0, blocks_for_tokens(peer_prompt_tokens + tokens) - prefix_blocks
            )
            resident = prefix_blocks + 2 * unique
        else:
            resident = 2 * resident_blocks(peer_prompt_tokens, tokens)
        if resident > capacity_blocks:
            return tokens
        tokens += 1

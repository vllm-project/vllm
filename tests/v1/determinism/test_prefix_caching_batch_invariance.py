# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Prefix-caching batch invariance test.

Prefix caching for Mamba and hybrid (attention+SSM) models is experimental,
and divergence is a known risk in both supported Mamba cache modes.

"align" mode caches attention KV blocks but recomputes Mamba recurrent state,
so cache reuse is only correct if every prefill split still lands on a valid
Mamba chunk boundary.

"all" mode caches both attention KV blocks and serialized Mamba state, so a
cache hit must reproduce exactly the same state that was materialized during
the cache-filling run.

This test forces chunked prefill with max_num_batched_tokens=2048, uses a long
shared prefix so several prefix-cache blocks are exercised, and verifies that a
cold run and a hot-cache rerun of the same batch produce bitwise-identical
logprobs across a sweep of batch sizes.
"""

import contextlib
import os
import random
import time

import pytest
import torch
from utils import (
    BACKENDS,
    TEST_MODEL,
    _extract_step_logprobs,
    is_device_capability_below_90,
    skip_unsupported,
)

from vllm import LLM, SamplingParams, TokensPrompt

IS_DEVICE_CAPABILITY_BELOW_90 = is_device_capability_below_90()

MAX_BATCH_SIZE = int(os.getenv("VLLM_DIVERGENCE_MAX_BATCH_SIZE", "8"))
EXTRA_BATCH_SIZES = os.getenv("VLLM_DIVERGENCE_EXTRA_BATCH_SIZES", "")
MAX_NUM_BATCHED_TOKENS = int(os.getenv("VLLM_TEST_MAX_NUM_BATCHED_TOKENS", "2048"))
PREFIX_CACHE_RESET_TIMEOUT_S = float(
    os.getenv("VLLM_PREFIX_CACHE_RESET_TIMEOUT_S", "5.0")
)

MAMBA_CACHE_MODES = ["align", "all"]


def _parse_positive_int_csv(raw: str) -> list[int]:
    values: list[int] = []
    if not raw:
        return values
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value <= 0:
            raise ValueError(f"Batch size must be positive, got {value}")
        values.append(value)
    return values


def _build_batch_sizes() -> list[int]:
    batch_sizes = list(range(1, MAX_BATCH_SIZE + 1))
    batch_sizes.extend(_parse_positive_int_csv(EXTRA_BATCH_SIZES))
    return sorted(set(batch_sizes))


BATCH_SIZES = _build_batch_sizes()

# Public-domain text from Alice's Adventures in Wonderland (Lewis Carroll,
# 1865). Repeating a stable excerpt keeps the test self-contained while still
# producing a shared prefix large enough to span multiple KV cache blocks.
ALICE_SHARED_PREFIX_SEGMENT = """\
Alice was beginning to get very tired of sitting by her sister on the bank,
and of having nothing to do: once or twice she had peeped into the book her
sister was reading, but it had no pictures or conversations in it, "and what
is the use of a book," thought Alice, "without pictures or conversations?"

So she was considering in her own mind, as well as she could, for the hot day
made her feel very sleepy and stupid, whether the pleasure of making a
daisy-chain would be worth the trouble of getting up and picking the daisies,
when suddenly a White Rabbit with pink eyes ran close by her.

There was nothing so very remarkable in that; nor did Alice think it so very
much out of the way to hear the Rabbit say to itself, "Oh dear! Oh dear! I
shall be late!" When she thought it over afterwards, it occurred to her that
she ought to have wondered at this, but at the time it all seemed quite
natural; but when the Rabbit actually took a watch out of its waistcoat-pocket,
and looked at it, and then hurried on, Alice started to her feet, for it
flashed across her mind that she had never before seen a rabbit with either a
waistcoat-pocket, or a watch to take out of it, and burning with curiosity,
she ran across the field after it, and was just in time to see it pop down a
large rabbit-hole under the hedge.

In another moment down went Alice after it, never once considering how in the
world she was to get out again. The rabbit-hole went straight on like a tunnel
for some way, and then dipped suddenly down, so suddenly that Alice had not a
moment to think about stopping herself before she found herself falling down
what seemed to be a very deep well.

Either the well was very deep, or she fell very slowly, for she had plenty of
time as she went down to look about her and to wonder what was going to happen
next. First, she tried to look down and make out what she was coming to, but
it was too dark to see anything; then she looked at the sides of the well, and
noticed that they were filled with cupboards and book-shelves; here and there
she saw maps and pictures hung upon pegs. She took down a jar from one of the
shelves as she passed; it was labeled ORANGE MARMALADE, but to her great
disappointment it was empty: she did not like to drop the jar for fear of
killing somebody underneath, so managed to put it into one of the cupboards as
she fell past it.

Down, down, down. Would the fall never come to an end? "I wonder how many
miles I've fallen by this time?" she said aloud. "I must be getting somewhere
near the center of the earth. Let me see: that would be four thousand miles
down, I think." She was rather glad there was no one listening, this time, as
it did not sound at all the right word, but still it was good practice to say
it over. "Yes, that's about the right distance, but then I wonder what
Latitude or Longitude I've got to?" Alice had no idea what Latitude was, or
Longitude either, but thought they were nice grand words to say.

Presently she began again. "I wonder if I shall fall right through the earth!
How funny it'll seem to come out among the people that walk with their heads
downward! The Antipathies, I think." She was rather glad there was no one
listening, as this time it did not sound at all the right word. "But I shall
have to ask them what the name of the country is, you know. Please, Ma'am, is
this New Zealand or Australia?" and she tried to curtsey as she spoke. Fancy
curtseying as you're falling through the air! Do you think you could manage
it?

At last, thump! thump! down she came upon a heap of sticks and dry leaves, and
the fall was over. Alice was not a bit hurt, and she jumped up on to her feet
in a moment. She looked up, but it was all dark overhead; before her was
another long passage, and the White Rabbit was still in sight, hurrying down
it. There was not a moment to be lost: away went Alice like the wind, and was
just in time to hear it say, as it turned a corner, "Oh my ears and whiskers,
how late it's getting!" She was close behind it when she turned the corner,
but the Rabbit was no longer to be seen.

She found herself in a long, low hall, which was lit up by a row of lamps
hanging from the roof. There were doors all round the hall, but they were all
locked; and when Alice had been all the way down one side and up the other,
trying every door, she walked sadly down the middle, wondering how she was
ever to get out again.

Suddenly she came upon a little three-legged table, all made of solid glass;
there was nothing on it except a tiny golden key, and Alice's first thought
was that it might belong to one of the doors of the hall; but, alas! either
the locks were too large, or the key was too small, but at any rate it would
not open any of them. However, on the second time round, she came upon a low
curtain she had not noticed before, and behind it was a little door about
fifteen inches high: she tried the little golden key in the lock, and to her
great delight it fitted.

Alice opened the door and found that it led into a small passage, not much
larger than a rat-hole: she knelt down and looked along the passage into the
loveliest garden you ever saw. How she longed to get out of that dark hall,
and wander about among those beds of bright flowers and those cool fountains,
but she could not even get her head through the doorway; "and even if my head
would go through," thought poor Alice, "it would be of very little use without
my shoulders."

Soon her eye fell on a little glass bottle that certainly was not there before.
Round the neck of the bottle was a paper label, with the words DRINK ME
beautifully printed on it in large letters. It was all very well to say,
"Drink me," but the wise little Alice was not going to do that in a hurry.
"No, I'll look first," she said, "and see whether it's marked poison or not."
However, this bottle was not marked poison, so Alice ventured to taste it, and
finding it very nice, she very soon finished it off.

"What a curious feeling!" said Alice. "I must be shutting up like a
telescope." And so it was indeed: she was now only ten inches high, and her
face brightened up at the thought that she was now the right size for going
through the little door into that lovely garden. First, however, she waited
for a few minutes to see if she was going to shrink any further; she felt a
little nervous about this, "for it might end, you know," said Alice to
herself, "in my going out altogether, like a candle. I wonder what I should be
like then?"

After a while, finding that nothing more happened, she decided on going into
the garden at once; but, alas for poor Alice! when she got to the door, she
found she had forgotten the little golden key, and when she went back to the
table for it, she found she could not possibly reach it: she could see it
quite plainly through the glass, and she tried her best to climb up one of the
legs of the table, but it was too slippery; and when she had tired herself out
with trying, the poor little thing sat down and cried.

Before long, there was another curious thing lying under the table, a very
small cake, on which the words EAT ME were beautifully marked in currants.
"Well, I'll eat it," said Alice, "and if it makes me grow larger, I can reach
the key; and if it makes me grow smaller, I can creep under the door; so either
way I'll get into the garden, and I don't care which happens!" She ate a
little bit, and said anxiously to herself, "Which way? Which way?" holding her
hand on the top of her head to feel which way it was growing, and she was
quite surprised to find that she remained the same size.

At last she stretched herself up on tiptoe, and peeped over the edge of the
table. "Curiouser and curiouser!" cried Alice; she was so much surprised, that
for the moment she quite forgot how to speak good English. Soon she had grown
so large that she was kneeling down in the hall, and presently there was not
room even for this, and she tried the effect of lying down with one elbow
against the door, and the other arm curled round her head. Still she went on
growing, and, as a last resource, she put one arm out of the window, and one
foot up the chimney.

All that remained to be done was to wonder what would become of her. She was
so much surprised that for some time she could say nothing, and when at last
the White Rabbit returned with a pair of white kid gloves in one hand and a
large fan in the other, Alice was still trying to make out how she could ever
get into that beautiful garden. The Rabbit muttered to itself, started when it
saw her enormous shape, and dropped the fan and gloves in its fright.

Alice took up the fan and the gloves. The hall was very hot, and she kept
fanning herself all the time she went on talking. "Dear, dear! How queer
everything is to-day! And yesterday things went on just as usual. I wonder if
I've been changed in the night? Let me think: was I the same when I got up
this morning? I almost think I can remember feeling a little different. But if
I'm not the same, the next question is, Who in the world am I? Ah, that's the
great puzzle!"

She went on planning lessons, naming girls she knew, and testing whether she
could still remember all the verses she had learned. The more she talked, the
more puzzled she became, until at last the Rabbit's fan had such an effect
that she began shrinking again with astonishing rapidity. Very soon she was so
small that she slipped one foot, and the next moment splash! she was up to her
chin in salt water. Whether the water came from the sea or from her own tears,
she could not at first imagine, but she soon understood that she had fallen
into the pool she had made when she was nine feet high and crying.
"""

LONG_SHARED_PREFIX = "\n\n".join([ALICE_SHARED_PREFIX_SEGMENT] * 2)

UNIQUE_SUFFIX_QUESTIONS = [
    "Why does Alice decide that her sister's book is dull?",
    "What detail makes the White Rabbit impossible to ignore?",
    "What does Alice hope the little golden key will unlock?",
    "Why does Alice drink from the bottle marked DRINK ME?",
    "What keeps Alice from entering the garden immediately after shrinking?",
    "Why does Alice think growing could still help her?",
    "What object causes Alice to shrink again after she grows too large?",
    "How does Alice end up in the pool of tears?",
]


def _reset_prefix_cache(llm: LLM) -> None:
    deadline = time.monotonic() + PREFIX_CACHE_RESET_TIMEOUT_S
    while not llm.reset_prefix_cache():
        if time.monotonic() > deadline:
            raise TimeoutError(
                "reset_prefix_cache did not succeed within "
                f"{PREFIX_CACHE_RESET_TIMEOUT_S}s."
            )
        time.sleep(0.1)


def _build_prompt_variants(num_prompts: int) -> list[str]:
    prompts: list[str] = []
    for idx in range(num_prompts):
        question = UNIQUE_SUFFIX_QUESTIONS[idx % len(UNIQUE_SUFFIX_QUESTIONS)]
        prompts.append(
            f"{LONG_SHARED_PREFIX}\n\n"
            f"Prompt variant {idx + 1}: {question} "
            "Answer in one sentence with at most eighteen words, and do not "
            "mention the variant number."
        )
    return prompts


def _tokenize_prompts(llm: LLM, prompts: list[str]) -> list[TokensPrompt]:
    tokenizer = llm.get_tokenizer()
    return [
        TokensPrompt(prompt=prompt, prompt_token_ids=tokenizer.encode(prompt))
        for prompt in prompts
    ]


def _extract_request_result(
    request_output,
) -> tuple[torch.Tensor, list[int]]:
    step_logprobs, token_ids = _extract_step_logprobs(request_output)
    if step_logprobs is None or token_ids is None:
        pytest.skip(
            "Logits are not available on RequestOutput; "
            "enable logprobs return to run this test."
        )
    return step_logprobs, token_ids


def _assert_outputs_equal(
    baseline_logprobs: torch.Tensor,
    baseline_tokens: list[int],
    current_logprobs: torch.Tensor,
    current_tokens: list[int],
    run_label: str,
) -> None:
    if len(baseline_logprobs) != len(current_logprobs):
        print(
            f"\n[DIVERGENCE] {run_label}: different step count "
            f"{len(baseline_logprobs)} vs {len(current_logprobs)}"
        )
        print(f"  Run-1 tokens: {baseline_tokens}")
        print(f"  Run-2 tokens: {current_tokens}")
        pytest.fail(
            f"Divergence at {run_label}: step count mismatch "
            f"({len(baseline_logprobs)} vs {len(current_logprobs)})."
        )

    if baseline_tokens != current_tokens:
        print(f"\n[DIVERGENCE] {run_label}: different tokens sampled")
        print(f"  Run-1 tokens: {baseline_tokens}")
        print(f"  Run-2 tokens: {current_tokens}")
        pytest.fail(f"Divergence at {run_label}: different tokens sampled.")

    for step_idx, (baseline_step, current_step) in enumerate(
        zip(baseline_logprobs, current_logprobs)
    ):
        if baseline_step.shape != current_step.shape:
            print(
                f"\n[DIVERGENCE] {run_label}, step {step_idx}: "
                f"shape mismatch {baseline_step.shape} vs {current_step.shape}"
            )
            pytest.fail(
                f"Divergence at {run_label}, step {step_idx}: "
                f"shape mismatch {baseline_step.shape} vs {current_step.shape}."
            )

        if not torch.equal(baseline_step, current_step):
            max_diff = torch.abs(baseline_step - current_step).max().item()
            baseline_tok = (
                baseline_tokens[step_idx] if step_idx < len(baseline_tokens) else "N/A"
            )
            current_tok = (
                current_tokens[step_idx] if step_idx < len(current_tokens) else "N/A"
            )
            print(
                f"\n[DIVERGENCE] {run_label}, step {step_idx}: max_diff={max_diff:.6e}"
            )
            print(f"  Token IDs: run1={baseline_tok}, run2={current_tok}")
            print(f"  Run-1 logprob: {baseline_step.tolist()}")
            print(f"  Run-2 logprob: {current_step.tolist()}")
            print(f"  Run-1 all logprobs: {baseline_logprobs.tolist()}")
            print(f"  Run-2 all logprobs: {current_logprobs.tolist()}")
            pytest.fail(
                f"Divergence at {run_label}, step {step_idx}: "
                f"bitwise mismatch (max_diff={max_diff:.6e})."
            )


@skip_unsupported
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("mamba_cache_mode", MAMBA_CACHE_MODES)
def test_logprobs_prefix_caching_batch_invariance(
    backend: str,
    mamba_cache_mode: str,
):
    seed = int(os.getenv("VLLM_TEST_SEED", "12345"))
    random.seed(seed)
    tp_size = int(os.getenv("VLLM_TEST_TP_SIZE", "1"))

    from vllm import envs

    disable_custom_ar = envs.VLLM_BATCH_INVARIANT

    if disable_custom_ar:
        print(f"\n{'=' * 80}")
        print(f"BATCH INVARIANCE MODE: Disabling custom all-reduce (TP={tp_size})")
        print(f"{'=' * 80}\n")

    llm = None
    try:
        llm = LLM(
            model=TEST_MODEL,
            tensor_parallel_size=tp_size,
            max_num_seqs=64,
            max_model_len=8192,
            max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
            enable_prefix_caching=True,
            mamba_cache_mode=mamba_cache_mode,
            dtype="bfloat16",
            gpu_memory_utilization=0.9,
            enforce_eager=IS_DEVICE_CAPABILITY_BELOW_90,
            attention_config={"backend": backend},
            mamba_cache_dtype="float32",
        )

        effective_mode = llm.llm_engine.vllm_config.cache_config.mamba_cache_mode
        if effective_mode != mamba_cache_mode:
            pytest.skip(
                f"Requested mamba_cache_mode={mamba_cache_mode!r}, "
                f"but model resolved to {effective_mode!r}."
            )

        sampling_params = SamplingParams(
            temperature=0.0,
            top_p=0.5,
            max_tokens=64,
            seed=1234,
            logprobs=5,
        )

        max_batch_size = max(BATCH_SIZES)
        prompts = _build_prompt_variants(max_batch_size)
        tokenized_prompts = _tokenize_prompts(llm, prompts)
        shared_prefix_token_ids = llm.get_tokenizer().encode(LONG_SHARED_PREFIX)
        total_request_comparisons = sum(BATCH_SIZES)

        print("\n" + "=" * 80)
        print(
            "STARTING PREFIX-CACHING BATCH INVARIANCE TEST "
            f"(backend={backend}, requested_mode={mamba_cache_mode}, "
            f"effective_mode={effective_mode}, batch sizes {BATCH_SIZES}, "
            f"shared_prefix_tokens={len(shared_prefix_token_ids)}, "
            f"comparisons={total_request_comparisons})"
        )
        print("=" * 80 + "\n")

        total_checked = 0
        for batch_size in BATCH_SIZES:
            _reset_prefix_cache(llm)

            batch_prompts = tokenized_prompts[:batch_size]
            run1_outputs = llm.generate(batch_prompts, sampling_params, use_tqdm=False)
            run2_outputs = llm.generate(batch_prompts, sampling_params, use_tqdm=False)

            assert len(run1_outputs) == batch_size
            assert len(run2_outputs) == batch_size

            run1_cached_tokens = [
                int(output.num_cached_tokens or 0) for output in run1_outputs
            ]
            run2_cached_tokens = [
                int(output.num_cached_tokens or 0) for output in run2_outputs
            ]

            if not all(cached_tokens > 0 for cached_tokens in run2_cached_tokens):
                pytest.fail(
                    f"Expected cache hits on second pass for batch_size={batch_size}, "
                    f"but got num_cached_tokens={run2_cached_tokens}."
                )

            for request_idx, (run1_output, run2_output) in enumerate(
                zip(run1_outputs, run2_outputs)
            ):
                run1_logprobs, run1_tokens = _extract_request_result(run1_output)
                run2_logprobs, run2_tokens = _extract_request_result(run2_output)
                _assert_outputs_equal(
                    run1_logprobs,
                    run1_tokens,
                    run2_logprobs,
                    run2_tokens,
                    run_label=(
                        f"backend={backend}, mode={effective_mode}, "
                        f"batch_size={batch_size}, request={request_idx}"
                    ),
                )
                total_checked += 1

            print(
                f"[batch_size={batch_size}] cold/hot runs matched "
                f"(run1 cached_tokens={run1_cached_tokens}, "
                f"run2 cached_tokens={run2_cached_tokens}, "
                f"total checked: {total_checked}/{total_request_comparisons})"
            )

        print(f"\n{'=' * 80}")
        print(
            f"SUCCESS: All {total_checked} requests produced bitwise-identical "
            f"logprobs between cold and hot prefix-cache runs across batch sizes "
            f"{BATCH_SIZES} in mamba_cache_mode={effective_mode!r}."
        )
        print(f"{'=' * 80}\n")
    finally:
        if llm is not None:
            with contextlib.suppress(Exception):
                llm.llm_engine.engine_core.shutdown()

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Chunked-prefill batch invariance test.

Chunked prefill divergence is a known hazard for Mamba and hybrid
(attention+SSM) models.  Mamba-2's chunked scan algorithm computes the
recurrent state over fixed-size token chunks; when a prefill is split by the
scheduler, each split must land on a Mamba chunk boundary.  An off-boundary
split produces a different intermediate scan state, which yields divergent
logprobs compared to processing the same sequence in a single chunk.  This
test sets max_num_batched_tokens=2048 so that batches of 5+ requests with a
~500-token prompt will trigger chunked prefill, exposing the divergence if the
implementation is incorrect.
"""

import os
import random

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

MAX_BATCH_SIZE = int(os.getenv("VLLM_DIVERGENCE_MAX_BATCH_SIZE", "10"))
EXTRA_BATCH_SIZES = os.getenv("VLLM_DIVERGENCE_EXTRA_BATCH_SIZES", "15,16,17")


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
    # Keep the original sweep and add boundary sizes where cudagraph buckets
    # commonly change behavior.
    batch_sizes = list(range(1, MAX_BATCH_SIZE + 1))
    batch_sizes.extend(_parse_positive_int_csv(EXTRA_BATCH_SIZES))
    return sorted(set(batch_sizes))


BATCH_SIZES = _build_batch_sizes()

# ~600-700 token excerpt from Pride and Prejudice, Chapters 1-2
# (Jane Austen, 1813 — Project Gutenberg, public domain).
LONG_NARRATIVE_PROMPT = """\
It is a truth universally acknowledged, that a single man in possession of a \
good fortune, must be in want of a wife.

However little known the feelings or views of such a man may be on his first \
entering a neighbourhood, this truth is so well fixed in the minds of the \
surrounding families, that he is considered as the rightful property of some \
one or other of their daughters.

"My dear Mr. Bennet," said his lady to him one day, "have you heard that \
Netherfield Park is let at last?"

Mr. Bennet replied that he had not.

"But it is," returned she; "for Mrs. Long has just been here, and she told me \
all about it."

Mr. Bennet made no answer.

"Do you not want to know who has taken it?" cried his wife impatiently.

"You want to tell me, and I have no objection to hearing it."

This was invitation enough.

"Why, my dear, you must know, Mrs. Long says that Netherfield is taken by a \
young man of large fortune from the north of England; that he came down on \
Monday in a chaise and four to see the place, and was so much delighted with \
it, that he agreed with Mr. Morris immediately; that he is to take possession \
before Michaelmas, and some of his servants are to be in the house by the end \
of next week."

"What is his name?"

"Bingley."

"Is he married or single?"

"Oh! Single, my dear, to be sure! A single man of large fortune; four or five \
thousand a year. What a fine thing for our girls!"

"How so? Can it affect them?"

"My dear Mr. Bennet," replied his wife, "how can you be so tiresome! You must \
know that I am thinking of his marrying one of them."

"Is that his design in settling here?"

"Design! Nonsense, how can you talk so! But it is very likely that he may fall \
in love with one of them, and therefore you must visit him as soon as he comes."

"I see no occasion for that. You and the girls may go, or you may send them by \
themselves, which perhaps will be still better, for as you are as handsome as \
any of them, Mr. Bingley may like you the best of the party."

"My dear, you flatter me. I certainly have had my share of beauty, but I do \
not pretend to be anything extraordinary now. When a woman has five grown-up \
daughters, she ought to give over thinking of her own beauty."

"In such cases, a woman has not often much beauty to think of."

"But, my dear, you must indeed go and see Mr. Bingley when he comes into the \
neighbourhood."

"It is more than I engage for, I assure you."

"But consider your daughters. Only think what an establishment it would be for \
one of them. Sir William and Lady Lucas are determined to go, merely on that \
account, for in general, you know, they visit no newcomers. Indeed you must go, \
for it will be impossible for us to visit him if you do not."

"You are over-scrupulous, surely. I dare say Mr. Bingley will be very glad to \
see you; and I will send a few lines by you to assure him of my hearty consent \
to his marrying whichever he chooses of our girls; though I must throw in a \
good word for my little Lizzy."

"I desire you will do no such thing. Lizzy is not a bit better than the others; \
and I am sure she is not half so handsome as Jane, nor half so good-humoured as \
Lydia. But you are always giving her the preference."

"They have none of them much to recommend them," replied he; "they are all \
silly and ignorant like other girls; but Lizzy has something more of quickness \
than her sisters."

"Mr. Bennet, how can you abuse your own children so? You take delight in \
vexing me. You have no compassion for my poor nerves."

"You mistake me, my dear. I have a high respect for your nerves. They are my \
old friends. I have heard you mention them with consideration these last twenty \
years at least."

"Ah, you do not know what I suffer."

"But I hope you will get over it, and live to see many young men of four \
thousand a year come into the neighbourhood."

"It will be no use to us, if twenty such should come, since you will not visit \
them."

"Depend upon it, my dear, that when there are twenty, I will visit them all."

Mr. Bennet was so odd a mixture of quick parts, sarcastic humour, reserve, and \
caprice, that the experience of three-and-twenty years had been insufficient to \
make his wife understand his character. Her mind was less difficult to develop. \
She was a woman of mean understanding, little information, and uncertain temper. \
When she was discontented, she fancied herself nervous. The business of her \
life was to get her daughters married; its solace was visiting and news.

Mr. Bennet had been obliged, from his account of the character of Netherfield's \
new tenant, to pay a visit to him; and Mrs. Bennet's curiosity to know the \
particulars of his reception and what he thought of it all filled her with \
anxious expectations. The visit was soon returned in due form. Miss Bennet's \
pleasing manners grew on the goodwill of Mrs. Hurst and Miss Bingley; and \
though the mother was found to be intolerable, and the younger sisters not \
worth speaking to, a wish of being better acquainted with them was expressed \
towards the two eldest. By Jane, this attention was received with the greatest \
pleasure, but Elizabeth still saw superciliousness in their treatment of \
everybody, hardly excepting even her sister, and could not like them; though \
their kindness to Jane, such as it was, had a value as arising in all \
probability from the influence of their brother's admiration. It was generally \
evident whenever they met, that he did admire her and to her it was equally \
evident that Jane was yielding to the preference which she had begun to \
entertain for him from the first, and was very anxious that no outward sign of \
admiration should now escape her, her colour changed, and she spoke with great \
animation. Elizabeth instantly read her feelings and at that moment solicitude \
for Wickham, resentment against his enemies, and everything else gave way \
before the hope of Jane's being in the fairest way for happiness.
"""


@skip_unsupported
@pytest.mark.parametrize(
    "backend",
    BACKENDS,
)
def test_logprobs_chunked_prefill_batch_invariance(
    backend,
):
    """
    Submits the same long narrative prompt in multiple batch sizes with chunked
    prefill enabled (max_num_batched_tokens=2048) and verifies that every output
    produces bitwise-identical logprobs regardless of batch size.

    With a ~500-token prompt and MAX_BATCH_SIZE=10, batches of 5+ requests will
    exceed the 2048-token window and trigger chunked prefill.  For Mamba and
    hybrid (attention+SSM) models the scheduler must ensure splits land on Mamba
    chunk boundaries; this test exposes any off-boundary divergence.

    The test fails fast on the first divergence to avoid wasting GPU time.
    """
    seed = int(os.getenv("VLLM_TEST_SEED", "12345"))
    random.seed(seed)
    tp_size = int(os.getenv("VLLM_TEST_TP_SIZE", "1"))

    from vllm import envs

    disable_custom_ar = envs.VLLM_BATCH_INVARIANT

    if disable_custom_ar:
        print(f"\n{'=' * 80}")
        print(f"BATCH INVARIANCE MODE: Disabling custom all-reduce (TP={tp_size})")
        print(f"{'=' * 80}\n")

    llm = LLM(
        model=TEST_MODEL,
        tensor_parallel_size=tp_size,
        max_num_seqs=128,
        max_model_len=8192,
        max_num_batched_tokens=2048,
        dtype="bfloat16",
        gpu_memory_utilization=0.9,
        enforce_eager=IS_DEVICE_CAPABILITY_BELOW_90,
        attention_config={"backend": backend},
        mamba_cache_dtype="float32",
    )

    sp = SamplingParams(
        temperature=0.0,
        top_p=0.5,
        max_tokens=128,
        seed=1234,
        logprobs=5,
    )

    prompt = LONG_NARRATIVE_PROMPT
    prompt_token_ids = llm.get_tokenizer().encode(prompt)
    total_runs = sum(BATCH_SIZES)

    print("\n" + "=" * 80)
    print(
        f"STARTING CHUNKED-PREFILL BATCH INVARIANCE TEST "
        f"(batch sizes {BATCH_SIZES}, {total_runs} total runs, "
        f"prompt tokens={len(prompt_token_ids)})"
    )
    print("=" * 80 + "\n")

    baseline_logprobs = None
    baseline_tokens = None
    total_checked = 0

    for batch_size in BATCH_SIZES:
        prompts = [
            TokensPrompt(prompt=prompt, prompt_token_ids=prompt_token_ids)
            for _ in range(batch_size)
        ]
        outs = llm.generate(prompts, sp, use_tqdm=False)
        assert len(outs) == batch_size

        for run_in_batch, o in enumerate(outs):
            step_logprobs, token_ids = _extract_step_logprobs(o)
            if step_logprobs is None:
                pytest.skip(
                    "Logits are not available on RequestOutput; "
                    "enable logprobs return to run this test."
                )

            if baseline_logprobs is None:
                baseline_logprobs = step_logprobs
                baseline_tokens = token_ids
                print(f"[Baseline] batch_size=1, idx 0 tokens={token_ids}")
                total_checked += 1
                continue
            assert baseline_tokens is not None
            total_checked += 1
            run_label = f"batch_size={batch_size}, idx {run_in_batch}"

            if len(baseline_logprobs) != len(step_logprobs):
                print(
                    f"\n[DIVERGENCE] {run_label}: "
                    f"different step count "
                    f"{len(baseline_logprobs)} vs {len(step_logprobs)}"
                )
                print(f"  Baseline tokens: {baseline_tokens}")
                print(f"  Current  tokens: {token_ids}")
                pytest.fail(
                    f"Divergence at {run_label}: step count mismatch "
                    f"({len(baseline_logprobs)} vs {len(step_logprobs)}). "
                    f"Checked {total_checked}/{total_runs} runs."
                )

            if baseline_tokens != token_ids:
                print(f"\n[DIVERGENCE] {run_label}: different tokens sampled")
                print(f"  Baseline tokens: {baseline_tokens}")
                print(f"  Current  tokens: {token_ids}")
                pytest.fail(
                    f"Divergence at {run_label}: different tokens sampled. "
                    f"Checked {total_checked}/{total_runs} runs."
                )

            for t, (a, b) in enumerate(zip(baseline_logprobs, step_logprobs)):
                if a.shape != b.shape:
                    print(
                        f"\n[DIVERGENCE] {run_label}, "
                        f"step {t}: shape mismatch {a.shape} vs {b.shape}"
                    )
                    pytest.fail(
                        f"Divergence at {run_label}, step {t}: "
                        f"shape mismatch {a.shape} vs {b.shape}. "
                        f"Checked {total_checked}/{total_runs} runs."
                    )

                if not torch.equal(a, b):
                    max_diff = torch.abs(a - b).max().item()
                    print(
                        f"\n[DIVERGENCE] {run_label}, step {t}: max_diff={max_diff:.6e}"
                    )
                    baseline_tok = (
                        baseline_tokens[t] if t < len(baseline_tokens) else "N/A"
                    )
                    current_tok = token_ids[t] if t < len(token_ids) else "N/A"
                    print(
                        f"  Token IDs: baseline={baseline_tok}, current={current_tok}"
                    )
                    print(f"  Baseline logprobs: {a.tolist()}")
                    print(f"  Current  logprobs: {b.tolist()}")
                    print(
                        "  Baseline all logprobs: "
                        + str(
                            [
                                baseline_logprobs[s].tolist()
                                for s in range(len(baseline_logprobs))
                            ]
                        )
                    )
                    print(
                        "  Current  all logprobs: "
                        + str(
                            [
                                step_logprobs[s].tolist()
                                for s in range(len(step_logprobs))
                            ]
                        )
                    )
                    pytest.fail(
                        f"Divergence at {run_label}, step {t}: "
                        f"bitwise mismatch (max_diff={max_diff:.6e}). "
                        f"Checked {total_checked}/{total_runs} runs."
                    )

        print(
            f"[batch_size={batch_size}] {batch_size} runs OK "
            f"(total checked: {total_checked}/{total_runs})"
        )

    print(f"\n{'=' * 80}")
    print(
        f"SUCCESS: All {total_checked} runs produced bitwise-identical "
        f"logprobs across batch sizes {BATCH_SIZES} with chunked prefill "
        f"(max_num_batched_tokens=2048)."
    )
    print(f"{'=' * 80}\n")

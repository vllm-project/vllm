# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerical checks against an independent, teacher-forced target model."""

import json
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

import torch
import xgrammar as xgr
from tokenizers.decoders import DecodeStream

from tests.conftest import HfRunner
from vllm import SamplingParams
from vllm.logprobs import Logprob
from vllm.outputs import RequestOutput
from vllm.tokenizers.detokenizer_utils import convert_ids_list_to_tokens
from vllm.v1.engine.logprobs import LogprobsProcessor


@dataclass(frozen=True)
class AccuracyTolerance:
    logprob_atol: float
    greedy_atol: float
    logprob_mean_atol: float = 0.1
    greedy_total_atol: float = 0.5

    def logprob_error(self, reference: torch.Tensor) -> torch.Tensor:
        return torch.full_like(reference, self.logprob_atol)


def check_logprobs(
    scores: dict[int, Logprob],
    reference: torch.Tensor,
    token: int,
    requested: int,
    tolerance: AccuracyTolerance,
    context: str,
) -> torch.Tensor:
    """Check every score, reported rank, and the complete requested top-k."""
    assert all(0 <= token_id < reference.numel() for token_id in scores), context
    assert token in scores, f"{context}: missing score for token {token}"
    assert max(1, requested) <= len(scores) <= requested + 1, context
    assert not torch.isnan(reference).any(), context
    assert all(s.rank is not None and s.rank >= 1 for s in scores.values()), context
    ids = torch.tensor(list(scores), device=reference.device)
    actual = torch.tensor(
        [score.logprob for score in scores.values()], device=reference.device
    )
    expected = reference[ids]
    finite = torch.isfinite(expected)
    assert torch.equal(torch.isfinite(actual), finite), (
        f"{context}: finite/masked logprobs differ: {scores}"
    )
    assert torch.equal(actual[~finite], expected[~finite]), context
    assert (actual[finite] <= 0).all(), f"{context}: positive log probability"
    errors = (actual[finite] - expected[finite]).abs()
    bounds = tolerance.logprob_error(expected[finite])
    assert (errors <= bounds).all(), (
        f"{context}: token IDs={ids[finite].tolist()}, "
        f"actual={actual[finite].tolist()}, reference={expected[finite].tolist()}, "
        f"error={errors.tolist()}, bound={bounds.tolist()}"
    )

    for token_id, score in scores.items():
        value = reference[token_id]
        if not torch.isfinite(value):
            continue
        delta = tolerance.logprob_error(value)
        lower = 1 + int((reference > value + 2 * delta).sum())
        upper = max(1, int((reference >= value - 2 * delta).sum()))
        assert score.rank is not None and lower <= score.rank <= upper, (
            f"{context}: token={token_id}, rank={score.rank}, "
            f"reference rank interval=[{lower}, {upper}]"
        )

    if requested:
        # The sampled token can be outside top-k even when it ties the kth
        # score. Its strict-greater rank can therefore duplicate a top-k rank.
        top_ids = (
            list(scores)
            if len(scores) == requested
            else [t for t in scores if t != token]
        )
        assert len(top_ids) == requested, f"{context}: incomplete top-k"
        assert {scores[t].rank for t in top_ids} == set(range(1, requested + 1)), (
            context
        )
        ordered = sorted(top_ids, key=lambda t: scores[t].rank)
        assert all(
            scores[a].logprob >= scores[b].logprob for a, b in zip(ordered, ordered[1:])
        ), f"{context}: top-k scores disagree with their reported ranks"
        boundary = reference.topk(requested).values[-1]
        if torch.isfinite(boundary):
            delta = tolerance.logprob_error(boundary)
            required = (reference > boundary + 2 * delta).nonzero().flatten()
            assert set(required.tolist()) <= scores.keys(), (
                f"{context}: missing an unambiguously top-{requested} token"
            )
            assert (reference[top_ids] >= boundary - 2 * delta).all(), (
                f"{context}: returned token below the top-{requested} boundary"
            )
    return errors


def check_greedy_token(
    logits: torch.Tensor,
    token: int,
    tolerance: AccuracyTolerance,
    context: str,
) -> float:
    assert not torch.isnan(logits).any(), context
    assert torch.isfinite(logits[token]), (
        f"{context}: selected token {token} is forbidden by the sampling parameters"
    )
    best = int(logits.argmax())
    gap = float(logits[best] - logits[token])
    assert gap <= tolerance.greedy_atol, (
        f"{context}: selected token={token}, reference best={best}, "
        f"logit gap={gap}, bound={tolerance.greedy_atol}"
    )
    return gap


def check_accuracy_budget(errors, gaps, tolerance, context):
    """Reject systematic sub-bound errors within each request and score stream."""
    if errors:
        finite_errors = torch.cat(errors)
        if finite_errors.numel():
            mean_error = float(finite_errors.mean())
            assert mean_error <= tolerance.logprob_mean_atol, (
                f"{context}: mean logprob error={mean_error}, "
                f"bound={tolerance.logprob_mean_atol}"
            )
    assert sum(gaps) <= tolerance.greedy_total_atol, (
        f"{context}: total greedy logit gap={sum(gaps)}, "
        f"bound={tolerance.greedy_total_atol}"
    )


def _check_decoded_tokens(processor, scores, history, context):
    ids = list(scores)
    decoded = convert_ids_list_to_tokens(processor.tokenizer, ids)
    decoded = processor._verify_tokens(decoded, ids, context_token_ids=history[-4:])
    for token, text in zip(ids, decoded, strict=True):
        assert scores[token].decoded_token == text, (
            f"{context}: incorrect decoded text for token {token}"
        )


@torch.inference_mode()
def check_request_accuracy(
    hf: HfRunner,
    request: RequestOutput,
    params: SamplingParams,
    tolerance: AccuracyTolerance,
    context: str,
) -> None:
    """Score the actual continuation, including positions after any divergence."""
    assert request.finished and len(request.outputs) == 1, context
    (completion,) = request.outputs
    prompt = request.prompt_token_ids
    assert prompt is not None and len(prompt) > 0, context
    tokens = list(completion.token_ids)
    assert params.max_tokens is not None and 0 < len(tokens) <= params.max_tokens, (
        context
    )
    # Incremental decoding buffers incomplete UTF-8; decoding the final token
    # list in one call can incorrectly append a replacement character.
    stream = DecodeStream(skip_special_tokens=params.skip_special_tokens)
    tokenizer = hf.tokenizer.backend_tokenizer
    for token in prompt:
        stream.step(tokenizer, token)
    visible = tokens
    if completion.finish_reason == "stop" and not params.include_stop_str_in_output:
        visible = tokens[:-1]
    expected_text = "".join(stream.step(tokenizer, token) or "" for token in visible)
    assert completion.text == expected_text, (
        f"{context}: decoded text does not match generated token IDs"
    )
    inputs = torch.tensor([prompt + tokens], device=hf.device)
    logits = hf.model(input_ids=inputs, use_cache=False).logits[0].float()
    # Use the existing byte-fallback handling with an independent request
    # history; numerical scores always come from the HF model above.
    decoder = LogprobsProcessor(hf.tokenizer, None, None, None, None, None)
    # Prompt token i is predicted by the causal logits at position i - 1.
    if params.prompt_logprobs is None:
        assert request.prompt_logprobs is None, context
    else:
        assert request.prompt_logprobs is not None, context
        assert len(request.prompt_logprobs) == len(prompt), context
        assert request.prompt_logprobs[0] is None, context
        errors = []
        for i, scores in enumerate(request.prompt_logprobs[1:], 1):
            assert scores is not None, context
            _check_decoded_tokens(decoder, scores, prompt[1:i], context)
            errors.append(
                check_logprobs(
                    scores,
                    logits[i - 1].log_softmax(-1),
                    prompt[i],
                    params.prompt_logprobs,
                    tolerance,
                    f"{context}, prompt position={i}",
                )
            )
        check_accuracy_budget(errors, [], tolerance, f"{context}, prompt logprobs")

    if params.logprobs is None:
        assert completion.logprobs is None, context
    else:
        assert completion.logprobs is not None, context
        assert len(completion.logprobs) == len(tokens), context

    reference = SamplingReference(
        hf.tokenizer,
        logits.shape[-1],
        hf.model.generation_config.to_dict(),
        params,
    )
    errors = []
    gaps = []
    for i, token in enumerate(tokens):
        raw, processed = reference.distributions(
            logits[len(prompt) + i - 1], tokens[:i]
        )
        position = f"{context}, generated position={i}"
        gaps.append(check_greedy_token(processed, token, tolerance, position))
        if completion.logprobs is not None:
            _check_decoded_tokens(decoder, completion.logprobs[i], tokens[:i], position)
            errors.append(
                check_logprobs(
                    completion.logprobs[i],
                    raw.log_softmax(-1),
                    token,
                    params.logprobs,
                    tolerance,
                    position,
                )
            )
    check_accuracy_budget(errors, gaps, tolerance, f"{context}, completion")
    # Advance the grammar through the last token too; an incomplete JSON prefix
    # is allowed at max_tokens, but every emitted token must remain valid.
    reference.accept_history(tokens)
    check_stopping(
        tokens, completion.finish_reason, reference.stop_ids, params, context
    )


def check_stopping(tokens, finish_reason, stop_ids, params, context):
    assert not params.ignore_eos and not params.stop, context
    assert not stop_ids.intersection(tokens[:-1]), (
        f"{context}: generation continued after a stop token"
    )
    if tokens[-1] in stop_ids:
        assert finish_reason == "stop", (
            f"{context}: stop token did not terminate output"
        )
    else:
        assert finish_reason == "length" and len(tokens) == params.max_tokens, context


@lru_cache(maxsize=4)
def _compiler(tokenizer, vocab_size: int):
    info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=vocab_size)
    return xgr.GrammarCompiler(info, max_threads=1, cache_enabled=True)


class SamplingReference:
    """One request's reference; histories must grow without changing their prefix."""

    def __init__(
        self,
        tokenizer,
        vocab_size: int,
        generation_config: dict,
        params: "SamplingParams",
    ):
        assert params.temperature == 0.0, "Only the original greedy matrix is supported"
        assert params.presence_penalty == 0.0 and params.repetition_penalty == 1.0
        assert params.allowed_token_ids is None and not params.logit_bias
        self.vocab_size = vocab_size
        self.min_tokens = params.min_tokens
        self.frequency_penalty = params.frequency_penalty
        self.stop_ids = set(params.stop_token_ids or [])
        for eos in (tokenizer.eos_token_id, generation_config.get("eos_token_id")):
            if eos is not None:
                self.stop_ids.update([eos] if isinstance(eos, int) else eos)
        assert all(0 <= token < vocab_size for token in self.stop_ids)

        # Match public bad-word tokenization semantics without using sampler code.
        self.bad_words: list[list[int]] = []
        for word in params.bad_words or []:
            plain = tokenizer.encode(word.lstrip(), add_special_tokens=False)
            spaced = tokenizer.encode(" " + word.lstrip(), add_special_tokens=False)
            assert plain, f"Empty bad-word tokenization: {word!r}"
            self.bad_words.append(plain)
            if spaced and len(spaced) == len(plain) and spaced[0] != plain[0]:
                self.bad_words.append(spaced)

        self.history: list[int] = []
        self.matcher = None
        self.bitmask = None
        if params.structured_outputs is not None:
            schema = params.structured_outputs.json
            assert schema is not None, "Only the original JSON schema is supported"
            # Preserve dictionary order: xgrammar's default grammar orders fields.
            schema = schema if isinstance(schema, str) else json.dumps(schema)
            compiled = _compiler(tokenizer, vocab_size).compile_json_schema(
                schema, any_whitespace=True
            )
            self.matcher = xgr.GrammarMatcher(
                compiled, override_stop_tokens=sorted(self.stop_ids) or None
            )
            self.bitmask = xgr.allocate_token_bitmask(1, vocab_size)

    def accept_history(self, generated_ids: list[int]) -> bool:
        """Validate newly generated grammar tokens, including the final EOS.

        Return whether the grammar has terminated. Use this after the final
        output token; no next-token distribution exists after termination.
        """
        assert generated_ids[: len(self.history)] == self.history, (
            "Each reference belongs to one monotonically growing request history"
        )
        if self.matcher is not None:
            for token in generated_ids[len(self.history) :]:
                assert self.matcher.accept_token(token), (
                    f"Grammar rejected token {token} at position {len(self.history)}"
                )
                self.history.append(token)
            return self.matcher.is_terminated()
        else:
            self.history = generated_ids.copy()
            return False

    def distributions(
        self, raw_logits: torch.Tensor, generated_ids: list[int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return FP32 logits before and after non-grammar sampling transforms.

        ``raw_logits`` scores the next token after the prompt and
        ``generated_ids``. Returned tensors do not alias the model's tensor.
        Normalize the first result to check reported completion logprobs;
        the second supplies the greedy decision reference.
        """
        assert raw_logits.shape == (self.vocab_size,)
        assert not self.accept_history(generated_ids), (
            "Cannot generate a distribution after grammar EOS"
        )

        grammar_logits = raw_logits.float().clone()
        if self.matcher is not None:
            assert self.bitmask is not None
            self.matcher.fill_next_token_bitmask(self.bitmask)
            words = self.bitmask[0, :, None]
            allowed = ((words >> torch.arange(32)) & 1).flatten()
            allowed = allowed[: self.vocab_size].bool().to(raw_logits.device)
            grammar_logits.masked_fill_(~allowed, -torch.inf)

        processed = grammar_logits.clone()
        for bad_word in self.bad_words:
            prefix = bad_word[:-1]
            if not prefix or generated_ids[-len(prefix) :] == prefix:
                processed[bad_word[-1]] = -torch.inf

        if len(generated_ids) < self.min_tokens and self.stop_ids:
            stop_ids = sorted(self.stop_ids)
            stop_scores = processed[stop_ids].clone()
            processed[stop_ids] = -torch.inf
            # A completed grammar may have no continuation except a stop token.
            if self.matcher is not None and torch.isneginf(processed).all():
                processed[stop_ids] = stop_scores

        if self.frequency_penalty:
            for token, count in Counter(generated_ids).items():
                processed[token] -= self.frequency_penalty * count
        return grammar_logits, processed

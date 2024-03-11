# Copyright (c) 2023, Tri Dao.
# Adapted from https://github.com/NVIDIA/Megatron-LM/blob/0bb597b42c53355a567aba2a1357cc34b9d99ddd/megatron/text_generation/forward_step.py#L31
import gc
import time
from collections import namedtuple
from dataclasses import dataclass, field
from functools import partial
from typing import Callable, Optional, Sequence, Union

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor
from torch.profiler import ProfilerActivity, profile, record_function
from transformers.generation import GreedySearchDecoderOnlyOutput, SampleDecoderOnlyOutput


@dataclass
class InferenceParams:
    """Inference parameters that are passed to the main model in order
    to efficienly calculate and store the context during inference."""

    max_seqlen: int
    max_batch_size: int
    seqlen_offset: int = 0
    batch_size_offset: int = 0
    key_value_memory_dict: dict = field(default_factory=dict)
    lengths_per_sample: Optional[Tensor] = None

    def reset(self, max_seqlen, max_batch_size):
        self.max_seqlen = max_seqlen
        self.max_batch_size = max_batch_size
        self.seqlen_offset = 0
        if self.lengths_per_sample is not None:
            self.lengths_per_sample.zero_()


# https://github.com/NVIDIA/Megatron-LM/blob/0bb597b42c53355a567aba2a1357cc34b9d99ddd/megatron/text_generation/sampling.py
# https://github.com/huggingface/transformers/blob/a44985b41cfa2de48a5e1de7f1f93b7483da25d1/src/transformers/generation/logits_process.py#L231
def modify_logits_for_top_k_filtering(logits, top_k):
    """Set the logits for none top-k values to -inf. Done in-place."""
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits.masked_fill_(indices_to_remove, float("-Inf"))


# https://github.com/NVIDIA/Megatron-LM/blob/0bb597b42c53355a567aba2a1357cc34b9d99ddd/megatron/text_generation/sampling.py
# https://github.com/huggingface/transformers/blob/a44985b41cfa2de48a5e1de7f1f93b7483da25d1/src/transformers/generation/logits_process.py#L170
def modify_logits_for_top_p_filtering(logits, top_p):
    """Set the logits for none top-p values to -inf. Done in-place."""
    if top_p <= 0.0 or top_p >= 1.0:
        return
    # First sort and calculate cumulative sum of probabilities.
    sorted_logits, sorted_indices = torch.sort(logits, descending=False)
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
    sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
    # scatter sorted tensors to original indexing
    indices_to_remove = sorted_indices_to_remove.scatter(
        1, sorted_indices, sorted_indices_to_remove
    )
    logits.masked_fill_(indices_to_remove, float("-inf"))


def sample(logits, top_k=1, top_p=0.0, temperature=1.0):
    """Sample from top-k logits.
    Arguments:
        logits: Tensor of shape (batch_size, vocab_size)
    """
    if top_k == 1:  # Short-circuit for greedy decoding
        return logits.argmax(dim=-1)
    else:
        if top_p > 0.0:
            assert top_p <= 1.0, "top-p should be in (0, 1]."
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))  # Safety check
            logits_top, indices = torch.topk(logits, top_k, dim=-1)
            if temperature != 1.0:
                logits_top /= temperature
            modify_logits_for_top_p_filtering(logits_top, top_p)
            return indices[
                torch.arange(indices.shape[0], device=indices.device),
                torch.multinomial(torch.softmax(logits_top, dim=-1), num_samples=1).squeeze(dim=-1),
            ]
        else:
            # Clone so that when we modify for top_p we don't change the original logits
            logits_top = logits / temperature if temperature != 1.0 else logits.clone()
            modify_logits_for_top_p_filtering(logits_top, top_p)
            return torch.multinomial(torch.softmax(logits_top, dim=-1), num_samples=1).squeeze(
                dim=-1
            )


@torch.inference_mode()
def decode(
    input_ids,
    model,
    max_length,
    top_k=1,
    top_p=0.0,
    temperature=1.0,
    eos_token_id=None,
    teacher_outputs=None,
    vocab_size=None,
    tensor_parallel=1,
    cg=False,
    enable_timing=False,
):
    """Decoding, either greedy or with top-k or top-p sampling.
    If top-k = 0, don't limit the number of candidates (pure sampling).
    Top-k and top-p can be used together. If top_k > 0 and top_p > 0, then top-k is applied first,
    then top-p.
    We assume that all sequences in the same batch have the same length.

    Arguments:
        input_ids: (batch, seq_len)
        max_length: int
        teacher_outputs (optional): (batch, seq_len). If provided, instead of sampling from the
            logits, the next token is taken from the teacher_outputs. Useful for testing.
    Returns: GreedySearchDecoderOnlyOutput or SampleDecoderOnlyOutput, with the following fields:
        sequences: (batch, max_length)
        scores: tuples of (batch, vocab_size)
    """
    batch_size, seqlen_og = input_ids.shape
    teacher_output_len = teacher_outputs.shape[1] if teacher_outputs is not None else 0
    if cg:
        if not hasattr(model, "_decoding_cache"):
            model._decoding_cache = None
        model._decoding_cache = update_graph_cache(
            model,
            model._decoding_cache,
            batch_size,
            seqlen_og,
            max_length,
            tensor_parallel=tensor_parallel,
        )
        inference_params = model._decoding_cache.inference_params
        inference_params.reset(max_length, batch_size)
    else:
        inference_params = InferenceParams(max_seqlen=max_length, max_batch_size=batch_size)

    def get_logits(input_ids, inference_params):
        decoding = inference_params.seqlen_offset > 0
        if decoding:
            position_ids = torch.full(
                (batch_size, 1),
                inference_params.seqlen_offset,
                dtype=torch.long,
                device=input_ids.device,
            )
        else:
            position_ids = None
        if not cg or not decoding:
            logits = model(
                input_ids,
                position_ids=position_ids,
                inference_params=inference_params,
                num_last_tokens=1,
            ).logits.squeeze(dim=1)
        else:
            logits = model._decoding_cache.run(
                input_ids, position_ids, inference_params.seqlen_offset
            ).squeeze(dim=1)
        return logits[..., :vocab_size] if vocab_size is not None else logits

    def sample_tokens(logits, inference_params):
        if teacher_outputs is None or teacher_output_len <= inference_params.seqlen_offset:
            token = sample(logits, top_k=top_k, top_p=top_p, temperature=temperature)
        else:
            token = teacher_outputs[:, inference_params.seqlen_offset]
        # return rearrange(token, "b -> b 1")
        return token.unsqueeze(1)

    def should_stop(current_token, inference_params):
        if inference_params.seqlen_offset == 0:
            return False
        if eos_token_id is not None and (current_token == eos_token_id).all():
            return True
        if inference_params.seqlen_offset >= max_length - 1:
            return True
        return False

    start = torch.cuda.Event(enable_timing=enable_timing)
    end = torch.cuda.Event(enable_timing=enable_timing)

    if enable_timing:
        if tensor_parallel > 1:
            torch.distributed.barrier()
        start.record()
    scores, sequences = [], [input_ids]
    while not should_stop(sequences[-1], inference_params):
        scores.append(get_logits(sequences[-1], inference_params))
        inference_params.seqlen_offset += sequences[-1].shape[1]
        sequences.append(sample_tokens(scores[-1], inference_params))
    if enable_timing:
        end.record()
        if tensor_parallel > 1:
            torch.distributed.barrier()
        torch.cuda.synchronize()
        print(f"Prompt processing + decoding time: {(start.elapsed_time(end)):.0f}ms")
    output_cls = GreedySearchDecoderOnlyOutput if top_k == 1 else SampleDecoderOnlyOutput
    return output_cls(sequences=torch.cat(sequences, dim=1), scores=tuple(scores))


def sample_speculative(logits, logits_draft, tokens_draft, top_k=1, top_p=0.0, temperature=1.0):
    """Algorithm 1 from [1]
    [1] Fast Inference from Transformers via Speculative Decoding
    Yaniv Leviathan, Matan Kalman, Yossi Matias
    https://arxiv.org/abs/2211.17192

    Arguments:
        logits: Tensor of shape (batch_size, seqlen + 1, vocab_size)
        logits_draft: Tensor of shape (batch_size, seqlen, vocab_size)
        tokens_draft: Tensor of shape (batch_size, seqlen)
    Return:
        tokens: Tensor of shape (batch_size, seqlen + 1)
        num_generated_tokens: Tensor of shape (batch_size), with value in [1, seqlen + 1].
            For each sequence in the batch, the number of valid tokens that were sampled by
            speculative sampling.
    """
    batch, seqlen_p_1, vocab_size = logits.shape
    seqlen = seqlen_p_1 - 1
    assert logits_draft.shape == (batch, seqlen, vocab_size)
    assert tokens_draft.shape == (batch, seqlen)
    assert tokens_draft.dtype in [torch.int64, torch.int32]
    # TODO: if top_k = 1 we can simplify things and only work with indices
    if top_p > 0.0:
        assert top_p <= 1.0, "top-p should be in (0, 1]."
    # Clone so that when we modify for top_p we don't change the original logits
    logits = logits / temperature if temperature != 1.0 else logits.clone()
    logits_draft = logits_draft / temperature if temperature != 1.0 else logits_draft.clone()
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))  # Safety check
        modify_logits_for_top_k_filtering(logits, top_k)
        modify_logits_for_top_k_filtering(logits_draft, top_k)
    modify_logits_for_top_p_filtering(logits, top_p)
    modify_logits_for_top_p_filtering(logits_draft, top_p)
    probs = torch.softmax(logits, dim=-1)
    probs_draft = torch.softmax(logits_draft, dim=-1)
    gather = lambda probs, tokens: rearrange(
        probs.gather(dim=-1, index=rearrange(tokens, "... -> ... 1")), "... 1 -> ..."
    )
    # (batch, seqlen)
    accepted = torch.rand(batch, seqlen, device=probs.device) * gather(
        probs_draft, tokens_draft
    ) <= gather(probs[:, :-1], tokens_draft)
    accepted_all = accepted.all(dim=-1)
    # (batch,)
    first_rejected_idx = torch.where(accepted_all, seqlen, accepted.int().argmin(dim=-1))
    probs_diff = torch.clamp(probs[:, :-1] - probs_draft, min=0.0)
    # torch.multinomial can deal with unnormalized probabilities
    # probs_diff /= probs_diff.sum(dim=-1, keepdim=True)
    resample_probs = torch.cat([probs_diff, probs[:, -1:]], dim=1)
    resample_probs = rearrange(
        resample_probs.gather(dim=1, index=repeat(first_rejected_idx, "b -> b 1 d", d=vocab_size)),
        "b 1 d -> b d",
    )
    resample = torch.multinomial(resample_probs, num_samples=1).squeeze(dim=-1)  # (batch,)
    tokens = F.pad(tokens_draft, (0, 1))
    tokens[:, first_rejected_idx] = resample
    return tokens, first_rejected_idx + 1


@torch.inference_mode()
def decode_speculative(
    input_ids,
    model,
    model_draft,
    max_length,
    speculative_lookahead=3,
    top_k=1,
    top_p=0.0,
    temperature=1.0,
    eos_token_id=None,
    vocab_size=None,
    tensor_parallel=1,
    cg=False,
    enable_timing=False,
    debug=False,
):
    """
    TD: WIP, for my own understanding, lightly tested. Only support batch_size == 1 for now.

    Speculative decoding, either greedy or with top-k or top-p sampling.
    If top-k = 0, don't limit the number of candidates (pure sampling).
    Top-k and top-p can be used together. If top_k > 0 and top_p > 0, then top-k is applied first,
    then top-p.
    We assume that all sequences in the same batch have the same length.

    Arguments:
        input_ids: (batch, seq_len)
        max_length: int
    Returns: GreedySearchDecoderOnlyOutput or SampleDecoderOnlyOutput, with the following fields:
        sequences: (batch, max_length)
        scores: tuples of (batch, vocab_size)
    """
    batch_size, seqlen_og = input_ids.shape
    assert batch_size == 1, "Speculative decoding implementation only supports batch_size=1"
    assert eos_token_id is None, "Speculative decoding implementation doesn't support eos_token_id"
    if cg:
        if not hasattr(model_draft, "_decoding_cache"):
            model_draft._decoding_cache = None
        model_draft._decoding_cache = update_graph_cache(
            model_draft,
            model_draft._decoding_cache,
            batch_size,
            seqlen_og,
            max_length,
            # draft model needs to process either 1 or 2 tokens at a time
            decoding_seqlens=(1, 2),
            tensor_parallel=tensor_parallel,
        )
        inference_params_draft = model_draft._decoding_cache.inference_params
        inference_params_draft.reset(max_length, batch_size)
        if not hasattr(model, "_decoding_cache"):
            model._decoding_cache = None
        model._decoding_cache = update_graph_cache(
            model,
            model._decoding_cache,
            batch_size,
            seqlen_og,
            max_length,
            decoding_seqlens=range(1, speculative_lookahead + 2),
            tensor_parallel=tensor_parallel,
        )
        inference_params = model._decoding_cache.inference_params
        inference_params.reset(max_length, batch_size)
    else:
        inference_params_draft = InferenceParams(max_seqlen=max_length, max_batch_size=batch_size)
        inference_params = InferenceParams(max_seqlen=max_length, max_batch_size=batch_size)

    def get_logits(input_ids, inference_params, model, num_last_tokens=1, cg=False):
        decoding = inference_params.seqlen_offset > 0
        if decoding:
            seqlen = input_ids.shape[1]
            # if inference_params.lengths_per_sample is None:
            # TODO: in the case of batched decoding where each sequence has a different length,
            # we need to compute the position_ids for each sequence using lengths_per_sample
            if True:
                cache_seqlens = torch.full(
                    (input_ids.shape[0],),
                    inference_params.seqlen_offset,
                    dtype=torch.int32,
                    device=input_ids.device,
                )
            else:
                cache_seqlens = inference_params.lengths_per_sample
            position_ids = cache_seqlens[:, None] + torch.arange(
                seqlen, dtype=torch.long, device=input_ids.device
            )
        else:
            position_ids = None
        if not cg or not decoding:
            logits = model(
                input_ids,
                position_ids=position_ids,
                inference_params=inference_params,
                num_last_tokens=num_last_tokens,
            ).logits
        else:
            # NOTE: careful, CUDA graph is set to have num_last_tokens=input_ids.shape[1].
            # This might not be compatible the num_last_tokens used here.
            assert num_last_tokens <= input_ids.shape[1]
            logits = model._decoding_cache.run(
                input_ids, position_ids, inference_params.seqlen_offset
            )[:, -num_last_tokens:]
        return logits[..., :vocab_size] if vocab_size is not None else logits

    def sample_tokens(input_ids, get_logits_fn, inference_params, sample_fn, num_tokens=1):
        """Sample `num_tokens` tokens from the model, given the previous logits.
        Also return the logits of the sampled tokens.
        Arguments:
            input_ids: (batch, seqlen)
        Return:
            tokens: (batch, num_tokens)
            scores: (batch, num_tokens), which contains @previous_logits and the logits of the next
                (num_tokens - 1) tokens. The logits of the last token isn't computed.
        """
        assert num_tokens >= 1
        sequences, scores = [input_ids], []
        for i in range(num_tokens):
            scores.append(get_logits_fn(sequences[-1], inference_params)[:, -1])
            inference_params.seqlen_offset += sequences[-1].shape[1]
            sequences.append(sample_fn(scores[-1]).unsqueeze(1))
        return torch.cat(sequences[1:], dim=1), torch.stack(scores, dim=1)

    sampling_kwargs = dict(top_k=top_k, top_p=top_p, temperature=temperature)
    sample_fn = partial(sample, **sampling_kwargs)
    get_logits_main = partial(get_logits, model=model, cg=cg)
    get_logits_draft = partial(get_logits, model=model_draft, cg=cg)
    sample_tokens_main = partial(
        sample_tokens,
        get_logits_fn=get_logits_main,
        sample_fn=sample_fn,
        inference_params=inference_params,
    )
    sample_tokens_draft = partial(
        sample_tokens,
        get_logits_fn=get_logits_draft,
        sample_fn=sample_fn,
        inference_params=inference_params_draft,
    )

    if debug:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if enable_timing:
        if tensor_parallel > 1:
            torch.distributed.barrier()
        torch.cuda.synchronize()
        start = time.time()

    sequences, scores = [input_ids], []
    num_main_model_calls = 0
    num_draft_tokens = 0
    num_accepted_tokens_history = []
    if seqlen_og >= max_length - 1:
        # Don't do speculative sampling, just sample 1 token from the model
        tokens, scores_new = sample_tokens_main(input_ids, num_tokens=1)
        sequences.append(tokens)
        scores.append(scores_new)
    else:
        # Sample from draft model, which produces @n_spec_tokens, and @model
        # will then use to produce between 1 and 1 + @n_spec_tokens tokens.
        # We want seqlen_og + 1 + @n_spec_tokens to be <= @max_length.
        n_spec_tokens = min(speculative_lookahead, max_length - seqlen_og - 1)
        tokens_draft, scores_draft = sample_tokens_draft(input_ids, num_tokens=n_spec_tokens)
        num_draft_tokens += n_spec_tokens
        if debug:
            scores_draft_ref = model_draft(
                torch.cat([input_ids, tokens_draft], dim=1), num_last_tokens=n_spec_tokens + 1
            ).logits
            print((scores_draft - scores_draft_ref[:, :-1]).abs().max())

        # Evaluate the draft tokens with the model
        logits = get_logits_main(
            torch.cat([input_ids, tokens_draft], dim=1),
            inference_params,
            num_last_tokens=n_spec_tokens + 1,
        )
        num_main_model_calls += 1
        if debug:
            logits_ref = model(
                torch.cat([input_ids, tokens_draft], dim=1), num_last_tokens=n_spec_tokens + 1
            ).logits
            print((logits - logits_ref).abs().max())
            # breakpoint()
        tokens, num_generated_tokens = sample_speculative(
            logits, scores_draft, tokens_draft, **sampling_kwargs
        )
        num_accepted_tokens_history.append(num_generated_tokens - 1)
        if debug:
            print(tokens)
            print(num_generated_tokens)
            # breakpoint()
        # TODO: we're using the fact that batch_size == 1
        # TODO: check eos_token_id
        sequences.append(tokens[:1, : num_generated_tokens[0]])
        scores.append(logits[:1, : num_generated_tokens[0]])
        # Note that @model has not evaluated the last sampled token yet, so we'll need to pass
        # that in the next time we call @model.
        num_generated = num_generated_tokens[0].item()
        inference_params.seqlen_offset = seqlen_og + num_generated - 1
        inference_params_draft.seqlen_offset = (
            inference_params.seqlen_offset - 1
            if num_generated > 1
            else inference_params.seqlen_offset
        )
        if debug:
            cur_ids = torch.cat([input_ids, sequences[-1]], dim=1)
            scores_ref = model(cur_ids, num_last_tokens=num_generated_tokens[0].item() + 1).logits
            print((scores[-1] - scores_ref[:, :-1]).abs().max())
            # breakpoint()

    while True:
        # seqlen_offset is total length generated - 1
        if inference_params.seqlen_offset >= max_length - 1:
            break
        if inference_params.seqlen_offset >= max_length - 2:
            # Don't do speculative sampling, just sample 1 token from the model
            tokens, scores_new = sample_tokens_main(sequences[-1][:, -1:], num_tokens=1)
            sequences.append(tokens)
            scores.append(scores_new)
            break
        # Sample from draft model
        n_spec_tokens = min(
            speculative_lookahead, max_length - inference_params_draft.seqlen_offset - 2
        )
        # If the main model accepts all the draft tokens, plus it samples one new token,
        # then at the next iteration the draft model need to evaluate the logits of the last draft
        # token and the logits of the newly sampled token. So here we pass in the last 2 tokens
        # of sequences[-1].
        # This exception is when the main model rejects all the draft tokens, in which case we
        # will only have 1 token to pass in.
        tokens_draft, scores_draft = sample_tokens_draft(
            sequences[-1][:, -2:], num_tokens=n_spec_tokens
        )
        num_draft_tokens += n_spec_tokens
        if debug:
            scores_draft_ref = model_draft(
                torch.cat([cur_ids, tokens_draft], dim=1), num_last_tokens=n_spec_tokens + 1
            ).logits
            print((scores_draft - scores_draft_ref[:, :-1]).abs().max())
            # breakpoint()
        # Evaluate the draft tokens with the model
        logits = get_logits_main(
            torch.cat([sequences[-1][:, -1:], tokens_draft], dim=1),
            inference_params,
            num_last_tokens=n_spec_tokens + 1,
        )  # (batch, n_spec_tokens + 1, vocab_size)
        num_main_model_calls += 1
        if debug:
            logits_ref = model(
                torch.cat([cur_ids, tokens_draft], dim=1), num_last_tokens=n_spec_tokens + 1
            ).logits
            print((logits - logits_ref).abs().max())
            # breakpoint()
        tokens, num_generated_tokens = sample_speculative(
            logits, scores_draft, tokens_draft, **sampling_kwargs
        )
        num_accepted_tokens_history.append(num_generated_tokens - 1)
        if debug:
            print(tokens)
            print(num_generated_tokens)
            # breakpoint()
        sequences.append(tokens[:1, : num_generated_tokens[0]])
        scores.append(logits[:1, : num_generated_tokens[0]])
        # We've evaluated 1 token from sequences[-1][:, -1:] above, plus
        # num_generated_tokens[0].item() - 1 tokens from the draft model.
        num_generated = num_generated_tokens[0].item()
        inference_params.seqlen_offset += num_generated
        inference_params_draft.seqlen_offset = (
            inference_params.seqlen_offset - 1
            if num_generated > 1
            else inference_params.seqlen_offset
        )
        if debug:
            cur_ids = torch.cat([cur_ids, sequences[-1]], dim=1)
            scores_ref = model(cur_ids, num_last_tokens=num_generated_tokens[0].item() + 1).logits
            print((scores[-1] - scores_ref[:, :-1]).abs().max())
            # breakpoint()

    if enable_timing:
        if tensor_parallel > 1:
            torch.distributed.barrier()
        torch.cuda.synchronize()
        print(f"Prompt processing + decoding time: {(time.time() - start) * 1000:.0f}ms")
        print(f"Number of calls to main model: {num_main_model_calls}")
        print(
            f"Acceptance rate: {torch.cat(num_accepted_tokens_history).sum().item() / num_draft_tokens * 100:.2f}%"
        )
    sequences = torch.cat(sequences, dim=1)
    scores = torch.cat(scores, dim=1)
    if debug:
        scores_ref = model(sequences).logits
        print((scores - scores_ref[:, seqlen_og - 1 : -1]).abs().max())
    output_cls = GreedySearchDecoderOnlyOutput if top_k == 1 else SampleDecoderOnlyOutput
    return output_cls(sequences=sequences, scores=scores)


class GenerationMixin:
    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        raise NotImplementedError

    def generate(
        self,
        input_ids,
        max_length,
        top_k=1,
        top_p=0.0,
        temperature=1.0,
        return_dict_in_generate=False,
        output_scores=False,
        **kwargs,
    ):
        output = decode(
            input_ids, self, max_length, top_k=top_k, top_p=top_p, temperature=temperature, **kwargs
        )
        if not output_scores:
            output.scores = None
        return output if return_dict_in_generate else output.sequences


def allocate_inference_cache(
    max_batch_size,
    max_seqlen,
    nheads,
    headdim,
    layers: Union[int, Sequence],
    device,
    dtype=torch.float16,
):
    assert dtype in [torch.float16, torch.bfloat16, torch.float32]
    kv_cache_shape = (max_batch_size, max_seqlen, 2, nheads, headdim)
    if isinstance(layers, int):
        layers = range(layers)
    return {i: torch.empty(kv_cache_shape, device=device, dtype=dtype) for i in layers}


@dataclass
class DecodingCGCache:
    max_batch_size: int = 0
    max_seqlen: int = 0
    device = None
    dtype = None
    callables: dict = field(default_factory=dict)
    mempool = None
    inference_params: Optional[InferenceParams] = None
    run: Optional[Callable] = None


@torch.inference_mode()
def update_graph_cache(
    model,
    cache,
    batch_size,
    seqlen_og,
    max_seqlen,
    decoding_seqlens=(1,),
    tensor_parallel=1,
    dtype=None,
    n_warmups=2,
):
    if cache is None:
        cache = DecodingCGCache()
    param_example = next(iter(model.parameters()))
    device = param_example.device
    if dtype is None:
        dtype = param_example.dtype
    if (
        (device, dtype) != (cache.device, cache.dtype)
        or batch_size > cache.max_batch_size
        or max_seqlen > cache.max_seqlen
    ):  # Invalidate the cache
        cache.callables = {}
        cache.mempool = None
        cache.inference_params = None
        gc.collect()
        cache.device, cache.dtype = device, dtype
        cache.max_batch_size, cache.max_seqlen = batch_size, max_seqlen
        if hasattr(model, "allocate_inference_cache"):
            inf_cache = model.allocate_inference_cache(batch_size, max_seqlen, dtype)
        else:
            headdim = getattr(
                model.config,
                "head_dim",
                model.config.hidden_size // model.config.num_attention_heads,
            )
            inf_cache = allocate_inference_cache(
                batch_size,
                max_seqlen,
                model.config.num_attention_heads // tensor_parallel,
                headdim,
                model.config.num_hidden_layers,
                device,
                dtype,
            )
        lengths_per_sample = torch.full((batch_size,), seqlen_og, dtype=torch.int32, device=device)
        cache.inference_params = InferenceParams(
            max_seqlen=max_seqlen,
            max_batch_size=batch_size,
            seqlen_offset=seqlen_og,
            key_value_memory_dict=inf_cache,
            lengths_per_sample=lengths_per_sample,
        )
        cache.mempool = torch.cuda.graphs.graph_pool_handle()
    for decoding_seqlen in decoding_seqlens:
        if (batch_size, decoding_seqlen) not in cache.callables:
            cache.callables[batch_size, decoding_seqlen] = capture_graph(
                model,
                cache.inference_params,
                batch_size,
                max_seqlen,
                decoding_seqlen=decoding_seqlen,
                mempool=cache.mempool,
                n_warmups=n_warmups,
            )

    def dispatch(input_ids, position_ids, seqlen):
        batch_size, decoding_seqlen = input_ids.shape[:2]
        return cache.callables[batch_size, decoding_seqlen](input_ids, position_ids, seqlen)

    cache.run = dispatch
    cache.inference_params.seqlen_offset = 0  # Reset so it's not confusing
    return cache


def capture_graph(
    model, inference_params, batch_size, max_seqlen, decoding_seqlen=1, mempool=None, n_warmups=2
):
    device = next(iter(model.parameters())).device
    input_ids = torch.full((batch_size, decoding_seqlen), 0, dtype=torch.long, device=device)
    position_ids = torch.full((batch_size, decoding_seqlen), 0, dtype=torch.long, device=device)
    seqlen_offset_og = inference_params.seqlen_offset
    inference_params.seqlen_offset = max_seqlen - decoding_seqlen
    inference_params.lengths_per_sample[:] = inference_params.seqlen_offset

    # Warmup before capture
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(n_warmups):
            logits = model(
                input_ids,
                position_ids=position_ids,
                inference_params=inference_params,
                num_last_tokens=decoding_seqlen,
            ).logits
        s.synchronize()
        # This might be needed for correctness if we run with NCCL_GRAPH_MIXING_SUPPORT=0,
        # which requires that graph launch and non-captured launch to not overlap (I think,
        # that's how I interpret the documentation). I'm not sure if this is required.
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
    torch.cuda.current_stream().wait_stream(s)
    # Captures the graph
    # To allow capture, automatically sets a side stream as the current stream in the context
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=mempool):
        logits = model(
            input_ids,
            position_ids=position_ids,
            inference_params=inference_params,
            num_last_tokens=decoding_seqlen,
        ).logits

    def run(new_input_ids, new_position_ids, seqlen):
        inference_params.lengths_per_sample[:] = seqlen
        input_ids.copy_(new_input_ids)
        position_ids.copy_(new_position_ids)
        graph.replay()
        return logits.clone()

    inference_params.seqlen_offset = seqlen_offset_og
    return run

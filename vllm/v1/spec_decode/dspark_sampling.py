# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable

import torch

from vllm.logger import init_logger
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.ops.topk_topp_sampler import (
    apply_top_k_top_p,
    empty_exponential_noise_like,
    sample_with_exponential_noise,
)
from vllm.v1.sample.sampler import _SAMPLING_EPS

MarkovBiasFn = Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor]

logger = init_logger(__name__)

# Per-step seed source for the fused Markov sampler's in-kernel Philox noise.
#
# CRITICAL for TP>1: the DSpark draft sampler runs redundantly on every TP rank
# over the same all-gathered logits, and all ranks MUST draw identical draft
# tokens — otherwise the ranks' KV/context diverge and generation corrupts. The
# reference sampler gets this for free from vLLM's cross-rank-synchronized global
# RNG (``torch.exponential_``). This counter must therefore start identical on
# every rank (NOT from per-process entropy like os.urandom) and advance in
# lockstep with the SPMD proposer loop, so every rank produces the same seed at
# the same step. Determinism of the *draft* noise across runs is harmless: output
# randomness still comes from the target's synchronized rejection sampler, and
# the counter keeps advancing across requests so proposals are not repeated.
# If sampling is ever CUDA-graph-captured, keep this seed outside replay or pass
# it as graph input; otherwise the replayed graph would reuse one draft seed.
_FUSED_MARKOV_SEED_COUNTER = 0x9E3779B1


def _next_fused_markov_seed() -> int:
    global _FUSED_MARKOV_SEED_COUNTER
    _FUSED_MARKOV_SEED_COUNTER = (_FUSED_MARKOV_SEED_COUNTER + 1) & 0x7FFFFFFF
    return _FUSED_MARKOV_SEED_COUNTER


def _expand_draft_sampling_tensor(
    tensor: torch.Tensor | None,
    num_tokens: int,
) -> torch.Tensor | None:
    if tensor is None or tensor.shape[0] == num_tokens:
        return tensor
    batch_size = tensor.shape[0]
    if num_tokens % batch_size != 0:
        raise ValueError(
            "Draft sampling metadata must either match the draft logits row "
            f"count or divide it evenly: metadata={batch_size}, rows={num_tokens}"
        )
    return tensor.repeat_interleave(num_tokens // batch_size)


def _expand_draft_sampling_generators(
    generators: dict[int, torch.Generator],
    batch_size: int,
    num_tokens: int,
) -> dict[int, torch.Generator]:
    if not generators or batch_size == num_tokens:
        return generators
    if num_tokens % batch_size != 0:
        raise ValueError(
            "Draft sampling generators must either match the draft logits row "
            f"count or divide it evenly: metadata={batch_size}, rows={num_tokens}"
        )
    repeat = num_tokens // batch_size
    return {
        req_idx * repeat + offset: generator
        for req_idx, generator in generators.items()
        for offset in range(repeat)
    }


def _sample_from_logits(
    logits: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    use_fp64_gumbel: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if sampling_metadata.all_greedy:
        return logits.argmax(dim=-1), logits

    assert sampling_metadata.temperature is not None
    num_tokens = logits.shape[0]
    logits = logits.to(torch.float32)
    temperature = _expand_draft_sampling_tensor(
        sampling_metadata.temperature, num_tokens
    )
    assert temperature is not None
    if not sampling_metadata.all_random:
        is_greedy = temperature < _SAMPLING_EPS
        temperature = torch.where(is_greedy, 1.0, temperature)
    logits.div_(temperature.view(-1, 1))
    top_k = _expand_draft_sampling_tensor(sampling_metadata.top_k, num_tokens)
    top_p = _expand_draft_sampling_tensor(sampling_metadata.top_p, num_tokens)
    logits = apply_top_k_top_p(logits, top_k, top_p)
    probs = logits.softmax(dim=-1, dtype=torch.float32)

    generators = _expand_draft_sampling_generators(
        sampling_metadata.generators,
        sampling_metadata.temperature.shape[0],
        num_tokens,
    )
    q = empty_exponential_noise_like(probs, use_fp64_gumbel)
    if len(generators) != num_tokens:
        q.exponential_()
    for i, generator in generators.items():
        q[i].exponential_(generator=generator)
    next_token_ids = sample_with_exponential_noise(probs.clone(), q)
    if not sampling_metadata.all_random:
        greedy_token_ids = probs.argmax(dim=-1)
        next_token_ids = torch.where(is_greedy, greedy_token_ids, next_token_ids)
    return next_token_ids, probs


def sample_dspark_markov_block(
    base_logits: torch.Tensor,
    first_prev_token_ids: torch.Tensor,
    apply_markov_bias: MarkovBiasFn,
    sampling_metadata: SamplingMetadata,
    *,
    return_probs: bool,
    use_fp64_gumbel: bool = False,
    tokens_out: torch.Tensor | None = None,
    draft_probs_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Sample a DSpark block with sequential Markov correction."""
    if base_logits.dim() != 3:
        raise ValueError(
            "DSpark Markov sampling expects [batch, block, vocab] logits, "
            f"got shape {tuple(base_logits.shape)}"
        )
    batch_size, block_size, _ = base_logits.shape
    if first_prev_token_ids.shape[0] != batch_size:
        raise ValueError(
            "first_prev_token_ids must have one entry per request: "
            f"got {first_prev_token_ids.shape[0]} for batch {batch_size}"
        )

    prev_token_ids = first_prev_token_ids.long()
    if tokens_out is not None and (
        tokens_out.shape[0] < batch_size
        or tokens_out.shape[1] < block_size
        or tokens_out.dtype is not torch.int64
        or tokens_out.device != base_logits.device
    ):
        tokens_out = None
    tokens = (
        tokens_out[:batch_size, :block_size]
        if tokens_out is not None
        else torch.empty(
            (batch_size, block_size),
            dtype=torch.int64,
            device=base_logits.device,
        )
    )
    draft_probs = None
    if return_probs and not sampling_metadata.all_greedy:
        vocab_size = base_logits.shape[-1]
        if draft_probs_out is not None and (
            draft_probs_out.shape[0] < batch_size
            or draft_probs_out.shape[1] < block_size
            or draft_probs_out.shape[2] != vocab_size
            or draft_probs_out.dtype is not torch.float32
            or draft_probs_out.device != base_logits.device
        ):
            draft_probs_out = None
        draft_probs = (
            draft_probs_out[:batch_size, :block_size, :]
            if draft_probs_out is not None
            else torch.empty(
                (batch_size, block_size, vocab_size),
                dtype=torch.float32,
                device=base_logits.device,
            )
        )

    for step_idx in range(block_size):
        step_logits = apply_markov_bias(
            base_logits[:, step_idx, :], prev_token_ids, step_idx
        )
        if return_probs and not sampling_metadata.all_greedy:
            next_token_ids, probs = _sample_from_logits(
                step_logits, sampling_metadata, use_fp64_gumbel
            )
            assert draft_probs is not None
            draft_probs[:, step_idx, :].copy_(probs)
        else:
            next_token_ids = step_logits.argmax(dim=-1)

        tokens[:, step_idx].copy_(next_token_ids)
        prev_token_ids = next_token_ids

    if draft_probs is None:
        return tokens, None
    return tokens, draft_probs


def sample_dspark_markov_block_fused(
    base_logits: torch.Tensor,
    first_prev_token_ids: torch.Tensor,
    apply_markov_bias: MarkovBiasFn,
    sampling_metadata: SamplingMetadata,
    *,
    use_fp64_gumbel: bool = False,
    tokens_out: torch.Tensor | None = None,
    draft_probs_out: torch.Tensor | None = None,
    block_v: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Fused-kernel variant of the probabilistic Markov block sampler.

    Replaces the per-step eager sampler (``to(f32)`` -> temperature div ->
    softmax -> exponential-noise -> ``clone`` -> div -> argmax -> probs copy,
    ~10 launches/step) with three Triton launches/step that fuse temperature
    scaling, softmax, in-kernel Gumbel sampling, and the draft-probs write. The
    Markov ``w1``/``w2`` still run through ``apply_markov_bias`` so the
    full-vocab ``w2`` weight is read exactly once per step.

    Falls back to :func:`sample_dspark_markov_block` (bit-identical reference)
    whenever the fused path's assumptions do not hold: non-CUDA tensors,
    all-greedy batches, fp64 Gumbel noise, or per-request seeded generators
    (reproducibility). Top-k/top-p constrained batches stay on the fused
    softmax/sample/probs writer after pre-masking the step logits with the
    shared sampler helper, preserving the reference distribution without
    duplicating its sort-based top-p logic.
    """
    from vllm.models.deepseek_v4.nvidia.dspark_triton import (
        dspark_markov_probs_sample,
    )

    if (
        not base_logits.is_cuda
        or sampling_metadata.all_greedy
        or use_fp64_gumbel
        or len(sampling_metadata.generators) > 0
    ):
        return sample_dspark_markov_block(
            base_logits,
            first_prev_token_ids,
            apply_markov_bias,
            sampling_metadata,
            return_probs=True,
            use_fp64_gumbel=use_fp64_gumbel,
            tokens_out=tokens_out,
            draft_probs_out=draft_probs_out,
        )

    if base_logits.dim() != 3:
        raise ValueError(
            "DSpark Markov sampling expects [batch, block, vocab] logits, "
            f"got shape {tuple(base_logits.shape)}"
        )
    batch_size, block_size, vocab_size = base_logits.shape
    if first_prev_token_ids.shape[0] < batch_size:
        raise ValueError(
            "first_prev_token_ids must have one entry per request: "
            f"got {first_prev_token_ids.shape[0]} for batch {batch_size}"
        )
    assert sampling_metadata.temperature is not None
    device = base_logits.device
    logger.info_once(
        "DSpark fused Markov sampler engaged: batch=%d block=%d vocab=%d",
        batch_size,
        block_size,
        vocab_size,
    )

    # Temperature and greedy-row selection are step-independent; compute once.
    temperature = _expand_draft_sampling_tensor(
        sampling_metadata.temperature, batch_size
    )
    assert temperature is not None
    if not sampling_metadata.all_random:
        is_greedy = temperature < _SAMPLING_EPS
        temp_eff = torch.where(is_greedy, torch.ones_like(temperature), temperature)
    else:
        is_greedy = torch.zeros(batch_size, dtype=torch.bool, device=device)
        temp_eff = temperature
    inv_temp = temp_eff.reciprocal().to(torch.float32)
    is_greedy_i32 = is_greedy.to(torch.int32)
    top_k = _expand_draft_sampling_tensor(sampling_metadata.top_k, batch_size)
    top_p = _expand_draft_sampling_tensor(sampling_metadata.top_p, batch_size)
    uses_top_k_top_p = top_k is not None or top_p is not None
    unit_inv_temp = torch.ones_like(inv_temp) if uses_top_k_top_p else None

    if tokens_out is not None and (
        tokens_out.shape[0] < batch_size
        or tokens_out.shape[1] < block_size
        or tokens_out.dtype is not torch.int64
        or tokens_out.device != device
    ):
        tokens_out = None
    tokens = (
        tokens_out[:batch_size, :block_size]
        if tokens_out is not None
        else torch.empty(
            (batch_size, block_size), dtype=torch.int64, device=device
        )
    )
    if draft_probs_out is not None and (
        draft_probs_out.shape[0] < batch_size
        or draft_probs_out.shape[1] < block_size
        or draft_probs_out.shape[2] != vocab_size
        or draft_probs_out.dtype is not torch.float32
        or draft_probs_out.device != device
    ):
        draft_probs_out = None
    draft_probs = (
        draft_probs_out[:batch_size, :block_size, :]
        if draft_probs_out is not None
        else torch.empty(
            (batch_size, block_size, vocab_size),
            dtype=torch.float32,
            device=device,
        )
    )
    num_blocks = (vocab_size + block_v - 1) // block_v
    scratch = {
        "block_max": torch.empty((batch_size, num_blocks), dtype=torch.float32, device=device),
        "block_sumexp": torch.empty((batch_size, num_blocks), dtype=torch.float32, device=device),
        "block_gval": torch.empty((batch_size, num_blocks), dtype=torch.float32, device=device),
        "block_maxid": torch.empty((batch_size, num_blocks), dtype=torch.int32, device=device),
        "block_gid": torch.empty((batch_size, num_blocks), dtype=torch.int32, device=device),
        "row_max": torch.empty((batch_size,), dtype=torch.float32, device=device),
        "row_invz": torch.empty((batch_size,), dtype=torch.float32, device=device),
    }

    prev_token_ids = first_prev_token_ids[:batch_size].long()
    for step_idx in range(block_size):
        step_logits = apply_markov_bias(
            base_logits[:, step_idx, :], prev_token_ids, step_idx
        )
        sample_logits = step_logits
        sample_inv_temp = inv_temp
        if uses_top_k_top_p:
            sample_logits = step_logits.to(torch.float32)
            if sample_logits.dtype == step_logits.dtype:
                sample_logits = sample_logits.clone()
            sample_logits.mul_(inv_temp.view(-1, 1))
            sample_logits = apply_top_k_top_p(sample_logits, top_k, top_p)
            assert unit_inv_temp is not None
            sample_inv_temp = unit_inv_temp
        dspark_markov_probs_sample(
            sample_logits,
            sample_inv_temp,
            is_greedy_i32,
            tokens[:, step_idx],
            draft_probs[:, step_idx, :],
            scratch,
            _next_fused_markov_seed(),
            block_v=block_v,
        )
        prev_token_ids = tokens[:, step_idx]

    return tokens, draft_probs

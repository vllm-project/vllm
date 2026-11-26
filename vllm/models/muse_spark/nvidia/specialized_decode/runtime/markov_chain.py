# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused Markov tails with the ordinary draft GEMMs and cache layout."""

from typing import Any

import torch

from ..cutedsl.draft.markov import MarkovKernels, anchor_embed
from .markov import fused_markov_params, unsupported


class MarkovChain:
    def __init__(self, model: Any, speculator: Any) -> None:
        cache = speculator.draft_logits.values
        self.params = fused_markov_params(model, cache)
        self.model = model
        self.steps = speculator.num_speculative_steps
        rows = speculator.max_num_reqs
        device = cache.device
        # Step-major so each step's rows are a contiguous `emb`: slot 0 the
        # anchor, slot `i + 1` the token step `i` samples.
        self.embeds = torch.empty(
            (self.steps + 1, rows, self.params.w1.shape[1]),
            device=device,
            dtype=torch.bfloat16,
        )
        self.anchor_idx = torch.arange(rows, device=device, dtype=torch.int64)
        blocks = (self.params.vocab_size + 511) // 512
        self.argmax = torch.empty(rows * blocks, device=device, dtype=torch.int64)
        self.maximum = torch.empty(rows * blocks, device=device, dtype=torch.float32)
        self.arrivals = torch.zeros(rows, device=device, dtype=torch.int32)
        self.packets = torch.empty((rows, 4), device=device, dtype=torch.int32)
        self.kernels = MarkovKernels(
            self.params.w1,
            self.embeds,
            width=self.params.vocab_size,
            has_cap=self.params.soft_cap is not None,
            tp=None,
            steps=tuple(
                (block, pdl, vector)
                for block, pdl in ((512, True), (1024, False), (1024, True))
                for vector in (False, True)
            ),
            pick_pdl=False,
            fill_vector=False,
        )

    def run(
        self,
        speculator,
        base_logits,
        prev,
        idx_map,
        sample_pos,
        out_tokens,
        *,
        prev_indices=None,
    ) -> torch.Tensor | None:
        """Sample the chain into `out_tokens`. Returns each step's Markov
        embedding, `[rows, steps, rank]` (a view), or None where the fused
        chain does not apply."""
        rows = base_logits.shape[0]
        cache = speculator.draft_logits
        if not (
            # A batch-sharded cache keeps the speculator's own loop.
            cache.row_mapping is None
            and base_logits.dtype == torch.bfloat16
            and base_logits.shape == (rows, self.steps, self.params.vocab_size)
            and base_logits.stride(-1) == 1
            and 0 < rows <= self.embeds.shape[1]
            and prev.dtype == torch.int32
            and prev.ndim == 1
            and prev.is_contiguous()
            and (
                (prev_indices is None and prev.shape == (rows,))
                or (
                    prev_indices is not None
                    and prev_indices.dtype == torch.int64
                    and prev_indices.shape == (rows,)
                    and prev_indices.is_contiguous()
                    and prev_indices.device == prev.device
                )
            )
            and out_tokens.dtype == torch.int64
            and out_tokens.shape == (rows, self.steps)
            and out_tokens.stride(-1) == 1
            and idx_map.dtype == sample_pos.dtype == torch.int64
            and idx_map.shape == sample_pos.shape == out_tokens.shape
        ):
            return None
        params = self.params
        embeds = self.embeds[:, :rows]
        anchor_embed(
            prev,
            self.anchor_idx[:rows] if prev_indices is None else prev_indices,
            params.w1,
            embeds[0],
            self.arrivals[:rows],
        )
        # Full-vocabulary w2 keeps the ordinary GEMM; its overlap differs
        # from the specialized path's vocabulary-sharded projection.
        block = 512 if rows <= 2 else 1024
        pdl = rows <= 2 or rows >= 8
        for step in range(self.steps):
            bias = self.model.markov_bias(embeds[step])
            # The builder accepts only the stock BF16 Markov projection.
            assert bias.dtype == torch.bfloat16
            self.kernels.step(
                base_logits[:, step],
                bias,
                cache.values,
                step,
                idx_map[:, step],
                sample_pos[:, step],
                speculator.seeds,
                speculator.temperature,
                self.argmax,
                self.maximum,
                self.arrivals[:rows],
                out_tokens,
                embeds[step + 1],
                self.packets,
                0,
                params.output_multiplier,
                params.inv_soft_cap,
                1.0 if params.soft_cap is None else params.soft_cap,
                block=block,
                pdl=pdl,
            )
        return embeds[: self.steps].transpose(0, 1)


def build_markov_chain(model, speculator) -> MarkovChain | None:
    if (
        speculator.use_acceptance_estimator
        or getattr(model, "has_recurrent_markov_head", False)
        or speculator.draft_logits is None
        or model.model.markov_head.markov_w2.weight.dtype != torch.bfloat16
    ):
        return None
    reason = unsupported(
        model,
        draft_logits=speculator.draft_logits.values,
        draft_topk=speculator._draft_topk,
        watermarked=speculator.draft_watermarker is not None,
        use_fp64_gumbel=speculator.use_fp64_gumbel,
        sample_from_anchor=speculator.sample_from_anchor,
    )
    if reason is not None:
        return None
    return MarkovChain(model, speculator)

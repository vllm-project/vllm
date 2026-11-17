# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.platforms import current_platform


@pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="the Muse fused Markov path is selected on SM100-family devices",
)
@pytest.mark.parametrize("rows", [1, 3, 48])
@torch.inference_mode()
def test_fused_markov_chain_feeds_confidence_head_its_sampled_tokens(rows):
    """Adaptive verification keeps the fused chain: each step's confidence
    reads the Markov embedding of the token that step actually sampled."""
    from types import SimpleNamespace

    from vllm.models.muse_spark.nvidia.specialized_decode.runtime.markov_chain import (  # noqa: E501
        build_markov_chain,
    )
    from vllm.v1.worker.gpu.spec_decode.draft_logits import DraftLogitsCache
    from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

    torch.manual_seed(185)
    steps, vocab, rank, hidden_size = 7, 2051, 512, 64
    with torch.device("cuda"):
        w1 = torch.randn(vocab, rank, dtype=torch.bfloat16)
        w2 = torch.randn(vocab, rank, dtype=torch.bfloat16) / rank**0.5
        conf_w = torch.randn(hidden_size + rank)

        def compute_confidence(hidden, markov_embed):
            x = torch.cat([hidden.float(), markov_embed.float()], dim=-1)
            return torch.sigmoid(x @ conf_w)

        draft = SimpleNamespace(
            model=SimpleNamespace(
                markov_head=SimpleNamespace(
                    markov_w1=SimpleNamespace(weight=w1),
                    markov_w2=SimpleNamespace(weight=w2),
                )
            ),
            config=SimpleNamespace(draft_vocab_size=vocab),
            logits_processor=SimpleNamespace(head_dtype=None),
            draft_id_to_target_id=None,
            output_logit_transform_params=lambda: (0.5, 30.0),
            markov_bias=lambda emb: emb @ w2.T,
            compute_confidence=compute_confidence,
        )
        spec = DSparkSpeculator.__new__(DSparkSpeculator)
        spec.model = draft
        spec.num_speculative_steps = steps
        spec.max_num_reqs = rows
        spec.use_confidence_head = True
        spec.use_acceptance_estimator = False
        spec._draft_topk = None
        spec.draft_watermarker = None
        spec.use_fp64_gumbel = False
        spec.sample_from_anchor = True
        spec.draft_logits = DraftLogitsCache(
            rows, steps, vocab, dtype=torch.bfloat16, fill=0.0, device="cuda"
        )
        spec.seeds = torch.arange(rows, dtype=torch.int64)
        spec.temperature = torch.tensor([0.0, 1.0], dtype=torch.float32).repeat(rows)[
            :rows
        ]
        chain = build_markov_chain(draft, spec)
        assert chain is not None
        draft.sample_markov_chain = lambda s, *args, **kwargs: chain.run(
            s, *args, **kwargs
        )

        input_ids = torch.randint(vocab, (rows * 3,), dtype=torch.int32)
        anchors = torch.arange(rows, dtype=torch.int64) * 3 + 1
        base_logits = torch.randn(rows, steps, vocab, dtype=torch.bfloat16)
        hidden = torch.randn(rows * steps, hidden_size, dtype=torch.bfloat16)
        idx_map = torch.arange(rows).unsqueeze(1).repeat(1, steps)
        sample_pos = 100 + torch.arange(steps).expand(rows, -1).contiguous()
        tokens = torch.full((rows, steps), -1, dtype=torch.int64)

        confidence = spec._sample_markov_chain(
            base_logits,
            hidden,
            input_ids,
            idx_map,
            sample_pos,
            tokens,
            prev_indices=anchors,
        )

        assert confidence is not None and confidence.shape == (rows, steps)
        assert torch.all((tokens >= 0) & (tokens < vocab))
        prev = torch.cat([input_ids[anchors].long().unsqueeze(1), tokens], dim=1)
        expected = compute_confidence(hidden, w1[prev[:, :steps]].flatten(0, 1)).view(
            rows, steps
        )
        torch.testing.assert_close(confidence, expected, rtol=0, atol=0)

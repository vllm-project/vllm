# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU: a segment-layout read (one forward for a shared prefix and several
candidates) matches a reference forward with the same isolating mask and
restarted positions."""

import pytest
import torch

from vllm import PoolingParams
from vllm.config import PoolerConfig
from vllm.pooling_params import SegmentLayoutParams

MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _reference_means(hf_model, prefix: list[int], candidates: list[list[int]]):
    """HF forward with the segment mask and positions, then the mean of each
    candidate's token states."""
    ids = prefix + [t for c in candidates for t in c]
    lengths = [len(prefix)] + [len(c) for c in candidates]
    segment = torch.cat(
        [torch.full((n,), k, dtype=torch.long) for k, n in enumerate(lengths)]
    )
    mask = (segment.view(1, -1) == 0) | (segment.view(-1, 1) == segment.view(1, -1))
    positions = torch.cat(
        [torch.arange(len(prefix))]
        + [torch.arange(len(prefix), len(prefix) + len(c)) for c in candidates]
    )
    device = next(hf_model.parameters()).device
    with torch.no_grad():
        out = (
            hf_model(
                input_ids=torch.tensor([ids], device=device),
                # 4D additive mask: used as is by the attention layers.
                attention_mask=torch.zeros(
                    1, 1, len(ids), len(ids), device=device
                ).masked_fill(~mask.to(device), torch.finfo(torch.float32).min),
                position_ids=positions.unsqueeze(0).to(device),
                token_type_ids=torch.zeros(
                    1, len(ids), dtype=torch.long, device=device
                ),
            )
            .last_hidden_state[0]
            .float()
        )
    rows = out[len(prefix) :].split(lengths[1:])
    return torch.stack([r.mean(dim=0) for r in rows]).cpu()


@pytest.mark.parametrize("num_candidates", [1, 4, 16])
def test_segment_layout_matches_masked_reference(vllm_runner, num_candidates):
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    prefix = tokenizer("which city is the capital of france?")["input_ids"]
    candidates = [
        tokenizer(f"document {k}: paris has {k} famous bridges.")["input_ids"][1:]
        for k in range(num_candidates)
    ]
    hf_model = (
        AutoModel.from_pretrained(MODEL, attn_implementation="eager")
        .to(torch.float32)
        .cuda()
        .eval()
    )
    expected = _reference_means(hf_model, prefix, candidates)

    ids = prefix + [t for c in candidates for t in c]
    layout = SegmentLayoutParams([len(prefix)] + [len(c) for c in candidates])
    with vllm_runner(
        MODEL,
        runner="pooling",
        pooler_config=PoolerConfig(task="token_embed"),
        dtype="float32",
        attention_backend="FLEX_ATTENTION",
        enforce_eager=True,
        max_model_len=256,
    ) as vllm_model:
        outputs = vllm_model.llm.encode(
            [{"prompt_token_ids": ids}],
            pooling_params=PoolingParams(
                use_activation=False, segment_layout_params=layout
            ),
            pooling_task="token_embed",
            use_tqdm=False,
        )
    got = outputs[0].outputs.data.float().cpu()
    assert got.shape == expected.shape
    torch.testing.assert_close(got, expected, atol=2e-3, rtol=2e-3)

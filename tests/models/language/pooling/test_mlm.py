# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from transformers import AutoModelForMaskedLM

from tests.models.utils import softmax
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed


@pytest.fixture(autouse=True)
def seed_everything():
    """Seed all random number generators for reproducibility."""
    seed = 0
    set_random_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    yield


@pytest.mark.parametrize(
    "model",
    [
        # Original Google checkpoint: legacy `gamma`/`beta` LayerNorm names, an
        # NSP head (`cls.seq_relationship.*`) and a decoder tied to the input
        # embeddings (no explicit decoder weight in the checkpoint).
        "google-bert/bert-base-uncased",
    ],
)
@pytest.mark.parametrize("dtype", ["float"])
@torch.inference_mode
def test_bert_for_masked_lm(
    hf_runner,
    vllm_runner,
    example_prompts,
    model: str,
    dtype: str,
) -> None:
    # vLLM exposes the masked-LM head as a token-level pooling task; the head
    # applies softmax over the vocabulary, so each output row is a distribution.
    with vllm_runner(model, max_model_len=None, dtype=dtype) as vllm_model:
        vllm_outputs = vllm_model.token_classify(example_prompts)

    # Use eager attention on ROCm to avoid HF Transformers flash attention
    # accuracy issues: https://github.com/vllm-project/vllm/issues/30167
    hf_model_kwargs = {}
    if current_platform.is_rocm():
        hf_model_kwargs["attn_implementation"] = "eager"

    with hf_runner(
        model,
        dtype=dtype,
        auto_cls=AutoModelForMaskedLM,
        model_kwargs=hf_model_kwargs,
    ) as hf_model:
        tokenizer = hf_model.tokenizer
        hf_outputs = []
        for prompt in example_prompts:
            inputs = tokenizer([prompt], return_tensors="pt")
            inputs = hf_model.wrap_device(inputs)
            output = hf_model.model(**inputs)
            hf_outputs.append(softmax(output.logits[0]))

    # Compare the per-token vocabulary distributions position by position.
    for hf_output, vllm_output in zip(hf_outputs, vllm_outputs):
        hf_output = hf_output.detach().clone().cpu().float()
        vllm_output = vllm_output.detach().clone().cpu().float()
        assert hf_output.shape == vllm_output.shape
        assert torch.equal(hf_output.argmax(dim=-1), vllm_output.argmax(dim=-1))
        torch.testing.assert_close(hf_output, vllm_output, atol=3.2e-2, rtol=1e-3)

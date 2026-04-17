# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest
import torch

MODEL = "Qwen/Qwen3-ForcedAligner-0.6B"
CLASSIFY_NUM = 5000
TIMESTAMP_TOKEN_ID = 151705


def build_prompt(words: list[str]) -> str:
    body = "<timestamp><timestamp>".join(words) + "<timestamp><timestamp>"
    return f"<|audio_start|><|audio_pad|><|audio_end|>{body}"


@pytest.mark.parametrize("model", [MODEL])
@pytest.mark.parametrize("dtype", ["bfloat16"])
@torch.inference_mode()
def test_qwen3_forced_aligner(
    vllm_runner,
    model: str,
    dtype: str,
) -> None:
    words = ["Hello", "world"]
    prompt = build_prompt(words)

    # 5-second silent audio at 16kHz
    audio = np.zeros(16000 * 5, dtype=np.float32)

    with vllm_runner(
        model,
        runner="pooling",
        dtype=dtype,
        enforce_eager=True,
        max_model_len=512,
        hf_overrides={
            "architectures": [
                "Qwen3ASRForcedAlignerForTokenClassification",
            ],
        },
    ) as vllm_model:
        outputs = vllm_model.llm.encode(
            [{"prompt": prompt, "multi_modal_data": {"audio": audio}}],
            pooling_task="token_classify",
        )

    # Validate output structure
    assert len(outputs) == 1
    logits = outputs[0].outputs.data
    assert logits.dim() == 2
    assert logits.shape[1] == CLASSIFY_NUM

    # Validate timestamp extraction
    token_ids = outputs[0].prompt_token_ids
    predictions = logits.argmax(dim=-1)
    ts_indices = [i for i, t in enumerate(token_ids) if t == TIMESTAMP_TOKEN_ID]

    # 2 words x 2 timestamps each (start + end) = 4
    assert len(ts_indices) == 4

    ts_preds = [predictions[i].item() for i in ts_indices]
    assert all(p >= 0 for p in ts_preds)
    # end >= start for each word
    assert ts_preds[1] >= ts_preds[0]  # Hello
    assert ts_preds[3] >= ts_preds[2]  # world

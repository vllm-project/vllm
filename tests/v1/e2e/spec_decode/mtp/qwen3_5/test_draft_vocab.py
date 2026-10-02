# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json

import pytest

from tests.evals.gsm8k.gsm8k_eval import evaluate_gsm8k_offline
from tests.utils import single_gpu_only
from vllm.config import CompilationConfig

from ...utils import compute_acceptance_len

MODEL = "Qwen/Qwen3.5-0.8B-Base"
# Low ids of a BPE vocabulary are roughly the most frequent tokens.
NUM_DRAFT_TOKENS = 32768


@single_gpu_only
def test_mtp_draft_token_map(monkeypatch: pytest.MonkeyPatch, tmp_path, vllm_runner):
    """MTP with a draft token map keeps GSM8K above the stock-MTP reference
    threshold, and its acceptance length within 10% of stock MTP."""
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    token_map = tmp_path / "draft_vocab.json"
    token_map.write_text(json.dumps(list(range(NUM_DRAFT_TOKENS))))

    acceptance_lens = []
    for draft_token_map in (None, str(token_map)):
        with vllm_runner(
            MODEL,
            max_model_len=2048,
            block_size=None,
            enable_chunked_prefill=None,
            compilation_config=CompilationConfig(),
            limit_mm_per_prompt={"image": 0, "video": 0},
            disable_log_stats=False,
            speculative_config={
                "method": "mtp",
                "num_speculative_tokens": 1,
                "draft_token_map": draft_token_map,
            },
        ) as runner:
            accuracy = evaluate_gsm8k_offline(runner.llm, num_questions=200)["accuracy"]
            acceptance_lens.append(compute_acceptance_len(runner.llm.get_metrics()))
        # Stock GSM8K for this model is about 34% (see test_mtp.py).
        assert accuracy >= 0.20, f"{draft_token_map=}: GSM8K {accuracy:.3f}"

    stock, reduced = acceptance_lens
    print(f"Acceptance length: stock {stock:.3f}, draft_token_map {reduced:.3f}")
    assert reduced > 1.0
    assert reduced >= 0.9 * stock

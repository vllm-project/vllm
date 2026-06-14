# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Optional end-to-end Wikitext PPL test for XiaomiMiMo/MiMo-V2.5-Pro FP8 loading.
"""

import lm_eval
import pytest

from tests.utils import large_gpu_mark, multi_gpu_marks
from vllm.platforms import current_platform

MODEL_NAME = "XiaomiMiMo/MiMo-V2.5-Pro"
TASK = "wikitext"
EXPECTED_WORD_PPL = 4.0
WORD_PPL_RTOL = 0.1


@pytest.mark.optional
@pytest.mark.skipif(
    not current_platform.supports_fp8(),
    reason="MiMo-V2.5-Pro FP8 loading requires FP8-capable hardware.",
)
@pytest.mark.parametrize(
    "tp_size",
    [
        pytest.param(
            4,
            marks=[*multi_gpu_marks(num_gpus=4), large_gpu_mark(min_gb=256)],
            id="tp4",
        ),
        pytest.param(
            8,
            marks=[*multi_gpu_marks(num_gpus=8), large_gpu_mark(min_gb=128)],
            id="tp8",
        ),
    ],
)
def test_mimo_v25_pro_fp8_wikitext_ppl(
    tp_size: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    model_args = {
        "pretrained": MODEL_NAME,
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "quantization": "fp8",
        "kv_cache_dtype": "bfloat16",
        "max_model_len": 1024,
        "max_num_seqs": 1,
        "tensor_parallel_size": tp_size,
        "attention_backend": "TRITON_ATTN_DIFFKV",
        "enforce_eager": True,
        "disable_custom_all_reduce": True,
    }
    results = lm_eval.simple_evaluate(
        model="vllm",
        model_args=model_args,
        tasks=TASK,
        batch_size=1,
    )
    word_ppl = results["results"][TASK]["word_perplexity,none"]

    print(f"MiMo-V2.5-Pro FP8 tp={tp_size} Wikitext word PPL: {word_ppl}")
    assert word_ppl == pytest.approx(EXPECTED_WORD_PPL, rel=WORD_PPL_RTOL)

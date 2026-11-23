# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real-weight TP8 decode against stock, including the MTP draft."""

import pytest

from vllm.platforms import current_platform

from ..utils import check_logprobs_close


def _has_specialized_decode(model):
    return model.language_model.specialized_decode is not None


@pytest.mark.distributed(num_gpus=8)
@pytest.mark.skipif(
    not current_platform.is_device_capability_family(100), reason="Requires SM100"
)
@pytest.mark.parametrize("mtp", [False, True])
def test_engine_tp8(vllm_runner, enable_pickle, monkeypatch, mtp, example_prompts):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    outputs = []
    for backend in ("stock", "latency_specialized"):
        with vllm_runner(
            "zai-org/GLM-5.3-Flash",
            tensor_parallel_size=8,
            dtype="bfloat16",
            trust_remote_code=True,
            max_model_len=256,
            max_num_seqs=8,
            kv_cache_dtype="fp8",
            limit_mm_per_prompt={"image": 0, "video": 0},
            compilation_config={
                "cudagraph_mode": "FULL_DECODE_ONLY",
                "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32],
            },
            speculative_config=(
                {"method": "mtp", "num_speculative_tokens": 3} if mtp else None
            ),
            additional_config={"decode_backend": backend},
        ) as runner:
            assert all(
                enabled == (backend != "stock")
                for enabled in runner.apply_model(_has_specialized_decode)
            )
            outputs.append(
                [
                    output
                    for batch in (1, 2, 8)
                    for output in runner.generate_greedy_logprobs(
                        example_prompts[:batch], max_tokens=16, num_logprobs=5
                    )
                ]
            )
    check_logprobs_close(
        outputs_0_lst=outputs[0],
        outputs_1_lst=outputs[1],
        name_0="stock TP8",
        name_1="specialized TP8",
    )

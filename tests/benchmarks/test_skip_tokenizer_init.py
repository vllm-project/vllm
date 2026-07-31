# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression test for --skip-tokenizer-init with --dataset-name custom.

Before the fix (introduced by #39896), running:

    vllm bench serve \
        --backend vllm-pooling \
        --dataset-name custom \
        --dataset-path <path> \
        --model ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11 \
        --endpoint /pooling \
        --skip-tokenizer-init \
        ...

raised immediately with:

    AssertionError: Tokenizer must be initialized before loading dataset

even though CustomDataset.sample() already handles tokenizer=None.
This test exercises main_async() directly so it catches any regression
re-introduced at the serve.py level, not just inside get_samples().
"""

import argparse
import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

import vllm.benchmarks.serve as serve_module

# Exact prompt payload from the failing benchmark run against a
# Prithvi-EO-2.0 pooling endpoint (URL-in / base64-out format).
_PRITHVI_PROMPT = {
    "data": {
        "data": "https://huggingface.co/christian-pinto/Prithvi-EO-2.0-300M-TL-VLLM/resolve/main/India_900498_S2Hand.tif",
        "data_format": "url",
        "out_data_format": "b64_json",
        "indices": [1, 2, 3, 8, 11, 12],
    },
    "priority": 0,
    "softmax": False,
}


def _write_dataset(path: Path) -> None:
    path.write_text(json.dumps({"prompt": _PRITHVI_PROMPT}) + "\n")


def _args(dataset_path: str) -> argparse.Namespace:
    """Reproduce the argparse.Namespace that serve.py builds from the
    failing command, including skip_tokenizer_init=True."""
    return argparse.Namespace(
        # dataset
        dataset_name="custom",
        dataset_path=dataset_path,
        disable_shuffle=False,
        num_prompts=1,
        custom_output_len=256,
        skip_chat_template=True,
        chat_template_kwargs=None,
        no_oversample=False,
        seed=0,
        request_id_prefix="bench-",
        # model / tokenizer
        model="ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11",
        served_model_name=None,
        tokenizer=None,
        tokenizer_mode="auto",
        trust_remote_code=False,
        skip_tokenizer_init=True,  # <-- the flag under test
        # backend / endpoint
        backend="vllm-pooling",
        base_url="http://127.0.0.1:8000",
        host="127.0.0.1",
        port=8000,
        endpoint="/pooling",
        header=None,
        insecure=False,
        # traffic
        request_rate=16.0,
        burstiness=1.0,
        max_concurrency=None,
        # misc serve args that main_async reads before reaching get_samples
        plot_timeline=False,
        plot_dataset_stats=False,
        self_timed=None,
        metadata=None,
        label=None,
        logprobs=None,
        use_beam_search=False,
        ignore_eos=False,
        goodput=None,
        percentile_metrics="ttft,tpot,itl,e2el",
        metric_percentiles="25,50,75,99",
        save_result=False,
        append_result=False,
        result_dir=".",
        result_filename=None,
        num_warmups=0,
        profile=False,
        disable_tqdm=True,
        lora_modules=None,
        lora_assignment="random",
        ramp_up_strategy=None,
        ramp_up_start_rps=None,
        ramp_up_end_rps=None,
        ready_check_timeout_sec=0,
        extra_body=None,
        top_p=None,
        top_k=None,
        min_p=None,
        temperature=None,
        frequency_penalty=None,
        presence_penalty=None,
        repetition_penalty=None,
        save_detailed=False,
        input_len=None,
        output_len=None,
    )


@pytest.mark.benchmark
def test_main_async_skip_tokenizer_init_does_not_raise(tmp_path: Path) -> None:
    """main_async must not raise AssertionError when skip_tokenizer_init=True
    and dataset_name='custom'.

    On main (before the fix) this test fails with:
        AssertionError: Tokenizer must be initialized before loading dataset
    """
    dataset_path = tmp_path / "dataset_url_input_india.jsonl"
    _write_dataset(dataset_path)

    args = _args(str(dataset_path))

    # Patch benchmark() so we never make real HTTP requests — the regression
    # triggers before benchmark() is ever called, so this just keeps the test
    # fast and hermetic.
    mock_result = {
        "completed": 1,
        "failed": 0,
        "total_input_tokens": 1,
        "total_output_tokens": 1,
        "request_throughput": 1.0,
        "output_throughput": 1.0,
        "total_token_throughput": 1.0,
        "input_lens": [],
        "output_lens": [],
        "ttfts": [],
        "itls": [],
        "generated_texts": [],
        "errors": [],
        "duration": 1.0,
    }
    with patch.object(
        serve_module, "benchmark", new=AsyncMock(return_value=mock_result)
    ):
        # Must NOT raise AssertionError
        asyncio.run(serve_module.main_async(args))

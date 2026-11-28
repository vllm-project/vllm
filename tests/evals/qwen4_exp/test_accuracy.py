# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GSM8K and AIME25 accuracy evaluation for Qwen3.8-Flash-Next-FP8."""

import asyncio
import math
import shlex
from pathlib import Path
from typing import Any

import yaml
from transformers import AutoTokenizer

from tests.utils import RemoteOpenAIServer, multi_gpu_test


def run_evalscope(
    eval_config: dict[str, Any], base_url: str, work_dir: Path
) -> dict[str, float]:
    from evalscope.run import run_task

    reports = run_task(
        {
            "model": eval_config["model_name"],
            "api_url": base_url,
            "api_key": "EMPTY_TOKEN",
            "datasets": list(eval_config["datasets"]),
            "eval_batch_size": eval_config.get("eval_batch_size", 32),
            "generation_config": eval_config["generation_config"],
            "work_dir": str(work_dir),
            "no_timestamp": True,
        }
    )
    if not isinstance(reports, dict):
        raise TypeError(f"Expected EvalScope reports to be a dict, got {type(reports)}")

    return {dataset: float(report.score) for dataset, report in reports.items()}


def test_qwen4_exp_accuracy(config_filename: Path, tmp_path: Path):
    eval_config = yaml.safe_load(config_filename.read_text(encoding="utf-8"))
    server_args = shlex.split(eval_config.get("server_args", ""))
    server_args.append("--disable-uvicorn-access-log")

    model_name = eval_config["model_name"]
    print(f"Starting Qwen4Exp evaluation for model: {model_name}")
    print(f"Datasets: {', '.join(eval_config['datasets'])}")
    print(f"Server args: {' '.join(server_args)}")

    with RemoteOpenAIServer(
        model_name,
        server_args,
        env_dict=eval_config.get("env"),
        max_wait_seconds=eval_config.get("startup_max_wait_seconds", 1800),
    ) as remote_server:
        scores = run_evalscope(eval_config, remote_server.url_for("v1"), tmp_path)

    for dataset, metric_config in eval_config["datasets"].items():
        score = scores[dataset]
        threshold = metric_config["metric_threshold"]
        tolerance = metric_config.get("tolerance", 0.0)
        minimum_score = threshold - tolerance

        print(
            f"{dataset}: measured={score:.4f}, expected={threshold:.4f}, "
            f"tolerance={tolerance:.4f}"
        )
        assert score >= minimum_score, (
            f"{dataset} score too low: {score:.4f} < {threshold:.4f} - "
            f"{tolerance:.4f} = {minimum_score:.4f}"
        )


@multi_gpu_test(num_gpus=4)
def test_qwen4_pcp_preserves_recurrent_state_for_concurrent_requests():
    """PCP segments must not become separate GDN requests or read the null slot."""
    model_name = "Qwen/Qwen3.8-Flash-Next"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    questions = [
        ("Calculate 17 * 23. Respond with only the integer.", "391"),
        ("Calculate 124 + 578. Respond with only the integer.", "702"),
        ("Sort 9, 2, 7 ascending. Output only 2, 7, 9.", "2, 7, 9"),
        ("What is the capital of France? Answer with one word.", "Paris"),
    ]
    server_args = shlex.split(
        "--tensor-parallel-size 1 --prefill-context-parallel-size 4 "
        "--enable-expert-parallel --quantization fp8 --dtype bfloat16 "
        "--kv-cache-dtype fp8 --max-model-len 8192 "
        "--max-num-batched-tokens 4096 --max-num-seqs 4 "
        "--enable-prefix-caching --no-enable-extensible-kv-cache "
        "--enforce-eager --language-model-only"
    )
    server_args += [
        "--engram-config",
        '{"cpu_offload":true}',
        "--attention-config",
        '{"indexer_kv_dtype":"fp8"}',
    ]
    with RemoteOpenAIServer(model_name, server_args, max_wait_seconds=1800) as server:
        client = server.get_async_client()

        async def complete(question, expected):
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": question}],
                tokenize=True,
                add_generation_prompt=True,
            )
            if hasattr(prompt, "keys"):
                prompt = prompt["input_ids"]
            response = await client.completions.create(
                model=model_name,
                prompt=prompt,
                max_tokens=512,
                temperature=0,
                logprobs=1,
                extra_body={"add_special_tokens": False},
            )
            choice = response.choices[0]
            final_answer = choice.text.rsplit("</think>", 1)[-1].strip()
            assert final_answer == expected
            assert choice.logprobs is not None
            assert choice.logprobs.token_logprobs
            assert all(math.isfinite(value) for value in choice.logprobs.token_logprobs)

        async def run():
            await asyncio.gather(*(complete(*question) for question in questions))
            await asyncio.gather(
                *(complete(*question) for question in reversed(questions))
            )

        asyncio.run(run())

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import asyncio
import json
from collections.abc import Callable
from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import AutoTokenizer, PreTrainedTokenizerBase, TokenizersBackend

from vllm.benchmarks.datasets import CustomDataset, add_dataset_parser, get_samples
from vllm.benchmarks.serve import add_cli_args, main_async
from vllm.utils.argparse_utils import FlexibleArgumentParser


@pytest.fixture(scope="session")
def hf_tokenizer() -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained("openai-community/gpt2")


def _write_jsonl(path: Path, n_rows: int) -> None:
    with path.open("w") as f:
        for i in range(n_rows):
            f.write(json.dumps({"prompt": f"row {i}: unique prompt content."}) + "\n")


def _args_for_custom(dataset_path: str, seed: int) -> argparse.Namespace:
    return argparse.Namespace(
        dataset_name="custom",
        dataset_path=dataset_path,
        disable_shuffle=False,
        num_prompts=30,
        custom_output_len=32,
        skip_chat_template=True,
        no_oversample=False,
        seed=seed,
        request_id_prefix="",
    )


@pytest.mark.benchmark
def test_custom_dataset_seed_propagates(
    hf_tokenizer: PreTrainedTokenizerBase, tmp_path: Path
) -> None:
    """--seed must control the CustomDataset shuffle used by get_samples.

    Without the fix, CustomDataset was instantiated without random_seed,
    so its load-time shuffle always used DEFAULT_SEED=0 regardless of
    args.seed, causing every run with --dataset-name custom to pick the
    same subset of rows from a larger file.
    """
    jsonl = tmp_path / "data.jsonl"
    _write_jsonl(jsonl, n_rows=60)

    samples_a = get_samples(_args_for_custom(str(jsonl), seed=0), hf_tokenizer)
    samples_b = get_samples(_args_for_custom(str(jsonl), seed=42), hf_tokenizer)

    prompts_a = {s.prompt for s in samples_a}
    prompts_b = {s.prompt for s in samples_b}

    assert len(prompts_a) == 30
    assert len(prompts_b) == 30
    assert prompts_a != prompts_b


@pytest.mark.benchmark
def test_custom_dataset_same_seed_is_deterministic(
    hf_tokenizer: PreTrainedTokenizerBase, tmp_path: Path
) -> None:
    """Same --seed must yield the same CustomDataset subset."""
    jsonl = tmp_path / "data.jsonl"
    _write_jsonl(jsonl, n_rows=60)

    samples_a = get_samples(_args_for_custom(str(jsonl), seed=7), hf_tokenizer)
    samples_b = get_samples(_args_for_custom(str(jsonl), seed=7), hf_tokenizer)

    prompts_a = [s.prompt for s in samples_a]
    prompts_b = [s.prompt for s in samples_b]

    assert prompts_a == prompts_b


def _write_spec_bench(tmp_path: Path, n_rows: int) -> str:
    path = tmp_path / "question.jsonl"
    with path.open("w") as f:
        for i in range(n_rows):
            row = {"category": "writing", "turns": [f"row {i}: unique prompt."]}
            f.write(json.dumps(row) + "\n")
    return str(path)


def _write_speed_bench(tmp_path: Path, n_rows: int) -> str:
    with (tmp_path / "qualitative.jsonl").open("w") as f:
        for i in range(n_rows):
            message = {"role": "user", "content": f"row {i}: unique prompt."}
            f.write(json.dumps({"category": "writing", "messages": [message]}) + "\n")
    return str(tmp_path)


_BENCH_DATASETS = [
    ("spec_bench", _write_spec_bench),
    ("speed_bench", _write_speed_bench),
]


@pytest.mark.benchmark
@pytest.mark.parametrize(("dataset_name", "write_dataset"), _BENCH_DATASETS)
def test_bench_dataset_seed_propagates(
    hf_tokenizer: PreTrainedTokenizerBase,
    tmp_path: Path,
    dataset_name: str,
    write_dataset: Callable[[Path, int], str],
) -> None:
    """--seed must control the load-time shuffle of spec_bench and speed_bench."""
    dataset_path = write_dataset(tmp_path, 60)
    parser = FlexibleArgumentParser()
    add_dataset_parser(parser)

    def args_with_seed(seed: int) -> argparse.Namespace:
        return parser.parse_args(
            ["--dataset-name", dataset_name, "--dataset-path", dataset_path]
            + ["--num-prompts", "30", "--seed", str(seed), "--skip-chat-template"]
        )

    prompts_a = {s.prompt for s in get_samples(args_with_seed(0), hf_tokenizer)}
    prompts_b = {s.prompt for s in get_samples(args_with_seed(42), hf_tokenizer)}

    assert len(prompts_a) == len(prompts_b) == 30
    assert prompts_a != prompts_b


async def _prompts_sent(argv: list[str]) -> set[str]:
    """Runs `vllm bench serve` against a stub server; returns the prompts it got."""
    prompts: set[str] = set()

    async def completions(request: web.Request) -> web.StreamResponse:
        prompts.add((await request.json())["prompt"])
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        for chunk in (
            {"choices": [{"text": "x"}]},
            {"usage": {"completion_tokens": 1}},
        ):
            await response.write(f"data: {json.dumps(chunk)}\n\n".encode())
        await response.write(b"data: [DONE]\n\n")
        return response

    app = web.Application()
    app.router.add_post("/v1/completions", completions)
    parser = FlexibleArgumentParser()
    add_cli_args(parser)
    async with TestServer(app) as server:
        base_url = str(server.make_url("")).rstrip("/")
        await main_async(parser.parse_args([*argv, "--base-url", base_url]))
    return prompts


@pytest.mark.benchmark
@pytest.mark.parametrize(("dataset_name", "write_dataset"), _BENCH_DATASETS)
def test_bench_serve_seed_changes_prompts_sent(
    tmp_path: Path, dataset_name: str, write_dataset: Callable[[Path, int], str]
) -> None:
    """`vllm bench serve --seed` must change which prompts reach the server."""
    dataset_path = write_dataset(tmp_path, 60)
    argv = ["--model", "stub", "--tokenizer", "openai-community/gpt2"]
    argv += ["--dataset-name", dataset_name, "--dataset-path", dataset_path]
    argv += ["--num-prompts", "30", "--skip-chat-template"]

    prompts_a = asyncio.run(_prompts_sent([*argv, "--seed", "0"]))
    prompts_b = asyncio.run(_prompts_sent([*argv, "--seed", "42"]))

    assert len(prompts_a) == len(prompts_b) == 30
    assert prompts_a != prompts_b


@pytest.mark.benchmark
def test_custom_dataset_keeps_numeric_string_prompts(tmp_path: Path) -> None:
    """All-numeric string prompts must not be inferred as numbers by pandas."""
    prompts = ["00123", "00456"]
    jsonl = tmp_path / "data.jsonl"
    jsonl.write_text(
        "".join(json.dumps({"prompt": p, "output_tokens": 4}) + "\n" for p in prompts)
    )
    tokenizer = TokenizersBackend(
        tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]")),
        unk_token="[UNK]",
    )

    dataset = CustomDataset(dataset_path=str(jsonl), disable_shuffle=True)
    requests = dataset.sample(tokenizer, num_requests=2, skip_chat_template=True)

    assert [r.prompt for r in requests] == prompts

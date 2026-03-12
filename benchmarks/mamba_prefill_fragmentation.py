# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import dataclasses
import random
import time

from transformers import PreTrainedTokenizerBase

from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser

try:
    from vllm.tokenizers import get_tokenizer
except ImportError:
    from backend_request_func import get_tokenizer


def _sample_tokens(tokenizer: PreTrainedTokenizerBase, length: int) -> list[int]:
    vocab = tokenizer.get_vocab()
    all_special_ids = set(tokenizer.all_special_ids)
    return random.choices(
        [v for v in vocab.values() if v not in all_special_ids],
        k=length,
    )


def _get_scheduler(llm: LLM):
    engine_core_client = getattr(llm.llm_engine, "engine_core", None)
    if engine_core_client is None:
        return None
    engine_core = getattr(engine_core_client, "engine_core", None)
    if engine_core is None:
        return None
    return getattr(engine_core, "scheduler", None)


def _run_once(llm: LLM, prompt_token_ids: list[int], max_tokens: int):
    sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0.0)
    start = time.perf_counter()
    outputs = llm.generate([prompt_token_ids], sampling_params, use_tqdm=False)
    end = time.perf_counter()
    output = outputs[0]
    metrics = output.metrics
    ttft = None
    if metrics is not None and metrics.first_token_latency:
        ttft = metrics.first_token_latency
    output_tokens = len(output.outputs[0].token_ids)
    return ttft, end - start, output_tokens


def main(args):
    random.seed(args.seed)
    tokenizer = get_tokenizer(args.model, trust_remote_code=True)
    prompt_token_ids = _sample_tokens(tokenizer, args.prompt_length)

    engine_args = EngineArgs.from_cli_args(args)
    llm = LLM(**dataclasses.asdict(engine_args))

    scheduler = _get_scheduler(llm)
    if scheduler is None:
        print("Warning: scheduler not available; fragmentation counters disabled.")

    if args.warmup_steps > 0:
        for _ in range(args.warmup_steps):
            _run_once(llm, prompt_token_ids, args.max_tokens)

    ttft_list: list[float] = []
    latency_list: list[float] = []
    tok_per_s_list: list[float] = []
    frag_deltas: list[int] = []
    zero_deltas: list[int] = []
    round_deltas: list[int] = []

    for _ in range(args.repeat_count):
        frag_before = getattr(scheduler, "mamba_fragmentation_count", 0)
        zero_before = getattr(scheduler, "mamba_zero_collapse_count", 0)
        round_before = getattr(scheduler, "_scheduler_iteration", 0)

        ttft, latency, output_tokens = _run_once(llm, prompt_token_ids, args.max_tokens)

        frag_after = getattr(scheduler, "mamba_fragmentation_count", 0)
        zero_after = getattr(scheduler, "mamba_zero_collapse_count", 0)
        round_after = getattr(scheduler, "_scheduler_iteration", 0)

        if ttft is not None:
            ttft_list.append(ttft)
        latency_list.append(latency)
        tok_per_s_list.append(output_tokens / latency if latency > 0 else 0.0)
        frag_deltas.append(frag_after - frag_before)
        zero_deltas.append(zero_after - zero_before)
        round_deltas.append(round_after - round_before)

    def _avg(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    print("=== Mamba Prefill Fragmentation Benchmark ===")
    print(f"model={args.model}")
    print(f"prompt_length={args.prompt_length} max_tokens={args.max_tokens}")
    print(f"repeat_count={args.repeat_count}")
    if ttft_list:
        print(f"ttft_avg_s={_avg(ttft_list):.6f}")
    else:
        print("ttft_avg_s=NA (metrics unavailable)")
    print(f"tokens_per_s_avg={_avg(tok_per_s_list):.2f}")
    print(f"latency_avg_s={_avg(latency_list):.6f}")
    print(f"scheduler_rounds_avg={_avg([float(x) for x in round_deltas]):.2f}")
    print(f"fragmentation_events_avg={_avg([float(x) for x in frag_deltas]):.2f}")
    print(f"zero_collapse_events_avg={_avg([float(x) for x in zero_deltas]):.2f}")


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="Benchmark Mamba prefill fragmentation."
    )
    parser.add_argument(
        "--prompt-length",
        type=int,
        default=2000,
        help="Prompt length in tokens.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32,
        help="Max output tokens.",
    )
    parser.add_argument(
        "--repeat-count",
        type=int,
        default=5,
        help="Number of measured iterations.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=1,
        help="Warmup iterations.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed.",
    )
    EngineArgs.add_cli_args(parser)
    parser.set_defaults(model="Qwen3-5.2B")
    args = parser.parse_args()
    main(args)

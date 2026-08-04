# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass
import argparse
import asyncio
import logging
import random
import time

# Third Party
from transformers import AutoTokenizer
from utils import (
    AsyncLoopWrapper,
    PromptBuildMethodType,
    build_rag_prompt,
    compute_f1,
    compute_rl,
    init_logger,
    load_dataset,
)
import openai
import pandas as pd

logger = init_logger(__name__, logging.INFO)

system_prompt_set = {
    PromptBuildMethodType.QA: "You will be asked a question after reading several passages. "  # noqa: E501
    "Please directly answer the question based on the given passages. "
    "Do NOT repeat the question. "
    "The answer should be within 5 words..\nPassages:\n",
    PromptBuildMethodType.FEW_SHOT: "Summarize the dialogue into a few short sentences. "  # noqa: E501
    "The following are some examples.\n\n",
}
query_prompt_set = {
    PromptBuildMethodType.QA: "\n\nAnswer the question directly based on the given passages."  # noqa: E501
    " Do NOT repeat the question. "
    "The answer should be within 5 words. \nQuestion:",
    PromptBuildMethodType.FEW_SHOT: "",
}


@dataclass
class WorkloadConfig:
    # Overall QPS
    qps: float
    # Model name
    model: str
    # Tokenizer name
    tokenizer: str
    # Dataset.
    dataset: str
    # Start index of the workload
    start_index: int
    # End index of the workload
    end_index: int
    # Random shuffle.
    shuffle: bool
    # System prompt.
    system_prompt: str
    # Separator.
    separator: str
    # Query prompt.
    query_prompt: str
    # Prompt build method.
    prompt_build_method: PromptBuildMethodType
    # Max tokens for each generation.
    max_tokens: int


@dataclass
class Response:
    request_id: int
    body: str
    ttft: float
    generation_time: float
    prompt_tokens: int
    generation_tokens: int
    launch_time: float
    finish_time: float


def parse_arguments():
    parser = argparse.ArgumentParser(description="Parse RAG benchmark configurations.")
    parser.add_argument("--qps", type=float, required=True, help="Overall QPS")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--tokenizer", type=str, default="", help="Tokenizer name")
    parser.add_argument("--dataset", type=str, required=True, help="The dataset path")
    parser.add_argument(
        "--start-index", type=int, default=0, help="Start index of the workload"
    )
    parser.add_argument(
        "--end-index", type=int, default=-1, help="End index of the workload"
    )
    parser.add_argument("--shuffle", action="store_true", help="Random shuffle")
    parser.add_argument("--system-prompt", type=str, default="", help="System prompt")
    parser.add_argument("--separator", type=str, default="", help="Separator")
    parser.add_argument("--query-prompt", type=str, default="", help="Query prompt")
    parser.add_argument(
        "--prompt-build-method",
        type=str,
        required=True,
        help="Prompt build method",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        required=True,
        help="Base URL of the serving engine endpoint",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="EMPTY",
        help="API key of the serving engine endpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="summary.csv",
        help="The output file name (ended with csv or txt) for the summary csv and txt",
    )
    parser.add_argument(
        "--warmup", action="store_true", help="Whether to enable warmup"
    )
    parser.add_argument(
        "--time",
        type=int,
        default=None,
        help="The total running time in seconds",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Whether to enable verbose logging",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32,
        help="Max tokens for each generation",
    )
    parser.add_argument(
        "--step-interval", type=float, default=0.02, help="Step interval"
    )
    args = parser.parse_args()
    return args


class RequestExecutor:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        prompt_build_method: PromptBuildMethodType,
        model: str,
    ):
        self.client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.loop = AsyncLoopWrapper.GetOrStartLoop()
        self.prompt_build_method = prompt_build_method

    async def _async_launch_request(self, request_id, prompt, max_tokens):
        start_time = time.time()
        first_token_time = None
        words = ""
        response = None
        if self.prompt_build_method == PromptBuildMethodType.QA:
            messages = [{"role": "user", "content": prompt}]
            response = await self.client.chat.completions.create(
                messages=messages,
                model=self.model,
                temperature=0,
                stream=True,
                max_tokens=max_tokens,
                stream_options={"include_usage": True},
            )
        elif self.prompt_build_method == PromptBuildMethodType.FEW_SHOT:
            response = await self.client.completions.create(
                prompt=prompt,
                model=self.model,
                temperature=0,
                stream=True,
                max_tokens=max_tokens,
                stream_options={"include_usage": True},
            )
        else:
            raise ValueError(f"Invalid prompt build method {self.prompt_build_method}")
        async for tok in response:
            if not tok.choices:
                continue
            chunk_message = tok.choices[0].delta.content
            if chunk_message is not None:
                if first_token_time is None and chunk_message != "":
                    first_token_time = time.time()
                words += chunk_message
        tokens_out = tok.usage.completion_tokens
        tokens_prefill = tok.usage.prompt_tokens
        finish_time = time.time()
        if first_token_time is None:
            first_token_time = finish_time
        return Response(
            request_id=request_id,
            body=words,
            ttft=first_token_time - start_time,
            generation_time=finish_time - first_token_time,
            prompt_tokens=tokens_prefill,
            generation_tokens=tokens_out,
            launch_time=start_time,
            finish_time=finish_time,
        )

    def launch_request(self, request_id: int, prompt, max_tokens, finish_callback):
        """
        finish_callback: Callable[[Response], None]
        """
        real_callback = lambda x: finish_callback(x.result())
        future = asyncio.run_coroutine_threadsafe(
            self._async_launch_request(request_id, prompt, max_tokens),
            self.loop,
        )
        future.add_done_callback(real_callback)


def warmup_engine(executor: RequestExecutor):
    logger.info("Warming up the engine")
    for i in range(10):
        prompt = f"WARMUP: Hi, I'm user {i}. Here are some text: {'hi ' * 100}."
        executor.launch_request(-1, prompt, 100, lambda x: None)

    AsyncLoopWrapper.WaitLoop()
    logger.info("Warm up finished.")


class RAGManager:
    def __init__(self, workload_config: WorkloadConfig):
        self.workload_config = workload_config
        eval_dataset = load_dataset(workload_config.dataset)
        start_index = workload_config.start_index
        end_index = workload_config.end_index
        if end_index < 0:
            end_index = len(eval_dataset)
        eval_dataset = eval_dataset[start_index:end_index]
        if workload_config.shuffle:
            random.shuffle(eval_dataset)
        self._prompts = []
        self._answers = []
        self._build_method = workload_config.prompt_build_method
        self._generated_text: list[str | None] = []
        self._generation_time: list[float | None] = []
        self._prefill_tok_cnt: list[int | None] = []
        self._generation_tok_cnt: list[int | None] = []
        self._ttft: list[float | None] = []
        self._tpot: list[float | None] = []
        for ex in eval_dataset:
            prompt, _ = build_rag_prompt(
                workload_config.system_prompt,
                ex,
                workload_config.query_prompt,
                workload_config.separator,
                workload_config.prompt_build_method,
            )
            self._prompts.append(prompt)
            self._answers.append(ex["answers"])
            self._generated_text.append(None)
            self._generation_time.append(None)
            self._prefill_tok_cnt.append(None)
            self._generation_tok_cnt.append(None)
            self._ttft.append(None)
            self._tpot.append(None)
        self._tokenizer = AutoTokenizer.from_pretrained(workload_config.tokenizer)
        self._last_request_time = -1.0
        self._last_request_index = 0
        assert workload_config.qps > 0
        self._gap = 1.0 / workload_config.qps
        self._max_tokens = workload_config.max_tokens

    def _update_result(self, response: Response):
        self._generated_text[response.request_id] = response.body
        self._ttft[response.request_id] = response.ttft
        self._tpot[response.request_id] = (
            response.generation_time / response.generation_tokens
        )
        self._generation_time[response.request_id] = response.generation_time
        self._prefill_tok_cnt[response.request_id] = response.prompt_tokens
        self._generation_tok_cnt[response.request_id] = response.generation_tokens

    def step(self, timestamp: float, executor: RequestExecutor) -> bool:
        if self._last_request_index >= len(self._prompts):
            return False
        if (
            self._last_request_time < 0
            or timestamp >= self._last_request_time + self._gap
        ):
            prompt = self._prompts[self._last_request_index]
            request_id = self._last_request_index
            self._last_request_time = timestamp
            self._last_request_index += 1
            executor.launch_request(
                request_id, prompt, self._max_tokens, self._update_result
            )
        return True

    def summary(self, start_time: float, end_time: float) -> pd.DataFrame:
        cnt = len(self._ttft)
        assert cnt > 0
        avg_ttft = sum(ttft for ttft in self._ttft if ttft is not None) / cnt
        avg_tpot = sum(tpot for tpot in self._tpot if tpot is not None) / cnt
        # Create a dataframe
        quality = []
        for i in range(cnt):
            if self._build_method == PromptBuildMethodType.QA:
                quality.append(
                    max(
                        [
                            compute_f1(self._generated_text[i], answer, self._tokenizer)
                            for answer in self._answers[i]
                        ]
                    )
                )
            elif self._build_method == PromptBuildMethodType.FEW_SHOT:
                quality.append(
                    max(
                        [
                            compute_rl(self._generated_text[i], answer)
                            for answer in self._answers[i]
                        ]
                    )
                )
            else:
                raise ValueError(f"Invalid prompt build method {self._build_method}")
        avg_quality = sum(quality) / cnt
        df = pd.DataFrame(
            {
                "quality": quality,
                "ttft": self._ttft,
                "tpot": self._tpot,
                "generation_time": self._generation_time,
                "prefill_token_cnt": self._prefill_tok_cnt,
                "generation_token_cnt": self._generation_tok_cnt,
            }
        )
        total_time = end_time - start_time
        thput = cnt / total_time
        logger.info(
            f"Summary: {cnt} requests, average_ttft={avg_ttft} (second)\n"
            f" average_tpot={avg_tpot} (second)\n"
            f"throughput={thput} (req/s)\n"
            f"average_quality={avg_quality}\n"
        )
        return df


def run_rag(args):
    build_prompt_method_str = args.prompt_build_method.upper()
    build_prompt_method = None
    if build_prompt_method_str == "QA":
        build_prompt_method = PromptBuildMethodType.QA
    elif build_prompt_method_str == "FEW_SHOT":
        build_prompt_method = PromptBuildMethodType.FEW_SHOT
    else:
        raise ValueError(f"Invalid prompt build method {build_prompt_method_str}")
    workload_config = WorkloadConfig(
        qps=args.qps,
        model=args.model,
        tokenizer=args.tokenizer,
        dataset=args.dataset,
        start_index=args.start_index,
        end_index=args.end_index,
        shuffle=args.shuffle,
        system_prompt=args.system_prompt,
        separator=args.separator,
        query_prompt=args.query_prompt,
        prompt_build_method=build_prompt_method,
        max_tokens=args.max_tokens,
    )
    executor = RequestExecutor(
        base_url=args.base_url,
        api_key=args.api_key,
        prompt_build_method=build_prompt_method,
        model=args.model,
    )
    if args.warmup:
        warmup_engine(executor)
    manager = RAGManager(workload_config)
    step_interval = args.step_interval
    num_steps = 0
    start_time = time.time()
    try:
        while True:
            num_steps += 1
            effective = manager.step(time.time(), executor)
            if not effective:
                break
            time.sleep(step_interval)
            if args.time is not None and time.time() - start_time > args.time:
                break

    except KeyboardInterrupt:
        logger.info("Interrupted, waiting for the final result")

    AsyncLoopWrapper.StopLoop()

    logger.info(f"Finished benchmarking, dumping summary to {args.output}")
    summary = manager.summary(start_time, time.time())
    summary.to_csv(args.output, index=False)


def main():
    args = parse_arguments()
    build_prompt_method_str = args.prompt_build_method.upper()
    build_prompt_method = None
    if build_prompt_method_str == "QA":
        build_prompt_method = PromptBuildMethodType.QA
    elif build_prompt_method_str == "FEW_SHOT":
        build_prompt_method = PromptBuildMethodType.FEW_SHOT
    else:
        raise ValueError(f"Invalid prompt build method {build_prompt_method_str}")
    if len(args.system_prompt) == 0:
        args.system_prompt = system_prompt_set[build_prompt_method]
    if len(args.query_prompt) == 0:
        args.query_prompt = query_prompt_set[build_prompt_method]
    if len(args.tokenizer) == 0:
        args.tokenizer = args.model
    args.system_prompt = args.system_prompt.encode().decode("unicode_escape")
    args.query_prompt = args.query_prompt.encode().decode("unicode_escape")
    if args.verbose:
        global logger
        logger = init_logger(__name__, logging.DEBUG)
    run_rag(args)


if __name__ == "__main__":
    main()

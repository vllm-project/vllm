# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass
from typing import Optional
import argparse
import asyncio
import json
import logging
import time

# Third Party
from utils import AsyncLoopWrapper, init_logger
import openai
import pandas as pd

logger = init_logger(__name__, logging.INFO)


@dataclass
class WorkloadConfig:
    # Max number of users in the system concurrently
    num_users: int

    # Length of shared system prompt
    system_prompt_len: int

    # Length of the user-specific data
    user_info_len: int

    # Length of the answer in one round
    answer_len: int

    # Number of rounds in the conversation
    num_rounds: int

    # Overall QPS
    qps: int

    # Model name
    model: str

    # Whether to include user id in request header
    enable_user_id: bool

    # Whether strictly cap active sessions at num_users
    enforce_strict_concurrent_users: bool = False

    # Whether to disable ramp up during the uphill stage
    disable_ramp_up: bool = False


@dataclass
class UserConfig:
    # User id
    user_id: int

    # System prompt length
    system_prompt_len: int

    # Length of the user-specific data
    user_info_len: int

    # Answer length
    answer_len: int

    # Gap between two requests
    gap_between_requests: float

    # Num rounds
    num_rounds: int

    # Whether to include user id in request header
    enable_user_id: bool

    @staticmethod
    def new_user_config(user_id: int, workload_config: WorkloadConfig) -> "UserConfig":
        return UserConfig(
            user_id=user_id,
            system_prompt_len=workload_config.system_prompt_len,
            user_info_len=workload_config.user_info_len,
            answer_len=workload_config.answer_len,
            gap_between_requests=workload_config.num_users / workload_config.qps,
            num_rounds=workload_config.num_rounds,
            enable_user_id=workload_config.enable_user_id,
        )


class ChatHistory:
    def __init__(
        self,
    ):
        self.history = []

    def on_user_query(self, query: str):
        if len(self.history) == 0:
            self.history.append({"role": "user", "content": query})
        else:
            assert self.history[-1]["role"] == "assistant", "Expect system response"
            self.history.append({"role": "user", "content": query})

    def on_system_response(self, response: str):
        assert len(self.history) > 0, "Expect user query"
        assert self.history[-1]["role"] == "user", "Expect user query"
        self.history.append({"role": "assistant", "content": response})

    def get_messages_for_openai(self):
        return self.history

    def __len__(self):
        return len(self.history)


@dataclass
class Response:
    body: str
    ttft: float
    generation_time: float
    prompt_tokens: int
    generation_tokens: int
    launch_time: float
    finish_time: float


class RequestExecutor:
    def __init__(self, base_url: str, api_key: str, model: str):
        self.client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.loop = AsyncLoopWrapper.GetOrStartLoop()

    async def _async_launch_request(self, messages, max_tokens, extra_headers=None):
        start_time = time.time()
        first_token_time = None
        words = ""

        response = await self.client.chat.completions.create(
            messages=messages,
            model=self.model,
            temperature=0,
            stream=True,
            max_tokens=max_tokens,
            stream_options={"include_usage": True},
            extra_headers=extra_headers,
        )

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

        if first_token_time is None:
            first_token_time = time.time()

        return Response(
            body=words,
            ttft=first_token_time - start_time,
            generation_time=time.time() - first_token_time,
            prompt_tokens=tokens_prefill,
            generation_tokens=tokens_out,
            launch_time=start_time,
            finish_time=time.time(),
        )

    def launch_request(
        self,
        chat_history: ChatHistory,
        max_tokens: int,
        finish_callback,
        extra_headers=None,
    ):
        """
        finish_callback: Callable[[Response], None]
        """
        messages = chat_history.get_messages_for_openai()
        real_callback = lambda x: finish_callback(x.result())
        future = asyncio.run_coroutine_threadsafe(
            self._async_launch_request(messages, max_tokens, extra_headers),
            self.loop,
        )
        future.add_done_callback(real_callback)


class UserSession:
    def __init__(self, user_config: UserConfig, use_sharegpt=False, sharegpt_data=None):
        self.user_config = user_config
        self.last_request_time: float | None = None
        self.chat_history = ChatHistory()
        self.question_id = 0
        self.use_sharegpt = use_sharegpt
        if self.use_sharegpt:
            self.sharegpt_data = sharegpt_data
            if self.sharegpt_data["num_round"] % 2 == 0:
                self.start_with_gpt = False
            else:
                self.start_with_gpt = True

        self.has_unfinished_request = False
        self.last_unfinished_log = 0.0

        self.prompt_lengths: list[int] = []
        self.generation_lengths: list[int] = []
        self.ttfts: list[float | None] = []
        self.generation_times: list[float | None] = []
        self.launch_times: list[float | None] = []
        self.finish_times: list[float | None] = []

        self.finished = False

    def _update_result(self, response: Response):
        self.prompt_lengths.append(response.prompt_tokens)
        self.generation_lengths.append(response.generation_tokens)
        self.ttfts.append(response.ttft)
        self.generation_times.append(response.generation_time)
        self.launch_times.append(response.launch_time)
        self.finish_times.append(response.finish_time)

    def _build_system_prompt(self):
        def gen_dummy_text(length):
            return " ".join(["hi"] * length)

        dummy_text_sys = gen_dummy_text(self.user_config.system_prompt_len)
        dummy_text_user = gen_dummy_text(self.user_config.user_info_len)
        system_prompt = (
            f"Hi, here's some system prompt: {dummy_text_sys}."
            + f"For user {self.user_config.user_id}, "
            + f"here are some other context: {dummy_text_user}."
        )
        return system_prompt

    def _build_new_question(self):
        self.question_id += 1
        return (
            f"Here's question #{self.question_id}: can you tell me "
            + "a new long story with a happy ending?"
        )

    def _launch_new_request(self, timestamp: float, request_executor: RequestExecutor):
        if self.use_sharegpt:
            if self.start_with_gpt:
                prompt = self.sharegpt_data["conversations"][2 * self.question_id + 1][
                    "value"
                ]
            else:
                prompt = self.sharegpt_data["conversations"][2 * self.question_id][
                    "value"
                ]
            self.question_id += 1
        else:
            prompt = self._build_new_question()
        if len(self.chat_history) == 0:
            prompt = self._build_system_prompt() + prompt
        self.chat_history.on_user_query(prompt)
        logger.debug(
            f"User {self.user_config.user_id} issues request {self.question_id}"
        )
        if self.use_sharegpt:
            if self.start_with_gpt:
                max_tokens = self.sharegpt_data["conversations"][2 * self.question_id][
                    "num_tokens"
                ]
            else:
                max_tokens = self.sharegpt_data["conversations"][
                    2 * self.question_id - 1
                ]["num_tokens"]
            max_tokens = min(max_tokens, self.user_config.answer_len)
        else:
            max_tokens = self.user_config.answer_len
        if request_executor is not None:
            request_executor.launch_request(
                self.chat_history,
                max_tokens,
                self._on_request_finished,
                extra_headers={"x-user-id": str(self.user_config.user_id)},
            )
            self.has_unfinished_request = True
        else:  # dry-run
            self.chat_history.on_system_response("")
            self.has_unfinished_request = False
        self.last_request_time = timestamp

    def _on_request_finished(self, response: Response):
        self.chat_history.on_system_response(response.body)
        self.has_unfinished_request = False
        logger.debug(
            f"User {self.user_config.user_id} finished one request. "
            f"Prompt tokens: {response.prompt_tokens}, "
            f"generation tokens: {response.generation_tokens}"
        )
        self._update_result(response)

    def set_internal_state(self, offset: float, timestamp: float):
        """Tell the session is the 'offset' seconds after the start"""
        assert len(self.chat_history) == 0, (
            "Internal state should be set before the first request"
        )

        num_passed_questions = int(offset / self.user_config.gap_between_requests) + 1

        passed_time = (num_passed_questions - 1) * self.user_config.gap_between_requests

        self.last_request_time = timestamp - offset + passed_time
        self.question_id = num_passed_questions
        logger.debug(
            f"Set internal state for user {self.user_config.user_id}, "
            f"question_id: {self.question_id}, "
            f"last_request_time: {self.last_request_time}"
        )

    def step(self, timestamp: float, request_executor: RequestExecutor):
        if (
            self.question_id >= self.user_config.num_rounds
            and not self.has_unfinished_request
        ):
            self.finished = True
            return

        if self.last_request_time is None:
            self._launch_new_request(timestamp, request_executor)
            return

        if timestamp - self.last_request_time > self.user_config.gap_between_requests:
            if self.has_unfinished_request:
                if timestamp - self.last_unfinished_log > 10:
                    logger.warning(
                        f"User {self.user_config.user_id} has an unfinished "
                        "request and unable to fit the QPS requirement."
                    )
                    self.last_unfinished_log = timestamp
                return

            self._launch_new_request(timestamp, request_executor)
            return

    def summary(self) -> pd.DataFrame:
        df = pd.DataFrame()
        df["prompt_tokens"] = self.prompt_lengths
        df["generation_tokens"] = self.generation_lengths
        df["ttft"] = self.ttfts
        df["generation_time"] = self.generation_times
        df["user_id"] = self.user_config.user_id
        df["question_id"] = range(1, len(self.prompt_lengths) + 1)
        df["launch_time"] = self.launch_times
        df["finish_time"] = self.finish_times
        return df


class UserSessionManager:
    def __init__(
        self,
        workload_config: WorkloadConfig,
        init_user_id=0,
        use_sharegpt=False,
    ):
        self.workload_config = workload_config
        self.sessions: list[UserSession] = []

        gap_between_requests_per_user = workload_config.num_users / workload_config.qps
        session_alive_time = gap_between_requests_per_user * (
            workload_config.num_rounds - 1
        )
        self.gap_between_users = session_alive_time / (workload_config.num_users + 0)
        self.ramp_up_time = workload_config.num_users * self.gap_between_users

        logger.info(
            f"Gap between users: {self.gap_between_users} secs.\n"
            f"Gap between user reqs: {gap_between_requests_per_user} secs.\n"
            f"Expected length of user session: {session_alive_time} secs."
        )

        self.user_id = init_user_id
        self.last_user_join = 0.0
        self.session_summaries: list[pd.DataFrame] = []
        self.start_time: float | None = None

        self.need_ramp_up = not workload_config.disable_ramp_up

        self.use_sharegpt = use_sharegpt
        if self.use_sharegpt:
            self._load_sharegpt_data()

        self.enforce_strict_concurrent_users = (
            workload_config.enforce_strict_concurrent_users
        )

    def _load_sharegpt_data(self):
        with open("ShareGPT.json", "r", encoding="utf-8") as file:
            self.sharegpt_data = json.load(file)
        self.sharegpt_data = [
            d
            for d in self.sharegpt_data
            if d["num_round"] > 2 * self.workload_config.num_rounds
        ]
        logger.info(f"There are {len(self.sharegpt_data)} users satisfying ")
        assert len(self.sharegpt_data) >= self.workload_config.num_users, (
            "Not enough data! Reduce --num-users or --num-rounds"
        )

    def _ramp_up(self, timestamp: float, ramp_up_time: float):
        for i in range(self.workload_config.num_users):
            new_session = self._create_user_session()
            offset = ramp_up_time - i * self.gap_between_users
            if offset < 0:
                break
            new_session.set_internal_state(offset, timestamp)
        self.need_ramp_up = False

    def _create_user_session(self):
        self.user_id += 1
        user_config = UserConfig.new_user_config(self.user_id, self.workload_config)
        if self.use_sharegpt:
            user_session = UserSession(
                user_config, self.use_sharegpt, self.sharegpt_data[self.user_id]
            )
        else:
            user_session = UserSession(user_config, self.use_sharegpt)
        self.sessions.append(user_session)
        return user_session

    def _remove_finished_sessions(self):
        sessions_to_remove = [s for s in self.sessions if s.finished]
        if len(sessions_to_remove) > 0:
            logger.info(
                f"Removing {len(sessions_to_remove)} finished sessions, now "
                f"active users: {len(self.sessions) - len(sessions_to_remove)}"
            )
            for session in sessions_to_remove:
                self.session_summaries.append(session.summary())
        self.sessions = [s for s in self.sessions if not s.finished]

    def _can_join_user(self, timestamp: float) -> bool:
        # No new user session if gap_between_users time interval not meets
        if timestamp - self.last_user_join <= self.gap_between_users:
            return False

        # No user seession if active user count is less than configured
        if (
            self.enforce_strict_concurrent_users
            and len(self.sessions) >= self.workload_config.num_users
        ):
            return False
        return True

    def step(self, timestamp: float, executor: RequestExecutor):
        if self.need_ramp_up:
            self._ramp_up(timestamp, self.ramp_up_time)

        if self.start_time is None:
            self.start_time = timestamp

        # Check if can join new user session
        if self._can_join_user(timestamp):
            self._create_user_session()
            self.last_user_join = timestamp
            logger.info(
                f"Joined a new user {self.user_id}, "
                f"now active users: {len(self.sessions)}"
            )

        for session in self.sessions:
            session.step(timestamp, executor)

        self._remove_finished_sessions()

    @staticmethod
    def ProcessSummary(
        df: pd.DataFrame,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        pending_queries: int = 0,
        config_qps: Optional[float] = None,
    ):
        if start_time and end_time:
            launched_queries = len(
                df.query(f"{start_time} <= launch_time <= {end_time}")
            )
            df = df.query(f"{start_time} <= finish_time <= {end_time}")
        else:
            launched_queries = len(df)

        logger.debug(
            f"Launched queries: {launched_queries}, "
            f"pending queries: {pending_queries}, "
            f"finished queries: {len(df)}"
        )

        if config_qps is None:
            config_qps = 0.0

        if start_time is None:
            start_time = df["launch_time"].min()
        if end_time is None:
            end_time = df["finish_time"].max()
        total_time = end_time - start_time

        total_requests = launched_queries + pending_queries
        actual_qps = total_requests / total_time

        total_finished_requests = len(df)
        finished_qps = total_finished_requests / total_time

        total_prompt_tokens = df["prompt_tokens"].sum()
        total_generation_tokens = df["generation_tokens"].sum()
        average_prefill_speed = total_prompt_tokens / total_time
        average_generation_speed = total_generation_tokens / total_time
        average_generation_speed_per_request = (
            df["generation_tokens"] / df["generation_time"]
        ).mean()
        average_ttft = df["ttft"].mean()
        logger.info("Calculating performance summary")
        print("\n")
        print("==================== Performance summary ======================")
        print(f"  \033[33mConfig QPS: \033[32m{config_qps:.4f} reqs/s\033[0m\n")
        print(f"  \033[33mActual QPS: \033[32m{actual_qps:.4f} reqs/s\033[0m\n")

        print(f"  \033[33mProcessing speed: \033[32m{finished_qps:.4f} reqs/s\033[0m\n")

        print(f"  \033[33mRequests on-the-fly: {pending_queries}\033[0m\n")

        print(
            "  \033[33mInput tokens per second: "
            f"\033[32m{average_prefill_speed:.4f} tokens/s\033[0m\n"
        )

        print(
            "  \033[33mOutput tokens per second: "
            f"\033[32m{average_generation_speed:.4f} tokens/s\033[0m\n"
        )

        print(
            "  \033[33mAverage generation throughput (per request): "
            f"\033[32m{average_generation_speed_per_request:.4f} "
            "tokens/req/s\033[0m\n"
        )

        print(f"  \033[33mAverage TTFT: \033[32m{average_ttft:.4f}s\033[0m\n")

        print(f"Time range: {start_time} - {end_time} ({total_time:.2f}s)")

        print("===============================================================")
        print("\n")
        return df

    def summary(self, start_time: float, end_time: float) -> pd.DataFrame:
        if len(self.session_summaries) == 0 and len(self.sessions) == 0:
            return pd.DataFrame()

        df = pd.concat(
            [s for s in self.session_summaries] + [s.summary() for s in self.sessions]
        )
        pending_queries = len([s for s in self.sessions if s.has_unfinished_request])
        assert self.start_time is not None
        start_time = max(self.start_time, start_time)
        qps = self.workload_config.qps

        df = UserSessionManager.ProcessSummary(
            df, start_time, end_time, pending_queries, qps
        )
        return df


def warmup_engine(executor):
    logger.info("Warming up the engine")
    for i in range(10):
        chat_history = ChatHistory()
        chat_history.on_user_query(
            f"WARMUP: Hi, I'm user {i}. Here are some text: {'hi ' * 100}."
        )
        executor.launch_request(chat_history, 100, lambda x: None)

    AsyncLoopWrapper.WaitLoop()


def parse_arguments():
    parser = argparse.ArgumentParser(description="Parse benchmark configurations.")

    parser.add_argument(
        "--num-users",
        type=int,
        required=True,
        help="Max number of users in the system concurrently",
    )
    parser.add_argument(
        "--shared-system-prompt",
        type=int,
        required=True,
        help="Length of the shared system prompt (tokens)",
    )
    parser.add_argument(
        "--user-history-prompt",
        type=int,
        required=True,
        help="Length of the user-specific history prompt (tokens)",
    )
    parser.add_argument(
        "--answer-len",
        type=int,
        required=True,
        help="Length of the answer in one round",
    )
    parser.add_argument(
        "--num-rounds",
        type=int,
        required=True,
        help="Number of rounds in the conversation",
    )
    parser.add_argument("--qps", type=float, required=True, help="Overall QPS")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument(
        "--base-url",
        type=str,
        required=True,
        help="Base URL of the serving engine endpoint",
    )
    parser.add_argument(
        "--time",
        type=int,
        required=False,
        help="The time to run the simulation in seconds",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="summary.csv",
        help="The output file name (ended with csv or txt) for the summary csv and txt",
    )
    parser.add_argument(
        "--init-user-id",
        type=int,
        default=0,
        help="The initial user id to start with",
    )
    parser.add_argument(
        "--request-with-user-id",
        action="store_true",
        help="Whether to enable user id in the request headers",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=30,
        help="The time between two summary loggings in seconds",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Whether to enable verbose logging",
    )
    parser.add_argument(
        "--sharegpt",
        action="store_true",
        help="Whether to use ShareGPT dataset",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Does not send requests to the endpoint (server)",
    )
    parser.add_argument(
        "--enforce-strict-concurrent-users",
        action="store_true",
        help="Strictly enforce concurrent users count to match --num-users",
    )
    parser.add_argument(
        "--disable-ramp-up",
        action="store_true",
        help="Disable the initial ramp-up phase, allowing users to join gradually",
    )
    args = parser.parse_args()
    return args


def parse_process_summary():
    parser = argparse.ArgumentParser(
        description="Parse benchmark configurations.", add_help=False
    )

    parser.add_argument("--process-summary", type=str, default=None)

    args, _ = parser.parse_known_args()
    return args


def process_output(filename):
    logger.warning(
        f"Processing the existing summary file {filename}"
        ", ignoring all the other arguments"
    )
    UserSessionManager.ProcessSummary(pd.read_csv(filename), pending_queries=0)


def main():
    args = parse_process_summary()
    if args.process_summary:
        process_output(args.process_summary)
        return

    args = parse_arguments()
    if args.verbose:
        global logger
        logger = init_logger(__name__, logging.DEBUG)

    step_interval = 0.1

    executor = None
    if not args.dry_run:
        executor = RequestExecutor(
            base_url=args.base_url, api_key="EMPTY", model=args.model
        )

        warmup_engine(executor)

    workload_config = WorkloadConfig(
        num_users=args.num_users,
        system_prompt_len=args.shared_system_prompt,
        user_info_len=args.user_history_prompt,
        answer_len=args.answer_len,
        num_rounds=args.num_rounds,
        qps=args.qps,
        model=args.model,
        enable_user_id=args.request_with_user_id,
        enforce_strict_concurrent_users=args.enforce_strict_concurrent_users,
        disable_ramp_up=args.disable_ramp_up,
    )

    manager = UserSessionManager(
        workload_config,
        init_user_id=args.init_user_id,
        use_sharegpt=args.sharegpt,
    )

    num_steps = 0
    start_time = time.time()
    last_summary_time = start_time
    try:
        while True:
            num_steps += 1
            manager.step(time.time(), executor)
            time.sleep(step_interval)

            if time.time() - last_summary_time > args.log_interval:
                manager.summary(last_summary_time, time.time())
                last_summary_time = time.time()

            if args.time is not None and time.time() - start_time > args.time:
                break

    except KeyboardInterrupt:
        logger.info("Interrupted, waiting for the final result")

    AsyncLoopWrapper.StopLoop()

    logger.info(f"Finished benchmarking, dumping summary to {args.output}")
    summary = manager.summary(0, time.time())
    summary.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()

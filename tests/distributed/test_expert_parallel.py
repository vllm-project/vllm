# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import List, Literal, NamedTuple, Optional

import pytest

from vllm.config import TaskOption
from vllm.logger import init_logger

from ..utils import compare_two_settings, fork_new_process_for_each_test

logger = init_logger("test_pipeline_parallel")


class ParallelSetup(NamedTuple):
    tp_size: int
    eager_mode: bool
    chunked_prefill: bool


class EPTestOptions(NamedTuple):
    trust_remote_code: bool
    tokenizer_mode: Optional[str]
    load_format: Optional[str] = None
    hf_overrides: Optional[str] = None


@dataclass
class EPTestSettings:
    parallel_setups: List[ParallelSetup]
    distributed_backends: List[str]
    task: TaskOption
    test_options: EPTestOptions

    @staticmethod
    def detailed(
        *,
        tp_base: int = 2,
        task: TaskOption = "auto",
        trust_remote_code: bool = False,
        tokenizer_mode: Optional[str] = None,
        load_format: Optional[str] = None,
        hf_overrides: Optional[str] = None,
    ):
        return EPTestSettings(
            parallel_setups=[
                ParallelSetup(tp_size=tp_base,
                              eager_mode=False,
                              chunked_prefill=False),
                ParallelSetup(tp_size=tp_base,
                              eager_mode=False,
                              chunked_prefill=True),
                ParallelSetup(tp_size=tp_base,
                              eager_mode=True,
                              chunked_prefill=False),
                ParallelSetup(tp_size=2 * tp_base,
                              eager_mode=False,
                              chunked_prefill=True),
                ParallelSetup(tp_size=2 * tp_base,
                              eager_mode=True,
                              chunked_prefill=False),
            ],
            distributed_backends=["mp", "ray"],
            task=task,
            test_options=EPTestOptions(trust_remote_code=trust_remote_code,
                                       tokenizer_mode=tokenizer_mode,
                                       load_format=load_format,
                                       hf_overrides=hf_overrides),
        )

    @staticmethod
    def fast(
        *,
        tp_base: int = 2,
        task: TaskOption = "auto",
        trust_remote_code: bool = False,
        tokenizer_mode: Optional[str] = None,
        load_format: Optional[str] = None,
        hf_overrides: Optional[str] = None,
    ):
        return EPTestSettings(
            parallel_setups=[
                ParallelSetup(tp_size=tp_base,
                              eager_mode=True,
                              chunked_prefill=False),
            ],
            distributed_backends=["mp"],
            task=task,
            test_options=EPTestOptions(trust_remote_code=trust_remote_code,
                                       tokenizer_mode=tokenizer_mode,
                                       load_format=load_format,
                                       hf_overrides=hf_overrides),
        )

    def iter_params(self, model_name: str):
        opts = self.test_options

        for parallel_setup in self.parallel_setups:
            for distributed_backend in self.distributed_backends:
                yield (model_name, parallel_setup, distributed_backend,
                       self.task, opts)


# NOTE: You can adjust tp_base locally to fit the model in GPU
# The values displayed here are only a rough indicator of the size of the model

# yapf: disable
TEST_MODELS = {
    # "ai21labs/Jamba-v0.1": EPTestSettings.fast(trust_remote_code=True),
    "deepseek-ai/deepseek-llm-7b-chat": EPTestSettings.fast(
        trust_remote_code=True),
    "deepseek-ai/DeepSeek-V2-Lite-Chat": EPTestSettings.fast(
        trust_remote_code=True),
    "mistralai/Mixtral-8x7B-Instruct-v0.1": EPTestSettings.fast(tp_base=4),
}


def _compare_tp(
    model_name: str,
    parallel_setup: ParallelSetup,
    distributed_backend: str,
    task: TaskOption,
    test_options: EPTestOptions,
    num_gpus_available: int,
    *,
    method: Literal["generate", "encode"],
):
    (
        tp_size,
        eager_mode,
        chunked_prefill,
    ) = parallel_setup
    (
        trust_remote_code,
        tokenizer_mode,
        load_format,
        hf_overrides,
    ) = test_options

    if num_gpus_available < tp_size:
        pytest.skip(f"Need at least {tp_size} GPUs")

    common_args = [
        # use half precision for speed and memory savings in CI environment
        "--dtype",
        "float16",
        "--max-model-len",
        "2048",
        "--max-num-seqs",
        "8",
    ]
    if chunked_prefill:
        common_args.append("--enable-chunked-prefill")
    if eager_mode:
        common_args.append("--enforce-eager")
    if task != "auto":
        common_args.extend(["--task", task])
    if trust_remote_code:
        common_args.append("--trust-remote-code")
    if tokenizer_mode:
        common_args.extend(["--tokenizer-mode", tokenizer_mode])
    if load_format:
        common_args.extend(["--load-format", load_format])
    if hf_overrides:
        common_args.extend(["--hf-overrides", hf_overrides])

    ep_env = {
        "VLLM_TEST_EP_PARALLEL": "1",
    }

    ep_args = [
        *common_args,
        "--tensor-parallel-size",
        str(tp_size),
        "--distributed-executor-backend",
        distributed_backend,
    ]

    # compare without pipeline parallelism
    # NOTE: use mp backend for TP
    # PP tests might involve multiple nodes, and ray might
    #  schedule all workers in a node other than the head node,
    #  which can cause the test to fail.
    tp_args = [
        *common_args,
        "--tensor-parallel-size",
        str(tp_size),
        "--distributed-executor-backend",
        "mp",
    ]

    try:
        compare_two_settings(model_name,
                             ep_args,
                             tp_args,
                             ep_env,
                             method=method)
    except Exception:
        raise


@pytest.mark.parametrize(
    ("model_name", "parallel_setup", "distributed_backend", "task",
     "test_options"),
    [
        params for model_name, settings in TEST_MODELS.items()
        for params in settings.iter_params(model_name)
    ],
)
@fork_new_process_for_each_test
def test_ep(
    model_name: str,
    parallel_setup: ParallelSetup,
    distributed_backend: str,
    task: TaskOption,
    test_options: EPTestOptions,
    num_gpus_available,
):
    _compare_tp(model_name,
                parallel_setup,
                distributed_backend,
                task,
                test_options,
                num_gpus_available,
                method="generate")

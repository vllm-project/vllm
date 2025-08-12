"""
WARNING: This test runs in both single-node (4 GPUs) and multi-node
 (2 node with 2 GPUs each) modes. If the test only uses 2 GPUs, it is
 important to set the distributed backend to "mp" to avoid Ray scheduling
 all workers in a node other than the head node, which can cause the test
 to fail.
"""
import os
from dataclasses import dataclass
from typing import List, Literal, NamedTuple, Optional

import pytest

from vllm.config import TaskOption
from vllm.logger import init_logger

from ..utils import compare_two_settings, fork_new_process_for_each_test

logger = init_logger("test_pipeline_parallel")

VLLM_MULTI_NODE = os.getenv("VLLM_MULTI_NODE", "0") == "1"


class ParallelSetup(NamedTuple):
    tp_size: int
    pp_size: int
    eager_mode: bool
    chunked_prefill: bool


@dataclass
class PPTestSettings:
    parallel_setups: List[ParallelSetup]
    distributed_backends: List[str]
    task: TaskOption
    trust_remote_code: bool
    tokenizer_mode: Optional[str]

    @staticmethod
    def detailed(
        *,
        tp_base: int = 1,
        pp_base: int = 2,
        task: TaskOption = "auto",
        trust_remote_code: bool = False,
        tokenizer_mode: Optional[str] = None,
    ):
        return PPTestSettings(
            parallel_setups=[
                ParallelSetup(tp_size=tp_base,
                              pp_size=pp_base,
                              eager_mode=False,
                              chunked_prefill=False),
                ParallelSetup(tp_size=tp_base,
                              pp_size=2 * pp_base,
                              eager_mode=False,
                              chunked_prefill=True),
                ParallelSetup(tp_size=tp_base,
                              pp_size=2 * pp_base,
                              eager_mode=True,
                              chunked_prefill=False),
                ParallelSetup(tp_size=2 * tp_base,
                              pp_size=pp_base,
                              eager_mode=False,
                              chunked_prefill=True),
                ParallelSetup(tp_size=2 * tp_base,
                              pp_size=pp_base,
                              eager_mode=True,
                              chunked_prefill=False),
            ],
            distributed_backends=["mp", "ray"],
            task=task,
            trust_remote_code=trust_remote_code,
            tokenizer_mode=tokenizer_mode,
        )

    @staticmethod
    def fast(
        *,
        tp_base: int = 1,
        pp_base: int = 2,
        task: TaskOption = "auto",
        trust_remote_code: bool = False,
        tokenizer_mode: Optional[str] = None,
    ):
        return PPTestSettings(
            parallel_setups=[
                ParallelSetup(tp_size=tp_base,
                              pp_size=pp_base,
                              eager_mode=True,
                              chunked_prefill=False),
            ],
            distributed_backends=["mp"],
            task=task,
            trust_remote_code=trust_remote_code,
            tokenizer_mode=tokenizer_mode,
        )

    def iter_params(self, model_name: str):
        for parallel_setup in self.parallel_setups:
            for distributed_backend in self.distributed_backends:
                yield (model_name, parallel_setup, distributed_backend,
                       self.task, self.trust_remote_code, self.tokenizer_mode)


# NOTE: You can adjust tp_base and/or pp_base locally to fit the model in GPU
# The values displayed here are only a rough indicator of the size of the model

# yapf: disable
GENERATION_MODEL_SETTINGS = {
    # [DETAILED TESTS]
    "meta-llama/Meta-Llama-3-8B": PPTestSettings.detailed(),
    # [FAST TESTS]
    # Uses Llama
    # "BAAI/AquilaChat-7B": PPTestSettings.fast(),
    "Snowflake/snowflake-arctic-instruct": PPTestSettings.fast(tp_base=8, trust_remote_code=True),  # noqa: E501
    "baichuan-inc/Baichuan-7B": PPTestSettings.fast(trust_remote_code=True),
    "baichuan-inc/Baichuan2-13B-Chat": PPTestSettings.fast(trust_remote_code=True),  # noqa: E501
    "bigscience/bloomz-1b1": PPTestSettings.fast(),
    "THUDM/chatglm3-6b": PPTestSettings.fast(trust_remote_code=True),
    "CohereForAI/c4ai-command-r-v01": PPTestSettings.fast(tp_base=2, trust_remote_code=True),  # noqa: E501
    "databricks/dbrx-instruct": PPTestSettings.fast(tp_base=8),
    "Deci/DeciLM-7B-instruct": PPTestSettings.fast(trust_remote_code=True),
    "deepseek-ai/deepseek-llm-7b-chat": PPTestSettings.fast(),
    "deepseek-ai/DeepSeek-V2-Lite-Chat": PPTestSettings.fast(trust_remote_code=True),  # noqa: E501
    "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct": PPTestSettings.fast(),
    "tiiuae/falcon-7b": PPTestSettings.fast(),
    "google/gemma-2b": PPTestSettings.fast(),
    "google/gemma-2-9b": PPTestSettings.fast(),
    "gpt2": PPTestSettings.fast(),
    "bigcode/starcoder": PPTestSettings.fast(),
    "EleutherAI/gpt-j-6b": PPTestSettings.fast(),
    "EleutherAI/pythia-12b": PPTestSettings.fast(),
    "ibm/PowerLM-3b": PPTestSettings.fast(),
    "ibm/PowerMoE-3b": PPTestSettings.fast(),
    # Uses Llama
    # "internlm/internlm-chat-7b": PPTestSettings.fast(),
    "internlm/internlm2-chat-7b": PPTestSettings.fast(trust_remote_code=True),
    "core42/jais-13b-chat": PPTestSettings.fast(),
    # TODO: Implement PP
    # "ai21labs/AI21-Jamba-1.5-Mini": PPTestSettings.fast(),
    "openbmb/MiniCPM-2B-sft-bf16": PPTestSettings.fast(trust_remote_code=True),
    "openbmb/MiniCPM3-4B": PPTestSettings.fast(trust_remote_code=True),
    # Uses Llama
    # "mistralai/Mistral-7B-Instruct-v0.1": PPTestSettings.fast(),
    "mistralai/Mixtral-8x7B-Instruct-v0.1": PPTestSettings.fast(tp_base=4),
    "mosaicml/mpt-7b": PPTestSettings.fast(),
    "nvidia/Minitron-8B-Base": PPTestSettings.fast(),
    "allenai/OLMoE-1B-7B-0924-Instruct": PPTestSettings.fast(),
    "allenai/OLMo-1B-hf": PPTestSettings.fast(),
    "facebook/opt-iml-max-1.3b": PPTestSettings.fast(),
    "OrionStarAI/Orion-14B-Chat": PPTestSettings.fast(trust_remote_code=True),
    "microsoft/phi-2": PPTestSettings.fast(),
    "microsoft/Phi-3-mini-4k-instruct": PPTestSettings.fast(),
    "microsoft/Phi-3-small-8k-instruct": PPTestSettings.fast(trust_remote_code=True),  # noqa: E501
    # FIXME: https://github.com/vllm-project/vllm/issues/8553
    # "microsoft/Phi-3.5-MoE-instruct": PPTestSettings.fast(trust_remote_code=True),  # noqa: E501
    "adept/persimmon-8b-chat": PPTestSettings.fast(),
    "Qwen/Qwen-7B-Chat": PPTestSettings.fast(trust_remote_code=True),
    "Qwen/Qwen2-beta-7B-Chat": PPTestSettings.fast(),
    "Qwen/Qwen1.5-MoE-A2.7B-Chat": PPTestSettings.fast(),
    "stabilityai/stablelm-3b-4e1t": PPTestSettings.fast(),
    "bigcode/starcoder2-3b": PPTestSettings.fast(),
    "upstage/solar-pro-preview-instruct": PPTestSettings.fast(tp_base=2),
    # FIXME: Cannot load tokenizer in latest transformers version
    # "xverse/XVERSE-7B-Chat": PPTestSettings.fast(trust_remote_code=True),
}

EMBEDDING_MODEL_SETTINGS = {  # type: ignore[var-annotated]
    # [FAST TESTS]
    "intfloat/e5-mistral-7b-instruct": PPTestSettings.fast(),
    "BAAI/bge-multilingual-gemma2": PPTestSettings.fast(),
    "Qwen/Qwen2.5-Math-RM-72B": PPTestSettings.fast(tp_base=4, trust_remote_code=True),  # noqa: E501
}

MULTIMODAL_MODEL_SETTINGS = {
    # [FAST TESTS]
    "Salesforce/blip2-opt-2.7b": PPTestSettings.fast(),
    "facebook/chameleon-7b": PPTestSettings.fast(),
    "adept/fuyu-8b": PPTestSettings.fast(),
    "OpenGVLab/InternVL2-1B": PPTestSettings.fast(trust_remote_code=True),
    "llava-hf/llava-1.5-7b-hf": PPTestSettings.fast(),
    "llava-hf/llava-v1.6-mistral-7b-hf": PPTestSettings.fast(),
    "llava-hf/LLaVA-NeXT-Video-7B-hf": PPTestSettings.fast(),
    "llava-hf/llava-onevision-qwen2-0.5b-ov-hf": PPTestSettings.fast(),
    "openbmb/MiniCPM-Llama3-V-2_5": PPTestSettings.fast(trust_remote_code=True),
    # TODO: Implement PP
    # "meta-llama/Llama-3.2-11B-Vision-Instruct": PPTestSettings.fast(),
    "microsoft/Phi-3-vision-128k-instruct": PPTestSettings.fast(trust_remote_code=True),  # noqa: E501
    "mistralai/Pixtral-12B-2409": PPTestSettings.fast(tp_base=2, tokenizer_mode="mistral"),  # noqa: E501
    "Qwen/Qwen-VL-Chat": PPTestSettings.fast(trust_remote_code=True),
    "Qwen/Qwen2-VL-2B-Instruct": PPTestSettings.fast(),
    "fixie-ai/ultravox-v0_3": PPTestSettings.fast(),
}

CONDITIONAL_GENERATION_MODEL_SETTINGS = {  # type: ignore[var-annotated]
    # [FAST TESTS]
    # TODO: Implement PP
    # "facebook/bart-base": PPTestSettings.fast(),
}
# yapf: enable

# NOTE: You can update this on your local machine to run specific tests
TEST_MODELS = [
    # [LANGUAGE GENERATION]
    "meta-llama/Meta-Llama-3-8B",
    "ibm/PowerLM-3b",
    # [LANGUAGE EMBEDDING]
    "intfloat/e5-mistral-7b-instruct",
    "BAAI/bge-multilingual-gemma2",
    # [MULTIMODAL GENERATION]
    "OpenGVLab/InternVL2-1B",
    "microsoft/Phi-3-vision-128k-instruct",
    "fixie-ai/ultravox-v0_3",
]


def _compare_tp(
    model_name: str,
    parallel_setup: ParallelSetup,
    distributed_backend: str,
    task: TaskOption,
    trust_remote_code: bool,
    tokenizer_mode: Optional[str],
    num_gpus_available: int,
    *,
    method: Literal["generate", "encode"] = "encode",
):
    tp_size, pp_size, eager_mode, chunked_prefill = parallel_setup

    if num_gpus_available < tp_size * pp_size:
        pytest.skip(f"Need at least {tp_size} x {pp_size} GPUs")
    if VLLM_MULTI_NODE and distributed_backend == "mp":
        pytest.skip("Skipping multi-node pipeline parallel test for "
                    "multiprocessing distributed backend")

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

    if (distributed_backend == "ray" and tp_size == 2 and pp_size == 2
            and chunked_prefill):
        # Test Ray ADAG for a subset of the tests
        pp_env = {
            "VLLM_USE_RAY_COMPILED_DAG": "1",
            "VLLM_USE_RAY_SPMD_WORKER": "1",
            "VLLM_USE_RAY_COMPILED_DAG_NCCL_CHANNEL": "1",
        }
        # Temporary. Currently when zeromq + SPMD is used, it does not properly
        # terminate because of aDAG issue.
        common_args.append("--disable-frontend-multiprocessing")
    else:
        pp_env = None

    pp_args = [
        *common_args,
        "--pipeline-parallel-size",
        str(pp_size),
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
                             pp_args,
                             tp_args,
                             pp_env,
                             method=method)
    except Exception:
        if pp_env is None:
            raise
        else:
            # Ray ADAG tests are flaky, so we don't want to fail the test
            logger.exception("Ray ADAG tests failed")


@pytest.mark.parametrize(
    ("model_name", "parallel_setup", "distributed_backend", "task",
     "trust_remote_code", "tokenizer_mode"),
    [
        params for model_name, settings in GENERATION_MODEL_SETTINGS.items()
        for params in settings.iter_params(model_name)
        if model_name in TEST_MODELS
    ],
)
@fork_new_process_for_each_test
def test_tp_language_generation(
    model_name: str,
    parallel_setup: ParallelSetup,
    distributed_backend: str,
    task: TaskOption,
    trust_remote_code: bool,
    tokenizer_mode: Optional[str],
    num_gpus_available,
):
    _compare_tp(model_name,
                parallel_setup,
                distributed_backend,
                task,
                trust_remote_code,
                tokenizer_mode,
                num_gpus_available,
                method="generate")


@pytest.mark.parametrize(
    ("model_name", "parallel_setup", "distributed_backend", "task",
     "trust_remote_code", "tokenizer_mode"),
    [
        params for model_name, settings in EMBEDDING_MODEL_SETTINGS.items()
        for params in settings.iter_params(model_name)
        if model_name in TEST_MODELS
    ],
)
@fork_new_process_for_each_test
def test_tp_language_embedding(
    model_name: str,
    parallel_setup: ParallelSetup,
    distributed_backend: str,
    task: TaskOption,
    trust_remote_code: bool,
    tokenizer_mode: Optional[str],
    num_gpus_available,
):
    _compare_tp(model_name,
                parallel_setup,
                distributed_backend,
                task,
                trust_remote_code,
                tokenizer_mode,
                num_gpus_available,
                method="encode")


@pytest.mark.parametrize(
    ("model_name", "parallel_setup", "distributed_backend", "task",
     "trust_remote_code", "tokenizer_mode"),
    [
        params for model_name, settings in MULTIMODAL_MODEL_SETTINGS.items()
        for params in settings.iter_params(model_name)
        if model_name in TEST_MODELS
    ],
)
@fork_new_process_for_each_test
def test_tp_multimodal_generation(
    model_name: str,
    parallel_setup: ParallelSetup,
    distributed_backend: str,
    task: TaskOption,
    trust_remote_code: bool,
    tokenizer_mode: Optional[str],
    num_gpus_available,
):
    _compare_tp(model_name,
                parallel_setup,
                distributed_backend,
                task,
                trust_remote_code,
                tokenizer_mode,
                num_gpus_available,
                method="generate")

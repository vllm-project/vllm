# SPDX-License-Identifier: Apache-2.0
"""
WARNING: This test runs in both single-node (4 GPUs) and multi-node
 (2 node with 2 GPUs each) modes. If the test only uses 2 GPUs, it is
 important to set the distributed backend to "mp" to avoid Ray scheduling
 all workers in a node other than the head node, which can cause the test
 to fail.
"""
import json
import os
from dataclasses import dataclass
from typing import Literal, NamedTuple, Optional

import pytest

from vllm.config import TaskOption
from vllm.logger import init_logger

from ..models.registry import HF_EXAMPLE_MODELS
from ..utils import compare_two_settings, create_new_process_for_each_test

logger = init_logger("test_pipeline_parallel")

VLLM_MULTI_NODE = os.getenv("VLLM_MULTI_NODE", "0") == "1"


@pytest.fixture(scope="function", autouse=True)
def use_v0_only(monkeypatch):
    """
    For PP, we fall back to V0 by default. This means
    that the TP baseline runs with V1 while the PP engine
    runs with V0. This gives divergent results with dummy
    weights. Once we enable V1 by default for PP, we can
    remove this.
    """
    monkeypatch.setenv('VLLM_USE_V1', '0')


class ParallelSetup(NamedTuple):
    tp_size: int
    pp_size: int
    eager_mode: bool
    chunked_prefill: bool


class PPTestOptions(NamedTuple):
    multi_node_only: bool
    load_format: Optional[str] = None


@dataclass
class PPTestSettings:
    parallel_setups: list[ParallelSetup]
    # NOTE: the length of distributed_backends and
    # vllm_major_versions should be the same, and they
    # are first zipped together to iterate over all
    # test settings.
    distributed_backends: list[str]
    # vllm major version: "0" for V0, "1" for V1
    vllm_major_versions: list[str]
    task: TaskOption
    test_options: PPTestOptions

    def __post_init__(self):
        if len(self.distributed_backends) != len(self.vllm_major_versions):
            raise ValueError(
                f"Length mismatch: distributed_backends "
                f"({len(self.distributed_backends)}) != "
                f"vllm_major_versions ({len(self.vllm_major_versions)})")

    @staticmethod
    def detailed(
        *,
        tp_base: int = 1,
        pp_base: int = 2,
        multi_node_only: bool = False,
        task: TaskOption = "auto",
        load_format: Optional[str] = None,
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
            # only ray is supported for V1
            distributed_backends=["mp", "ray", "ray"],
            vllm_major_versions=["0", "0", "1"],
            task=task,
            test_options=PPTestOptions(multi_node_only=multi_node_only,
                                       load_format=load_format),
        )

    @staticmethod
    def fast(
        *,
        tp_base: int = 1,
        pp_base: int = 2,
        task: TaskOption = "auto",
        multi_node_only: bool = False,
        load_format: Optional[str] = None,
    ):
        return PPTestSettings(
            parallel_setups=[
                ParallelSetup(tp_size=tp_base,
                              pp_size=pp_base,
                              eager_mode=True,
                              chunked_prefill=False),
            ],
            distributed_backends=["mp"],
            vllm_major_versions=["0"],
            task=task,
            test_options=PPTestOptions(multi_node_only=multi_node_only,
                                       load_format=load_format),
        )

    def iter_params(self, model_id: str):
        opts = self.test_options

        for parallel_setup in self.parallel_setups:
            for backend, vllm_major_version in zip(self.distributed_backends,
                                                   self.vllm_major_versions):
                yield (model_id, parallel_setup, backend, vllm_major_version,
                       self.task, opts)


# NOTE: You can adjust tp_base and/or pp_base locally to fit the model in GPU
# The values displayed here are only a rough indicator of the size of the model

# yapf: disable
TEXT_GENERATION_MODELS = {
    # [Decoder-only]
    # Uses Llama
    # "BAAI/AquilaChat-7B": PPTestSettings.fast(),
    "Snowflake/snowflake-arctic-instruct": PPTestSettings.fast(load_format="dummy"),  # noqa: E501
    "baichuan-inc/Baichuan-7B": PPTestSettings.fast(),
    "baichuan-inc/Baichuan2-13B-Chat": PPTestSettings.fast(),
    "bigscience/bloomz-1b1": PPTestSettings.fast(),
    "THUDM/chatglm3-6b": PPTestSettings.fast(),
    "CohereForAI/c4ai-command-r-v01": PPTestSettings.fast(load_format="dummy"),
    "databricks/dbrx-instruct": PPTestSettings.fast(load_format="dummy"),
    "Deci/DeciLM-7B-instruct": PPTestSettings.fast(),
    "deepseek-ai/deepseek-llm-7b-chat": PPTestSettings.fast(),
    "deepseek-ai/DeepSeek-V2-Lite-Chat": PPTestSettings.fast(),
    "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct": PPTestSettings.fast(),
    "tiiuae/falcon-7b": PPTestSettings.fast(),
    "google/gemma-1.1-2b-it": PPTestSettings.fast(),
    "google/gemma-2-9b": PPTestSettings.fast(),
    "gpt2": PPTestSettings.fast(),
    "bigcode/starcoder": PPTestSettings.fast(),
    "EleutherAI/gpt-j-6b": PPTestSettings.fast(),
    "EleutherAI/pythia-1.4b": PPTestSettings.fast(),
    "ibm/PowerLM-3b": PPTestSettings.fast(),
    "ibm/PowerMoE-3b": PPTestSettings.fast(),
    # Uses Llama
    # "internlm/internlm-chat-7b": PPTestSettings.fast(),
    "internlm/internlm2-chat-7b": PPTestSettings.fast(),
    "inceptionai/jais-13b-chat": PPTestSettings.fast(),
    "ai21labs/Jamba-tiny-dev": PPTestSettings.fast(),
    "meta-llama/Llama-3.2-1B-Instruct": PPTestSettings.detailed(),
    # Tests TransformersForCausalLM
    "ArthurZ/Ilama-3.2-1B": PPTestSettings.fast(),
    "openbmb/MiniCPM-2B-sft-bf16": PPTestSettings.fast(),
    "openbmb/MiniCPM3-4B": PPTestSettings.fast(),
    # Uses Llama
    # "mistralai/Mistral-7B-Instruct-v0.1": PPTestSettings.fast(),
    "state-spaces/mamba-130m-hf": PPTestSettings.fast(),
    "mistralai/Mixtral-8x7B-Instruct-v0.1": PPTestSettings.fast(load_format="dummy"),  # noqa: E501
    "mosaicml/mpt-7b": PPTestSettings.fast(),
    "nvidia/Minitron-8B-Base": PPTestSettings.fast(),
    "allenai/OLMo-1B-hf": PPTestSettings.fast(),
    "shanearora/OLMo-7B-1124-hf": PPTestSettings.fast(),
    "allenai/OLMoE-1B-7B-0924-Instruct": PPTestSettings.fast(),
    "facebook/opt-iml-max-1.3b": PPTestSettings.fast(),
    "OrionStarAI/Orion-14B-Chat": PPTestSettings.fast(),
    "adept/persimmon-8b-chat": PPTestSettings.fast(),
    "microsoft/phi-2": PPTestSettings.fast(),
    "microsoft/Phi-3-small-8k-instruct": PPTestSettings.fast(),
    "microsoft/Phi-3.5-MoE-instruct": PPTestSettings.detailed(multi_node_only=True, load_format="dummy"),  # noqa: E501
    "Qwen/Qwen-7B-Chat": PPTestSettings.fast(),
    "Qwen/Qwen2.5-0.5B-Instruct": PPTestSettings.fast(),
    "Qwen/Qwen1.5-MoE-A2.7B-Chat": PPTestSettings.fast(),
    "stabilityai/stablelm-3b-4e1t": PPTestSettings.fast(),
    "bigcode/starcoder2-3b": PPTestSettings.fast(),
    "upstage/solar-pro-preview-instruct": PPTestSettings.fast(load_format="dummy"),  # noqa: E501
    # FIXME: Cannot load tokenizer in latest transformers version.
    # Need to use tokenizer from `meta-llama/Llama-2-7b-chat-hf`
    # "xverse/XVERSE-7B-Chat": PPTestSettings.fast(),
    # [Encoder-only]
    # TODO: Implement PP
    # "facebook/bart-base": PPTestSettings.fast(),
}

EMBEDDING_MODELS = {  # type: ignore[var-annotated]
    # [Text-only]
    "intfloat/e5-mistral-7b-instruct": PPTestSettings.fast(),
    "BAAI/bge-multilingual-gemma2": PPTestSettings.fast(),
    "Qwen/Qwen2.5-Math-RM-72B": PPTestSettings.fast(load_format="dummy"),
}

MULTIMODAL_MODELS = {
    # [Decoder-only]
    "Salesforce/blip2-opt-6.7b": PPTestSettings.fast(),
    "facebook/chameleon-7b": PPTestSettings.fast(),
    "adept/fuyu-8b": PPTestSettings.fast(),
    "THUDM/glm-4v-9b": PPTestSettings.fast(),
    "OpenGVLab/InternVL2-1B": PPTestSettings.fast(),
    "llava-hf/llava-1.5-7b-hf": PPTestSettings.fast(),
    "llava-hf/llava-v1.6-mistral-7b-hf": PPTestSettings.fast(),
    "llava-hf/LLaVA-NeXT-Video-7B-hf": PPTestSettings.fast(),
    "llava-hf/llava-onevision-qwen2-0.5b-ov-hf": PPTestSettings.fast(),
    "openbmb/MiniCPM-Llama3-V-2_5": PPTestSettings.fast(),
    "allenai/Molmo-7B-D-0924": PPTestSettings.fast(),
    "microsoft/Phi-3.5-vision-instruct": PPTestSettings.fast(),
    "mistralai/Pixtral-12B-2409": PPTestSettings.fast(load_format="dummy"),
    "Qwen/Qwen-VL-Chat": PPTestSettings.fast(),
    "Qwen/Qwen2-Audio-7B-Instruct": PPTestSettings.fast(),
    "Qwen/Qwen2-VL-2B-Instruct": PPTestSettings.fast(),
    "fixie-ai/ultravox-v0_5-llama-3_2-1b": PPTestSettings.fast(),
    # [Encoder-decoder]
    # TODO: Implement PP
    # "meta-llama/Llama-3.2-11B-Vision-Instruct": PPTestSettings.fast(),
}
# yapf: enable

# NOTE: You can update this on your local machine to run specific tests
TEST_MODELS = [
    # [LANGUAGE GENERATION]
    "microsoft/Phi-3.5-MoE-instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    "ArthurZ/Ilama-3.2-1B",
    "ibm/PowerLM-3b",
    # [LANGUAGE EMBEDDING]
    "intfloat/e5-mistral-7b-instruct",
    "BAAI/bge-multilingual-gemma2",
    # [MULTIMODAL GENERATION]
    "OpenGVLab/InternVL2-1B",
    "microsoft/Phi-3.5-vision-instruct",
    "fixie-ai/ultravox-v0_5-llama-3_2-1b",
    # [LANGUAGE GENERATION - HYBRID ARCH]
    "ai21labs/Jamba-tiny-dev",
]


def _compare_tp(
    model_id: str,
    parallel_setup: ParallelSetup,
    distributed_backend: str,
    vllm_major_version: str,
    task: TaskOption,
    test_options: PPTestOptions,
    num_gpus_available: int,
    *,
    method: Literal["generate", "encode"],
    is_multimodal: bool,
):
    (
        tp_size,
        pp_size,
        eager_mode,
        chunked_prefill,
    ) = parallel_setup

    multi_node_only, load_format = test_options

    model_info = HF_EXAMPLE_MODELS.find_hf_info(model_id)
    model_info.check_transformers_version(on_fail="skip")

    trust_remote_code = model_info.trust_remote_code
    tokenizer_mode = model_info.tokenizer_mode
    hf_overrides = model_info.hf_overrides

    if load_format == "dummy":
        # Avoid OOM
        text_overrides = {
            "num_hidden_layers": 4,
            "hidden_size": 512,
            "intermediate_size": 800,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
        }

        if is_multimodal:
            hf_overrides.update({"text_config": text_overrides})
        else:
            hf_overrides.update(text_overrides)
    else:
        model_info.check_available_online(on_fail="skip")

    if num_gpus_available < tp_size * pp_size:
        pytest.skip(f"Need at least {tp_size} x {pp_size} GPUs")
    if VLLM_MULTI_NODE and distributed_backend == "mp":
        pytest.skip("Skipping multi-node pipeline parallel test for "
                    "multiprocessing distributed backend")
    if multi_node_only and not VLLM_MULTI_NODE:
        pytest.skip("Not in multi-node setting")

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
        common_args.extend(["--hf-overrides", json.dumps(hf_overrides)])

    specific_case = tp_size == 2 and pp_size == 2 and chunked_prefill
    if distributed_backend == "ray" and (vllm_major_version == "1"
                                         or specific_case):
        # For V1, test Ray Compiled Graph for all the tests
        # For V0, test Ray Compiled Graph for a subset of the tests
        pp_env = {
            "VLLM_USE_V1": vllm_major_version,
            "VLLM_USE_RAY_COMPILED_DAG": "1",
            "VLLM_USE_RAY_SPMD_WORKER": "1",
            "VLLM_USE_RAY_COMPILED_DAG_NCCL_CHANNEL": "1",
        }
        # Temporary. Currently when zeromq + SPMD is used, it does not properly
        # terminate because of a Ray Compiled Graph issue.
        common_args.append("--disable-frontend-multiprocessing")
    else:
        pp_env = None

    tp_env = {
        "VLLM_USE_V1": vllm_major_version,
    }

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
        compare_two_settings(model_id,
                             pp_args,
                             tp_args,
                             pp_env,
                             tp_env,
                             method=method)
    except Exception:
        testing_ray_compiled_graph = pp_env is not None
        if testing_ray_compiled_graph and vllm_major_version == "0":
            # Ray Compiled Graph tests are flaky for V0,
            # so we don't want to fail the test
            logger.exception("Ray Compiled Graph tests failed")
        else:
            raise


@pytest.mark.parametrize(
    ("model_id", "parallel_setup", "distributed_backend", "vllm_major_version",
     "task", "test_options"),
    [
        params for model_id, settings in TEXT_GENERATION_MODELS.items()
        for params in settings.iter_params(model_id) if model_id in TEST_MODELS
    ],
)
@create_new_process_for_each_test()
def test_tp_language_generation(
    model_id: str,
    parallel_setup: ParallelSetup,
    distributed_backend: str,
    vllm_major_version: str,
    task: TaskOption,
    test_options: PPTestOptions,
    num_gpus_available,
):
    _compare_tp(model_id,
                parallel_setup,
                distributed_backend,
                vllm_major_version,
                task,
                test_options,
                num_gpus_available,
                method="generate",
                is_multimodal=False)


@pytest.mark.parametrize(
    ("model_id", "parallel_setup", "distributed_backend", "vllm_major_version",
     "task", "test_options"),
    [
        params for model_id, settings in EMBEDDING_MODELS.items()
        for params in settings.iter_params(model_id) if model_id in TEST_MODELS
    ],
)
@create_new_process_for_each_test()
def test_tp_language_embedding(
    model_id: str,
    parallel_setup: ParallelSetup,
    distributed_backend: str,
    vllm_major_version: str,
    task: TaskOption,
    test_options: PPTestOptions,
    num_gpus_available,
):
    _compare_tp(model_id,
                parallel_setup,
                distributed_backend,
                vllm_major_version,
                task,
                test_options,
                num_gpus_available,
                method="encode",
                is_multimodal=False)


@pytest.mark.parametrize(
    ("model_id", "parallel_setup", "distributed_backend", "vllm_major_version",
     "task", "test_options"),
    [
        params for model_id, settings in MULTIMODAL_MODELS.items()
        for params in settings.iter_params(model_id) if model_id in TEST_MODELS
    ],
)
@create_new_process_for_each_test()
def test_tp_multimodal_generation(
    model_id: str,
    parallel_setup: ParallelSetup,
    distributed_backend: str,
    vllm_major_version: str,
    task: TaskOption,
    test_options: PPTestOptions,
    num_gpus_available,
):
    _compare_tp(model_id,
                parallel_setup,
                distributed_backend,
                vllm_major_version,
                task,
                test_options,
                num_gpus_available,
                method="generate",
                is_multimodal=True)

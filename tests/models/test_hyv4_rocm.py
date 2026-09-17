# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import pytest

from tests.utils import multi_gpu_marks
from vllm import SamplingParams
from vllm.config import CompilationConfig
from vllm.config.compilation import CompilationMode, CUDAGraphMode
from vllm.platforms import current_platform
from vllm.v1.metrics.reader import Counter

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="ROCm-specific tests"
)


def _small_hyv4_config(config):
    # Apply to both target and MTP draft while retaining MLA/indexer dimensions.
    config.update(
        {
            "hidden_size": 512,
            "num_hidden_layers": 2,
            "num_attention_heads": 64,
            "q_lora_rank": 128,
            "intermediate_size": 512,
            "moe_intermediate_size": 256,
            "n_routed_experts": 4,
            "num_experts_per_tok": 2,
            "n_shared_experts": 1,
            "enable_ihc": True,
            "index_n_heads": 32,
            "index_topk": 2048,
            "layer_types": ["deepseek_sparse_attention"] * 2,
            "indexer_types": ["full", "shared"],
            "num_nextn_predict_layers": 1,
            "vocab_size": 256,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "pad_token_id": 0,
        }
    )
    return config


def _num_captured_graphs(_worker):
    from vllm.compilation.counter import compilation_counter

    return compilation_counter.num_cudagraph_captured


@pytest.mark.parametrize(
    "tp,enforce_eager,mtp,dtype,kv_cache_dtype",
    [
        pytest.param(1, True, False, "bfloat16", "auto", id="eager-tp1"),
        pytest.param(1, True, False, "float16", "auto", id="eager-fp16"),
        pytest.param(1, True, False, "bfloat16", "fp8", id="eager-fp8-kv"),
        pytest.param(
            2,
            False,
            False,
            "bfloat16",
            "auto",
            id="graphs-tp2",
            marks=multi_gpu_marks(num_gpus=2),
        ),
        pytest.param(1, True, True, "bfloat16", "auto", id="eager-mtp"),
    ],
)
def test_hyv4_dummy_generation(
    vllm_runner,
    enable_pickle,
    tp: int,
    enforce_eager: bool,
    mtp: bool,
    dtype: str,
    kv_cache_dtype: str,
) -> None:
    """Exercise prefill, decode, and batch consistency with dummy HY weights."""
    with vllm_runner(
        "tencent/Hy4-preview",
        trust_remote_code=True,
        skip_tokenizer_init=True,
        load_format="dummy",
        disable_log_stats=not mtp,
        dtype=dtype,
        kv_cache_dtype=kv_cache_dtype,
        hf_overrides=_small_hyv4_config,
        max_model_len=256,
        max_num_batched_tokens=128,
        max_num_seqs=8,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        kv_cache_memory_bytes=128 * 1024 * 1024,
        enforce_eager=enforce_eager,
        tensor_parallel_size=tp,
        speculative_config={"method": "mtp", "num_speculative_tokens": 2}
        if mtp
        else None,
        seed=17,
        # Exercise supported decode graphs independently of default graph policy.
        compilation_config=CompilationConfig(
            mode=CompilationMode.NONE,
            cudagraph_mode=CUDAGraphMode.NONE
            if enforce_eager
            else CUDAGraphMode.FULL_DECODE_ONLY,
            cudagraph_capture_sizes=[] if enforce_eager else [1, 2, 4],
        ),
    ) as model:
        if not enforce_eager:
            assert model.collective_rpc(_num_captured_graphs) == [3] * tp
        prompts = [
            {"prompt_token_ids": [11, 12, 13, 14]},
            {"prompt_token_ids": list(range(20, 84))},
        ]
        params = SamplingParams(
            temperature=0, max_tokens=4, ignore_eos=True, logprobs=1
        )
        outputs = model.llm.generate(prompts, params)
        assert len(outputs) == len(prompts)
        for prompt, output in zip(prompts, outputs):
            completion = output.outputs[0]
            assert len(completion.token_ids) == 4
            assert completion.cumulative_logprob is not None
            assert math.isfinite(completion.cumulative_logprob)
            single = model.llm.generate([prompt], params)[0].outputs[0]
            assert single.token_ids == completion.token_ids

        if mtp:
            assert any(
                isinstance(metric, Counter)
                and metric.name == "vllm:spec_decode_num_drafts"
                and metric.value > 0
                for metric in model.llm.get_metrics()
            ), "MTP drafter did not run"

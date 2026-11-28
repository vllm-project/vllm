# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

import pytest

from tests.evals.gsm8k.gsm8k_eval import evaluate_gsm8k_offline
from tests.utils import single_gpu_only
from vllm.config import CompilationConfig

from ...utils import compute_acceptance_len
from ..utils import run_acceptance_length_eval


@dataclass(frozen=True, slots=True)
class DFlashCorrectnessConfig:
    model: str
    draft_model: str
    expected_accuracy: float
    expected_acceptance_len: float
    num_speculative_tokens: int = 16
    max_model_len: int = 4096
    max_num_seqs: int = 128
    num_questions: int = 1319
    use_chat_completions: bool = False
    enforce_eager: bool = False
    disable_flashinfer_sampler: bool = False
    language_model_only: bool = False
    chat_template_kwargs: dict[str, object] | None = None


QWEN3_DFLASH = DFlashCorrectnessConfig(
    model="Qwen/Qwen3-8B",
    draft_model="z-lab/Qwen3-8B-DFlash-b16",
    expected_accuracy=0.8,
    expected_acceptance_len=3.5,
)

QWEN3_8_DFLASH2_NVFP4 = DFlashCorrectnessConfig(
    model="nvidia/Qwen3.8-27B-NVFP4",
    draft_model="z-lab/Qwen3.8-27B-DFlash2",
    expected_accuracy=0.90,
    expected_acceptance_len=6.14 * 0.95,
    num_speculative_tokens=7,
    max_num_seqs=32,
    use_chat_completions=True,
    language_model_only=True,
    chat_template_kwargs={"enable_thinking": False},
)

# Anchor-first DFlash2 draft (`sample_from_anchor: true`): K queries per request.
QWEN3_8_DFLASH2_ANCHOR = DFlashCorrectnessConfig(
    model="Qwen/Qwen3.8-27B",
    draft_model="DaoCloud/Qwen3.8-27B-DFlash2-Exp",
    expected_accuracy=0.90,
    expected_acceptance_len=6.03 * 0.95,
    num_speculative_tokens=7,
    max_num_seqs=32,
    use_chat_completions=True,
    language_model_only=True,
    chat_template_kwargs={"enable_thinking": False},
)

QWEN3_8_DSPINE = DFlashCorrectnessConfig(
    model="Qwen/Qwen3.8-27B",
    draft_model="tencent/Qwen3.8-27B-DSpine-b16",
    expected_accuracy=0.90,
    expected_acceptance_len=7.87 * 0.95,
    num_speculative_tokens=15,
    max_num_seqs=32,
    use_chat_completions=True,
    language_model_only=True,
    chat_template_kwargs={"enable_thinking": False},
)

QWEN3_5_4B_DFLASH2 = DFlashCorrectnessConfig(
    model="Qwen/Qwen3.5-4B",
    draft_model="vllm/Qwen3.5-4B-DFlash2",
    expected_accuracy=0.8,
    expected_acceptance_len=4.55 * 0.9,
    num_speculative_tokens=15,
    max_num_seqs=32,
    use_chat_completions=True,
    language_model_only=True,
    chat_template_kwargs={"enable_thinking": False},
)

LAGUNA_DFLASH_NVFP4 = DFlashCorrectnessConfig(
    model="poolside/Laguna-XS-2.1-NVFP4",
    draft_model="poolside/Laguna-XS-2.1-DFlash-NVFP4",
    expected_accuracy=0.7,  # Standard GSM8K sanity floor.
    expected_acceptance_len=3.05 * 0.9,
    num_speculative_tokens=15,
    max_model_len=8192,
    max_num_seqs=32,
    num_questions=200,
    use_chat_completions=True,
    enforce_eager=True,
    disable_flashinfer_sampler=True,
)


def test_dflash_reference_acceptance_lengths(vllm_runner):
    run_acceptance_length_eval(
        vllm_runner,
        spec_config={
            "model": QWEN3_DFLASH.model,
            "speculative_config": {
                "method": "dflash",
                "model": QWEN3_DFLASH.draft_model,
                "num_speculative_tokens": QWEN3_DFLASH.num_speculative_tokens,
                "max_model_len": 32768,
            },
            "max_model_len": 32768,
            "max_num_seqs": 128,
            "enforce_eager": False,
            "disable_log_stats": False,
        },
        # Table 1 in https://arxiv.org/pdf/2602.06036.
        expected_acceptance_lengths={
            "mt-bench": 4.24,
            "humaneval": 6.50,
            "gsm8k": 6.54 * 0.975,
        },
        chat_template_kwargs={"enable_thinking": False},
    )


@single_gpu_only
def test_dflash2_mixed_kv_pages_do_not_inflate_after_allocation(
    monkeypatch: pytest.MonkeyPatch, vllm_runner
):
    """BF16 draft KV must fit the hybrid page planned for FP8 target KV."""
    # The geometry callback stays local; no callable serialization is needed.
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    def allocated_geometry(worker):
        from vllm.config.vllm import get_layers_from_vllm_config
        from vllm.model_executor.layers.attention import Attention
        from vllm.v1.kv_cache_interface import AttentionSpec

        config = worker.vllm_config
        cache = worker.model_runner.kv_cache_config
        layers = get_layers_from_vllm_config(config, Attention)
        dtypes: dict[str, set[str]] = {"target": set(), "draft": set()}
        pages = set()
        for group in cache.kv_cache_groups:
            for name, spec in group.kv_cache_specs.items():
                pages.add(spec.page_size_bytes)
                if isinstance(spec, AttentionSpec):
                    layer = layers[name]
                    role = "draft" if layer.is_draft_layer else "target"
                    dtypes[role].add(layer.kv_cache_dtype)
        views = worker.model_runner.kv_caches_by_layer
        for tensor in cache.kv_cache_tensors:
            view = views[tensor.layers[0]]
            assert view.is_cuda
            assert view.untyped_storage().nbytes() == tensor.size
        return (
            config.cache_config.mamba_page_size_padded,
            pages,
            dtypes,
        )

    config = QWEN3_5_4B_DFLASH2
    with vllm_runner(
        config.model,
        dtype="bfloat16",
        block_size=None,
        distributed_executor_backend="uni",
        kv_cache_dtype="fp8_e4m3",
        mamba_cache_dtype="bfloat16",
        mamba_ssm_cache_dtype="float32",
        kv_cache_memory_bytes=8 * 1024**3,
        max_model_len=4096,
        max_num_seqs=1,
        max_num_batched_tokens=512,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        enforce_eager=True,
        language_model_only=True,
        speculative_config={
            "method": "dflash",
            "model": config.draft_model,
            "num_speculative_tokens": config.num_speculative_tokens,
            "kv_cache_dtype": "bfloat16",
        },
    ) as runner:
        llm = runner.llm
        padded_page, pages, dtypes = llm.collective_rpc(allocated_geometry)[0]
        print(
            f"Mixed KV geometry: planned_page={padded_page}, allocated_pages={pages}, "
            f"dtypes={dtypes}"
        )
        assert dtypes == {"target": {"fp8_e4m3"}, "draft": {"bfloat16"}}
        assert pages == {2621440}, "Draft KV inflated the shared physical page"
        assert padded_page == 2621440


@single_gpu_only
@pytest.mark.parametrize(
    "config",
    [
        pytest.param(QWEN3_DFLASH, id="qwen3"),
        pytest.param(LAGUNA_DFLASH_NVFP4, id="laguna-nvfp4"),
        pytest.param(QWEN3_8_DFLASH2_NVFP4, id="qwen3.8-dflash2-nvfp4"),
        pytest.param(QWEN3_8_DFLASH2_ANCHOR, id="qwen3.8-dflash2-anchor"),
        pytest.param(QWEN3_8_DSPINE, id="qwen3.8-dspine"),
        pytest.param(QWEN3_5_4B_DFLASH2, id="qwen3.5-4b-dflash2"),
    ],
)
def test_dflash_correctness(
    monkeypatch: pytest.MonkeyPatch,
    config: DFlashCorrectnessConfig,
    vllm_runner,
):
    """Guard GSM8K accuracy and batched acceptance length for DFlash models."""
    if config.disable_flashinfer_sampler:
        monkeypatch.setenv("VLLM_USE_FLASHINFER_SAMPLER", "0")

    with vllm_runner(
        config.model,
        block_size=None,
        speculative_config={
            "method": "dflash",
            "model": config.draft_model,
            "num_speculative_tokens": config.num_speculative_tokens,
            "max_model_len": config.max_model_len,
        },
        max_model_len=config.max_model_len,
        max_num_seqs=config.max_num_seqs,
        enforce_eager=config.enforce_eager,
        disable_log_stats=False,
        enable_chunked_prefill=None,
        compilation_config=CompilationConfig(),
        language_model_only=config.language_model_only,
    ) as spec_runner:
        spec_llm = spec_runner.llm
        results = evaluate_gsm8k_offline(
            spec_llm,
            num_questions=config.num_questions,
            use_chat_completions=config.use_chat_completions,
            chat_template_kwargs=config.chat_template_kwargs,
        )
        accuracy = results["accuracy"]
        acceptance_len = compute_acceptance_len(spec_llm.get_metrics())
        context = f"DFlash target={config.model}, draft={config.draft_model}"
        print(
            f"{context}: GSM8K accuracy={accuracy:.3f}, "
            f"acceptance_len={acceptance_len:.2f}"
        )

        assert accuracy >= config.expected_accuracy, (
            f"{context}: GSM8K accuracy {accuracy:.3f} is below "
            f"{config.expected_accuracy:.3f}; acceptance_len={acceptance_len:.3f}"
        )
        assert acceptance_len >= config.expected_acceptance_len, (
            f"{context}: acceptance_len {acceptance_len:.3f} is below "
            f"{config.expected_acceptance_len:.3f}; accuracy={accuracy:.3f}"
        )

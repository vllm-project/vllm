# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

import pytest
import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import Qwen3Config, Qwen3ForCausalLM, TokenizersBackend

from tests.evals.gsm8k.gsm8k_eval import evaluate_gsm8k_offline
from tests.utils import single_gpu_only
from vllm import SamplingParams
from vllm.config import CompilationConfig, CUDAGraphMode

from ...utils import compute_acceptance_len
from ..utils import run_acceptance_length_eval


@dataclass(frozen=True)
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

QWEN3_5_4B_DFLASH2 = DFlashCorrectnessConfig(
    model="Qwen/Qwen3.5-4B",
    draft_model="vllm-project/Qwen3.5-4B-DFlash2",
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


def _draft_causality_metadata(worker):
    from vllm.v1.worker.gpu.attn_utils import build_attn_metadata

    spec = worker.model_runner.speculator
    names = spec.model.get_draft_kv_cache_layer_names()
    causal = getattr(spec, "_layer_causal", None)
    if causal is None:
        causal = spec._group_causal
    device = worker.device
    query_start = torch.tensor([0, 4], dtype=torch.int32)
    seq_lens = torch.tensor([4], dtype=torch.int32)
    num_groups = len(spec.kv_cache_config.kv_cache_groups)
    results = []
    for capture in (False, True):
        metadata = build_attn_metadata(
            attn_groups=spec.attn_groups,
            num_reqs=1,
            num_tokens=4,
            query_start_loc_gpu=query_start.to(device),
            query_start_loc_cpu=query_start,
            max_query_len=4,
            seq_lens=seq_lens.to(device),
            seq_lens_cpu_upper_bound=seq_lens,
            max_seq_len=4,
            block_tables=[
                torch.zeros((1, 2), dtype=torch.int32, device=device)
                for _ in range(num_groups)
            ],
            slot_mappings=torch.zeros(
                (num_groups, 4), dtype=torch.int64, device=device
            ),
            kv_cache_config=spec.kv_cache_config,
            causal=causal,
            for_cudagraph_capture=capture,
        )
        results.append([metadata[name].causal for name in names])
    groups = [[group.layer_names for group in groups] for groups in spec.attn_groups]
    return names, results, groups, len(spec.query_cudagraph_manager.graphs)


@single_gpu_only
@pytest.mark.parametrize("enforce_eager", [True, False])
def test_dflash_merged_cache_preserves_layer_masks(
    tmp_path, monkeypatch, vllm_runner, enforce_eager
):
    """Mixed causal masks survive merged cache storage and graph capture."""
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    target, draft = tmp_path / "target", tmp_path / "draft"
    target.mkdir()
    draft.mkdir()
    vocab = {f"token{i}": i for i in range(128)}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="token0"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    TokenizersBackend(
        tokenizer_object=tokenizer,
        unk_token="token0",
        bos_token="token1",
        eos_token="token2",
        pad_token="token3",
    ).save_pretrained(target)
    config = Qwen3Config(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=256,
        architectures=["Qwen3ForCausalLM"],
        tie_word_embeddings=True,
    )
    torch.manual_seed(17)
    target_model = Qwen3ForCausalLM(config)
    target_model.save_pretrained(target)
    weights = {
        name.removeprefix("model."): value.detach().contiguous().clone()
        for name, value in target_model.state_dict().items()
        if name.startswith("model.layers.") or name == "model.norm.weight"
    }
    weights["fc.weight"] = torch.randn(128, 512) * 0.02
    weights["hidden_norm.weight"] = torch.ones(128)
    save_file(weights, str(draft / "model.safetensors"), metadata={"format": "pt"})
    config.architectures = ["DFlashDraftModel"]
    config.layer_types = ["sliding_attention"] * 3 + ["full_attention"]
    config.use_sliding_window = True
    config.sliding_window = 32
    config.dflash_config = {"mask_token_id": 4, "target_layer_ids": [0, 1, 2, 3]}
    config.is_causal = None
    config.save_pretrained(draft)

    with vllm_runner(
        str(target),
        dtype="bfloat16",
        max_model_len=128,
        max_num_seqs=2,
        kv_cache_memory_bytes=128 * 1024 * 1024,
        enforce_eager=enforce_eager,
        compilation_config=CompilationConfig(
            cudagraph_mode=CUDAGraphMode.NONE if enforce_eager else CUDAGraphMode.FULL,
            cudagraph_capture_sizes=[4, 8],
        ),
        disable_hybrid_kv_cache_manager=True,
        kernel_config={"enable_flashinfer_autotune": False},
        speculative_config={
            "method": "dflash",
            "model": str(draft),
            "num_speculative_tokens": 3,
        },
    ) as runner:
        llm = runner.get_llm()
        names, masks, groups, captured_graphs = llm.collective_rpc(
            _draft_causality_metadata
        )[0]
        assert len(groups) == 1
        assert set(names) == {name for group in groups[0] for name in group}
        assert masks == [[True, True, True, False]] * 2
        if not enforce_eager:
            assert captured_graphs > 0
        outputs = llm.generate(
            "token10 token11 token12", SamplingParams(temperature=0, max_tokens=8)
        )
        assert len(outputs[0].outputs[0].token_ids) > 0


def test_dflash_reference_acceptance_lengths(vllm_runner):
    run_acceptance_length_eval(
        vllm_runner,
        spec_config={
            "model": QWEN3_DFLASH.model,
            "trust_remote_code": True,
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
@pytest.mark.parametrize(
    "config",
    [
        pytest.param(QWEN3_DFLASH, id="qwen3"),
        pytest.param(LAGUNA_DFLASH_NVFP4, id="laguna-nvfp4"),
        pytest.param(QWEN3_8_DFLASH2_NVFP4, id="qwen3.8-dflash2-nvfp4"),
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
        trust_remote_code=True,
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

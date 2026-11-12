# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from tests.utils import single_gpu_only
from vllm import SamplingParams, TokensPrompt
from vllm.config import CompilationMode, CUDAGraphMode
from vllm.platforms import current_platform
from vllm.v1.metrics.reader import Vector

from ...utils import get_spec_decode_metric_value
from .._correctness import check_mtp_correctness

TARGET_TOKEN, MODULE_1_TOKEN = 1, 2


def _write_two_module_mtp_checkpoint(path: Path) -> Path:
    """Tiny DeepSeek-V3 (MLA, dense MLP) with two MTP modules whose greedy
    tokens are fixed by construction, with large logit margins.

    `o_proj` and `down_proj` are zero, so attention and MLP add nothing and
    every hidden state is a scaled copy of its input. All token embeddings are
    `u` (ones in the first half), and `lm_head` maps `u` to TARGET_TOKEN and
    `v` (ones in the second half) to MODULE_1_TOKEN, all other rows zero. The
    target and MTP module 0 (`eh_proj` passes `u` through) emit TARGET_TOKEN;
    module 1 (`eh_proj` maps `u` to `v`) drafts MODULE_1_TOKEN. The modules
    differ inside the compiled drafter forward, not in `compute_logits`, so
    step 2 drafts are accepted iff module 0 runs.
    """
    hidden, vocab, inter, layers, heads = 512, 1024, 1024, 2, 16
    nope, rope, v_dim, kv_lora = 128, 64, 128, 512
    config = {
        "architectures": ["DeepseekV3ForCausalLM"],
        "model_type": "deepseek_v3",
        "hidden_size": hidden,
        "intermediate_size": inter,
        "num_hidden_layers": layers,
        "num_nextn_predict_layers": 2,
        # All layers, including the MTP blocks, use a dense MLP.
        "first_k_dense_replace": 1000,
        "n_routed_experts": 8,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 256,
        "n_group": 1,
        "topk_group": 1,
        "num_attention_heads": heads,
        "num_key_value_heads": heads,
        "q_lora_rank": None,
        "kv_lora_rank": kv_lora,
        "qk_nope_head_dim": nope,
        "qk_rope_head_dim": rope,
        "v_head_dim": v_dim,
        "vocab_size": vocab,
        "max_position_embeddings": 4096,
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
    }
    (path / "config.json").write_text(json.dumps(config))

    generator = torch.Generator().manual_seed(0)

    def randn(*shape: int) -> torch.Tensor:
        return (0.02 * torch.randn(shape, generator=generator)).bfloat16()

    def zeros(*shape: int) -> torch.Tensor:
        return torch.zeros(shape, dtype=torch.bfloat16)

    def ones(*shape: int) -> torch.Tensor:
        return torch.ones(shape, dtype=torch.bfloat16)

    half = hidden // 2
    u, v = zeros(hidden), zeros(hidden)
    u[:half], v[half:] = 1, 1
    lm_head = zeros(vocab, hidden)
    lm_head[TARGET_TOKEN], lm_head[MODULE_1_TOKEN] = u, v
    identity = torch.eye(hidden, dtype=torch.bfloat16)
    swap_halves = identity.roll(half, dims=0)
    # eh_proj sees [enorm(embedding), hnorm(previous hidden)], both along u.
    eh_proj = [identity.repeat(1, 2), swap_halves.repeat(1, 2)]

    weights = {
        "model.embed_tokens.weight": u.repeat(vocab, 1),
        "model.norm.weight": ones(hidden),
        "lm_head.weight": lm_head,
    }
    for i in range(layers + 2):
        prefix = f"model.layers.{i}."
        weights |= {
            prefix + "input_layernorm.weight": ones(hidden),
            prefix + "post_attention_layernorm.weight": ones(hidden),
            prefix + "self_attn.q_proj.weight": randn(heads * (nope + rope), hidden),
            prefix + "self_attn.kv_a_proj_with_mqa.weight": randn(
                kv_lora + rope, hidden
            ),
            prefix + "self_attn.kv_a_layernorm.weight": ones(kv_lora),
            prefix + "self_attn.kv_b_proj.weight": randn(
                heads * (nope + v_dim), kv_lora
            ),
            prefix + "self_attn.o_proj.weight": zeros(hidden, heads * v_dim),
            prefix + "mlp.gate_proj.weight": randn(inter, hidden),
            prefix + "mlp.up_proj.weight": randn(inter, hidden),
            prefix + "mlp.down_proj.weight": zeros(hidden, inter),
        }
        if i >= layers:
            weights |= {
                prefix + "enorm.weight": ones(hidden),
                prefix + "hnorm.weight": ones(hidden),
                prefix + "eh_proj.weight": eh_proj[i - layers].clone(),
                prefix + "shared_head.norm.weight": ones(hidden),
                prefix + "shared_head.head.weight": lm_head.clone(),
            }
    save_file(weights, path / "model.safetensors")
    return path


@single_gpu_only
def test_mtp_draft_steps_use_their_own_module(tmp_path: Path, vllm_runner):
    """With two MTP modules, compilation and full CUDA graphs enabled, draft
    step 2 must run module 1. A compiled drafter would reuse module 0."""
    model = _write_two_module_mtp_checkpoint(tmp_path)
    with vllm_runner(
        str(model),
        skip_tokenizer_init=True,
        speculative_config={"method": "mtp", "num_speculative_tokens": 2},
        compilation_config={
            "mode": CompilationMode.VLLM_COMPILE,
            "cudagraph_mode": CUDAGraphMode.FULL_AND_PIECEWISE,
        },
        disable_log_stats=False,
    ) as runner:
        compilation_config = runner.llm.llm_engine.vllm_config.compilation_config
        assert compilation_config.mode == CompilationMode.VLLM_COMPILE
        assert compilation_config.cudagraph_mode.has_full_cudagraphs()
        prompts = [
            TokensPrompt(prompt_token_ids=list(range(3 + i, 35 + i))) for i in range(8)
        ]
        outputs = runner.llm.generate(
            prompts, SamplingParams(temperature=0, max_tokens=64, ignore_eos=True)
        )
        metrics = runner.llm.get_metrics()

    assert all(set(o.outputs[0].token_ids) == {TARGET_TOKEN} for o in outputs)
    num_drafts = get_spec_decode_metric_value(metrics, "vllm:spec_decode_num_drafts")
    accepted_per_pos = next(
        m.values
        for m in metrics
        if isinstance(m, Vector)
        and m.name == "vllm:spec_decode_num_accepted_tokens_per_pos"
    )
    assert num_drafts > 0
    assert len(accepted_per_pos) == 2
    # Every counted draft has a first token; drafts trimmed near max_tokens may
    # lack a second, which only lowers step 2.
    step_1, step_2 = (accepted / num_drafts for accepted in accepted_per_pos)
    assert step_1 == 1.0, f"module 0 acceptance {step_1:.3f}"
    assert step_2 == 0.0, f"module 1 acceptance {step_2:.3f}: module 0 was reused"


@pytest.mark.parametrize(
    ["model_setup", "mm_enabled", "expected_accuracy_threshold"],
    [
        pytest.param(
            ("mtp", "vllm-project/DeepSeek-V3-4layers-MTP-FP8", 1, None),
            False,
            0.0,
            marks=pytest.mark.skipif(
                current_platform.is_device_capability_family(100),
                reason="DeepSeek MTP: TRTLLM MoE top_k check fails on Blackwell",
            ),
        ),  # dummy model
    ],
    ids=["deepseek"],
)
@single_gpu_only
def test_mtp_correctness(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
    model_setup: tuple[str, str, int, str | None],
    mm_enabled: bool,
    expected_accuracy_threshold: float,
    vllm_runner,
):
    check_mtp_correctness(
        monkeypatch,
        sampling_config,
        model_setup,
        mm_enabled,
        expected_accuracy_threshold,
        vllm_runner,
    )

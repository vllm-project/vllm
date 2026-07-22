# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Integration tests for Ultravox tower/connector LoRA (#31479).

The adapters are built procedurally (fixed seed) instead of loading a trained
adapter: released Ultravox adapters keep the audio stack frozen
(lora_B == 0), so they cannot distinguish a working tower/connector LoRA path
from a silent no-op.

Correctness of the applied delta is checked against a reference checkpoint that
folds the adapter into the base weights (`W + (alpha / r) * B @ A`). HF's custom
Ultravox `forward` and `save_pretrained` are both incompatible with Transformers
v5 (the HF-vs-vLLM comparison in `test_common.py` is skipped for the same
reason), so the reference is built by merging into the checkpoint tensors
directly rather than through an HF forward.
"""

import json
import os
import shutil

import pytest
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoTokenizer

from vllm.lora.request import LoRARequest

from ..conftest import AudioTestAssets, VllmRunner
from ..models.registry import HF_EXAMPLE_MODELS
from ..models.utils import check_logprobs_close

MODEL_NAME = "fixie-ai/ultravox-v0_5-llama-3_2-1b"

TOWER_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "out_proj"]
CONNECTOR_TARGET_MODULES = ["linear_1", "linear_2"]

RANK = 8
LORA_ALPHA = 16


def _adapter_module_shapes(
    config, target_modules: list[str]
) -> dict[str, tuple[int, int]]:
    """(dim_out, dim_in) of each base module the adapter targets."""
    d_model = config.audio_config.d_model
    num_layers = config.audio_config.encoder_layers
    stacked_dim = config.audio_config.hidden_size * config.stack_factor
    linear_2_in = (
        config.hidden_size // 2
        if config.projector_act == "swiglu"
        else config.hidden_size
    )
    connector_shapes = {
        "multi_modal_projector.linear_1": (config.hidden_size, stacked_dim),
        "multi_modal_projector.linear_2": (config.text_config.hidden_size, linear_2_in),
    }

    shapes: dict[str, tuple[int, int]] = {}
    for layer in range(num_layers):
        for proj in TOWER_TARGET_MODULES:
            if proj in target_modules:
                shapes[f"audio_tower.layers.{layer}.self_attn.{proj}"] = (
                    d_model,
                    d_model,
                )
    for proj in CONNECTOR_TARGET_MODULES:
        if proj in target_modules:
            shapes[f"multi_modal_projector.{proj}"] = connector_shapes[
                f"multi_modal_projector.{proj}"
            ]
    return shapes


def _save_adapter(dst, target_modules: list[str]) -> str:
    config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
    shapes = _adapter_module_shapes(config, target_modules)

    gen = torch.Generator().manual_seed(0)
    tensors = {}
    for name, (dim_out, dim_in) in shapes.items():
        base = f"base_model.model.{name}"
        tensors[f"{base}.lora_A.weight"] = (
            torch.randn((RANK, dim_in), generator=gen, dtype=torch.float32) * 0.02
        )
        tensors[f"{base}.lora_B.weight"] = (
            torch.randn((dim_out, RANK), generator=gen, dtype=torch.float32) * 0.02
        )

    save_file(tensors, str(dst / "adapter_model.safetensors"))
    adapter_config = {
        "peft_type": "LORA",
        "r": RANK,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": 0.0,
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "target_modules": target_modules,
    }
    (dst / "adapter_config.json").write_text(json.dumps(adapter_config))
    return str(dst)


@pytest.fixture(scope="module")
def tower_only_adapter(tmp_path_factory) -> str:
    return _save_adapter(
        tmp_path_factory.mktemp("ultravox_tower_lora"), TOWER_TARGET_MODULES
    )


@pytest.fixture(scope="module")
def connector_only_adapter(tmp_path_factory) -> str:
    return _save_adapter(
        tmp_path_factory.mktemp("ultravox_connector_lora"), CONNECTOR_TARGET_MODULES
    )


@pytest.fixture(scope="module")
def tower_connector_adapter(tmp_path_factory) -> str:
    return _save_adapter(
        tmp_path_factory.mktemp("ultravox_tower_connector_lora"),
        TOWER_TARGET_MODULES + CONNECTOR_TARGET_MODULES,
    )


@pytest.fixture(scope="module")
def merged_model(tmp_path_factory, tower_connector_adapter: str) -> str:
    """Base checkpoint with the adapter folded into the tower/connector weights.

    Loaded in vLLM without LoRA, this is the reference for what applying the
    adapter should produce: each targeted weight becomes
    `W + (lora_alpha / r) * B @ A`. The released checkpoint stores only the
    audio tower and connector (the language model is loaded from
    `text_model_id`), so only those tensors need folding; fusing q/k/v happens
    in vLLM at load time, on both the merged and the base+LoRA paths.
    """
    from huggingface_hub import snapshot_download

    src = snapshot_download(MODEL_NAME)
    dst = tmp_path_factory.mktemp("ultravox_merged")
    for name in os.listdir(src):
        shutil.copy2(os.path.realpath(os.path.join(src, name)), dst / name)

    base_sd = load_file(str(dst / "model.safetensors"))
    adapter_sd = load_file(
        os.path.join(tower_connector_adapter, "adapter_model.safetensors")
    )
    scaling = LORA_ALPHA / RANK
    prefix, suffix = "base_model.model.", ".lora_A.weight"
    for key in adapter_sd:
        if not key.endswith(suffix):
            continue
        weight_name = key[len(prefix) : -len(suffix)] + ".weight"
        lora_a = adapter_sd[key]
        lora_b = adapter_sd[key.replace(".lora_A.", ".lora_B.")]
        delta = (scaling * (lora_b @ lora_a)).to(base_sd[weight_name].dtype)
        base_sd[weight_name] = base_sd[weight_name] + delta

    save_file(base_sd, str(dst / "model.safetensors"), metadata={"format": "pt"})
    return str(dst)


def _get_prompt(question: str) -> str:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": f"<|audio|>\n{question}"}],
        tokenize=False,
        add_generation_prompt=True,
    )


def _check_model_available() -> None:
    model_info = HF_EXAMPLE_MODELS.find_hf_info(MODEL_NAME)
    model_info.check_available_online(on_fail="skip")
    model_info.check_transformers_version(on_fail="skip")


@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [16])
def test_tower_and_connector_lora_are_applied(
    vllm_runner: type[VllmRunner],
    audio_assets: AudioTestAssets,
    tower_only_adapter: str,
    connector_only_adapter: str,
    dtype: str,
    max_tokens: int,
) -> None:
    """Tower-only and connector-only adapters must each shift the logprobs.

    Greedy tokens can be insensitive to small perturbations (basin effects),
    but punica adds exact zeros when no adapter weights are active, so with
    prefix caching disabled the engine is deterministic and ANY logprob
    deviation proves the corresponding LoRA delta was applied (guards
    against the silent no-op behavior of #31479).
    """
    _check_model_available()

    prompt = _get_prompt("Transcribe this audio.")
    audio = audio_assets[0].audio_and_sample_rate

    with vllm_runner(
        MODEL_NAME,
        dtype=dtype,
        enforce_eager=True,
        # Prefix caching would let later runs reuse earlier runs' KV
        # blocks, hiding (or leaking) the adapter's effect.
        enable_prefix_caching=False,
        limit_mm_per_prompt={"audio": 1},
        enable_lora=True,
        enable_tower_connector_lora=True,
        max_loras=2,
        max_lora_rank=RANK,
    ) as vllm_model:

        def greedy_logprobs(lora_request):
            [(token_ids, _, logprobs)] = vllm_model.generate_greedy_logprobs(
                [prompt],
                max_tokens,
                num_logprobs=1,
                audios=[[audio]],
                lora_request=lora_request,
            )
            return {tok: lps[tok].logprob for tok, lps in zip(token_ids, logprobs)}

        base = greedy_logprobs(None)
        base_repeat = greedy_logprobs(None)
        assert base == base_repeat, "engine is not deterministic; test is invalid"

        adapters = [
            ("tower", tower_only_adapter),
            ("connector", connector_only_adapter),
        ]
        for lora_id, (name, adapter) in enumerate(adapters, start=1):
            with_lora = greedy_logprobs(LoRARequest(f"{name}-lora", lora_id, adapter))
            shared = base.keys() & with_lora.keys()
            max_delta = max(
                (abs(base[tok] - with_lora[tok]) for tok in shared), default=0.0
            )
            assert len(base) != len(with_lora) or max_delta > 1e-4, (
                f"{name}-only LoRA left all logprobs unchanged; {name} LoRA "
                "weights appear to be silently ignored"
            )


@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [32])
@pytest.mark.parametrize("num_logprobs", [5])
def test_tower_connector_lora_matches_merged_weights(
    vllm_runner: type[VllmRunner],
    audio_assets: AudioTestAssets,
    tower_connector_adapter: str,
    merged_model: str,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
) -> None:
    """Applying the adapter through the LoRA path must match the reference
    checkpoint that folds the same adapter into the base weights.

    This checks the delta's value (not just that it is nonzero): both runs go
    through vLLM's forward, so any divergence isolates the tower/connector LoRA
    application against `W + (alpha / r) * B @ A`.
    """
    _check_model_available()

    prompt = _get_prompt("Transcribe this audio.")
    audio = audio_assets[0].audio_and_sample_rate

    with vllm_runner(
        MODEL_NAME,
        dtype=dtype,
        enforce_eager=True,
        limit_mm_per_prompt={"audio": 1},
        enable_lora=True,
        enable_tower_connector_lora=True,
        max_lora_rank=RANK,
    ) as vllm_model:
        lora_outputs = vllm_model.generate_greedy_logprobs(
            [prompt],
            max_tokens,
            num_logprobs=num_logprobs,
            audios=[[audio]],
            lora_request=LoRARequest(
                "tower-connector-lora", 1, tower_connector_adapter
            ),
        )

    with vllm_runner(
        merged_model,
        dtype=dtype,
        enforce_eager=True,
        limit_mm_per_prompt={"audio": 1},
    ) as vllm_model:
        merged_outputs = vllm_model.generate_greedy_logprobs(
            [prompt],
            max_tokens,
            num_logprobs=num_logprobs,
            audios=[[audio]],
        )

    check_logprobs_close(
        outputs_0_lst=merged_outputs,
        outputs_1_lst=lora_outputs,
        name_0="merged-weights",
        name_1="tower-connector-lora",
    )

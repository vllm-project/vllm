# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from functools import partial
from pathlib import Path

import pytest
import torch
from transformers import Gemma3nTextConfig

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from vllm.platforms import current_platform

from ..models.utils import dummy_hf_overrides
from ..utils import create_new_process_for_each_test


@dataclass(frozen=True)
class SleepModelCase:
    name: str
    architecture: str
    model: str | None
    tensor_names: tuple[str, ...]
    modalities: tuple[str, ...] = ()
    revision: str | None = None
    trust_remote_code: bool = False
    tokenizer_mode: str | None = None


SLEEP_MODEL_CASES = (
    SleepModelCase(
        name="fireredasr2",
        architecture="FireRedASR2ForConditionalGeneration",
        model="allendou/FireRedASR2-LLM-vllm",
        tensor_names=("pe",),
        trust_remote_code=True,
    ),
    SleepModelCase(
        name="gemma3n",
        architecture="Gemma3nForCausalLM",
        model=None,
        tensor_names=(
            "router_input_scale",
            "embed_scale",
            "embed_scale_per_layer",
            "per_layer_input_scale",
            "per_layer_projection_scale",
        ),
    ),
    SleepModelCase(
        name="voxtral",
        architecture="VoxtralForConditionalGeneration",
        model="mistralai/Voxtral-Mini-3B-2507",
        tensor_names=("mel_filters",),
        revision="3060fe34b35ba5d44202ce9ff3c097642914f8f3",
        tokenizer_mode="mistral",
    ),
    SleepModelCase(
        name="ernie45_vl",
        architecture="Ernie4_5_VLMoeForConditionalGeneration",
        model="baidu/ERNIE-4.5-VL-28B-A3B-PT",
        tensor_names=("inv_freq", "_visual_token_ids_tensor_cache"),
        revision="refs/pr/17",
        trust_remote_code=True,
    ),
)


def _create_local_model_config(case: SleepModelCase, model_dir: Path) -> str:
    if case.name == "gemma3n":
        config = Gemma3nTextConfig(
            architectures=[case.architecture],
            vocab_size=128,
            hidden_size=64,
            intermediate_size=[128, 128, 128],
            num_hidden_layers=3,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=32,
            max_position_embeddings=256,
            sliding_window=64,
            layer_types=[
                "sliding_attention",
                "full_attention",
                "sliding_attention",
            ],
            vocab_size_per_layer_input=128,
            hidden_size_per_layer_input=16,
            num_kv_shared_layers=1,
            laurel_rank=16,
            activation_sparsity_pattern=[0.0, 0.0, 0.0],
        )
    else:
        raise AssertionError(f"No local config builder for {case.name}")

    model_dir.mkdir()
    config.save_pretrained(model_dir)
    return str(model_dir)


def _sleep_test_hf_overrides(hf_config, *, model_arch: str):
    hf_config = dummy_hf_overrides(hf_config, model_arch=model_arch)
    if model_arch == "FireRedASR2ForConditionalGeneration":
        audio_encoder_conf = getattr(hf_config, "audio_encoder_conf", {})
        audio_encoder_conf.update(
            {
                "idim": 80,
                "n_layers_enc": 1,
                "n_head": 1,
                "d_model": 32,
                "kernel_size": 3,
                # MM profiling uses the repository's real feature extractor
                # and can produce roughly 750 encoder frames. Keep the real
                # positional-encoding capacity instead of shrinking it with
                # the number of test decoder tokens.
                "pe_maxlen": 5000,
            }
        )
        hf_config.audio_encoder_conf = audio_encoder_conf
    if (
        model_arch == "Ernie4_5_VLMoeForConditionalGeneration"
        and getattr(hf_config, "im_patch_id", None) is None
    ):
        # Some revisions expose this id as image_token_id, while the native
        # model intentionally caches it under the older im_patch_id name.
        hf_config.im_patch_id = getattr(hf_config, "image_token_id", 100295)
    return hf_config


def _get_sleep_tensor_targets(
    model_name: str,
) -> tuple[tuple[type[torch.nn.Module], tuple[str, ...]], ...]:
    if model_name == "fireredasr2":
        from vllm.model_executor.models.conformer_encoder import (
            RelPositionalEncoding,
        )

        return ((RelPositionalEncoding, ("pe",)),)
    if model_name == "gemma3n":
        from vllm.model_executor.models.gemma3n import (
            Gemma3nAltUp,
            Gemma3nSelfDecoder,
        )

        return (
            (Gemma3nAltUp, ("router_input_scale",)),
            (
                Gemma3nSelfDecoder,
                (
                    "embed_scale",
                    "embed_scale_per_layer",
                    "per_layer_input_scale",
                    "per_layer_projection_scale",
                ),
            ),
        )
    if model_name == "voxtral":
        from vllm.model_executor.models.voxtral import VoxtralEncoderModel

        return ((VoxtralEncoderModel, ("mel_filters",)),)
    if model_name == "ernie45_vl":
        from vllm.model_executor.models.ernie45_vl import (
            Ernie4_5_VisionRotaryEmbedding,
            Ernie4_5_VLMoeForConditionalGeneration,
        )

        return (
            (Ernie4_5_VisionRotaryEmbedding, ("inv_freq",)),
            (
                Ernie4_5_VLMoeForConditionalGeneration,
                ("_visual_token_ids_tensor_cache",),
            ),
        )
    raise AssertionError(f"No sleep tensor targets for {model_name}")


def _snapshot_sleep_tensors(
    model: torch.nn.Module,
    *,
    model_name: str,
    tensor_names: tuple[str, ...],
) -> dict[str, tuple[int, torch.Tensor]]:
    targets = _get_sleep_tensor_targets(model_name)
    snapshot: dict[str, tuple[int, torch.Tensor]] = {}
    found_tensor_names = set[str]()

    for module_path, module in model.named_modules():
        for module_type, target_names in targets:
            if not isinstance(module, module_type):
                continue
            for tensor_name in target_names:
                tensor = getattr(module, tensor_name, None)
                if tensor is None:
                    continue
                assert isinstance(tensor, torch.Tensor)
                qualified_name = (
                    f"{module_path}.{tensor_name}" if module_path else tensor_name
                )
                snapshot[qualified_name] = (
                    tensor.data_ptr(),
                    tensor.detach().cpu().clone(),
                )
                found_tensor_names.add(tensor_name)

    assert found_tensor_names == set(tensor_names), (
        f"{model_name}: expected tensors {tensor_names}, "
        f"found {tuple(sorted(found_tensor_names))}"
    )
    return snapshot


def _save_dummy_weights(worker) -> None:
    model = worker.model_runner.model
    worker._sleep_test_dummy_weights = [
        (name, parameter.detach().cpu().clone())
        for name, parameter in model.named_parameters()
    ]


def _reload_dummy_weights(worker) -> None:
    with torch.no_grad():
        worker.reload_weights(
            weights_iterator=iter(worker._sleep_test_dummy_weights),
            is_checkpoint_format=False,
        )


def _generation_signature(output) -> tuple[tuple[int, ...], tuple[float, ...]]:
    completion = output[0].outputs[0]
    token_ids = tuple(completion.token_ids)
    assert completion.logprobs is not None
    chosen_logprobs = tuple(
        step[token_id].logprob
        for token_id, step in zip(token_ids, completion.logprobs, strict=True)
    )
    return token_ids, chosen_logprobs


def _assert_tensor_snapshots_equal(
    before: dict[str, tuple[int, torch.Tensor]],
    after: dict[str, tuple[int, torch.Tensor]],
    model_name: str,
) -> None:
    assert after.keys() == before.keys(), model_name
    for name, (before_ptr, before_value) in before.items():
        after_ptr, after_value = after[name]
        assert after_ptr == before_ptr, f"{model_name}: {name} address changed"
        torch.testing.assert_close(
            after_value,
            before_value,
            msg=lambda msg, name=name: (f"{model_name}: {name} was corrupted: {msg}"),
        )


@create_new_process_for_each_test("spawn")
@pytest.mark.slow_test
@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA sleep mode test")
@pytest.mark.parametrize("case", SLEEP_MODEL_CASES, ids=lambda case: case.name)
def test_static_model_tensors_survive_level2_restore(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    case: SleepModelCase,
) -> None:
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    if case.name == "fireredasr2":
        # FireRedASR2 has a regular decoder and compute_logits implementation,
        # but advertises only the transcription task. Enable its text path so
        # this ownership test can exercise the common LLM.generate lifecycle
        # without constructing an unrelated audio fixture.
        from vllm.model_executor.models.fireredasr2 import (
            FireRedASR2ForConditionalGeneration,
        )

        monkeypatch.setattr(
            FireRedASR2ForConditionalGeneration,
            "supports_transcription_only",
            False,
        )

    model = case.model or _create_local_model_config(case, tmp_path / case.name)
    llm_kwargs = {
        "model": model,
        "load_format": "dummy",
        "enable_sleep_mode": True,
        "enforce_eager": True,
        "attention_backend": "TRITON_ATTN",
        "max_model_len": 128,
        "max_num_seqs": 1,
        "gpu_memory_utilization": 0.5,
        "seed": 0,
        "enable_prefix_caching": False,
        "disable_log_stats": True,
        "trust_remote_code": case.trust_remote_code,
    }
    if case.revision is not None:
        llm_kwargs["revision"] = case.revision
    if case.name not in ("fireredasr2", "voxtral", "ernie45_vl"):
        llm_kwargs["skip_tokenizer_init"] = True
    if case.tokenizer_mode is not None:
        llm_kwargs["tokenizer_mode"] = case.tokenizer_mode
    if case.model is not None:
        llm_kwargs["hf_overrides"] = partial(
            _sleep_test_hf_overrides, model_arch=case.architecture
        )
    if case.modalities:
        llm_kwargs["limit_mm_per_prompt"] = {
            modality: 0 for modality in case.modalities
        }

    llm = LLM(**llm_kwargs)
    prompt = TokensPrompt(prompt_token_ids=[2, 5, 9, 12, 7, 3])
    sampling_params = SamplingParams(temperature=0.0, max_tokens=8, logprobs=1)

    before_generation = _generation_signature(llm.generate(prompt, sampling_params))
    snapshot_fn = partial(
        _snapshot_sleep_tensors,
        model_name=case.name,
        tensor_names=case.tensor_names,
    )
    (before_tensors,) = llm.apply_model(snapshot_fn)
    llm.collective_rpc(_save_dummy_weights)

    llm.sleep(level=2)
    llm.wake_up(tags=["weights"])
    (after_tensors,) = llm.apply_model(snapshot_fn)
    _assert_tensor_snapshots_equal(before_tensors, after_tensors, case.name)

    llm.collective_rpc(_reload_dummy_weights)
    llm.wake_up(tags=["kv_cache"])
    after_generation = _generation_signature(llm.generate(prompt, sampling_params))

    assert after_generation[0] == before_generation[0], case.name
    assert after_generation[1] == pytest.approx(
        before_generation[1], rel=1e-5, abs=1e-5
    ), case.name

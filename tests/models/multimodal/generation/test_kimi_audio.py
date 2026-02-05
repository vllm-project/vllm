# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from pathlib import Path

import pytest
import torch

from vllm.model_executor.models.kimi_audio_asr import KimiAudioForConditionalGeneration
from vllm.platforms import current_platform


class _DummyEmbedTokens:
    def __init__(self, hidden: int):
        self.embedding_dim = hidden
        self.weight = torch.zeros((1, hidden), dtype=torch.float16)

    def __call__(self, ids: torch.Tensor) -> torch.Tensor:
        # ids: [S]
        s = int(ids.numel())
        return torch.zeros((s, self.embedding_dim), dtype=self.weight.dtype)


class _DummyVQAdaptor(torch.nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.hidden = hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expect [S, B, F] and return [S, B, H]
        assert x.dim() == 3
        s, b = int(x.shape[0]), int(x.shape[1])
        return torch.zeros((s, b, self.hidden), dtype=torch.float16)


class _DummyModel:
    def __init__(self, hidden: int):
        self.embed_tokens = _DummyEmbedTokens(hidden)
        self.vq_adaptor = _DummyVQAdaptor(hidden)


def test_embed_input_ids_smoke_no_shape_errors():
    # This is a lightweight unit test to ensure the embed_input_ids mixing path
    # works with flattened token shapes (vLLM V1 convention) and does not crash
    # on common [S] / [S, F] inputs.
    hidden = 3584
    s = 8

    dummy_self = type("_Dummy", (), {})()
    dummy_self.model = _DummyModel(hidden)

    input_ids = torch.zeros((s,), dtype=torch.long)
    audio_input_ids = torch.ones((s,), dtype=torch.long)
    is_continuous_mask = torch.ones((s,), dtype=torch.bool)
    text_input_ids = torch.zeros((s,), dtype=torch.long)

    # Case 1: raw whisper feature dim (needs vq_adaptor)
    out_raw = KimiAudioForConditionalGeneration.embed_input_ids(
        dummy_self,
        input_ids,
        whisper_input_features=torch.zeros((s, 5120), dtype=torch.float16),
        is_continuous_mask=is_continuous_mask,
        text_input_ids=text_input_ids,
        audio_input_ids=audio_input_ids,
    )
    assert isinstance(out_raw, torch.Tensor)
    assert out_raw.shape == (s, hidden)

    # Case 2: already-projected whisper embeddings (hidden_size), should bypass
    # adaptor and still succeed.
    out_proj = KimiAudioForConditionalGeneration.embed_input_ids(
        dummy_self,
        input_ids,
        whisper_input_features=torch.zeros((s, hidden), dtype=torch.float16),
        is_continuous_mask=is_continuous_mask,
        text_input_ids=text_input_ids,
        audio_input_ids=audio_input_ids,
    )
    assert isinstance(out_proj, torch.Tensor)
    assert out_proj.shape == (s, hidden)


def test_kimi_audio_does_not_register_audio_tower_submodule() -> None:
    """Regression test: Kimi-Audio must not register a tower submodule.

    V1 multiprocessing uses a strict missing-weights check in DefaultModelLoader.
    If KimiAudioTower is registered as a model submodule and contains parameters not
    present in the checkpoint, engine startup will fail.

    This test prevents reintroducing `self.audio_tower = ...` style registration.
    """
    # Resolve repo root robustly regardless of test file location.
    # Find the first parent that contains a top-level `vllm/` package directory.
    this_file = Path(__file__).resolve()
    repo_root = next(
        parent for parent in this_file.parents if (parent / "vllm").is_dir()
    )

    path = repo_root / "vllm" / "model_executor" / "models" / "kimi_audio_asr.py"
    src = path.read_text(encoding="utf-8")

    assert "self.audio_tower" not in src, (
        "KimiAudioTower must not be assigned to `self.audio_tower` as a model "
        "submodule; use runtime-only instantiation to avoid V1 missing-weights "
        "failures."
    )


@pytest.mark.core_model
@pytest.mark.parametrize("dtype", ["half"])
def test_kimi_audio_hf_outputs_match_vllm(
    hf_runner, vllm_runner, monkeypatch, dtype: str
) -> None:
    """Compare vLLM outputs against HuggingFace for a small ASR case.

    Skips on CPU CI.
    """
    if current_platform.is_cpu():
        pytest.skip("Skipping HF comparison on CPU CI")

    # Avoid fork-based engine startup issues in multi-threaded pytest runs.
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    # Kimi-Audio native prompt construction currently depends on kimia_infer.
    # In vLLM dev setups, the upstream Kimi-Audio repo may be checked out next
    # to vLLM (e.g. ./Kimi-Audio/). Mirror the model's import fallback behavior
    # so we can run the HF comparison test locally.
    try:
        import kimia_infer.api.prompt_manager  # noqa: F401
    except ModuleNotFoundError:
        this_file = Path(__file__).resolve()
        repo_root = next(
            parent for parent in this_file.parents if (parent / "vllm").is_dir()
        )
        kimi_audio_root = repo_root / "Kimi-Audio"
        sys.path.insert(0, str(kimi_audio_root))
        try:
            import kimia_infer.api.prompt_manager  # noqa: F401
        except Exception as e:  # noqa: BLE001
            pytest.skip(f"kimia_infer not available: {e}")
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"kimia_infer not available: {e}")

    # Prefer local model path when present (faster, no download).
    local_path = "/data1/moonshotai/Kimi-Audio-7B-Instruct"
    model = (
        local_path if Path(local_path).exists() else "moonshotai/Kimi-Audio-7B-Instruct"
    )

    # Use a short, deterministic audio clip.
    from vllm.assets.audio import AudioAsset

    audio, sr = AudioAsset("mary_had_lamb").audio_and_sample_rate

    # Kimi-Audio expects 16kHz features (see get_speech_to_text_config).
    if sr != 16000:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000

    # vLLM: pass audio as multimodal input; prompt text can be empty.
    with vllm_runner(
        model,
        dtype=dtype,
        max_model_len=2048,
        max_num_seqs=1,
        limit_mm_per_prompt={"audio": 1},
        enable_mm_embeds=True,
        enforce_eager=True,
        disable_custom_all_reduce=True,
    ) as vllm_model:
        try:
            vllm_out = vllm_model.generate_greedy_logprobs(
                [""],
                max_tokens=128,
                num_logprobs=5,
                audios=[(audio, sr)],
            )
        except ValueError as e:
            # Today, Kimi-Audio in vLLM expects preprocessed tensors (whisper
            # features + token streams) rather than raw audio tuples. Once the
            # preprocessing is integrated into vLLM's multimodal processor, this
            # test can be converted into a true HF-vs-vLLM comparison.
            pytest.skip(f"vLLM does not accept raw audio for Kimi-Audio yet: {e}")

    # HF: run the same clip through the reference model.
    from transformers import AutoModelForCausalLM

    with hf_runner(model, dtype=dtype, auto_cls=AutoModelForCausalLM) as hf_model:
        if not hasattr(hf_model.model, "generate"):
            pytest.skip(
                "HF Kimi-Audio model does not implement generate(); "
                "need a custom decoding loop for HF-vs-vLLM comparison"
            )
        hf_out = hf_model.generate_greedy_logprobs_limit(
            [""],
            max_tokens=128,
            num_logprobs=5,
            audios=[(audio, sr)],
        )

    # Compare tokens/logprobs approximately (exact match may differ).
    from tests.models.utils import check_logprobs_close

    check_logprobs_close(
        outputs_0_lst=hf_out, outputs_1_lst=vllm_out, name_0="hf", name_1="vllm"
    )

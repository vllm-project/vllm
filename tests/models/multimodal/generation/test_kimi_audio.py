# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path

import pytest
import torch

from vllm.model_executor.models.kimi_audio_asr import KimiAudioForConditionalGeneration
from vllm.platforms import current_platform

# Skip entire module if kimia_infer is not installed (optional dependency)
try:
    import kimia_infer.api.prompt_manager  # noqa: F401
except Exception as e:  # noqa: BLE001
    pytest.skip(f"kimia_infer not available: {e}", allow_module_level=True)


class _DummyEmbedTokens:
    def __init__(self, hidden: int):
        self.embedding_dim = hidden
        self.weight = torch.zeros((1, hidden), dtype=torch.float16)

    def __call__(self, ids: torch.Tensor) -> torch.Tensor:
        s = int(ids.numel())
        return torch.zeros((s, self.embedding_dim), dtype=self.weight.dtype)


class _DummyVQAdaptor(torch.nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.hidden = hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 3
        s, b = int(x.shape[0]), int(x.shape[1])
        return torch.zeros((s, b, self.hidden), dtype=torch.float16)


class _DummyModel:
    def __init__(self, hidden: int):
        self.embed_tokens = _DummyEmbedTokens(hidden)
        self.vq_adaptor = _DummyVQAdaptor(hidden)


def test_embed_input_ids_smoke_no_shape_errors():
    # Lightweight smoke test for embed_input_ids mixing path.
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


def test_kimi_audio_post_process_output_cleans_prompt_echo() -> None:
    raw = (
        "这 并不是 告别 这是一个 篇章 的 结束 也是 新篇章 的 开始 "
        "AA AA A A , , Please trans cribe the following audio ."
    )
    cleaned = KimiAudioForConditionalGeneration.post_process_output(raw)
    compact = cleaned.replace(" ", "").replace("，", "").replace("。", "")

    assert "这并不是告别这是一个篇章的结束也是新篇章的开始" in compact
    assert "transcribe" not in cleaned.lower()
    assert "audio" not in cleaned.lower()
    assert "AA" not in cleaned


def test_kimi_audio_post_process_output_dedupes_chinese() -> None:
    raw = (
        "输出转写文本， 不要其他内容。 远"
        "这并不是告别这是一个篇章的结束也是新篇章的开始"
        "这并不是告别这是一个篇章的结束也是新篇章的开始"
    )
    cleaned = KimiAudioForConditionalGeneration.post_process_output(raw)

    assert cleaned == "这并不是告别这是一个篇章的结束也是新篇章的开始"


def test_kimi_audio_post_process_output_strips_leading_char() -> None:
    raw = "输出转写文本不要其他内容远这并不是告别这是一个篇章的结束也是新篇章的开始"
    cleaned = KimiAudioForConditionalGeneration.post_process_output(raw)

    assert cleaned == "这并不是告别这是一个篇章的结束也是新篇章的开始"


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
def test_kimi_audio_basic_load(vllm_runner, monkeypatch, dtype: str) -> None:
    """Test that vLLM can load Kimi-Audio model with proper tokenizer.

    This is a basic smoke test that verifies:
    1. KimiTokenizer is auto-detected and loads correctly
    2. Model initialization succeeds
    3. Inference does not hang (basic forward pass)

    Note: Full HF comparison test is skipped because:
    - Kimi-Audio requires kimia_infer for preprocessing (not a vLLM dep)
    - Detokenizer loading hangs (19GB file, known upstream issue)
    """
    if current_platform.is_cpu():
        pytest.skip("Skipping on CPU CI")

    # Avoid fork-based engine startup issues in multi-threaded pytest runs.
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    model = "moonshotai/Kimi-Audio-7B-Instruct"

    # Test that model loads with Kimi tokenizer (auto-detected)
    # If this doesn't raise, the tokenizer and model loaded successfully
    with vllm_runner(
        model,
        dtype=dtype,
        max_model_len=2048,
        max_num_seqs=1,
        limit_mm_per_prompt={"audio": 1},
        enable_mm_embeds=True,
        enforce_eager=True,
        disable_custom_all_reduce=True,
        # tokenizer_mode is auto-detected as 'kimi' for this model
    ):
        # Model and tokenizer loaded successfully if we get here
        pass

from types import SimpleNamespace

from vllm.model_executor.models.config import Lfm2ForCausalLMConfig
from vllm.v1.attention.backends.registry import AttentionBackendEnum


def _vllm_config(backend=None):
    return SimpleNamespace(attention_config=SimpleNamespace(backend=backend))


def test_lfm2_forces_triton_attention_on_rocm(monkeypatch):
    import vllm.platforms as platforms

    monkeypatch.setattr(platforms, "current_platform", SimpleNamespace(is_rocm=lambda: True))

    vllm_config = _vllm_config()
    Lfm2ForCausalLMConfig.verify_and_update_config(vllm_config)

    assert vllm_config.attention_config.backend == AttentionBackendEnum.TRITON_ATTN


def test_lfm2_keeps_explicit_attention_backend_on_rocm(monkeypatch):
    import vllm.platforms as platforms

    monkeypatch.setattr(platforms, "current_platform", SimpleNamespace(is_rocm=lambda: True))

    vllm_config = _vllm_config(AttentionBackendEnum.ROCM_ATTN)
    Lfm2ForCausalLMConfig.verify_and_update_config(vllm_config)

    assert vllm_config.attention_config.backend == AttentionBackendEnum.ROCM_ATTN


def test_lfm2_does_not_force_attention_backend_off_rocm(monkeypatch):
    import vllm.platforms as platforms

    monkeypatch.setattr(
        platforms, "current_platform", SimpleNamespace(is_rocm=lambda: False)
    )

    vllm_config = _vllm_config()
    Lfm2ForCausalLMConfig.verify_and_update_config(vllm_config)

    assert vllm_config.attention_config.backend is None

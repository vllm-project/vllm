import pytest


def test_get_layers_from_vllm_config_normalizes_model_prefix():
    # Skip if torch isn't installed in the test environment. The full
    # vLLM package expects torch; this keeps the test runnable (skipped)
    # in lightweight developer environments while ensuring CI with torch
    # installed exercises the behavior.
    pytest.importorskip("torch")

    from vllm.config.vllm import get_layers_from_vllm_config

    class DummyLayer:
        pass

    # Build a fake vllm_config with the minimal structure needed.
    class FakeCompilationConfig:
        def __init__(self, mapping):
            self.static_forward_context = mapping

    class FakeVllmConfig:
        def __init__(self, mapping):
            self.compilation_config = FakeCompilationConfig(mapping)

    # The forward context key (runtime) is missing the extra '.model.' segment
    forward_key = "language_model.layers.0.self_attn.attn"
    layer_obj = DummyLayer()

    fake_cfg = FakeVllmConfig({forward_key: layer_obj})

    # Request the name with the extra '.model.' segment (as seen in some
    # checkpoint mappings). The function should resolve it to the object
    # present in static_forward_context.
    requested = ["language_model.model.layers.0.self_attn.attn"]
    result = get_layers_from_vllm_config(fake_cfg, DummyLayer, requested)

    assert requested[0] in result
    assert result[requested[0]] is layer_obj

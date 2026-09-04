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

    # Variant 1: Remove a redundant '.model.' segment (only first occurrence)
    forward_key1 = "language_model.layers.0.self_attn.attn"
    layer_obj1 = DummyLayer()
    fake_cfg1 = FakeVllmConfig({forward_key1: layer_obj1})
    requested1 = ["language_model.model.layers.0.self_attn.attn"]
    result1 = get_layers_from_vllm_config(fake_cfg1, DummyLayer, requested1)
    assert requested1[0] in result1
    assert result1[requested1[0]] is layer_obj1

    # Variant 2: Insert '.model.' after 'language_model.'
    forward_key2 = "language_model.model.layers.1.self_attn.attn"
    layer_obj2 = DummyLayer()
    fake_cfg2 = FakeVllmConfig({forward_key2: layer_obj2})
    requested2 = ["language_model.layers.1.self_attn.attn"]
    result2 = get_layers_from_vllm_config(fake_cfg2, DummyLayer, requested2)
    assert requested2[0] in result2
    assert result2[requested2[0]] is layer_obj2

    # Variant 3: Collapse duplicate 'model.model.' to single 'model.'
    forward_key3 = "language_model.model.layers.2.self_attn.attn"
    layer_obj3 = DummyLayer()
    fake_cfg3 = FakeVllmConfig({forward_key3: layer_obj3})
    requested3 = ["language_model.model.model.layers.2.self_attn.attn"]
    result3 = get_layers_from_vllm_config(fake_cfg3, DummyLayer, requested3)
    assert requested3[0] in result3
    assert result3[requested3[0]] is layer_obj3

def test_platform_plugins():
    from vllm.platforms import current_platform
    assert current_platform.device_name == "DummyDevice"

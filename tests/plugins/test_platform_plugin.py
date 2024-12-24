from vllm.platforms import PlatformRegistry, current_platform


def test_current_platform_register():
    # make sure the platform is registered
    assert PlatformRegistry.current_platform == "my_platform"
    # make sure the platform is loaded
    assert current_platform.device_name == "dummy"
    assert current_platform.is_async_output_supported(enforce_eager=True) \
        is False

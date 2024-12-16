from vllm import PlatformRegistry


def register():
    # Register the dummy platform
    PlatformRegistry.register_platform(
        "my_platform", "vllm_add_dummy_platform.my_platform.DummyPlatform")
    # Set the current platform to the dummy platform
    PlatformRegistry.set_current_platform("my_platform")

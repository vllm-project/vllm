from typing import Tuple


def dummy_platform_plugin() -> Tuple[bool, str]:
    is_dummy = True
    return is_dummy, "vllm_add_dummy_platform.dummy_platform.DummyPlatform"

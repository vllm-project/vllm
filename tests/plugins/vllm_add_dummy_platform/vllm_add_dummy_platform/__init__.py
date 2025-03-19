# SPDX-License-Identifier: Apache-2.0

from typing import Optional


def dummy_platform_plugin() -> Optional[str]:
    return "vllm_add_dummy_platform.dummy_platform.DummyPlatform"

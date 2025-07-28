# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional


def dummy_platform_plugin() -> Optional[str]:
    return "vllm_add_dummy_platform.dummy_platform.DummyPlatform"


def register_ops():
    import vllm_add_dummy_platform.dummy_custom_ops  # noqa

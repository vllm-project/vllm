# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


def dummy_platform_plugin() -> str | None:
    return "vllm_add_dummy_platform.dummy_platform.DummyPlatform"


def register_ops():
    import vllm_add_dummy_platform.dummy_custom_ops  # noqa

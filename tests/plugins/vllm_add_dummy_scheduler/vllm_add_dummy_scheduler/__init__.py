# SPDX-License-Identifier: Apache-2.0

from typing import Optional


def dummy_scheduler_plugin() -> Optional[str]:
    return "vllm_add_dummy_scheduler.dummy_platform.DummyPlatform"

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

import pytest

from vllm.assets.audio import AudioAsset


@pytest.fixture(scope="module")
def rocm_aiter_fa_attention():
    """
    Sets VLLM_ATTENTION_BACKEND=ROCM_AITER_FA for ROCm
    for the duration of this test module.
    """
    from vllm.platforms import current_platform

    if current_platform.is_rocm():
        old_backend = os.environ.get("VLLM_ATTENTION_BACKEND")
        os.environ["VLLM_ATTENTION_BACKEND"] = "ROCM_AITER_FA"
        yield
        if old_backend is None:
            del os.environ["VLLM_ATTENTION_BACKEND"]
        else:
            os.environ["VLLM_ATTENTION_BACKEND"] = old_backend
    else:
        yield


def pytest_collection_modifyitems(session, config, items):
    """Auto-use rocm_aiter_fa_attention fixture for specific test files."""
    for item in items:
        if item.nodeid and (
            "test_transcription_validation.py" in item.nodeid
            or "test_translation_validation.py" in item.nodeid
        ):
            item.fixturenames.append("rocm_aiter_fa_attention")


@pytest.fixture
def mary_had_lamb():
    path = AudioAsset("mary_had_lamb").get_local_path()
    with open(str(path), "rb") as f:
        yield f


@pytest.fixture
def winning_call():
    path = AudioAsset("winning_call").get_local_path()
    with open(str(path), "rb") as f:
        yield f


@pytest.fixture
def foscolo():
    # Test translation it->en
    path = AudioAsset("azacinto_foscolo").get_local_path()
    with open(str(path), "rb") as f:
        yield f

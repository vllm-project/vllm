# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.assets.audio import AudioAsset


def add_attention_backend(server_args, attention_config):
    """Append attention backend CLI arg if specified.

    Args:
        server_args: List of server arguments to extend in-place.
        attention_config: Dict with 'backend' key, or None.
    """
    if attention_config and "backend" in attention_config:
        server_args.extend(["--attention-backend", attention_config["backend"]])


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
    # NOTE: "azacinto_foscolo" is served from the same asset bucket and is used
    # by examples/ as well, but it is missing from the ``AudioAssetName``
    # literal in vllm/assets/audio.py, so the call does not type check.
    path = AudioAsset("azacinto_foscolo").get_local_path()  # type: ignore[arg-type]
    with open(str(path), "rb") as f:
        yield f

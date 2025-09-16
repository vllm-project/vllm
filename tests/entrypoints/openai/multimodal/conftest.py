# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from ....utils import RemoteOpenAIServer

VISION_SERVER_ARGS = [
    "--runner", "generate", "--dtype", "bfloat16", "--max-model-len", "1024",
    "--enforce-eager", "--max-num-seqs", "4", "--gpu-memory-utilization",
    "0.7", "--trust-remote-code", "--limit-mm-per-prompt", '{"image": 2}',
    "--disable-log-stats", "--disable-log-requests"
]

AUDIO_SERVER_ARGS = [
    "--dtype", "float32", "--max-model-len", "1024", "--enforce-eager",
    "--max-num-seqs", "4", "--gpu-memory-utilization", "0.7",
    "--trust-remote-code", "--limit-mm-per-prompt", '{"audio": 2}',
    "--disable-log-stats", "--disable-log-requests"
]

VIDEO_SERVER_ARGS = [
    "--runner", "generate", "--dtype", "bfloat16", "--max-model-len", "1024",
    "--enforce-eager", "--max-num-seqs", "4", "--gpu-memory-utilization",
    "0.7", "--trust-remote-code", "--limit-mm-per-prompt", '{"video": 1}',
    "--disable-log-stats", "--disable-log-requests"
]


@pytest.fixture(scope="package")
def vision_server():
    with RemoteOpenAIServer("microsoft/Phi-3.5-vision-instruct",
                            VISION_SERVER_ARGS,
                            max_wait_seconds=120) as server:
        yield server


@pytest.fixture(scope="package")
def audio_server():
    with RemoteOpenAIServer("fixie-ai/ultravox-v0_5-llama-3_2-1b",
                            AUDIO_SERVER_ARGS,
                            max_wait_seconds=120) as server:
        yield server


@pytest.fixture(scope="package")
def video_server():
    with RemoteOpenAIServer("microsoft/Phi-3.5-vision-instruct",
                            VIDEO_SERVER_ARGS,
                            max_wait_seconds=120) as server:
        yield server

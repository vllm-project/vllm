# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from ....utils import RemoteOpenAIServer

BASIC_SERVER_ARGS = [
    "--dtype", "bfloat16", "--max-model-len", "1024", "--enforce-eager",
    "--max-num-seqs", "32", "--gpu-memory-utilization", "0.7",
    "--enable-server-load-tracking", "--chat-template",
    "{% for message in messages %}{{message['role'] + ': ' \
    + message['content'] + '\\n'}}{% endfor %}", "--enable-auto-tool-choice",
    "--tool-call-parser", "hermes", "--trust-remote-code"
]


@pytest.fixture(scope="package")
def server():
    with RemoteOpenAIServer("microsoft/DialoGPT-small",
                            BASIC_SERVER_ARGS,
                            max_wait_seconds=120) as server:
        yield server

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from ....utils import RemoteOpenAIServer

DUMMY_CHAT_TEMPLATE = """{% for message in messages %}{{message['role'] + ': ' + message['content'] + '\\n'}}{% endfor %}"""  # noqa: E501
UNIVERSAL_EMBEDDING_ARGS = [
    "--runner", "pooling", "--dtype", "bfloat16", "--enforce-eager",
    "--max-model-len", "512", "--gpu-memory-utilization", "0.7",
    "--max-num-seqs", "4", "--disable-log-stats", "--disable-log-requests",
    "--chat-template", DUMMY_CHAT_TEMPLATE
]


@pytest.fixture(scope="package")
def embedding_server():
    with RemoteOpenAIServer("intfloat/multilingual-e5-small",
                            UNIVERSAL_EMBEDDING_ARGS,
                            max_wait_seconds=120) as server:
        yield server

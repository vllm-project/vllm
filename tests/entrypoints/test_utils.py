# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.entrypoints.utils import sanitize_message


def test_sanitize_message():
    assert (
        sanitize_message("<_io.BytesIO object at 0x7a95e299e750>")
        == "<_io.BytesIO object>"
    )
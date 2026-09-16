# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import patch

from vllm.snapshot.utils import is_restore


def test_is_restore_delegates_to_current_platform():
    with patch(
        "vllm.snapshot.utils.current_platform.is_restore", return_value=True
    ) as platform_is_restore:
        assert is_restore()

    platform_is_restore.assert_called_once_with()

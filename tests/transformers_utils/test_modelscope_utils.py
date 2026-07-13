# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import tempfile

from vllm.transformers_utils.modelscope_utils import configure_modelscope_runtime


def test_configure_modelscope_runtime_sets_writable_defaults(monkeypatch):
    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_MODELSCOPE", "True")
        m.setenv("NO_PROXY", "localhost,127.0.0.1,::1")
        m.delenv("MODELSCOPE_CACHE", raising=False)
        m.delenv("MODELSCOPE_CREDENTIALS_PATH", raising=False)

        configure_modelscope_runtime()

        assert "modelscope.cn" in os.environ["NO_PROXY"]
        expected_root = tempfile.gettempdir() + "/modelscope"
        assert os.environ["MODELSCOPE_CACHE"] == expected_root
        assert (
            os.environ["MODELSCOPE_CREDENTIALS_PATH"]
            == expected_root + "/credentials"
        )

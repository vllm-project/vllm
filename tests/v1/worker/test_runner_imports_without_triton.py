# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The model runner imports where vLLM disables Triton.

There ``vllm.triton_utils.tl`` is ``TritonLanguagePlaceholder``, which has only
a few dtypes, so a module the runner imports must not read Triton attributes at
module scope (only inside ``@triton.jit`` bodies or at launch).
"""

import subprocess
import sys
import textwrap


def test_model_runner_imports_with_the_triton_placeholder():
    code = textwrap.dedent(
        """
        import sys

        sys.modules["triton"] = None  # find_spec("triton") raises -> no Triton
        import vllm.triton_utils as triton_utils
        from vllm.triton_utils.importing import TritonLanguagePlaceholder

        assert not triton_utils.HAS_TRITON
        assert isinstance(triton_utils.tl, TritonLanguagePlaceholder)
        import vllm.v1.worker.gpu.model_runner  # noqa: F401
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=600
    )
    assert result.returncode == 0, result.stderr[-4000:]

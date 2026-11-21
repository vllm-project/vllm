# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import subprocess
import sys
from pathlib import Path


def test_broken_boto3_does_not_break_model_inspection(tmp_path: Path):
    (tmp_path / "boto3.py").write_text(
        'raise AttributeError("broken boto3 import")\n',
        encoding="utf-8",
    )

    repo_root = Path(__file__).parents[2]
    env = os.environ.copy()
    pythonpath = [str(tmp_path), str(repo_root)]
    if existing_pythonpath := env.get("PYTHONPATH"):
        pythonpath.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    env["VLLM_CACHE_ROOT"] = str(tmp_path / "cache")

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import vllm.transformers_utils.s3_utils as s3_utils; "
                "import vllm.model_executor.model_loader; "
                "from vllm.model_executor.models.registry import ModelRegistry; "
                "from vllm.utils.import_utils import PlaceholderModule; "
                "assert isinstance(s3_utils.boto3, PlaceholderModule); "
                "assert ModelRegistry._try_inspect_model_cls("
                "'LlamaForCausalLM'"
                ") is not None"
            ),
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )

    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "broken boto3 import" in output

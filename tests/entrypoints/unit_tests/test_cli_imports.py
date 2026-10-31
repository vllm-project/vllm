# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import subprocess
import sys


def test_version_does_not_import_multimedia_decoders():
    """The version-only CLI path must not load native multimedia libraries."""
    script = """
import json
import runpy
import sys

sys.argv = ["vllm", "--version"]
try:
    runpy.run_module("vllm.entrypoints.cli.main", run_name="__main__")
except SystemExit as exc:
    if exc.code != 0:
        raise

print(json.dumps({
    "cv2": "cv2" in sys.modules,
    "torchcodec": "torchcodec" in sys.modules,
}))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )

    imported = json.loads(result.stdout.splitlines()[-1])
    assert imported == {"cv2": False, "torchcodec": False}

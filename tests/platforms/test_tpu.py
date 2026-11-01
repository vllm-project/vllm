# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
import sys

import pytest


def test_tpu_platform_missing_tpu_inference_raises_helpful_error(monkeypatch):
    """Importing the TPU platform without tpu_inference must fail fast.

    Previously the missing import was only logged and the module was left
    without TpuPlatform, so platform resolution later failed with a
    confusing AttributeError. Now the import itself raises an ImportError
    pointing at the missing package.
    """
    monkeypatch.setitem(sys.modules, "tpu_inference", None)
    monkeypatch.setitem(sys.modules, "tpu_inference.platforms", None)
    sys.modules.pop("vllm.platforms.tpu", None)
    with pytest.raises(ImportError, match="tpu_inference not found"):
        importlib.import_module("vllm.platforms.tpu")

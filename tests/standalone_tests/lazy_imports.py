# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Description: Test the lazy import module
# The utility function cannot be placed in `vllm.utils`
# this needs to be a standalone script
import sys

# List of modules that should not be imported too early.
# Lazy import `torch._inductor.async_compile` to avoid creating
# too many processes before we set the number of compiler threads.
# Lazy import `cv2` to avoid bothering users who only use text models.
# `cv2` can easily mess up the environment.
module_names = ["torch._inductor.async_compile", "cv2"]

# set all modules in `module_names` to be None.
# if we import any modules during `import vllm`, there would be a
# hard error and nice stacktrace on the first import.
for module_name in module_names:
    sys.modules[module_name] = None  # type: ignore[assignment]

import vllm  # noqa

# Positive check: the compile stack must stay out of `import vllm`. The
# sys.modules trick above cannot catch a regression here, because an eager
# `from torch._inductor import lowering` guarded by `except ImportError`
# silently swallows the poisoned-module error (this is how #42129 undid
# #40056 unnoticed). Use `python -X importtime -c "import vllm"` to find
# the offending import chain when this fails.
# vllm.env_override still carries version-gated backports that import
# Dynamo/Inductor eagerly on torch < 2.12, so the guarantee applies to the
# torch versions vLLM pins today (>= 2.12).
from packaging.version import Version  # noqa: E402

import torch  # noqa: E402

if Version(torch.__version__) >= Version("2.12.0.dev"):
    eagerly_loaded = [
        module_name
        for module_name in ("torch._inductor", "torch._dynamo")
        if module_name in sys.modules
    ]
    assert not eagerly_loaded, (
        f"`import vllm` must not import {eagerly_loaded}; only processes that "
        "actually compile a model should pay for the Inductor stack."
    )

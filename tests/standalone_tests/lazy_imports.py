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
# `openai`, `fastapi` and `starlette` belong to the frontend layer; `vllm.config`
# (imported by every process) and the engine core's `vllm.v1.structured_output`
# must not pull them in.
module_names = [
    "torch._inductor.async_compile",
    "cv2",
    "openai",
    "fastapi",
    "starlette",
]

# set all modules in `module_names` to be None.
# if we import any modules during `import vllm`, there would be a
# hard error and nice stacktrace on the first import.
for module_name in module_names:
    sys.modules[module_name] = None

import vllm  # noqa
import vllm.config  # noqa
import vllm.v1.structured_output  # noqa

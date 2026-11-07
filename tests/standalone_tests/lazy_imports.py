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
# `openai` belongs to the frontend layer; `vllm.config` is imported by
# every process and must not pull it in.
module_names = ["torch._inductor.async_compile", "cv2", "openai"]

# set all modules in `module_names` to be None.
# if we import any modules during `import vllm`, there would be a
# hard error and nice stacktrace on the first import.
for module_name in module_names:
    sys.modules[module_name] = None

import vllm  # noqa
import vllm.config  # noqa

# Resolving config defaults must not import the allreduce fusion pass module:
# it eagerly imports `flashinfer.comm`, whose import initializes CUDA in the
# API server process and makes vLLM force the `spawn` multiprocessing method.
from vllm.config.compilation import PassConfig  # noqa

PassConfig().flashinfer_max_size(2)
_FUSION_PASS = "vllm.compilation.passes.fusion.allreduce_rms_fusion"
assert _FUSION_PASS not in sys.modules, (
    "PassConfig.flashinfer_max_size() imported the allreduce fusion pass "
    "module, which initializes CUDA as an import side effect"
)

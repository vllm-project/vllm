# SPDX-License-Identifier: Apache-2.0

# Description: Test the lazy import module
# The utility function cannot be placed in `vllm.utils`
# this needs to be a standalone script
import sys
from contextlib import nullcontext

from vllm_test_utils import BlameResult, blame

# List of modules that should not be imported too early.
# Lazy import `torch._inductor.async_compile` to avoid creating
# too many processes before we set the number of compiler threads.
# Lazy import `cv2` to avoid bothering users who only use text models.
# `cv2` can easily mess up the environment.
module_names = ["torch._inductor.async_compile", "cv2"]


def any_module_imported():
    return any(module_name in sys.modules for module_name in module_names)


# In CI, we only check finally if the module is imported.
# If it is indeed imported, we can rerun the test with `use_blame=True`,
# which will trace every function call to find the first import location,
# and help find the root cause.
# We don't run it in CI by default because it is slow.
use_blame = False
context = blame(any_module_imported) if use_blame else nullcontext()
with context as result:
    import vllm  # noqa

if use_blame:
    assert isinstance(result, BlameResult)
    print(f"the first import location is:\n{result.trace_stack}")

assert not any_module_imported(), (
    f"Some the modules in {module_names} are imported. To see the first"
    f" import location, run the test with `use_blame=True`.")

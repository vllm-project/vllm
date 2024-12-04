# Description: Test the lazy import module
# The utility function cannot be placed in `vllm.utils`
# this needs to be a standalone script
import sys
from contextlib import nullcontext

from vllm_test_utils import BlameResult, blame

module_name = "torch._inductor.async_compile"

# In CI, we only check finally if the module is imported.
# If it is indeed imported, we can rerun the test with `use_blame=True`,
# which will trace every function call to find the first import location,
# and help find the root cause.
# We don't run it in CI by default because it is slow.
use_blame = False
context = blame(
    lambda: module_name in sys.modules) if use_blame else nullcontext()
with context as result:
    import vllm  # noqa

if use_blame:
    assert isinstance(result, BlameResult)
    print(f"the first import location is:\n{result.trace_stack}")

assert module_name not in sys.modules, (
    f"Module {module_name} is imported. To see the first"
    f" import location, run the test with `use_blame=True`.")

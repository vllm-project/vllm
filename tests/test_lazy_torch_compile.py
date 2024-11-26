# Description: Test the lazy import module
# The utility function cannot be placed in `vllm.utils`
# this needs to be a standalone script
import sys

from vllm_test_utils import blame

module_name = "torch._inductor.async_compile"

with blame(lambda: module_name in sys.modules) as result:
    import vllm  # noqa

assert not result.found, (f"Module {module_name} is already imported, the"
                          f" first import location is:\n{result.trace_stack}")

print(f"Module {module_name} is not imported yet")

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Description: Test the lazy import module
# The utility function cannot be placed in `vllm.utils`
# this needs to be a standalone script
import subprocess
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
    sys.modules[module_name] = None

import vllm  # noqa

# `vllm --version` loads every CLI subcommand before parsing argv, and those
# reach `vllm.multimodal.media`, so the CLI path needs guarding too.
CLI_SUBCOMMAND_IMPORTS = """
import vllm.entrypoints.cli.benchmark.main
import vllm.entrypoints.cli.collect_env
import vllm.entrypoints.cli.download_kernels
import vllm.entrypoints.cli.launch
import vllm.entrypoints.cli.openai
import vllm.entrypoints.cli.preload
import vllm.entrypoints.cli.run_batch
import vllm.entrypoints.cli.serve
import vllm.entrypoints.cli.snapshot
"""

# Checked in a clean process, by inspecting `sys.modules` rather than poisoning
# it: an eager import wrapped in `try: import cv2 / except ImportError:` would
# swallow the error the poisoning above relies on, and still load the library.
subprocess.run(
    [
        sys.executable,
        "-c",
        CLI_SUBCOMMAND_IMPORTS
        + """
import sys

eager = [name for name in ("cv2",) if name in sys.modules]
assert not eager, f"loading the vLLM CLI must not import {eager}"
""",
    ],
    check=True,
)

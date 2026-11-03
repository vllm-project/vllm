# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Keep vLLM custom ops on the eager-bypass path.

Ops registered with `direct_register_custom_op` must be called through the
returned handle or `vllm_ops.<name>`, which skip the dispatcher in eager mode.
Compilation passes are exempt: they match and emit the dispatcher ops.
"""

import sys

import regex as re

_PATTERNS = {
    r"\btorch\.ops\.vllm\.\w+\s*\(|=\s*torch\.ops\.vllm\.\w+\s*$": (
        "Call vLLM custom ops via `vllm_ops.<name>` (vllm.utils.torch_utils) "
        "or the handle returned by direct_register_custom_op, not "
        "`torch.ops.vllm.<name>`."
    ),
    r"\btorch\.library\.custom_op\s*\(": (
        "Register vLLM custom ops with direct_register_custom_op instead of "
        "torch.library.custom_op."
    ),
}

ALLOWED_PREFIXES = ("vllm/compilation/",)


def scan_file(path: str) -> int:
    returncode = 0
    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if line.lstrip().startswith("#"):
                continue
            for pattern, message in _PATTERNS.items():
                if re.search(pattern, line):
                    print(f"{path}:{line_num}: \033[91merror:\033[0m {message}")
                    returncode = 1
    return returncode


def main():
    returncode = 0
    for filename in sys.argv[1:]:
        if filename.startswith(ALLOWED_PREFIXES):
            continue
        returncode |= scan_file(filename)
    return returncode


if __name__ == "__main__":
    sys.exit(main())

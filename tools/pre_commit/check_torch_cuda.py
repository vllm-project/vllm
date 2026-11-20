# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import sys

import regex as re

_ACCELERATOR = (
    "Please refer RFC https://github.com/vllm-project/vllm/issues/30679, "
    "use torch.accelerator API instead."
)

# --------------------------------------------------------------------------- #
# Regex: match `torch.cuda.xxx` but allow `torch.accelerator.xxx`
# --------------------------------------------------------------------------- #
_TORCH_CUDA_PATTERNS = [
    (
        r"\btorch\.cuda\.(empty_cache|synchronize|device_count|current_device|memory_reserved|memory_allocated|max_memory_allocated|max_memory_reserved|reset_peak_memory_stats|memory_stats|mem_get_info|set_device|device)\b",
        _ACCELERATOR,
    ),
    (r"\btorch\.cuda\.(manual_seed|manual_seed_all)\b", "Use set_random_seed instead."),
    # Calls torch.cuda.{_is_compiled/_device_count_amdsmi/_device_count_nvml} internally
    (
        r"\bcuda_device_count_stateless\(",
        "Use current_platform.device_count() instead.",
    ),
    # Platform.__getattr__ forwards this to torch.cuda.mem_get_info.
    (
        r"\bcurrent_platform\.mem_get_info\(",
        "Use torch.accelerator.get_memory_info() instead.",
    ),
]

ALLOWED_FILES = {
    "vllm/platforms/",
    "vllm/device_allocator/",
    "vllm/distributed/weight_transfer/ipc_engine.py",
    "vllm/distributed/weight_transfer/sharded_rdt_engine.py",
    "tests/distributed/test_packed_tensor.py",
    # Wraps (rather than calls) the torch.cuda sync APIs.
    "vllm/v1/conf_compute_utils.py",
    # Predate the working current_platform.mem_get_info check; migrate to
    # torch.accelerator.get_memory_info and drop from this list.
    "tests/kernels/core/test_mrope.py",
    "tests/kernels/fla/test_chunk_int64_offsets.py",
    "tests/kernels/fla/test_fla_layernorm_guard.py",
    "tests/kernels/fla/test_fused_gdn_post_conv.py",
    "tests/kernels/mamba/test_checkpoint_exporter.py",
    "tests/kernels/test_fused_qk_norm_rope_gate.py",
    "tests/kernels/turboquant/test_flydsl_turboquant_decode.py",
}


def scan_file(path: str) -> int:
    with open(path, encoding="utf-8") as f:
        content = f.read()
    found = []
    for pattern, message in _TORCH_CUDA_PATTERNS:
        for match in re.finditer(pattern, content):
            line_start = content.rfind("\n", 0, match.start()) + 1
            # shortcut: skips `#` comments only; docstrings and strings still match.
            if "#" in content[line_start : match.start()]:
                continue
            line_num = content.count("\n", 0, match.start()) + 1
            found.append((line_num, match.group(0), message))
    for line_num, matched_text, message in sorted(found):
        print(
            f"{path}:{line_num}: "
            "\033[91merror:\033[0m "  # red color
            f"Found {matched_text} API call. {message}"
        )
    return int(bool(found))


def main():
    returncode = 0
    for filename in sys.argv[1:]:
        if any(filename.startswith(prefix) for prefix in ALLOWED_FILES):
            continue
        returncode |= scan_file(filename)
    return returncode


if __name__ == "__main__":
    sys.exit(main())

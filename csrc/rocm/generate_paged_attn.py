#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generate the ROCm paged attention translation units and their dispatch.

Instantiating the paged attention kernels is by far the most expensive part of
building _rocm_C: the device backend emits code for every (dtype, kv-cache
dtype, head size, block size, alibi, out dtype, V-cache layout, gqa ratio)
combination, so compiling all of them in a single file serialized ~10 min of
device code generation onto one core.

This script emits, from the single VARIANTS list below:

  * attention_paged_attn_<variant>_bs<block_size>.cu: one translation unit per
    (variant, block size) pair, each instantiating only its own slice of the
    kernel set, so ninja compiles the slices in parallel (~48 s per unit instead
    of ~95 s for a whole (dtype, kv-cache dtype, head size) set),
  * attention_paged_attn_entries.inc: the prototypes of the entry points those
    translation units define (included at file scope by attention_dispatch.cu),
  * attention_paged_attn_dispatch.inc: the head-size/block-size/dtype dispatch
    of paged_attention() (included inside the function body).

Variants, entry-point prototypes and the dispatch arms are all derived from
VARIANTS, so a translation unit can no longer be compiled without being
dispatched (which used to need a configure-time guard regexing
attention_dispatch.cu).

Generated files are written next to the kernels, like
csrc/quantization/marlin/generate_kernels.py, and are gitignored.
"""

import glob
import hashlib
import os
import sys

# Block sizes the kernels are instantiated for (see CALL_CUSTOM_LAUNCHER in
# attention_kernels.cuh). One translation unit is generated per (variant, block
# size).
BLOCK_SIZES = [16, 32]

# ALIBI_ENABLED and the out dtype are *not* split into their own translation
# units: both are only a runtime branch on the optional alibi_slopes /
# fp8_out_scale arguments (CALL_CUSTOM_LAUNCHER_OUT in attention_kernels.cuh),
# and splitting them was measured to be a loss. It halves the per-unit codegen
# (61.6 s -> 33.1 s for the h128 units) but every unit pays ~20 s just to parse
# attention_kernels.cuh, so _rocm_C's serial work grew from 1434 s to 2974 s,
# the -j64 wall stayed at 64 s (skinny_gemms is the limit there) and -j16 went
# from 142 s to 240 s.

# One entry per supported (query dtype, kv-cache dtype, head size) kernel set.
#   name:               suffix of the generated file/entry point names
#   query_type:         C++ type of the query/out (template parameter T)
#   kv_type:            C++ type of the key/value cache (template parameter KVT)
#                       (half/bfloat16 are written with their HIP spellings:
#                       these sources are only compiled through hipify, which
#                       maps __nv_bfloat16 to __hip_bfloat16 but leaves _Float16
#                       alone - `__half` is a distinct type from _Float16 for the
#                       MFMA builtins)
#   kv_cache_enum:      Fp8KVCacheDataType passed to the launcher
#   head_size:          head size instantiated
#   mfma:               MFMAType template parameter
#   query_scalar_type:  at::ScalarType of the query this variant serves
#   kv_cache_dtype:     kv_cache_dtype string this variant serves ("auto",
#                       "fp8"); the exact accepted spellings and the mfma_type
#                       gating live below.
#   mfma_type:          mfma_type string this variant serves, or None when the
#                       kv-cache dtype does not split on mfma_type.
VARIANTS = [
    {
        "name": "h64_f16_auto",
        "query_type": "_Float16",
        "kv_type": "_Float16",
        "kv_cache_enum": "vllm::Fp8KVCacheDataType::kAuto",
        "head_size": 64,
        "mfma": "MFMAType::F16",
        "query_scalar_type": "Half",
        "kv_cache_dtype": "auto",
        "mfma_type": None,
    },
    {
        "name": "h128_f16_auto",
        "query_type": "_Float16",
        "kv_type": "_Float16",
        "kv_cache_enum": "vllm::Fp8KVCacheDataType::kAuto",
        "head_size": 128,
        "mfma": "MFMAType::F16",
        "query_scalar_type": "Half",
        "kv_cache_dtype": "auto",
        "mfma_type": None,
    },
    {
        "name": "h64_bf16_auto",
        "query_type": "__hip_bfloat16",
        "kv_type": "__hip_bfloat16",
        "kv_cache_enum": "vllm::Fp8KVCacheDataType::kAuto",
        "head_size": 64,
        "mfma": "MFMAType::F16",
        "query_scalar_type": "BFloat16",
        "kv_cache_dtype": "auto",
        "mfma_type": None,
    },
    {
        "name": "h128_bf16_auto",
        "query_type": "__hip_bfloat16",
        "kv_type": "__hip_bfloat16",
        "kv_cache_enum": "vllm::Fp8KVCacheDataType::kAuto",
        "head_size": 128,
        "mfma": "MFMAType::F16",
        "query_scalar_type": "BFloat16",
        "kv_cache_dtype": "auto",
        "mfma_type": None,
    },
    {
        "name": "h64_f16_fp8_f16",
        "query_type": "_Float16",
        "kv_type": "uint8_t",
        "kv_cache_enum": "vllm::Fp8KVCacheDataType::kFp8E4M3",
        "head_size": 64,
        "mfma": "MFMAType::F16",
        "query_scalar_type": "Half",
        "kv_cache_dtype": "fp8",
        "mfma_type": "f16",
    },
    {
        "name": "h128_f16_fp8_f16",
        "query_type": "_Float16",
        "kv_type": "uint8_t",
        "kv_cache_enum": "vllm::Fp8KVCacheDataType::kFp8E4M3",
        "head_size": 128,
        "mfma": "MFMAType::F16",
        "query_scalar_type": "Half",
        "kv_cache_dtype": "fp8",
        "mfma_type": "f16",
    },
    {
        "name": "h64_f16_fp8_fp8",
        "query_type": "_Float16",
        "kv_type": "uint8_t",
        "kv_cache_enum": "vllm::Fp8KVCacheDataType::kFp8E4M3",
        "head_size": 64,
        "mfma": "MFMAType::Fp8",
        "query_scalar_type": "Half",
        "kv_cache_dtype": "fp8",
        "mfma_type": "fp8",
    },
    {
        "name": "h128_f16_fp8_fp8",
        "query_type": "_Float16",
        "kv_type": "uint8_t",
        "kv_cache_enum": "vllm::Fp8KVCacheDataType::kFp8E4M3",
        "head_size": 128,
        "mfma": "MFMAType::Fp8",
        "query_scalar_type": "Half",
        "kv_cache_dtype": "fp8",
        "mfma_type": "fp8",
    },
    {
        "name": "h64_bf16_fp8_f16",
        "query_type": "__hip_bfloat16",
        "kv_type": "uint8_t",
        "kv_cache_enum": "vllm::Fp8KVCacheDataType::kFp8E4M3",
        "head_size": 64,
        "mfma": "MFMAType::F16",
        "query_scalar_type": "BFloat16",
        "kv_cache_dtype": "fp8",
        "mfma_type": "f16",
    },
    {
        "name": "h128_bf16_fp8_f16",
        "query_type": "__hip_bfloat16",
        "kv_type": "uint8_t",
        "kv_cache_enum": "vllm::Fp8KVCacheDataType::kFp8E4M3",
        "head_size": 128,
        "mfma": "MFMAType::F16",
        "query_scalar_type": "BFloat16",
        "kv_cache_dtype": "fp8",
        "mfma_type": "f16",
    },
    {
        "name": "h64_bf16_fp8_fp8",
        "query_type": "__hip_bfloat16",
        "kv_type": "uint8_t",
        "kv_cache_enum": "vllm::Fp8KVCacheDataType::kFp8E4M3",
        "head_size": 64,
        "mfma": "MFMAType::Fp8",
        "query_scalar_type": "BFloat16",
        "kv_cache_dtype": "fp8",
        "mfma_type": "fp8",
    },
    {
        "name": "h128_bf16_fp8_fp8",
        "query_type": "__hip_bfloat16",
        "kv_type": "uint8_t",
        "kv_cache_enum": "vllm::Fp8KVCacheDataType::kFp8E4M3",
        "head_size": 128,
        "mfma": "MFMAType::Fp8",
        "query_scalar_type": "BFloat16",
        "kv_cache_dtype": "fp8",
        "mfma_type": "fp8",
    },
]

# kv_cache_dtype values accepted per VARIANTS["kv_cache_dtype"], in dispatch
# order (the first matching arm wins).
KV_CACHE_DTYPE_PREDICATES = {
    "auto": 'kv_cache_dtype == "auto"',
    "fp8": 'kv_cache_dtype == "fp8" || kv_cache_dtype == "fp8_e4m3"',
}

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# MD5 of this script, recorded in the stamp so CMake can tell whether the files
# on disk were produced by the current version (same MD5 as `file(MD5 ...)`).
# CMake passes the key it compares against so the format is defined in one place;
# a manual run falls back to the hash above.
with open(__file__, "rb") as _self:
    SCRIPT_HASH = hashlib.md5(_self.read()).hexdigest()
KEY = sys.argv[1] if len(sys.argv) > 1 else SCRIPT_HASH
STAMP_NAME = ".paged_attn_generated"

# Whether the Navi (gfx11/gfx12) kernel set is instantiated is a compile
# definition decided by CMake (`VLLM_ROCM_ATTN_NO_NAVI`, see CMakeLists.txt), so
# the units written here are identical for every build directory and arch list.

FILE_HEAD = """\
// auto generated by generate_paged_attn.py
// clang-format off
"""

TU_HEAD = FILE_HEAD + """
#include "attention_kernels.cuh"
"""

TU_ENTRY = """
ATTN_ENTRY(paged_attention_{name}_bs{block_size}) {{
  CALL_CUSTOM_LAUNCHER_OUT({query_type}, {kv_type}, {kv_cache_enum},
                           {block_size}, {head_size}, {mfma});
}}
"""

ENTRIES_HEAD = FILE_HEAD + """
// Prototypes of the entry points defined in the generated
// attention_paged_attn_*.cu translation units.
"""

DISPATCH_HEAD = FILE_HEAD + """
  // Dispatch on kv_cache_dtype, query dtype, mfma_type, block size and head
  // size, mirroring the template parameters of the generated entry points.
"""


def entry_point(variant, block_size):
    return f"paged_attention_{variant['name']}_bs{block_size}"


def block_size_switch(variants, indent_level):
    """Switch on block_size then head_size for one dispatch group.

    `variants` is the list of variants sharing (kv_cache_dtype, query dtype,
    mfma_type); they differ in head size only.
    """
    indent = " " * indent_level
    inner = " " * (indent_level + 4)
    lines = [f"{indent}switch (block_size) {{"]
    for block_size in BLOCK_SIZES:
        lines.append(f"{indent}  case {block_size}:")
        lines.append(f"{inner}switch (head_size) {{")
        for variant in variants:
            lines.append(f"{inner}  case {variant['head_size']}:")
            lines.append(f"{inner}    "
                         f"{entry_point(variant, block_size)}(ATTN_FWD);")
            lines.append(f"{inner}    break;")
        lines.append(f"{inner}  default:")
        lines.append(f'{inner}    TORCH_CHECK(false, '
                     f'"Unsupported head size: ", head_size);')
        lines.append(f"{inner}    break;")
        lines.append(f"{inner}}}")
        lines.append(f"{indent}    break;")
    lines.append(f"{indent}  default:")
    lines.append(f'{indent}    TORCH_CHECK(false, '
                 f'"Unsupported block size: ", block_size);')
    lines.append(f"{indent}    break;")
    lines.append(f"{indent}}}")
    return lines


def group_variants():
    """Group VARIANTS by (kv_cache_dtype, query dtype, mfma_type), in order."""
    groups = {}
    for variant in VARIANTS:
        key = (variant["kv_cache_dtype"], variant["query_scalar_type"],
               variant["mfma_type"])
        groups.setdefault(key, []).append(variant)
    return groups


def generate_dispatch():
    groups = group_variants()
    # kv_cache_dtype -> query dtype -> mfma_type -> variants
    by_kv_cache_dtype = {}
    for (kv_cache_dtype, query_scalar_type, mfma_type), variants in groups.items():
        by_kv_cache_dtype.setdefault(kv_cache_dtype, {}).setdefault(
            query_scalar_type, {})[mfma_type] = variants

    lines = [DISPATCH_HEAD.rstrip("\n")]
    kv_cache_dtypes = list(by_kv_cache_dtype)
    for index, kv_cache_dtype in enumerate(kv_cache_dtypes):
        keyword = "if" if index == 0 else "else if"
        predicate = KV_CACHE_DTYPE_PREDICATES[kv_cache_dtype]
        lines.append(f"  {keyword} ({predicate}) {{")

        by_query_dtype = by_kv_cache_dtype[kv_cache_dtype]
        for dtype_index, (query_scalar_type, by_mfma) in enumerate(
                by_query_dtype.items()):
            dtype_keyword = "if" if dtype_index == 0 else "else if"
            lines.append(f"    {dtype_keyword} (query.dtype() == "
                         f"at::ScalarType::{query_scalar_type}) {{")

            mfma_types = list(by_mfma)
            if len(mfma_types) == 1 and mfma_types[0] is None:
                lines.extend(block_size_switch(by_mfma[None], 6))
            else:
                # ``mfma_type`` is matched with a plain ``else`` on the last arm,
                # as the hand-written dispatch used to: any value other than the
                # recognised ones keeps selecting that arm instead of failing.
                if "f16" not in mfma_types:
                    raise SystemExit(
                        f"the {mfma_types} group needs an f16 arm, which is "
                        "emitted as the catch-all else")
                ordered = sorted(mfma_types,
                                 key=lambda m: m == "f16")  # f16 last
                for mfma_index, mfma_type in enumerate(ordered):
                    if mfma_index == len(ordered) - 1:
                        lines.append("      else {")
                    else:
                        mfma_keyword = "if" if mfma_index == 0 else "else if"
                        lines.append(f'      {mfma_keyword} '
                                     f'(mfma_type == "{mfma_type}") {{')
                    lines.extend(block_size_switch(by_mfma[mfma_type], 8))
                    lines.append("      }")
            lines.append("    }")

        lines.append("    else {")
        lines.append('      TORCH_CHECK(false, "Unsupported data type: ", '
                     "query.dtype());")
        lines.append("    }")
        lines.append("  }")

    lines.append("  else {")
    lines.append('    TORCH_CHECK(false, "Unsupported KV cache dtype: ", '
                 "kv_cache_dtype);")
    lines.append("  }")
    return "\n".join(lines) + "\n"


def generate_entries():
    lines = [ENTRIES_HEAD.rstrip("\n")]
    for variant in VARIANTS:
        for block_size in BLOCK_SIZES:
            lines.append(f"ATTN_ENTRY({entry_point(variant, block_size)});")
    return "\n".join(lines) + "\n"


def generate_tu(variant, block_size):
    return TU_HEAD + TU_ENTRY.format(block_size=block_size, **variant)


def remove_old_files():
    # Only remove files this script produced: a name match alone would delete a
    # developer's scratch copy in the same directory.
    for path in glob.glob(
            os.path.join(OUTPUT_DIR, "attention_paged_attn_*.cu")):
        with open(path) as fd:
            if fd.readline() != FILE_HEAD.splitlines()[0] + "\n":
                continue
        os.remove(path)
    for name in ("attention_paged_attn_entries.inc",
                 "attention_paged_attn_dispatch.inc"):
        path = os.path.join(OUTPUT_DIR, name)
        if os.path.exists(path):
            os.remove(path)


def write(name, content):
    with open(os.path.join(OUTPUT_DIR, name), "w") as fd:
        fd.write(content)


def check_dispatch_covers_entries(entries, dispatch):
    # Cheap replacement for the configure-time guard that used to regex
    # attention_dispatch.cu: every instantiated entry point must be named by the
    # generated dispatch, otherwise its unit is compiled and linked for nothing.
    for name in entries:
        if f"{name}(" not in dispatch:
            raise SystemExit(
                f"{name} is instantiated but never dispatched: add it to "
                "VARIANTS ordering/generate_dispatch(), or drop it.")


def main():
    remove_old_files()
    outputs = []
    entries = []
    for variant in VARIANTS:
        for block_size in BLOCK_SIZES:
            name = f"attention_paged_attn_{variant['name']}_bs{block_size}.cu"
            write(name, generate_tu(variant, block_size))
            outputs.append(name)
            entries.append(entry_point(variant, block_size))
    entries_inc = generate_entries()
    dispatch_inc = generate_dispatch()
    check_dispatch_covers_entries(entries, dispatch_inc)
    for name, content in (("attention_paged_attn_entries.inc", entries_inc),
                          ("attention_paged_attn_dispatch.inc", dispatch_inc)):
        write(name, content)
        outputs.append(name)
    # Record what is on disk next to the outputs: CMake checks this list instead
    # of trusting a glob, and compares the key (script hash + arch list) so that
    # a build directory with a different arch list cannot reuse these units.
    write(STAMP_NAME, KEY + "\n" + "\n".join(outputs) + "\n")
    print(f"Generated {len(outputs)} paged attention files.")


if __name__ == "__main__":
    main()

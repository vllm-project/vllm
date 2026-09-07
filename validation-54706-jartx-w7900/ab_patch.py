#!/usr/bin/env python3
"""Apply the A/B control patch: force the LEGACY (pre-#54706) CAS epilogue
so accuracy/nondeterminism of the old behavior can be measured on this
machine. Revert with: git checkout -- csrc/rocm/q_gemm_rdna3*.cu
"""
import sys

SCALAR = "csrc/rocm/q_gemm_rdna3.cu"
WMMA = "csrc/rocm/q_gemm_rdna3_wmma.cu"


def must(src, find, rep, tag, count=1):
    n = src.count(find)
    if n != count:
        sys.exit(f"AB patch '{tag}': expected {count} occurrence(s), found {n}")
    return src.replace(find, rep)


def apply_patches():
    # --- scalar: zero-init output, bypass partials, skip reduce ---
    src = open(SCALAR).read()
    src = must(src, "  at::Tensor c = torch::empty({size_m, size_n}, opts);",
               "  at::Tensor c = torch::zeros({size_m, size_n}, opts);  "
               "// AB-CONTROL", "scalar-c-zeros")
    src = must(src, "  float* partials_ptr = partials.data_ptr<float>();",
               "  float* partials_ptr = nullptr;  // AB-CONTROL",
               "scalar-no-partials")
    src = must(src,
               "    reduce_partials_rdna3<T><<<blocks, threads, 0, stream>>>(\n"
               "        partials_ptr, c + (long)row0 * size_n, z_count, rows, size_n);",
               "    // AB-CONTROL: reduce disabled (scratch is unused)",
               "scalar-no-reduce")
    open(SCALAR, "w").write(src)
    print("patched scalar")

    # --- wmma: zero-init output, never use partials, no reduce ---
    src = open(WMMA).read()
    src = must(src, "  at::Tensor c = torch::empty({size_m, size_n}, opts);",
               "  at::Tensor c = torch::zeros({size_m, size_n}, opts);  "
               "// AB-CONTROL", "wmma-c-zeros")
    src = must(src, "    partials_ptr = partials.data_ptr<float>();",
               "    partials_ptr = nullptr;  // AB-CONTROL",
               "wmma-ptr-guarded", count=4)  # launchers 16x16/32x16/64x16/64x32
    src = must(src, "  float* partials_ptr = partials.data_ptr<float>();",
               "  float* partials_ptr = nullptr;  // AB-CONTROL",
               "wmma-ptr-64x64", count=2)     # 64x64 fast + tiled paths
    src = must(src, "    float* partials_ptr = partials.data_ptr<float>();",
               "    float* partials_ptr = nullptr;  // AB-CONTROL",
               "wmma-ptr-128x64", count=1)     # 128x64 tiled path
    src = must(src,
               "  reduce_partials_wmma<T><<<blocks, threads, 0, stream>>>(\n"
               "      partials.data_ptr<float>(), c, k_split, size_m, size_n);",
               "  // AB-CONTROL: reduce disabled (scratch is unused)",
               "wmma-no-reduce")
    open(WMMA, "w").write(src)
    print("patched wmma")


if __name__ == "__main__":
    if sys.argv[1] != "apply":
        sys.exit("revert via: git checkout -- csrc/rocm/q_gemm_rdna3*.cu")
    apply_patches()

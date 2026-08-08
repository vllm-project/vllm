"""
Patch the vendored Triton MXFP4 matmul kernel to fix OOB scale reads.

Triton v3.5.1's _matmul_ogs performs an unmasked tl.load of Hopper-swizzled
scale values. When EVEN_K=False, the last tile reads past K, pulling 0xff
from uninitialized memory which gets interpreted as NaN in MXFP4 scale decode.

Upstream fix: triton-lang/triton commit 0add6826
"""

import sys


def patch_file(filepath: str) -> bool:
    with open(filepath) as f:
        content = f.read()

    old = "w_scales = unswizzle_mxfp4_scale_hopper(tl.load(WMxScalePtrs), mx_axis=1, num_warps=num_warps)"

    new = (
        "if EVEN_K:\n"
        "                    hopper_scale_mask = tl.full([PACKED_MX_BLOCK], True, dtype=tl.int1)\n"
        "                else:\n"
        '                    # Mask out the swizzled K tail to prevent OOB reads.\n'
        "                    hopper_scale_mask = (offs_k_scale // 32) * MX_PACK_DIVISOR < k\n"
        "                w_scales = unswizzle_mxfp4_scale_hopper(\n"
        "                    tl.load(WMxScalePtrs, mask=hopper_scale_mask[None, :], other=0),\n"
        "                    mx_axis=1,\n"
        "                    num_warps=num_warps,\n"
        "                )"
    )

    if old not in content:
        print(f"No match in {filepath} -- already patched or unexpected version")
        return False

    content = content.replace(old, new, 1)

    with open(filepath, "w") as f:
        f.write(content)
    print(f"Patched {filepath}")
    return True


if __name__ == "__main__":
    filepath = sys.argv[1]
    success = patch_file(filepath)
    sys.exit(0 if success else 1)

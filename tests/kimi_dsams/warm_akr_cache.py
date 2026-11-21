# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compile (compile-only) every amd_kernels variant the DSA wiring uses, into the FlyDSL
disk cache.

Needs the amd-kernels source checkout (for its `tests/compile_only.py`):
`AMD_KERNELS_SRC`, by default the directory above the imported `amd_kernels` package (an
editable install).

    HIP_VISIBLE_DEVICES=0 python tests/kimi_dsams/warm_akr_cache.py [kernel names]
"""

import os
import sys
import time

os.environ["AKR_COMPILE_ONLY"] = "1"
import importlib.util

import torch


def _load_compile_only():
    src = os.environ.get("AMD_KERNELS_SRC")
    if not src:
        pkg = importlib.util.find_spec("amd_kernels")
        assert pkg is not None and pkg.origin is not None, "amd_kernels not installed"
        src = os.path.dirname(os.path.dirname(pkg.origin))
    spec = importlib.util.spec_from_file_location(
        "akr_compile_only", os.path.join(src, "tests/compile_only.py")
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


compile_only = _load_compile_only()
compile_only.install()
from amd_kernels import warm_up  # noqa: E402, F401  (after compile_only.install())

which = set(sys.argv[1:])
from vllm.models.kimi_k3.amd import akr_dsa as A  # noqa: E402

dev = torch.device("cuda", 0)
torch.accelerator.set_device_index(0)
bf = torch.bfloat16
z = lambda *s, dtype: torch.zeros(s, dtype=dtype, device=dev)
t0 = time.time()


def W(fn, *args, **kwargs):
    spec = fn.entry
    mode = spec.mode(*args, **kwargs)
    if (
        which
        and spec.name.split("/")[-1] not in which
        and spec.name not in which
        and not any(w in spec.name for w in which)
    ):
        return
    for n in spec.sizes(mode):
        a, k = spec.inputs(mode, n, *args, **kwargs)
        spec.fn(*a, **k)
    print("warmed", spec.name, mode, time.time() - t0, flush=True)


from amd_kernels.attention import mla_decode_common as mla  # noqa: E402
from amd_kernels.indexer import mxfp4_indexer_common as ix  # noqa: E402

# (the q gather and the combine need a live 8-rank communicator: the server's warm-up,
# akr_dsa.warm_up, compiles them)
cache = A._dummy_index_cache(dev)
index = dict(
    ln_weight=z(128, dtype=bf),
    ln_bias=z(128, dtype=bf),
    cache=cache,
    slot_mapping=z(1, dtype=torch.int64),
    weights=z(1, 32, dtype=torch.float32),
    weight_scale=1.0,
)
A.prewarm(
    mla.qkv_a_norm_kv_insert_kernel,
    z(1, 2272, dtype=bf),
    z(1536, dtype=bf),
    z(512, dtype=bf),
    1e-5,
    z(1, 1536, dtype=bf),
    z(1, 512, dtype=bf),
    A._dummy_kv_cache(dev),
    z(1, dtype=torch.int64),
    index,
)
print("K_A", time.time() - t0, flush=True)
A.prewarm(
    mla.indexer_prep_kernel,
    z(1, 32, 128, dtype=bf),
    None,
    None,
    None,
    None,
    1e-5,
    None,
    cache,
    z(1, 2048, dtype=torch.int32),
    1.0,
    z(1, 32, 64, dtype=torch.uint8),
    z(1, 1, 4, 16, 4, dtype=torch.uint8),
    None,
)
print("K_B", time.time() - t0, flush=True)
A.prewarm(
    ix.mxfp4_paged_score_topk_kernel,
    z(1, 32, 64, dtype=torch.uint8),
    z(1, 1, 4, 16, 4, dtype=torch.uint8),
    z(1, 32, dtype=torch.float32),
    cache,
    z(1, 1, dtype=torch.int32),
    z(2, dtype=torch.int32),
    z(1, dtype=torch.int32),
    1,
    z(1, 2048, dtype=torch.int32),
    z(1, 2048, dtype=torch.float32),
)
print("score_topk", time.time() - t0, flush=True)
A.prewarm(
    mla.fp8_sparse_mla_decode_kernel,
    z(1, 96, 576, dtype=bf),
    z(64, 656, dtype=torch.uint8),
    z(2, dtype=torch.int32),
    z(1, dtype=torch.int32),
    1.0,
)
print("sparse", time.time() - t0, flush=True)

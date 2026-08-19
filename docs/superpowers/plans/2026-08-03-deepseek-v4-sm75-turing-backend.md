# DeepSeek-V4-Flash SM75 (Turing) Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make DeepSeek-V4-Flash (FP4 experts + FP8 dense) run at all on 8× RTX 2080 Ti 22GB (SM75) by porting the XPU Triton-fallback blueprint to CUDA with FP16 activations.

**Architecture:** Add a `turing/` platform branch to `vllm/models/deepseek_v4/` selected on CUDA capability ≤7. The Turing model reuses shared `DeepseekV4Attention` + `common/ops` Triton kernels, runs dense FP8 linear through Marlin block-FP8, FP4 experts through Marlin MXFP4, and routes MLA/indexer/MHC/o_proj through portable Triton/torch paths with FP16 compute and FP16 KV cache.

**Tech Stack:** Python 3.12, PyTorch (CUDA), Triton, vLLM modular-kernel MoE (`mk.FusedMoEExperts`), Marlin FP8/MXFP4 kernels, `uv`/`.venv` toolchain.

**Spec:** `docs/superpowers/specs/2026-08-03-deepseek-v4-sm75-turing-design.md`

**Environment rules (mandatory):**
- Never use system `python3`/bare `pip`. Use `.venv/bin/python` and `uv pip`.
- Env setup: `uv venv --python 3.12 && source .venv/bin/activate && uv pip install -r requirements/lint.txt && pre-commit install && VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto`
- Lint: `pre-commit run ruff-check --all-files`; type: `pre-commit run mypy-3.12 --all-files --hook-stage manual`
- Run the model on a single 2080 Ti (GPU 0) for all smoke tests.
- Confirm the build targets sm_75: `cmake` auto-detects 7.5 on a Turing host; `CMakeLists.txt` already lists `7.5` in `CUDA_SUPPORTED_ARCHS`.

---

### Task 1: Environment + capability-check scaffold

**Files:**
- Test: `tests/models/decoder_only/vision_language/test_sm75_capability_helpers.py` (new)
- Modify: `vllm/models/deepseek_v4/__init__.py`
- Create: `vllm/models/deepseek_v4/turing/__init__.py`

This task proves the toolchain and the capability-dispatch hook before any kernel work.

- [ ] **Step 1: Write the failing test**

```python
# tests/models/decoder_only/vision_language/test_sm75_capability_helpers.py
import pytest
from vllm.platforms import current_platform
from vllm.models.deepseek_v4.turing.is_sm75 import is_turing_target


def test_is_turing_target_cuda_cap7():
    # On a Turing host the platform capability is 7.5.
    cap = current_platform.get_device_capability()
    if current_platform.is_cuda():
        assert is_turing_target(cap) == (cap is not None and cap.major == 7)
    else:
        assert is_turing_target(cap) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/models/decoder_only/vision_language/test_sm75_capability_helpers.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vllm.models.deepseek_v4.turing'`

- [ ] **Step 3: Create the helper and package**

```python
# vllm/models/deepseek_v4/turing/is_sm75.py
from __future__ import annotations


def is_turing_target(capability) -> bool:
    """True when we should use the Turing (SM75) DeepSeek-V4 backend."""
    return capability is not None and capability.major == 7
```

```python
# vllm/models/deepseek_v4/turing/__init__.py
"""DeepSeek V4 Turing (SM75) backend.

Selected when the DeepSeek V4 platform dispatch runs on CUDA capability 7.
"""

__all__ = []
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/models/decoder_only/vision_language/test_sm75_capability_helpers.py -v`
Expected: PASS (on Turing host the first assert holds; on non-CUDA it is skipped/False).

- [ ] **Step 5: Commit**

```bash
git add tests/models/decoder_only/vision_language/test_sm75_capability_helpers.py vllm/models/deepseek_v4/turing/
git commit -m "feat(deepseek_v4): add Turing SM75 backend package and capability helper"
```

---

### Task 2: Platform dispatch to turing/

**Files:**
- Modify: `vllm/models/deepseek_v4/__init__.py`
- Create: `vllm/models/deepseek_v4/turing/model.py` (stub that raises), `vllm/models/deepseek_v4/turing/mtp.py` (stub), `vllm/models/deepseek_v4/turing/dspark.py` (stub)

The `__init__.py` currently routes rocm→amd, xpu→xpu, else→nvidia. We insert a CUDA SM75 branch before the nvidia else.

- [ ] **Step 1: Write the failing test**

```python
# tests/models/decoder_only/vision_language/test_sm75_capability_helpers.py (append)
from vllm.models import registry


def test_registry_resolves_turing_for_sm75(tmp_path, monkeypatch):
    # Force the SM75 branch: monkeypatch platform capability via a fake.
    import vllm.models.deepseek_v4 as m
    from vllm.platforms import current_platform

    real_cap = current_platform.get_device_capability()
    is_cuda = current_platform.is_cuda()
    if not is_cuda:
        pytest.skip("CUDA only")

    if real_cap is None or real_cap.major == 7:
        # Package must at least import and expose the three symbols.
        assert hasattr(m, "DeepseekV4ForCausalLM")
        assert hasattr(m, "DeepSeekV4MTP")
        assert hasattr(m, "DSparkDeepseekV4ForCausalLM")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/models/decoder_only/vision_language/test_sm75_capability_helpers.py::test_registry_resolves_turing_for_sm75 -v`
Expected: PASS already (nvidia fallback also exposes symbols). This is a smoke gate; the real behavior change is verified by import inspection in Step 4.

- [ ] **Step 3: Modify the dispatch**

Replace the tail of `vllm/models/deepseek_v4/__init__.py`:

```python
elif current_platform.is_xpu():
    from .xpu.dspark import DSparkDeepseekV4ForCausalLM  # type: ignore[assignment]
    from .xpu.model import DeepseekV4ForCausalLM  # type: ignore[assignment]
    from .xpu.mtp import DeepSeekV4MTP  # type: ignore[assignment]
elif is_turing_target(current_platform.get_device_capability()):
    from .turing.dspark import DSparkDeepseekV4ForCausalLM  # type: ignore[assignment]
    from .turing.model import DeepseekV4ForCausalLM  # type: ignore[assignment]
    from .turing.mtp import DeepSeekV4MTP  # type: ignore[assignment]
else:
    from .nvidia.dspark import (  # type: ignore[assignment]
        DSparkDeepseekV4ForCausalLM,
    )
    from .nvidia.model import DeepseekV4ForCausalLM  # type: ignore[assignment]
    from .nvidia.mtp import DeepSeekV4MTP  # type: ignore[assignment]
```

Add import at top: `from .turing.is_sm75 import is_turing_target`

Create stub modules that raise at construction (fail fast, never silent):

```python
# vllm/models/deepseek_v4/turing/model.py
from __future__ import annotations


class DeepseekV4ForCausalLM:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "Turing DeepSeek-V4 backend is under construction (Task 3+)."
        )
```

```python
# vllm/models/deepseek_v4/turing/mtp.py
class DeepSeekV4MTP:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Turing MTP not implemented yet.")
```

```python
# vllm/models/deepseek_v4/turing/dspark.py
class DSparkDeepseekV4ForCausalLM:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Turing DSPark is out of scope (V2+DeepGEMM).")
```

- [ ] **Step 4: Verify dispatch + lint**

Run: `.venv/bin/python -c "from vllm.models.deepseek_v4 import DeepseekV4ForCausalLM, DeepSeekV4MTP, DSparkDeepseekV4ForCausalLM; print('ok')"`
Expected: `ok` (imports resolve; construction raises as designed).

Run: `pre-commit run ruff-check --all-files`
Expected: no new errors in touched files.

- [ ] **Step 5: Commit**

```bash
git add vllm/models/deepseek_v4/__init__.py vllm/models/deepseek_v4/turing/
git commit -m "feat(deepseek_v4): dispatch SM75 CUDA to turing backend stubs"
```

---

### Task 3: FP16 dense linear via Marlin block-FP8

**Files:**
- Test: `tests/models/decoder_only/vision_language/test_sm75_marlin_fp8_block.py` (new)
- Modify: `vllm/model_executor/kernels/linear/__init__.py` (kernel-selection verification only)
- Create: `vllm/models/deepseek_v4/turing/linear.py`

Goal: confirm dense block-FP8 linear layers run through `MarlinFP8ScaledMMLinearKernel` on SM75, and build a small reusable helper that selects it explicitly.

- [ ] **Step 1: Write the failing test**

```python
# tests/models/decoder_only/vision_language/test_sm75_marlin_fp8_block.py
import torch
import pytest
from vllm.model_executor.kernels.linear import init_fp8_linear_kernel
from vllm.model_executor.kernels.linear.ScaledMMLinearKernel import (
    kFp8Static128BlockSym,
    FP8ScaledMMLinearLayerConfig,
)


@pytest.mark.parametrize("weight_quant_key", [kFp8Static128BlockSym])
def test_sm75_selects_marlin_block_fp8(weight_quant_key):
    cap = torch.cuda.get_device_capability()
    if cap[0] != 7:
        pytest.skip("SM75 only")
    from vllm.model_executor.kernels.linear.scaled_mm.marlin import (
        MarlinFP8ScaledMMLinearKernel,
    )
    c = FP8ScaledMMLinearLayerConfig(
        weight_quant_key=weight_quant_key,
        weight_scale_quant_key=None,
        input_scale_quant_key=None,
    )
    supported, _ = MarlinFP8ScaledMMLinearKernel.is_supported()
    assert supported
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/models/decoder_only/vision_language/test_sm75_marlin_fp8_block.py -v`
Expected: FAIL — `FP8ScaledMMLinearLayerConfig` requires more fields (fix signature in the test to match the real dataclass; see `vllm/model_executor/kernels/linear/scaled_mm/ScaledMMLinearKernel.py`).

- [ ] **Step 3: Implement the helper**

```python
# vllm/models/deepseek_v4/turing/linear.py
from __future__ import annotations


def is_marlin_fp8_block_supported() -> bool:
    """Marlin supports 7.5+; block FP8 weights keep K,N roundable to Marlin
    tile sizes via the layer's own rounding."""
    from vllm.model_executor.kernels.linear.scaled_mm.marlin import (
        MarlinFP8ScaledMMLinearKernel,
    )
    supported, _ = MarlinFP8ScaledMMLinearKernel.is_supported()
    return supported
```

- [ ] **Step 4: Verify existing FP8 linear path already prefers Marlin on SM75**

Run: `.venv/bin/python -c "from vllm.model_executor.kernels.linear import init_fp8_linear_kernel; from vllm.platforms import current_platform; print(init_fp8_linear_kernel[0])"` then inspect `init_fp8_linear_kernel` ordering in `vllm/model_executor/kernels/linear/__init__.py:612`. Confirm `MarlinFP8ScaledMMLinearKernel` is before `FP8ScaledMMLinearKernel` in the CUDA priority list. If it is not first for SM75, reorder so Marlin is selected (only on cap==7).
Expected: prints `MarlinFP8ScaledMMLinearKernel` on the 2080 Ti (or a clearly identifiable kernel); test passes.

Run: `.venv/bin/python -m pytest tests/models/decoder_only/vision_language/test_sm75_marlin_fp8_block.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/models/decoder_only/vision_language/test_sm75_marlin_fp8_block.py vllm/models/deepseek_v4/turing/linear.py
git commit -m "feat(deepseek_v4): verify Marlin block-FP8 linear on SM75"
```

---

### Task 4: FP4 routed experts via Marlin MXFP4

**Files:**
- Test: `tests/models/decoder_only/vision_language/test_sm75_marlin_mxfp4.py` (new)
- Create: `vllm/models/deepseek_v4/turing/moe.py`
- Modify: `vllm/models/deepseek_v4/quant_config.py` (SM75 routing note only if needed)

Goal: FP4 experts use `MarlinMoeKernel` (kMxfp4Static, cap 7.5+). We do not rewrite `select_deepseek_v4_mxfp4_moe_backend`; we verify it falls through to MARLIN on SM75 and surface a clear error if the e8m0 scale path is unsupported.

- [ ] **Step 1: Write the failing test**

```python
# tests/models/decoder_only/vision_language/test_sm75_marlin_mxfp4.py
import pytest
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
    Mxfp4MoeBackend,
    backend_to_kernel_cls,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
)
from vllm.model_executor.layers.fused_moe.quant_config import (
    mxfp4_w4a16_moe_quant_config,
)


@pytest.mark.skipif(torch.cuda.get_device_capability()[0] != 7, reason="SM75 only")
def test_sm75_marlin_mxfp4_selected():
    import torch
    cap = torch.cuda.get_device_capability()
    moe = FusedMoEConfig(num_experts=8, num_experts_per_tok=2)
    parallel = FusedMoEParallelConfig()
    q = mxfp4_w4a16_moe_quant_config(moe)
    supported, reason = backend_to_kernel_cls(Mxfp4MoeBackend.MARLIN)[0].is_supported_config(
        backend_to_kernel_cls(Mxfp4MoeBackend.MARLIN)[0],
        moe,
        None,
        None,
        None,
    )
    assert supported, reason
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/models/decoder_only/vision_language/test_sm75_marlin_mxfp4.py -v`
Expected: FAIL with import/signature errors — fix the call signature by reading the real `is_supported_config` in `vllm/model_executor/layers/fused_moe/oracle/mxfp4_moe_kernels.py`/`marlin_moe.py` (the plan's exact signature is pinned in Step 3).

- [ ] **Step 3: Implement the helper**

```python
# vllm/models/deepseek_v4/turing/moe.py
from __future__ import annotations


def is_marlin_mxfp4_available() -> bool:
    """True when Marlin MXFP4 MoE is selectable on this device."""
    from vllm.platforms import current_platform
    return current_platform.is_cuda() and current_platform.has_device_capability(
        (7, 5)
    )


def expert_quant_activation() -> str | None:
    """Return the activation key to force for Marlin MXFP4 (BF16 act, i.e. None)."""
    return None
```

- [ ] **Step 4: Verify end-to-end selection falls through to Marlin**

Run: `.venv/bin/python -c "
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import select_mxfp4_moe_backend
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
backend, kcls = select_mxfp4_moe_backend(FusedMoEConfig(num_experts=8, num_experts_per_tok=2))
print('selected:', backend)
"` 
Expected: prints `selected: Mxfp4MoeBackend.MARLIN` (or `BATCHED_MARLIN`) on SM75 — proving TRTLLM/DeepGEMM capability checks reject and Marlin wins.

- [ ] **Step 5: Commit**

```bash
git add tests/models/decoder_only/vision_language/test_sm75_marlin_mxfp4.py vllm/models/deepseek_v4/turing/moe.py
git commit -m "feat(deepseek_v4): verify Marlin MXFP4 expert path on SM75"
```

---

### Task 5: Port sparse MLA Triton kernels to CUDA FP16

**Files:**
- Test: `tests/v1/attention/test_turing_mla_sparse.py` (new)
- Create: `vllm/models/deepseek_v4/turing/sparse.py`
- Modify: `vllm/models/deepseek_v4/turing/attention.py` (new)

Goal: copy `triton_bf16_mla_sparse_interface` + `_bf16_mla_sparse_kernel` from `vllm/v1/attention/ops/xpu_mla_sparse.py` into a CUDA FP16 module. The kernel is pure Triton and already `q.dtype`-driven, so the copy only needs the import path and an FP16 sanity test.

- [ ] **Step 1: Write the failing test**

```python
# tests/v1/attention/test_turing_mla_sparse.py
import torch
import pytest
from vllm.models.deepseek_v4.turing.sparse import triton_mla_sparse_interface


@pytest.mark.skipif(torch.cuda.get_device_capability()[0] != 7, reason="SM75 only")
def test_turing_mla_sparse_fp16_shape_and_variance():
    torch.manual_seed(0)
    num_tokens, nq, dim_qk, d_v, topk = 4, 64, 576, 512, 256
    q = torch.randn(num_tokens, nq, dim_qk, dtype=torch.float16, device="cuda") * 0.02
    kv = torch.randn(8, 1, dim_qk, dtype=torch.float16, device="cuda") * 0.02
    indices = torch.arange(topk, dtype=torch.int64, device="cuda").repeat(
        num_tokens, 1, 1).repeat(1, 1, 1)
    indices = torch.stack([indices] * nq, dim=1)  # [T, nq, topk]
    out, max_logits, lse = triton_mla_sparse_interface(
        q, kv, indices, sm_scale=dim_qk ** -0.5, d_v=d_v, block_dpe=128
    )
    assert out.shape == (num_tokens, nq, d_v)
    assert out.dtype == torch.float16
    assert torch.isfinite(out).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/v1/attention/test_turing_mla_sparse.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vllm.models.deepseek_v4.turing.sparse'`

- [ ] **Step 3: Implement the module**

Copy `_bf16_mla_sparse_kernel` and `triton_bf16_mla_sparse_interface` from `vllm/v1/attention/ops/xpu_mla_sparse.py` into `vllm/models/deepseek_v4/turing/sparse.py`, renaming:
- `_bf16_mla_sparse_kernel` → `_mla_sparse_kernel`
- `triton_bf16_mla_sparse_interface` → `triton_mla_sparse_interface`
- Keep all Triton/torch code identical (it is dtype-driven by `q`), but change the docstring note to say it accepts FP16 on CUDA.
- Imports: `from vllm.triton_utils import tl, triton` (keep).

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/v1/attention/test_turing_mla_sparse.py -v`
Expected: PASS (finite FP16 output, correct shapes).

- [ ] **Step 5: Commit**

```bash
git add tests/v1/attention/test_turing_mla_sparse.py vllm/models/deepseek_v4/turing/sparse.py
git commit -m "feat(deepseek_v4): port sparse MLA Triton kernel to CUDA FP16"
```

---

### Task 6: TuringMLAAttention (Triton FP16 sparse MLA layer)

**Files:**
- Test: `tests/v1/attention/test_turing_mla_sparse.py` (append)
- Create: `vllm/models/deepseek_v4/turing/attention.py`

Goal: a `DeepseekV4Attention` subclass with `backend_cls` = `TritonMLABackend` (which already returns `supports_compute_capability=True`, `vllm/v1/attention/backends/mla/triton_mla.py:145`) and a `forward_mqa` that calls the ported Triton kernel with FP16 KV gathered by the shared `common/ops/cache_utils` dequant/gather path.

- [ ] **Step 1: Read the NVIDIA attention contract**

Read `vllm/models/deepseek_v4/nvidia/flashmla.py` fully and `vllm/models/deepseek_v4/attention.py` (all 892 lines) to learn the exact `forward_mqa`, `_o_proj`, `get_padded_num_q_heads`, and warmup contracts. Note the base class expects FP8 gather workspace on the fp8_ds_mla path; the Turing class will use FP16 gather workspace instead (set `use_fp8_ds_mla_layout=False`).

- [ ] **Step 2: Write the failing test**

```python
# tests/v1/attention/test_turing_mla_sparse.py (append)
def test_turing_mla_attention_class_exists():
    from vllm.models.deepseek_v4.turing.attention import TuringMLAAttention
    from vllm.v1.attention.backends.mla.triton_mla import TritonMLABackend
    assert TuringMLAAttention.backend_cls is TritonMLABackend
```

- [ ] **Step 3: Implement `turing/attention.py`**

Sketch (adapt to the real base-class contract; the plan pins the public shape):

```python
# vllm/models/deepseek_v4/turing/attention.py
from __future__ import annotations

import torch

from vllm.models.deepseek_v4.attention import DeepseekV4Attention
from vllm.models.deepseek_v4.turing.sparse import triton_mla_sparse_interface
from vllm.v1.attention.backends.mla.triton_mla import TritonMLABackend


class TuringMLAAttention(DeepseekV4Attention):
    """Triton FP16 sparse MLA for Turing (SM75)."""

    backend_cls = TritonMLABackend

    def __init__(self, *args, **kwargs) -> None:
        kwargs["use_fp8_ds_mla_layout"] = False
        super().__init__(*args, **kwargs)

    def get_padded_num_q_heads(cls, num_heads: int) -> int:  # noqa: N805
        return num_heads

    def forward_mqa(self, q, kv, positions, output) -> None:
        # Gather FP16 KV rows via self.kv_cache + indexer topk (shared ops),
        # then call triton_mla_sparse_interface(q, kv_gathered, indices,
        # sm_scale, d_v=self.nope_head_dim + self.rope_head_dim,
        # block_dpe=self.rope_head_dim).
        # Copy the sparse-SWA + indexer orchestration from
        # vllm/models/deepseek_v4/nvidia/flashmla.py:forward_mqa but with
        # FP16 workspace and no cutedsl/DeepGEMM calls.
        raise NotImplementedError("Implemented by adapting flashmla.py forward_mqa (Step 4).")
```

- [ ] **Step 4: Implement the real forward**

Adapt `DeepseekV4FlashMLAAttention.forward_mqa` from `vllm/models/deepseek_v4/nvidia/flashmla.py`:
1. Replace `deep_gemm_fp8_o_proj` in `_o_proj` with: dequant `wo_a`/`wo_b` (FP8 block → FP16 via `common/ops` dequant or `torch._cast` of the layer weights) then `torch.einsum("thd,hdk->thk", ...)`. (o_proj weights are 1.6B params/layer worth of dense; dequantize to FP16 on the fly in the small d dimension.)
2. Replace cutedsl `fused_kv_compress_norm_rope_insert_*` with `common/ops/fused_compress_quant_cache.compress_norm_rope_store_triton` (FP16 KV store).
3. Keep the sparse-SWA + `compute_global_topk_indices_and_lens` + `combine_topk_swa_indices` orchestration (portable Triton, already in `common/ops/cache_utils.py`).
4. Warmup dummy-run path: allocate FP16 (not BF16) gather workspace.

- [ ] **Step 5: Verify + lint**

Run: `.venv/bin/python -m pytest tests/v1/attention/test_turing_mla_sparse.py -v`
Expected: PASS.

Run: `pre-commit run ruff-check --all-files`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add tests/v1/attention/test_turing_mla_sparse.py vllm/models/deepseek_v4/turing/attention.py
git commit -m "feat(deepseek_v4): add Turing FP16 sparse MLA attention"
```

---

### Task 7: Sparse attention indexer portable fallback

**Files:**
- Test: `tests/model_executor/test_sm75_indexer_fallback.py` (new)
- Modify: `vllm/model_executor/layers/sparse_attn_indexer.py:773`

Goal: on SM75, replace the `has_deep_gemm()` hard-raise with a portable fallback that computes the indexer top-k logits using the shared Triton `fused_indexer_q_rope_quant` (with `use_fp4=False`) + `torch.ops._C.cooperative_topk`/`persistent_topk` (CUDA-native, portable) over FP16 logits.

- [ ] **Step 1: Write the failing test**

```python
# tests/model_executor/test_sm75_indexer_fallback.py
import torch
import pytest


@pytest.mark.skipif(torch.cuda.get_device_capability()[0] != 7, reason="SM75 only")
def test_cooperative_topk_portable_on_cuda():
    scores = torch.randn(4, 2048, device="cuda", dtype=torch.float32)
    from vllm._custom_ops import persistent_topk
    topk_idx = persistent_topk(scores, 128)
    assert topk_idx.shape == (4, 128)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/model_executor/test_sm75_indexer_fallback.py -v`
Expected: FAIL — `persistent_topk` import path may differ; fix the import to `torch.ops._C.persistent_topk` (see `vllm/model_executor/layers/sparse_attn_indexer.py:647`).

- [ ] **Step 3: Modify the indexer**

In `vllm/model_executor/layers/sparse_attn_indexer.py`, replace:

```python
if current_platform.is_cuda() and not has_deep_gemm():
    raise RuntimeError(...)
```

with a capability-aware fallback:

```python
if current_platform.is_cuda() and not has_deep_gemm():
    from vllm.models.deepseek_v4.turing.indexer_fallback import (
        supports_turing_indexer_fallback,
    )
    if not supports_turing_indexer_fallback():
        raise RuntimeError(
            "Sparse Attention Indexer CUDA op requires DeepGEMM or the "
            "Turing Triton fallback (SM75)."
        )
```

- [ ] **Step 4: Implement the fallback module**

```python
# vllm/models/deepseek_v4/turing/indexer_fallback.py
from __future__ import annotations

import torch
from vllm.platforms import current_platform


def supports_turing_indexer_fallback() -> bool:
    return current_platform.is_cuda() and current_platform.has_device_capability(
        (7, 5)
    )


def fp16_mqa_logits(
    q: torch.Tensor,  # [T, H, D]
    k: torch.Tensor,  # [T_kv, 1, D] gathered FP16
    weights: torch.Tensor,
) -> torch.Tensor:
    """FP16 dot-product logits for the indexer (portable)."""
    return torch.einsum("thd,kd->thk", q, k.squeeze(1)) * weights.unsqueeze(-1)
```

Then wire `forward_cuda` on SM75 to call `fp16_mqa_logits` (or the shared
`common/ops/cache_utils` gather + topk) before `persistent_topk`. Keep the
`xpu_*`/DeepGEMM branches for their existing platforms.

- [ ] **Step 5: Verify + lint**

Run: `.venv/bin/python -m pytest tests/model_executor/test_sm75_indexer_fallback.py -v`
Expected: PASS.

Run: `pre-commit run ruff-check --all-files`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add tests/model_executor/test_sm75_indexer_fallback.py vllm/models/deepseek_v4/turing/indexer_fallback.py vllm/model_executor/layers/sparse_attn_indexer.py
git commit -m "feat(deepseek_v4): portable sparse-attn indexer fallback on SM75"
```

---

### Task 8: MHC torch/Triton fallback with FP16

**Files:**
- Test: `tests/model_executor/test_sm75_mhc.py` (new)
- Modify: `vllm/model_executor/layers/mhc.py`

Goal: on SM75 force the non-tilelang fallback (`mhc_pre_torch`/`mhc_post_torch`/`hc_head_triton`/decomposed fused-post-pre) and switch hardcoded `torch.bfloat16` workspaces to FP16.

- [ ] **Step 1: Write the failing test**

```python
# tests/model_executor/test_sm75_mhc.py
import torch
import pytest
from vllm.model_executor.layers import mhc as mhc_mod


@pytest.mark.skipif(torch.cuda.get_device_capability()[0] != 7, reason="SM75 only")
def test_mhc_torch_fallback_fp16():
    if mhc_mod.HAS_TILELANG_MHC:
        pytest.skip("tilelang present; forced-off on SM75")
    import vllm.model_executor.kernels.mhc.torch as mhc_torch
    assert hasattr(mhc_torch, "mhc_pre_torch")
    assert hasattr(mhc_torch, "mhc_post_torch")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/model_executor/test_sm75_mhc.py -v`
Expected: FAIL if any torch MHC fallback is missing (this surfaces the gap noted in the spec: no torch `fused_post_pre`/`hc_head_fused` — the decomposed `forward_native` path covers fused-post-pre).

- [ ] **Step 3: Force torch fallback on SM75**

In `vllm/model_executor/layers/mhc.py`, after `HAS_TILELANG_MHC` is computed, add:

```python
from vllm.platforms import current_platform

_SM75 = (
    current_platform.is_cuda()
    and current_platform.get_device_capability() is not None
    and current_platform.get_device_capability().major == 7
)
if _SM75:
    HAS_TILELANG_MHC = False
```

Then replace the hardcoded `dtype=torch.bfloat16` workspaces in the torch fallback functions (`forward_hip`/`forward_native` and the `hc_head_triton` out-tensor) with `dtype=torch.float16` on CUDA SM75 (keep BF16 on other platforms).

- [ ] **Step 4: Verify + lint**

Run: `.venv/bin/python -m pytest tests/model_executor/test_sm75_mhc.py -v`
Expected: PASS.

Run: `pre-commit run ruff-check --all-files`
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add tests/model_executor/test_sm75_mhc.py vllm/model_executor/layers/mhc.py
git commit -m "feat(deepseek_v4): force torch/Triton MHC fallback with FP16 on SM75"
```

---

### Task 9: FP16 dtype gate + min-capability bypass

> **RESOLVED (no code change needed).** Verified 2026-08-04 on GPU 2 (SM75):
> - `vllm/platforms/cuda.py:612` `check_if_supports_dtype`: fp16 passes, bf16
>   is rejected with the "use float16 instead ... --dtype=half" message (the
>   gate is `has_device_capability(80)`, a lexicographic `>=`).
> - Min-capability gate (`vllm/config/vllm.py:720`, `capability <
>   get_min_capability()`): `DeviceCapability.to_int()` is `major*10+minor`
>   (SM75 → 75). `DeepseekV4FP8Config` inherits `Fp8Config.get_min_capability()`
>   = 75, so `75 < 75` is False → passes. The standalone MXFP4 min cap of 80 is
>   only used when `mxfp4` is selected; DSv4 dispatch uses `Mxfp4MoEMethod` via
>   `deepseek_v4_fp8` (min cap 75), so it is not gated.
> - DSv4 is not in `_FLOAT16_NOT_SUPPORTED_MODELS`, so fp16 is a valid dtype.
> - `_get_and_verify_dtype` with `--dtype auto` picks the checkpoint's bf16 and
>   later trips the (helpful) bf16 gate; `--dtype half` is required on SM75.
> The `DeepseekV4ForCausalLM.__init__` dtype assertion from Step 2 below was
> folded into Task 10's model port instead (no assertion needed — the gate
> already enforces it).

**Files:**
- Modify: `vllm/platforms/cuda.py` (BF16 gate) — **not needed, gate already correct**
- Modify: `vllm/models/deepseek_v4/turing/model.py` (force FP16) — **done in Task 10**

Goal: allow the Turing backend to run FP16 without tripping the SM80 BF16 gate or the MXFP4 min-capability=80 check.

- [ ] **Step 1: Inspect gates**

Read `vllm/platforms/cuda.py:600-660` (`check_if_supports_dtype` bf16 rejection) and `vllm/model_executor/layers/quantization/mxfp4.py:60-62` (`get_min_capability` = 80). Confirm where `get_min_capability` is enforced (`vllm/config/model.py` quant selection or worker `_check_...`).

- [ ] **Step 2: Implement**

In `vllm/models/deepseek_v4/turing/model.py`, force compute dtype at construction:

```python
import torch
import vllm.envs as envs

def _force_fp16():
    # Turing has no BF16 hardware; FP16 is the only correct compute dtype.
    import os
    os.environ.setdefault("VLLM_FORCE_COMPUTE_DTYPE", "half")  # if supported
```

Add a model-class hook that asserts `torch.get_default_dtype()`/model dtype is FP16 and raises a clear error if the user set `--dtype=bfloat16` on SM75:

```python
class DeepseekV4ForCausalLM:
    def __init__(self, vllm_config, *args, **kwargs):
        from vllm.config import get_current_vllm_config
        dtype = get_current_vllm_config().model_config.dtype
        if dtype != torch.float16:
            raise ValueError(
                "DeepSeek-V4 on Turing (SM75) requires --dtype=half; "
                f"got {dtype}."
            )
        ...  # (real init added in Task 10)
```

Then, if `check_if_supports_dtype` still rejects FP16 (it should not — it rejects bf16 only), leave the gate as-is and rely on `--dtype=half`. Only if a BF16 request is unavoidable, gate the bf16 rejection to `not is_turing_target(...)`.

- [ ] **Step 3: Verify**

Run: `.venv/bin/python -c "
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
print('fp8 min cap:', Fp8Config.get_min_capability())
from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4Config
print('mxfp4 min cap:', Mxfp4Config.get_min_capability())
"` 
Expected: prints `75` and `80`. Confirm the DSv4 config is `DeepseekV4FP8Config` (subclass of Fp8Config) so the checkpoint's `deepseek_v4_fp8` method min-capability is 75, not 80 (the MXFP4 min-capability is only used for the standalone `mxfp4` method, which we do not select).

- [ ] **Step 4: Lint + commit**

Run: `pre-commit run ruff-check --all-files`
Expected: clean.

```bash
git add vllm/platforms/cuda.py vllm/models/deepseek_v4/turing/model.py
git commit -m "feat(deepseek_v4): enforce FP16 and correct min-capability on SM75"
```

---

### Task 10: Turing model assembly (DeepseekV4ForCausalLM + layers)

**Files:**
- Modify: `vllm/models/deepseek_v4/turing/model.py` (real implementation)
- Modify: `vllm/models/deepseek_v4/turing/__init__.py`

Goal: assemble the Turing model by porting `xpu/model.py` (1381 lines) with these mechanical changes:
1. `from vllm.models.deepseek_v4.xpu.xpu_sparse import DeepseekV4XPUAttention` → `from vllm.models.deepseek_v4.turing.attention import TuringMLAAttention` (and all `DeepseekV4XPU*` references → `Turing*`).
2. All `torch.bfloat16` literals → `torch.float16`.
3. `torch.ops._xpu_C.*` → the corresponding `common/ops` Triton or torch fallback (list every occurrence; the XPU module uses `_xpu_C` for fused indexer/prefill ops — map each to the shared `common/ops` equivalent).
4. Keep `FusedMoEFactory`, `GateLinear`, `fused_moe_make_expert_params_mapping`, `fused_topk_bias`, RMSNorm, LogitsProcessor, PPMissingLayer, PP-support scaffolding unchanged.
5. Remove or gate the MegaMoE path (`use_mega_moe=False` always on SM75).

- [ ] **Step 1: Port the module**

Copy `vllm/models/deepseek_v4/xpu/model.py` → `vllm/models/deepseek_v4/turing/model.py` and apply the mechanical changes above. Keep the class names `DeepseekV4ForCausalLM` (and module-level `DeepseekV4MLP`).

- [ ] **Step 2: Write the import/smoke test**

```python
# tests/models/decoder_only/vision_language/test_sm75_capability_helpers.py (append)
def test_turing_model_imports():
    from vllm.models.deepseek_v4.turing.model import DeepseekV4ForCausalLM
    assert DeepseekV4ForCausalLM.__name__ == "DeepseekV4ForCausalLM"
```

- [ ] **Step 3: Verify import + lint**

Run: `.venv/bin/python -m pytest tests/models/decoder_only/vision_language/test_sm75_capability_helpers.py -v`
Expected: PASS.

Run: `pre-commit run ruff-check --all-files`
Expected: clean (expect type-ignore annotations matching the xpu originals).

- [ ] **Step 4: Commit**

```bash
git add vllm/models/deepseek_v4/turing/model.py vllm/models/deepseek_v4/turing/__init__.py tests/models/decoder_only/vision_language/test_sm75_capability_helpers.py
git commit -m "feat(deepseek_v4): port XPU model to Turing backend"
```

---

### Task 11: MTP + DSPark on SM75

**Files:**
- Modify: `vllm/models/deepseek_v4/turing/mtp.py`
- Modify: `vllm/models/deepseek_v4/turing/dspark.py`

Goal: MTP uses only portable ops (shared `common/ops/fused_mtp_input_rmsnorm.py`, `save_partial_states.py`, Triton MLA, Marlin FP8/MXFP4). DSPark stays out of scope (raises).

- [ ] **Step 1: Port MTP**

Copy `vllm/models/deepseek_v4/xpu/mtp.py` (525 lines) → `vllm/models/deepseek_v4/turing/mtp.py`, applying the same mechanical changes as Task 10 (XPU attention → Turing attention, bf16 → fp16, `_xpu_C` → shared ops). Keep `DeepSeekV4MTP` class name.

- [ ] **Step 2: Verify import**

Run: `.venv/bin/python -c "from vllm.models.deepseek_v4.turing.mtp import DeepSeekV4MTP; print('ok')"`
Expected: `ok`.

- [ ] **Step 3: Keep DSPark out of scope**

Leave `vllm/models/deepseek_v4/turing/dspark.py` raising `NotImplementedError` (V2-runner + DeepGEMM only). Document in the module docstring.

- [ ] **Step 4: Lint + commit**

Run: `pre-commit run ruff-check --all-files`
Expected: clean.

```bash
git add vllm/models/deepseek_v4/turing/mtp.py vllm/models/deepseek_v4/turing/dspark.py
git commit -m "feat(deepseek_v4): port Turing MTP; keep DSPark out of scope"
```

---

### Task 12: Weight loader + long-context KV sizing

**Files:**
- Test: `tests/models/decoder_only/vision_language/test_sm75_weights.py` (new)
- Create: `vllm/models/deepseek_v4/turing/weights.py`

Goal: confirm weights load without dequantizing (FP8/FP4 stay packed) and the FP16 KV cache sizing fits in the ~17GB headroom for 100K+ context at small batch.

- [ ] **Step 1: Write the test**

```python
# tests/models/decoder_only/vision_language/test_sm75_weights.py
import torch
import pytest


@pytest.mark.skipif(torch.cuda.get_device_capability()[0] != 7, reason="SM75 only")
def test_fp16_kv_bytes_per_token():
    # Turing FP16 KV: per-token bytes for compressed MLA row.
    nope, rope = 512, 128          # dim_qk split (from checkpoint config)
    kv_bytes = (nope + rope) * 2   # fp16
    assert kv_bytes == 1280
```

- [ ] **Step 2: Implement**

`vllm/models/deepseek_v4/turing/weights.py`:

```python
from __future__ import annotations


def kv_bytes_per_token_fp16(nope_head_dim: int, rope_head_dim: int) -> int:
    """Bytes per token in the FP16 MLA KV cache (dense row layout)."""
    return (nope_head_dim + rope_head_dim) * 2
```

- [ ] **Step 3: Verify + commit**

Run: `.venv/bin/python -m pytest tests/models/decoder_only/vision_language/test_sm75_weights.py -v`
Expected: PASS.

```bash
git add tests/models/decoder_only/vision_language/test_sm75_weights.py vllm/models/deepseek_v4/turing/weights.py
git commit -m "feat(deepseek_v4): FP16 KV sizing helper for Turing"
```

---

### Task 13: End-to-end smoke test on 2080 Ti

**Files:**
- Run-only (no code) unless a fix is needed.

Goal: prove the full stack loads and produces tokens on one 2080 Ti.

- [ ] **Step 1: Download the checkpoint**

```bash
huggingface-cli download deepseek-ai/DeepSeek-V4-Flash --local-dir ~/.cache/huggingface/DeepSeek-V4-Flash
```
(If the HF repo is not accessible, use the first layer only to smoke-test the stack.)

- [ ] **Step 2: Read the config**

Run: `.venv/bin/python -c "
import json
cfg = json.load(open('~/.cache/huggingface/DeepSeek-V4-Flash/config.json'))
print({k: cfg[k] for k in ['model_type','hidden_size','num_hidden_layers','num_attention_heads','moe_intermediate_size','compress_ratios','expert_dtype'] if k in cfg})
"` 
Record `compress_ratios` and `head_dim` for kernel tuning. If `expert_dtype` is not in the JSON, it defaults to `fp4`.

- [ ] **Step 3: Launch a tiny server**

```bash
.venv/bin/vllm serve deepseek-ai/DeepSeek-V4-Flash \
  --dtype half \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.90 \
  --enforce-eager \
  --tensor-parallel-size 1 \
  --kv-cache-dtype auto \
  --disable-log-stats
```
Expected: model loads, `engine startup` reaches serving. If it fails, capture the first failing op and return to the relevant task.

- [ ] **Step 4: Completion sanity check**

```bash
curl -s http://localhost:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-ai/DeepSeek-V4-Flash","prompt":"The capital of France is","max_tokens":32}'
```
Expected: a 32-token completion (any coherent or plausible text). Log the generated tokens and wall time per token.

- [ ] **Step 5: Commit any fixes**

If fixes were needed, commit them with the test command and result in the message body.

```bash
git add -A
git commit -m "fix(deepseek_v4): SM75 smoke-test fixes"
```

---

### Task 14: Long-context smoke test (100K+) + correctness record

**Files:**
- Run-only unless fixes needed. Write results to `docs/superpowers/results/2026-08-03-sm75-results.md`.

Goal: prove long-context works and log correctness evidence.

- [ ] **Step 1: Long-context run**

```bash
.venv/bin/vllm serve deepseek-ai/DeepSeek-V4-Flash \
  --dtype half \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.98 \
  --enforce-eager \
  --tensor-parallel-size 1 \
  --kv-cache-dtype auto
```
Send a prompt built from a repeated sentence to exceed 100K tokens (`max_tokens=1`). Record success/failure and GPU memory (nvidia-smi).

- [ ] **Step 2: Correctness spot-check**

Generate the same prompt's top-1 token with `--dtype half` Turing path and, if a reference build exists, with the stock FP8 path; compare. Record both in the results file.

- [ ] **Step 3: Write results**

Create `docs/superpowers/results/2026-08-03-sm75-results.md` with:
- Model/arch/dtype/KV-cache-layout.
- Load time, first-token latency, tok/s (decode), peak VRAM.
- Long-context (100K+) result and any memory limit hit.
- Correctness comparison if available.
- Which kernels are used per layer (Marlin FP8 block, Marlin MXFP4, Triton MLA, indexer fallback, MHC torch).

- [ ] **Step 4: Lint + commit**

```bash
git add docs/superpowers/results/2026-08-03-sm75-results.md
git commit -m "docs: SM75 DeepSeek-V4 results"
```

---

## Self-Review Notes

- **Spec coverage:** every spec component maps to a task: dispatch (2), dense linear (3), FP4 experts (4), sparse MLA (5,6), indexer (7), MHC (8), dtype gate (9), model assembly (10), MTP/DSPark (11), weights/long-context (12), e2e (13), long-context/correctness (14).
- **Open spec questions** are resolved during Task 13 Step 2 (config read) and Task 4 Step 4 (e8m0 scale check); the plan keeps them as explicit verification steps, not placeholders.
- **No placeholders:** every task pins files, code, and commands. Task 6/10 leave the final op-by-op mapping to the implementer reading the real NVIDIA/XPU sources, but name the exact source files to adapt.

# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch 100 — Native FlashInfer FULL CUDA graph for spec-decode.

Backport of vllm#41127 ("Enable native FlashInfer full CUDA graph support
for SpecDec w/out TRT-LLM"). PR open 2026-04-28. Per Sander direct request:
"не ждём, изучаем, импортируем".

================================================================
WHY THIS MATTERS
================================================================

NEW vllm: 27B variants (Minachist INT8 / Lorbus INT4 / gs128) auto-select
FlashInferBackend with fp8_e5m2 KV. With spec-decode (MTP K=3) the backend
falls back to PIECEWISE cudagraph because:

  CUDAGraphMode.FULL_AND_PIECEWISE is not supported with spec-decode for
  attention backend FlashInferBackend (support: UNIFORM_SINGLE_TOKEN_DECODE)

PIECEWISE cudagraph is significantly slower than FULL on Ampere — large
per-step CPU launch overhead.

PR #41127 adds a native FISpecDecode path: when decode bucket has uniform
query_len > 1 (i.e. K+1 spec verify), route through
BatchPrefillWithPagedKVCacheWrapper (instead of decode wrapper) in
cudagraph mode. Verified zero_rows padding gives bit-identical numerics.

Cross-engine note (per agent a91bc4ecd9967da81): SGLang has had this
exact pattern for 1+ year in production
(`python/sglang/srt/layers/attention/flashinfer_backend.py:555-700`).
PR #41127 is vLLM finally catching up.

================================================================
WHAT THIS PATCH DOES (faithful port of #41127)
================================================================

7 sub-patches on `vllm/v1/attention/backends/flashinfer.py`:

  1. **imports** — drop `UniformTypeKVCacheSpecs` (unused after rewrite)
  2. **FISpecDecode dataclass** — new type wrapping
     `BatchPrefillWithPagedKVCacheWrapper`
  3. **FlashInferMetadata.decode union** — extend to include FISpecDecode
  4. **__init__ buffers + dicts** — `_spec_decode_wrapper`,
     `_spec_decode_wrappers_cudagraph`, `spec_decode_qo_indptr`,
     `native_spec_as_decode` flag
  5. **get_cudagraph_support** — return UNIFORM_BATCH unconditionally
     for non-DCP (was: UNIFORM_SINGLE_TOKEN_DECODE if no TRTLLM)
  6. **_get_spec_decode_prefill_wrapper method** — NEW method, lazy
     wrapper allocation cached per padded batch_size
  7. **build() routing** — per-row qo_indptr delta scan + branch on
     query_len: ≤1 → FIDecode (existing), >1 → FISpecDecode (new)
  8. **forward() FISpecDecode case** — call decode_wrapper.run() with
     causal=True instead of FIDecode path

================================================================
EXPECTED IMPACT
================================================================

Per agent analysis on Ampere SM 8.6 (A5000):
- Author claim: +2-3% per-token on SM120
- Ampere has higher CG launch-overhead share → +5-10% expected
- Specifically for 27B INT8/INT4/gs128 with MTP K=3
- Currently 63 TPS sustained → expected 67-70 TPS (modest)
- Combined with potential PIECEWISE→FULL transition: bigger gain at
  high concurrency (max-num-seqs > 1)

NOT applicable to PROD (PROD uses TurboQuantAttentionImpl, not FlashInfer).
applies_to: any backend using FlashInfer + spec-decode + non-DCP.

================================================================
SAFETY MODEL
================================================================

- Default OFF; opt-in via `GENESIS_ENABLE_P100=1`
- Idempotent via marker
- 7 anchor sites — drift detection on each
- DCP guard preserved (BatchDCPPrefillWrapper not wired for CG spec-decode)

Author backport: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
Original PR: vllm#41127. Cross-reference: SGLang flashinfer_backend.py.
"""
from __future__ import annotations

import logging
import os

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import (
    TextPatch,
    TextPatcher,
    TextPatchResult,
)

log = logging.getLogger("genesis.wiring.p100_flashinfer_full_cg_specdec")


GENESIS_P100_MARKER = (
    "Genesis P100 FlashInfer FULL CUDA graph for spec-decode (vllm#41127) v7.62.17"
)


# ─── Sub-patch 1: imports — drop UniformTypeKVCacheSpecs ─────────────────

P100_IMPORTS_OLD = (
    "from vllm.v1.kv_cache_interface import (\n"
    "    AttentionSpec,\n"
    "    KVQuantMode,\n"
    "    UniformTypeKVCacheSpecs,\n"
    ")\n"
)

P100_IMPORTS_NEW = (
    "from vllm.v1.kv_cache_interface import (\n"
    "    AttentionSpec,\n"
    "    KVQuantMode,\n"
    ")\n"
)


# ─── Sub-patch 2: Add FISpecDecode dataclass after FIDecode ──────────────
# Anchor on the FIDecode class definition + closing line + blank +
# next class TRTLLMPrefill — insert FISpecDecode between.

P100_FISPECDECODE_OLD = (
    "@dataclass\n"
    "class FIDecode:\n"
    '    """Metadata for the native FlashInfer decode pathway (non-TRTLLM)."""\n'
    "\n"
    "    wrapper: BatchDecodeWithPagedKVCacheWrapper\n"
    "\n"
    "\n"
    "@dataclass\n"
    "class TRTLLMPrefill:\n"
)

P100_FISPECDECODE_NEW = (
    "@dataclass\n"
    "class FIDecode:\n"
    '    """Metadata for the native FlashInfer decode pathway (non-TRTLLM)."""\n'
    "\n"
    "    wrapper: BatchDecodeWithPagedKVCacheWrapper\n"
    "\n"
    "\n"
    "# [Genesis P100 vllm#41127 backport] FISpecDecode dataclass for native\n"
    "# FlashInfer spec-decode verification through prefill wrapper in CG mode.\n"
    "@dataclass\n"
    "class FISpecDecode:\n"
    '    """Metadata for native FlashInfer spec-decode verification (non-TRTLLM).\n'
    "\n"
    "    Used when the decode bucket has uniform query_len > 1 (1 + num_spec_tokens)\n"
    "    and TRTLLM decode attention is unavailable. Routes through the prefill\n"
    "    wrapper in cudagraph mode with zero_rows padding for padded request slots.\n"
    '    """\n'
    "\n"
    "    wrapper: BatchPrefillWithPagedKVCacheWrapper\n"
    "\n"
    "\n"
    "@dataclass\n"
    "class TRTLLMPrefill:\n"
)


# ─── Sub-patch 3: extend FlashInferMetadata.decode union type ────────────

P100_METADATA_DECODE_OLD = (
    "    decode: FIDecode | TRTLLMDecode | None\n"
)

P100_METADATA_DECODE_NEW = (
    "    # [Genesis P100 vllm#41127 backport] add FISpecDecode variant\n"
    "    decode: FIDecode | FISpecDecode | TRTLLMDecode | None\n"
)


# ─── Sub-patch 4: replace get_cudagraph_support body ─────────────────────
# Anchor on the method signature + comment + the entire body.

P100_CGSUPPORT_OLD = (
    '        """Get the cudagraph support level for FlashInfer attention.\n'
    "\n"
    "        This depends on whether we can use TRTLLM attention for decodes, since we can\n"
    "        only do UNIFORM_SINGLE_TOKEN_DECODE if it is unavailable.\n"
    "        To check this, we must call can_use_trtllm_attention with the number of KV\n"
    "        heads from the kv_cache_spec. We check all available KV cache specs and\n"
    "        only return UNIFORM_BATCH if all of them support TRTLLM attention.\n"
    '        """\n'
    "        # For UniformTypeKVCacheSpecs, check all contained specs\n"
    "        kv_specs = (\n"
    "            kv_cache_spec.kv_cache_specs.values()\n"
    "            if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs)\n"
    "            else [kv_cache_spec]\n"
    "        )\n"
    "        num_qo_heads = vllm_config.model_config.get_num_attention_heads(\n"
    "            vllm_config.parallel_config\n"
    "        )\n"
    "        has_trtllm_support: bool = len(kv_specs) > 0\n"
    "        for spec in kv_specs:\n"
    "            if not isinstance(spec, AttentionSpec):\n"
    "                # FlashInfer only applies to attention, so we don't consider other types\n"
    "                # of KV spec (e.g. Mamba) here. This is mostly for type checking.\n"
    "                continue\n"
    "            if not can_use_trtllm_attention(\n"
    "                num_qo_heads=num_qo_heads,\n"
    "                num_kv_heads=spec.num_kv_heads,\n"
    "            ):\n"
    "                has_trtllm_support = False\n"
    "                break\n"
    "\n"
    "        if has_trtllm_support:\n"
    "            return AttentionCGSupport.UNIFORM_BATCH\n"
    "        else:\n"
    "            return AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE\n"
)

P100_CGSUPPORT_NEW = (
    '        """Get the cudagraph support level for FlashInfer attention.\n'
    "\n"
    "        [Genesis P100 vllm#41127 backport]\n"
    "        Native FlashInfer can capture UNIFORM_BATCH full cudagraphs for\n"
    "        spec-decode by routing uniform query_len > 1 batches through the\n"
    "        prefill wrapper in cudagraph mode (verified zero_rows padding\n"
    "        yields bit-identical real-row numerics). TRTLLM decode attention\n"
    "        is not required for this path.\n"
    "\n"
    "        DCP uses BatchDCPPrefillWrapper which is not wired for cudagraph\n"
    "        spec-decode; downgrade to UNIFORM_SINGLE_TOKEN_DECODE there.\n"
    '        """\n'
    "        if vllm_config.parallel_config.decode_context_parallel_size > 1:\n"
    "            return AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE\n"
    "        return AttentionCGSupport.UNIFORM_BATCH\n"
)


# ─── Sub-patch 5: __init__ — add spec-decode buffers + dicts + flag ─────
# Anchor on the existing block where _decode_wrappers_cudagraph is initialized.

P100_INIT_CGDICT_OLD = (
    "            self._decode_wrappers_cudagraph: dict[\n"
    "                int, BatchDecodeWithPagedKVCacheWrapper\n"
    "            ] = {}\n"
)

P100_INIT_CGDICT_NEW = (
    "            self._decode_wrappers_cudagraph: dict[\n"
    "                int, BatchDecodeWithPagedKVCacheWrapper\n"
    "            ] = {}\n"
    "            # [Genesis P100 vllm#41127 backport] Parallel dict for the\n"
    "            # spec-decode prefill wrapper, keyed by request batch size\n"
    "            # (not token count) because the prefill CUDAGraph wrapper\n"
    "            # fixes batch_size == len(qo_indptr) - 1.\n"
    "            self._spec_decode_wrappers_cudagraph: dict[\n"
    "                int, BatchPrefillWithPagedKVCacheWrapper\n"
    "            ] = {}\n"
)


# Anchor on existing _decode_wrapper = None to add _spec_decode_wrapper.

P100_INIT_DECODE_WRAP_OLD = (
    "        self._decode_wrapper = None  # Wrapper for decode (general shape)\n"
)

P100_INIT_DECODE_WRAP_NEW = (
    "        self._decode_wrapper = None  # Wrapper for decode (general shape)\n"
    "        # [Genesis P100 vllm#41127 backport] Separate prefill-shaped\n"
    "        # wrapper reserved for spec-decode verification so real-prefill\n"
    "        # and spec-decode plan() calls cannot stomp each other inside a\n"
    "        # mixed batch.\n"
    "        self._spec_decode_wrapper: BatchPrefillWithPagedKVCacheWrapper | None = None\n"
)


# Anchor on _init_reorder_batch_threshold. Replace flag computation.

P100_INIT_REORDER_OLD = (
    "        self._init_reorder_batch_threshold(1, supports_spec_as_decode=can_use_trtllm)\n"
)

P100_INIT_REORDER_NEW = (
    "        # [Genesis P100 vllm#41127 backport] Non-DCP native FlashInfer\n"
    "        # can also route spec-decode through the decode bucket by using\n"
    "        # the prefill wrapper in cudagraph mode with zero_rows padding.\n"
    "        # DCP keeps threshold=1 regardless (enforced inside\n"
    "        # _init_reorder_batch_threshold when supports_dcp_with_varlen is False).\n"
    "        _genesis_p100_native_spec_as_decode = self.dcp_world_size <= 1\n"
    "        self._init_reorder_batch_threshold(\n"
    "            1,\n"
    "            supports_spec_as_decode=can_use_trtllm or _genesis_p100_native_spec_as_decode,\n"
    "        )\n"
)


# Anchor on paged_kv_last_page_len buffer creation — add spec_decode_qo_indptr after.

P100_INIT_BUFFER_OLD = (
    "        self.paged_kv_indices = self._make_buffer(max_num_pages)\n"
    "        self.paged_kv_last_page_len = self._make_buffer(max_num_reqs)\n"
)

P100_INIT_BUFFER_NEW = (
    "        self.paged_kv_indices = self._make_buffer(max_num_pages)\n"
    "        self.paged_kv_last_page_len = self._make_buffer(max_num_reqs)\n"
    "        # [Genesis P100 vllm#41127 backport] Persistent qo_indptr buffer\n"
    "        # for the spec-decode prefill wrapper. Sized for the padded request\n"
    "        # count (one extra slot for the inclusive end). Populated by plan()\n"
    "        # each step; the CUDAGraph-mode wrapper holds a fixed-address view\n"
    "        # into this buffer.\n"
    "        self.spec_decode_qo_indptr = self._make_buffer(max_num_reqs + 1)\n"
)


# ─── Sub-patch 6: Add _get_spec_decode_prefill_wrapper method ────────────
# Anchor on the END of _get_decode_wrapper method (its `return decode_wrapper`)
# + the next method to insert between.

P100_NEW_METHOD_OLD = (
    "        return decode_wrapper\n"
    "\n"
    "    def _get_cascade_wrapper(self):\n"
)

P100_NEW_METHOD_NEW = (
    "        return decode_wrapper\n"
    "\n"
    "    # ════════════════════════════════════════════════════════════════\n"
    "    # [Genesis P100 vllm#41127 backport] _get_spec_decode_prefill_wrapper\n"
    "    # ════════════════════════════════════════════════════════════════\n"
    "    def _get_spec_decode_prefill_wrapper(\n"
    "        self, batch_size: int, use_cudagraph: bool = False\n"
    "    ) -> BatchPrefillWithPagedKVCacheWrapper:\n"
    "        \"\"\"Return a BatchPrefillWithPagedKVCacheWrapper for spec-decode.\n"
    "\n"
    "        In cudagraph mode, a separate wrapper is cached per padded request\n"
    "        batch size; the wrapper holds fixed-address views into\n"
    "        `spec_decode_qo_indptr`, `paged_kv_indptr`, `paged_kv_indices`, and\n"
    "        `paged_kv_last_page_len` so that per-step plan() calls only update\n"
    "        buffer contents, not pointers.\n"
    "\n"
    "        `batch_size` is the padded request count, not the token count.\n"
    "        \"\"\"\n"
    "        if use_cudagraph:\n"
    "            wrapper = self._spec_decode_wrappers_cudagraph.get(batch_size, None)\n"
    "        else:\n"
    "            wrapper = self._spec_decode_wrapper\n"
    "\n"
    "        if wrapper is None:\n"
    "            if use_cudagraph:\n"
    "                wrapper = BatchPrefillWithPagedKVCacheWrapper(\n"
    "                    self._get_workspace_buffer(),\n"
    "                    get_kv_cache_layout(),\n"
    "                    use_cuda_graph=True,\n"
    "                    qo_indptr_buf=self.spec_decode_qo_indptr.gpu[: batch_size + 1],\n"
    "                    paged_kv_indptr_buf=self.paged_kv_indptr.gpu[: batch_size + 1],\n"
    "                    paged_kv_indices_buf=self.paged_kv_indices.gpu,\n"
    "                    paged_kv_last_page_len_buf=(\n"
    "                        self.paged_kv_last_page_len.gpu[:batch_size]\n"
    "                    ),\n"
    "                )\n"
    "                self._spec_decode_wrappers_cudagraph[batch_size] = wrapper\n"
    "            else:\n"
    "                wrapper = BatchPrefillWithPagedKVCacheWrapper(\n"
    "                    self._get_workspace_buffer(),\n"
    "                    get_kv_cache_layout(),\n"
    "                )\n"
    "                self._spec_decode_wrapper = wrapper\n"
    "\n"
    "        return wrapper\n"
    "\n"
    "    def _get_cascade_wrapper(self):\n"
)


# ─── Sub-patch 7: build() — replace decode block with per-row scan + branch ──
# Anchor on the EXACT decode block. Replace with query_len detection + branch.

P100_BUILD_OLD = (
    "                num_input_tokens = num_decode_tokens\n"
    "\n"
    "                decode_wrapper = self._get_decode_wrapper(\n"
    "                    num_input_tokens, use_cudagraph\n"
    "                )\n"
    "                # Use the persistent buffer with padding length,\n"
    "                # instead of the same address but chunked version\n"
    "                # in atten_metadata when using cudagraph.\n"
    "                fast_plan_decode(\n"
    "                    decode_wrapper,\n"
    "                    indptr_cpu=self.paged_kv_indptr.cpu[: num_input_tokens + 1],\n"
    "                    indices=paged_kv_indices,\n"
    "                    last_page_len_cpu=self.paged_kv_last_page_len.cpu[\n"
    "                        :num_input_tokens\n"
    "                    ],\n"
    "                    num_qo_heads=self.num_qo_heads * self.dcp_world_size,\n"
    "                    num_kv_heads=self.num_kv_heads,\n"
    "                    head_dim=self.head_dim,\n"
    "                    page_size=self.page_size,\n"
    "                    # Disable flashinfer's pos encoding and use vllm's rope.\n"
    "                    pos_encoding_mode=\"NONE\",\n"
    "                    sm_scale=self.sm_scale,\n"
    "                    window_left=self.window_left,\n"
    "                    logits_soft_cap=self.logits_soft_cap,\n"
    "                    q_data_type=self.q_data_type,\n"
    "                    kv_data_type=self.kv_cache_dtype,\n"
    "                    o_data_type=self.model_config.dtype,\n"
    "                    fixed_split_size=self.decode_fixed_split_size,\n"
    "                    disable_split_kv=self.disable_split_kv,\n"
    "                )\n"
    "                attn_metadata.decode = FIDecode(wrapper=decode_wrapper)\n"
)

P100_BUILD_NEW = (
    "                # ════════════════════════════════════════════════════════════════\n"
    "                # [Genesis P100 vllm#41127 backport] Per-row qo_indptr delta scan\n"
    "                # ════════════════════════════════════════════════════════════════\n"
    "                # require_uniform=True (see split_decodes_and_prefills above)\n"
    "                # guarantees every decode-bucket request has the same\n"
    "                # query_len, except padded slots which carry zero-length rows\n"
    "                # under the zero_rows CG padding strategy. Derive query_len\n"
    "                # from per-request qo_indptr deltas instead of\n"
    "                # num_decode_tokens / num_decodes — the aggregate form\n"
    "                # misroutes mixed real+padded batches (e.g. [5, 5, 0] gives\n"
    "                # 10 % 3 == 1, falsely selecting the FIDecode path).\n"
    "                _genesis_p100_decode_query_lens = (\n"
    "                    qo_indptr_cpu[1 : num_decodes + 1] - qo_indptr_cpu[:num_decodes]\n"
    "                )\n"
    "                _genesis_p100_nonzero = _genesis_p100_decode_query_lens[\n"
    "                    _genesis_p100_decode_query_lens > 0\n"
    "                ]\n"
    "                if _genesis_p100_nonzero.numel() == 0:\n"
    "                    _genesis_p100_query_len = 1\n"
    "                else:\n"
    "                    _genesis_p100_query_len = int(_genesis_p100_nonzero[0].item())\n"
    "\n"
    "                if _genesis_p100_query_len <= 1:\n"
    "                    num_input_tokens = num_decode_tokens\n"
    "\n"
    "                    decode_wrapper = self._get_decode_wrapper(\n"
    "                        num_input_tokens, use_cudagraph\n"
    "                    )\n"
    "                    # Use the persistent buffer with padding length,\n"
    "                    # instead of the same address but chunked version\n"
    "                    # in atten_metadata when using cudagraph.\n"
    "                    fast_plan_decode(\n"
    "                        decode_wrapper,\n"
    "                        indptr_cpu=self.paged_kv_indptr.cpu[: num_input_tokens + 1],\n"
    "                        indices=paged_kv_indices,\n"
    "                        last_page_len_cpu=self.paged_kv_last_page_len.cpu[\n"
    "                            :num_input_tokens\n"
    "                        ],\n"
    "                        num_qo_heads=self.num_qo_heads * self.dcp_world_size,\n"
    "                        num_kv_heads=self.num_kv_heads,\n"
    "                        head_dim=self.head_dim,\n"
    "                        page_size=self.page_size,\n"
    "                        # Disable flashinfer's pos encoding and use vllm's rope.\n"
    "                        pos_encoding_mode=\"NONE\",\n"
    "                        sm_scale=self.sm_scale,\n"
    "                        window_left=self.window_left,\n"
    "                        logits_soft_cap=self.logits_soft_cap,\n"
    "                        q_data_type=self.q_data_type,\n"
    "                        kv_data_type=self.kv_cache_dtype,\n"
    "                        o_data_type=self.model_config.dtype,\n"
    "                        fixed_split_size=self.decode_fixed_split_size,\n"
    "                        disable_split_kv=self.disable_split_kv,\n"
    "                    )\n"
    "                    attn_metadata.decode = FIDecode(wrapper=decode_wrapper)\n"
    "                else:\n"
    "                    # [Genesis P100] Spec-decode: uniform query_len > 1\n"
    "                    # in decode bucket. Route through prefill wrapper in CG mode.\n"
    "                    # zero_rows padding: trailing padded slots have duplicate\n"
    "                    # qo_indptr / paged_kv_indptr entries and last_page_len == 0,\n"
    "                    # which FlashInfer accepts with bit-identical real-row numerics.\n"
    "                    _genesis_p100_spec_wrapper = self._get_spec_decode_prefill_wrapper(\n"
    "                        num_decodes, use_cudagraph\n"
    "                    )\n"
    "                    _genesis_p100_spec_wrapper.plan(\n"
    "                        qo_indptr=qo_indptr_cpu[: num_decodes + 1],\n"
    "                        paged_kv_indptr=self.paged_kv_indptr.cpu[: num_decodes + 1],\n"
    "                        paged_kv_indices=paged_kv_indices,\n"
    "                        paged_kv_last_page_len=self.paged_kv_last_page_len.cpu[\n"
    "                            :num_decodes\n"
    "                        ],\n"
    "                        num_qo_heads=self.num_qo_heads * self.dcp_world_size,\n"
    "                        num_kv_heads=self.num_kv_heads,\n"
    "                        head_dim_qk=self.head_dim,\n"
    "                        page_size=self.page_size,\n"
    "                        causal=True,\n"
    "                        sm_scale=self.sm_scale,\n"
    "                        window_left=self.window_left,\n"
    "                        logits_soft_cap=self.logits_soft_cap,\n"
    "                        q_data_type=self.q_data_type,\n"
    "                        kv_data_type=self.kv_cache_dtype,\n"
    "                        o_data_type=self.model_config.dtype,\n"
    "                        fixed_split_size=self.prefill_fixed_split_size,\n"
    "                        disable_split_kv=self.disable_split_kv,\n"
    "                    )\n"
    "                    attn_metadata.decode = FISpecDecode(wrapper=_genesis_p100_spec_wrapper)\n"
)


# ─── Sub-patch 8: forward() — handle FISpecDecode case ────────────────────

P100_FORWARD_OLD = (
    "            if not decode_use_trtllm:\n"
    "                assert isinstance(attn_metadata.decode, FIDecode)\n"
    "                decode_wrapper = attn_metadata.decode.wrapper\n"
    "                assert decode_wrapper is not None\n"
    "                assert decode_wrapper._window_left == self.window_left\n"
    "                assert decode_wrapper._logits_soft_cap == (self.logits_soft_cap or 0.0)\n"
    "                assert decode_wrapper._sm_scale == self.scale\n"
    "\n"
    "                if use_dcp:\n"
)

P100_FORWARD_NEW = (
    "            if not decode_use_trtllm:\n"
    "                # [Genesis P100 vllm#41127 backport] Allow FISpecDecode too\n"
    "                assert isinstance(attn_metadata.decode, (FIDecode, FISpecDecode))\n"
    "                decode_wrapper = attn_metadata.decode.wrapper\n"
    "                assert decode_wrapper is not None\n"
    "                assert decode_wrapper._window_left == self.window_left\n"
    "                assert decode_wrapper._logits_soft_cap == (self.logits_soft_cap or 0.0)\n"
    "                assert decode_wrapper._sm_scale == self.scale\n"
    "\n"
    "                if isinstance(attn_metadata.decode, FISpecDecode):\n"
    "                    # [Genesis P100] Spec-decode verification through\n"
    "                    # prefill wrapper. Non-DCP only — DCP downgrades CG\n"
    "                    # support to UNIFORM_SINGLE_TOKEN_DECODE upstream.\n"
    "                    assert not use_dcp, (\n"
    "                        \"FISpecDecode is not supported under DCP\"\n"
    "                    )\n"
    "                    assert decode_wrapper._causal\n"
    "                    decode_wrapper.run(\n"
    "                        decode_query,\n"
    "                        kv_cache_permute,\n"
    "                        k_scale=layer._k_scale_float,\n"
    "                        v_scale=layer._v_scale_float,\n"
    "                        out=output[:num_decode_tokens],\n"
    "                    )\n"
    "                elif use_dcp:\n"
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("v1/attention/backends/flashinfer.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="P100 flashinfer.py — native FULL CG for spec-decode (vllm#41127)",
        target_file=str(target),
        marker=GENESIS_P100_MARKER,
        sub_patches=[
            TextPatch(
                name="p100_imports_drop_uniform",
                anchor=P100_IMPORTS_OLD,
                replacement=P100_IMPORTS_NEW,
                required=True,
            ),
            TextPatch(
                name="p100_fispecdecode_dataclass",
                anchor=P100_FISPECDECODE_OLD,
                replacement=P100_FISPECDECODE_NEW,
                required=True,
            ),
            TextPatch(
                name="p100_metadata_decode_union",
                anchor=P100_METADATA_DECODE_OLD,
                replacement=P100_METADATA_DECODE_NEW,
                required=True,
            ),
            TextPatch(
                name="p100_cgsupport_uniform_batch",
                anchor=P100_CGSUPPORT_OLD,
                replacement=P100_CGSUPPORT_NEW,
                required=True,
            ),
            TextPatch(
                name="p100_init_decode_wrap_field",
                anchor=P100_INIT_DECODE_WRAP_OLD,
                replacement=P100_INIT_DECODE_WRAP_NEW,
                required=True,
            ),
            TextPatch(
                name="p100_init_cgdict",
                anchor=P100_INIT_CGDICT_OLD,
                replacement=P100_INIT_CGDICT_NEW,
                required=True,
            ),
            TextPatch(
                name="p100_init_reorder_threshold",
                anchor=P100_INIT_REORDER_OLD,
                replacement=P100_INIT_REORDER_NEW,
                required=True,
            ),
            TextPatch(
                name="p100_init_qo_indptr_buffer",
                anchor=P100_INIT_BUFFER_OLD,
                replacement=P100_INIT_BUFFER_NEW,
                required=True,
            ),
            TextPatch(
                name="p100_get_spec_decode_prefill_wrapper_method",
                anchor=P100_NEW_METHOD_OLD,
                replacement=P100_NEW_METHOD_NEW,
                required=True,
            ),
            TextPatch(
                name="p100_build_query_len_scan_branch",
                anchor=P100_BUILD_OLD,
                replacement=P100_BUILD_NEW,
                required=True,
            ),
            TextPatch(
                name="p100_forward_fispecdecode_case",
                anchor=P100_FORWARD_OLD,
                replacement=P100_FORWARD_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "[Genesis P100",
            # Upstream-side markers if vllm#41127 (or equivalent) merges:
            "FISpecDecode",
            "spec_decode_qo_indptr",
            "_get_spec_decode_prefill_wrapper",
            "_genesis_p100_native_spec_as_decode",
        ],
    )


def apply() -> tuple[str, str]:
    """Apply P100 v1 — FlashInfer FULL CG for spec-decode (vllm#41127 backport).

    Full 11-sub-patch backport. When applied, FlashInfer backend with
    spec-decode + non-DCP gets FULL cudagraph capture (was PIECEWISE).

    Composability:
    - PROD (TurboQuantAttentionImpl backend) — NOT affected, P100 only
      patches FlashInferImpl. Co-exists with P67/P67b/P98/P99.
    - 27B variants (FlashInfer backend with fp8_e5m2 KV) — directly
      benefits from FULL cudagraph routing.

    Expected: +5-10% TPS on Ampere SM 8.6 (per agent estimate vs author's
    +2-3% on SM120). 27B 63 → 67-70 TPS.
    """
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("P100")
    log_decision("P100", decision, reason)
    if not decision:
        return "skipped", reason

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "flashinfer.py not found"

    if not os.path.isfile(patcher.target_file):
        return "skipped", f"target disappeared: {patcher.target_file}"
    with open(patcher.target_file) as f:
        content = f.read()
    if patcher.marker in content:
        log.info("[P100] marker present — skip (idempotent)")
        return "applied", "idempotent (marker present)"
    for m in patcher.upstream_drift_markers:
        if m.startswith("[Genesis"):
            continue
        if m in content:
            return (
                "skipped",
                f"upstream drift marker {m!r} in {patcher.target_file} "
                "— upstream PR #41127 (or equivalent) appears merged",
            )

    result, failure = patcher.apply()
    # Audit P1 fix 2026-05-05: 11-subpatch hotfix MUST surface SKIPPED honestly
    # — was the highest-blast-radius silent-mask in the original 35-file set.
    from vllm._genesis.wiring.text_patch import result_to_wiring_status
    return result_to_wiring_status(
        result, failure,
        applied_message=(
            "P100 v7.62.17 applied: 11 sub-patches on flashinfer.py for native "
            "FULL CUDA graph + spec-decode without TRTLLM. 27B variants now "
            "get UNIFORM_BATCH cudagraph (was PIECEWISE) for K+1 spec-verify. "
            "Expected: +5-10% TPS on Ampere SM 8.6. NO-OP for PROD (TQ backend). "
            "Composes with P67/P67b/P98/P99 (different backends)."
        ),
        patch_name=patcher.patch_name,
    )


def is_applied() -> bool:
    if vllm_install_root() is None:
        return False
    patcher = _make_patcher()
    if patcher is None:
        return False
    try:
        with open(patcher.target_file) as f:
            return patcher.marker in f.read()
    except Exception:
        return False

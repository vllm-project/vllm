# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Test MLA AITER fused kernels for DeepSeek-V3 - COMPREHENSIVE VERSION.

This test validates the AITER fused kernel implementation with thorough comparisons:
1. Determinism check (same input → same output)
2. Fused vs Unfused correctness (same input → similar outputs)

Test scenarios:
1. Pure prefill (16 prefill tokens, 0 decode tokens)
2. Pure decode (0 prefill tokens, 16 decode tokens)
3. Mixed batch (7 prefill tokens, 9 decode tokens)

For each scenario:
- Phase 1: Run unfused path with random input A
- Phase 2: Run unfused path with SAME input A - verify outputs are identical (determinism)
- Phase 3: Run fused path with same input A
- Phase 4: Compare all results

Comparisons:
- Unfused run 1 vs run 2: Outputs MUST be identical (proves determinism)
- Fused vs Unfused: Outputs should MATCH within tolerance (proves correctness)
"""

import os
from unittest.mock import Mock

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm() or not torch.cuda.is_available(),
    reason="Requires ROCm GPU",
)


class TestMLAAiterFusedKernel:
    """Test MLA AITER fused kernels with comprehensive comparisons."""

    @pytest.mark.parametrize(
        "scenario,num_prefill,num_decode,prefill_history",
        [
            ("Pure Prefill", 16, 0, 0),
            (
                "Pure Decode",
                0,
                16,
                16,
            ),  # 16 decode tokens attending to 16 prefill history
            ("Mixed Batch", 7, 9, 0),
        ],
    )
    def test_aiter_fused_vs_unfused_comprehensive(
        self, default_vllm_config, scenario, num_prefill, num_decode, prefill_history
    ):
        """
        Comprehensive test comparing fused and unfused MLA attention.

        Phase 1: Unfused path with input A (run 1)
        Phase 2: Unfused path with input A (run 2) - verify determinism
        Phase 3: Fused path with input A
        Phase 4: Comparisons
          - Unfused run 1 vs run 2 (must be identical - determinism)
          - Fused vs Unfused (should match within tolerance - correctness)
        """
        import torch.nn as nn

        from vllm.config import CacheConfig, ModelConfig
        from vllm.forward_context import set_forward_context
        from vllm.model_executor.layers.mla import (
            MLAModules,
            MultiHeadLatentAttentionWrapper,
        )
        from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
        from vllm.v1.attention.backend import AttentionMetadata

        os.environ["VLLM_ROCM_USE_AITER"] = "1"
        os.environ["VLLM_USE_AITER_FUSED"] = "1"

        # CRITICAL: Reload env variables after setting them
        from vllm._aiter_ops import rocm_aiter_ops

        rocm_aiter_ops.refresh_env_variables()

        print("\n" + "=" * 80)
        print(f"[Test] Scenario: {scenario}")
        print(f"       Prefill tokens: {num_prefill}, Decode tokens: {num_decode}")
        if prefill_history > 0:
            print(f"       Prefill history for decode: {prefill_history} tokens")
        print("=" * 80)

        device = "cuda"
        dtype = torch.bfloat16
        num_tokens = num_prefill + num_decode

        # Create proper model_config with bfloat16 dtype for backend selection
        if default_vllm_config is None:
            default_vllm_config = Mock()
        if not hasattr(default_vllm_config, "model_config"):
            default_vllm_config.model_config = Mock(spec=ModelConfig)
        default_vllm_config.model_config.dtype = dtype

        # Config (DeepSeek-V3 dimensions)
        hidden_size = 2048
        num_heads = 128
        qk_nope_head_dim = 128
        qk_rope_head_dim = 64
        qk_head_dim = 192
        v_head_dim = 128
        q_lora_rank = 1536
        kv_lora_rank = 512
        scale = qk_head_dim**-0.5

        print("\nStep 1: Creating wrapper components...")

        # Create RoPE
        rotary_emb = RotaryEmbedding(
            head_size=qk_rope_head_dim,
            rotary_dim=qk_rope_head_dim,
            max_position_embeddings=8192,
            base=10000.0,
            is_neox_style=True,
            dtype=dtype,
        ).to(device)

        # Create projections that return tuples
        class LinearWithTupleOutput(nn.Module):
            def __init__(self, in_features, out_features, bias=False):
                super().__init__()
                self.linear = nn.Linear(in_features, out_features, bias=bias)
                self.quant_method = None

            def forward(self, x):
                return (self.linear(x), None)

            @property
            def weight(self):
                return self.linear.weight

        # DeepSeek-V3 uses q_lora_rank path
        fused_qkv_a_proj = LinearWithTupleOutput(
            hidden_size, q_lora_rank + kv_lora_rank + qk_rope_head_dim, bias=False
        ).to(device, dtype)
        q_a_layernorm = nn.RMSNorm(q_lora_rank, eps=1e-6).to(device, dtype)
        q_b_proj = LinearWithTupleOutput(
            q_lora_rank, num_heads * qk_head_dim, bias=False
        ).to(device, dtype)
        kv_a_layernorm = nn.RMSNorm(kv_lora_rank, eps=1e-6).to(device, dtype)
        kv_b_proj = LinearWithTupleOutput(
            kv_lora_rank, num_heads * (qk_nope_head_dim + v_head_dim), bias=False
        ).to(device, dtype)
        o_proj = LinearWithTupleOutput(
            num_heads * v_head_dim, hidden_size, bias=False
        ).to(device, dtype)

        mla_modules = MLAModules(
            kv_a_proj_with_mqa=None,
            kv_a_layernorm=kv_a_layernorm,
            kv_b_proj=kv_b_proj,
            rotary_emb=rotary_emb,
            o_proj=o_proj,
            fused_qkv_a_proj=fused_qkv_a_proj,
            q_a_layernorm=q_a_layernorm,
            q_b_proj=q_b_proj,
            q_proj=None,
            indexer=None,
            is_sparse=False,
            topk_indices_buffer=None,
        )

        cache_config = CacheConfig(block_size=16, cache_dtype="fp8")

        print("  ✓ Components created")

        # Create KV cache and slot mapping
        print("\nStep 2: Creating KV cache and slot mapping...")
        # Need enough slots for prefill history + current tokens
        num_blocks = (prefill_history + num_tokens + 15) // 16
        kv_cache_dim = kv_lora_rank + qk_rope_head_dim
        num_kv_heads = 1  # MLA uses 1 KV head

        # AITER fused kernel expects FLAT format
        total_slots = num_blocks * 16
        # Decode tokens write to slots AFTER prefill history
        slot_mapping = torch.arange(
            prefill_history,
            prefill_history + num_tokens,
            device=device,
            dtype=torch.long,
        )
        print("[TEST DEBUG] Created slot_mapping:")
        print(f"  slot_mapping shape: {slot_mapping.shape}")
        print(f"  slot_mapping values: {slot_mapping.cpu().numpy()}")
        print(
            f"  ✓ slot_mapping: {slot_mapping.shape} (slots [{prefill_history}:{prefill_history + num_tokens}])"
        )

        # Set PyTorch default dtype to bfloat16
        original_default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(dtype)

        # Create wrapper
        print("\nStep 3: Creating wrapper...")
        wrapper = MultiHeadLatentAttentionWrapper(
            hidden_size=hidden_size,
            num_heads=num_heads,
            scale=scale,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            mla_modules=mla_modules,
            cache_config=cache_config,
            quant_config=None,
            prefix="test_layer",
        ).to(device)

        from vllm import envs
        from vllm.platforms import current_platform

        print(f"  ✓ Backend: {wrapper.mla_attn.attn_backend.get_name()}")
        print(f"  ✓ Impl type: {type(wrapper.mla_attn.impl).__name__}")

        # FORCE use_direct_call = True
        wrapper.mla_attn.use_direct_call = True

        # RECOMPUTE use_aiter_fused_decode
        wrapper.mla_attn.use_aiter_fused_decode = (
            wrapper.mla_attn.use_direct_call
            and current_platform.is_rocm()
            and wrapper.mla_attn.is_aiter_triton_fp8_bmm_enabled
            and envs.VLLM_USE_AITER_FUSED
            and wrapper.mla_attn.cos_cache is not None
            and wrapper.mla_attn.sin_cache is not None
        )
        print(f"  ✓ use_aiter_fused_decode = {wrapper.mla_attn.use_aiter_fused_decode}")

        # Import fused kernel
        if wrapper.mla_attn.use_aiter_fused_decode:
            if wrapper.mla_attn.is_aiter_triton_fp8_bmm_enabled:
                from aiter.ops.triton.fusions.fused_bmm_rope_kv_cache import (
                    fused_fp8_bmm_rope_cat_and_cache_mla,
                )

                wrapper.mla_attn._fused_decode_kernel = (
                    fused_fp8_bmm_rope_cat_and_cache_mla
                )
                wrapper.mla_attn._fused_kernel_type = "fp8"

        # Initialize DCP attributes
        if wrapper.mla_attn.impl.dcp_world_size == -1:
            wrapper.mla_attn.impl.dcp_world_size = 1

        # Process weights
        wrapper.mla_attn.process_weights_after_loading(act_dtype=dtype)

        torch.set_default_dtype(original_default_dtype)

        # Set up vllm_config
        print("\nStep 4: Setting up vllm_config...")
        mock_attn_layer = Mock()
        mock_attn_layer.impl = wrapper.mla_attn.impl
        mock_attn_layer._real_mla_attn = wrapper.mla_attn

        if default_vllm_config is None:
            default_vllm_config = Mock()
        if (
            not hasattr(default_vllm_config, "compilation_config")
            or default_vllm_config.compilation_config is None
        ):
            default_vllm_config.compilation_config = Mock()

        default_vllm_config.compilation_config.static_forward_context = {
            "test_layer.attn": mock_attn_layer
        }
        default_vllm_config.compilation_config.fast_moe_cold_start = False

        if (
            not hasattr(default_vllm_config, "parallel_config")
            or default_vllm_config.parallel_config is None
        ):
            default_vllm_config.parallel_config = Mock()
        default_vllm_config.parallel_config.data_parallel_size = 1
        default_vllm_config.parallel_config.is_moe_model = False

        # Create inputs
        print("\nStep 5: Creating inputs...")
        positions = torch.randint(
            0, 1000, (num_tokens,), device=device, dtype=torch.long
        )
        hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)

        # Create fixed prefill history (for determinism in Pure Decode scenario)
        if prefill_history > 0:
            prefill_hidden_states_for_history = torch.randn(
                prefill_history, hidden_size, device=device, dtype=dtype
            )
        else:
            prefill_hidden_states_for_history = None

        # Create metadata
        print("\nStep 6: Setting up attention metadata...")
        mock_metadata = Mock(spec=AttentionMetadata)
        mock_metadata.num_decode_tokens = num_decode
        mock_metadata.num_prefill_tokens = num_prefill
        mock_metadata.num_actual_tokens = num_tokens
        mock_metadata.num_decodes = num_decode
        mock_metadata.num_prefills = num_prefill

        if num_prefill > 0:
            mock_prefill_metadata = Mock()
            mock_prefill_metadata.q_data_type = dtype
            mock_prefill_metadata.chunked_context = None
            mock_prefill_metadata.max_query_len = num_prefill
            mock_prefill_metadata.query_start_loc = torch.tensor(
                [0, num_prefill], dtype=torch.int32, device=device
            )
            mock_metadata.prefill = mock_prefill_metadata
        else:
            mock_metadata.prefill = None

        if num_decode > 0:
            mock_decode_metadata = Mock()
            # Each decode token attends to:
            # 1. prefill_history (pre-populated KV cache from previous runs)
            # 2. num_prefill (prefill tokens in THIS batch)
            # 3. +1 (itself)
            # For Mixed Batch: decode tokens attend to prefill tokens in same batch + themselves
            decode_seq_len = prefill_history + num_prefill + 1
            mock_decode_metadata.block_table = torch.zeros(
                (num_decode, 1), dtype=torch.int32, device=device
            )
            mock_decode_metadata.seq_lens = torch.full(
                (num_decode,), decode_seq_len, dtype=torch.int32, device=device
            )
            mock_decode_metadata.dcp_tot_seq_lens = None

            # Set up paging indices to point to the correct cache slots
            # Slots layout:
            #   [0:prefill_history] - pre-populated history
            #   [prefill_history:prefill_history+num_prefill] - prefill tokens from THIS batch
            #   [prefill_history+num_prefill:...] - decode tokens
            paged_kv_indptr = []
            paged_kv_indices = []
            paged_kv_last_page_len = []

            for i in range(num_decode):
                # This decode token attends to:
                # 1. All prefill history slots [0:prefill_history]
                # 2. All prefill tokens from this batch [prefill_history:prefill_history+num_prefill]
                # 3. Itself at [prefill_history+num_prefill+i]
                start_idx = len(paged_kv_indices)

                # Add prefill history slots
                for j in range(prefill_history):
                    paged_kv_indices.append(j)

                # Add prefill tokens from THIS batch
                for j in range(num_prefill):
                    paged_kv_indices.append(prefill_history + j)

                # Add this decode token's slot
                paged_kv_indices.append(prefill_history + num_prefill + i)

                paged_kv_indptr.append(start_idx)
                paged_kv_last_page_len.append(1)  # Last page has 1 token

            paged_kv_indptr.append(len(paged_kv_indices))  # Final entry

            mock_decode_metadata.paged_kv_indptr = torch.tensor(
                paged_kv_indptr, dtype=torch.int32, device=device
            )
            mock_decode_metadata.paged_kv_indices = torch.tensor(
                paged_kv_indices, dtype=torch.int32, device=device
            )
            mock_decode_metadata.paged_kv_last_page_len = torch.tensor(
                paged_kv_last_page_len, dtype=torch.int32, device=device
            )
            mock_decode_metadata.qo_indptr = torch.arange(
                num_decode + 1, dtype=torch.int32, device=device
            )
            mock_decode_metadata.attn_out_dtype = dtype
            mock_decode_metadata.max_qo_len = 1
            mock_metadata.decode = mock_decode_metadata

            print("\n[DEBUG] Decode metadata setup:")
            print(f"  seq_lens (each decode attends to): {decode_seq_len} tokens")
            print(
                f"    = {prefill_history} (history) + {num_prefill} (prefill in batch) + 1 (self)"
            )
            print(f"  paged_kv_indptr[:5]: {paged_kv_indptr[:5]}")
            print(f"  paged_kv_indices[:20]: {paged_kv_indices[:20]}")
        else:
            mock_metadata.decode = None

        # ==================================================================
        # PHASE 0: POPULATE KV CACHE WITH PREFILL HISTORY (if needed)
        # ==================================================================
        def populate_kv_cache_with_prefill(kv_cache, use_fused, phase_name):
            """Populate KV cache with prefill_history tokens."""
            if prefill_history == 0:
                return

            print(
                f"\n[SETUP {phase_name}] Populating KV cache with {prefill_history} prefill history tokens (for decode to attend to)..."
            )
            print(f"  use_fused: {use_fused}")
            print(
                f"  prefill_hidden_states_for_history id: {id(prefill_hidden_states_for_history)}"
            )
            print(
                f"  prefill_hidden_states_for_history[0,:5]: {prefill_hidden_states_for_history[0, :5].cpu().float().numpy()}"
            )

            # Create prefill-only inputs (use fixed hidden states for determinism)
            prefill_positions = torch.arange(
                prefill_history, device=device, dtype=torch.long
            )
            prefill_slot_mapping = torch.arange(
                prefill_history, device=device, dtype=torch.long
            )

            print(
                f"  prefill_positions: {prefill_positions[:5].cpu().numpy()} ... (total {len(prefill_positions)})"
            )
            print(
                f"  prefill_slot_mapping: {prefill_slot_mapping[:5].cpu().numpy()} ... (total {len(prefill_slot_mapping)})"
            )

            # Create prefill-only metadata
            prefill_only_metadata = Mock(spec=AttentionMetadata)
            prefill_only_metadata.num_decode_tokens = 0
            prefill_only_metadata.num_prefill_tokens = prefill_history
            prefill_only_metadata.num_actual_tokens = prefill_history
            prefill_only_metadata.num_decodes = 0
            prefill_only_metadata.num_prefills = prefill_history
            prefill_only_metadata.decode = None

            prefill_meta = Mock()
            prefill_meta.q_data_type = dtype
            prefill_meta.chunked_context = None
            prefill_meta.max_query_len = prefill_history
            prefill_meta.query_start_loc = torch.tensor(
                [0, prefill_history], dtype=torch.int32, device=device
            )
            prefill_only_metadata.prefill = prefill_meta

            # Set KV cache
            mock_attn_layer.kv_cache = {0: kv_cache}
            wrapper.mla_attn.kv_cache[0] = kv_cache
            wrapper.mla_attn.use_aiter_fused_decode = use_fused

            # Check KV cache BEFORE setup
            kv_cache_before = kv_cache.cpu().view(torch.uint8).numpy()
            non_zero_before = (kv_cache_before != 0).sum()
            print("\n  KV cache BEFORE setup:")
            print(f"    Non-zero bytes: {non_zero_before}/{kv_cache_before.size}")

            # Run prefill forward pass
            with (
                torch.no_grad(),
                set_forward_context(
                    attn_metadata=prefill_only_metadata,
                    vllm_config=default_vllm_config,
                    virtual_engine=0,
                    slot_mapping={"test_layer.attn": prefill_slot_mapping},
                ),
            ):
                _ = wrapper.forward(
                    positions=prefill_positions,
                    hidden_states=prefill_hidden_states_for_history,
                )

            # Check KV cache AFTER setup
            kv_cache_after = kv_cache.cpu().view(torch.uint8).numpy()
            non_zero_after = (kv_cache_after != 0).sum()
            print("\n  KV cache AFTER setup:")
            print(f"    Non-zero bytes: {non_zero_after}/{kv_cache_after.size}")
            print(f"    Slots written: {non_zero_after // (kv_cache.shape[-1])}")

            # Show first few slots
            for slot_idx in range(min(3, prefill_history)):
                slot_data = kv_cache_after[slot_idx, 0, :]
                kv_c_part = slot_data[:kv_lora_rank]
                k_pe_part = slot_data[kv_lora_rank : kv_lora_rank + qk_rope_head_dim]
                print(f"    Slot {slot_idx}:")
                print(f"      kv_c (first 5): {kv_c_part[:5]}")
                print(f"      k_pe (first 5): {k_pe_part[:5]}")

            print(f"\n[SETUP {phase_name}] ✓ Done. Now running actual test...\n")

        # ==================================================================
        # PHASE 1: UNFUSED PATH (First Run)
        # ==================================================================
        print(f"\n{'=' * 80}")
        print("PHASE 1: UNFUSED PATH (Run 1)")
        print(f"{'=' * 80}")

        # DEBUG: Check hidden_states at start of Phase 1
        print("\n[DEBUG PHASE 1 START] hidden_states check:")
        print(f"  hidden_states id: {id(hidden_states)}")
        print(f"  hidden_states data_ptr: {hidden_states.data_ptr()}")
        print(f"  hidden_states[0,:10]: {hidden_states[0, :10].cpu().float().numpy()}")

        # Create fresh KV cache for unfused run 1
        kv_cache_unfused_1 = torch.zeros(
            total_slots, num_kv_heads, kv_cache_dim, device=device, dtype=torch.uint8
        )

        # Disable fused kernel
        original_use_aiter_fused = wrapper.mla_attn.use_aiter_fused_decode
        wrapper.mla_attn.use_aiter_fused_decode = False

        # Populate KV cache with prefill history (if needed)
        print("\n>>> SETUP START: Populating KV cache for prefill history <<<")
        populate_kv_cache_with_prefill(
            kv_cache_unfused_1, use_fused=False, phase_name="UNFUSED RUN 1"
        )

        print(">>> SETUP END: KV cache populated <<<\n")

        # Capture KV cache state after setup (baseline for slot comparison)
        kv_cache_after_setup_unfused = kv_cache_unfused_1.clone().detach()

        mock_attn_layer.kv_cache = {0: kv_cache_unfused_1}
        wrapper.mla_attn.kv_cache[0] = kv_cache_unfused_1

        captured_unfused_1 = {}

        def capture_unfused_1(q, kv_c_normed, k_pe, **kwargs):
            # Deep copy inputs (clone + detach)
            captured_unfused_1["q_before_bmm"] = q.clone().detach()
            captured_unfused_1["kv_c_normed"] = kv_c_normed.clone().detach()
            captured_unfused_1["k_pe"] = k_pe.clone().detach()

            # DEBUG: Print Q before Flash Attention
            print("\n[DEBUG UNFUSED RUN 1] Q before Flash Attention:")
            print(f"  q shape: {q.shape}")
            print(f"  q[0,0,:5]: {q[0, 0, :5].cpu().float().numpy()}")
            print(f"  k_pe[0,0,:5]: {k_pe[0, 0, :5].cpu().float().numpy()}")

            # Enable test capture
            test_storage = {}
            wrapper.mla_attn._test_q_after_rope_storage = test_storage

            # Call forward
            result = wrapper.mla_attn.__class__.forward(
                wrapper.mla_attn, q, kv_c_normed, k_pe, **kwargs
            )

            # Capture Q after forward (to see if BMM modified it)
            captured_unfused_1["q_after_bmm"] = q.clone().detach()

            # Capture test storage
            captured_unfused_1["_test_storage"] = test_storage.copy()

            return result

        original_forward = wrapper.mla_attn.forward
        wrapper.mla_attn.forward = capture_unfused_1

        with (
            torch.no_grad(),
            set_forward_context(
                attn_metadata=mock_metadata,
                vllm_config=default_vllm_config,
                virtual_engine=0,
                slot_mapping={"test_layer.attn": slot_mapping},
            ),
        ):
            print("\n>>> RUN EXECUTING: UNFUSED PATH RUN 1 <<<\n")
            output_unfused_1 = wrapper.forward(
                positions=positions, hidden_states=hidden_states
            )

        wrapper.mla_attn.forward = original_forward

        # Capture deep copies (clone + detach from computation graph)
        captured_unfused_1["output"] = output_unfused_1.clone().detach()
        captured_unfused_1["kv_cache"] = kv_cache_unfused_1.clone().detach()

        # DEBUG: Check KV cache after population
        print("[DEBUG] PHASE 1 KV cache after prefill history population:")
        kv_cache_before_decode = kv_cache_unfused_1.clone()
        print(f"  kv_cache[0,0,:5]: {kv_cache_unfused_1[0, 0, :5].cpu().numpy()}")
        print(f"  kv_cache[5,0,:5]: {kv_cache_unfused_1[5, 0, :5].cpu().numpy()}")
        print(
            f"  kv_cache non-zero: {(kv_cache_unfused_1 != 0).sum().item()} / {kv_cache_unfused_1.numel()}"
        )

        # Check which slots were written during decode
        kv_diff_decode = kv_cache_unfused_1 != kv_cache_before_decode
        print("\n[DEBUG PHASE 1] Slots written during UNFUSED decode:")
        for slot in range(min(32, kv_cache_unfused_1.shape[0])):
            if kv_diff_decode[slot].sum().item() > 0:
                print(
                    f"  Slot {slot}: {kv_diff_decode[slot].sum().item()} bytes written"
                )

        print("  ✓ Unfused run 1 complete")
        print(f"    Output shape: {output_unfused_1.shape}")
        print(f"    Q shape: {captured_unfused_1['q_after_bmm'].shape}")
        print(
            f"    KV cache non-zero: {(kv_cache_unfused_1 != 0).sum().item()} / {kv_cache_unfused_1.numel()}"
        )

        # ==================================================================
        # PHASE 2: UNFUSED PATH (Second Run - SAME Input for Determinism Check)
        # ==================================================================
        print(f"\n{'=' * 80}")
        print("PHASE 2: UNFUSED PATH (Run 2 - SAME Input, Determinism Check)")
        print(f"{'=' * 80}")

        # Create fresh KV cache for unfused run 2
        kv_cache_unfused_2 = torch.zeros(
            total_slots, num_kv_heads, kv_cache_dim, device=device, dtype=torch.uint8
        )

        # Populate KV cache with SAME prefill history as run 1 (for determinism)
        populate_kv_cache_with_prefill(
            kv_cache_unfused_2, use_fused=False, phase_name="UNFUSED RUN 2"
        )

        print(">>> SETUP END: KV cache populated <<<\n")

        mock_attn_layer.kv_cache = {0: kv_cache_unfused_2}
        wrapper.mla_attn.kv_cache[0] = kv_cache_unfused_2

        captured_unfused_2 = {}

        def capture_unfused_2(q, kv_c_normed, k_pe, **kwargs):
            # Deep copy inputs (clone + detach)
            captured_unfused_2["q_before_bmm"] = q.clone().detach()
            captured_unfused_2["kv_c_normed"] = kv_c_normed.clone().detach()
            captured_unfused_2["k_pe"] = k_pe.clone().detach()

            # DEBUG: Print Q before Flash Attention
            print("\n[DEBUG UNFUSED RUN 2] Q before Flash Attention:")
            print(f"  q shape: {q.shape}")
            print(f"  q[0,0,:5]: {q[0, 0, :5].cpu().float().numpy()}")
            print(f"  k_pe[0,0,:5]: {k_pe[0, 0, :5].cpu().float().numpy()}")

            # Call forward
            result = wrapper.mla_attn.__class__.forward(
                wrapper.mla_attn, q, kv_c_normed, k_pe, **kwargs
            )

            # Capture Q after forward (to see if BMM modified it)
            captured_unfused_2["q_after_bmm"] = q.clone().detach()

            return result

        wrapper.mla_attn.forward = capture_unfused_2

        print("\n[DEBUG] About to run PHASE 2 forward pass...")
        print(f"  hidden_states id: {id(hidden_states)}")
        print(f"  hidden_states[0,:5]: {hidden_states[0, :5].cpu().float().numpy()}")

        with torch.no_grad():
            with set_forward_context(
                attn_metadata=mock_metadata,
                vllm_config=default_vllm_config,
                virtual_engine=0,
                slot_mapping={"test_layer.attn": slot_mapping},
            ):
                print("\n>>> RUN EXECUTING: UNFUSED PATH RUN 2 <<<\n")
                output_unfused_2 = wrapper.forward(
                    positions=positions,
                    hidden_states=hidden_states,  # Use SAME input as run 1 (determinism check)
                )

        print("[DEBUG] PHASE 2 forward completed")
        print(f"  output_unfused_2 id: {id(output_unfused_2)}")
        print(
            f"  output_unfused_2[0,:5]: {output_unfused_2[0, :5].cpu().float().numpy()}"
        )

        wrapper.mla_attn.forward = original_forward

        # Capture deep copies (clone + detach from computation graph)
        captured_unfused_2["output"] = output_unfused_2.clone().detach()
        captured_unfused_2["kv_cache"] = kv_cache_unfused_2.clone().detach()

        # DEBUG: Check KV cache after population
        print("[DEBUG] PHASE 2 KV cache after prefill history population:")
        print(f"  kv_cache[0,0,:5]: {kv_cache_unfused_2[0, 0, :5].cpu().numpy()}")
        print(f"  kv_cache[5,0,:5]: {kv_cache_unfused_2[5, 0, :5].cpu().numpy()}")
        print(
            f"  kv_cache non-zero: {(kv_cache_unfused_2 != 0).sum().item()} / {kv_cache_unfused_2.numel()}"
        )

        # Compare with Phase 1 KV cache
        kv_diff_bytes = kv_cache_unfused_2 != captured_unfused_1["kv_cache"]
        kv_diff_float = torch.abs(
            kv_cache_unfused_2.float() - captured_unfused_1["kv_cache"].float()
        )
        print("\n[DEBUG] Detailed KV cache comparison (Phase 2 vs Phase 1):")
        print(f"  Total elements: {kv_cache_unfused_2.numel()}")
        print(
            f"  Differing bytes: {kv_diff_bytes.sum().item()} / {kv_cache_unfused_2.numel()}"
        )
        print(f"  Max diff: {kv_diff_float.max().item()}")
        print(f"  Mean diff: {kv_diff_float.mean().item()}")
        print("  First 16 slots comparison:")
        for slot in range(min(16, kv_cache_unfused_2.shape[0])):
            slot_diff = kv_diff_bytes[slot].sum().item()
            if slot_diff > 0:
                print(f"    Slot {slot}: {slot_diff} bytes differ")
        print(
            f"  Differing bytes: {(kv_cache_unfused_2 != captured_unfused_1['kv_cache']).sum().item()} / {kv_cache_unfused_2.numel()}"
        )

        print("  ✓ Unfused run 2 complete with SAME input")

        # ==================================================================
        # COMPARE: Unfused Run 1 vs Run 2 (SAME Input - Determinism Check)
        # ==================================================================
        print(f"\n{'=' * 80}")
        print("COMPARISON: Unfused Run 1 vs Run 2 (SAME Input)")
        print(f"{'=' * 80}")
        print("  Run 1: Input A")
        print("  Run 2: Input A (same)")
        print("  EXPECTATION: Outputs MUST be identical (determinism)")

        # Compare outputs - they MUST be identical
        # IMPORTANT: Use captured CLONES
        print("\n  [DEBUG] Comparing captured tensors:")
        print(f"    Run 1 output data_ptr: {captured_unfused_1['output'].data_ptr()}")
        print(f"    Run 2 output data_ptr: {captured_unfused_2['output'].data_ptr()}")
        print(
            f"    Different memory? {captured_unfused_1['output'].data_ptr() != captured_unfused_2['output'].data_ptr()}"
        )
        print(
            f"    Run 1 output[0,:5]: {captured_unfused_1['output'][0, :5].cpu().float().numpy()}"
        )
        print(
            f"    Run 2 output[0,:5]: {captured_unfused_2['output'][0, :5].cpu().float().numpy()}"
        )

        # Check byte-level equality
        bytes_equal = torch.equal(
            captured_unfused_1["output"], captured_unfused_2["output"]
        )
        print(f"    Byte-perfect match? {bytes_equal}")

        output_diff = torch.abs(
            captured_unfused_1["output"] - captured_unfused_2["output"]
        )
        output_match = torch.allclose(
            captured_unfused_1["output"], captured_unfused_2["output"], atol=1e-6
        )

        print("\n  Output Comparison:")
        print(f"    Max difference: {output_diff.max().item():.6e}")
        print(f"    Mean difference: {output_diff.mean().item():.6e}")
        print(f"    Match (atol=1e-6): {'✓ PASS' if output_match else '✗ FAIL'}")

        if output_match:
            print("\n  ✓ SUCCESS: Same input produces identical outputs")
            print("  → Unfused path is deterministic")
        else:
            print("\n  ✗ FAIL: Same input produces different outputs")
            print("  → NON-DETERMINISM BUG DETECTED!")
            print("  → This could be due to:")
            print("    - Uninitialized memory")
            print("    - Race conditions")
            print("    - Non-deterministic CUDA operations")
            print("    - Flash Attention non-determinism")

        assert output_match, (
            f"Unfused path should be deterministic! Got max diff {output_diff.max().item():.6e}"
        )

        print(f"\n  {'✓' * 40}")
        print("  ✓✓✓ UNFUSED PATH IS DETERMINISTIC! ✓✓✓")
        print(f"  {'✓' * 40}")

        # ==================================================================
        # PHASE 3: FUSED PATH
        # ==================================================================
        print(f"\n{'=' * 80}")
        print("PHASE 3: FUSED PATH")
        print(f"{'=' * 80}")

        # DEBUG: Check hidden_states at start of Phase 3
        print("\n[DEBUG PHASE 3 START] hidden_states check:")
        print(f"  hidden_states id: {id(hidden_states)}")
        print(f"  hidden_states data_ptr: {hidden_states.data_ptr()}")
        print(f"  hidden_states[0,:10]: {hidden_states[0, :10].cpu().float().numpy()}")

        # Create fresh KV cache for fused run
        kv_cache_fused = torch.zeros(
            total_slots, num_kv_heads, kv_cache_dim, device=device, dtype=torch.uint8
        )

        # Enable fused kernel
        wrapper.mla_attn.use_aiter_fused_decode = original_use_aiter_fused

        # NOTE: Fused kernel only works for decode-only batches (num_prefill == 0)
        # For Mixed Batch, decode tokens will still use unfused path
        if num_prefill > 0:
            print(
                f"\n[NOTE] Mixed batch detected ({num_prefill} prefill + {num_decode} decode)"
            )
            print(
                "       Fused kernel requires decode-only batch, so decode tokens will use unfused path"
            )
            print(
                "       Only prefill RoPE timing will differ between unfused and 'fused' phases\n"
            )

        # Populate KV cache with prefill history (if needed)
        print("\n>>> SETUP START: Populating KV cache for prefill history <<<")
        populate_kv_cache_with_prefill(
            kv_cache_fused, use_fused=True, phase_name="FUSED RUN"
        )

        print(">>> SETUP END: KV cache populated <<<\n")

        # Capture KV cache state after setup (baseline for slot comparison)
        kv_cache_after_setup_fused = kv_cache_fused.clone().detach()

        mock_attn_layer.kv_cache = {0: kv_cache_fused}
        wrapper.mla_attn.kv_cache[0] = kv_cache_fused

        captured_fused = {}

        # Storage for Q after RoPE (captured inside forward_impl)
        q_after_rope_storage = {}

        def capture_fused(q, kv_c_normed, k_pe, **kwargs):
            # Deep copy inputs (clone + detach)
            captured_fused["q_before_rope"] = q.clone().detach()
            captured_fused["kv_c_normed"] = kv_c_normed.clone().detach()
            captured_fused["k_pe_before_rope"] = k_pe.clone().detach()

            # DEBUG: Print Q BEFORE RoPE (fused path applies RoPE internally)
            print("\n[DEBUG FUSED] Q BEFORE RoPE:")
            print(f"  q shape: {q.shape}")
            print(f"  q[0,0,:5]: {q[0, 0, :5].cpu().float().numpy()}")
            print(
                f"  k_pe[0,0,:5] (before RoPE): {k_pe[0, 0, :5].cpu().float().numpy()}"
            )

            # Store reference to capture Q after RoPE
            wrapper.mla_attn._test_q_after_rope_storage = q_after_rope_storage

            # Pass SAME Q to fused path (no modification)
            result = wrapper.mla_attn.__class__.forward(
                wrapper.mla_attn, q, kv_c_normed, k_pe, **kwargs
            )

            # DEBUG: Print Q AFTER forward (after RoPE)
            print("\n[DEBUG FUSED] Q AFTER forward (RoPE applied internally):")
            print(f"  q[0,0,:5]: {q[0, 0, :5].cpu().float().numpy()}")

            # Retrieve all captured values from forward_impl
            if "k_pe_after_rope" in q_after_rope_storage:
                captured_fused["k_pe_after_rope"] = (
                    q_after_rope_storage["k_pe_after_rope"].clone().detach()
                )

            # Copy test storage
            captured_fused["_test_storage"] = q_after_rope_storage.copy()

            return result

        wrapper.mla_attn.forward = capture_fused

        with torch.no_grad():
            print("\n[TEST DEBUG PHASE 3] Passing to forward context:")
            print(f"  slot_mapping: {slot_mapping.cpu().numpy()}")

            with set_forward_context(
                attn_metadata=mock_metadata,
                vllm_config=default_vllm_config,
                virtual_engine=0,
                slot_mapping={"test_layer.attn": slot_mapping},
            ):
                print("\n>>> RUN EXECUTING: FUSED PATH <<<\n")
                output_fused = wrapper.forward(
                    positions=positions,
                    hidden_states=hidden_states,  # Same input as unfused run 1
                )

        wrapper.mla_attn.forward = original_forward

        # Capture deep copies (clone + detach from computation graph)
        captured_fused["output"] = output_fused.clone().detach()
        captured_fused["kv_cache"] = kv_cache_fused.clone().detach()

        # DEBUG: Check KV cache after population
        print("[DEBUG] PHASE 3 KV cache after prefill history population:")
        kv_cache_before_fused_decode = kv_cache_fused.clone()
        print(f"  kv_cache[0,0,:5]: {kv_cache_fused[0, 0, :5].cpu().numpy()}")
        print(f"  kv_cache[5,0,:5]: {kv_cache_fused[5, 0, :5].cpu().numpy()}")
        print(
            f"  kv_cache non-zero: {(kv_cache_fused != 0).sum().item()} / {kv_cache_fused.numel()}"
        )

        # Compare with Phase 1 KV cache
        # Check prefill history slots (0-15) vs decode slots (16-31)
        prefill_slots_diff = (
            torch.abs(kv_cache_fused[:16] - captured_unfused_1["kv_cache"][:16])
            .sum()
            .item()
        )
        decode_slots_diff = (
            torch.abs(kv_cache_fused[16:32] - captured_unfused_1["kv_cache"][16:32])
            .sum()
            .item()
        )
        print("\n[DEBUG] KV cache slot range comparison (Phase 3 vs Phase 1):")
        print(
            f"  Prefill history slots [0-15]:  diff = {prefill_slots_diff:.2f}  (should be 0 - not modified during decode)"
        )
        print(
            f"  Decode slots [16-31]:          diff = {decode_slots_diff:.2f}  (expected to differ)"
        )

        # Detailed comparison
        kv_diff_bytes = kv_cache_fused != captured_unfused_1["kv_cache"]
        kv_diff_float = torch.abs(
            kv_cache_fused.float() - captured_unfused_1["kv_cache"].float()
        )
        prefill_diff_bytes = kv_diff_bytes[:16].sum().item()
        decode_diff_bytes = kv_diff_bytes[16:32].sum().item()
        print(f"  Prefill slots differing bytes: {prefill_diff_bytes}")
        print(f"  Decode slots differing bytes:  {decode_diff_bytes}")
        print(
            f"  Total differing bytes: {kv_diff_bytes.sum().item()} / {kv_cache_fused.numel()}"
        )
        print(f"  Max diff: {kv_diff_float.max().item()}")
        print(f"  Mean diff: {kv_diff_float.mean().item()}")
        print(f"  Max diff: {kv_diff_float.max().item()}")
        print(f"  Mean diff: {kv_diff_float.mean().item()}")
        print(
            f"\n  Checking which slots differ (KV cache dim = {kv_cache_fused.shape[-1]}):"
        )
        print(
            "  Note: KV cache layout = [kv_c (512 dims) || k_pe (64 dims)] = 576 total"
        )
        decode_start_slot = 16  # Decode tokens write starting from slot 16
        for slot in range(16, min(32, kv_cache_fused.shape[0])):
            slot_diff_mask = kv_diff_bytes[slot, 0, :]
            slot_diff_count = slot_diff_mask.sum().item()
            if slot_diff_count > 0:
                # Find which positions differ
                diff_positions = torch.where(slot_diff_mask)[0].cpu().numpy()
                phase1_vals = (
                    captured_unfused_1["kv_cache"][slot, 0, diff_positions[:10]]
                    .cpu()
                    .numpy()
                )
                phase3_vals = kv_cache_fused[slot, 0, diff_positions[:10]].cpu().numpy()
                print(
                    f"\n    Slot {slot:2d} (decode token {slot - 16}): {slot_diff_count} bytes differ"
                )
                print(f"      Diff positions (first 10): {diff_positions[:10]}")
                if diff_positions[0] >= 512:
                    print("      → Differences in K_PE part (positions >= 512)")
                else:
                    print("      → Differences in KV_C part (positions < 512)")
                print(f"      Phase1 values: {phase1_vals}")
                print(f"      Phase3 values: {phase3_vals}")
        print(
            f"  Differing bytes: {(kv_cache_fused != captured_unfused_1['kv_cache']).sum().item()} / {kv_cache_fused.numel()}"
        )

        # Check which slots were written during fused decode
        kv_diff_fused_decode = kv_cache_fused != kv_cache_before_fused_decode
        print("\n[DEBUG PHASE 3] Slots written during FUSED decode:")
        for slot in range(min(32, kv_cache_fused.shape[0])):
            if kv_diff_fused_decode[slot].sum().item() > 0:
                print(
                    f"  Slot {slot}: {kv_diff_fused_decode[slot].sum().item()} bytes written"
                )

        print("  ✓ Fused run complete")
        print(f"    Output shape: {output_fused.shape}")
        q_shape = captured_fused.get(
            "q_after_rope", captured_fused["q_before_rope"]
        ).shape
        print(f"    Q shape: {q_shape}")
        print(
            f"    Q captured after RoPE: {'✓ YES' if 'q_after_rope' in captured_fused else '✗ NO (using before)'}"
        )
        print(
            f"    KV cache non-zero: {(kv_cache_fused != 0).sum().item()} / {kv_cache_fused.numel()}"
        )

        # ==================================================================
        # PHASE 4: COMPARE FUSED VS UNFUSED
        # ==================================================================
        print(f"\n{'=' * 80}")
        print(f"COMPARISON: Fused vs Unfused ({scenario})")
        print(f"{'=' * 80}")

        # Split Q into nope and rope parts for detailed analysis
        # Q shape: [batch, num_heads, qk_head_dim] where qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        qk_nope_head_dim = 128  # DeepSeek-V3
        qk_rope_head_dim = 64

        # Compare the ACTUAL final FP8 values used for attention
        # Both paths end up with FP8 after _decode_concat_quant_fp8_op

        # For decode tokens (if any):
        if num_decode > 0:
            unfused_storage = captured_unfused_1.get("_test_storage", {})
            fused_storage = captured_fused.get("_test_storage", {})

            # Compare final mqa_q (FP8 in both paths, used for attention)
            if (
                "decode_mqa_q_final_fp8" in unfused_storage
                and "decode_mqa_q_final_fp8" in fused_storage
            ):
                mqa_q_unfused = unfused_storage["decode_mqa_q_final_fp8"]
                mqa_q_fused = fused_storage["decode_mqa_q_final_fp8"]

                mqa_q_diff = torch.abs(mqa_q_fused.float() - mqa_q_unfused.float())
                mqa_q_match = torch.allclose(
                    mqa_q_fused.float(), mqa_q_unfused.float(), atol=0.10
                )

                print("\n  Final Q for Attention (FP8, concat of Q_NOPE + Q_ROPE):")
                print(f"    Shape: {mqa_q_unfused.shape}")
                print(f"    Unfused dtype: {mqa_q_unfused.dtype}")
                print(f"    Fused dtype:   {mqa_q_fused.dtype}")
                print(
                    f"    Unfused[0,0,:5] (Q_NOPE part): {mqa_q_unfused[0, 0, :5].cpu().float().numpy()}"
                )
                print(
                    f"    Fused[0,0,:5] (Q_NOPE part):   {mqa_q_fused[0, 0, :5].cpu().float().numpy()}"
                )
                print(
                    f"    Unfused[0,0,512:517] (Q_ROPE part): {mqa_q_unfused[0, 0, 512:517].cpu().float().numpy()}"
                )
                print(
                    f"    Fused[0,0,512:517] (Q_ROPE part):   {mqa_q_fused[0, 0, 512:517].cpu().float().numpy()}"
                )
                print(f"    Max difference: {mqa_q_diff.max().item():.6e}")
                print(f"    Mean difference: {mqa_q_diff.mean().item():.6e}")
                print(f"    Match (atol=0.10): {'✓ PASS' if mqa_q_match else '✗ FAIL'}")
                print("    → Both paths use FP8 for attention")
                print("    → Unfused: bf16 BMM + bf16 RoPE → quantized to FP8")
                print("    → Fused: FP8 BMM + FP8 RoPE (in kernel) → stays FP8")
                print("    → Small difference due to different quantization order")

        # For prefill tokens (if any): Compare Q directly (goes to flash attention)
        if num_prefill > 0:
            unfused_storage = captured_unfused_1.get("_test_storage", {})
            fused_storage = captured_fused.get("_test_storage", {})

            if (
                "prefill_q_after_rope" in unfused_storage
                and "prefill_q_after_rope" in fused_storage
            ):
                q_prefill_unfused = unfused_storage["prefill_q_after_rope"]
                q_prefill_fused = fused_storage["prefill_q_after_rope"]

                q_prefill_diff = torch.abs(q_prefill_fused - q_prefill_unfused)
                q_prefill_match = torch.allclose(
                    q_prefill_fused, q_prefill_unfused, atol=0.2
                )

                print("\n  Prefill Q (after RoPE, goes to Flash Attention):")
                print(f"    Max difference: {q_prefill_diff.max().item():.6e}")
                print(f"    Mean difference: {q_prefill_diff.mean().item():.6e}")
                print(
                    f"    Match (tol=0.2): {'✓ PASS' if q_prefill_match else '✗ FAIL'}"
                )

        # Compare k_pe (position embeddings, AFTER RoPE in both paths)
        # NOTE: Kernel DOES return k_out with K_PE (RoPE applied) for decode tokens!
        # So we can now compare K_PE for ALL scenarios (Prefill, Decode, Mixed)
        k_pe_comparison_valid = (
            "k_pe_after_rope" in captured_fused
            and "k_pe_decode_no_capture" not in captured_fused
        )

        if k_pe_comparison_valid:
            # All scenarios: kernel returns k_out for decode, RoPE applied in forward_impl for prefill
            k_pe_unfused = captured_unfused_1["k_pe"]
            k_pe_fused = captured_fused["k_pe_after_rope"]

            # Debug: Check if we're comparing the same tensor
            print("\n  [DEBUG K_PE Comparison]:")
            print(f"    k_pe_unfused shape: {k_pe_unfused.shape}")
            print(f"    k_pe_fused shape: {k_pe_fused.shape}")
            print(f"    k_pe_unfused data_ptr: {k_pe_unfused.data_ptr()}")
            print(f"    k_pe_fused data_ptr: {k_pe_fused.data_ptr()}")
            print(
                f"    Same tensor? {k_pe_unfused.data_ptr() == k_pe_fused.data_ptr()}"
            )
            print(
                f"    k_pe_unfused[0,0,:5]: {k_pe_unfused[0, 0, :5].cpu().float().numpy()}"
            )
            print(
                f"    k_pe_fused[0,0,:5]:   {k_pe_fused[0, 0, :5].cpu().float().numpy()}"
            )
            if num_prefill > 0:
                print(
                    f"    k_pe_unfused[{num_decode},0,:5]: {k_pe_unfused[num_decode, 0, :5].cpu().float().numpy()}"
                )
                print(
                    f"    k_pe_fused[{num_decode},0,:5]:   {k_pe_fused[num_decode, 0, :5].cpu().float().numpy()}"
                )

            k_pe_diff = torch.abs(k_pe_fused - k_pe_unfused)
            k_pe_match = torch.allclose(k_pe_fused, k_pe_unfused, atol=0.2)

            print(
                f"\n  K_PE (position embeddings, {qk_rope_head_dim} dims, AFTER RoPE):"
            )
            print(f"    Max difference: {k_pe_diff.max().item():.6e}")
            print(f"    Mean difference: {k_pe_diff.mean().item():.6e}")
            print(f"    Match (tol=0.2): {'✓ PASS' if k_pe_match else '✗ FAIL'}")
            print("    → Both paths have RoPE applied, values should be close")
        else:
            # Fallback: kernel didn't return k_out (shouldn't happen)
            k_pe_match = True  # Skip comparison
            print(f"\n  K_PE (position embeddings, {qk_rope_head_dim} dims):")
            print("    Comparison skipped: Kernel k_out was empty")
            print("    → K correctness validated by final output comparison instead")

        # Compare kv_c_normed (compressed KV before expansion, should be identical)
        kv_c_diff = torch.abs(
            captured_fused["kv_c_normed"] - captured_unfused_1["kv_c_normed"]
        )
        kv_c_match = torch.allclose(
            captured_fused["kv_c_normed"], captured_unfused_1["kv_c_normed"], atol=1e-6
        )
        print("\n  kv_c_normed (compressed KV, should be identical):")
        print(f"    Max difference: {kv_c_diff.max().item():.6e}")
        print(f"    Mean difference: {kv_c_diff.mean().item():.6e}")
        print(f"    Match (atol=1e-6): {'✓ PASS' if kv_c_match else '✗ FAIL'}")

        # Compare KV cache values (FP8 quantized cache)
        # IMPORTANT: Use captured CLONES
        cache_diff_bytes = (
            (captured_fused["kv_cache"] != captured_unfused_1["kv_cache"]).sum().item()
        )
        cache_total_bytes = captured_fused["kv_cache"].numel()
        # Note: KV cache may differ slightly due to:
        # 1. Quantization order (unfused quantizes all at once, fused splits by prefill/decode)
        # 2. RoPE timing affects k_pe values going into cache
        fused_cache_nonzero = (captured_fused["kv_cache"] != 0).sum().item()
        unfused_cache_nonzero = (captured_unfused_1["kv_cache"] != 0).sum().item()

        print("\n  KV Cache (FP8 quantized values):")
        print(f"    Differing bytes: {cache_diff_bytes} / {cache_total_bytes}")
        print(f"    Fused non-zero:   {fused_cache_nonzero} bytes")
        print(f"    Unfused non-zero: {unfused_cache_nonzero} bytes")
        print("    Note: May differ due to RoPE timing and quantization order")

        # Compare which slots were modified by each path (excluding setup)
        print(
            "\n  KV Cache Slot Modification Comparison (RUN phase only, excluding SETUP):"
        )

        # Get final KV caches and baselines
        unfused_cache_final = (
            captured_unfused_1["kv_cache"].cpu().view(torch.uint8).numpy()
        )
        fused_cache_final = captured_fused["kv_cache"].cpu().view(torch.uint8).numpy()
        unfused_cache_baseline = (
            kv_cache_after_setup_unfused.cpu().view(torch.uint8).numpy()
        )
        fused_cache_baseline = (
            kv_cache_after_setup_fused.cpu().view(torch.uint8).numpy()
        )

        # Find slots modified during RUN (changed from baseline)
        unfused_run_modified_slots = set()
        fused_run_modified_slots = set()

        num_slots = unfused_cache_final.shape[0]
        for slot_idx in range(num_slots):
            # Check if slot was modified during RUN (different from baseline)
            if (
                unfused_cache_final[slot_idx] != unfused_cache_baseline[slot_idx]
            ).any():
                unfused_run_modified_slots.add(slot_idx)
            if (fused_cache_final[slot_idx] != fused_cache_baseline[slot_idx]).any():
                fused_run_modified_slots.add(slot_idx)

        # Compare slot sets
        slots_match = unfused_run_modified_slots == fused_run_modified_slots
        only_unfused = unfused_run_modified_slots - fused_run_modified_slots
        only_fused = fused_run_modified_slots - unfused_run_modified_slots

        print(f"    Unfused RUN modified slots: {sorted(unfused_run_modified_slots)}")
        print(f"    Fused RUN modified slots:   {sorted(fused_run_modified_slots)}")

        if slots_match:
            print("    ✓ MATCH: Both paths modified exactly the same slots during RUN!")
        else:
            print("    ✗ MISMATCH: Different slots modified during RUN")
            if only_unfused:
                print(f"      Only unfused: {sorted(only_unfused)}")
            if only_fused:
                print(f"      Only fused:   {sorted(only_fused)}")

        # Compare attention output (final result after MLA + o_proj)
        # Note: This is AFTER o_proj, not before (wrapper.forward returns after o_proj)
        # IMPORTANT: Use captured CLONES to avoid comparing modified tensors

        print("\n  [DEBUG] Output comparison:")
        print(
            f"    output_unfused_1 (captured) data_ptr: {captured_unfused_1['output'].data_ptr()}"
        )
        print(
            f"    output_fused (captured) data_ptr: {captured_fused['output'].data_ptr()}"
        )
        print(
            f"    Same tensor? {torch.equal(captured_unfused_1['output'], captured_fused['output'])}"
        )
        print(
            f"    output_unfused_1[0,:5]: {captured_unfused_1['output'][0, :5].cpu().float().numpy()}"
        )
        print(
            f"    output_fused[0,:5]:     {captured_fused['output'][0, :5].cpu().float().numpy()}"
        )

        output_diff = torch.abs(captured_fused["output"] - captured_unfused_1["output"])
        output_max_diff = output_diff.max().item()
        output_mean_diff = output_diff.mean().item()

        # Also compare in float32 to see tiny differences that bfloat16 can't represent
        output_diff_f32 = torch.abs(
            captured_fused["output"].float() - captured_unfused_1["output"].float()
        )
        output_max_diff_f32 = output_diff_f32.max().item()
        output_mean_diff_f32 = output_diff_f32.mean().item()

        # Tolerance: 0.2 for bfloat16 with complex operations (Flash Attention, RoPE, etc.)
        # Prefill path has more numerical variance due to Flash Attention varlen implementation
        tolerance = 0.2
        output_match = output_max_diff < tolerance

        print("\n  Attention Output (final result after MLA + o_proj):")
        print(f"    Max difference (bfloat16): {output_max_diff:.6e}")
        print(f"    Mean difference (bfloat16): {output_mean_diff:.6e}")
        print(f"    Max difference (float32): {output_max_diff_f32:.10f}")
        print(f"    Mean difference (float32): {output_mean_diff_f32:.10f}")
        print(f"    Match (tol={tolerance}): {'✓ PASS' if output_match else '✗ FAIL'}")
        print("    → Fused and unfused use SAME input")
        print(
            "    → Small differences expected due to RoPE timing and quantization order"
        )

        # Summary
        print(f"\n{'=' * 80}")
        print(f"SUMMARY: {scenario}")
        print(f"{'=' * 80}")
        print(
            "  ✓ Unfused run 1 vs run 2: Same input → identical outputs (deterministic)"
        )

        if num_prefill > 0 and num_decode > 0:
            # Mixed batch: both prefill and decode tokens
            print("  ✓ Fused vs Unfused: Same input → similar outputs (correctness)")
            print("    → Decode uses fused kernel, prefill RoPE timing differs")
        else:
            # Pure prefill or pure decode
            print("  ✓ Fused vs Unfused: Same input → similar outputs (correctness)")

        print("  ✓ Both paths write KV cache")

        # Assertions for decode tokens
        if num_decode > 0:
            unfused_storage = captured_unfused_1.get("_test_storage", {})
            fused_storage = captured_fused.get("_test_storage", {})

            if (
                "decode_mqa_q_final_fp8" in unfused_storage
                and "decode_mqa_q_final_fp8" in fused_storage
            ):
                assert mqa_q_match, (
                    f"Final Q (FP8) for attention difference {mqa_q_diff.max().item():.6e} exceeds tolerance 0.10"
                )

        # Assertions for prefill tokens
        if num_prefill > 0:
            unfused_storage = captured_unfused_1.get("_test_storage", {})
            fused_storage = captured_fused.get("_test_storage", {})

            if (
                "prefill_q_after_rope" in unfused_storage
                and "prefill_q_after_rope" in fused_storage
            ):
                assert q_prefill_match, (
                    f"Prefill Q difference {q_prefill_diff.max().item():.6e} exceeds tolerance 0.2"
                )

        # K_PE comparison
        if k_pe_comparison_valid:
            assert k_pe_match, (
                f"K_PE difference {k_pe_diff.max().item():.6e} exceeds tolerance 0.2"
            )

        assert kv_c_match, "kv_c_normed should match"
        assert output_match, (
            f"Output difference {output_max_diff:.6e} exceeds tolerance {tolerance}"
        )
        assert fused_cache_nonzero > 0, "Fused path should write to KV cache"
        assert unfused_cache_nonzero > 0, "Unfused path should write to KV cache"

        print(f"\n  {'✓' * 40}")
        print("  ✓✓✓ ALL COMPARISONS PASSED! ✓✓✓")
        print(f"  {'✓' * 40}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

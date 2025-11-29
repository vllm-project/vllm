#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Generate FlashInfer Bench traces from vLLM inference.

This script runs inference on various models with FlashInfer enabled and
captures workload traces using FlashInfer-Bench. These traces can then be
used to optimize FlashInfer operations with custom kernels.

The script automatically creates Definition files for vLLM's FlashInfer operations
based on the model's configuration (hidden sizes, head dims, etc.).

Traced operators:
- Attention:
  - GQA Paged Decode (attention decode with paged KV cache)
  - GQA Paged Prefill (attention prefill with paged KV cache)
  - GQA Ragged Prefill (attention prefill with ragged tensors)
  - MLA Paged (Multi-head Latent Attention for DeepSeek models)
- Normalization:
  - RMSNorm (fused_add_rmsnorm)
- Sampling:
  - Top-k sampling
  - Top-p sampling
  - Top-k + Top-p sampling
- Activations:
  - SiLU and mul
  - GELU and mul
  - GELU tanh and mul
- Positional Encoding:
  - RoPE (apply_rope_with_cos_sin_cache_inplace)
- MoE:
  - CUTLASS fused MoE
  - TRT-LLM FP8 MoE
  - TRT-LLM FP4 MoE
- GEMM (for quantized models):
  - mm_fp4 (FP4 matrix multiplication)
  - bmm_fp8 (FP8 batched matrix multiplication)
  - grouped_gemm_nt_masked (MoE CUTEDSL grouped GEMM)
- Communication:
  - AllReduce fusion
  - MnnvlMoe dispatch/combine (All2All)

NOTE: Tracing requires `enforce_eager=True` because FlashInfer-Bench adapters
are not compatible with torch.compile. See flashinfer_integration_issues.md
for details.

Reference: https://bench.flashinfer.ai/docs/start/quickstart

Installation:
    pip install flashinfer-bench --no-deps

Usage:
    python tests/kernels/generate_flashinfer_traces.py --model qwen
    python tests/kernels/generate_flashinfer_traces.py --model llama
    python tests/kernels/generate_flashinfer_traces.py --model all --output-dir /path/to/traces
"""

import argparse
import json
import os
import sys
from pathlib import Path


def check_compatibility():
    """Check if the environment is compatible for trace generation.
    
    Returns:
        tuple: (is_compatible, error_message)
    """
    # Check torch version
    try:
        import torch
        torch_version = torch.__version__
    except ImportError:
        return False, "torch is not installed"
    
    # Check if flashinfer-bench is installed
    try:
        import importlib.util
        spec = importlib.util.find_spec("flashinfer_bench")
        if spec is None:
            return False, (
                "flashinfer-bench is not installed.\n"
                "Install with: pip install flashinfer-bench --no-deps"
            )
    except Exception as e:
        return False, f"Error checking flashinfer-bench: {e}"
    
    # Check if vLLM can be imported
    try:
        import vllm  # noqa: F401
    except Exception as e:
        return False, f"vLLM import failed: {e}"
    
    return True, f"Compatible (torch {torch_version})"


def create_rmsnorm_definitions(hidden_sizes: list[int], output_dir: str):
    """Create RMSNorm definitions for given hidden sizes."""
    from flashinfer_bench.data import (
        AxisConst, AxisVar, Definition, TensorSpec
    )
    
    definitions_dir = Path(output_dir) / "definitions"
    definitions_dir.mkdir(parents=True, exist_ok=True)
    
    definitions = {}
    for hidden_size in hidden_sizes:
        def_name = f"fused_add_rmsnorm_h{hidden_size}"
        
        definition = Definition(
            name=def_name,
            op_type="rmsnorm",
            axes={"M": AxisVar(), "H": AxisConst(value=hidden_size)},
            inputs={
                "hidden_states": TensorSpec(shape=["M", "H"], dtype="bfloat16"),
                "residual": TensorSpec(shape=["M", "H"], dtype="bfloat16"),
                "weight": TensorSpec(shape=["H"], dtype="bfloat16"),
            },
            outputs={"output": TensorSpec(shape=["M", "H"], dtype="bfloat16")},
            reference="def run(hidden_states, residual, weight):\n    return hidden_states\n",
        )
        
        definitions[def_name] = definition
        
        # Save to file
        def_file = definitions_dir / f"{def_name}.json"
        with open(def_file, 'w') as f:
            json.dump(definition.model_dump(), f, indent=2)
    
    return definitions


def create_attention_definitions(
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    output_dir: str,
    page_size: int = 16,
):
    """Create attention definitions for GQA paged decode/prefill.
    
    Note: For paged attention, the k_cache/v_cache have shape [num_pages, num_kv_heads, head_dim]
    after normalization (NHD layout). The batch dimension B is separate from the cache dimension.
    """
    from flashinfer_bench.data import (
        AxisConst, AxisVar, Definition, TensorSpec
    )
    
    definitions_dir = Path(output_dir) / "definitions"
    definitions_dir.mkdir(parents=True, exist_ok=True)
    
    definitions = {}
    
    # GQA Paged Decode definition
    # q: [B, H, D] - batch of queries
    # k_cache/v_cache: [N, KV, D] - paged KV cache (N = total pages, independent of B)
    # kv_indptr: [B1] - indptr for batch (length = batch_size + 1)
    # kv_indices: [I] - indices into cache
    decode_def_name = f"gqa_paged_decode_h{num_heads}_kv{num_kv_heads}_d{head_dim}_ps{page_size}"
    decode_definition = Definition(
        name=decode_def_name,
        op_type="gqa_paged_decode",
        axes={
            "B": AxisVar(),  # Batch size (queries)
            "B1": AxisVar(),  # Batch size + 1 (for indptr)
            "N": AxisVar(),  # Number of cache pages (independent of B)
            "I": AxisVar(),  # Number of indices
            "H": AxisConst(value=num_heads),
            "KV": AxisConst(value=num_kv_heads),
            "D": AxisConst(value=head_dim),
        },
        inputs={
            "q": TensorSpec(shape=["B", "H", "D"], dtype="bfloat16"),
            "k_cache": TensorSpec(shape=["N", "KV", "D"], dtype="bfloat16"),
            "v_cache": TensorSpec(shape=["N", "KV", "D"], dtype="bfloat16"),
            "kv_indptr": TensorSpec(shape=["B1"], dtype="int32"),
            "kv_indices": TensorSpec(shape=["I"], dtype="int32"),
            "sm_scale": TensorSpec(shape=[], dtype="float32"),
            "page_size": TensorSpec(shape=[], dtype="int32"),
        },
        outputs={"output": TensorSpec(shape=["B", "H", "D"], dtype="bfloat16")},
        reference="def run(q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale, page_size):\n    return q\n",
    )
    definitions[decode_def_name] = decode_definition
    
    # Save decode definition
    def_file = definitions_dir / f"{decode_def_name}.json"
    with open(def_file, 'w') as f:
        json.dump(decode_definition.model_dump(), f, indent=2)
    
    # GQA Paged Prefill (causal) definition
    # q: [T, H, D] - total tokens across batch
    # k_cache/v_cache: [N, KV, D] - paged KV cache
    # qo_indptr: [B1] - indptr for query batch (length = batch_size + 1)
    # kv_indptr: [B1] - indptr for KV batch (length = batch_size + 1)
    # kv_indices: [I] - indices into cache
    prefill_def_name = f"gqa_paged_prefill_causal_h{num_heads}_kv{num_kv_heads}_d{head_dim}_ps{page_size}"
    prefill_definition = Definition(
        name=prefill_def_name,
        op_type="gqa_paged_prefill",
        axes={
            "T": AxisVar(),  # Total tokens
            "B1": AxisVar(),  # Batch size + 1 (for indptr)
            "N": AxisVar(),  # Number of cache pages
            "I": AxisVar(),  # Number of indices
            "H": AxisConst(value=num_heads),
            "KV": AxisConst(value=num_kv_heads),
            "D": AxisConst(value=head_dim),
        },
        inputs={
            "q": TensorSpec(shape=["T", "H", "D"], dtype="bfloat16"),
            "k_cache": TensorSpec(shape=["N", "KV", "D"], dtype="bfloat16"),
            "v_cache": TensorSpec(shape=["N", "KV", "D"], dtype="bfloat16"),
            "qo_indptr": TensorSpec(shape=["B1"], dtype="int32"),
            "kv_indptr": TensorSpec(shape=["B1"], dtype="int32"),
            "kv_indices": TensorSpec(shape=["I"], dtype="int32"),
            "sm_scale": TensorSpec(shape=[], dtype="float32"),
            "page_size": TensorSpec(shape=[], dtype="int32"),
            "causal": TensorSpec(shape=[], dtype="bool"),
        },
        outputs={"output": TensorSpec(shape=["T", "H", "D"], dtype="bfloat16")},
        reference="def run(q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale, page_size, causal):\n    return q\n",
    )
    definitions[prefill_def_name] = prefill_definition
    
    # Save prefill definition
    def_file = definitions_dir / f"{prefill_def_name}.json"
    with open(def_file, 'w') as f:
        json.dump(prefill_definition.model_dump(), f, indent=2)
    
    # GQA Ragged Prefill (causal) definition
    # q: [T, H, D] - total tokens
    # k: [T, KV, D] - keys (same token count as q)
    # v: [T, KV, D] - values (same token count as q)
    # qo_indptr: [B1] - indptr for query batch (length = batch_size + 1)
    # kv_indptr: [B1] - indptr for KV batch (length = batch_size + 1)
    ragged_def_name = f"gqa_ragged_prefill_causal_h{num_heads}_kv{num_kv_heads}_d{head_dim}"
    ragged_definition = Definition(
        name=ragged_def_name,
        op_type="gqa_ragged_prefill",
        axes={
            "T": AxisVar(),  # Total tokens
            "B1": AxisVar(),  # Batch size + 1 (for indptr)
            "H": AxisConst(value=num_heads),
            "KV": AxisConst(value=num_kv_heads),
            "D": AxisConst(value=head_dim),
        },
        inputs={
            "q": TensorSpec(shape=["T", "H", "D"], dtype="bfloat16"),
            "k": TensorSpec(shape=["T", "KV", "D"], dtype="bfloat16"),
            "v": TensorSpec(shape=["T", "KV", "D"], dtype="bfloat16"),
            "qo_indptr": TensorSpec(shape=["B1"], dtype="int32"),
            "kv_indptr": TensorSpec(shape=["B1"], dtype="int32"),
            "sm_scale": TensorSpec(shape=[], dtype="float32"),
            "causal": TensorSpec(shape=[], dtype="bool"),
        },
        outputs={"output": TensorSpec(shape=["T", "H", "D"], dtype="bfloat16")},
        reference="def run(q, k, v, qo_indptr, kv_indptr, sm_scale, causal):\n    return q\n",
    )
    definitions[ragged_def_name] = ragged_definition
    
    # Save ragged definition
    def_file = definitions_dir / f"{ragged_def_name}.json"
    with open(def_file, 'w') as f:
        json.dump(ragged_definition.model_dump(), f, indent=2)
    
    return definitions


def create_mla_definitions(
    num_heads: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    output_dir: str,
    page_size: int = 16,
):
    """Create MLA attention definitions for DeepSeek models.
    
    Note: For MLA paged attention, the cache tensors have shape [N, CKV] or [N, KPE]
    where N is the total number of cache entries (independent of batch size).
    """
    from flashinfer_bench.data import (
        AxisConst, AxisVar, Definition, TensorSpec
    )
    
    definitions_dir = Path(output_dir) / "definitions"
    definitions_dir.mkdir(parents=True, exist_ok=True)
    
    definitions = {}
    
    # MLA Paged Decode definition
    # q_nope: [B, H, CKV] - batch of queries (non-positional)
    # q_pe: [B, H, KPE] - batch of queries (positional)
    # ckv_cache: [N, CKV] - paged compressed KV cache
    # kpe_cache: [N, KPE] - paged key positional encoding cache
    # kv_indptr: [B1] - indptr for batch (length = batch_size + 1)
    decode_def_name = f"mla_paged_decode_h{num_heads}_ckv{kv_lora_rank}_kpe{qk_rope_head_dim}_ps{page_size}"
    decode_definition = Definition(
        name=decode_def_name,
        op_type="mla_paged_decode",
        axes={
            "B": AxisVar(),  # Batch size
            "B1": AxisVar(),  # Batch size + 1 (for indptr)
            "N": AxisVar(),  # Number of cache entries
            "I": AxisVar(),  # Number of indices
            "H": AxisConst(value=num_heads),
            "CKV": AxisConst(value=kv_lora_rank),
            "KPE": AxisConst(value=qk_rope_head_dim),
        },
        inputs={
            "q_nope": TensorSpec(shape=["B", "H", "CKV"], dtype="bfloat16"),
            "q_pe": TensorSpec(shape=["B", "H", "KPE"], dtype="bfloat16"),
            "ckv_cache": TensorSpec(shape=["N", "CKV"], dtype="bfloat16"),
            "kpe_cache": TensorSpec(shape=["N", "KPE"], dtype="bfloat16"),
            "kv_indptr": TensorSpec(shape=["B1"], dtype="int32"),
            "kv_indices": TensorSpec(shape=["I"], dtype="int32"),
            "sm_scale": TensorSpec(shape=[], dtype="float32"),
            "page_size": TensorSpec(shape=[], dtype="int32"),
        },
        outputs={"output": TensorSpec(shape=["B", "H", "CKV"], dtype="bfloat16")},
        reference="def run(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale, page_size):\n    return q_nope\n",
    )
    definitions[decode_def_name] = decode_definition
    
    # Save decode definition
    def_file = definitions_dir / f"{decode_def_name}.json"
    with open(def_file, 'w') as f:
        json.dump(decode_definition.model_dump(), f, indent=2)
    
    # MLA Paged Prefill (causal) definition
    # q_nope: [T, H, CKV] - total tokens queries (non-positional)
    # q_pe: [T, H, KPE] - total tokens queries (positional)
    # ckv_cache: [N, CKV] - paged compressed KV cache
    # kpe_cache: [N, KPE] - paged key positional encoding cache
    # qo_indptr: [B1] - indptr for query batch (length = batch_size + 1)
    # kv_indptr: [B1] - indptr for KV batch (length = batch_size + 1)
    prefill_def_name = f"mla_paged_prefill_causal_h{num_heads}_ckv{kv_lora_rank}_kpe{qk_rope_head_dim}_ps{page_size}"
    prefill_definition = Definition(
        name=prefill_def_name,
        op_type="mla_paged_prefill",
        axes={
            "T": AxisVar(),  # Total tokens
            "B1": AxisVar(),  # Batch size + 1 (for indptr)
            "N": AxisVar(),  # Number of cache entries
            "I": AxisVar(),  # Number of indices
            "H": AxisConst(value=num_heads),
            "CKV": AxisConst(value=kv_lora_rank),
            "KPE": AxisConst(value=qk_rope_head_dim),
        },
        inputs={
            "q_nope": TensorSpec(shape=["T", "H", "CKV"], dtype="bfloat16"),
            "q_pe": TensorSpec(shape=["T", "H", "KPE"], dtype="bfloat16"),
            "ckv_cache": TensorSpec(shape=["N", "CKV"], dtype="bfloat16"),
            "kpe_cache": TensorSpec(shape=["N", "KPE"], dtype="bfloat16"),
            "qo_indptr": TensorSpec(shape=["B1"], dtype="int32"),
            "kv_indptr": TensorSpec(shape=["B1"], dtype="int32"),
            "kv_indices": TensorSpec(shape=["I"], dtype="int32"),
            "sm_scale": TensorSpec(shape=[], dtype="float32"),
            "page_size": TensorSpec(shape=[], dtype="int32"),
            "causal": TensorSpec(shape=[], dtype="bool"),
        },
        outputs={"output": TensorSpec(shape=["T", "H", "CKV"], dtype="bfloat16")},
        reference="def run(q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale, page_size, causal):\n    return q_nope\n",
    )
    definitions[prefill_def_name] = prefill_definition
    
    # Save prefill definition
    def_file = definitions_dir / f"{prefill_def_name}.json"
    with open(def_file, 'w') as f:
        json.dump(prefill_definition.model_dump(), f, indent=2)
    
    return definitions


def create_sampling_definitions(vocab_size: int, output_dir: str):
    """Create sampling definitions for top-k/top-p sampling.
    
    Note: k and p parameters are not included in definitions as they can be
    tensors with per-batch values, not just scalars.
    """
    from flashinfer_bench.data import (
        AxisConst, AxisVar, Definition, TensorSpec
    )
    
    definitions_dir = Path(output_dir) / "definitions"
    definitions_dir.mkdir(parents=True, exist_ok=True)
    
    definitions = {}
    
    # Top-k sampling
    topk_def_name = f"top_k_sampling_v{vocab_size}"
    topk_definition = Definition(
        name=topk_def_name,
        op_type="sampling_top_k",
        axes={
            "B": AxisVar(),  # Batch size
            "V": AxisConst(value=vocab_size),
        },
        inputs={
            "probs": TensorSpec(shape=["B", "V"], dtype="float32"),
        },
        outputs={"output": TensorSpec(shape=["B"], dtype="int64")},
        reference="def run(probs):\n    return torch.argmax(probs, dim=-1)\n",
    )
    definitions[topk_def_name] = topk_definition
    
    def_file = definitions_dir / f"{topk_def_name}.json"
    with open(def_file, 'w') as f:
        json.dump(topk_definition.model_dump(), f, indent=2)
    
    # Top-p sampling
    topp_def_name = f"top_p_sampling_v{vocab_size}"
    topp_definition = Definition(
        name=topp_def_name,
        op_type="sampling_top_p",
        axes={
            "B": AxisVar(),
            "V": AxisConst(value=vocab_size),
        },
        inputs={
            "probs": TensorSpec(shape=["B", "V"], dtype="float32"),
        },
        outputs={"output": TensorSpec(shape=["B"], dtype="int64")},
        reference="def run(probs):\n    return torch.argmax(probs, dim=-1)\n",
    )
    definitions[topp_def_name] = topp_definition
    
    def_file = definitions_dir / f"{topp_def_name}.json"
    with open(def_file, 'w') as f:
        json.dump(topp_definition.model_dump(), f, indent=2)
    
    # Top-k + Top-p sampling
    topkp_def_name = f"top_k_top_p_sampling_v{vocab_size}"
    topkp_definition = Definition(
        name=topkp_def_name,
        op_type="sampling_top_k_top_p",
        axes={
            "B": AxisVar(),
            "V": AxisConst(value=vocab_size),
        },
        inputs={
            "logits": TensorSpec(shape=["B", "V"], dtype="float32"),
        },
        outputs={"output": TensorSpec(shape=["B"], dtype="int64")},
        reference="def run(logits):\n    return torch.argmax(logits, dim=-1)\n",
    )
    definitions[topkp_def_name] = topkp_definition
    
    def_file = definitions_dir / f"{topkp_def_name}.json"
    with open(def_file, 'w') as f:
        json.dump(topkp_definition.model_dump(), f, indent=2)
    
    return definitions


def create_activation_definitions(hidden_sizes: list[int], output_dir: str):
    """Create activation definitions for SiLU, GELU."""
    from flashinfer_bench.data import (
        AxisConst, AxisVar, Definition, TensorSpec
    )
    
    definitions_dir = Path(output_dir) / "definitions"
    definitions_dir.mkdir(parents=True, exist_ok=True)
    
    definitions = {}
    
    for hidden_size in hidden_sizes:
        # SiLU and mul
        silu_def_name = f"silu_and_mul_d{hidden_size}"
        silu_definition = Definition(
            name=silu_def_name,
            op_type="activation_silu",
            axes={
                "T": AxisVar(),  # Total tokens
                "D": AxisConst(value=hidden_size),
            },
            inputs={
                "input": TensorSpec(shape=["T", "D"], dtype="bfloat16"),
            },
            outputs={"output": TensorSpec(shape=["T", "D"], dtype="bfloat16")},
            reference="def run(input):\n    return input\n",
        )
        definitions[silu_def_name] = silu_definition
        
        def_file = definitions_dir / f"{silu_def_name}.json"
        with open(def_file, 'w') as f:
            json.dump(silu_definition.model_dump(), f, indent=2)
        
        # GELU and mul
        gelu_def_name = f"gelu_and_mul_d{hidden_size}"
        gelu_definition = Definition(
            name=gelu_def_name,
            op_type="activation_gelu",
            axes={
                "T": AxisVar(),
                "D": AxisConst(value=hidden_size),
            },
            inputs={
                "input": TensorSpec(shape=["T", "D"], dtype="bfloat16"),
            },
            outputs={"output": TensorSpec(shape=["T", "D"], dtype="bfloat16")},
            reference="def run(input):\n    return input\n",
        )
        definitions[gelu_def_name] = gelu_definition
        
        def_file = definitions_dir / f"{gelu_def_name}.json"
        with open(def_file, 'w') as f:
            json.dump(gelu_definition.model_dump(), f, indent=2)
        
        # GELU tanh and mul
        gelu_tanh_def_name = f"gelu_tanh_and_mul_d{hidden_size}"
        gelu_tanh_definition = Definition(
            name=gelu_tanh_def_name,
            op_type="activation_gelu_tanh",
            axes={
                "T": AxisVar(),
                "D": AxisConst(value=hidden_size),
            },
            inputs={
                "input": TensorSpec(shape=["T", "D"], dtype="bfloat16"),
            },
            outputs={"output": TensorSpec(shape=["T", "D"], dtype="bfloat16")},
            reference="def run(input):\n    return input\n",
        )
        definitions[gelu_tanh_def_name] = gelu_tanh_definition
        
        def_file = definitions_dir / f"{gelu_tanh_def_name}.json"
        with open(def_file, 'w') as f:
            json.dump(gelu_tanh_definition.model_dump(), f, indent=2)
    
    return definitions


def create_rope_definitions(head_dims: list[int], output_dir: str):
    """Create RoPE definitions."""
    from flashinfer_bench.data import (
        AxisConst, AxisVar, Definition, TensorSpec
    )
    
    definitions_dir = Path(output_dir) / "definitions"
    definitions_dir.mkdir(parents=True, exist_ok=True)
    
    definitions = {}
    
    for head_dim in head_dims:
        for interleave in [False, True]:
            interleave_str = "_interleave" if interleave else ""
            def_name = f"rope_inplace_h{head_dim}{interleave_str}"
            definition = Definition(
                name=def_name,
                op_type="rope_inplace",
                axes={
                    "T": AxisVar(),  # Total tokens
                    "H": AxisVar(),  # Number of heads
                    "D": AxisConst(value=head_dim),
                    "S": AxisVar(),  # Max sequence length in cache
                },
                inputs={
                    "q": TensorSpec(shape=["T", "H", "D"], dtype="bfloat16"),
                    "k": TensorSpec(shape=["T", "H", "D"], dtype="bfloat16"),
                    "cos_cache": TensorSpec(shape=["S", "D"], dtype="bfloat16"),
                    "sin_cache": TensorSpec(shape=["S", "D"], dtype="bfloat16"),
                    "pos_ids": TensorSpec(shape=["T"], dtype="int64"),
                    "interleave": TensorSpec(shape=[], dtype="bool"),
                },
                outputs={
                    "q_out": TensorSpec(shape=["T", "H", "D"], dtype="bfloat16"),
                    "k_out": TensorSpec(shape=["T", "H", "D"], dtype="bfloat16"),
                },
                reference="def run(q, k, cos_cache, sin_cache, pos_ids, interleave):\n    return q, k\n",
            )
            definitions[def_name] = definition
            
            def_file = definitions_dir / f"{def_name}.json"
            with open(def_file, 'w') as f:
                json.dump(definition.model_dump(), f, indent=2)
    
    return definitions


def create_moe_definitions(
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    topk: int,
    output_dir: str,
):
    """Create MoE definitions for CUTLASS and TRT-LLM MoE kernels."""
    from flashinfer_bench.data import (
        AxisConst, AxisVar, Definition, TensorSpec
    )
    
    definitions_dir = Path(output_dir) / "definitions"
    definitions_dir.mkdir(parents=True, exist_ok=True)
    
    definitions = {}
    
    # CUTLASS MoE
    cutlass_def_name = f"cutlass_moe_e{num_experts}_h{hidden_size}_i{intermediate_size}_k{topk}"
    cutlass_definition = Definition(
        name=cutlass_def_name,
        op_type="fused_moe_cutlass",
        axes={
            "T": AxisVar(),  # Total tokens
            "E": AxisConst(value=num_experts),
            "H": AxisConst(value=hidden_size),
            "I": AxisConst(value=intermediate_size),
            "K": AxisConst(value=topk),
        },
        inputs={
            "hidden_states": TensorSpec(shape=["T", "H"], dtype="bfloat16"),
            "w1": TensorSpec(shape=["E", "I", "H"], dtype="bfloat16"),
            "w2": TensorSpec(shape=["E", "H", "I"], dtype="bfloat16"),
            "topk_weights": TensorSpec(shape=["T", "K"], dtype="float32"),
            "topk_ids": TensorSpec(shape=["T", "K"], dtype="int32"),
        },
        outputs={"output": TensorSpec(shape=["T", "H"], dtype="bfloat16")},
        reference="def run(hidden_states, w1, w2, topk_weights, topk_ids):\n    return hidden_states\n",
    )
    definitions[cutlass_def_name] = cutlass_definition
    
    def_file = definitions_dir / f"{cutlass_def_name}.json"
    with open(def_file, 'w') as f:
        json.dump(cutlass_definition.model_dump(), f, indent=2)
    
    # TRT-LLM FP8 MoE
    fp8_def_name = f"trtllm_fp8_moe_e{num_experts}_h{hidden_size}_i{intermediate_size}_k{topk}"
    fp8_definition = Definition(
        name=fp8_def_name,
        op_type="fused_moe_fp8",
        axes={
            "T": AxisVar(),
            "E": AxisConst(value=num_experts),
            "H": AxisConst(value=hidden_size),
            "I": AxisConst(value=intermediate_size),
            "K": AxisConst(value=topk),
        },
        inputs={
            "hidden_states": TensorSpec(shape=["T", "H"], dtype="bfloat16"),
        },
        outputs={"output": TensorSpec(shape=["T", "H"], dtype="bfloat16")},
        reference="def run(hidden_states):\n    return hidden_states\n",
    )
    definitions[fp8_def_name] = fp8_definition
    
    def_file = definitions_dir / f"{fp8_def_name}.json"
    with open(def_file, 'w') as f:
        json.dump(fp8_definition.model_dump(), f, indent=2)
    
    # TRT-LLM FP4 MoE
    fp4_def_name = f"trtllm_fp4_moe_e{num_experts}_h{hidden_size}"
    fp4_definition = Definition(
        name=fp4_def_name,
        op_type="fused_moe_fp4",
        axes={
            "T": AxisVar(),
            "E": AxisConst(value=num_experts),
            "H": AxisConst(value=hidden_size),
        },
        inputs={
            "hidden_states": TensorSpec(shape=["T", "H"], dtype="bfloat16"),
        },
        outputs={"output": TensorSpec(shape=["T", "H"], dtype="bfloat16")},
        reference="def run(hidden_states):\n    return hidden_states\n",
    )
    definitions[fp4_def_name] = fp4_definition
    
    def_file = definitions_dir / f"{fp4_def_name}.json"
    with open(def_file, 'w') as f:
        json.dump(fp4_definition.model_dump(), f, indent=2)
    
    return definitions


def create_gemm_definitions(hidden_size: int, intermediate_size: int, output_dir: str):
    """Create GEMM definitions for MLP layers.
    
    MLP typically uses:
    - Gate projection: [hidden_size] -> [intermediate_size]
    - Up projection: [hidden_size] -> [intermediate_size]  
    - Down projection: [intermediate_size] -> [hidden_size]
    """
    from flashinfer_bench.data import (
        AxisConst, AxisVar, Definition, TensorSpec
    )
    
    definitions_dir = Path(output_dir) / "definitions"
    definitions_dir.mkdir(parents=True, exist_ok=True)
    
    definitions = {}
    
    # FP4 GEMM for gate/up projection (hidden -> intermediate)
    # Note: FP4 data is packed as int8 (2 FP4 values per byte)
    fp4_gate_def_name = f"mm_fp4_k{hidden_size}_n{intermediate_size}"
    fp4_gate_definition = Definition(
        name=fp4_gate_def_name,
        op_type="gemm_fp4",
        axes={
            "M": AxisVar(),  # Batch/sequence dimension
            "K": AxisConst(value=hidden_size // 2),  # FP4 packed (2 values per byte)
            "N": AxisConst(value=intermediate_size),
        },
        inputs={
            "a": TensorSpec(shape=["M", "K"], dtype="int8"),  # FP4 packed as int8
            "b": TensorSpec(shape=["N", "K"], dtype="int8"),  # FP4 packed, transposed
            "a_descale": TensorSpec(shape=["M"], dtype="float32"),
            "b_descale": TensorSpec(shape=["N"], dtype="float32"),
        },
        outputs={"output": TensorSpec(shape=["M", "N"], dtype="bfloat16")},
        reference="def run(a, b, a_descale, b_descale):\n    return torch.zeros(a.shape[0], b.shape[0])\n",
    )
    definitions[fp4_gate_def_name] = fp4_gate_definition
    
    def_file = definitions_dir / f"{fp4_gate_def_name}.json"
    with open(def_file, 'w') as f:
        json.dump(fp4_gate_definition.model_dump(), f, indent=2)
    
    # FP4 GEMM for down projection (intermediate -> hidden)
    fp4_down_def_name = f"mm_fp4_k{intermediate_size}_n{hidden_size}"
    fp4_down_definition = Definition(
        name=fp4_down_def_name,
        op_type="gemm_fp4",
        axes={
            "M": AxisVar(),
            "K": AxisConst(value=intermediate_size // 2),  # FP4 packed
            "N": AxisConst(value=hidden_size),
        },
        inputs={
            "a": TensorSpec(shape=["M", "K"], dtype="int8"),  # FP4 packed as int8
            "b": TensorSpec(shape=["N", "K"], dtype="int8"),
            "a_descale": TensorSpec(shape=["M"], dtype="float32"),
            "b_descale": TensorSpec(shape=["N"], dtype="float32"),
        },
        outputs={"output": TensorSpec(shape=["M", "N"], dtype="bfloat16")},
        reference="def run(a, b, a_descale, b_descale):\n    return torch.zeros(a.shape[0], b.shape[0])\n",
    )
    definitions[fp4_down_def_name] = fp4_down_definition
    
    def_file = definitions_dir / f"{fp4_down_def_name}.json"
    with open(def_file, 'w') as f:
        json.dump(fp4_down_definition.model_dump(), f, indent=2)
    
    # FP8 GEMM for gate/up projection
    fp8_gate_def_name = f"mm_fp8_k{hidden_size}_n{intermediate_size}"
    fp8_gate_definition = Definition(
        name=fp8_gate_def_name,
        op_type="gemm_fp8",
        axes={
            "M": AxisVar(),
            "K": AxisConst(value=hidden_size),
            "N": AxisConst(value=intermediate_size),
        },
        inputs={
            "A": TensorSpec(shape=["M", "K"], dtype="float8_e4m3fn"),
            "B": TensorSpec(shape=["K", "N"], dtype="float8_e4m3fn"),
            "A_scale": TensorSpec(shape=["M"], dtype="float32"),
            "B_scale": TensorSpec(shape=["N"], dtype="float32"),
        },
        outputs={"output": TensorSpec(shape=["M", "N"], dtype="bfloat16")},
        reference="def run(A, B, A_scale, B_scale):\n    return torch.zeros(A.shape[0], B.shape[1])\n",
    )
    definitions[fp8_gate_def_name] = fp8_gate_definition
    
    def_file = definitions_dir / f"{fp8_gate_def_name}.json"
    with open(def_file, 'w') as f:
        json.dump(fp8_gate_definition.model_dump(), f, indent=2)
    
    # FP8 GEMM for down projection
    fp8_down_def_name = f"mm_fp8_k{intermediate_size}_n{hidden_size}"
    fp8_down_definition = Definition(
        name=fp8_down_def_name,
        op_type="gemm_fp8",
        axes={
            "M": AxisVar(),
            "K": AxisConst(value=intermediate_size),
            "N": AxisConst(value=hidden_size),
        },
        inputs={
            "A": TensorSpec(shape=["M", "K"], dtype="float8_e4m3fn"),
            "B": TensorSpec(shape=["K", "N"], dtype="float8_e4m3fn"),
            "A_scale": TensorSpec(shape=["M"], dtype="float32"),
            "B_scale": TensorSpec(shape=["N"], dtype="float32"),
        },
        outputs={"output": TensorSpec(shape=["M", "N"], dtype="bfloat16")},
        reference="def run(A, B, A_scale, B_scale):\n    return torch.zeros(A.shape[0], B.shape[1])\n",
    )
    definitions[fp8_down_def_name] = fp8_down_definition
    
    def_file = definitions_dir / f"{fp8_down_def_name}.json"
    with open(def_file, 'w') as f:
        json.dump(fp8_down_definition.model_dump(), f, indent=2)
    
    return definitions


def create_comm_definitions(hidden_sizes: list[int], output_dir: str):
    """Create communication definitions for AllReduce."""
    from flashinfer_bench.data import (
        AxisConst, AxisVar, Definition, TensorSpec
    )
    
    definitions_dir = Path(output_dir) / "definitions"
    definitions_dir.mkdir(parents=True, exist_ok=True)
    
    definitions = {}
    
    for hidden_size in hidden_sizes:
        # AllReduce fusion
        allreduce_def_name = f"allreduce_fusion_h{hidden_size}"
        allreduce_definition = Definition(
            name=allreduce_def_name,
            op_type="comm_allreduce",
            axes={
                "T": AxisVar(),  # Total tokens
                "H": AxisConst(value=hidden_size),
            },
            inputs={
                "input": TensorSpec(shape=["T", "H"], dtype="bfloat16"),
            },
            outputs={"output": TensorSpec(shape=["T", "H"], dtype="bfloat16")},
            reference="def run(input):\n    return input\n",
        )
        definitions[allreduce_def_name] = allreduce_definition
        
        def_file = definitions_dir / f"{allreduce_def_name}.json"
        with open(def_file, 'w') as f:
            json.dump(allreduce_definition.model_dump(), f, indent=2)
        
        # MnnvlMoe dispatch
        dispatch_def_name = f"mnnvl_moe_dispatch_h{hidden_size}"
        dispatch_definition = Definition(
            name=dispatch_def_name,
            op_type="moe_dispatch",
            axes={
                "T": AxisVar(),
                "H": AxisConst(value=hidden_size),
            },
            inputs={
                "hidden_states": TensorSpec(shape=["T", "H"], dtype="bfloat16"),
            },
            outputs={"output": TensorSpec(shape=["T", "H"], dtype="bfloat16")},
            reference="def run(hidden_states):\n    return hidden_states\n",
        )
        definitions[dispatch_def_name] = dispatch_definition
        
        def_file = definitions_dir / f"{dispatch_def_name}.json"
        with open(def_file, 'w') as f:
            json.dump(dispatch_definition.model_dump(), f, indent=2)
        
        # MnnvlMoe combine
        combine_def_name = f"mnnvl_moe_combine_h{hidden_size}"
        combine_definition = Definition(
            name=combine_def_name,
            op_type="moe_combine",
            axes={
                "T": AxisVar(),
                "H": AxisConst(value=hidden_size),
            },
            inputs={
                "hidden_states": TensorSpec(shape=["T", "H"], dtype="bfloat16"),
            },
            outputs={"output": TensorSpec(shape=["T", "H"], dtype="bfloat16")},
            reference="def run(hidden_states):\n    return hidden_states\n",
        )
        definitions[combine_def_name] = combine_definition
        
        def_file = definitions_dir / f"{combine_def_name}.json"
        with open(def_file, 'w') as f:
            json.dump(combine_definition.model_dump(), f, indent=2)
    
    return definitions


def setup_tracing_for_model(model_config, output_dir: str, page_size: int = 16):
    """Set up tracing definitions and configs for a specific model.
    
    Args:
        model_config: The model configuration from transformers
        output_dir: Directory to store traces
        page_size: KV cache page size (default 16, matching vLLM default)
    """
    from flashinfer_bench.data import TraceSet
    from flashinfer_bench.tracing import TracingConfig, enable_tracing
    
    all_definitions = {}
    
    # Determine hidden sizes to trace based on model
    hidden_sizes = []
    
    # Add model's hidden size
    if hasattr(model_config, 'hidden_size'):
        hidden_sizes.append(model_config.hidden_size)
    
    # For MoE models, add intermediate size
    if hasattr(model_config, 'intermediate_size'):
        hidden_sizes.append(model_config.intermediate_size)
    
    # Add some common sizes
    for size in [2048, 4096, 7168, 8192]:
        if size not in hidden_sizes:
            hidden_sizes.append(size)
    
    # Get model dimensions
    num_heads = getattr(model_config, 'num_attention_heads', 32)
    num_kv_heads = getattr(model_config, 'num_key_value_heads', num_heads)
    head_dim = getattr(model_config, 'head_dim', None)
    if head_dim is None and hasattr(model_config, 'hidden_size'):
        head_dim = model_config.hidden_size // num_heads
    vocab_size = getattr(model_config, 'vocab_size', 32000)
    intermediate_size = getattr(model_config, 'intermediate_size', 4096)
    
    # MoE parameters
    num_experts = getattr(model_config, 'num_local_experts', None)
    if num_experts is None:
        num_experts = getattr(model_config, 'num_experts', 8)
    topk = getattr(model_config, 'num_experts_per_tok', 2)
    
    print(f"\n✓ Creating definitions for model with:")
    print(f"    hidden_sizes: {hidden_sizes}")
    print(f"    vocab_size: {vocab_size}")
    print(f"    num_heads: {num_heads}, kv_heads: {num_kv_heads}, head_dim: {head_dim}")
    
    # 1. RMSNorm definitions
    print(f"✓ Creating RMSNorm definitions")
    rmsnorm_defs = create_rmsnorm_definitions(hidden_sizes, output_dir)
    all_definitions.update(rmsnorm_defs)
    
    # 2. Attention definitions
    if head_dim:
        print(f"✓ Creating attention definitions (page_size={page_size})")
        attn_defs = create_attention_definitions(num_heads, num_kv_heads, head_dim, output_dir, page_size=page_size)
        all_definitions.update(attn_defs)
    
    # 3. MLA definitions (DeepSeek models)
    kv_lora_rank = getattr(model_config, 'kv_lora_rank', None)
    qk_rope_head_dim = getattr(model_config, 'qk_rope_head_dim', None)
    if kv_lora_rank and qk_rope_head_dim:
        print(f"✓ Creating MLA definitions")
        mla_defs = create_mla_definitions(num_heads, kv_lora_rank, qk_rope_head_dim, output_dir, page_size=page_size)
        all_definitions.update(mla_defs)
    
    # 4. Sampling definitions
    print(f"✓ Creating sampling definitions (vocab_size={vocab_size})")
    sampling_defs = create_sampling_definitions(vocab_size, output_dir)
    all_definitions.update(sampling_defs)
    
    # 5. Activation definitions
    print(f"✓ Creating activation definitions (SiLU, GELU)")
    activation_defs = create_activation_definitions(hidden_sizes, output_dir)
    all_definitions.update(activation_defs)
    
    # 6. RoPE definitions
    if head_dim:
        head_dims = [head_dim]
        # Add common head dims
        for hd in [64, 128, 256]:
            if hd not in head_dims:
                head_dims.append(hd)
        print(f"✓ Creating RoPE definitions (head_dims={head_dims})")
        rope_defs = create_rope_definitions(head_dims, output_dir)
        all_definitions.update(rope_defs)
    
    # 7. MoE definitions (if model has MoE)
    if hasattr(model_config, 'num_local_experts') or hasattr(model_config, 'num_experts'):
        print(f"✓ Creating MoE definitions (experts={num_experts}, topk={topk})")
        moe_defs = create_moe_definitions(
            num_experts=num_experts,
            hidden_size=model_config.hidden_size,
            intermediate_size=intermediate_size,
            topk=topk,
            output_dir=output_dir,
        )
        all_definitions.update(moe_defs)
    
    # 8. Communication definitions (AllReduce, All2All)
    print(f"✓ Creating communication definitions")
    comm_defs = create_comm_definitions(hidden_sizes, output_dir)
    all_definitions.update(comm_defs)
    
    # 9. GEMM definitions (for MLP layers)
    print(f"✓ Creating GEMM definitions (FP4/FP8 for MLP)")
    gemm_defs = create_gemm_definitions(
        hidden_size=model_config.hidden_size,
        intermediate_size=intermediate_size,
        output_dir=output_dir,
    )
    all_definitions.update(gemm_defs)
    
    # Create TraceSet
    trace_set = TraceSet(
        root=output_dir,
        definitions=all_definitions,
        solutions={},
        traces={}
    )
    
    # Create tracing configs for all definitions
    tracing_configs = {}
    for def_name in all_definitions.keys():
        tracing_configs[def_name] = TracingConfig(
            input_dump_policy="dump_all",
            filter_policy="keep_first_by_axes"
        )
    
    # Enable tracing
    runtime = enable_tracing(dataset_path=output_dir, tracing_configs=tracing_configs)
    
    print(f"\n✓ Tracing enabled for {len(tracing_configs)} operation types")
    
    return runtime


def generate_traces(
    model_id: str,
    model_name: str,
    tensor_parallel_size: int,
    num_prompts: int,
    max_tokens: int,
    output_dir: str,
    quantization: str | None = None,
    trust_remote_code: bool = True,
    extra_env_vars: dict | None = None,
):
    """Generate FlashInfer traces for a specific model."""
    
    # Set any extra environment variables
    if extra_env_vars:
        for key, value in extra_env_vars.items():
            os.environ[key] = value
    
    # Set FlashInfer environment variables
    os.environ.setdefault("VLLM_USE_FLASHINFER", "1")
    os.environ.setdefault("VLLM_USE_FLASHINFER_NORM", "1")  # Enable FlashInfer RMSNorm
    
    # Set FlashInfer-Bench tracing environment variables
    # CRITICAL: These must be set BEFORE vLLM spawns worker processes
    os.environ["FIB_ENABLE_TRACING"] = "1"
    os.environ["FIB_DATASET_PATH"] = output_dir
    os.environ["FIB_ENABLE_APPLY"] = "1"
    
    print(f"\n✓ Environment variables set for tracing:")
    print(f"  FIB_ENABLE_TRACING=1")
    print(f"  FIB_DATASET_PATH={output_dir}")
    print(f"  FIB_ENABLE_APPLY=1")
    
    print(f"\n{'='*70}")
    print(f"Generating traces for: {model_name}")
    print(f"Model ID: {model_id}")
    print(f"TP: {tensor_parallel_size}, Quantization: {quantization or 'none'}")
    print(f"Output: {output_dir}")
    print(f"{'='*70}\n")
    
    # Import vLLM
    from vllm import LLM, SamplingParams
    from transformers import AutoConfig
    
    # Load model config to get dimensions
    print("Loading model config...")
    model_config = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    print(f"✓ Model config loaded")
    print(f"  Hidden size: {model_config.hidden_size}")
    if hasattr(model_config, 'intermediate_size'):
        print(f"  Intermediate size: {model_config.intermediate_size}")
    if hasattr(model_config, 'num_attention_heads'):
        print(f"  Attention heads: {model_config.num_attention_heads}")
    if hasattr(model_config, 'num_key_value_heads'):
        print(f"  KV heads: {model_config.num_key_value_heads}")
    
    # Set up tracing with model-specific definitions
    runtime = setup_tracing_for_model(model_config, output_dir)
    
    # Initialize the model
    # NOTE: enforce_eager=True is REQUIRED for FlashInfer-Bench tracing to work.
    # When using torch.compile, the adapter wrappers are traced and compiled away,
    # preventing the tracing logic from executing at runtime.
    # See docs/source/design/flashinfer_integration_issues.md for details.
    llm_kwargs = {
        "model": model_id,
        "tensor_parallel_size": tensor_parallel_size,
        "max_model_len": 2048,
        "trust_remote_code": trust_remote_code,
        "gpu_memory_utilization": 0.7,
        "enforce_eager": True,  # REQUIRED for FlashInfer-Bench tracing
    }
    
    if quantization:
        llm_kwargs["quantization"] = quantization
    
    print(f"\nInitializing LLM (enforce_eager=True required for tracing)...")
    llm = LLM(**llm_kwargs)
    
    # Generate diverse prompts
    prompts = [
        "What is 2+2?",
        "Hello!",
        "What is the capital of France? Please answer briefly.",
        "Write a haiku about programming.",
        "Explain machine learning in simple terms.",
        "Write a short story about a robot.",
        "Compare the Renaissance and Enlightenment periods.",
    ]
    
    test_prompts = [prompts[i % len(prompts)] for i in range(num_prompts)]
    
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=max_tokens,
    )
    
    print(f"\nRunning inference with {len(test_prompts)} prompts...")
    outputs = llm.generate(test_prompts, sampling_params)
    
    print(f"\n✓ Generated {len(outputs)} outputs")
    if outputs:
        print(f"Sample: {outputs[0].outputs[0].text[:100]}...")
    
    del llm
    
    # Flush traces
    if runtime:
        runtime.flush()
    
    print(f"\n✓ Traces flushed to {output_dir}")
    
    # Check if any traces were actually captured
    import glob
    workload_files = glob.glob(os.path.join(output_dir, "workloads/**/*.jsonl"), recursive=True)
    trace_count = 0
    for f in workload_files:
        with open(f) as fh:
            trace_count += len(fh.readlines())
    
    if trace_count > 0:
        print(f"✓ Captured {trace_count} workload trace(s)")
        for f in workload_files:
            with open(f) as fh:
                count = len(fh.readlines())
                print(f"  - {os.path.basename(f)}: {count} traces")
    else:
        print(f"\n⚠ No traces captured.")
        print(f"This may happen if:")
        print(f"  1. FlashInfer operators are not being used (check VLLM_USE_FLASHINFER=1)")
        print(f"  2. The adapters are not matching the operator signatures")
        print(f"  3. The model architecture doesn't use the traced operators")
    
    return trace_count > 0


def main():
    parser = argparse.ArgumentParser(
        description="Generate FlashInfer Bench traces from vLLM inference"
    )
    parser.add_argument(
        "--model", type=str, default="qwen",
        choices=["qwen", "llama", "gpt-oss", "all"],
        help="Model to generate traces for",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory to store traces",
    )
    parser.add_argument(
        "--num-prompts", type=int, default=5,
        help="Number of prompts to run",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=128,
        help="Maximum tokens per prompt",
    )
    parser.add_argument(
        "--tp", type=int, default=2,
        help="Tensor parallel size",
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("FLASHINFER BENCH TRACE GENERATION")
    print("=" * 70)
    print("\nNOTE: This script uses enforce_eager=True because FlashInfer-Bench")
    print("      tracing is not compatible with torch.compile.")
    print("      See docs/source/design/flashinfer_integration_issues.md")
    
    # Check compatibility first
    is_compatible, message = check_compatibility()
    if not is_compatible:
        print(f"\n✗ ERROR: {message}")
        return 1
    
    print(f"\n✓ {message}")
    
    output_dir = args.output_dir or str(Path.home() / ".cache" / "flashinfer_bench" / "vllm_traces")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    models = {
        "qwen": {
            "model_id": "Qwen/Qwen3-30B-A3B-Instruct-2507",
            "model_name": "Qwen3-30B-A3B MoE",
            "tensor_parallel_size": args.tp,
        },
        "llama": {
            "model_id": "meta-llama/Llama-3.1-70B-Instruct",
            "model_name": "Llama-3.1-70B",
            "tensor_parallel_size": max(args.tp, 4),
        },
        "gpt-oss": {
            "model_id": "openai/gpt-oss-120b",
            "model_name": "GPT-OSS-120B",
            "tensor_parallel_size": max(args.tp, 4),
            "extra_env_vars": {"VLLM_ATTENTION_BACKEND": "FLASH_ATTN"},
        },
    }
    
    models_to_run = list(models.keys()) if args.model == "all" else [args.model]
    results = {}
    
    for model_key in models_to_run:
        config = models[model_key]
        model_output_dir = os.path.join(output_dir, model_key)
        Path(model_output_dir).mkdir(parents=True, exist_ok=True)
        
        try:
            success = generate_traces(
                model_id=config["model_id"],
                model_name=config["model_name"],
                tensor_parallel_size=config["tensor_parallel_size"],
                num_prompts=args.num_prompts,
                max_tokens=args.max_tokens,
                output_dir=model_output_dir,
                extra_env_vars=config.get("extra_env_vars"),
            )
            results[model_key] = "✓ SUCCESS" if success else "⚠ NO TRACES"
        except Exception as e:
            print(f"\n✗ Error: {e}")
            import traceback
            traceback.print_exc()
            results[model_key] = f"✗ ERROR: {e}"
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for model_key, result in results.items():
        print(f"  {model_key}: {result}")
    
    print(f"\nTraces saved to: {output_dir}")
    print("\nTraced operators:")
    print("  Attention:")
    print("    ✓ GQA Paged Decode/Prefill")
    print("    ✓ GQA Ragged Prefill")
    print("    ✓ MLA Paged (DeepSeek)")
    print("  Normalization:")
    print("    ✓ RMSNorm (fused_add_rmsnorm)")
    print("  Sampling:")
    print("    ✓ Top-k/Top-p sampling")
    print("  Activations:")
    print("    ✓ SiLU, GELU, GELU-tanh")
    print("  Positional Encoding:")
    print("    ✓ RoPE")
    print("  MoE:")
    print("    ✓ CUTLASS MoE, TRT-LLM FP8/FP4 MoE")
    print("  GEMM (quantized models only):")
    print("    ✓ mm_fp4 (FP4 matmul)")
    print("    ✓ bmm_fp8 (FP8 batched matmul)")
    print("    ✓ grouped_gemm_nt_masked (MoE CUTEDSL)")
    print("  Communication:")
    print("    ✓ AllReduce, MnnvlMoe All2All")
    print("\nTo analyze traces:")
    print(f"  flashinfer-bench run --local {output_dir}")
    
    return 0 if all("SUCCESS" in r for r in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())

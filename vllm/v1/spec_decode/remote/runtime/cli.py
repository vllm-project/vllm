# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU-only startup kept out of the protocol server's CPU import path."""

import argparse

import torch

from vllm.config import AttentionConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.spec_decode.remote.runtime.adapter import TargetConditionedDraftAdapter
from vllm.v1.spec_decode.remote.runtime.gpu import make_gpu_adapter
from vllm.v1.spec_decode.remote.runtime.schema import (
    RuntimeContract,
    create_runtime_contract,
)


def create_gpu_adapter(
    args: argparse.Namespace,
) -> tuple[TargetConditionedDraftAdapter, RuntimeContract]:
    if not args.target_model or not args.draft_model:
        raise ValueError("--adapter gpu requires --target-model and --draft-model")
    attention = AttentionConfig(
        backend=(
            AttentionBackendEnum[args.target_attention_backend]
            if args.target_attention_backend
            else None
        )
    )
    config = EngineArgs(
        model=args.target_model,
        revision=args.target_revision,
        tokenizer_revision=args.target_revision,
        trust_remote_code=args.trust_remote_code,
        dtype=args.dtype,
        tensor_parallel_size=1,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_batch_size,
        max_num_batched_tokens=args.max_feature_tokens,
        enforce_eager=True,
        async_scheduling=False,
        enable_prefix_caching=False,
        enable_chunked_prefill=True,
        attention_config=attention,
        moe_backend=args.moe_backend,
        block_size=args.block_size,
        speculative_config={
            "model": args.draft_model,
            "revision": args.draft_revision,
            "method": args.method,
            "num_speculative_tokens": args.num_speculative_tokens,
            "draft_tensor_parallel_size": 1,
            "attention_backend": args.attention_backend,
        },
    ).create_engine_config()
    contract = create_runtime_contract(config)
    adapter = make_gpu_adapter(
        config,
        device=torch.device(args.device),
        kv_cache_bytes=int(args.draft_kv_cache_gib * 1024**3),
        shared_weights_source=args.shared_weights_source,
    )
    return adapter, contract

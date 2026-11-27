# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run with torchrun to measure fine-grained TP vocabulary latency and memory."""

import argparse
import json
import os

import torch

from vllm.config import (
    FineGrainedTPConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.distributed import (
    cleanup_dist_env_and_memory,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dp", type=int, default=2)
    parser.add_argument("--pcp", type=int, default=1)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--embedding-tp-size", type=int, default=1)
    parser.add_argument("--lmhead-tp-size", type=int, default=1)
    parser.add_argument("--vocab", type=int, default=154880)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    assert world == args.dp * args.pcp * args.tp
    torch.accelerator.set_device_index(local_rank)
    init_distributed_environment(world, rank, "env://", local_rank)

    config = VllmConfig(
        parallel_config=ParallelConfig(
            data_parallel_size=args.dp,
            data_parallel_rank=rank // (args.pcp * args.tp),
            prefill_context_parallel_size=args.pcp,
            tensor_parallel_size=args.tp,
            enable_expert_parallel=True,
            fine_grained_tp_config=FineGrainedTPConfig(
                embedding_tp_size=args.embedding_tp_size,
                lmhead_tp_size=args.lmhead_tp_size,
            ),
        ),
        scheduler_config=SchedulerConfig(
            max_model_len=args.tokens,
            is_encoder_decoder=False,
            max_num_batched_tokens=args.tokens,
            max_num_seqs=args.requests,
        ),
    )
    with set_current_vllm_config(config), torch.device(f"cuda:{local_rank}"):
        initialize_model_parallel(
            tensor_model_parallel_size=args.tp,
            prefill_context_model_parallel_size=args.pcp,
        )
        embedding = VocabParallelEmbedding(
            args.vocab,
            args.hidden,
            params_dtype=torch.bfloat16,
            prefix="model.embed_tokens",
        )
        head = ParallelLMHead(
            args.vocab,
            args.hidden,
            params_dtype=torch.bfloat16,
            prefix="lm_head",
        )
        torch.manual_seed(7)
        embedding.weight.normal_(std=0.01)
        head.weight.normal_(std=0.01)
        processor = LogitsProcessor(args.vocab)

        embed_owner = rank // args.tp
        ids = (torch.arange(args.tokens) + embed_owner * 13) % args.vocab
        states = torch.randn(args.requests, args.hidden, dtype=torch.bfloat16)
        weight_bytes = sum(
            layer.weight.numel() * layer.weight.element_size()
            for layer in (embedding, head)
        )
        results = {}
        for name, fn in (
            ("embedding", lambda: embedding(ids)),
            ("lm_head", lambda: processor(head, states)),
        ):
            for _ in range(10):
                fn()
            torch.accelerator.synchronize()
            torch.distributed.barrier()
            torch.accelerator.reset_peak_memory_stats()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(args.iterations):
                fn()
            end.record()
            end.synchronize()
            free, total = torch.accelerator.get_memory_info()
            results[name] = {
                "ms": start.elapsed_time(end) / args.iterations,
                "peak_allocated_bytes": torch.accelerator.max_memory_allocated(),
                "peak_reserved_bytes": torch.accelerator.max_memory_reserved(),
                "device_used_bytes": total - free,
            }
        print(
            json.dumps(
                {
                    "rank": rank,
                    "args": vars(args),
                    "embed_shards": embedding.tp_size,
                    "lmhead_shards": head.tp_size,
                    "weight_bytes": weight_bytes,
                    "results": results,
                }
            ),
            flush=True,
        )
    cleanup_dist_env_and_memory()


if __name__ == "__main__":
    main()

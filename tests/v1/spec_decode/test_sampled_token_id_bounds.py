# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tile-local argmax must never emit a token id >= vocab_size.

The reductions compute ``token_id = block_idx * BLOCK_SIZE + idx`` where the
tile's out-of-vocab tail lanes are loaded as -inf. When every in-vocab lane is
also -inf/NaN, the reduction can settle on a tail lane. Nothing downstream
bounds a sampled token id, and DeepSeek-V4's hash-MoE router gathers
``tid2eid[token_id * topk]`` directly, so an unbounded id faults the context.
Reported by alexbi29 in vllm-project/vllm#41834; adjacent to d8885a3335 and
upstream #50183 but distinct — those keep NaN from winning the reduction,
this bounds the id the winner composes."""

import re

MODULES = (
    "vllm/v1/worker/gpu/sample/gumbel.py",
    "vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py",
)

UNBOUNDED = re.compile(r"token_id\s*=\s*block_idx \* BLOCK_SIZE \+ idx\b")
BOUNDED = re.compile(
    r"token_id\s*=\s*tl\.minimum\(block_idx \* BLOCK_SIZE \+ idx, vocab_size - 1\)"
)


def test_every_tile_argmax_token_id_is_clamped():
    import vllm

    root = vllm.__path__[0].rsplit("/vllm", 1)[0]
    for rel in MODULES:
        with open(f"{root}/{rel}") as f:
            src = f.read()
        assert not UNBOUNDED.search(src), (
            f"{rel} composes an unclamped tile-local token id; a degenerate "
            "tile can emit id >= vocab_size and fault the hash-MoE gather"
        )
        assert BOUNDED.search(src), f"{rel} lost the clamped composition"

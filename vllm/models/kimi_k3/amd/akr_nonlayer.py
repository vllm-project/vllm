# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sub-switches and shared helpers of the Kimi-K3.1 non-layer amd_kernels routes
(embedding, lm_head + logits exchange, sampling, DSpark drafter).

Each piece has a sub-switch `VLLM_KIMI_AKR_<NAME>` (`0` off, `1` on, unset = its
default below) that applies on top of its route (`akr_config.on(route)`, which
`VLLM_KIMI_AKR` and `--kernel-config` control):

| NAME      | route       | default | piece                                            |
|-----------|-------------|---------|--------------------------------------------------|
| EMBED     | collectives | on      | `vocab_gather_comm` (target + drafter block ids) |
| LMHEAD    | gemm        | on      | lm_head GEMM + all-gather (target, drafter)      |
| SHARDED   | sampling    | off     | `--enable-batch-sharded-sampling` owner exchange |
| TOPP      | sampling    | on      | `top_p_mask` in `split_top_p_override`           |
| MARKOV    | sampling    | on      | fused persistent Markov draft chain              |
| CTXPROJ   | draft       | on      | sharded drafter context projection + norm        |
| DMLP      | draft       | on      | drafter dense MLP                                |
| DMLA      | draft       | on      | drafter dense MLA decode (MLA drafts)            |
| DAR       | draft       | on      | drafter fused all-reduce + RMSNorm               |
| DQKV      | draft       | on      | drafter q / k norm + RoPE + KV insert (GQA)      |
| DCTXKV    | draft       | on      | drafter context K norm + RoPE + KV inserts (GQA) |
| FINALNORM | norm        | off     | final attn_res + norm in `allreduce_attnres`     |
"""

import os

from vllm.logger import init_logger
from vllm.models.kimi_k3.amd import akr_config

logger = init_logger(__name__)

# name -> (route, default)
SWITCHES: dict[str, tuple[str, bool]] = {
    "EMBED": ("collectives", True),
    "LMHEAD": ("gemm", True),
    "SHARDED": ("sampling", False),
    "TOPP": ("sampling", True),
    "MARKOV": ("sampling", True),
    "CTXPROJ": ("draft", True),
    "DMLP": ("draft", True),
    "DMLA": ("draft", True),
    "DAR": ("draft", True),
    "DQKV": ("draft", True),
    "DCTXKV": ("draft", True),
    "FINALNORM": ("norm", False),
}


def switch(name: str) -> bool:
    """Whether the piece `name` is requested: its route is on and its sub-switch
    env is not `0` (`1` forces it on when its default is off)."""
    route, default = SWITCHES[name]
    if not akr_config.on(route):
        return False
    v = os.environ.get(f"VLLM_KIMI_AKR_{name}", "").strip().lower()
    if v in ("0", "false", "off", "no"):
        return False
    if v in ("1", "true", "on", "yes"):
        return True
    return default


def tp_group():
    from vllm.distributed.parallel_state import get_tp_group

    return get_tp_group()


def tp_is_eight() -> bool:
    return tp_group().world_size == 8


def blocked(name: str, reason: str | None) -> bool:
    """True (warned once) when `reason` keeps the piece `name` on the stock path."""
    return akr_config.unsupported_reason(name.lower(), reason)

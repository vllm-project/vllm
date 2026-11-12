# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Which MiniMax-M3 monokernels `--monokernel` may run.

By default every one that fits the deployment. `additional_config` can narrow
that, e.g. `--additional-config '{"minimax_m3_monokernel": ["drafter"]}'`.
"""

from vllm.config import VllmConfig

KEY = "minimax_m3_monokernel"
DECODE = "decode"  # the whole-layer mono decode, else the FFN-only decode
DRAFTER = "drafter"  # the DSpark drafter kernels
ALL = frozenset((DECODE, DRAFTER))


def enabled_monokernels(vllm_config: VllmConfig) -> frozenset[str]:
    extra = vllm_config.additional_config
    names = extra.get(KEY) if isinstance(extra, dict) else None
    if names is None:
        return ALL
    selected = frozenset([names] if isinstance(names, str) else names)
    if not selected or not selected <= ALL:
        raise ValueError(
            f"additional_config[{KEY!r}] must name one or more of {sorted(ALL)}, "
            f"got {names!r}"
        )
    return selected

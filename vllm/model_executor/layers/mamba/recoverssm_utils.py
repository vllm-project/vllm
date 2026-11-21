# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-independent RecoverSSM Python helpers: the activation predicate shared by
layers, models and backends, and argument validation that raises ``ValueError``
before any raw-pointer kernel launch."""

import torch


def recoverssm_require(cond: bool, msg: str, component: str = "RecoverSSM") -> None:
    if not cond:
        raise ValueError(f"{component}: {msg}")


def uses_recoverssm(cache_config, num_speculative_tokens: int) -> bool:
    """True when this Qwen GDN / PLE layer runs RecoverSSM for its recurrent states.

    VllmConfig sets ``cache_config.use_gdn_recoverssm`` for the Qwen GDN
    architectures (Kimi-K3 KDA has its own ``use_kda_recoverssm``). Without
    speculative tokens there is nothing to recover, so the stock path runs."""
    return (
        bool(getattr(cache_config, "use_gdn_recoverssm", False))
        and num_speculative_tokens > 0
    )


def recoverssm_ptr_table(tensors, device) -> torch.Tensor:
    """int64 table of the tensors' data pointers, keeping each uint64 bit pattern.

    Device pointers with the top bit set (XPU) overflow a plain int64 conversion;
    the kernels only cast the value back to a pointer."""
    ptrs = [t.data_ptr() for t in tensors]
    ptrs = [p if p < (1 << 63) else p - (1 << 64) for p in ptrs]
    return torch.tensor(ptrs, dtype=torch.int64, device=device)


__all__ = ["recoverssm_ptr_table", "recoverssm_require", "uses_recoverssm"]

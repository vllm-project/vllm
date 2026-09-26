# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-independent RecoverSSM Python helpers: the activation predicate shared by
layers, models and backends, and argument validation that raises ``ValueError``
before any raw-pointer kernel launch."""


def recoverssm_require(cond: bool, msg: str, component: str = "RecoverSSM") -> None:
    if not cond:
        raise ValueError(f"{component}: {msg}")


def uses_recoverssm(cache_config, num_speculative_tokens: int) -> bool:
    """True when this model instance runs RecoverSSM for its recurrent states.

    VllmConfig sets ``cache_config.use_recoverssm`` for every model whose recurrent
    states support it (Kimi-K3 KDA, Qwen GDN/PLE). Without speculative tokens there is
    nothing to recover, so the stock path runs."""
    return (
        bool(getattr(cache_config, "use_recoverssm", False))
        and num_speculative_tokens > 0
    )


__all__ = ["recoverssm_require", "uses_recoverssm"]

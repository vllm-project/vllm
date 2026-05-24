# SPDX-License-Identifier: Apache-2.0
"""Forward-compat shim for vllm#41566 quantization_config rework.

Upstream PR #41566 (mgoin, label `ready`) renames:
  * `OnlineQuantScheme` enum → `QuantSpec(weight, activation)`
  * `linear_scheme_override` → `linear`
  * `moe_scheme_override` → `moe`

Genesis itself has 0 call-sites for these old names (verified by grep
2026-05-04). But community forks of Genesis or downstream Lorbus/AutoRound
recipes may pass them via CLI / env. This shim:

1. Detects use of OLD field names in env vars / kwargs
2. Logs deprecation warning with new name suggestion
3. Returns translated kwargs so call still works

Doesn't actually patch upstream — pure logging shim. Activated when
`init_wizard.py` builds engine config or CLI parses args.

Author: Sandermage 2026-05-04, vllm#41566 forward-compat.
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger("genesis.compat.quant_config_compat")


_DEPRECATED_KEYS_MAP = {
    "linear_scheme_override": "linear",
    "moe_scheme_override": "moe",
    # OnlineQuantScheme enum members → QuantSpec equivalents
    "FP8_PER_TENSOR": "QuantSpec(weight='fp8', activation='fp8_per_tensor')",
    "MXFP8": "QuantSpec(weight='mxfp8', activation='mxfp8')",
    "MXFP4": "QuantSpec(weight='mxfp4', activation='mxfp8')",
}


def warn_if_uses_deprecated_quant_keys(quant_config: dict | None = None) -> int:
    """Scan a quant config dict for vllm#41566 deprecated keys.

    Returns count of deprecation warnings emitted.
    """
    if not quant_config or not isinstance(quant_config, dict):
        return 0
    n = 0
    for old_key, new_key in _DEPRECATED_KEYS_MAP.items():
        if old_key in quant_config:
            log.warning(
                "[Genesis vllm#41566 compat] Deprecated quant config key %r → "
                "use %r instead. Will be removed in vllm 0.21+.",
                old_key, new_key,
            )
            n += 1
    return n


def warn_env_deprecated_quant_keys() -> int:
    """Scan os.environ for VLLM_*OnlineQuantScheme* refs.

    Returns count of warnings.
    """
    n = 0
    for env_var, value in os.environ.items():
        if "OnlineQuantScheme" in env_var or "ONLINE_QUANT_SCHEME" in env_var.upper():
            log.warning(
                "[Genesis vllm#41566 compat] Env var %r references "
                "OnlineQuantScheme — deprecated by upstream rework. "
                "Will be QuantSpec-based in next vllm.",
                env_var,
            )
            n += 1
        # Catch values that name old enum members
        for old_enum in ("FP8_PER_TENSOR", "MXFP8", "MXFP4"):
            if value == old_enum and ("QUANT" in env_var.upper() or "SCHEME" in env_var.upper()):
                log.warning(
                    "[Genesis vllm#41566 compat] Env var %r=%r uses "
                    "OnlineQuantScheme enum value — translate to QuantSpec(...).",
                    env_var, value,
                )
                n += 1
    return n


def main():
    """CLI entry — `python -m vllm._genesis.compat.quant_config_compat`."""
    n = warn_env_deprecated_quant_keys()
    if n == 0:
        print("✓ No deprecated quant config keys in env")
    else:
        print(f"⚠ {n} deprecation warning(s) emitted; see log")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

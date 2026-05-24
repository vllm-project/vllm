# SPDX-License-Identifier: Apache-2.0
"""Genesis compat — unified compatibility / UX / diagnostic layer.

This module is the single home for everything Genesis needs to detect
about the environment (hardware, software, model, vllm version) and
the surface that the operator interacts with for setup + diagnosis +
maintenance.

Modules
-------
- doctor          : `python3 -m vllm._genesis.compat.doctor`
                    Unified diagnostic report (hw + sw + model + patches)
- init_wizard     : `python3 -m vllm._genesis.compat.init_wizard`
                    First-run interactive setup
- version_check   : version-range matching (vllm / torch / cuda / triton / driver)
- predicates      : richer applies_to evaluator (AND / OR / NOT trees)
- lifecycle       : patch lifecycle state machine (experimental / stable / ...)
- gpu_profile     : per-GPU datasheet (HBM bandwidth, L2, sm class)
                    [re-exported from legacy vllm._genesis.gpu_profile for now]
- model_detect    : model class + hybrid + MoE detection
                    [re-exported from legacy vllm._genesis.model_detect for now]
- config_detect   : runtime vllm config introspection
                    [re-exported from legacy vllm._genesis.config_detect for now]
- models.registry : SUPPORTED_MODELS dict + utilities
- models.pull     : `genesis pull <key>` — HF download + verify + launch
                    script generator
- fingerprints/   : reference benchmark JSONs per (hardware × model × patch_set)

Backwards compatibility
-----------------------
The legacy import paths
    vllm._genesis.gpu_profile
    vllm._genesis.model_detect
    vllm._genesis.config_detect
continue to work as before — no production code is broken by this
reorganisation. The new home for these in `compat/` re-exports the
same symbols, so old + new imports both resolve.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

# Re-export the main detectors so callers can `from vllm._genesis.compat
# import gpu_profile, model_detect, config_detect` with one import statement.
# Lazy imports to avoid pulling torch / vllm into compat/__init__ unless
# actually needed (some compat tools — version_check, predicates, lifecycle —
# don't need torch).

__all__ = [
    "version_check",
    "predicates",
    "lifecycle",
]

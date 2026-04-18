# SPDX-License-Identifier: Apache-2.0
"""Pairwise FP4 rotation quantization – research prototype.

Modules:
    rotation_applier  – Givens rotation engine
    channel_monitor   – risk-score computation and caching
    pair_constructor  – channel pairing strategies
    angle_solver      – rotation angle computation
    fp4_quant_policy  – FP4 quantization helpers (wraps nvfp4 utils)
    rotation_plan     – RotationPlan builder (orchestrator)
    utils             – shared types (RotationPlan dataclass, constants)
"""

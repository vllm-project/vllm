# Specialized Models

This directory contains opt-in implementations for checkpoints that need a
model-specific execution path. Each implementation targets a concrete model
architecture, quantization format, attention backend, and hardware profile.

Set `VLLM_USE_SPECIALIZED_MODELS=1` to allow the model registry to prefer a
specialized implementation when one is available. Specialized wrappers must
validate their target checkpoint and runtime requirements before replacing the
generic model implementation.

The generic implementations in `../models/` remain the default and should stay
the long-term home for broadly reusable model support.

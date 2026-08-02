# [CI] Solidify speculative decoding E2E coverage

- Standardize the reorganized spec-decode tests on `vllm_runner` for consistent setup, defaults, and cleanup.
- Add AMD CI mirrors for DFlash and DSpark, including NVFP4 targets through ROCm emulation.
- Enable supported ROCm cases and retain skips only for failures reproduced locally.
- Make prompt selection deterministic, remove unnecessary CI progress output, and select platform-appropriate backends.
- Strengthen acceptance thresholds, expected-failure handling, and failure diagnostics.
- Keep the directory-scoped CI contract introduced by the original test reorganization.

The review on #50330 highlighted that the reorganized tests should consistently use shared runner infrastructure and receive stable AMD coverage. This follow-up applies that feedback across the new spec-decode area while hardening deterministic setup and actionable failure reporting. Local MI300 validation covers DFlash, DSpark, Eagle, MTP, and LoRA paths, including NVFP4 targets through ROCm emulation.

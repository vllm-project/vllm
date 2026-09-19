# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RL HTTP contracts and lifecycle integration (RFCs #45585 and #55421).

State transitions cover pause/resume, sleep/wake, and weight synchronization.
Information retrieval covers response schemas and weight-version inspection.
Consistency covers routing and numerical results across lifecycle boundaries.

Component coverage remains next to its implementation (paths relative to tests/):
    v1/engine/test_engine_core_sleep.py: drain-before-offload and partial wake.
    v1/engine/test_async_llm.py: pause modes and queued-request cancellation.
    v1/distributed/test_async_llm_dp.py: DP pause, drain, and rank coordination.
    v1/worker/test_sleep_mode_backend.py: sleep backend factory and capabilities.
    v1/worker/test_gpu_worker_weight_transfer.py: update-session error recovery.
    basic_correctness/test_mem.py: allocator, graphs, and offline sleep/reload.
    models/language/generation/test_gdn_sleep_wake.py: hybrid state restoration.
    entrypoints/weight_transfer/test_weight_transfer_llm.py: LLM API delegation.
    distributed/test_weight_transfer.py: transport and trainer/client protocols.
    model_executor/model_loader/test_reload.py: checkpoint/quantization reload.
    model_executor/test_routed_experts_capture.py: capture and rank mapping.
    kernels/moe/test_routed_experts_capture_monolithic.py: backend capture.
    v1/sample/ and v1/determinism/: general logprobs and determinism.

These references document complementary coverage; they do not collect tests.
CI configuration owns resource selection. PR progress is tracked in RFC #55421.
"""

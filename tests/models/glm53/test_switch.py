# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3 low-latency switch (vllm/models/glm53/nvidia/switch.py)."""

from types import SimpleNamespace

import pytest

from vllm.models.glm53.nvidia import switch


@pytest.fixture
def clean_env(monkeypatch):
    monkeypatch.setattr(switch, "_resolved", None)
    return monkeypatch


def _config(
    tp=8,
    pp=1,
    kv="fp8",
    backend="FLASHINFER_MLA_SPARSE",
    v2=True,
    seqs=1,
    ep=False,
    dp=1,
    dcp=1,
    lora=None,
    kv_transfer=None,
    moe="deep_gemm",
):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp,
            pipeline_parallel_size=pp,
            enable_expert_parallel=ep,
            data_parallel_size=dp,
            decode_context_parallel_size=dcp,
        ),
        cache_config=SimpleNamespace(cache_dtype=kv),
        attention_config=SimpleNamespace(backend=SimpleNamespace(name=backend)),
        scheduler_config=SimpleNamespace(max_num_seqs=seqs),
        lora_config=lora,
        kv_transfer_config=kv_transfer,
        kernel_config=SimpleNamespace(moe_backend=moe),
        use_v2_model_runner=v2,
    )


def test_off_by_default(clean_env):
    assert switch.from_env() == switch.LowLatencyDecode()
    assert not switch.current().enabled


def test_resolve_off_needs_no_collective(clean_env):
    record = switch.resolve(_config(tp=4, kv="auto", v2=False))
    assert record == switch.LowLatencyDecode()
    assert switch.current() is record


def test_unsupported_lists_every_reason(clean_env):
    missing = switch._unsupported(_config(tp=4, kv="auto", v2=False))
    joined = "; ".join(missing)
    for need in ("tensor parallel size 8", "FLASHINFER_MLA_SPARSE", "Model Runner V2"):
        assert need in joined


@pytest.mark.parametrize(
    ("override", "need"),
    [
        ({"seqs": 2}, "--max-num-seqs 1"),
        ({"ep": True}, "no expert parallelism"),
        ({"dp": 2}, "no data or decode context parallelism"),
        ({"dcp": 2}, "no data or decode context parallelism"),
        ({"lora": object()}, "no LoRA"),
        ({"kv_transfer": object()}, "no KV transfer connector"),
        ({"moe": "flashinfer_trtllm"}, "--moe-backend deep_gemm or auto"),
        ({"moe": "triton"}, "--moe-backend deep_gemm or auto"),
    ],
)
def test_refuses_unservable_deployments(clean_env, override, need):
    assert not any(need in m for m in switch._unsupported(_config()))
    assert any(need in m for m in switch._unsupported(_config(**override)))


def test_refuses_batch_invariant(clean_env):
    clean_env.setenv("VLLM_BATCH_INVARIANT", "1")
    assert any("batch-invariant" in m for m in switch._unsupported(_config()))


@pytest.mark.parametrize(
    ("kv", "backend", "supported"),
    [
        ("fp8", "FLASHINFER_MLA_SPARSE", True),
        ("fp8_e4m3", "FLASHINFER_MLA_SPARSE", True),
        ("fp8_ds_mla", "FLASHMLA_SPARSE", False),  # the removed 656-byte rows
        ("fp8", "FLASHMLA_SPARSE", False),
    ],
)
def test_only_flashinfer_fp8_rows(clean_env, kv, backend, supported):
    missing = switch._unsupported(_config(kv=kv, backend=backend))
    assert any("--kv-cache-dtype fp8" in need for need in missing) != supported


def test_latency_specialized_turns_on_the_path(clean_env):
    config = _config()
    config.additional_config = {"decode_backend": "latency_specialized"}
    assert switch.from_config(config).enabled
    config.additional_config = {}
    assert switch.from_config(config) == switch.LowLatencyDecode()
    config.additional_config = {"decode_backend": "cute"}
    with pytest.raises(ValueError, match="latency_specialized"):
        switch.from_config(config)


@pytest.mark.parametrize(
    ("moe", "selected"), [("auto", "deep_gemm"), ("deep_gemm", "deep_gemm")]
)
def test_moe_backend_auto_selects_deep_gemm(clean_env, moe, selected):
    config = _config(moe=moe)
    assert not any("--moe-backend" in m for m in switch._unsupported(config))
    switch._select_moe_backend(config)
    assert config.kernel_config.moe_backend == selected

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The locality placement degrades to plain memory where the domains are missing."""

from types import SimpleNamespace

import pytest
import torch

from vllm.models.glm53.nvidia import locality
from vllm.models.glm53.nvidia.ops.cute_dsl.mla_worker import Handoff, MlaWorker

SUCCESS = "success"


def _driver(count: int = 2, **missing: bool) -> SimpleNamespace:
    enums = {
        "CUdevice_attribute": "CU_DEVICE_ATTRIBUTE_LOCALITY_DOMAIN_COUNT",
        "CUmemLocationType": "CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN",
        "CUdevSmResourceGroup_flags": "CU_DEV_SM_RESOURCE_GROUP_LOCALITY_DOMAIN_ID",
        "CUpointer_attribute": "CU_POINTER_ATTRIBUTE_LOCALITY_DOMAIN_ORDINAL",
    }
    fields = {
        enum: SimpleNamespace(**({} if missing.get(enum) else {member: 1}))
        for enum, member in enums.items()
    }
    return SimpleNamespace(
        **fields,
        CUresult=SimpleNamespace(CUDA_SUCCESS=SUCCESS),
        cuDevSmResourceSplit=None,
        cuGreenCtxCreate=None,
        cuDeviceGetAttribute=lambda attribute, device: (SUCCESS, count),
    )


def test_detection_accepts_two_domains() -> None:
    assert locality.unsupported_reason(_driver(), 0) == (None, 2)


@pytest.mark.parametrize(
    "enum",
    [
        "CUdevice_attribute",
        "CUmemLocationType",
        "CUdevSmResourceGroup_flags",
        "CUpointer_attribute",
    ],
)
def test_detection_reports_a_missing_enum(enum: str) -> None:
    reason, _ = locality.unsupported_reason(_driver(**{enum: True}), 0)
    assert reason is not None and enum in reason


def test_detection_reports_old_bindings_without_the_calls() -> None:
    driver = _driver()
    del driver.cuGreenCtxCreate
    reason, _ = locality.unsupported_reason(driver, 0)
    assert reason is not None and "cuGreenCtxCreate" in reason


def test_detection_reports_one_domain_and_a_failing_query() -> None:
    reason, count = locality.unsupported_reason(_driver(count=1), 0)
    assert reason is not None and count == 1
    failing = _driver()
    failing.cuDeviceGetAttribute = lambda attribute, device: ("error", 0)
    assert locality.unsupported_reason(failing, 0)[0] is not None


@pytest.fixture
def fallback(monkeypatch):
    """A :class:`LocalityDomains` on device 0 whose bindings lack the domain enum, and
    the info lines it logged.
    """
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    from cuda.bindings import driver

    lines: list[str] = []
    monkeypatch.setattr(
        locality.logger, "info", lambda message, *args: lines.append(message % args)
    )
    monkeypatch.setattr(driver, "CUdevice_attribute", SimpleNamespace())
    domains = locality.LocalityDomains(torch.device("cuda", 0))
    monkeypatch.setattr(locality, "_BY_DEVICE", {0: domains})
    yield domains, lines
    domains.close()


def test_fallback_is_chosen_and_logged_once(fallback) -> None:
    domains, lines = fallback
    assert not domains.active
    assert set(domains.sm_domain) == {0}
    assert len(domains.sm_domain) == domains.sms
    assert len(lines) == 1 and "placement off" in lines[0]


@pytest.mark.parametrize("by_sm", [False, True])
def test_fallback_keeps_sizes_and_layouts(fallback, by_sm: bool) -> None:
    from vllm.models.glm53.nvidia.arena import Glm53Placement
    from vllm.models.glm53.nvidia.ops.cute_dsl.fused_moe import Flag
    from vllm.models.glm53.nvidia.ops.launch import sizes_of

    placement = Glm53Placement(torch.device("cuda", 0))
    worker = placement.worker_handoff(by_sm)
    assert worker.handoff.numel() == worker.mirror.numel() == Handoff.WORDS
    assert worker.mirror.data_ptr() == worker.handoff.data_ptr() + 4 * Handoff.MIRROR
    table = worker.handoff[Handoff.ROLE_TABLE : Handoff.ROLE_TABLE + MlaWorker.GRID]
    assert table.tolist() == list(range(MlaWorker.GRID))
    flags, mirror = placement.moe_flags()
    words = sizes_of("fused_moe_q4_ckpt_shared")["kFlagWords"]
    assert flags.numel() == mirror.numel() == words
    assert mirror.data_ptr() == flags.data_ptr() + 4 * Flag.MIRROR
    sm_table = flags[Flag.SM_DOMAIN : Flag.SM_DOMAIN + len(placement.domains.sm_domain)]
    assert not bool(sm_table.any())
    worker.mirror.fill_(3)
    assert int(worker.mirror.sum()) == 3 * Handoff.WORDS
    placement.release()

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU tests of the locality-domain pools (``vllm.models.glm53.nvidia.locality``)."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@pytest.fixture(scope="module")
def domains():
    from vllm.models.glm53.nvidia.locality import domains_for

    try:
        yield domains_for(torch.device("cuda", 0))
    except RuntimeError as error:
        pytest.skip(str(error))


@pytest.fixture(scope="module")
def geometry():
    """GLM-5.3's geometry on TP8 (from the package conftest, which a bare run skips)."""
    import importlib.util
    from pathlib import Path
    from types import SimpleNamespace

    from vllm.models.glm53.nvidia.ops.launch import Glm53Geometry

    path = Path(__file__).with_name("conftest.py")
    spec = importlib.util.spec_from_file_location("glm53_test_conftest", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return Glm53Geometry.from_config(SimpleNamespace(**module.GLM53_CONFIG), world=8)


def _ordinal(address: int) -> int:
    from cuda.bindings import driver

    attribute = driver.CUpointer_attribute.CU_POINTER_ATTRIBUTE_LOCALITY_DOMAIN_ORDINAL
    status, value = driver.cuPointerGetAttribute(attribute, address)
    assert status == driver.CUresult.CUDA_SUCCESS
    return int(value)


def test_sm_table_covers_both_domains(domains) -> None:
    table = domains.sm_domain
    assert len(table) == torch.cuda.get_device_properties(0).multi_processor_count
    assert all(table.count(domain) > 0 for domain in range(domains.count))


@pytest.mark.parametrize("domain", [0, 1])
def test_words_are_homed_and_zeroed(domains, domain: int) -> None:
    words = domains.words(domain, 1000)
    assert _ordinal(words.data_ptr()) == domain
    assert not bool(words.any())
    words.fill_(3)
    assert int(words.sum()) == 3000


@pytest.mark.parametrize("first", [0, 1])
def test_mirrored_copies_sit_in_both_domains(domains, first: int) -> None:
    half = 1 << 21
    flags = domains.mirrored_words(first, half, 5000)
    assert _ordinal(flags.data_ptr()) == first
    assert _ordinal(flags.data_ptr() + 4 * half) == 1 - first
    other = domains.mirrored_words(first, half, 100)
    assert other.data_ptr() >= flags.data_ptr() + 4 * 5000
    bits = domains.copy_bits(first)
    assert set(bits) <= {0, 1} and len(bits) == len(domains.sm_domain)


def test_idle_grid_maps_every_cta(domains) -> None:
    grid = len(domains.sm_domain)
    assert sorted(domains.grid_domains(grid)) == sorted(domains.sm_domain)


def _placement():
    """A placement on device 0, or skip when the driver has no locality domains."""
    from vllm.models.glm53.nvidia.arena import Glm53Placement

    try:
        return Glm53Placement(torch.device("cuda", 0))
    except RuntimeError as error:
        pytest.skip(str(error))


@pytest.mark.parametrize("by_sm", [False, True])
def test_worker_handoff_pools_sit_in_their_domains(by_sm: bool) -> None:
    from vllm.models.glm53.nvidia.arena import GROUP_A, GROUP_B
    from vllm.models.glm53.nvidia.ops.cute_dsl.mla_worker import Handoff, MlaWorker

    placement = _placement()
    worker = placement.worker_handoff(by_sm)
    first = worker.domain_a
    assert _ordinal(worker.handoff.data_ptr()) == first
    assert _ordinal(worker.mirror.data_ptr()) == 1 - first
    assert worker.mirror.data_ptr() == worker.handoff.data_ptr() + 4 * Handoff.MIRROR
    assert worker.handoff.numel() == worker.mirror.numel() == Handoff.WORDS
    grid = MlaWorker.GRID
    table = worker.handoff[Handoff.ROLE_TABLE : Handoff.ROLE_TABLE + grid].tolist()
    assert table == list(worker.role_table)
    assert sorted(entry & Handoff.ROLE_MASK for entry in table) == list(range(grid))
    pool_of_role = {
        entry & Handoff.ROLE_MASK: entry >> Handoff.POOL_SHIFT for entry in table
    }
    assert {pool_of_role[role] for role in GROUP_A} == {0}
    assert {pool_of_role[role] for role in GROUP_B} == {1}
    if by_sm:
        sm_domain = placement.domains.sm_domain
        assert [entry >> Handoff.POOL_SHIFT for entry in table] == [
            int(domain != first) for domain in sm_domain
        ]


def test_moe_flags_are_mirrored_with_the_sm_table() -> None:
    from vllm.models.glm53.nvidia.ops.cute_dsl.fused_moe import Flag

    placement = _placement()
    flags, mirror = placement.moe_flags()
    first = placement.preferred_domain
    assert _ordinal(flags.data_ptr()) == first
    assert mirror.data_ptr() == flags.data_ptr() + 4 * Flag.MIRROR
    assert mirror.numel() == flags.numel()
    assert _ordinal(mirror.data_ptr()) == 1 - first
    table = flags[Flag.SM_DOMAIN : Flag.SM_DOMAIN + len(placement.domains.sm_domain)]
    assert table.tolist() == placement.domains.copy_bits(first)


def test_release_reuses_and_rezeroes(domains) -> None:
    half = 1 << 21
    mark = domains.mark()
    first = domains.words(0, 1000)
    first.fill_(7)
    flags = domains.mirrored_words(0, half, 5000)
    flags.fill_(5)
    domains.mirror_of(flags, half).fill_(5)
    domains.release(mark)
    again = domains.words(0, 1000)
    assert again.data_ptr() == first.data_ptr() and not bool(again.any())
    flags_again = domains.mirrored_words(0, half, 5000)
    assert flags_again.data_ptr() == flags.data_ptr()
    assert not bool(flags_again.any())
    assert not bool(domains.mirror_of(flags_again, half).any())
    domains.release(mark)


def test_rebuilding_a_segment_does_not_exhaust_the_pools(geometry) -> None:
    placement = _placement()
    moe_layers = geometry.layers - geometry.dense_layers
    for _ in range(3):
        for _ in range(moe_layers):
            placement.moe_flags()
        for _ in range(geometry.layers):
            placement.worker_handoff(False)
        placement.release()

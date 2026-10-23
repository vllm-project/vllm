# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3 kernel registry and launch program, on the CPU."""

import dataclasses
import struct
from types import SimpleNamespace

import torch

from vllm.models.glm53.nvidia import providers
from vllm.models.glm53.nvidia.ops import launch as glm53_launch
from vllm.models.glm53.nvidia.ops.cute_dsl.mla_selector import IndexScale, StateWords
from vllm.models.glm53.nvidia.ops.cute_dsl.mla_worker import Handoff, MlaWorker


class _AnyTensors(dict):
    """Every input resolves to a small CPU tensor (the target providers); no replicated
    embedding.
    """

    def __missing__(self, name: str) -> torch.Tensor:
        if name == glm53_launch.BLOCK_TABLE:
            tensor = torch.zeros(16384 // 64, dtype=torch.int32)
        elif name == "rope_table":
            tensor = torch.zeros(16384, 64)
        else:
            tensor = torch.zeros(8, dtype=torch.int32)
        self[name] = tensor
        return tensor

    def __contains__(self, name: object) -> bool:
        return name != "embedding"


class _ReplicatedEmbedding(_AnyTensors):
    """A replicated ``embedding`` table instead of vLLM's shards."""

    def __contains__(self, name: object) -> bool:
        return True


class _Placement:
    """CPU stand-in for ``Glm53Placement``: plain handoff and flags."""

    def worker_handoff(self, by_sm: bool) -> SimpleNamespace:
        return SimpleNamespace(
            handoff=torch.zeros(Handoff.WORDS, dtype=torch.int32),
            mirror=torch.zeros(Handoff.WORDS, dtype=torch.int32),
        )

    def moe_flags(self) -> tuple[torch.Tensor, torch.Tensor]:
        words = glm53_launch.sizes_of("fused_moe_q4_ckpt_shared")["kFlagWords"]
        return (
            torch.zeros(words, dtype=torch.int32),
            torch.zeros(words, dtype=torch.int32),
        )


def _bound(program: list[providers.Provider]) -> list[glm53_launch.Launch]:
    """Bind ``program`` to a stand-in arena and return its launches."""
    launches = []
    for provider in program:
        provider.bind(lambda name: [peer << 40 for peer in range(8)])
        assert provider.launch is not None
        launches.append(provider.launch)
    return launches


def test_shipped_kernels_are_the_served_kernels(geometry) -> None:
    from vllm.models.glm53.nvidia.ops.cute_dsl import registry

    launched: list[glm53_launch.Launch] = []
    for packed in (False, True):
        shape = dataclasses.replace(geometry, index_scale_packed=packed)
        for rank in range(shape.world):
            for inputs in (_AnyTensors(), _ReplicatedEmbedding()):
                program = providers.program(shape, rank, inputs, _Placement())
                launched += _bound(program)
    for launch in launched:
        assert len(launch.arguments) == len(registry.arguments(launch.kernel)), (
            launch.site
        )
    kernels = {launch.kernel for launch in launched}
    assert set(registry.ENTRIES) == kernels, sorted(set(registry.ENTRIES) ^ kernels)


def test_layer_layout(geometry) -> None:
    layers = range(geometry.layers)
    shared = [layer for layer in layers if geometry.is_shared(layer)]
    assert shared[:6] == [3, 4, 5, 7, 8, 9] and len(shared) == 78 - 3 - 18
    kinds = [geometry.ffn_kind(layer) for layer in layers]
    assert kinds[:7] == ["dense"] * 3 + ["moe_seam", "moe_seam", "moe", "moe_seam"]
    assert kinds[-1] == "moe" and kinds.count("moe") == 19
    assert kinds.count("moe_seam") == 56


def test_placed_handoff_layout() -> None:
    assert Handoff.ROLE_TABLE == Handoff.VO + 4 * Handoff.VO_ROW
    assert Handoff.ROLE_TABLE + MlaWorker.GRID <= Handoff.SPLIT_DONE
    assert Handoff.SPLIT_DONE + 4 * 32 <= Handoff.WORDS
    assert Handoff.WORDS % 32 == 0 and Handoff.WORDS < Handoff.MIRROR


def test_seed_words(geometry) -> None:
    Selector = providers.Selector
    assert Selector.nominal_seed_words("layers.2", 4) == [0] * 4
    key = providers.ordered_key(100.0) | 1
    assert Selector.nominal_seed_words("layers.1", 4) == [key - (1 << 32)] * 4
    inputs = _AnyTensors()
    x = torch.zeros(geometry.rows, geometry.hidden, dtype=torch.bfloat16)
    selector = Selector(geometry, 1, 0, inputs, x)
    selector.reset_seeds()
    state = selector.select_state
    seeds = state[StateWords.SEEDS : StateWords.SEEDS + 4]
    assert seeds.tolist() == [key - (1 << 32)] * 4
    assert int(state.count_nonzero()) == 4


def test_selector_entry_follows_the_cache_scale_format(
    geometry, hf_config, monkeypatch
) -> None:
    from vllm.models.glm53.nvidia.ops.cute_dsl import registry

    for packed, suffix in ((False, ""), (True, "_mx")):
        monkeypatch.setattr(glm53_launch, "index_scale_packed", lambda p=packed: p)
        shape = glm53_launch.Glm53Geometry.from_config(hf_config, world=8)
        assert shape.selector_ranks == geometry.selector_ranks == 3
        for rank in range(3):
            name = shape.selector_kernel(rank)
            assert name == f"mla_selector_q4_shard3r{rank}_paged_vlin{suffix}"
            knobs = dict(registry.ENTRIES[name].constants)
            assert knobs == {
                "SHARD": rank,
                **({"PACKED_SCALE": True} if packed else {}),
            }


def test_index_scale_words() -> None:
    """Both cache scale formats carry the same power of two, as vLLM writes them: an
    fp32 (exponent byte at bit 23) or four UE8M0 bytes of that exponent.
    """
    from cutlass import Uint32

    for exponent in range(1, 255):
        scale = struct.unpack("<f", struct.pack("<I", exponent << 23))[0]
        assert scale == 2.0 ** (exponent - 127)
        packed = ((exponent << 23) >> 23 & 0xFF) * 0x01010101
        assert int(IndexScale.word(Uint32(exponent), True)) == packed
        assert int(IndexScale.word(Uint32(exponent), False)) == exponent << 23
        decoded = struct.unpack("<f", struct.pack("<I", (packed & 255) << 23))[0]
        assert decoded == scale

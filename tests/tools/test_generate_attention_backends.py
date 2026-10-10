# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
import importlib.util
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GEN_FILES = REPO_ROOT / "docs" / "mkdocs" / "gen_files"
GENERATOR = GEN_FILES / "generate_attention_backends.py"


@pytest.fixture(scope="module")
def generator():
    """Import the generator with its mkdocs dependency stubbed out.

    Importing runs the whole generation pass, so the module also carries the
    blocks it produced in ``captured_blocks``.
    """
    blocks: dict[str, str] = {}
    stub = types.ModuleType("generated_content")
    stub.fill_markers = lambda page, produced: blocks.update(produced)
    sys.modules["generated_content"] = stub
    sys.path.insert(0, str(GENERATOR.parent))
    spec = importlib.util.spec_from_file_location(
        "generate_attention_backends", GENERATOR
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.captured_blocks = blocks
    return module


def _row(table: str, backend: str) -> list[str]:
    for line in table.splitlines():
        if line.startswith(f"| `{backend}`"):
            return [cell.strip() for cell in line.strip("|").split("|")]
    raise AssertionError(f"{backend} is missing from the table")


@pytest.mark.parametrize(
    "backend,head_sizes,block_sizes,compute_cap",
    [
        ("TRITON_FLASHINFER", "256, 512", "64", "10.x"),
        ("TRITON_FLASH_ATTN", "256, 512", "%16", "9.x"),
    ],
)
def test_composite_backends_are_documented(
    generator, backend, head_sizes, block_sizes, compute_cap
):
    """Backends built by the composite factory belong in the table."""
    cells = _row(generator.captured_blocks["table-standard"], backend)
    # Backend, Version, Dtypes, KV Dtypes, Block Sizes, Head Sizes, ...
    assert cells[4] == block_sizes
    assert cells[5] == head_sizes
    assert cells[-1] == compute_cap


def test_composite_kv_dtypes_intersect_both_children(generator):
    """`supports_kv_cache_dtype` requires both children, so the row must too."""
    registry = generator.parse_registry()
    composite = generator.analyze_backend(
        "TRITON_FLASHINFER", registry["TRITON_FLASHINFER"]
    )
    triton = generator.analyze_backend("TRITON_ATTN", registry["TRITON_ATTN"])
    flashinfer = generator.analyze_backend("FLASHINFER", registry["FLASHINFER"])

    def dtypes(info):
        return set(generator._csv_items(info["kv_cache_dtypes"]))

    assert dtypes(composite) == dtypes(triton) & dtypes(flashinfer)


def test_composite_child_inherits_from_its_base(generator):
    """A child that only overrides behaviour still has its base's capabilities."""
    module = "vllm.v1.attention.backends.triton_flash_attn"
    own = generator.analyze_backend("_FABackend", f"{module}._FABackend")
    inherited = generator._analyze_composite_child("_FABackend", f"{module}._FABackend")
    base = generator.analyze_backend(
        "FLASH_ATTN", "vllm.v1.attention.backends.flash_attn.FlashAttentionBackend"
    )
    assert own["kv_cache_dtypes"] == "auto"
    assert inherited["kv_cache_dtypes"] == base["kv_cache_dtypes"]


def test_composite_disables_dcp(generator):
    """`CompositeAttentionBackend.supports_dcp` returns False unconditionally."""
    registry = generator.parse_registry()
    for backend in ("TRITON_FLASHINFER", "TRITON_FLASH_ATTN"):
        info = generator.analyze_backend(backend, registry[backend])
        assert info["supports_dcp"] is False


def test_unanalyzable_backend_fails_the_build(generator, monkeypatch):
    """A registered backend that produces no row must not vanish silently."""
    monkeypatch.setattr(
        generator,
        "parse_registry",
        lambda: {"MADE_UP": "vllm.v1.attention.backends.made_up.MadeUpBackend"},
    )
    with pytest.raises(ValueError, match="MADE_UP"):
        generator.build_blocks()


@pytest.mark.parametrize(
    "spec,size,allowed",
    [
        ("Any", 64, True),
        ("%16", 64, True),
        ("%16", 24, False),
        ("16, 32, 64", 64, True),
        ("16, 32", 64, False),
    ],
)
def test_spec_allows_block_size(generator, spec, size, allowed):
    assert generator._spec_allows_block_size(spec, size) is allowed


@pytest.mark.parametrize(
    "first,second,expected",
    [
        ("Any", "%16", "%16"),
        ("%16", "Any", "%16"),
        ("%16", "%32", "%32"),
        ("16, 32, 64", "%32", "32, 64"),
        ("16", "%32", None),
    ],
)
def test_intersect_block_specs(generator, first, second, expected):
    assert generator._intersect_block_specs(first, second) == expected


def test_find_composite_factory_call(generator):
    tree = ast.parse(
        "Backend = create_composite_attention_backend(A, B, name='x')\n"
        "Other = something_else(A, B)\n"
    )
    assert generator._find_composite_factory_call(tree, "Backend") is not None
    assert generator._find_composite_factory_call(tree, "Other") is None
    assert generator._find_composite_factory_call(tree, "Missing") is None

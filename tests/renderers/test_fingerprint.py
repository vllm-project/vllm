# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`render_fingerprint` layout `rf1`: the bytes must match any other
implementation, so the canonical JSON and the manifest shape are pinned."""

import hashlib
import json
from pathlib import Path

import pytest

from vllm.renderers.fingerprint import (
    CONFIG_FILES,
    PROCESSOR_FILES,
    TOKENIZER_FILES,
    build_manifest,
    canonical_json,
    collect_files,
    render_fingerprint,
    renderer_tag,
)

FIXTURE = Path(__file__).parent / "render_fingerprint_fixture"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1e-7, "1e-7"),
        (1.5e-7, "1.5e-7"),
        (0.000001, "0.000001"),
        (-0.0, "0"),
        (4.5, "4.5"),
        (100.0, "100"),
        (1e21, "1e+21"),
        (1e20, "100000000000000000000"),
        (333333333.33333329, "333333333.3333333"),
        (2**53 - 1, "9007199254740991"),
        (True, "true"),
        (None, "null"),
    ],
)
def test_canonical_json_scalars_follow_rfc8785(value, expected):
    assert canonical_json(value) == expected.encode()


def test_canonical_json_sorts_keys_by_utf16_code_units():
    """The RFC 8785 section 3.2.3 vector; code-point order would differ."""
    keys = ["€", "\r", "דּ", "1", "\U0001f600", "\u0080", "ö"]
    expected = ["\r", "1", "\u0080", "ö", "€", "\U0001f600", "דּ"]

    encoded = canonical_json(dict.fromkeys(keys, 0)).decode()

    positions = [encoded.index(canonical_json(key).decode()) for key in expected]
    assert positions == sorted(positions)


def test_canonical_json_keeps_non_ascii_and_escapes_controls():
    assert canonical_json({"k": "é\n\u0001"}) == '{"k":"é\\n\\u0001"}'.encode()


@pytest.mark.parametrize("value", [2**53, float("nan"), float("inf")])
def test_canonical_json_rejects_numbers_without_a_portable_encoding(value):
    with pytest.raises(ValueError):
        canonical_json({"kwargs": {"x": value}})


def test_collect_files_lists_every_fixed_key_and_tells_absent_from_empty(tmp_path):
    (tmp_path / "tokenizer.json").write_bytes(b"{}")
    (tmp_path / "merges.txt").write_bytes(b"")

    files = collect_files(tmp_path, tmp_path, tmp_path)

    assert set(files) == set(TOKENIZER_FILES + CONFIG_FILES + PROCESSOR_FILES)
    assert files["tokenizer.json"] == hashlib.sha256(b"{}").hexdigest()
    assert files["merges.txt"] == hashlib.sha256(b"").hexdigest()
    assert files["vocab.json"] is None


def test_collect_files_adds_glob_matches_as_extra_keys(tmp_path):
    for name in ("tokenizer.model.v3", "cl100k.tiktoken", "tool_use.jinja"):
        (tmp_path / name).write_bytes(b"x")
    (tmp_path / "additional_chat_templates").mkdir()
    (tmp_path / "additional_chat_templates" / "rag.jinja").write_bytes(b"x")
    (tmp_path / "README.md").write_bytes(b"x")

    extra = set(collect_files(tmp_path, tmp_path, tmp_path)) - set(
        TOKENIZER_FILES + CONFIG_FILES + PROCESSOR_FILES
    )

    assert extra == {
        "tokenizer.model.v3",
        "cl100k.tiktoken",
        "tool_use.jinja",
        "additional_chat_templates/rag.jinja",
    }


def test_each_file_group_resolves_against_its_own_source(tmp_path):
    """`--tokenizer` and `--hf-config-path` can point away from the model; the
    processor configs are always loaded from the model itself."""
    dirs = {name: tmp_path / name for name in ("tokenizer", "config", "model")}
    for name, path in dirs.items():
        path.mkdir()
        for file in ("tokenizer.json", "config.json", "preprocessor_config.json"):
            (path / file).write_text(name)

    files = collect_files(dirs["tokenizer"], dirs["config"], dirs["model"])

    assert files["tokenizer.json"] == hashlib.sha256(b"tokenizer").hexdigest()
    assert files["config.json"] == hashlib.sha256(b"config").hexdigest()
    assert files["preprocessor_config.json"] == hashlib.sha256(b"model").hexdigest()


def test_render_fingerprint_is_pinned_for_a_known_manifest():
    """A second implementation must reproduce this exact string."""
    manifest = build_manifest(
        {"tokenizer.json": "ab", "vocab.json": None},
        {"tokenizer_mode": "auto", "default_chat_template_kwargs": {"t": 0.5}},
    )
    tag = renderer_tag(version="1.2.3", renderer="hf", tokenizer="tokenizers/0.22.0")

    canonical = (
        '{"files":{"tokenizer.json":"ab","vocab.json":null},'
        '"flags":{"default_chat_template_kwargs":{"t":0.5},"tokenizer_mode":"auto"},'
        '"schema":1}'
    )
    assert canonical_json(manifest) == canonical.encode()
    assert render_fingerprint(manifest, tag) == (
        f"rf1:{hashlib.sha256(canonical.encode()).hexdigest()}"
        ":vllm-py/1.2.3;renderer=hf;tokenizer=tokenizers/0.22.0"
    )


def test_renderer_tag_rejects_a_colon_so_the_string_splits_in_three():
    with pytest.raises(ValueError):
        renderer_tag(version="1.0", renderer="hf", tokenizer="tok:1")


def test_shared_fixture_pins_the_manifest_and_digest():
    """The fixture is the cross-frontend contract: another implementation
    must build the same `files` from `model/` and the same digest from the
    manifest. A failure shows the entry that differs."""
    expected = json.loads((FIXTURE / "expected.json").read_text(encoding="utf-8"))
    model = FIXTURE / "model"

    assert collect_files(model, model, model) == expected["manifest"]["files"]
    digest = hashlib.sha256(canonical_json(expected["manifest"])).hexdigest()
    assert digest == expected["digest"]

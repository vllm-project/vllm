# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib.util
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "mkdocs"
    / "gen_files"
    / "generate_examples.py"
)
_spec = importlib.util.spec_from_file_location("generate_examples", SCRIPT)
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)
Example = gen.Example

GH = "https://github.com/vllm-project/vllm"


def _write(path: Path, text: str = "") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A fake repo root so relative links resolve against a known tree."""
    root = tmp_path.resolve()
    monkeypatch.setattr(gen, "ROOT_DIR", root)
    _write(root / "docs" / "features" / "guide.md", "# Guide\n")
    (root / "docs" / "features" / "grafana").mkdir()
    return root


def test_collect_examples_dedups_multi_file_directories(tmp_path):
    category = tmp_path / "pooling"
    for name in ("a.py", "b.py", "c.sh", "README.md"):
        _write(category / "multi" / name)
    _write(category / "single.py")

    examples = gen.collect_examples(tmp_path)

    assert sorted((e.path.name, e.category) for e in examples) == [
        ("multi", "pooling"),
        ("single.py", "pooling"),
    ]


def test_collect_examples_is_deterministic(tmp_path):
    for sub in ("zeta", "alpha", "mid"):
        _write(tmp_path / "cat" / sub / "x.py")

    names = [e.path.name for e in gen.collect_examples(tmp_path)]

    assert names == sorted(names)


def test_main_file_picks_first_markdown_alphabetically(tmp_path):
    for name in ("b.md", "a.md", "c.md"):
        _write(tmp_path / "ex" / name, "# T\n")

    assert Example(tmp_path / "ex", "cat").main_file.name == "a.md"


@pytest.mark.parametrize(
    ("link", "expected"),
    [
        ("../../../docs/features/guide.md", f"{GH}/blob/main/docs/features/guide.md"),
        ("../../../docs/features/grafana", f"{GH}/tree/main/docs/features/grafana"),
        (
            "../../../docs/features/guide.md#usage",
            f"{GH}/blob/main/docs/features/guide.md#usage",
        ),
    ],
)
def test_fix_relative_links_uses_target_kind(repo, link, expected):
    readme = _write(repo / "examples" / "cat" / "demo" / "README.md", "# Demo\n")
    example = Example(readme.parent, "cat")

    assert example.fix_relative_links(f"[x]({link})") == f"[x]({expected})"


def test_fix_relative_links_leaves_outside_repo_untouched(repo):
    readme = _write(repo / "examples" / "cat" / "demo" / "README.md", "# Demo\n")
    example = Example(readme.parent, "cat")
    text = "[x](../../../../../outside.md)"

    assert example.fix_relative_links(text) == text


def test_generate_source_url_is_posix(repo):
    readme = _write(repo / "examples" / "cat" / "demo" / "README.md", "# Demo\n")
    content = Example(readme.parent, "cat").generate()

    assert f"Source <{GH}/tree/main/examples/cat/demo>." in content
    assert "\\" not in content.split("\n")[2]

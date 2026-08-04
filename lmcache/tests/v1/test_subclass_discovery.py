# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``lmcache.v1.utils.subclass_discovery``."""

# Standard
from pathlib import Path
from typing import Iterable, List, Tuple
import importlib
import sys
import textwrap
import uuid

# Third Party
import pytest

# First Party
from lmcache.v1.utils.subclass_discovery import discover_subclasses


def _write_module(pkg_dir: Path, name: str, source: str) -> None:
    (pkg_dir / f"{name}.py").write_text(textwrap.dedent(source))


@pytest.fixture
def temp_pkg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Create a fresh, importable Python package on disk and yield its
    fully-qualified name plus its filesystem path.

    The fixture also takes care of cleaning ``sys.modules`` so each test
    starts from a clean slate even when reusing the same Python
    interpreter.
    """
    pkg_name = f"_subclass_discovery_pkg_{uuid.uuid4().hex}"
    pkg_dir = tmp_path / pkg_name
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("")

    # Make the parent dir importable.
    monkeypatch.syspath_prepend(str(tmp_path))

    yield pkg_name, pkg_dir

    # Drop every module that the test imported under this package so
    # subsequent runs cannot observe stale state.
    for mod_name in [
        m for m in list(sys.modules) if m == pkg_name or m.startswith(pkg_name + ".")
    ]:
        sys.modules.pop(mod_name, None)


def _populate_basic_pkg(pkg_dir: Path) -> None:
    """A minimal layout used by the majority of the tests:
    base.py defines Base + AbstractChild, child_a.py & child_b.py define
    one concrete subclass each.
    """
    _write_module(
        pkg_dir,
        "base",
        """
        from abc import ABC, abstractmethod

        class Base(ABC):
            @abstractmethod
            def name(self) -> str: ...

        class AbstractChild(Base):
            # Still abstract: does not implement name().
            pass
        """,
    )
    _write_module(
        pkg_dir,
        "child_a",
        """
        from .base import Base

        class ChildA(Base):
            def name(self) -> str:
                return "a"
        """,
    )
    _write_module(
        pkg_dir,
        "child_b",
        """
        from .base import Base

        class ChildB(Base):
            def name(self) -> str:
                return "b"
        """,
    )


def _names(classes: Iterable[type]) -> List[str]:
    return sorted(c.__name__ for c in classes)


class TestDiscoverSubclassesBasic:
    def test_finds_concrete_subclasses(self, temp_pkg: Tuple[str, Path]) -> None:
        pkg_name, pkg_dir = temp_pkg
        _populate_basic_pkg(pkg_dir)

        base_mod = importlib.import_module(f"{pkg_name}.base")

        result = list(
            discover_subclasses(
                pkg_name,
                base_mod.Base,
                module_filter=lambda n: n != "base",
            )
        )

        assert _names(result) == ["ChildA", "ChildB"]

    def test_accepts_module_object(self, temp_pkg: Tuple[str, Path]) -> None:
        pkg_name, pkg_dir = temp_pkg
        _populate_basic_pkg(pkg_dir)

        pkg_mod = importlib.import_module(pkg_name)
        base_mod = importlib.import_module(f"{pkg_name}.base")

        result = list(
            discover_subclasses(
                pkg_mod,
                base_mod.Base,
                module_filter=lambda n: n != "base",
            )
        )

        assert _names(result) == ["ChildA", "ChildB"]

    def test_base_class_itself_is_skipped(self, temp_pkg: Tuple[str, Path]) -> None:
        pkg_name, pkg_dir = temp_pkg
        _populate_basic_pkg(pkg_dir)
        base_mod = importlib.import_module(f"{pkg_name}.base")

        # Even when scanning base.py the Base class itself is not yielded.
        result = list(
            discover_subclasses(
                pkg_name,
                base_mod.Base,
                include_abstract=True,
            )
        )

        assert base_mod.Base not in result


class TestAbstractFiltering:
    def test_skips_abstract_by_default(self, temp_pkg: Tuple[str, Path]) -> None:
        pkg_name, pkg_dir = temp_pkg
        _populate_basic_pkg(pkg_dir)
        base_mod = importlib.import_module(f"{pkg_name}.base")

        result = list(discover_subclasses(pkg_name, base_mod.Base))

        # AbstractChild lives in base.py; with default filter
        # (include_abstract=False) it must be excluded.
        assert "AbstractChild" not in _names(result)
        assert _names(result) == ["ChildA", "ChildB"]

    def test_include_abstract_keeps_them(self, temp_pkg: Tuple[str, Path]) -> None:
        pkg_name, pkg_dir = temp_pkg
        _populate_basic_pkg(pkg_dir)
        base_mod = importlib.import_module(f"{pkg_name}.base")

        result = list(
            discover_subclasses(
                pkg_name,
                base_mod.Base,
                include_abstract=True,
            )
        )
        assert "AbstractChild" in _names(result)


class TestModuleFilter:
    def test_module_filter_skips_modules(self, temp_pkg: Tuple[str, Path]) -> None:
        pkg_name, pkg_dir = temp_pkg
        _populate_basic_pkg(pkg_dir)
        base_mod = importlib.import_module(f"{pkg_name}.base")

        result = list(
            discover_subclasses(
                pkg_name,
                base_mod.Base,
                module_filter=lambda n: n == "child_a",
            )
        )
        assert _names(result) == ["ChildA"]


class TestReExportHandling:
    def test_default_excludes_reexports(self, temp_pkg: Tuple[str, Path]) -> None:
        """A class re-exported from another module must not be yielded
        twice when ``require_defined_in_module=True`` (the default)."""
        pkg_name, pkg_dir = temp_pkg
        _populate_basic_pkg(pkg_dir)
        # extra.py only re-imports ChildA; it does not define a new class.
        _write_module(
            pkg_dir,
            "extra",
            """
            from .child_a import ChildA  # re-export
            """,
        )
        base_mod = importlib.import_module(f"{pkg_name}.base")

        result = list(
            discover_subclasses(
                pkg_name,
                base_mod.Base,
                module_filter=lambda n: n != "base",
            )
        )
        # Each class appears at most once - the re-export does not
        # cause duplication.
        assert _names(result) == ["ChildA", "ChildB"]

    def test_disable_require_defined_keeps_reexports_but_dedups(
        self, temp_pkg: Tuple[str, Path]
    ) -> None:
        pkg_name, pkg_dir = temp_pkg
        _populate_basic_pkg(pkg_dir)
        _write_module(
            pkg_dir,
            "extra",
            """
            from .child_a import ChildA  # re-export
            """,
        )
        base_mod = importlib.import_module(f"{pkg_name}.base")

        result = list(
            discover_subclasses(
                pkg_name,
                base_mod.Base,
                module_filter=lambda n: n != "base",
                require_defined_in_module=False,
            )
        )
        # Even though ChildA is visible in two modules, dedup ensures
        # it is yielded a single time.
        assert _names(result) == ["ChildA", "ChildB"]
        # And it is exactly the same class object.
        assert result.count(importlib.import_module(f"{pkg_name}.child_a").ChildA) == 1


class TestImportErrorHandling:
    def test_callback_invoked_on_failure(self, temp_pkg: Tuple[str, Path]) -> None:
        pkg_name, pkg_dir = temp_pkg
        _populate_basic_pkg(pkg_dir)
        _write_module(
            pkg_dir,
            "broken",
            """
            raise RuntimeError("boom")
            """,
        )
        base_mod = importlib.import_module(f"{pkg_name}.base")

        captured: List[Tuple[str, BaseException]] = []

        result = list(
            discover_subclasses(
                pkg_name,
                base_mod.Base,
                module_filter=lambda n: n != "base",
                on_import_error=lambda mod, exc: captured.append((mod, exc)),
            )
        )

        # Discovery still produced the healthy modules.
        assert _names(result) == ["ChildA", "ChildB"]
        assert len(captured) == 1
        failed_mod, failed_exc = captured[0]
        assert failed_mod == f"{pkg_name}.broken"
        assert isinstance(failed_exc, RuntimeError)

    def test_default_handler_logs_and_continues(
        self,
        temp_pkg: Tuple[str, Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        pkg_name, pkg_dir = temp_pkg
        _populate_basic_pkg(pkg_dir)
        _write_module(
            pkg_dir,
            "broken",
            """
            raise RuntimeError("boom")
            """,
        )
        base_mod = importlib.import_module(f"{pkg_name}.base")

        # Spy directly on the module-level logger to avoid coupling the
        # test to pytest's caplog interaction with the project's
        # custom logger setup.
        # First Party
        from lmcache.v1.utils import subclass_discovery as sd

        warnings: List[Tuple[str, Tuple[object, ...]]] = []
        monkeypatch.setattr(
            sd.logger,
            "warning",
            lambda msg, *args, **kwargs: warnings.append((msg, args)),
        )

        result = list(
            discover_subclasses(
                pkg_name,
                base_mod.Base,
                module_filter=lambda n: n != "base",
            )
        )

        assert _names(result) == ["ChildA", "ChildB"]
        # The default handler must surface the failure via logging,
        # without aborting the iteration.
        assert any(f"{pkg_name}.broken" in str(args) for _, args in warnings)


class TestInvalidPackage:
    def test_non_package_raises_type_error(self) -> None:
        # A regular (non-package) module: pytest itself has no __path__.
        # Standard
        import io  # any builtin module without __path__

        with pytest.raises(TypeError):
            list(discover_subclasses(io, object))


def _populate_nested_pkg(pkg_dir: Path) -> None:
    """Layout exercising multi-level discovery:

    * depth 1: ``base.py`` (Base + AbstractChild), ``leaf_top.py`` (TopChild)
    * depth 1 sub-pkg ``mid/`` with its own ``leaf_mid.py`` (MidChild)
    * depth 2 sub-pkg ``mid/deep/`` with ``leaf_deep.py`` (DeepChild)
    """
    _write_module(
        pkg_dir,
        "base",
        """
        from abc import ABC, abstractmethod

        class Base(ABC):
            @abstractmethod
            def name(self) -> str: ...

        class AbstractChild(Base):
            pass
        """,
    )
    _write_module(
        pkg_dir,
        "leaf_top",
        """
        from .base import Base

        class TopChild(Base):
            def name(self) -> str:
                return "top"
        """,
    )
    mid = pkg_dir / "mid"
    mid.mkdir()
    (mid / "__init__.py").write_text("")
    _write_module(
        mid,
        "leaf_mid",
        """
        from ..base import Base

        class MidChild(Base):
            def name(self) -> str:
                return "mid"
        """,
    )
    deep = mid / "deep"
    deep.mkdir()
    (deep / "__init__.py").write_text("")
    _write_module(
        deep,
        "leaf_deep",
        """
        from ...base import Base

        class DeepChild(Base):
            def name(self) -> str:
                return "deep"
        """,
    )


class TestLevels:
    def test_default_keeps_legacy_top_level_only(
        self, temp_pkg: Tuple[str, Path]
    ) -> None:
        pkg_name, pkg_dir = temp_pkg
        _populate_nested_pkg(pkg_dir)
        base_mod = importlib.import_module(f"{pkg_name}.base")

        result = list(
            discover_subclasses(
                pkg_name,
                base_mod.Base,
                module_filter=lambda n: n != "base",
            )
        )
        # No ``levels`` argument: behaves exactly like the historic
        # direct-children-only scan.
        assert _names(result) == ["TopChild"]

    def test_levels_1_1_matches_default(self, temp_pkg: Tuple[str, Path]) -> None:
        pkg_name, pkg_dir = temp_pkg
        _populate_nested_pkg(pkg_dir)
        base_mod = importlib.import_module(f"{pkg_name}.base")

        result = list(
            discover_subclasses(
                pkg_name,
                base_mod.Base,
                module_filter=lambda n: n != "base",
                levels=[1, 1],
            )
        )
        assert _names(result) == ["TopChild"]

    def test_levels_2_2_only_grandchildren(self, temp_pkg: Tuple[str, Path]) -> None:
        pkg_name, pkg_dir = temp_pkg
        _populate_nested_pkg(pkg_dir)
        base_mod = importlib.import_module(f"{pkg_name}.base")

        result = list(
            discover_subclasses(
                pkg_name,
                base_mod.Base,
                levels=[2, 2],
            )
        )
        # Only ``mid/leaf_mid.py`` lives at depth 2.
        assert _names(result) == ["MidChild"]

    def test_levels_1_2_combines_both_layers(self, temp_pkg: Tuple[str, Path]) -> None:
        pkg_name, pkg_dir = temp_pkg
        _populate_nested_pkg(pkg_dir)
        base_mod = importlib.import_module(f"{pkg_name}.base")

        result = list(
            discover_subclasses(
                pkg_name,
                base_mod.Base,
                module_filter=lambda n: n != "base",
                levels=[1, 2],
            )
        )
        assert _names(result) == ["MidChild", "TopChild"]

    def test_unlimited_via_empty_list(self, temp_pkg: Tuple[str, Path]) -> None:
        pkg_name, pkg_dir = temp_pkg
        _populate_nested_pkg(pkg_dir)
        base_mod = importlib.import_module(f"{pkg_name}.base")

        result = list(
            discover_subclasses(
                pkg_name,
                base_mod.Base,
                module_filter=lambda n: n != "base",
                levels=[],
            )
        )
        assert _names(result) == ["DeepChild", "MidChild", "TopChild"]

    def test_unlimited_via_zero_zero(self, temp_pkg: Tuple[str, Path]) -> None:
        pkg_name, pkg_dir = temp_pkg
        _populate_nested_pkg(pkg_dir)
        base_mod = importlib.import_module(f"{pkg_name}.base")

        result = list(
            discover_subclasses(
                pkg_name,
                base_mod.Base,
                module_filter=lambda n: n != "base",
                levels=[0, 0],
            )
        )
        assert _names(result) == ["DeepChild", "MidChild", "TopChild"]

    def test_module_filter_applies_per_level(self, temp_pkg: Tuple[str, Path]) -> None:
        """The short-name filter is evaluated for every leaf module
        regardless of its depth, matching the documented contract."""
        pkg_name, pkg_dir = temp_pkg
        _populate_nested_pkg(pkg_dir)
        # Add another depth-2 module that also happens to have the
        # name ``leaf_mid`` shadowed under ``deep/`` -- proves the
        # filter looks at the leaf short name, not the full path.
        deep_dir = pkg_dir / "mid" / "deep"
        _write_module(
            deep_dir,
            "leaf_mid",
            """
            from ...base import Base

            class DeepShadow(Base):
                def name(self) -> str:
                    return "deep_shadow"
            """,
        )
        base_mod = importlib.import_module(f"{pkg_name}.base")

        result = list(
            discover_subclasses(
                pkg_name,
                base_mod.Base,
                module_filter=lambda n: n == "leaf_mid",
                levels=[],
            )
        )
        assert _names(result) == ["DeepShadow", "MidChild"]

    def test_discovers_class_in_subpackage_init(
        self, temp_pkg: Tuple[str, Path]
    ) -> None:
        """A class defined directly in a sub-package ``__init__.py``
        must be discovered at the sub-package's depth (depth 1).  This
        matches the pre-levels legacy behaviour where sub-packages were
        treated identically to leaf modules."""
        pkg_name, pkg_dir = temp_pkg
        _populate_nested_pkg(pkg_dir)
        # Place a class inside mid/__init__.py (depth 1 sub-package).
        (pkg_dir / "mid" / "__init__.py").write_text(
            textwrap.dedent(
                """
                from ..base import Base

                class InitChild(Base):
                    def name(self) -> str:
                        return "init"
                """
            )
        )
        base_mod = importlib.import_module(f"{pkg_name}.base")

        # Default levels [1, 1] -- depth 1, should see InitChild in
        # mid/__init__.py at depth 1 + TopChild in leaf_top.py.
        result = list(
            discover_subclasses(
                pkg_name,
                base_mod.Base,
                module_filter=lambda n: n != "base",
            )
        )
        assert _names(result) == ["InitChild", "TopChild"]

    def test_subpackage_init_not_found_at_deeper_level(
        self, temp_pkg: Tuple[str, Path]
    ) -> None:
        """A class in a sub-package ``__init__.py`` at depth 1 must not
        appear when the scan window excludes depth 1."""
        pkg_name, pkg_dir = temp_pkg
        _populate_nested_pkg(pkg_dir)
        (pkg_dir / "mid" / "__init__.py").write_text(
            textwrap.dedent(
                """
                from ..base import Base

                class InitChild(Base):
                    def name(self) -> str:
                        return "init"
                """
            )
        )
        base_mod = importlib.import_module(f"{pkg_name}.base")

        # levels=[2, 2] only inspects depth-2 leaf modules.
        result = list(
            discover_subclasses(
                pkg_name,
                base_mod.Base,
                levels=[2, 2],
            )
        )
        # InitChild lives at depth 1, MidChild (mid/leaf_mid.py) at
        # depth 2.
        assert _names(result) == ["MidChild"]

    @pytest.mark.parametrize(
        "bad_levels",
        [
            [1],
            [1, 2, 3],
            [-1, 2],
            [2, 1],
            [0, 3],
            [3, 0],
        ],
    )
    def test_invalid_levels_raise(
        self, temp_pkg: Tuple[str, Path], bad_levels: List[int]
    ) -> None:
        pkg_name, pkg_dir = temp_pkg
        _populate_nested_pkg(pkg_dir)
        base_mod = importlib.import_module(f"{pkg_name}.base")

        with pytest.raises(ValueError):
            list(
                discover_subclasses(
                    pkg_name,
                    base_mod.Base,
                    levels=bad_levels,
                )
            )

# SPDX-License-Identifier: Apache-2.0
# Standard
from types import ModuleType
from unittest.mock import AsyncMock, patch

# Third Party
import pytest

# First Party
from lmcache.v1.basic_check import main, parse_args
from lmcache.v1.check import CheckModeRegistry, check_mode

# --------------- CheckModeRegistry tests ---------------


class TestCheckModeRegistry:
    """Tests for CheckModeRegistry."""

    def test_register_and_get(self) -> None:
        reg = CheckModeRegistry()
        reg.loaded = True  # skip auto-loading

        async def dummy(**kw):
            pass

        reg.register("foo", dummy)
        assert reg.get_mode("foo") is dummy

    def test_register_duplicate_raises(self) -> None:
        reg = CheckModeRegistry()
        reg.loaded = True

        reg.register("dup", lambda: None)
        with pytest.raises(ValueError, match="already registered"):
            reg.register("dup", lambda: None)

    def test_get_mode_returns_none_for_unknown(self) -> None:
        reg = CheckModeRegistry()
        reg.loaded = True
        assert reg.get_mode("nonexistent") is None

    def test_load_modes_discovers_decorated_functions(
        self, tmp_path, monkeypatch
    ) -> None:
        """Verify that load_modes discovers check_mode_ modules."""
        reg = CheckModeRegistry()

        # Create a fake check_mode_ module
        mod = ModuleType("check_mode_fake")

        async def fake_func(**kw):
            pass

        fake_func.is_check_mode = True  # type: ignore[attr-defined]
        fake_func.mode_name = "fake"  # type: ignore[attr-defined]
        mod.fake_func = fake_func  # type: ignore[attr-defined]

        # Patch os.listdir to return our fake module
        monkeypatch.setattr(
            "os.listdir",
            lambda _: ["check_mode_fake.py", "utils.py"],
        )
        monkeypatch.setattr(
            "importlib.import_module",
            lambda name, package=None: mod,
        )

        reg.load_modes()
        assert reg.loaded is True
        assert "fake" in reg.modes
        assert reg.modes["fake"] is fake_func

    def test_load_modes_skips_import_errors(self, monkeypatch) -> None:
        """load_modes should not crash on ImportError."""
        reg = CheckModeRegistry()

        def raiser(name, package=None):
            raise ImportError("boom")

        monkeypatch.setattr(
            "os.listdir",
            lambda _: ["check_mode_bad.py"],
        )
        monkeypatch.setattr("importlib.import_module", raiser)

        reg.load_modes()
        assert reg.loaded is True
        assert len(reg.modes) == 0

    def test_load_modes_called_once(self, monkeypatch) -> None:
        """Second call to load_modes should be a no-op."""
        reg = CheckModeRegistry()

        call_count = 0

        def counting_listdir(path):
            nonlocal call_count
            call_count += 1
            return []

        monkeypatch.setattr("os.listdir", counting_listdir)

        reg.load_modes()
        reg.load_modes()
        assert call_count == 1

    def test_get_mode_triggers_load(self, monkeypatch) -> None:
        """get_mode should call load_modes if not yet loaded."""
        reg = CheckModeRegistry()
        assert reg.loaded is False

        monkeypatch.setattr("os.listdir", lambda _: [])

        reg.get_mode("any")
        assert reg.loaded is True


# --------------- check_mode decorator tests ---------------


class TestCheckModeDecorator:
    """Tests for the @check_mode decorator."""

    def test_decorator_sets_attributes(self) -> None:
        @check_mode("my_mode")
        async def handler(**kw):
            pass

        assert handler.is_check_mode is True  # type: ignore[attr-defined]
        assert handler.mode_name == "my_mode"  # type: ignore[attr-defined]

    def test_decorated_function_still_callable(self) -> None:
        @check_mode("callable_check")
        def sync_fn():
            return 42

        assert sync_fn() == 42


# --------------- parse_args tests ---------------


class TestParseArgs:
    """Tests for basic_check.parse_args."""

    def test_required_mode(self) -> None:
        with patch(
            "sys.argv",
            ["basic_check", "--mode", "test_remote"],
        ):
            args = parse_args()
        assert args.mode == "test_remote"

    @pytest.mark.parametrize(
        "flag,value,attr,expected",
        [
            ("--model", "/my/model", "model", "/my/model"),
            ("--num-keys", "50", "num_keys", 50),
            ("--concurrency", "8", "concurrency", 8),
            ("--offset", "10", "offset", 10),
        ],
    )
    def test_optional_flags(
        self,
        flag: str,
        value: str,
        attr: str,
        expected: object,
    ) -> None:
        with patch(
            "sys.argv",
            ["basic_check", "--mode", "x", flag, value],
        ):
            args = parse_args()
        assert getattr(args, attr) == expected

    def test_defaults(self) -> None:
        with patch("sys.argv", ["basic_check", "--mode", "x"]):
            args = parse_args()
        assert args.num_keys == 5
        assert args.concurrency == 16
        assert args.offset == 0
        assert args.model == "/lmcache_test_model/"
        assert args.l2_adapter == []
        assert args.obj_size is None
        assert args.kv_dtype is None
        assert args.settle_time == 0.0


# --------------- main() tests ---------------


class TestMain:
    """Tests for basic_check.main (async)."""

    @pytest.mark.asyncio
    async def test_list_mode(self, monkeypatch, capsys) -> None:
        """--mode list should print available modes."""
        monkeypatch.setattr("sys.argv", ["basic_check", "--mode", "list"])

        fake_reg = CheckModeRegistry()
        fake_reg.loaded = True
        fake_reg.modes["alpha"] = lambda: None
        fake_reg.modes["beta"] = lambda: None

        monkeypatch.setattr("lmcache.v1.basic_check.registry", fake_reg)

        await main()
        out = capsys.readouterr().out
        assert "alpha" in out
        assert "beta" in out

    @pytest.mark.asyncio
    async def test_unknown_mode(self, monkeypatch, capsys) -> None:
        """Unknown mode should print an error."""
        monkeypatch.setattr(
            "sys.argv",
            ["basic_check", "--mode", "no_such_mode"],
        )

        fake_reg = CheckModeRegistry()
        fake_reg.loaded = True  # empty registry

        monkeypatch.setattr("lmcache.v1.basic_check.registry", fake_reg)

        await main()
        out = capsys.readouterr().out
        assert "Unknown mode" in out

    @pytest.mark.asyncio
    async def test_valid_mode_invokes_function(self, monkeypatch) -> None:
        """A valid mode should be called with parsed args."""
        handler = AsyncMock()

        fake_reg = CheckModeRegistry()
        fake_reg.loaded = True
        fake_reg.register("run_it", handler)

        monkeypatch.setattr("lmcache.v1.basic_check.registry", fake_reg)
        monkeypatch.setattr(
            "sys.argv",
            [
                "basic_check",
                "--mode",
                "run_it",
                "--num-keys",
                "5",
                "--concurrency",
                "2",
                "--offset",
                "3",
            ],
        )

        await main()

        handler.assert_awaited_once_with(
            model="/lmcache_test_model/",
            num_keys=5,
            concurrency=2,
            offset=3,
            l2_adapter=[],
            obj_size=None,
            kv_dtype=None,
            settle_time=0.0,
        )

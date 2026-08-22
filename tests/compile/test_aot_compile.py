# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools
import hashlib
import json
import os
import pickle
import subprocess
import sys
import tempfile
from contextlib import contextmanager, nullcontext
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch

import vllm.compilation.decorators as compilation_decorators
import vllm.envs as envs
from vllm.compilation.backends import VllmBackend
from vllm.compilation.caching import (
    StandaloneCompiledArtifacts,
    VllmSerializableFunction,
)
from vllm.compilation.counter import compilation_counter
from vllm.compilation.decorators import support_torch_compile
from vllm.config import (
    CompilationConfig,
    CompilationMode,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.envs import disable_envs_cache
from vllm.forward_context import set_forward_context
from vllm.utils.torch_utils import is_torch_equal_or_newer

from ..utils import create_new_process_for_each_test


@pytest.fixture
def vllm_tmp_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Fixture that sets VLLM_CACHE_ROOT to a temporary directory."""
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path / "vllm_cache"))
    return tmp_path


def reference_fn(x: torch.Tensor):
    assert x.shape[0] <= 42
    assert x.shape[0] % 2 == 0
    for _ in range(30):
        x = x + x.shape[0]
    return x


def reference_fn_tuple(x: torch.Tensor):
    """Reference function that returns a tuple of tensors."""
    assert x.shape[0] <= 42
    assert x.shape[0] % 2 == 0
    for _ in range(30):
        x = x + x.shape[0]
    return x, x * 2


@support_torch_compile
class CompiledMod(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return reference_fn(x)


@support_torch_compile
class CompiledModTuple(torch.nn.Module):
    """A compiled module that returns a tuple of tensors."""

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return reference_fn_tuple(x)


def _capture_forward_state(enabled: bool):
    def decorate(forward):
        @functools.wraps(forward)
        def wrapped(*args, **kwargs):
            # Make the captured flag affect both the fingerprint and behavior.
            output = forward(*args, **kwargs)
            return output + 1 if enabled else output

        return wrapped

    return decorate


def _capture_unsupported_forward_state():
    # Capture a mutable value that the fingerprint protocol deliberately rejects.
    mutable_state: list[object] = []

    def decorate(forward):
        @functools.wraps(forward)
        def wrapped(*args, **kwargs):
            if mutable_state:
                return forward(*args, **kwargs)
            return forward(*args, **kwargs)

        return wrapped

    return decorate


class ExternalDataDecoratedMod(torch.nn.Module):
    @_capture_forward_state(True)
    @_capture_forward_state(False)
    def forward(self, x: torch.Tensor):
        return x + 1


_EXTERNAL_DATA_AOT_FORWARD_FLAG = (
    os.environ.get("VLLM_TEST_AOT_FORWARD_FLAG", "0") == "1"
)


@support_torch_compile
class ExternalDataAOTMod(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    @_capture_forward_state(True)
    @_capture_forward_state(_EXTERNAL_DATA_AOT_FORWARD_FLAG)
    def forward(self, x: torch.Tensor):
        return x + 1


@support_torch_compile
class UnsupportedExternalDataAOTMod(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    @_capture_unsupported_forward_state()
    def forward(self, x: torch.Tensor):
        return x + 1


def _install_canonical_forward_model(
    monkeypatch: pytest.MonkeyPatch, model_name: str, forward
) -> torch.nn.Module:
    """Install a dynamic model so module/qualname lookup resolves its forward."""
    forward.__module__ = __name__
    forward.__qualname__ = f"{model_name}.forward"
    model_type = type(
        model_name,
        (torch.nn.Module,),
        {"__module__": __name__, "forward": forward},
    )
    monkeypatch.setattr(sys.modules[__name__], model_name, model_type, raising=False)
    return model_type()


@pytest.mark.parametrize("inherited", [False, True])
def test_forward_external_data_describes_direct_and_inherited_forward(inherited: bool):
    if inherited:

        class Model(ExternalDataDecoratedMod):
            pass

        model = Model()
    else:
        model = ExternalDataDecoratedMod()

    outer = model.forward.__func__
    inner = outer.__wrapped__
    original = inner.__wrapped__

    external_data = compilation_decorators._get_forward_external_data(model)

    assert external_data is not None
    assert set(external_data.values()) == {inner, original}
    expected_prefix = (
        f"aot:model-forward:{outer.__module__}:{outer.__qualname__}:wrapped:"
    )
    assert all(key.startswith(expected_prefix) for key in external_data)


@pytest.mark.parametrize("case", ["instance_override", "noncanonical_outer"])
def test_forward_external_data_rejects_unstable_outer_function(case: str):
    if case == "instance_override":
        model = ExternalDataDecoratedMod()
        object.__setattr__(model, "forward", lambda x: x)
    else:

        class NonCanonicalOuterMod(torch.nn.Module):
            @_capture_forward_state(False)
            def forward(self, x):
                return x

        model = NonCanonicalOuterMod()

    assert compilation_decorators._get_forward_external_data(model) is None


@pytest.mark.parametrize("case", ["non_function", "cycle", "excessive_chain"])
def test_forward_external_data_rejects_malformed_wrapped_chain(
    case: str, monkeypatch: pytest.MonkeyPatch
):
    def original(self, x):
        return x

    if case == "excessive_chain":
        outer = original
        for _ in range(33):
            outer = _capture_forward_state(False)(outer)
    else:
        outer = _capture_forward_state(False)(original)
        monkeypatch.setattr(
            outer,
            "__wrapped__",
            object() if case == "non_function" else outer,
        )

    model = _install_canonical_forward_model(
        monkeypatch, "_MalformedWrappedChainMod", outer
    )

    assert compilation_decorators._get_forward_external_data(model) is None


@pytest.mark.parametrize("case", ["empty_closure_cell", "missing_inner_metadata"])
def test_forward_external_data_rejects_invalid_inner_state(
    case: str, monkeypatch: pytest.MonkeyPatch
):
    def original(self, x):
        return x

    if case == "empty_closure_cell":
        flag = True

        @functools.wraps(original)
        def outer(*args, **kwargs):
            if flag:
                return original(*args, **kwargs)
            return original(*args, **kwargs)

        del flag
    else:
        outer = _capture_forward_state(False)(original)
        outer.__wrapped__.__module__ = None

    model = _install_canonical_forward_model(
        monkeypatch, "_InvalidInnerStateMod", outer
    )

    assert compilation_decorators._get_forward_external_data(model) is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, ("none",)),
        (False, ("bool", "0")),
        (True, ("bool", "1")),
        (10**100, ("int", str(10**100))),
        (-12345678901234567890, ("int", "-12345678901234567890")),
        (0.0, ("float", "0x0.0p+0")),
        (-0.0, ("float", "-0x0.0p+0")),
        ("forward-state", ("str", "forward-state")),
        (b"\x00\xff", ("bytes", "00ff")),
        (
            (None, True, 7, "nested"),
            (
                "tuple",
                (
                    ("none",),
                    ("bool", "1"),
                    ("int", "7"),
                    ("str", "nested"),
                ),
            ),
        ),
    ],
)
def test_normalize_aot_external_state_uses_tagged_values(value, expected):
    assert compilation_decorators._normalize_aot_external_state(value) == expected


def test_normalize_aot_external_state_only_accepts_next_wrapped_function():
    def next_wrapped():
        pass

    assert compilation_decorators._normalize_aot_external_state(
        next_wrapped,
        next_wrapped=next_wrapped,
        next_depth=2,
    ) == ("wrapped", "2")

    with pytest.raises(compilation_decorators._UnsupportedAOTExternalState):
        compilation_decorators._normalize_aot_external_state(lambda: None)


@pytest.mark.parametrize(
    "value",
    [
        [],
        {},
        set(),
        frozenset(),
        object(),
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_normalize_aot_external_state_rejects_unsupported_values(value):
    with pytest.raises(compilation_decorators._UnsupportedAOTExternalState):
        compilation_decorators._normalize_aot_external_state(value)


def test_normalize_aot_external_state_rejects_excessive_depth():
    value: object = None
    for _ in range(33):
        value = (value,)

    with pytest.raises(compilation_decorators._UnsupportedAOTExternalState):
        compilation_decorators._normalize_aot_external_state(value)


@pytest.mark.parametrize("flag_depth", [0, 1])
def test_forward_external_data_key_covers_complete_decorator_chain(
    monkeypatch: pytest.MonkeyPatch,
    flag_depth: int,
):
    def build_model(flag: bool):
        flags = [False, False]
        flags[flag_depth] = flag

        @_capture_forward_state(flags[0])
        @_capture_forward_state(flags[1])
        @_capture_forward_state(False)
        def forward(self, x):
            return x

        return _install_canonical_forward_model(
            monkeypatch, "_CanonicalChainHolder", forward
        )

    false_model = build_model(False)
    false_keys = compilation_decorators._get_forward_external_data(false_model)

    true_model = build_model(True)
    true_keys = compilation_decorators._get_forward_external_data(true_model)

    assert false_keys is not None
    assert true_keys is not None
    assert false_keys.keys().isdisjoint(true_keys.keys())


@_capture_forward_state(False)
def _canonical_middle_forward(self, x):
    return x


def test_forward_external_data_skips_canonical_inner_function(
    monkeypatch: pytest.MonkeyPatch,
):
    next_forward = _canonical_middle_forward

    def outer(self, x):
        return next_forward(self, x)

    outer.__wrapped__ = next_forward
    model = _install_canonical_forward_model(
        monkeypatch, "_CanonicalInnerHolder", outer
    )

    external_data = compilation_decorators._get_forward_external_data(model)

    assert external_data is not None
    assert next_forward not in external_data.values()
    assert next_forward.__wrapped__ in external_data.values()


def test_forward_external_data_does_not_hide_internal_errors(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        compilation_decorators,
        "_normalize_aot_external_state",
        Mock(side_effect=RuntimeError("fingerprint bug")),
    )

    with pytest.raises(RuntimeError, match="fingerprint bug"):
        compilation_decorators._get_forward_external_data(ExternalDataDecoratedMod())


def test_forward_external_data_rejects_unsupported_closure_state():
    # This helper inspects class metadata only; avoid the compile-aware __init__.
    model = UnsupportedExternalDataAOTMod.__new__(UnsupportedExternalDataAOTMod)

    assert compilation_decorators._get_forward_external_data(model) is None


@pytest.mark.parametrize(
    ("supports_external_data", "external_data"),
    [(False, None), (True, None), (True, {})],
)
def test_save_aot_compiled_function_uses_compatible_signature(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    supports_external_data: bool,
    external_data: dict | None,
):
    monkeypatch.setenv("VLLM_DISABLE_COMPILE_CACHE", "0")
    monkeypatch.setattr(
        compilation_decorators,
        "_SUPPORTS_AOT_EXTERNAL_DATA",
        supports_external_data,
    )
    disable_envs_cache()
    model = ExternalDataDecoratedMod()
    model.was_aot_compile_fn_loaded_from_disk = False
    model._aot_cache_dir = str(tmp_path)
    model._aot_compilation_path = str(tmp_path / "model")
    saved_paths = []

    class StandardCompiledFunction:
        def save_compiled_function(self, path):
            saved_paths.append(path)
            Path(path).touch()

    model.aot_compiled_fn = StandardCompiledFunction()
    helper = Mock(return_value=external_data)
    if not supports_external_data:
        helper.side_effect = AssertionError("helper requires torch 2.11+")

    with patch.object(
        compilation_decorators,
        "_get_forward_external_data",
        helper,
    ):
        CompiledMod.save_aot_compiled_function(model)

    assert helper.call_count == int(supports_external_data)
    assert saved_paths == [f"{model._aot_compilation_path}.{os.getpid()}.tmp"]
    assert Path(model._aot_compilation_path).exists()


@pytest.mark.parametrize(
    ("supports_external_data", "external_data"),
    [(False, None), (True, None), (True, {})],
)
def test_load_aot_compiled_function_uses_compatible_signature(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    supports_external_data: bool,
    external_data: dict | None,
):
    monkeypatch.setenv("VLLM_FORCE_AOT_LOAD", "0")
    monkeypatch.setattr(
        compilation_decorators,
        "_SUPPORTS_AOT_EXTERNAL_DATA",
        supports_external_data,
    )
    disable_envs_cache()
    artifact = tmp_path / "model"
    artifact.write_bytes(b"invalid")
    model = ExternalDataDecoratedMod()
    model.vllm_config = Mock()
    model._is_encoder = False
    calls = []

    def standard_load(file, *, f_globals=None):
        calls.append((file, f_globals))
        raise RuntimeError("stop after argument capture")

    helper = Mock(return_value=external_data)
    if not supports_external_data:
        helper.side_effect = AssertionError("helper requires torch 2.11+")

    with (
        patch.object(
            compilation_decorators,
            "monitor_torch_compile",
            return_value=nullcontext(),
        ),
        patch.object(
            compilation_decorators,
            "_get_forward_external_data",
            helper,
        ),
        patch.object(torch.compiler, "load_compiled_function", standard_load),
    ):
        loaded = compilation_decorators._try_load_aot_compiled_fn(model, str(artifact))

    assert helper.call_count == int(supports_external_data)
    assert loaded is None
    assert len(calls) == 1
    assert calls[0][1] is model.forward.__globals__


@pytest.mark.parametrize("force_load", [False, True])
def test_forward_external_data_failure_obeys_load_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, force_load: bool
):
    monkeypatch.setenv("VLLM_FORCE_AOT_LOAD", "1" if force_load else "0")
    disable_envs_cache()
    artifact = tmp_path / "model"
    artifact.write_bytes(b"unused")
    model = ExternalDataDecoratedMod()
    model.vllm_config = Mock()
    model._is_encoder = False

    with (
        patch.object(
            compilation_decorators,
            "monitor_torch_compile",
            return_value=nullcontext(),
        ),
        patch.object(
            compilation_decorators,
            "_get_forward_external_data",
            side_effect=RuntimeError("unsafe decorator state"),
        ),
        patch.object(torch.compiler, "load_compiled_function") as load_mock,
    ):
        if force_load:
            with pytest.raises(RuntimeError, match="unsafe decorator state"):
                compilation_decorators._try_load_aot_compiled_fn(model, str(artifact))
        else:
            assert (
                compilation_decorators._try_load_aot_compiled_fn(model, str(artifact))
                is None
            )

    load_mock.assert_not_called()


def _run_external_data_aot_process(result_path: str) -> None:
    """Run one AOT phase in a fresh interpreter and persist its outcome."""
    disable_envs_cache()
    torch.manual_seed(0)
    args = (torch.arange(100, device="cuda", dtype=torch.float32).reshape(10, 10),)
    vllm_config = make_vllm_config()
    model = None
    try:
        with use_vllm_config(vllm_config):
            model_type = (
                UnsupportedExternalDataAOTMod
                if os.environ.get("VLLM_TEST_AOT_UNSUPPORTED_STATE") == "1"
                else ExternalDataAOTMod
            )
            model = model_type(vllm_config=vllm_config)
            output = model(*args)
        result = {
            "output": output.cpu().tolist(),
            "loaded_from_disk": model.was_aot_compile_fn_loaded_from_disk,
        }
    except Exception as exc:
        result = {
            "error": f"{type(exc).__name__}: {exc}",
            "loaded_from_disk": False,
        }
        raise
    finally:
        result.update(
            {
                "num_aot_compiles": compilation_counter.num_aot_compiles,
                "num_aot_artifacts_saved": (
                    compilation_counter.num_aot_artifacts_saved
                ),
                "num_aot_artifacts_loaded": (
                    compilation_counter.num_aot_artifacts_loaded
                ),
                "artifact_exists": bool(
                    model is not None
                    and (artifact_path := getattr(model, "_aot_compilation_path", None))
                    and Path(artifact_path).exists()
                ),
            }
        )
        Path(result_path).write_text(json.dumps(result))


def _launch_external_data_aot_process(
    cache_root: Path,
    result_path: Path,
    *,
    force_load: bool,
    forward_flag: bool = False,
    unsupported_state: bool = False,
) -> tuple[subprocess.CompletedProcess[str], dict]:
    """Launch an isolated process so no in-memory compile state is reused."""
    env = os.environ.copy()
    env.update(
        {
            "VLLM_CACHE_ROOT": str(cache_root),
            "VLLM_USE_AOT_COMPILE": "1",
            "VLLM_USE_MEGA_AOT_ARTIFACT": "1",
            "VLLM_USE_STANDALONE_COMPILE": "1",
            "VLLM_TEST_AOT_FORWARD_FLAG": "1" if forward_flag else "0",
            "VLLM_TEST_AOT_UNSUPPORTED_STATE": ("1" if unsupported_state else "0"),
        }
    )
    if force_load:
        env["VLLM_FORCE_AOT_LOAD"] = "1"
    else:
        env.pop("VLLM_FORCE_AOT_LOAD", None)
    code = (
        "import sys; "
        "from tests.compile.test_aot_compile import "
        "_run_external_data_aot_process; "
        "_run_external_data_aot_process(sys.argv[1])"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code, str(result_path)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
    )
    result = json.loads(result_path.read_text()) if result_path.exists() else {}
    return completed, result


@pytest.mark.skipif(not is_torch_equal_or_newer("2.10.0"), reason="requires torch 2.10")
def test_unsupported_closure_state_is_nonfatal_during_real_save(tmp_path: Path):
    process, result = _launch_external_data_aot_process(
        tmp_path / "cache",
        tmp_path / "unsupported.json",
        force_load=False,
        unsupported_state=True,
    )

    assert process.returncode == 0, process.stderr
    assert result["output"] == torch.arange(100).reshape(10, 10).add(1).tolist()
    assert result["num_aot_artifacts_saved"] == 0
    assert result["artifact_exists"] is False
    logs = f"{process.stdout}\n{process.stderr}".lower()
    assert "unable to save aot compiled function" in logs


@pytest.mark.skipif(
    not is_torch_equal_or_newer("2.11.0"),
    reason="requires AOT external_data support in torch 2.11+",
)
def test_external_data_aot_cache_loads_across_fresh_processes(tmp_path: Path):
    cache_root = tmp_path / "cache"
    cold_process, cold = _launch_external_data_aot_process(
        cache_root, tmp_path / "cold.json", force_load=False
    )
    assert cold_process.returncode == 0, cold_process.stderr
    assert cold["num_aot_compiles"] == 1
    assert cold["num_aot_artifacts_saved"] == 1
    assert cold["num_aot_artifacts_loaded"] == 0
    assert cold["loaded_from_disk"] is False

    warm_process, warm = _launch_external_data_aot_process(
        cache_root, tmp_path / "warm.json", force_load=True
    )
    assert warm_process.returncode == 0, warm_process.stderr
    assert warm["num_aot_compiles"] == 0
    assert warm["num_aot_artifacts_saved"] == 0
    assert warm["num_aot_artifacts_loaded"] == 1
    assert warm["loaded_from_disk"] is True
    assert warm["output"] == cold["output"]


@pytest.mark.skipif(
    not is_torch_equal_or_newer("2.11.0"),
    reason="requires AOT external_data support in torch 2.11+",
)
def test_external_data_fingerprint_mismatch_obeys_load_mode(tmp_path: Path):
    cache_root = tmp_path / "cache"

    # Save an artifact whose wrapper flag uses the old behavior.
    cold_process, cold = _launch_external_data_aot_process(
        cache_root,
        tmp_path / "cold.json",
        force_load=False,
        forward_flag=False,
    )
    assert cold_process.returncode == 0, cold_process.stderr
    assert cold["num_aot_artifacts_saved"] == 1

    # Forced loading must reject the old fingerprint before compiling.
    forced_process, forced = _launch_external_data_aot_process(
        cache_root,
        tmp_path / "forced.json",
        force_load=True,
        forward_flag=True,
    )
    assert forced_process.returncode != 0
    assert "external reference" in forced["error"].lower()
    assert forced["num_aot_compiles"] == 0
    assert forced["num_aot_artifacts_loaded"] == 0

    # Normal mode falls back to compilation and must run the new behavior.
    warm_process, warm = _launch_external_data_aot_process(
        cache_root,
        tmp_path / "warm.json",
        force_load=False,
        forward_flag=True,
    )
    assert warm_process.returncode == 0, warm_process.stderr
    assert warm["num_aot_compiles"] == 1
    assert warm["num_aot_artifacts_loaded"] == 0
    assert warm["loaded_from_disk"] is False
    assert warm["output"] != cold["output"]
    assert torch.equal(torch.tensor(warm["output"]), torch.tensor(cold["output"]) + 1)
    logs = f"{warm_process.stdout}\n{warm_process.stderr}".lower()
    assert "compiling model again due to a load failure" in logs


def make_vllm_config() -> VllmConfig:
    return VllmConfig(
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            backend="inductor",
        )
    )


@contextmanager
def use_vllm_config(vllm_config: VllmConfig):
    with set_forward_context({}, vllm_config), set_current_vllm_config(vllm_config):
        yield


@pytest.mark.skipif(not is_torch_equal_or_newer("2.10.0"), reason="requires torch 2.10")
def test_no_dynamo_cache_entry(monkeypatch: pytest.MonkeyPatch):
    with monkeypatch.context() as m:
        vllm_config = make_vllm_config()
        args = (torch.randn(10, 10),)
        expected = reference_fn(*args)
        with use_vllm_config(vllm_config):
            m.setenv("VLLM_USE_AOT_COMPILE", "0")
            m.setenv("VLLM_USE_MEGA_AOT_ARTIFACT", "1")
            m.setenv("VLLM_USE_STANDALONE_COMPILE", "1")
            with (
                pytest.raises(RuntimeError, match="Detected recompile"),
                torch.compiler.set_stance("fail_on_recompile"),
            ):
                CompiledMod(vllm_config=vllm_config)(*args)
            disable_envs_cache()

            m.setenv("VLLM_USE_AOT_COMPILE", "1")
            torch._dynamo.reset()
            with torch.compiler.set_stance("fail_on_recompile"):
                actual = CompiledMod(vllm_config=vllm_config)(*args)
            assert torch.allclose(actual, expected)


@pytest.mark.skipif(not is_torch_equal_or_newer("2.10.0"), reason="requires torch 2.10")
def test_force_aot_load(monkeypatch: pytest.MonkeyPatch):
    with tempfile.TemporaryDirectory() as tmpdirname, monkeypatch.context() as m:
        args = (torch.randn(10, 10),)
        m.setenv("VLLM_USE_AOT_COMPILE", "1")
        m.setenv("VLLM_USE_MEGA_AOT_ARTIFACT", "1")
        m.setenv("VLLM_USE_STANDALONE_COMPILE", "1")
        m.setenv("VLLM_FORCE_AOT_LOAD", "1")
        m.setenv("VLLM_CACHE_ROOT", tmpdirname)
        vllm_config = make_vllm_config()
        with use_vllm_config(vllm_config), pytest.raises(FileNotFoundError):
            CompiledMod(vllm_config=vllm_config)(*args)


@pytest.mark.skipif(not is_torch_equal_or_newer("2.10.0"), reason="requires torch 2.10")
def test_save_and_load(monkeypatch: pytest.MonkeyPatch):
    with monkeypatch.context() as m:
        args = (torch.randn(10, 10),)

        with tempfile.TemporaryDirectory() as tmpdirname:
            m.setenv("VLLM_CACHE_ROOT", tmpdirname)
            m.setenv("VLLM_USE_AOT_COMPILE", "1")
            m.setenv("VLLM_USE_MEGA_AOT_ARTIFACT", "1")
            m.setenv("VLLM_USE_STANDALONE_COMPILE", "1")
            disable_envs_cache()
            vllm_config = make_vllm_config()
            with (
                use_vllm_config(vllm_config),
                compilation_counter.expect(
                    num_aot_compiles=1,
                    num_aot_artifacts_saved=1,
                    num_aot_artifacts_loaded=0,
                ),
            ):
                compiled_mod = CompiledMod(vllm_config=vllm_config)
                expected = compiled_mod(*args)
            assert isinstance(expected, torch.Tensor)

            disable_envs_cache()

            m.setenv("VLLM_FORCE_AOT_LOAD", "1")
            vllm_config = make_vllm_config()
            with (
                use_vllm_config(vllm_config),
                compilation_counter.expect(
                    num_aot_compiles=0,
                    num_aot_artifacts_saved=0,
                    num_aot_artifacts_loaded=1,
                ),
            ):
                cached_mod = CompiledMod(vllm_config=vllm_config)
                ret = cached_mod(*args)
            assert isinstance(ret, torch.Tensor)
            assert cached_mod.was_aot_compile_fn_loaded_from_disk, (
                "Expected was_aot_compile_fn_loaded_from_disk to be True"
            )
            assert torch.allclose(ret, expected)


@pytest.mark.skipif(not is_torch_equal_or_newer("2.10.0"), reason="requires torch 2.10")
def test_save_and_load_slice(monkeypatch: pytest.MonkeyPatch):
    from torch._subclasses import FakeTensorMode
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    def foo(x: torch.Tensor):
        return x[slice(0, x.shape[0])]

    vllm_config = make_vllm_config()

    example_input = torch.randn(10, 10)
    torch._dynamo.mark_dynamic(example_input, 0)
    gm = torch.fx.symbolic_trace(foo)
    assert "getitem_1 = x[slice(0, getitem, None)]" in gm.code
    with use_vllm_config(vllm_config):
        payload = VllmSerializableFunction.serialize_graph_module(gm)
        fake_mode = FakeTensorMode(shape_env=ShapeEnv())
        loaded_gm = VllmSerializableFunction.deserialize_graph_module(
            payload, fake_mode
        )

    assert gm.code == loaded_gm.code


@pytest.mark.skipif(not is_torch_equal_or_newer("2.10.0"), reason="requires torch 2.10")
def test_cache_load_returns_tuple_consistency_tuple_output(
    monkeypatch: pytest.MonkeyPatch,
):
    """
    Test that cache loading correctly handles models that return tuples.

    This verifies that when a model returns a tuple of tensors, the output
    type is preserved as a tuple between fresh compilation and cache load.
    """
    with monkeypatch.context() as m:
        args = (torch.randn(10, 10),)

        with tempfile.TemporaryDirectory() as tmpdirname:
            m.setenv("VLLM_CACHE_ROOT", tmpdirname)
            m.setenv("VLLM_USE_AOT_COMPILE", "1")
            m.setenv("VLLM_USE_MEGA_AOT_ARTIFACT", "1")
            m.setenv("VLLM_USE_STANDALONE_COMPILE", "1")
            vllm_config = make_vllm_config()

            # Fresh compilation with tuple-returning model
            with use_vllm_config(vllm_config):
                compiled_mod = CompiledModTuple(vllm_config=vllm_config)
                fresh_result = compiled_mod(*args)
                fresh_result_type = type(fresh_result)

            # Verify fresh result is a tuple
            assert isinstance(fresh_result, tuple), (
                f"Fresh compile should return tuple, got {fresh_result_type}"
            )
            assert len(fresh_result) == 2, (
                f"Fresh compile should return 2-tuple, got {len(fresh_result)}"
            )

            disable_envs_cache()

            # Load from cache
            m.setenv("VLLM_FORCE_AOT_LOAD", "1")
            vllm_config = make_vllm_config()
            with use_vllm_config(vllm_config):
                cached_mod = CompiledModTuple(vllm_config=vllm_config)
                cached_result = cached_mod(*args)
                cached_result_type = type(cached_result)

            # Verify cache was actually loaded
            assert cached_mod.was_aot_compile_fn_loaded_from_disk, (
                "Expected was_aot_compile_fn_loaded_from_disk to be True after "
                "loading from cache"
            )

            # Verify cached result is also a tuple
            assert isinstance(cached_result, tuple), (
                f"Cache load should return tuple, got {cached_result_type}. "
                "This indicates the returns_tuple logic is not preserving "
                "tuple outputs when loading from cache."
            )
            assert len(cached_result) == 2, (
                f"Cache load should return 2-tuple, got {len(cached_result)}"
            )

            # Verify values match
            assert torch.allclose(cached_result[0], fresh_result[0]), (
                "Cached result[0] values should match fresh compilation"
            )
            assert torch.allclose(cached_result[1], fresh_result[1]), (
                "Cached result[1] values should match fresh compilation"
            )


@pytest.mark.skipif(not is_torch_equal_or_newer("2.10.0"), reason="requires torch 2.10")
def test_shape_env(monkeypatch: pytest.MonkeyPatch):
    """
    Test that the shape environment is correctly serialized and preserved
    when loading from cache.
    """
    with monkeypatch.context() as m:
        args = (torch.randn(10, 10),)

        with tempfile.TemporaryDirectory() as tmpdirname:
            m.setenv("VLLM_CACHE_ROOT", tmpdirname)
            m.setenv("VLLM_USE_AOT_COMPILE", "1")
            m.setenv("VLLM_USE_MEGA_AOT_ARTIFACT", "1")
            m.setenv("VLLM_USE_STANDALONE_COMPILE", "1")
            vllm_config = make_vllm_config()
            with use_vllm_config(vllm_config):
                compiled_mod = CompiledMod(vllm_config=vllm_config)
                compiled_mod(*args)
                artifacts = compiled_mod.aot_compiled_fn._artifacts
                guards_string = artifacts.compiled_fn.shape_env.format_guards()
                assert guards_string == " - s77 <= 42\n - Eq(Mod(s77, 2), 0)"

            disable_envs_cache()

            m.setenv("VLLM_FORCE_AOT_LOAD", "1")
            vllm_config = make_vllm_config()
            with use_vllm_config(vllm_config):
                compiled_mod = CompiledMod(vllm_config=vllm_config)
                compiled_mod(*args)
                assert compiled_mod.was_aot_compile_fn_loaded_from_disk, (
                    "Expected was_aot_compile_fn_loaded_from_disk to be True"
                )
                artifacts = compiled_mod.aot_compiled_fn._artifacts
                guards_string = artifacts.compiled_fn.shape_env.format_guards()
                assert guards_string == " - s77 <= 42\n - Eq(Mod(s77, 2), 0)"


@pytest.mark.skipif(not is_torch_equal_or_newer("2.10.0"), reason="requires torch 2.10")
def test_partition_wrapper_applied_on_aot_load(
    monkeypatch: pytest.MonkeyPatch, vllm_tmp_cache: Path, mocker
):
    """
    Test that partition wrappers are applied when loading AOT cached functions.

    This test verifies the fix for GitHub issue #31439 where AOT compile
    caused 2x latency regression when use_inductor_graph_partition=True.
    The root cause was that partition wrapper context was bypassed when
    loading from AOT cache.
    """
    from vllm.config import CUDAGraphMode

    args = (torch.randn(10, 10),)
    monkeypatch.setenv("VLLM_USE_AOT_COMPILE", "1")

    # Create config with partition enabled
    vllm_config = VllmConfig(
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            use_inductor_graph_partition=True,
            cudagraph_mode=CUDAGraphMode.PIECEWISE,
        )
    )

    # First compilation - save to cache
    with use_vllm_config(vllm_config):
        compiled_mod = CompiledMod(vllm_config=vllm_config)
        compiled_mod(*args)

    disable_envs_cache()

    # Second run - load from cache, verify partition wrapper applied
    monkeypatch.setenv("VLLM_FORCE_AOT_LOAD", "1")
    vllm_config = VllmConfig(
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            use_inductor_graph_partition=True,
            cudagraph_mode=CUDAGraphMode.PIECEWISE,
        )
    )

    # Use mocker to spy on set_customized_partition_wrappers
    spy = mocker.spy(torch._inductor.utils, "set_customized_partition_wrappers")

    with use_vllm_config(vllm_config):
        compiled_mod = CompiledMod(vllm_config=vllm_config)

        # First call after restart: loads from AOT cache.
        # This tests the fix for the first call after a restart.
        compiled_mod(*args)

        # Verify cache was loaded
        assert compiled_mod.was_aot_compile_fn_loaded_from_disk, (
            "Expected was_aot_compile_fn_loaded_from_disk to be True"
        )

        # Verify partition wrapper was called on AOT load.
        assert spy.call_count >= 2, (
            "Expected partition wrapper to be set and cleared on AOT load, "
            f"got {spy.call_count} calls"
        )
        # First call should set a wrapper, last call should clear it
        assert spy.call_args_list[0][0][0] is not None, (
            "First call on AOT load should set a wrapper function"
        )
        assert spy.call_args_list[-1][0][0] is None, (
            "Last call on AOT load should clear the wrapper"
        )

        # Reset for the next check.
        spy.reset_mock()

        # Subsequent call: uses the cached `aot_compiled_fn`.
        # This tests the fix for subsequent calls.
        compiled_mod(*args)

        # Verify partition wrapper was called on the subsequent call.
        assert spy.call_count >= 2, (
            "Expected partition wrapper set and cleared on subsequent "
            f"call, got {spy.call_count} calls"
        )
        assert spy.call_args_list[0][0][0] is not None, (
            "First call on subsequent call should set a wrapper function"
        )
        assert spy.call_args_list[-1][0][0] is None, (
            "Last call on subsequent call should clear the wrapper"
        )


@create_new_process_for_each_test("spawn")
def test_standalone_compile_correctness():
    """Outputs must match regardless of VLLM_USE_STANDALONE_COMPILE."""
    import json

    from ..utils import compare_two_settings

    compilation_config = json.dumps(
        {
            "mode": CompilationMode.VLLM_COMPILE,
        }
    )

    common_args = [
        "--dtype",
        "float16",
        "--max-model-len",
        "256",
        "--compilation_config",
        compilation_config,
    ]

    compare_two_settings(
        "facebook/opt-125m",
        common_args,
        common_args,
        env1={"VLLM_USE_STANDALONE_COMPILE": "1"},
        env2={
            "VLLM_USE_STANDALONE_COMPILE": "0",
            "VLLM_USE_MEGA_AOT_ARTIFACT": "0",
        },
    )


@pytest.mark.skipif(not is_torch_equal_or_newer("2.10.0"), reason="requires torch 2.10")
@create_new_process_for_each_test("spawn")
def test_gpt2_cache_hit(monkeypatch: pytest.MonkeyPatch):
    """
    Test that compiling gpt2 twice results in a cache hit.

    Counter values are read from the EngineCore subprocess via
    ``LLM.collective_rpc`` so the test works under default V1
    multiprocessing (no shared memory between test and engine).
    """

    from vllm import LLM

    def _snap(self):
        from vllm.compilation.counter import compilation_counter

        return (
            compilation_counter.num_aot_compiles,
            compilation_counter.num_aot_artifacts_saved,
            compilation_counter.num_aot_artifacts_loaded,
        )

    # collective_rpc(callable) requires pickle-based serialization.
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    with monkeypatch.context() as m, tempfile.TemporaryDirectory() as tmpdirname:
        m.setenv("VLLM_CACHE_ROOT", tmpdirname)
        m.setenv("VLLM_USE_AOT_COMPILE", "1")
        # First compilation - initialize model and generate
        llm_model = LLM(
            model="openai-community/gpt2",
            compilation_config=CompilationConfig(
                mode=CompilationMode.VLLM_COMPILE,
            ),
            max_model_len=256,
        )

        llm_model.generate("Hello, my name is")
        assert llm_model.collective_rpc(_snap)[0] == (1, 1, 0)

        # Clean up first model
        del llm_model
        disable_envs_cache()

        # Second compilation - should hit cache
        m.setenv("VLLM_FORCE_AOT_LOAD", "1")
        llm_model = LLM(
            model="openai-community/gpt2",
            compilation_config=CompilationConfig(
                mode=CompilationMode.VLLM_COMPILE,
            ),
            max_model_len=256,
        )
        llm_model.generate("Hello, my name is")
        assert llm_model.collective_rpc(_snap)[0] == (0, 0, 1)


@pytest.mark.skipif(not is_torch_equal_or_newer("2.10.0"), reason="requires torch 2.10")
class TestStandaloneCompiledArtifacts:
    def test_init(self):
        cache = StandaloneCompiledArtifacts()
        assert cache.submodule_bytes == {}
        assert cache.submodule_bytes_store == {}
        assert cache.loaded_submodule_store == {}

    def test_insert_new_artifact(self):
        cache = StandaloneCompiledArtifacts()
        test_data = b"test_artifact_data"
        submod_name = "test_submod"
        shape = "s1"

        hasher = hashlib.sha256()
        hasher.update(test_data)
        expected_hash = hasher.hexdigest()

        cache.insert(submod_name, shape, test_data)

        assert f"{submod_name}_{shape}" in cache.submodule_bytes
        assert cache.submodule_bytes[f"{submod_name}_{shape}"] == expected_hash
        assert expected_hash in cache.submodule_bytes_store
        assert cache.submodule_bytes_store[expected_hash] == test_data

    def test_insert_duplicate_artifact(self):
        cache = StandaloneCompiledArtifacts()

        test_data = b"duplicate_test_data"
        submod_name1 = "submod1"
        submod_name2 = "submod2"
        shape = "s2"

        cache.insert(submod_name1, shape, test_data)
        cache.insert(submod_name2, shape, test_data)

        hash1 = cache.submodule_bytes[f"{submod_name1}_{shape}"]
        hash2 = cache.submodule_bytes[f"{submod_name2}_{shape}"]
        assert hash1 == hash2

        assert len(cache.submodule_bytes_store) == 1
        assert len(cache.submodule_bytes) == 2

    def test_get_artifact(self):
        cache = StandaloneCompiledArtifacts()
        test_data = b"retrievable_data"
        submod_name = "mod1"
        shape = "shape16"

        cache.insert(submod_name, shape, test_data)
        retrieved_data = cache.get(submod_name, shape)

        assert retrieved_data == test_data

    def test_get_nonexistent_artifact(self):
        cache = StandaloneCompiledArtifacts()

        with pytest.raises(KeyError):
            cache.get("nonexistent", "shape")

    def test_size_bytes(self):
        cache = StandaloneCompiledArtifacts()

        assert cache.size_bytes() == 0

        data1 = b"x" * 100
        data2 = b"y" * 200
        cache.insert("mod1", "shape1", data1)
        cache.insert("mod2", "shape2", data2)

        assert cache.size_bytes() == 300

    def test_num_artifacts_and_entries(self):
        cache = StandaloneCompiledArtifacts()

        assert cache.num_artifacts() == 0
        assert cache.num_entries() == 0

        cache.insert("mod1", "shape1", b"data1")
        cache.insert("mod2", "shape2", b"data2")
        assert cache.num_artifacts() == 2
        assert cache.num_entries() == 2

        cache.insert("mod3", "shape3", b"data1")
        assert cache.num_artifacts() == 2
        assert cache.num_entries() == 3

    @patch("torch._inductor.standalone_compile.AOTCompiledArtifact.deserialize")
    def test_load_all_success(self, mock_deserialize):
        """Test successful loading of all artifacts"""
        cache = StandaloneCompiledArtifacts()

        mock_artifact1 = Mock()
        mock_artifact2 = Mock()
        mock_deserialize.side_effect = [mock_artifact1, mock_artifact2]

        cache.insert("mod1", "shape1", pickle.dumps(b"data1"))
        cache.insert("mod2", "shape2", pickle.dumps(b"data2"))

        cache.load_all()

        assert len(cache.loaded_submodule_store) == 2
        assert mock_deserialize.call_count == 2

    @patch("torch._inductor.standalone_compile.AOTCompiledArtifact.deserialize")
    def test_load_all_already_loaded(self, mock_deserialize):
        """Test that load_all skips if already loaded"""
        cache = StandaloneCompiledArtifacts()

        mock_artifact = Mock()
        cache.submodule_bytes_store["hash1"] = pickle.dumps(b"data1")
        cache.loaded_submodule_store["hash1"] = mock_artifact

        cache.load_all()

        mock_deserialize.assert_not_called()

    @patch("torch._inductor.standalone_compile.AOTCompiledArtifact.deserialize")
    def test_get_loaded_artifact(self, mock_deserialize):
        """Test retrieving loaded artifacts"""
        cache = StandaloneCompiledArtifacts()

        mock_artifact = Mock()
        mock_deserialize.return_value = mock_artifact

        submod_name = "test_mod"
        shape = "test_shape"
        cache.insert(submod_name, shape, pickle.dumps(b"test_data"))
        cache.load_all()

        retrieved_artifact = cache.get_loaded(submod_name, shape)
        assert retrieved_artifact == mock_artifact

    def test_getstate_setstate(self):
        cache = StandaloneCompiledArtifacts()

        cache.insert("mod1", "shape1", b"data1")
        cache.insert("mod2", "shape2", b"data2")

        cache.loaded_submodule_store["hash1"] = Mock()

        state = cache.__getstate__()

        assert "submodule_bytes" in state
        assert "submodule_bytes_store" in state
        assert "loaded_submodule_store" not in state

        new_cache = StandaloneCompiledArtifacts()
        new_cache.__setstate__(state)

        assert new_cache.submodule_bytes == cache.submodule_bytes
        assert new_cache.submodule_bytes_store == cache.submodule_bytes_store
        assert new_cache.loaded_submodule_store == {}

    def test_pickle_roundtrip(self):
        cache = StandaloneCompiledArtifacts()

        test_data1 = b"pickle_test_data_1"
        test_data2 = b"pickle_test_data_2"
        cache.insert("mod1", "shape1", test_data1)
        cache.insert("mod2", "shape2", test_data2)

        pickled_data = pickle.dumps(cache)
        restored_cache = pickle.loads(pickled_data)

        assert restored_cache.get("mod1", "shape1") == test_data1
        assert restored_cache.get("mod2", "shape2") == test_data2
        assert restored_cache.num_artifacts() == cache.num_artifacts()
        assert restored_cache.num_entries() == cache.num_entries()
        assert restored_cache.size_bytes() == cache.size_bytes()

        assert len(restored_cache.loaded_submodule_store) == 0


@pytest.mark.skipif(not is_torch_equal_or_newer("2.10.0"), reason="requires torch 2.10")
class TestStandaloneCompiledArtifactsIntegration:
    def test_add_pickle_unpickle(self):
        cache = StandaloneCompiledArtifacts()

        artifacts = {
            ("mod1", "shape1"): b"m1s1_artifact",
            ("mod1", "shape2"): b"m1s2_artifact",
            ("mod2", "shape1"): b"m2s1_artifact",
            ("mod2", "shape2"): b"m2s2_artifact",
        }

        for (submod, shape), data in artifacts.items():
            cache.insert(submod, shape, data)

        assert cache.num_entries() == 4
        assert cache.num_artifacts() == 4

        for (submod, shape), expected_data in artifacts.items():
            retrieved_data = cache.get(submod, shape)
            assert retrieved_data == expected_data

        pickled = pickle.dumps(cache)
        restored_cache = pickle.loads(pickled)

        for (submod, shape), expected_data in artifacts.items():
            retrieved_data = restored_cache.get(submod, shape)
            assert retrieved_data == expected_data

    def test_deduplication(self):
        cache = StandaloneCompiledArtifacts()

        shared_data = b"shared_artifact_data" * 1000

        cache.insert("mod1", "shape1", shared_data)
        cache.insert("mod2", "shape1", shared_data)
        cache.insert("mod1", "shape2", shared_data)
        cache.insert("mod3", "shape3", shared_data)

        assert cache.num_entries() == 4
        assert cache.num_artifacts() == 1
        assert cache.size_bytes() == len(shared_data)

        for submod, shape in [
            ("mod1", "shape1"),
            ("mod2", "shape1"),
            ("mod1", "shape2"),
            ("mod3", "shape3"),
        ]:
            assert cache.get(submod, shape) == shared_data

    @pytest.mark.skipif(
        envs.VLLM_USE_MEGA_AOT_ARTIFACT,
        reason="There's no AOT Autograd run with mega artifact",
    )
    def test_functorch_config(self):
        vllm_config = make_vllm_config()
        example_inputs = (torch.randn(10, 10),)

        def add_1(x: torch.Tensor):
            return x + 1

        gm = torch._dynamo.functional_export.dynamo_graph_capture_for_export(add_1)(
            *example_inputs
        )

        gm.graph._codegen = torch.fx.graph.CodeGen()
        gm._dynamo_bytecode_flatten = None
        gm._dynamo_bytecode_unflatten = None

        with (
            torch._functorch.config.patch(bundled_autograd_cache=False),
            set_current_vllm_config(vllm_config),
        ):
            with torch._functorch.config.patch(bundled_autograd_cache=True):
                fn = VllmSerializableFunction(gm, example_inputs, "", add_1)

            payload = VllmSerializableFunction.serialize_compile_artifacts(fn)

            config = None

            def backend(*args, **kwargs) -> VllmSerializableFunction:
                nonlocal config
                # bundled_autograd_cache should be True even compiler backend
                # runs with bundled_autograd_cache=False in ambient context.
                config = torch._functorch.config.save_config_portable()
                return fn

            loaded_fn = VllmSerializableFunction.deserialize_compile_artifacts(payload)
            with patch.object(VllmBackend, "__call__", backend):
                loaded_fn(*example_inputs)

        assert isinstance(config, dict)
        assert "bundled_autograd_cache" in config
        assert config["bundled_autograd_cache"] is True


@pytest.mark.skipif(not is_torch_equal_or_newer("2.10.0"), reason="requires torch 2.10")
def test_disable_compile_cache_skips_aot_save(
    monkeypatch: pytest.MonkeyPatch, fresh_vllm_cache: str
):
    """When VLLM_DISABLE_COMPILE_CACHE=1, AOT artifacts must not be saved."""
    monkeypatch.setenv("VLLM_DISABLE_COMPILE_CACHE", "1")
    monkeypatch.setenv("VLLM_USE_AOT_COMPILE", "1")
    disable_envs_cache()

    args = (torch.randn(10, 10),)
    expected = reference_fn(*args)
    vllm_config = make_vllm_config()

    with (
        use_vllm_config(vllm_config),
        compilation_counter.expect(
            num_aot_compiles=1,
            num_aot_artifacts_saved=0,
            num_aot_artifacts_loaded=0,
        ),
    ):
        mod = CompiledMod(vllm_config=vllm_config)
        actual = mod(*args)

    assert torch.allclose(actual, expected)

    # No cached artifact should exist on disk
    aot_dir = os.path.join(fresh_vllm_cache, "torch_compile_cache", "torch_aot_compile")
    if os.path.isdir(aot_dir):
        for root, _dirs, files in os.walk(aot_dir):
            for f in files:
                assert f != "model", (
                    f"AOT artifact unexpectedly saved at {os.path.join(root, f)}"
                )


@pytest.mark.skipif(not is_torch_equal_or_newer("2.10.0"), reason="requires torch 2.10")
def test_disable_compile_cache_skips_aot_load(
    monkeypatch: pytest.MonkeyPatch, fresh_vllm_cache: str
):
    """When VLLM_DISABLE_COMPILE_CACHE=1, AOT artifacts must not be loaded."""
    # Phase 1: compile and save with cache enabled
    monkeypatch.setenv("VLLM_USE_AOT_COMPILE", "1")
    disable_envs_cache()

    args = (torch.randn(10, 10),)
    vllm_config = make_vllm_config()

    with (
        use_vllm_config(vllm_config),
        compilation_counter.expect(num_aot_artifacts_saved=1),
    ):
        CompiledMod(vllm_config=vllm_config)(*args)

    # Phase 2: disable cache, compile again — should NOT load from disk
    monkeypatch.setenv("VLLM_DISABLE_COMPILE_CACHE", "1")
    disable_envs_cache()
    torch._dynamo.reset()

    vllm_config = make_vllm_config()
    with (
        use_vllm_config(vllm_config),
        compilation_counter.expect(
            num_aot_compiles=1,
            num_aot_artifacts_saved=0,
            num_aot_artifacts_loaded=0,
        ),
    ):
        mod = CompiledMod(vllm_config=vllm_config)
        mod(*args)

    assert not mod.was_aot_compile_fn_loaded_from_disk

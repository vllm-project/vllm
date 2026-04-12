from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "vllm"
    / "v1"
    / "attention"
    / "backends"
    / "rocm_aiter_fa_utils.py"
)


def load_helper_module():
    assert MODULE_PATH.is_file(), f"Missing helper module: {MODULE_PATH}"
    spec = spec_from_file_location("rocm_aiter_fa_utils", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_small_head_size_uses_unified_decode_fallback():
    module = load_helper_module()

    assert (
        module.should_use_unified_decode_fallback(
            head_size=32,
            sliding_window=(-1, -1),
        )
        is True
    )


def test_sliding_window_uses_unified_decode_fallback():
    module = load_helper_module()

    assert (
        module.should_use_unified_decode_fallback(
            head_size=128,
            sliding_window=(-1, -1),
        )
        is False
    )
    assert (
        module.should_use_unified_decode_fallback(
            head_size=128,
            sliding_window=(255, 0),
        )
        is True
    )

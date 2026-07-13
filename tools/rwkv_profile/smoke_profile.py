# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Clean-environment smoke checks for an installed RWKV-only artifact."""

from __future__ import annotations

import importlib.metadata
import importlib.util

OMITTED_DISTRIBUTIONS = (
    "apache-tvm-ffi",
    "compressed-tensors",
    "flashinfer-cubin",
    "flashinfer-python",
    "humming-kernels",
    "mistral-common",
    "model-hosting-container-standards",
    "numba",
    "nvidia-cutlass-dsl",
    "opencv-python-headless",
    "outlines-core",
    "PyNvVideoCodec",
    "quack-kernels",
    "tilelang",
    "tokenspeed-mla",
    "torchaudio",
    "torchvision",
    "xgrammar",
)


def assert_distribution_absent(name: str) -> None:
    try:
        importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError:
        return
    raise AssertionError(f"omitted distribution is installed: {name}")


def main() -> None:
    from vllm.build_profile import get_build_profile_metadata

    metadata = get_build_profile_metadata()
    assert metadata.profile == "rwkv", metadata
    assert "rwkv7_ops" in metadata.configured_targets, metadata
    assert "_rapid_sampling" in metadata.configured_targets, metadata
    assert "_C_stable_libtorch" not in metadata.configured_targets, metadata
    assert not metadata.external_projects, metadata
    assert "rapid_sampling" in metadata.supported_serving_features, metadata
    assert "openai_chat" in metadata.supported_serving_features, metadata

    assert importlib.util.find_spec("vllm._C_stable_libtorch") is None
    import vllm._rapid_sampling  # noqa: F401
    import vllm.rwkv7_ops  # noqa: F401

    import vllm.platforms.cuda  # noqa: F401
    from vllm.entrypoints.openai import api_server  # noqa: F401
    from vllm.tokenizers.registry import get_tokenizer
    from vllm.utils.import_utils import import_pynvml

    assert import_pynvml().__name__ == "vllm.third_party.pynvml"
    tokenizer = get_tokenizer("BlinkDL/rwkv7-g1", tokenizer_mode="rwkv")
    assert tokenizer.encode("Hello world", add_special_tokens=False) == [33155, 40213]
    assert tokenizer.decode([33155, 40213]) == "Hello world"
    assert (
        tokenizer.apply_chat_template(
            [{"role": "user", "content": "Hi"}],
            tokenize=False,
            add_generation_prompt=True,
        )
        == "User: Hi\n\nAssistant: <think"
    )

    for distribution in OMITTED_DISTRIBUTIONS:
        assert_distribution_absent(distribution)

    print("RWKV-only clean-environment smoke passed")


if __name__ == "__main__":
    main()

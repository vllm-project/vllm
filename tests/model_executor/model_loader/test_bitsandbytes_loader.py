# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional

from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader.bitsandbytes_loader import (
    BitsAndBytesModelLoader,
)


class _DummyBitsAndBytesLoader(BitsAndBytesModelLoader):
    """Test helper that bypasses any real HF interactions."""

    def __init__(
        self, load_config: LoadConfig, mock_result: tuple[str, list[str], str]
    ):
        super().__init__(load_config)
        self._mock_result = mock_result

    def _get_weight_files(  # type: ignore[override]
        self,
        model_name_or_path: str,
        allowed_patterns: list[str],
        revision: Optional[str] = None,
    ) -> tuple[str, list[str], str]:
        return self._mock_result


def test_bitsandbytes_loader_detects_safetensors_from_files(tmp_path):
    """Even if the allow-pattern looks like *.bin, safetensors files are detected."""

    llm_dir = tmp_path / "llm"
    llm_dir.mkdir()
    safetensor = llm_dir / "model-00001-of-00002.safetensors"
    safetensor.write_bytes(b"test")

    load_config = LoadConfig()
    loader = _DummyBitsAndBytesLoader(
        load_config,
        mock_result=(str(tmp_path), [str(safetensor)], "*.bin"),
    )

    files, use_safetensors = loader._prepare_weights(str(tmp_path), revision=None)

    assert use_safetensors is True
    assert files == [str(safetensor)]

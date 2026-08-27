from types import SimpleNamespace
from unittest.mock import patch

from vllm.model_executor.layers.fused_moe.prepare_finalize import (
    flashinfer_nvlink_one_sided,
)


def test_get_local_sizes_without_dp_metadata():
    context = SimpleNamespace(dp_metadata=None)

    with patch.object(
        flashinfer_nvlink_one_sided,
        "get_forward_context",
        return_value=context,
    ):
        assert flashinfer_nvlink_one_sided.get_local_sizes() is None


def test_get_local_sizes_with_dp_metadata():
    metadata = SimpleNamespace(get_chunk_sizes_across_dp_rank=lambda: [3, 5])
    context = SimpleNamespace(dp_metadata=metadata)

    with patch.object(
        flashinfer_nvlink_one_sided,
        "get_forward_context",
        return_value=context,
    ):
        assert flashinfer_nvlink_one_sided.get_local_sizes() == [3, 5]

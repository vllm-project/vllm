# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.model_executor.models.fireredlid import FireRedLIDProcessingInfo
from vllm.transformers_utils.processors.fireredlid import (
    FIREREDLID_EXPECTED_CONTEXT,
    FireRedLIDFeatureExtractor,
)

pytestmark = pytest.mark.skip_global_cleanup


def test_fireredlid_feature_extractor_accepts_fixed_frontend_geometry() -> None:
    extractor = FireRedLIDFeatureExtractor(
        dim=128,
        num_mel_bins=128,
        left_context=2,
        right_context=4,
    )

    assert extractor.num_mel_bins == 128
    assert extractor.context == FIREREDLID_EXPECTED_CONTEXT


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"right_context": 1_000_000}, "fixed seven-frame frontend context"),
        ({"num_mel_bins": 1_000_000}, "num_mel_bins to match dim"),
    ],
)
def test_fireredlid_feature_extractor_rejects_unsafe_geometry_before_use(
    kwargs: dict[str, int],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        FireRedLIDFeatureExtractor(**kwargs)


class _ProcessorContext:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.processor = SimpleNamespace(
            feature_extractor=SimpleNamespace(
                dim=80,
                left_context=3,
                num_mel_bins=80,
                right_context=3,
            )
        )

    def get_hf_processor(self, **kwargs: object):
        self.calls.append(kwargs)
        return self.processor


@pytest.mark.parametrize("field", ["right_context", "num_mel_bins", "dim"])
def test_fireredlid_processing_info_rejects_static_processor_overrides(
    field: str,
) -> None:
    ctx = _ProcessorContext()
    info = FireRedLIDProcessingInfo(ctx)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="static feature extractor fields"):
        info.get_hf_processor(**{field: 1_000_000})

    assert ctx.calls == [{}]


@pytest.mark.parametrize(
    ("field", "value"),
    [("right_context", 3), ("num_mel_bins", 80), ("dim", 80)],
)
def test_fireredlid_processing_info_allows_noop_static_overrides(
    field: str,
    value: int,
) -> None:
    ctx = _ProcessorContext()
    info = FireRedLIDProcessingInfo(ctx)  # type: ignore[arg-type]

    processor = info.get_hf_processor(**{field: value})

    assert processor is ctx.processor
    assert ctx.calls == [{}]


def test_fireredlid_processing_info_rejects_matching_attacker_width_overrides() -> None:
    ctx = _ProcessorContext()
    info = FireRedLIDProcessingInfo(ctx)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="static feature extractor fields"):
        info.get_hf_processor(dim=1_000_000, num_mel_bins=1_000_000)

    assert ctx.calls == [{}]

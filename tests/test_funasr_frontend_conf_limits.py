# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
import types
from types import SimpleNamespace

import pytest

try:
    import torchaudio.compliance.kaldi  # noqa: F401
except ModuleNotFoundError:
    torchaudio = types.ModuleType("torchaudio")
    torchaudio.__path__ = []  # type: ignore[attr-defined]
    compliance = types.ModuleType("torchaudio.compliance")
    compliance.__path__ = []  # type: ignore[attr-defined]
    kaldi = types.ModuleType("torchaudio.compliance.kaldi")
    torchaudio.compliance = compliance  # type: ignore[attr-defined]
    compliance.kaldi = kaldi  # type: ignore[attr-defined]
    sys.modules["torchaudio"] = torchaudio
    sys.modules["torchaudio.compliance"] = compliance
    sys.modules["torchaudio.compliance.kaldi"] = kaldi

from vllm.model_executor.models.funasr import FunASRProcessingInfo
from vllm.transformers_utils.processors.funasr import FunASRFeatureExtractor

pytestmark = pytest.mark.skip_global_cleanup


class _FakeProcessingContext:
    def __init__(self, frontend_conf: dict[str, object]):
        self.frontend_conf = frontend_conf
        self.processor_calls: list[dict[str, object]] = []

    def get_hf_processor(self, **kwargs: object):
        self.processor_calls.append(dict(kwargs))
        frontend_conf = kwargs.get("frontend_conf", self.frontend_conf)
        return SimpleNamespace(
            feature_extractor=FunASRFeatureExtractor(frontend_conf=frontend_conf)
        )


def test_funasr_allows_request_lfr_m_within_configured_frontend() -> None:
    ctx = _FakeProcessingContext({"lfr_m": 7, "lfr_n": 6, "n_mels": 80})
    info = FunASRProcessingInfo(ctx)  # type: ignore[arg-type]

    feature_extractor = info.get_feature_extractor(
        frontend_conf={"lfr_m": 5, "lfr_n": 6, "n_mels": 80}
    )

    assert feature_extractor.frontend_conf["lfr_m"] == 5


def test_funasr_rejects_request_lfr_m_above_configured_frontend() -> None:
    ctx = _FakeProcessingContext({"lfr_m": 7, "lfr_n": 6, "n_mels": 80})
    info = FunASRProcessingInfo(ctx)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="may not exceed the configured frontend"):
        info.get_feature_extractor(
            frontend_conf={"lfr_m": 1000, "lfr_n": 1, "n_mels": 80}
        )

    assert ctx.processor_calls == [{}]


def test_funasr_preserves_explicit_server_lfr_m_budget() -> None:
    ctx = _FakeProcessingContext({"lfr_m": 9, "lfr_n": 6, "n_mels": 80})
    info = FunASRProcessingInfo(ctx)  # type: ignore[arg-type]

    feature_extractor = info.get_feature_extractor(
        frontend_conf={"lfr_m": 9, "lfr_n": 6, "n_mels": 80}
    )

    assert feature_extractor.frontend_conf["lfr_m"] == 9

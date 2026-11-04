# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import sys
import types

import pytest

from vllm.tokenizers import fastokens as fastokens_patch


class _FakeFastokens(types.ModuleType):
    patched = False

    def patch_transformers(self) -> None:
        self.patched = True


@pytest.fixture
def fake_fastokens(monkeypatch):
    module = _FakeFastokens("fastokens")
    monkeypatch.setitem(sys.modules, "fastokens", module)
    return module


@pytest.mark.parametrize("installed", ["0.2.0", "0.3.4"])
def test_fastokens_before_0_3_5_is_rejected(monkeypatch, fake_fastokens, installed):
    # Earlier releases can return wrong token ids (fastokens#78, fixed in 0.3.4)
    # or deadlock in encode_batch (fastokens#84, fixed in 0.3.5).
    monkeypatch.setattr(fastokens_patch, "version", lambda _: installed)
    with pytest.raises(
        ImportError, match=rf"fastokens >= 0\.3\.5 .*\(found {installed}\)"
    ):
        fastokens_patch.apply_fastokens_patch()
    assert not fake_fastokens.patched


def test_fastokens_0_3_5_is_applied(monkeypatch, fake_fastokens):
    monkeypatch.setattr(fastokens_patch, "version", lambda _: "0.3.5")
    fastokens_patch.apply_fastokens_patch()
    assert fake_fastokens.patched

# SPDX-License-Identifier: Apache-2.0

import io
import platform

from vllm.distributed import nixl_utils as nu


def test_cpu_no_avx_skips_nixl(monkeypatch):
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(
        "builtins.open",
        lambda path, *a, **k: io.StringIO("flags\t: fpu vme de\n"),
    )
    nu.__dict__.pop("NixlWrapper", None)
    assert nu._load_nixl_attr("NixlWrapper") is None

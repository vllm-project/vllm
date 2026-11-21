# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run as a script so spawn imports the entry point as __mp_main__."""

import multiprocessing
import pickle

import pytest

from vllm.v1.attention.backends.registry import (
    AttentionBackendEnum,
    MambaAttentionBackendEnum,
    register_backend,
)


@register_backend("ENTRYPOINT")
class Backend:
    pass


path = f"{Backend.__module__}.Backend"
register_backend("ENTRYPOINT", path, is_mamba=True)


def check(decorated, direct, late, late_direct, conflicting):
    assert __name__ == "__mp_main__"
    assert decorated.get_class() is direct.get_class() is Backend
    register_backend("LATE")(Backend)
    register_backend("LATE", path, is_mamba=True)
    assert late.get_class() is late_direct.get_class() is Backend
    for payload in conflicting:
        with pytest.raises(ValueError, match="Conflicting registration"):
            pickle.loads(payload)


if __name__ == "__main__":
    register_backend("LATE")(Backend)
    register_backend("LATE", path, is_mamba=True)
    member = AttentionBackendEnum["ENTRYPOINT"]
    conflicting = []
    for other in ("__main__.OtherBackend", "other_module.Backend"):
        register_backend(member, other)
        conflicting.append(pickle.dumps(member))
    register_backend(member, path)

    child = multiprocessing.get_context("spawn").Process(
        target=check,
        args=(
            member,
            MambaAttentionBackendEnum["ENTRYPOINT"],
            AttentionBackendEnum["LATE"],
            MambaAttentionBackendEnum["LATE"],
            conflicting,
        ),
    )
    child.start()
    child.join()
    raise SystemExit(child.exitcode)

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A store that runs out of space partway through a block raises the real
errno (EFBIG here, ENOSPC on a full filesystem), not EIO or no errno."""

import errno
import os
import subprocess
import sys
import textwrap

import pytest

import vllm.v1.kv_offload.tiering.fs.io as io_mod

_BLOCK = 192 * 1024


def _paths_in(directory) -> list[str]:
    return sorted(os.listdir(directory))


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs POSIX rlimits")
@pytest.mark.parametrize("use_c_ext", [True, False])
def test_store_hitting_file_size_limit_reports_efbig(tmp_path, use_c_ext):
    """RLIMIT_FSIZE below the block size makes write(2) return a short count;
    the next write fails with EFBIG. Runs in a subprocess to scope the rlimit."""
    if use_c_ext and not io_mod._HAS_FSIO_C:
        pytest.skip("fs_io_C extension not built")
    script = textwrap.dedent(
        f"""
        import os, resource, signal, sys
        import vllm.v1.kv_offload.tiering.fs.io as io_mod
        io_mod._HAS_FSIO_C = {use_c_ext!r}
        signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
        resource.setrlimit(resource.RLIMIT_FSIZE, (64 * 1024, 64 * 1024))
        buf = memoryview(bytearray({_BLOCK}))
        try:
            io_mod.batch_store_block(
                [sys.argv[1]], buf, [0], {_BLOCK}, use_o_direct=False
            )
        except OSError as exc:
            print(exc.errno)
        else:
            print("stored")
        """
    )
    dest = str(tmp_path / "block.bin")
    out = subprocess.run(
        [sys.executable, "-c", script, dest],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )

    assert out.stdout.strip().splitlines()[-1] == str(errno.EFBIG), out.stderr
    assert _paths_in(tmp_path) == []

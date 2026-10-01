"""Tests for suppress_stdout() fd handling (issues #59268, #59269)."""
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import vllm

REPO_ROOT = Path(vllm.__file__).resolve().parent.parent


def _run(snippet: str) -> subprocess.CompletedProcess:
    bootstrap = (
        "import os, sys\n"
        f"sys.path.insert(0, {str(REPO_ROOT)!r})\n"
    )
    code = bootstrap + textwrap.dedent(snippet)
    return subprocess.run([sys.executable, "-c", code],
                          capture_output=True,
                          text=True,
                          timeout=60)


def test_suppresses_stdout_when_bound_to_stdout_fd():
    """Regression: fd-1 output from C libraries is suppressed while inside."""
    p = _run("""
        import os
        from vllm.utils.system_utils import suppress_stdout
        with suppress_stdout():
            os.write(1, b"SHOULD_BE_SUPPRESSED\\n")
        os.write(1, b"VISIBLE_AFTER\\n")
    """)
    assert p.returncode == 0, p.stderr
    assert "SHOULD_BE_SUPPRESSED" not in p.stdout
    assert "VISIBLE_AFTER" in p.stdout


def test_no_crash_when_stdout_has_no_fileno():
    """Issue #59268: sys.stdout without an fd (StringIO, Jupyter) must not crash."""
    p = _run("""
        import contextlib
        import io
        import os
        from vllm.utils.system_utils import suppress_stdout
        with contextlib.redirect_stdout(io.StringIO()):
            with suppress_stdout():
                os.write(1, b"HI\\n")
        os.write(1, b"OK_NO_CRASH\\n")
    """)
    assert p.returncode == 0, p.stderr
    assert "OK_NO_CRASH" in p.stdout


def test_stderr_preserved_when_stdout_bound_to_stderr():
    """Issue #59269: when sys.stdout is bound to stderr's fd, stderr survives."""
    p = _run("""
        import os
        import sys
        sys.stdout = sys.stderr
        from vllm.utils.system_utils import suppress_stdout
        with suppress_stdout():
            os.write(2, b"CRITICAL_KEPT\\n")
    """)
    assert p.returncode == 0, p.stderr
    assert "CRITICAL_KEPT" in p.stderr

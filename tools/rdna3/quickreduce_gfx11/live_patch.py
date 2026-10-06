# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Wire the rebuilt QuickReduce into the installed vLLM.

Three edits, all reversible from the .p2pbak copies:
  1. quick_all_reduce.py loads libqr_rdna3.so (the gfx11 buffer-descriptor fix)
     and routes the qr_* ops to it.
  2. the gfx94/gfx95 arch gate also admits gfx11.
  3. custom_all_reduce.py stays disabled on gfx11: it is a pull collective and
     this root complex returns zeros for peer reads.
"""

import py_compile
import shutil

BASE = "/usr/local/lib/python3.12/dist-packages/vllm/distributed/device_communicators"
QR = f"{BASE}/quick_all_reduce.py"
CA = f"{BASE}/custom_all_reduce.py"
LIB = "/tmp/qr_build/libqr_rdna3.so"

for f in (QR, CA):
    shutil.copy(f, f + ".p2pbak")

with open(QR) as f:
    s = f.read()

anchor = "logger = init_logger(__name__)\n"
shim = '''logger = init_logger(__name__)

# The in-tree QuickReduce builds its buffer resource descriptor with the gfx9
# word3 encoding, so every buffer_load returns zero on RDNA and the collective
# reduces garbage. libqr_rdna3.so is the same kernel rebuilt with the RDNA
# encoding; point VLLM_QR_RDNA3_LIB at it to use it instead.
import os as _os

_qr_lib = _os.environ.get("VLLM_QR_RDNA3_LIB")
if _qr_lib:
    torch.ops.load_library(_qr_lib)

    class _QrOps:
        """Routes qr_* to the rebuilt library, everything else to vLLM."""

        def __init__(self, inner, rebuilt):
            self._inner, self._rebuilt = inner, rebuilt

        def __getattr__(self, name):
            if name.startswith("qr_") or name == "init_custom_qr":
                return getattr(self._rebuilt, name)
            return getattr(self._inner, name)

    ops = _QrOps(ops, torch.ops._qr_rdna3)
    logger.info("QuickReduce: using rebuilt RDNA3 kernels from %s", _qr_lib)
'''
assert anchor in s
s = s.replace(anchor, shim, 1)

old = 'supported_archs = ["gfx94", "gfx95"]'
new = 'supported_archs = ["gfx94", "gfx95", "gfx11"]'
assert old in s
s = s.replace(old, new)
with open(QR, "w") as f:
    f.write(s)
print("QR: libreria reconstruida + gfx11 permitido")

with open(CA) as f:
    s = f.read()
anchor = """        # test P2P capability, this checks software/cudaruntime support
        # this is expensive to compute at the first time
        # then we cache the result"""
guard = """\
        # A desktop root complex forwards peer WRITES but answers peer
        # READS with zeros, and this collective is a pull: every rank reads
        # every peer's buffer. Leave it off; QuickReduce pushes instead.
        if current_platform.is_rocm() and "gfx11" in getattr(
            torch.cuda.get_device_properties(0), "gcnArchName", ""
        ):
            logger.warning(
                "Custom allreduce disabled on gfx11: peer reads are not "
                "routed on this platform."
            )
            return

"""
assert anchor in s and guard not in s
s = s.replace(anchor, guard + anchor)
with open(CA, "w") as f:
    f.write(s)
print("CA: apagado en gfx11")

for f in (QR, CA):
    py_compile.compile(f, doraise=True)
print("los dos compilan")

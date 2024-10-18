import triton
import triton.language as tl
from packaging import version

TRITON3 = version.parse(triton.__version__) >= version.parse("3.0.0")

if TRITON3:

    @triton.jit
    def softplus(dt):
        return tl.where(dt <= 20.0, tl.math.log(tl.math.exp(dt) + 1), dt)
else:

    @triton.jit
    def softplus(dt):
        return tl.where(dt <= 20.0, tl.math.log1p(tl.exp(dt)), dt)

from std.atomic import Atomic
from std.gpu import lane_id
from std.gpu.memory import AddressSpace
from std.sys._assembly import inlined_assembly

from mojo.common import dtype_in


@always_inline
def s_wait_lds_ops():
    inlined_assembly[
        "s_waitcnt lgkmcnt(0)", NoneType, constraints="", has_side_effect=True
    ]()


@always_inline
def wait_for_counter(
    counter: UnsafePointer[
        mut=True, Int32, _, address_space=AddressSpace.SHARED
    ],
    threshold: Int32,
):
    while Atomic.load(counter) < threshold:
        inlined_assembly[
            "s_sleep 0", NoneType, constraints="", has_side_effect=True
        ]()


@always_inline
def increment_counter_if_first_lane(
    counter: UnsafePointer[
        mut=True, Int32, _, address_space=AddressSpace.SHARED
    ]
):
    if lane_id() == 0:
        _ = Atomic.fetch_add(counter, Int32(1))


@always_inline
def wait_for_ring_stage_stores():
    # RDNA 3.5 tracks indexed LDS operations with LGKMcnt. Wait before
    # publishing the stage counter so consumers cannot observe partial tiles.
    s_wait_lds_ops()


@always_inline
def stage_ptr[
    stride: Int,
](
    base: UnsafePointer[
        Scalar[dtype_in], MutAnyOrigin, address_space=AddressSpace.SHARED
    ],
    stage: Int,
) -> UnsafePointer[
    Scalar[dtype_in], MutAnyOrigin, address_space=AddressSpace.SHARED
]:
    return base + stage * stride

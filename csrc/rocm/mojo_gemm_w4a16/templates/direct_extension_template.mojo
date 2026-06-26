from std.gpu.host import DeviceContext
from std.math import ceildiv
from std.memory import OpaquePointer, UnsafePointer
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder

from layout import TileTensor

from common import (
    ALayout,
    BM,
    BN,
    BK,
    CLayout,
    COMPUTE_WARPS,
    SPLITK_BLOCK_K,
    SPLITK_ROWS_PER_CTA,
    SPLITK_PARTITIONS,
    SPLITK_THREADS,
    PRODUCTION_TOTAL_THREADS,
    MAX_M,
    MAX_N,
    PartialLayout,
    QZerosLayout,
    QWeightKPackedLayout,
    QWeightLayout,
    ScalesLayout,
    a_layout,
    c_layout,
    dtype_acc,
    dtype_in,
    dtype_out,
    dtype_q,
    partial_layout,
    qzeros_layout,
    qweight_kpacked_layout,
    qweight_layout,
    scales_layout,
)
__REDUCE_IMPORT__
__KERNEL_IMPORT__


comptime active_kernel = __ACTIVE_KERNEL__
__REDUCE_KERNEL_DEF__
comptime NEED_PARTIAL = __NEED_PARTIAL__
comptime PARTIAL_COUNT = (
    SPLITK_PARTITIONS * MAX_M * MAX_N if NEED_PARTIAL else 1
)


struct W4A16Runner(Movable, Writable):
    var ctx: DeviceContext
    var compiled_kernel: type_of(DeviceContext().compile_function[active_kernel]())
__REDUCE_FIELD__

    def __init__(out self, _stream_addr: Int) raises:
        self.ctx = DeviceContext()
        self.compiled_kernel = self.ctx.compile_function[active_kernel]()
__REDUCE_INIT__

    def write_to(self, mut writer: Some[Writer]):
        writer.write("W4A16Runner")

    def write_repr_to(self, mut writer: Some[Writer]):
        writer.write("W4A16Runner")

    @staticmethod
    def py_init(out self: W4A16Runner, args: PythonObject, kwargs: PythonObject) raises:
        if len(args) != 1:
            raise Error("W4A16Runner expects one stream address argument")
        self = Self(Int(py=args[0]))

    @staticmethod
    def gemm(
        self_ptr: UnsafePointer[W4A16Runner, MutAnyOrigin],
        stream_addr_obj: PythonObject,
        c_obj: PythonObject,
        a_obj: PythonObject,
        qweight_obj: PythonObject,
        qweight_kpacked_obj: PythonObject,
        qzeros_obj: PythonObject,
        scales_obj: PythonObject,
        partial_obj: PythonObject,
    ) raises -> PythonObject:
        var m = Int(py=a_obj.shape[0])
        var k = Int(py=a_obj.shape[1])
        var n = Int(py=scales_obj.shape[1])

        var c_ptr = UnsafePointer[Scalar[dtype_out], MutExternalOrigin](
            unsafe_from_address=Int(py=c_obj.data_ptr())
        )
        var a_ptr = UnsafePointer[Scalar[dtype_in], ImmutExternalOrigin](
            unsafe_from_address=Int(py=a_obj.data_ptr())
        )
        var qweight_ptr = UnsafePointer[Scalar[dtype_q], ImmutExternalOrigin](
            unsafe_from_address=Int(py=qweight_obj.data_ptr())
        )
        var qweight_kpacked_ptr = UnsafePointer[
            Scalar[dtype_q], ImmutExternalOrigin
        ](unsafe_from_address=Int(py=qweight_kpacked_obj.data_ptr()))
        var qzeros_ptr = UnsafePointer[Scalar[dtype_q], ImmutExternalOrigin](
            unsafe_from_address=Int(py=qzeros_obj.data_ptr())
        )
        var scales_ptr = UnsafePointer[Scalar[dtype_in], ImmutExternalOrigin](
            unsafe_from_address=Int(py=scales_obj.data_ptr())
        )
        var partial_ptr = UnsafePointer[Scalar[dtype_acc], MutAnyOrigin](
            unsafe_from_address=Int(py=partial_obj.data_ptr())
        )
        var partial_read_ptr = UnsafePointer[
            Scalar[dtype_acc], ImmutAnyOrigin
        ](unsafe_from_address=Int(py=partial_obj.data_ptr()))

        var c_tt = TileTensor[
            mut=True, dtype_out, CLayout, MutExternalOrigin
        ](c_ptr, c_layout)
        var a_tt = TileTensor[
            dtype_in, ALayout, ImmutExternalOrigin
        ](a_ptr, a_layout)
        var qweight_tt = TileTensor[
            dtype_q, QWeightLayout, ImmutExternalOrigin
        ](qweight_ptr, qweight_layout)
        var qweight_kpacked_tt = TileTensor[
            dtype_q, QWeightKPackedLayout, ImmutExternalOrigin
        ](qweight_kpacked_ptr, qweight_kpacked_layout)
        var qzeros_tt = TileTensor[
            dtype_q, QZerosLayout, ImmutExternalOrigin
        ](qzeros_ptr, qzeros_layout)
        var scales_tt = TileTensor[
            dtype_in, ScalesLayout, ImmutExternalOrigin
        ](scales_ptr, scales_layout)
        var partial_tt = TileTensor[
            mut=True, dtype_acc, PartialLayout, MutAnyOrigin
        ](partial_ptr, partial_layout)
        var partial_read_tt = TileTensor[
            dtype_acc, PartialLayout, ImmutAnyOrigin
        ](partial_read_ptr, partial_layout)

        var external = OpaquePointer[MutExternalOrigin](
            unsafe_from_address=Int(py=stream_addr_obj)
        )
        var stream = self_ptr[].ctx.create_external_stream(external)
__PY_LAUNCH_BODY__
        return PythonObject(None)


@always_inline
def _gemm_native(
    self_ptr: UnsafePointer[W4A16Runner, MutExternalOrigin],
    stream_addr: Int,
    c_addr: Int,
    a_addr: Int,
    qweight_addr: Int,
    qweight_kpacked_addr: Int,
    qzeros_addr: Int,
    scales_addr: Int,
    partial_addr: Int,
    m: Int,
    n: Int,
    k: Int,
) raises:
    var c_ptr = UnsafePointer[Scalar[dtype_out], MutExternalOrigin](
        unsafe_from_address=c_addr
    )
    var a_ptr = UnsafePointer[Scalar[dtype_in], ImmutExternalOrigin](
        unsafe_from_address=a_addr
    )
    var qweight_ptr = UnsafePointer[Scalar[dtype_q], ImmutExternalOrigin](
        unsafe_from_address=qweight_addr
    )
    var qweight_kpacked_ptr = UnsafePointer[
        Scalar[dtype_q], ImmutExternalOrigin
    ](unsafe_from_address=qweight_kpacked_addr)
    var qzeros_ptr = UnsafePointer[Scalar[dtype_q], ImmutExternalOrigin](
        unsafe_from_address=qzeros_addr
    )
    var scales_ptr = UnsafePointer[Scalar[dtype_in], ImmutExternalOrigin](
        unsafe_from_address=scales_addr
    )
    var partial_ptr = UnsafePointer[Scalar[dtype_acc], MutAnyOrigin](
        unsafe_from_address=partial_addr
    )
    var partial_read_ptr = UnsafePointer[
        Scalar[dtype_acc], ImmutAnyOrigin
    ](unsafe_from_address=partial_addr)

    var c_tt = TileTensor[
        mut=True, dtype_out, CLayout, MutExternalOrigin
    ](c_ptr, c_layout)
    var a_tt = TileTensor[dtype_in, ALayout, ImmutExternalOrigin](
        a_ptr, a_layout
    )
    var qweight_tt = TileTensor[
        dtype_q, QWeightLayout, ImmutExternalOrigin
    ](qweight_ptr, qweight_layout)
    var qweight_kpacked_tt = TileTensor[
        dtype_q, QWeightKPackedLayout, ImmutExternalOrigin
    ](qweight_kpacked_ptr, qweight_kpacked_layout)
    var qzeros_tt = TileTensor[
        dtype_q, QZerosLayout, ImmutExternalOrigin
    ](qzeros_ptr, qzeros_layout)
    var scales_tt = TileTensor[
        dtype_in, ScalesLayout, ImmutExternalOrigin
    ](scales_ptr, scales_layout)
    var partial_tt = TileTensor[
        mut=True, dtype_acc, PartialLayout, MutAnyOrigin
    ](partial_ptr, partial_layout)
    var partial_read_tt = TileTensor[
        dtype_acc, PartialLayout, ImmutAnyOrigin
    ](partial_read_ptr, partial_layout)

    var external = OpaquePointer[MutExternalOrigin](
        unsafe_from_address=stream_addr
    )
    var stream = self_ptr[].ctx.create_external_stream(external)
__NATIVE_LAUNCH_BODY__


@export
def mojo_w4a16_runner_create(
    stream_addr: Int
) -> UnsafePointer[W4A16Runner, MutExternalOrigin]:
    try:
        var runner_ptr = alloc[W4A16Runner](1)
        runner_ptr.init_pointee_move(W4A16Runner(stream_addr))
        return runner_ptr
    except e:
        abort(String("failed to create native W4A16Runner: ", e))


@export
def mojo_w4a16_runner_launch(
    self_ptr: UnsafePointer[W4A16Runner, MutExternalOrigin],
    stream_addr: Int,
    c_addr: Int,
    a_addr: Int,
    qweight_addr: Int,
    qweight_kpacked_addr: Int,
    qzeros_addr: Int,
    scales_addr: Int,
    partial_addr: Int,
    m: Int,
    n: Int,
    k: Int,
):
    try:
        _gemm_native(
            self_ptr,
            stream_addr,
            c_addr,
            a_addr,
            qweight_addr,
            qweight_kpacked_addr,
            qzeros_addr,
            scales_addr,
            partial_addr,
            m,
            n,
            k,
        )
    except e:
        abort(String("failed to launch native W4A16Runner: ", e))


@export
def PyInit___MODULE_NAME__() -> PythonObject:
    try:
        var m = PythonModuleBuilder("__MODULE_NAME__")
        _ = (
            m.add_type[W4A16Runner]("W4A16Runner")
            .def_py_init[W4A16Runner.py_init]()
        )
        return m.finalize()
    except e:
        abort(String("failed to create __MODULE_NAME__: ", e))

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""This file is a pure Python wrapper for the Level Zero (oneAPI L0) library.
It avoids the need to compile a separate shared library, and is convenient for
use when we just need to call a few functions.

Unlike CUDA/HIP, Level Zero does not expose a flat, stateless C runtime such as
``cudart``. Memory allocation, fills and copies are all bound to an explicit
``ze_context_handle_t`` / ``ze_device_handle_t`` and require a command list.
This wrapper therefore keeps the CUDA-like surface (``xpuSetDevice``,
``xpuMalloc``, ``xpuMemset``, ``xpuMemcpy``, ``xpuIpcGetMemHandle``,
``xpuIpcOpenMemHandle`` ...) while internally managing the driver, device,
context and an immediate (synchronous) command list.

For the original Level Zero definitions, please check
https://oneapi-src.github.io/level-zero-spec/level-zero/latest/index.html
"""

import ctypes
from dataclasses import dataclass
from typing import Any

# this line makes it possible to directly load `libze_loader.so` using `ctypes`
import torch  # noqa

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.utils.system_utils import find_loaded_library

logger = init_logger(__name__)

# === Level Zero types and constants ===
# ze_result_t is an enum (uint32); ZE_RESULT_SUCCESS == 0.
ze_result_t = ctypes.c_uint32

# All Level Zero object handles are opaque pointers.
ze_driver_handle_t = ctypes.c_void_p
ze_device_handle_t = ctypes.c_void_p
ze_context_handle_t = ctypes.c_void_p
ze_command_list_handle_t = ctypes.c_void_p
ze_event_handle_t = ctypes.c_void_p

# #define ZE_MAX_IPC_HANDLE_SIZE 64
ZE_MAX_IPC_HANDLE_SIZE = 64

# ze_init_flag_t
ZE_INIT_FLAG_GPU_ONLY = 1

# ze_structure_type_t
ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC = 0x15
ZE_STRUCTURE_TYPE_CONTEXT_DESC = 0xD
ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC = 0xE

# ze_command_queue_mode_t
ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS = 1
# ze_command_queue_priority_t
ZE_COMMAND_QUEUE_PRIORITY_NORMAL = 0

# wait forever, used as the timeout for host synchronization (UINT64_MAX)
ZE_TIMEOUT_INFINITE = 0xFFFFFFFFFFFFFFFF

# human readable names for the ze_result_t codes we are likely to hit
_ZE_RESULT_NAMES: dict[int, str] = {
    0x00000000: "ZE_RESULT_SUCCESS",
    0x00000001: "ZE_RESULT_NOT_READY",
    0x70000001: "ZE_RESULT_ERROR_DEVICE_LOST",
    0x70000002: "ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY",
    0x70000003: "ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY",
    0x70000004: "ZE_RESULT_ERROR_MODULE_BUILD_FAILURE",
    0x78000001: "ZE_RESULT_ERROR_UNINITIALIZED",
    0x78000002: "ZE_RESULT_ERROR_UNSUPPORTED_VERSION",
    0x78000003: "ZE_RESULT_ERROR_UNSUPPORTED_FEATURE",
    0x78000004: "ZE_RESULT_ERROR_INVALID_ARGUMENT",
    0x78000005: "ZE_RESULT_ERROR_INVALID_NULL_HANDLE",
    0x78000006: "ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE",
    0x78000007: "ZE_RESULT_ERROR_INVALID_NULL_POINTER",
    0x78000008: "ZE_RESULT_ERROR_INVALID_SIZE",
    0x78000009: "ZE_RESULT_ERROR_UNSUPPORTED_SIZE",
    0x7800000A: "ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT",
    0x7FFFFFFE: "ZE_RESULT_ERROR_UNKNOWN",
}


class ze_ipc_mem_handle_t(ctypes.Structure):
    """struct _ze_ipc_mem_handle_t { char data[ZE_MAX_IPC_HANDLE_SIZE]; }"""

    _fields_ = [("data", ctypes.c_char * ZE_MAX_IPC_HANDLE_SIZE)]


# keep a CUDA-flavored alias so this is a drop-in for cudaIpcMemHandle_t
xpuIpcMemHandle_t = ze_ipc_mem_handle_t


class ze_context_desc_t(ctypes.Structure):
    _fields_ = [
        ("stype", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
    ]


class ze_device_mem_alloc_desc_t(ctypes.Structure):
    _fields_ = [
        ("stype", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
        ("ordinal", ctypes.c_uint32),
    ]


class ze_command_queue_desc_t(ctypes.Structure):
    _fields_ = [
        ("stype", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("ordinal", ctypes.c_uint32),
        ("index", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("mode", ctypes.c_uint32),
        ("priority", ctypes.c_uint32),
    ]


@dataclass
class Function:
    name: str
    restype: Any
    argtypes: list[Any]


class XpuRTLibrary:
    exported_functions = [
        # ze_result_t zeInit(ze_init_flags_t flags)
        Function("zeInit", ze_result_t, [ctypes.c_uint32]),
        # ze_result_t zeDriverGet(uint32_t* pCount,
        #                         ze_driver_handle_t* phDrivers)
        Function(
            "zeDriverGet",
            ze_result_t,
            [ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ze_driver_handle_t)],
        ),
        # ze_result_t zeDeviceGet(ze_driver_handle_t hDriver, uint32_t* pCount,
        #                         ze_device_handle_t* phDevices)
        Function(
            "zeDeviceGet",
            ze_result_t,
            [
                ze_driver_handle_t,
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.POINTER(ze_device_handle_t),
            ],
        ),
        # ze_result_t zeContextCreate(ze_driver_handle_t hDriver,
        #                             const ze_context_desc_t* desc,
        #                             ze_context_handle_t* phContext)
        Function(
            "zeContextCreate",
            ze_result_t,
            [
                ze_driver_handle_t,
                ctypes.POINTER(ze_context_desc_t),
                ctypes.POINTER(ze_context_handle_t),
            ],
        ),
        # ze_result_t zeContextDestroy(ze_context_handle_t hContext)
        Function("zeContextDestroy", ze_result_t, [ze_context_handle_t]),
        # ze_result_t zeCommandListCreateImmediate(
        #     ze_context_handle_t hContext, ze_device_handle_t hDevice,
        #     const ze_command_queue_desc_t* altdesc,
        #     ze_command_list_handle_t* phCommandList)
        Function(
            "zeCommandListCreateImmediate",
            ze_result_t,
            [
                ze_context_handle_t,
                ze_device_handle_t,
                ctypes.POINTER(ze_command_queue_desc_t),
                ctypes.POINTER(ze_command_list_handle_t),
            ],
        ),
        # ze_result_t zeCommandListDestroy(
        #     ze_command_list_handle_t hCommandList)
        Function("zeCommandListDestroy", ze_result_t, [ze_command_list_handle_t]),
        # ze_result_t zeCommandListHostSynchronize(
        #     ze_command_list_handle_t hCommandList, uint64_t timeout)
        Function(
            "zeCommandListHostSynchronize",
            ze_result_t,
            [ze_command_list_handle_t, ctypes.c_uint64],
        ),
        # ze_result_t zeMemAllocDevice(
        #     ze_context_handle_t hContext,
        #     const ze_device_mem_alloc_desc_t* device_desc, size_t size,
        #     size_t alignment, ze_device_handle_t hDevice, void** pptr)
        Function(
            "zeMemAllocDevice",
            ze_result_t,
            [
                ze_context_handle_t,
                ctypes.POINTER(ze_device_mem_alloc_desc_t),
                ctypes.c_size_t,
                ctypes.c_size_t,
                ze_device_handle_t,
                ctypes.POINTER(ctypes.c_void_p),
            ],
        ),
        # ze_result_t zeMemFree(ze_context_handle_t hContext, void* ptr)
        Function("zeMemFree", ze_result_t, [ze_context_handle_t, ctypes.c_void_p]),
        # ze_result_t zeCommandListAppendMemoryFill(
        #     ze_command_list_handle_t hCommandList, void* ptr,
        #     const void* pattern, size_t pattern_size, size_t size,
        #     ze_event_handle_t hSignalEvent, uint32_t numWaitEvents,
        #     ze_event_handle_t* phWaitEvents)
        Function(
            "zeCommandListAppendMemoryFill",
            ze_result_t,
            [
                ze_command_list_handle_t,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_size_t,
                ze_event_handle_t,
                ctypes.c_uint32,
                ctypes.POINTER(ze_event_handle_t),
            ],
        ),
        # ze_result_t zeCommandListAppendMemoryCopy(
        #     ze_command_list_handle_t hCommandList, void* dstptr,
        #     const void* srcptr, size_t size, ze_event_handle_t hSignalEvent,
        #     uint32_t numWaitEvents, ze_event_handle_t* phWaitEvents)
        Function(
            "zeCommandListAppendMemoryCopy",
            ze_result_t,
            [
                ze_command_list_handle_t,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_size_t,
                ze_event_handle_t,
                ctypes.c_uint32,
                ctypes.POINTER(ze_event_handle_t),
            ],
        ),
        # ze_result_t zeMemGetIpcHandle(ze_context_handle_t hContext,
        #     const void* ptr, ze_ipc_mem_handle_t* pIpcHandle)
        Function(
            "zeMemGetIpcHandle",
            ze_result_t,
            [
                ze_context_handle_t,
                ctypes.c_void_p,
                ctypes.POINTER(ze_ipc_mem_handle_t),
            ],
        ),
        # ze_result_t zeMemOpenIpcHandle(ze_context_handle_t hContext,
        #     ze_device_handle_t hDevice, ze_ipc_mem_handle_t handle,
        #     ze_ipc_memory_flags_t flags, void** pptr)
        Function(
            "zeMemOpenIpcHandle",
            ze_result_t,
            [
                ze_context_handle_t,
                ze_device_handle_t,
                ze_ipc_mem_handle_t,
                ctypes.c_uint32,
                ctypes.POINTER(ctypes.c_void_p),
            ],
        ),
        # ze_result_t zeMemCloseIpcHandle(ze_context_handle_t hContext,
        #     const void* ptr)
        Function(
            "zeMemCloseIpcHandle",
            ze_result_t,
            [ze_context_handle_t, ctypes.c_void_p],
        ),
    ]

    # class attribute to store the mapping from the path to the library
    # to avoid loading the same library multiple times
    path_to_library_cache: dict[str, Any] = {}

    # class attribute to store the mapping from library path
    #  to the corresponding dictionary
    path_to_dict_mapping: dict[str, dict[str, Any]] = {}

    def __init__(self, so_file: str | None = None):
        if so_file is None:
            so_file = (
                find_loaded_library("libze_loader")
                or envs.VLLM_XPURT_SO_PATH  # fallback to env var
                or "libze_loader.so.1"
            )
            assert so_file is not None, (
                "libze_loader is not loaded in the current process, "
                "try setting VLLM_XPURT_SO_PATH"
            )
        if so_file not in XpuRTLibrary.path_to_library_cache:
            lib = ctypes.CDLL(so_file)
            XpuRTLibrary.path_to_library_cache[so_file] = lib
        self.lib = XpuRTLibrary.path_to_library_cache[so_file]

        if so_file not in XpuRTLibrary.path_to_dict_mapping:
            _funcs = {}
            for func in XpuRTLibrary.exported_functions:
                f = getattr(self.lib, func.name)
                f.restype = func.restype
                f.argtypes = func.argtypes
                _funcs[func.name] = f
            XpuRTLibrary.path_to_dict_mapping[so_file] = _funcs
        self.funcs = XpuRTLibrary.path_to_dict_mapping[so_file]

        # Level Zero state managed by this wrapper.
        self._driver: ze_driver_handle_t | None = None
        # all devices exposed by the selected driver
        self._devices: list[ze_device_handle_t] = []
        # currently selected device ordinal
        self._device_ordinal: int = 0
        # lazily created per-device (context, immediate command list)
        self._device_state: dict[
            int, tuple[ze_context_handle_t, ze_command_list_handle_t]
        ] = {}

        self._init_driver()

    # === internal helpers ===

    def XPU_CHECK(self, result: int) -> None:
        if result != 0:
            name = _ZE_RESULT_NAMES.get(result, "UNKNOWN")
            raise RuntimeError(f"Level Zero error: {name} (0x{result:08x})")

    def _init_driver(self) -> None:
        self.XPU_CHECK(self.funcs["zeInit"](ZE_INIT_FLAG_GPU_ONLY))

        # pick the first driver
        driver_count = ctypes.c_uint32(1)
        driver = ze_driver_handle_t()
        self.XPU_CHECK(
            self.funcs["zeDriverGet"](ctypes.byref(driver_count), ctypes.byref(driver))
        )
        assert driver_count.value >= 1, "No Level Zero driver found"
        self._driver = driver

        # enumerate all devices of that driver
        device_count = ctypes.c_uint32(0)
        self.XPU_CHECK(
            self.funcs["zeDeviceGet"](self._driver, ctypes.byref(device_count), None)
        )
        assert device_count.value >= 1, "No Level Zero device found"
        devices = (ze_device_handle_t * device_count.value)()
        self.XPU_CHECK(
            self.funcs["zeDeviceGet"](self._driver, ctypes.byref(device_count), devices)
        )
        self._devices = [
            ze_device_handle_t(devices[i]) for i in range(device_count.value)
        ]

    def _get_device_state(
        self, ordinal: int
    ) -> tuple[ze_context_handle_t, ze_device_handle_t, ze_command_list_handle_t]:
        device = self._devices[ordinal]
        if ordinal not in self._device_state:
            # create a context bound to this driver
            ctx_desc = ze_context_desc_t(
                stype=ZE_STRUCTURE_TYPE_CONTEXT_DESC, pNext=None, flags=0
            )
            context = ze_context_handle_t()
            self.XPU_CHECK(
                self.funcs["zeContextCreate"](
                    self._driver, ctypes.byref(ctx_desc), ctypes.byref(context)
                )
            )

            # create a synchronous immediate command list so that memory
            # fill/copy operations block until they complete on return.
            queue_desc = ze_command_queue_desc_t(
                stype=ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
                pNext=None,
                ordinal=0,
                index=0,
                flags=0,
                mode=ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS,
                priority=ZE_COMMAND_QUEUE_PRIORITY_NORMAL,
            )
            cmd_list = ze_command_list_handle_t()
            self.XPU_CHECK(
                self.funcs["zeCommandListCreateImmediate"](
                    context, device, ctypes.byref(queue_desc), ctypes.byref(cmd_list)
                )
            )
            self._device_state[ordinal] = (context, cmd_list)

        context, cmd_list = self._device_state[ordinal]
        return context, device, cmd_list

    @property
    def _current(
        self,
    ) -> tuple[ze_context_handle_t, ze_device_handle_t, ze_command_list_handle_t]:
        return self._get_device_state(self._device_ordinal)

    # === CUDA-flavored public API ===

    def xpuGetErrorString(self, error: int) -> str:
        return _ZE_RESULT_NAMES.get(error, f"UNKNOWN (0x{error:08x})")

    def xpuSetDevice(self, device: int) -> None:
        assert 0 <= device < len(self._devices), (
            f"invalid device ordinal {device}, "
            f"only {len(self._devices)} device(s) available"
        )
        self._device_ordinal = device
        # eagerly create the context / command list for this device
        self._get_device_state(device)

    def xpuDeviceSynchronize(self) -> None:
        _, _, cmd_list = self._current
        self.XPU_CHECK(
            self.funcs["zeCommandListHostSynchronize"](cmd_list, ZE_TIMEOUT_INFINITE)
        )

    def xpuDeviceReset(self) -> None:
        # destroy all per-device state (command lists then contexts)
        for context, cmd_list in self._device_state.values():
            self.funcs["zeCommandListDestroy"](cmd_list)
            self.funcs["zeContextDestroy"](context)
        self._device_state.clear()

    def xpuMalloc(self, size: int) -> ctypes.c_void_p:
        context, device, _ = self._current
        mem_desc = ze_device_mem_alloc_desc_t(
            stype=ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC,
            pNext=None,
            flags=0,
            ordinal=0,
        )
        devPtr = ctypes.c_void_p()
        self.XPU_CHECK(
            self.funcs["zeMemAllocDevice"](
                context, ctypes.byref(mem_desc), size, 64, device, ctypes.byref(devPtr)
            )
        )
        return devPtr

    def xpuFree(self, devPtr: ctypes.c_void_p) -> None:
        context, _, _ = self._current
        self.XPU_CHECK(self.funcs["zeMemFree"](context, devPtr))

    def xpuMemset(self, devPtr: ctypes.c_void_p, value: int, count: int) -> None:
        _, _, cmd_list = self._current
        pattern = ctypes.c_ubyte(value & 0xFF)
        self.XPU_CHECK(
            self.funcs["zeCommandListAppendMemoryFill"](
                cmd_list, devPtr, ctypes.byref(pattern), 1, count, None, 0, None
            )
        )

    def xpuMemcpy(self, dst: ctypes.c_void_p, src: ctypes.c_void_p, count: int) -> None:
        _, _, cmd_list = self._current
        self.XPU_CHECK(
            self.funcs["zeCommandListAppendMemoryCopy"](
                cmd_list, dst, src, count, None, 0, None
            )
        )

    def xpuIpcGetMemHandle(self, devPtr: ctypes.c_void_p) -> ze_ipc_mem_handle_t:
        context, _, _ = self._current
        handle = ze_ipc_mem_handle_t()
        self.XPU_CHECK(
            self.funcs["zeMemGetIpcHandle"](context, devPtr, ctypes.byref(handle))
        )
        return handle

    def xpuIpcOpenMemHandle(self, handle: ze_ipc_mem_handle_t) -> ctypes.c_void_p:
        context, device, _ = self._current
        devPtr = ctypes.c_void_p()
        self.XPU_CHECK(
            self.funcs["zeMemOpenIpcHandle"](
                context, device, handle, 0, ctypes.byref(devPtr)
            )
        )
        return devPtr

    def xpuIpcCloseMemHandle(self, devPtr: ctypes.c_void_p) -> None:
        context, _, _ = self._current
        self.XPU_CHECK(self.funcs["zeMemCloseIpcHandle"](context, devPtr))

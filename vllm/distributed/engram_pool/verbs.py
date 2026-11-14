# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ctypes bindings to libibverbs: just enough for RC QPs and RDMA WRITEs.

Offsets follow the stable LP64 ABI of rdma-core's verbs.h. `ibv_post_send` and
`ibv_poll_cq` are inline in verbs.h, so they are called through the context's
ops table. Only the standard library is imported: the RDMA proxy process loads
this file without vllm or torch.
"""

import ctypes
import os
import struct
from functools import cache

SEND_WR_BYTES = 128
WC_BYTES = 48

# enum ibv_access_flags
_LOCAL_WRITE, _REMOTE_WRITE, _REMOTE_READ = 1, 2, 4
_RELAXED_ORDERING = 1 << 20
# Access of registered memory: read only by this NIC (shards, descriptors);
# written by peers in any order (landing); written by peers in order, and read
# back by them (flags, mailboxes).
LOCAL = _RELAXED_ORDERING
BULK = _LOCAL_WRITE | _REMOTE_WRITE | _RELAXED_ORDERING
FLAGS = _LOCAL_WRITE | _REMOTE_WRITE | _REMOTE_READ
_QPT_RC = 2
_QPS_INIT, _QPS_RTR, _QPS_RTS = 1, 2, 3
# Words of ibv_context (its device pointer, then struct ibv_context_ops) that
# hold the calls verbs.h inlines.
_OPS_POLL_CQ, _OPS_NOTIFY_CQ, _OPS_POST_SEND, _OPS_POST_RECV = 12, 13, 26, 27
MTU_BYTES = {1: 256, 2: 512, 3: 1024, 4: 2048, 5: 4096}

_POST_SEND = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)
)
_NOTIFY_CQ = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_int)
_POLL_CQ = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p
)


@cache
def _lib() -> ctypes.CDLL:
    lib = ctypes.CDLL("libibverbs.so.1", use_errno=True)
    p, i = ctypes.c_void_p, ctypes.c_int
    for name, res, args in (
        ("ibv_get_device_list", p, [ctypes.POINTER(i)]),
        ("ibv_get_device_name", ctypes.c_char_p, [p]),
        ("ibv_open_device", p, [p]),
        ("ibv_alloc_pd", p, [p]),
        ("ibv_query_device", i, [p, p]),
        ("ibv_query_port", i, [p, ctypes.c_uint8, p]),
        ("ibv_query_gid", i, [p, ctypes.c_uint8, i, p]),
        ("ibv_create_cq", p, [p, i, p, p, i]),
        ("ibv_create_comp_channel", p, [p]),
        ("ibv_get_cq_event", i, [p, p, p]),
        ("ibv_ack_cq_events", None, [p, ctypes.c_uint]),
        ("ibv_create_qp", p, [p, p]),
        ("ibv_modify_qp", i, [p, p, i]),
        ("ibv_reg_mr", p, [p, p, ctypes.c_size_t, i]),
        (
            "ibv_reg_dmabuf_mr",
            p,
            [p, ctypes.c_uint64, ctypes.c_size_t, ctypes.c_uint64, i, i],
        ),
        ("ibv_dereg_mr", i, [p]),
    ):
        fn = getattr(lib, name)
        fn.restype, fn.argtypes = res, args
    return lib


def _check(result, what: str):
    if not result:
        err = ctypes.get_errno()
        raise RuntimeError(f"{what} failed: {os.strerror(err)} ({err})")
    return result


def _sysfs(device: str, *parts: str) -> str:
    path = os.path.join("/sys/class/infiniband", device, *parts)
    with open(path) as f:
        return f.read().strip()


def active_devices() -> list[str]:
    """Names of RDMA devices whose port 1 is active."""
    root = "/sys/class/infiniband"
    if not os.path.isdir(root):
        return []
    return sorted(
        name
        for name in os.listdir(root)
        if _sysfs(name, "ports", "1", "state").endswith("ACTIVE")
    )


def device_numa_node(device: str) -> int:
    return int(_sysfs(device, "device", "numa_node"))


def local_cpus(device: str) -> set[int]:
    """CPUs on the device's NUMA node."""
    cpus: set[int] = set()
    for part in _sysfs(device, "device", "local_cpulist").split(","):
        first, _, last = part.partition("-")
        cpus.update(range(int(first), int(last or first) + 1))
    return cpus


def _pick_gid_index(device: str) -> int:
    """RoCE v2 with an IPv4-mapped address if present, else index 0 (IB)."""
    best = 0
    for index in range(16):
        try:
            kind = _sysfs(device, "ports", "1", "gid_attrs", "types", str(index))
            gid = _sysfs(device, "ports", "1", "gids", str(index))
        except OSError:
            continue
        if kind == "RoCE v2" and gid.startswith("0000:0000:0000:0000:0000:ffff"):
            return index
        if kind == "RoCE v2" and int(gid.replace(":", ""), 16):
            best = index
    return best


class Device:
    """One opened RDMA device and protection domain on port 1."""

    def __init__(self, name: str) -> None:
        lib = _lib()
        count = ctypes.c_int()
        devices = _check(lib.ibv_get_device_list(ctypes.byref(count)), "list")
        array = ctypes.cast(devices, ctypes.POINTER(ctypes.c_void_p))
        handle = next(
            (
                array[i]
                for i in range(count.value)
                if lib.ibv_get_device_name(array[i]).decode() == name
            ),
            None,
        )
        if handle is None:
            raise RuntimeError(f"RDMA device {name} not found")
        self.name = name
        self.context = _check(lib.ibv_open_device(handle), f"open {name}")
        self.pd = _check(lib.ibv_alloc_pd(self.context), "ibv_alloc_pd")
        attr = ctypes.create_string_buffer(256)
        if lib.ibv_query_device(self.context, attr):
            raise RuntimeError(f"ibv_query_device({name}) failed")
        self.max_qp_wr = struct.unpack_from("<i", attr, 112)[0]
        self.max_sge = struct.unpack_from("<i", attr, 120)[0]
        self.max_cqe = struct.unpack_from("<i", attr, 132)[0]
        # READs a QP may have in flight, as initiator and as target.
        self.rd_atomic = min(
            struct.unpack_from("<i", attr, 144)[0],
            struct.unpack_from("<i", attr, 156)[0],
        )
        port = ctypes.create_string_buffer(64)
        if lib.ibv_query_port(self.context, 1, port):
            raise RuntimeError(f"ibv_query_port({name}) failed")
        self.mtu = struct.unpack_from("<I", port, 8)[0]
        self.lid = struct.unpack_from("<H", port, 34)[0]
        self.is_ib = port.raw[46] == 1
        self.gid_index = _pick_gid_index(name)
        gid = ctypes.create_string_buffer(16)
        if lib.ibv_query_gid(self.context, 1, self.gid_index, gid):
            raise RuntimeError(f"ibv_query_gid({name}) failed")
        self.gid = gid.raw
        ops = ctypes.cast(self.context, ctypes.POINTER(ctypes.c_void_p))
        self._post = _POST_SEND(ops[_OPS_POST_SEND])
        self._poll = _POLL_CQ(ops[_OPS_POLL_CQ])
        self._post_recv = _POST_SEND(ops[_OPS_POST_RECV])
        self._notify = _NOTIFY_CQ(ops[_OPS_NOTIFY_CQ])
        self._bad = ctypes.c_void_p()
        self._bad_ref = ctypes.byref(self._bad)
        self._event = ctypes.c_void_p(), ctypes.c_void_p()
        self._event_refs = tuple(ctypes.byref(p) for p in self._event)

    def describe(self) -> str:
        kind = "InfiniBand" if self.is_ib else f"RoCE, GID index {self.gid_index}"
        return f"{self.name} ({kind}, MTU {MTU_BYTES.get(self.mtu, 0)})"

    def address(self) -> tuple[bytes, int]:
        return self.gid, self.lid

    def reg_host(
        self, address: int, num_bytes: int, access: int
    ) -> tuple[int, int, int]:
        """Register memory with `access` LOCAL, BULK or FLAGS."""
        mr = _check(
            _lib().ibv_reg_mr(self.pd, address, num_bytes, access),
            f"ibv_reg_mr({num_bytes} B)",
        )
        return mr, *self._keys(mr)

    def reg_dmabuf(
        self, fd: int, address: int, num_bytes: int, access: int
    ) -> tuple[int, int, int]:
        mr = _check(
            _lib().ibv_reg_dmabuf_mr(self.pd, 0, num_bytes, address, fd, access),
            f"ibv_reg_dmabuf_mr({num_bytes} B)",
        )
        return mr, *self._keys(mr)

    @staticmethod
    def _keys(mr: int) -> tuple[int, int]:
        lkey, rkey = struct.unpack_from("<II", ctypes.string_at(mr + 36, 8))
        return lkey, rkey

    def create_cq(self, depth: int, wakeable: bool = False) -> tuple[int, int]:
        """A CQ, and if `wakeable` a completion channel (else 0) whose events
        (`cq_event`) wake a sleeper."""
        channel = 0
        if wakeable:
            channel = _check(_lib().ibv_create_comp_channel(self.context), "channel")
        cq = _check(
            _lib().ibv_create_cq(self.context, depth, None, channel, 0),
            "ibv_create_cq",
        )
        return cq, channel

    def notify_cq(self, cq: int) -> None:
        """Raise one event on the CQ's channel at its next completion."""
        if error := self._notify(cq, 0):
            raise RuntimeError(f"ibv_req_notify_cq failed: {os.strerror(error)}")

    def cq_event(self, channel: int) -> None:
        """Sleep until the next event of `channel` and consume it."""
        lib = _lib()
        _check(lib.ibv_get_cq_event(channel, *self._event_refs) == 0, "get_cq_event")
        lib.ibv_ack_cq_events(self._event[0], 1)

    def post_recv(self, qp: int, wr: int) -> None:
        if error := self._post_recv(qp, wr, self._bad_ref):
            raise RuntimeError(f"ibv_post_recv failed: {os.strerror(error)}")

    def create_qp(
        self,
        send_cq: int,
        recv_cq: int,
        max_send_wr: int,
        max_send_sge: int = 1,
        max_inline: int = 0,
        max_recv_wr: int = 1,
    ) -> tuple[int, int]:
        """An RC QP; it receives only WRITE_WITH_IMM notices. Returns it and
        its number."""
        attr = ctypes.create_string_buffer(64)
        struct.pack_into("<QQ", attr, 8, send_cq, recv_cq)
        struct.pack_into(
            "<IIIII", attr, 32, max_send_wr, max_recv_wr, max_send_sge, 1, max_inline
        )
        struct.pack_into("<i", attr, 52, _QPT_RC)
        qp = _check(_lib().ibv_create_qp(self.pd, attr), "ibv_create_qp")
        return qp, struct.unpack_from("<I", ctypes.string_at(qp + 52, 4))[0]

    def connect(self, qp: int, remote_qpn: int, remote: tuple[bytes, int]) -> None:
        """Move a QP through INIT, RTR and RTS towards `remote_qpn`."""
        lib = _lib()
        attr = ctypes.create_string_buffer(144)
        struct.pack_into("<I", attr, 0, _QPS_INIT)
        struct.pack_into("<i", attr, 32, _REMOTE_WRITE | _REMOTE_READ)
        struct.pack_into("<H", attr, 120, 0)
        struct.pack_into("<B", attr, 129, 1)  # port
        mask = 1 | 8 | 16 | 32  # state, access, pkey, port
        if lib.ibv_modify_qp(qp, attr, mask):
            raise RuntimeError("QP INIT failed")

        attr = ctypes.create_string_buffer(144)
        gid, lid = remote
        struct.pack_into("<IIi", attr, 0, _QPS_RTR, 0, self.mtu)
        struct.pack_into("<II", attr, 20, 0, 0)
        struct.pack_into("<I", attr, 28, remote_qpn)
        ah = 56
        if self.is_ib:
            struct.pack_into("<H", attr, ah + 24, lid)
        else:
            # dgid, sgid_index, hop_limit, then is_global.
            struct.pack_into("<16s4xBB", attr, ah, gid, self.gid_index, 64)
            struct.pack_into("<B", attr, ah + 29, 1)
        struct.pack_into("<B", attr, ah + 30, 1)  # port
        # max_dest_rd_atomic, min_rnr_timer
        struct.pack_into("<BB", attr, 127, self.rd_atomic, 12)
        # state, av, path_mtu, dest_qpn, rq_psn, max_dest_rd_atomic, min_rnr_timer
        mask = 1 | 128 | 256 | (1 << 20) | (1 << 12) | (1 << 17) | (1 << 15)
        if lib.ibv_modify_qp(qp, attr, mask):
            raise RuntimeError(f"QP RTR failed: {os.strerror(ctypes.get_errno())}")

        attr = ctypes.create_string_buffer(144)
        struct.pack_into("<I", attr, 0, _QPS_RTS)
        struct.pack_into("<I", attr, 24, 0)  # sq_psn
        struct.pack_into("<B", attr, 126, self.rd_atomic)  # max_rd_atomic
        struct.pack_into("<BBB", attr, 130, 14, 7, 7)  # timeout, retries, RNR retries
        # state, timeout, retry, rnr_retry, sq_psn, max_qp_rd_atomic
        mask = 1 | 512 | 1024 | 2048 | (1 << 16) | (1 << 13)
        if lib.ibv_modify_qp(qp, attr, mask):
            raise RuntimeError(f"QP RTS failed: {os.strerror(ctypes.get_errno())}")

    @staticmethod
    def mlx5_queue(qp: int) -> tuple[int, int, int, int, int]:
        """An mlx5 QP's send queue for posting without libibverbs: its WQE
        buffer, WQE count, doorbell record and doorbell register (BlueFlame
        register and size), from mlx5dv_init_obj."""
        mlx5 = ctypes.CDLL("libmlx5.so.1")
        mlx5.mlx5dv_init_obj.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
        # struct mlx5dv_qp: dbrec, sq {buf, wqe_cnt, stride}, rq {..}, bf {reg,
        # size}; struct mlx5dv_obj starts with qp {in, out}.
        out = (ctypes.c_uint64 * 16)()
        obj = (ctypes.c_uint64 * 16)(qp, ctypes.addressof(out))
        if mlx5.mlx5dv_init_obj(ctypes.addressof(obj), 1):
            raise RuntimeError("mlx5dv_init_obj failed: not an mlx5 device?")
        return out[1], out[2] & 0xFFFFFFFF, out[0], out[5], out[6] & 0xFFFFFFFF

    def post_send(self, qp: int, wr: int) -> None:
        if error := self._post(qp, wr, self._bad_ref):
            raise RuntimeError(f"ibv_post_send failed: {os.strerror(error)}")

    def poll_cq(self, cq: int, wc: ctypes.Array, num_entries: int) -> int:
        count = self._poll(cq, num_entries, ctypes.addressof(wc))
        if count < 0:
            raise RuntimeError("ibv_poll_cq failed")
        return count

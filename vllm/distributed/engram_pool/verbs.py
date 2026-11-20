# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ctypes bindings to libibverbs: just enough for RC QPs and RDMA WRITEs.

Structs and constants mirror rdma-core's verbs.h and mlx5dv.h under their C
names (LP64 ABI), as `pynccl_wrapper` does for NCCL. `ibv_post_send`,
`ibv_post_recv`, `ibv_poll_cq` and `ibv_req_notify_cq` are inline in verbs.h,
so they are called through the context's ops table. Only the standard library
is imported: the RDMA proxy process loads this file without vllm or torch.
"""

import ctypes
import os
from ctypes import (
    POINTER,
    Structure,
    Union,
    c_char,
    c_int,
    c_int64,
    c_size_t,
    c_uint8,
    c_uint16,
    c_uint32,
    c_uint64,
    c_void_p,
)
from functools import cache

IBV_ACCESS_LOCAL_WRITE = 1 << 0
IBV_ACCESS_REMOTE_WRITE = 1 << 1
IBV_ACCESS_REMOTE_READ = 1 << 2
IBV_ACCESS_RELAXED_ORDERING = 1 << 20
IBV_QPT_RC = 2
IBV_QPS_INIT, IBV_QPS_RTR, IBV_QPS_RTS = 1, 2, 3
IBV_QP_STATE = 1 << 0
IBV_QP_ACCESS_FLAGS = 1 << 3
IBV_QP_PKEY_INDEX = 1 << 4
IBV_QP_PORT = 1 << 5
IBV_QP_AV = 1 << 7
IBV_QP_PATH_MTU = 1 << 8
IBV_QP_TIMEOUT = 1 << 9
IBV_QP_RETRY_CNT = 1 << 10
IBV_QP_RNR_RETRY = 1 << 11
IBV_QP_RQ_PSN = 1 << 12
IBV_QP_MAX_QP_RD_ATOMIC = 1 << 13
IBV_QP_MIN_RNR_TIMER = 1 << 15
IBV_QP_SQ_PSN = 1 << 16
IBV_QP_MAX_DEST_RD_ATOMIC = 1 << 17
IBV_QP_DEST_QPN = 1 << 20
IBV_WR_RDMA_WRITE, IBV_WR_RDMA_READ = 0, 4
IBV_SEND_FENCE, IBV_SEND_SIGNALED, IBV_SEND_INLINE = 1 << 0, 1 << 1, 1 << 3
IBV_LINK_LAYER_INFINIBAND = 1
MLX5_OPCODE_RDMA_WRITE_IMM = 0x09
MLX5_SEND_WQE_BB = 64
MLX5DV_OBJ_QP = 1 << 0

# Memory only this NIC reads (shards, descriptors), peers write in any order
# (landing), or peers write in order and read back (flags, mailboxes).
NIC_READ = IBV_ACCESS_RELAXED_ORDERING
PEER_WRITE = (
    IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_RELAXED_ORDERING
)
PEER_ORDERED = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ


def _fields(ctype, names: str) -> list:
    return [(name, ctype) for name in names.split()]


class ibv_device_attr(Structure):
    _fields_ = [
        ("fw_ver", c_char * 64),
        *_fields(c_uint64, "node_guid sys_image_guid max_mr_size page_size_cap"),
        *_fields(
            c_int,
            """vendor_id vendor_part_id hw_ver max_qp max_qp_wr device_cap_flags
            max_sge max_sge_rd max_cq max_cqe max_mr max_pd max_qp_rd_atom
            max_ee_rd_atom max_res_rd_atom max_qp_init_rd_atom max_ee_init_rd_atom
            atomic_cap max_ee max_rdd max_mw max_raw_ipv6_qp max_raw_ethy_qp
            max_mcast_grp max_mcast_qp_attach max_total_mcast_qp_attach max_ah
            max_fmr max_map_per_fmr max_srq max_srq_wr max_srq_sge""",
        ),
        ("max_pkeys", c_uint16),
        *_fields(c_uint8, "local_ca_ack_delay phys_port_cnt"),
    ]


class ibv_port_attr(Structure):
    _fields_ = [
        *_fields(
            c_int,
            """state max_mtu active_mtu gid_tbl_len port_cap_flags max_msg_sz
            bad_pkey_cntr qkey_viol_cntr""",
        ),
        *_fields(c_uint16, "pkey_tbl_len lid sm_lid"),
        *_fields(
            c_uint8,
            """lmc max_vl_num sm_sl subnet_timeout init_type_reply active_width
            active_speed phys_state link_layer flags""",
        ),
        ("port_cap_flags2", c_uint16),
        ("active_speed_ex", c_uint32),
    ]


class ibv_qp_cap(Structure):
    _fields_ = _fields(
        c_uint32, "max_send_wr max_recv_wr max_send_sge max_recv_sge max_inline_data"
    )


class ibv_qp_init_attr(Structure):
    _fields_ = [
        *_fields(c_void_p, "qp_context send_cq recv_cq srq"),
        ("cap", ibv_qp_cap),
        *_fields(c_int, "qp_type sq_sig_all"),
    ]


class ibv_gid(Union):
    _fields_ = [("raw", c_uint8 * 16), ("global_", c_uint64 * 2)]


class ibv_global_route(Structure):
    _fields_ = [
        ("dgid", ibv_gid),
        ("flow_label", c_uint32),
        *_fields(c_uint8, "sgid_index hop_limit traffic_class"),
    ]


class ibv_ah_attr(Structure):
    _fields_ = [
        ("grh", ibv_global_route),
        ("dlid", c_uint16),
        *_fields(c_uint8, "sl src_path_bits static_rate is_global port_num"),
    ]


class ibv_qp_attr(Structure):
    _fields_ = [
        *_fields(c_int, "qp_state cur_qp_state path_mtu path_mig_state"),
        *_fields(c_uint32, "qkey rq_psn sq_psn dest_qp_num qp_access_flags"),
        ("cap", ibv_qp_cap),
        *_fields(ibv_ah_attr, "ah_attr alt_ah_attr"),
        *_fields(c_uint16, "pkey_index alt_pkey_index"),
        *_fields(
            c_uint8,
            """en_sqd_async_notify sq_draining max_rd_atomic max_dest_rd_atomic
            min_rnr_timer port_num timeout retry_cnt rnr_retry alt_port_num
            alt_timeout""",
        ),
        ("rate_limit", c_uint32),
    ]


class ibv_sge(Structure):
    _fields_ = [("addr", c_uint64), *_fields(c_uint32, "length lkey")]


class ibv_send_wr(Structure):
    """With `wr.rdma` of the `wr` union. RDMA WRITEs and READs leave the
    trailing `bind_mw`/`tso` union (`spare`) unused."""

    _fields_ = [
        ("wr_id", c_uint64),
        *_fields(c_void_p, "next sg_list"),
        *_fields(c_int, "num_sge opcode send_flags imm_data"),
        ("remote_addr", c_uint64),
        ("rkey", c_uint32),
        ("_wr", c_uint8 * 20),
        ("remote_srqn", c_uint32),
        ("spare", c_uint64 * 6),
    ]


class ibv_recv_wr(Structure):
    _fields_ = [
        ("wr_id", c_uint64),
        *_fields(c_void_p, "next sg_list"),
        ("num_sge", c_int),
    ]


class ibv_wc(Structure):
    _fields_ = [
        ("wr_id", c_uint64),
        *_fields(
            c_uint32,
            "status opcode vendor_err byte_len imm_data qp_num src_qp wc_flags",
        ),
        *_fields(c_uint16, "pkey_index slid"),
        *_fields(c_uint8, "sl dlid_path_bits"),
    ]


class ibv_mr(Structure):
    _fields_ = [
        *_fields(c_void_p, "context pd addr"),
        ("length", c_size_t),
        *_fields(c_uint32, "handle lkey rkey"),
    ]


class ibv_qp(Structure):
    """Up to the fields read here."""

    _fields_ = [
        *_fields(c_void_p, "context qp_context pd send_cq recv_cq srq"),
        *_fields(c_uint32, "handle qp_num"),
    ]


class ibv_context(Structure):
    """Up to the ops called here; the skipped ops are `_compat_*` stubs or
    unused."""

    _fields_ = [
        ("device", c_void_p),
        ("_ops_0", c_void_p * 11),
        *_fields(c_void_p, "poll_cq req_notify_cq"),
        ("_ops_13", c_void_p * 12),
        *_fields(c_void_p, "post_send post_recv"),
    ]


class mlx5dv_qp_queue(Structure):
    """The `sq` and `rq` members of struct mlx5dv_qp."""

    _fields_ = [("buf", c_void_p), *_fields(c_uint32, "wqe_cnt stride")]


class mlx5dv_qp(Structure):
    _fields_ = [
        ("dbrec", c_void_p),
        *_fields(mlx5dv_qp_queue, "sq rq"),
        ("bf_reg", c_void_p),
        ("bf_size", c_uint32),
        ("comp_mask", c_uint64),
        ("uar_mmap_offset", c_int64),
        *_fields(c_uint32, "tirn tisn rqn sqn"),
        ("tir_icm_addr", c_uint64),
    ]


class mlx5dv_obj(Structure):
    """Its `qp` member's `in` and `out`; the other members stay null."""

    _fields_ = [
        ("qp_in", c_void_p),
        ("qp_out", POINTER(mlx5dv_qp)),
        ("_others", c_void_p * 12),
    ]


class mlx5_wqe_ctrl_seg(Structure):
    _pack_ = 1
    _fields_ = [
        *_fields(c_uint32, "opmod_idx_opcode qpn_ds"),
        ("signature", c_uint8),
        ("dci_stream_channel_id", c_uint16),
        ("fm_ce_se", c_uint8),
        ("imm", c_uint32),
    ]


class mlx5_wqe_raddr_seg(Structure):
    _fields_ = [("raddr", c_uint64), *_fields(c_uint32, "rkey reserved")]


class mlx5_wqe_data_seg(Structure):
    _fields_ = [*_fields(c_uint32, "byte_count lkey"), ("addr", c_uint64)]


_post_send_t = ctypes.CFUNCTYPE(c_int, c_void_p, c_void_p, POINTER(c_void_p))
_post_recv_t = ctypes.CFUNCTYPE(
    c_int, c_void_p, POINTER(ibv_recv_wr), POINTER(c_void_p)
)
_poll_cq_t = ctypes.CFUNCTYPE(c_int, c_void_p, c_int, POINTER(ibv_wc))
_req_notify_cq_t = ctypes.CFUNCTYPE(c_int, c_void_p, c_int)


@cache
def _lib() -> ctypes.CDLL:
    lib = ctypes.CDLL("libibverbs.so.1", use_errno=True)
    p, i, context = c_void_p, c_int, POINTER(ibv_context)
    for name, res, args in (
        ("ibv_get_device_list", POINTER(p), [POINTER(i)]),
        ("ibv_get_device_name", ctypes.c_char_p, [p]),
        ("ibv_open_device", context, [p]),
        ("ibv_alloc_pd", p, [context]),
        ("ibv_query_device", i, [context, POINTER(ibv_device_attr)]),
        ("ibv_query_port", i, [context, c_uint8, POINTER(ibv_port_attr)]),
        ("ibv_query_gid", i, [context, c_uint8, i, POINTER(ibv_gid)]),
        ("ibv_create_cq", p, [context, i, p, p, i]),
        ("ibv_create_comp_channel", p, [context]),
        ("ibv_get_cq_event", i, [p, POINTER(p), POINTER(p)]),
        ("ibv_ack_cq_events", None, [p, ctypes.c_uint]),
        ("ibv_create_qp", POINTER(ibv_qp), [p, POINTER(ibv_qp_init_attr)]),
        ("ibv_modify_qp", i, [p, POINTER(ibv_qp_attr), i]),
        ("ibv_reg_mr", POINTER(ibv_mr), [p, p, c_size_t, i]),
        ("ibv_reg_dmabuf_mr", POINTER(ibv_mr), [p, c_uint64, c_size_t, c_uint64, i, i]),
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
    """Names of mlx5 RDMA devices (the WQEs we build are mlx5's) whose port 1
    is active. By driver, since udev may rename devices."""
    root = "/sys/class/infiniband"
    if not os.path.isdir(root):
        return []
    return sorted(
        name
        for name in os.listdir(root)
        if os.path.basename(os.path.realpath(f"{root}/{name}/device/driver"))
        == "mlx5_core"
        and _sysfs(name, "ports", "1", "state").endswith("ACTIVE")
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
        count = c_int()
        devices = _check(lib.ibv_get_device_list(ctypes.byref(count)), "list")
        handle = next(
            (
                devices[i]
                for i in range(count.value)
                if lib.ibv_get_device_name(devices[i]).decode() == name
            ),
            None,
        )
        if handle is None:
            raise RuntimeError(f"RDMA device {name} not found")
        self.name = name
        self.context = _check(lib.ibv_open_device(handle), f"open {name}")
        self.pd = _check(lib.ibv_alloc_pd(self.context), "ibv_alloc_pd")
        attr = ibv_device_attr()
        if lib.ibv_query_device(self.context, attr):
            raise RuntimeError(f"ibv_query_device({name}) failed")
        self.max_qp_wr, self.max_sge = attr.max_qp_wr, attr.max_sge
        self.max_cqe = attr.max_cqe
        # READs a QP may have in flight, as initiator and as target.
        self.rd_atomic = min(attr.max_qp_rd_atom, attr.max_qp_init_rd_atom)
        port = ibv_port_attr()
        if lib.ibv_query_port(self.context, 1, port):
            raise RuntimeError(f"ibv_query_port({name}) failed")
        self.mtu, self.lid = port.active_mtu, port.lid
        self.is_ib = port.link_layer == IBV_LINK_LAYER_INFINIBAND
        self.gid_index = _pick_gid_index(name)
        self.gid = ibv_gid()
        if lib.ibv_query_gid(self.context, 1, self.gid_index, self.gid):
            raise RuntimeError(f"ibv_query_gid({name}) failed")
        ops = self.context.contents
        self._post_send = _post_send_t(ops.post_send)
        self._post_recv = _post_recv_t(ops.post_recv)
        self._poll_cq = _poll_cq_t(ops.poll_cq)
        self._req_notify_cq = _req_notify_cq_t(ops.req_notify_cq)
        self._bad = ctypes.byref(c_void_p())
        self._event = c_void_p(), c_void_p()
        self._event_refs = tuple(ctypes.byref(p) for p in self._event)

    @property
    def mtu_bytes(self) -> int:
        """The path MTU (enum ibv_mtu) in bytes."""
        return 128 << self.mtu

    def describe(self) -> str:
        kind = "InfiniBand" if self.is_ib else f"RoCE, GID index {self.gid_index}"
        return f"{self.name} ({kind}, MTU {self.mtu_bytes})"

    def address(self) -> tuple[bytes, int]:
        return bytes(self.gid.raw), self.lid

    def reg_host(self, address: int, num_bytes: int, access: int) -> tuple[int, int]:
        """Register memory with `access` NIC_READ, PEER_WRITE or PEER_ORDERED;
        returns its lkey and rkey."""
        mr = _check(
            _lib().ibv_reg_mr(self.pd, address, num_bytes, access),
            f"ibv_reg_mr({num_bytes} B)",
        ).contents
        return mr.lkey, mr.rkey

    def reg_dmabuf(
        self, fd: int, address: int, num_bytes: int, access: int
    ) -> tuple[int, int]:
        mr = _check(
            _lib().ibv_reg_dmabuf_mr(self.pd, 0, num_bytes, address, fd, access),
            f"ibv_reg_dmabuf_mr({num_bytes} B)",
        ).contents
        return mr.lkey, mr.rkey

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
        if error := self._req_notify_cq(cq, 0):
            raise RuntimeError(f"ibv_req_notify_cq failed: {os.strerror(error)}")

    def cq_event(self, channel: int) -> None:
        """Sleep until the next event of `channel` and consume it."""
        lib = _lib()
        _check(lib.ibv_get_cq_event(channel, *self._event_refs) == 0, "get_cq_event")
        lib.ibv_ack_cq_events(self._event[0], 1)

    def post_recv(self, qp: int, wr: ibv_recv_wr) -> None:
        if error := self._post_recv(qp, wr, self._bad):
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
        cap = ibv_qp_cap(max_send_wr, max_recv_wr, max_send_sge, 1, max_inline)
        attr = ibv_qp_init_attr(
            send_cq=send_cq, recv_cq=recv_cq, cap=cap, qp_type=IBV_QPT_RC
        )
        qp = _check(_lib().ibv_create_qp(self.pd, attr), "ibv_create_qp").contents
        return ctypes.addressof(qp), qp.qp_num

    def connect(self, qp: int, remote_qpn: int, remote: tuple[bytes, int]) -> None:
        """Move a QP through INIT, RTR and RTS towards `remote_qpn`."""
        init = ibv_qp_attr(
            qp_state=IBV_QPS_INIT,
            qp_access_flags=IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ,
            port_num=1,
        )
        rtr = ibv_qp_attr(
            qp_state=IBV_QPS_RTR,
            path_mtu=self.mtu,
            dest_qp_num=remote_qpn,
            max_dest_rd_atomic=self.rd_atomic,
            min_rnr_timer=12,
        )
        rtr.ah_attr.port_num = 1
        gid, lid = remote
        if self.is_ib:
            rtr.ah_attr.dlid = lid
        else:
            rtr.ah_attr.is_global = 1
            rtr.ah_attr.grh.dgid.raw[:] = gid
            rtr.ah_attr.grh.sgid_index = self.gid_index
            rtr.ah_attr.grh.hop_limit = 64
        rts = ibv_qp_attr(
            qp_state=IBV_QPS_RTS,
            max_rd_atomic=self.rd_atomic,
            timeout=14,
            retry_cnt=7,
            rnr_retry=7,
        )
        for name, attr, mask in (
            ("INIT", init, IBV_QP_ACCESS_FLAGS | IBV_QP_PKEY_INDEX | IBV_QP_PORT),
            (
                "RTR",
                rtr,
                IBV_QP_AV
                | IBV_QP_PATH_MTU
                | IBV_QP_DEST_QPN
                | IBV_QP_RQ_PSN
                | IBV_QP_MAX_DEST_RD_ATOMIC
                | IBV_QP_MIN_RNR_TIMER,
            ),
            (
                "RTS",
                rts,
                IBV_QP_TIMEOUT
                | IBV_QP_RETRY_CNT
                | IBV_QP_RNR_RETRY
                | IBV_QP_SQ_PSN
                | IBV_QP_MAX_QP_RD_ATOMIC,
            ),
        ):
            if _lib().ibv_modify_qp(qp, attr, IBV_QP_STATE | mask):
                err = os.strerror(ctypes.get_errno())
                raise RuntimeError(f"QP {name} failed: {err}")

    @staticmethod
    def mlx5_queue(qp: int) -> mlx5dv_qp:
        """An mlx5 QP's send queue, doorbell record and BlueFlame register, for
        posting without libibverbs."""
        mlx5 = ctypes.CDLL("libmlx5.so.1")
        mlx5.mlx5dv_init_obj.argtypes = [POINTER(mlx5dv_obj), c_uint64]
        out = mlx5dv_qp()
        if mlx5.mlx5dv_init_obj(mlx5dv_obj(qp, ctypes.pointer(out)), MLX5DV_OBJ_QP):
            raise RuntimeError("mlx5dv_init_obj failed: not an mlx5 device?")
        return out

    def post_send(self, qp: int, wr: int) -> None:
        if error := self._post_send(qp, wr, self._bad):
            raise RuntimeError(f"ibv_post_send failed: {os.strerror(error)}")

    def poll_cq(self, cq: int, wc: ctypes.Array, num_entries: int) -> int:
        count = self._poll_cq(cq, num_entries, wc)
        if count < 0:
            raise RuntimeError("ibv_poll_cq failed")
        return count

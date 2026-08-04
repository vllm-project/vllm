// SPDX-License-Identifier: Apache-2.0
#![allow(unknown_lints)]

//! Raw block device I/O extension for LMCache.
//! Provides direct block device access with optional O_DIRECT support.
//!
//! Design notes (for reviewers unfamiliar with Rust / Linux I/O):
//! - This module exposes a very small surface to Python via PyO3.
//! - We wrap Linux `pread` / `pwrite` on a file descriptor opened from a
//!   block device (e.g., /dev/nvmeXnY) or a regular file (for tests).
//! - When O_DIRECT is enabled, Linux requires aligned offsets and I/O sizes.
//!   If Python buffers are aligned, we use them directly; otherwise we fallback
//!   to a bounce buffer (aligned via `posix_memalign`) for safety.
//! - For io_uring case a dedicated worker thread drives the io_uring
//!   submission/completion loop. All alignment checks are performed before
//!   enqueuing; violations result in an immediate Python `ValueError`.

use pyo3::exceptions::{PyOSError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyAny;
use std::collections::HashMap;
use std::ffi::CString;
use std::io;
use std::os::unix::io::RawFd;
use std::slice;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::Duration;

use io_uring::cqueue::{Entry, Entry32};
use io_uring::squeue::{Entry as SqueueEntry, Entry128};
use io_uring::types::Fd;
use io_uring::{opcode, IoUring};

// Wrapper enum to support both standard and big io_uring entries
// This allows fallback to standard entries on kernels < 5.19
#[derive(Clone)]
enum IoUringWrapper {
    Standard(Arc<Mutex<IoUring<SqueueEntry, Entry>>>),
    Big(Arc<Mutex<IoUring<Entry128, Entry32>>>),
}

impl IoUringWrapper {
    // Get the submission queue length
    fn submission_len(&self) -> usize {
        match self {
            IoUringWrapper::Standard(ring) => {
                let mut ring = ring.lock().unwrap();
                let len = ring.submission().len();
                len
            }
            IoUringWrapper::Big(ring) => {
                let mut ring = ring.lock().unwrap();
                let len = ring.submission().len();
                len
            }
        }
    }

    // Sync the submission queue
    fn submission_sync(&self) {
        match self {
            IoUringWrapper::Standard(ring) => {
                let mut ring = ring.lock().unwrap();
                ring.submission().sync();
            }
            IoUringWrapper::Big(ring) => {
                let mut ring = ring.lock().unwrap();
                ring.submission().sync();
            }
        }
    }
}

// NVMe identify namespace data structure
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct NvmeIdNs {
    nsze: u64,
    ncap: u64,
    nuse: u64,
    nsfeat: u8,
    nlbaf: u8,
    flbas: u8,
    mc: u8,
    dpc: u8,
    dps: u8,
    nmic: u8,
    rescap: u8,
    fpi: u8,
    dlfeat: u8,
    nawun: u16,
    nawupf: u16,
    nacwu: u16,
    nabsn: u16,
    nabo: u16,
    nabspf: u16,
    noiob: u16,
    nvmcap: [u8; 16],
    npwg: u16,
    npwa: u16,
    npdg: u16,
    npda: u16,
    nows: u16,
    mssrl: u16,
    mcl: u32,
    msrc: u8,
    rsvd81: [u8; 11],
    anagrpid: u32,
    rsvd96: [u8; 3],
    nsattr: u8,
    nvmsetid: u16,
    endgid: u16,
    nguid: [u8; 16],
    eui64: [u8; 8],
    lbaf: [NvmeLbaf; 64],
    vs: [u8; 3712],
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct NvmeLbaf {
    ms: u16,
    ds: u8,
    rp: u8,
}

// NVMe admin opcodes
const NVME_ADMIN_IDENTIFY: u8 = 0x06;

// NVMe identify CNS values
const NVME_IDENTIFY_CNS_NS: u32 = 0x00;

// NVMe I/O opcodes
const NVME_IO_READ: u8 = 0x02;
const NVME_IO_WRITE: u8 = 0x01;

// NVMe uring command structure (80 bytes)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct NvmeUringCmd {
    opcode: u8,
    flags: u8,
    rsvd1: u16,
    nsid: u32,
    cdw2: u32,
    cdw3: u32,
    metadata: u64,
    addr: u64,
    metadata_len: u32,
    data_len: u32,
    cdw10: u32,
    cdw11: u32,
    cdw12: u32,
    cdw13: u32,
    cdw14: u32,
    cdw15: u32,
    rsvd2: [u32; 4],
}

// Linux ioctl for NVMe admin command
// Defined in <linux/nvme_ioctl.h>: NVME_IOCTL_ADMIN_CMD _IOWR ('N', 0x41)
const NVME_IOCTL_ADMIN_CMD: libc::c_ulong = 0xC048_4E41;

// Defined in <linux/nvme_ioctl.h>: NVME_IOCTL_IO_CMD _IOWR ('N', 0x43)
// const NVME_IOCTL_IO_CMD: libc::c_ulong = 0xC048_4E43;

// NVMe io_uring_cmd opcodes
const NVME_URING_CMD_IO: u32 = 0xC048_4E80;

// Linux ioctl for NVMe namespace ID
// Defined in <linux/nvme_ioctl.h>: NVME_IOCTL_ID _IO ('N', 0x40)
const NVME_IOCTL_ID: libc::c_ulong = 0x4e40;

// Linux ioctl for block device size in bytes.
// Defined in <linux/fs.h>: BLKGETSIZE64 _IOR(0x12,114,size_t)
const BLKGETSIZE64: libc::c_ulong = 0x8008_1272; // ioctl op to query block size

// Buffer protocol flags (from CPython C-API).
const PYBUF_WRITABLE: i32 = 0x0001; // buffer must be writable
const PYBUF_ND: i32 = 0x0008; // request N-dimensional buffer
const PYBUF_STRIDES: i32 = 0x0010 | PYBUF_ND; // request strides info
const PYBUF_ANY_CONTIGUOUS: i32 = 0x0080 | PYBUF_STRIDES; // accept any contiguous layout

// O_DIRECT is Linux-only; define a no-op fallback for other platforms.
#[cfg(target_os = "linux")]
const O_DIRECT: i32 = libc::O_DIRECT;
#[cfg(not(target_os = "linux"))]
const O_DIRECT: i32 = 0;
const RING_SIZE: usize = 256;

fn parse_use_iouring(io_engine: Option<String>, use_iouring: bool) -> PyResult<bool> {
    match io_engine {
        Some(engine) if !engine.is_empty() => match engine.as_str() {
            "posix" => Ok(false),
            "io_uring" => Ok(true),
            other => Err(PyValueError::new_err(format!(
                "io_engine must be one of posix, io_uring; got {other}"
            ))),
        },
        _ => Ok(use_iouring),
    }
}

///Per batch tracking for in flight I/O operation
type BatchTracking = (Arc<AtomicU64>, Arc<Condvar>);

/// Round up to nearest multiple of alignment (required for O_DIRECT).
#[allow(clippy::manual_div_ceil)]
// Small helper used to align sizes for O_DIRECT I/O.
fn round_up(x: usize, align: usize) -> usize {
    (x + align - 1) / align * align
}

// Fetch errno for the last libc call on this thread.
fn errno() -> i32 {
    // SAFETY: libc call.
    #[cfg(target_os = "linux")]
    unsafe {
        *libc::__errno_location()
    }
    #[cfg(target_os = "macos")]
    unsafe {
        *libc::__error()
    }
}

// Convert errno to a Python OSError with a message.
fn os_err(msg: &str) -> PyErr {
    PyOSError::new_err((errno(), msg.to_string()))
}

// Low-level write loop that retries until all bytes are written.
// This isolates the raw syscalls from Python-facing logic.
fn pwrite_from_ptr(
    fd: RawFd,
    mut offset: u64,
    mut ptr: *const u8,
    mut len: usize,
) -> Result<(), PyErr> {
    while len > 0 {
        // SAFETY: caller guarantees ptr is valid for len bytes.
        let chunk = unsafe { slice::from_raw_parts(ptr, len) };
        let n = unsafe {
            libc::pwrite(
                fd,
                chunk.as_ptr() as *const libc::c_void,
                chunk.len(),
                offset as libc::off_t,
            )
        };
        if n < 0 {
            return Err(os_err("pwrite failed"));
        }
        let n = n as usize;
        offset += n as u64;
        // SAFETY: advance ptr by n bytes.
        unsafe {
            ptr = ptr.add(n);
        }
        len -= n;
    }
    Ok(())
}

// Low-level read loop that retries until all bytes are read.
// We treat EOF as an error because the caller expects a full read.
fn pread_into(fd: RawFd, offset: u64, mut dst: *mut u8, mut size: usize) -> Result<(), PyErr> {
    let mut off = offset;
    while size > 0 {
        // SAFETY: pread writes into dst for size bytes.
        let n = unsafe { libc::pread(fd, dst as *mut libc::c_void, size, off as libc::off_t) };
        if n < 0 {
            return Err(os_err("pread failed"));
        }
        if n == 0 {
            return Err(PyRuntimeError::new_err("unexpected EOF"));
        }
        let n = n as usize;
        // SAFETY: advance dst by n bytes.
        unsafe {
            dst = dst.add(n);
        }
        off += n as u64;
        size -= n;
    }
    Ok(())
}

// Determine file/device size in bytes (ioctl for block device, fstat fallback).
fn fd_size_bytes(fd: RawFd) -> Result<u64, PyErr> {
    // Try ioctl first (block device / loop device).
    let mut size: u64 = 0;
    // SAFETY: ioctl expects pointer to u64 for BLKGETSIZE64.
    let rc = unsafe { libc::ioctl(fd, BLKGETSIZE64, &mut size as *mut u64) };
    if rc == 0 {
        return Ok(size);
    }

    // Fallback to fstat for regular files.
    let mut st: libc::stat = unsafe { std::mem::zeroed() };
    let rc2 = unsafe { libc::fstat(fd, &mut st as *mut libc::stat) };
    if rc2 != 0 {
        return Err(os_err("fstat failed"));
    }
    Ok(st.st_size as u64)
}

// NVMe helper functions for io_uring command support

// Calculate NVMe namespace size in bytes from identify namespace data
fn nvme_ns_size_bytes(id_ns: &NvmeIdNs, lba_size: u32) -> u64 {
    id_ns.nsze * lba_size as u64
}

/// Check if device path is a character device (e.g., /dev/ng0n1)
fn is_character_device(path: &str) -> Result<bool, PyErr> {
    let cpath = CString::new(path).map_err(|_| PyValueError::new_err("path contains NUL"))?;

    // SAFETY: stat call
    let mut st: libc::stat = unsafe { std::mem::zeroed() };
    let rc = unsafe { libc::stat(cpath.as_ptr(), &mut st as *mut libc::stat) };

    if rc != 0 {
        return Err(os_err("stat failed"));
    }

    Ok((st.st_mode & libc::S_IFMT) == libc::S_IFCHR)
}

/// Get namespace ID from NVMe device using ioctl.
fn nvme_get_nsid_from_fd(fd: RawFd) -> Result<u32, PyErr> {
    //SAFETY: ioctl call with request that returns an integer
    let ret = unsafe { libc::ioctl(fd, NVME_IOCTL_ID) };
    if ret < 0 {
        return Err(os_err("Failed to get namespace ID via ioctl"));
    }
    Ok(ret as u32)
}

/// Get LBA shift (log2 of LBA size) from identify namespace data
fn nvme_get_lba_shift(id_ns: &NvmeIdNs) -> Result<u32, PyErr> {
    // Extract LBA format index from FLBAS

    let lbaf_index = if id_ns.nlbaf < 16 {
        (id_ns.flbas & 0x0F) as usize
    } else {
        let lsb = (id_ns.flbas & 0x0F) as usize;
        let msb = ((id_ns.flbas >> 5) & 0x03) as usize;
        lsb + (msb << 4)
    };

    if lbaf_index >= 64 {
        return Err(PyValueError::new_err("Invalid LBA format index"));
    }

    // Get LBA data size from LBAF
    let ds = id_ns.lbaf[lbaf_index].ds;
    if ds == 0 {
        return Err(PyValueError::new_err("Invalid LBA data size"));
    }

    // Check for metadata support
    let ms = id_ns.lbaf[lbaf_index].ms;
    if ms != 0 {
        return Err(PyValueError::new_err(
            "Device is formatted with metadata, can't be supported.",
        ));
    }

    Ok(ds as u32)
}

/// Get LBA size in bytes from identify namespace data
fn nvme_get_lba_size(id_ns: &NvmeIdNs) -> Result<u32, PyErr> {
    let lba_shift = nvme_get_lba_shift(id_ns)?;
    Ok(1u32 << lba_shift)
}

/// NVMe passthrough command structure for ioctl
#[repr(C)]
struct NvmePassthruCmd {
    opcode: u8,
    flags: u8,
    rsvd1: u16,
    nsid: u32,
    cdw2: u32,
    cdw3: u32,
    metadata: u64,
    addr: u64,
    metadata_len: u32,
    data_len: u32,
    cdw10: u32,
    cdw11: u32,
    cdw12: u32,
    cdw13: u32,
    cdw14: u32,
    cdw15: u32,
    timeout_ms: u32,
    result: u32,
}

/// Send NVMe identify namespace command via ioctl
fn nvme_identify_ns(fd: RawFd, nsid: u32) -> Result<NvmeIdNs, PyErr> {
    let mut id_ns: NvmeIdNs = unsafe { std::mem::zeroed() };

    let cmd = NvmePassthruCmd {
        opcode: NVME_ADMIN_IDENTIFY,
        nsid,
        addr: &mut id_ns as *mut NvmeIdNs as u64,
        data_len: std::mem::size_of::<NvmeIdNs>() as u32,
        cdw10: NVME_IDENTIFY_CNS_NS,
        timeout_ms: 0,
        ..unsafe { std::mem::zeroed() }
    };

    // SAFETY: ioctl with properly initialized command structure
    let rc = unsafe { libc::ioctl(fd, NVME_IOCTL_ADMIN_CMD, &cmd as *const NvmePassthruCmd) };

    if rc < 0 {
        return Err(os_err("NVMe identify namespace ioctl failed"));
    }

    Ok(id_ns)
}

/// Prepare NVMe uring command for read/write operations
#[allow(clippy::too_many_arguments)]
fn nvme_uring_cmd_prep(
    cmd: &mut NvmeUringCmd,
    is_write: bool,
    nsid: u32,
    offset: u64,
    len: usize,
    lba_shift: u32,
    ptr: *const u8,
    dtype: u8,
    dspec: u16,
) -> Result<(), PyErr> {
    let lba_size = 1usize << lba_shift;

    // Validate offset alignment
    if !(offset as usize).is_multiple_of(lba_size) {
        return Err(PyValueError::new_err(format!(
            "offset must be aligned to LBA size ({} bytes), got offset={}",
            lba_size, offset
        )));
    }

    // Validate length alignment
    if !len.is_multiple_of(lba_size) {
        return Err(PyValueError::new_err(format!(
            "length must be aligned to LBA size ({} bytes), got len={}",
            lba_size, len
        )));
    }

    // Validate non-zero length
    if len == 0 {
        return Err(PyValueError::new_err("length must be non-zero"));
    }

    // Calculate SLBA (Starting LBA) and NLB (Number of LBAs)
    let slba = offset >> lba_shift;
    let nlb = (len >> lba_shift) - 1; // NLB is 0-based

    // Validate NLB fits in NVMe field (16 bits, max 0xFFFF)
    if nlb > 0xFFFF {
        return Err(PyValueError::new_err(format!(
            "NLB ({}) exceeds NVMe field maximum (65535)",
            nlb
        )));
    }

    // Set opcode
    cmd.opcode = if is_write {
        NVME_IO_WRITE
    } else {
        NVME_IO_READ
    };
    cmd.nsid = nsid;

    // Set SLBA in cdw10 and cdw11
    cmd.cdw10 = (slba & 0xFFFFFFFF) as u32;
    cmd.cdw11 = (slba >> 32) as u32;

    // Set NLB in cdw12 (bits 0-15) and dtype in bits 20-23
    cmd.cdw12 = nlb as u32 | ((dtype as u32) << 20);

    // Set dspec in cdw13 bits 16-31
    cmd.cdw13 = (dspec as u32) << 16;

    // Set data address and length
    cmd.addr = ptr as u64;
    cmd.data_len = len as u32;

    // No metadata support for now
    cmd.metadata = 0;
    cmd.metadata_len = 0;

    Ok(())
}

/// NVMe command data for io_uring_cmd submissions.
///
/// This structure contains NVMe-specific information needed for
/// passthrough commands via io_uring_cmd.
#[derive(Clone, Debug)]
struct NvmeCmdData {
    nsid: u32,      // Namespace ID
    lba_shift: u32, // LBA shift (log2 of LBA size)
    dtype: u8,      // Directive Type
    dspec: u16,     // Directive Specific
}

/// Aligned buffer for O_DIRECT I/O.
/// Allocated with posix_memalign so the pointer satisfies alignment requirements.
/// Automatically freed on drop.
struct AlignedBuf {
    ptr: *mut u8,
    #[allow(dead_code)]
    len: usize,
    #[allow(dead_code)]
    align: usize,
}

unsafe impl Send for AlignedBuf {}
unsafe impl Sync for AlignedBuf {}

impl AlignedBuf {
    // Allocate an aligned buffer suitable for O_DIRECT.
    fn new(len: usize, align: usize) -> Result<Self, PyErr> {
        let mut p: *mut libc::c_void = std::ptr::null_mut();
        // SAFETY: posix_memalign writes to p.
        let rc = unsafe { libc::posix_memalign(&mut p as *mut *mut libc::c_void, align, len) };
        if rc != 0 {
            return Err(PyRuntimeError::new_err(format!(
                "posix_memalign failed rc={rc}"
            )));
        }
        if p.is_null() {
            return Err(PyRuntimeError::new_err("posix_memalign returned null"));
        }
        Ok(Self {
            ptr: p as *mut u8,
            len,
            align,
        })
    }

    // Mutable pointer for read/write syscalls.
    fn as_mut_ptr(&self) -> *mut u8 {
        self.ptr
    }

    // Const pointer for write syscalls.
    fn as_ptr(&self) -> *const u8 {
        self.ptr as *const u8
    }
}

impl Drop for AlignedBuf {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                libc::free(self.ptr as *mut libc::c_void);
            }
            self.ptr = std::ptr::null_mut();
        }
    }
}

// Acquire a Python buffer view with the requested mutability.
fn get_pybuffer<'py>(
    py: Python<'py>,
    obj: &Bound<'py, PyAny>,
    writable: bool,
) -> Result<pyo3::ffi::Py_buffer, PyErr> {
    // SAFETY: PyObject_GetBuffer follows CPython buffer protocol.
    unsafe {
        let mut view: pyo3::ffi::Py_buffer = std::mem::zeroed();
        // Request a contiguous byte-view. This lets Rust issue a single syscall
        // against a flat pointer instead of handling Python strides/shapes.
        let flags = if writable {
            PYBUF_WRITABLE | PYBUF_ANY_CONTIGUOUS
        } else {
            PYBUF_ANY_CONTIGUOUS
        };
        let rc = pyo3::ffi::PyObject_GetBuffer(obj.as_ptr(), &mut view, flags);
        if rc != 0 {
            return Err(PyErr::fetch(py));
        }
        Ok(view)
    }
}

// Release a buffer view previously acquired by get_pybuffer.
fn release_pybuffer(mut view: pyo3::ffi::Py_buffer) {
    // SAFETY: view was created by PyObject_GetBuffer.
    unsafe {
        pyo3::ffi::PyBuffer_Release(&mut view);
    }
}

/// Completion primitive for synchronous io_uring operations.
///
/// When a Python call needs to wait for an I/O operation to complete,
/// we use this primitive. The worker thread will call `set()` with the
/// result, and the caller calls `wait()` to block until the result is ready.
///
/// Fields:
/// - `result`: Stores the completion status (Ok or Err)
/// - `cvar`: Condition variable for signaling when result is available
struct IoCompletion {
    result: Mutex<Option<PyResult<()>>>,
    cvar: Condvar,
}

impl IoCompletion {
    fn new() -> Self {
        Self {
            result: Mutex::new(None),
            cvar: Condvar::new(),
        }
    }
    fn set(&self, r: PyResult<()>) {
        let mut guard = self
            .result
            .lock()
            .expect("IoCompletion: mutex poisoned in set()");
        *guard = Some(r);
        self.cvar.notify_one();
    }
    fn wait(&self) -> PyResult<()> {
        let mut guard = self
            .result
            .lock()
            .expect("IoCompletion: mutex poisoned in wait()");
        while guard.is_none() {
            guard = self
                .cvar
                .wait(guard)
                .expect("IoCompletion: condition variable wait failed");
        }
        guard.take().unwrap()
    }
}

/// Manages io_uring worker thread notification, using one `epoll` instance
/// over two eventfds to wait on two event sources at once: a producer-side
/// eventfd signalled when Python pushes a new submission, and a CQ-side
/// eventfd signalled by the kernel when a completion is posted.
struct UringNotify {
    /// Epoll instance watching both `producer_efd` and `cq_efd`. A single
    /// `epoll_wait` on this fd blocks until either eventfd becomes readable,
    /// so the worker can react to user-space queue pushes and kernel CQE
    /// posts from the same call site.
    epoll_fd: RawFd,
    /// Eventfd written by Python producer threads after pushing into the
    /// submission queue. Replaces the `Condvar::notify_one` used in the
    /// pre-eventfd design. Also written by `do_close` so the worker can
    /// break out of `epoll_wait` and observe the shutdown flag.
    producer_efd: RawFd,
    /// Eventfd registered with the io_uring instance via
    /// `Submitter::register_eventfd`. The kernel writes to it whenever a
    /// CQE is posted, so the worker is woken without having to drain the
    /// completion queue speculatively.
    cq_efd: RawFd,
}

impl UringNotify {
    /// Builds the three fds (two eventfds + one epoll) and wires the
    /// eventfds into the epoll instance. Cleans up partially-built state
    /// on any error path so no fd leaks if construction fails midway.
    fn new() -> io::Result<Self> {
        // Producer-side eventfd. Counter starts at 0 (no pending events).
        // EFD_CLOEXEC prevents leaking the fd into a child process via
        // execve. EFD_NONBLOCK makes read() return EAGAIN (instead of
        // blocking) when the counter is already 0; wait() relies on this
        // non-blocking behaviour to drain safely without hanging.
        let producer_efd = unsafe { libc::eventfd(0, libc::EFD_CLOEXEC | libc::EFD_NONBLOCK) };
        if producer_efd < 0 {
            // Nothing else has been allocated yet -- just bubble the error.
            return Err(io::Error::last_os_error());
        }

        // CQ-side eventfd. The kernel writes to this one after
        // register_eventfd() is called on the io_uring instance.
        let cq_efd = unsafe { libc::eventfd(0, libc::EFD_CLOEXEC | libc::EFD_NONBLOCK) };
        if cq_efd < 0 {
            // producer_efd is live -- close it before bubbling the error.
            let e = io::Error::last_os_error();
            unsafe { libc::close(producer_efd) };
            return Err(e);
        }

        // Epoll instance fd. EPOLL_CLOEXEC for the same hygiene reason as
        // EFD_CLOEXEC above.
        let epoll_fd = unsafe { libc::epoll_create1(libc::EPOLL_CLOEXEC) };
        if epoll_fd < 0 {
            // Both eventfds are live; close them before returning.
            let e = io::Error::last_os_error();
            unsafe {
                libc::close(producer_efd);
                libc::close(cq_efd);
            }
            return Err(e);
        }

        // Register both eventfds with the epoll instance so epoll_wait
        // can block on either source.
        //
        // We use EPOLLIN: wake when the counter becomes non-zero
        // (someone signalled). Alternatives we deliberately don't use:
        //   - EPOLLOUT (writable): pointless -- eventfd is effectively
        //     always writable, so it would just busy-loop epoll_wait.
        //   - EPOLLET (edge-triggered): would require fully draining
        //     every wake-up in one go to avoid missing edges. Default
        //     level-triggered fits our drain pattern in wait().
        //   - EPOLLONESHOT: would auto-disarm after each fire and force
        //     us to re-register; not worth the complexity.
        // u64 stores the fd value itself so wait() can identify which
        // fd fired without keeping a side map.
        for fd in [producer_efd, cq_efd] {
            let mut ev = libc::epoll_event {
                events: libc::EPOLLIN as u32,
                u64: fd as u64,
            };
            let rc = unsafe { libc::epoll_ctl(epoll_fd, libc::EPOLL_CTL_ADD, fd, &mut ev) };
            if rc < 0 {
                // All three fds are live; clean up all of them.
                let e = io::Error::last_os_error();
                unsafe {
                    libc::close(epoll_fd);
                    libc::close(producer_efd);
                    libc::close(cq_efd);
                }
                return Err(e);
            }
        }

        // Ownership of all three fds moves into Self; Drop handles the
        // happy-path close.
        Ok(Self {
            epoll_fd,
            producer_efd,
            cq_efd,
        })
    }

    /// Wakes the worker thread by writing 1 to `producer_efd`. Called by
    /// producers after pushing a submission into the queue and by `do_close`
    /// to break the worker out of `epoll_wait` for shutdown.
    fn signal_producer(&self) {
        let v: u64 = 1;
        // Given EFD_NONBLOCK and counter < u64::MAX - 1, the 8-byte write
        // always succeeds, so the return value is intentionally ignored.
        unsafe {
            libc::write(
                self.producer_efd,
                &v as *const u64 as *const libc::c_void,
                8,
            );
        }
    }

    /// Blocks the worker until either eventfd is readable, then drains
    /// each fired fd. Drain is required because epoll is level-triggered:
    /// without consuming the counter, the next epoll_wait would return
    /// immediately on the same already-handled signal.
    fn wait(&self) {
        // A capacity of 2 is enough: only two fds are registered with this
        // epoll instance, so at most two events can come back per call.
        let mut events = [libc::epoll_event { events: 0, u64: 0 }; 2];

        // Timeout = -1 means "block indefinitely". Shutdown wakes us by
        // writing producer_efd from do_close, so we never need a timeout.
        let n = unsafe { libc::epoll_wait(self.epoll_fd, events.as_mut_ptr(), 2, -1) };

        // n < 0 is usually EINTR (signal interruption); we just return and
        // the worker's outer loop will call wait() again. n == 0 should
        // not happen with timeout=-1 but is handled defensively.
        if n <= 0 {
            return;
        }

        // For each event reported, read 8 bytes from the corresponding
        // eventfd to reset its counter to 0. The fd value was stashed in
        // ev.u64 during epoll_ctl registration. We discard the read value
        // (we only care that a signal arrived, not how many).
        let mut buf = [0u8; 8];
        for ev in &events[..n as usize] {
            let fd = ev.u64 as RawFd;
            // Discard the result. The wake-up was already delivered by
            // epoll_wait; this read only exists to reset the eventfd
            // counter so epoll stops reporting the fd as readable. If it
            // fails, the worst case is one spurious wake-up next
            // iteration. No work is lost, because the real submissions
            // live in the queue and CQ, not in the bytes we just read.
            unsafe {
                libc::read(fd, buf.as_mut_ptr() as *mut libc::c_void, 8);
            }
        }
    }
}

impl Drop for UringNotify {
    fn drop(&mut self) {
        unsafe {
            libc::close(self.epoll_fd);
            libc::close(self.producer_efd);
            libc::close(self.cq_efd);
        }
    }
}

/// Represents a single I/O submission to io_uring.
///
/// This struct is sent from Python threads to the worker thread via a queue.
/// It contains all information needed to perform the I/O operation.
///
/// Fields:
/// - `fd`: File descriptor for the block device
/// - `offset`: Byte offset on the device to read/write
/// - `len`: Number of bytes to transfer
/// - `ptr_addr`: Memory address of the buffer (as usize for Send)
/// - `is_write`: true for write, false for read
/// - `completion`: Shared completion primitive for signaling result
/// - `fixed_buffer_idx`: Index into registered fixed buffers (if using zero-copy)
/// - `bounce`: optional bounce buffer when O_DIRECT requires alignment.
/// - `original_ptr`: For reads with bounce buffer, the original destination pointer.
/// - `payload_len`: For reads with bounce buffer, the actual payload length to copy back.
/// - `batch_id`: The batch ID this submission belongs to (for per-batch tracking)
/// - `nvme_cmd_data`: Optional NVMe command for io_uring_cmd submission
#[derive(Clone)]
struct IoSubmission {
    fd: RawFd,
    offset: u64,
    len: usize,
    ptr_addr: usize,
    is_write: bool,
    completion: Arc<IoCompletion>,
    fixed_buffer_idx: Option<u16>,
    bounce: Option<std::sync::Arc<AlignedBuf>>,
    original_ptr: Option<usize>,        // For bounce buffer reads
    payload_len: Option<usize>,         // For bounce buffer reads
    batch_id: u64,                      // Batch ID for per-batch tracking
    nvme_cmd_data: Option<NvmeCmdData>, // NVMe command data for io_uring_cmd
}

impl Default for IoSubmission {
    fn default() -> Self {
        IoSubmission {
            fd: -1 as RawFd,
            offset: 0,
            len: 0,
            ptr_addr: 0,
            is_write: false,
            completion: Arc::new(IoCompletion::new()),
            fixed_buffer_idx: None,
            bounce: None,
            original_ptr: None,
            payload_len: None,
            batch_id: 0,
            nvme_cmd_data: None,
        }
    }
}

/// Raw block device I/O interface for Python.
///
/// - Synchronous I/O (pread/pwrite) - always available
/// - Asynchronous I/O via io_uring - optional, enabled with use_iouring flag
/// Higher-level policies (slotting, manifests, etc.) live in Python.
#[pyclass]
struct RawBlockDevice {
    fd: RawFd,           // raw file descriptor
    size: u64,           // cached device size in bytes
    closed: AtomicBool,  // avoid double-close
    use_odirect: bool,   // enforce alignment + bypass page cache
    alignment: usize,    // required alignment in bytes
    use_iouring: bool,   // Enable io_uring
    use_uring_cmd: bool, // Enable io_uring_cmd for NVMe passthrough
    // NVMe device data (only when use_uring_cmd=true)
    nvme_nsid: Option<u32>,      // Namespace ID
    nvme_lba_shift: Option<u32>, // LBA shift (log2 of LBA size)
    nvme_lba_size: Option<u32>,  // LBA size in bytes
    // io_uring ring instance (only when use_iouring=true)
    // Uses wrapper to support both standard and big entries for kernel compatibility
    ring: Option<IoUringWrapper>,
    // Queue for sending I/O requests from Python to worker thread
    queue: Option<Arc<Mutex<Vec<IoSubmission>>>>,
    // Background worker thread handle
    worker: Option<thread::JoinHandle<()>>,
    // Shutdown signal for worker thread
    shutdown: Option<Arc<AtomicBool>>,
    // Map from buffer pointer address to registered fixed buffer index
    // Used for zero-copy I/O with pre-registered buffers
    fixed_buffer_map: Arc<Mutex<HashMap<usize, (u16, usize)>>>,
    // Flag indicating if fixed buffers have been registered
    fixed_buffers_registered: Arc<AtomicBool>,
    // Count of currently in-flight I/O operations (global)
    // Used for shutdown and cleanup
    in_flight_count: Arc<AtomicU64>,
    // Condition variable for signaling when in_flight_count reaches 0
    in_flight_cvar: Arc<Condvar>,
    // Per-batch in-flight count tracking
    // Maps batch_id -> (in_flight_count, condition_variable)
    batch_in_flight: Arc<Mutex<HashMap<u64, BatchTracking>>>,
    // Worker wake-up: producer eventfd + ring CQ eventfd, both polled via
    // a single epoll_fd. Replaces the previous `Arc<Condvar>` which couldn't
    // be signaled from the kernel-side completion queue.
    batch_ready: Option<Arc<UringNotify>>,
    // Store Python buffer objects for writes, reads to keep them alive until they complete
    // This prevents premature garbage collection while io_uring is using the buffers
    // Keyed by batch_id to isolate concurrent batches
    batched_buffer_objs: Arc<Mutex<HashMap<u64, Vec<Py<PyAny>>>>>,
    // Store IoCompletion objects for batched operations to check for I/O errors
    // Keyed by batch_id to isolate concurrent batches
    batched_completions: Arc<Mutex<HashMap<u64, Vec<Arc<IoCompletion>>>>>,
    // Counter for generating unique batch IDs
    next_batch_id: Arc<AtomicU64>,
}

/// RAII guard for a raw file descriptor
struct FdGuard {
    fd: RawFd,
}

impl FdGuard {
    /// Takes ownership of an open file descriptor.
    ///
    /// Args:
    /// - `fd`: a descriptor returned by a successful `open()`.
    fn new(fd: RawFd) -> Self {
        FdGuard { fd }
    }

    /// Releases ownership of the fd to the caller, disarming the guard so the
    /// descriptor is not closed on drop.
    ///
    /// Returns the raw descriptor, now owned by the caller.
    fn disarm(self) -> RawFd {
        let fd = self.fd;
        std::mem::forget(self);
        fd
    }
}

impl Drop for FdGuard {
    fn drop(&mut self) {
        // SAFETY: `fd` was returned by a successful `open()` and ownership has
        // not been released via `disarm()`, so this is the only close of this
        // descriptor.
        unsafe {
            libc::close(self.fd);
        }
    }
}

impl RawBlockDevice {
    /// Internal constructor performs all low level setup.
    #[allow(clippy::too_many_arguments)]
    fn new_internal(
        path: String,
        writable: bool,
        use_odirect: bool,
        alignment: usize,
        use_iouring: bool,
        use_uring_cmd: bool,
        io_engine: Option<String>,
        iouring_queue_depth: usize,
    ) -> PyResult<Self> {
        let use_iouring = parse_use_iouring(io_engine, use_iouring)?;
        let iouring_queue_depth = iouring_queue_depth.max(1);
        // use_uring_cmd requires use_iouring to be enabled
        if use_uring_cmd && !use_iouring {
            return Err(PyValueError::new_err(
                "use_uring_cmd requires use_iouring to be enabled",
            ));
        }
        let cpath =
            CString::new(path.clone()).map_err(|_| PyValueError::new_err("path contains NUL"))?;
        let mut flags = if writable {
            libc::O_RDWR
        } else {
            libc::O_RDONLY
        };
        if use_odirect && !use_uring_cmd {
            flags |= O_DIRECT;
        }
        // SAFETY: open returns fd or -1.
        let fd = unsafe { libc::open(cpath.as_ptr(), flags) };
        if fd < 0 {
            return Err(os_err("open failed"));
        }
        // Take ownership of the fd so it is closed if any fallible setup step
        // below returns early before the fd is moved into the RawBlockDevice.
        // Disarmed once the struct is successfully constructed.
        let fd_guard = FdGuard::new(fd);

        // Initialize NVMe data if io_uring command support is enabled
        let (nvme_nsid, nvme_lba_shift, nvme_lba_size, nvme_id_ns) = if use_uring_cmd {
            // Validate that device is a character device (required for io_uring_cmd)
            let is_char_dev = is_character_device(&path)?;
            if !is_char_dev {
                return Err(PyValueError::new_err(
                    "use_uring_cmd requires an NVMe namespace character device (e.g., /dev/ng0n1)",
                ));
            }

            // Get namespace ID from device path
            let nsid = nvme_get_nsid_from_fd(fd)?;
            // Send identify namespace command to get LBA size
            let id_ns = nvme_identify_ns(fd, nsid)?;
            let lba_shift = nvme_get_lba_shift(&id_ns)?;
            let lba_size = nvme_get_lba_size(&id_ns)?;
            if alignment == 0 || !alignment.is_multiple_of(lba_size as usize) {
                return Err(PyValueError::new_err(format!(
                    "alignment ({alignment}) must be a non-zero multiple of NVMe LBA size \
                     ({lba_size})"
                )));
            }

            (Some(nsid), Some(lba_shift), Some(lba_size), Some(id_ns))
        } else {
            (None, None, None, None)
        };

        // Calculate device size. Use NVMe ns info for character device
        // Use ioctl/fstat for block devices and regular files
        let size = if use_uring_cmd {
            if let (Some(id_ns), Some(lba_size)) = (nvme_id_ns, nvme_lba_size) {
                nvme_ns_size_bytes(&id_ns, lba_size)
            } else {
                0
            }
        } else {
            fd_size_bytes(fd)?
        };

        let (
            ring_opt,
            queue_opt,
            shutdown_opt,
            worker_opt,
            batch_ready_opt,
            in_flight_count_opt,
            in_flight_cvar_opt,
            batched_buffer_objs_opt,
            batched_completions_opt,
            next_batch_id_opt,
            batch_in_flight_opt,
        ) = if use_iouring {
            let notify = UringNotify::new()
                .map_err(|e| PyRuntimeError::new_err(format!("UringNotify init failed: {}", e)))?;
            // Try to create IoUring with big entries (Entry128/Entry32) first
            // This is required for io_uring_cmd support (kernel 5.19+)
            // If that fails, fall back to standard entries (Entry/Entry) for kernel 5.4-5.18
            let ring = match IoUring::<Entry128, Entry32>::builder()
                .build(iouring_queue_depth as u32)
            {
                Ok(big_ring) => {
                    // Big entries supported - io_uring_cmd can be used
                    if use_uring_cmd {
                        // Validate that device is a character device (required for io_uring_cmd)
                        let is_char_dev = is_character_device(&path)?;
                        if !is_char_dev {
                            return Err(PyValueError::new_err(
                                "use_uring_cmd requires an NVMe namespace character device (e.g., /dev/ng0n1)",
                            ));
                        }
                    }
                    // Register the CQ eventfd with the ring so the kernel writes to it
                    // whenever a CQE is posted. Must happen before the ring is wrapped
                    // in a Mutex / handed to the worker.
                    big_ring
                        .submitter()
                        .register_eventfd(notify.cq_efd)
                        .map_err(|e| {
                            PyRuntimeError::new_err(format!("register_eventfd failed: {}", e))
                        })?;
                    let big_ring = Arc::new(Mutex::new(big_ring));
                    IoUringWrapper::Big(big_ring)
                }
                Err(_) => {
                    // Big entries not supported (kernel < 5.19), fall back to standard entries
                    // io_uring_cmd is not available on these kernels
                    if use_uring_cmd {
                        return Err(PyRuntimeError::new_err(
                            "io_uring_cmd requires kernel 5.19 or later (big SQE/CQE entries not supported)",
                        ));
                    }
                    let std_ring = IoUring::<SqueueEntry, Entry>::builder()
                        .build(iouring_queue_depth as u32)
                        .map_err(|e| {
                            PyRuntimeError::new_err(format!("io_uring init failed: {}", e))
                        })?;
                    // Register the CQ eventfd with the ring so the kernel writes to it
                    // whenever a CQE is posted. Must happen before the ring is wrapped
                    // in a Mutex / handed to the worker.
                    std_ring
                        .submitter()
                        .register_eventfd(notify.cq_efd)
                        .map_err(|e| {
                            PyRuntimeError::new_err(format!("register_eventfd failed: {}", e))
                        })?;
                    let std_ring = Arc::new(Mutex::new(std_ring));
                    IoUringWrapper::Standard(std_ring)
                }
            };
            let queue = Arc::new(Mutex::new(Vec::<IoSubmission>::new()));
            let shutdown = Arc::new(AtomicBool::new(false));
            let batch_ready = Arc::new(notify);
            let in_flight_count = Arc::new(AtomicU64::new(0));
            let in_flight_cvar = Arc::new(Condvar::new());
            let batched_buffer_objs = Arc::new(Mutex::new(HashMap::<u64, Vec<Py<PyAny>>>::new()));
            let batched_completions =
                Arc::new(Mutex::new(HashMap::<u64, Vec<Arc<IoCompletion>>>::new()));
            let next_batch_id = Arc::new(AtomicU64::new(1));
            let batch_in_flight = Arc::new(Mutex::new(HashMap::<u64, BatchTracking>::new()));

            let ring_clone = ring.clone();
            let queue_clone = Arc::clone(&queue);
            let shutdown_clone = Arc::clone(&shutdown);
            let batch_ready_clone = Arc::clone(&batch_ready);
            let in_flight_count_clone = Arc::clone(&in_flight_count);
            let in_flight_cvar_clone = Arc::clone(&in_flight_cvar);
            let batch_in_flight_clone = Arc::clone(&batch_in_flight);
            let ring_size = iouring_queue_depth;

            // Helper function to copy data from bounce buffer to original buffer
            fn copy_from_bounce_buffer(bounce: &AlignedBuf, orig_ptr: usize, payload_len: usize) {
                unsafe {
                    libc::memcpy(
                        orig_ptr as *mut libc::c_void,
                        bounce.as_ptr() as *const libc::c_void,
                        payload_len,
                    );
                }
            }

            // Helper function to handle completion result and set IoCompletion
            // Returns Ok(()) for successful completion, Err for errors
            // Note: Short I/O resubmission for regular I/O is handled BEFORE calling this function
            // in the main completion loop. This function only handles:
            // - Full completions
            // - Errors (negative results)
            // - Short I/O during shutdown (cannot resubmit)
            fn handle_completion_result(
                sub: &mut IoSubmission,
                cqe_result: i32,
                is_shutdown: bool,
            ) -> PyResult<()> {
                let is_uring_cmd = sub.nvme_cmd_data.is_some();

                if cqe_result < 0 {
                    let code = -cqe_result;
                    let _ = sub.bounce.take();
                    Err(PyOSError::new_err((code, "io_uring I/O error")))
                } else if is_uring_cmd {
                    // Non-zero result indicates NVMe command error
                    if cqe_result != 0 {
                        let code = cqe_result;
                        let _ = sub.bounce.take();
                        Err(PyOSError::new_err((code, "io_uring_cmd NVMe error")))
                    } else {
                        // io_uring_cmd successful completion (result == 0)
                        // For reads with bounce buffer, copy data back to original buffer
                        if !sub.is_write {
                            if let (Some(bounce), Some(orig_ptr), Some(payload_len)) =
                                (sub.bounce.take(), sub.original_ptr, sub.payload_len)
                            {
                                copy_from_bounce_buffer(&bounce, orig_ptr, payload_len);
                            }
                        } else {
                            let _ = sub.bounce.take();
                        }
                        Ok(())
                    }
                } else {
                    // Regular io_uring read/write
                    let bytes_transferred = cqe_result as usize;
                    if bytes_transferred < sub.len {
                        if is_shutdown {
                            // Short read/write during shutdown: fail the request
                            let _ = sub.bounce.take();
                            Err(PyRuntimeError::new_err(
                                "io_uring worker shutting down: short I/O during shutdown",
                            ))
                        } else {
                            // This should never happen
                            let _ = sub.bounce.take();
                            Err(PyRuntimeError::new_err(
                                "Unexpected short I/O: internal error",
                            ))
                        }
                    } else {
                        // Full completion
                        // For reads with bounce buffer, copy data back to original buffer
                        if !sub.is_write {
                            if let (Some(bounce), Some(orig_ptr), Some(payload_len)) =
                                (sub.bounce.take(), sub.original_ptr, sub.payload_len)
                            {
                                copy_from_bounce_buffer(&bounce, orig_ptr, payload_len);
                            }
                        } else {
                            let _ = sub.bounce.take();
                        }
                        Ok(())
                    }
                }
            }

            // Helper function to decrement in-flight counts and notify condition variables
            fn decrement_in_flight(
                in_flight_count: &Arc<AtomicU64>,
                in_flight_cvar: &Arc<Condvar>,
                batch_in_flight: &Arc<Mutex<HashMap<u64, BatchTracking>>>,
                batch_id: u64,
            ) {
                let prev = in_flight_count.fetch_sub(1, Ordering::Relaxed);
                if prev == 1 {
                    in_flight_cvar.notify_all();
                }
                // Decrement per-batch in-flight count and notify if batch is complete
                if batch_id != 0 {
                    let batch_map = batch_in_flight.lock().unwrap();
                    if let Some((batch_count, batch_cvar)) = batch_map.get(&batch_id) {
                        let prev_batch = batch_count.fetch_sub(1, Ordering::Relaxed);
                        if prev_batch == 1 {
                            batch_cvar.notify_all();
                        }
                    }
                }
            }

            // Helper function to build and submit an SQE for a submission
            fn build_and_submit_sqe(
                ring: &IoUringWrapper,
                sub: &IoSubmission,
                user_data: u64,
            ) -> Result<(), PyErr> {
                let ptr = sub.ptr_addr as *mut u8;

                // Check if this is an io_uring_cmd submission
                if let Some(nvme_data) = &sub.nvme_cmd_data {
                    // Prepare NVMe uring command
                    let mut nvme_cmd: NvmeUringCmd = unsafe { std::mem::zeroed() };
                    nvme_uring_cmd_prep(
                        &mut nvme_cmd,
                        sub.is_write,
                        nvme_data.nsid,
                        sub.offset,
                        sub.len,
                        nvme_data.lba_shift,
                        ptr,
                        nvme_data.dtype,
                        nvme_data.dspec,
                    )?;

                    // Convert NvmeUringCmd to byte array for UringCmd80
                    let cmd_bytes: [u8; 80] = unsafe { std::mem::transmute_copy(&nvme_cmd) };

                    // Build UringCmd80 with big SQE entry
                    let mut uring_cmd =
                        opcode::UringCmd80::new(Fd(sub.fd), NVME_URING_CMD_IO).cmd(cmd_bytes);

                    // Set buf_index if using fixed buffers
                    if let Some(idx) = sub.fixed_buffer_idx {
                        uring_cmd = uring_cmd.buf_index(Some(idx));
                    }

                    let sqe128 = uring_cmd.build().user_data(user_data);

                    // Push the big SQE entry (128 bytes)
                    match ring {
                        IoUringWrapper::Big(ring) => {
                            let mut ring = ring.lock().unwrap();
                            unsafe {
                                ring.submission()
                                    .push(&sqe128)
                                    .expect("failed to push sqe128");
                            }
                        }
                        IoUringWrapper::Standard(_) => {
                            return Err(PyRuntimeError::new_err(
                                "io_uring_cmd requires big entries (kernel 5.19+)",
                            ));
                        }
                    }
                } else {
                    // Regular read/write operations
                    let sqe = if sub.is_write {
                        if let Some(idx) = sub.fixed_buffer_idx {
                            opcode::WriteFixed::new(
                                Fd(sub.fd),
                                ptr as *const u8,
                                sub.len as u32,
                                idx,
                            )
                            .offset(sub.offset)
                            .build()
                        } else {
                            opcode::Write::new(Fd(sub.fd), ptr as *const u8, sub.len as u32)
                                .offset(sub.offset)
                                .build()
                        }
                    } else if let Some(idx) = sub.fixed_buffer_idx {
                        opcode::ReadFixed::new(Fd(sub.fd), ptr, sub.len as u32, idx)
                            .offset(sub.offset)
                            .build()
                    } else {
                        opcode::Read::new(Fd(sub.fd), ptr, sub.len as u32)
                            .offset(sub.offset)
                            .build()
                    };
                    let sqe = sqe.user_data(user_data);
                    // Convert to appropriate entry type based on ring type
                    match ring {
                        IoUringWrapper::Big(ring) => {
                            let mut ring = ring.lock().unwrap();
                            let sqe128: Entry128 = sqe.into();
                            unsafe {
                                ring.submission().push(&sqe128).expect("failed to push sqe");
                            }
                        }
                        IoUringWrapper::Standard(ring) => {
                            let mut ring = ring.lock().unwrap();
                            unsafe {
                                ring.submission().push(&sqe).expect("failed to push sqe");
                            }
                        }
                    }
                }
                Ok(())
            }

            // Worker thread that handles io_uring submissions and completions.
            //
            // Runs a continuous loop that:
            // - Processes completion queue events (CQ) from the kernel
            // - Waits for new I/O requests from Python via the queue
            // - Submits new requests to the kernel (SQ)
            //
            // On shutdown, we must process ALL pending requests to avoid deadlocks:
            // - Requests still in the queue
            // - Requests already submitted to kernel (in_flight)
            // Each waiting Python thread must be woken up with an error.
            //
            // Thread safety: All access to shared state (ring, queue, in_flight)
            // is protected by mutexes. The worker is the only thread that:
            // - Reads from the submission queue
            // - Submits to io_uring
            // - Processes completions
            let worker = thread::Builder::new()
                .name("rust-rawblock-uring".into())
                .spawn(move || {
                    let mut in_flight: HashMap<u64, IoSubmission> = HashMap::new();
                    let mut next_user_data: u64 = 1;

                    while !shutdown_clone.load(Ordering::Relaxed) {
                        // This drains all completed I/O operations from the completion queue (CQ).
                        // For each completion:
                        //   - Remove the request from our in_flight tracking HashMap
                        //   - Signal the waiting Python thread via IoCompletion
                        //   - Decrement the in_flight_count atomic
                        //   - Wake up any threads waiting for all I/O to complete
                        {
                            // Process completions for standard ring
                            if let IoUringWrapper::Standard(ring) = &ring_clone {
                                let completions: Vec<_> = {
                                    let mut ring = ring.lock().unwrap();
                                    ring.completion().collect()
                                };
                                for cqe in completions {
                                    let user_data = cqe.user_data();
                                    if let Some(mut sub) = in_flight.remove(&user_data) {
                                        let batch_id = sub.batch_id;
                                        let cqe_result = cqe.result();

                                        // Handle short I/O with resubmission (only for regular I/O, not io_uring_cmd)
                                        if cqe_result >= 0
                                            && (cqe_result as usize) < sub.len
                                            && sub.nvme_cmd_data.is_none()
                                        {
                                            let bytes_transferred = cqe_result as usize;
                                            // Update offset and length for resubmission
                                            sub.offset += bytes_transferred as u64;
                                            sub.len -= bytes_transferred;
                                            // Update buffer pointer for writes and direct reads
                                            if sub.is_write || sub.bounce.is_none() {
                                                sub.ptr_addr += bytes_transferred;
                                            }
                                            // For read with bounce buffer, copy partial data back
                                            if !sub.is_write {
                                                if let (
                                                    Some(bounce),
                                                    Some(orig_ptr),
                                                    Some(payload_len),
                                                ) = (
                                                    sub.bounce.as_ref(),
                                                    sub.original_ptr,
                                                    sub.payload_len,
                                                ) {
                                                    copy_from_bounce_buffer(
                                                        bounce,
                                                        orig_ptr,
                                                        bytes_transferred.min(payload_len),
                                                    );
                                                    sub.original_ptr =
                                                        Some(orig_ptr + bytes_transferred);
                                                    sub.payload_len = Some(
                                                        payload_len
                                                            .saturating_sub(bytes_transferred),
                                                    );
                                                }
                                            }
                                            // Re-insert into in_flight with updated values
                                            // Don't decrement in_flight_count since we're resubmitting
                                            in_flight.insert(user_data, sub.clone());
                                            // Push a new SQE for the remaining data
                                            let _ =
                                                build_and_submit_sqe(&ring_clone, &sub, user_data);
                                            let _ = match &ring_clone {
                                                IoUringWrapper::Standard(ring) => {
                                                    let ring = ring.lock().unwrap();
                                                    ring.submitter().submit()
                                                }
                                                IoUringWrapper::Big(ring) => {
                                                    let ring = ring.lock().unwrap();
                                                    ring.submitter().submit()
                                                }
                                            };
                                            continue;
                                        }

                                        // Handle completion result
                                        let result =
                                            handle_completion_result(&mut sub, cqe_result, false);
                                        sub.completion.set(result);

                                        // Decrement in-flight counts and notify
                                        decrement_in_flight(
                                            &in_flight_count_clone,
                                            &in_flight_cvar_clone,
                                            &batch_in_flight_clone,
                                            batch_id,
                                        );
                                    }
                                }
                            } else if let IoUringWrapper::Big(ring) = &ring_clone {
                                let completions: Vec<_> = {
                                    let mut ring = ring.lock().unwrap();
                                    ring.completion().collect()
                                };
                                for cqe in completions {
                                    let user_data = cqe.user_data();
                                    if let Some(mut sub) = in_flight.remove(&user_data) {
                                        let batch_id = sub.batch_id;
                                        let cqe_result = cqe.result();

                                        // Handle short I/O with resubmission (only for regular I/O, not io_uring_cmd)
                                        if cqe_result >= 0
                                            && (cqe_result as usize) < sub.len
                                            && sub.nvme_cmd_data.is_none()
                                        {
                                            let bytes_transferred = cqe_result as usize;
                                            // Update offset and length for resubmission
                                            sub.offset += bytes_transferred as u64;
                                            sub.len -= bytes_transferred;
                                            // Update buffer pointer for writes and direct reads
                                            if sub.is_write || sub.bounce.is_none() {
                                                sub.ptr_addr += bytes_transferred;
                                            }
                                            // For read with bounce buffer, copy partial data back
                                            if !sub.is_write {
                                                if let (
                                                    Some(bounce),
                                                    Some(orig_ptr),
                                                    Some(payload_len),
                                                ) = (
                                                    sub.bounce.as_ref(),
                                                    sub.original_ptr,
                                                    sub.payload_len,
                                                ) {
                                                    copy_from_bounce_buffer(
                                                        bounce,
                                                        orig_ptr,
                                                        bytes_transferred.min(payload_len),
                                                    );
                                                    sub.original_ptr =
                                                        Some(orig_ptr + bytes_transferred);
                                                    sub.payload_len = Some(
                                                        payload_len
                                                            .saturating_sub(bytes_transferred),
                                                    );
                                                }
                                            }
                                            // Re-insert into in_flight with updated values
                                            // Don't decrement in_flight_count since we're resubmitting
                                            in_flight.insert(user_data, sub.clone());
                                            // Push a new SQE for the remaining data
                                            let _ =
                                                build_and_submit_sqe(&ring_clone, &sub, user_data);
                                            let _ = match &ring_clone {
                                                IoUringWrapper::Standard(ring) => {
                                                    let ring = ring.lock().unwrap();
                                                    ring.submitter().submit()
                                                }
                                                IoUringWrapper::Big(ring) => {
                                                    let ring = ring.lock().unwrap();
                                                    ring.submitter().submit()
                                                }
                                            };
                                            continue;
                                        }

                                        // Handle completion result
                                        let result =
                                            handle_completion_result(&mut sub, cqe_result, false);
                                        sub.completion.set(result);

                                        // Decrement in-flight counts and notify
                                        decrement_in_flight(
                                            &in_flight_count_clone,
                                            &in_flight_cvar_clone,
                                            &batch_in_flight_clone,
                                            batch_id,
                                        );
                                    }
                                }
                            }
                            ring_clone.submission_sync();
                        }

                        // Block on epoll only if there's truly nothing pending. The empty +
                        // shutdown checks short-circuit so we don't sleep when a producer or
                        // do_close() already left work for us. Race-free against a late
                        // signal_producer(): eventfd is a counter, so a wake-up between the
                        // check and wait() is buffered, not lost.
                        if !shutdown_clone.load(Ordering::Relaxed)
                            && queue_clone.lock().unwrap().is_empty()
                        {
                            batch_ready_clone.wait();
                        }

                        let mut q = queue_clone.lock().unwrap();
                        if !q.is_empty() {
                            // Take all pending requests from our queue and submit them to io_uring.
                            //
                            // - Remove all pending requests from queue
                            // - Check how much space is available in the ring (max 256 entries)
                            // - If batch is larger than available space, put excess back in queue
                            // - Increment in_flight_count for each request we're about to submit
                            // - Build SQE (Submission Queue Entry) for each request
                            // - Push SQEs to the ring
                            // - Call submit() to send them to the kernel
                            //
                            // Fixed Buffer Support:
                            // - If the buffer was pre-registered with register_fixed_buffers(),
                            //   we use ReadFixed/WriteFixed for true zero-copy I/O
                            // - Otherwise we use regular Read/Write with user-space pointers
                            let batch: Vec<IoSubmission> = std::mem::take(&mut *q);
                            let batch_len = batch.len();

                            let available = ring_size - ring_clone.submission_len();
                            let to_submit_count = std::cmp::min(available, batch_len);

                            if to_submit_count < batch_len {
                                let remaining: Vec<_> = batch[to_submit_count..].to_vec();
                                if !remaining.is_empty() {
                                    q.extend(remaining);
                                }
                            }

                            drop(q);

                            // Track only successfully built submissions so submit results
                            // remain aligned even when one build fails in the middle.
                            let mut user_data_list: Vec<u64> = Vec::with_capacity(to_submit_count);
                            let mut built_submissions: Vec<IoSubmission> =
                                Vec::with_capacity(to_submit_count);
                            for sub in batch.iter().take(to_submit_count) {
                                let user_data = next_user_data;
                                next_user_data = next_user_data.wrapping_add(1);
                                match build_and_submit_sqe(&ring_clone, sub, user_data) {
                                    Ok(()) => {
                                        user_data_list.push(user_data);
                                        built_submissions.push(sub.clone());
                                        in_flight.insert(user_data, sub.clone());
                                    }
                                    Err(e) => {
                                        sub.completion.set(Err(e));
                                        decrement_in_flight(
                                            &in_flight_count_clone,
                                            &in_flight_cvar_clone,
                                            &batch_in_flight_clone,
                                            sub.batch_id,
                                        );
                                    }
                                }
                            }

                            let built_count = built_submissions.len();
                            let submit_result = match &ring_clone {
                                IoUringWrapper::Standard(ring) => {
                                    let ring = ring.lock().unwrap();
                                    ring.submitter().submit()
                                }
                                IoUringWrapper::Big(ring) => {
                                    let ring = ring.lock().unwrap();
                                    ring.submitter().submit()
                                }
                            };
                            // Handle EAGAIN (ring full) and EINTR (interrupted syscall)
                            match submit_result {
                                Ok(submitted) => {
                                    // Any remaining requests in batch that weren't submitted
                                    // will be retried in the next iteration of the loop
                                    if submitted < built_count {
                                        // Remove in_flight entries for unsubmitted requests
                                        for user_data in user_data_list[submitted..].iter() {
                                            in_flight.remove(user_data);
                                        }
                                        // Put unsubmitted requests back in the queue for retry
                                        let unsubmitted: Vec<_> =
                                            built_submissions[submitted..].to_vec();
                                        if !unsubmitted.is_empty() {
                                            let mut q = queue_clone.lock().unwrap();
                                            // Insert unsubmitted requests back at the front preserving order
                                            q.splice(0..0, unsubmitted);
                                        }
                                    }
                                }
                                Err(e) => {
                                    // Handle submission errors
                                    let error_code = e.raw_os_error();
                                    match error_code {
                                        Some(libc::EAGAIN) | Some(libc::EINTR) => {
                                            // Ring is full, or the operation was interrupted due
                                            // to signal. We need to wait for completions and then retry
                                            // Remove in_flight entries for all submissions in this batch
                                            for user_data in user_data_list.iter() {
                                                in_flight.remove(user_data);
                                            }
                                            // Put unsubmitted requests back in queue for next iteration
                                            if built_count > 0 {
                                                let unsubmitted = built_submissions.clone();
                                                let mut q = queue_clone.lock().unwrap();
                                                // Insert unsubmitted requests back at the front preserving order
                                                q.splice(0..0, unsubmitted);
                                            }
                                        }
                                        _ => {
                                            // Error: fail all pending submissions in this batch.
                                            // Remove in_flight entries since these won't generate completions
                                            for user_data in user_data_list.iter() {
                                                in_flight.remove(user_data);
                                            }
                                            for sub in built_submissions.iter_mut() {
                                                let batch_id = sub.batch_id;
                                                sub.completion.set(Err(PyRuntimeError::new_err(
                                                    format!("io_uring submit error: {:?}", e),
                                                )));
                                                let _ = sub.bounce.take();
                                                decrement_in_flight(
                                                    &in_flight_count_clone,
                                                    &in_flight_cvar_clone,
                                                    &batch_in_flight_clone,
                                                    batch_id,
                                                );
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // SHUTDOWN: Wake up all waiting Python threads
                    // Drain the queue and wake up all waiting threads with error
                    {
                        let mut q = queue_clone
                            .lock()
                            .expect("Worker: queue mutex poisoned during shutdown");
                        while let Some(mut sub) = q.pop() {
                            let batch_id = sub.batch_id;
                            let _ = sub.bounce.take();
                            sub.completion.set(Err(PyRuntimeError::new_err(
                                "io_uring worker shutting down",
                            )));
                            decrement_in_flight(
                                &in_flight_count_clone,
                                &in_flight_cvar_clone,
                                &batch_in_flight_clone,
                                batch_id,
                            );
                        }
                    }

                    // Process any remaining in-flight requests
                    // Wait for kernel to complete the requests or force-cancel them
                    // Note: This 1000 milliseconds is a rough estimate
                    let graceful_shutdown = Duration::from_millis(1000);
                    thread::sleep(graceful_shutdown);
                    {
                        // Process completions for standard ring
                        if let IoUringWrapper::Standard(ring) = &ring_clone {
                            let completions: Vec<_> = {
                                let mut ring = ring.lock().unwrap();
                                ring.completion().collect()
                            };
                            for cqe in completions {
                                let user_data = cqe.user_data();
                                if let Some(mut sub) = in_flight.remove(&user_data) {
                                    let batch_id = sub.batch_id;
                                    let result =
                                        handle_completion_result(&mut sub, cqe.result(), true);
                                    sub.completion.set(result);
                                    decrement_in_flight(
                                        &in_flight_count_clone,
                                        &in_flight_cvar_clone,
                                        &batch_in_flight_clone,
                                        batch_id,
                                    );
                                }
                            }
                        } else if let IoUringWrapper::Big(ring) = &ring_clone {
                            let completions: Vec<_> = {
                                let mut ring = ring.lock().unwrap();
                                ring.completion().collect()
                            };
                            for cqe in completions {
                                let user_data = cqe.user_data();
                                if let Some(mut sub) = in_flight.remove(&user_data) {
                                    let batch_id = sub.batch_id;
                                    let result =
                                        handle_completion_result(&mut sub, cqe.result(), true);
                                    sub.completion.set(result);
                                    decrement_in_flight(
                                        &in_flight_count_clone,
                                        &in_flight_cvar_clone,
                                        &batch_in_flight_clone,
                                        batch_id,
                                    );
                                }
                            }
                        }
                        ring_clone.submission_sync();
                    }

                    // Any remaining in_flight requests, force wake with error
                    // (these were submitted to kernel but won't get completions)
                    for (_user_data, mut sub) in in_flight.drain() {
                        let batch_id = sub.batch_id;
                        let _ = sub.bounce.take();
                        sub.completion.set(Err(PyRuntimeError::new_err(
                            "io_uring worker shutting down - request cancelled",
                        )));
                        decrement_in_flight(
                            &in_flight_count_clone,
                            &in_flight_cvar_clone,
                            &batch_in_flight_clone,
                            batch_id,
                        );
                    }

                    // Final notification in case any thread is waiting on in_flight_count
                    in_flight_cvar_clone.notify_all();
                })
                .expect("spawn rust-rawblock-uring worker");

            (
                Some(ring),
                Some(queue),
                Some(shutdown),
                Some(worker),
                Some(batch_ready),
                Some(in_flight_count),
                Some(in_flight_cvar),
                Some(batched_buffer_objs),
                Some(batched_completions),
                Some(next_batch_id),
                Some(batch_in_flight),
            )
        } else {
            (
                None, None, None, None, None, None, None, None, None, None, None,
            )
        };

        // All fallible setup succeeded: hand the fd over to the struct, whose
        // Drop is now responsible for closing it. Disarm so the guard does not
        // close the same descriptor a second time.
        let fd = fd_guard.disarm();

        Ok(Self {
            fd,
            size,
            closed: AtomicBool::new(false),
            use_odirect,
            alignment,
            use_iouring,
            use_uring_cmd,
            nvme_nsid,
            nvme_lba_shift,
            nvme_lba_size,
            ring: ring_opt,
            queue: queue_opt,
            worker: worker_opt,
            shutdown: shutdown_opt,
            fixed_buffer_map: Arc::new(Mutex::new(HashMap::new())),
            fixed_buffers_registered: Arc::new(AtomicBool::new(false)),
            in_flight_count: in_flight_count_opt.unwrap_or_else(|| Arc::new(AtomicU64::new(0))),
            in_flight_cvar: in_flight_cvar_opt.unwrap_or_else(|| Arc::new(Condvar::new())),
            batch_ready: batch_ready_opt,
            batched_buffer_objs: batched_buffer_objs_opt
                .unwrap_or_else(|| Arc::new(Mutex::new(HashMap::new()))),
            batched_completions: batched_completions_opt
                .unwrap_or_else(|| Arc::new(Mutex::new(HashMap::new()))),
            next_batch_id: next_batch_id_opt.unwrap_or_else(|| Arc::new(AtomicU64::new(1))),
            batch_in_flight: batch_in_flight_opt
                .unwrap_or_else(|| Arc::new(Mutex::new(HashMap::new()))),
        })
    }

    /// Build NVMe command data for io_uring_cmd operations.
    /// Returns None if use_uring_cmd is disabled.
    fn _build_nvme_cmd_data(&self, dtype: u8, dspec: u16) -> PyResult<Option<NvmeCmdData>> {
        if !self.use_uring_cmd {
            return Ok(None);
        }
        Ok(Some(NvmeCmdData {
            nsid: self
                .nvme_nsid
                .ok_or_else(|| PyRuntimeError::new_err("NVMe namespace ID not available"))?,
            lba_shift: self
                .nvme_lba_shift
                .ok_or_else(|| PyRuntimeError::new_err("NVMe LBA shift not available"))?,
            dtype,
            dspec,
        }))
    }
}

#[pymethods]
impl RawBlockDevice {
    #[new]
    #[pyo3(
        signature = (
            path,
            writable,
            use_odirect = false,
            use_iouring = false,
            use_uring_cmd = false,
            alignment = 4096,
            io_engine = None,
            iouring_queue_depth = RING_SIZE
        )
    )]
    #[allow(clippy::too_many_arguments)]
    fn new(
        path: String,
        writable: bool,
        use_odirect: bool,
        use_iouring: bool,
        use_uring_cmd: bool,
        alignment: usize,
        io_engine: Option<String>,
        iouring_queue_depth: usize,
    ) -> PyResult<Self> {
        Self::new_internal(
            path,
            writable,
            use_odirect,
            alignment,
            use_iouring,
            use_uring_cmd,
            io_engine,
            iouring_queue_depth,
        )
    }

    // Expose cached size to Python.
    fn size_bytes(&self) -> PyResult<u64> {
        Ok(self.size)
    }

    /// Get NVMe namespace ID (only available when use_uring_cmd=true)
    fn nvme_nsid(&self) -> PyResult<u32> {
        self.nvme_nsid.ok_or_else(|| {
            PyRuntimeError::new_err("NVMe namespace ID not available (use_uring_cmd not enabled)")
        })
    }

    /// Get NVMe LBA shift (log2 of LBA size, only available when use_uring_cmd=true)
    fn nvme_lba_shift(&self) -> PyResult<u32> {
        self.nvme_lba_shift.ok_or_else(|| {
            PyRuntimeError::new_err("NVMe LBA shift not available (use_uring_cmd not enabled)")
        })
    }

    /// Get NVMe LBA size in bytes (only available when use_uring_cmd=true)
    fn nvme_lba_size(&self) -> PyResult<u32> {
        self.nvme_lba_size.ok_or_else(|| {
            PyRuntimeError::new_err("NVMe LBA size not available (use_uring_cmd not enabled)")
        })
    }

    /// Register fixed buffers for zero-copy io_uring operations.
    ///
    /// - Pre-registering memory buffers with the kernel
    /// - Using indexed descriptors (instead of pointers) in I/O operations
    /// - The kernel then does DMA directly from/to the registered buffers
    ///
    /// This is more efficient than regular I/O because:
    /// - No buffer copying between user space and kernel
    /// - The kernel can pin the memory pages for the duration of I/O
    ///
    /// Registration must happen BEFORE any I/O using these buffers.
    /// The buffers must remain valid (not freed) until unregistered.
    #[pyo3(signature = (buffer_ptrs, buffer_sizes))]
    fn register_fixed_buffers(
        &self,
        buffer_ptrs: Vec<usize>,
        buffer_sizes: Vec<usize>,
    ) -> PyResult<()> {
        if !self.use_iouring {
            return Err(PyRuntimeError::new_err("io_uring not enabled"));
        }
        if buffer_ptrs.len() != buffer_sizes.len() {
            return Err(PyValueError::new_err(
                "buffer_ptrs and buffer_sizes must have same length",
            ));
        }
        if buffer_ptrs.is_empty() {
            return Err(PyValueError::new_err(
                "at least one buffer must be provided",
            ));
        }

        {
            let mut map = self.fixed_buffer_map.lock().unwrap();
            map.clear();
            for (idx, (ptr, size)) in buffer_ptrs.iter().zip(buffer_sizes.iter()).enumerate() {
                map.insert(*ptr, (idx as u16, *size));
            }
        }

        if let Some(ring) = &self.ring {
            let mut iovecs: Vec<libc::iovec> = Vec::new();
            for (ptr, size) in buffer_ptrs.iter().zip(buffer_sizes.iter()) {
                iovecs.push(libc::iovec {
                    iov_base: *ptr as *mut libc::c_void,
                    iov_len: *size,
                });
            }
            unsafe {
                let result = match ring {
                    IoUringWrapper::Standard(ring) => {
                        let ring = ring.lock().unwrap();
                        ring.submitter().register_buffers(&iovecs)
                    }
                    IoUringWrapper::Big(ring) => {
                        let ring = ring.lock().unwrap();
                        ring.submitter().register_buffers(&iovecs)
                    }
                };
                match result {
                    Ok(_) => {
                        self.fixed_buffers_registered.store(true, Ordering::Relaxed);
                    }
                    Err(e) => {
                        return Err(PyRuntimeError::new_err(format!(
                            "register_buffers failed: {}",
                            e
                        )))
                    }
                }
            }
        }
        Ok(())
    }

    /// Batched write: submit multiple writes at once via io_uring.
    /// All writes are queued to the worker thread, which processes them
    /// in batches to maximize throughput.
    ///
    /// Returns a batch_id that must be passed to wait_iouring() to wait
    /// for completions for that batch.
    #[pyo3(signature = (offsets, buffers, total_lens))]
    fn batched_write(
        &self,
        py: Python<'_>,
        offsets: Vec<u64>,
        buffers: Vec<Bound<'_, PyAny>>,
        total_lens: Vec<usize>,
    ) -> PyResult<u64> {
        if !self.use_iouring {
            return Err(PyRuntimeError::new_err("io_uring not enabled"));
        }
        if self.closed.load(Ordering::Relaxed) {
            return Err(PyRuntimeError::new_err("device is closed"));
        }

        let n = offsets.len();
        if n == 0 {
            // Return a valid batch_id even for empty batches
            return Ok(self.next_batch_id.fetch_add(1, Ordering::Relaxed));
        }
        if buffers.len() != n || total_lens.len() != n {
            return Err(PyValueError::new_err("All vectors must have same length"));
        }

        // Acquire buffer views to keep them alive until wait_iouring() completes
        let mut views = Vec::with_capacity(n);
        for buffer in &buffers {
            let view = get_pybuffer(py, buffer, false)?;
            if view.buf.is_null() {
                for v in views {
                    release_pybuffer(v);
                }
                release_pybuffer(view);
                return Err(PyValueError::new_err("null buffer pointer"));
            }
            views.push(view);
        }

        // Generate a unique batch ID for this batch
        let batch_id = self.next_batch_id.fetch_add(1, Ordering::Relaxed);

        // Initialize per-batch tracking for this batch
        {
            let mut batch_map = self.batch_in_flight.lock().unwrap();
            batch_map.insert(
                batch_id,
                (Arc::new(AtomicU64::new(0)), Arc::new(Condvar::new())),
            );
        }

        // Store buffer objects to keep them alive until they are complete
        {
            let mut stored_objs = self.batched_buffer_objs.lock().unwrap();
            let batch_buffers = stored_objs.entry(batch_id).or_default();
            for buffer in &buffers {
                batch_buffers.push(buffer.clone().unbind());
            }
        }

        // Extract pointers as usize before releasing GIL (raw pointers are not Send)
        let mut ptrs = Vec::with_capacity(n);
        for view in &views {
            ptrs.push(view.buf as usize);
        }

        for view in views {
            release_pybuffer(view);
        }

        let fd = self.fd;
        let use_odirect = self.use_odirect;
        let alignment = self.alignment;
        let use_uring_cmd = self.use_uring_cmd;
        let fixed_buffers_registered = self.fixed_buffers_registered.load(Ordering::Relaxed);
        // Clone the fixed buffer map before releasing GIL to avoid lock contention
        let fixed_buffer_map: HashMap<usize, (u16, usize)> = if fixed_buffers_registered {
            let map = self.fixed_buffer_map.lock().unwrap();
            map.clone()
        } else {
            HashMap::new()
        };

        let nvme_cmd_data_base = if use_uring_cmd {
            Some((
                self.nvme_nsid
                    .ok_or_else(|| PyRuntimeError::new_err("NVMe namespace ID not available"))?,
                self.nvme_lba_shift
                    .ok_or_else(|| PyRuntimeError::new_err("NVMe LBA shift not available"))?,
            ))
        } else {
            None
        };
        let in_flight_count = Arc::clone(&self.in_flight_count);
        let queue = Arc::clone(self.queue.as_ref().unwrap());
        let batch_ready = Arc::clone(self.batch_ready.as_ref().unwrap());
        let batched_completions = Arc::clone(&self.batched_completions);
        let batch_in_flight = Arc::clone(&self.batch_in_flight);
        // Additional clones for cleanup on error path
        let batch_in_flight_cleanup = Arc::clone(&batch_in_flight);
        let batched_completions_cleanup = Arc::clone(&batched_completions);
        let batched_buffer_objs_cleanup = Arc::clone(&self.batched_buffer_objs);

        // Release the GIL while submitting I/O operations
        let res = py.allow_threads(move || {
            let mut submissions: Vec<(IoSubmission, Arc<IoCompletion>)> = Vec::with_capacity(n);

            // Prepare all requests, bounce buffers (if needed) and collect submission data.
            for i in 0..n {
                let ptr = ptrs[i] as *const u8;
                let total_len = total_lens[i];
                let offset = offsets[i];

                let comp = Arc::new(IoCompletion::new());

                // Fixed buffers are pre-registered with io_uring, enabling true zero-copy I/O
                let fixed_idx = fixed_buffer_map.get(&ptrs[i]).map(|(idx, _)| *idx);

                if use_odirect {
                    #[allow(clippy::manual_is_multiple_of)]
                    if (offset as usize) % alignment != 0 {
                        return Err(PyValueError::new_err("O_DIRECT requires aligned offset"));
                    }
                    #[allow(clippy::manual_is_multiple_of)]
                    if total_len % alignment != 0 {
                        return Err(PyValueError::new_err("O_DIRECT requires aligned total_len"));
                    }
                }

                // Misaligned pointer is handled via bounce buffer for writes
                let (final_ptr, bounce_opt, fixed_idx) = if use_odirect {
                    let align = alignment;
                    #[allow(clippy::manual_is_multiple_of)]
                    if ptrs[i] % align != 0 {
                        let bounce = AlignedBuf::new(total_len, align)?;
                        unsafe {
                            libc::memcpy(
                                bounce.as_mut_ptr() as *mut libc::c_void,
                                ptr as *const libc::c_void,
                                total_len,
                            );
                        }
                        let bounce_arc = std::sync::Arc::new(bounce);
                        let bounce_ptr = bounce_arc.as_ptr();
                        (bounce_ptr, Some(bounce_arc), None)
                    } else {
                        (ptr, None, fixed_idx)
                    }
                } else {
                    (ptr, None, fixed_idx)
                };

                // Build NVMe command data
                let nvme_cmd_data = if let Some((nsid, lba_shift)) = nvme_cmd_data_base {
                    Some(NvmeCmdData {
                        nsid,
                        lba_shift,
                        dtype: 0,
                        dspec: 0,
                    })
                } else {
                    None
                };

                let sub = IoSubmission {
                    fd,
                    offset,
                    len: total_len,
                    ptr_addr: final_ptr as usize,
                    is_write: true,
                    completion: comp.clone(),
                    fixed_buffer_idx: fixed_idx,
                    bounce: bounce_opt,
                    original_ptr: None,
                    payload_len: None,
                    batch_id,
                    nvme_cmd_data,
                };

                submissions.push((sub, comp));
            }

            // Queue all submissions atomically. At this point no further errors can
            // occur during queuing.
            for (sub, comp) in submissions {
                in_flight_count.fetch_add(1, Ordering::Relaxed);

                // Increment per-batch in-flight count
                {
                    let batch_map = batch_in_flight.lock().unwrap();
                    if let Some((batch_count, _)) = batch_map.get(&batch_id) {
                        batch_count.fetch_add(1, Ordering::Relaxed);
                    }
                }
                {
                    let mut q = queue.lock().unwrap();
                    q.push(sub);
                }
                batch_ready.signal_producer();

                // Store completion for error checking in wait_iouring
                {
                    let mut completions = batched_completions.lock().unwrap();
                    let batch_completions = completions.entry(batch_id).or_default();
                    batch_completions.push(comp);
                }
            }
            Ok::<(), PyErr>(())
        });

        // Preparation failed, clean up the tracking entries to prevent leaks
        if let Err(e) = res {
            {
                let mut batch_map = batch_in_flight_cleanup.lock().unwrap();
                batch_map.remove(&batch_id);
            }
            {
                let mut stored_objs = batched_buffer_objs_cleanup.lock().unwrap();
                stored_objs.remove(&batch_id);
            }
            {
                let mut completions = batched_completions_cleanup.lock().unwrap();
                completions.remove(&batch_id);
            }
            return Err(e);
        }

        Ok(batch_id)
    }

    /// Wait for all in-flight I/O for a specific batch to complete.
    /// The method waits on a per-batch condition variable that gets signaled
    /// when the batch's in-flight count reaches 0.
    ///
    /// Args:
    ///     batch_id: The batch ID returned by batched_write() or batched_read().
    ///               Only completions from this batch are checked.
    ///
    /// Returns an error if any I/O operation in this batch failed. The error message
    /// includes details about the first failure encountered.
    #[pyo3(signature = (batch_id))]
    fn wait_iouring(&self, py: Python<'_>, batch_id: u64) -> PyResult<()> {
        if !self.use_iouring {
            return Ok(());
        }

        // Get the per-batch tracking for this batch
        let (batch_count, batch_cvar) = {
            let batch_map = self.batch_in_flight.lock().unwrap();
            match batch_map.get(&batch_id) {
                Some((count, cvar)) => (Arc::clone(count), Arc::clone(cvar)),
                None => {
                    // Batch not found. This could be an empty batch or already completed
                    // Check if there are any completions for this batch
                    let mut completions = self.batched_completions.lock().unwrap();
                    let batch_completions = completions.remove(&batch_id);
                    let mut first_error: Option<PyErr> = None;
                    if let Some(comp_vec) = batch_completions {
                        for comp in comp_vec.iter() {
                            if let Err(e) = comp.wait() {
                                if first_error.is_none() {
                                    first_error = Some(e);
                                }
                            }
                        }
                    }
                    // Clear stored buffer objects for this batch
                    let mut stored_objs = self.batched_buffer_objs.lock().unwrap();
                    stored_objs.remove(&batch_id);
                    return if let Some(e) = first_error {
                        Err(e)
                    } else {
                        Ok(())
                    };
                }
            }
        };

        // Release the GIL while waiting for I/O to complete
        py.allow_threads(move || {
            let mutex = Mutex::new(());
            let mut guard = mutex.lock().unwrap();
            while batch_count.load(Ordering::Relaxed) > 0 {
                let (g, _) = batch_cvar
                    .wait_timeout(guard, Duration::from_micros(10))
                    .unwrap();
                guard = g;
            }
        });

        // Check all completion results for errors for this specific batch
        let mut completions = self.batched_completions.lock().unwrap();
        let batch_completions = completions.remove(&batch_id);
        let mut first_error: Option<PyErr> = None;
        if let Some(comp_vec) = batch_completions {
            for comp in comp_vec.iter() {
                if let Err(e) = comp.wait() {
                    if first_error.is_none() {
                        first_error = Some(e);
                    }
                }
            }
        }

        // Clear stored buffer objects for this batch now that I/O is complete
        let mut stored_objs = self.batched_buffer_objs.lock().unwrap();
        stored_objs.remove(&batch_id);

        // Clean up per-batch tracking
        let mut batch_map = self.batch_in_flight.lock().unwrap();
        batch_map.remove(&batch_id);

        if let Some(e) = first_error {
            Err(e)
        } else {
            Ok(())
        }
    }

    /// Synchronous read using io_uring.
    #[pyo3(signature = (offset, data, payload_len, total_len = None))]
    fn read_uring(
        &self,
        py: Python<'_>,
        offset: u64,
        data: &Bound<'_, PyAny>,
        payload_len: usize,
        total_len: Option<usize>,
    ) -> PyResult<()> {
        if !self.use_iouring {
            return Err(PyRuntimeError::new_err("io_uring not enabled"));
        }
        if self.closed.load(Ordering::Relaxed) {
            return Err(PyRuntimeError::new_err("device is closed"));
        }

        let view = get_pybuffer(py, data, true)?;
        if view.readonly != 0 {
            release_pybuffer(view);
            return Err(PyValueError::new_err("output buffer is readonly"));
        }
        let ptr = view.buf as *mut u8;
        if ptr.is_null() {
            release_pybuffer(view);
            return Err(PyValueError::new_err("null buffer pointer"));
        }

        let cap = view.len as usize;
        let total_len = total_len.unwrap_or(payload_len);
        if cap < payload_len {
            release_pybuffer(view);
            return Err(PyValueError::new_err(format!(
                "output buffer too small: cap={cap} need={payload_len}"
            )));
        }
        if total_len < payload_len {
            release_pybuffer(view);
            return Err(PyValueError::new_err("total_len must be >= payload_len"));
        }

        let align = self.alignment;
        if self.use_odirect {
            #[allow(clippy::manual_is_multiple_of)]
            if (offset as usize) % align != 0 {
                release_pybuffer(view);
                return Err(PyValueError::new_err("O_DIRECT requires aligned offset"));
            }
            #[allow(clippy::manual_is_multiple_of)]
            if total_len % align != 0 {
                release_pybuffer(view);
                return Err(PyValueError::new_err("O_DIRECT requires aligned total_len"));
            }
        }

        // Check if the buffer is aligned for O_DIRECT
        let ptr_aligned = if self.use_odirect {
            (ptr as usize).is_multiple_of(align)
        } else {
            true
        };

        // Fixed buffers are pre-registered with io_uring, enabling true zero-copy I/O
        let use_fixed = self.fixed_buffers_registered.load(Ordering::Relaxed);
        let fixed_idx = if use_fixed && ptr_aligned {
            let map = self.fixed_buffer_map.lock().unwrap();
            let ptr_addr = ptr as usize;
            map.get(&ptr_addr).map(|(idx, _)| *idx)
        } else {
            None
        };

        // Use bounce buffer if:
        // Buffer is not aligned (O_DIRECT requirement)
        // Buffer capacity is less than total_len
        let use_bounce = !ptr_aligned || cap < total_len;

        let res = if !use_bounce {
            self.in_flight_count.fetch_add(1, Ordering::Relaxed);
            let comp = Arc::new(IoCompletion::new());
            let sub = IoSubmission {
                fd: self.fd,
                offset,
                len: total_len,
                ptr_addr: ptr as usize,
                is_write: false,
                completion: comp.clone(),
                fixed_buffer_idx: fixed_idx,
                bounce: None,
                original_ptr: None,
                payload_len: None,
                batch_id: 0,
                nvme_cmd_data: self._build_nvme_cmd_data(0, 0)?,
            };
            {
                let q = self.queue.as_ref().expect("queue must exist");
                let mut q = q.lock().unwrap();
                q.push(sub);
            }
            if let Some(batch_ready) = &self.batch_ready {
                batch_ready.signal_producer();
            }
            py.allow_threads(move || comp.wait())
        } else {
            let bounce = AlignedBuf::new(total_len, align)?;
            let bounce_arc = std::sync::Arc::new(bounce);
            let bounce_ptr = bounce_arc.as_mut_ptr();
            self.in_flight_count.fetch_add(1, Ordering::Relaxed);
            let comp = Arc::new(IoCompletion::new());
            let sub = IoSubmission {
                fd: self.fd,
                offset,
                len: total_len,
                ptr_addr: bounce_ptr as usize,
                is_write: false,
                completion: comp.clone(),
                fixed_buffer_idx: None,
                bounce: Some(bounce_arc),
                original_ptr: Some(ptr as usize),
                payload_len: Some(payload_len),
                batch_id: 0,
                nvme_cmd_data: self._build_nvme_cmd_data(0, 0)?,
            };
            {
                let q = self.queue.as_ref().expect("queue must exist");
                let mut q = q.lock().unwrap();
                q.push(sub);
            }
            if let Some(batch_ready) = &self.batch_ready {
                batch_ready.signal_producer();
            }
            py.allow_threads(move || comp.wait())
        };

        release_pybuffer(view);
        res?;
        Ok(())
    }

    /// Synchronous write using io_uring.
    #[pyo3(signature = (offset, data, payload_len, total_len = None))]
    fn write_uring(
        &self,
        py: Python<'_>,
        offset: u64,
        data: &Bound<'_, PyAny>,
        payload_len: usize,
        total_len: Option<usize>,
    ) -> PyResult<()> {
        if !self.use_iouring {
            return Err(PyRuntimeError::new_err("io_uring not enabled"));
        }
        if self.closed.load(Ordering::Relaxed) {
            return Err(PyRuntimeError::new_err("device is closed"));
        }

        let view = get_pybuffer(py, data, false)?;
        let ptr = view.buf as *const u8;
        if ptr.is_null() {
            release_pybuffer(view);
            return Err(PyValueError::new_err("null buffer pointer"));
        }

        let cap = view.len as usize;
        let total_len = total_len.unwrap_or(payload_len);
        if cap < payload_len {
            release_pybuffer(view);
            return Err(PyValueError::new_err(format!(
                "input buffer too small: cap={cap} need={payload_len}"
            )));
        }
        if total_len < payload_len {
            release_pybuffer(view);
            return Err(PyValueError::new_err("total_len must be >= payload_len"));
        }

        let align = self.alignment;
        if self.use_odirect {
            #[allow(clippy::manual_is_multiple_of)]
            if (offset as usize) % align != 0 {
                release_pybuffer(view);
                return Err(PyValueError::new_err("O_DIRECT requires aligned offset"));
            }
            #[allow(clippy::manual_is_multiple_of)]
            if total_len % align != 0 {
                release_pybuffer(view);
                return Err(PyValueError::new_err("O_DIRECT requires aligned total_len"));
            }
        }

        // Check if the buffer is aligned for O_DIRECT
        let ptr_aligned = if self.use_odirect {
            (ptr as usize).is_multiple_of(align)
        } else {
            true
        };

        // Fixed buffers are pre-registered with io_uring, enabling true zero-copy I/O
        let use_fixed = self.fixed_buffers_registered.load(Ordering::Relaxed);
        let fixed_idx = if use_fixed && ptr_aligned {
            let map = self.fixed_buffer_map.lock().unwrap();
            let ptr_addr = ptr as usize;
            map.get(&ptr_addr).map(|(idx, _)| *idx)
        } else {
            None
        };

        // Use bounce buffer if:
        // Buffer is not aligned (O_DIRECT requirement)
        // Buffer capacity is less than total_len
        let use_bounce = !ptr_aligned || cap < total_len;

        let res = if !use_bounce {
            self.in_flight_count.fetch_add(1, Ordering::Relaxed);
            let comp = Arc::new(IoCompletion::new());
            let sub = IoSubmission {
                fd: self.fd,
                offset,
                len: total_len,
                ptr_addr: ptr as usize,
                is_write: true,
                completion: comp.clone(),
                fixed_buffer_idx: fixed_idx,
                bounce: None,
                original_ptr: None,
                payload_len: None,
                batch_id: 0,
                nvme_cmd_data: self._build_nvme_cmd_data(0, 0)?,
            };
            {
                let q = self.queue.as_ref().expect("queue must exist");
                let mut q = q.lock().unwrap();
                q.push(sub);
            }
            if let Some(batch_ready) = &self.batch_ready {
                batch_ready.signal_producer();
            }
            py.allow_threads(move || comp.wait())
        } else {
            let bounce = AlignedBuf::new(total_len, align)?;
            let bounce_arc = std::sync::Arc::new(bounce);
            let bounce_ptr = bounce_arc.as_mut_ptr();
            // Copy data to bounce buffer before submission
            unsafe {
                std::ptr::copy_nonoverlapping(ptr, bounce_ptr, payload_len);
            }
            self.in_flight_count.fetch_add(1, Ordering::Relaxed);
            let comp = Arc::new(IoCompletion::new());
            let sub = IoSubmission {
                fd: self.fd,
                offset,
                len: total_len,
                ptr_addr: bounce_ptr as usize,
                is_write: true,
                completion: comp.clone(),
                fixed_buffer_idx: None,
                bounce: Some(bounce_arc),
                original_ptr: None,
                payload_len: Some(payload_len),
                batch_id: 0,
                nvme_cmd_data: self._build_nvme_cmd_data(0, 0)?,
            };
            {
                let q = self.queue.as_ref().expect("queue must exist");
                let mut q = q.lock().unwrap();
                q.push(sub);
            }
            if let Some(batch_ready) = &self.batch_ready {
                batch_ready.signal_producer();
            }
            py.allow_threads(move || comp.wait())
        };

        release_pybuffer(view);
        res?;
        Ok(())
    }

    /// Batched read: submit multiple reads at once via io_uring.
    /// All reads are queued to the worker thread, which processes them
    /// in batches to maximize throughput.
    ///
    /// Returns a batch_id that must be passed to wait_iouring() to wait
    /// for completions for that batch
    #[pyo3(signature = (offsets, buffers, total_lens))]
    fn batched_read(
        &self,
        py: Python<'_>,
        offsets: Vec<u64>,
        buffers: Vec<Bound<'_, PyAny>>,
        total_lens: Vec<usize>,
    ) -> PyResult<u64> {
        if !self.use_iouring {
            return Err(PyRuntimeError::new_err("io_uring not enabled"));
        }
        if self.closed.load(Ordering::Relaxed) {
            return Err(PyRuntimeError::new_err("device is closed"));
        }

        let n = offsets.len();
        if n == 0 {
            // Return a valid batch_id even for empty batches
            return Ok(self.next_batch_id.fetch_add(1, Ordering::Relaxed));
        }
        if buffers.len() != n || total_lens.len() != n {
            return Err(PyValueError::new_err("All vectors must have same length"));
        }

        // Acquire buffer views to keep them alive until wait_iouring() completes
        let mut views = Vec::with_capacity(n);
        let mut caps = Vec::with_capacity(n);
        for buffer in &buffers {
            let view = get_pybuffer(py, buffer, true)?;
            if view.readonly != 0 {
                for v in views {
                    release_pybuffer(v);
                }
                release_pybuffer(view);
                return Err(PyValueError::new_err("output buffer is readonly"));
            }
            if view.buf.is_null() {
                for v in views {
                    release_pybuffer(v);
                }
                release_pybuffer(view);
                return Err(PyValueError::new_err("null buffer pointer"));
            }
            caps.push(view.len as usize);
            views.push(view);
        }

        // Generate a unique batch ID for this batch
        let batch_id = self.next_batch_id.fetch_add(1, Ordering::Relaxed);

        // Initialize per-batch tracking for this batch
        {
            let mut batch_map = self.batch_in_flight.lock().unwrap();
            batch_map.insert(
                batch_id,
                (Arc::new(AtomicU64::new(0)), Arc::new(Condvar::new())),
            );
        }

        // Store buffer objects to keep them alive until they complete
        {
            let mut stored_objs = self.batched_buffer_objs.lock().unwrap();
            let batch_buffers = stored_objs.entry(batch_id).or_default();
            for buffer in &buffers {
                batch_buffers.push(buffer.clone().unbind());
            }
        }

        // Extract pointers as usize before releasing GIL (raw pointers are not Send)
        let mut ptrs = Vec::with_capacity(n);
        for view in &views {
            ptrs.push(view.buf as usize);
        }

        for view in views {
            release_pybuffer(view);
        }

        let fd = self.fd;
        let use_odirect = self.use_odirect;
        let alignment = self.alignment;
        let fixed_buffers_registered = self.fixed_buffers_registered.load(Ordering::Relaxed);
        // Clone the fixed buffer map before releasing GIL to avoid lock contention
        let fixed_buffer_map: HashMap<usize, (u16, usize)> = if fixed_buffers_registered {
            let map = self.fixed_buffer_map.lock().unwrap();
            map.clone()
        } else {
            HashMap::new()
        };
        // Get NVMe data for io_uring_cmd
        let nvme_cmd_data = self._build_nvme_cmd_data(0, 0)?;
        let in_flight_count = Arc::clone(&self.in_flight_count);
        let queue = Arc::clone(self.queue.as_ref().unwrap());
        let batch_ready = Arc::clone(self.batch_ready.as_ref().unwrap());
        let batched_completions = Arc::clone(&self.batched_completions);
        let batch_in_flight = Arc::clone(&self.batch_in_flight);

        // Additional clones for cleanup on error path
        let batch_in_flight_cleanup = Arc::clone(&batch_in_flight);
        let batched_completions_cleanup = Arc::clone(&batched_completions);
        let batched_buffer_objs_cleanup = Arc::clone(&self.batched_buffer_objs);

        // Release the GIL while submitting I/O operations
        let res = py.allow_threads(move || {
            let mut submissions: Vec<(IoSubmission, Arc<IoCompletion>)> = Vec::with_capacity(n);

            // Prepare all requests, validate buffers and collect submission data.
            for i in 0..n {
                let total_len = total_lens[i];
                let offset = offsets[i];
                let cap = caps[i];

                // Validate buffer capacity
                if cap < total_len {
                    return Err(PyValueError::new_err(format!(
                        "output buffer too small: cap={} need={}",
                        cap, total_len
                    )));
                }

                if use_odirect {
                    #[allow(clippy::manual_is_multiple_of)]
                    if (offset as usize) % alignment != 0 {
                        return Err(PyValueError::new_err("O_DIRECT requires aligned offset"));
                    }
                    #[allow(clippy::manual_is_multiple_of)]
                    if total_len % alignment != 0 {
                        return Err(PyValueError::new_err("O_DIRECT requires aligned total_len"));
                    }
                    #[allow(clippy::manual_is_multiple_of)]
                    if ptrs[i] % alignment != 0 {
                        return Err(PyValueError::new_err("O_DIRECT requires aligned buffers"));
                    }
                }

                let comp = Arc::new(IoCompletion::new());

                // Fixed buffers are pre-registered with io_uring, enabling true zero-copy I/O
                let fixed_idx = fixed_buffer_map.get(&ptrs[i]).map(|(idx, _)| *idx);

                let sub = IoSubmission {
                    fd,
                    offset,
                    len: total_len,
                    ptr_addr: ptrs[i],
                    is_write: false, // read operation
                    completion: comp.clone(),
                    fixed_buffer_idx: fixed_idx,
                    bounce: None,
                    original_ptr: None,
                    payload_len: None,
                    batch_id,
                    nvme_cmd_data: nvme_cmd_data.clone(),
                };

                submissions.push((sub, comp));
            }

            // Queue all submissions atomically. At this point no further errors can
            // occur during queuing.
            for (sub, comp) in submissions {
                in_flight_count.fetch_add(1, Ordering::Relaxed);

                // Increment per-batch in-flight count
                {
                    let batch_map = batch_in_flight.lock().unwrap();
                    if let Some((batch_count, _)) = batch_map.get(&batch_id) {
                        batch_count.fetch_add(1, Ordering::Relaxed);
                    }
                }

                {
                    let mut q = queue.lock().unwrap();
                    q.push(sub);
                }
                batch_ready.signal_producer();

                // Store completion for error checking in wait_iouring
                {
                    let mut completions = batched_completions.lock().unwrap();
                    let batch_completions = completions.entry(batch_id).or_default();
                    batch_completions.push(comp);
                }
            }
            Ok::<(), PyErr>(())
        });

        // Preparation failed, clean up the tracking entries to prevent leaks
        if let Err(e) = res {
            {
                let mut batch_map = batch_in_flight_cleanup.lock().unwrap();
                batch_map.remove(&batch_id);
            }
            {
                let mut stored_objs = batched_buffer_objs_cleanup.lock().unwrap();
                stored_objs.remove(&batch_id);
            }
            {
                let mut completions = batched_completions_cleanup.lock().unwrap();
                completions.remove(&batch_id);
            }
            return Err(e);
        }

        Ok(batch_id)
    }

    /// Write bytes from any Python buffer object into the device.
    /// For O_DIRECT, we use direct pointer I/O when aligned and fallback to
    /// bounce buffering only for the unaligned/padded tail.
    #[pyo3(signature=(offset, data, payload_len=None, total_len=None))]
    fn pwrite_from_buffer(
        &self,
        py: Python<'_>,
        offset: u64,
        data: &Bound<'_, PyAny>,
        payload_len: Option<usize>,
        total_len: Option<usize>,
    ) -> PyResult<()> {
        if self.closed.load(Ordering::Relaxed) {
            return Err(PyRuntimeError::new_err("device is closed"));
        }
        let fd = self.fd;

        let view = get_pybuffer(py, data, false)?;
        let ptr = view.buf as *const u8;
        let buf_len = view.len as usize;
        if ptr.is_null() {
            release_pybuffer(view);
            return Err(PyValueError::new_err("null buffer pointer"));
        }

        // `payload_len`: user bytes to write.
        // `total_len`: actual I/O length. For O_DIRECT this is often aligned up.
        // Example: payload=4100, align=4096 -> total_len=8192.
        let payload_len = payload_len.unwrap_or(buf_len);
        if payload_len > buf_len {
            release_pybuffer(view);
            return Err(PyValueError::new_err("payload_len exceeds buffer length"));
        }
        let total_len = total_len.unwrap_or(payload_len);
        if total_len < payload_len {
            release_pybuffer(view);
            return Err(PyValueError::new_err("total_len must be >= payload_len"));
        }

        let align = self.alignment;
        if self.use_odirect {
            #[allow(clippy::manual_is_multiple_of)]
            if (offset as usize) % align != 0 {
                release_pybuffer(view);
                return Err(PyValueError::new_err("O_DIRECT requires aligned offset"));
            }
            #[allow(clippy::manual_is_multiple_of)]
            if total_len % align != 0 {
                release_pybuffer(view);
                return Err(PyValueError::new_err("O_DIRECT requires aligned total_len"));
            }
        }

        // Store pointer as integer before releasing the GIL. The closure passed
        // to `allow_threads` must own plain data and cannot borrow `view`.
        // We still keep `view` alive until I/O finishes, then release it below.
        let ptr_usize = ptr as usize;
        let res = py.allow_threads(move || {
            let src = ptr_usize as *const u8;
            let src_aligned = (src as usize).is_multiple_of(align);
            if total_len == payload_len && !self.use_odirect {
                // direct write without padding
                return pwrite_from_ptr(fd, offset, src, payload_len);
            }

            if self.use_odirect && src_aligned {
                if total_len == payload_len {
                    // Fully aligned fast path: no copies.
                    return pwrite_from_ptr(fd, offset, src, total_len);
                }

                // Hybrid path for O_DIRECT with padding:
                // - If the Python pointer is aligned, we avoid copying the large
                //   aligned prefix and write it directly.
                // - Only the tail is copied into an aligned bounce buffer, then
                //   zero-padded to satisfy O_DIRECT full-block writes.
                //
                // This keeps copy cost proportional to tail size, not payload size.
                let aligned_prefix = payload_len / align * align;
                if aligned_prefix > 0 {
                    pwrite_from_ptr(fd, offset, src, aligned_prefix)?;
                }
                let tail_payload = payload_len - aligned_prefix;
                let tail_total = total_len - aligned_prefix;
                if tail_total > 0 {
                    let tail_offset = offset
                        .checked_add(aligned_prefix as u64)
                        .ok_or_else(|| PyValueError::new_err("offset overflow"))?;
                    let bounce = AlignedBuf::new(tail_total, align)?;
                    unsafe {
                        if tail_payload > 0 {
                            libc::memcpy(
                                bounce.as_mut_ptr() as *mut libc::c_void,
                                src.add(aligned_prefix) as *const libc::c_void,
                                tail_payload,
                            );
                        }
                        if tail_total > tail_payload {
                            libc::memset(
                                bounce.as_mut_ptr().add(tail_payload) as *mut libc::c_void,
                                0,
                                tail_total - tail_payload,
                            );
                        }
                    }
                    pwrite_from_ptr(fd, tail_offset, bounce.as_ptr(), tail_total)?;
                }
                return Ok(());
            }

            // Full bounce path:
            // - required when source pointer is not alignment-safe for O_DIRECT.
            // - also used when non-O_DIRECT call asks for padding behavior.
            let bounce = AlignedBuf::new(total_len, align)?;
            unsafe {
                libc::memcpy(
                    bounce.as_mut_ptr() as *mut libc::c_void,
                    src as *const libc::c_void,
                    payload_len,
                );
                if total_len > payload_len {
                    libc::memset(
                        bounce.as_mut_ptr().add(payload_len) as *mut libc::c_void,
                        0,
                        total_len - payload_len,
                    );
                }
            }
            pwrite_from_ptr(fd, offset, bounce.as_ptr(), total_len)
        });
        // Always release the CPython buffer view once the blocking I/O closure
        // completes. This decrements exporter-side view count correctly.
        release_pybuffer(view);
        res?;
        Ok(())
    }

    /// Read exactly `payload_len` bytes into a writable Python buffer.
    /// For O_DIRECT, use direct reads when destination is aligned and fallback
    /// to a hybrid/read-bounce path when needed.
    #[pyo3(signature=(offset, out, payload_len, total_len=None))]
    fn pread_into(
        &self,
        py: Python<'_>,
        offset: u64,
        out: &Bound<'_, PyAny>,
        payload_len: usize,
        total_len: Option<usize>,
    ) -> PyResult<()> {
        if self.closed.load(Ordering::Relaxed) {
            return Err(PyRuntimeError::new_err("device is closed"));
        }
        let fd = self.fd;
        let view = get_pybuffer(py, out, true)?;
        if view.readonly != 0 {
            release_pybuffer(view);
            return Err(PyValueError::new_err("output buffer is readonly"));
        }
        let cap = view.len as usize;
        if cap < payload_len {
            release_pybuffer(view);
            return Err(PyValueError::new_err(format!(
                "output buffer too small: cap={cap} need={payload_len}"
            )));
        }
        let ptr = view.buf as *mut u8;
        if ptr.is_null() {
            release_pybuffer(view);
            return Err(PyValueError::new_err("null buffer pointer"));
        }

        // `payload_len`: bytes caller wants copied into `out`.
        // `total_len`: bytes to read from device. For O_DIRECT this is usually
        // aligned up and can be larger than payload_len.
        let total_len = total_len.unwrap_or(payload_len);
        if total_len < payload_len {
            release_pybuffer(view);
            return Err(PyValueError::new_err("total_len must be >= payload_len"));
        }

        let align = self.alignment;
        if self.use_odirect {
            #[allow(clippy::manual_is_multiple_of)]
            if (offset as usize) % align != 0 {
                release_pybuffer(view);
                return Err(PyValueError::new_err("O_DIRECT requires aligned offset"));
            }
            #[allow(clippy::manual_is_multiple_of)]
            if total_len % align != 0 {
                release_pybuffer(view);
                return Err(PyValueError::new_err("O_DIRECT requires aligned total_len"));
            }
        }

        // Same pattern as write path: move raw address into closure-safe value
        // while retaining `view` lifetime until closure completion.
        let dst_usize = ptr as usize;
        let res = py.allow_threads(move || {
            let dst = dst_usize as *mut u8;
            let dst_aligned = (dst as usize).is_multiple_of(align);
            if total_len == payload_len && !self.use_odirect {
                return pread_into(fd, offset, dst, payload_len);
            }

            if self.use_odirect && dst_aligned {
                if cap >= total_len {
                    // Fully aligned fast path: no copies.
                    return pread_into(fd, offset, dst, total_len);
                }

                // Hybrid path for O_DIRECT with smaller destination capacity:
                // - read aligned prefix directly into destination.
                // - read aligned tail into bounce buffer.
                // - copy only payload tail bytes back into destination.
                //
                // This avoids writing beyond Python buffer capacity while still
                // honoring O_DIRECT aligned read requirements.
                let aligned_prefix = payload_len / align * align;
                if aligned_prefix > 0 {
                    pread_into(fd, offset, dst, aligned_prefix)?;
                }
                let tail_payload = payload_len - aligned_prefix;
                let tail_total = total_len - aligned_prefix;
                if tail_total > 0 {
                    let tail_offset = offset
                        .checked_add(aligned_prefix as u64)
                        .ok_or_else(|| PyValueError::new_err("offset overflow"))?;
                    let bounce = AlignedBuf::new(tail_total, align)?;
                    pread_into(fd, tail_offset, bounce.as_mut_ptr(), tail_total)?;
                    unsafe {
                        if tail_payload > 0 {
                            libc::memcpy(
                                dst.add(aligned_prefix) as *mut libc::c_void,
                                bounce.as_ptr() as *const libc::c_void,
                                tail_payload,
                            );
                        }
                    }
                }
                return Ok(());
            }

            // Full bounce read path:
            // read aligned size into temporary aligned memory, then copy the
            // requested payload portion to Python output buffer.
            let bounce = AlignedBuf::new(round_up(total_len, align), align)?;
            pread_into(fd, offset, bounce.as_mut_ptr(), total_len)?;
            unsafe {
                libc::memcpy(
                    dst as *mut libc::c_void,
                    bounce.as_ptr() as *const libc::c_void,
                    payload_len,
                );
            }
            Ok(())
        });
        release_pybuffer(view);
        res?;
        Ok(())
    }

    /// Internal function to perform the cleanup operation.
    fn do_close(&mut self) -> Result<(), PyErr> {
        if self.use_iouring {
            if let Some(shutdown) = &self.shutdown {
                shutdown.store(true, Ordering::Relaxed);
            }
            if let Some(batch_ready) = &self.batch_ready {
                batch_ready.signal_producer();
            }

            let mutex = Mutex::new(());
            let mut guard = mutex.lock().unwrap();
            while self.in_flight_count.load(Ordering::Relaxed) > 0 {
                let (g, _) = self
                    .in_flight_cvar
                    .wait_timeout(guard, Duration::from_millis(10))
                    .unwrap();
                guard = g;
            }

            if self.fixed_buffers_registered.load(Ordering::Relaxed) {
                if let Some(ring) = &self.ring {
                    let _ = match ring {
                        IoUringWrapper::Standard(ring) => {
                            let ring = ring.lock().unwrap();
                            ring.submitter().unregister_buffers()
                        }
                        IoUringWrapper::Big(ring) => {
                            let ring = ring.lock().unwrap();
                            ring.submitter().unregister_buffers()
                        }
                    };
                }
                self.fixed_buffers_registered
                    .store(false, Ordering::Relaxed);
                self.fixed_buffer_map.lock().unwrap().clear();
            }
        }

        if let Some(handle) = self.worker.take() {
            let _ = handle.join();
        }

        let rc = unsafe { libc::close(self.fd) };
        if rc != 0 {
            return Err(os_err("close failed"));
        }
        self.closed.store(true, Ordering::Relaxed);
        Ok(())
    }

    fn close(&mut self) -> PyResult<()> {
        if !self.closed.load(Ordering::Relaxed) {
            self.do_close()?;
        }
        Ok(())
    }
}

impl Drop for RawBlockDevice {
    fn drop(&mut self) {
        if !self.closed.load(Ordering::Relaxed) {
            let _ = self.do_close();
        }
    }
}

#[pymodule]
fn lmcache_rust_raw_block_io(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RawBlockDevice>()?;
    Ok(())
}

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A fork server ("zygote") for EngineCore and worker processes.

With `spawn`, every EngineCore and worker process starts a fresh interpreter
that re-imports vLLM: thousands of modules and several seconds per process, on
the startup critical path. `fork` reuses the parent's imports but inherits its
state, which is unsafe once CUDA or other threads are live.

The zygote starts a clean interpreter, imports the engine and worker modules
once, and serves fork requests on its main Python thread.
Like a spawned process, a child gets the requester's environment, `sys.path`,
`sys.argv` and working directory, then unpickles and runs the `Process` object
it is sent. Its OS parent is the zygote, but the requester owns it through the
usual `Popen` (pid, sentinel and exit code), following the protocol of
`multiprocessing.forkserver`.

Requests the zygote cannot serve faithfully fall back to `spawn`: when it is
unreachable, when preloading started a Python thread or initialized CUDA, when an
environment variable that preloading depended on has changed since, or when
the requester's processes would run another Python executable.

Enabled with `VLLM_WORKER_MULTIPROC_METHOD=zygote`.
"""

import atexit
import contextlib
import ctypes
import gc
import io
import json
import os
import selectors
import shutil
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from collections.abc import Iterable, Sequence
from importlib import import_module
from multiprocessing import (
    connection,
    context,
    popen_fork,
    popen_spawn_posix,
    process,
    reduction,
    resource_tracker,
    spawn,
    util,
)
from multiprocessing.forkserver import read_signed, write_signed

from vllm.logger import init_logger
from vllm.utils.numa_utils import NumaBinding, _apply_numa_binding, zygote_numa_binding

logger = init_logger(__name__)

ADDRESS_ENV = "VLLM_ZYGOTE_ADDRESS"
"""Socket of the zygote serving this process tree, inherited by its processes."""

# Imported by the zygote in this order. Requests are served between imports,
# so EngineCore can start before the worker modules are loaded.
_ENGINE_MODULES = (
    "vllm.v1.engine.core",
    "vllm.v1.executor.multiproc_executor",
    "vllm.v1.executor.uniproc_executor",
)

# Read only when a process starts, so a forked child cannot honor a new value.
_EXEC_TIME_ENV = ("LD_PRELOAD", "PYTHONHASHSEED", "PYTHONPATH")

_SUPPORTED = sys.platform.startswith("linux")
_MAX_FDS = 256
_TIMEOUT_S = 300.0
_QUERY_TIMEOUT_S = 30.0
_ORPHAN_GRACE_S = 10.0
_ORPHAN_REAP_S = 2.0
_LENGTH = struct.Struct("!I")
_UCRED = struct.Struct("3i")

# Fds passed along with the request that created this process, in order.
_inherited_fds: list[int] = []


class ZygoteUnavailableError(RuntimeError):
    """The zygote cannot serve a request."""


def _close_fds(*fds: int) -> None:
    for fd in fds:
        os.close(fd)


def _interpreter_flags() -> list[str]:
    """Command-line flags reproducing this interpreter's settings."""
    return subprocess._args_from_interpreter_flags()  # type: ignore[attr-defined]


def _flush_std_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        if stream is not None:
            stream.flush()


# Client side.


# A transport failure or a permanent preload hazard disables forks from
# this address for the rest of the requester's lifetime.
_unavailable_addresses: set[str] = set()


class _DupFd:
    """An fd passed to a zygote child, by position (cf. `popen_forkserver`)."""

    def __init__(self, ind: int) -> None:
        self.ind = ind

    def detach(self) -> int:
        return _inherited_fds[self.ind]


class _ZygotePopen(popen_fork.Popen):
    method = "zygote"
    DupFd = _DupFd

    def __init__(self, process_obj: process.BaseProcess, address: str) -> None:
        self._fds: list[int] = []
        self._poll_lock = threading.Lock()
        self._address = address
        super().__init__(process_obj)

    def duplicate_for_child(self, fd: int) -> int:
        self._fds.append(fd)
        return len(self._fds) - 1

    def _launch(self, process_obj: process.BaseProcess) -> None:
        prep_data = spawn.get_preparation_data(process_obj.name)
        buf = io.BytesIO()
        context.set_spawning_popen(self)
        try:
            reduction.dump(prep_data, buf)
            reduction.dump(process_obj, buf)
        finally:
            context.set_spawning_popen(None)

        # The zygote writes the child's pid, then its exit code, to the status
        # pipe. The data pipe carries `buf`; the child also watches it as its
        # parent sentinel, so we keep its write end open for our lifetime.
        status_r, status_w = os.pipe()
        data_r, data_w = os.pipe()
        try:
            with socket.socket(socket.AF_UNIX) as sock:
                sock.settimeout(_TIMEOUT_S)
                sock.connect(self._address)
                _peer_pid(sock)
                tracker_fd = resource_tracker.getfd()
                assert tracker_fd is not None
                fds = [data_r, status_w, tracker_fd, *self._fds]
                request = {
                    "op": "fork",
                    "numa_binding": zygote_numa_binding.get(),
                    "env": dict(os.environ),
                    "executable": os.fsdecode(spawn.get_executable()),
                }
                _send_msg(sock, request, fds)
                reply = _recv_msg(sock)[0]
        except (OSError, EOFError, ValueError) as e:
            reply = {
                "error": f"no zygote at {self._address} ({e!r})",
                "unavailable": isinstance(
                    e,
                    (
                        ConnectionError,
                        TimeoutError,
                        FileNotFoundError,
                        PermissionError,
                        EOFError,
                    ),
                ),
            }
        finally:
            os.close(data_r)
            os.close(status_w)
        if "error" in reply:
            os.close(status_r)
            os.close(data_w)
            if reply.get("unavailable"):
                _unavailable_addresses.add(self._address)
            raise ZygoteUnavailableError(reply["error"])

        self.sentinel = status_r
        parent_w = os.dup(data_w)
        self.finalizer = util.Finalize(self, _close_fds, (parent_w, self.sentinel))
        with open(data_w, "wb", closefd=True) as f:
            f.write(buf.getbuffer())
        self.pid = read_signed(self.sentinel)

    def poll(self, flag: int = os.WNOHANG) -> int | None:
        if self.returncode is None:
            timeout = 0 if flag == os.WNOHANG else None
            if not connection.wait([self.sentinel], timeout):
                return None
            # wait() and the engine liveness monitor share this one-shot status.
            with self._poll_lock:
                if self.returncode is None:
                    try:
                        self.returncode = read_signed(self.sentinel)
                    except (OSError, EOFError):
                        # The zygote exited before reporting the exit code.
                        self.returncode = 255
        return self.returncode


def _popen(process_obj: process.BaseProcess):
    if _SUPPORTED:
        try:
            address = os.environ.get(ADDRESS_ENV) or start()
            if address not in _unavailable_addresses:
                return _ZygotePopen(process_obj, address)
        except ZygoteUnavailableError as e:
            logger.warning_once("Zygote unavailable, using spawn: %s", e)
    if (binding := zygote_numa_binding.get()) is not None:
        from concurrent.futures import ThreadPoolExecutor

        # Linux inherits affinity and memory policy across fork/exec from
        # the calling thread. Bind a disposable thread for this spawn only.
        def spawn_bound_child():
            _apply_numa_binding(binding)
            return popen_spawn_posix.Popen(process_obj)

        with ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(spawn_bound_child).result()
    return popen_spawn_posix.Popen(process_obj)


class ZygoteProcess(process.BaseProcess):
    # Leaves the child's default start method alone (see
    # `BaseProcess._bootstrap`): vLLM passes contexts explicitly.
    _start_method = None

    @property
    def forked_by_zygote(self) -> bool:
        return isinstance(self._popen, _ZygotePopen)  # type: ignore[attr-defined]

    @staticmethod
    def _Popen(process_obj: process.BaseProcess):
        return _popen(process_obj)


class ZygoteContext(context.BaseContext):
    _name = "zygote"
    Process = ZygoteProcess


_context = ZygoteContext()
_start_lock = threading.Lock()
_zygote: subprocess.Popen | None = None
# Write end of the pipe whose closing, when this process exits, tells the
# zygote to stop serving. Never closed explicitly.
_alive_w: int | None = None


def get_context() -> ZygoteContext:
    """The multiprocessing context whose processes are forked by the zygote.

    The zygote is started on first use if none serves this process tree.
    """
    return _context


def descendants(pids: Iterable[int]) -> list[int]:
    """The processes the zygote forked at the request of one of `pids`, or of
    another such process.

    They descend from `pids` as spawned processes would, but their parent is
    the zygote, so `psutil` does not find them.
    """
    address = os.environ.get(ADDRESS_ENV)
    if not (_SUPPORTED and address):
        return []
    try:
        with socket.socket(socket.AF_UNIX) as sock:
            sock.settimeout(_QUERY_TIMEOUT_S)
            sock.connect(address)
            _peer_pid(sock)
            _send_msg(sock, {"op": "descendants", "pids": list(pids)})
            return _recv_msg(sock)[0]["pids"]
    except (OSError, EOFError, ValueError, KeyError) as e:
        logger.debug("Could not query the zygote at %s: %r", address, e)
        return []


def maybe_start_for_cli(argv: Sequence[str]) -> None:
    """Start the opted-in zygote early for commands that launch engines."""
    if len(argv) < 2 or argv[1] not in ("serve", "run-batch"):
        return
    if any(arg in ("-h", "--help") or arg.startswith("--help=") for arg in argv):
        return
    try:
        start()
    except ZygoteUnavailableError as e:
        logger.warning("%s; processes will be spawned.", e)
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


def start(preload: Sequence[str] | None = None) -> str:
    """Start a zygote owned by this process, unless one is running, and
    advertise it to the processes this process starts.

    When this process exits, the zygote stops serving and sends SIGTERM to
    orphaned roots. After a grace period it kills any remaining children,
    waits a bounded time to reap them, and removes its socket directory.

    Args:
        preload: Modules to import before forking. Defaults to the EngineCore,
            executor and GPU worker modules.

    Returns:
        The zygote's socket address.

    Raises:
        ZygoteUnavailableError: If the zygote could not be started.

    """
    global _zygote, _alive_w
    if not _SUPPORTED:
        raise ZygoteUnavailableError("zygote requires Linux")
    with _start_lock:
        if _zygote is not None and _zygote.poll() is None:
            return os.environ[ADDRESS_ENV]
        alive_r, alive_w = os.pipe()
        sock_dir = None
        try:
            sock_dir = tempfile.mkdtemp(prefix="vllm-zygote-")
            address = os.path.join(sock_dir, "sock")
            with socket.socket(socket.AF_UNIX) as listener:
                listener.bind(address)
                listener.listen(64)
                # The zygote starts with the GC off (it never collects;
                # children freeze its heap) and ignoring Ctrl-C, like the
                # stdlib forkserver.
                cmd = (
                    "import gc, signal, sys; gc.disable(); "
                    "signal.signal(signal.SIGINT, signal.SIG_IGN); "
                    f"sys.path[:] = {sys.path!r}; "
                    "from vllm.utils.zygote import _serve; "
                    f"_serve({listener.fileno()}, {alive_r}, {sock_dir!r}, "
                    f"{None if preload is None else list(preload)!r})"
                )
                _zygote = subprocess.Popen(
                    [spawn.get_executable(), *_interpreter_flags(), "-c", cmd],
                    pass_fds=(listener.fileno(), alive_r),
                    stdin=subprocess.DEVNULL,
                    env={**os.environ, ADDRESS_ENV: address},
                )
        except OSError as e:
            os.close(alive_w)
            if sock_dir is not None:
                shutil.rmtree(sock_dir, ignore_errors=True)
            raise ZygoteUnavailableError(f"could not start a zygote ({e!r})") from e
        finally:
            os.close(alive_r)
        atexit.register(shutil.rmtree, sock_dir, ignore_errors=True)
        _alive_w = alive_w
        os.environ[ADDRESS_ENV] = address
        logger.debug("Started zygote pid %d at %s", _zygote.pid, address)
        return address


# Wire format: a length-prefixed JSON message, with any fds attached to the
# length prefix.


def _peer_pid(sock: socket.socket) -> int:
    """Authenticate the other endpoint before exchanging process state or fds."""
    assert hasattr(socket, "SO_PEERCRED")
    creds = sock.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, _UCRED.size)
    pid, uid, _ = _UCRED.unpack(creds)
    if uid != os.getuid():
        raise PermissionError("zygote peer has a different uid")
    return pid


def _send_msg(sock: socket.socket, msg: dict, fds: Sequence[int] = ()) -> None:
    data = json.dumps(msg).encode()
    socket.send_fds(sock, [_LENGTH.pack(len(data))], list(fds))
    sock.sendall(data)


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise EOFError("connection closed")
        buf += chunk
    return bytes(buf)


def _recv_msg(sock: socket.socket, maxfds: int = 0) -> tuple[dict, list[int]]:
    head, fds, flags, _ = socket.recv_fds(sock, _LENGTH.size, maxfds)
    try:
        if flags & socket.MSG_CTRUNC:
            raise ValueError("too many fds")
        if not head:
            raise EOFError("connection closed")
        head += _recv_exact(sock, _LENGTH.size - len(head))
        (length,) = _LENGTH.unpack(head)
        return json.loads(_recv_exact(sock, length)), fds
    except BaseException:
        for fd in fds:
            os.close(fd)
        raise


# Zygote side.


def _default_preload() -> list[str] | None:
    """The modules to preload, or None if the platform is not supported.

    Other platforms configure their workers' OpenMP runtime through variables
    that it reads when loaded, which a forked child cannot honor.
    """
    from vllm.platforms import current_platform

    if not current_platform.is_cuda_alike():
        return None
    return [
        *_ENGINE_MODULES,
        "vllm.v1.worker.gpu_worker",
        "vllm.v1.worker.gpu.model_runner",
    ]


def _fork_hazard() -> str | None:
    """Why forking this process would be unsafe, if it would be."""
    threads = [
        t.name for t in threading.enumerate() if t is not threading.main_thread()
    ]
    if threads:
        return f"started Python threads {threads}"
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_initialized():
        return "initialized CUDA"

    # An extension can initialize the driver without going through torch.
    # RTLD_NOLOAD also avoids loading a driver solely to perform the probe.
    try:
        driver = ctypes.CDLL("libcuda.so.1", mode=os.RTLD_NOLOAD)
    except OSError:
        return None
    try:
        result = driver.cuDeviceGetCount(ctypes.byref(ctypes.c_int()))
    except AttributeError:
        return "cannot query the loaded CUDA driver"
    if result != 3:  # CUDA_ERROR_NOT_INITIALIZED
        return f"CUDA driver is unsafe to fork (cuDeviceGetCount returned {result})"
    return None


def _child_env(
    start_env: dict[str, str],
    current_env: dict[str, str],
    request_env: dict[str, str],
    depends_on: Iterable[str],
) -> dict[str, str] | str:
    """The environment of a child forked for a request, or why there is none.

    The child gets the requester's environment, plus the changes preloading
    made to the zygote's (as its own imports would have in a spawned process).
    Variables that preloading read or changed must be as the zygote saw them,
    before or after preloading: modules imported under other values would be
    stale in the child. Requester edits to `LD_LIBRARY_PATH` are forwarded
    for programs executed later in the child.
    """
    env = dict(request_env)
    changed = {
        k
        for k in start_env.keys() | current_env.keys()
        if start_env.get(k) != current_env.get(k)
    }
    for name in changed.union(depends_on):
        value = current_env.get(name)
        if request_env.get(name) not in (start_env.get(name), value):
            # Imports such as cv2 can edit LD_LIBRARY_PATH in both processes.
            # Preserve the requester's value for executables the child starts.
            if name == "LD_LIBRARY_PATH":
                continue
            return f"{name} changed after the zygote started"
        if value is None:
            env.pop(name, None)
        else:
            env[name] = value
    return env


class _ZygoteServer:
    def __init__(
        self, listener_fd: int, alive_fd: int, preload: list[str] | None
    ) -> None:
        import vllm.envs as envs

        # Preloaded modules were imported under this environment, and may
        # have captured the vLLM variables they read.
        self.start_env = dict(os.environ)
        self.env_reads: set[str] = set()
        self.envs_getattr = envs.__getattr__

        def recording_getattr(name: str):
            self.env_reads.add(name)
            return self.envs_getattr(name)

        envs.__getattr__ = recording_getattr  # type: ignore[assignment]

        from vllm.utils.system_utils import decorate_logs, set_process_title

        set_process_title("Zygote")
        decorate_logs("Zygote")
        self.listener = socket.socket(fileno=listener_fd)
        self.alive_fd = alive_fd
        self.children: dict[int, int] = {}  # pid -> write end of status pipe
        self.requesters: dict[int, int] = {}  # pid -> pid of its requester
        self.hazard: str | None = None
        if preload is None and (preload := _default_preload()) is None:
            self.hazard = "it supports only CUDA and ROCm"
            preload = []
        else:
            self.hazard = _fork_hazard()
        self.pending = deque(preload)

        self.sig_r, self.sig_w = os.pipe()
        os.set_blocking(self.sig_r, False)
        os.set_blocking(self.sig_w, False)
        signal.set_wakeup_fd(self.sig_w)
        signal.signal(signal.SIGCHLD, lambda *_: None)
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.listener, selectors.EVENT_READ)
        self.selector.register(self.alive_fd, selectors.EVENT_READ)
        self.selector.register(self.sig_r, selectors.EVENT_READ)

    def _drain_sigchld(self) -> None:
        with contextlib.suppress(BlockingIOError):
            os.read(self.sig_r, 4096)

    def run(self) -> None:
        start = time.monotonic()
        while True:
            preloading = bool(self.pending)
            for key, _ in self.selector.select(0 if preloading else None):
                if key.fileobj is self.listener:
                    self._accept()
                elif key.fd == self.sig_r:
                    self._drain_sigchld()
                    self._reap()
                elif key.fd == self.alive_fd:
                    self.selector.unregister(self.listener)
                    self.selector.unregister(self.alive_fd)
                    self.listener.close()
                    os.close(self.alive_fd)
                    self._stop_orphans()
                    return
            if preloading:
                self._preload(self.pending.popleft())
                if not self.pending:
                    logger.debug("Preloaded in %.2fs", time.monotonic() - start)

    def _stop_orphans(self) -> None:
        self._reap()
        if not self.children:
            return
        for sig, timeout in (
            (signal.SIGTERM, _ORPHAN_GRACE_S),
            (signal.SIGKILL, _ORPHAN_REAP_S),
        ):
            deadline = time.monotonic() + timeout
            signaled: set[int] = set()
            while self.children:
                targets = set(self.children)
                if sig == signal.SIGTERM:
                    # Give each requester a chance to stop its own workers.
                    targets = {
                        pid
                        for pid in targets
                        if self.requesters.get(pid) not in targets
                    } or targets
                for pid in targets - signaled:
                    with contextlib.suppress(ProcessLookupError):
                        os.kill(pid, sig)
                signaled.update(targets)
                # Only the SIGCHLD wakeup pipe is still registered.
                if self.selector.select(max(0, deadline - time.monotonic())):
                    self._drain_sigchld()
                self._reap()
                if time.monotonic() >= deadline:
                    break
            if not self.children:
                return
        logger.warning("Timed out reaping zygote children %s", list(self.children))

    def _preload(self, name: str) -> None:
        try:
            import_module(name)
        except Exception:
            logger.warning(
                "Zygote could not import %s; processes will import it themselves.",
                name,
                exc_info=True,
            )
        if self.hazard is None and (hazard := _fork_hazard()) is not None:
            self.hazard = f"importing {name} {hazard}"
            logger.warning("Zygote disabled: %s", self.hazard)

    def _accept(self) -> None:
        try:
            conn = self.listener.accept()[0]
        except OSError:
            return
        fds: list[int] = []
        with conn:
            try:
                conn.settimeout(_TIMEOUT_S)
                peer_pid = _peer_pid(conn)
                msg, fds = _recv_msg(conn, _MAX_FDS)
                if msg["op"] == "fork":
                    env = self._environment_for(msg)
                    if isinstance(env, str):
                        _send_msg(
                            conn, {"error": env, "unavailable": self.hazard is not None}
                        )
                    else:
                        self._fork(conn, env, fds, peer_pid, msg.get("numa_binding"))
                elif msg["op"] == "descendants":
                    _send_msg(conn, {"pids": self._descendants(msg["pids"])})
            except Exception as e:
                # A malformed request must not take down the zygote.
                logger.debug("Dropped a zygote request: %r", e)
            finally:
                for fd in fds:
                    with contextlib.suppress(OSError):
                        os.close(fd)

    def _environment_for(self, request: dict) -> dict[str, str] | str:
        """The environment of the child to fork for `request`, or why there is
        none."""
        if self.hazard is not None:
            return self.hazard
        # The child runs our interpreter, not the one the requester's spawned
        # processes would run (e.g. another virtualenv's).
        executable = os.fsdecode(spawn.get_executable())
        if request["executable"] != executable:
            return f"it runs {executable}, not {request['executable']}"
        return _child_env(
            self.start_env,
            dict(os.environ),
            request["env"],
            (*_EXEC_TIME_ENV, *self.env_reads),
        )

    def _fork(
        self,
        conn: socket.socket,
        env: dict[str, str],
        fds: list[int],
        requester: int,
        numa_binding: NumaBinding | None,
    ) -> None:
        from vllm.utils.system_utils import _release_libgomp_threads

        data_r, status_w, tracker_fd, *passed = fds
        _flush_std_streams()
        _release_libgomp_threads()
        pid = os.fork()
        if pid == 0:
            code = 1
            try:
                conn.close()
                self.listener.close()
                self.selector.close()
                for fd in (self.alive_fd, self.sig_r, self.sig_w, status_w):
                    os.close(fd)
                for fd in self.children.values():
                    os.close(fd)
                code = self._run_child(env, data_r, tracker_fd, passed, numa_binding)
            except Exception:
                sys.excepthook(*sys.exc_info())
                sys.stderr.flush()
            finally:
                # Exit handlers run when a spawned process exits, but not on
                # os._exit.
                atexit._run_exitfuncs()
                os._exit(code)
        # The caller closes the other fds.
        fds.remove(status_w)
        self.children[pid] = status_w
        self.requesters[pid] = requester
        try:
            _send_msg(conn, {"pid": pid})
            write_signed(status_w, pid)
        except OSError:
            pass  # The requester is gone; the child sees EOF on its data pipe.

    def _run_child(
        self,
        env: dict[str, str],
        data_r: int,
        tracker_fd: int,
        passed: list[int],
        numa_binding: NumaBinding | None,
    ) -> int:
        import vllm.envs as envs

        signal.set_wakeup_fd(-1)
        signal.signal(signal.SIGCHLD, signal.SIG_DFL)
        signal.signal(signal.SIGINT, signal.default_int_handler)
        envs.__getattr__ = self.envs_getattr  # type: ignore[assignment]
        os.environ.clear()
        os.environ.update(env)
        if numa_binding is not None:
            _apply_numa_binding(numa_binding)
        # Torch sized its intra-op thread pool from the zygote's environment
        # when the zygote imported it; a spawned process sizes it from its own
        # (e.g. the share of CPUs EngineCore picks for each worker).
        torch = sys.modules.get("torch")
        num_threads = env.get("OMP_NUM_THREADS")
        if (
            torch is not None
            and num_threads is not None
            and num_threads != self.start_env.get("OMP_NUM_THREADS")
        ):
            with contextlib.suppress(ValueError):
                torch.set_num_threads(int(num_threads))
        global _inherited_fds
        _inherited_fds = passed
        resource_tracker._resource_tracker._fd = tracker_fd  # type: ignore[attr-defined]
        # The preloaded heap is shared with the zygote and its other children:
        # keep collections, and the page copies their writes cause, out of it.
        gc.freeze()
        gc.enable()
        return spawn._main(data_r, os.dup(data_r))  # type: ignore[attr-defined]

    def _descendants(self, pids: list[int]) -> list[int]:
        found: set[int] = set()
        owners = set(pids)
        while owners:
            owners = {
                pid
                for pid, requester in self.requesters.items()
                if requester in owners and pid not in found
            }
            found |= owners
        return sorted(found)

    def _reap(self) -> None:
        while True:
            try:
                pid, status = os.waitpid(-1, os.WNOHANG)
            except ChildProcessError:
                return
            if pid == 0:
                return
            self.requesters.pop(pid, None)
            if (fd := self.children.pop(pid, None)) is not None:
                with contextlib.suppress(OSError):
                    write_signed(fd, os.waitstatus_to_exitcode(status))
                os.close(fd)


def _serve(
    listener_fd: int, alive_fd: int, sock_dir: str, preload: list[str] | None
) -> None:
    """The zygote process's main function."""
    code = 0
    try:
        _ZygoteServer(listener_fd, alive_fd, preload).run()
    except BaseException:
        logger.exception("Zygote failed")
        code = 1
    finally:
        shutil.rmtree(sock_dir, ignore_errors=True)
        # Skip the preloaded modules' exit handlers: they belong to children.
        os._exit(code)

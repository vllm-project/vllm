#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""mpolset - set memory policy and execute a command

Usage: mpolset [options] command [args...]

Options:
 -h, --help            show this help message and exit.
 -i, --interleave SPEC set memory policy to MPOL_INTERLEAVED on nodes
                       specified by SPEC. SPEC can be one of:
                       - "all": all nodes
                       - "local:nodes": nodes within distance 10 from CPUs
                       - "local:socket": nodes within distance 19 from CPUs
                       - "dist:14": nodes within distance 14 from CPUs
                       - "mems:0,1,2": list of node numbers
                       (see numactl -H for distance matrix)
                       The default is to not set memory policy.
 -o, --override        override existing memory policy if one is already set.
 -E <ignore|exit>      ignore: exec command even if setting memory policy fails
                       exit: exit with error if setting memory policy fails
                       (default: exit)
 -q, --quiet           suppress all output from mpolset.
 -v, --verbose         increase verbosity level.
 -t, --test            run a test function that hogs memory
                       without running a command.
"""

import ctypes
import glob
import logging
import os
import sys
import time

_env_mpol_interleave = "MPOL_INTERLEAVE"

_g_logfile = sys.stderr

log = logging.getLogger(__name__).info


def debug(level, msg):
    if level < _g_debug_level:
        log(msg)


def set_debug_level(level):
    global _g_debug_level
    _g_debug_level = level


## mempolicy.h

# Policies
MPOL_DEFAULT = 0
MPOL_PREFERRED = 1
MPOL_BIND = 2
MPOL_INTERLEAVED = 3
MPOL_LOCAL = 4
MPOL_PREFERRED_MANY = 5
MPOL_WEIGHTED_INTERLEAVE = 6
MPOL_MAX = 7

# Flags for get_mempolicy
MPOL_F_NODE = (1 << 0)
MPOL_F_ADDR = (1 << 1)
MPOL_F_MEMS_ALLOWED = (1 < 2)

## syscall.h
_SYS_SET_MEMPOLICY = 238
_SYS_GET_MEMPOLICY = 239

_strmpol = {}
for _mpol in globals().copy():
    if _mpol.startswith("MPOL_") and not _mpol.startswith("MPOL_F_"):
        _strmpol[globals()[_mpol]] = _mpol

_g_libc = None


def _libc():
    """Return libc with necessary function argtypes and restypes set."""
    global _g_libc
    if _g_libc is None:
        if os.name != 'posix':
            raise NotImplementedError(
                "This module only works on POSIX systems")
        _g_libc = ctypes.CDLL(None, use_errno=True)
        _g_libc.syscall.restype = ctypes.c_long
        _g_libc.sched_getaffinity.argtypes = (ctypes.c_int, ctypes.c_size_t,
                                              ctypes.c_void_p)
        _g_libc.sched_getaffinity.restype = ctypes.c_int
    return _g_libc


def set_mempolicy(mpol, nodes):
    """Set the memory policy of the calling process."""
    libc = _libc()
    if max(nodes) > 63 or min(nodes) < 0:
        raise ValueError("set_mempolicy: not all nodes are between 0 and 63")
    c_mpol = ctypes.c_int(mpol)
    c_maxnode = 8 * ctypes.sizeof(ctypes.c_ulong) - 1
    nodemask = 0
    for n in nodes:
        nodemask |= 1 << n
    c_nodemask = ctypes.c_ulong(nodemask)
    _g_libc.syscall.argtypes = (ctypes.c_long, ctypes.c_int, ctypes.c_void_p,
                                ctypes.c_ulong)
    result = libc.syscall(_SYS_SET_MEMPOLICY, c_mpol, ctypes.byref(c_nodemask),
                          c_maxnode)
    if result != 0:
        raise OSError(
            ctypes.get_errno(),
            f"set_mempolicy() returned {os.strerror(ctypes.get_errno())}")


def get_mempolicy(flags):
    """Get the memory policy of the calling process."""
    libc = _libc()
    c_mode = ctypes.c_int()
    c_nodemask = ctypes.c_ulong()
    c_maxnode = 8 * ctypes.sizeof(ctypes.c_ulong) - 1
    c_flags = ctypes.c_ulong(flags)
    _g_libc.syscall.argtypes = (ctypes.c_long, ctypes.c_void_p,
                                ctypes.c_void_p, ctypes.c_ulong,
                                ctypes.c_void_p, ctypes.c_ulong)
    result = libc.syscall(_SYS_GET_MEMPOLICY, ctypes.byref(c_mode),
                          ctypes.byref(c_nodemask), c_maxnode, 0, c_flags)
    if result != 0:
        raise OSError(ctypes.get_errno(),
                      f"get_mempolicy failed with errno {ctypes.get_errno()}")
    nodes = []
    for i in range(c_maxnode):
        if c_nodemask.value & (1 << i):
            nodes.append(i)
    return c_mode.value, nodes


def sched_getaffinity():
    """Return the list of CPUs in the current process's affinity mask."""
    libc = _libc()
    c_cpu_set = ctypes.create_string_buffer(80)
    c_cpuset_size = ctypes.sizeof(c_cpu_set)
    c_pid = 0
    result = libc.sched_getaffinity(c_pid, c_cpuset_size,
                                    ctypes.byref(c_cpu_set))
    if result != 0:
        errno = ctypes.get_errno()
        raise OSError(
            errno, f"sched_getaffinity returned {errno}: {os.strerror(errno)}")
    cpus = []
    for char_idx, char in enumerate(c_cpu_set.raw):
        for bit_idx in range(8):
            if char & (1 << bit_idx):
                cpus.append(char_idx * 8 + bit_idx)
    return cpus


def _listset2list(s):
    """Parse list set syntax ("0,61-63,2") into a list ([0, 61, 62, 63, 2])"""
    lst = []
    if s == "":
        return lst
    for comma in s.split(","):
        rng = comma.split("-")
        if len(rng) == 1:
            lst.append(int(rng[0]))
            continue
        for i in range(int(rng[0]), int(rng[1]) + 1):
            lst.append(i)
    return lst


def _list2listset(lst):
    """Write list elements ([0, 61, 62, 63, 2]) as a string ("0,2,61-63")"""
    lst = sorted(lst)
    s = []
    i = 0
    while i < len(lst):
        start, end = i, i
        while end + 1 < len(lst) and lst[end + 1] == lst[end] + 1:
            end += 1
        if start == end:
            s.append(str(lst[start]))
        else:
            s.append(f"{lst[start]}-{lst[end]}")
        i = end + 1
    return ",".join(s)


_g_cpu_node, _g_node_cpu, _g_node_node_dist = None, None, None


def _node_topology():
    """Return a tuple of three dictionaries:
        - cpu_node: {cpuid: nodeid}
        - node_cpu: {nodeid: set(cpuid)}
        - node_node_dist: {from_nodeid: {to_nodeid: distance_int}}"""
    global _g_cpu_node, _g_node_cpu, _g_node_node_dist
    if _g_cpu_node is None:
        parse_nodeid = lambda d: int(d[len("/sys/devices/system/node/node"):
                                       -len("/cpulist")])
        cpu_node = {}
        node_cpu = {}
        node_node_dist = {}
        for d in glob.glob("/sys/devices/system/node/node*/cpulist"):
            nodeid = parse_nodeid(d)
            if nodeid not in node_cpu:
                node_cpu[nodeid] = set()
                node_node_dist[nodeid] = {}
            with open(d) as f:
                cpulist = _listset2list(f.read())
            for cpuid in cpulist:
                cpu_node[cpuid] = nodeid
                node_cpu[nodeid].add(cpuid)
            with open(d.replace("cpulist", "distance")) as f:
                for target_node_id, dist in enumerate(
                        int(n) for n in f.read().strip().split()):
                    node_node_dist[nodeid][target_node_id] = dist
        _g_cpu_node = cpu_node
        _g_node_cpu = node_cpu
        _g_node_node_dist = node_node_dist
    return _g_cpu_node, _g_node_cpu, _g_node_node_dist


def _nodes_within_dist(max_node_dist, from_cpus):
    """Return a list of nodeids within max_node_dist from the closest
    node of from_cpus."""
    cpu_node, node_cpu, node_node_dist = _node_topology()
    nodes = set()
    source_nodes = set(cpu_node[cpu] for cpu in from_cpus)
    debug(
        1, f"looking for nodes within distance {max_node_dist} "
        f"from {sorted(source_nodes)} of CPUs {_list2listset(from_cpus)}")
    for source_node in source_nodes:
        for target_node, dist in node_node_dist[source_node].items():
            if dist <= max_node_dist:
                nodes.add(target_node)
    return sorted(set.union(source_nodes, nodes))


def set_mpol_interleaved(interleaved_spec):
    """Set memory policy to MPOL_INTERLEAVED on nodes specified by
    interleaved_spec. Supported spec formats:
    - "all": interleave on all nodes
    - "mems:0,1,2": interleave on nodes 0, 1, and 2
    - "dist:14": interleave on nodes within distance 14 from allowed CPUs
    - "local:node": interleave on nodes within distance 10 from allowed CPUs
    - "local:socket": interleave on nodes within distance 19 from allowed CPUs
    The allowed CPUs are the current process's affinity mask.

    Raises error if the spec is invalid or if set_mempolicy fails.
    """
    cpu_node, node_cpu, node_node_dist = _node_topology()
    nodes = None
    mpol = MPOL_INTERLEAVED
    if interleaved_spec == "all":
        nodes = sorted(node_cpu.keys())
    elif interleaved_spec.startswith("mems:"):
        nodes = _listset2list(interleaved_spec[len("mems:"):])
    elif interleaved_spec.startswith("dist:"):
        nodes = _nodes_within_dist(int(interleaved_spec[len("dist:"):]),
                                   sched_getaffinity())
    elif interleaved_spec in ["local:node"]:
        nodes = _nodes_within_dist(10, sched_getaffinity())
    elif interleaved_spec in ["local:socket"]:
        nodes = _nodes_within_dist(19, sched_getaffinity())
    if nodes is None:
        raise ValueError(f"Invalid interleaved spec: \"{interleaved_spec}\"")
    debug(
        1,
        f"set_mempolicy {_strmpol[mpol]} ({mpol}) on {_list2listset(nodes)}")
    set_mempolicy(mpol, nodes)


def configure_memory_policy(mpol_spec, override=False):
    """Configure memory policy. Returns False if configuring failed."""
    try:
        mpol_mode, mpol_nodes = get_mempolicy(0)
    except Exception:
        log("skip setting memory policy: cannot read existing policy ({err})")
        return False
    existing_policy = (f"{_strmpol.get(mpol_mode, 'N/A')} ({mpol_mode}) "
                       f"on nodes: {_list2listset(mpol_nodes)}")
    debug(1, f"existing memory policy: {existing_policy}")
    if mpol_spec == "":
        debug(0, f"no memory policy to set, keep policy: {existing_policy}")
        return True
    if not override and mpol_mode != MPOL_DEFAULT:
        debug(0, f"not overriding existing memory policy: {existing_policy}")
        return True
    if mpol_spec.startswith("interleave:"):
        try:
            set_mpol_interleaved(mpol_spec[len("interleave:"):])
        except Exception as err:
            log(f"failed to set memory policy: {err}")
            return False
    else:
        log(f"unsupported memory policy specification: {mpol_spec}")
        return False
    try:
        mpol_mode, mpol_nodes = get_mempolicy(0)
        debug(
            0, f"effective memory policy: {_strmpol.get(mpol_mode, 'N/A')} "
            f"({mpol_mode}) "
            f"on nodes: {_list2listset(mpol_nodes)}")
    except Exception as err:
        log(f"failed to get effective memory policy after configuring: {err}")
    return True


def _hog():
    import subprocess
    s = []
    for i in range(4):
        log(f"hogging memory {i}")
        s.append(chr(ord('a') + i) * (1024 * 1024 * 100))
        time.sleep(1.0)
        subprocess.call(["stdbuf", "-oL", "numastat", "-p", str(os.getpid())])


def _main(argv):
    global _g_debug_level
    global log
    import getopt

    # Read command line options until the first option that does not
    # start with "-".
    opt_override = False
    opt_error_action = "exit"
    opt_interleave = os.getenv(_env_mpol_interleave, "")
    opt_test = False
    opt_error_action = "exit"
    try:
        opts, remainder = getopt.getopt(
            argv, "hi:toE:qv",
            ["help", "interleave=", "test", "override", "quiet", "verbose"])
        for opt, arg in opts:
            if opt in ("-h", "--help"):
                print(__doc__)
                return 0
            elif opt in ("-i", "--interleave"):
                opt_interleave = arg
            elif opt in ("-t", "--test"):
                opt_test = True
            elif opt in ("-q", "--quiet"):
                log = lambda msg: None
            elif opt in ("-E", ):
                opt_error_action = arg.lower()
                if opt_error_action not in ["exit", "ignore"]:
                    raise ValueError("invalid action on error: "
                                     f"-E: {opt_error_action}, "
                                     "expected: exit or ignore")
            elif opt in ("-o", "--override"):
                opt_override = True
            elif opt in ("-v", "--verbose"):
                _g_debug_level += 1
            else:
                raise ValueError(f"unhandled option: {opt}")
    except Exception as err:
        log(f"error: {err}")
        return 1

    mpol_spec = ""
    if opt_interleave:
        mpol_spec = "interleave:" + opt_interleave

    if mpol_spec:
        success = configure_memory_policy(mpol_spec, override=opt_override)
        if (not success) and opt_error_action == "exit":
            return 1

    if opt_test:
        _hog()
        return 0

    if not remainder:
        log("missing: command to execute")
        return 1

    debug(1, f"executing: {remainder}")
    try:
        os.execvp(remainder[0], remainder)
    except Exception as err:
        log(f"execvp failed: {err}")
        return 1


_g_debug_level = int(os.getenv("MPOL_DEBUG", "0"))
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    log = logging.getLogger("mpolset").info
    sys.exit(_main(sys.argv[1:]))
else:
    log = logging.getLogger(__name__).info

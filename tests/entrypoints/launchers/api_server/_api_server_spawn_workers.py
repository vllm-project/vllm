# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Spawn worker targets kept free of heavy imports.

``multiprocessing`` with the ``spawn`` start method re-imports the module that
defines a process target in the child. Housing these stubs in a stdlib-only
module keeps child startup fast and deterministic, instead of paying a multi-
second ``import vllm`` before the child can run.
"""


def exit_before_report_worker(listen_address, sock, args, client_config=None):
    """Exit immediately without touching ``actual_address_pipe``."""
    return


def report_listener_worker(listen_address, sock, args, client_config=None):
    """Report the kernel listener identity without starting an engine."""
    import os

    sock.listen()
    pipe = client_config["actual_address_pipe"]
    pipe.send((sock.getsockname(), os.fstat(sock.fileno()).st_ino))
    pipe.close()
    sock.close()

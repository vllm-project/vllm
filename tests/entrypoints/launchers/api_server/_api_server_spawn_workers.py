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

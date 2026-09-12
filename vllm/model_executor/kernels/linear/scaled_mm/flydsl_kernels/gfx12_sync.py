# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Fine-grained LDS synchronization helpers for gfx12+ kernels."""

from flydsl.expr import rocdl

WORKGROUP_BARRIER_ID = -1


def lds_wait(outstanding=0):
    """Wait until no more than ``outstanding`` DS operations remain."""
    rocdl.s_wait_dscnt(outstanding)


def lds_fence_signal(outstanding=0):
    """Make this wave's required LDS writes visible, then signal peers."""
    lds_wait(outstanding)
    rocdl.s_barrier_signal(WORKGROUP_BARRIER_ID)


def lds_fence_wait():
    """Wait until every wave has signaled the matching LDS fence."""
    rocdl.s_barrier_wait(WORKGROUP_BARRIER_ID)

# SPDX-License-Identifier: Apache-2.0

from vllm.core.scheduler import Scheduler


class DummyScheduler(Scheduler):

    def schedule(self):
        raise Exception("Exception raised by DummyScheduler")

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import threading

from vllm.v1.offloading.abstract import LoadStoreSpec
from vllm.v1.offloading.worker.worker import (OffloadingQueueManager,
                                              TransferSpec)


class LoadStoreSpec1(LoadStoreSpec):

    def __init__(self, success: bool = True, exception: bool = False):
        self.called_event = threading.Event()
        self.finished_event = threading.Event()
        self.success = success
        self.exception = exception

    @staticmethod
    def medium() -> str:
        return "1"


class LoadStoreSpec2(LoadStoreSpec):

    @staticmethod
    def medium() -> str:
        return "2"


def transfer_function_1_to_2(transfer_spec: TransferSpec) -> bool:
    src, dst = transfer_spec
    assert isinstance(src, LoadStoreSpec1)
    assert isinstance(dst, LoadStoreSpec2)
    src.called_event.set()
    src.finished_event.wait()
    if src.exception:
        raise Exception()
    return src.success


def transfer_function_2_to_1(transfer_spec: TransferSpec) -> bool:
    src, dst = transfer_spec
    assert isinstance(src, LoadStoreSpec2)
    assert isinstance(dst, LoadStoreSpec1)
    dst.called_event.set()
    dst.finished_event.wait()
    if dst.exception:
        raise Exception()
    return dst.success


def test_offloading_queue_manager():
    """
    Tests OffloadingQueueManager with 2 workers.
    One worker performs 1->2 transfers, and the other handles 2->1.
    """
    offloading_queue_manager = OffloadingQueueManager()
    offloading_queue_manager.register_worker(LoadStoreSpec1, LoadStoreSpec2,
                                             transfer_function_1_to_2)
    offloading_queue_manager.register_worker(LoadStoreSpec2, LoadStoreSpec1,
                                             transfer_function_2_to_1)

    # 1st transfer 1->2 (exception)
    src1 = LoadStoreSpec1(exception=True)
    dst1 = LoadStoreSpec2()
    job_id = offloading_queue_manager.transfer_async((src1, dst1))
    assert job_id == 0

    # 2ed transfer 1->2 (failure)
    src2 = LoadStoreSpec1(success=False)
    dst2 = LoadStoreSpec2()
    job_id = offloading_queue_manager.transfer_async((src2, dst2))
    assert job_id == 1

    # 3rd transfer 1->2 (success)
    src3 = LoadStoreSpec1()
    dst3 = LoadStoreSpec2()
    job_id = offloading_queue_manager.transfer_async((src3, dst3))
    assert job_id == 2

    # 4th transfer 2->1
    src4 = LoadStoreSpec2()
    dst4 = LoadStoreSpec1()
    job_id = offloading_queue_manager.transfer_async((src4, dst4))
    assert job_id == 3

    # 1st transfer started
    assert src1.called_event.wait(timeout=1)

    # 4th transfer started
    assert dst4.called_event.wait(timeout=1)

    # 2ed transfer have not started (blocked by 1st)
    assert not src2.called_event.is_set()

    # no transfer completed yet
    assert offloading_queue_manager.get_finished() == []

    # complete 1st transfer
    src1.finished_event.set()

    # 2ed transfer started
    src2.called_event.wait(timeout=1)

    # 1st transfer finished with failure (exception)
    assert offloading_queue_manager.get_finished() == [(0, False)]

    # complete 2ed, 3rd and 4th transfers
    src2.finished_event.set()
    src3.finished_event.set()
    dst4.finished_event.set()

    # 5th transfer 1->2
    src5 = LoadStoreSpec1()
    dst5 = LoadStoreSpec2()
    job_id = offloading_queue_manager.transfer_async((src5, dst5))
    assert job_id == 4

    # 6th transfer 2->1
    src6 = LoadStoreSpec2()
    dst6 = LoadStoreSpec1()
    job_id = offloading_queue_manager.transfer_async((src6, dst6))
    assert job_id == 5

    # 5th and 6th transfers started
    assert src5.called_event.wait(timeout=1)
    assert dst6.called_event.wait(timeout=1)

    # verify result of 2ed, 3rd and 4th transfers
    assert (sorted(offloading_queue_manager.get_finished()) == [(1, False),
                                                                (2, True),
                                                                (3, True)])

    # complete 5th and 6th transfers
    src5.finished_event.set()
    dst6.finished_event.set()

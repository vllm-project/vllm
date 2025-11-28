# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.v1.kv_offload.abstract import LoadStoreSpec
from vllm.v1.kv_offload.worker.worker import (
    OffloadingHandler,
    OffloadingWorker,
    TransferResult,
    TransferSpec,
)


class LoadStoreSpec1(LoadStoreSpec):
    def __init__(
        self,
        submit_success: bool = True,
        async_success: bool = True,
        exception: bool = False,
    ):
        self.finished = False
        self.submit_success = submit_success
        self.async_success = async_success
        self.exception = exception

    @staticmethod
    def medium() -> str:
        return "1"

    def __repr__(self):
        return f"{self.medium()}: {id(self)}"


class LoadStoreSpec2(LoadStoreSpec):
    @staticmethod
    def medium() -> str:
        return "2"

    def __repr__(self):
        return f"{self.medium()}: {id(self)}"


class OffloadingHandler1To2(OffloadingHandler):
    def __init__(self):
        self.transfers: dict[int, LoadStoreSpec1] = {}

    def transfer_async(self, job_id: int, spec: TransferSpec) -> bool:
        src, dst = spec
        assert isinstance(src, LoadStoreSpec1)
        assert isinstance(dst, LoadStoreSpec2)

        if src.exception:
            raise Exception("An expected exception. Don't worry!")
        if not src.submit_success:
            return False

        self.transfers[job_id] = src
        return True

    def get_finished(self) -> list[TransferResult]:
        finished = []
        for job_id, spec in list(self.transfers.items()):
            if spec.finished:
                finished.append((job_id, spec.async_success))
                del self.transfers[job_id]
        return finished


class OffloadingHandler2To1(OffloadingHandler):
    def __init__(self):
        self.transfers: dict[int, LoadStoreSpec1] = {}

    def transfer_async(self, job_id: int, spec: TransferSpec) -> bool:
        src, dst = spec
        assert isinstance(src, LoadStoreSpec2)
        assert isinstance(dst, LoadStoreSpec1)

        self.transfers[job_id] = dst
        return True

    def get_finished(self) -> list[TransferResult]:
        finished = []
        for job_id, spec in list(self.transfers.items()):
            if spec.finished:
                finished.append((job_id, spec.async_success))
                del self.transfers[job_id]
        return finished


def test_offloading_worker():
    """
    Tests OffloadingWorker with 2 handlers.
    One handler performs 1->2 transfers, and the other handles 2->1.
    """
    worker = OffloadingWorker()
    handler1to2 = OffloadingHandler1To2()
    handler2to1 = OffloadingHandler2To1()
    worker.register_handler(LoadStoreSpec1, LoadStoreSpec2, handler1to2)
    worker.register_handler(LoadStoreSpec2, LoadStoreSpec1, handler2to1)

    # 1st transfer 1->2 (exception)
    src1 = LoadStoreSpec1(exception=True)
    dst1 = LoadStoreSpec2()
    assert not worker.transfer_async(1, (src1, dst1))

    # 2ed transfer 1->2 (failure to submit)
    src2 = LoadStoreSpec1(submit_success=False)
    dst2 = LoadStoreSpec2()
    assert not worker.transfer_async(2, (src2, dst2))

    # 3rd transfer 1->2 (failure)
    src3 = LoadStoreSpec1(async_success=False)
    dst3 = LoadStoreSpec2()
    assert worker.transfer_async(3, (src3, dst3))

    # 4th transfer 1->2 (success)
    src4 = LoadStoreSpec1()
    dst4 = LoadStoreSpec2()
    worker.transfer_async(4, (src4, dst4))
    assert set(handler1to2.transfers.keys()) == {3, 4}

    # 5th transfer 2->1
    src5 = LoadStoreSpec2()
    dst5 = LoadStoreSpec1()
    worker.transfer_async(5, (src5, dst5))
    assert set(handler2to1.transfers.keys()) == {5}

    # no transfer completed yet
    assert worker.get_finished() == []

    # complete 3rd, 4th
    src3.finished = True
    src4.finished = True

    # 6th transfer 1->2
    src6 = LoadStoreSpec1()
    dst6 = LoadStoreSpec2()
    worker.transfer_async(6, (src6, dst6))

    # 7th transfer 2->1
    src7 = LoadStoreSpec2()
    dst7 = LoadStoreSpec1()
    worker.transfer_async(7, (src7, dst7))

    # 6th and 7th transfers started
    assert 6 in handler1to2.transfers
    assert 7 in handler2to1.transfers

    # verify result of 3rd and 4th transfers
    assert sorted(worker.get_finished()) == [(3, False), (4, True)]

    # complete 6th and 7th transfers
    src6.finished = True
    dst7.finished = True
    assert sorted(worker.get_finished()) == [(6, True), (7, True)]

# Copyright (c) 2025, Tri Dao.

# import math
from dataclasses import dataclass

from cutlass import Boolean, Int32, const_expr
from cutlass.cutlass_dsl import if_generate
from cutlass.pipeline import PipelineState, PipelineUserType
from cutlass.pipeline import PipelineTmaAsync as PipelineTmaAsyncOg
from cutlass.pipeline import PipelineTmaUmma as PipelineTmaUmmaOg


class PipelineStateSimple:
    """
    Pipeline state contains an index and phase bit corresponding to the current position in the circular buffer.
    Use a single Int32 to store both the index and phase bit, then we use divmod to get the
    index and phase. If stages is a power of 2, divmod turns into bit twiddling.
    """

    def __init__(self, stages: int, phase_index: Int32):
        # assert stages < 2**16
        # self._log_stages = int(math.log2(stages))
        # assert 1 << self._log_stages == stages, "Number of stages must be a power of 2."
        self._stages = stages
        self._phase_index = phase_index

    def clone(self) -> "PipelineStateSimple":
        return PipelineStateSimple(self.stages, self._phase_index)

    @property
    def stages(self) -> int:
        # return 1 << self._log_stages
        return self._stages

    @property
    def index(self) -> Int32:
        # return self._phase_index & 0xFFFF
        # return self._phase_index & ((1 << self._log_stages) - 1)
        if const_expr(self._stages == 1):
            return Int32(0)
        else:
            return self._phase_index % self._stages

    @property
    def phase(self) -> Int32:
        # return self._phase_index >> 16
        # PTX docs say that the phase parity needs to be 0 or 1, so by right we need to
        # take modulo 2. But in practice just passing the phase in without modulo works fine.
        # return (self._phase_index >> self._log_stages) % 2
        # return self._phase_index >> self._log_stages
        if const_expr(self._stages == 1):
            return self._phase_index
        else:
            return self._phase_index // self._stages

    def advance(self):
        if const_expr(self._stages == 1):
            self._phase_index ^= 1
        else:
            self._phase_index += 1

        # def then_body(phase_index):
        #     # XOR the phase bit and set the index to 0
        #     return (phase_index & 0xFFFF0000) ^ (1 << 16)

        # def else_body(phase_index):
        #     return phase_index

        # self._phase_index = if_generate(
        #     (self._phase_index & 0xFFFF) == self.stages,
        #     then_body,
        #     else_body,
        #     [self._phase_index],
        #     [Int32],
        # )

    def __extract_mlir_values__(self):
        phase_index = self._phase_index
        return [phase_index.ir_value()]

    def __new_from_mlir_values__(self, values):
        return PipelineStateSimple(self.stages, Int32(values[0]))


def make_pipeline_state(type: PipelineUserType, stages: int):
    """
    Creates a pipeline state. Producers are assumed to start with an empty buffer and have a flipped phase bit of 1.
    """
    if type is PipelineUserType.Producer:
        # return PipelineStateSimple(stages, Int32(1 << 16))
        return PipelineStateSimple(stages, Int32(stages))
    elif type is PipelineUserType.Consumer:
        return PipelineStateSimple(stages, Int32(0))
    else:
        assert False, (
            "Error: invalid PipelineUserType specified for make_pipeline_state."
        )


@dataclass(frozen=True)
class PipelineTmaAsync(PipelineTmaAsyncOg):
    """
    Override producer_acquire to take in extra_tx_count parameter.
    """

    @staticmethod
    def create(*args, **kwargs):
        obj = PipelineTmaAsyncOg.create(*args, **kwargs)
        # Can't assign to __class__ directly since the dataclass is frozen
        # obj.__class__ = PipelineTmaAsync
        object.__setattr__(obj, "__class__", PipelineTmaAsync)
        return obj

    def producer_acquire(
        self,
        state: PipelineState,
        try_acquire_token: Boolean | None = None,
        extra_tx_count: int = 0,
        *,
        loc=None,
        ip=None,
    ):
        """
        TMA producer commit conditionally waits on buffer empty and sets the transaction barrier for leader threadblocks.
        """
        if_generate(
            try_acquire_token is None or try_acquire_token == 0,
            lambda: self.sync_object_empty.wait(
                state.index, state.phase, loc=loc, ip=ip
            ),
            loc=loc,
            ip=ip,
        )
        if const_expr(extra_tx_count == 0):
            self.sync_object_full.arrive(
                state.index, self.producer_mask, loc=loc, ip=ip
            )
        else:
            tx_count = self.sync_object_full.tx_count + extra_tx_count
            self.sync_object_full.arrive_and_expect_tx(
                state.index, tx_count, loc=loc, ip=ip
            )


@dataclass(frozen=True)
class PipelineTmaUmma(PipelineTmaUmmaOg):
    """
    Override producer_acquire to take in extra_tx_count parameter.
    """

    @staticmethod
    def create(*args, **kwargs):
        obj = PipelineTmaUmmaOg.create(*args, **kwargs)
        # Can't assign to __class__ directly since the dataclass is frozen
        # obj.__class__ = PipelineTmaUmma
        object.__setattr__(obj, "__class__", PipelineTmaUmma)
        return obj

    def producer_acquire(
        self,
        state: PipelineState,
        try_acquire_token: Boolean | None = None,
        extra_tx_count: int = 0,
        *,
        loc=None,
        ip=None,
    ):
        """
        TMA producer commit conditionally waits on buffer empty and sets the transaction barrier for leader threadblocks.
        """
        if_generate(
            try_acquire_token is None or try_acquire_token == 0,
            lambda: self.sync_object_empty.wait(
                state.index, state.phase, loc=loc, ip=ip
            ),
            loc=loc,
            ip=ip,
        )
        if const_expr(extra_tx_count == 0):
            if_generate(
                self.is_leader_cta,
                lambda: self.sync_object_full.arrive(
                    state.index, self.producer_mask, loc=loc, ip=ip
                ),
                loc=loc,
                ip=ip,
            )
        else:
            tx_count = self.sync_object_full.tx_count + extra_tx_count
            if_generate(
                self.is_leader_cta,
                lambda: self.sync_object_full.arrive_and_expect_tx(
                    state.index, tx_count, loc=loc, ip=ip
                ),
                loc=loc,
                ip=ip,
            )

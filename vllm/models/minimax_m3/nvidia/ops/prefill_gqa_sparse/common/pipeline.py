# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa
# import math
from dataclasses import dataclass

import cutlass.cute as cute
import cutlass.pipeline as cutlass_pipeline
from cutlass import Boolean, Int32, const_expr
from cutlass.cutlass_dsl import dsl_user_op, if_generate
from cutlass.pipeline import NamedBarrier as NamedBarrierOg
from cutlass.pipeline import PipelineAsync as PipelineAsyncOg
from cutlass.pipeline import PipelineAsyncUmma as PipelineAsyncUmmaOg
from cutlass.pipeline import PipelineState, PipelineUserType
from cutlass.pipeline import PipelineTmaAsync as PipelineTmaAsyncOg
from cutlass.pipeline import PipelineTmaUmma as PipelineTmaUmmaOg
from cutlass.pipeline import PipelineUmmaAsync as PipelineUmmaAsyncOg


def make_pipeline_state(type: PipelineUserType, stages: int):
    """Compatibility wrapper for FA-style helpers now vendored into common."""
    return cutlass_pipeline.make_pipeline_state(type, stages)


@dataclass(frozen=True)
class NamedBarrier(NamedBarrierOg):
    @staticmethod
    def create(*args, **kwargs):
        obj = NamedBarrierOg.create(*args, **kwargs)
        # Can't assign to __class__ directly since the dataclass is frozen
        object.__setattr__(obj, "__class__", NamedBarrier)
        return obj

    @dsl_user_op
    def arrive_w_index(self, index: Int32, *, loc=None, ip=None) -> None:
        """
        The aligned flavor of arrive is used when all threads in the CTA will execute the
        same instruction. See PTX documentation.
        """
        cute.arch.barrier_arrive(
            barrier_id=self.barrier_id + index,
            number_of_threads=self.num_threads,
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def arrive_and_wait_w_index(self, index: Int32, *, loc=None, ip=None) -> None:
        cute.arch.barrier(
            barrier_id=self.barrier_id + index,
            number_of_threads=self.num_threads,
            loc=loc,
            ip=ip,
        )


@dataclass(frozen=True)
class PipelineAsync(PipelineAsyncOg):
    @staticmethod
    def create(*args, **kwargs):
        obj = PipelineAsyncOg.create(*args, **kwargs)
        # Can't assign to __class__ directly since the dataclass is frozen
        # obj.__class__ = PipelineAsync
        object.__setattr__(obj, "__class__", PipelineAsync)
        return obj

    @dsl_user_op
    def producer_acquire_w_index_phase(
        self,
        index: Int32,
        phase: Int32,
        try_acquire_token: Boolean | None = None,
        *,
        loc=None,
        ip=None,
    ):
        if_generate(
            try_acquire_token is None or try_acquire_token == 0,
            lambda: self.sync_object_empty.wait(index, phase, loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def producer_commit_w_index(self, index: Int32, *, loc=None, ip=None):
        self.sync_object_full.arrive(index, self.producer_mask, loc=loc, ip=ip)

    @dsl_user_op
    def consumer_wait_w_index_phase(
        self,
        index: Int32,
        phase: Int32,
        try_wait_token: Boolean | None = None,
        *,
        loc=None,
        ip=None,
    ):
        if_generate(
            try_wait_token is None or try_wait_token == 0,
            lambda: self.sync_object_full.wait(index, phase, loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def consumer_release_w_index(self, index: Int32, *, loc=None, ip=None):
        self.sync_object_empty.arrive(index, self.consumer_mask, loc=loc, ip=ip)


@dataclass(frozen=True)
class PipelineTmaAsync(PipelineTmaAsyncOg):
    """
    Override producer_acquire to take in extra_tx_count parameter.
    """

    @staticmethod
    def create(*args, **kwargs):
        obj = PipelineTmaAsyncOg.create(*args, **kwargs)
        # Can't assign to __class__ directly since the dataclass is frozen
        object.__setattr__(obj, "__class__", PipelineTmaAsync)
        return obj

    @dsl_user_op
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

    @dsl_user_op
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

    @dsl_user_op
    def producer_acquire_w_index_phase(
        self,
        index: Int32,
        phase: Int32,
        try_acquire_token: Boolean | None = None,
        *,
        loc=None,
        ip=None,
    ):
        """
        TMA producer commit conditionally waits on buffer empty and sets the transaction barrier for leader threadblocks.
        """
        if_generate(
            try_acquire_token is None or try_acquire_token == 0,
            lambda: self.sync_object_empty.wait(index, phase, loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )
        if_generate(
            self.is_leader_cta,
            lambda: self.sync_object_full.arrive(
                index, self.producer_mask, loc=loc, ip=ip
            ),
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def consumer_wait_w_index_phase(
        self,
        index: Int32,
        phase: Int32,
        try_wait_token: Boolean | None = None,
        *,
        loc=None,
        ip=None,
    ):
        if_generate(
            try_wait_token is None or try_wait_token == 0,
            lambda: self.sync_object_full.wait(index, phase, loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def consumer_release_w_index(self, index: Int32, *, loc=None, ip=None):
        """
        UMMA consumer release buffer empty, cta_group needs to be provided.
        """
        self.sync_object_empty.arrive(
            index, self.consumer_mask, self.cta_group, loc=loc, ip=ip
        )


@dataclass(frozen=True)
class PipelineUmmaAsync(PipelineUmmaAsyncOg):
    @staticmethod
    def create(*args, **kwargs):
        obj = PipelineUmmaAsyncOg.create(*args, **kwargs)
        # Can't assign to __class__ directly since the dataclass is frozen
        object.__setattr__(obj, "__class__", PipelineUmmaAsync)
        return obj

    @dsl_user_op
    def producer_acquire_w_index_phase(
        self,
        index: Int32,
        phase: Int32,
        try_acquire_token: Boolean | None = None,
        *,
        loc=None,
        ip=None,
    ):
        if_generate(
            try_acquire_token is None or try_acquire_token == 0,
            lambda: self.sync_object_empty.wait(index, phase, loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def producer_commit_w_index(self, index: Int32, *, loc=None, ip=None):
        """
        UMMA producer commit buffer full, cta_group needs to be provided.
        """
        self.sync_object_full.arrive(
            index, self.producer_mask, self.cta_group, loc=loc, ip=ip
        )

    @dsl_user_op
    def consumer_wait_w_index_phase(
        self,
        index: Int32,
        phase: Int32,
        try_wait_token: Boolean | None = None,
        *,
        loc=None,
        ip=None,
    ):
        if_generate(
            try_wait_token is None or try_wait_token == 0,
            lambda: self.sync_object_full.wait(index, phase, loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def consumer_release_w_index(self, index: Int32, *, loc=None, ip=None):
        self.sync_object_empty.arrive(index, self.consumer_mask, loc=loc, ip=ip)


@dataclass(frozen=True)
class PipelineAsyncUmma(PipelineAsyncUmmaOg):
    @staticmethod
    def create(*args, **kwargs):
        obj = PipelineAsyncUmmaOg.create(*args, **kwargs)
        # Can't assign to __class__ directly since the dataclass is frozen
        object.__setattr__(obj, "__class__", PipelineAsyncUmma)
        return obj

    @dsl_user_op
    def producer_acquire_w_index_phase(
        self,
        index: Int32,
        phase: Int32,
        try_acquire_token: Boolean | None = None,
        *,
        loc=None,
        ip=None,
    ):
        if_generate(
            try_acquire_token is None or try_acquire_token == 0,
            lambda: self.sync_object_empty.wait(index, phase, loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def producer_commit_w_index(self, index: Int32, *, loc=None, ip=None):
        self.sync_object_full.arrive(index, self.producer_mask, loc=loc, ip=ip)

    @dsl_user_op
    def consumer_wait_w_index_phase(
        self,
        index: Int32,
        phase: Int32,
        try_wait_token: Boolean | None = None,
        *,
        loc=None,
        ip=None,
    ):
        if_generate(
            try_wait_token is None or try_wait_token == 0,
            lambda: self.sync_object_full.wait(index, phase, loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def consumer_release_w_index(self, index: Int32, *, loc=None, ip=None):
        """
        UMMA consumer release buffer empty, cta_group needs to be provided.
        """
        self.sync_object_empty.arrive(
            index, self.consumer_mask, self.cta_group, loc=loc, ip=ip
        )

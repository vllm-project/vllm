# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa

# Vendored from TokenSpeed MLA 2.0.3 (tokenspeed_mla/mla_helpers.py, MIT, see below)
# for the Kimi K3 KV-balanced MLA decode; adds MLAWorkListTileScheduler.
#
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import cutlass
import cutlass.cute as cute


def get_mla_decode_arch(compute_capability: tuple[int, int]) -> str:
    """Return the CuTe architecture name for a supported MLA decode device."""
    if compute_capability not in ((10, 0), (10, 3), (10, 7)):
        raise ValueError(
            f"MLA decode requires SM100, SM103 or SM107, got {compute_capability}"
        )
    major, minor = compute_capability
    return f"sm_{major}{minor}"


def select_mla_decode_tilers(
    num_heads: int,
    seq_len_q: int,
    *,
    is_fp8: bool,
    compute_capability: tuple[int, int],
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Select decode MMA tile shapes from runtime head/q_len configuration.

    FP16/BF16 path currently supports only M=128 kernels.
    FP8 path supports an additional M=64 kernel family on SM100 for the tuned
    H=16, S_q=4 decode shape.
    """
    get_mla_decode_arch(compute_capability)
    default_qk = (128, 128)
    default_pv = (128, 256)
    if not is_fp8:
        return default_qk, default_pv

    is_sm100 = compute_capability == (10, 0)

    if num_heads * seq_len_q <= 64 and is_sm100:
        return (64, 128), (64, 256)
    return default_qk, default_pv


def compute_q_tile_layout(
    num_heads: int, seq_len_q: int, m_tile: int
) -> tuple[int, int, int]:
    """Return total rows, tile count and valid rows in the final query tile.

    ``num_heads`` is the positive head count per query, ``seq_len_q`` is the
    positive query length, and ``m_tile`` is the positive MMA row capacity.
    Heads must fit within one MMA tile. Following FlashInfer PR #4178,
    consecutive rows represent ``query_token * num_heads + head`` and may
    cross query boundaries. The returned tail is in ``[1, m_tile]``.
    """
    if min(num_heads, seq_len_q, m_tile) <= 0 or num_heads > m_tile:
        raise ValueError("Require positive H, Sq and M, with H <= M")
    total_rows = num_heads * seq_len_q
    num_tiles = (total_rows + m_tile - 1) // m_tile
    return total_rows, num_tiles, total_rows - (num_tiles - 1) * m_tile


def get_mla_decode_fold_sq_factor(
    num_heads: int, seq_len_q: int, mma_m_tile: int = 128
) -> int:
    """Return the largest query-token fold factor that exactly divides seq_len_q."""
    if num_heads <= 0 or seq_len_q <= 0 or mma_m_tile <= 0:
        return 1

    max_fold = min(seq_len_q, mma_m_tile // num_heads)
    for candidate in range(max_fold, 1, -1):
        if seq_len_q % candidate == 0:
            return candidate
    return 1


class MLAStaticTileSchedulerParams:
    def __init__(
        self,
        is_persistent: bool,
        problem_shape_b: cute.Int32,
        problem_shape_s: cute.Int32,
        cluster_shape_mnk: cute.Shape,
        split_kv: cutlass.Int32,
        *,
        problem_shape_b_fdd: cute.FastDivmodDivisorV2 = None,
        problem_shape_s_fdd: cute.FastDivmodDivisorV2 = None,
        split_kv_fdd: cute.FastDivmodDivisorV2 = None,
        loc=None,
        ip=None,
    ):
        """The static tile scheduler parameters prepared for MLA static tile scheduler.

        :param is_persistent: Whether to use persistent kernel mode
        :type is_persistent: bool
        :param problem_shape_b: The shape of the problem
        :type problem_shape_b: cute.Int32
        :param problem_shape_s: The shape of the problem in sequence length Q dimension
        :type problem_shape_s: cute.Int32
        :param cluster_shape_mnk: The shape of the cluster
        :type cluster_shape_mnk: cute.Shape
        :param split_kv: The scalar factor for split KV
        """
        self.is_persistent = is_persistent
        self.problem_shape_b = problem_shape_b
        self.problem_shape_s = problem_shape_s
        self.problem_shape_b_fdd = problem_shape_b_fdd
        self.problem_shape_s_fdd = problem_shape_s_fdd
        self.cluster_shape_mnk = cluster_shape_mnk
        self.split_kv = split_kv
        self.split_kv_fdd = split_kv_fdd
        if cutlass.const_expr(problem_shape_b_fdd is None):
            self.problem_shape_b_fdd = cute.fast_divmod_create_divisor_v2(
                problem_shape_b, loc=loc, ip=ip
            )
        if cutlass.const_expr(problem_shape_s_fdd is None):
            self.problem_shape_s_fdd = cute.fast_divmod_create_divisor_v2(
                problem_shape_s, loc=loc, ip=ip
            )
        if cutlass.const_expr(split_kv_fdd is None):
            self.split_kv_fdd = cute.fast_divmod_create_divisor_v2(
                split_kv, loc=loc, ip=ip
            )
        self.loc = loc
        self.ip = ip

    def __extract_mlir_values__(self):
        values = cutlass.extract_mlir_values(self.problem_shape_b)
        values += cutlass.extract_mlir_values(self.problem_shape_s)
        values += cutlass.extract_mlir_values(self.split_kv)
        values += cutlass.extract_mlir_values(self.problem_shape_b_fdd)
        values += cutlass.extract_mlir_values(self.problem_shape_s_fdd)
        values += cutlass.extract_mlir_values(self.split_kv_fdd)
        return values

    def __new_from_mlir_values__(self, values):
        problem_shape_b = cutlass.new_from_mlir_values(
            self.problem_shape_b, (values[0],)
        )
        problem_shape_s = cutlass.new_from_mlir_values(
            self.problem_shape_s, (values[1],)
        )
        split_kv = cutlass.new_from_mlir_values(self.split_kv, (values[2],))
        problem_shape_b_fdd = cutlass.new_from_mlir_values(
            self.problem_shape_b_fdd, values[3:5]
        )
        problem_shape_s_fdd = cutlass.new_from_mlir_values(
            self.problem_shape_s_fdd, values[5:7]
        )
        split_kv_fdd = cutlass.new_from_mlir_values(self.split_kv_fdd, values[7:9])
        return MLAStaticTileSchedulerParams(
            self.is_persistent,
            problem_shape_b,
            problem_shape_s,
            self.cluster_shape_mnk,
            split_kv,
            problem_shape_b_fdd=problem_shape_b_fdd,
            problem_shape_s_fdd=problem_shape_s_fdd,
            split_kv_fdd=split_kv_fdd,
            loc=self.loc,
        )


def create_mla_static_tile_scheduler_params(
    is_persistent: bool,
    problem_shape_b: cute.Int32,
    problem_shape_s: cute.Int32,
    cluster_shape_mnk: cute.Shape,
    split_kv: cutlass.Int32,
) -> MLAStaticTileSchedulerParams:
    return MLAStaticTileSchedulerParams(
        is_persistent, problem_shape_b, problem_shape_s, cluster_shape_mnk, split_kv
    )


class WorkTileInfo:
    def __init__(self, blk_coord: cute.Coord, is_valid: bool):
        self.blk_coord = blk_coord
        self.is_valid = cutlass.Boolean(is_valid)

    def __extract_mlir_values__(self):
        values = cutlass.extract_mlir_values(self.blk_coord)
        values += cutlass.extract_mlir_values(self.is_valid)
        return values

    def __new_from_mlir_values__(self, values):
        new_tile_idx = cutlass.new_from_mlir_values(self.blk_coord, values[:-1])
        new_is_valid_tile = cutlass.new_from_mlir_values(self.is_valid, [values[-1]])
        return WorkTileInfo(new_tile_idx, new_is_valid_tile)

    @property
    def is_valid_tile(self) -> cutlass.Boolean:
        return self.is_valid

    @property
    def tile_idx(self) -> cute.Coord:
        return self.blk_coord


class MLAStaticTileScheduler:
    def __init__(
        self,
        params: MLAStaticTileSchedulerParams,
        current_work_linear_idx: cutlass.Int32,
        blk_coord: cute.Coord,
        grid_shape: cute.Shape,
        *,
        is_valid: bool = True,
        loc=None,
        ip=None,
    ):
        """The static tile scheduler for MLA split kv kernel.
        Based on `is_persistent`, it provides 2 modes for use:
        - Persistent mode: Launch fixed blocks and reschedule the data blocks.
        - Non-persistent mode: Launch dynamic blocks and exit when the current work is done.

        :param params: The static tile scheduler parameters
        :type params: MLAStaticTileSchedulerParams
        :param current_work_linear_idx: The linear index of the current work
        :type current_work_linear_idx: cutlass.Int32
        :param blk_coord: The coordinate of the current work
        :type blk_coord: cute.Coord
        :param grid_shape: The shape of the grid
        :type grid_shape: cute.Shape
        :param is_valid: Whether the current work is valid
        :type is_valid: bool
        """
        self.params = params
        self.blk_coord = blk_coord
        self.grid_shape = grid_shape
        self.current_work_linear_idx = current_work_linear_idx
        if params.is_persistent:
            self.persistent_blk_layout = cute.make_layout(
                (
                    params.cluster_shape_mnk[0],
                    params.problem_shape_s,
                    params.problem_shape_b,
                    params.split_kv,
                ),
                loc=loc,
                ip=ip,
            )
            self.num_blocks = cute.size(self.persistent_blk_layout, loc=loc, ip=ip)
            # Used for persistent scheduling
            self.num_persistent_sm = cute.size(grid_shape, loc=loc, ip=ip)
        else:
            self.is_valid = is_valid
        self.loc = loc
        self.ip = ip

    @staticmethod
    def get_grid_shape(
        params: MLAStaticTileSchedulerParams,
        max_active_clusters: int,
        *,
        loc=None,
        ip=None,
    ) -> cute.Shape:
        # called by host
        grid_shape = (
            params.cluster_shape_mnk[0],
            params.problem_shape_b * params.problem_shape_s,
            params.split_kv,
        )
        if params.is_persistent:
            return (
                cutlass.min(
                    max_active_clusters * cute.size(params.cluster_shape_mnk),
                    cute.size(grid_shape, loc=loc, ip=ip),
                ),
                1,
                1,
            )
        else:
            return grid_shape

    def get_current_work(self, *, loc=None, ip=None) -> WorkTileInfo:
        is_valid = (
            self.current_work_linear_idx < self.num_blocks
            if self.params.is_persistent
            else self.is_valid
        )

        if self.params.is_persistent:
            current_work_cluster_batch, cluster_idx = (
                self.current_work_linear_idx // self.params.cluster_shape_mnk[0],
                self.current_work_linear_idx % self.params.cluster_shape_mnk[0],
            )
            current_work_s_batch, s_idx = divmod(
                current_work_cluster_batch, self.params.problem_shape_s_fdd
            )
            current_work_b_batch, b_idx = divmod(
                current_work_s_batch, self.params.problem_shape_b_fdd
            )
            _, split_kv_idx = divmod(current_work_b_batch, self.params.split_kv_fdd)

            blk_coord = (cluster_idx, s_idx, b_idx, split_kv_idx)
        else:
            s_idx, b_idx = divmod(self.blk_coord[1], self.params.problem_shape_b_fdd)
            blk_coord = (self.blk_coord[0], s_idx, b_idx, self.blk_coord[2])

        return WorkTileInfo(blk_coord, is_valid)

    def initial_work_tile_info(self, *, loc=None, ip=None):
        return self.get_current_work(loc=loc, ip=ip)

    def advance_to_next_work(self, *, advance_count=1, loc=None, ip=None):
        if self.params.is_persistent:
            self.current_work_linear_idx += advance_count * self.num_persistent_sm
        else:
            self.is_valid = False

    def __extract_mlir_values__(self):
        values = cutlass.extract_mlir_values(self.params)
        values.extend(cutlass.extract_mlir_values(self.current_work_linear_idx))
        values.extend(cutlass.extract_mlir_values(self.blk_coord))
        values.extend(cutlass.extract_mlir_values(self.grid_shape))
        return values

    def __new_from_mlir_values__(self, values):
        assert len(values) == 16
        new_params = cutlass.new_from_mlir_values(self.params, values[0:9])
        new_current_work_linear_idx = cutlass.new_from_mlir_values(
            self.current_work_linear_idx, [values[9]]
        )
        new_blk_coord = cutlass.new_from_mlir_values(self.blk_coord, values[10:13])
        new_grid_shape = cutlass.new_from_mlir_values(self.grid_shape, values[13:])
        return MLAStaticTileScheduler(
            new_params, new_current_work_linear_idx, new_blk_coord, new_grid_shape
        )


def create_mla_static_tile_scheduler(
    params: MLAStaticTileSchedulerParams,
    blk_coord: cute.Coord,
    grid_shape: cute.Shape,
) -> MLAStaticTileScheduler:
    return MLAStaticTileScheduler(params, blk_coord[0], blk_coord, grid_shape)


class MLAWorkListTileScheduler:
    """Persistent tile scheduler over a device work list (added for the Kimi K3
    KV-balanced MLA decode). CTA ``i`` of the grid walks clusters ``i // cluster``,
    then every ``grid / cluster`` further; entry ``c`` of the list is
    ``(q_tile * B + batch) << 8 | split``, or < 0. The valid entries are a prefix
    followed by at least one wave of negative entries, so every walk ends inside
    the list."""

    def __init__(
        self,
        params: MLAStaticTileSchedulerParams,
        work_list: cute.Tensor,
        current_work_linear_idx: cutlass.Int32,
        num_ctas: cutlass.Int32,
    ):
        self.params = params
        self.work_list = work_list
        self.current_work_linear_idx = current_work_linear_idx
        self.num_ctas = num_ctas

    def get_current_work(self, *, loc=None, ip=None) -> WorkTileInfo:
        cluster_size = self.params.cluster_shape_mnk[0]
        cluster = self.current_work_linear_idx // cluster_size
        cta = self.current_work_linear_idx % cluster_size
        entry = self.work_list[cluster]
        packed = cutlass.max(entry, cutlass.Int32(0))
        s_idx, b_idx = divmod(packed >> 8, self.params.problem_shape_b_fdd)
        return WorkTileInfo((cta, s_idx, b_idx, packed & 255), entry >= 0)

    def initial_work_tile_info(self, *, loc=None, ip=None):
        return self.get_current_work(loc=loc, ip=ip)

    def advance_to_next_work(self, *, advance_count=1, loc=None, ip=None):
        self.current_work_linear_idx += advance_count * self.num_ctas

    def __extract_mlir_values__(self):
        values = cutlass.extract_mlir_values(self.params)
        values.extend(cutlass.extract_mlir_values(self.current_work_linear_idx))
        values.extend(cutlass.extract_mlir_values(self.num_ctas))
        return values

    def __new_from_mlir_values__(self, values):
        assert len(values) == 11
        return MLAWorkListTileScheduler(
            cutlass.new_from_mlir_values(self.params, values[0:9]),
            self.work_list,
            cutlass.new_from_mlir_values(self.current_work_linear_idx, [values[9]]),
            cutlass.new_from_mlir_values(self.num_ctas, [values[10]]),
        )


LOG2_E = 1.4426950408889634074
# avoid register indexing on array.
MAX_SPLITS = 256


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b

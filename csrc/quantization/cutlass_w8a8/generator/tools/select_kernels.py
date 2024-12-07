import pickle as pkl
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from utils import Data, make_heatmap_data, measurement_to_data, to_cutlass_dtype_str


@dataclass
class Interval:
    s: int  # start of interval
    e: int  # end of interval
    eff: float  # efficiency of the kernel in that range.

    def x_in_interval(self, x: int) -> bool:
        return self.s <= x and x <= self.e

    def is_overlap(self, s, e):
        return s <= self.e and self.s <= e


@dataclass
class KernelIntervals:
    name: str
    intervals: List[Interval]

    def spanning_interval(self, pi: int) -> Optional[Interval]:
        for i in self.intervals:
            if i.x_in_interval(pi):
                return i
        return None


class SelectKernelMeta:

    def __init__(self, gemm_ops: np.array, gemm_problems: List[str],
                 ops: List[str], min_gemm_efficiency: float):
        self.gemm_ops = np.copy(gemm_ops)
        self.gemm_problems = gemm_problems
        self.ops = ops
        self.min_gemm_efficiency = min_gemm_efficiency

        self.n_problems = len(self.gemm_problems)
        self.n_kernels = len(self.ops)

        # Convert to kernel ranges
        self.problem_indices = {x: idx for idx, x in enumerate(gemm_problems)}
        self.kernel_indices = {x: idx for idx, x in enumerate(ops)}

        self.kernel_intervals: List[KernelIntervals] = []
        for ki in range(self.n_kernels):
            self.kernel_intervals.append(self.make_kernel_intervals(ki))

    def avg_efficiency(self, p_s: int, p_e: int, ki: int) -> float:
        """
        Average efficiency of the ki kernel for the gemm shapes in
        range [p_s, p_e]
        """
        vals = self.gemm_ops[:, ki].tolist()[p_s:p_e + 1]
        return sum(vals) / len(vals)

    # TODO (varun) : Revisit kernel scores to use only the intervals we actually
    # use for specific kernels.
    def kernel_set_score(self, p_s: int, p_e: int, kernel_indices: set[int]):
        """
        Compute a score for a set of kernels for the gemm shape indices in
        range [p_s, p_e]
        """
        if len(kernel_indices) == 0:
            return 0.0
        ki_scores = []
        for ki in kernel_indices:
            interval_scores = []
            for i in self.kernel_intervals[ki].intervals:
                if i.is_overlap(p_s, p_e):
                    interval_scores.append(i.eff)
            assert len(interval_scores) > 0
            ki_scores.append(sum(interval_scores) / len(interval_scores))
        assert len(ki_scores) > 0
        return sum(ki_scores) / len(ki_scores)

    def make_kernel_intervals(self, ki: int) -> KernelIntervals:
        s = None
        e = None
        kernel_intervals: KernelIntervals = KernelIntervals(self.ops[ki], [])
        for pi in range(self.n_problems):
            if self.gemm_ops[pi][ki] < self.min_gemm_efficiency:
                # record range
                if e:
                    assert s is not None
                    kernel_intervals.intervals.append(
                        Interval(s, e, eff=self.avg_efficiency(s, e, ki)))
                s, e = None, None
            else:
                s = pi if s is None else s
                e = pi
        if e:
            assert s is not None
            kernel_intervals.intervals.append(
                Interval(s, e, eff=self.avg_efficiency(s, e, ki)))
        # sort intervals in the kernel
        kernel_intervals.intervals = sorted(kernel_intervals.intervals,
                                            key=lambda x: x.s)
        return kernel_intervals


def map_gemm_to_kernel(kernel_indices: List[int],
                       meta: SelectKernelMeta) -> Dict[int, int]:
    """
    For every gemm problem in meta.gemm_problems, select a kernel from
    kernel_indices and return as a dict.
    """
    gemm_to_kernel_map = {}

    for pi in range(meta.n_problems):
        kernels_for_pi = []
        for ki in kernel_indices:
            if meta.kernel_intervals[ki].spanning_interval(pi):
                kernels_for_pi.append(ki)
        assert len(kernels_for_pi) != 0

        # select the kernel with max efficiency
        eff_ki = [(meta.gemm_ops[pi][ki], ki) for ki in kernels_for_pi]
        max_eff_ki = max(eff_ki, key=lambda x: x[0])[1]
        gemm_to_kernel_map[pi] = max_eff_ki

    return gemm_to_kernel_map


def select_kernels_dp(
        p_s: int,
        p_e: int,  # Problem start index and problem end index
        meta: SelectKernelMeta,
        solution_cache: Dict[Tuple[int, int], set]) -> set[int]:
    """
    Compute the best set of kernels for the gemm problem shapes,
    meta.gemm_problems[p_s:p_e].
    """
    if p_s > p_e:
        return set([])
    assert p_s <= p_e
    assert p_s >= 0 and p_e >= 0
    assert p_s < meta.n_problems and p_e < meta.n_problems

    if solution_cache.get((p_s, p_e), None) is not None:
        return solution_cache.get((p_s, p_e))

    spanning_kernels: List[Tuple[int, Interval]] = []
    for ki in range(meta.n_kernels):
        span_i = meta.kernel_intervals[ki].spanning_interval(p_s)
        assert span_i is None or (span_i.s <= p_s and span_i.e >= p_s)
        if span_i is not None:
            spanning_kernels.append((ki, span_i))

    assert len(spanning_kernels) != 0, \
            (f"Cannot find a spanning kernel in range ({p_s}, {p_e})"
            f"- gemm {meta.gemm_problems[p_s]} to {meta.gemm_problems[p_e]}"
            f". Try reducing the min_gemm_efficiency")
    ki_solutions: List[set[int]] = []
    for ki, span in spanning_kernels:
        ki_solutions.append(
            set([ki]).union(
                select_kernels_dp(span.e + 1, p_e, meta, solution_cache)))

    # find the solution with minimum number of kernels.
    sol = min(ki_solutions, key=lambda x: len(x))
    solution_cache[(p_s, p_e)] = sol
    return sol


def generate_struct_from_kernel_name(kernel_name: str, kernels_dict: dict) -> str:
    # Sample kernel name:
    # autogen_scaled_mm_90_64x64x256_8x1x1_KernelTmaWarpSpecializedPingpong_\
    # TmaWarpSpecializedCooperative_PersistentScheduler_kGemm_int32_t_int8
    import re
    pattern = r'(\d+)x(\d+)x(\d+)_(\d+)x(\d+)x(\d+)_([A-Za-z0-9]+)_([A-Za-z0-9]+)_([A-Za-z0-9]+)_([A-Za-z0-9]+)_([a-z][a-z0-9_]+)_([a-z][a-z0-9_]+)'
    match = re.search(pattern, kernel_name)
    if not match:
        raise ValueError(f"Cannot parse kernel name {kernel_name}")
    
    tile_dims = match.group(1, 2, 3)
    cluster_dims = match.group(4, 5, 6)
    kernel_schedule = match.group(7)
    epilogue_schedule = match.group(8)
    tile_schedule = match.group(9)
    mode = match.group(10)
    acc_type = match.group(11)
    input_type = match.group(12)

    # Check if the kernel is already in the kernels_dict
    if (tile_dims, cluster_dims, kernel_schedule, epilogue_schedule, tile_schedule, mode, acc_type, input_type) in kernels_dict:
        kernel_idx = kernels_dict[(tile_dims, cluster_dims, kernel_schedule, epilogue_schedule, tile_schedule, mode, acc_type, input_type)]
        return f'sm90_{input_type}_config_{kernel_idx}', ""
    
    # Create new kernel
    kernel_idx = len(kernels_dict)
    kernels_dict[(tile_dims, cluster_dims, kernel_schedule, epilogue_schedule, tile_schedule, mode, acc_type, input_type)] = kernel_idx

    # Create struct template
    struct_template = f"""
template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_{input_type}_config_{kernel_idx} {{
    static_assert(std::is_same<InType, {to_cutlass_dtype_str(input_type)}>());
    using TileShape = Shape<_{tile_dims[0]}, _{tile_dims[1]}, _{tile_dims[2]}>;
    using ClusterShape = Shape<_{cluster_dims[0]}, _{cluster_dims[1]}, _{cluster_dims[2]}>;
    using KernelSchedule = typename cutlass::gemm::{kernel_schedule};
    using EpilogueSchedule = typename cutlass::epilogue::{epilogue_schedule};
    using TileSchedule = typename cutlass::gemm::{tile_schedule};
    using AccType = {acc_type};
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::{mode};
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
}};
"""
    return f'sm90_{input_type}_config_{kernel_idx}', struct_template


def print_conditionals(meta: SelectKernelMeta, gemm_to_kernel_map: Dict[int, int]):
    # Sort the rows based on the first dimension of the gemm shape, then second, then third.
    meta.gemm_problems = sorted(meta.gemm_problems, key=lambda x: (x[0], x[1], x[2]))
    prev_m = -1
    configs_str = ""
    conds_str = ""
    kernels_dict = {}
    for pi in range(meta.n_problems):
        config_name, config_body = generate_struct_from_kernel_name(meta.ops[gemm_to_kernel_map[pi]], kernels_dict)
        configs_str += config_body
        if prev_m != meta.gemm_problems[pi][0]:
            if prev_m == -1:
                conds_str += f"if (m == {meta.gemm_problems[pi][0]}) {{\n"
            elif meta.gemm_problems[pi][0] == meta.gemm_problems[-1][0]:
                conds_str += f"}} else {{ // m{meta.gemm_problems[pi][0]} kernels\n"
            else:
                conds_str += f"}} else if (m <= {meta.gemm_problems[pi][0]}) {{\n"
            prev_m = meta.gemm_problems[pi][0]
        conds_str += f"    if (n == {meta.gemm_problems[pi][1]} && k == {meta.gemm_problems[pi][2]})\n"
        conds_str += f"        return cutlass_gemm_caller<typename {config_name}\n"
        conds_str += f"            <InType, OutType, Epilogue>::Cutlass3xGemm >(\n"
        conds_str += f"            out, a, b, std::forward<EpilogueArgs>(args)...);\n"
        # conds_str += f"    }}\n"
    conds_str += f"}}\n"

    print(configs_str)
    print(conds_str)


def select_kernels(gemm_ops: np.array, gemm_problems: List[str],
                   ops: List[str], min_gemm_efficiency: float) -> List[str]:
    """
    Given a list of gemm problem shapes, gemm_problems, a list of autogen
    kernel operations ops, normalized benchmarking information and a
    minimum operation efficiency to consider, this function, finds that
    smallest set of kernels such that kernels in the satisfies the
    min_gemm_efficiency for all the gemm shapes. 
    """
    solution_cache = {}
    meta = SelectKernelMeta(gemm_ops, gemm_problems, ops, min_gemm_efficiency)
    kernels = select_kernels_dp(0, meta.n_problems - 1, meta, solution_cache)

    gemm_to_kernel_map = map_gemm_to_kernel(list(kernels), meta)

    print(f"#kernels found {len(kernels)}")
    for pi in range(meta.n_problems):
        print(f"Problem {meta.gemm_problems[pi]} - "
              f"Kernel {meta.ops[gemm_to_kernel_map[pi]]} "
              f"eff. ({gemm_ops[pi][gemm_to_kernel_map[pi]]}) ")
    
    print_conditionals(meta, gemm_to_kernel_map)

    kernel_names = [ops[ki] for ki in kernels]
    return kernel_names


def from_measurements(pkl_files: List[str], min_gemm_efficiency: float):
    data: List[Data] = []

    for pkl_file in pkl_files:
        with open(pkl_file, 'rb') as f:
            pkl_data = pkl.load(f)
            data.extend(list(map(lambda x: measurement_to_data(x), pkl_data)))

    ops = list(map(lambda x: x.description, data))
    ops = sorted(list(set(ops)))
    # have only autogen kernels
    ops = list(filter(lambda x: 'autogen' in x, ops))

    gemm_problems = list(map(lambda x: (x.m, x.n, x.k), data))
    gemm_problems = sorted(list(set(gemm_problems)))

    print(f"#gemm_problems {len(gemm_problems)}")
    print(f"#gemm_ops {len(ops)}")

    gemm_ops: np.array = make_heatmap_data(gemm_problems, ops, data)
    select_kernels(gemm_ops, gemm_problems, ops, min_gemm_efficiency)


def main(pkl_files: List[str], min_gemm_efficiency: float):
    from_measurements(pkl_files, min_gemm_efficiency)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=("Select minimal set of kernels in some model_bench "
                     "pkl file such that the set of kernels satisfy"
                     "the min-gemm-efficiency for all the gemm shapes in"
                     "the model_bench"))
    parser.add_argument("--input-pkl",
                        "-i",
                        nargs="+",
                        required=True,
                        type=str)
    parser.add_argument(
        "--min-gemm-efficiency",
        type=float,
        default=0.95,
        help="Gemms that are less than this for a particular gemm shape is"
        "disregarded")
    args = parser.parse_args()

    main(args.input_pkl, args.min_gemm_efficiency)

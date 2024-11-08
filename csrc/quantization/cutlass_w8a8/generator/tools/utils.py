from dataclasses import dataclass
from typing import List

import numpy as np
from torch.utils.benchmark import Measurement as TMeasurement


@dataclass
class Data:
    m: int
    k: int
    n: int
    description: str
    time: float
    tflops: float


def parse_mkn(mkn_str: str):
    # mkn_str : MKN=(16x1024x512)
    mkn_tuple = mkn_str.split("=")[1]
    # mkn_tuple : (16x1024x512)
    mkn_prod = mkn_tuple[1:-1]
    # mkn_prod : 16x1024x512
    mkn_tuple = tuple(mkn_prod.split("x"))
    return (int(mkn_tuple[0]), int(mkn_tuple[1]), int(mkn_tuple[2]))


def measurement_to_data(measurement: TMeasurement) -> Data:
    m, k, n = parse_mkn(measurement.sub_label)
    t_ops = 2 * m * k * n / 1024 / 1024 / 1024 / 1024
    tflops = t_ops / measurement.median
    return Data(m, k, n, measurement.task_spec.description, measurement.median,
                tflops)


def make_heatmap_data(gemm_problems: List[str], ops: List[str],
                      data: List[Data]) -> np.array:
    """
        gemm_problems : List of gemm problem shapes
        ops : List of operations (kernels)
        data : List of Data that contains benchmark information for all
            op-gemmshape pairs.
        Normalize all the benchmark information w.r.t. to its gemm-shape
        and return the normalized benchmark information as a numpy array.
    """
    gemm_ops: List[List[float]] = [[0.0] * len(ops)
                                   for _ in range(len(gemm_problems))]
    for op_idx, op in enumerate(ops):
        op_data = list(filter(lambda x: x.description == op, data))
        for gemm_idx, gemm in enumerate(gemm_problems):
            m, n, k = gemm
            selected = list(
                filter(lambda x: x.m == m and x.n == n and x.k == k, op_data))
            if len(selected) >= 1:
                gemm_ops[gemm_idx][op_idx] = float(selected[0].tflops)

    for gemm_idx in range(len(gemm_problems)):
        max_tflops = max(gemm_ops[gemm_idx])
        for op_idx in range(len(ops)):
            gemm_ops[gemm_idx][op_idx] = round(
                gemm_ops[gemm_idx][op_idx] / max_tflops, 2)

    return np.array(gemm_ops)

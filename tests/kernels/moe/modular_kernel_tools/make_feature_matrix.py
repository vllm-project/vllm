# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
from enum import Enum
from itertools import product
from typing import Optional

import torch
from tqdm import tqdm

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.platforms import current_platform

from .common import (Config, RankTensors, WeightTensors, reference_moe_impl,
                     run_modular_kernel)
from .mk_objects import (MK_FUSED_EXPERT_TYPES,
                         MK_MULTI_GPU_PREPARE_FINALIZE_TYPES, MK_QUANT_CONFIGS)
from .parallel_utils import ProcessGroupInfo, parallel_launch_with_config


class Result(Enum):
    PASS = 1
    FAIL = 2
    SKIP = 3


def rank_worker(
    pgi: ProcessGroupInfo,
    vllm_config: VllmConfig,
    cpu_group,
    config: Config,
    weights: WeightTensors,
):
    current_platform.seed_everything(pgi.rank)

    # sanity check
    from vllm import envs
    if config.fused_moe_chunk_size is not None:
        assert (config.fused_moe_chunk_size == envs.VLLM_FUSED_MOE_CHUNK_SIZE)

    # get weights to this device
    weights.to_current_device()

    Ms = config.Ms
    assert isinstance(Ms, list)
    TOPKs = config.topks
    assert isinstance(TOPKs, list)

    for m, topk in product(Ms, TOPKs):
        print(f"Running m={m}, topk={topk} ...")
        # override m and topk
        cfgx = copy.deepcopy(config)
        cfgx.Ms = m
        cfgx.topks = topk

        # inputs for rank
        rank_tensors = RankTensors.make(cfgx, pgi)

        # modular kernel out
        mk_out = run_modular_kernel(pgi, vllm_config, cfgx, weights,
                                    rank_tensors)

        with set_current_vllm_config(vllm_config):
            ref_out = reference_moe_impl(cfgx, weights, rank_tensors)

        torch.testing.assert_close(ref_out, mk_out, atol=3e-2, rtol=3e-2)


def make_feature_matrix(csv_file_path: str):

    from dataclasses import asdict

    import pandas as pd

    def add_to_results(config: Config,
                       success: Result,
                       results_df: Optional[pd.DataFrame] = None):
        config_dict = asdict(config)
        config_dict['prepare_finalize_type'] = config_dict[
            'prepare_finalize_type'].__name__
        config_dict['fused_experts_type'] = config_dict[
            'fused_experts_type'].__name__
        config_dict['per_tensor_act_quant'] = config.is_per_tensor_act_quant
        quant_config_dict = config_dict['quant_config']
        del config_dict['quant_config']
        if quant_config_dict is None:
            quant_config = FusedMoEQuantConfig(None)
            quant_config_dict = asdict(quant_config)

        config_dict |= quant_config_dict
        result_dict = config_dict | {'success': success.name}

        result_df = pd.DataFrame([result_dict])
        if results_df is None:
            results_df = result_df
        else:
            results_df = pd.concat([results_df, result_df], ignore_index=True)

        return results_df

    Ms = [64]
    Ks = [7168]  # hidden sizes
    Ns = [2048]
    TOPKs = [[4, 1]]
    Es = [32]
    DTYPEs = [torch.bfloat16]
    PF_TYPES = MK_MULTI_GPU_PREPARE_FINALIZE_TYPES
    FE_TYPES = MK_FUSED_EXPERT_TYPES
    Q_TYPES = MK_QUANT_CONFIGS

    combinations = list(
        product(Ms, Ks, Ns, Es, TOPKs, DTYPEs, PF_TYPES, FE_TYPES, Q_TYPES))

    results_df: Optional[pd.DataFrame] = None
    for m, k, n, e, topks, dtype, pf_type, experts_type, quant_config in tqdm(
            combinations):  #noqa: E501
        config = Config(Ms=[m],
                        K=k,
                        N=n,
                        E=e,
                        topks=topks,
                        dtype=dtype,
                        prepare_finalize_type=pf_type,
                        fused_experts_type=experts_type,
                        quant_config=quant_config,
                        world_size=2,
                        fused_moe_chunk_size=None)

        success = None
        if config.is_valid():
            print(f"Running config : {config.describe()} ...")
            try:
                weights: WeightTensors = WeightTensors.make(config)
                vllm_config, env_dict = config.make_env_data()
                parallel_launch_with_config(config.world_size, rank_worker,
                                            vllm_config, env_dict, config,
                                            weights)
                success = Result.PASS
            except Exception as _:
                success = Result.FAIL
        else:
            success = Result.SKIP

        results_df = add_to_results(config, success, results_df)

    if results_df is not None:
        results_df.to_csv(f"{csv_file_path}")


if __name__ == '__main__':
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(description=(
        "Make ModularKernel feature matrix \n"
        "Example : python3 -m tests.kernels.moe.modular_kernel_tools.make_feature_matrix "  #noqa: E501
        "-f ./feature_matrices/feature_matrix.csv"))

    parser.add_argument("-f",
                        "--feature-matrix-csv-file-path",
                        type=str,
                        required=True,
                        help="File name to Generate a .csv file")
    args = parser.parse_args()

    csv_path = args.feature_matrix_csv_file_path
    assert csv_path.endswith(
        'csv'), f"Need a file path ending with .csv, got {csv_path}"
    assert Path(csv_path).parent.is_dir(
    ), f"Cannot find parent directory for {Path(csv_path).parent}"

    make_feature_matrix(args.feature_matrix_csv_file_path)

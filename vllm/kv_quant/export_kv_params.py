# Copyright (c) OpenMMLab. All rights reserved.
from pathlib import Path
from typing import Union

import fire
import numpy as np
import torch


def _export_sym(key_stats: dict,
                value_stats: dict,
                bits: int,
                out_dir: Union[str, Path],
                tp: int = 1) -> None:
    """Export symmetric quantization parameters to specified directory."""
    keys_absmax = key_stats['absmax']
    values_absmax = value_stats['absmax']
    for layer_idx, name in enumerate(keys_absmax.keys()):
        k_absmax = keys_absmax[name]
        v_absmax = values_absmax[name]

        heads, _ = k_absmax.shape
        assert heads % tp == 0

        mp_k_absmax = torch.chunk(k_absmax, tp)
        mp_v_absmax = torch.chunk(v_absmax, tp)
        for i in range(tp):
            # quant: q = f / scale
            # dequant: f = q * scale
            k_s = mp_k_absmax[i].max() / (2**(bits - 1) - 1)
            v_s = mp_v_absmax[i].max() / (2**(bits - 1) - 1)

            kv_qparams = np.array([k_s, v_s], dtype=np.float32)
            out_path = out_dir / f'layers.{layer_idx}.past_kv_scale.{i}.weight'  # noqa: E501
            kv_qparams.tofile(out_path)
            print(f'Layer {layer_idx} MP {i} qparam: {k_s} \t{v_s}')


def _export_asym(key_stats: dict,
                 value_stats: dict,
                 bits: int,
                 out_dir: Union[str, Path],
                 tp: int = 1) -> None:
    """Export asymmetric quantization parameters to specified directory."""
    keys_min = key_stats['min']
    values_min = value_stats['min']

    keys_max = key_stats['max']
    values_max = value_stats['max']
    for layer_idx, name in enumerate(keys_min.keys()):
        k_max = keys_max[name]
        v_max = values_max[name]

        k_min = keys_min[name]
        v_min = values_min[name]

        heads, _ = k_min.shape
        assert heads % tp == 0

        tp_k_min = torch.chunk(k_min, tp)
        tp_v_min = torch.chunk(v_min, tp)

        tp_k_max = torch.chunk(k_max, tp)
        tp_v_max = torch.chunk(v_max, tp)
        for i in range(tp):
            # zp = (min+max) / 2
            # scale = (max-min) / 255
            # quant: q = (f-zp) / scale
            # dequant: f = q * scale + zp
            k_min = tp_k_min[i].min()
            v_min = tp_v_min[i].min()

            k_max = tp_k_max[i].max()
            v_max = tp_v_max[i].max()

            k_scale = (k_max - k_min) / (2**bits - 1)
            v_scale = (v_max - v_min) / (2**bits - 1)

            k_zp = (k_max + k_min) / 2
            v_zp = (v_max + v_min) / 2

            kv_qparams = np.array([k_scale, k_zp, v_scale, v_zp],
                                  dtype=np.float32)
            out_path = out_dir / f'layers.{layer_idx}.past_kv_scale.{i}.weight'
            kv_qparams.tofile(out_path)
            print(f'Layer {layer_idx} MP {i} qparam: '
                  f'\t{k_scale} \t{k_zp} \t{v_scale} \t{v_zp}')


def main(work_dir: str,
         kv_params_dir: str,
         kv_bits: int = 8,
         kv_sym: bool = False,
         num_tp: int = 1) -> None:
    """Main function to export key and value stats.

    Args:
        work_dir (Union[str, Path]): Directory path where the stats are saved.
        kv_params_dir (Union[str, Path]): Directory path where to
            save the results.
        kv_bits (int, optional): Number of bits for quantization.
            Defaults to 8.
        kv_sym (bool, optional): Whether to use symmetric quantizaiton.
            Defaults to False.
        num_tp (int, optional): Number of tensor parallelism. Defaults to 1.
    """

    work_dir = Path(work_dir)

    tm_dir = Path(kv_params_dir)
    tm_dir.mkdir(parents=True, exist_ok=True)

    key_stats = torch.load(work_dir / 'key_stats.pth')
    value_stats = torch.load(work_dir / 'value_stats.pth')

    if kv_sym:
        _export_sym(key_stats, value_stats, kv_bits, tm_dir, num_tp)
    else:
        _export_asym(key_stats, value_stats, kv_bits, tm_dir, num_tp)


if __name__ == '__main__':
    fire.Fire(main)

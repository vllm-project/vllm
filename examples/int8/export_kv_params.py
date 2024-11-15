# Copyright (c) OpenMMLab. All rights reserved.
import os
from pathlib import Path
from typing import Union
import matplotlib.pyplot as plt
import seaborn as sns
import json

import fire
import numpy as np
import torch

n_reques=1
n_layer=32
n_tokens=1
kv_head=8
head_size=128
plot = False
use_max = False
n_max = 1
if not use_max:
    n_max = 10

plt.figure(figsize=(80,40))
font_size = 20

def format(i, x_axis_name, y_axis_name, png_name):
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.xlabel(x_axis_name, fontsize=font_size)
    plt.ylabel(y_axis_name, fontsize=font_size)
    plt.title('layer %i'%i,fontsize=font_size)
    plt.rcParams.update({'font.size': font_size})
    plt.savefig(png_name)

def plot_hideen_size(t:np, png_name, quant_group): # t.shape [n_req, n_layer, input_len, (kv_head*head_size)//quant_group]
    t = np.transpose(t, (1,0,2,3))
    t = t.reshape(n_layer, -1)
    for i in range(t.shape[0]):
        print("Ploting %s layer %i "%(png_name, i))
        y=t[i:i+1].reshape(t.shape[1])
        x = np.arange(kv_head*head_size//quant_group)
        x = np.repeat(x, t.shape[1]//(kv_head*head_size//quant_group))
        # print(y.shape)
        # print(x.shape)
        plt.subplot(4,8,i+1)
        plt.plot(x, y, '*')
        # plot1=plt.plot(x, y, '*',label=(f'layer %i', i))
        # z1 = np.polyfit(x, y, 4)
        # p1 = np.poly1d(z1)
        # # print(p1)
        # yvals=np.polyval(z1,x)
        # plot2=plt.plot(x, yvals, 'r',label=(f'polyfit layer %i', i))
        # plt.legend(loc=4)s
        format(i, 'head_idx','scaling factor', png_name)

def plot_per_value(t:np, png_name, quant_group):
    t = np.transpose(t, (1,0,2,3))
    t = t.reshape(n_layer, -1, kv_head*head_size//quant_group)
    for i in range(t.shape[0]):
        print("Ploting %s layer %i "%(png_name, i))
        y= t[:,i,:]
        y = y.tolist()
        plt.subplot(4,8,i+1)
        sns.histplot(y, bins=100, legend=False)
        format(i, 'scaling factor', 'count bin', png_name)

def loadtxt(txtname, quant_group):
    key = np.loadtxt(txtname, delimiter='\n')
    key = key.reshape(-1, n_layer, n_tokens, (kv_head*head_size)//quant_group)
    return key

def sorted_np(a:np, axis):
    b=np.sort(a, axis)[::-1]
    print( " ", a.shape[axis])
    global n_max
    if n_max > a.shape[axis]:
        n_max=0
    if axis == 0 or (len(a.shape) == 1 and axis ==-1):
        c = b[n_max:n_max+1]
    elif axis == 1 or (len(a.shape) == 2 and axis ==-1):
        c = b[:,n_max:n_max+1]
    elif axis == 2 or (len(a.shape) == 3 and axis ==-1):
        c = b[:,:,n_max:n_max+1]
    elif axis == 3 or (len(a.shape) == 4 and axis ==-1):
        c = b[:,:,:,n_max:n_max+1]
    return c

def find_max(tensors, axis):
    print(tensors.shape)
    sorted_tensor = sorted_np(tensors, axis)
    print("sorted_tensor.shape ", sorted_tensor.shape)
    # print("sorted_tensor ", sorted_tensor)
    # scale = np.reshape(scale, (-1))
    if use_max:
        scale = np.max(tensors, axis=axis, keepdims=True)
    else:
        scale = sorted_tensor
    print("scale.shape", scale.shape)
    # print("scale, ", scale)
    return scale

def save_txt(save_name, tensor):
    with open(save_name,'w', encoding='utf-8') as k_file:
        for i in range(tensor.size):
            k_file.write("%f\n"%tensor[i])

class NumpyEncoder(json.JSONEncoder):  
    def default(self, obj):  
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,  
            np.int16, np.int32, np.int64, np.uint8,  
            np.uint16, np.uint32, np.uint64)):  
            return int(obj)  
        elif isinstance(obj, (np.float_, np.float16, np.float32,np.float64)):  
            return float(obj)  
        elif isinstance(obj, (np.ndarray,)):  
            return obj.tolist()  
        return json.JSONEncoder.default(self, obj)

def values_to_scaling_factor(scale, zp=None):
    s = {}
    z = {}
    scale = np.reshape(scale, (n_layer, -1))
    # np.set_printoptions(threshold=np.inf)
    print(scale.shape)
    for i in range(scale.shape[0]):
        layer_i_s = {}
        layer_i_z = {}
        for j in range(scale.shape[1]):
            layer_i_s[f"%i"%j] = scale[i][j]
            # print(scale[i][j])
            if zp is not None:
                zp = np.reshape(zp, (n_layer, -1))
                layer_i_z[f"%i"%j] = zp[i][j]
            else:
                layer_i_z[f"%i"%j] = 0.0
        s[f"%i"%i] = layer_i_s
        z[f"%i"%i] = layer_i_z
    return s, z

def save_to_json(out_dir, quant_group, k_scale, v_scale, k_zps=None, v_zps=None):
    info = {
                "model_type":"llama",
                "kv_cache": {
                    "dtype": "int8",
                    "scaling_factor": {
                    }
                }
        }
    data = json.loads(json.dumps(info))

    if k_zps is not None:
        k_s, k_z = values_to_scaling_factor(k_scale, k_zps)
        v_s, v_z = values_to_scaling_factor(v_scale, v_zps)
    else:
        k_s, k_z = values_to_scaling_factor(k_scale)
        v_s, v_z = values_to_scaling_factor(v_scale)

    scaling_factor = {"k_scale":k_s}
    scaling_factor.update({"v_scale":v_s})
    scaling_factor.update({"k_zero_point":k_z})
    scaling_factor.update({"v_zero_point":v_z})
    # scaling_factor = {"scaling_factor": {k_s_info, v_s_info, k_z_info, v_z_info}}
    data['kv_cache']['scaling_factor'] = scaling_factor
    # print("json_data ", data)
    if quant_group==-1:
        json_name = "./kv_cache_scales_layer_level.json"
        save_json = os.path.join(out_dir,json_name)
        with open(save_json, 'w') as f:
            json.dump(data, f, indent=4, cls=NumpyEncoder)
    else:
        json_name = "./kv_cache_scales_quant_group"+str(quant_group)+".json"
        save_json = os.path.join(out_dir,json_name)
        with open(save_json, 'w') as f:
            json.dump(data, f, indent=4, cls=NumpyEncoder)

def get_tensors_for_json(lists):
    tensor = np.stack(lists, axis=0 )
    tensor_layer_level = torch.Tensor(tensor)
    tensor_layer_level,_ = torch.max(tensor_layer_level, 1, True)
    tensor_layer_level = tensor_layer_level.numpy()
    tensor_layer_level = np.reshape(tensor_layer_level, (-1)).astype("float32")
    tensor = np.reshape(tensor, (-1)).astype("float32")
    return tensor, tensor_layer_level

def _export_sym(key_stats: dict,
                value_stats: dict,
                bits: int,
                out_dir: Union[str, Path],
                tp: int = 1,
                quant_group: int = 32) -> None:
    """Export symmetric quantization parameters to specified directory."""
    keys_absmax = key_stats['absmax']
    values_absmax = value_stats['absmax']
    ks_lists, vs_lists = [], []
    for layer_idx, name in enumerate(keys_absmax.keys()):
        k_absmax = keys_absmax[name]
        v_absmax = values_absmax[name]

        heads, _ = k_absmax.shape
        assert heads % tp == 0

        mp_k_absmax = torch.chunk(k_absmax, tp)
        mp_v_absmax = torch.chunk(v_absmax, tp)
        for i in range(tp):
            k_max = mp_k_absmax[i].reshape(-1, quant_group)
            v_max = mp_v_absmax[i].reshape(-1, quant_group)
            kmax, k_max_sp = torch.max(k_max, -1, True)
            vmax, v_max_sp = torch.max(v_max, -1, True)

            k_scale = kmax / (2**(bits-1) - 1)
            v_scale = vmax / (2**(bits-1) - 1)

            ks_lists.append(k_scale)
            vs_lists.append(v_scale)
    
    k_scales, k_scales_layer_level = get_tensors_for_json(ks_lists)
    v_scales, v_scales_layer_level = get_tensors_for_json(vs_lists)
    # print("kkk ", k_scales.shape)
    save_to_json(out_dir, quant_group, k_scales, v_scales)
    save_to_json(out_dir, -1, k_scales_layer_level, v_scales_layer_level)

    if plot:
        k_png = "savefig_k_cache.png"
        v_png = "savefig_v_cache.png"
        plot_hideen_size(k_scales, k_png, quant_group)
        plt.clf()
        plot_hideen_size(v_scales, v_png, quant_group)
        plt.clf()
        k_png_ = "savefig_k_cache_per_value.png"
        v_png_ = "savefig_v_cache_per_value.png"
        plot_per_value(k_scales, k_png_, quant_group)
        plt.clf()
        plot_per_value(v_scales, v_png_, quant_group)
        plt.clf()

def _export_asym(key_stats: dict,
                 value_stats: dict,
                 bits: int,
                 out_dir: Union[str, Path],
                 tp: int = 1,
                 quant_group: int = 32) -> None:
    """Export asymmetric quantization parameters to specified directory."""
    keys_min = key_stats['min']
    values_min = value_stats['min']

    keys_max = key_stats['max']
    values_max = value_stats['max']
    # print("key_stat ", type(key_stats))
    # print("value_stat ", type(value_stats))
    # print("key_stat ", key_stats.keys())
    # print("value_stat ", value_stats.keys())
    # print("key_stat ", key_stats)
    # print("value_stat ", value_stats)
    # print("key_stat[min].shape ", key_stats['min']['model.layers.0'].shape)
    # print("value_stat[min].shape ", value_stats['min']['model.layers.0'].shape)
    # print("key_stat[min] ", key_stats['min']['model.layers.0'])
    # print("value_stat[min] ", value_stats['min']['model.layers.0'])
    # print("key_stat[max] ", key_stats['max']['model.layers.0'])
    # print("value_stat[max] ", value_stats['max']['model.layers.0'])
    # print("key_stat[absmax] ", key_stats['absmax']['model.layers.0'])
    # print("value_stat[absmax] ", value_stats['absmax']['model.layers.0'])
    ks_lists, vs_lists = [], []
    kz_lists, vz_lists = [], []
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
            k_min = tp_k_min[i].reshape(-1, quant_group)
            v_min = tp_v_min[i].reshape(-1, quant_group)
            k_max = tp_k_max[i].reshape(-1, quant_group)
            v_max = tp_v_max[i].reshape(-1, quant_group)
            kmin, k_min_sp = torch.min(torch.abs(k_min), -1, True)
            vmin, v_min_sp = torch.min(torch.abs(v_min), -1, True)
            kmax, k_max_sp = torch.max(torch.abs(k_max), -1, True)
            vmax, v_max_sp = torch.max(torch.abs(v_max), -1, True)

            k_scale = (kmax - kmin) / (2**bits - 1)
            v_scale = (vmax - vmin) / (2**bits - 1)
            k_zp = (kmax + kmin) / 2
            v_zp = (vmax + vmin) / 2

            ks_lists.append(k_scale)
            vs_lists.append(v_scale)
            kz_lists.append(k_zp)
            vz_lists.append(v_zp)

    k_scales, k_scales_layer_level = get_tensors_for_json(ks_lists)
    v_scales, v_scales_layer_level = get_tensors_for_json(vs_lists)
    k_zps, k_zps_layer_level = get_tensors_for_json(kz_lists)
    v_zps, v_zps_layer_level = get_tensors_for_json(vz_lists)

    # print("kkk ", k_scales.shape)
    save_to_json(out_dir, quant_group, k_scales, v_scales, k_zps, v_zps)
    save_to_json(out_dir, -1, k_scales_layer_level, v_scales_layer_level, k_zps_layer_level, v_zps_layer_level)

    if plot:
        k_png = "savefig_k_cache.png"
        v_png = "savefig_v_cache.png"
        plot_hideen_size(k_scales, k_png, quant_group)
        plt.clf()
        plot_hideen_size(v_scales, v_png, quant_group)
        plt.clf()
        k_png_ = "savefig_k_cache_per_value.png"
        v_png_ = "savefig_v_cache_per_value.png"
        plot_per_value(k_scales, k_png_, quant_group)
        plt.clf()
        plot_per_value(v_scales, v_png_, quant_group)
        plt.clf()

def main(work_dir: str,
         kv_params_dir: str = './work_dir/',
         kv_bits: int = 8,
         quant_group: int = 32,
         kv_sym: bool = True,
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
        _export_sym(key_stats, value_stats, kv_bits, tm_dir, num_tp, quant_group)
    else:
        _export_asym(key_stats, value_stats, kv_bits, tm_dir, num_tp, quant_group)


if __name__ == '__main__':
    fire.Fire(main)

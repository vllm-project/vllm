# coding=utf-8
# Adapted from
# https://github.com/InternLM/lmdeploy/blob/main/lmdeploy/lite/apis/calibrate.py

# Copyright (c) OpenMMLab. All rights reserved.

from pathlib import Path

import fire
import torch
from accelerate import (infer_auto_device_map, init_empty_weights,
                        load_checkpoint_in_model)
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from calib_dataloader import get_calib_loaders
from calibration import CalibrationContext
from utils import collect_target_modules

LAYER_TYPE_MAP = {
    'InternLMForCausalLM': 'InternLMDecoderLayer',
    'QWenLMHeadModel': 'QWenBlock',
    'BaiChuanForCausalLM': 'DecoderLayer',
    'LlamaForCausalLM': 'LlamaDecoderLayer',
}
NORM_TYPE_MAP = {
    'InternLMForCausalLM': 'InternLMRMSNorm',
    'QWenLMHeadModel': 'RMSNorm',
    'BaiChuanForCausalLM': 'RMSNorm',
    'LlamaForCausalLM': 'LlamaRMSNorm',
}


def calibrate(model: str = '/home/model_weights/Llama3-Chinese-8B-Instruct/',
              calib_dataset: str = 'ptb',
              dataset_path: str = None,
              work_dir: str = './work_dir',
              calib_samples: int = 128,
              calib_seqlen: int = 2048,
              device: str = 'cuda',
              ) -> None:
    """The main function for loading the model and performing calibration on a
    given dataset.

    Args:
        model (str): The model to be loaded.
        calib_dataset (str, optional): The calibration dataset name.
            Defaults to 'ptb'.
        calib_samples (int, optional): The number of samples for calibration.
            Defaults to 128.
        calib_seqlen (int, optional): The sequence length for calibration.
            Defaults to 2048.
        work_dir (str): The working directory for outputs.
            Defaults to './work_dir'.
        device (str, optional): The device to be used for calculation.
            Defaults to 'cuda'.
    """
    # ceval_val_cmcc.jsonl

    assert calib_dataset in ['c4', 'ptb', 'wikitext2', 'pileval', 'ceval_val_cmcc',
                              'ceval', 'cmmlu', 'cmb', 'medmcqa', 'medqa', 'mmlu'], \
                              'Support only `c4`, `ptb`, `wikitext2` or `pileval`, \
                              `ceval_val_cmcc`, `ceval`, `cmmlu`, `cmb`, `medmcqa`,\
                              `medqa`, `mmlu`'

    # Load tokenizer and configuration
    tokenizer = AutoTokenizer.from_pretrained(model,
                                              use_fast=False,
                                              trust_remote_code=True)
    hf_config = AutoConfig.from_pretrained(model, trust_remote_code=True)
    checkpoint = hf_config._name_or_path

    with init_empty_weights():
        # Load model
        model = AutoModelForCausalLM.from_pretrained(model,
                                                     torch_dtype=torch.float16,
                                                     trust_remote_code=True)
        model.config.use_cache = False

    layer_type = LAYER_TYPE_MAP[type(model).__name__]
    norm_type = NORM_TYPE_MAP[type(model).__name__]

    decoder_layers = collect_target_modules(model, layer_type)

    # Infer device map
    device_map = infer_auto_device_map(model,
                                       no_split_module_classes=[layer_type])
    for name in device_map:
        if name in decoder_layers or 'lm_head' in name:
            device_map[name] = 'cpu'
        else:
            device_map[name] = 0
    load_checkpoint_in_model(model, checkpoint, device_map)

    print('Loading calibrate dataset ...')
    calib_loader, _ = get_calib_loaders(calib_dataset,
                                        tokenizer,
                                        nsamples=calib_samples,
                                        seqlen=calib_seqlen,
                                        path=dataset_path)

    # Initialize calibration context
    calib_ctx = CalibrationContext(model,
                                   tokenizer,
                                   layer_type=layer_type,
                                   norm_type=norm_type,
                                   device=device)

    with calib_ctx:
        all_data = torch.cat([
            data if isinstance(data, torch.Tensor) else data[0]
            for data in calib_loader
        ]).to(device)
        calib_ctx.calibrate(all_data)

    # Create work directory if not exists
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    calib_ctx.export(work_dir)


if __name__ == '__main__':
    fire.Fire(calibrate)

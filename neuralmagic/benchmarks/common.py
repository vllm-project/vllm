import itertools
import json

from argparse import Namespace
from pathlib import Path
from typing import NamedTuple, Iterable

from vllm.model_executor.weight_utils import prepare_hf_model_weights
from vllm.transformers_utils.tokenizer import get_tokenizer
from vllm.transformers_utils.config import get_config
# TODO (varun) : find a workaround so we avoid using private methods
from vllm.config import _get_and_verify_max_len


def download_model(model: str) -> None:
    """
     Downloads a hugging face model to cache
     """
    prepare_hf_model_weights(model)
    get_tokenizer(model)


def max_model_length_from_model_id(model: str,
                                   trust_remote_code: bool = False) -> int:
    config = get_config(model, trust_remote_code=trust_remote_code)
    return _get_and_verify_max_len(config, max_model_len=None)


def script_args_to_cla(config: NamedTuple) -> Iterable[list[str]]:
    #config is a NamedTuple constructed from some JSON in neuralmagic/benchmarks/configs

    kv = vars(config.script_args)

    keys = kv.keys()
    arg_lists = kv.values()
    assert all(map(lambda le: isinstance(le, list), arg_lists))

    # Empty lists are arguments without any values (e.g. boolean args)
    key_args = []
    for k, v in zip(keys, arg_lists):
        if len(v) == 0:
            key_args.append(k)

    key_args_cla = list(map(lambda k: f"--{k}", key_args))

    # Remove empty lists from arg_lists and remove key args from keys
    arg_lists = list(filter(lambda arg_list: len(arg_list) != 0, arg_lists))
    keys = list(filter(lambda k: k not in key_args, keys))
    assert len(keys) == len(arg_lists)

    for args in itertools.product(*arg_lists):
        cla = key_args_cla
        for name, value in zip(keys, args):
            cla.extend([f"--{name}", f"{value}"])
        yield cla


def benchmark_configs(config_file_path: Path) -> Iterable[NamedTuple]:
    """
    Give a path to a config file in `neuralmagic/benchmarks/configs/*` return an Iterable of
    (sub)configs in the file
    """
    assert config_file_path.exists()

    configs = None
    with open(config_file_path, "r") as f:
        configs = json.load(f, object_hook=lambda d: Namespace(**d))
    assert configs is not None

    for config in configs.configs:
        yield config

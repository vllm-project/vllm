# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import sys
from argparse import ArgumentParser, BooleanOptionalAction


# copied from: vllm/transformers_utils/configs/nemotron_h.py
def compute_layers_block_type(pattern):
    return [
        "mamba"
        if pattern[i] == "M" else "attention" if pattern[i] == "*" else "mlp"
        for i in range(len(pattern))
    ]


def main(args):
    if args.pattern is not None and args.config_path is not None:
        raise RuntimeError("cannot specify both --pattern and config-path")
    default_pattern = ("M-M-M-M*-M-M-M-M*-M-M-M-M*-"
                       "M-M-M-M*-M-M-M-M*-M-M-M-M*-"
                       "M-M-M-M-")
    pattern = None
    config = None
    if args.config_path is not None:
        try:
            with open(args.config_path) as file:
                data = file.read()
            config = json.loads(data.strip())
            if "text_config" not in config:
                if "llm_config" not in config:
                    raise RuntimeError(
                        "provided config.json is missing 'text_config' and "
                        "'llm_config'")
                else:
                    config["text_config"] = config.pop("llm_config")
            pattern = config["text_config"]["hybrid_override_pattern"]
        except Exception as err:
            raise RuntimeError("error reading config path = "
                               f"{repr(args.config_path)} "
                               f"exception = {type(err).__name__} "
                               f"detail = {err}") from err
    elif args.pattern is not None:
        pattern = default_pattern
    else:
        print(
            "DEBUG: convert_layers_block_type.main: using default pattern",
            file=sys.stderr,
        )
        pattern = default_pattern

    msg_pattern = ("DEBUG: convert_layers_block_type.main: "
                   f"pattern      = {repr(pattern)}")
    print(msg_pattern, file=sys.stderr)
    msg_len_pattern = ("DEBUG: convert_layers_block_type.main: "
                       f"len(pattern) = {len(pattern)}")
    print(msg_len_pattern, file=sys.stderr)

    layers_block_type = compute_layers_block_type(pattern)

    msg_layers = ("DEBUG: convert_layers_block_type.main: "
                  f"layers       = {repr(layers_block_type)}")
    print(msg_layers, file=sys.stderr)
    msg_len_layers = ("DEBUG: convert_layers_block_type.main: "
                      f"len(layers)  = {len(layers_block_type)}")
    print(msg_len_layers, file=sys.stderr)

    if args.dump_full_config:
        extra_text_config = {"layers_block_type": layers_block_type}
        config["text_config"] |= extra_text_config
        print(json.dumps(config, indent=2))
    else:
        print(json.dumps(layers_block_type))


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--pattern", type=str, default=None)
    args.add_argument("--config-path", type=str, default=None)
    args.add_argument("--dump-full-config",
                      action=BooleanOptionalAction,
                      default=None)
    args = args.parse_args()
    main(args)

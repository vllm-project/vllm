# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import json
import os


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=("prefill", "decode"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--lookup-rpc-port", type=int, required=True)
    parser.add_argument("--cache-prefix", required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--kv-cache-dtype", default="auto")
    parser.add_argument("--linear-backend", default="auto")
    parser.add_argument("--device-name", default="mlx5_bond_0")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--disable-store-lookup", action="store_true")
    args = parser.parse_args()

    role = "kv_producer" if args.role == "prefill" else "kv_consumer"
    store_extra = {
        "cache_prefix": args.cache_prefix,
        "lookup_rpc_port": args.lookup_rpc_port,
        "store_tp_size": args.tensor_parallel_size,
    }
    if args.role == "decode":
        store_extra["save_decode_cache"] = True
    if args.disable_store_lookup:
        store_extra["enable_lookup"] = False

    config = {
        "kv_connector": "MultiConnector",
        "kv_role": role,
        "kv_connector_extra_config": {
            "load_policy": "range_aware",
            "connectors": [
                {
                    "kv_connector": "MooncakeStoreConnector",
                    "kv_role": "kv_both" if args.role == "prefill" else role,
                    "kv_connector_extra_config": store_extra,
                },
                {
                    "kv_connector": "MooncakeConnector",
                    "kv_role": role,
                    "kv_connector_extra_config": {
                        "mooncake_protocol": "rdma",
                        "device_name": args.device_name,
                    },
                },
            ],
        },
    }
    command = [
        os.environ.get("VLLM_BIN", "vllm"),
        "serve",
        args.model,
        "--served-model-name",
        args.served_model_name,
        "--host",
        "0.0.0.0",
        "--port",
        str(args.port),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--block-size",
        str(args.block_size),
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--kv-cache-dtype",
        args.kv_cache_dtype,
        "--linear-backend",
        args.linear_backend,
        "--enable-prefix-caching",
        "--enable-prompt-tokens-details",
        "--kv-transfer-config",
        json.dumps(config),
    ]
    if args.enforce_eager:
        command.append("--enforce-eager")
    os.execvp(command[0], command)


if __name__ == "__main__":
    main()

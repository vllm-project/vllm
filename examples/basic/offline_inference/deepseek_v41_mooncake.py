# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Publish DeepSeek V4.1 Engram tables to Mooncake Store.

Mooncake connection settings use the standard ``MOONCAKE_*`` environment
variables. Keep this process alive while vLLM workers are serving because its
Store segment owns the published table bytes.
"""

import argparse
import gc
import json
import signal
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoConfig

from vllm.models.deepseek_v4_1.common.engram import EngramLayout
from vllm.models.deepseek_v4_1.nvidia.engram_mooncake import engram_head_shard


def _mooncake_api():
    try:
        from mooncake.store import (
            EngramStore,
            EngramStoreConfig,
            MooncakeDistributedStore,
            ReplicateConfig,
        )
    except ImportError:
        from store import (  # type: ignore[no-redef]
            EngramStore,
            EngramStoreConfig,
            MooncakeDistributedStore,
            ReplicateConfig,
        )
    return EngramStore, EngramStoreConfig, MooncakeDistributedStore, ReplicateConfig


def _weight_map(model: Path) -> dict[str, str]:
    index_path = model / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path, encoding="utf-8") as index_file:
            return json.load(index_file)["weight_map"]
    result = {}
    for shard in sorted(model.glob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as checkpoint:
            result.update({name: shard.name for name in checkpoint})
    if not result:
        raise ValueError(f"No safetensors checkpoints found under {model}")
    return result


def _find_weight(weight_map: dict[str, str], suffix: str) -> tuple[str, str]:
    matches = [
        (name, file) for name, file in weight_map.items() if name.endswith(suffix)
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one checkpoint weight ending in {suffix!r}")
    return matches[0]


def _setup_store():
    from mooncake.mooncake_config import MooncakeConfig

    _, _, MooncakeDistributedStore, _ = _mooncake_api()
    connection = MooncakeConfig.load_from_env()
    store = MooncakeDistributedStore()
    rc = store.setup(
        local_hostname=connection.local_hostname,
        metadata_server=connection.metadata_server,
        global_segment_size=connection.global_segment_size,
        local_buffer_size=connection.local_buffer_size,
        protocol=connection.protocol,
        rdma_devices=connection.device_name or "",
        master_server_addr=connection.master_server_address,
        enable_ssd_offload=connection.enable_ssd_offload,
        ssd_offload_path=connection.ssd_offload_path,
        tenant_id=connection.tenant_id,
        enable_client_http_server=connection.enable_client_http_server,
        client_http_port=connection.client_http_port,
    )
    if rc != 0:
        raise RuntimeError(f"Mooncake Store setup failed, rc={rc}")
    return store


def _pack_head(weight, scale, offset: int, rows: int, dim: int) -> np.ndarray:
    row_bytes = dim + dim // 32
    packed = np.empty((rows, row_bytes), dtype=np.uint8)
    for start in range(0, rows, 65536):
        end = min(start + 65536, rows)
        packed[start:end, :dim] = (
            weight[offset + start : offset + end].view(torch.uint8).numpy()
        )
        packed[start:end, dim:] = (
            scale[offset + start : offset + end].view(torch.uint8).numpy()
        )
    return packed


def _verify_published_shard(store, table, layer_id: int, buffers) -> None:
    """Read the boundary rows back before releasing the source buffers."""
    num_heads = len(buffers)
    row_bytes = buffers[0].shape[1]
    row_ids = np.empty((1, 2, num_heads), dtype=np.int64)
    row_ids[0, 0] = 0
    row_ids[0, 1] = [buffer.shape[0] - 1 for buffer in buffers]
    output = np.empty((1, 2, num_heads, row_bytes), dtype=np.uint8)

    rc = store.register_buffer(output.ctypes.data, output.nbytes)
    if rc != 0:
        raise RuntimeError(f"Could not register verification buffer, rc={rc}")
    try:
        table.lookup_into(layer_id, row_ids, output)
        for head, source in enumerate(buffers):
            if not np.array_equal(output[0, 0, head], source[0]):
                raise RuntimeError(
                    f"Mooncake verification failed for layer {layer_id}, "
                    f"head {head}, first row"
                )
            if not np.array_equal(output[0, 1, head], source[-1]):
                raise RuntimeError(
                    f"Mooncake verification failed for layer {layer_id}, "
                    f"head {head}, last row"
                )
    finally:
        rc = store.unregister_buffer(output.ctypes.data)
        if rc != 0:
            raise RuntimeError(f"Could not unregister verification buffer, rc={rc}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--num-shards", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    hf_config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    text_config = getattr(hf_config, "text_config", hf_config)
    layout = EngramLayout(text_config)
    weight_map = _weight_map(args.model)
    EngramStore, EngramStoreConfig, _, ReplicateConfig = _mooncake_api()
    store = _setup_store()

    manifest = {"version": 1, "num_shards": args.num_shards, "layers": {}}
    configs = {}
    flat_sizes_by_layer = {}
    for layer_index, model_layer_id in enumerate(layout.layer_ids):
        head_sizes = tuple(
            size for order in layout.primes[layer_index] for size in order
        )
        flat_sizes_by_layer[model_layer_id] = head_sizes
        manifest["layers"][str(model_layer_id)] = {
            "table_vocab_sizes": list(head_sizes),
            "head_dim": layout.head_dim,
            "row_bytes": layout.head_dim + layout.head_dim // 32,
        }
        for shard_rank in range(args.num_shards):
            _, local_sizes, _ = engram_head_shard(
                head_sizes, args.num_shards, shard_rank
            )
            config = EngramStoreConfig()
            config.table_vocab_sizes = list(local_sizes)
            config.row_bytes = layout.head_dim + layout.head_dim // 32
            configs[model_layer_id * args.num_shards + shard_rank] = config

    table = EngramStore(configs, store_client=store)
    all_keys = [
        key
        for store_layer_id in sorted(configs)
        for key in table.get_store_keys(store_layer_id)
    ]
    exists = store.batch_is_exist(all_keys)
    if len(exists) != len(all_keys) or any(value != 0 for value in exists):
        store.close()
        raise RuntimeError("Mooncake destination already contains Engram table keys")

    replicate = ReplicateConfig()
    replicate.with_hard_pin = True
    try:
        for layer_index, model_layer_id in enumerate(layout.layer_ids):
            head_sizes = flat_sizes_by_layer[model_layer_id]
            weight_name, weight_file = _find_weight(
                weight_map, f"layers.{model_layer_id}.engram.embed.weight"
            )
            scale_name, scale_file = _find_weight(
                weight_map, f"layers.{model_layer_id}.engram.embed.scale"
            )
            with (
                safe_open(
                    args.model / weight_file, framework="pt", device="cpu"
                ) as weight_checkpoint,
                safe_open(
                    args.model / scale_file, framework="pt", device="cpu"
                ) as scale_checkpoint,
            ):
                weight = weight_checkpoint.get_slice(weight_name)
                scale = scale_checkpoint.get_slice(scale_name)
                if sum(head_sizes) != layout.num_embeddings[layer_index]:
                    raise ValueError(
                        f"Layer {model_layer_id} head sizes do not match its table"
                    )
                global_offsets = np.cumsum((0, *head_sizes[:-1]), dtype=np.int64)
                for shard_rank in range(args.num_shards):
                    head_start, local_sizes, _ = engram_head_shard(
                        head_sizes, args.num_shards, shard_rank
                    )
                    buffers = [
                        _pack_head(
                            weight,
                            scale,
                            int(global_offsets[head]),
                            rows,
                            layout.head_dim,
                        )
                        for head, rows in enumerate(local_sizes, start=head_start)
                    ]
                    store_layer_id = model_layer_id * args.num_shards + shard_rank
                    table.populate(store_layer_id, buffers, replicate)
                    _verify_published_shard(store, table, store_layer_id, buffers)
                    print(
                        f"Published model_layer={model_layer_id} "
                        f"shard={shard_rank}/{args.num_shards} "
                        f"heads={len(buffers)} "
                        f"bytes={sum(buffer.nbytes for buffer in buffers)}",
                        flush=True,
                    )
                    del buffers
                    gc.collect()

        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(manifest, indent=2) + "\n")
        temporary.replace(args.output)
        print(f"READY: {args.output}; keep this process alive", flush=True)
        signal.signal(
            signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt())
        )
        while True:
            signal.pause()
    except KeyboardInterrupt:
        pass
    finally:
        store.close()


if __name__ == "__main__":
    main()

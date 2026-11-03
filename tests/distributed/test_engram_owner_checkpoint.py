# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import struct
import tempfile
import unittest
from pathlib import Path

from vllm.distributed.engram_rdma.owner import (
    MAX_COPY_CHUNK,
    _copy_and_sample,
    _copy_and_sample_row264,
    build_owner_metadata,
    inspect_checkpoint,
)


def write_safetensors(path, tensors, payload):
    header = json.dumps(tensors, separators=(",", ":")).encode()
    header += b" " * ((-len(header)) % 8)
    path.write_bytes(struct.pack("<Q", len(header)) + header + payload)


class Row264OwnerCpuTests(unittest.TestCase):
    def make_checkpoint(self, root):
        config = {
            "text_config": {
                "engram_layer_ids": [1, 14],
                "engram_num_embeddings": [2, 3],
                "engram_head_dim": 256,
            }
        }
        (root / "config.json").write_text(json.dumps(config))
        tensors, weight_map = {}, {}
        payloads = {}
        for layer, rows, shard in (
            (1, 2, "model-00047-of-00048.safetensors"),
            (14, 3, "model-00048-of-00048.safetensors"),
        ):
            weight_name = f"layers.{layer}.engram.embed.weight"
            scale_name = f"layers.{layer}.engram.embed.scale"
            weight = bytes((layer + i) % 256 for i in range(rows * 256))
            scale = bytes((layer + i + 31) % 256 for i in range(rows * 8))
            tensors[weight_name] = {
                "dtype": "F8_E4M3",
                "shape": [rows, 256],
                "data_offsets": [0, len(weight)],
            }
            tensors[scale_name] = {
                "dtype": "F8_E8M0",
                "shape": [rows, 8],
                "data_offsets": [len(weight), len(weight) + len(scale)],
            }
            weight_map[weight_name] = weight_map[scale_name] = shard
            payloads[shard] = (weight, scale)
        (root / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": weight_map})
        )
        for shard, (weight, scale) in payloads.items():
            write_safetensors(
                root / shard,
                {k: v for k, v in tensors.items() if weight_map[k] == shard},
                weight + scale,
            )
        return payloads

    def test_split_and_row264_have_same_bytes_and_checkpoint_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payloads = self.make_checkpoint(root)
            manifest = inspect_checkpoint(root)
            total = sum(spec["nbytes"] for spec in manifest["specs"])
            split_buffer = bytearray(total)
            split_tables, split_samples = _copy_and_sample(
                manifest, memoryview(split_buffer), chunk_bytes=64
            )
            packed_buffer = bytearray(total)
            packed_tables, packed_samples = _copy_and_sample_row264(
                manifest, memoryview(packed_buffer), chunk_bytes=2 * 264
            )

            self.assertEqual(len(split_buffer), len(packed_buffer))
            self.assertEqual(len(split_tables), 4)
            self.assertEqual(
                [(x["layer"], x["kind"]) for x in packed_tables],
                [(1, "row264"), (14, "row264")],
            )
            self.assertTrue(
                all(x["exact_match"] for x in split_samples + packed_samples)
            )
            self.assertEqual(
                manifest["identity_material"],
                inspect_checkpoint(root)["identity_material"],
            )

            offset = 0
            for layer, rows, shard in (
                (1, 2, "model-00047-of-00048.safetensors"),
                (14, 3, "model-00048-of-00048.safetensors"),
            ):
                weight, scale = payloads[shard]
                for row in range(rows):
                    expected = (
                        weight[row * 256 : (row + 1) * 256]
                        + scale[row * 8 : (row + 1) * 8]
                    )
                    self.assertEqual(
                        bytes(
                            packed_buffer[offset + row * 264 : offset + (row + 1) * 264]
                        ),
                        expected,
                    )
                offset += rows * 264

            split_md = build_owner_metadata(
                manifest,
                split_tables,
                [11, 12, 13, 14],
                [x["nbytes"] for x in split_tables],
                0,
                55000,
                b"native-md",
            )
            row_md = build_owner_metadata(
                manifest,
                packed_tables,
                [21, 22],
                [x["nbytes"] for x in packed_tables],
                0,
                55000,
                b"native-md",
                storage_layout="row264",
            )
            self.assertEqual(split_md["storage_layout"], "split")
            self.assertEqual(row_md["storage_layout"], "row264")
            self.assertEqual(split_md["checkpoint_id"], row_md["checkpoint_id"])
            self.assertEqual(len(split_md["addresses"]), 4)
            self.assertEqual(len(row_md["addresses"]), 2)
            self.assertEqual(
                [
                    (x["layer"], x["kind"], x["rows"], x["columns"])
                    for x in row_md["table_specs"]
                ],
                [(1, "row264", 2, 264), (14, "row264", 3, 264)],
            )
            self.assertEqual(
                row_md["table_specs"][0]["component_layout"]["scale"]["offset"], 256
            )

    def test_row264_copy_enforces_bounded_chunk(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_checkpoint(root)
            manifest = inspect_checkpoint(root)
            backing = bytearray(sum(spec["nbytes"] for spec in manifest["specs"]))
            with self.assertRaisesRegex(ValueError, "copy chunk"):
                _copy_and_sample_row264(
                    manifest, memoryview(backing), MAX_COPY_CHUNK + 1
                )

    def test_row264_metadata_rejects_wrong_mr_count_or_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_checkpoint(root)
            manifest = inspect_checkpoint(root)
            tables = [
                {
                    "layer": 1,
                    "kind": "row264",
                    "nbytes": 528,
                    "rows": 2,
                    "columns": 264,
                    "shape": [2, 264],
                },
                {
                    "layer": 14,
                    "kind": "row264",
                    "nbytes": 792,
                    "rows": 3,
                    "columns": 264,
                    "shape": [3, 264],
                },
            ]
            with self.assertRaisesRegex(ValueError, "ordered table MRs"):
                build_owner_metadata(
                    manifest,
                    tables,
                    [1, 2, 3],
                    [528, 792, 0],
                    0,
                    55000,
                    b"md",
                    storage_layout="row264",
                )
            with self.assertRaisesRegex(ValueError, "MR order"):
                build_owner_metadata(
                    manifest,
                    list(reversed(tables)),
                    [1, 2],
                    [792, 528],
                    0,
                    55000,
                    b"md",
                    storage_layout="row264",
                )


if __name__ == "__main__":
    unittest.main()

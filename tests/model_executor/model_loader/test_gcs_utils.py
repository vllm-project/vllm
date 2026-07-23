# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock

from vllm.transformers_utils.gcs_utils import glob, list_files


def _make_mock_blob(name: str) -> MagicMock:
    blob = MagicMock()
    blob.name = name
    return blob


def _make_mock_client(blob_names: list[str]) -> MagicMock:
    client = MagicMock()
    bucket = MagicMock()
    bucket.list_blobs.return_value = [_make_mock_blob(n) for n in blob_names]
    client.bucket.return_value = bucket
    return client


class TestListFiles:
    def test_basic_listing(self):
        mock_client = _make_mock_client([
            "models/model-rank-0-part-0.safetensors",
            "models/model-rank-0-part-1.safetensors",
            "models/config.json",
        ])
        bucket_name, prefix, paths = list_files(
            mock_client, path="gs://my-bucket/models/"
        )
        assert bucket_name == "my-bucket"
        assert prefix == "models/"
        assert len(paths) == 3

    def test_allow_pattern(self):
        mock_client = _make_mock_client([
            "models/model-rank-0-part-0.safetensors",
            "models/model-rank-0-part-1.safetensors",
            "models/model-rank-1-part-0.safetensors",
            "models/config.json",
        ])
        _, _, paths = list_files(
            mock_client,
            path="gs://my-bucket/models/",
            allow_pattern=["*model-rank-0-part-*.safetensors"],
        )
        assert len(paths) == 2
        assert all("rank-0" in p for p in paths)

    def test_ignore_pattern(self):
        mock_client = _make_mock_client([
            "models/model-rank-0-part-0.safetensors",
            "models/config.json",
        ])
        _, _, paths = list_files(
            mock_client,
            path="gs://my-bucket/models/",
            ignore_pattern=["*.json"],
        )
        assert len(paths) == 1
        assert paths[0].endswith(".safetensors")


class TestGlob:
    def test_returns_full_gs_paths(self):
        mock_client = _make_mock_client([
            "models/model-rank-0-part-0.safetensors",
            "models/model-rank-0-part-1.safetensors",
        ])
        result = glob(
            client=mock_client,
            path="gs://my-bucket/models/",
            allow_pattern=["*model-rank-0-part-*.safetensors"],
        )
        assert len(result) == 2
        assert all(p.startswith("gs://my-bucket/") for p in result)
        assert all(p.endswith(".safetensors") for p in result)

    def test_adds_trailing_slash(self):
        mock_client = _make_mock_client([])
        glob(client=mock_client, path="gs://my-bucket/models")
        mock_client.bucket.assert_called_once_with("my-bucket")
        bucket = mock_client.bucket.return_value
        bucket.list_blobs.assert_called_once_with(prefix="models/")

    def test_filters_by_rank_pattern(self):
        mock_client = _make_mock_client([
            "models/model-rank-0-part-0.safetensors",
            "models/model-rank-1-part-0.safetensors",
            "models/model-rank-2-part-0.safetensors",
            "models/config.json",
            "models/tokenizer.json",
        ])
        result = glob(
            client=mock_client,
            path="gs://my-bucket/models",
            allow_pattern=["*model-rank-2-part-*.safetensors"],
        )
        assert len(result) == 1
        assert "rank-2" in result[0]

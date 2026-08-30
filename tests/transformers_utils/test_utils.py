# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.transformers_utils.utils import (
    is_azure,
    is_cloud_storage,
    is_gcs,
    is_s3,
    normalize_atomgit_repo_id,
)


def test_is_gcs():
    assert is_gcs("gs://model-path")
    assert not is_gcs("s3://model-path/path-to-model")
    assert not is_gcs("/unix/local/path")
    assert not is_gcs("nfs://nfs-fqdn.local")


def test_is_s3():
    assert is_s3("s3://model-path/path-to-model")
    assert not is_s3("gs://model-path")
    assert not is_s3("/unix/local/path")
    assert not is_s3("nfs://nfs-fqdn.local")


def test_is_azure():
    assert is_azure("az://model-container/path")
    assert not is_azure("s3://model-path/path-to-model")
    assert not is_azure("/unix/local/path")
    assert not is_azure("nfs://nfs-fqdn.local")


def test_is_cloud_storage():
    assert is_cloud_storage("gs://model-path")
    assert is_cloud_storage("s3://model-path/path-to-model")
    assert is_cloud_storage("az://model-container/path")
    assert not is_cloud_storage("/unix/local/path")
    assert not is_cloud_storage("nfs://nfs-fqdn.local")


def test_normalize_atomgit_repo_id():
    # 1-2 segment ids are already in AtomGit's canonical form.
    assert normalize_atomgit_repo_id("zai-org/GLM-4.6") == "zai-org/GLM-4.6"
    assert normalize_atomgit_repo_id("someuser") == "someuser"
    # 3+ segment ids collapse everything before the last slash into the owner.
    assert normalize_atomgit_repo_id("hf_mirrors/Qwen/Qwen2.5-7B-Instruct") == \
        "hf_mirrors-Qwen/Qwen2.5-7B-Instruct"
    assert normalize_atomgit_repo_id("hf_mirrors/BAAI/bge-reranker-v2-m3") == \
        "hf_mirrors-BAAI/bge-reranker-v2-m3"
    assert normalize_atomgit_repo_id("a/b/c/d") == "a-b-c/d"
    # Normalization is idempotent: the result is already in owner/repo form.
    assert normalize_atomgit_repo_id("hf_mirrors-Qwen/Qwen2.5-7B-Instruct") == \
        "hf_mirrors-Qwen/Qwen2.5-7B-Instruct"
    # URLs, object-storage URIs and paths are not repo ids.
    assert normalize_atomgit_repo_id("s3://bucket/prefix/model") == \
        "s3://bucket/prefix/model"
    assert normalize_atomgit_repo_id("gs://bucket/prefix/model") == \
        "gs://bucket/prefix/model"
    assert normalize_atomgit_repo_id("az://container/prefix/model") == \
        "az://container/prefix/model"
    assert normalize_atomgit_repo_id("runai://org/model") == "runai://org/model"
    assert normalize_atomgit_repo_id("https://host/a/b/c") == \
        "https://host/a/b/c"
    assert normalize_atomgit_repo_id("/abs/path/to/model") == \
        "/abs/path/to/model"
    assert normalize_atomgit_repo_id("./rel/path/to/model") == \
        "./rel/path/to/model"
    assert normalize_atomgit_repo_id("../rel/path/to/model") == \
        "../rel/path/to/model"

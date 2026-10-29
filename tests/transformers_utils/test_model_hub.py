# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import fnmatch
import json
import shutil
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from transformers import Qwen3Config

from vllm.config import LoadConfig, ParallelConfig
from vllm.config.model import get_served_model_name
from vllm.engine.arg_utils import EngineArgs
from vllm.transformers_utils.model_hub import (
    parse_model_source,
    resolve_model_source,
)


@pytest.fixture
def hubs(monkeypatch, tmp_path):
    calls = []
    remote = tmp_path / "remote"
    Qwen3Config(
        architectures=["Qwen3ForCausalLM"], tie_word_embeddings=True
    ).save_pretrained(remote)
    save_file(
        {"lm_head.weight": torch.zeros(1, dtype=torch.bfloat16)},
        remote / "model.safetensors",
    )
    (remote / "pytorch_model.bin").write_bytes(b"unused")

    def download(source, repo, **kwargs):
        calls.append((source, repo, kwargs))
        snapshot = tmp_path / source / repo
        snapshot.mkdir(parents=True, exist_ok=True)
        ignore = kwargs.get("ignore_patterns") or []
        if isinstance(ignore, str):
            ignore = [ignore]
        for file in remote.rglob("*"):
            if not file.is_file():
                continue
            relative = file.relative_to(remote)
            if any(
                fnmatch.fnmatch(str(relative), p)
                for p in kwargs.get("allow_patterns") or ["*"]
            ) and not any(fnmatch.fnmatch(str(relative), p) for p in ignore):
                (snapshot / relative).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(file, snapshot / relative)
        return str(snapshot)

    monkeypatch.setattr(
        "vllm.transformers_utils.model_hub.hf_snapshot_download",
        lambda repo, **kw: download("hf", repo, **kw),
    )
    monkeypatch.setattr(
        "modelscope.snapshot_download",
        lambda repo, **kw: download("modelscope", repo, **kw),
    )
    return remote, calls


@pytest.mark.parametrize("source", ["hf", "modelscope"])
@pytest.mark.parametrize(
    "repo",
    ["model", "owner/model/file", "owner/model@rev", "owner/model?x=1", "../model"],
)
def test_invalid_reference(source, repo, hubs):
    with pytest.raises(ValueError):
        resolve_model_source(f"{source}://{repo}")
    assert not hubs[1]


@pytest.mark.parametrize("path", ["owner/model", "/models/local", "s3://bucket/model"])
def test_existing_paths_are_unchanged(path, hubs):
    args = EngineArgs(model=path, tokenizer="owner/tokenizer", revision="revision")
    assert args.model == path
    assert args.tokenizer == "owner/tokenizer"
    assert args.tokenizer_revision is None
    assert args.model_weights == ""
    assert resolve_model_source(path) == path
    assert parse_model_source(path) is None
    assert not hubs[1]


@pytest.mark.parametrize(
    "target_source,draft_source", [("hf", "modelscope"), ("modelscope", "hf")]
)
@pytest.mark.parametrize("use_modelscope", ["True", "False"])
def test_mixed_sources_load_local_weights(
    monkeypatch, tmp_path, hubs, target_source, draft_source, use_modelscope
):
    monkeypatch.setenv("VLLM_USE_MODELSCOPE", use_modelscope)
    args = EngineArgs(
        model=f"{target_source}://owner/target",
        revision="target-rev",
        download_dir=str(tmp_path),
        hf_token="test-token",
        speculative_config={
            "model": f"{draft_source}://owner/draft",
            "method": "draft_model",
            "num_speculative_tokens": 2,
            "revision": "draft-rev",
        },
    )
    assert args.model == args.tokenizer
    assert args.served_model_name == ["owner/target"]
    target = args.create_model_config()
    # Cold-cache weights are available before dtype / tied-head inspection.
    assert target.dtype == torch.bfloat16
    target.maybe_untie_word_embeddings()
    assert not target.hf_config.tie_word_embeddings
    config = args.create_speculative_config(target, ParallelConfig())
    assert config is not None
    assert Path(config.draft_model_config.model).parent.name == "owner"
    assert Path(config.draft_model_config.model).parents[1].name == draft_source
    for source, repo, kwargs in hubs[1]:
        assert source == (target_source if repo == "owner/target" else draft_source)
        assert kwargs["revision"] == (
            "target-rev" if repo == "owner/target" else "draft-rev"
        )
        assert kwargs["cache_dir"] == str(tmp_path)
        assert kwargs.get("token") == ("test-token" if source == "hf" else None)
    assert (Path(args.model) / "model.safetensors").is_file()


def test_independent_tokenizer_and_default_reuse(hubs):
    args = EngineArgs(model="hf://owner/model")
    assert args.distributed_executor_backend is None
    count = len(hubs[1])
    assert args.model == args.tokenizer
    args = EngineArgs(
        model="owner/model",
        revision="model-rev",
        tokenizer="modelscope://owner/tokenizer",
        tokenizer_revision="tokenizer-rev",
    )
    assert args.model == "owner/model"
    assert hubs[1][-1][2]["revision"] == "tokenizer-rev"
    assert len(hubs[1]) == count + 1  # Tokenizer does not download weights.
    assert not list(Path(args.tokenizer).glob("*.safetensors"))
    assert get_served_model_name("hf://owner/model", None) == "owner/model"
    assert get_served_model_name("hf://owner/model", ["alias"]) == "alias"


def test_invalid_tokenizer_fails_before_model_download(hubs):
    with pytest.raises(ValueError):
        EngineArgs(model="hf://owner/model", tokenizer="hf://invalid")
    assert not hubs[1]


def test_weight_ignore_patterns_preserve_config(hubs):
    args = EngineArgs(model="hf://owner/model", ignore_patterns=["*.json"])
    assert args.create_model_config().hf_config.model_type == "qwen3"


@pytest.mark.parametrize("content", ['[{"stage":"train"}]', "not-json"])
def test_unrelated_configs_do_not_affect_loading(hubs, content):
    (hubs[0] / "training_config.json").write_text(content)
    args = EngineArgs(model="hf://owner/model")
    assert args.create_model_config().hf_config.model_type == "qwen3"


def test_load_format_is_case_insensitive(hubs):
    args = EngineArgs(model="hf://owner/model", load_format="SAFETENSORS")
    assert args.create_load_config().load_format == "safetensors"
    assert args.create_model_config().hf_config.model_type == "qwen3"


@pytest.mark.parametrize("subfolder", [None, "audio"])
def test_auxiliary_weights_remain_available(hubs, subfolder):
    from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

    folder = hubs[0] / subfolder if subfolder else hubs[0]
    folder.mkdir(exist_ok=True)
    torch.save({"weight": torch.zeros(1)}, folder / "sparse_linear.pt")
    local = resolve_model_source("hf://owner/model")
    _, files, _, _ = DefaultModelLoader(LoadConfig())._prepare_weights(
        local, subfolder, None, True, ["sparse_linear.pt"]
    )
    assert [Path(f).name for f in files] == ["sparse_linear.pt"]


def test_explicit_draft_load_config_does_not_inherit_target(hubs, tmp_path):
    remote, calls = hubs
    args = EngineArgs(
        model=str(remote),
        load_format="safetensors",
        ignore_patterns=["*.bin"],
        speculative_config={
            "model": "hf://owner/draft",
            "method": "draft_model",
            "num_speculative_tokens": 2,
            "draft_load_config": {"download_dir": str(tmp_path)},
        },
    )
    target = args.create_model_config()
    (remote / "model.safetensors").unlink()
    torch.save({"lm_head.weight": torch.zeros(1)}, remote / "pytorch_model.bin")
    config = args.create_speculative_config(target, ParallelConfig())
    assert config is not None
    assert config.draft_load_config is not None
    assert config.draft_load_config.load_format == "auto"
    assert (Path(config.draft_model_config.model) / "pytorch_model.bin").is_file()
    assert calls[-1][2]["ignore_patterns"] == config.draft_load_config.ignore_patterns


def test_snapshot_ignore_and_offline(monkeypatch, hubs):
    remote, calls = hubs
    (remote / "unused.safetensors").write_bytes(b"unused")
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_OFFLINE", True)
    path = resolve_model_source("hf://owner/model", ignore_patterns=["unused*"])
    assert sorted(p.name for p in Path(path).glob("*.safetensors")) == [
        "model.safetensors"
    ]
    assert calls[-1][2]["allow_patterns"] is None
    assert all(c[2]["local_files_only"] for c in calls)


@pytest.mark.parametrize("source", ["hf", "modelscope"])
def test_download_failure_never_falls_back(monkeypatch, hubs, source):
    failure = OSError("Hub unavailable")

    def fail(*args, **kwargs):
        raise failure

    sdk = (
        "vllm.transformers_utils.model_hub.hf_snapshot_download"
        if source == "hf"
        else "modelscope.snapshot_download"
    )
    monkeypatch.setattr(sdk, fail)
    with pytest.raises(OSError) as caught:
        resolve_model_source(f"{source}://owner/model")
    assert caught.value is failure
    assert source in caught.value.__notes__[0]
    assert not hubs[1]


def test_modelscope_is_optional(monkeypatch, hubs):
    monkeypatch.setitem(sys.modules, "modelscope", None)
    assert Path(resolve_model_source("hf://owner/model")).is_dir()
    with pytest.raises(ImportError, match="pip install modelscope"):
        resolve_model_source("modelscope://owner/model")


@pytest.mark.parametrize(
    "options",
    [
        {"nnodes": 2},
        {"distributed_executor_backend": "ray"},
        {"data_parallel_size": 2, "data_parallel_size_local": 1},
    ],
)
def test_unsupported_distribution_fails_before_download(hubs, options):
    with pytest.raises(ValueError, match="single-node"):
        EngineArgs(model="hf://owner/model", **options)
    assert not hubs[1]


def test_unsupported_loading_and_code(hubs):
    with pytest.raises(ValueError, match="loading only"):
        EngineArgs(model="hf://owner/model", load_format="tensorizer")
    with pytest.raises(ValueError, match="code_revision"):
        EngineArgs(model="hf://owner/model", code_revision="other")
    assert not hubs[1]
    (hubs[0] / "config.json").write_text(
        json.dumps(
            {"auto_map": {"AutoConfig": "owner/code--configuration.CustomConfig"}}
        )
    )
    with pytest.raises(ValueError, match="cross-repository"):
        EngineArgs(model="hf://owner/model", trust_remote_code=True)


def test_speculator_requires_explicit_target(hubs):
    (hubs[0] / "config.json").write_text(
        json.dumps({"speculators_config": {"verifier": {}}})
    )
    with pytest.raises(ValueError, match="target model explicitly"):
        EngineArgs(model="hf://owner/model")
    assert Path(resolve_model_source("hf://owner/draft", draft=True)).is_dir()


def test_builtin_mtp_reuses_local_target(hubs):
    from transformers import DeepseekV3Config

    DeepseekV3Config(
        architectures=["DeepseekV3ForCausalLM"],
        num_nextn_predict_layers=1,
    ).save_pretrained(hubs[0])
    args = EngineArgs(
        model="modelscope://owner/model",
        speculative_config={"method": "mtp", "num_speculative_tokens": 1},
    )
    count = len(hubs[1])
    config = args.create_speculative_config(
        args.create_model_config(), ParallelConfig()
    )
    assert config is not None
    assert config.draft_model_config.model == args.model
    assert len(hubs[1]) == count


def test_hf_downloader_survives_modelscope_patch(tmp_path):
    import httpx
    from huggingface_hub import set_client_factory
    from modelscope.utils.hf_util import patch_context

    sha = "a" * 40
    content = b'{"model_type":"qwen3"}'

    def respond(request):
        assert request.url.host == "huggingface.co"
        if "/tree/" in request.url.path:
            return httpx.Response(
                200,
                json=[
                    dict(type="file", path="config.json", size=len(content), oid=sha)
                ],
            )
        if request.url.path.startswith("/api/models/"):
            return httpx.Response(
                200,
                json=dict(
                    id="owner/model", sha=sha, siblings=[{"rfilename": "config.json"}]
                ),
            )
        return httpx.Response(
            200, content=content, headers={"X-Repo-Commit": sha, "ETag": '"config"'}
        )

    set_client_factory(lambda: httpx.Client(transport=httpx.MockTransport(respond)))
    try:
        with patch_context():
            path = resolve_model_source(
                "hf://owner/model", download_dir=str(tmp_path), tokenizer_only=True
            )
        assert (Path(path) / "config.json").read_bytes() == content
    finally:
        set_client_factory(httpx.Client)

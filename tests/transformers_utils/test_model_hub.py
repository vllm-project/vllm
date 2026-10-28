# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import fnmatch
import io
import json
import shutil
import struct
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import Mock

import pytest
import requests
import torch
from transformers import AutoConfig, AutoTokenizer, Qwen3Config

from vllm.config import ParallelConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.model_hub import parse_model_source, resolve_model_source, strip_model_source
from vllm.transformers_utils.repo_utils import hf_api

HUB_DOWNLOADS = {
    "hf": "vllm.transformers_utils.repo_utils._hf_snapshot_download",
    "modelscope": "modelscope.snapshot_download",
}


@pytest.fixture(autouse=True)
def hub_metadata(monkeypatch):
    from huggingface_hub.utils import (
        SafetensorsFileMetadata,
        SafetensorsRepoMetadata,
        TensorInfo,
    )
    from modelscope.hub.api import HubApi

    HfApi = type(hf_api())
    monkeypatch.setattr(
        HfApi, "resolve_revision", lambda self, repo, **kw: kw.get("revision")
    )
    monkeypatch.setattr(
        HfApi, "list_repo_files", lambda self, repo, **kw: ["config.json"]
    )
    monkeypatch.setattr(HubApi, "get_model_files", lambda self, model_id, **kw: [])
    monkeypatch.setattr(
        HfApi,
        "get_safetensors_metadata",
        lambda self, repo, **kw: SafetensorsRepoMetadata(
            None,
            False,
            {},
            {
                "model.safetensors": SafetensorsFileMetadata(
                    {}, {"lm_head.weight": TensorInfo("BF16", [2], (0, 4))}
                )
            },
        ),
    )


@pytest.fixture
def hubs(monkeypatch, tmp_path):
    downloads = {}
    for source, sdk in HUB_DOWNLOADS.items():
        path = tmp_path / source
        Qwen3Config().save_pretrained(path)
        downloads[source] = Mock(return_value=str(path))
        monkeypatch.setattr(sdk, downloads[source])
    return downloads


@pytest.fixture
def custom_code(tmp_path):
    code = tmp_path / "code"
    code.mkdir()
    (code / "helper.py").write_text("VALUE = 42\n")
    (code / "configuration.py").write_text(
        "from transformers import PretrainedConfig\n"
        "from .helper import VALUE\n"
        "class CustomConfig(PretrainedConfig):\n"
        "    marker = VALUE\n"
    )
    (code / "tokenization.py").write_text(
        "from transformers import PreTrainedTokenizer\n"
        "from .helper import VALUE\n"
        "class CustomTokenizer(PreTrainedTokenizer):\n"
        "    vocab_files_names = {}\n"
        "    def get_vocab(self): return {'hello': VALUE}\n"
        "    def _tokenize(self, text): return [text]\n"
        "    def _convert_token_to_id(self, token): return VALUE\n"
    )
    return code


@pytest.mark.parametrize("source", ["hf", "modelscope"])
@pytest.mark.parametrize(
    "repo",
    ["model", "owner/model/file", "owner/model@rev", "owner/model?x=1", "../model"],
)
def test_invalid_reference(source, repo):
    with pytest.raises(ValueError):
        resolve_model_source(f"{source}://{repo}")


@pytest.mark.parametrize("path", ["owner/model", "/models/local", "s3://bucket/model"])
def test_existing_paths_are_unchanged(path):
    assert parse_model_source(path) is None
    assert resolve_model_source(path) == path
    assert strip_model_source(path) == path
    args = EngineArgs(model=path, tokenizer="owner/tokenizer", revision="revision")
    assert args.model == path
    assert args.model_weights == ""
    assert args.tokenizer == "owner/tokenizer"
    assert args.tokenizer_revision is None


@pytest.mark.parametrize("source", ["hf", "modelscope"])
def test_served_name_omits_only_the_source(source):
    from vllm.config.model import get_served_model_name

    assert strip_model_source(f"{source}://owner/model") == "owner/model"
    assert get_served_model_name(f"{source}://owner/model", None) == "owner/model"
    assert get_served_model_name(f"{source}://owner/model", ["custom"]) == "custom"


@pytest.mark.parametrize("source", ["hf", "modelscope"])
@pytest.mark.parametrize("use_modelscope", ["True", "False"])
def test_explicit_source_options_and_failure(
    monkeypatch, tmp_path, hubs, source, use_modelscope
):
    monkeypatch.setenv("VLLM_USE_MODELSCOPE", use_modelscope)
    selected = hubs[source]
    other = hubs["modelscope" if source == "hf" else "hf"]
    other.side_effect = AssertionError("Accessed the other Hub")
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_OFFLINE", True)
    uri = f"{source}://owner/model"
    assert (
        resolve_model_source(
            uri,
            revision="revision",
            download_dir=str(tmp_path),
            hf_token="hf-token",
        )
        == selected.return_value
    )
    assert selected.call_args.args == ("owner/model",)
    kwargs = selected.call_args.kwargs
    assert kwargs["revision"] == "revision"
    assert kwargs["cache_dir"] == str(tmp_path)
    assert kwargs["local_files_only"] is True
    assert "*.json" in kwargs["allow_patterns"]
    assert "*.safetensors" not in kwargs["allow_patterns"]
    assert kwargs.get("token") == ("hf-token" if source == "hf" else None)

    failure = OSError("download failed")
    selected.side_effect = failure
    with pytest.raises(OSError) as caught:
        resolve_model_source(uri)
    assert caught.value is failure
    assert uri in caught.value.__notes__[0]
    other.assert_not_called()


def test_hf_download_survives_modelscope_patch(tmp_path):
    """Exercise the real HF downloader with ModelScope's global patch enabled."""
    import httpx
    from huggingface_hub import set_client_factory
    from modelscope.utils.hf_util import patch_context

    sha = "a" * 40
    content = b'{"model_type":"qwen3"}'

    def respond(request):
        assert request.url.host == "huggingface.co"
        if "/tree/" in request.url.path:
            file = dict(type="file", path="config.json", size=len(content), oid=sha)
            return httpx.Response(200, json=[file])
        if request.url.path.startswith("/api/models/"):
            info = dict(
                id="owner/model", sha=sha, siblings=[{"rfilename": "config.json"}]
            )
            return httpx.Response(200, json=info)
        return httpx.Response(
            200,
            content=content,
            headers={"X-Repo-Commit": sha, "ETag": '"config"'},
        )

    set_client_factory(lambda: httpx.Client(transport=httpx.MockTransport(respond)))
    try:
        with patch_context():
            path = resolve_model_source("hf://owner/model", download_dir=str(tmp_path))
        assert (Path(path) / "config.json").read_bytes() == content
    finally:
        set_client_factory(httpx.Client)


@pytest.mark.parametrize("source", ["hf", "modelscope"])
@pytest.mark.parametrize("patched", [False, True])
@pytest.mark.parametrize("code_revision", [None, "code-rev"])
def test_cross_repo_code_keeps_source_and_revision(
    monkeypatch, tmp_path, custom_code, source, patched, code_revision
):
    from modelscope.utils.hf_util import patch_context

    snapshot = tmp_path / "models--owner--model"
    snapshot.mkdir()
    config = {"auto_map": {"AutoConfig": "owner/code--configuration.CustomConfig"}}
    tokenizer_config = {
        "auto_map": {
            "AutoTokenizer": ["owner/code--tokenization.CustomTokenizer", None]
        }
    }
    for name, data in (
        ("config.json", config),
        ("tokenizer_config.json", tokenizer_config),
    ):
        (snapshot / name).write_text(json.dumps(data))
    (snapshot / "weights.safetensors").write_bytes(b"weight-placeholder")

    def download(repo, **kwargs):
        assert kwargs.get("revision") == (
            "weights-rev" if repo == "owner/model" else code_revision
        )
        return str(snapshot if repo == "owner/model" else custom_code)

    with patch_context() if patched else nullcontext():
        downloads = {}
        for hub, sdk in HUB_DOWNLOADS.items():
            downloads[hub] = Mock(
                side_effect=download if hub == source else AssertionError("Wrong Hub")
            )
            monkeypatch.setattr(sdk, downloads[hub])
        path = resolve_model_source(
            f"{source}://owner/model",
            revision="weights-rev",
            code_revision=code_revision,
            download_dir=str(tmp_path),
            trust_remote_code=True,
        )
        assert AutoConfig.from_pretrained(path, trust_remote_code=True).marker == 42
        with pytest.raises(ValueError):
            AutoConfig.from_pretrained(path, trust_remote_code=False)
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        assert tokenizer.encode("hello", add_special_tokens=False) == [42]
        assert (Path(path) / "weights.safetensors").samefile(
            snapshot / "weights.safetensors"
        )
        assert json.loads((snapshot / "config.json").read_text()) == config
        selected = downloads.pop(source)
        other = next(iter(downloads.values()))
        assert [call.args[0] for call in selected.call_args_list] == [
            "owner/model",
            "owner/code",
        ]
        failure = OSError("code download failed")
        selected.side_effect = [str(snapshot), failure]
        with pytest.raises(OSError) as caught:
            resolve_model_source(f"{source}://owner/model", trust_remote_code=True)
        assert caught.value is failure
        assert f"{source}://owner/code" in failure.__notes__[0]
        other.assert_not_called()


def test_same_repo_code_revision_and_cached_overlay(monkeypatch, tmp_path, custom_code):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text(
        json.dumps({"auto_map": {"AutoConfig": "configuration.CustomConfig"}})
    )

    def download(repo, **kwargs):
        assert repo == "owner/model"
        return str(custom_code if kwargs.get("revision") == "code-rev" else snapshot)

    monkeypatch.setattr(
        "vllm.transformers_utils.repo_utils._hf_snapshot_download", download
    )
    options = dict(
        code_revision="code-rev", download_dir=str(tmp_path), trust_remote_code=True
    )
    path = resolve_model_source("hf://owner/model", **options)
    assert AutoConfig.from_pretrained(path, trust_remote_code=True).marker == 42
    assert resolve_model_source("hf://owner/model", **options) == path
    # A later full download must not reuse a metadata-only overlay missing weights.
    (snapshot / "model.safetensors").write_bytes(b"weight-placeholder")
    full = resolve_model_source("hf://owner/model", **options)
    assert (Path(full) / "model.safetensors").samefile(snapshot / "model.safetensors")


@pytest.mark.parametrize(
    "target_source,draft_source", [("hf", "modelscope"), ("modelscope", "hf")]
)
@pytest.mark.parametrize("use_modelscope", ["True", "False"])
def test_engine_args_mixed_hubs(
    monkeypatch, tmp_path, hubs, target_source, draft_source, use_modelscope
):
    monkeypatch.setenv("VLLM_USE_MODELSCOPE", use_modelscope)
    target, draft = hubs[target_source], hubs[draft_source]
    args = EngineArgs(
        model=f"{target_source}://owner/target",
        revision="target-rev",
        download_dir=str(tmp_path),
        load_format="dummy",
        speculative_config={
            "model": f"{draft_source}://owner/draft",
            "method": "draft_model",
            "num_speculative_tokens": 2,
            "revision": "draft-rev",
        },
    )
    assert args.tokenizer == args.model == target.return_value
    assert args.served_model_name == ["owner/target"]
    assert target.call_count == 1  # Default tokenizer reuses the model snapshot.
    config = args.create_speculative_config(
        args.create_model_config(), ParallelConfig()
    )
    assert config is not None
    assert config.draft_model_config.model == draft.return_value
    assert config.draft_model_config.model_weights == f"{draft_source}://owner/draft"
    assert args.model_weights == f"{target_source}://owner/target"
    assert target.call_args.args == ("owner/target",)
    assert draft.call_args.args == ("owner/draft",)
    assert target.call_args.kwargs["revision"] == "target-rev"
    assert draft.call_args.kwargs["revision"] == "draft-rev"
    assert draft.call_args.kwargs["cache_dir"] == str(tmp_path)

    args = EngineArgs(
        model="owner/target",
        tokenizer=f"{draft_source}://owner/tokenizer",
        tokenizer_revision="tokenizer-rev",
    )
    assert args.model == "owner/target"
    assert args.tokenizer == draft.return_value
    assert draft.call_args.args == ("owner/tokenizer",)
    assert draft.call_args.kwargs["revision"] == "tokenizer-rev"


@pytest.mark.parametrize("source", ["hf", "modelscope"])
def test_untrusted_auto_map_does_not_fetch_code(hubs, source):
    snapshot = Path(hubs[source].return_value)
    config = json.loads((snapshot / "config.json").read_text())
    config["auto_map"] = {"AutoConfig": "unavailable/code--configuration.Config"}
    (snapshot / "config.json").write_text(json.dumps(config))
    args = EngineArgs(model=f"{source}://owner/model", trust_remote_code=False)
    assert args.create_model_config().hf_config.model_type == "qwen3"
    assert {call.args[0] for call in hubs[source].call_args_list} == {"owner/model"}


def test_builtin_mtp_keeps_target_source_and_revision(hubs):
    from transformers import DeepseekV3Config

    DeepseekV3Config(
        architectures=["DeepseekV3ForCausalLM"], num_nextn_predict_layers=1
    ).save_pretrained(hubs["modelscope"].return_value)
    args = EngineArgs(
        model="modelscope://owner/model",
        revision="target-rev",
        speculative_config={"method": "mtp", "num_speculative_tokens": 1},
    )
    target = args.create_model_config()
    speculative = args.create_speculative_config(target, ParallelConfig())
    assert speculative is not None
    assert speculative.draft_model_config.model_weights == "modelscope://owner/model"
    assert speculative.draft_model_config.revision == "target-rev"
    assert all(
        call.kwargs["revision"] == "target-rev"
        for call in hubs["modelscope"].call_args_list
    )
    hubs["hf"].assert_not_called()


def test_independent_tokenizer_does_not_use_model_code_revision(
    monkeypatch, tmp_path, custom_code
):
    model, tokenizer = tmp_path / "model", custom_code
    Qwen3Config().save_pretrained(model)
    (tokenizer / "tokenizer_config.json").write_text(
        json.dumps(
            {"auto_map": {"AutoTokenizer": ["tokenization.CustomTokenizer", None]}}
        )
    )

    def download(repo, **kwargs):
        assert repo == "owner/tokenizer"
        assert kwargs["revision"] == "tokenizer-rev"
        return str(tokenizer)

    monkeypatch.setattr("modelscope.snapshot_download", download)
    args = EngineArgs(
        model=str(model),
        tokenizer="modelscope://owner/tokenizer",
        code_revision="model-code-rev",
        tokenizer_revision="tokenizer-rev",
        trust_remote_code=True,
    )
    loaded = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    assert loaded.encode("hello", add_special_tokens=False) == [42]


@pytest.mark.parametrize("source", ["hf", "modelscope"])
@pytest.mark.parametrize(
    "load_format,filename", [("auto", "model.safetensors"), ("pt", "model.pt")]
)
def test_loader_filters_weights_and_resolves_each_node(
    monkeypatch, tmp_path, source, load_format, filename
):
    from safetensors.torch import save_file

    from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

    repo = tmp_path / "repo"
    Qwen3Config().save_pretrained(repo)
    weights = {"weight": torch.tensor([1.0, 2.0])}
    save_file(weights, repo / "model.safetensors")
    torch.save(weights, repo / "model.pt")
    torch.save(weights, repo / "sparse_linear.pt")
    for name in (
        "pytorch_model.bin",
        "model.h5",
        "model.msgpack",
        "model.onnx",
        "model.gguf",
        "original/model.safetensors",
        "unused.safetensors",
    ):
        path = repo / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"must not download")
    (repo / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": "model.safetensors"}})
    )
    cache = tmp_path / "head"

    def download(repo_id, **kwargs):
        assert repo_id == "owner/model"
        assert kwargs["revision"] == "weights-rev"
        assert kwargs.get("token") == ("test-token" if source == "hf" else None)
        cache.mkdir(exist_ok=True)
        for file in repo.rglob("*"):
            name = file.relative_to(repo).as_posix()
            if (
                file.is_file()
                and any(fnmatch.fnmatch(name, p) for p in kwargs["allow_patterns"])
                and not any(fnmatch.fnmatch(name, p) for p in kwargs["ignore_patterns"])
            ):
                dest = cache / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(file, dest)
        return str(cache)

    monkeypatch.setattr(HUB_DOWNLOADS[source], download)
    monkeypatch.setenv("VLLM_USE_MODELSCOPE", "True" if source == "hf" else "False")
    args = EngineArgs(
        model=f"{source}://owner/model",
        revision="weights-rev",
        hf_token="test-token",
        load_format=load_format,
    )
    config = args.create_model_config()
    assert sorted(p.name for p in cache.iterdir()) == [
        "config.json",
        "model.safetensors.index.json",
    ]
    # A second node cannot read the head's cache directory.
    shutil.rmtree(cache)
    cache = tmp_path / "worker"
    loader = DefaultModelLoader(args.create_load_config())
    model = torch.nn.Linear(1, 1)
    model.secondary_weights = [
        DefaultModelLoader.Source(
            config.model,
            config.revision,
            prefix="sparse.",
            allow_patterns_overrides=["sparse_linear.pt"],
        )
    ]
    actual = dict(loader.get_all_weights(config, model))
    assert torch.equal(actual["weight"], weights["weight"])
    assert torch.equal(actual["sparse.weight"], weights["weight"])
    assert sorted(p.name for p in cache.iterdir()) == sorted(
        [filename, "sparse_linear.pt", "model.safetensors.index.json"]
    )


@pytest.mark.parametrize("source", ["hf", "modelscope"])
def test_cold_hub_snapshot_preserves_weight_metadata(monkeypatch, hubs, source):
    from modelscope.hub.api import HubApi

    snapshot = Path(hubs[source].return_value)
    Qwen3Config(tie_word_embeddings=True).save_pretrained(snapshot)
    config = json.loads((snapshot / "config.json").read_text())
    config.pop("dtype", None)
    (snapshot / "config.json").write_text(json.dumps(config))

    header = json.dumps(
        {"lm_head.weight": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]}}
    ).encode()
    data = struct.pack("<Q", len(header)) + header
    monkeypatch.setenv("MODELSCOPE_API_TOKEN", "test-token")
    monkeypatch.setattr(
        HubApi,
        "get_model_files",
        lambda self, model_id, **kw: [{"Path": "model.safetensors"}],
    )

    def send(self, request, **kwargs):
        assert source == "modelscope"
        assert request.url.startswith("https://modelscope.cn/")
        assert request.headers["Authorization"] == "Bearer test-token"
        assert request.headers["Range"].startswith("bytes=0-")
        assert kwargs["stream"] is True
        response = requests.Response()
        response.status_code = 206
        response.raw = io.BytesIO(data)
        return response

    monkeypatch.setattr(requests.Session, "send", send)
    model = EngineArgs(model=f"{source}://owner/model").create_model_config()
    model.maybe_untie_word_embeddings()
    assert model.hf_config.tie_word_embeddings is False
    assert model.dtype == torch.bfloat16
    assert not list(snapshot.glob("*.safetensors"))
    hubs["modelscope" if source == "hf" else "hf"].assert_not_called()


@pytest.mark.parametrize("source", ["hf", "modelscope"])
@pytest.mark.parametrize("served_name", [None, ["custom"]])
def test_speculator_switches_weights_but_keeps_draft_source(
    monkeypatch, tmp_path, source, served_name
):
    from copy import deepcopy

    from tests.transformers_utils.test_speculators_override import (
        _SPECULATORS_CONFIG_DICT,
    )

    draft, target = tmp_path / "draft", tmp_path / "target"
    draft.mkdir()
    config = deepcopy(_SPECULATORS_CONFIG_DICT)
    config["speculators_config"]["verifier"]["name_or_path"] = "owner/target"
    (draft / "config.json").write_text(json.dumps(config))
    Qwen3Config().save_pretrained(target)

    def download(repo, **kwargs):
        assert repo in ("owner/draft", "owner/target")
        assert kwargs["revision"] == ("draft-rev" if repo.endswith("draft") else None)
        return str(draft if repo.endswith("draft") else target)

    monkeypatch.setattr(HUB_DOWNLOADS[source], download)
    args = EngineArgs(
        model=f"{source}://owner/draft",
        revision="draft-rev",
        served_model_name=served_name,
    )
    model = args.create_model_config()
    assert model.model == str(target)
    assert model.model_weights == f"{source}://owner/target"
    assert model.served_model_name == (served_name or ["owner/target"])[0]
    assert args.speculative_config is not None
    assert args.speculative_config["model"] == f"{source}://owner/draft"
    assert args.speculative_config["revision"] == "draft-rev"


def test_main_and_draft_downloads_use_resolved_commits(monkeypatch, hubs):
    HfApi = type(hf_api())
    monkeypatch.setenv("VLLM_USE_MODELSCOPE", "True")

    def resolve(self, repo, **kwargs):
        assert repo in ("owner/target", "owner/draft")
        return ("a" if repo.endswith("target") else "b") * 40

    monkeypatch.setattr(HfApi, "resolve_revision", resolve)
    args = EngineArgs(
        model="hf://owner/target",
        speculative_config={"model": "hf://owner/draft", "num_speculative_tokens": 2},
    )
    config = args.create_speculative_config(
        args.create_model_config(), ParallelConfig()
    )
    assert config is not None
    assert args.revision == "a" * 40
    assert config.draft_model_config.revision == "b" * 40
    assert {(c.args[0], c.kwargs["revision"]) for c in hubs["hf"].call_args_list} == {
        ("owner/target", "a" * 40),
        ("owner/draft", "b" * 40),
    }


@pytest.mark.parametrize("field", ["model", "tokenizer"])
def test_rust_frontend_receives_local_hub_paths(monkeypatch, hubs, field):
    import socket
    import subprocess

    import zmq

    from vllm.entrypoints.common.cli_args import make_arg_parser
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    from vllm.utils.network_utils import make_zmq_listener
    from vllm.v1.utils import RustFrontendProcessManager

    model = "modelscope://owner/model" if field == "model" else hubs["hf"].return_value
    argv = ["--model", model]
    if field == "tokenizer":
        argv += ["--tokenizer", "modelscope://owner/tokenizer"]
    args = make_arg_parser(FlexibleArgumentParser()).parse_args(argv)
    config = EngineArgs.from_cli_args(args).create_model_config()
    launched = Mock()
    monkeypatch.setattr(subprocess, "Popen", launched)
    with socket.socket() as sock:
        inputs = make_zmq_listener("tcp://127.0.0.1:0", zmq.ROUTER)
        outputs = make_zmq_listener("tcp://127.0.0.1:0", zmq.PULL)
        try:
            RustFrontendProcessManager(
                binary_path="vllm-rs",
                sock=sock,
                args=args,
                input_listener=inputs,
                output_listener=outputs,
                engine_start_index=0,
                engine_count=1,
                data_parallel_size=1,
                model_config=config,
            )
            command = launched.call_args.args[0]
            bootstrap = json.loads(command[command.index("--args-json") + 1])
            assert bootstrap["tokenizer"] == hubs["modelscope"].return_value
            if field == "model":
                assert bootstrap["model_tag"] == config.model
                assert bootstrap["served_model_name"] == ["owner/model"]
        finally:
            inputs.socket.close()
            outputs.socket.close()


def test_specialized_loader_requires_local_snapshot(hubs):
    from vllm.config import DeviceConfig, VllmConfig
    from vllm.model_executor.model_loader import get_model

    args = EngineArgs(model="hf://owner/model", load_format="sharded_state")
    config = VllmConfig(
        model_config=args.create_model_config(),
        load_config=args.create_load_config(),
        device_config=DeviceConfig(device="cpu"),
    )
    with pytest.raises(ValueError, match="download to a local directory"):
        get_model(vllm_config=config)

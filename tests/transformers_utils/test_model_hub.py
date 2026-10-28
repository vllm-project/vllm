# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import Mock

import pytest

from vllm.model_hub import parse_model_source, resolve_model_source, strip_model_source


@pytest.fixture
def hubs(monkeypatch, tmp_path):
    from transformers import Qwen3Config

    downloads = {}
    for source, sdk in (
        ("hf", "vllm.model_hub.hf_snapshot_download"),
        ("modelscope", "modelscope.snapshot_download"),
    ):
        path = tmp_path / source
        Qwen3Config().save_pretrained(path)
        downloads[source] = Mock(return_value=str(path))
        monkeypatch.setattr(sdk, downloads[source])
    return downloads


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


@pytest.mark.parametrize("source", ["hf", "modelscope"])
def test_served_name_omits_only_the_source(source):
    assert strip_model_source(f"{source}://owner/model") == "owner/model"


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
            metadata_only=True,
        )
        == selected.return_value
    )
    assert selected.call_args.args == ("owner/model",)
    kwargs = selected.call_args.kwargs
    assert kwargs["revision"] == "revision"
    assert kwargs["cache_dir"] == str(tmp_path)
    assert kwargs["local_files_only"] is True
    assert "*.safetensors" in kwargs["ignore_patterns"]
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
    monkeypatch, tmp_path, source, patched, code_revision
):
    from modelscope.utils.hf_util import patch_context
    from transformers import AutoConfig, AutoTokenizer

    snapshot = tmp_path / "models--owner--model"
    code = tmp_path / "models--owner--code"
    snapshot.mkdir()
    code.mkdir()
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

    def download(repo, **kwargs):
        assert kwargs.get("revision") == (
            "weights-rev" if repo == "owner/model" else code_revision
        )
        return str(snapshot if repo == "owner/model" else code)

    with patch_context() if patched else nullcontext():
        hf = Mock(
            side_effect=download if source == "hf" else AssertionError("Wrong Hub")
        )
        ms = Mock(
            side_effect=download
            if source == "modelscope"
            else AssertionError("Wrong Hub")
        )
        monkeypatch.setattr("vllm.model_hub.hf_snapshot_download", hf)
        monkeypatch.setattr("modelscope.snapshot_download", ms)
        path = resolve_model_source(
            f"{source}://owner/model",
            revision="weights-rev",
            code_revision=code_revision,
            download_dir=str(tmp_path),
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
        selected, other = (hf, ms) if source == "hf" else (ms, hf)
        assert [call.args[0] for call in selected.call_args_list] == [
            "owner/model",
            "owner/code",
        ]
        failure = OSError("code download failed")
        selected.side_effect = [str(snapshot), failure]
        with pytest.raises(OSError) as caught:
            resolve_model_source(f"{source}://owner/model")
        assert caught.value is failure
        assert f"{source}://owner/code" in failure.__notes__[0]
        other.assert_not_called()


def test_same_repo_code_revision_and_cached_overlay(monkeypatch, tmp_path):
    from transformers import AutoConfig

    snapshot, code = tmp_path / "snapshot", tmp_path / "code"
    snapshot.mkdir()
    code.mkdir()
    (snapshot / "config.json").write_text(
        json.dumps({"auto_map": {"AutoConfig": "configuration.CustomConfig"}})
    )
    (code / "configuration.py").write_text(
        "from transformers import PretrainedConfig\n"
        "class CustomConfig(PretrainedConfig):\n"
        "    marker = 43\n"
    )

    def download(repo, **kwargs):
        assert repo == "owner/model"
        return str(code if kwargs.get("revision") == "code-rev" else snapshot)

    monkeypatch.setattr("vllm.model_hub.hf_snapshot_download", download)
    options = dict(code_revision="code-rev", download_dir=str(tmp_path))
    path = resolve_model_source("hf://owner/model", **options)
    assert AutoConfig.from_pretrained(path, trust_remote_code=True).marker == 43
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
    from vllm.config import ParallelConfig
    from vllm.engine.arg_utils import EngineArgs

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
    assert config.draft_model_config.model == draft.return_value
    assert target.call_args.args == ("owner/target",)
    assert draft.call_args.args == ("owner/draft",)
    assert target.call_args.kwargs["revision"] == "target-rev"
    assert draft.call_args.kwargs["revision"] == "draft-rev"

    args = EngineArgs(
        model="owner/target",
        tokenizer=f"{draft_source}://owner/tokenizer",
        tokenizer_revision="tokenizer-rev",
    )
    assert args.model == "owner/target"
    assert args.tokenizer == draft.return_value
    assert draft.call_args.args == ("owner/tokenizer",)
    assert draft.call_args.kwargs["revision"] == "tokenizer-rev"

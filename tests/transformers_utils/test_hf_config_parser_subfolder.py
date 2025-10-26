# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional, Union

from transformers import GenerationConfig, PretrainedConfig

from vllm.transformers_utils import config as config_module
from vllm.transformers_utils.config import HFConfigParser, try_get_generation_config


def test_hf_config_parser_uses_llm_subfolder(monkeypatch):
    parser = HFConfigParser()
    base_config = PretrainedConfig()
    subfolder_config = PretrainedConfig()

    def fake_get_config_dict(
        cls,
        model: Union[str, bytes],
        revision: Optional[str] = None,
        code_revision: Optional[str] = None,
        **kwargs,
    ):
        return {"llm_cfg": {}}, base_config

    def fake_file_exists(
        model: Union[str, bytes], config_name: str, revision: Optional[str]
    ):
        return config_name == "llm/config.json"

    auto_called = {}

    def fake_auto_from_pretrained(cls, *args, **kwargs):
        auto_called["subfolder"] = kwargs.get("subfolder")
        return subfolder_config

    monkeypatch.setattr(
        PretrainedConfig,
        "get_config_dict",
        classmethod(fake_get_config_dict),
    )
    monkeypatch.setattr(config_module, "file_or_path_exists", fake_file_exists)
    monkeypatch.setattr(
        config_module.AutoConfig,
        "from_pretrained",
        classmethod(fake_auto_from_pretrained),
    )

    returned_dict, returned_config = parser.parse("fake/model", trust_remote_code=False)

    assert returned_dict == {"llm_cfg": {}}
    assert returned_config is subfolder_config
    assert auto_called["subfolder"] == "llm"


def test_try_get_generation_config_llm_subfolder(monkeypatch):
    calls = []

    def fake_from_pretrained(cls, model: str, **kwargs):
        calls.append(kwargs.get("subfolder"))
        if len(calls) == 1:
            raise OSError("missing")
        return GenerationConfig()

    monkeypatch.setattr(
        config_module.GenerationConfig,
        "from_pretrained",
        classmethod(fake_from_pretrained),
    )

    result = try_get_generation_config("fake/model", trust_remote_code=False)

    assert isinstance(result, GenerationConfig)
    assert calls == [None, "llm"]

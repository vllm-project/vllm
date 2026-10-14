# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from pathlib import Path

import pytest
from transformers import PreTrainedConfig

from vllm.transformers_utils.config import (
    get_config,
    get_config_parser,
    register_config_parser,
)
from vllm.transformers_utils.config_parser_base import ConfigParserBase


@register_config_parser("custom_config_parser")
class CustomConfigParser(ConfigParserBase):
    def parse(
        self,
        model: str | Path,
        trust_remote_code: bool,
        revision: str | None = None,
        code_revision: str | None = None,
        **kwargs,
    ) -> tuple[dict, PreTrainedConfig]:
        raise NotImplementedError


def test_register_config_parser():
    assert isinstance(get_config_parser("custom_config_parser"), CustomConfigParser)


def test_invalid_config_parser():
    with pytest.raises(ValueError):

        @register_config_parser("invalid_config_parser")
        class InvalidConfigParser:
            pass


def test_parse_uses_checkpoint_context_length(tmp_path):
    """Remote Parse configs must not acquire BART's 1024-token default."""
    (tmp_path / "configuration_parse.py").write_text(
        "from transformers import PreTrainedConfig\n"
        "class ParseConfig(PreTrainedConfig):\n"
        "    model_type = 'nemotron_parse'\n"
        "    def __init__(self, decoder=None, **kwargs):\n"
        "        self.decoder = PreTrainedConfig(**(decoder or {}))\n"
        "        super().__init__(**kwargs)\n"
        "    def get_text_config(self, **kwargs):\n"
        "        return self.decoder\n"
    )
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "nemotron_parse",
                "auto_map": {"AutoConfig": "configuration_parse.ParseConfig"},
                "decoder": {"max_sequence_length": 16384},
            }
        )
    )

    config = get_config(str(tmp_path), trust_remote_code=True)

    assert type(config).__name__ == "ParseConfig"
    assert config.get_text_config().max_sequence_length == 16384
    assert not hasattr(config.get_text_config(), "max_position_embeddings")

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass, field
from typing import ClassVar, Dict, List, Optional

from transformers import PretrainedConfig


@dataclass
class HSTUEmbeddingConfig:
    table_name: str = ""
    feature_names: List[str] = field(default_factory=list)
    vocab_size: int = 0
    dim: int = 0
    associated_feature_table: Optional[str] = None
    associated_feature_name: Optional[str] = None


@dataclass
class HSTUModelConfig:
    hidden_size: int = 4096
    head_dim: int = 256
    num_heads: int = 16
    num_layers: int = 12
    is_causal: bool = True
    target_group_size: int = 1
    layernorm_epsilon: float = 1e-7
    residual: bool = True
    has_ffn: bool = True
    ffn_expand: int = 4
    dropout_ratio: float = 0.0


@dataclass
class RankingConfig:
    embedding_configs: List[HSTUEmbeddingConfig] = field(
        default_factory=list)
    prediction_head_arch: List[List[int]] = field(
        default_factory=lambda: [[256, 1]])
    prediction_head_act_type: str = "relu"
    embedding_stack_seq_cnt: List[int] = field(default_factory=list)
    embedding_feature_cnt: List[int] = field(default_factory=list)


class HSTUConfig(PretrainedConfig):
    model_type = "hstu"
    is_generative_recommend_model = True

    def __init__(
        self,
        # Accept both naming conventions: the config.json key name
        # (hstu_config / task_config) and the internal storage name
        # (*_dict).  PretrainedConfig.from_dict passes the exact keys
        # found in config.json, which are "hstu_config" / "task_config".
        hstu_config: Optional[Dict] = None,
        task_config: Optional[Dict] = None,
        hstu_config_dict: Optional[Dict] = None,
        task_config_dict: Optional[Dict] = None,
        merged_table: bool = True,
        features_cnt: int = 20,
        num_ratings: int = 5,
        max_seq_len: int = 8192,
        max_batch_size: int = 32,
        max_num_candidates: int = 4096,
        model_module_mapping: Optional[Dict[str, str]] = None,
        use_random_model: bool = False,
        input_seq: Optional[List[str]] = None,
        multi_value_prefix: Optional[str] = None,
        graph_model_compile_config: Optional[Dict] = None,
        # PretrainedConfig may pass extra keys from config.json
        # (e.g. "vocab_size", "torch_dtype", "num_hidden_layers").
        # We capture them in **kwargs and forward to super.
        **kwargs,
    ):
        # Prefer the canonical key name; fall back to internal name.
        self.hstu_config_dict = hstu_config or hstu_config_dict or {}
        self.task_config_dict = task_config or task_config_dict or {}
        self.merged_table = merged_table
        self.features_cnt = features_cnt
        self.num_ratings = num_ratings
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size
        self.max_num_candidates = max_num_candidates
        self.model_module_mapping = model_module_mapping
        self.use_random_model = use_random_model
        self.input_seq = input_seq
        self.multi_value_prefix = multi_value_prefix
        self.graph_model_compile_config = graph_model_compile_config or {}

        super().__init__(**kwargs)

    # Mapping from config.json field names to HSTUModelConfig fields.
    # Keys that exist in both names (e.g. hidden_size, head_dim) don't need
    # an entry here — only the *renamed* ones do.
    # ClassVar so Pydantic does not treat these as mutable defaults.
    _HSTU_CONFIG_KEY_MAP: ClassVar[Dict[str, str]] = {
        "num_attention_heads": "num_heads",
        "norm_epsilon": "layernorm_epsilon",
    }
    # Keys present in config.json but not consumed by HSTUModelConfig.
    _HSTU_CONFIG_IGNORED_KEYS: ClassVar[set] = {
        "dtype",
        "position_encoding_config",
        "rab",
    }

    @property
    def hstu_config(self) -> HSTUModelConfig:
        """Return the typed HSTUModelConfig dataclass."""
        if not isinstance(self.hstu_config_dict, HSTUModelConfig):
            raw = dict(self.hstu_config_dict)
            # Rename keys according to the mapping
            for json_key, dataclass_key in self._HSTU_CONFIG_KEY_MAP.items():
                if json_key in raw:
                    raw[dataclass_key] = raw.pop(json_key)
            # Drop keys that the dataclass doesn't accept
            for ignored in self._HSTU_CONFIG_IGNORED_KEYS:
                raw.pop(ignored, None)
            self.hstu_config_dict = HSTUModelConfig(**raw)
        return self.hstu_config_dict

    @hstu_config.setter
    def hstu_config(self, value):
        if isinstance(value, HSTUModelConfig):
            self.hstu_config_dict = value
        elif isinstance(value, dict):
            self.hstu_config_dict = value
        else:
            self.hstu_config_dict = value

    @property
    def task_config(self) -> RankingConfig:
        """Return the typed RankingConfig dataclass."""
        if not isinstance(self.task_config_dict, RankingConfig):
            raw = dict(self.task_config_dict)
            # Convert embedding_configs list-of-dict → list of dataclass
            emb_cfgs = raw.pop("embedding_configs", [])
            typed_embs = []
            for ec in emb_cfgs:
                if isinstance(ec, HSTUEmbeddingConfig):
                    typed_embs.append(ec)
                elif isinstance(ec, dict):
                    # Filter out keys not in HSTUEmbeddingConfig
                    known_keys = {
                        "table_name",
                        "feature_names",
                        "vocab_size",
                        "dim",
                        "associated_feature_table",
                        "associated_feature_name",
                    }
                    filtered = {k: v for k, v in ec.items() if k in known_keys}
                    typed_embs.append(HSTUEmbeddingConfig(**filtered))
                else:
                    typed_embs.append(ec)
            raw["embedding_configs"] = typed_embs
            # Filter out keys not in RankingConfig
            known_keys = {
                "embedding_configs",
                "prediction_head_arch",
                "prediction_head_act_type",
                "embedding_stack_seq_cnt",
                "embedding_feature_cnt",
            }
            raw = {k: v for k, v in raw.items() if k in known_keys}
            self.task_config_dict = RankingConfig(**raw)
        return self.task_config_dict

    def to_dict(self):
        """Override to convert typed dataclasses back to plain dicts."""
        result = super().to_dict()
        # Ensure hstu_config / task_config are plain dicts in the output
        if isinstance(self.hstu_config_dict, HSTUModelConfig):
            result["hstu_config"] = {
                f.name: getattr(self.hstu_config_dict, f.name)
                for f in self.hstu_config_dict.__dataclass_fields__.values()
            }
        elif self.hstu_config_dict:
            result["hstu_config"] = self.hstu_config_dict

        if isinstance(self.task_config_dict, RankingConfig):
            tc = self.task_config_dict
            tc_dict = {
                f.name: getattr(tc, f.name) for f in tc.__dataclass_fields__.values()
            }
            # Convert HSTUEmbeddingConfig objects to dicts
            tc_dict["embedding_configs"] = [
                (
                    {
                        f.name: getattr(ec, f.name)
                        for f in ec.__dataclass_fields__.values()
                    }
                    if isinstance(ec, HSTUEmbeddingConfig)
                    else ec
                )
                for ec in tc.embedding_configs
            ]
            result["task_config"] = tc_dict
        elif self.task_config_dict:
            result["task_config"] = self.task_config_dict

        # Remove internal storage keys from output
        result.pop("hstu_config_dict", None)
        result.pop("task_config_dict", None)
        return result

    @task_config.setter
    def task_config(self, value):
        if isinstance(value, RankingConfig):
            self.task_config_dict = value
        elif isinstance(value, dict):
            self.task_config_dict = value
        else:
            self.task_config_dict = value

    # -- Convenience accessors -------------------------------------------

    @property
    def hidden_size(self) -> int:
        """Convenience accessor: hidden_size delegates to hstu_config."""
        return self.hstu_config.hidden_size

    @hidden_size.setter
    def hidden_size(self, value: int):
        """Allow PretrainedConfig to set hidden_size."""
        self.hstu_config.hidden_size = value

# SPDX-License-Identifier: Apache-2.0

from enum import IntEnum
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig
from typing_extensions import assert_never

from vllm.config import PoolerConfig, ModelConfig, ExtraModuleConfig
from vllm.model_executor.pooling_metadata import (PoolingMetadata,
                                                  PoolingTensors)
from vllm.sequence import PoolerOutput, PoolingSequenceGroupOutput
from vllm.transformers_utils.config import (
    get_cross_encoder_activation_function)
from vllm.logger import init_logger
import importlib

class PoolingType(IntEnum):
    """Enumeration for different types of pooling methods."""
    LAST = 0
    ALL = 1
    CLS = 2
    STEP = 3
    MEAN = 4


class SimplePooler(nn.Module):
    """A layer that pools specific information from hidden states.

    This layer does the following:
    1. Extracts specific tokens or aggregates data based on pooling method.
    2. Normalizes output if specified.
    3. Returns structured results as `PoolerOutput`.

    Attributes:
        pooling_type: The type of pooling to use.
        normalize: Whether to normalize the pooled data.
    """

    @staticmethod
    def from_pooling_type(
        pooling_type: PoolingType,
        *,
        hidden_size: int,
        normalize: bool,
        softmax: bool,
        extra_modules_config: Optional[List[ExtraModuleConfig]] = None,
        step_tag_id: Optional[int] = None,
        returned_token_ids: Optional[List[int]] = None,
    ) -> "SimplePooler":
        """Creates a pooler instance based on the configuration.

        If extra_modules_config is provided, an EmbeddingPooler is created
        which wraps the base pooling logic and applies the extra modules.
        Otherwise, a simple base pooler (e.g., LastPool) is created.
        """
        # If extra modules are specified, create the wrapper EmbeddingPooler
        if extra_modules_config:
            # TODO: Support step_tag_id and returned_token_ids in EmbeddingPooler
            if pooling_type == PoolingType.STEP:
                logger.warning(
                    "STEP pooling with extra_modules is not fully supported yet "
                    "regarding step_tag_id and returned_token_ids.")

            return EmbeddingPooler(
                pooling_type=pooling_type,
                hidden_size=hidden_size,
                normalize=normalize,
                softmax=softmax,
                extra_modules_config=extra_modules_config,
                # step_tag_id=step_tag_id,
                # returned_token_ids=returned_token_ids,
            )

        # Otherwise, create the specific base pooler
        if pooling_type == PoolingType.LAST:
            assert step_tag_id is None and returned_token_ids is None
            return LastPool(normalize=normalize, softmax=softmax)
        if pooling_type == PoolingType.ALL:
            assert step_tag_id is None and returned_token_ids is None
            return AllPool(normalize=normalize, softmax=softmax)
        if pooling_type == PoolingType.CLS:
            assert step_tag_id is None and returned_token_ids is None
            return CLSPool(normalize=normalize, softmax=softmax)
        if pooling_type == PoolingType.MEAN:
            assert step_tag_id is None and returned_token_ids is None
            return MeanPool(normalize=normalize, softmax=softmax)
        if pooling_type == PoolingType.STEP:
            return StepPool(normalize=normalize,
                            softmax=softmax,
                            step_tag_id=step_tag_id,
                            returned_token_ids=returned_token_ids)

        assert_never(pooling_type)

    def __init__(self, *, normalize: bool, softmax: bool) -> None:
        super().__init__()

        self.head = PoolerHead(normalize=normalize, softmax=softmax)

    def get_prompt_lens(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> torch.Tensor:
        return PoolingTensors.from_pooling_metadata(
            pooling_metadata, hidden_states.device).prompt_lens

    def extract_states(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> Union[list[torch.Tensor], torch.Tensor]:
        raise NotImplementedError

    def build_output(self, data: torch.Tensor) -> PoolingSequenceGroupOutput:
        return PoolingSequenceGroupOutput(data)

    def forward(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> PoolerOutput:
        pooled_data = self.extract_states(hidden_states, pooling_metadata)
        pooled_data = self.head(pooled_data, pooling_metadata)
        pooled_outputs = [self.build_output(data) for data in pooled_data]
        return PoolerOutput(outputs=pooled_outputs)


class CLSPool(SimplePooler):

    def extract_states(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> Union[list[torch.Tensor], torch.Tensor]:
        prompt_lens = self.get_prompt_lens(hidden_states, pooling_metadata)

        first_token_flat_indices = torch.zeros_like(prompt_lens)
        first_token_flat_indices[1:] += torch.cumsum(prompt_lens, dim=0)[:-1]
        return hidden_states[first_token_flat_indices]


class LastPool(SimplePooler):

    def extract_states(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> Union[list[torch.Tensor], torch.Tensor]:
        prompt_lens = self.get_prompt_lens(hidden_states, pooling_metadata)

        last_token_flat_indices = torch.cumsum(prompt_lens, dim=0) - 1
        return hidden_states[last_token_flat_indices]


class AllPool(SimplePooler):

    def extract_states(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> Union[list[torch.Tensor], torch.Tensor]:
        prompt_lens = self.get_prompt_lens(hidden_states, pooling_metadata)

        offset = 0
        pooled_data = list[torch.Tensor]()
        for prompt_len in prompt_lens:
            pooled_data.append(hidden_states[offset:offset + prompt_len])
            offset += prompt_len

        return pooled_data


class MeanPool(SimplePooler):

    def extract_states(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> Union[list[torch.Tensor], torch.Tensor]:
        prompt_lens = self.get_prompt_lens(hidden_states, pooling_metadata)

        cumsum = torch.cumsum(hidden_states, dim=0)
        start_indices = torch.cat([
            torch.tensor([0], device=hidden_states.device),
            torch.cumsum(prompt_lens[:-1], dim=0)
        ])
        end_indices = torch.cumsum(prompt_lens, dim=0)
        return (cumsum[end_indices - 1] - cumsum[start_indices] +
                hidden_states[start_indices]) / prompt_lens.unsqueeze(1)


class StepPool(SimplePooler):

    def __init__(
        self,
        *,
        normalize: bool,
        softmax: bool,
        step_tag_id: Optional[int] = None,
        returned_token_ids: Optional[List[int]] = None,
    ):
        super().__init__(normalize=normalize, softmax=softmax)

        self.step_tag_id = step_tag_id
        self.returned_token_ids = returned_token_ids

    def extract_states(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> Union[list[torch.Tensor], torch.Tensor]:
        prompt_lens = self.get_prompt_lens(hidden_states, pooling_metadata)

        returned_token_ids = self.returned_token_ids
        if returned_token_ids is not None and len(returned_token_ids) > 0:
            hidden_states = hidden_states[:, returned_token_ids]

        step_tag_id = self.step_tag_id

        offset = 0
        pooled_data = list[torch.Tensor]()
        for prompt_len, seq_data_i in zip(prompt_lens,
                                          pooling_metadata.seq_data.values()):
            pooled_data_i = hidden_states[offset:offset + prompt_len]
            if step_tag_id is not None:
                token_ids = torch.tensor(seq_data_i.prompt_token_ids)
                pooled_data_i = pooled_data_i[token_ids == step_tag_id]

            offset += prompt_len
            pooled_data.append(pooled_data_i)

        return pooled_data


class PoolerHead(nn.Module):

    def __init__(self, *, normalize: bool, softmax: bool) -> None:
        super().__init__()

        self.normalize = normalize
        self.softmax = softmax

    def forward(self, pooled_data: Union[list[torch.Tensor], torch.Tensor],
                pooling_metadata: PoolingMetadata):

        dimensions_list = [
            pooling_param.dimensions
            for _, pooling_param in pooling_metadata.seq_groups
        ]
        if any(d is not None for d in dimensions_list):
            # change the output dimension
            assert len(pooled_data) == len(dimensions_list)
            pooled_data = [
                vecs if d is None else vecs[..., :d]
                for vecs, d in zip(pooled_data, dimensions_list)
            ]

        if self.normalize:
            if isinstance(pooled_data, list):
                pooled_data = [
                    F.normalize(data, p=2, dim=-1) for data in pooled_data
                ]
            else:
                pooled_data = F.normalize(pooled_data, p=2, dim=-1)

        if self.softmax:
            if isinstance(pooled_data, list):
                pooled_data = [F.softmax(data, dim=-1) for data in pooled_data]
            else:
                pooled_data = F.softmax(pooled_data, dim=-1)

        return pooled_data


class Pooler(nn.Module):

    @classmethod
    def from_config_with_defaults(
        cls,
        pooler_config: PoolerConfig,
        *,
        hidden_size: int,
        pooling_type: PoolingType,
        normalize: bool,
        softmax: bool,
        step_tag_id: Optional[int] = None,
        returned_token_ids: Optional[List[int]] = None,
    ) -> SimplePooler:
        resolved_pooling_type = (PoolingType[pooler_config.pooling_type]
                                  if pooler_config.pooling_type is not None
                                  else pooling_type)
        resolved_normalize = (pooler_config.normalize
                               if pooler_config.normalize is not None
                               else normalize)
        resolved_softmax = (pooler_config.softmax
                             if pooler_config.softmax is not None
                             else softmax)
        resolved_step_tag_id = (pooler_config.step_tag_id
                                 if pooler_config.step_tag_id is not None
                                 else step_tag_id)
        resolved_returned_token_ids = (
            pooler_config.returned_token_ids
            if pooler_config.returned_token_ids is not None
            else returned_token_ids)

        # Pass hidden_size and extra_modules down
        return SimplePooler.from_pooling_type(
            pooling_type=resolved_pooling_type,
            hidden_size=hidden_size,
            normalize=resolved_normalize,
            softmax=resolved_softmax,
            extra_modules_config=pooler_config.extra_modules,
            step_tag_id=resolved_step_tag_id,
            returned_token_ids=resolved_returned_token_ids,
        )


class CrossEncodingPooler(nn.Module):
    """A layer that pools specific information from hidden states.

    This layer does the following:
    1. Extracts specific tokens or aggregates data based on pooling method.
    2. Normalizes output if specified.
    3. Returns structured results as `PoolerOutput`.

    Attributes:
        pooling_type: The type of pooling to use.
        normalize: Whether to normalize the pooled data.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        classifier: nn.Module,
        pooler: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.classifier = classifier
        self.pooler = pooler
        self.default_activation_function = \
            get_cross_encoder_activation_function(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> PoolerOutput:
        """Pools sentence pair scores from the hidden_states."""

        prompt_lens = PoolingTensors.from_pooling_metadata(
            pooling_metadata, hidden_states.device).prompt_lens

        offset = 0
        pooled_data_lst = []
        for prompt_len in prompt_lens:
            pooled_data_i = hidden_states[offset:offset + prompt_len]

            if self.pooler is not None:
                final_shape_tensor = self.pooler(pooled_data_i)
            else:
                final_shape_tensor = self.classifier(pooled_data_i)

            pooled_data_lst.append(final_shape_tensor)
            offset += prompt_len

        pooled_output = torch.stack(pooled_data_lst)

        if self.pooler is not None:
            # apply classifier once on the full batch if possible
            pooled_output = self.classifier(pooled_output)

        scores = self.default_activation_function(pooled_output).squeeze(-1)

        pooled_outputs = [PoolingSequenceGroupOutput(data) for data in scores]
        return PoolerOutput(outputs=pooled_outputs)


class EmbeddingPooler(nn.Module):
    """A Pooler that applies a base pooling method and then passes the
    result through a sequence of additional modules (e.g., Dense layers).
    """

    def __init__(
        self,
        pooling_type: PoolingType,
        hidden_size: int,
        normalize: bool,
        softmax: bool,
        extra_modules_config: List[ExtraModuleConfig],
        # TODO: Add support for step_tag_id and returned_token_ids if needed
        # step_tag_id: Optional[int] = None,
        # returned_token_ids: Optional[List[int]] = None,
    ):
        super().__init__()
        # Store configuration for final normalization/softmax
        self.normalize = normalize
        self.softmax = softmax

        # Instantiate the base pooler (norm/softmax applied at the end)
        self.base_pooler = SimplePooler.from_pooling_type(
            pooling_type=pooling_type,
            hidden_size=hidden_size,
            normalize=False,
            softmax=False,
            # Pass other relevant args if/when supported
            # step_tag_id=step_tag_id,
            # returned_token_ids=returned_token_ids,
        )

        # Load extra modules, passing the initial hidden size
        self.extra_modules_list = self._load_extra_modules(extra_modules_config,
                                                       hidden_size)

    def _load_extra_modules(self, extra_modules_config: List[ExtraModuleConfig],
                          initial_in_features: int) -> nn.ModuleList:
        extra_modules_list = nn.ModuleList()
        current_in_features = initial_in_features

        for module_config in extra_modules_config:
            try:
                module_name = module_config.type
                st_models = importlib.import_module("sentence_transformers.models")
                module_class = getattr(st_models, module_name)

                # Prepare params, injecting in_features for Dense layer
                params = module_config.params.copy()
                if module_name == "Dense":
                    if "in_features" in params:
                        logger.warning(
                            f"'in_features' ({params['in_features']}) provided in config "
                            f"for Dense layer will be overridden by calculated value "
                            f"({current_in_features}).")
                    if "out_features" not in params:
                        raise ValueError("'out_features' must be specified in params "
                                         "for Dense layer config.")

                    params["in_features"] = current_in_features
                    logger.info(
                        f"Loading extra module: {module_name} with calculated "
                        f"in_features={current_in_features}, params: {params}")
                    module_instance = module_class(**params)
                    # Update in_features for the next layer
                    current_in_features = params["out_features"]
                else:
                    # For other module types, pass params as is
                    logger.info(
                        f"Loading extra module: {module_name} with params: {params}"
                    )
                    module_instance = module_class(**params)
                    # We cannot know the output shape of arbitrary modules easily.
                    # If a non-Dense module changes the dimension, subsequent Dense
                    # layers might fail. Consider adding an 'output_features' hint
                    # to ExtraModuleConfig or restricting sequences.
                    logger.warning(
                        f"Output dimension of {module_name} is unknown. "
                        f"If subsequent layers are Dense, they might fail if the "
                        f"dimension changed.")

                extra_modules_list.append(module_instance)
            except (ImportError, AttributeError) as e:
                logger.error(
                    f"Could not find or import extra module class "
                    f"'sentence_transformers.models.{module_config.type}'. Error: {e}"
                )
                raise ImportError(
                    f"Failed to load extra module: {module_config.type}") from e
            except TypeError as e:
                logger.error(
                    f"Error initializing extra module '{module_config.type}' "
                    f"with params {params}. Error: {e}")
                raise TypeError(
                    f"Failed to initialize extra module: {module_config.type}") from e
            except ValueError as e:
                logger.error(
                    f"Configuration error for extra module '{module_config.type}'. "
                    f"Error: {e}")
                raise e # Reraise config errors
            except Exception as e:
                logger.error(
                    f"An unexpected error occurred while loading extra module "
                    f"'{module_config.type}': {e}")
                raise e  # Reraise unexpected errors

        return extra_modules_list

    def forward(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> torch.Tensor:
        # 1. Apply base pooling
        # Note: The base pooler was initialized with normalize=False, softmax=False
        pooled_output = self.base_pooler(hidden_states, pooling_metadata)

        # 2. Apply extra modules sequentially
        # sentence-transformers modules typically expect/return a dict.
        features = {'sentence_embedding': pooled_output}
        current_embedding = pooled_output
        for i, extra_module in enumerate(self.extra_modules_list):
            try:
                features = extra_module(features)
                if 'sentence_embedding' in features:
                    current_embedding = features['sentence_embedding']
                else:
                    logger.warning(f"Extra module {i} ({extra_module.__class__.__name__}) did not "
                                   f"return 'sentence_embedding' in its output dict. "
                                   f"Using the output from the previous step.")
                    # Keep using the last known good embedding
                    features['sentence_embedding'] = current_embedding

            except Exception as e:
                logger.error(f"Error during forward pass of extra module {i} "
                             f"({extra_module.__class__.__name__}): {e}")
                # Depending on desired behavior, could raise, or return intermediate
                # For now, return the last successful embedding
                logger.warning("Returning embedding from before the failing module.")
                break

        # Get the final embedding after all extra modules
        final_embedding = current_embedding

        # 3. Apply final normalization if configured
        if self.normalize:
            final_embedding = F.normalize(final_embedding, p=2, dim=-1)

        # 4. Apply final softmax if configured (less common for embeddings)
        if self.softmax:
            final_embedding = F.softmax(final_embedding, dim=-1)

        return final_embedding

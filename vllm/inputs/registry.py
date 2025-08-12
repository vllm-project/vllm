import functools
from collections import UserDict
from dataclasses import dataclass
from typing import (TYPE_CHECKING, Any, Callable, Dict, Mapping, NamedTuple,
                    Optional, Protocol, Type)

from torch import nn
from transformers import PretrainedConfig
from typing_extensions import TypeVar

from vllm.logger import init_logger
from vllm.utils import (get_allowed_kwarg_only_overrides, print_warning_once,
                        resolve_mm_processor_kwargs)

from .data import DecoderOnlyInputs

if TYPE_CHECKING:
    from vllm.config import ModelConfig
    from vllm.multimodal import (MultiModalDataDict, MultiModalPlaceholderDict,
                                 MultiModalRegistry)
    from vllm.sequence import SequenceData

logger = init_logger(__name__)

C = TypeVar("C", bound=PretrainedConfig, default=PretrainedConfig)


@dataclass(frozen=True)
class InputContext:
    """
    Contains information about the model which may be used to
    modify the inputs.
    """

    model_config: "ModelConfig"
    """The configuration of the model."""

    def get_hf_config(self, hf_config_type: Type[C] = PretrainedConfig) -> C:
        """
        Get the HuggingFace configuration
        (:class:`transformers.PretrainedConfig`) of the model,
        additionally checking its type.

        Raises:
            TypeError: If the model is not of the specified type.
        """

        hf_config = self.model_config.hf_config
        if not isinstance(hf_config, hf_config_type):
            raise TypeError("Invalid type of HuggingFace config. "
                            f"Expected type: {hf_config_type}, but "
                            f"found type: {type(hf_config)}")

        return hf_config

    def get_hf_image_processor_config(self) -> Dict[str, Any]:
        """
        Get the HuggingFace image processor configuration of the model.
        """

        return self.model_config.hf_image_processor_config


N = TypeVar("N", bound=Type[nn.Module])


class DummyData(NamedTuple):
    """Dummy data used for profiling."""

    seq_data: "SequenceData"
    multi_modal_data: Optional["MultiModalDataDict"] = None
    multi_modal_placeholders: Optional["MultiModalPlaceholderDict"] = None


class DummyDataFactory(Protocol):

    def __call__(
        self,
        ctx: InputContext,
        seq_len: int,
        mm_counts: Mapping[str, int],
        **mm_processor_kwargs: Any,
    ) -> DummyData:
        """
        Create dummy data to be inputted into the model.

        Note:
            :data:`InputProcessor` is not applied to the dummy data.

            The :code:`mm_processor_kwargs` are overrides provided at
            initialization time to values in the config whose values
            may affect the number of tokens per instance.
        """
        ...


class _MultiModalCounts(UserDict):
    """
    Wraps `mm_counts` for a more informative error message
    when attempting to access a plugin that does not exist.
    """

    def __getitem__(self, key: str) -> int:
        try:
            return super().__getitem__(key)
        except KeyError as exc:
            msg = (f"There is no multi-modal plugin with the key: {key}. "
                   f"Available keys: {set(self.keys())}")
            raise KeyError(msg) from exc


InputProcessor = Callable[[InputContext, DecoderOnlyInputs], DecoderOnlyInputs]
"""Preprocess the inputs to the model."""


class InputRegistry:
    """
    A registry to dispatch data processing
    according to the target model.
    """

    def __init__(self) -> None:
        self._dummy_factories_by_model_type: Dict[Type[nn.Module],
                                                  DummyDataFactory] = {}
        self._dummy_encoder_factories_by_model_type: Dict[
            Type[nn.Module], DummyDataFactory] = {}
        self._input_processors_by_model_type: Dict[Type[nn.Module],
                                                   InputProcessor] = {}

    def _default_dummy_data_factory(
        self,
        ctx: InputContext,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> DummyData:
        """
        The default dummy data factory represents the longest possible text
        that can be inputted to the model.

        Note:
            :data:`InputProcessor` is not applied to the dummy data.
        """
        # Avoid circular import
        from vllm.sequence import SequenceData

        return DummyData(SequenceData.from_prompt_token_counts((0, seq_len)))

    def register_dummy_data(self, factory: DummyDataFactory):
        """
        Register a dummy data factory to a model class.

        During memory profiling, the provided function is invoked to create
        dummy data to be inputted into the model. The resulting memory usage
        should be an upper bound of what the model would use at inference time.
        """

        def wrapper(model_cls: N) -> N:
            if model_cls in self._dummy_factories_by_model_type:
                logger.warning(
                    "Model class %s already has dummy data "
                    "registered to %s. It is overwritten by the new one.",
                    model_cls, self)

            self._dummy_factories_by_model_type[model_cls] = factory

            return model_cls

        return wrapper

    def _get_dummy_data_factory(self, model_cls: Type[nn.Module]):
        return self._dummy_factories_by_model_type \
            .get(model_cls, self._default_dummy_data_factory)

    def register_dummy_encoder_data(self, factory: DummyDataFactory):
        """
        Register a dummy encoder data factory to a model class

        This is similar to :meth:`~register_dummy_data`, but for encoder input.
        """

        def wrapper(model_cls: N) -> N:
            if model_cls in self._dummy_encoder_factories_by_model_type:
                logger.warning(
                    "Model class %s already has dummy encoder data "
                    "registered to %s. It is overwritten by the new one.",
                    model_cls, self)

            self._dummy_encoder_factories_by_model_type[model_cls] = factory

            return model_cls

        return wrapper

    def _get_dummy_encoder_data_factory(self, model_cls: Type[nn.Module]):
        return self._dummy_encoder_factories_by_model_type \
            .get(model_cls, self._default_dummy_data_factory)

    def dummy_data_for_profiling(
        self,
        model_config: "ModelConfig",
        seq_len: int,
        mm_registry: "MultiModalRegistry",
        is_encoder_data: bool = False,
    ) -> DummyData:
        """
        Create dummy data for profiling the memory usage of a model.

        The model is identified by ``model_config``.

        See also:
            :ref:`enabling_multimodal_inputs`

        Note:
            This should be called after
            :meth:`~MultiModalRegistry.init_mm_limits_per_prompt`.
        """
        # Avoid circular import
        from vllm.model_executor.model_loader import get_model_architecture

        model_cls, _ = get_model_architecture(model_config)
        if is_encoder_data:
            dummy_factory = self._get_dummy_encoder_data_factory(model_cls)
        else:
            dummy_factory = self._get_dummy_data_factory(model_cls)
        mm_counts = mm_registry.get_mm_limits_per_prompt(model_config)
        mm_processor_kwargs = get_allowed_kwarg_only_overrides(
            dummy_factory, overrides=model_config.mm_processor_kwargs)

        dummy_data = dummy_factory(InputContext(model_config), seq_len,
                                   _MultiModalCounts(mm_counts),
                                   **mm_processor_kwargs)

        # Having more tokens is over-conservative but otherwise fine
        num_tokens = dummy_data.seq_data.prompt_token_ids
        if len(num_tokens) < seq_len:
            if is_encoder_data:
                print_warning_once(
                    f"Expected at least {seq_len} dummy encoder tokens for "
                    f"profiling, but found {len(num_tokens)} tokens instead.")
            else:
                raise AssertionError(
                    f"Expected at least {seq_len} dummy tokens for profiling, "
                    f"but found {len(num_tokens)} tokens instead.")
        if dummy_data.multi_modal_data is not None:
            for k, v in dummy_data.multi_modal_data.items():
                num_items = len(v) if isinstance(v, list) else 1
                num_expected = mm_counts[k]
                assert num_items >= num_expected, (
                    f"Expected at least {num_expected} dummy '{k}' instances "
                    f"for profiling, but found {num_items} instances instead.")

        return dummy_data

    def _default_input_processor(
        self,
        ctx: InputContext,
        inputs: DecoderOnlyInputs,
    ) -> DecoderOnlyInputs:
        """The default input processor is a no-op."""
        return inputs

    def register_input_processor(self, processor: InputProcessor):
        """
        Register an input processor to a model class.

        The provided function is invoked on each input to the model. This
        happens before :meth:`~vllm.multimodal.MultiModalRegistry.map_input`.

        See also:
            :ref:`input_processing_pipeline`
        """

        def wrapper(model_cls: N) -> N:
            if model_cls in self._input_processors_by_model_type:
                logger.warning(
                    "Model class %s already has input processor "
                    "registered to %s. It is overwritten by the new one.",
                    model_cls, self)

            self._input_processors_by_model_type[model_cls] = processor

            return model_cls

        return wrapper

    def _get_model_input_processor(self, model_cls: Type[nn.Module]):
        return self._input_processors_by_model_type \
            .get(model_cls, self._default_input_processor)

    def process_input(self, model_config: "ModelConfig",
                      inputs: DecoderOnlyInputs) -> DecoderOnlyInputs:
        """
        Apply an input processor to an instance of model inputs.

        The model is identified by ``model_config``.

        See also:
            :ref:`input_processing_pipeline`
        """
        # Avoid circular import
        from vllm.model_executor.model_loader import get_model_architecture

        model_cls, _ = get_model_architecture(model_config)
        processor = self._get_model_input_processor(model_cls)

        # Handle multimodal processor kwargs with priority:
        #     Inference kwargs -> Init kwargs -> {}
        # If it's empty, it'll fall back to the default kwarg values
        mm_processor_kwargs = resolve_mm_processor_kwargs(
            model_config.mm_processor_kwargs,
            inputs.get("mm_processor_kwargs"),
            processor,
        )

        return processor(InputContext(model_config), inputs,
                         **mm_processor_kwargs)

    def create_input_processor(self, model_config: "ModelConfig"):
        """
        Create an input processor (see :meth:`_process_input`) for a
        specific model.
        """
        return functools.partial(self.process_input, model_config)

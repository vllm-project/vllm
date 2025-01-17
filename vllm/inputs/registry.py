import functools
from collections import UserDict
from dataclasses import dataclass
from typing import (TYPE_CHECKING, Any, Callable, Mapping, NamedTuple,
                    Optional, Protocol, Union)

from torch import nn
from transformers import BatchFeature, PretrainedConfig, ProcessorMixin
from typing_extensions import TypeVar, assert_never

from vllm.logger import init_logger
from vllm.transformers_utils.processor import cached_get_processor
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.utils import (ClassRegistry, get_allowed_kwarg_only_overrides,
                        resolve_mm_processor_kwargs)

from .data import ProcessorInputs, SingletonInputs
from .parse import is_encoder_decoder_inputs

if TYPE_CHECKING:
    from vllm.config import ModelConfig
    from vllm.multimodal import (MultiModalDataDict, MultiModalPlaceholderDict,
                                 MultiModalRegistry)
    from vllm.sequence import SequenceData

logger = init_logger(__name__)

C = TypeVar("C", bound=PretrainedConfig, default=PretrainedConfig)
P = TypeVar("P", bound=ProcessorMixin, default=ProcessorMixin)


@dataclass(frozen=True)
class InputContext:
    """
    Contains information about the model which may be used to
    modify the inputs.
    """

    model_config: "ModelConfig"
    """The configuration of the model."""

    def get_hf_config(
        self,
        typ: Union[type[C], tuple[type[C], ...]] = PretrainedConfig,
        /,
    ) -> C:
        """
        Get the HuggingFace configuration
        (:class:`transformers.PretrainedConfig`) of the model,
        additionally checking its type.

        Raises:
            TypeError: If the configuration is not of the specified type.
        """
        hf_config = self.model_config.hf_config
        if not isinstance(hf_config, typ):
            raise TypeError("Invalid type of HuggingFace config. "
                            f"Expected type: {typ}, but "
                            f"found type: {type(hf_config)}")

        return hf_config

    def get_hf_image_processor_config(self) -> dict[str, Any]:
        """
        Get the HuggingFace image processor configuration of the model.
        """
        return self.model_config.hf_image_processor_config

    def get_mm_config(self):
        """
        Get the multimodal config of the model.

        Raises:
            RuntimeError: If the model is not a multimodal model.
        """
        mm_config = self.model_config.multimodal_config
        if mm_config is None:
            raise RuntimeError("Not a multimodal model")

        return mm_config

    def get_hf_processor(
        self,
        typ: Union[type[P], tuple[type[P], ...]] = ProcessorMixin,
        /,
        **kwargs: object,
    ) -> P:
        """
        Get the HuggingFace processor
        (:class:`transformers.ProcessorMixin`) of the model,
        additionally checking its type.

        Raises:
            TypeError: If the processor is not of the specified type.
        """
        base_kwargs = self.model_config.mm_processor_kwargs
        if base_kwargs is None:
            base_kwargs = {}

        merged_kwargs = {**base_kwargs, **kwargs}

        if isinstance(typ, type):
            merged_kwargs["processor_cls"] = typ

        hf_processor = cached_get_processor(
            self.model_config.model,
            trust_remote_code=self.model_config.trust_remote_code,
            **merged_kwargs,
        )
        if not isinstance(hf_processor, typ):
            raise TypeError("Invalid type of HuggingFace processor. "
                            f"Expected type: {typ}, but "
                            f"found type: {type(hf_processor)}")

        return hf_processor


@dataclass(frozen=True)
class InputProcessingContext(InputContext):
    tokenizer: AnyTokenizer
    """The tokenizer used to tokenize the inputs."""

    def get_hf_processor(
        self,
        typ: Union[type[P], tuple[type[P], ...]] = ProcessorMixin,
        /,
        **kwargs: object,
    ) -> P:
        return super().get_hf_processor(
            typ,
            tokenizer=self.tokenizer,
            **kwargs,
        )

    def call_hf_processor(
        self,
        hf_processor: ProcessorMixin,
        data: Mapping[str, object],
        kwargs: Mapping[str, object] = {},
    ) -> BatchFeature:
        """
        Call :code:`hf_processor` on the prompt :code:`data`
        (text, image, audio...) with configurable options :code:`kwargs`.
        """
        assert callable(hf_processor)

        base_kwargs = self.model_config.mm_processor_kwargs
        if base_kwargs is None:
            base_kwargs = {}

        merged_kwargs = resolve_mm_processor_kwargs(
            base_kwargs,
            kwargs,
            hf_processor,
            requires_kw_only=False,
            allow_var_kwargs=True,
        )

        try:
            return hf_processor(**data, **merged_kwargs, return_tensors="pt")
        except Exception as exc:
            msg = (f"Failed to apply {type(hf_processor).__name__} "
                   f"on data={data} with kwargs={merged_kwargs}")

            raise RuntimeError(msg) from exc


N = TypeVar("N", bound=type[nn.Module])


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


class _MultiModalCounts(UserDict[str, int]):
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


InputProcessor = Callable[[InputContext, ProcessorInputs], ProcessorInputs]
"""Preprocess the inputs to the model."""


class InputRegistry:
    """
    A registry to dispatch data processing
    according to the target model.
    """

    def __init__(self) -> None:
        self._dummy_factories_by_model_type = \
            ClassRegistry[nn.Module, DummyDataFactory]()
        self._dummy_encoder_factories_by_model_type = \
            ClassRegistry[nn.Module, DummyDataFactory]()
        self._input_processors_by_model_type = \
            ClassRegistry[nn.Module, InputProcessor]()

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
            if self._dummy_factories_by_model_type.contains(model_cls,
                                                            strict=True):
                logger.warning(
                    "Model class %s already has dummy data "
                    "registered to %s. It is overwritten by the new one.",
                    model_cls, self)

            self._dummy_factories_by_model_type[model_cls] = factory

            return model_cls

        return wrapper

    def _get_dummy_data_factory(self, model_cls: type[nn.Module]):
        return self._dummy_factories_by_model_type \
            .get(model_cls, self._default_dummy_data_factory)

    def register_dummy_encoder_data(self, factory: DummyDataFactory):
        """
        Register a dummy encoder data factory to a model class

        This is similar to :meth:`~register_dummy_data`, but for encoder input.
        """

        def wrapper(model_cls: N) -> N:
            if self._dummy_encoder_factories_by_model_type.contains(
                    model_cls, strict=True):
                logger.warning(
                    "Model class %s already has dummy encoder data "
                    "registered to %s. It is overwritten by the new one.",
                    model_cls, self)

            self._dummy_encoder_factories_by_model_type[model_cls] = factory

            return model_cls

        return wrapper

    def _get_dummy_encoder_data_factory(self, model_cls: type[nn.Module]):
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

        Note:
            This should be called after
            :meth:`~MultiModalRegistry.init_mm_limits_per_prompt`.
        """
        # Avoid circular import
        from vllm.model_executor.model_loader import get_model_architecture
        from vllm.multimodal import MultiModalKwargs
        from vllm.multimodal.profiling import MultiModalProfiler
        from vllm.multimodal.utils import cached_get_tokenizer

        if mm_registry.has_processor(model_config):
            tokenizer = cached_get_tokenizer(
                model_config.tokenizer,
                trust_remote_code=model_config.trust_remote_code,
            )
            processor = mm_registry.create_processor(model_config, tokenizer)
            profiler = MultiModalProfiler(processor)
            dummy_data = profiler.get_dummy_data(seq_len)
        else:
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
                logger.warning_once(
                    f"Expected at least {seq_len} dummy encoder tokens for "
                    f"profiling, but found {len(num_tokens)} tokens instead.")
            else:
                raise AssertionError(
                    f"Expected at least {seq_len} dummy tokens for profiling, "
                    f"but found {len(num_tokens)} tokens instead.")

        if (dummy_data.multi_modal_data is not None and
                not isinstance(dummy_data.multi_modal_data, MultiModalKwargs)):
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
        inputs: ProcessorInputs,
    ) -> ProcessorInputs:
        """The default input processor is a no-op."""
        return inputs

    def register_input_processor(self, processor: InputProcessor):
        """
        Register an input processor to a model class.

        The provided function is invoked on each input to the model. This
        happens before
        :meth:`~vllm.multimodal.registry.MultiModalRegistry.map_input`.
        """

        def wrapper(model_cls: N) -> N:
            if self._input_processors_by_model_type.contains(model_cls,
                                                             strict=True):
                logger.warning(
                    "Model class %s already has input processor "
                    "registered to %s. It is overwritten by the new one.",
                    model_cls, self)

            self._input_processors_by_model_type[model_cls] = processor

            return model_cls

        return wrapper

    def _get_model_input_processor(self, model_cls: type[nn.Module]):
        return self._input_processors_by_model_type \
            .get(model_cls, self._default_input_processor)

    def _ensure_mm_kwargs(
        self,
        inputs: SingletonInputs,
        mm_processor_kwargs: dict[str, Any],
    ):
        if inputs["type"] == "token":
            # In case the input processor for that model fails to set it
            if "mm_processor_kwargs" not in inputs:
                inputs["mm_processor_kwargs"] = mm_processor_kwargs
        elif inputs["type"] == "multimodal":
            # Be more strict in V2
            assert "mm_kwargs" in inputs
        else:
            assert_never(inputs["type"])  # type: ignore[arg-type]

    def process_input(self, model_config: "ModelConfig",
                      inputs: ProcessorInputs) -> ProcessorInputs:
        """
        Apply an input processor to an instance of model inputs.

        The model is identified by ``model_config``.
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
            inputs.get("mm_processor_kwargs", {}),  # type: ignore
            processor,
        )

        processed_inputs = processor(
            InputContext(model_config),
            inputs,
            **mm_processor_kwargs,
        )

        if is_encoder_decoder_inputs(processed_inputs):
            self._ensure_mm_kwargs(processed_inputs["encoder"],
                                   mm_processor_kwargs)
            self._ensure_mm_kwargs(processed_inputs["decoder"],
                                   mm_processor_kwargs)
        else:
            self._ensure_mm_kwargs(processed_inputs, mm_processor_kwargs)

        return processed_inputs

    def create_input_processor(self, model_config: "ModelConfig"):
        """
        Create an input processor (see :meth:`_process_input`) for a
        specific model.
        """
        return functools.partial(self.process_input, model_config)

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import (TYPE_CHECKING, Any, Callable, Dict, List, NamedTuple,
                    Optional, Sequence, Tuple, Type, TypeVar, Union)

from torch import nn

from vllm.inputs import InputContext
from vllm.logger import init_logger
from vllm.utils import (get_allowed_kwarg_only_overrides,
                        resolve_mm_processor_kwargs)

if TYPE_CHECKING:
    from vllm.config import ModelConfig
    from vllm.sequence import SequenceGroupMetadata

from .inputs import (MultiModalData, MultiModalDataDict, MultiModalKwargs,
                     PlaceholderRange)

logger = init_logger(__name__)

MultiModalInputMapper = Callable[[InputContext, MultiModalData[object]],
                                 MultiModalKwargs]
"""
Return a dictionary to be passed as keyword arguments to
:meth:`~torch.nn.Module.forward`. This is similar in concept to tokenizers
and processors in HuggingFace Transformers.

If the data is not supported, throw :exc:`TypeError`.
"""

MultiModalTokensCalc = Union[int, Callable[[InputContext], int]]
"""
Calculate the maximum number of multimodal tokens input to the language
model. This does not include tokens that correspond to the input text.
"""

_T = TypeVar("_T")
N = TypeVar("N", bound=Type[nn.Module])


class MultiModalPlugin(ABC):
    """
    Base class that defines data processing logic for a specific modality.

    In particular, we adopt a registry pattern to dispatch data processing
    according to the model being used (considering that different models may
    process the same data differently). This registry is in turn used by
    :class:`~MultiModalRegistry` which acts at a higher level
    (i.e., the modality of the data).

    See also:
        :ref:`adding_multimodal_plugin`
    """

    def __init__(self) -> None:
        self._input_mappers: Dict[Type[nn.Module], MultiModalInputMapper] = {}
        self._max_mm_tokens: Dict[Type[nn.Module], MultiModalTokensCalc] = {}

    @abstractmethod
    def get_data_key(self) -> str:
        """
        Get the data key corresponding to the modality.
        """
        raise NotImplementedError

    @abstractmethod
    def _default_input_mapper(
        self,
        ctx: InputContext,
        data: MultiModalData[Any],
        **mm_processor_kwargs,
    ) -> MultiModalKwargs:
        """
        Return a dictionary to be passed as keyword arguments to
        :meth:`~torch.nn.Module.forward`. This is similar in concept to
        tokenizers and processors in HuggingFace Transformers.

        If the data is not supported, throw :exc:`TypeError`.
        """
        raise NotImplementedError

    def register_input_mapper(
        self,
        mapper: Optional[MultiModalInputMapper] = None,
    ):
        """
        Register an input mapper to a model class.

        When the model receives input data that matches the modality served by
        this plugin (see :meth:`get_data_key`), the provided function is
        invoked to transform the data into a dictionary of model inputs.

        If `None` is provided, then the default input mapper is used instead.

        See also:
            - :ref:`input_processing_pipeline`
            - :ref:`enabling_multimodal_inputs`
        """

        def wrapper(model_cls: N) -> N:
            if model_cls in self._input_mappers:
                logger.warning(
                    "Model class %s already has an input mapper "
                    "registered to %s. It is overwritten by the new one.",
                    model_cls,
                    self,
                )

            self._input_mappers[model_cls] = (mapper
                                              or self._default_input_mapper)

            return model_cls

        return wrapper

    def map_input(
        self,
        model_config: "ModelConfig",
        data: MultiModalData[Any],
        mm_processor_kwargs: Optional[Dict[str, Any]],
    ) -> MultiModalKwargs:
        """
        Transform the data into a dictionary of model inputs using the
        input mapper registered for that model.

        The model is identified by ``model_config``.

        Raises:
            TypeError: If the data type is not supported.

        See also:
            - :ref:`input_processing_pipeline`
            - :ref:`enabling_multimodal_inputs`
        """

        # Avoid circular import
        from vllm.model_executor.model_loader import get_model_architecture

        model_cls, _ = get_model_architecture(model_config)

        mapper = self._input_mappers.get(model_cls)

        if mapper is None:
            raise KeyError(f"No input mapper in {self} is registered for "
                           f"model class {model_cls.__name__}.")

        if mm_processor_kwargs is None:
            mm_processor_kwargs = {}

        # In the case of the default mapper, we have to get resource
        # processor through its HuggingFace autoclass; since this goes
        # through **kwargs, we can't inspect it the same way, so we allow
        # drop mm_processor_kwargs based on signature inspection
        # if we're using the default mapper.
        #
        # This should be safe in general due to the sanitation, since the
        # transformers resource should filter unused kwargs anyway.
        uses_default_mapper = mapper == self._default_input_mapper
        mm_processor_kwargs = resolve_mm_processor_kwargs(
            model_config.mm_processor_kwargs,
            mm_processor_kwargs,
            callable=mapper,
            allow_var_kwargs=uses_default_mapper,
        )
        return mapper(InputContext(model_config), data, **mm_processor_kwargs)

    @abstractmethod
    def _default_max_multimodal_tokens(self, ctx: InputContext) -> int:
        """
        Calculate the maximum number of tokens, corresponding to a single
        instance of multimodal data, that are passed to the language model.
        """
        raise NotImplementedError

    def _validate_max_multimodal_tokens(self, max_mm_tokens: int):
        if max_mm_tokens < 1:
            raise ValueError("You should set the number of tokens to a "
                             f"positive integer. Found: {max_mm_tokens}")

    def register_max_multimodal_tokens(
        self,
        max_mm_tokens: Optional[MultiModalTokensCalc] = None,
    ):
        """
        Register the maximum number of tokens, corresponding to a single
        instance of multimodal data, that are passed to the language model
        for a model class.

        If `None` is provided, then the default calculation is used instead.

        See also:
            :ref:`enabling_multimodal_inputs`
        """

        def wrapper(model_cls: N) -> N:
            if model_cls in self._max_mm_tokens:
                logger.warning(
                    "Model class %s already calculates maximum number of "
                    "tokens in %s. It is overwritten by the new one.",
                    model_cls,
                    self,
                )

            if isinstance(max_mm_tokens, int):
                self._validate_max_multimodal_tokens(max_mm_tokens)

            self._max_mm_tokens[model_cls] = (
                max_mm_tokens or self._default_max_multimodal_tokens)

            return model_cls

        return wrapper

    def get_max_multimodal_tokens(self, model_config: "ModelConfig") -> int:
        """
        Get the maximum number of multi-modal tokens
        for profiling the memory usage of a model.

        If this registry is not applicable to the model, `0` is returned.

        The model is identified by ``model_config``.

        See also:
            :ref:`enabling_multimodal_inputs`
        """
        # Avoid circular import
        from vllm.model_executor.model_loader import get_model_architecture

        model_cls, _ = get_model_architecture(model_config)

        if model_cls not in self._input_mappers:
            return 0

        max_mm_tokens = self._max_mm_tokens.get(model_cls)
        if max_mm_tokens is None:
            raise KeyError(f"No maximum number of multi-modal tokens is given "
                           f"for model class {model_cls.__name__} in {self}.")

        if callable(max_mm_tokens):
            mm_processor_kwargs = get_allowed_kwarg_only_overrides(
                max_mm_tokens, overrides=model_config.mm_processor_kwargs)
            max_mm_tokens = max_mm_tokens(InputContext(model_config),
                                          **mm_processor_kwargs)

        self._validate_max_multimodal_tokens(max_mm_tokens)

        return max_mm_tokens


class MultiModalPlaceholderMap:
    """
    Relates multi-modal embeddings to their corresponding placeholders.
    """

    class IndexMap(NamedTuple):
        src: List[int]
        dest: List[int]

    src_ranges: List[range]
    """
    The indices of the multi-modal embeddings that will replace the
    corresponding placeholder embeddings pointed to by ``dest_ranges``.
    """

    src_len: int
    """
    The total number of flattened multi-modal embeddings.
    """

    dest_ranges: List[range]
    """
    The indices of the placeholder embeddings that will be replaced by the
    multimodal embeddings.
    """

    dest_len: int
    """
    The total number of embeddings in the destination tensor.
    """

    def __init__(self):
        self.src_ranges = []
        self.src_len = 0
        self.dest_ranges = []
        self.dest_len = 0

    @classmethod
    def from_seq_group(
        cls, seq_group: "SequenceGroupMetadata", positions: range
    ) -> Tuple[Optional[MultiModalDataDict], Dict[str,
                                                  "MultiModalPlaceholderMap"]]:
        """
        Returns the multi-modal items that intersect with the portion of a
        prompt (``seq_group``) represented by ``positions``, as well as a
        ``MultiModalPlaceholderMap`` that relates the multi-modal embedding
        vectors to their corresponding placeholders.

        Consider the following scenarios:

           Prompt: |AAAA BBBB What's in these images?|
        Positions: |.................................|

            images      = [A, B]
            src_ranges  = [(0, 4), (4, 8)]
            dest_ranges = [(0, 4), (5, 9)]

           Prompt: |AAAA BBBB What's in these images?|
        Positions: |  .....                          |

            images      = [A, B]
            src_ranges  = [(2, 4), (4, 6)]
            dest_ranges = [(0, 2), (3, 5)]

           Prompt: |AAAA BBBB What's in these images?|
        Positions: |     .........                   |

            images      = [B]
            src_ranges  = [(0, 4)]
            dest_ranges = [(0, 4)]

           Prompt: |AAAA BBBB What's in these images?|
        Positions: |          .......................|

            images      = []
            src_ranges  = []
            dest_ranges = []
        """
        if (not seq_group.multi_modal_data
                or not seq_group.multi_modal_placeholders):
            return seq_group.multi_modal_data, {}

        mm_data = {**seq_group.multi_modal_data}
        placeholder_maps: Dict[str, MultiModalPlaceholderMap] = defaultdict(
            MultiModalPlaceholderMap)

        for (
                modality,
                placeholders,
        ) in seq_group.multi_modal_placeholders.items():
            mm_items = mm_data.pop(modality)
            if not isinstance(mm_items, list):
                mm_items = [mm_items]

            if positions:
                intersecting_items = placeholder_maps[
                    modality].append_items_from_seq_group(
                        positions, mm_items, placeholders)

                if intersecting_items:
                    mm_data[modality] = intersecting_items

        return mm_data, placeholder_maps

    def append_items_from_seq_group(
        self,
        positions: range,
        multi_modal_items: List[_T],
        multi_modal_placeholders: Sequence[PlaceholderRange],
    ) -> List[_T]:
        """
        Adds the multi-modal items that intersect ```positions`` to this
        placeholder map and returns the intersecting items.
        """
        intersecting_items = []

        if len(multi_modal_items) != len(multi_modal_placeholders):
            raise ValueError(
                "Multi-modal placeholders and items must have the same length."
            )
        for placeholder_dict, mm_item in zip(multi_modal_placeholders,
                                             multi_modal_items):
            placeholder = range(
                placeholder_dict["offset"],
                placeholder_dict["offset"] + placeholder_dict["length"],
            )
            intersection = range(
                max(positions.start, placeholder.start),
                min(positions.stop, placeholder.stop),
            )

            if not intersection:
                # Skip this multi-modal item.
                continue

            token_embedding_range = range(
                intersection.start - positions.start,
                intersection.stop - positions.start,
            )

            multimodal_embedding_range = range(
                intersection.start - placeholder.start + self.src_len,
                intersection.stop - placeholder.start + self.src_len,
            )

            intersecting_items.append(mm_item)
            self.dest_ranges.append(token_embedding_range)
            self.src_ranges.append(multimodal_embedding_range)
            self.src_len += len(placeholder)

        self.dest_len += len(positions)
        return intersecting_items

    def extend(self, other: "MultiModalPlaceholderMap"):
        """
        Adds the placeholders from another ``MultiModalPlaceholderMap`` to this
        instance based on the source and destination tensors being
        concatenated.
        """

        self.src_ranges.extend(
            range(self.src_len + r.start, self.src_len + r.stop)
            for r in other.src_ranges)
        self.src_len += other.src_len
        self.dest_ranges.extend(
            range(self.dest_len + r.start, self.dest_len + r.stop)
            for r in other.dest_ranges)
        self.dest_len += other.dest_len

    def index_map(self) -> "IndexMap":
        """
        Finalizes the placeholder map into lists of indices that can be used to
        index the source and destination tensors.
        """

        src_indices = [i for r in self.src_ranges for i in r]
        dest_indices = [i for r in self.dest_ranges for i in r]

        if len(src_indices) != len(dest_indices):
            raise ValueError(
                f"The number of source ({len(src_indices)}) and destination "
                f"indices ({len(dest_indices)}) must be the same.")

        return MultiModalPlaceholderMap.IndexMap(src=src_indices,
                                                 dest=dest_indices)


def __getattr__(name: str):
    import warnings

    if name == "MultiModalInputs":
        msg = ("MultiModalInputs has been renamed to MultiModalKwargs. "
               "The original name will take another meaning in an upcoming "
               "version.")

        warnings.warn(DeprecationWarning(msg), stacklevel=2)

        return MultiModalKwargs

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

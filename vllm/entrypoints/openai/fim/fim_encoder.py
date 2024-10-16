from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Type, Union

from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.utils import is_list_of


class FIMEncoder(ABC):
    """
    An encoder of fill-in-the-middle (FIM) prompts comprising prefix
    and suffix strings.
    """

    def __init__(self, tokenizer: AnyTokenizer):
        self.tokenizer = tokenizer

    @abstractmethod
    def encode_with_suffix(self, prompt: str, suffix: str) -> List[int]:
        """
        Encode the provided prompt prefix and suffix
        to a list of token ids
        """
        pass

    @classmethod
    def for_tokenizer(cls: Type, tokenizer: AnyTokenizer) -> "FIMEncoder":
        fim_encoder = getattr(tokenizer, "fim_encoder", None)
        if fim_encoder is None:
            fim_encoder = cls(tokenizer)
            tokenizer.fim_encoder = fim_encoder  # type: ignore[union-attr]
        return fim_encoder


class FIMEncoderManager:
    fim_encoders: Dict[str, Type] = {}

    @classmethod
    def get_fim_encoder_class(cls, name: Optional[str]) -> Optional[Type]:
        """
        Get FIM encoder by name which is registered by `register_module`.

        Raise a KeyError exception if the name is not registered.
        """
        if name is None:
            return None

        if (encoder := cls.fim_encoders.get(name)) is None:
            raise KeyError(f"fim encoder '{name}' not recognized")

        return encoder

    @classmethod
    def _register_module(cls,
                         module: Type,
                         module_name: Optional[Union[str, List[str]]] = None,
                         force: bool = True) -> None:
        if not issubclass(module, FIMEncoder):
            raise TypeError(
                f'module must be subclass of FIMEncoder, but got {type(module)}'
            )
        if module_name is None:
            module_name = module.__name__
        if isinstance(module_name, str):
            module_name = [module_name]
        for name in module_name:
            if not force and (exist_module := cls.fim_encoders.get(name)
                              is not None):
                raise KeyError(f'{name} is already registered '
                               f'at {exist_module.__module__}')
            cls.fim_encoders[name] = module

    @classmethod
    def register_module(
            cls,
            name: Optional[Union[str, List[str]]] = None,
            force: bool = True,
            module: Union[Type, None] = None) -> Union[type, Callable]:
        """
        Register module with the given name or name list. it can be used as a
        decoder(with module as None) or normal function(with module as not
        None).
        """
        if not isinstance(force, bool):
            raise TypeError(f'force must be a boolean, but got {type(force)}')

        # raise the error ahead of time
        if not (name is None or isinstance(name, str)
                or is_list_of(name, str)):
            raise TypeError(
                'name must be None, an instance of str, or a sequence of str, '
                f'but got {type(name)}')

        # use it as a normal method: x.register_module(module=SomeClass)
        if module is not None:
            cls._register_module(module=module, module_name=name, force=force)
            return module

        # use it as a decorator: @x.register_module()
        def _register(module):
            cls._register_module(module=module, module_name=name, force=force)
            return module

        return _register

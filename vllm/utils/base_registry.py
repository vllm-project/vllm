from abc import ABC, abstractmethod
from typing import Any

from vllm.utils.base_int_enum import BaseIntEnum


class BaseRegistry(ABC):
    _key_class = BaseIntEnum

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._registry = {}
    
    @classmethod
    def register(cls, key: BaseIntEnum, implementation_class: Any) -> None:
        if key in cls._registry:
            return

        cls._registry[key] = implementation_class
    
    @classmethod
    def unregister(cls, key: BaseIntEnum) -> None:
        if key not in cls._registry:
            raise ValueError(f"{key} is not registered")
        
        del cls._registry[key]
    
    @classmethod
    def get(cls, key: BaseIntEnum, *args, **kwargs) -> Any:
        if key not in cls._registry:
            raise ValueError(f"{key} is not registered")
        
        return cls._registry[key](*args, **kwargs)

    @classmethod
    def get_class(cls, key: BaseIntEnum) -> Any:
        if key not in cls._registry:
            raise ValueError(f"{key} is not registered")
        
        return cls._registry[key]
    
    @classmethod
    @abstractmethod
    def get_key_from_str(cls, key_str: str) -> BaseIntEnum:
        pass

    @classmethod
    def get_from_str(cls, key_str: str, *args, **kwargs) -> Any:
        return cls.get(cls.get_key_from_str(key_str), *args, **kwargs)

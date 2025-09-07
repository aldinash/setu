from abc import ABC
from enum import Enum
from typing import Any

from setu.core.configs.utils import get_all_subclasses
from setu.utils.dataclasses import frozen_dataclass


@frozen_dataclass
class BasePolyConfig(ABC):
    """Base class for polymorphic configuration objects.

    This abstract base class provides a framework for creating configuration
    objects that can be dynamically instantiated based on an enum type. It's
    designed to support a factory pattern where concrete configuration classes
    can be created from their corresponding enum values.

    Subclasses must implement the `get_type()` method to return their
    corresponding enum type.

    Example:
        >>> class ModelType(Enum):
        ...     LLAMA = "llama"
        ...     MIXTRAL = "mixtral"
        >>>
        >>> class LlamaConfig(BasePolyConfig):
        ...     @staticmethod
        ...     def get_type():
        ...         return ModelType.LLAMA
        >>>
        >>> config = BasePolyConfig.create_from_type(ModelType.LLAMA)
    """

    @classmethod
    def create_from_type(cls, type_: Enum) -> Any:
        for subclass in get_all_subclasses(cls):
            if subclass.get_type() == type_:
                return subclass()
        raise ValueError(f"Invalid type: {type_}")

    @staticmethod
    def get_type() -> Enum:
        raise NotImplementedError

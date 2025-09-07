from abc import ABC
from enum import Enum
from typing import Any, ClassVar, Dict


class BaseRegistry(ABC):
    """A generic registry pattern implementation for managing implementations.

    This abstract base class provides a registry pattern where implementations
    can be registered with enum keys and retrieved dynamically. Each subclass
    maintains its own isolated registry, allowing different components to have
    separate implementation registries.

    The registry pattern is useful for:
    - Plugin systems where implementations can be added dynamically
    - Factory patterns where objects are created based on configuration
    - Strategy patterns where algorithms are selected at runtime

    Each subclass automatically gets its own empty registry upon definition,
    ensuring that registrations don't leak between different registry types.

    Example:
        >>> from enum import Enum
        >>>
        >>> class ModelType(Enum):
        ...     LLAMA = "llama"
        ...     MIXTRAL = "mixtral"
        >>>
        >>> class ModelRegistry(BaseRegistry):
        ...     pass
        >>>
        >>> ModelRegistry.register(ModelType.LLAMA, LlamaModel)
        >>> class LlamaModel:
        ...     def __init__(self, size):
        ...         self.size = size
        >>>
        >>> # Create a model dynamically
        >>> model = ModelRegistry.get(ModelType.LLAMA, size="7B")

    Attributes:
        _registry: Class-level dictionary mapping enum keys to implementation
                  classes. Each subclass has its own registry.
    """

    # For MyPy to recognize that all subclasses have a _registry class attribute,
    # declare it here as a ClassVar (class-level variable).
    _registry: ClassVar[Dict[Enum, Any]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Each subclass gets its own empty registry.
        cls._registry = {}

    @classmethod
    def register(cls, key: Enum, implementation_class: Any) -> None:
        if key in cls._registry:
            return

        cls._registry[key] = implementation_class

    @classmethod
    def unregister(cls, key: Enum) -> None:
        if key not in cls._registry:
            raise ValueError(f"{key} is not registered")

        del cls._registry[key]

    @classmethod
    def get(cls, key: Enum, *args: Any, **kwargs: Any) -> Any:
        if key not in cls._registry:
            raise ValueError(f"{key} is not registered")

        return cls._registry[key](*args, **kwargs)

# Setu Python Style Guide

## Introduction

This style guide outlines the coding conventions and best practices for writing Python code within the Setu project. It builds upon [PEP 8](https://peps.python.org/pep-0008/) and incorporates Setu-specific patterns. We use **Black** for code formatting, which automatically handles most style decisions.

## Key Goals

- **Readability**: Code should be easy to understand and follow by any developer
- **Consistency**: Uniformity in style across the project reduces cognitive load
- **Type Safety**: Comprehensive type hints improve code reliability and IDE support
- **Maintainability**: Well-structured code is easier to modify and extend
- **Integration**: Seamless interop with C++ components through native handles

## Code Formatting

### Running Black

```bash
# Format all Python files
make format

# Check formatting without making changes
black --check .
```

## Import Organization

### Import Order

Follow this exact order with blank lines between sections:

```python
# 1. Standard library imports
import logging
import time
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import field

# 2. Third-party imports  
import torch
import numpy as np
from transformers import AutoTokenizer

# 3. Setu native imports (C++ bindings)
from setu._native.core.configs import ModelConfig as ModelConfig_C
from setu._native.core.datatypes import Session, SamplingParams

# 4. Setu Python imports
from setu.core.configs.base_poly_config import BasePolyConfig
from setu.core.configs.parallel_config import ParallelConfig
from setu.logger import init_logger
from setu.utils.dataclasses import frozen_dataclass
```

### Import Guidelines

- Use absolute imports for all Setu modules
- Group native (C++) imports separately from Python imports
- Import specific items rather than using `import *`

## Naming Conventions

### Variables and Functions
- **snake_case**: `model_config`, `process_session`, `token_ids`
- **Boolean variables**: Use descriptive predicates: `is_finished`, `has_cache`, `can_schedule`

```python
model_name: str = "llama-7b"
is_ready: bool = False
can_process: bool = True

def process_sessions(sessions: List[Session]) -> List[SamplerOutput]:
    """Process a batch of sessions."""
    pass
```

### Classes and Types
- **PascalCase**: `ModelConfig`, `InferenceEngine`, `SamplingParams`
- **Type aliases**: Descriptive PascalCase names

```python
from typing import Dict, List

# Type aliases
SessionId = str
TokenId = int
SessionMapping = Dict[SessionId, Session]
BatchTokens = List[List[TokenId]]
```

### Constants
- **UPPER_SNAKE_CASE**: `MAX_SEQUENCE_LENGTH`, `DEFAULT_TEMPERATURE`

```python
MAX_SEQUENCE_LENGTH: int = 4096
DEFAULT_TEMPERATURE: float = 1.0
SUPPORTED_MODELS: List[str] = ["llama", "mistral", "mixtral"]
```

### Files and Modules
- **snake_case**: `model_config.py`, `session_manager.py`, `inference_engine.py`

## Type Hints

### Comprehensive Type Annotations

All functions, methods, and class attributes must have type hints:

```python
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass

@dataclass
class ModelConfig:
    model_name: str
    max_model_len: int
    vocab_size: Optional[int] = None
    dtype: str = "float16"
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.max_model_len <= 0:
            raise ValueError(f"max_model_len must be positive, got {self.max_model_len}")

def process_batch(
    sessions: List[Session],
    sampling_params: SamplingParams,
    kv_caches: List[torch.Tensor],
) -> Tuple[List[SamplerOutput], Dict[str, Any]]:
    """Process a batch of sessions with given sampling parameters."""
    # Implementation
    return outputs, metrics
```

### Generic Types

Use generic types for flexible, reusable code:

```python
from typing import TypeVar, Generic, Protocol

T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')

class Cacheable(Protocol):
    def cache_key(self) -> str: ...

class Cache(Generic[T]):
    def __init__(self) -> None:
        self._cache: Dict[str, T] = {}
    
    def get(self, item: Cacheable) -> Optional[T]:
        return self._cache.get(item.cache_key())
    
    def put(self, item: Cacheable, value: T) -> None:
        self._cache[item.cache_key()] = value
```

### Custom Type Annotations

Define domain-specific types for clarity:

```python
from typing import NewType

# Strong type aliases for domain concepts
SessionId = NewType('SessionId', str)
TokenId = NewType('TokenId', int)
PageId = NewType('PageId', int)
GPUDeviceId = NewType('GPUDeviceId', int)

def schedule_session(session_id: SessionId, device: GPUDeviceId) -> None:
    """Schedule session on specific GPU device."""
    pass
```

## Dataclasses and Configuration

### Frozen Dataclasses

Use the Setu `@frozen_dataclass` decorator for configuration classes:

```python
from setu.utils.dataclasses import frozen_dataclass
from dataclasses import field
from typing import List, Optional

@frozen_dataclass
class ModelConfig:
    model_name: str = field(
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        metadata={"help": "Name or path of the HuggingFace model to use."}
    )
    max_model_len: int = field(
        default=4096,
        metadata={"help": "Maximum session length supported by the model."}
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={"help": "Trust remote code when loading model."}
    )
    
    def __post_init__(self) -> None:
        """Validate and initialize derived fields."""
        self._validate_parameters()
        self.hf_config = get_config(self.model_name, self.trust_remote_code)
        # Create native handle for C++ interop
        object.__setattr__(self, 'native_handle', ModelConfig_C(
            model_name=self.model_name,
            max_model_len=self.max_model_len,
            # ... other parameters
        ))
    
    def _validate_parameters(self) -> None:
        """Validate configuration parameters."""
        if self.max_model_len <= 0:
            raise ValueError(f"max_model_len must be positive, got {self.max_model_len}")
        
        if not self.model_name.strip():
            raise ValueError("model_name cannot be empty")
```

### Field Metadata

Use field metadata for documentation and CLI generation:

```python
@frozen_dataclass
class ParallelConfig:
    tensor_parallel_size: int = field(
        default=1,
        metadata={
            "help": "Number of tensor parallel replicas.",
            "constraints": "Must be a power of 2 and <= number of GPUs"
        }
    )
    pipeline_parallel_size: int = field(
        default=1, 
        metadata={
            "help": "Number of pipeline parallel stages.",
            "constraints": "Must be >= 1"
        }
    )
```

## Logging

### Logger Initialization

Use the Setu logging system:

```python
from setu.logger import init_logger

logger = init_logger(__name__)

class InferenceEngine:
    def __init__(self, config: InferenceEngineConfig):
        logger.info("Initializing InferenceEngine with config: %s", config.model_name)
        self.config = config
        # ... initialization
```

### Logging Patterns

Use structured logging with appropriate levels:

```python
def process_session(self, session: Session) -> None:
    """Process an inference session."""
    logger.debug("Processing session %s with %d tokens", 
                session.session_id, len(session.prompt_token_ids))
    
    try:
        # Process session
        result = self._execute_inference(session)
        logger.info("Successfully processed session %s in %.2fs", 
                   session.session_id, result.processing_time)
    
    except Exception as e:
        logger.error("Failed to process session %s: %s", 
                    session.session_id, str(e))
        raise
    
    logger.debug("Session %s generated %d tokens", 
                session.session_id, len(result.output_tokens))
```

## Class Design Patterns

### Polymorphic Base Classes

Use the BasePolyConfig pattern for extensible configurations:

```python
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, ClassVar, Dict

class SchedulerType(Enum):
    FCFS = "fcfs"
    PRIORITY = "priority"
    FAIR_SHARE = "fair_share"

@frozen_dataclass
class BaseSchedulerConfig(BasePolyConfig):
    """Base class for scheduler configurations."""
    
    @classmethod
    @abstractmethod
    def get_type(cls) -> SchedulerType:
        """Return the scheduler type for this config."""
        pass

@frozen_dataclass  
class FCFSSchedulerConfig(BaseSchedulerConfig):
    """First-Come-First-Served scheduler configuration."""
    queue_timeout_s: float = 30.0
    
    @classmethod
    def get_type(cls) -> SchedulerType:
        return SchedulerType.FCFS

@frozen_dataclass
class PrioritySchedulerConfig(BaseSchedulerConfig):
    """Priority-based scheduler configuration."""
    default_priority: int = 5
    max_priority: int = 10
    
    @classmethod 
    def get_type(cls) -> SchedulerType:
        return SchedulerType.PRIORITY
```

### Registry Pattern

Use registries for extensible component systems:

```python
from typing import Type, Dict, Any
from abc import ABC, abstractmethod

class BaseRegistry(ABC):
    """Base class for component registries."""
    _registry: ClassVar[Dict[Enum, Type[Any]]] = {}
    
    @classmethod
    def register(cls, key: Enum, implementation_class: Type[Any]) -> None:
        """Register an implementation for the given key."""
        if key in cls._registry:
            logger.warning("Overriding existing registration for %s", key)
        cls._registry[key] = implementation_class
    
    @classmethod
    def get(cls, key: Enum, *args: Any, **kwargs: Any) -> Any:
        """Create an instance of the registered implementation."""
        if key not in cls._registry:
            raise ValueError(f"{key} is not registered in {cls.__name__}")
        return cls._registry[key](*args, **kwargs)
    
    @classmethod
    def list_registered(cls) -> List[Enum]:
        """List all registered keys."""
        return list(cls._registry.keys())

class BatcherRegistry(BaseRegistry):
    """Registry for batcher implementations."""
    _registry: ClassVar[Dict[BatcherType, Type[BaseBatcher]]] = {}

# Register implementations
BatcherRegistry.register(BatcherType.FIXED_CHUNK, FixedChunkBatcher)
BatcherRegistry.register(BatcherType.DYNAMIC_CHUNK, DynamicChunkBatcher)

# Use registry
batcher = BatcherRegistry.get(BatcherType.FIXED_CHUNK, config)
```

### Native Handle Integration

Integrate with C++ components through native handles:

```python
import setu._native as setu_native
from typing import Optional

class InferenceEngine:
    """Python wrapper for C++ InferenceEngine."""
    
    def __init__(self, config: InferenceEngineConfig):
        self.config = config
        self._tokenizer = AutoTokenizer.from_pretrained(config.model_config.model_name)
        self._session_tracker: Dict[str, SessionInfo] = {}
        
        # Create C++ engine through native handle
        self._native_handle = setu_native.create_inference_engine(
            config.to_native_config()
        )
    
    def add_session(
        self,
        prompt: str,
        sampling_params: SamplingParams,
        session_id: Optional[str] = None,
    ) -> str:
        """Add a new inference session."""
        # Tokenize in Python (ecosystem integration)
        prompt_token_ids = self._tokenizer.encode(prompt)
        
        # Generate ID if needed
        if session_id is None:
            session_id = f"session_{uuid.uuid4().hex[:8]}"
        
        # Create native session
        native_session = setu_native.Session(
            session_id=session_id,
            prompt=prompt,
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params.to_native()
        )
        
        # Pass to C++ engine
        self._native_handle.add_session(native_session)
        
        # Track in Python for API responses
        self._session_tracker[session_id] = SessionInfo(
            prompt=prompt,
            arrival_time=time.time(),
            status="pending"
        )
        
        return session_id
    
    def __del__(self) -> None:
        """Ensure cleanup of native resources."""
        if hasattr(self, '_native_handle'):
            self._native_handle.stop()
```

## Testing Patterns

### Test Class Organization

```python
import pytest
import torch
from unittest.mock import Mock, patch
from setu.core.configs.model_config import ModelConfig
from setu.engine.inference_engine import InferenceEngine

class TestModelConfig:
    """Test ModelConfig functionality."""
    
    def test_valid_config_creation(self):
        """Test creating a valid ModelConfig."""
        config = ModelConfig(
            model_name="meta-llama/Meta-Llama-3-8B-Instruct",
            max_model_len=4096
        )
        assert config.model_name == "meta-llama/Meta-Llama-3-8B-Instruct"
        assert config.max_model_len == 4096
    
    def test_invalid_max_model_len_raises_error(self):
        """Test that invalid max_model_len raises ValueError."""
        with pytest.raises(ValueError, match="max_model_len must be positive"):
            ModelConfig(
                model_name="test-model",
                max_model_len=-1
            )
    
    def test_empty_model_name_raises_error(self):
        """Test that empty model_name raises ValueError."""
        with pytest.raises(ValueError, match="model_name cannot be empty"):
            ModelConfig(
                model_name="",
                max_model_len=4096
            )

class TestInferenceEngine:
    """Test InferenceEngine functionality."""
    
    @pytest.fixture
    def mock_config(self) -> ModelConfig:
        """Create a mock configuration for testing."""
        return ModelConfig(
            model_name="test-model",
            max_model_len=2048
        )
    
    @pytest.fixture  
    def engine(self, mock_config: ModelConfig) -> InferenceEngine:
        """Create an InferenceEngine for testing."""
        with patch('setu._native.create_inference_engine'):
            return InferenceEngine(mock_config)
    
    def test_add_session_returns_session_id(self, engine: InferenceEngine):
        """Test that add_session returns a valid session ID."""
        session_id = engine.add_session(
            prompt="Test prompt",
            sampling_params=SamplingParams(temperature=0.7)
        )
        assert isinstance(session_id, str)
        assert len(session_id) > 0
    
    def test_add_session_with_empty_prompt_raises_error(self, engine: InferenceEngine):
        """Test that empty prompt raises ValueError."""
        with pytest.raises(ValueError, match="Prompt cannot be empty"):
            engine.add_session(
                prompt="",
                sampling_params=SamplingParams()
            )
```

### Pytest Configuration

Use pytest with appropriate fixtures and markers:

```python
# conftest.py
import pytest
import torch

@pytest.fixture(scope="session")
def cuda_available():
    """Check if CUDA is available for testing."""
    return torch.cuda.is_available()

@pytest.fixture
def sample_model_config():
    """Provide a standard model config for testing."""
    return ModelConfig(
        model_name="test-model",
        max_model_len=1024,
        dtype="float16"
    )

# Mark GPU tests
pytestmark = pytest.mark.gpu

class TestGPUOperations:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_memory_allocation(self):
        """Test GPU memory allocation."""
        pass
```

## Documentation

### Docstring Style

Use Google-style docstrings with type information:

```python
def schedule_sessions(
    self,
    sessions: List[Session],
    available_pages: int,
    current_time: float,
) -> Tuple[List[Session], List[Session]]:
    """Schedule sessions based on available memory and priorities.
    
    This method implements the core scheduling logic, deciding which sessions
    can be executed based on memory constraints and scheduling policies.
    
    Args:
        sessions: List of sessions to schedule.
        available_pages: Number of memory pages available for allocation.
        current_time: Current timestamp for scheduling decisions.
        
    Returns:
        A tuple containing:
            - List of sessions that can be scheduled immediately
            - List of sessions that must wait for resources
            
    Raises:
        ValueError: If available_pages is negative.
        RuntimeError: If scheduling algorithm fails to converge.
        
    Example:
        >>> scheduler = PriorityScheduler()
        >>> ready, waiting = scheduler.schedule_sessions(sessions, 100, time.time())
        >>> print(f"Scheduled {len(ready)} sessions")
    """
    if available_pages < 0:
        raise ValueError(f"available_pages must be non-negative, got {available_pages}")
    
    # Implementation...
```

### Module-Level Documentation

```python
"""Setu model configuration management.

This module provides configuration classes for model loading, validation,
and parameter management. It integrates with the HuggingFace transformers
library for model discovery and with Setu's native C++ implementation
for performance-critical operations.

Example:
    Basic usage of ModelConfig:
    
    >>> config = ModelConfig(
    ...     model_name="meta-llama/Meta-Llama-3-8B-Instruct",
    ...     max_model_len=4096
    ... )
    >>> config.validate()
    >>> engine = InferenceEngine(config)

Classes:
    ModelConfig: Configuration for model loading and execution.
    ModelRegistry: Registry for supported model types.
    
Functions:
    validate_model_config: Validate model configuration parameters.
    load_model_config: Load configuration from file or dictionary.
"""

from typing import Dict, List, Optional
# ... rest of module
```

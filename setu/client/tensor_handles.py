"""
Read/Write handles for thread-safe tensor shard access.

These handles provide context managers for accessing tensor data through
CUDA IPC (Inter-Process Communication). The tensor memory is allocated
on the NodeAgent and shared with the client process.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import torch
from torch.multiprocessing.reductions import rebuild_cuda_tensor

from setu._commons.datatypes import TensorShardRef

if TYPE_CHECKING:
    from setu.client.client import Client


def _rebuild_tensor_from_ipc_spec(spec_dict: dict) -> torch.Tensor:
    """
    Rebuild a PyTorch tensor from an IPC specification dictionary.

    Args:
        spec_dict: Dictionary containing tensor IPC specification fields

    Returns:
        PyTorch tensor backed by the shared CUDA memory
    """
    args = {
        **spec_dict,
        "tensor_cls": torch.Tensor,
        "storage_cls": torch.storage.UntypedStorage,
    }
    return rebuild_cuda_tensor(**args)


class TensorReadHandle:
    """
    Context manager for read access to tensor shard device memory.

    Provides a PyTorch tensor view backed by CUDA IPC shared memory.
    The tensor should only be read, not modified, within this context.

    Example:
        >>> with TensorReadHandle(client, shard_ref) as tensor:
        ...     data = tensor.clone()
        ...     result = tensor.sum()
    """

    def __init__(self, client: Client, shard_ref: TensorShardRef) -> None:
        """
        Initialize read handle.

        Args:
            client: Client instance for accessing tensor operations
            shard_ref: TensorShardRef to acquire read access for
        """
        self._client = client
        self._shard_ref = shard_ref
        self._tensor: Optional[torch.Tensor] = None

    def __enter__(self) -> torch.Tensor:
        """
        Acquire read access and return tensor view.

        Gets the IPC handle from the NodeAgent and reconstructs the tensor
        in this process. The returned tensor is backed by the same GPU
        memory as the original tensor on the NodeAgent.

        Returns:
            PyTorch tensor view of the shard's device memory
        """
        # Get IPC spec from NodeAgent
        tensor_ipc_spec = self._client.get_tensor_handle(self._shard_ref)
        spec_dict = tensor_ipc_spec.to_dict()

        # Rebuild tensor from IPC handle
        self._tensor = _rebuild_tensor_from_ipc_spec(spec_dict)

        return self._tensor

    def __exit__(
        self, exc_type: Optional[type], exc_val: Optional[BaseException], exc_tb: Any
    ) -> None:
        """
        Release read access.
        """
        self._tensor = None


class TensorWriteHandle:
    """
    Context manager for write access to tensor shard device memory.

    Provides a PyTorch tensor view backed by CUDA IPC shared memory.
    The tensor can be read and modified within this context.

    Example:
        >>> with TensorWriteHandle(client, shard_ref) as tensor:
        ...     tensor.fill_(1.0)
        ...     tensor[0, :] = some_data
    """

    def __init__(self, client: Client, shard_ref: TensorShardRef) -> None:
        """
        Initialize write handle.

        Args:
            client: Client instance for accessing tensor operations
            shard_ref: TensorShardRef to acquire write access for
        """
        self._client = client
        self._shard_ref = shard_ref
        self._tensor: Optional[torch.Tensor] = None

    def __enter__(self) -> torch.Tensor:
        """
        Acquire write access and return tensor view.

        Gets the IPC handle from the NodeAgent and reconstructs the tensor
        in this process. The returned tensor is backed by the same GPU
        memory as the original tensor on the NodeAgent.

        Returns:
            PyTorch tensor view of the shard's device memory
        """
        # Get IPC spec from NodeAgent
        tensor_ipc_spec = self._client.get_tensor_handle(self._shard_ref)
        spec_dict = tensor_ipc_spec.to_dict()

        # Rebuild tensor from IPC handle
        self._tensor = _rebuild_tensor_from_ipc_spec(spec_dict)

        return self._tensor

    def __exit__(
        self, exc_type: Optional[type], exc_val: Optional[BaseException], exc_tb: Any
    ) -> None:
        """
        Release write access.
        """
        self._tensor = None

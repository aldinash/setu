"""
Read/Write handles for thread-safe tensor shard access.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

import torch
from torch.multiprocessing.reductions import rebuild_cuda_tensor

from setu._commons.datatypes import (
    TensorShard,
    TensorShardMetadata,
    TensorShardReadHandle as NativeReadHandle,
    TensorShardRef,
    TensorShardWriteHandle as NativeWriteHandle,
)

if TYPE_CHECKING:
    from setu.client.client import Client


class CachedTensorHandle(NamedTuple):
    """Cached result of a get_tensor_handle RPC + tensor rebuild."""

    tensor: torch.Tensor
    metadata: TensorShardMetadata
    lock_base_dir: str


def _rebuild_tensor_from_ipc_spec(tensor_ipc_spec: Any) -> torch.Tensor:
    """Rebuild a CUDA tensor from an IPC spec.

    Args:
        tensor_ipc_spec: TensorIPCSpec from C++ containing CUDA IPC handle info

    Returns:
        Reconstructed PyTorch tensor pointing to shared GPU memory
    """
    spec_dict = tensor_ipc_spec.to_dict()
    args = {
        **spec_dict,
        "tensor_cls": torch.Tensor,
        "storage_cls": torch.storage.UntypedStorage,
    }
    return rebuild_cuda_tensor(**args)


def _get_or_create_cached_handle(
    client: "Client", shard_ref: TensorShardRef
) -> CachedTensorHandle:
    """Get a cached tensor handle, or fetch and cache on first access.

    Args:
        client: Python Client instance
        shard_ref: TensorShardRef to look up

    Returns:
        CachedTensorHandle with rebuilt tensor, metadata, and lock_base_dir
    """
    shard_id = shard_ref.shard_id
    cached = client.get_cached_tensor_handle(shard_id)
    if cached is not None:
        return cached

    # Cache miss — fetch from NodeAgent and rebuild
    response = client.get_tensor_handle(shard_ref)
    tensor = _rebuild_tensor_from_ipc_spec(response.tensor_ipc_spec)
    cached = CachedTensorHandle(
        tensor=tensor,
        metadata=response.metadata,
        lock_base_dir=response.lock_base_dir,
    )
    client.set_cached_tensor_handle(shard_id, cached)
    return cached


class TensorReadHandle:
    """Context manager for read access to tensor shard device memory."""

    def __init__(
        self, client: "Client", shard_ref: TensorShardRef
    ) -> None:
        """
        Initialize read handle.

        Args:
            client: Python Client instance for accessing tensor operations
            shard_ref: TensorShardRef to acquire read access for
        """
        self._client = client
        self._shard_ref = shard_ref
        self._tensor: torch.Tensor | None = None
        self._shard: TensorShard | None = None
        self._handle: NativeReadHandle | None = None

    def __enter__(self) -> torch.Tensor:
        """
        Acquire read lock and return tensor view.

        Returns:
            PyTorch tensor view of the shard's device memory
        """
        # 1. Get cached (or fetch) tensor, metadata, and lock_base_dir
        cached = _get_or_create_cached_handle(self._client, self._shard_ref)
        self._tensor = cached.tensor

        # 2. Create TensorShard (C++ object) with same lock file path
        self._shard = TensorShard(
            metadata=cached.metadata,
            tensor=self._tensor,
            lock_base_dir=cached.lock_base_dir,
        )

        # 3. Create C++ read handle - acquires shared lock automatically (RAII)
        self._handle = NativeReadHandle(self._shard)

        return self._tensor

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Release read lock."""
        # Handle destructor releases the lock automatically (RAII)
        self._handle = None
        self._shard = None
        self._tensor = None


class TensorWriteHandle:
    """Context manager for write access to tensor shard device memory."""

    def __init__(
        self, client: "Client", shard_ref: TensorShardRef
    ) -> None:
        """
        Initialize write handle.

        Args:
            client: Python Client instance for accessing tensor operations
            shard_ref: TensorShardRef to acquire write access for
        """
        self._client = client
        self._shard_ref = shard_ref
        self._tensor: torch.Tensor | None = None
        self._shard: TensorShard | None = None
        self._handle: NativeWriteHandle | None = None

    def __enter__(self) -> torch.Tensor:
        """
        Acquire write lock and return tensor view.

        Returns:
            PyTorch tensor view of the shard's device memory
        """
        # 1. Get cached (or fetch) tensor, metadata, and lock_base_dir
        cached = _get_or_create_cached_handle(self._client, self._shard_ref)
        self._tensor = cached.tensor

        # 2. Create TensorShard (C++ object) with same lock file path
        self._shard = TensorShard(
            metadata=cached.metadata,
            tensor=self._tensor,
            lock_base_dir=cached.lock_base_dir,
        )

        # 3. Create C++ write handle - acquires exclusive lock automatically (RAII)
        self._handle = NativeWriteHandle(self._shard)

        return self._tensor

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Release write lock."""
        # Handle destructor releases the lock automatically (RAII)
        self._handle = None
        self._shard = None
        self._tensor = None

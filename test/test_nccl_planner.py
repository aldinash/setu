"""
Tests for Planner compilation with NCCL backend.
"""

import uuid

import torch

from setu._commons.datatypes import (
    CopySpec,
    Device,
    TensorDim,
    TensorDimSpec,
    TensorSelection,
    TensorShardSpec,
)
from setu._coordinator import (
    MetaStore,
    NCCLBackend,
    PassManager,
    Planner,
)


def test_nccl_planner_basic_copy():
    backend = NCCLBackend()
    pass_manager = PassManager()
    planner = Planner(backend, pass_manager)
    metastore = MetaStore()

    node_id = uuid.uuid4()
    device = Device(torch.device("cpu"))

    dim_spec = TensorDimSpec("dim0", 128, 0, 128)

    src_shard_spec = TensorShardSpec("tensor_a", [dim_spec], torch.float32, device)
    dst_shard_spec = TensorShardSpec("tensor_b", [dim_spec], torch.float32, device)

    metastore.register_tensor_shard(src_shard_spec, node_id)
    metastore.register_tensor_shard(dst_shard_spec, node_id)

    dim_map = {"dim0": TensorDim("dim0", 128)}
    src_selection = TensorSelection("tensor_a", dim_map)
    dst_selection = TensorSelection("tensor_b", dim_map)

    copy_spec = CopySpec("tensor_a", "tensor_b", src_selection, dst_selection)

    plan = planner.compile(copy_spec, metastore)

    assert plan is not None
    print(plan)


def test_nccl_planner_with_pass_manager():
    backend = NCCLBackend()
    pass_manager = PassManager()
    assert pass_manager.num_passes() == 0

    planner = Planner(backend, pass_manager)
    metastore = MetaStore()

    node_id = uuid.uuid4()
    device = Device(torch.device("cpu"))

    dim_spec = TensorDimSpec("dim0", 128, 0, 128)

    src_shard_spec = TensorShardSpec("tensor_a", [dim_spec], torch.float32, device)
    dst_shard_spec = TensorShardSpec("tensor_b", [dim_spec], torch.float32, device)

    metastore.register_tensor_shard(src_shard_spec, node_id)
    metastore.register_tensor_shard(dst_shard_spec, node_id)

    dim_map = {"dim0": TensorDim("dim0", 128)}
    src_selection = TensorSelection("tensor_a", dim_map)
    dst_selection = TensorSelection("tensor_b", dim_map)

    copy_spec = CopySpec("tensor_a", "tensor_b", src_selection, dst_selection)

    plan = planner.compile(copy_spec, metastore)

    assert plan is not None
    print(plan)

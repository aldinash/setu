"""
Tests for NCCLPlanner compilation.
"""

from __future__ import annotations

import torch
from expecttest import assert_expected_inline
from planner_test_utils import PlannerTestHelper

from setu._commons.datatypes import TensorDimSpec


def test_nccl_planner_basic_copy():
    helper = PlannerTestHelper()

    helper.register_tensor(
        "tensor_a",
        "shard_a",
        torch.device("cpu"),
        "node_1",
        [TensorDimSpec("dim0", 128, 0, 128)],
    )
    helper.register_tensor(
        "tensor_b",
        "shard_b",
        torch.device("cpu"),
        "node_1",
        [TensorDimSpec("dim0", 128, 0, 128)],
    )

    plan = helper.compile(helper.select("tensor_a"), helper.select("tensor_b"))

    assert plan is not None
    assert len(plan.participants) == 1
    assert_expected_inline(
        helper.format_plan(plan),
        """\
Plan(num_participants=1):
  Participant(node=node_1, device=Device(torch_device=cpu)):
    InitComm(comm=COMM_0, ranks={'Participant(node=node_1, device=Device(torch_device=cpu))': 0})
    Copy(src=(shard_a, node=node_1), src_offset=0, dst=(shard_b, node=node_1), dst_offset=0, count=128, dtype=torch.float32)""",
    )


def test_nccl_planner_partial_copy():
    """Test compile with explicit TensorSelection using .where() slicing."""
    helper = PlannerTestHelper()

    helper.register_tensor(
        "tensor_a",
        "shard_a",
        torch.device("cpu"),
        "node_1",
        [TensorDimSpec("dim0", 128, 0, 128)],
    )
    helper.register_tensor(
        "tensor_b",
        "shard_b",
        torch.device("cpu"),
        "node_1",
        [TensorDimSpec("dim0", 128, 0, 128)],
    )

    # Select only the first 64 elements from each tensor
    src_sel = helper.select("tensor_a", {"dim0": (0, 64)})
    dst_sel = helper.select("tensor_b", {"dim0": (0, 64)})

    plan = helper.compile(src_sel, dst_sel)

    assert plan is not None
    assert_expected_inline(
        helper.format_plan(plan),
        """\
Plan(num_participants=1):
  Participant(node=node_1, device=Device(torch_device=cpu)):
    InitComm(comm=COMM_0, ranks={'Participant(node=node_1, device=Device(torch_device=cpu))': 0})
    Copy(src=(shard_a, node=node_1), src_offset=0, dst=(shard_b, node=node_1), dst_offset=0, count=64, dtype=torch.float32)""",
    )


def test_nccl_planner_2d_sharded_copy():
    """Test copy from a full 2D CPU tensor to a tensor sharded across two GPUs."""
    helper = PlannerTestHelper()

    # tensor_a: 2D [128, 64] fully on CPU, single shard
    helper.register_tensor(
        "tensor_a",
        "shard_a",
        torch.device("cpu"),
        "node_1",
        [TensorDimSpec("rows", 128, 0, 128), TensorDimSpec("cols", 64, 0, 32)],
    )

    helper.register_tensor(
        "tensor_a",
        "shard_a",
        torch.device("cuda:0"),
        "node_1",
        [TensorDimSpec("rows", 128, 0, 128), TensorDimSpec("cols", 64, 32, 64)],
    )

    # tensor_b: 2D [128, 64] sharded along rows across two GPUs
    helper.register_tensor(
        "tensor_b",
        "shard_b_0",
        torch.device("cuda:1"),
        "node_2",
        [TensorDimSpec("rows", 128, 0, 64), TensorDimSpec("cols", 64, 0, 64)],
    )
    helper.register_tensor(
        "tensor_b",
        "shard_b_1",
        torch.device("cuda:2"),
        "node_3",
        [TensorDimSpec("rows", 128, 64, 128), TensorDimSpec("cols", 64, 0, 64)],
    )

    plan = helper.compile(helper.select("tensor_a"), helper.select("tensor_b"))

    assert plan is not None
    assert_expected_inline(
        helper.format_plan(plan),
        """\
Plan(num_participants=4):
  Participant(node=node_1, device=Device(torch_device=cpu)):
    InitComm(comm=COMM_0, ranks={'Participant(node=node_1, device=Device(torch_device=cpu))': 0, 'Participant(node=node_1, device=Device(torch_device=cuda:0))': 1, 'Participant(node=node_2, device=Device(torch_device=cuda:1))': 2, 'Participant(node=node_3, device=Device(torch_device=cuda:2))': 3})
    Send(src=(shard_a, node=node_1), offset=0, count=4096, dtype=torch.float32, peer_rank=2)
  Participant(node=node_1, device=Device(torch_device=cuda:0)):
    InitComm(comm=COMM_0, ranks={'Participant(node=node_1, device=Device(torch_device=cpu))': 0, 'Participant(node=node_1, device=Device(torch_device=cuda:0))': 1, 'Participant(node=node_2, device=Device(torch_device=cuda:1))': 2, 'Participant(node=node_3, device=Device(torch_device=cuda:2))': 3})
    Send(src=(shard_a, node=node_1), offset=0, count=4096, dtype=torch.float32, peer_rank=3)
  Participant(node=node_2, device=Device(torch_device=cuda:1)):
    InitComm(comm=COMM_0, ranks={'Participant(node=node_1, device=Device(torch_device=cpu))': 0, 'Participant(node=node_1, device=Device(torch_device=cuda:0))': 1, 'Participant(node=node_2, device=Device(torch_device=cuda:1))': 2, 'Participant(node=node_3, device=Device(torch_device=cuda:2))': 3})
    Receive(dst=(shard_b_0, node=node_2), offset=0, count=4096, dtype=torch.float32, peer_rank=0)
  Participant(node=node_3, device=Device(torch_device=cuda:2)):
    InitComm(comm=COMM_0, ranks={'Participant(node=node_1, device=Device(torch_device=cpu))': 0, 'Participant(node=node_1, device=Device(torch_device=cuda:0))': 1, 'Participant(node=node_2, device=Device(torch_device=cuda:1))': 2, 'Participant(node=node_3, device=Device(torch_device=cuda:2))': 3})
    Receive(dst=(shard_b_1, node=node_3), offset=0, count=4096, dtype=torch.float32, peer_rank=1)""",
    )

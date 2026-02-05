"""
Integration tests for the Python Client API.

Tests the high-level Python Client class with:
- register_tensor_shard
- select with TensorSelection comparisons
- read / write context managers
"""

import time
import uuid

import pytest
import torch
import torch.multiprocessing as mp

import os

def _run_coordinator(port: int, ready_event, stop_event):
    """Run the Coordinator in a separate process."""
    from setu._coordinator import Coordinator

    coordinator = Coordinator(port)
    coordinator.start()
    ready_event.set()

    while not stop_event.is_set():
        time.sleep(0.05)

    coordinator.stop()


def _run_node_agent(
    port: int,
    coordinator_endpoint: str,
    ready_event,
    stop_event,
    node_id=None,
    device_index: int = 0,
):
    """Run the NodeAgent in a separate process."""
    from setu._commons.datatypes import Device
    from setu._node_manager import NodeAgent

    if node_id is None:
        node_id = uuid.uuid4()

    devices = [Device(torch_device=torch.device(f"cuda:{device_index}"))]

    node_agent = NodeAgent(
        node_id=node_id,
        port=port,
        coordinator_endpoint=coordinator_endpoint,
        devices=devices,
    )
    node_agent.start()
    ready_event.set()

    while not stop_event.is_set():
        time.sleep(0.05)

    node_agent.stop()


@pytest.fixture(scope="module")
def infrastructure():
    """Start Coordinator and NodeAgent for all tests in this module."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    coordinator_port = 29500
    node_agent_port = 29501
    coordinator_endpoint = f"tcp://localhost:{coordinator_port}"
    client_endpoint = f"tcp://localhost:{node_agent_port}"

    ctx = mp.get_context("spawn")
    coordinator_ready = ctx.Event()
    node_agent_ready = ctx.Event()
    stop_event = ctx.Event()

    coordinator_proc = ctx.Process(
        target=_run_coordinator,
        args=(coordinator_port, coordinator_ready, stop_event),
    )
    coordinator_proc.start()
    assert coordinator_ready.wait(timeout=10), "Coordinator failed to start"

    node_agent_proc = ctx.Process(
        target=_run_node_agent,
        args=(
            node_agent_port,
            coordinator_endpoint,
            node_agent_ready,
            stop_event,
        ),
    )
    node_agent_proc.start()
    assert node_agent_ready.wait(timeout=10), "NodeAgent failed to start"

    time.sleep(0.2)

    yield {"client_endpoint": client_endpoint}

    stop_event.set()
    time.sleep(0.3)
    for proc in [node_agent_proc, coordinator_proc]:
        proc.join(timeout=2)
        if proc.is_alive():
            proc.kill()  # Use SIGKILL instead of SIGTERM
            proc.join(timeout=1)


def make_expected_selection(name, dims, owned_ranges):
    """
    Create an expected TensorSelection with specific owned ranges.

    Args:
        name: Tensor name
        dims: Dict of {dim_name: total_size}
        owned_ranges: Dict of {dim_name: (start, end)} for owned indices

    Returns:
        Native TensorSelection with only owned indices selected
    """
    from setu._commons.datatypes import TensorDim, TensorSelection

    # Create TensorDimMap with full sizes
    dim_map = {dim_name: TensorDim(dim_name, size) for dim_name, size in dims.items()}

    # Create full selection
    full_selection = TensorSelection(name, dim_map)

    # Narrow to owned ranges using where()
    result = full_selection
    for dim_name, (start, end) in owned_ranges.items():
        result = result.where(dim_name, set(range(start, end)))

    return result


# ==============================================================================
# Basic Client Tests
# ==============================================================================


@pytest.mark.gpu
def test_client_connect_and_properties(infrastructure):
    """Test Client connection and basic properties."""
    from setu.client import Client

    client = Client(infrastructure["client_endpoint"])

    assert client.is_connected
    assert client.endpoint == infrastructure["client_endpoint"]

    client.disconnect()


@pytest.mark.gpu
def test_client_register_tensor_shard(infrastructure):
    """Test registering a tensor shard through Python Client."""
    from setu._commons.datatypes import Device, TensorDimSpec, TensorShardSpec
    from setu.client import Client

    client = Client(infrastructure["client_endpoint"])

    device = Device(torch_device=torch.device("cuda:0"))
    dims = [
        TensorDimSpec("rows", 64, 0, 64),
        TensorDimSpec("cols", 128, 0, 128),
    ]
    spec = TensorShardSpec(
        name="test_register_tensor",
        dims=dims,
        dtype=torch.float32,
        device=device,
    )

    shard_ref = client.register_tensor_shard(spec)

    assert shard_ref is not None
    assert shard_ref.name == "test_register_tensor"

    client.disconnect()


# ==============================================================================
# Select Tests - Compare Native Selections
# ==============================================================================


@pytest.mark.gpu
def test_select_full_shard_matches_expected(infrastructure):
    """Test select() on full shard matches expected selection."""
    from setu._commons.datatypes import Device, TensorDimSpec, TensorShardSpec
    from setu.client import Client

    client = Client(infrastructure["client_endpoint"])

    device = Device(torch_device=torch.device("cuda:0"))
    dims = [
        TensorDimSpec("rows", 32, 0, 32),
        TensorDimSpec("cols", 64, 0, 64),
    ]
    spec = TensorShardSpec(
        name="select_full_match",
        dims=dims,
        dtype=torch.float32,
        device=device,
    )

    client.register_tensor_shard(spec)

    actual = client.select("select_full_match")
    expected = make_expected_selection(
        "select_full_match",
        {"rows": 32, "cols": 64},
        {"rows": (0, 32), "cols": (0, 64)},
    )

    assert actual.native == expected

    client.disconnect()


@pytest.mark.gpu
def test_select_partial_rows_matches_expected(infrastructure):
    """Test select() on partial rows matches expected selection."""
    from setu._commons.datatypes import Device, TensorDimSpec, TensorShardSpec
    from setu.client import Client

    client = Client(infrastructure["client_endpoint"])

    device = Device(torch_device=torch.device("cuda:0"))
    dims = [
        TensorDimSpec("rows", 100, 20, 50),  # Only rows 20-50 of 100
        TensorDimSpec("cols", 64, 0, 64),
    ]
    spec = TensorShardSpec(
        name="select_partial_rows",
        dims=dims,
        dtype=torch.float32,
        device=device,
    )

    client.register_tensor_shard(spec)

    actual = client.select("select_partial_rows")
    expected = make_expected_selection(
        "select_partial_rows",
        {"rows": 100, "cols": 64},
        {"rows": (20, 50), "cols": (0, 64)},
    )

    assert actual.native == expected

    client.disconnect()


@pytest.mark.gpu
def test_select_partial_both_dims_matches_expected(infrastructure):
    """Test select() on partial shard in both dims matches expected."""
    from setu._commons.datatypes import Device, TensorDimSpec, TensorShardSpec
    from setu.client import Client

    client = Client(infrastructure["client_endpoint"])

    device = Device(torch_device=torch.device("cuda:0"))
    dims = [
        TensorDimSpec("rows", 100, 25, 75),  # rows 25-75 of 100
        TensorDimSpec("cols", 80, 10, 50),  # cols 10-50 of 80
    ]
    spec = TensorShardSpec(
        name="select_partial_both",
        dims=dims,
        dtype=torch.float32,
        device=device,
    )

    client.register_tensor_shard(spec)

    actual = client.select("select_partial_both")
    expected = make_expected_selection(
        "select_partial_both",
        {"rows": 100, "cols": 80},
        {"rows": (25, 75), "cols": (10, 50)},
    )

    assert actual.native == expected

    client.disconnect()


@pytest.mark.gpu
def test_select_3d_tensor_matches_expected(infrastructure):
    """Test select() on 3D tensor matches expected selection."""
    from setu._commons.datatypes import Device, TensorDimSpec, TensorShardSpec
    from setu.client import Client

    client = Client(infrastructure["client_endpoint"])

    device = Device(torch_device=torch.device("cuda:0"))
    dims = [
        TensorDimSpec("batch", 16, 4, 12),  # batch 4-12 of 16
        TensorDimSpec("rows", 64, 0, 32),  # rows 0-32 of 64
        TensorDimSpec("cols", 128, 64, 128),  # cols 64-128 of 128
    ]
    spec = TensorShardSpec(
        name="select_3d_tensor",
        dims=dims,
        dtype=torch.float32,
        device=device,
    )

    client.register_tensor_shard(spec)

    actual = client.select("select_3d_tensor")
    expected = make_expected_selection(
        "select_3d_tensor",
        {"batch": 16, "rows": 64, "cols": 128},
        {"batch": (4, 12), "rows": (0, 32), "cols": (64, 128)},
    )

    assert actual.native == expected

    client.disconnect()


@pytest.mark.gpu
def test_select_multiple_shards_union_matches_expected(infrastructure):
    """Test select() with multiple shards returns union of owned indices."""
    from setu._commons.datatypes import Device, TensorDimSpec, TensorShardSpec
    from setu.client import Client

    client = Client(infrastructure["client_endpoint"])

    device = Device(torch_device=torch.device("cuda:0"))

    # First shard: rows 0-32
    dims1 = [
        TensorDimSpec("rows", 64, 0, 32),
        TensorDimSpec("cols", 32, 0, 32),
    ]
    spec1 = TensorShardSpec(
        name="select_multi_shard",
        dims=dims1,
        dtype=torch.float32,
        device=device,
    )
    client.register_tensor_shard(spec1)

    # Second shard: rows 32-64
    dims2 = [
        TensorDimSpec("rows", 64, 32, 64),
        TensorDimSpec("cols", 32, 0, 32),
    ]
    spec2 = TensorShardSpec(
        name="select_multi_shard",
        dims=dims2,
        dtype=torch.float32,
        device=device,
    )
    client.register_tensor_shard(spec2)

    actual = client.select("select_multi_shard")

    # Union should cover all rows 0-64
    expected = make_expected_selection(
        "select_multi_shard",
        {"rows": 64, "cols": 32},
        {"rows": (0, 64), "cols": (0, 32)},
    )

    assert actual.native == expected
    assert actual.is_spanning()

    client.disconnect()


@pytest.mark.gpu
def test_select_multiple_shards_partial_union(infrastructure):
    """Test select() with multiple non-contiguous shards."""
    from setu._commons.datatypes import Device, TensorDimSpec, TensorShardSpec
    from setu._commons.datatypes import TensorDim, TensorSelection
    from setu.client import Client

    client = Client(infrastructure["client_endpoint"])

    device = Device(torch_device=torch.device("cuda:0"))

    # First shard: rows 0-20
    dims1 = [
        TensorDimSpec("rows", 100, 0, 20),
        TensorDimSpec("cols", 50, 0, 50),
    ]
    spec1 = TensorShardSpec(
        name="select_partial_union",
        dims=dims1,
        dtype=torch.float32,
        device=device,
    )
    client.register_tensor_shard(spec1)

    # Second shard: rows 50-70 (gap between 20-50)
    dims2 = [
        TensorDimSpec("rows", 100, 50, 70),
        TensorDimSpec("cols", 50, 0, 50),
    ]
    spec2 = TensorShardSpec(
        name="select_partial_union",
        dims=dims2,
        dtype=torch.float32,
        device=device,
    )
    client.register_tensor_shard(spec2)

    actual = client.select("select_partial_union")

    # Build expected with union of [0,20) and [50,70)
    dim_map = {"rows": TensorDim("rows", 100), "cols": TensorDim("cols", 50)}
    full_sel = TensorSelection("select_partial_union", dim_map)
    expected = full_sel.where("rows", set(range(0, 20)) | set(range(50, 70))).where(
        "cols", set(range(0, 50))
    )

    assert actual.native == expected
    assert not actual.is_spanning()

    client.disconnect()


# ==============================================================================
# TensorSelection where() Tests
# ==============================================================================


@pytest.mark.gpu
def test_select_where_narrows_selection(infrastructure):
    """Test where() narrows selection and comparison works."""
    from setu._commons.datatypes import Device, TensorDimSpec, TensorShardSpec
    from setu.client import Client

    client = Client(infrastructure["client_endpoint"])

    device = Device(torch_device=torch.device("cuda:0"))
    dims = [
        TensorDimSpec("rows", 32, 0, 32),
        TensorDimSpec("cols", 64, 0, 64),
    ]
    spec = TensorShardSpec(
        name="where_narrow",
        dims=dims,
        dtype=torch.float32,
        device=device,
    )
    client.register_tensor_shard(spec)

    base = client.select("where_narrow")
    narrowed = base.where("rows", [0, 5, 10, 15])

    expected = make_expected_selection(
        "where_narrow",
        {"rows": 32, "cols": 64},
        {"rows": (0, 32), "cols": (0, 64)},
    ).where("rows", {0, 5, 10, 15})

    assert narrowed.native == expected
    assert not narrowed.is_spanning()

    client.disconnect()


@pytest.mark.gpu
def test_select_where_outside_owned_is_empty(infrastructure):
    """Test where() selecting indices outside owned range results in empty."""
    from setu._commons.datatypes import Device, TensorDimSpec, TensorShardSpec
    from setu.client import Client

    client = Client(infrastructure["client_endpoint"])

    device = Device(torch_device=torch.device("cuda:0"))
    dims = [
        TensorDimSpec("rows", 100, 50, 60),  # Only owns rows 50-60
        TensorDimSpec("cols", 50, 0, 50),
    ]
    spec = TensorShardSpec(
        name="where_outside",
        dims=dims,
        dtype=torch.float32,
        device=device,
    )
    client.register_tensor_shard(spec)

    selection = client.select("where_outside").where("rows", [0, 10, 20])

    assert selection.is_empty()

    client.disconnect()


@pytest.mark.gpu
def test_select_intersection_matches_expected(infrastructure):
    """Test get_intersection() returns correct intersection."""
    from setu._commons.datatypes import Device, TensorDimSpec, TensorShardSpec
    from setu.client import Client

    client = Client(infrastructure["client_endpoint"])

    device = Device(torch_device=torch.device("cuda:0"))
    dims = [
        TensorDimSpec("rows", 32, 0, 32),
        TensorDimSpec("cols", 32, 0, 32),
    ]
    spec = TensorShardSpec(
        name="intersection_test",
        dims=dims,
        dtype=torch.float32,
        device=device,
    )
    client.register_tensor_shard(spec)

    sel1 = client.select("intersection_test").where("rows", [0, 1, 2, 3, 4, 5])
    sel2 = client.select("intersection_test").where("rows", [3, 4, 5, 6, 7, 8])

    intersection = sel1.get_intersection(sel2)

    # Expected: rows {3, 4, 5}, cols all
    expected = make_expected_selection(
        "intersection_test",
        {"rows": 32, "cols": 32},
        {"rows": (0, 32), "cols": (0, 32)},
    ).where("rows", {3, 4, 5})

    assert intersection.native == expected

    client.disconnect()


# ==============================================================================
# Read/Write Context Manager Tests
# ==============================================================================


@pytest.mark.gpu
def test_write_and_read_tensor(infrastructure):
    """Test write and read context managers."""
    from setu._commons.datatypes import Device, TensorDimSpec, TensorShardSpec
    from setu.client import Client

    client = Client(infrastructure["client_endpoint"])

    device = Device(torch_device=torch.device("cuda:0"))
    dims = [
        TensorDimSpec("rows", 8, 0, 8),
        TensorDimSpec("cols", 16, 0, 16),
    ]
    spec = TensorShardSpec(
        name="rw_test_tensor",
        dims=dims,
        dtype=torch.float32,
        device=device,
    )

    shard_ref = client.register_tensor_shard(spec)
    time.sleep(0.3)

    test_value = 42.0
    with client.write(shard_ref) as tensor:
        tensor.fill_(test_value)

    with client.read(shard_ref) as tensor:
        assert tensor.shape == (8, 16)
        assert torch.allclose(tensor, torch.full((8, 16), test_value, device="cuda:0"))

    client.disconnect()


@pytest.mark.gpu
def test_write_specific_pattern(infrastructure):
    """Test writing specific values to tensor."""
    from setu._commons.datatypes import Device, TensorDimSpec, TensorShardSpec
    from setu.client import Client

    client = Client(infrastructure["client_endpoint"])

    device = Device(torch_device=torch.device("cuda:0"))
    dims = [
        TensorDimSpec("rows", 4, 0, 4),
        TensorDimSpec("cols", 4, 0, 4),
    ]
    spec = TensorShardSpec(
        name="pattern_tensor",
        dims=dims,
        dtype=torch.float32,
        device=device,
    )

    shard_ref = client.register_tensor_shard(spec)
    time.sleep(0.3)

    with client.write(shard_ref) as tensor:
        tensor.zero_()
        tensor[0, 0] = 1.0
        tensor[1, 1] = 2.0
        tensor[2, 2] = 3.0
        tensor[3, 3] = 4.0

    with client.read(shard_ref) as tensor:
        assert tensor[0, 0].item() == 1.0
        assert tensor[1, 1].item() == 2.0
        assert tensor[2, 2].item() == 3.0
        assert tensor[3, 3].item() == 4.0
        assert tensor[0, 1].item() == 0.0

    client.disconnect()


@pytest.mark.gpu
def test_multiple_shards_write_read(infrastructure):
    """Test writing and reading from multiple shards."""
    from setu._commons.datatypes import Device, TensorDimSpec, TensorShardSpec
    from setu.client import Client

    client = Client(infrastructure["client_endpoint"])

    device = Device(torch_device=torch.device("cuda:0"))

    dims1 = [
        TensorDimSpec("rows", 16, 0, 8),
        TensorDimSpec("cols", 8, 0, 8),
    ]
    spec1 = TensorShardSpec(
        name="multi_rw_tensor",
        dims=dims1,
        dtype=torch.float32,
        device=device,
    )
    shard1 = client.register_tensor_shard(spec1)

    dims2 = [
        TensorDimSpec("rows", 16, 8, 16),
        TensorDimSpec("cols", 8, 0, 8),
    ]
    spec2 = TensorShardSpec(
        name="multi_rw_tensor",
        dims=dims2,
        dtype=torch.float32,
        device=device,
    )
    shard2 = client.register_tensor_shard(spec2)

    time.sleep(0.5)  # Allow time for ZMQ cleanup

    with client.write(shard1) as tensor:
        tensor.fill_(1.0)

    with client.write(shard2) as tensor:
        tensor.fill_(2.0)

    with client.read(shard1) as tensor:
        assert torch.allclose(tensor, torch.full((8, 8), 1.0, device="cuda:0"))

    with client.read(shard2) as tensor:
        assert torch.allclose(tensor, torch.full((8, 8), 2.0, device="cuda:0"))

    client.disconnect()


if __name__ == "__main__":
    mp.set_start_method("spawn")
    pytest.main([__file__, "-v", "--gpu"])

"""
E2E test for shortest-path-optimized copy across two nodes.

Setup:
- Node 0: 4 devices (cuda:0..3), hosts tensor_a sharded by 2 on devices 0,1
- Node 1: 4 devices (cuda:4..7), hosts tensor_b sharded by 4 on devices 4,5,6,7
- Topology: full mesh between all 8 participants
    - Intra-node: latency=0us, bw=200 Gbps
    - Inter-node symmetric (i <-> i+4): latency=10us, bw=100 Gbps
    - Inter-node cross (i <-> j+4, i!=j): latency=10us, bw=50 Gbps
- Planner: NCCL backend + ShortestPathRouting pass
- Pull-based copy: each tensor_b shard pulls from tensor_a
"""

import os
import time
import uuid

import pytest
import torch
import torch.multiprocessing as mp
from torch.multiprocessing.reductions import rebuild_cuda_tensor

os.environ["SETU_LOG_LEVEL"] = "DEBUG"


def _build_planner(config):
    """Build planner with topology + shortest-path routing + NCCL backend."""
    from setu._commons.datatypes import Device
    from setu._coordinator import (
        Link,
        NCCLBackend,
        Participant,
        PassManager,
        Planner,
        RegisterSet,
        ShortestPathRouting,
        Topology,
    )

    topo = Topology()
    for ni, di, nj, dj, lat, bw in config["topo_edges"]:
        pi = Participant(ni, Device(torch.device(f"cuda:{di}")))
        pj = Participant(nj, Device(torch.device(f"cuda:{dj}")))
        topo.add_bidirectional_link(pi, pj, Link(lat, bw))

    pass_manager = PassManager()
    pass_manager.add_pass(ShortestPathRouting(topo))

    reg = RegisterSet.uniform(
        config["registers_per_device"], config["register_size_bytes"]
    )
    register_sets = {}
    for nid, dev_indices in [
        (config["n0_id"], config["n0_device_indices"]),
        (config["n1_id"], config["n1_device_indices"]),
    ]:
        for di in dev_indices:
            p = Participant(nid, Device(torch.device(f"cuda:{di}")))
            register_sets[p] = reg

    backend = NCCLBackend(register_sets)
    return Planner(backend, pass_manager)


def _make_dims_data(dim_name, dim_global_size, num_shards, shard_idx):
    shard_size = dim_global_size // num_shards
    start = shard_idx * shard_size
    end = start + shard_size
    return [(dim_name, dim_global_size, start, end)]


def _rebuild_tensor_from_handle(tensor_ipc_spec):
    """Rebuild a CUDA tensor from an IPC spec."""
    spec_dict = tensor_ipc_spec.to_dict()
    return rebuild_cuda_tensor(
        **spec_dict,
        tensor_cls=torch.Tensor,
        storage_cls=torch.storage.UntypedStorage,
    )


# ==============================================================================
# Process entry points
# ==============================================================================


def _run_coordinator(config, ready_event, stop_event):
    """Spawn coordinator in its own process."""
    from setu._coordinator import Coordinator

    planner = _build_planner(config)
    coordinator = Coordinator(config["coordinator_port"], planner)
    coordinator.start()
    ready_event.set()

    while not stop_event.is_set():
        time.sleep(0.05)

    coordinator.stop()


def _run_node_agent(
    node_id, port, coordinator_endpoint, device_indices, ready_event, stop_event
):
    """Spawn a node agent in its own process."""
    from setu._commons.datatypes import Device
    from setu._node_manager import NodeAgent

    devices = [Device(torch_device=torch.device(f"cuda:{i}")) for i in device_indices]
    agent = NodeAgent(
        node_id=node_id,
        port=port,
        coordinator_endpoint=coordinator_endpoint,
        devices=devices,
    )
    agent.start()
    ready_event.set()

    while not stop_event.is_set():
        time.sleep(0.05)

    agent.stop()


def _run_source_client(
    client_endpoint,
    tensor_name,
    dims_data,
    init_value,
    init_done_event,
    result_queue,
    client_id,
    device_index,
):
    """Source client: register shard, fill tensor, signal ready."""
    from setu._client import Client
    from setu._commons.datatypes import Device, TensorDimSpec, TensorShardSpec

    try:
        print(f"[Source {client_id}] Starting on cuda:{device_index}...", flush=True)
        client = Client()
        client.connect(client_endpoint)

        dims_spec = [
            TensorDimSpec(name, global_size, start, end)
            for name, global_size, start, end in dims_data
        ]
        device = Device(torch_device=torch.device(f"cuda:{device_index}"))
        shard_spec = TensorShardSpec(
            name=tensor_name, dims=dims_spec, dtype=torch.float32, device=device
        )

        shard_ref = client.register_tensor_shard(shard_spec)
        if shard_ref is None:
            raise RuntimeError(f"Failed to register source shard {tensor_name}")
        client.wait_for_shard_allocation(shard_ref.shard_id)

        tensor_ipc_spec, _, _ = client.get_tensor_handle(shard_ref)
        tensor = _rebuild_tensor_from_handle(tensor_ipc_spec)
        tensor.fill_(init_value)
        torch.cuda.synchronize(device_index)
        print(f"[Source {client_id}] Tensor initialized to {init_value}", flush=True)

        result_queue.put({"success": True, "client_id": client_id})
        init_done_event.set()

        while True:
            time.sleep(0.1)

    except Exception as e:
        import traceback

        print(f"[Source {client_id}] ERROR: {e}", flush=True)
        result_queue.put(
            {
                "success": False,
                "client_id": client_id,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
        )


def _run_dest_client(
    client_endpoint,
    src_tensor_name,
    dst_tensor_name,
    dims_data,
    all_sources_ready_event,
    expected_value,
    result_queue,
    client_id,
    device_index,
):
    """Dest client: wait for sources, register shard, pull, verify."""
    from setu._client import Client
    from setu._commons.datatypes import (
        CopySpec,
        Device,
        TensorDim,
        TensorDimSpec,
        TensorSelection,
        TensorShardSpec,
    )

    try:
        print(f"[Dest {client_id}] Waiting for sources...", flush=True)
        if not all_sources_ready_event.wait(timeout=30):
            raise RuntimeError("Timeout waiting for sources")

        client = Client()
        client.connect(client_endpoint)

        dims_spec = [
            TensorDimSpec(name, global_size, start, end)
            for name, global_size, start, end in dims_data
        ]
        device = Device(torch_device=torch.device(f"cuda:{device_index}"))
        shard_spec = TensorShardSpec(
            name=dst_tensor_name, dims=dims_spec, dtype=torch.float32, device=device
        )

        shard_ref = client.register_tensor_shard(shard_spec)
        if shard_ref is None:
            raise RuntimeError(f"Failed to register dest shard {dst_tensor_name}")
        client.wait_for_shard_allocation(shard_ref.shard_id)

        dim_map = {
            name: TensorDim(name, global_size) for name, global_size, _, _ in dims_data
        }
        src_selection = TensorSelection(src_tensor_name, dim_map)
        dst_selection = TensorSelection(dst_tensor_name, dim_map)
        copy_spec = CopySpec(
            src_tensor_name, dst_tensor_name, src_selection, dst_selection
        )

        print(f"[Dest {client_id}] Submitting pull...", flush=True)
        copy_op_id = client.submit_pull(copy_spec)
        if copy_op_id is None:
            raise RuntimeError("Failed to submit pull")

        client.wait_for_copy(copy_op_id)
        print(f"[Dest {client_id}] Copy complete!", flush=True)

        tensor_ipc_spec, _, _ = client.get_tensor_handle(shard_ref)
        tensor = _rebuild_tensor_from_handle(tensor_ipc_spec)
        torch.cuda.synchronize(device_index)

        print(f"[Dest {client_id}] {tensor}")

        actual_value = tensor.mean().item()
        values_match = abs(actual_value - expected_value) < 1e-5
        print(
            f"[Dest {client_id}] expected={expected_value}, "
            f"actual={actual_value}, match={values_match}",
            flush=True,
        )

        result_queue.put(
            {
                "success": True,
                "client_id": client_id,
                "expected_value": expected_value,
                "actual_value": actual_value,
                "values_match": values_match,
            }
        )
        client.disconnect()

    except Exception as e:
        import traceback

        print(f"[Dest {client_id}] ERROR: {e}", flush=True)
        result_queue.put(
            {
                "success": False,
                "client_id": client_id,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
        )


# ==============================================================================
# Test
# ==============================================================================


@pytest.mark.gpu
def test_shortest_path_optim_e2e():
    """
    E2E pull with shortest-path routing across 2 nodes / 8 devices.

    - tensor_a on node0, sharded [0..4) on cuda:0 and [4..8) on cuda:1
    - tensor_b on node1, sharded across cuda:4, cuda:5, cuda:6, cuda:7
    - Each tensor_b shard pulls its slice from tensor_a
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if torch.cuda.device_count() < 8:
        pytest.skip(f"Need 8 CUDA devices, got {torch.cuda.device_count()}")

    n0_id = uuid.UUID("00000000-0000-0000-0000-000000000000")
    n1_id = uuid.UUID("00000000-0000-0000-0000-000000000001")

    n0_device_indices = [0, 1, 2, 3]
    n1_device_indices = [4, 5, 6, 7]

    coordinator_port = 29510
    node0_port = 29610
    node1_port = 29611
    coordinator_endpoint = f"tcp://localhost:{coordinator_port}"
    node0_endpoint = f"tcp://localhost:{node0_port}"
    node1_endpoint = f"tcp://localhost:{node1_port}"

    dim_name = "dim0"
    dim_global_size = 1024
    num_src_shards = 2
    num_dst_shards = 4
    init_value = 42.0

    # Build topology edges (picklable plain data)
    # Intra-node: latency=0us, bw=200 Gbps
    # Inter-node symmetric (i <-> i+4): latency=10us, bw=100 Gbps
    # Inter-node cross (i <-> j+4, i!=j): latency=10us, bw=50 Gbps
    topo_edges = []
    # Intra-node0
    for i in range(4):
        for j in range(i + 1, 4):
            topo_edges.append((n0_id, i, n0_id, j, 0, 200))
    # Intra-node1
    for i in range(4, 8):
        for j in range(i + 1, 8):
            topo_edges.append((n1_id, i, n1_id, j, 0, 200))
    # Inter-node
    for i in range(4):
        for j in range(4, 8):
            symmetric = (j - 4) == i  # 0<->4, 1<->5, 2<->6, 3<->7
            topo_edges.append((n0_id, i, n1_id, j, 10, 100 if symmetric else 50))

    config = {
        "n0_id": n0_id,
        "n1_id": n1_id,
        "n0_device_indices": n0_device_indices,
        "n1_device_indices": n1_device_indices,
        "coordinator_port": coordinator_port,
        "topo_edges": topo_edges,
        "registers_per_device": 1,
        "register_size_bytes": 1024 * 1024,
    }

    ctx = mp.get_context("spawn")

    coordinator_ready = ctx.Event()
    node0_ready = ctx.Event()
    node1_ready = ctx.Event()
    stop_event = ctx.Event()

    source_init_events = [ctx.Event() for _ in range(num_src_shards)]
    all_sources_ready = ctx.Event()
    source_results = ctx.Queue()
    dest_results = ctx.Queue()

    processes = []

    try:
        # -- Coordinator --
        coordinator_proc = ctx.Process(
            target=_run_coordinator,
            args=(config, coordinator_ready, stop_event),
        )
        coordinator_proc.start()
        processes.append(coordinator_proc)
        assert coordinator_ready.wait(timeout=10), "Coordinator failed to start"

        # -- Node agents --
        node0_proc = ctx.Process(
            target=_run_node_agent,
            args=(
                n0_id,
                node0_port,
                coordinator_endpoint,
                n0_device_indices,
                node0_ready,
                stop_event,
            ),
        )
        node0_proc.start()
        processes.append(node0_proc)

        node1_proc = ctx.Process(
            target=_run_node_agent,
            args=(
                n1_id,
                node1_port,
                coordinator_endpoint,
                n1_device_indices,
                node1_ready,
                stop_event,
            ),
        )
        node1_proc.start()
        processes.append(node1_proc)

        assert node0_ready.wait(timeout=10), "Node0 agent failed to start"
        assert node1_ready.wait(timeout=10), "Node1 agent failed to start"
        time.sleep(0.5)

        # -- Source clients: tensor_a on node0, devices 0 and 1 --
        for i in range(num_src_shards):
            proc = ctx.Process(
                target=_run_source_client,
                args=(
                    node0_endpoint,
                    "tensor_a",
                    _make_dims_data(dim_name, dim_global_size, num_src_shards, i),
                    init_value,
                    source_init_events[i],
                    source_results,
                    i,
                    i,  # device_index: 0, 1
                ),
            )
            proc.start()
            processes.append(proc)

        for i, event in enumerate(source_init_events):
            assert event.wait(timeout=30), f"Source {i} failed to initialize"
        all_sources_ready.set()
        print("[Main] All sources ready", flush=True)

        for _ in range(num_src_shards):
            result = source_results.get(timeout=30)
            assert result["success"], f"Source failed: {result.get('error')}"

        # -- Dest clients: tensor_b on node1, devices 4,5,6,7 --
        for i in range(num_dst_shards):
            proc = ctx.Process(
                target=_run_dest_client,
                args=(
                    node1_endpoint,
                    "tensor_a",
                    "tensor_b",
                    _make_dims_data(dim_name, dim_global_size, num_dst_shards, i),
                    all_sources_ready,
                    init_value,
                    dest_results,
                    i,
                    4 + i,  # device_index: 4, 5, 6, 7
                ),
            )
            proc.start()
            processes.append(proc)

        for _ in range(num_dst_shards):
            result = dest_results.get(timeout=60)
            print(f"[Main] Dest result: {result}", flush=True)
            assert result["success"], (
                f"Dest {result.get('client_id')} failed: "
                f"{result.get('error')}\n{result.get('traceback', '')}"
            )
            assert result["values_match"], (
                f"Dest {result['client_id']}: "
                f"expected {result['expected_value']}, got {result['actual_value']}"
            )

        print("[Main] All dest clients verified!", flush=True)

    finally:
        stop_event.set()
        time.sleep(0.2)

        for proc in processes:
            proc.terminate()
            proc.join(timeout=2)
            if proc.is_alive():
                proc.kill()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

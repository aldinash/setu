#!/usr/bin/env python3
"""Simple test for Client-NodeAgent communication."""

import concurrent.futures
import threading
import time
from typing import Tuple

from setu._client import Client, ErrorCode
from setu._commons.enums import MsgType
from setu._commons.messages import (
    Header,
    RegisterTensorShardRequest,
    RegisterTensorShardResponse,
)
from setu._node_manager import NodeAgent


def main() -> None:
    router_port = 5555
    dealer_executor_port = 5556
    dealer_handler_port = 5557

    print("=== Testing Client-NodeAgent Communication ===\n")

    # Test message types
    print("--- Testing Message Types ---")
    header = Header(MsgType.REGISTER_TENSOR_SHARD_REQUEST)
    print(f"Header: {header}")

    request = RegisterTensorShardRequest("test_tensor")
    print(f"Request: {request}")

    response = RegisterTensorShardResponse(ErrorCode.SUCCESS)
    print(f"Response: {response}")
    print()

    # Test enum values
    print("--- Testing Enum Values ---")
    print(f"MsgType.REGISTER_TENSOR_SHARD_REQUEST = {MsgType.REGISTER_TENSOR_SHARD_REQUEST}")
    print(f"MsgType.REGISTER_TENSOR_SHARD_RESPONSE = {MsgType.REGISTER_TENSOR_SHARD_RESPONSE}")
    print(f"ErrorCode.SUCCESS = {ErrorCode.SUCCESS}")
    print(f"ErrorCode.INVALID_ARGUMENTS = {ErrorCode.INVALID_ARGUMENTS}")
    print()

    # Create NodeAgent
    print("--- Testing Client-NodeAgent RPC ---")
    print(f"Creating NodeAgent on ports {router_port}, {dealer_executor_port}, {dealer_handler_port}")
    node_agent = NodeAgent(router_port, dealer_executor_port, dealer_handler_port)

    # Start NodeAgent in background
    print("Starting NodeAgent...")
    node_agent.start()
    time.sleep(0.5)  # Wait for NodeAgent to be ready

    # Create Client
    endpoint = f"tcp://localhost:{router_port}"
    print(f"Creating Client connected to {endpoint}")
    client = Client(endpoint)

    # Test RegisterTensorShard
    print("\nSending RegisterTensorShard request...")
    result = client.register_tensor_shard("my_tensor")
    print(f"Result: {result}")

    if result == ErrorCode.SUCCESS:
        print("SUCCESS: Tensor shard registered successfully!")
    else:
        print(f"FAILED: Got error code {result}")

    # Test multiple clients sequentially
    print("\n--- Testing Multiple Clients (Sequential) ---")
    client2 = Client(endpoint)
    client3 = Client(endpoint)

    result1 = client.register_tensor_shard("tensor_from_client1")
    result2 = client2.register_tensor_shard("tensor_from_client2")
    result3 = client3.register_tensor_shard("tensor_from_client3")

    print(f"Client 1 result: {result1}")
    print(f"Client 2 result: {result2}")
    print(f"Client 3 result: {result3}")

    # Test multiple clients concurrently
    print("\n--- Testing Multiple Clients (Concurrent) ---")

    def register_tensor(client_id: int) -> Tuple[int, ErrorCode]:
        c = Client(endpoint)
        result = c.register_tensor_shard(f"concurrent_tensor_{client_id}")
        return (client_id, result)

    num_concurrent_clients = 10
    print(f"Launching {num_concurrent_clients} concurrent clients...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent_clients) as executor:
        futures = [executor.submit(register_tensor, i) for i in range(num_concurrent_clients)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    results.sort(key=lambda x: x[0])  # Sort by client_id
    success_count = sum(1 for _, r in results if r == ErrorCode.SUCCESS)
    print(f"Results: {success_count}/{num_concurrent_clients} succeeded")
    for client_id, result in results:
        print(f"  Client {client_id}: {result}")

    # Stop NodeAgent
    print("\nStopping NodeAgent...")
    node_agent.stop()

    print("\n=== Test Complete ===")


if __name__ == "__main__":
    main()

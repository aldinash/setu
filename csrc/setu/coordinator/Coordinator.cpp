//==============================================================================
// Copyright (c) 2025 Vajra Team; Georgia Institute of Technology; Microsoft
// Corporation.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//==============================================================================
#include "coordinator/Coordinator.h"
//==============================================================================
#include "commons/Logging.h"
#include "commons/utils/SetuCommHelper.h"
//==============================================================================
namespace setu::coordinator {
//==============================================================================
using setu::commons::GenerateUUID;
using setu::commons::RequestId;
using setu::commons::enums::ErrorCode;
using setu::commons::messages::AllocateTensorRequest;
using setu::commons::messages::CoordinatorMessage;
using setu::commons::messages::NodeAgentRequest;
using setu::commons::messages::RegisterTensorShardResponse;
using setu::commons::messages::SubmitCopyResponse;
using setu::commons::utils::SetuCommHelper;
using setu::commons::utils::ZmqHelper;
//==============================================================================
constexpr std::chrono::milliseconds kHandleLoopSleepMs(10);
//==============================================================================
Coordinator::Coordinator(std::size_t router_executor_port,
                         std::size_t router_handler_port)
    : router_executor_port_(router_executor_port),
      router_handler_port_(router_handler_port) {
  InitZmqSockets();
}

Coordinator::~Coordinator() {
  Stop();
  CloseZmqSockets();
}

void Coordinator::Start() {
  LOG_DEBUG("Starting Coordinator");
  StartHandlerLoop();
  StartExecutorLoop();
}

void Coordinator::Stop() {
  LOG_DEBUG("Stopping Coordinator");
  StopHandlerLoop();
  StopExecutorLoop();
}

std::optional<TensorShardRef> Coordinator::RegisterTensorShard(
    const TensorShardSpec& shard_spec) {
  LOG_DEBUG("Registering tensor shard: {}", shard_spec.name);

  TensorShardRef shard_ref = metastore_.RegisterTensorShard(shard_spec);
  return shard_ref;
}

std::optional<CopyOperationId> Coordinator::SubmitCopy(
    const CopySpec& copy_spec) {
  LOG_DEBUG("Submitting copy operation from {} to {}", copy_spec.src_name,
            copy_spec.dst_name);

  // TODO: Implement copy submission and plan generation
  return std::nullopt;
}

void Coordinator::PlanExecuted(CopyOperationId copy_op_id) {
  LOG_DEBUG("Plan executed for copy operation ID: {}", copy_op_id);

  // TODO: Implement plan execution completion handling
}

void Coordinator::InitZmqSockets() {
  LOG_DEBUG("Initializing ZMQ sockets");

  zmq_context_ = std::make_shared<zmq::context_t>();

  node_agent_router_executor_socket_ = ZmqHelper::CreateAndBindSocket(
      zmq_context_, zmq::socket_type::router, router_executor_port_);
  node_agent_router_handler_socket_ = ZmqHelper::CreateAndBindSocket(
      zmq_context_, zmq::socket_type::router, router_handler_port_);

  LOG_DEBUG("Initialized ZMQ sockets successfully");
}

void Coordinator::CloseZmqSockets() {
  LOG_DEBUG("Closing ZMQ sockets");

  if (node_agent_router_executor_socket_)
    node_agent_router_executor_socket_->close();
  if (node_agent_router_handler_socket_)
    node_agent_router_handler_socket_->close();
  if (zmq_context_) zmq_context_->close();

  LOG_DEBUG("Closed ZMQ sockets successfully");
}

void Coordinator::StartHandlerLoop() {
  LOG_DEBUG("Starting handler loop");

  handler_thread_ = std::thread(SETU_LAUNCH_THREAD(
      [this]() { this->HandlerLoop(); }, "CoordinatorHandlerThread"));
}

void Coordinator::StopHandlerLoop() {
  LOG_DEBUG("Stopping handler loop");

  handler_running_ = false;

  if (handler_thread_.joinable()) {
    handler_thread_.join();
  }

  LOG_DEBUG("Handler loop stopped");
}

void Coordinator::StartExecutorLoop() {
  LOG_DEBUG("Starting executor loop");

  executor_thread_ = std::thread(SETU_LAUNCH_THREAD(
      [this]() { this->ExecutorLoop(); }, "CoordinatorExecutorThread"));
}

void Coordinator::StopExecutorLoop() {
  LOG_DEBUG("Stopping executor loop");

  executor_running_ = false;

  if (executor_thread_.joinable()) {
    executor_thread_.join();
  }

  LOG_DEBUG("Executor loop stopped");
}

void Coordinator::HandlerLoop() {
  LOG_DEBUG("Entering handler loop");

  handler_running_ = true;
  while (handler_running_) {
    auto [node_agent_identity, request] =
        SetuCommHelper::RecvWithIdentity<NodeAgentRequest, false>(
            node_agent_router_handler_socket_);
    std::visit(
        [&](const auto& msg) {
          using T = std::decay_t<decltype(msg)>;
          if constexpr (std::is_same_v<T, RegisterTensorShardRequest>) {
            HandleRegisterTensorShardRequest(node_agent_identity, msg);
          } else if constexpr (std::is_same_v<T, SubmitCopyRequest>) {
            HandleSubmitCopyRequest(node_agent_identity, msg);
          }
        },
        request);
  }
}

void Coordinator::HandleRegisterTensorShardRequest(
    const Identity& node_agent_identity,
    const RegisterTensorShardRequest& request) {
  LOG_INFO("Coordinator received RegisterTensorShardRequest for tensor: {}",
           request.tensor_shard_spec.name);

  // Register the tensor shard in the metastore
  TensorShardRef shard_ref =
      metastore_.RegisterTensorShard(request.tensor_shard_spec);

  // Send response
  RegisterTensorShardResponse response(request.request_id, ErrorCode::kSuccess,
                                       shard_ref);
  SetuCommHelper::SendWithIdentity<CoordinatorMessage, false>(
      node_agent_router_handler_socket_, node_agent_identity, response);

  // Check if all shards for this tensor are registered
  if (metastore_.AllShardsRegistered(request.tensor_shard_spec.name)) {
    LOG_INFO(
        "All shards registered for tensor: {}, sending AllocateTensorRequest",
        request.tensor_shard_spec.name);

    // Send AllocateTensorRequest to NodeAgent to allocate the tensor
    AllocateTensorRequest allocate_request(request.tensor_shard_spec.name);
    SetuCommHelper::SendWithIdentity<CoordinatorMessage, false>(
        node_agent_router_handler_socket_, node_agent_identity,
        allocate_request);
  }
}

void Coordinator::HandleSubmitCopyRequest(const Identity& node_agent_identity,
                                          const SubmitCopyRequest& request) {
  LOG_INFO("Coordinator received SubmitCopyRequest from {} to {}",
           request.copy_spec.src_name, request.copy_spec.dst_name);

  auto copy_key =
      std::make_pair(request.copy_spec.src_name, request.copy_spec.dst_name);

  // Check if this is the first request for this (src, dst) pair
  auto pending_it = pending_copy_specs_.find(copy_key);
  if (pending_it == pending_copy_specs_.end()) {
    // First request - store the CopySpec for validation
    pending_copy_specs_.emplace(copy_key, request.copy_spec);
    copies_received_[copy_key] = 1;
  } else {
    // Subsequent request - verify TensorSelections match
    const CopySpec& first_spec = pending_it->second;

    ASSERT_VALID_RUNTIME(
        *request.copy_spec.src_selection == *first_spec.src_selection,
        "SubmitCopy {} -> {}: source selection mismatch",
        request.copy_spec.src_name, request.copy_spec.dst_name);

    ASSERT_VALID_RUNTIME(
        *request.copy_spec.dst_selection == *first_spec.dst_selection,
        "SubmitCopy {} -> {}: destination selection mismatch",
        request.copy_spec.src_name, request.copy_spec.dst_name);

    copies_received_[copy_key]++;
  }

  // Track this client for later response
  pending_copy_clients_[copy_key].emplace_back(node_agent_identity,
                                               request.request_id);

  // Get the expected number of clients (number of shards for source tensor)
  std::size_t expected_clients =
      metastore_.GetNumShardsForTensor(request.copy_spec.src_name);

  LOG_DEBUG("SubmitCopy {} -> {}: received {}/{} requests",
            request.copy_spec.src_name, request.copy_spec.dst_name,
            copies_received_[copy_key], expected_clients);

  // Check if all clients have sent the request
  if (copies_received_[copy_key] == expected_clients) {
    // Generate CopyOperationId
    CopyOperationId copy_op_id = GenerateUUID();

    LOG_INFO(
        "All clients submitted copy request {} -> {}, "
        "copy_op_id={}, adding to planner queue",
        request.copy_spec.src_name, request.copy_spec.dst_name, copy_op_id);

    // Store the mapping
    copy_operations_.emplace(copy_op_id, request.copy_spec);

    // Add to planner queue
    planner_queue_.push(request.copy_spec);

    // Send responses to all waiting clients with copy_op_id
    for (const auto& [client_identity, client_request_id] :
         pending_copy_clients_[copy_key]) {
      SubmitCopyResponse response(client_request_id, copy_op_id,
                                  ErrorCode::kSuccess);
      SetuCommHelper::SendWithIdentity<CoordinatorMessage, false>(
          node_agent_router_handler_socket_, client_identity, response);
    }

    // Clean up maps
    copies_received_.erase(copy_key);
    pending_copy_specs_.erase(copy_key);
    pending_copy_clients_.erase(copy_key);
  }
}

void Coordinator::ExecutorLoop() {
  LOG_DEBUG("Entering executor loop");

  executor_running_ = true;
  while (executor_running_) {
    // TODO: Implement executor loop to dispatch plans to NodeAgents
    std::this_thread::sleep_for(kHandleLoopSleepMs);
  }
}
//==============================================================================
}  // namespace setu::coordinator
//==============================================================================

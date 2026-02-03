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
#include "node_manager/NodeAgent.h"
//==============================================================================
#include "commons/Logging.h"
#include "commons/utils/Comm.h"
#include "commons/utils/TorchTensorIPC.h"
//==============================================================================
namespace setu::node_manager {
//==============================================================================
using setu::commons::DeviceRank;
using setu::commons::RequestId;
using setu::commons::ShardId;
using setu::commons::TensorName;
using setu::commons::datatypes::Device;
using setu::commons::datatypes::TensorDim;
using setu::commons::datatypes::TensorDimMap;
using setu::commons::datatypes::TensorShardMetadata;
using setu::commons::datatypes::TensorShardMetadataMap;
using setu::commons::datatypes::TensorShardMetadataPtr;
using setu::commons::datatypes::TensorShardRef;
using setu::commons::enums::DeviceKind;
using setu::commons::enums::ErrorCode;
using setu::commons::messages::AllocateTensorRequest;
using setu::commons::messages::ClientRequest;
using setu::commons::messages::CoordinatorMessage;
using setu::commons::messages::CopyOperationFinishedRequest;
using setu::commons::messages::ExecuteProgramRequest;
using setu::commons::messages::ExecuteProgramResponse;
using setu::commons::messages::ExecuteRequest;
using setu::commons::messages::ExecuteResponse;
using setu::commons::messages::NodeAgentRequest;
using setu::commons::messages::RegisterTensorShardCoordinatorResponse;
using setu::commons::messages::RegisterTensorShardNodeAgentResponse;
using setu::commons::messages::RegisterTensorShardRequest;
using setu::commons::messages::SubmitCopyRequest;
using setu::commons::messages::SubmitCopyResponse;
using setu::commons::messages::WaitForCopyRequest;
using setu::commons::messages::WaitForCopyResponse;
using setu::commons::utils::Comm;
using setu::commons::utils::PrepareTensorIPCSpec;
using setu::commons::utils::ZmqHelper;
using setu::planner::Plan;
//==============================================================================
constexpr std::int32_t kPollTimeoutMs = 100;
//==============================================================================
// NodeAgent Implementation
//==============================================================================
NodeAgent::NodeAgent(NodeId node_id, std::size_t port,
                     std::string coordinator_endpoint,
                     const std::vector<Device>& devices)
    : node_id_(node_id),
      port_(port),
      coordinator_endpoint_(std::move(coordinator_endpoint)),
      devices_(devices),
      zmq_context_(std::make_shared<zmq::context_t>()) {
  handler_ = std::make_unique<Handler>(node_id_, zmq_context_, port_,
                                       coordinator_endpoint_, executor_queue_);
  executor_ = std::make_unique<Executor>(
      node_id_, zmq_context_, coordinator_endpoint_, devices_, executor_queue_);
}

NodeAgent::~NodeAgent() {
  Stop();
  if (zmq_context_) {
    zmq_context_->close();
  }
}

void NodeAgent::Start() {
  LOG_DEBUG("Starting NodeAgent");
  handler_->Start();
  executor_->Start();
}

void NodeAgent::Stop() {
  LOG_DEBUG("Stopping NodeAgent");

  executor_queue_.close();
  handler_->Stop();
  executor_->Stop();
}

std::optional<TensorShardRef> NodeAgent::RegisterTensorShard(
    const TensorShardSpec& shard_spec) {
  LOG_DEBUG("Registering tensor shard: {}", shard_spec.name);

  // TODO: Implement
  return std::nullopt;
}

std::optional<CopyOperationId> NodeAgent::SubmitCopy(
    const CopySpec& copy_spec) {
  LOG_DEBUG("Submitting copy operation from {} to {}", copy_spec.src_name,
            copy_spec.dst_name);

  // TODO: Implement copy submission
  return std::nullopt;
}

void NodeAgent::WaitForCopy(CopyOperationId copy_op_id) {
  LOG_DEBUG("Waiting for copy operation ID: {}", copy_op_id);

  // TODO: Implement wait for copy
}

void NodeAgent::CopyOperationFinished(CopyOperationId copy_op_id) {
  LOG_DEBUG("Marking copy operation ID: {} as finished", copy_op_id);
}

void NodeAgent::Execute(Plan plan) {
  LOG_DEBUG("Executing Plan {}", plan.ToString());
}

//==============================================================================
// Handler Implementation
//==============================================================================
NodeAgent::Handler::Handler(
    NodeId node_id, std::shared_ptr<zmq::context_t> zmq_context,
    std::size_t port, const std::string& coordinator_endpoint,
    Queue<std::pair<CopyOperationId, Plan>>& executor_queue)
    : node_id_(node_id),
      zmq_context_(zmq_context),
      port_(port),
      coordinator_endpoint_(coordinator_endpoint),
      executor_queue_(executor_queue) {
  InitSockets();
}

NodeAgent::Handler::~Handler() {
  Stop();
  CloseSockets();
}

void NodeAgent::Handler::InitSockets() {
  LOG_DEBUG("Handler: Initializing ZMQ sockets");

  client_socket_ = ZmqHelper::CreateAndBindSocket(
      zmq_context_, zmq::socket_type::router, port_);

  Identity identity = to_string(node_id_);
  coordinator_socket_ = ZmqHelper::CreateAndConnectSocket(
      zmq_context_, zmq::socket_type::dealer, coordinator_endpoint_, identity);

  LOG_DEBUG("Handler: Initialized ZMQ sockets with identity={}", identity);
}

void NodeAgent::Handler::CloseSockets() {
  LOG_DEBUG("Handler: Closing ZMQ sockets");

  if (client_socket_) {
    client_socket_->close();
  }
  if (coordinator_socket_) {
    coordinator_socket_->close();
  }

  LOG_DEBUG("Handler: Closed ZMQ sockets successfully");
}

void NodeAgent::Handler::Start() {
  if (running_.load()) {
    return;
  }
  LOG_DEBUG("Starting handler loop");
  thread_ = std::thread(
      SETU_LAUNCH_THREAD([this]() { this->Loop(); }, "HandlerLoopThread"));
}

void NodeAgent::Handler::Stop() {
  LOG_DEBUG("Stopping handler loop");
  running_ = false;

  if (thread_.joinable()) {
    thread_.join();
  }
  LOG_DEBUG("Handler loop stopped");
}

void NodeAgent::Handler::Loop() {
  LOG_DEBUG("Entering handler loop");

  running_ = true;
  while (running_) {
    auto ready = Comm::PollForRead({client_socket_, coordinator_socket_},
                                   kPollTimeoutMs);

    for (const auto& socket : ready) {
      if (socket == client_socket_) {
        auto [identity, request] =
            Comm::RecvWithIdentity<ClientRequest>(socket);
        HandleClientMessage(identity, request);
      } else if (socket == coordinator_socket_) {
        auto message = Comm::Recv<CoordinatorMessage>(socket);
        HandleCoordinatorMessage(message);
      }
    }
  }
}

void NodeAgent::Handler::HandleClientMessage(const Identity& client_identity,
                                             const ClientRequest& request) {
  std::visit(
      [&](const auto& msg) {
        using T = std::decay_t<decltype(msg)>;
        if constexpr (std::is_same_v<T, RegisterTensorShardRequest>) {
          HandleRegisterTensorShardRequest(client_identity, msg);
        } else if constexpr (std::is_same_v<T, SubmitCopyRequest>) {
          HandleSubmitCopyRequest(client_identity, msg);
        } else if constexpr (std::is_same_v<T, WaitForCopyRequest>) {
          HandleWaitForCopyRequest(client_identity, msg);
        } else if constexpr (std::is_same_v<T, GetReadHandleRequest>) {
          HandleGetReadHandleRequest(client_identity, msg);
        } else if constexpr (std::is_same_v<T, ReleaseReadHandleRequest>) {
          HandleReleaseReadHandleRequest(client_identity, msg);
        } else if constexpr (std::is_same_v<T, GetWriteHandleRequest>) {
          HandleGetWriteHandleRequest(client_identity, msg);
        } else if constexpr (std::is_same_v<T, ReleaseWriteHandleRequest>) {
          HandleReleaseWriteHandleRequest(client_identity, msg);
        }
      },
      request);
}

void NodeAgent::Handler::HandleCoordinatorMessage(
    const CoordinatorMessage& message) {
  std::visit(
      [&](const auto& msg) {
        using T = std::decay_t<decltype(msg)>;
        if constexpr (std::is_same_v<T, AllocateTensorRequest>) {
          HandleAllocateTensorRequest(msg);
        } else if constexpr (std::is_same_v<T, CopyOperationFinishedRequest>) {
          HandleCopyOperationFinishedRequest(msg);
        } else if constexpr (std::is_same_v<T, ExecuteRequest>) {
          HandleExecuteRequest(msg);
        } else if constexpr (std::is_same_v<
                                 T, RegisterTensorShardCoordinatorResponse>) {
          HandleRegisterTensorShardCoordinatorResponse(msg);
        } else if constexpr (std::is_same_v<T, SubmitCopyResponse>) {
          HandleSubmitCopyResponse(msg);
        } else if constexpr (std::is_same_v<T, WaitForCopyResponse>) {
          HandleWaitForCopyResponse(msg);
        }
      },
      message);
}

void NodeAgent::Handler::HandleRegisterTensorShardRequest(
    const Identity& client_identity,
    const RegisterTensorShardRequest& request) {
  LOG_DEBUG("Handling RegisterTensorShardRequest for tensor: {}",
            request.tensor_shard_spec.name);

  request_id_to_client_identity_[request.request_id] = client_identity;

  Comm::Send<NodeAgentRequest>(coordinator_socket_, request);
}

void NodeAgent::Handler::HandleSubmitCopyRequest(
    const Identity& client_identity, const SubmitCopyRequest& request) {
  LOG_DEBUG("Handling SubmitCopyRequest from {} to {}",
            request.copy_spec.src_name, request.copy_spec.dst_name);

  request_id_to_client_identity_[request.request_id] = client_identity;

  Comm::Send<NodeAgentRequest>(coordinator_socket_, request);
}

void NodeAgent::Handler::HandleWaitForCopyRequest(
    const Identity& client_identity, const WaitForCopyRequest& request) {
  LOG_DEBUG("Handling WaitForCopyRequest for copy operation ID: {}",
            request.copy_operation_id);

  pending_waits_[request.copy_operation_id].push_back(client_identity);

  WaitForCopyResponse response(RequestId{}, ErrorCode::kSuccess);
}

void NodeAgent::Handler::HandleGetReadHandleRequest(
    const Identity& client_identity, const GetReadHandleRequest& request) {
  LOG_DEBUG("Handling GetReadHandleRequest for shard: {} from client: {}",
            request.shard_id, request.client_id);

  TensorShardPtr shard;
  bool found = tensor_shards_.visit(
      request.shard_id, [&shard](const auto& entry) { shard = entry.second; });

  if (!found) {
    LOG_ERROR("Shard not found: {}", request.shard_id);
    GetReadHandleResponse response(request.request_id,
                                   ErrorCode::kTensorNotFound);
    Comm::SendWithIdentity<GetReadHandleResponse>(client_socket_,
                                                  client_identity, response);
    return;
  }

  // Create read handle and store it (acquires shared lock)
  auto read_handle = std::make_shared<TensorShardReadHandle>(shard);
  active_read_locks_[request.client_id][request.shard_id] = read_handle;

  auto tensor_ipc_spec = PrepareTensorIPCSpec(shard->GetTensor());
  GetReadHandleResponse response(request.request_id, ErrorCode::kSuccess,
                                 std::move(tensor_ipc_spec));
  Comm::SendWithIdentity<GetReadHandleResponse>(client_socket_, client_identity,
                                                response);

  LOG_DEBUG("Sent read handle response for shard: {}", request.shard_id);
}

void NodeAgent::Handler::HandleReleaseReadHandleRequest(
    const Identity& client_identity, const ReleaseReadHandleRequest& request) {
  LOG_DEBUG("Handling ReleaseReadHandleRequest for shard: {} from client: {}",
            request.shard_id, request.client_id);

  auto client_it = active_read_locks_.find(request.client_id);
  if (client_it == active_read_locks_.end()) {
    LOG_ERROR("No read locks found for client: {}", request.client_id);
    ReleaseReadHandleResponse response(request.request_id,
                                       ErrorCode::kInternalError);
    Comm::SendWithIdentity<ReleaseReadHandleResponse>(
        client_socket_, client_identity, response);
    return;
  }

  auto shard_it = client_it->second.find(request.shard_id);
  if (shard_it == client_it->second.end()) {
    LOG_ERROR("No read lock found for shard: {} from client: {}",
              request.shard_id, request.client_id);
    ReleaseReadHandleResponse response(request.request_id,
                                       ErrorCode::kInternalError);
    Comm::SendWithIdentity<ReleaseReadHandleResponse>(
        client_socket_, client_identity, response);
    return;
  }

  // Remove the handle (releases shared lock)
  client_it->second.erase(shard_it);
  if (client_it->second.empty()) {
    active_read_locks_.erase(client_it);
  }

  ReleaseReadHandleResponse response(request.request_id, ErrorCode::kSuccess);
  Comm::SendWithIdentity<ReleaseReadHandleResponse>(client_socket_,
                                                    client_identity, response);

  LOG_DEBUG("Released read handle for shard: {}", request.shard_id);
}

void NodeAgent::Handler::HandleGetWriteHandleRequest(
    const Identity& client_identity, const GetWriteHandleRequest& request) {
  LOG_DEBUG("Handling GetWriteHandleRequest for shard: {} from client: {}",
            request.shard_id, request.client_id);

  TensorShardPtr shard;
  bool found = tensor_shards_.visit(
      request.shard_id, [&shard](const auto& entry) { shard = entry.second; });

  if (!found) {
    LOG_ERROR("Shard not found: {}", request.shard_id);
    GetWriteHandleResponse response(request.request_id,
                                    ErrorCode::kTensorNotFound);
    Comm::SendWithIdentity<GetWriteHandleResponse>(client_socket_,
                                                   client_identity, response);
    return;
  }

  // Create write handle and store it (acquires exclusive lock)
  auto write_handle = std::make_shared<TensorShardWriteHandle>(shard);
  active_write_locks_[request.client_id][request.shard_id] = write_handle;

  auto tensor_ipc_spec = PrepareTensorIPCSpec(shard->GetTensor());
  GetWriteHandleResponse response(request.request_id, ErrorCode::kSuccess,
                                  std::move(tensor_ipc_spec));
  Comm::SendWithIdentity<GetWriteHandleResponse>(client_socket_,
                                                 client_identity, response);

  LOG_DEBUG("Sent write handle response for shard: {}", request.shard_id);
}

void NodeAgent::Handler::HandleReleaseWriteHandleRequest(
    const Identity& client_identity, const ReleaseWriteHandleRequest& request) {
  LOG_DEBUG("Handling ReleaseWriteHandleRequest for shard: {} from client: {}",
            request.shard_id, request.client_id);

  auto client_it = active_write_locks_.find(request.client_id);
  if (client_it == active_write_locks_.end()) {
    LOG_ERROR("No write locks found for client: {}", request.client_id);
    ReleaseWriteHandleResponse response(request.request_id,
                                        ErrorCode::kInternalError);
    Comm::SendWithIdentity<ReleaseWriteHandleResponse>(
        client_socket_, client_identity, response);
    return;
  }

  auto shard_it = client_it->second.find(request.shard_id);
  if (shard_it == client_it->second.end()) {
    LOG_ERROR("No write lock found for shard: {} from client: {}",
              request.shard_id, request.client_id);
    ReleaseWriteHandleResponse response(request.request_id,
                                        ErrorCode::kInternalError);
    Comm::SendWithIdentity<ReleaseWriteHandleResponse>(
        client_socket_, client_identity, response);
    return;
  }

  // Remove the handle (releases exclusive lock)
  client_it->second.erase(shard_it);
  if (client_it->second.empty()) {
    active_write_locks_.erase(client_it);
  }

  ReleaseWriteHandleResponse response(request.request_id, ErrorCode::kSuccess);
  Comm::SendWithIdentity<ReleaseWriteHandleResponse>(client_socket_,
                                                     client_identity, response);

  LOG_DEBUG("Released write handle for shard: {}", request.shard_id);
}

void NodeAgent::Handler::HandleAllocateTensorRequest(
    const AllocateTensorRequest& request) {
  LOG_DEBUG("Handling AllocateTensorRequest for request: {}", request);

  for (const auto& shard_id : request.shard_ids) {
    auto it = tensor_shard_metadata_map_.find(shard_id);
    ASSERT_VALID_RUNTIME(it != tensor_shard_metadata_map_.end(),
                         "No metadata found for shard: {}", shard_id);

    AllocateTensor(*it->second);
  }
}

void NodeAgent::Handler::HandleCopyOperationFinishedRequest(
    const CopyOperationFinishedRequest& request) {
  LOG_DEBUG("Handling CopyOperationFinishedRequest for request: {}", request);

  // Get and remove all clients waiting for this copy operation
  auto it = pending_waits_.find(request.copy_operation_id);
  if (it != pending_waits_.end()) {
    for (const auto& client_id : it->second) {
      WaitForCopyResponse response(RequestId{}, ErrorCode::kSuccess);

      // unblock waiting clients
      Comm::SendWithIdentity<WaitForCopyResponse>(client_socket_, client_id,
                                                  response);
    }
    pending_waits_.erase(it);
  }
}

void NodeAgent::Handler::HandleExecuteRequest(const ExecuteRequest& request) {
  LOG_DEBUG("Handling ExecuteRequest for request: {}", request);

  // Put (copy_op_id, node_plan) into executor queue
  executor_queue_.push(std::make_pair(request.copy_op_id, request.node_plan));
}

void NodeAgent::Handler::HandleRegisterTensorShardCoordinatorResponse(
    const RegisterTensorShardCoordinatorResponse& response) {
  auto it = request_id_to_client_identity_.find(response.request_id);
  if (it == request_id_to_client_identity_.end()) {
    LOG_WARNING(
        "Received RegisterTensorShardCoordinatorResponse for unknown "
        "request_id: {}, ignoring",
        response.request_id);
    return;
  }
  const auto& client_identity = it->second;

  // Reconstruct TensorShardRef from TensorShardMetadata
  std::optional<TensorShardRef> shard_ref;
  if (response.shard_metadata.has_value()) {
    const auto& metadata = response.shard_metadata.value();

    // Store the metadata for later allocation
    auto metadata_ptr = std::make_shared<TensorShardMetadata>(metadata);
    tensor_shard_metadata_map_.emplace(metadata.id, metadata_ptr);

    // Build TensorDimMap from the spec's dims
    TensorDimMap dims;
    for (const auto& dim_spec : metadata.spec.dims) {
      dims.emplace(dim_spec.name, TensorDim(dim_spec.name, dim_spec.size));
    }

    shard_ref.emplace(metadata.spec.name, metadata.id, std::move(dims));
  }

  // Send RegisterTensorShardNodeAgentResponse to client
  RegisterTensorShardNodeAgentResponse client_response(
      response.request_id, response.error_code, std::move(shard_ref));
  Comm::SendWithIdentity<RegisterTensorShardNodeAgentResponse>(
      client_socket_, client_identity, client_response);

  request_id_to_client_identity_.erase(it);
}

void NodeAgent::Handler::HandleSubmitCopyResponse(
    const SubmitCopyResponse& response) {
  auto it = request_id_to_client_identity_.find(response.request_id);
  if (it == request_id_to_client_identity_.end()) {
    LOG_WARNING(
        "Received SubmitCopyResponse for unknown request_id: {}, ignoring",
        response.request_id);
    return;
  }
  const auto& client_identity = it->second;

  Comm::SendWithIdentity<SubmitCopyResponse>(client_socket_, client_identity,
                                             response);

  request_id_to_client_identity_.erase(it);
}

void NodeAgent::Handler::HandleWaitForCopyResponse(
    const WaitForCopyResponse& response) {
  auto it = request_id_to_client_identity_.find(response.request_id);
  if (it == request_id_to_client_identity_.end()) {
    LOG_WARNING(
        "Received WaitForCopyResponse for unknown request_id: {}, ignoring",
        response.request_id);
    return;
  }
  const auto& client_identity = it->second;

  Comm::SendWithIdentity<WaitForCopyResponse>(client_socket_, client_identity,
                                              response);

  request_id_to_client_identity_.erase(it);
}

void NodeAgent::Handler::AllocateTensor(
    const TensorShardMetadata& shard_metadata) {
  LOG_DEBUG("Allocating tensor shard: shard_metadata={}", shard_metadata);

  const auto& spec = shard_metadata.spec;

  // Build the shape from dims (using owned size for each dimension)
  std::vector<std::int64_t> shape;
  shape.reserve(spec.dims.size());
  for (const auto& dim_spec : spec.dims) {
    shape.push_back(static_cast<std::int64_t>(dim_spec.GetOwnedSize()));
  }

  // Create tensor options with dtype and device from spec
  auto options =
      torch::TensorOptions().dtype(spec.dtype).device(spec.device.torch_device);

  torch::Tensor tensor = torch::empty(shape, options);

  // Create TensorShard with metadata and tensor, then store it
  auto shard = std::make_shared<TensorShard>(shard_metadata, std::move(tensor));
  tensor_shards_.insert({shard_metadata.id, std::move(shard)});

  LOG_DEBUG("Successfully allocated shard {} with shape {} on device {}",
            shard_metadata.id, shape, spec.device.torch_device.str());
}

//==============================================================================
// Executor Implementation
//==============================================================================
NodeAgent::Executor::Executor(
    NodeId node_id, std::shared_ptr<zmq::context_t> zmq_context,
    const std::string& coordinator_endpoint, const std::vector<Device>& devices,
    Queue<std::pair<CopyOperationId, Plan>>& executor_queue)
    : node_id_(node_id),
      zmq_context_(zmq_context),
      coordinator_endpoint_(coordinator_endpoint),
      devices_(devices),
      executor_queue_(executor_queue) {
  InitSockets();
}

NodeAgent::Executor::~Executor() {
  Stop();
  CloseSockets();
}

void NodeAgent::Executor::InitSockets() {
  LOG_DEBUG("Executor: Initializing ZMQ sockets");

  Identity identity = to_string(node_id_);
  coordinator_socket_ = ZmqHelper::CreateAndConnectSocket(
      zmq_context_, zmq::socket_type::dealer, coordinator_endpoint_, identity);

  // TODO: Initialize worker sockets based on devices
  LOG_DEBUG("Executor: devices={}", devices_);

  LOG_DEBUG("Executor: Initialized ZMQ sockets with identity={}", identity);
}

void NodeAgent::Executor::CloseSockets() {
  LOG_DEBUG("Executor: Closing ZMQ sockets");

  // Close worker REQ sockets
  for (auto& [device_rank, socket] : worker_sockets_) {
    if (socket) {
      socket->close();
    }
  }
  worker_sockets_.clear();

  if (coordinator_socket_) {
    coordinator_socket_->close();
  }

  LOG_DEBUG("Executor: Closed ZMQ sockets successfully");
}

void NodeAgent::Executor::Start() {
  if (running_.load()) {
    return;
  }
  LOG_DEBUG("Starting executor loop");
  thread_ = std::thread(
      SETU_LAUNCH_THREAD([this]() { this->Loop(); }, "ExecutorLoopThread"));
}

void NodeAgent::Executor::Stop() {
  LOG_DEBUG("Stopping executor loop");
  running_ = false;

  if (thread_.joinable()) {
    thread_.join();
  }
  LOG_DEBUG("Executor loop stopped");
}

void NodeAgent::Executor::Loop() {
  LOG_DEBUG("Entering executor loop");

  running_ = true;
  while (running_) {
    // Block until we receive a (copy_op_id, plan) pair from the queue
    try {
      auto [copy_op_id, plan] = executor_queue_.pull();

      LOG_DEBUG("Executor received plan for copy_op_id: {}", copy_op_id);

      // For each worker program in the plan, send it to the corresponding
      // worker
      for (const auto& [participant, program] : plan.program) {
        // Ensure worker is ready before sending
        auto device_rank = participant.LocalDeviceIndex();
        auto it = worker_sockets_.find(device_rank);
        ASSERT_VALID_RUNTIME(it != worker_sockets_.end(),
                             "No socket found for device_rank: {}",
                             device_rank);

        // Send ExecuteProgramRequest to worker
        LOG_DEBUG("Sending program with {} instructions to worker {}",
                  program.size(), device_rank);
        ExecuteProgramRequest request(program);
        Comm::Send(it->second, request);

        // Wait for acknowledgment from worker
        auto response = Comm::Recv<ExecuteProgramResponse>(it->second);
        LOG_DEBUG("Received acknowledgment from worker {}: {}", device_rank,
                  response);
      }

      LOG_DEBUG("All workers completed execution for copy_op_id: {}",
                copy_op_id);

      // Notify coordinator that execution is complete
      ExecuteResponse response(RequestId{}, ErrorCode::kSuccess);
      Comm::Send(coordinator_socket_, response);
    } catch (const boost::concurrent::sync_queue_is_closed&) {
      LOG_DEBUG("Executor: executor_queue_ closed, exiting");
      return;
    }
  }
}
//==============================================================================
}  // namespace setu::node_manager
//==============================================================================

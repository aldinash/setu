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
#include "client/Client.h"
//==============================================================================
#include "commons/BoostCommon.h"
#include "commons/Logging.h"
#include "commons/messages/Messages.h"
#include "commons/utils/Comm.h"
#include "commons/utils/ZmqHelper.h"
//==============================================================================
namespace setu::client {
//==============================================================================
using setu::commons::GenerateUUID;
using setu::commons::datatypes::TensorShardRefPtr;
using setu::commons::messages::ClientRequest;
using setu::commons::messages::GetReadHandleRequest;
using setu::commons::messages::GetReadHandleResponse;
using setu::commons::messages::GetWriteHandleRequest;
using setu::commons::messages::GetWriteHandleResponse;
using setu::commons::messages::RegisterTensorShardNodeAgentResponse;
using setu::commons::messages::RegisterTensorShardRequest;
using setu::commons::messages::ReleaseReadHandleRequest;
using setu::commons::messages::ReleaseReadHandleResponse;
using setu::commons::messages::ReleaseWriteHandleRequest;
using setu::commons::messages::ReleaseWriteHandleResponse;
using setu::commons::messages::SubmitCopyRequest;
using setu::commons::messages::SubmitCopyResponse;
using setu::commons::messages::WaitForCopyRequest;
using setu::commons::messages::WaitForCopyResponse;
using setu::commons::utils::Comm;
using setu::commons::utils::ZmqHelper;
//==============================================================================
Client::Client() : client_id_(GenerateUUID()) {
  zmq_context_ = std::make_shared<zmq::context_t>();
  LOG_DEBUG("Client created with ID: {}", client_id_);
}

Client::~Client() {
  if (is_connected_) {
    Disconnect();
  }
  if (zmq_context_) {
    zmq_context_->close();
  }
}

void Client::Connect(const std::string& endpoint) {
  ASSERT_VALID_ARGUMENTS(!is_connected_,
                         "Client is already connected to {}. Disconnect first.",
                         endpoint_);
  ASSERT_VALID_ARGUMENTS(!endpoint.empty(), "Endpoint cannot be empty");

  LOG_DEBUG("Client connecting to {}", endpoint);

  request_socket_ = ZmqHelper::CreateAndConnectSocket(
      zmq_context_, zmq::socket_type::req, endpoint, to_string(client_id_));

  endpoint_ = endpoint;
  is_connected_ = true;

  LOG_DEBUG("Client connected to {} successfully", endpoint_);
}

void Client::Disconnect() {
  ASSERT_VALID_RUNTIME(is_connected_, "Client is not connected");

  LOG_DEBUG("Client disconnecting from {}", endpoint_);

  if (request_socket_) {
    request_socket_->close();
    request_socket_.reset();
  }

  endpoint_.clear();
  is_connected_ = false;

  LOG_DEBUG("Client disconnected successfully");
}

bool Client::IsConnected() const { return is_connected_; }

const std::string& Client::GetEndpoint() const { return endpoint_; }

std::optional<TensorShardRef> Client::RegisterTensorShard(
    const TensorShardSpec& shard_spec) {
  LOG_DEBUG("Client registering tensor shard: {}", shard_spec.name);

  ClientRequest request = RegisterTensorShardRequest(shard_spec);
  Comm::Send(request_socket_, request);

  auto response =
      Comm::Recv<RegisterTensorShardNodeAgentResponse>(request_socket_);

  LOG_DEBUG("Client received response for tensor shard: {} with error code: {}",
            shard_spec.name, response.error_code);

  if (response.error_code != ErrorCode::kSuccess) {
    return std::nullopt;
  }

  if (!response.shard_ref.has_value()) {
    LOG_ERROR("Client receieved success response but shard_ref is missing {}",
              shard_spec.name);
    return std::nullopt;
  }

  client_shards_.push_back(
      std::make_shared<TensorShardRef>(response.shard_ref.value()));

  return response.shard_ref;
}

std::optional<CopyOperationId> Client::SubmitCopy(const CopySpec& copy_spec) {
  LOG_DEBUG("Client submitting copy operation from {} to {}",
            copy_spec.src_name, copy_spec.dst_name);

  ClientRequest request = SubmitCopyRequest(copy_spec);
  Comm::Send(request_socket_, request);

  auto response = Comm::Recv<SubmitCopyResponse>(request_socket_);

  LOG_DEBUG("Client received copy operation ID: {}",
            response.copy_operation_id);

  if (response.error_code != ErrorCode::kSuccess) {
    return std::nullopt;
  }

  return response.copy_operation_id;
}

void Client::WaitForCopy(CopyOperationId copy_op_id) {
  LOG_DEBUG("Client waiting for copy operation ID: {}", copy_op_id);

  ClientRequest request = WaitForCopyRequest(copy_op_id);
  Comm::Send(request_socket_, request);

  auto response = Comm::Recv<WaitForCopyResponse>(request_socket_);

  LOG_DEBUG(
      "Client finished waiting for copy operation ID: {} with error code: {}",
      copy_op_id, response.error_code);
}

TensorIPCSpec Client::GetReadHandle(const TensorShardRef& shard_ref) {
  LOG_DEBUG("Client requesting read handle for shard: {}", shard_ref.shard_id);

  ClientRequest request = GetReadHandleRequest(client_id_, shard_ref.shard_id);
  Comm::Send(request_socket_, request);

  auto response = Comm::Recv<GetReadHandleResponse>(request_socket_);

  LOG_DEBUG(
      "Client received read handle response for shard: {} with error code: {}",
      shard_ref.shard_id, response.error_code);

  ASSERT_VALID_RUNTIME(response.error_code == ErrorCode::kSuccess,
                       "Failed to get read handle for shard {}: {}",
                       shard_ref.shard_id, response.error_code);
  ASSERT_VALID_RUNTIME(response.tensor_ipc_spec.has_value(),
                       "Tensor IPC spec is missing for shard {}",
                       shard_ref.shard_id);

  return response.tensor_ipc_spec.value();
}

void Client::ReleaseReadHandle(const TensorShardRef& shard_ref) {
  LOG_DEBUG("Client releasing read handle for shard: {}", shard_ref.shard_id);

  ClientRequest request =
      ReleaseReadHandleRequest(client_id_, shard_ref.shard_id);
  Comm::Send(request_socket_, request);

  auto response = Comm::Recv<ReleaseReadHandleResponse>(request_socket_);

  LOG_DEBUG("Client released read handle for shard: {} with error code: {}",
            shard_ref.shard_id, response.error_code);

  ASSERT_VALID_RUNTIME(response.error_code == ErrorCode::kSuccess,
                       "Failed to release read handle for shard {}: {}",
                       shard_ref.shard_id, response.error_code);
}

TensorIPCSpec Client::GetWriteHandle(const TensorShardRef& shard_ref) {
  LOG_DEBUG("Client requesting write handle for shard: {}", shard_ref.shard_id);

  ClientRequest request = GetWriteHandleRequest(client_id_, shard_ref.shard_id);
  Comm::Send(request_socket_, request);

  auto response = Comm::Recv<GetWriteHandleResponse>(request_socket_);

  LOG_DEBUG(
      "Client received write handle response for shard: {} with error code: {}",
      shard_ref.shard_id, response.error_code);

  ASSERT_VALID_RUNTIME(response.error_code == ErrorCode::kSuccess,
                       "Failed to get write handle for shard {}: {}",
                       shard_ref.shard_id, response.error_code);
  ASSERT_VALID_RUNTIME(response.tensor_ipc_spec.has_value(),
                       "Tensor IPC spec is missing for shard {}",
                       shard_ref.shard_id);

  return response.tensor_ipc_spec.value();
}

void Client::ReleaseWriteHandle(const TensorShardRef& shard_ref) {
  LOG_DEBUG("Client releasing write handle for shard: {}", shard_ref.shard_id);

  ClientRequest request =
      ReleaseWriteHandleRequest(client_id_, shard_ref.shard_id);
  Comm::Send(request_socket_, request);

  auto response = Comm::Recv<ReleaseWriteHandleResponse>(request_socket_);

  LOG_DEBUG("Client released write handle for shard: {} with error code: {}",
            shard_ref.shard_id, response.error_code);

  ASSERT_VALID_RUNTIME(response.error_code == ErrorCode::kSuccess,
                       "Failed to release write handle for shard {}: {}",
                       shard_ref.shard_id, response.error_code);
}

const std::vector<TensorShardRefPtr>& Client::GetShards() const {
  return client_shards_;
}

const ClientId& Client::GetClientId() const { return client_id_; }
//==============================================================================
}  // namespace setu::client
//==============================================================================

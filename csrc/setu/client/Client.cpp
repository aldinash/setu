#include "client/Client.h"

#include "commons/Logging.h"
#include "commons/messages/MessagesHelper.cpp"

namespace setu::client {
using setu::commons::enums::MsgType;
using setu::commons::messages::MessagesHelper;
using setu::commons::messages::RegisterTensorShardRequest;
using setu::commons::messages::RegisterTensorShardResponse;
using setu::commons::utils::ZmqHelper;

Client::Client(const std::string& endpoint) : endpoint_(endpoint) {
  InitZmqSockets();
}

Client::~Client() { CloseZmqSockets(); }

ErrorCode Client::RegisterTensorShard(const std::string& tensor_name) {
  LOG_DEBUG("Client registering tensor shard: {}", tensor_name);

  RegisterTensorShardRequest request(tensor_name);
  MessagesHelper::SendRequest(request_socket_,
                              MsgType::kRegisterTensorShardRequest, request);

  auto response_msg = MessagesHelper::RecvResponse(request_socket_);
  setu::commons::utils::BinaryRange range(response_msg.body.begin(),
                                          response_msg.body.end());
  auto response = RegisterTensorShardResponse::Deserialize(range);

  LOG_DEBUG("Client received response for tensor shard: {} with error code: {}",
            tensor_name, response.error_code);

  return response.error_code;
}

void Client::InitZmqSockets() {
  LOG_DEBUG("Client initializing ZMQ sockets");

  zmq_context_ = std::make_shared<zmq::context_t>();

  request_socket_ = ZmqHelper::CreateAndConnectSocket(
      zmq_context_, zmq::socket_type::req, endpoint_);

  LOG_DEBUG("Client initialized ZMQ sockets successfully");
}

void Client::CloseZmqSockets() {
  LOG_DEBUG("Client closing ZMQ sockets");

  if (request_socket_) request_socket_->close();

  if (zmq_context_) zmq_context_->close();

  LOG_DEBUG("Client closed ZMQ sockets successfully");
}
}  // namespace setu::client
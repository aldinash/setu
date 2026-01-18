#include "node_manager/NodeAgent.h"

#include "commons/Logging.h"
#include "commons/messages/MessagesHelper.cpp"

namespace setu::node_manager {
using setu::commons::enums::ErrorCode;
using setu::commons::enums::MsgType;
using setu::commons::messages::MessagesHelper;
using setu::commons::messages::ReceivedRouterMessage;
using setu::commons::messages::RegisterTensorShardResponse;
using setu::commons::utils::ZmqHelper;
//==============================================================================
constexpr std::chrono::milliseconds kHandleLoopSleepMs(10);
//==============================================================================
NodeAgent::NodeAgent(std::size_t router_port, std::size_t dealer_executor_port,
                     std::size_t dealer_handler_port)
    : router_port_(router_port),
      dealer_executor_port_(dealer_executor_port),
      dealer_handler_port_(dealer_handler_port) {
  InitZmqSockets();
}

NodeAgent::~NodeAgent() {
  Stop();
  CloseZmqSockets();
}

void NodeAgent::Start() {
  LOG_DEBUG("Starting NodeAgent");
  StartHandlerLoop();
}

void NodeAgent::Stop() {
  LOG_DEBUG("Stopping NodeAgent");
  StopHandlerLoop();
}

void NodeAgent::RegisterTensorShard(const TensorName& tensor_name) {
  LOG_DEBUG("Registering tensor shard: {}", tensor_name);
}

void NodeAgent::InitZmqSockets() {
  LOG_DEBUG("Initializing ZMQ sockets");

  zmq_context_ = std::make_shared<zmq::context_t>();

  client_router_socket_ = ZmqHelper::CreateAndBindSocket(
      zmq_context_, zmq::socket_type::router, router_port_);

  coordinator_dealer_executor_socket_ = ZmqHelper::CreateAndBindSocket(
      zmq_context_, zmq::socket_type::dealer, dealer_executor_port_);
  coordinator_dealer_handler_socket_ = ZmqHelper::CreateAndBindSocket(
      zmq_context_, zmq::socket_type::dealer, dealer_handler_port_);

  LOG_DEBUG("Initialized ZMQ sockets successfully");
}

void NodeAgent::CloseZmqSockets() {
  LOG_DEBUG("Closing ZMQ sockets");

  if (client_router_socket_) client_router_socket_->close();
  if (coordinator_dealer_executor_socket_)
    coordinator_dealer_executor_socket_->close();
  if (coordinator_dealer_handler_socket_)
    coordinator_dealer_handler_socket_->close();
  if (zmq_context_) zmq_context_->close();

  LOG_DEBUG("Closed ZMQ sockets successfully");
}

void NodeAgent::InitializeThreads() {
  LOG_DEBUG("Initializing worker threads");

  // Implementation for initializing worker threads
}

void NodeAgent::StopThreads() {
  LOG_DEBUG("Stopping worker threads");

  // Implementation for stopping worker threads
}

void NodeAgent::StartHandlerLoop() {
  LOG_DEBUG("Starting handler loop");

  handler_thread_ = std::thread(SETU_LAUNCH_THREAD(
      [this]() { this->HandlerLoop(); }, "HandlerLoopThread"));
}

void NodeAgent::StopHandlerLoop() {
  LOG_DEBUG("Stopping handler loop");

  handler_running_ = false;

  if (handler_thread_.joinable()) {
    handler_thread_.join();
  }

  LOG_DEBUG("Handler loop stopped");
}

void NodeAgent::StartExecutorLoop() {
  LOG_DEBUG("Starting executor loop");

  executor_thread_ = std::thread(SETU_LAUNCH_THREAD(
      [this]() { this->ExecutorLoop(); }, "ExecutorLoopThread"));
}

void NodeAgent::StopExecutorLoop() {
  LOG_DEBUG("Stopping executor loop");

  executor_running_ = false;

  if (executor_thread_.joinable()) {
    executor_thread_.join();
  }

  LOG_DEBUG("Executor loop stopped");
}

void NodeAgent::HandlerLoop() {
  LOG_DEBUG("Entering handler loop");

  handler_running_ = true;
  while (handler_running_) {
    auto msg_opt = MessagesHelper::TryRecvFromClient(client_router_socket_);
    if (!msg_opt.has_value()) {
      std::this_thread::sleep_for(kHandleLoopSleepMs);
      continue;
    }

    auto& msg = msg_opt.value();
    switch (msg.header.msg_type) {
      case MsgType::kRegisterTensorShardRequest: {
        auto request = std::get<RegisterTensorShardRequest>(
            MessagesHelper::DecodeRequest(msg));
        HandleRegisterTensorShardRequest(msg.identity, request);
        break;
      }
      default:
        LOG_WARNING("Unknown message type: {}",
                    static_cast<std::uint16_t>(msg.header.msg_type));
        break;
    }
  }
}

void NodeAgent::HandleRegisterTensorShardRequest(
    const std::string& client_identity,
    const RegisterTensorShardRequest& request) {
  LOG_DEBUG("Handling RegisterTensorShardRequest for tensor: {}",
            request.tensor_name);

  RegisterTensorShard(request.tensor_name);

  RegisterTensorShardResponse response(ErrorCode::kSuccess);
  MessagesHelper::SendToClient(client_router_socket_, client_identity,
                               MsgType::kRegisterTensorShardResponse, response);
}

void NodeAgent::ExecutorLoop() {
  LOG_DEBUG("Entering executor loop");
}
}  // namespace setu::node_manager
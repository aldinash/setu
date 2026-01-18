#pragma once

#include "commons/StdCommon.h"
#include "commons/Types.h"
#include "commons/datatypes/CopySpec.h"
#include "commons/datatypes/TensorShard.h"
#include "commons/datatypes/TensorShardRef.h"
#include "commons/messages/RegisterTensorShardRequest.h"
#include "commons/utils/ThreadingUtils.h"
#include "commons/utils/ZmqHelper.h"

namespace setu::node_manager {
using setu::commons::Queue;
using setu::commons::TensorName;
using setu::commons::datatypes::TensorShardRef;
using setu::commons::messages::RegisterTensorShardRequest;
using setu::commons::utils::ZmqContextPtr;
using setu::commons::utils::ZmqSocketPtr;

class NodeAgent {
 public:
  NodeAgent(std::size_t router_port, std::size_t dealer_executor_port,
            std::size_t dealer_handler_port);
  ~NodeAgent();

  void RegisterTensorShard(const TensorName& tensor_name);

  void Start();
  void Stop();

 private:
  void InitializeThreads();

  void StopThreads();

  void StartHandlerLoop();

  void StopHandlerLoop();

  void StartExecutorLoop();

  void StopExecutorLoop();

  void HandlerLoop();

  void ExecutorLoop();

  void HandleRegisterTensorShardRequest(
      const std::string& client_identity,
      const RegisterTensorShardRequest& request);

  void InitZmqSockets();

  void CloseZmqSockets();

  std::shared_ptr<zmq::context_t> zmq_context_;
  ZmqSocketPtr client_router_socket_;
  ZmqSocketPtr coordinator_dealer_executor_socket_;
  ZmqSocketPtr coordinator_dealer_handler_socket_;

  std::thread handler_thread_;
  std::thread executor_thread_;

  std::size_t router_port_;
  std::size_t dealer_executor_port_;
  std::size_t dealer_handler_port_;

  std::atomic<bool> handler_running_{false};
  std::atomic<bool> executor_running_{false};
};
}  // namespace setu::node_manager
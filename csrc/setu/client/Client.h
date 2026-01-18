#pragma once

#include "commons/StdCommon.h"
#include "commons/Types.h"
#include "commons/datatypes/CopySpec.h"
#include "commons/datatypes/TensorShardRef.h"
#include "commons/enums/Enums.h"
#include "commons/utils/ZmqHelper.h"

namespace setu::client {
using setu::commons::datatypes::TensorShardRef;
using setu::commons::enums::ErrorCode;
using setu::commons::utils::ZmqContextPtr;
using setu::commons::utils::ZmqSocketPtr;

class Client {
 public:
  Client(const std::string& endpoint);
  ~Client();

  ErrorCode RegisterTensorShard(const std::string& tensor_name);

 private:
  void InitZmqSockets();

  void CloseZmqSockets();

  // Zmq context and sockets
  ZmqContextPtr zmq_context_;
  ZmqSocketPtr request_socket_;

  std::string endpoint_;
};
//==============================================================================
}  // namespace setu::client
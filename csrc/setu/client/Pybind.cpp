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
#include "commons/Logging.h"
#include "commons/StdCommon.h"
#include "commons/TorchCommon.h"
#include "commons/datatypes/CopySpec.h"
#include "commons/datatypes/TensorShardRef.h"
#include "commons/datatypes/TensorShardSpec.h"
#include "commons/enums/Enums.h"
//==============================================================================
namespace setu::client {
//==============================================================================
using setu::commons::ClientRank;
using setu::commons::CopyOperationId;
using setu::commons::datatypes::CopySpec;
using setu::commons::datatypes::TensorShardRef;
using setu::commons::datatypes::TensorShardSpec;
using setu::commons::enums::ErrorCode;
//==============================================================================
void InitClientPybindClass(py::module_& m) {
  py::class_<Client, std::shared_ptr<Client>>(m, "Client")
      .def(py::init<ClientRank>(), py::arg("client_rank") = ClientRank{0},
           "Create a new client instance")
      .def("connect", &Client::Connect, py::arg("endpoint"),
           "Connect to a NodeAgent at the specified endpoint")
      .def("disconnect", &Client::Disconnect,
           "Disconnect from the current NodeAgent")
      .def("is_connected", &Client::IsConnected,
           "Check if the client is connected")
      .def("get_endpoint", &Client::GetEndpoint,
           "Get the endpoint the client is connected to")
      .def("register_tensor_shard", &Client::RegisterTensorShard,
           py::arg("shard_spec"),
           "Register a tensor shard and return a reference to it")
      .def("submit_copy", &Client::SubmitCopy, py::arg("copy_spec"),
           "Submit a copy operation and return an operation ID")
      .def("wait_for_copy", &Client::WaitForCopy, py::arg("copy_op_id"),
           "Wait for a copy operation to complete");
}
//==============================================================================
void InitEnumsPybindClass(py::module_& m) {
  py::enum_<ErrorCode>(m, "ErrorCode")
      .value("SUCCESS", ErrorCode::kSuccess)
      .value("INVALID_ARGUMENTS", ErrorCode::kInvalidArguments)
      .value("TIMEOUT", ErrorCode::kTimeout)
      .value("INTERNAL_ERROR", ErrorCode::kInternalError);
}
//==============================================================================
}  // namespace setu::client
//==============================================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  setu::commons::Logger::InitializeLogLevel();

  setu::client::InitEnumsPybindClass(m);
  setu::client::InitClientPybindClass(m);
}
//==============================================================================

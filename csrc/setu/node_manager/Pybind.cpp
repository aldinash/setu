//==============================================================================
// Copyright 2025 Vajra Team; Georgia Institute of Technology; Microsoft
// Corporation
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
#include "node_manager/datatypes/Pybind.h"

#include "commons/Logging.h"
#include "commons/StdCommon.h"
#include "commons/TorchCommon.h"
#include "node_manager/NodeAgent.h"
//==============================================================================
namespace setu::node_manager {
//==============================================================================
void InitNodeAgentPybindClass(py::module_& m) {
  py::class_<NodeAgent, std::shared_ptr<NodeAgent>>(m, "NodeAgent")
      .def(py::init<std::size_t, std::size_t, std::size_t>(),
           py::arg("router_port"), py::arg("dealer_executor_port"),
           py::arg("dealer_handler_port"))
      .def("start", &NodeAgent::Start)
      .def("stop", &NodeAgent::Stop)
      .def("register_tensor_shard", &NodeAgent::RegisterTensorShard,
           py::arg("tensor_name"));
}
//==============================================================================
}  // namespace setu::node_manager
//==============================================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  setu::commons::Logger::InitializeLogLevel();

  setu::node_manager::datatypes::InitDatatypesPybindSubmodule(m);
  setu::node_manager::InitNodeAgentPybindClass(m);
}
//==============================================================================

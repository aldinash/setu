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
#include "Pybind.h"
//==============================================================================
#include "commons/Logging.h"
#include "commons/StdCommon.h"
#include "commons/TorchCommon.h"
#include "commons/messages/Messages.h"
//==============================================================================
namespace setu::commons::messages {
//==============================================================================
using setu::commons::enums::ErrorCode;
using setu::commons::enums::MsgType;
//==============================================================================
void InitHeaderPybind(py::module_& m) {
  py::class_<Header>(m, "Header", py::module_local())
      .def(py::init<MsgType>(), py::arg("msg_type"))
      .def_readonly("msg_type", &Header::msg_type)
      .def("__str__", &Header::ToString)
      .def("__repr__", &Header::ToString);
}
//==============================================================================
void InitRegisterTensorShardRequestPybind(py::module_& m) {
  py::class_<RegisterTensorShardRequest>(m, "RegisterTensorShardRequest",
                                         py::module_local())
      .def(py::init<TensorName>(), py::arg("tensor_name"))
      .def_readonly("tensor_name", &RegisterTensorShardRequest::tensor_name)
      .def("__str__", &RegisterTensorShardRequest::ToString)
      .def("__repr__", &RegisterTensorShardRequest::ToString);
}
//==============================================================================
void InitRegisterTensorShardResponsePybind(py::module_& m) {
  py::class_<RegisterTensorShardResponse>(m, "RegisterTensorShardResponse",
                                          py::module_local())
      .def(py::init<ErrorCode>(), py::arg("error_code"))
      .def_readonly("error_code", &RegisterTensorShardResponse::error_code)
      .def("__str__", &RegisterTensorShardResponse::ToString)
      .def("__repr__", &RegisterTensorShardResponse::ToString);
}
//==============================================================================
void InitMessagesPybindSubmodule(py::module_& pm) {
  auto m = pm.def_submodule("messages", "Messages submodule");

  InitHeaderPybind(m);
  InitRegisterTensorShardRequestPybind(m);
  InitRegisterTensorShardResponsePybind(m);
}
//==============================================================================
}  // namespace setu::commons::messages
//==============================================================================

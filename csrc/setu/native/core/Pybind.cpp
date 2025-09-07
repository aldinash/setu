//==============================================================================
// Copyright 2025 Setu Team; Georgia Institute of Technology
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

#include "Core.h"
#include "commons/TorchCommon.h"
//==============================================================================
namespace setu::native::core {
//==============================================================================
void InitPybindSubmodule(py::module_& pm) {
  auto m = pm.def_submodule("core", "Core submodule");

  // Bind CoreProcessor class
  py::class_<CoreProcessor>(m, "CoreProcessor")
      .def(py::init<>())
      .def("process", &CoreProcessor::Process, py::arg("input"),
           "Process input strings")
      .def("get_processed_count", &CoreProcessor::GetProcessedCount,
           "Get number of processed items");
}
//==============================================================================
}  // namespace setu::native::core
//==============================================================================

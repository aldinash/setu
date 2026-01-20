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
#include "worker/Pybind.h"
//==============================================================================
#include "commons/Logging.h"
#include "commons/datatypes/Device.h"
#include "coordinator/datatypes/Program.h"
#include "worker/Worker.h"
//==============================================================================
namespace setu::worker {
//==============================================================================
using setu::commons::datatypes::Device;
using setu::coordinator::datatypes::Program;
//==============================================================================
void InitWorkerPybindClass(py::module_& m) {
  py::class_<Worker, std::shared_ptr<Worker>>(m, "Worker")
      .def(py::init<Device, std::size_t>(), py::arg("device"),
           py::arg("reply_port"),
           "Create a worker bound to a device and reply port")
      .def("start", &Worker::Start, "Start the worker executor loop")
      .def("stop", &Worker::Stop, "Stop the worker executor loop")
      .def("execute", &Worker::Execute, py::arg("program"),
           "Execute a program on the worker")
      .def("is_running", &Worker::IsRunning, "Check if worker is running")
      .def_property_readonly("device", &Worker::GetDevice,
                             "Get the device this worker is bound to");
}
//==============================================================================
}  // namespace setu::worker
//==============================================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  setu::commons::Logger::InitializeLogLevel();

  setu::worker::InitWorkerPybindClass(m);
}
//==============================================================================

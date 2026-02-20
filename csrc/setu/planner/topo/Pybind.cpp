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
#include "planner/topo/Pybind.h"
//==============================================================================
#include "commons/Logging.h"
#include "commons/StdCommon.h"
#include "commons/TorchCommon.h"
//==============================================================================
#include "planner/topo/Topology.h"
//==============================================================================
namespace setu::planner::topo {
//==============================================================================
void InitTopoPybind(py::module_& m) {
  py::class_<Link>(m, "Link").def(
      py::init<float, float, std::optional<std::string>>(),
      py::arg("latency_us"), py::arg("bandwidth_gbps"),
      py::arg("tag") = std::nullopt,
      "Create a link with latency and bandwidth");

  py::class_<Topology, TopologyPtr>(m, "Topology")
      .def(py::init<>(), "Create an empty topology")
      .def("add_link", &Topology::AddLink, py::arg("src"), py::arg("dst"),
           py::arg("link"), "Add a directed link between participants")
      .def("add_bidirectional_link", &Topology::AddBidirectionalLink,
           py::arg("src"), py::arg("dst"), py::arg("link"),
           "Add a bidirectional link between participants");
}
//==============================================================================
}  // namespace setu::planner::topo
//==============================================================================

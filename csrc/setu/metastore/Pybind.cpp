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
#include "metastore/datatypes/Pybind.h"
//==============================================================================
#include "commons/Logging.h"
#include "commons/StdCommon.h"
#include "commons/TorchCommon.h"
#include "commons/datatypes/Pybind.h"
#include "commons/enums/Pybind.h"
#include "commons/utils/Pybind.h"
#include "metastore/MetaStore.h"
//==============================================================================
namespace setu::metastore {
//==============================================================================
using setu::commons::NodeId;
using setu::commons::TensorName;
using setu::commons::datatypes::TensorShardSpec;
//==============================================================================
void InitMetaStorePybindClass(py::module_& m) {
  py::class_<MetaStore, std::shared_ptr<MetaStore>>(m, "MetaStore")
      .def(py::init<>(), "Create a new MetaStore instance")
      .def("register_tensor_shard", &MetaStore::RegisterTensorShard,
           py::arg("shard_spec"), py::arg("owner_node_id"),
           "Register a new tensor shard and return its metadata")
      .def("all_shards_registered", &MetaStore::AllShardsRegistered,
           py::arg("tensor_name"),
           "Check if all shards are registered for a tensor")
      .def("get_num_shards_for_tensor", &MetaStore::GetNumShardsForTensor,
           py::arg("tensor_name"),
           "Get number of shards registered for a tensor")
      .def("get_tensor_metadata", &MetaStore::GetTensorMetadata,
           py::arg("tensor_name"),
           "Get complete tensor metadata once all shards are registered");
}
//==============================================================================
}  // namespace setu::metastore
//==============================================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  setu::commons::Logger::InitializeLogLevel();

  setu::commons::enums::InitEnumsPybindSubmodule(m);
  setu::commons::datatypes::InitDatatypesPybindSubmodule(m);
  setu::metastore::datatypes::InitDatatypesPybindSubmodule(m);
  setu::metastore::InitMetaStorePybindClass(m);
}
//==============================================================================

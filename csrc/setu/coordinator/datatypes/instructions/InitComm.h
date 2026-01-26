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
#pragma once
//==============================================================================
#include <nccl.h>
#include <utility>

#include "setu/commons/StdCommon.h"
#include "setu/commons/Types.h"
#include "setu/commons/utils/Serialization.h"
//==============================================================================
namespace setu::coordinator::datatypes::instructions {
//==============================================================================
using setu::commons::DeviceRank;
using setu::commons::utils::BinaryBuffer;
using setu::commons::utils::BinaryRange;
using setu::commons::utils::BinaryReader;
using setu::commons::utils::BinaryWriter;
//==============================================================================

struct InitCommInstruction {
  InitCommInstruction() = default;
  InitCommInstruction(ncclUniqueId id,
                      std::unordered_map<DeviceRank, int> device_to_rank)
      : comm_id(std::move(id)), device_to_rank(std::move(device_to_rank)) {}

  ~InitCommInstruction() = default;

  [[nodiscard]] std::string ToString() const {
    return std::format("InitCommInstruction(device_to_rank_size={})",
                       device_to_rank.size());
  }

  void Serialize(BinaryBuffer& buffer) const {
    BinaryWriter w(buffer);
    // ncclUniqueId is a POD containing bytes; write it directly
    w.Write(comm_id);
    w.Write(device_to_rank);
  }

  static InitCommInstruction Deserialize(const BinaryRange& range) {
    BinaryReader r(range);
    ncclUniqueId id = r.Read<ncclUniqueId>();
    auto map = r.Read<std::unordered_map<DeviceRank, int>>();
    return InitCommInstruction(id, map);
  }

  const ncclUniqueId comm_id;
  const std::unordered_map<DeviceRank, int> device_to_rank;
};

//==============================================================================
}  // namespace setu::coordinator::datatypes::instructions
//==============================================================================

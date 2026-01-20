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
#include "setu/commons/StdCommon.h"
#include "setu/commons/Types.h"
#include "setu/commons/utils/Serialization.h"
#include "setu/coordinator/datatypes/Program.h"
//==============================================================================
namespace setu::coordinator::datatypes {
//==============================================================================
using setu::commons::DeviceRank;
using setu::commons::utils::BinaryBuffer;
using setu::commons::utils::BinaryRange;
using setu::commons::utils::BinaryReader;
using setu::commons::utils::BinaryWriter;
//==============================================================================

struct Plan {
  Plan() = default;
  ~Plan() = default;

  [[nodiscard]] std::string ToString() const {
    return std::format("Plan(worker_programs={})", worker_programs.size());
  }

  void Serialize(BinaryBuffer& buffer) const {
    BinaryWriter writer(buffer);
    writer.WriteFields(worker_programs);
  }

  static Plan Deserialize(const BinaryRange& range) {
    BinaryReader reader(range);
    Plan plan;
    std::tie(plan.worker_programs) =
        reader.ReadFields<std::unordered_map<DeviceRank, Program>>();
    return plan;
  }

  std::unordered_map<DeviceRank, Program> worker_programs;
};

//==============================================================================
}  // namespace setu::coordinator::datatypes
//==============================================================================

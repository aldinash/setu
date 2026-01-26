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

#include "setu/commons/StdCommon.h"
#include "setu/commons/Types.h"
#include "setu/commons/utils/Serialization.h"
//==============================================================================
namespace setu::coordinator::datatypes::instructions {
//==============================================================================
using setu::commons::utils::BinaryBuffer;
using setu::commons::utils::BinaryRange;
using setu::commons::utils::BinaryReader;
using setu::commons::utils::BinaryWriter;
//==============================================================================

struct UseCommInstruction {
  UseCommInstruction() = default;
  explicit UseCommInstruction(ncclUniqueId id) : comm_id(std::move(id)) {}

  ~UseCommInstruction() = default;

  [[nodiscard]] std::string ToString() const {
    return std::format("UseCommInstruction(comm_id_present={})", true);
  }

  void Serialize(BinaryBuffer& buffer) const {
    BinaryWriter w(buffer);
    w.Write(comm_id);
  }

  static UseCommInstruction Deserialize(const BinaryRange& range) {
    BinaryReader r(range);
    auto id = r.Read<ncclUniqueId>();
    return UseCommInstruction(id);
  }

  const ncclUniqueId comm_id;
};

//==============================================================================
}  // namespace setu::coordinator::datatypes::instructions
//==============================================================================

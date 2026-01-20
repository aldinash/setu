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
#include "setu/commons/utils/Serialization.h"
#include "setu/coordinator/datatypes/Program.h"
//==============================================================================
namespace setu::commons::messages {
//==============================================================================
using setu::commons::utils::BinaryBuffer;
using setu::commons::utils::BinaryRange;
using setu::commons::utils::BinaryReader;
using setu::commons::utils::BinaryWriter;
using setu::coordinator::datatypes::Program;
//==============================================================================

struct ExecuteProgramRequest {
  Program program;

  ExecuteProgramRequest() = default;
  explicit ExecuteProgramRequest(Program program_param)
      : program(std::move(program_param)) {}

  [[nodiscard]] std::string ToString() const {
    return std::format("ExecuteProgramRequest(program={})", program.ToString());
  }

  void Serialize(BinaryBuffer& buffer) const {
    BinaryWriter writer(buffer);
    writer.WriteFields(program);
  }

  static ExecuteProgramRequest Deserialize(const BinaryRange& range) {
    BinaryReader reader(range);
    ExecuteProgramRequest request;
    std::tie(request.program) = reader.ReadFields<Program>();
    return request;
  }
};
using ExecuteProgramRequestPtr = std::shared_ptr<ExecuteProgramRequest>;

//==============================================================================
}  // namespace setu::commons::messages
//==============================================================================

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
#include "setu/coordinator/datatypes/instructions/Copy.h"
#include "setu/coordinator/datatypes/instructions/Send.h"
#include "setu/coordinator/datatypes/instructions/Receive.h"
#include "setu/coordinator/datatypes/instructions/InitComm.h"
#include "setu/coordinator/datatypes/instructions/UseComm.h"
#include <variant>
//==============================================================================
namespace setu::coordinator::datatypes {
//==============================================================================
using setu::commons::utils::BinaryBuffer;
using setu::commons::utils::BinaryRange;
using setu::commons::utils::BinaryReader;
using setu::commons::utils::BinaryWriter;
//==============================================================================

enum class InstructionType : std::uint8_t {
  kInitComm = 1,
  kUseComm = 2,
  kCopy = 3,
  kSend = 4,
  kReceive = 5,
};

struct Instruction {
  Instruction() = delete;

  template <typename T>
  explicit Instruction(T inst) : instruction(std::move(inst)) {}

  [[nodiscard]] std::string ToString() const {
    return std::visit([](auto&& instr) { return instr.ToString(); }, instruction);
  }

  void Serialize(BinaryBuffer& buffer) const {
    BinaryWriter writer(buffer);

    std::visit([&](auto&& inst) {
      using T = std::decay_t<decltype(inst)>;
      InstructionType t = InstructionType::kInitComm;
      if constexpr (std::is_same_v<T, InitCommInstruction>)
        t = InstructionType::kInitComm;
      else if constexpr (std::is_same_v<T, UseCommInstruction>)
        t = InstructionType::kUseComm;
      else if constexpr (std::is_same_v<T, CopyInstruction>)
        t = InstructionType::kCopy;
      else if constexpr (std::is_same_v<T, SendInstruction>)
        t = InstructionType::kSend;
      else if constexpr (std::is_same_v<T, ReceiveInstruction>)
        t = InstructionType::kReceive;

      writer.Write<std::uint8_t>(static_cast<std::uint8_t>(t));
      writer.Write(inst);
    }, instruction);
  }

  static Instruction Deserialize(const BinaryRange& range) {
    BinaryReader reader(range);
    
    const auto type_id = reader.Read<std::uint8_t>();
    switch (static_cast<InstructionType>(type_id)) {
      case InstructionType::kInitComm: {
        auto v = reader.Read<InitCommInstruction>();
        return Instruction(v);
      }
      case InstructionType::kUseComm: {
        auto v = reader.Read<UseCommInstruction>();
        return Instruction(v);
      }
      case InstructionType::kCopy: {
        auto v = reader.Read<CopyInstruction>();
        return Instruction(v);
      }
      case InstructionType::kSend: {
        auto v = reader.Read<SendInstruction>();
        return Instruction(v);
      }
      case InstructionType::kReceive: {
        auto v = reader.Read<ReceiveInstruction>();
        return Instruction(v);
      }
      default:
        RAISE_RUNTIME_ERROR("Unknown instruction type id {}", type_id);
    }
  }

  std::variant<InitCommInstruction, UseCommInstruction, CopyInstruction, SendInstruction, ReceiveInstruction> instruction;
};

//==============================================================================
}  // namespace setu::coordinator::datatypes
//==============================================================================

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
//==============================================================================
#include "setu/coordinator/datatypes/instructions/Copy.h"
#include "setu/coordinator/datatypes/instructions/InitComm.h"
#include "setu/coordinator/datatypes/instructions/Receive.h"
#include "setu/coordinator/datatypes/instructions/Send.h"
#include "setu/coordinator/datatypes/instructions/UseComm.h"
//==============================================================================
namespace setu::coordinator::datatypes {
//==============================================================================
using setu::commons::utils::BinaryBuffer;
using setu::commons::utils::BinaryRange;
using setu::commons::utils::BinaryReader;
using setu::commons::utils::BinaryWriter;
using instructions::CopyInstruction;
using instructions::InitCommInstruction;
using instructions::ReceiveInstruction;
using instructions::SendInstruction;
using instructions::UseCommInstruction;
//==============================================================================

enum class InstructionType : std::uint8_t {
  kInitComm = 1,
  kUseComm = 2,
  kCopy = 3,
  kSend = 4,
  kReceive = 5,
};

using InstructionVariant =
    std::variant<InitCommInstruction, UseCommInstruction, CopyInstruction,
                 SendInstruction, ReceiveInstruction>;

struct Instruction {
  Instruction() = delete;

  template <typename T>
  explicit Instruction(T inst) : instr(std::move(inst)) {}

  ~Instruction() = default;
  Instruction(const Instruction&) = default;
  Instruction& operator=(const Instruction&) = default;
  Instruction(Instruction&&) = default;
  Instruction& operator=(Instruction&&) = default;

  [[nodiscard]] std::string ToString() const {
    return std::visit([](const auto& instr) { return instr.ToString(); },
                      instr);
  }

  void Serialize(BinaryBuffer& buffer) const {
    BinaryWriter writer(buffer);

    std::visit(
        [&writer](const auto& inst) {
          using T = std::decay_t<decltype(inst)>;
          InstructionType type = InstructionType::kInitComm;
          if constexpr (std::is_same_v<T, InitCommInstruction>) {
            type = InstructionType::kInitComm;
          } else if constexpr (std::is_same_v<T, UseCommInstruction>) {
            type = InstructionType::kUseComm;
          } else if constexpr (std::is_same_v<T, CopyInstruction>) {
            type = InstructionType::kCopy;
          } else if constexpr (std::is_same_v<T, SendInstruction>) {
            type = InstructionType::kSend;
          } else if constexpr (std::is_same_v<T, ReceiveInstruction>) {
            type = InstructionType::kReceive;
          }

          writer.Write<std::uint8_t>(static_cast<std::uint8_t>(type));
          writer.Write(inst);
        },
        instr);
  }

  static Instruction Deserialize(const BinaryRange& range) {
    BinaryReader reader(range);

    const auto type_id = reader.Read<std::uint8_t>();
    switch (static_cast<InstructionType>(type_id)) {
      case InstructionType::kInitComm:
        return Instruction(reader.Read<InitCommInstruction>());
      case InstructionType::kUseComm:
        return Instruction(reader.Read<UseCommInstruction>());
      case InstructionType::kCopy:
        return Instruction(reader.Read<CopyInstruction>());
      case InstructionType::kSend:
        return Instruction(reader.Read<SendInstruction>());
      case InstructionType::kReceive:
        return Instruction(reader.Read<ReceiveInstruction>());
      default:
        RAISE_RUNTIME_ERROR("Unknown instruction type id {}", type_id);
    }
  }

  InstructionVariant instr;
};

//==============================================================================
}  // namespace setu::coordinator::datatypes
//==============================================================================

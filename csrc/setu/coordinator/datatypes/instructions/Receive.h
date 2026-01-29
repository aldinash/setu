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
#include "setu/commons/enums/Enums.h"
#include "setu/commons/utils/Serialization.h"
//==============================================================================
namespace setu::coordinator::datatypes::instructions {
//==============================================================================
using setu::commons::utils::BinaryBuffer;
using setu::commons::utils::BinaryRange;
using setu::commons::utils::BinaryReader;
using setu::commons::utils::BinaryWriter;
using setu::commons::DeviceRank;
using setu::commons::enums::DType;
using setu::commons::TensorName;
using setu::commons::ShardId;
//==============================================================================

struct ReceiveInstruction {
  ReceiveInstruction(DeviceRank src_device_id,
                     std::pair<TensorName, ShardId> dst_tensor,
                     DType dtype,
                     std::size_t memory_offset_bytes,
                     std::size_t num_elements)
      : src_device_id(src_device_id),
        dst_tensor(std::move(dst_tensor)),
        dtype(dtype),
        memory_offset_bytes(memory_offset_bytes),
        num_elements(num_elements) {}

  ~ReceiveInstruction() = default;
  ReceiveInstruction(const ReceiveInstruction&) = default;
  ReceiveInstruction& operator=(const ReceiveInstruction&) = default;
  ReceiveInstruction(ReceiveInstruction&&) = default;
  ReceiveInstruction& operator=(ReceiveInstruction&&) = default;

  [[nodiscard]] std::string ToString() const {
    return std::format(
        "ReceiveInstruction(src_rank={}, tensor=({}, {}), dtype={}, "
        "memory_offset={}, num_elements={})",
        src_device_id, dst_tensor.first, dst_tensor.second,
        static_cast<int>(dtype), memory_offset_bytes, num_elements);
  }

  void Serialize(BinaryBuffer& buffer) const {
    BinaryWriter writer(buffer);
    writer.WriteFields(src_device_id, dst_tensor.first, dst_tensor.second,
                       dtype, memory_offset_bytes, num_elements);
  }

  static ReceiveInstruction Deserialize(const BinaryRange& range) {
    BinaryReader reader(range);
    auto [src_device_id, tensor_name, shard_id, dtype, memory_offset_bytes,
          num_elements] =
        reader.ReadFields<DeviceRank, TensorName, ShardId, DType, std::size_t,
                          std::size_t>();
    return ReceiveInstruction(src_device_id,
                              {std::move(tensor_name), std::move(shard_id)},
                              dtype, memory_offset_bytes, num_elements);
  }

  DeviceRank src_device_id;
  std::pair<TensorName, ShardId> dst_tensor;
  DType dtype;
  std::size_t memory_offset_bytes;
  std::size_t num_elements;
};

//==============================================================================
}  // namespace setu::coordinator::datatypes::instructions
//==============================================================================

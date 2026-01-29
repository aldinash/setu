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
using setu::commons::enums::DType;
using setu::commons::TensorName;
using setu::commons::ShardId;
//==============================================================================

struct CopyInstruction {
  CopyInstruction(std::pair<TensorName, ShardId> src_tensor,
                  std::size_t src_memory_offset_bytes,
                  std::pair<TensorName, ShardId> dst_tensor,
                  std::size_t dst_memory_offset_bytes,
                  DType dtype,
                  std::size_t num_elements)
      : src_tensor(std::move(src_tensor)),
        src_memory_offset_bytes(src_memory_offset_bytes),
        dst_tensor(std::move(dst_tensor)),
        dst_memory_offset_bytes(dst_memory_offset_bytes),
        dtype(dtype),
        num_elements(num_elements) {}

  ~CopyInstruction() = default;
  CopyInstruction(const CopyInstruction&) = default;
  CopyInstruction& operator=(const CopyInstruction&) = default;
  CopyInstruction(CopyInstruction&&) = default;
  CopyInstruction& operator=(CopyInstruction&&) = default;

  [[nodiscard]] std::string ToString() const {
    return std::format(
        "CopyInstruction(src_tensor=({}, {}), src_offset={}, dst_tensor=({}, "
        "{}), dst_offset={}, dtype={}, num_elements={})",
        src_tensor.first, src_tensor.second, src_memory_offset_bytes,
        dst_tensor.first, dst_tensor.second, dst_memory_offset_bytes,
        static_cast<int>(dtype), num_elements);
  }

  void Serialize(BinaryBuffer& buffer) const {
    BinaryWriter writer(buffer);
    writer.WriteFields(src_tensor.first, src_tensor.second,
                       src_memory_offset_bytes, dst_tensor.first,
                       dst_tensor.second, dst_memory_offset_bytes, dtype,
                       num_elements);
  }

  static CopyInstruction Deserialize(const BinaryRange& range) {
    BinaryReader reader(range);
    auto [src_tensor_name, src_shard_id, src_memory_offset_bytes,
          dst_tensor_name, dst_shard_id, dst_memory_offset_bytes, dtype,
          num_elements] =
        reader.ReadFields<TensorName, ShardId, std::size_t, TensorName, ShardId,
                          std::size_t, DType, std::size_t>();

    return CopyInstruction({std::move(src_tensor_name), std::move(src_shard_id)},
                           src_memory_offset_bytes,
                           {std::move(dst_tensor_name), std::move(dst_shard_id)},
                           dst_memory_offset_bytes, dtype, num_elements);
  }

  std::pair<TensorName, ShardId> src_tensor;
  std::size_t src_memory_offset_bytes;
  std::pair<TensorName, ShardId> dst_tensor;
  std::size_t dst_memory_offset_bytes;
  DType dtype;
  std::size_t num_elements;
};

//==============================================================================
}  // namespace setu::coordinator::datatypes::instructions
//==============================================================================

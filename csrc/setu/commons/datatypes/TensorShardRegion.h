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
#include "commons/StdCommon.h"
#include "commons/Types.h"
#include "commons/utils/Serialization.h"
//==============================================================================
namespace setu::commons::datatypes {
using setu::commons::utils::BinaryBuffer;
using setu::commons::utils::BinaryRange;
using setu::commons::utils::BinaryReader;
using setu::commons::utils::BinaryWriter;
//==============================================================================

/// @brief Describes a contiguous element region within a tensor shard.
///
/// Offsets and counts are expressed in elements (not bytes). The worker is
/// responsible for resolving the associated `ShardId` to a local
/// `TensorShard` to obtain the device pointer and dtype; it then computes
/// bytes = count * dtype_size for actual memcpy/NCCL operations.
struct TensorShardRegion {
  TensorShardRegion(const TensorName& tensor_name_param,
                    const ShardId& shard_id_param,
                    std::uint64_t shard_offset_param,
                    std::uint64_t count_param)
      : tensor_name(tensor_name_param),
        shard_id(shard_id_param),
        shard_offset(shard_offset_param),
        count(count_param) {
    ASSERT_VALID_ARGUMENTS(count_param > 0, "count must be > 0");
  }

  [[nodiscard]] std::string ToString() const {
    return std::format("TensorShardRegion(tensor={}, shard={}, offset={}, count={})",
                       tensor_name, shard_id, shard_offset, count);
  }

  void Serialize(BinaryBuffer& buffer) const {
    BinaryWriter writer(buffer);
    writer.Write(tensor_name);
    writer.Write(shard_id);
    writer.Write(shard_offset);
    writer.Write(count);
  }

  static TensorShardRegion Deserialize(const BinaryRange& range) {
    BinaryReader reader(range);
    auto name = reader.Read<TensorName>();
    auto id = reader.Read<ShardId>();
    auto offset = reader.Read<std::uint64_t>();
    auto cnt = reader.Read<std::uint64_t>();
    return TensorShardRegion(name, id, offset, cnt);
  }

  [[nodiscard]] std::uint64_t GetStart() const { return shard_offset; }
  [[nodiscard]] std::uint64_t GetEnd() const { return shard_offset + count; }
  [[nodiscard]] std::uint64_t NumElements() const { return count; }

  const TensorName tensor_name;
  const ShardId shard_id;
  const std::uint64_t shard_offset;  ///< element offset within shard
  const std::uint64_t count;         ///< number of elements
};

//==============================================================================
}  // namespace setu::commons::datatypes
//==============================================================================

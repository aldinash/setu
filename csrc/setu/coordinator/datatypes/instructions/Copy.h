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
#include "setu/commons/datatypes/TensorShardRegion.h"
#include "setu/commons/utils/Serialization.h"
//==============================================================================
namespace setu::coordinator::datatypes::instructions {
//==============================================================================
using setu::commons::utils::BinaryBuffer;
using setu::commons::utils::BinaryRange;
using setu::commons::utils::BinaryReader;
using setu::commons::utils::BinaryWriter;
//==============================================================================

struct CopyInstruction {
  // Construct from two reusable TensorShardRegion values
  CopyInstruction(TensorShardRegion src_region,
                  TensorShardRegion dst_region)
      : src_shard_region(std::move(src_region)), dst_shard_region(std::move(dst_region)) {}

  ~CopyInstruction() = default;

  [[nodiscard]] std::string ToString() const {
  return std::format("CopyInstruction(src={}, dst={})", src_shard_region.ToString(),
             dst_shard_region.ToString());
  }

  void Serialize(BinaryBuffer& buffer) const {
    BinaryWriter writer(buffer);
    src_shard_region.Serialize(buffer);
    dst_shard_region.Serialize(buffer);
  }

  static CopyInstruction Deserialize(const BinaryRange& range) {
    BinaryReader reader(range);
    // Read src region fields
    auto src = reader.Read<TensorShardRegion>();
    auto dst = reader.Read<TensorShardRegion>();

    return CopyInstruction(src, dst);
  }

  const TensorShardRegion src_shard_region;
  const TensorShardRegion dst_shard_region;
};

//==============================================================================
}  // namespace setu::coordinator::datatypes
//==============================================================================

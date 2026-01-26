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
using setu::commons::DeviceRank;
//==============================================================================

struct ReceiveInstruction {
  ReceiveInstruction(DeviceRank src_device,
                     TensorShardRegion region)
      : src_device_id(std::move(src_device)), dst_shard_region(region_param) {}

  ~ReceiveInstruction() = default;

  [[nodiscard]] std::string ToString() const {
    return std::format("ReceiveInstruction(src_device_id={}, dst_shard_region={})", src_device_id,
                       dst_shard_region.ToString());
  }

  void Serialize(BinaryBuffer& buffer) const {
    BinaryWriter w(buffer);
    w.Write(src_device_id);
    dst_shard_region.Serialize(buffer);
  }

  static ReceiveInstruction Deserialize(const BinaryRange& range) {
    BinaryReader reader(range);
    auto src_device = reader.Read<DeviceRank>();
    auto dst_shard_region = reader.Read<TensorShardRegion>();
    return ReceiveInstruction(src_device, dst_shard_region);
  }

  const DeviceRank src_device_id;
  const TensorShardRegion dst_shard_region;
};

//==============================================================================
}  // namespace setu::coordinator::datatypes::instructions
//==============================================================================

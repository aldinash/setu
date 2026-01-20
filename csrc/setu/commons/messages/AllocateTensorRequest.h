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
#include "commons/BoostCommon.h"
#include "commons/StdCommon.h"
//==============================================================================
#include "commons/Types.h"
#include "commons/utils/Serialization.h"
//==============================================================================
namespace setu::commons::messages {
//==============================================================================
using setu::commons::DeviceRank;
using setu::commons::ShardId;
using setu::commons::TensorName;
using setu::commons::utils::BinaryBuffer;
using setu::commons::utils::BinaryRange;
//==============================================================================

struct AllocateTensorRequest {
  TensorName tensor_id;
  ShardId shard_id;
  DeviceRank device;

  AllocateTensorRequest() = default;
  AllocateTensorRequest(TensorName tensor_id_param, ShardId shard_id_param,
                        DeviceRank device_param)
      : tensor_id(std::move(tensor_id_param)),
        shard_id(shard_id_param),
        device(device_param) {}

  [[nodiscard]] std::string ToString() const {
    return std::format(
        "AllocateTensorRequest(tensor_id={}, shard_id={}, device={})",
        tensor_id, boost::uuids::to_string(shard_id), device);
  }

  void Serialize(BinaryBuffer& buffer) const;

  static AllocateTensorRequest Deserialize(const BinaryRange& range);
};
using AllocateTensorRequestPtr = std::shared_ptr<AllocateTensorRequest>;

//==============================================================================
}  // namespace setu::commons::messages
//==============================================================================

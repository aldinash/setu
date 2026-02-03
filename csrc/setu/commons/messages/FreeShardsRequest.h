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
//==============================================================================
#include "commons/Types.h"
#include "commons/messages/BaseRequest.h"
#include "commons/utils/Serialization.h"
//==============================================================================
namespace setu::commons::messages {
//==============================================================================
using setu::commons::RequestId;
using setu::commons::ShardId;
using setu::commons::utils::BinaryBuffer;
using setu::commons::utils::BinaryRange;
//==============================================================================

struct FreeShardsRequest : public BaseRequest {
  /// @brief Constructs a request with auto-generated request ID.
  explicit FreeShardsRequest(std::vector<ShardId> shard_ids_param)
      : BaseRequest(), shard_ids(std::move(shard_ids_param)) {
    ASSERT_VALID_ARGUMENTS(!shard_ids.empty(),
                           "Shard IDs list cannot be empty");
  }

  /// @brief Constructs a request with explicit request ID (for
  /// deserialization).
  FreeShardsRequest(RequestId request_id_param,
                    std::vector<ShardId> shard_ids_param)
      : BaseRequest(request_id_param), shard_ids(std::move(shard_ids_param)) {
    ASSERT_VALID_ARGUMENTS(!shard_ids.empty(),
                           "Shard IDs list cannot be empty");
  }

  [[nodiscard]] std::string ToString() const {
    return std::format("FreeShardsRequest(request_id={}, num_shards={})",
                       request_id, shard_ids.size());
  }

  void Serialize(BinaryBuffer& buffer) const;

  static FreeShardsRequest Deserialize(const BinaryRange& range);

  const std::vector<ShardId> shard_ids;
};
using FreeShardsRequestPtr = std::shared_ptr<FreeShardsRequest>;

//==============================================================================
}  // namespace setu::commons::messages
//==============================================================================

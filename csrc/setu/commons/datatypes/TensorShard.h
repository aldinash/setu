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
#include "commons/TorchCommon.h"
#include "commons/Types.h"
//==============================================================================
#include "commons/datatypes/TensorShardMetadata.h"
//==============================================================================
namespace setu::commons::datatypes {
//==============================================================================
/**
 * @brief Represents a physical tensor shard with device memory
 *
 * TensorShard combines metadata about a shard with the underlying torch::Tensor
 * and thread-safe locking mechanisms. Use TensorShardReadHandle and
 * TensorShardWriteHandle for safe access to the device memory.
 */
struct TensorShard {
  /**
   * @brief Constructs a tensor shard with metadata and tensor storage
   *
   * @param metadata_param Metadata describing this shard
   * @param tensor_param The torch::Tensor that owns the device memory
   *
   * @throws std::invalid_argument if tensor is not defined
   */
  TensorShard(TensorShardMetadata metadata_param, torch::Tensor tensor_param)
      : metadata(std::move(metadata_param)), tensor(std::move(tensor_param)) {
    ASSERT_VALID_ARGUMENTS(tensor.defined(), "Tensor must be defined");
  }

  /**
   * @brief Returns the raw device pointer to the tensor data
   *
   * @return Pointer to device memory location
   */
  [[nodiscard]] DevicePtr GetDevicePtr() const { return tensor.data_ptr(); }

  /**
   * @brief Returns the underlying torch::Tensor
   *
   * @return Reference to the tensor
   */
  [[nodiscard]] const torch::Tensor& GetTensor() const { return tensor; }

  /**
   * @brief Returns a string representation of the tensor shard
   *
   * @return String containing metadata and device pointer
   */
  [[nodiscard]] std::string ToString() const {
    return std::format("TensorShard(metadata={}, device_ptr={})",
                       metadata.ToString(), GetDevicePtr());
  }

  const TensorShardMetadata metadata;  ///< Immutable metadata for this shard

 private:
  friend class TensorShardReadHandle;
  friend class TensorShardWriteHandle;

  const torch::Tensor tensor;       ///< Tensor that owns the device memory
  mutable std::shared_mutex mutex;  ///< Mutex for thread-safe access to tensor
};
//==============================================================================
/// @brief Shared pointer to a TensorShard object
using TensorShardPtr = std::shared_ptr<TensorShard>;

/// @brief Map of shard IDs to TensorShard objects
using TensorShardsMap = std::unordered_map<ShardId, TensorShardPtr>;

/// @brief Concurrent map of shard IDs to TensorShard objects
using TensorShardsConcurrentMap = ConcurrentMap<ShardId, TensorShardPtr>;
//==============================================================================
}  // namespace setu::commons::datatypes
//==============================================================================

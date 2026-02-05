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
#include "commons/Logging.h"
#include "commons/StdCommon.h"
#include "commons/TorchCommon.h"
#include "commons/Types.h"
#include "commons/datatypes/TensorShardMetadata.h"
//==============================================================================
namespace setu::commons::datatypes {
//==============================================================================
/**
 * @brief Represents a physical tensor shard with device memory
 *
 * TensorShard combines metadata about a shard with the physical device pointer
 * and thread-safe locking mechanisms. Use TensorShardReadHandle and
 * TensorShardWriteHandle for safe access to the device memory.
 */
struct TensorShard {
  /**
   * @brief Constructs a tensor shard with metadata, tensor, and lock directory
   *
   * @param metadata_param Metadata describing this shard
   * @param tensor_param The torch tensor holding the shard data
   * @param lock_dir Directory for lock files (default: /tmp/setu/locks)
   *
   * @throws std::invalid_argument if tensor_param is not defined
   */
  TensorShard(TensorShardMetadata metadata_param, torch::Tensor tensor_param,
              std::string lock_dir = "/tmp/setu/locks")
      : metadata(std::move(metadata_param)),
        tensor(std::move(tensor_param)),
        lock_file_path_(
            (std::filesystem::path(lock_dir) /
             (boost::uuids::to_string(metadata.id) + ".lock"))
                .string()) {
    ASSERT_VALID_ARGUMENTS(tensor.defined() && tensor.numel() > 0,
                           "Invalid tensor argument: tensor is not defined");
    // Ensure the lock directory and file exist for cross-process locking
    std::filesystem::create_directories(
        std::filesystem::path(lock_file_path_).parent_path());
    std::int32_t fd =
        ::open(lock_file_path_.c_str(), O_RDWR | O_CREAT, 0666);
    ASSERT_VALID_RUNTIME(fd >= 0, "Failed to create lock file '{}': {}",
                         lock_file_path_, std::strerror(errno));
    ::close(fd);
  }

  /**
   * @brief Destructor. Removes the lock file.
   */
  ~TensorShard() {
    std::error_code ec;
    std::filesystem::remove(lock_file_path_, ec);
    if (ec) {
      LOG_WARNING("Failed to remove lock file '{}': {}", lock_file_path_,
                  ec.message());
    }
  }

  /**
   * @brief Returns a string representation of the tensor shard
   *
   * @return String containing metadata and device pointer
   */
  [[nodiscard]] std::string ToString() const {
    return std::format("TensorShard(metadata={}, tensor={})",
                       metadata.ToString(), tensor);
  }

  /**
   * @brief Get read-only pointer to device memory
   *
   * @return Const pointer to device memory
   */
  [[nodiscard]] DevicePtr GetDevicePtr() const { return tensor.data_ptr(); }

  const TensorShardMetadata metadata;  ///< Immutable metadata for this shard
  const torch::Tensor tensor;          ///< The torch tensor holding shard data

 private:
  friend class TensorShardReadHandle;
  friend class TensorShardWriteHandle;
  const std::string lock_file_path_;  ///< Path to flock file for cross-process
                                      ///< locking
};
//==============================================================================
/// @brief Shared pointer to a TensorShard object
using TensorShardPtr = std::shared_ptr<TensorShard>;

/// @brief Map of shard IDs to TensorShard objects
using TensorShardsMap = std::unordered_map<ShardId, TensorShardPtr>;
//==============================================================================
}  // namespace setu::commons::datatypes
//==============================================================================

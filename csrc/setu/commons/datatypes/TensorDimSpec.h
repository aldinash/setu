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
#include "commons/Logging.h"
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
/**
 * @brief Specification for a tensor dimension including its owned range
 *
 * TensorDimSpec defines a single dimension of a tensor shard, including its
 * name, total size, and the range of indices [start, end) owned by this shard.
 * This is used in distributed tensor sharding to specify the local view of
 * each dimension.
 */
struct TensorDimSpec {
  /**
   * @brief Constructs a tensor dimension specification
   *
   * @param name_param Name of the dimension (e.g., "batch", "sequence",
   * "hidden")
   * @param size_param Total size of the dimension across all shards
   * @param start_param Start index of owned range (inclusive)
   * @param end_param End index of owned range (exclusive)
   *
   * @throws std::invalid_argument if size_param is 0
   * @throws std::invalid_argument if start >= end
   * @throws std::invalid_argument if owned range extends beyond dimension size
   */
  TensorDimSpec(TensorDimName name_param, std::size_t size_param,
                TensorIndex start_param, TensorIndex end_param)
      : name(std::move(name_param)),
        size(size_param),
        start(start_param),
        end(end_param) {
    ASSERT_VALID_ARGUMENTS(size > 0, "Size {} must be greater than 0", size);
    ASSERT_VALID_ARGUMENTS(start >= 0, "Start {} must be non-negative", start);
    ASSERT_VALID_ARGUMENTS(start < end, "Start {} must be less than end {}",
                           start, end);
    ASSERT_VALID_ARGUMENTS(static_cast<std::size_t>(end) <= size,
                           "End {} must not exceed dimension size {}", end,
                           size);
  }

  /**
   * @brief Returns the size of the owned portion of this dimension
   *
   * @return Number of indices owned by this shard
   */
  [[nodiscard]] std::size_t GetOwnedSize() const {
    return static_cast<std::size_t>(end - start);
  }

  /**
   * @brief Checks if an index is within the owned range
   *
   * @param index The index to check
   * @return true if index is in [start, end)
   */
  [[nodiscard]] bool ContainsIndex(TensorIndex index) const {
    return index >= start && index < end;
  }

  /**
   * @brief Returns a string representation of the tensor dimension spec
   *
   * @return String containing the dimension properties
   */
  [[nodiscard]] std::string ToString() const {
    return std::format("TensorDimSpec(name={}, size={}, start={}, end={})",
                       name, size, start, end);
  }

  void Serialize(BinaryBuffer& buffer) const;

  static TensorDimSpec Deserialize(const BinaryRange& range);

  const TensorDimName name;  ///< Name of the tensor dimension
  const std::size_t size;    ///< Total size of the dimension
  const TensorIndex start;   ///< Start index of owned range (inclusive)
  const TensorIndex end;     ///< End index of owned range (exclusive)
};
//==============================================================================
}  // namespace setu::commons::datatypes
//==============================================================================

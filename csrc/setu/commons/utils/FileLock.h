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
#include "commons/ClassTraits.h"
//==============================================================================
namespace setu::commons::utils {
//==============================================================================
/**
 * @brief Lock mode for file-based locks
 *
 * Maps directly to flock() operation modes.
 */
enum class FileLockMode : std::int32_t {
  kShared = LOCK_SH,     ///< Multiple readers allowed
  kExclusive = LOCK_EX,  ///< Single writer, no readers
};
//==============================================================================
/**
 * @brief RAII wrapper for file-based read/write locks using flock()
 *
 * Provides cross-process locking semantics using POSIX flock(). The lock is
 * acquired in the constructor and released in the destructor. Supports both
 * shared (read) and exclusive (write) modes.
 *
 * This is movable but not copyable (inherits from NonCopyable).
 */
class FileLock : public NonCopyable {
 public:
  /**
   * @brief Opens lock file and acquires flock
   *
   * Creates the lock file and parent directories if they don't exist.
   * Blocks until the lock is acquired.
   *
   * @param lock_file_path Path to the lock file
   * @param mode Lock mode (shared or exclusive)
   */
  FileLock(const std::string& lock_file_path, FileLockMode mode);

  /**
   * @brief Move constructor, transfers ownership of the lock
   */
  FileLock(FileLock&& other) noexcept;

  /**
   * @brief Move assignment, releases current lock and takes ownership
   */
  FileLock& operator=(FileLock&& other) noexcept;

  /**
   * @brief Releases the flock and closes the file descriptor
   */
  ~FileLock();

  /**
   * @brief Get the lock file path
   * @return Const reference to the lock file path
   */
  [[nodiscard]] const std::string& GetFilePath() const;

  /**
   * @brief Get the lock mode
   * @return The lock mode (shared or exclusive)
   */
  [[nodiscard]] FileLockMode GetMode() const;

 private:
  /**
   * @brief Release the flock and close the file descriptor
   */
  void Release();

  std::int32_t fd_;       ///< File descriptor for the lock file
  std::string filepath_;  ///< Path to the lock file
  FileLockMode mode_;     ///< Lock mode (shared or exclusive)
};
//==============================================================================
}  // namespace setu::commons::utils
//==============================================================================

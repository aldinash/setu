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
#include "commons/utils/FileLock.h"
//==============================================================================
#include "commons/Logging.h"
//==============================================================================
namespace setu::commons::utils {
//==============================================================================

FileLock::FileLock(const std::string& lock_file_path, FileLockMode mode)
    : fd_(-1), filepath_(lock_file_path), mode_(mode) {
  // Create parent directories if they don't exist
  std::filesystem::path parent_dir =
      std::filesystem::path(filepath_).parent_path();
  if (!parent_dir.empty() && !std::filesystem::exists(parent_dir)) {
    std::filesystem::create_directories(parent_dir);
  }

  // Open (or create) the lock file
  fd_ = ::open(filepath_.c_str(), O_RDWR | O_CREAT, 0666);
  if (fd_ < 0) {
    RAISE_RUNTIME_ERROR("Failed to open lock file '{}': {}", filepath_,
                        std::strerror(errno));
  }

  // Acquire the flock (blocks until lock is granted)
  std::int32_t flock_op = static_cast<std::int32_t>(mode_);
  if (::flock(fd_, flock_op) != 0) {
    ::close(fd_);
    fd_ = -1;
    RAISE_RUNTIME_ERROR("Failed to acquire flock on '{}': {}", filepath_,
                        std::strerror(errno));
  }

  LOG_DEBUG("Acquired {} flock on '{}'",
            mode_ == FileLockMode::kShared ? "shared" : "exclusive", filepath_);
}

//==============================================================================

FileLock::FileLock(FileLock&& other) noexcept
    : fd_(other.fd_),
      filepath_(std::move(other.filepath_)),
      mode_(other.mode_) {
  other.fd_ = -1;
  other.filepath_.clear();
}

//==============================================================================

FileLock& FileLock::operator=(FileLock&& other) noexcept {
  if (this != &other) {
    Release();
    fd_ = other.fd_;
    filepath_ = std::move(other.filepath_);
    mode_ = other.mode_;
    other.fd_ = -1;
    other.filepath_.clear();
  }
  return *this;
}

//==============================================================================

FileLock::~FileLock() { Release(); }

//==============================================================================

const std::string& FileLock::GetFilePath() const { return filepath_; }

//==============================================================================

FileLockMode FileLock::GetMode() const { return mode_; }

//==============================================================================

void FileLock::Release() {
  if (fd_ >= 0) {
    if (::flock(fd_, LOCK_UN) != 0) {
      LOG_WARNING("Failed to release flock on '{}': {}", filepath_,
                  std::strerror(errno));
    } else {
      LOG_DEBUG("Released flock on '{}'", filepath_);
    }

    if (::close(fd_) < 0) {
      LOG_WARNING("Failed to close lock file descriptor {} for '{}': {}", fd_,
                  filepath_, std::strerror(errno));
    }
    fd_ = -1;
  }
}

//==============================================================================
}  // namespace setu::commons::utils
//==============================================================================

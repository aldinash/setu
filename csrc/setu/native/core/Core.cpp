//==============================================================================
// Copyright 2025 Setu Team, Georgia Institute of Technology
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
#include "Core.h"
//==============================================================================
#include "commons/StdCommon.h"
//==============================================================================
namespace setu::native::core {
//==============================================================================
std::vector<std::string> CoreProcessor::Process(
    const std::vector<std::string>& input) const {
  std::vector<std::string> result;
  result.reserve(input.size());

  for (const auto& item : input) {
    result.push_back("processed: " + item);
    processed_count_.fetch_add(1, std::memory_order_relaxed);
  }

  return result;
}
//==============================================================================
std::size_t CoreProcessor::GetProcessedCount() const noexcept {
  return processed_count_.load(std::memory_order_relaxed);
}
//==============================================================================
}  // namespace setu::native::core
//==============================================================================

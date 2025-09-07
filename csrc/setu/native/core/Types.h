//==============================================================================
// Copyright 2025 Setu Team; Georgia Institute of Technology
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
#include "commons/TorchCommon.h"
//==============================================================================
namespace setu::native::core {
//==============================================================================
using TimeMS = double;  // time in milliseconds
using TimeS = double;   // time in seconds
using Rank = std::size_t;
using TensorSize = std::int64_t;
using SerialNumber = std::uint64_t;
using BinaryBuffer = std::vector<std::uint8_t>;
using BinaryIterator = BinaryBuffer::const_iterator;
using BinaryRange = std::pair<BinaryIterator, BinaryIterator>;
//==============================================================================
}  // namespace setu::native::core
//==============================================================================

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
#include "native/core/Pybind.h"

#include "commons/Logging.h"
#include "commons/TorchCommon.h"
#include "native/utils/Pybind.h"
//==============================================================================
namespace pybind11 {
namespace detail {
/// @brief Type caster for converting Python sets to C++ std::set<int>
///
/// Provides conversion between Python set/frozenset and C++ std::set<int>.
template <>
struct type_caster<std::set<int>> {
 public:
  PYBIND11_TYPE_CASTER(std::set<int>, _("Set[int]"));
  bool load(handle src, bool) {
    if (!py::isinstance<py::set>(src) && !py::isinstance<py::frozenset>(src))
      return false;
    for (auto item : src) {
      if (!py::isinstance<py::int_>(item)) return false;
      value.insert(item.cast<int>());
    }
    return true;
  }
  static handle cast(const std::set<int>& src, return_value_policy, handle) {
    py::set s;
    for (int v : src) s.add(py::cast(v));
    return s.release();
  }
};
}  // namespace detail
}  // namespace pybind11
namespace setu::native {
//==============================================================================
/**
 * @brief Main Pybind11 module initialization
 *
 * Initializes the native Setu Python bindings and all submodules.
 */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  commons::Logger::InitializeLogLevel();

  setu::native::core::InitPybindSubmodule(m);
  setu::native::utils::InitPybindSubmodule(m);
}
//==============================================================================
}  // namespace setu::native
//==============================================================================

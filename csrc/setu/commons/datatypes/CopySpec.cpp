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
#include "commons/datatypes/CopySpec.h"
//==============================================================================
namespace setu::commons::datatypes {
using setu::commons::utils::BinaryBuffer;
using setu::commons::utils::BinaryRange;
using setu::commons::utils::BinaryReader;
using setu::commons::utils::BinaryWriter;
//==============================================================================
void CopySpec::Serialize(BinaryBuffer& buffer) const {
  BinaryWriter writer(buffer);
  writer.WriteFields(src_name, dst_name, src_selection, dst_selection);
}

CopySpec CopySpec::Deserialize(const BinaryRange& range) {
  BinaryReader reader(range);
  auto [src_name_val, dst_name_val, src_selection_val, dst_selection_val] =
      reader.ReadFields<TensorName, TensorName, TensorSelectionPtr,
                        TensorSelectionPtr>();
  return CopySpec(src_name_val, dst_name_val, src_selection_val,
                  dst_selection_val);
}
//==============================================================================
}  // namespace setu::commons::datatypes
//==============================================================================

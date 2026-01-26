#pragma once
#include "commons/StdCommon.h"
#include "commons/datatypes/CopySpec.h"
#include "commons/datatypes/TensorShardRef.h"
#include "commons/enums/Enums.h"
#include "commons/utils/Serialization.h"

namespace setu::commons::messages {
using setu::commons::datatypes::CopySpec;
using setu::commons::datatypes::TensorShardRefPtr;
using setu::commons::enums::ErrorCode;
using setu::commons::utils::BinaryBuffer;
using setu::commons::utils::BinaryRange;
//==============================================================================

struct SubmitCopyRequest {
  SubmitCopyRequest(CopySpec copy_spec_param)
      : copy_spec(std::move(copy_spec_param)) {}

  std::string ToString() const {
    return std::format("SubmitCopyRequest(copy_spec={})", copy_spec);
  }

  void Serialize(BinaryBuffer& buffer) const;

  static SubmitCopyRequest Deserialize(const BinaryRange& range);

  const CopySpec copy_spec;
};
using SubmitCopyRequestPtr = std::shared_ptr<SubmitCopyRequest>;
//==============================================================================
}  // namespace setu::commons::messages
//==============================================================================

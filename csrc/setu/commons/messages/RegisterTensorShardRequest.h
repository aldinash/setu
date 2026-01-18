#pragma once
#include "commons/StdCommon.h"
#include "commons/Types.h"
#include "commons/utils/Serialization.h"

namespace setu::commons::messages {
using setu::commons::TensorName;
using setu::commons::utils::BinaryBuffer;
using setu::commons::utils::BinaryRange;

struct RegisterTensorShardRequest {
  RegisterTensorShardRequest(TensorName tensor_name_param)
      : tensor_name(std::move(tensor_name_param)) {
    ASSERT_VALID_ARGUMENTS(!tensor_name.empty(), "Tensor name cannot be empty");
  }

  std::string ToString() const {
    return std::format("RegisterTensorShardRequest(tensor_name={})",
                       tensor_name);
  }

  void Serialize(BinaryBuffer& buffer) const;

  static RegisterTensorShardRequest Deserialize(const BinaryRange& range);

  const TensorName tensor_name;
};
using RegisterTensorShardRequestPtr =
    std::shared_ptr<RegisterTensorShardRequest>;
//==============================================================================
}  // namespace setu::commons::messages
//==============================================================================

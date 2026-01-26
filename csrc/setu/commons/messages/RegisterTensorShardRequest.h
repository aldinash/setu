#pragma once
#include "commons/StdCommon.h"
#include "commons/Types.h"
#include "commons/datatypes/TensorShardSpec.h"
#include "commons/utils/Serialization.h"

namespace setu::commons::messages {
using setu::commons::datatypes::TensorShardSpec;
using setu::commons::utils::BinaryBuffer;
using setu::commons::utils::BinaryRange;

struct RegisterTensorShardRequest {
  RegisterTensorShardRequest(TensorShardSpec tensor_shard_spec_param)
      : tensor_shard_spec(std::move(tensor_shard_spec_param)) {
    ASSERT_VALID_ARGUMENTS(!tensor_shard_spec.name.empty(),
                           "Tensor name cannot be empty");
  }

  std::string ToString() const {
    return std::format("RegisterTensorShardRequest(tensor_name={})",
                       tensor_shard_spec.name);
  }

  void Serialize(BinaryBuffer& buffer) const;

  static RegisterTensorShardRequest Deserialize(const BinaryRange& range);

  const TensorShardSpec tensor_shard_spec;
};
using RegisterTensorShardRequestPtr =
    std::shared_ptr<RegisterTensorShardRequest>;
//==============================================================================
}  // namespace setu::commons::messages
//==============================================================================

#pragma once
#include "commons/StdCommon.h"
#include "commons/enums/Enums.h"
#include "commons/utils/Serialization.h"

namespace setu::commons::messages {
using setu::commons::enums::ErrorCode;
using setu::commons::utils::BinaryBuffer;
using setu::commons::utils::BinaryRange;

struct RegisterTensorShardResponse {
  RegisterTensorShardResponse(ErrorCode error_code_param)
      : error_code(error_code_param) {}

  std::string ToString() const {
    return std::format("RegisterTensorShardResponse(error_code={})",
                       error_code);
  }

  void Serialize(BinaryBuffer& buffer) const;

  static RegisterTensorShardResponse Deserialize(const BinaryRange& range);

  const ErrorCode error_code;
};
using RegisterTensorShardResponsePtr =
    std::shared_ptr<RegisterTensorShardResponse>;
//==============================================================================
}  // namespace setu::commons::messages
//==============================================================================

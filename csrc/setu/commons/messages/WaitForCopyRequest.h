
#pragma once
#include "commons/StdCommon.h"
#include "commons/Types.h"
#include "commons/enums/Enums.h"
#include "commons/utils/Serialization.h"

namespace setu::commons::messages {
using setu::commons::CopyOperationId;
using setu::commons::enums::ErrorCode;
using setu::commons::utils::BinaryBuffer;
using setu::commons::utils::BinaryRange;
//==============================================================================

struct WaitForCopyRequest {
  WaitForCopyRequest(CopyOperationId copy_operation_id_param)
      : copy_operation_id(std::move(copy_operation_id_param)) {}

  std::string ToString() const {
    return std::format("WaitForCopyRequest(copy_operation_id={})",
                       copy_operation_id);
  }

  void Serialize(BinaryBuffer& buffer) const;

  static WaitForCopyRequest Deserialize(const BinaryRange& range);

  const CopyOperationId copy_operation_id;
};
using WaitForCopyRequestPtr = std::shared_ptr<WaitForCopyRequest>;
//==============================================================================
}  // namespace setu::commons::messages
//==============================================================================

#include "setu/commons/messages/WaitForCopyRequest.h"

namespace setu::commons::messages {
using setu::commons::utils::BinaryBuffer;
using setu::commons::utils::BinaryRange;
using setu::commons::utils::BinaryReader;
using setu::commons::utils::BinaryWriter;
void WaitForCopyRequest::Serialize(BinaryBuffer& buffer) const {
  BinaryWriter writer(buffer);
  writer.WriteFields(copy_operation_id);
}

WaitForCopyRequest WaitForCopyRequest::Deserialize(const BinaryRange& range) {
  BinaryReader reader(range);
  auto [copy_operation_id_val] = reader.ReadFields<CopyOperationId>();

  return WaitForCopyRequest(copy_operation_id_val);
}
}  // namespace setu::commons::messages
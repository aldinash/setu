#include "commons/messages/SubmitCopyRequest.h"

namespace setu::commons::messages {
using setu::commons::utils::BinaryBuffer;
using setu::commons::utils::BinaryRange;
using setu::commons::utils::BinaryReader;
using setu::commons::utils::BinaryWriter;

void SubmitCopyRequest::Serialize(BinaryBuffer& buffer) const {
  BinaryWriter writer(buffer);
  writer.WriteFields(copy_spec);
}

SubmitCopyRequest SubmitCopyRequest::Deserialize(const BinaryRange& range) {
  BinaryReader reader(range);
  auto [copy_spec_val] = reader.ReadFields<CopySpec>();

  return SubmitCopyRequest(copy_spec_val);
}
}  // namespace setu::commons::messages

#include "commons/messages/RegisterTensorShardResponse.h"

namespace setu::commons::messages {
using setu::commons::utils::BinaryBuffer;
using setu::commons::utils::BinaryRange;
using setu::commons::utils::BinaryReader;
using setu::commons::utils::BinaryWriter;

void RegisterTensorShardResponse::Serialize(BinaryBuffer& buffer) const {
  BinaryWriter writer(buffer);
  writer.Write(static_cast<std::uint32_t>(error_code));
}

RegisterTensorShardResponse RegisterTensorShardResponse::Deserialize(
    const BinaryRange& range) {
  BinaryReader reader(range);
  auto error_code_val = reader.Read<std::uint32_t>();

  return RegisterTensorShardResponse(static_cast<ErrorCode>(error_code_val));
}
}  // namespace setu::commons::messages

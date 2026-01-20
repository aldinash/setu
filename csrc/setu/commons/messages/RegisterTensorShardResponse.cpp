#include "commons/messages/RegisterTensorShardResponse.h"

namespace setu::commons::messages {
using setu::commons::utils::BinaryBuffer;
using setu::commons::utils::BinaryRange;
using setu::commons::utils::BinaryReader;
using setu::commons::utils::BinaryWriter;

void RegisterTensorShardResponse::Serialize(BinaryBuffer& buffer) const {
  BinaryWriter writer(buffer);
  writer.WriteFields(error_code, shard_ref);
}

RegisterTensorShardResponse RegisterTensorShardResponse::Deserialize(
    const BinaryRange& range) {
  BinaryReader reader(range);
  auto [error_code_val, shard_ref_val] =
      reader.ReadFields<ErrorCode, std::optional<TensorShardRef>>();

  return RegisterTensorShardResponse(error_code_val, shard_ref_val);
}
}  // namespace setu::commons::messages

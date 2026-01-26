#include "commons/messages/RegisterTensorShardRequest.h"

namespace setu::commons::messages {
using setu::commons::utils::BinaryBuffer;
using setu::commons::utils::BinaryRange;
using setu::commons::utils::BinaryReader;
using setu::commons::utils::BinaryWriter;

void RegisterTensorShardRequest::Serialize(BinaryBuffer& buffer) const {
  BinaryWriter writer(buffer);
  writer.WriteFields(tensor_shard_spec);
}

RegisterTensorShardRequest RegisterTensorShardRequest::Deserialize(
    const BinaryRange& range) {
  BinaryReader reader(range);
  auto [tensor_shard_spec] = reader.ReadFields<TensorShardSpec>();

  return RegisterTensorShardRequest(tensor_shard_spec);
}
}  // namespace setu::commons::messages
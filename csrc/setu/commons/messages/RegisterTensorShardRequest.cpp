#include "commons/messages/RegisterTensorShardRequest.h"

namespace setu::commons::messages {
using setu::commons::utils::BinaryBuffer;
using setu::commons::utils::BinaryRange;
using setu::commons::utils::BinaryReader;
using setu::commons::utils::BinaryWriter;

void RegisterTensorShardRequest::Serialize(BinaryBuffer& buffer) const {
  BinaryWriter writer(buffer);
  writer.WriteFields(tensor_name);
}

RegisterTensorShardRequest RegisterTensorShardRequest::Deserialize(
    const BinaryRange& range) {
  BinaryReader reader(range);
  auto [tensor_name] = reader.ReadFields<TensorName>();

  return RegisterTensorShardRequest(tensor_name);
}
}  // namespace setu::commons::messages
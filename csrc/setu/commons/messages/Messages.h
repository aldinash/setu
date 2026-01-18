#pragma once
//==============================================================================
#include "commons/StdCommon.h"
//==============================================================================
#include "commons/enums/Enums.h"
#include "commons/messages/RegisterTensorShardRequest.h"
#include "commons/messages/RegisterTensorShardResponse.h"
#include "commons/utils/Serialization.h"
//==============================================================================
namespace setu::commons::messages {
//==============================================================================
using setu::commons::enums::MsgType;
using setu::commons::utils::BinaryBuffer;
using setu::commons::utils::BinaryRange;
using setu::commons::utils::BinaryReader;
using setu::commons::utils::BinaryWriter;
//==============================================================================
struct Header {
  MsgType msg_type;

  void Serialize(BinaryBuffer& buffer) const {
    BinaryWriter writer(buffer);
    writer.Write(static_cast<std::uint16_t>(msg_type));
  }

  static Header Deserialize(const BinaryRange& range) {
    BinaryReader reader(range);
    const auto msg_type_val = reader.Read<std::uint16_t>();
    return Header{.msg_type = static_cast<MsgType>(msg_type_val)};
  }

  std::string ToString() const {
    return std::format("Header(msg_type={})",
                       static_cast<std::uint16_t>(msg_type));
  }
};
//==============================================================================
using Request = std::variant<RegisterTensorShardRequest>;
//==============================================================================
}  // namespace setu::commons::messages
//==============================================================================
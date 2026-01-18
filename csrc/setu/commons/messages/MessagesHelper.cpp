#include "commons/ClassTraits.h"
#include "commons/Logging.h"
#include "commons/ZmqCommon.h"
//==============================================================================
#include "commons/enums/Enums.h"
#include "commons/messages/Messages.h"
#include "commons/utils/Serialization.h"
#include "commons/utils/ZmqHelper.h"
//==============================================================================
namespace setu::commons::messages {
//==============================================================================
using setu::commons::enums::MsgType;
using setu::commons::utils::BinaryBuffer;
using setu::commons::utils::Serializable;
using setu::commons::utils::Serialize;
using setu::commons::utils::ZmqSocketPtr;
//==============================================================================
using ClientIdentity = std::string;
//==============================================================================
struct ReceivedMessage {
  Header header;
  BinaryBuffer body;
};
//==============================================================================
struct ReceivedRouterMessage {
  ClientIdentity identity;
  Header header;
  BinaryBuffer body;
};
//==============================================================================
class MessagesHelper : public NonCopyableNonMovable {
 public:
  template <Serializable T>
  static void SendRequest(ZmqSocketPtr socket, MsgType msg_type,
                          const T& message) {
    ASSERT_VALID_POINTER_ARGUMENT(socket);

    // Send header frame with SNDMORE flag
    Header header{.msg_type = msg_type};
    SendFrame(socket, header, zmq::send_flags::sndmore);

    // Send body frame (last frame, no SNDMORE)
    SendFrame(socket, message, zmq::send_flags::none);
  }

  [[nodiscard]] static ReceivedMessage RecvRequest(ZmqSocketPtr socket) {
    ASSERT_VALID_POINTER_ARGUMENT(socket);

    // Receive header frame
    zmq::message_t header_msg;
    auto result = socket->recv(header_msg, zmq::recv_flags::none);
    ASSERT_VALID_RUNTIME(result.has_value(), "Failed to receive header frame");
    ASSERT_VALID_RUNTIME(header_msg.more(),
                         "Expected multipart message, but header was last");

    Header header = DeserializeFrame<Header>(header_msg);

    // Receive body frame
    zmq::message_t body_msg;
    result = socket->recv(body_msg, zmq::recv_flags::none);
    ASSERT_VALID_RUNTIME(result.has_value(), "Failed to receive body frame");

    BinaryBuffer body(
        static_cast<const std::uint8_t*>(body_msg.data()),
        static_cast<const std::uint8_t*>(body_msg.data()) + body_msg.size());

    return ReceivedMessage{.header = header, .body = std::move(body)};
  }

  template <Serializable T>
  static void SendResponse(ZmqSocketPtr socket, MsgType msg_type,
                           const T& response) {
    // Response format is identical to request
    SendRequest(socket, msg_type, response);
  }

  [[nodiscard]] static ReceivedMessage RecvResponse(ZmqSocketPtr socket) {
    // Response format is identical to request
    return RecvRequest(socket);
  }

  template <Serializable T>
  static void SendToClient(ZmqSocketPtr socket, const ClientIdentity& identity,
                           MsgType msg_type, const T& message) {
    ASSERT_VALID_POINTER_ARGUMENT(socket);
    ASSERT_VALID_ARGUMENTS(!identity.empty(),
                           "Client identity cannot be empty");

    zmq::message_t identity_msg(identity.data(), identity.size());
    auto result = socket->send(identity_msg, zmq::send_flags::sndmore);
    ASSERT_VALID_RUNTIME(result.has_value(), "Failed to send identity frame");

    zmq::message_t delimiter_msg;
    result = socket->send(delimiter_msg, zmq::send_flags::sndmore);
    ASSERT_VALID_RUNTIME(result.has_value(), "Failed to send delimiter frame");

    Header header{.msg_type = msg_type};
    SendFrame(socket, header, zmq::send_flags::sndmore);

    SendFrame(socket, message, zmq::send_flags::none);
  }

  [[nodiscard]] static ReceivedRouterMessage RecvFromClient(
      ZmqSocketPtr socket) {
    ASSERT_VALID_POINTER_ARGUMENT(socket);

    zmq::message_t identity_msg;
    auto result = socket->recv(identity_msg, zmq::recv_flags::none);
    ASSERT_VALID_RUNTIME(result.has_value(),
                         "Failed to receive identity frame");
    ASSERT_VALID_RUNTIME(identity_msg.more(),
                         "Expected multipart message, but identity was last");

    ClientIdentity identity(static_cast<const char*>(identity_msg.data()),
                            identity_msg.size());

    zmq::message_t delimiter_msg;
    result = socket->recv(delimiter_msg, zmq::recv_flags::none);
    ASSERT_VALID_RUNTIME(result.has_value(),
                         "Failed to receive delimiter frame");
    ASSERT_VALID_RUNTIME(delimiter_msg.more(),
                         "Expected multipart message, but delimiter was last");

    zmq::message_t header_msg;
    result = socket->recv(header_msg, zmq::recv_flags::none);
    ASSERT_VALID_RUNTIME(result.has_value(), "Failed to receive header frame");
    ASSERT_VALID_RUNTIME(header_msg.more(),
                         "Expected multipart message, but header was last");

    Header header = DeserializeFrame<Header>(header_msg);

    zmq::message_t body_msg;
    result = socket->recv(body_msg, zmq::recv_flags::none);
    ASSERT_VALID_RUNTIME(result.has_value(), "Failed to receive body frame");

    BinaryBuffer body(
        static_cast<const std::uint8_t*>(body_msg.data()),
        static_cast<const std::uint8_t*>(body_msg.data()) + body_msg.size());

    return ReceivedRouterMessage{.identity = std::move(identity),
                                 .header = header,
                                 .body = std::move(body)};
  }

  template <Serializable T>
  static void SendViaDealer(ZmqSocketPtr socket, MsgType msg_type,
                            const T& message) {
    // DEALER sends same format as REQ, identity is handled automatically
    SendRequest(socket, msg_type, message);
  }

  [[nodiscard]] static ReceivedMessage RecvViaDealer(ZmqSocketPtr socket) {
    // DEALER receives same format as REQ
    return RecvRequest(socket);
  }

  [[nodiscard]] static std::optional<ReceivedMessage> TryRecvRequest(
      ZmqSocketPtr socket) {
    ASSERT_VALID_POINTER_ARGUMENT(socket);

    zmq::message_t header_msg;
    auto result = socket->recv(header_msg, zmq::recv_flags::dontwait);
    if (!result.has_value()) {
      return std::nullopt;
    }

    ASSERT_VALID_RUNTIME(header_msg.more(),
                         "Expected multipart message, but header was last");

    Header header = DeserializeFrame<Header>(header_msg);

    // Must receive body since header indicated more frames
    zmq::message_t body_msg;
    result = socket->recv(body_msg, zmq::recv_flags::none);
    ASSERT_VALID_RUNTIME(result.has_value(), "Failed to receive body frame");

    BinaryBuffer body(
        static_cast<const std::uint8_t*>(body_msg.data()),
        static_cast<const std::uint8_t*>(body_msg.data()) + body_msg.size());

    return ReceivedMessage{.header = header, .body = std::move(body)};
  }

  [[nodiscard]] static std::optional<ReceivedRouterMessage> TryRecvFromClient(
      ZmqSocketPtr socket) {
    ASSERT_VALID_POINTER_ARGUMENT(socket);

    zmq::message_t identity_msg;
    auto result = socket->recv(identity_msg, zmq::recv_flags::dontwait);
    if (!result.has_value()) {
      return std::nullopt;
    }

    ASSERT_VALID_RUNTIME(identity_msg.more(),
                         "Expected multipart message, but identity was last");

    ClientIdentity identity(static_cast<const char*>(identity_msg.data()),
                            identity_msg.size());

    zmq::message_t delimiter_msg;
    result = socket->recv(delimiter_msg, zmq::recv_flags::none);
    ASSERT_VALID_RUNTIME(result.has_value(),
                         "Failed to receive delimiter frame");
    ASSERT_VALID_RUNTIME(delimiter_msg.more(),
                         "Expected multipart message, but delimiter was last");

    zmq::message_t header_msg;
    result = socket->recv(header_msg, zmq::recv_flags::none);
    ASSERT_VALID_RUNTIME(result.has_value(), "Failed to receive header frame");
    ASSERT_VALID_RUNTIME(header_msg.more(),
                         "Expected multipart message, but header was last");

    Header header = DeserializeFrame<Header>(header_msg);

    zmq::message_t body_msg;
    result = socket->recv(body_msg, zmq::recv_flags::none);
    ASSERT_VALID_RUNTIME(result.has_value(), "Failed to receive body frame");

    BinaryBuffer body(
        static_cast<const std::uint8_t*>(body_msg.data()),
        static_cast<const std::uint8_t*>(body_msg.data()) + body_msg.size());

    return ReceivedRouterMessage{.identity = std::move(identity),
                                 .header = header,
                                 .body = std::move(body)};
  }

  [[nodiscard]] static Request DecodeRequest(const Header& header,
                                             const BinaryBuffer& body_bytes) {
    const setu::commons::utils::BinaryRange range(body_bytes.begin(),
                                                  body_bytes.end());
    switch (header.msg_type) {
      case MsgType::kRegisterTensorShardRequest: {
        return RegisterTensorShardRequest::Deserialize(range);
      }
      default:
        RAISE_RUNTIME_ERROR(
            "Unknown MsgType {} in DecodeRequest",
            static_cast<std::underlying_type_t<MsgType>>(header.msg_type));
    }
  }

  [[nodiscard]] static Request DecodeRequest(const ReceivedMessage& msg) {
    return DecodeRequest(msg.header, msg.body);
  }

  [[nodiscard]] static Request DecodeRequest(const ReceivedRouterMessage& msg) {
    return DecodeRequest(msg.header, msg.body);
  }

 private:
  MessagesHelper() = delete;
  ~MessagesHelper() = delete;

  template <Serializable T>
  static void SendFrame(ZmqSocketPtr socket, const T& obj,
                        zmq::send_flags flags) {
    const auto serialized_data = Serialize(obj);
    zmq::message_t message(serialized_data.size());
    std::memcpy(message.data(), serialized_data.data(), serialized_data.size());

    const auto result = socket->send(message, flags);
    ASSERT_VALID_RUNTIME(result.has_value(), "Failed to send frame of size {}",
                         serialized_data.size());
  }

  template <Serializable T>
  [[nodiscard]] static T DeserializeFrame(const zmq::message_t& msg) {
    const auto* data = static_cast<const std::uint8_t*>(msg.data());
    const std::span<const std::uint8_t> message_span(data, msg.size());
    return setu::commons::utils::Deserialize<T>(message_span);
  }
};
//==============================================================================
}  // namespace setu::commons::messages
//==============================================================================
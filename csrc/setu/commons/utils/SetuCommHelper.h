#pragma once
//==============================================================================
#include "commons/ClassTraits.h"
#include "commons/Logging.h"
#include "commons/Types.h"
#include "commons/ZmqCommon.h"
//==============================================================================
#include "commons/enums/Enums.h"
#include "commons/messages/Messages.h"
#include "commons/utils/Serialization.h"
#include "commons/utils/ZmqHelper.h"
//==============================================================================
namespace setu::commons::utils {
//==============================================================================
using setu::commons::ClientIdentity;
using setu::commons::NonCopyableNonMovable;
using setu::commons::enums::MsgType;
using setu::commons::messages::AllocateTensorRequest;
using setu::commons::messages::AnyClientRequest;
using setu::commons::messages::AnyCoordinatorRequest;
using setu::commons::messages::CopyOperationFinishedRequest;
using setu::commons::messages::ExecuteRequest;
using setu::commons::messages::Header;
using setu::commons::messages::MsgTypeFor;
using setu::commons::messages::RegisterTensorShardRequest;
using setu::commons::messages::SubmitCopyRequest;
using setu::commons::messages::WaitForCopyRequest;
//==============================================================================
// SetuCommHelper - Static helper for Setu protocol communication over ZMQ
//
// Supported socket patterns:
//   - REQ/REP/DEALER: Use Send(), Recv<T>(), TryRecv<T>(), TryRecvRequest()
//   - ROUTER: Use SendToClient(), RecvRequestFromClient(),
//   TryRecvRequestFromClient()
//
// The header (MsgType) is handled internally - callers just work with typed
// request/response objects.
//==============================================================================
class SetuCommHelper : public NonCopyableNonMovable {
 public:
  //============================================================================
  // REQ/REP/DEALER pattern: [Header][Body]
  // Works for REQ, REP, and DEALER sockets
  //============================================================================

  /// @brief Send a typed message (MsgType deduced automatically from T)
  template <Serializable T>
  static void Send(ZmqSocketPtr socket, const T& message) {
    ASSERT_VALID_POINTER_ARGUMENT(socket);

    constexpr MsgType msg_type = MsgTypeFor<T>::value;
    Header header{.msg_type = msg_type};
    SendFrame(socket, header, zmq::send_flags::sndmore);
    SendFrame(socket, message, zmq::send_flags::none);
  }

  /// @brief Receive a typed message (blocking)
  template <Serializable T>
  [[nodiscard]] static T Recv(ZmqSocketPtr socket) {
    ASSERT_VALID_POINTER_ARGUMENT(socket);

    // Receive and discard header (internal detail)
    zmq::message_t header_msg;
    auto result = socket->recv(header_msg, zmq::recv_flags::none);
    ASSERT_VALID_RUNTIME(result.has_value(), "Failed to receive header frame");
    ASSERT_VALID_RUNTIME(header_msg.more(),
                         "Expected multipart message, but header was last");

    // Receive and deserialize body to typed T
    zmq::message_t body_msg;
    result = socket->recv(body_msg, zmq::recv_flags::none);
    ASSERT_VALID_RUNTIME(result.has_value(), "Failed to receive body frame");

    return DeserializeFrame<T>(body_msg);
  }

  /// @brief Try to receive a typed message (non-blocking)
  template <Serializable T>
  [[nodiscard]] static std::optional<T> TryRecv(ZmqSocketPtr socket) {
    ASSERT_VALID_POINTER_ARGUMENT(socket);

    // Try to receive header (non-blocking)
    zmq::message_t header_msg;
    auto result = socket->recv(header_msg, zmq::recv_flags::dontwait);
    if (!result.has_value()) {
      return std::nullopt;
    }

    ASSERT_VALID_RUNTIME(header_msg.more(),
                         "Expected multipart message, but header was last");

    // Receive body (blocking - we already started receiving)
    zmq::message_t body_msg;
    result = socket->recv(body_msg, zmq::recv_flags::none);
    ASSERT_VALID_RUNTIME(result.has_value(), "Failed to receive body frame");

    return DeserializeFrame<T>(body_msg);
  }

  //============================================================================
  // ROUTER pattern: [Identity][Delimiter][Header][Body]
  // ROUTER sockets must explicitly handle client identity
  //============================================================================

  /// @brief Send a typed response to a specific client (MsgType deduced from T)
  template <Serializable T>
  static void SendToClient(ZmqSocketPtr socket, const ClientIdentity& identity,
                           const T& message) {
    ASSERT_VALID_POINTER_ARGUMENT(socket);
    ASSERT_VALID_ARGUMENTS(!identity.empty(),
                           "Client identity cannot be empty");

    // Identity frame
    zmq::message_t identity_msg(identity.data(), identity.size());
    auto result = socket->send(identity_msg, zmq::send_flags::sndmore);
    ASSERT_VALID_RUNTIME(result.has_value(), "Failed to send identity frame");

    // Empty delimiter frame
    zmq::message_t delimiter_msg;
    result = socket->send(delimiter_msg, zmq::send_flags::sndmore);
    ASSERT_VALID_RUNTIME(result.has_value(), "Failed to send delimiter frame");

    // Header and body frames (MsgType deduced from T)
    constexpr MsgType msg_type = MsgTypeFor<T>::value;
    Header header{.msg_type = msg_type};
    SendFrame(socket, header, zmq::send_flags::sndmore);
    SendFrame(socket, message, zmq::send_flags::none);
  }

  /// @brief Receive a request from a client (blocking)
  /// @return Tuple of (client identity, request variant)
  [[nodiscard]] static std::tuple<ClientIdentity, AnyClientRequest>
  RecvRequestFromClient(ZmqSocketPtr socket) {
    ASSERT_VALID_POINTER_ARGUMENT(socket);

    // Identity frame
    zmq::message_t identity_msg;
    auto result = socket->recv(identity_msg, zmq::recv_flags::none);
    ASSERT_VALID_RUNTIME(result.has_value(),
                         "Failed to receive identity frame");
    ASSERT_VALID_RUNTIME(identity_msg.more(),
                         "Expected multipart message, but identity was last");

    ClientIdentity identity(static_cast<const char*>(identity_msg.data()),
                            identity_msg.size());

    // Delimiter frame
    zmq::message_t delimiter_msg;
    result = socket->recv(delimiter_msg, zmq::recv_flags::none);
    ASSERT_VALID_RUNTIME(result.has_value(),
                         "Failed to receive delimiter frame");
    ASSERT_VALID_RUNTIME(delimiter_msg.more(),
                         "Expected multipart message, but delimiter was last");

    // Header frame
    zmq::message_t header_msg;
    result = socket->recv(header_msg, zmq::recv_flags::none);
    ASSERT_VALID_RUNTIME(result.has_value(), "Failed to receive header frame");
    ASSERT_VALID_RUNTIME(header_msg.more(),
                         "Expected multipart message, but header was last");

    Header header = DeserializeFrame<Header>(header_msg);

    // Body frame
    zmq::message_t body_msg;
    result = socket->recv(body_msg, zmq::recv_flags::none);
    ASSERT_VALID_RUNTIME(result.has_value(), "Failed to receive body frame");

    // Deserialize to correct type based on header
    AnyClientRequest request =
        DeserializeClientRequest(header.msg_type, body_msg);

    return {std::move(identity), std::move(request)};
  }

  /// @brief Try to receive a request from a client (non-blocking)
  /// @return Optional tuple of (client identity, request variant)
  [[nodiscard]] static std::optional<
      std::tuple<ClientIdentity, AnyClientRequest>>
  TryRecvRequestFromClient(ZmqSocketPtr socket) {
    ASSERT_VALID_POINTER_ARGUMENT(socket);

    // Identity frame (non-blocking)
    zmq::message_t identity_msg;
    auto result = socket->recv(identity_msg, zmq::recv_flags::dontwait);
    if (!result.has_value()) {
      return std::nullopt;
    }

    ASSERT_VALID_RUNTIME(identity_msg.more(),
                         "Expected multipart message, but identity was last");

    ClientIdentity identity(static_cast<const char*>(identity_msg.data()),
                            identity_msg.size());

    // Delimiter frame (blocking - we already started receiving)
    zmq::message_t delimiter_msg;
    result = socket->recv(delimiter_msg, zmq::recv_flags::none);
    ASSERT_VALID_RUNTIME(result.has_value(),
                         "Failed to receive delimiter frame");
    ASSERT_VALID_RUNTIME(delimiter_msg.more(),
                         "Expected multipart message, but delimiter was last");

    // Header frame
    zmq::message_t header_msg;
    result = socket->recv(header_msg, zmq::recv_flags::none);
    ASSERT_VALID_RUNTIME(result.has_value(), "Failed to receive header frame");
    ASSERT_VALID_RUNTIME(header_msg.more(),
                         "Expected multipart message, but header was last");

    Header header = DeserializeFrame<Header>(header_msg);

    // Body frame
    zmq::message_t body_msg;
    result = socket->recv(body_msg, zmq::recv_flags::none);
    ASSERT_VALID_RUNTIME(result.has_value(), "Failed to receive body frame");

    // Deserialize to correct type based on header
    AnyClientRequest request =
        DeserializeClientRequest(header.msg_type, body_msg);

    return std::make_tuple(std::move(identity), std::move(request));
  }

  //============================================================================
  // DEALER pattern for receiving coordinator requests: [Header][Body]
  // Used when DEALER socket receives requests from coordinator
  //============================================================================

  /// @brief Receive a coordinator request from a DEALER socket (blocking)
  /// @return The coordinator request variant
  [[nodiscard]] static AnyCoordinatorRequest RecvCoordinatorRequest(
      ZmqSocketPtr socket) {
    ASSERT_VALID_POINTER_ARGUMENT(socket);

    // Header frame
    zmq::message_t header_msg;
    auto result = socket->recv(header_msg, zmq::recv_flags::none);
    ASSERT_VALID_RUNTIME(result.has_value(), "Failed to receive header frame");
    ASSERT_VALID_RUNTIME(header_msg.more(),
                         "Expected multipart message, but header was last");

    Header header = DeserializeFrame<Header>(header_msg);

    // Body frame
    zmq::message_t body_msg;
    result = socket->recv(body_msg, zmq::recv_flags::none);
    ASSERT_VALID_RUNTIME(result.has_value(), "Failed to receive body frame");

    return DeserializeCoordinatorRequest(header.msg_type, body_msg);
  }

  /// @brief Try to receive a coordinator request from a DEALER socket
  /// (non-blocking)
  /// @return Optional coordinator request variant
  [[nodiscard]] static std::optional<AnyCoordinatorRequest>
  TryRecvCoordinatorRequest(ZmqSocketPtr socket) {
    ASSERT_VALID_POINTER_ARGUMENT(socket);

    // Header frame (non-blocking)
    zmq::message_t header_msg;
    auto result = socket->recv(header_msg, zmq::recv_flags::dontwait);
    if (!result.has_value()) {
      return std::nullopt;
    }

    ASSERT_VALID_RUNTIME(header_msg.more(),
                         "Expected multipart message, but header was last");

    Header header = DeserializeFrame<Header>(header_msg);

    // Body frame (blocking - we already started receiving)
    zmq::message_t body_msg;
    result = socket->recv(body_msg, zmq::recv_flags::none);
    ASSERT_VALID_RUNTIME(result.has_value(), "Failed to receive body frame");

    return DeserializeCoordinatorRequest(header.msg_type, body_msg);
  }

  //============================================================================
  // ROUTER pattern for receiving from DEALER: [Identity][Header][Body]
  // Used when ROUTER socket receives from DEALER (no delimiter frame)
  // This is different from REQ→ROUTER which has
  // [Identity][Delimiter][Header][Body]
  //============================================================================

  /// @brief Try to receive a client request from a DEALER via ROUTER socket
  /// (non-blocking)
  /// @note DEALER→ROUTER has no delimiter frame, unlike REQ→ROUTER
  /// @return Optional tuple of (node agent identity, request variant)
  [[nodiscard]] static std::optional<
      std::tuple<ClientIdentity, AnyClientRequest>>
  TryRecvRequestFromNodeAgent(ZmqSocketPtr socket) {
    ASSERT_VALID_POINTER_ARGUMENT(socket);

    // Identity frame (non-blocking)
    zmq::message_t identity_msg;
    auto result = socket->recv(identity_msg, zmq::recv_flags::dontwait);
    if (!result.has_value()) {
      return std::nullopt;
    }

    ASSERT_VALID_RUNTIME(identity_msg.more(),
                         "Expected multipart message, but identity was last");

    ClientIdentity identity(static_cast<const char*>(identity_msg.data()),
                            identity_msg.size());

    // Header frame (blocking - we already started receiving)
    // NOTE: No delimiter frame in DEALER→ROUTER pattern
    zmq::message_t header_msg;
    result = socket->recv(header_msg, zmq::recv_flags::none);
    ASSERT_VALID_RUNTIME(result.has_value(), "Failed to receive header frame");
    ASSERT_VALID_RUNTIME(header_msg.more(),
                         "Expected multipart message, but header was last");

    Header header = DeserializeFrame<Header>(header_msg);

    // Body frame
    zmq::message_t body_msg;
    result = socket->recv(body_msg, zmq::recv_flags::none);
    ASSERT_VALID_RUNTIME(result.has_value(), "Failed to receive body frame");

    // Deserialize to correct type based on header
    AnyClientRequest request =
        DeserializeClientRequest(header.msg_type, body_msg);

    return std::make_tuple(std::move(identity), std::move(request));
  }

  /// @brief Send a response to a NODE AGENT via ROUTER socket
  /// @note NODE AGENT→ROUTER response has no delimiter frame
  template <Serializable T>
  static void SendToNodeAgent(ZmqSocketPtr socket,
                              const ClientIdentity& identity,
                              const T& message) {
    ASSERT_VALID_POINTER_ARGUMENT(socket);
    ASSERT_VALID_ARGUMENTS(!identity.empty(),
                           "Dealer identity cannot be empty");

    // Identity frame
    zmq::message_t identity_msg(identity.data(), identity.size());
    auto result = socket->send(identity_msg, zmq::send_flags::sndmore);
    ASSERT_VALID_RUNTIME(result.has_value(), "Failed to send identity frame");

    // NOTE: No delimiter frame in DEALER→ROUTER pattern

    // Header + Body frames
    constexpr MsgType msg_type = MsgTypeFor<T>::value;
    Header header{.msg_type = msg_type};
    SendFrame(socket, header, zmq::send_flags::sndmore);
    SendFrame(socket, message, zmq::send_flags::none);
  }

 private:
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
  static T DeserializeFrame(const zmq::message_t& msg) {
    const auto* data = static_cast<const std::uint8_t*>(msg.data());
    const std::span<const std::uint8_t> message_span(data, msg.size());
    return Deserialize<T>(message_span);
  }

  /// @brief Deserialize client request based on MsgType
  [[nodiscard]] static AnyClientRequest DeserializeClientRequest(
      MsgType msg_type, const zmq::message_t& body_msg) {
    const auto* data = static_cast<const std::uint8_t*>(body_msg.data());
    const BinaryRange range(data, data + body_msg.size());

    switch (msg_type) {
      case MsgType::kRegisterTensorShardRequest:
        return RegisterTensorShardRequest::Deserialize(range);
      case MsgType::kSubmitCopyRequest:
        return SubmitCopyRequest::Deserialize(range);
      case MsgType::kWaitForCopyRequest:
        return WaitForCopyRequest::Deserialize(range);
      default:
        RAISE_RUNTIME_ERROR(
            "Unknown client request MsgType {} in DeserializeClientRequest",
            static_cast<std::underlying_type_t<MsgType>>(msg_type));
    }
  }

  /// @brief Deserialize coordinator request based on MsgType
  [[nodiscard]] static AnyCoordinatorRequest DeserializeCoordinatorRequest(
      MsgType msg_type, const zmq::message_t& body_msg) {
    const auto* data = static_cast<const std::uint8_t*>(body_msg.data());
    const BinaryRange range(data, data + body_msg.size());

    switch (msg_type) {
      case MsgType::kAllocateTensorRequest:
        return AllocateTensorRequest::Deserialize(range);
      case MsgType::kCopyOperationFinishedRequest:
        return CopyOperationFinishedRequest::Deserialize(range);
      case MsgType::kExecuteRequest:
        return ExecuteRequest::Deserialize(range);
      default:
        RAISE_RUNTIME_ERROR(
            "Unknown coordinator request MsgType {} in "
            "DeserializeCoordinatorRequest",
            static_cast<std::underlying_type_t<MsgType>>(msg_type));
    }
  }
};
//==============================================================================
}  // namespace setu::commons::utils
//==============================================================================

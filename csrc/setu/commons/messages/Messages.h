//==============================================================================
// Copyright (c) 2025 Vajra Team; Georgia Institute of Technology; Microsoft
// Corporation.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//==============================================================================
#pragma once
//==============================================================================
#include "commons/StdCommon.h"
#include "commons/Types.h"
//==============================================================================
#include "commons/enums/Enums.h"
#include "commons/messages/AllocateTensorRequest.h"
#include "commons/messages/AllocateTensorResponse.h"
#include "commons/messages/BaseResponse.h"
#include "commons/messages/CopyOperationFinishedRequest.h"
#include "commons/messages/CopyOperationFinishedResponse.h"
#include "commons/messages/ExecuteProgramRequest.h"
#include "commons/messages/ExecuteProgramResponse.h"
#include "commons/messages/ExecuteRequest.h"
#include "commons/messages/ExecuteResponse.h"
#include "commons/messages/RegisterTensorShardRequest.h"
#include "commons/messages/RegisterTensorShardResponse.h"
#include "commons/messages/SubmitCopyRequest.h"
#include "commons/messages/SubmitCopyResponse.h"
#include "commons/messages/WaitForCopyRequest.h"
#include "commons/messages/WaitForCopyResponse.h"
#include "commons/utils/Serialization.h"
//==============================================================================
namespace setu::commons::messages {
//==============================================================================
using setu::commons::BinaryBuffer;
using setu::commons::ClientIdentity;
using setu::commons::enums::MsgType;
using setu::commons::utils::BinaryRange;
using setu::commons::utils::BinaryReader;
using setu::commons::utils::BinaryWriter;
//==============================================================================
// Header - Wire format for message type identification (internal use)
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
// MsgTypeFor<T> - Compile-time mapping from message type to MsgType enum
//==============================================================================
template <typename T>
struct MsgTypeFor;

// Request type mappings
template <>
struct MsgTypeFor<RegisterTensorShardRequest> {
  static constexpr MsgType value = MsgType::kRegisterTensorShardRequest;
};

template <>
struct MsgTypeFor<SubmitCopyRequest> {
  static constexpr MsgType value = MsgType::kSubmitCopyRequest;
};

template <>
struct MsgTypeFor<WaitForCopyRequest> {
  static constexpr MsgType value = MsgType::kWaitForCopyRequest;
};

template <>
struct MsgTypeFor<AllocateTensorRequest> {
  static constexpr MsgType value = MsgType::kAllocateTensorRequest;
};

template <>
struct MsgTypeFor<CopyOperationFinishedRequest> {
  static constexpr MsgType value = MsgType::kCopyOperationFinishedRequest;
};

template <>
struct MsgTypeFor<ExecuteRequest> {
  static constexpr MsgType value = MsgType::kExecuteRequest;
};

template <>
struct MsgTypeFor<ExecuteProgramRequest> {
  static constexpr MsgType value = MsgType::kExecuteProgramRequest;
};

// Response type mappings
template <>
struct MsgTypeFor<RegisterTensorShardResponse> {
  static constexpr MsgType value = MsgType::kRegisterTensorShardResponse;
};

template <>
struct MsgTypeFor<SubmitCopyResponse> {
  static constexpr MsgType value = MsgType::kSubmitCopyResponse;
};

template <>
struct MsgTypeFor<WaitForCopyResponse> {
  static constexpr MsgType value = MsgType::kWaitForCopyResponse;
};

template <>
struct MsgTypeFor<AllocateTensorResponse> {
  static constexpr MsgType value = MsgType::kAllocateTensorResponse;
};

template <>
struct MsgTypeFor<CopyOperationFinishedResponse> {
  static constexpr MsgType value = MsgType::kCopyOperationFinishedResponse;
};

template <>
struct MsgTypeFor<ExecuteResponse> {
  static constexpr MsgType value = MsgType::kExecuteResponse;
};

template <>
struct MsgTypeFor<ExecuteProgramResponse> {
  static constexpr MsgType value = MsgType::kExecuteProgramResponse;
};

//==============================================================================
// Request variants by source
//==============================================================================
// Requests from clients to NodeAgent
using AnyClientRequest = std::variant<RegisterTensorShardRequest,
                                      SubmitCopyRequest, WaitForCopyRequest>;

// Requests from coordinator to NodeAgent
using AnyCoordinatorRequest =
    std::variant<AllocateTensorRequest, CopyOperationFinishedRequest,
                 ExecuteRequest>;

// All request types (for generic handling if needed)
using AnyRequest = std::variant<RegisterTensorShardRequest, SubmitCopyRequest,
                                WaitForCopyRequest, AllocateTensorRequest,
                                CopyOperationFinishedRequest, ExecuteRequest>;
//==============================================================================
// AnyResponse - Variant of all response types
//==============================================================================
using AnyResponse =
    std::variant<RegisterTensorShardResponse, SubmitCopyResponse,
                 WaitForCopyResponse, AllocateTensorResponse,
                 CopyOperationFinishedResponse, ExecuteResponse>;
//==============================================================================
// Helper for std::visit with lambdas
//==============================================================================
template <class... Ts>
struct Overloaded : Ts... {
  using Ts::operator()...;
};
template <class... Ts>
Overloaded(Ts...) -> Overloaded<Ts...>;
//==============================================================================
}  // namespace setu::commons::messages
//==============================================================================

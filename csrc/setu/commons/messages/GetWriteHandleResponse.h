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
//==============================================================================
#include "commons/Types.h"
#include "commons/messages/BaseResponse.h"
#include "commons/utils/Serialization.h"
#include "commons/utils/TorchTensorIPC.h"
//==============================================================================
namespace setu::commons::messages {
//==============================================================================
using setu::commons::RequestId;
using setu::commons::utils::BinaryBuffer;
using setu::commons::utils::BinaryRange;
using setu::commons::utils::TensorIPCSpec;
//==============================================================================

/// @brief Response containing the IPC handle for write access to a tensor
/// shard. The server has acquired the exclusive write lock and will hold it
/// until ReleaseWriteHandleRequest is received.
struct GetWriteHandleResponse : public BaseResponse {
  GetWriteHandleResponse(
      RequestId request_id_param,
      ErrorCode error_code_param = ErrorCode::kSuccess,
      std::optional<TensorIPCSpec> tensor_ipc_spec_param = std::nullopt)
      : BaseResponse(request_id_param, error_code_param),
        tensor_ipc_spec(std::move(tensor_ipc_spec_param)) {}

  [[nodiscard]] std::string ToString() const {
    return std::format("GetWriteHandleResponse(request_id={}, error_code={})",
                       request_id, error_code);
  }

  void Serialize(BinaryBuffer& buffer) const;

  static GetWriteHandleResponse Deserialize(const BinaryRange& range);

  const std::optional<TensorIPCSpec> tensor_ipc_spec;
};
using GetWriteHandleResponsePtr = std::shared_ptr<GetWriteHandleResponse>;

//==============================================================================
}  // namespace setu::commons::messages
//==============================================================================

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
#include "commons/BoostCommon.h"
#include "commons/StdCommon.h"
//==============================================================================
#include "commons/Types.h"
//==============================================================================
namespace setu::commons::utils {
//==============================================================================

enum class AddWaiterResult : std::uint8_t {
  kWaiterAdded,
  kAlreadyComplete,
  kNotRegistered
};

/// @brief Tracks pending async operations with registration, waiting, and
/// completion.
///
/// Operations are registered via RegisterOperation (refcounted — multiple
/// registrations for the same key are allowed). Clients wait via AddWaiter.
/// When the operation completes, MarkComplete + DrainWaiters serves existing
/// waiters. Late waiters (arriving after completion) get kAlreadyComplete from
/// AddWaiter, which drains one completion. The key is fully cleaned up once all
/// registrations have been drained.
///
/// When PayloadType is not void, each operation can carry associated data that
/// is stored on registration and consumed on completion via ConsumePayload.
///
/// @tparam KeyType The key type (e.g., CopyOperationId, ShardId).
/// @tparam PayloadType Optional payload type to associate with each operation.
///         Defaults to void (no payload, zero overhead).
/// @tparam Hash The hash function for KeyType (defaults to boost::hash).
template <typename KeyType, typename PayloadType = void,
          typename Hash = boost::hash<KeyType>>
class PendingOperations {
 public:
  /// @brief Register an operation without payload.
  /// Only available when PayloadType is void.
  /// Can be called multiple times for the same key; each call increments the
  /// expected number of waiters.
  /// @param key [in] The operation key to register.
  void RegisterOperation(const KeyType& key /*[in]*/)
    requires std::is_void_v<PayloadType>
  {
    registration_count_[key]++;
  }

  /// @brief Register an operation with associated payload data.
  /// Only available when PayloadType is not void.
  /// Can be called multiple times for the same key; each call increments the
  /// registration count and replaces any existing payload.
  /// @param key [in] The operation key to register.
  /// @param payload [in] The payload to associate with this key.
  template <typename P = PayloadType>
    requires (!std::is_void_v<P>)
  void RegisterOperation(const KeyType& key /*[in]*/, P payload /*[in]*/) {
    registration_count_[key]++;
    payloads_.erase(key);
    payloads_.emplace(key, std::move(payload));
  }

  /// @brief Consume and return the payload for a registered operation.
  /// Only available when PayloadType is not void.
  /// Returns std::nullopt if the key has no stored payload.
  /// @param key [in] The operation key whose payload to consume.
  /// @return The payload if present, std::nullopt otherwise.
  template <typename P = PayloadType>
    requires (!std::is_void_v<P>)
  [[nodiscard]] std::optional<P> ConsumePayload(
      const KeyType& key /*[in]*/) {
    auto it = payloads_.find(key);
    if (it == payloads_.end()) {
      return std::nullopt;
    }
    P result = std::move(it->second);
    payloads_.erase(it);
    return result;
  }

  /// @brief Mark the operation as complete. Future AddWaiter calls for this
  /// key will return kAlreadyComplete.
  /// @param key [in] The operation key to mark complete.
  void MarkComplete(const KeyType& key /*[in]*/) { completed_.insert(key); }

  /// @brief Add a waiter for the given key.
  ///
  /// If the operation is already complete, drains one completion (decrements
  /// the registration count) and returns kAlreadyComplete. If the key was
  /// never registered, returns kNotRegistered. Otherwise, queues the waiter.
  ///
  /// @param key [in] The operation key to wait on.
  /// @param identity [in] The client identity to register.
  /// @return The result indicating what action was taken.
  [[nodiscard]] AddWaiterResult AddWaiter(const KeyType& key /*[in]*/,
                                          const Identity& identity /*[in]*/) {
    if (completed_.contains(key)) {
      DecrementRegistrations(key, 1);
      return AddWaiterResult::kAlreadyComplete;
    }
    if (!registration_count_.contains(key)) {
      return AddWaiterResult::kNotRegistered;
    }
    waiters_[key].push_back(identity);
    return AddWaiterResult::kWaiterAdded;
  }

  /// @brief Retrieve and remove all waiters for the given key. Each drained
  /// waiter decrements the registration count.
  /// @param key [in] The operation key whose waiters to drain.
  /// @return Vector of waiting client identities. Empty if no waiters.
  [[nodiscard]] std::vector<Identity> DrainWaiters(
      const KeyType& key /*[in]*/) {
    auto it = waiters_.find(key);
    if (it == waiters_.end()) {
      return {};
    }
    std::vector<Identity> result = std::move(it->second);
    waiters_.erase(it);
    DecrementRegistrations(key, result.size());
    return result;
  }

  /// @brief Check whether any waiters exist for the given key.
  /// @param key [in] The operation key to check.
  /// @return True if at least one waiter is registered.
  [[nodiscard]] bool HasWaiters(const KeyType& key /*[in]*/) const {
    return waiters_.contains(key);
  }

 private:
  void DecrementRegistrations(const KeyType& key /*[in]*/,
                              std::size_t count /*[in]*/) {
    auto it = registration_count_.find(key);
    if (it == registration_count_.end()) {
      return;
    }
    if (it->second <= count) {
      registration_count_.erase(it);
      completed_.erase(key);
      if constexpr (!std::is_void_v<PayloadType>) {
        payloads_.erase(key);
      }
    } else {
      it->second -= count;
    }
  }

  std::unordered_map<KeyType, std::size_t, Hash> registration_count_;
  std::unordered_set<KeyType, Hash> completed_;
  std::unordered_map<KeyType, std::vector<Identity>, Hash> waiters_;

  [[no_unique_address]] std::conditional_t<
      std::is_void_v<PayloadType>, std::monostate,
      std::unordered_map<KeyType, PayloadType, Hash>>
      payloads_;
};

//==============================================================================
}  // namespace setu::commons::utils
//==============================================================================

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
#include <gtest/gtest.h>
//==============================================================================
#include "commons/StdCommon.h"
#include "commons/Types.h"
#include "commons/utils/ShardAggregator.h"
//==============================================================================
namespace setu::test::native {
//==============================================================================
using setu::commons::GenerateUUID;
using setu::commons::Identity;
using setu::commons::RequestId;
using setu::commons::ShardId;
using setu::commons::TensorName;
using setu::commons::utils::AggregationParticipant;
using setu::commons::utils::ShardAggregator;
//==============================================================================
namespace {
//==============================================================================

/// @brief Simple key type for testing (mimics CopyKey in Coordinator)
struct TestKey {
  TensorName src;
  TensorName dst;

  bool operator<(const TestKey& other) const {
    if (src != other.src) return src < other.src;
    return dst < other.dst;
  }

  bool operator==(const TestKey& other) const {
    return src == other.src && dst == other.dst;
  }
};

/// @brief Simple payload type for testing
struct TestPayload {
  std::string data;

  bool operator==(const TestPayload& other) const { return data == other.data; }
};

/// @brief No-op validation function for tests
auto NoopValidate = [](const TestPayload& /*stored*/,
                       const TestPayload& /*incoming*/) {};

/// @brief Helper to create an AggregationParticipant
AggregationParticipant MakeParticipant(const std::string& identity) {
  return AggregationParticipant{identity, GenerateUUID()};
}

//==============================================================================
}  // namespace
//==============================================================================
// CancelIf tests
//==============================================================================

TEST(ShardAggregatorTest, CancelIf_NoGroups_ReturnsEmpty) {
  ShardAggregator<TestKey, TestPayload> aggregator;

  auto cancelled =
      aggregator.CancelIf([](const TestKey& /*key*/) { return true; });

  EXPECT_TRUE(cancelled.empty());
}

TEST(ShardAggregatorTest, CancelIf_NoMatchingGroups_ReturnsEmpty) {
  ShardAggregator<TestKey, TestPayload> aggregator;

  // Submit a shard to group ("tensor_a", "tensor_b")
  auto shard_id = GenerateUUID();
  auto result = aggregator.Submit(TestKey{"tensor_a", "tensor_b"}, shard_id,
                                  TestPayload{"payload"},
                                  MakeParticipant("client_1"), 2, NoopValidate);
  EXPECT_FALSE(result.has_value());

  // Cancel groups matching a different tensor — should find none
  auto cancelled = aggregator.CancelIf([](const TestKey& key) {
    return key.src == "tensor_x" || key.dst == "tensor_x";
  });

  EXPECT_TRUE(cancelled.empty());

  // Original group should still be intact — submitting second shard completes
  // it
  auto shard_id_2 = GenerateUUID();
  auto result_2 = aggregator.Submit(
      TestKey{"tensor_a", "tensor_b"}, shard_id_2, TestPayload{"payload"},
      MakeParticipant("client_2"), 2, NoopValidate);
  ASSERT_TRUE(result_2.has_value());
  EXPECT_EQ(result_2->participants.size(), 2);
}

TEST(ShardAggregatorTest, CancelIf_MatchesSingleGroup_ReturnsParticipants) {
  ShardAggregator<TestKey, TestPayload> aggregator;

  auto shard_id = GenerateUUID();
  auto participant = MakeParticipant("client_1");
  auto expected_identity = participant.identity;
  auto expected_request_id = participant.request_id;

  (void)aggregator.Submit(TestKey{"tensor_a", "tensor_b"}, shard_id,
                          TestPayload{"payload"}, std::move(participant), 2,
                          NoopValidate);

  // Cancel groups involving "tensor_a"
  auto cancelled = aggregator.CancelIf([](const TestKey& key) {
    return key.src == "tensor_a" || key.dst == "tensor_a";
  });

  ASSERT_EQ(cancelled.size(), 1);
  EXPECT_EQ(cancelled[0].identity, expected_identity);
  EXPECT_EQ(cancelled[0].request_id, expected_request_id);
}

TEST(ShardAggregatorTest,
     CancelIf_MatchesMultipleGroups_ReturnsAllParticipants) {
  ShardAggregator<TestKey, TestPayload> aggregator;

  // Group 1: tensor_a -> tensor_b (1 participant)
  (void)aggregator.Submit(TestKey{"tensor_a", "tensor_b"}, GenerateUUID(),
                          TestPayload{"p1"}, MakeParticipant("client_1"), 2,
                          NoopValidate);

  // Group 2: tensor_a -> tensor_c (1 participant)
  (void)aggregator.Submit(TestKey{"tensor_a", "tensor_c"}, GenerateUUID(),
                          TestPayload{"p2"}, MakeParticipant("client_2"), 2,
                          NoopValidate);

  // Group 3: tensor_x -> tensor_y (1 participant, should NOT be cancelled)
  (void)aggregator.Submit(TestKey{"tensor_x", "tensor_y"}, GenerateUUID(),
                          TestPayload{"p3"}, MakeParticipant("client_3"), 2,
                          NoopValidate);

  // Cancel groups involving "tensor_a" (should match groups 1 and 2)
  auto cancelled = aggregator.CancelIf([](const TestKey& key) {
    return key.src == "tensor_a" || key.dst == "tensor_a";
  });

  EXPECT_EQ(cancelled.size(), 2);

  // Verify identities
  std::set<Identity> cancelled_identities;
  for (const auto& p : cancelled) {
    cancelled_identities.insert(p.identity);
  }
  EXPECT_TRUE(cancelled_identities.count("client_1") > 0);
  EXPECT_TRUE(cancelled_identities.count("client_2") > 0);

  // Group 3 should still be intact
  auto shard_id_2 = GenerateUUID();
  auto result = aggregator.Submit(TestKey{"tensor_x", "tensor_y"}, shard_id_2,
                                  TestPayload{"p3"},
                                  MakeParticipant("client_4"), 2, NoopValidate);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->participants.size(), 2);
}

TEST(ShardAggregatorTest, CancelIf_GroupWithMultipleParticipants_ReturnsAll) {
  ShardAggregator<TestKey, TestPayload> aggregator;

  // Submit 2 shards to the same group (expected_count = 3, so not complete yet)
  (void)aggregator.Submit(TestKey{"tensor_a", "tensor_b"}, GenerateUUID(),
                          TestPayload{"payload"}, MakeParticipant("client_1"),
                          3, NoopValidate);
  (void)aggregator.Submit(TestKey{"tensor_a", "tensor_b"}, GenerateUUID(),
                          TestPayload{"payload"}, MakeParticipant("client_2"),
                          3, NoopValidate);

  // Cancel groups involving "tensor_a"
  auto cancelled = aggregator.CancelIf([](const TestKey& key) {
    return key.src == "tensor_a" || key.dst == "tensor_a";
  });

  ASSERT_EQ(cancelled.size(), 2);
  std::set<Identity> cancelled_identities;
  for (const auto& p : cancelled) {
    cancelled_identities.insert(p.identity);
  }
  EXPECT_TRUE(cancelled_identities.count("client_1") > 0);
  EXPECT_TRUE(cancelled_identities.count("client_2") > 0);
}

TEST(ShardAggregatorTest, CancelIf_CancelledGroupCannotBeCompleted) {
  ShardAggregator<TestKey, TestPayload> aggregator;

  // Submit 1 shard (expected_count = 2)
  (void)aggregator.Submit(TestKey{"tensor_a", "tensor_b"}, GenerateUUID(),
                          TestPayload{"payload"}, MakeParticipant("client_1"),
                          2, NoopValidate);

  // Cancel the group
  auto cancelled = aggregator.CancelIf(
      [](const TestKey& key) { return key.src == "tensor_a"; });
  EXPECT_EQ(cancelled.size(), 1);

  // Submitting to the same key again should start a fresh group
  // (the old group was cleaned up)
  auto result = aggregator.Submit(TestKey{"tensor_a", "tensor_b"},
                                  GenerateUUID(), TestPayload{"payload"},
                                  MakeParticipant("client_2"), 2, NoopValidate);

  // Should not complete because this is a new group with only 1 of 2 shards
  EXPECT_FALSE(result.has_value());
}

TEST(ShardAggregatorTest, CancelIf_MatchByDstName_Works) {
  ShardAggregator<TestKey, TestPayload> aggregator;

  // Group where tensor_x is the destination
  (void)aggregator.Submit(TestKey{"tensor_a", "tensor_x"}, GenerateUUID(),
                          TestPayload{"payload"}, MakeParticipant("client_1"),
                          2, NoopValidate);

  // Cancel groups where tensor_x is involved (check dst)
  auto cancelled = aggregator.CancelIf([](const TestKey& key) {
    return key.src == "tensor_x" || key.dst == "tensor_x";
  });

  ASSERT_EQ(cancelled.size(), 1);
  EXPECT_EQ(cancelled[0].identity, "client_1");
}

TEST(ShardAggregatorTest, CancelIf_AllGroupsCancelled_LeavesEmptyState) {
  ShardAggregator<TestKey, TestPayload> aggregator;

  // Add several groups
  (void)aggregator.Submit(TestKey{"a", "b"}, GenerateUUID(), TestPayload{"p1"},
                          MakeParticipant("c1"), 2, NoopValidate);
  (void)aggregator.Submit(TestKey{"c", "d"}, GenerateUUID(), TestPayload{"p2"},
                          MakeParticipant("c2"), 2, NoopValidate);

  // Cancel all groups
  auto cancelled =
      aggregator.CancelIf([](const TestKey& /*key*/) { return true; });

  EXPECT_EQ(cancelled.size(), 2);

  // Cancelling again should return empty
  auto cancelled_again =
      aggregator.CancelIf([](const TestKey& /*key*/) { return true; });
  EXPECT_TRUE(cancelled_again.empty());
}

//==============================================================================
// Submit tests (complementary - ensure Submit still works correctly)
//==============================================================================

TEST(ShardAggregatorTest, Submit_SingleShard_CompletesImmediately) {
  ShardAggregator<TestKey, TestPayload> aggregator;

  auto shard_id = GenerateUUID();
  auto result =
      aggregator.Submit(TestKey{"src", "dst"}, shard_id, TestPayload{"data"},
                        MakeParticipant("client_1"), 1, NoopValidate);

  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->payload.data, "data");
  EXPECT_EQ(result->participants.size(), 1);
  EXPECT_EQ(result->participants[0].identity, "client_1");
}

TEST(ShardAggregatorTest, Submit_TwoShards_CompletesOnSecond) {
  ShardAggregator<TestKey, TestPayload> aggregator;
  TestKey key{"src", "dst"};

  auto result_1 =
      aggregator.Submit(key, GenerateUUID(), TestPayload{"data"},
                        MakeParticipant("client_1"), 2, NoopValidate);
  EXPECT_FALSE(result_1.has_value());

  auto result_2 =
      aggregator.Submit(key, GenerateUUID(), TestPayload{"data"},
                        MakeParticipant("client_2"), 2, NoopValidate);
  ASSERT_TRUE(result_2.has_value());
  EXPECT_EQ(result_2->participants.size(), 2);
}

//==============================================================================
}  // namespace setu::test::native
//==============================================================================

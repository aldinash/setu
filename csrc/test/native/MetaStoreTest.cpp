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
#include "commons/BoostCommon.h"
#include "commons/StdCommon.h"
#include "commons/TorchCommon.h"
#include "commons/datatypes/Device.h"
#include "commons/datatypes/TensorDimSpec.h"
#include "commons/datatypes/TensorShardSpec.h"
#include "metastore/MetaStore.h"
//==============================================================================
namespace setu::test::native {
//==============================================================================
using setu::commons::GenerateUUID;
using setu::commons::NodeId;
using setu::commons::TensorName;
using setu::commons::datatypes::Device;
using setu::commons::datatypes::TensorDimSpec;
using setu::commons::datatypes::TensorShardSpec;
using setu::metastore::MetaStore;
//==============================================================================
namespace {
//==============================================================================
// Helper to create a simple 1D tensor shard spec
TensorShardSpec Make1DShardSpec(const TensorName& name, std::size_t total_size,
                                std::int32_t start, std::int32_t end) {
  std::vector<TensorDimSpec> dims;
  dims.emplace_back("x", total_size, start, end);
  return TensorShardSpec(name, dims, torch::kFloat32, Device(torch::kCPU));
}

// Helper to create a 2D tensor shard spec
TensorShardSpec Make2DShardSpec(const TensorName& name, std::size_t rows,
                                std::size_t cols, std::int32_t row_start,
                                std::int32_t row_end, std::int32_t col_start,
                                std::int32_t col_end) {
  std::vector<TensorDimSpec> dims;
  dims.emplace_back("row", rows, row_start, row_end);
  dims.emplace_back("col", cols, col_start, col_end);
  return TensorShardSpec(name, dims, torch::kFloat32, Device(torch::kCPU));
}
//==============================================================================
}  // namespace
//==============================================================================
// RegisterTensorShard tests
//==============================================================================
TEST(MetaStoreTest, RegisterTensorShard_SingleShard_ReturnsValidRef) {
  MetaStore store;
  const TensorName tensor_name = "test_tensor";
  const NodeId owner_node = GenerateUUID();

  // Register a single shard that covers the entire tensor
  auto spec = Make1DShardSpec(tensor_name, 100, 0, 100);
  auto ref = store.RegisterTensorShard(spec, owner_node);

  // Verify all TensorShardRef fields
  EXPECT_EQ(ref.name, tensor_name);
  EXPECT_FALSE(ref.shard_id.is_nil());
  EXPECT_EQ(ref.dims.size(), 1);

  // Verify dimension name and size
  EXPECT_EQ(ref.dims.at("x").name, "x");
  EXPECT_EQ(ref.dims.at("x").size, 100);
}

TEST(MetaStoreTest, RegisterTensorShard_MultipleShards_AllRegistered) {
  MetaStore store;
  const TensorName tensor_name = "test_tensor";
  const NodeId node_0 = GenerateUUID();
  const NodeId node_1 = GenerateUUID();

  // Register two shards that together cover a 1D tensor of size 100
  auto spec1 = Make1DShardSpec(tensor_name, 100, 0, 50);
  auto spec2 = Make1DShardSpec(tensor_name, 100, 50, 100);

  auto ref1 = store.RegisterTensorShard(spec1, node_0);
  auto ref2 = store.RegisterTensorShard(spec2, node_1);

  // Verify ref1 contains expected values
  EXPECT_EQ(ref1.name, tensor_name);
  EXPECT_FALSE(ref1.shard_id.is_nil());
  EXPECT_EQ(ref1.dims.size(), 1);
  EXPECT_EQ(ref1.dims.at("x").name, "x");
  EXPECT_EQ(ref1.dims.at("x").size, 100);

  // Verify ref2 contains expected values
  EXPECT_EQ(ref2.name, tensor_name);
  EXPECT_FALSE(ref2.shard_id.is_nil());
  EXPECT_EQ(ref2.dims.size(), 1);
  EXPECT_EQ(ref2.dims.at("x").name, "x");
  EXPECT_EQ(ref2.dims.at("x").size, 100);

  // Shard IDs should be unique
  EXPECT_NE(ref1.shard_id, ref2.shard_id);
  EXPECT_EQ(store.GetNumShardsForTensor(tensor_name), 2);
}

TEST(MetaStoreTest, RegisterTensorShard_2DTensor_CorrectDims) {
  MetaStore store;
  const TensorName tensor_name = "matrix";
  const NodeId owner_node = GenerateUUID();

  // Register a 2D shard
  auto spec = Make2DShardSpec(tensor_name, 10, 20, 0, 10, 0, 20);
  auto ref = store.RegisterTensorShard(spec, owner_node);

  // Verify all TensorShardRef fields
  EXPECT_EQ(ref.name, tensor_name);
  EXPECT_FALSE(ref.shard_id.is_nil());
  EXPECT_EQ(ref.dims.size(), 2);

  // Verify row dimension
  EXPECT_EQ(ref.dims.at("row").name, "row");
  EXPECT_EQ(ref.dims.at("row").size, 10);

  // Verify col dimension
  EXPECT_EQ(ref.dims.at("col").name, "col");
  EXPECT_EQ(ref.dims.at("col").size, 20);
}
//==============================================================================
// GetNumShardsForTensor tests
//==============================================================================
TEST(MetaStoreTest, GetNumShardsForTensor_UnknownTensor_ReturnsZero) {
  MetaStore store;

  EXPECT_EQ(store.GetNumShardsForTensor("nonexistent"), 0);
}

TEST(MetaStoreTest,
     GetNumShardsForTensor_AfterRegistration_ReturnsCorrectCount) {
  MetaStore store;
  const TensorName tensor_name = "test_tensor";
  const NodeId node_0 = GenerateUUID();
  const NodeId node_1 = GenerateUUID();
  const NodeId node_2 = GenerateUUID();

  // Register 3 shards and verify each ref
  auto ref1 = store.RegisterTensorShard(Make1DShardSpec(tensor_name, 90, 0, 30),
                                        node_0);
  EXPECT_EQ(ref1.name, tensor_name);
  EXPECT_FALSE(ref1.shard_id.is_nil());
  EXPECT_EQ(ref1.dims.at("x").size, 90);
  EXPECT_EQ(store.GetNumShardsForTensor(tensor_name), 1);

  auto ref2 = store.RegisterTensorShard(
      Make1DShardSpec(tensor_name, 90, 30, 60), node_1);
  EXPECT_EQ(ref2.name, tensor_name);
  EXPECT_FALSE(ref2.shard_id.is_nil());
  EXPECT_EQ(ref2.dims.at("x").size, 90);
  EXPECT_NE(ref1.shard_id, ref2.shard_id);
  EXPECT_EQ(store.GetNumShardsForTensor(tensor_name), 2);

  auto ref3 = store.RegisterTensorShard(
      Make1DShardSpec(tensor_name, 90, 60, 90), node_2);
  EXPECT_EQ(ref3.name, tensor_name);
  EXPECT_FALSE(ref3.shard_id.is_nil());
  EXPECT_EQ(ref3.dims.at("x").size, 90);
  EXPECT_NE(ref1.shard_id, ref3.shard_id);
  EXPECT_NE(ref2.shard_id, ref3.shard_id);
  EXPECT_EQ(store.GetNumShardsForTensor(tensor_name), 3);
}
//==============================================================================
// AllShardsRegistered tests
//==============================================================================
TEST(MetaStoreTest, AllShardsRegistered_UnknownTensor_ReturnsFalse) {
  MetaStore store;

  EXPECT_FALSE(store.AllShardsRegistered("nonexistent"));
}

TEST(MetaStoreTest, AllShardsRegistered_PartialRegistration_ReturnsFalse) {
  MetaStore store;
  const TensorName tensor_name = "test_tensor";
  const NodeId owner_node = GenerateUUID();

  // Register only half of the tensor
  auto ref = store.RegisterTensorShard(Make1DShardSpec(tensor_name, 100, 0, 50),
                                       owner_node);
  EXPECT_EQ(ref.name, tensor_name);
  EXPECT_FALSE(ref.shard_id.is_nil());

  EXPECT_FALSE(store.AllShardsRegistered(tensor_name));
}

TEST(MetaStoreTest, AllShardsRegistered_FullRegistration_ReturnsTrue) {
  MetaStore store;
  const TensorName tensor_name = "test_tensor";
  const NodeId node_0 = GenerateUUID();
  const NodeId node_1 = GenerateUUID();

  // Register all shards covering the full tensor
  auto ref1 = store.RegisterTensorShard(
      Make1DShardSpec(tensor_name, 100, 0, 50), node_0);
  auto ref2 = store.RegisterTensorShard(
      Make1DShardSpec(tensor_name, 100, 50, 100), node_1);
  EXPECT_FALSE(ref1.shard_id.is_nil());
  EXPECT_FALSE(ref2.shard_id.is_nil());

  EXPECT_TRUE(store.AllShardsRegistered(tensor_name));
}

TEST(MetaStoreTest, AllShardsRegistered_SingleShardFullTensor_ReturnsTrue) {
  MetaStore store;
  const TensorName tensor_name = "test_tensor";
  const NodeId owner_node = GenerateUUID();

  // Single shard covers entire tensor
  auto ref = store.RegisterTensorShard(
      Make1DShardSpec(tensor_name, 100, 0, 100), owner_node);
  EXPECT_EQ(ref.name, tensor_name);
  EXPECT_FALSE(ref.shard_id.is_nil());

  EXPECT_TRUE(store.AllShardsRegistered(tensor_name));
}
//==============================================================================
// GetTensorMetadata tests
//==============================================================================
TEST(MetaStoreTest, GetTensorMetadata_PartialRegistration_ReturnsNullptr) {
  MetaStore store;
  const TensorName tensor_name = "test_tensor";
  const NodeId owner_node = GenerateUUID();

  // Register only part of the tensor
  auto ref = store.RegisterTensorShard(Make1DShardSpec(tensor_name, 100, 0, 50),
                                       owner_node);
  EXPECT_FALSE(ref.shard_id.is_nil());

  EXPECT_EQ(store.GetTensorMetadata(tensor_name), nullptr);
}

TEST(MetaStoreTest, GetTensorMetadata_UnknownTensor_ReturnsNullptr) {
  MetaStore store;

  EXPECT_EQ(store.GetTensorMetadata("nonexistent"), nullptr);
}

TEST(MetaStoreTest, GetTensorMetadata_FullRegistration_ReturnsValidMetadata) {
  MetaStore store;
  const TensorName tensor_name = "test_tensor";
  const NodeId node_0 = GenerateUUID();
  const NodeId node_1 = GenerateUUID();

  // Register all shards
  auto ref1 = store.RegisterTensorShard(
      Make1DShardSpec(tensor_name, 100, 0, 50), node_0);
  auto ref2 = store.RegisterTensorShard(
      Make1DShardSpec(tensor_name, 100, 50, 100), node_1);
  EXPECT_FALSE(ref1.shard_id.is_nil());
  EXPECT_FALSE(ref2.shard_id.is_nil());

  auto metadata = store.GetTensorMetadata(tensor_name);

  ASSERT_NE(metadata, nullptr);
  EXPECT_EQ(metadata->name, tensor_name);
  EXPECT_EQ(metadata->size, 100);
  EXPECT_EQ(metadata->shards.size(), 2);
  EXPECT_EQ(metadata->dtype, torch::kFloat32);
}

TEST(MetaStoreTest, GetTensorMetadata_CachesResult_ReturnsSamePointer) {
  MetaStore store;
  const TensorName tensor_name = "test_tensor";
  const NodeId owner_node = GenerateUUID();

  auto ref = store.RegisterTensorShard(
      Make1DShardSpec(tensor_name, 100, 0, 100), owner_node);
  EXPECT_FALSE(ref.shard_id.is_nil());

  auto metadata1 = store.GetTensorMetadata(tensor_name);
  auto metadata2 = store.GetTensorMetadata(tensor_name);

  // Should return the same cached pointer
  EXPECT_EQ(metadata1.get(), metadata2.get());
}

TEST(MetaStoreTest, GetTensorMetadata_2DTensor_CorrectMetadata) {
  MetaStore store;
  const TensorName tensor_name = "matrix";
  const NodeId node_0 = GenerateUUID();
  const NodeId node_1 = GenerateUUID();
  const NodeId node_2 = GenerateUUID();
  const NodeId node_3 = GenerateUUID();

  // Register 4 shards for a 10x20 matrix (2x2 grid of shards)
  auto ref1 = store.RegisterTensorShard(
      Make2DShardSpec(tensor_name, 10, 20, 0, 5, 0, 10), node_0);
  auto ref2 = store.RegisterTensorShard(
      Make2DShardSpec(tensor_name, 10, 20, 0, 5, 10, 20), node_1);
  auto ref3 = store.RegisterTensorShard(
      Make2DShardSpec(tensor_name, 10, 20, 5, 10, 0, 10), node_2);
  auto ref4 = store.RegisterTensorShard(
      Make2DShardSpec(tensor_name, 10, 20, 5, 10, 10, 20), node_3);

  // Verify all refs have correct tensor name and valid shard IDs
  EXPECT_EQ(ref1.name, tensor_name);
  EXPECT_EQ(ref2.name, tensor_name);
  EXPECT_EQ(ref3.name, tensor_name);
  EXPECT_EQ(ref4.name, tensor_name);
  EXPECT_FALSE(ref1.shard_id.is_nil());
  EXPECT_FALSE(ref2.shard_id.is_nil());
  EXPECT_FALSE(ref3.shard_id.is_nil());
  EXPECT_FALSE(ref4.shard_id.is_nil());

  // Verify all shard IDs are unique
  EXPECT_NE(ref1.shard_id, ref2.shard_id);
  EXPECT_NE(ref1.shard_id, ref3.shard_id);
  EXPECT_NE(ref1.shard_id, ref4.shard_id);
  EXPECT_NE(ref2.shard_id, ref3.shard_id);
  EXPECT_NE(ref2.shard_id, ref4.shard_id);
  EXPECT_NE(ref3.shard_id, ref4.shard_id);

  // Verify dimension structure in refs
  EXPECT_EQ(ref1.dims.size(), 2);
  EXPECT_EQ(ref1.dims.at("row").size, 10);
  EXPECT_EQ(ref1.dims.at("col").size, 20);

  auto metadata = store.GetTensorMetadata(tensor_name);

  ASSERT_NE(metadata, nullptr);
  EXPECT_EQ(metadata->name, tensor_name);
  EXPECT_EQ(metadata->size, 200);  // 10 * 20
  EXPECT_EQ(metadata->shards.size(), 4);
  EXPECT_EQ(metadata->dims.size(), 2);
  EXPECT_EQ(metadata->dims.at("row").size, 10);
  EXPECT_EQ(metadata->dims.at("col").size, 20);
}

TEST(MetaStoreTest, GetTensorMetadata_GetOwnerNodeIds_ReturnsAllOwners) {
  MetaStore store;
  const TensorName tensor_name = "test_tensor";
  const NodeId node_0 = GenerateUUID();
  const NodeId node_1 = GenerateUUID();
  const NodeId node_2 = GenerateUUID();

  auto ref1 = store.RegisterTensorShard(Make1DShardSpec(tensor_name, 90, 0, 30),
                                        node_0);
  auto ref2 = store.RegisterTensorShard(
      Make1DShardSpec(tensor_name, 90, 30, 60), node_1);
  auto ref3 = store.RegisterTensorShard(
      Make1DShardSpec(tensor_name, 90, 60, 90), node_2);
  EXPECT_FALSE(ref1.shard_id.is_nil());
  EXPECT_FALSE(ref2.shard_id.is_nil());
  EXPECT_FALSE(ref3.shard_id.is_nil());

  auto metadata = store.GetTensorMetadata(tensor_name);
  ASSERT_NE(metadata, nullptr);

  auto owner_ids = metadata->GetOwnerNodeIds();
  EXPECT_EQ(owner_ids.size(), 3);
  EXPECT_TRUE(owner_ids.count(node_0) > 0);
  EXPECT_TRUE(owner_ids.count(node_1) > 0);
  EXPECT_TRUE(owner_ids.count(node_2) > 0);
}
//==============================================================================
// Multiple tensors tests
//==============================================================================
TEST(MetaStoreTest, MultipleTensors_IndependentTracking) {
  MetaStore store;
  const NodeId node_0 = GenerateUUID();
  const NodeId node_1 = GenerateUUID();

  // Register shards for two different tensors
  auto ref_a = store.RegisterTensorShard(
      Make1DShardSpec("tensor_a", 100, 0, 50), node_0);
  auto ref_b = store.RegisterTensorShard(
      Make1DShardSpec("tensor_b", 200, 0, 200), node_1);

  // Verify ref_a contains correct values for tensor_a
  EXPECT_EQ(ref_a.name, "tensor_a");
  EXPECT_FALSE(ref_a.shard_id.is_nil());
  EXPECT_EQ(ref_a.dims.size(), 1);
  EXPECT_EQ(ref_a.dims.at("x").size, 100);

  // Verify ref_b contains correct values for tensor_b
  EXPECT_EQ(ref_b.name, "tensor_b");
  EXPECT_FALSE(ref_b.shard_id.is_nil());
  EXPECT_EQ(ref_b.dims.size(), 1);
  EXPECT_EQ(ref_b.dims.at("x").size, 200);

  EXPECT_EQ(store.GetNumShardsForTensor("tensor_a"), 1);
  EXPECT_EQ(store.GetNumShardsForTensor("tensor_b"), 1);

  EXPECT_FALSE(store.AllShardsRegistered("tensor_a"));
  EXPECT_TRUE(store.AllShardsRegistered("tensor_b"));

  EXPECT_EQ(store.GetTensorMetadata("tensor_a"), nullptr);
  ASSERT_NE(store.GetTensorMetadata("tensor_b"), nullptr);
  EXPECT_EQ(store.GetTensorMetadata("tensor_b")->size, 200);
}
//==============================================================================
// Same owner tests
//==============================================================================
TEST(MetaStoreTest,
     RegisterTensorShard_SameOwnerMultipleShards_TracksCorrectly) {
  MetaStore store;
  const TensorName tensor_name = "test_tensor";
  const NodeId same_owner = GenerateUUID();

  // Same node owns all shards
  auto ref1 = store.RegisterTensorShard(Make1DShardSpec(tensor_name, 90, 0, 30),
                                        same_owner);
  auto ref2 = store.RegisterTensorShard(
      Make1DShardSpec(tensor_name, 90, 30, 60), same_owner);
  auto ref3 = store.RegisterTensorShard(
      Make1DShardSpec(tensor_name, 90, 60, 90), same_owner);

  // Shard IDs should still be unique even with same owner
  EXPECT_NE(ref1.shard_id, ref2.shard_id);
  EXPECT_NE(ref1.shard_id, ref3.shard_id);
  EXPECT_NE(ref2.shard_id, ref3.shard_id);

  EXPECT_EQ(store.GetNumShardsForTensor(tensor_name), 3);
  EXPECT_TRUE(store.AllShardsRegistered(tensor_name));

  auto metadata = store.GetTensorMetadata(tensor_name);
  ASSERT_NE(metadata, nullptr);

  // Only one unique owner
  auto owner_ids = metadata->GetOwnerNodeIds();
  EXPECT_EQ(owner_ids.size(), 1);
  EXPECT_TRUE(owner_ids.count(same_owner) > 0);
}
//==============================================================================
// Metadata shard ID verification tests
//==============================================================================
TEST(MetaStoreTest, GetTensorMetadata_ShardIdsMatchRegistered) {
  MetaStore store;
  const TensorName tensor_name = "test_tensor";
  const NodeId node_0 = GenerateUUID();
  const NodeId node_1 = GenerateUUID();

  auto ref1 = store.RegisterTensorShard(
      Make1DShardSpec(tensor_name, 100, 0, 50), node_0);
  auto ref2 = store.RegisterTensorShard(
      Make1DShardSpec(tensor_name, 100, 50, 100), node_1);

  auto metadata = store.GetTensorMetadata(tensor_name);
  ASSERT_NE(metadata, nullptr);

  // Verify that the shard IDs in metadata match the ones returned during
  // registration
  EXPECT_TRUE(metadata->shards.count(ref1.shard_id) > 0);
  EXPECT_TRUE(metadata->shards.count(ref2.shard_id) > 0);

  // Verify shard metadata contents
  auto shard1 = metadata->shards.at(ref1.shard_id);
  EXPECT_EQ(shard1->owner, node_0);
  EXPECT_EQ(shard1->spec.name, tensor_name);

  auto shard2 = metadata->shards.at(ref2.shard_id);
  EXPECT_EQ(shard2->owner, node_1);
  EXPECT_EQ(shard2->spec.name, tensor_name);
}
//==============================================================================
// Different dtype tests
//==============================================================================
TEST(MetaStoreTest, RegisterTensorShard_Float16Dtype_CorrectMetadata) {
  MetaStore store;
  const TensorName tensor_name = "float16_tensor";
  const NodeId owner_node = GenerateUUID();

  // Create shard spec with float16 dtype
  std::vector<TensorDimSpec> dims;
  dims.emplace_back("x", 100, 0, 100);
  TensorShardSpec spec(tensor_name, dims, torch::kFloat16, Device(torch::kCPU));

  auto ref = store.RegisterTensorShard(spec, owner_node);
  EXPECT_EQ(ref.name, tensor_name);
  EXPECT_FALSE(ref.shard_id.is_nil());

  auto metadata = store.GetTensorMetadata(tensor_name);
  ASSERT_NE(metadata, nullptr);
  EXPECT_EQ(metadata->dtype, torch::kFloat16);
}
//==============================================================================
// Metadata dimension verification tests
//==============================================================================
TEST(MetaStoreTest, GetTensorMetadata_DimensionNamesCorrect) {
  MetaStore store;
  const TensorName tensor_name = "named_dims_tensor";
  const NodeId owner_node = GenerateUUID();

  // Create 3D tensor with named dimensions
  std::vector<TensorDimSpec> dims;
  dims.emplace_back("batch", 8, 0, 8);
  dims.emplace_back("sequence", 512, 0, 512);
  dims.emplace_back("hidden", 768, 0, 768);
  TensorShardSpec spec(tensor_name, dims, torch::kFloat32, Device(torch::kCPU));

  auto ref = store.RegisterTensorShard(spec, owner_node);
  EXPECT_EQ(ref.dims.size(), 3);

  auto metadata = store.GetTensorMetadata(tensor_name);
  ASSERT_NE(metadata, nullptr);

  // Verify all dimension names and sizes in metadata
  EXPECT_EQ(metadata->dims.size(), 3);
  EXPECT_EQ(metadata->dims.at("batch").name, "batch");
  EXPECT_EQ(metadata->dims.at("batch").size, 8);
  EXPECT_EQ(metadata->dims.at("sequence").name, "sequence");
  EXPECT_EQ(metadata->dims.at("sequence").size, 512);
  EXPECT_EQ(metadata->dims.at("hidden").name, "hidden");
  EXPECT_EQ(metadata->dims.at("hidden").size, 768);

  // Verify total size
  EXPECT_EQ(metadata->size, 8 * 512 * 768);
}
//==============================================================================
}  // namespace setu::test::native
//==============================================================================

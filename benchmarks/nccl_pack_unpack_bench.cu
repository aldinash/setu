// =============================================================================
// Benchmark: Setu-style Send/Recv pack/unpack vs NCCL Gather/Scatter
//
// Compares two approaches for pack (gather) and unpack (scatter):
//   1. Grouped ncclSend/ncclRecv (what Setu's NCCL backend lowers to)
//   2. Native ncclGather / ncclScatter collectives
//
// Build:
//   NCCL_ROOT=$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/nccl
//   nvcc -O2 -o nccl_pack_unpack_bench nccl_pack_unpack_bench.cu \
//        -I$NCCL_ROOT/include -L$NCCL_ROOT/lib -lnccl -lcudart \
//        -Xlinker -rpath=$NCCL_ROOT/lib
//
// Run (requires N GPUs):
//   ./nccl_pack_unpack_bench [num_elements_per_rank] [num_warmup] [num_iters]
//
// Defaults: 1M elements per rank, 10 warmup, 100 iterations
// =============================================================================

#include <cuda_runtime.h>
#include <nccl.h>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

// =============================================================================

#define CUDA_CHECK(cmd)                                                        \
  do {                                                                         \
    cudaError_t err = (cmd);                                                   \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,           \
              cudaGetErrorString(err));                                         \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

#define NCCL_CHECK(cmd)                                                        \
  do {                                                                         \
    ncclResult_t res = (cmd);                                                  \
    if (res != ncclSuccess) {                                                  \
      fprintf(stderr, "NCCL error %s:%d: %s\n", __FILE__, __LINE__,           \
              ncclGetErrorString(res));                                         \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// =============================================================================

struct BenchResult {
  double avg_us;
  double min_us;
  double max_us;
};

/// Time a lambda over `iters` iterations (after `warmup` warmups).
/// The lambda receives (iteration_index) and is expected to synchronize
/// all streams internally. Uses host-side timing since NCCL work spans
/// multiple device streams.
template <typename Fn>
BenchResult benchmark(Fn&& fn, int32_t warmup, int32_t iters) {
  // Warmup
  for (int32_t i = 0; i < warmup; ++i) {
    fn(i);
  }

  double total_us = 0.0;
  double min_us = 1e12;
  double max_us = 0.0;

  for (int32_t i = 0; i < iters; ++i) {
    // Device sync before timing to ensure clean start
    CUDA_CHECK(cudaDeviceSynchronize());

    auto t0 = std::chrono::high_resolution_clock::now();
    fn(i);
    // fn() is expected to synchronize all streams
    auto t1 = std::chrono::high_resolution_clock::now();

    double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    total_us += us;
    if (us < min_us) min_us = us;
    if (us > max_us) max_us = us;
  }

  return {total_us / iters, min_us, max_us};
}

// =============================================================================

int main(int argc, char** argv) {
  // --- Parse args ---
  size_t count_per_rank = (argc > 1) ? static_cast<size_t>(atol(argv[1]))
                                     : 1024 * 1024;  // 1M elements
  int32_t warmup = (argc > 2) ? atoi(argv[2]) : 10;
  int32_t iters = (argc > 3) ? atoi(argv[3]) : 100;

  // --- Discover GPUs ---
  int32_t num_gpus = 0;
  CUDA_CHECK(cudaGetDeviceCount(&num_gpus));
  if (num_gpus < 2) {
    fprintf(stderr, "Need at least 2 GPUs, found %d\n", num_gpus);
    return 1;
  }
  printf("Found %d GPUs\n", num_gpus);
  printf("Elements per rank: %zu  (%.2f MB per rank as float)\n",
         count_per_rank, count_per_rank * sizeof(float) / (1024.0 * 1024.0));
  printf("Warmup: %d, Iterations: %d\n\n", warmup, iters);

  size_t total_count = count_per_rank * static_cast<size_t>(num_gpus);

  // --- Create NCCL communicators ---
  std::vector<ncclComm_t> comms(num_gpus);
  {
    std::vector<int32_t> dev_list(num_gpus);
    for (int32_t i = 0; i < num_gpus; ++i) dev_list[i] = i;
    NCCL_CHECK(ncclCommInitAll(comms.data(), num_gpus, dev_list.data()));
  }

  // --- Allocate per-GPU buffers and streams ---
  // send_buf[i]: each rank's local data          (count_per_rank floats)
  // recv_buf[i]: gather destination on root /
  //              scatter source on root           (total_count floats)
  std::vector<float*> send_buf(num_gpus);
  std::vector<float*> recv_buf(num_gpus);
  std::vector<cudaStream_t> streams(num_gpus);

  for (int32_t i = 0; i < num_gpus; ++i) {
    CUDA_CHECK(cudaSetDevice(i));
    CUDA_CHECK(cudaMalloc(&send_buf[i], count_per_rank * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&recv_buf[i], total_count * sizeof(float)));
    CUDA_CHECK(cudaStreamCreate(&streams[i]));

    // Fill send buffer with rank index for verification
    float fill_val = static_cast<float>(i);
    std::vector<float> host_data(count_per_rank, fill_val);
    CUDA_CHECK(cudaMemcpy(send_buf[i], host_data.data(),
                          count_per_rank * sizeof(float),
                          cudaMemcpyHostToDevice));
  }

  int32_t root = 0;  // Gather destination / Scatter source

  // =========================================================================
  // Benchmark 1: GATHER via grouped Send/Recv (Setu's approach)
  // =========================================================================
  // All ranks send to root. Root receives from all ranks into contiguous
  // slots of recv_buf. This is what Setu's PackOp lowers to.
  // =========================================================================

  auto gather_sendrecv = [&](int32_t /*iter*/) {
    NCCL_CHECK(ncclGroupStart());
    for (int32_t r = 0; r < num_gpus; ++r) {
      // Every rank sends its data to root
      CUDA_CHECK(cudaSetDevice(r));
      NCCL_CHECK(ncclSend(send_buf[r], count_per_rank, ncclFloat, root,
                           comms[r], streams[r]));
    }
    // Root receives from all ranks
    CUDA_CHECK(cudaSetDevice(root));
    for (int32_t r = 0; r < num_gpus; ++r) {
      NCCL_CHECK(ncclRecv(
          static_cast<char*>(static_cast<void*>(recv_buf[root])) +
              r * count_per_rank * sizeof(float),
          count_per_rank, ncclFloat, r, comms[root], streams[root]));
    }
    NCCL_CHECK(ncclGroupEnd());

    // Sync all streams
    for (int32_t r = 0; r < num_gpus; ++r) {
      CUDA_CHECK(cudaSetDevice(r));
      CUDA_CHECK(cudaStreamSynchronize(streams[r]));
    }
  };

  // =========================================================================
  // Benchmark 2: GATHER via ncclGather collective
  // =========================================================================

  auto gather_collective = [&](int32_t /*iter*/) {
    NCCL_CHECK(ncclGroupStart());
    for (int32_t r = 0; r < num_gpus; ++r) {
      CUDA_CHECK(cudaSetDevice(r));
      NCCL_CHECK(ncclGather(send_buf[r], recv_buf[r], count_per_rank,
                             ncclFloat, root, comms[r], streams[r]));
    }
    NCCL_CHECK(ncclGroupEnd());

    for (int32_t r = 0; r < num_gpus; ++r) {
      CUDA_CHECK(cudaSetDevice(r));
      CUDA_CHECK(cudaStreamSynchronize(streams[r]));
    }
  };

  // =========================================================================
  // Benchmark 3: SCATTER via grouped Send/Recv (Setu's approach)
  // =========================================================================
  // Root sends slices to all ranks. Each rank receives into its send_buf
  // (reused as destination). This is what Setu's UnpackOp lowers to.
  // =========================================================================

  auto scatter_sendrecv = [&](int32_t /*iter*/) {
    NCCL_CHECK(ncclGroupStart());
    // Root sends to all ranks
    CUDA_CHECK(cudaSetDevice(root));
    for (int32_t r = 0; r < num_gpus; ++r) {
      NCCL_CHECK(ncclSend(
          static_cast<char*>(static_cast<void*>(recv_buf[root])) +
              r * count_per_rank * sizeof(float),
          count_per_rank, ncclFloat, r, comms[root], streams[root]));
    }
    // Every rank receives from root
    for (int32_t r = 0; r < num_gpus; ++r) {
      CUDA_CHECK(cudaSetDevice(r));
      NCCL_CHECK(ncclRecv(send_buf[r], count_per_rank, ncclFloat, root,
                           comms[r], streams[r]));
    }
    NCCL_CHECK(ncclGroupEnd());

    for (int32_t r = 0; r < num_gpus; ++r) {
      CUDA_CHECK(cudaSetDevice(r));
      CUDA_CHECK(cudaStreamSynchronize(streams[r]));
    }
  };

  // =========================================================================
  // Benchmark 4: SCATTER via ncclScatter collective
  // =========================================================================

  auto scatter_collective = [&](int32_t /*iter*/) {
    NCCL_CHECK(ncclGroupStart());
    for (int32_t r = 0; r < num_gpus; ++r) {
      CUDA_CHECK(cudaSetDevice(r));
      NCCL_CHECK(ncclScatter(recv_buf[r], send_buf[r], count_per_rank,
                              ncclFloat, root, comms[r], streams[r]));
    }
    NCCL_CHECK(ncclGroupEnd());

    for (int32_t r = 0; r < num_gpus; ++r) {
      CUDA_CHECK(cudaSetDevice(r));
      CUDA_CHECK(cudaStreamSynchronize(streams[r]));
    }
  };

  // =========================================================================
  // Run benchmarks
  // =========================================================================

  printf("=== GATHER (all-to-one pack) ===\n");

  {
    auto r = benchmark(gather_sendrecv, warmup, iters);
    printf("  Send/Recv (Setu):  avg=%8.1f us  min=%8.1f us  max=%8.1f us\n",
           r.avg_us, r.min_us, r.max_us);
  }
  {
    auto r = benchmark(gather_collective, warmup, iters);
    printf("  ncclGather:        avg=%8.1f us  min=%8.1f us  max=%8.1f us\n",
           r.avg_us, r.min_us, r.max_us);
  }

  printf("\n=== SCATTER (one-to-all unpack) ===\n");

  {
    auto r = benchmark(scatter_sendrecv, warmup, iters);
    printf("  Send/Recv (Setu):  avg=%8.1f us  min=%8.1f us  max=%8.1f us\n",
           r.avg_us, r.min_us, r.max_us);
  }
  {
    auto r = benchmark(scatter_collective, warmup, iters);
    printf("  ncclScatter:       avg=%8.1f us  min=%8.1f us  max=%8.1f us\n",
           r.avg_us, r.min_us, r.max_us);
  }

  // =========================================================================
  // Run at multiple sizes for a sweep
  // =========================================================================

  printf("\n=== SIZE SWEEP (gather: Send/Recv vs ncclGather) ===\n");
  printf("  %12s  %10s  %14s  %14s  %8s\n",
         "elems/rank", "MB/rank", "sendrecv(us)", "gather(us)", "ratio");

  std::vector<size_t> sweep_sizes = {1024,           4096,           16384,
                                     65536,          262144,         1024 * 1024,
                                     4 * 1024 * 1024, 16 * 1024 * 1024};

  for (size_t sz : sweep_sizes) {
    // Reallocate if needed
    if (sz > count_per_rank) {
      for (int32_t i = 0; i < num_gpus; ++i) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaFree(send_buf[i]));
        CUDA_CHECK(cudaFree(recv_buf[i]));
        CUDA_CHECK(cudaMalloc(&send_buf[i], sz * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&recv_buf[i],
                              sz * num_gpus * sizeof(float)));
      }
      count_per_rank = sz;
      total_count = sz * num_gpus;
    }

    size_t this_count = sz;

    auto sr = [&](int32_t) {
      NCCL_CHECK(ncclGroupStart());
      for (int32_t r = 0; r < num_gpus; ++r) {
        CUDA_CHECK(cudaSetDevice(r));
        NCCL_CHECK(ncclSend(send_buf[r], this_count, ncclFloat, root,
                             comms[r], streams[r]));
      }
      CUDA_CHECK(cudaSetDevice(root));
      for (int32_t r = 0; r < num_gpus; ++r) {
        NCCL_CHECK(ncclRecv(
            static_cast<char*>(static_cast<void*>(recv_buf[root])) +
                r * this_count * sizeof(float),
            this_count, ncclFloat, r, comms[root], streams[root]));
      }
      NCCL_CHECK(ncclGroupEnd());
      for (int32_t r = 0; r < num_gpus; ++r) {
        CUDA_CHECK(cudaSetDevice(r));
        CUDA_CHECK(cudaStreamSynchronize(streams[r]));
      }
    };

    auto gc = [&](int32_t) {
      NCCL_CHECK(ncclGroupStart());
      for (int32_t r = 0; r < num_gpus; ++r) {
        CUDA_CHECK(cudaSetDevice(r));
        NCCL_CHECK(ncclGather(send_buf[r], recv_buf[r], this_count,
                               ncclFloat, root, comms[r], streams[r]));
      }
      NCCL_CHECK(ncclGroupEnd());
      for (int32_t r = 0; r < num_gpus; ++r) {
        CUDA_CHECK(cudaSetDevice(r));
        CUDA_CHECK(cudaStreamSynchronize(streams[r]));
      }
    };

    auto r1 = benchmark(sr, 5, 50);
    auto r2 = benchmark(gc, 5, 50);

    printf("  %12zu  %8.2f MB  %12.1f us  %12.1f us  %7.2fx\n", sz,
           sz * sizeof(float) / (1024.0 * 1024.0), r1.avg_us, r2.avg_us,
           r1.avg_us / r2.avg_us);
  }

  // =========================================================================
  // Size sweep: scatter comparison
  // =========================================================================

  printf("\n=== SIZE SWEEP (scatter: Send/Recv vs ncclScatter) ===\n");
  printf("  %12s  %10s  %14s  %14s  %8s\n",
         "elems/rank", "MB/rank", "sendrecv(us)", "scatter(us)", "ratio");

  for (size_t sz : sweep_sizes) {
    if (sz > count_per_rank) {
      for (int32_t i = 0; i < num_gpus; ++i) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaFree(send_buf[i]));
        CUDA_CHECK(cudaFree(recv_buf[i]));
        CUDA_CHECK(cudaMalloc(&send_buf[i], sz * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&recv_buf[i],
                              sz * num_gpus * sizeof(float)));
      }
      count_per_rank = sz;
      total_count = sz * num_gpus;
    }

    size_t this_count = sz;

    auto sr = [&](int32_t) {
      NCCL_CHECK(ncclGroupStart());
      CUDA_CHECK(cudaSetDevice(root));
      for (int32_t r = 0; r < num_gpus; ++r) {
        NCCL_CHECK(ncclSend(
            static_cast<char*>(static_cast<void*>(recv_buf[root])) +
                r * this_count * sizeof(float),
            this_count, ncclFloat, r, comms[root], streams[root]));
      }
      for (int32_t r = 0; r < num_gpus; ++r) {
        CUDA_CHECK(cudaSetDevice(r));
        NCCL_CHECK(ncclRecv(send_buf[r], this_count, ncclFloat, root,
                             comms[r], streams[r]));
      }
      NCCL_CHECK(ncclGroupEnd());
      for (int32_t r = 0; r < num_gpus; ++r) {
        CUDA_CHECK(cudaSetDevice(r));
        CUDA_CHECK(cudaStreamSynchronize(streams[r]));
      }
    };

    auto sc = [&](int32_t) {
      NCCL_CHECK(ncclGroupStart());
      for (int32_t r = 0; r < num_gpus; ++r) {
        CUDA_CHECK(cudaSetDevice(r));
        NCCL_CHECK(ncclScatter(recv_buf[r], send_buf[r], this_count,
                                ncclFloat, root, comms[r], streams[r]));
      }
      NCCL_CHECK(ncclGroupEnd());
      for (int32_t r = 0; r < num_gpus; ++r) {
        CUDA_CHECK(cudaSetDevice(r));
        CUDA_CHECK(cudaStreamSynchronize(streams[r]));
      }
    };

    auto r1 = benchmark(sr, 5, 50);
    auto r2 = benchmark(sc, 5, 50);

    printf("  %12zu  %8.2f MB  %12.1f us  %12.1f us  %7.2fx\n", sz,
           sz * sizeof(float) / (1024.0 * 1024.0), r1.avg_us, r2.avg_us,
           r1.avg_us / r2.avg_us);
  }

  // =========================================================================
  // Cleanup
  // =========================================================================

  for (int32_t i = 0; i < num_gpus; ++i) {
    CUDA_CHECK(cudaSetDevice(i));
    CUDA_CHECK(cudaFree(send_buf[i]));
    CUDA_CHECK(cudaFree(recv_buf[i]));
    CUDA_CHECK(cudaStreamDestroy(streams[i]));
    ncclCommDestroy(comms[i]);
  }

  printf("\nDone.\n");
  return 0;
}

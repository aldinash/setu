# ~~~
# ParallelismSetup.cmake
# Configures build parallelism for both general compilation and NVCC
#
# This module handles:
# - Automatic detection of CPU cores
# - Setting CMAKE_BUILD_PARALLEL_LEVEL
# - Configuring NVCC thread parallelism for CUDA 11.2+
# - Adjusting job counts to prevent CPU oversubscription
# - Setting up Ninja job pools
#
# Environment Variables:
# - CMAKE_BUILD_PARALLEL_LEVEL: Override number of parallel jobs
# - MAX_JOBS: Alternative to CMAKE_BUILD_PARALLEL_LEVEL
# - NVCC_THREADS: Number of threads for nvcc compilation (default: 8)
# ~~~

# ##################################################################################################
# Determine the number of parallel jobs
# ##################################################################################################
if(NOT DEFINED CMAKE_BUILD_PARALLEL_LEVEL)
  # Check environment variables first
  if(DEFINED ENV{CMAKE_BUILD_PARALLEL_LEVEL})
    set(CMAKE_BUILD_PARALLEL_LEVEL $ENV{CMAKE_BUILD_PARALLEL_LEVEL})
  elseif(DEFINED ENV{MAX_JOBS})
    set(CMAKE_BUILD_PARALLEL_LEVEL $ENV{MAX_JOBS})
  else()
    # Auto-detect CPU cores
    include(ProcessorCount)
    ProcessorCount(NUM_CORES)
    if(NUM_CORES EQUAL 0)
      set(NUM_CORES 1)
    endif()
    set(CMAKE_BUILD_PARALLEL_LEVEL ${NUM_CORES})
  endif()
  set(CMAKE_BUILD_PARALLEL_LEVEL
      ${CMAKE_BUILD_PARALLEL_LEVEL}
      CACHE STRING "Number of parallel build jobs")
endif()

message(STATUS "CMAKE_BUILD_PARALLEL_LEVEL: ${CMAKE_BUILD_PARALLEL_LEVEL}")

# ##################################################################################################
# Configure NVCC parallelism for CUDA builds
# ##################################################################################################
if(DEFINED SETU_GPU_LANG AND SETU_GPU_LANG STREQUAL "CUDA")
  # Check if CUDA version supports --threads flag (11.2+)
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "11.2")
    # Determine NVCC threads
    if(DEFINED ENV{NVCC_THREADS})
      set(NVCC_THREADS $ENV{NVCC_THREADS})
    else()
      set(NVCC_THREADS 8)
    endif()

    message(STATUS "NVCC parallel threads: ${NVCC_THREADS}")

    # Add --threads flag to GPU compilation flags
    list(APPEND SETU_GPU_FLAGS "--threads=${NVCC_THREADS}")

    # Adjust CMAKE_BUILD_PARALLEL_LEVEL to prevent CPU oversubscription NVCC with --threads uses
    # more CPU resources per compilation unit Formula: adjusted_jobs = original_jobs * 4 /
    # nvcc_threads
    math(EXPR ADJUSTED_JOBS "${CMAKE_BUILD_PARALLEL_LEVEL} * 4 / ${NVCC_THREADS}")
    if(ADJUSTED_JOBS LESS 1)
      set(ADJUSTED_JOBS 1)
    endif()

    if(NOT ADJUSTED_JOBS EQUAL CMAKE_BUILD_PARALLEL_LEVEL)
      message(
        STATUS "Adjusted parallel jobs from ${CMAKE_BUILD_PARALLEL_LEVEL} to ${ADJUSTED_JOBS} "
               "due to NVCC thread parallelism")
      set(CMAKE_BUILD_PARALLEL_LEVEL ${ADJUSTED_JOBS})
    endif()
  else()
    message(STATUS "CUDA ${CMAKE_CUDA_COMPILER_VERSION} does not support --threads flag "
                   "(requires 11.2+)")
  endif()
endif()

# ##################################################################################################
# Configure build tool specific settings
# ##################################################################################################
if(CMAKE_GENERATOR STREQUAL "Ninja")
  # Set up job pools for Ninja to respect parallelism limits
  set(CMAKE_JOB_POOL_COMPILE compile)
  set(CMAKE_JOB_POOLS compile=${CMAKE_BUILD_PARALLEL_LEVEL})
  message(STATUS "Configured Ninja job pool with ${CMAKE_BUILD_PARALLEL_LEVEL} jobs")
elseif(CMAKE_GENERATOR MATCHES "Unix Makefiles")
  # For Make, the -j flag is passed externally, but we can set this for reference
  message(STATUS "Unix Makefiles generator: use -j${CMAKE_BUILD_PARALLEL_LEVEL} externally")
endif()

# ##################################################################################################
# Export parallelism settings for subprojects
# ##################################################################################################
set(_SETU_BUILD_PARALLEL_LEVEL
    ${CMAKE_BUILD_PARALLEL_LEVEL}
    CACHE INTERNAL "Setu build parallelism level")

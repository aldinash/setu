include(FetchContent)

FetchContent_Declare(googletest
                     URL https://github.com/google/googletest/archive/refs/tags/v1.13.0.tar.gz)

set(BOOST_ENABLE_CMAKE ON)
set(BOOST_INCLUDE_LIBRARIES thread uuid heap container_hash stacktrace dynamic_bitset)

FetchContent_Declare(
  Boost
  URL https://github.com/boostorg/boost/releases/download/boost-1.88.0/boost-1.88.0-cmake.tar.gz)

FetchContent_Declare(
  zmq URL https://github.com/zeromq/libzmq/releases/download/v4.3.4/zeromq-4.3.4.tar.gz)

FetchContent_Declare(cppzmq URL https://github.com/zeromq/cppzmq/archive/refs/tags/v4.10.0.tar.gz)

set(ZMQ_BUILD_TESTS
    OFF
    CACHE BOOL "Build ZeroMQ tests" FORCE)
set(ZMQ_BUILD_DRAFT_API
    OFF
    CACHE BOOL "Build ZeroMQ draft API" FORCE)

FetchContent_MakeAvailable(googletest Boost zmq cppzmq)

# ---------------------------------------------------------------------------
# NCCL: required (nccl.h + libnccl)
# We try to locate nccl.h and libnccl from various locations including pip packages
# ---------------------------------------------------------------------------

# Get conda prefix from SETU_PYTHON_EXECUTABLE if not already set
if(NOT CONDA_PREFIX AND DEFINED SETU_PYTHON_EXECUTABLE)
  get_filename_component(_python_bin_dir "${SETU_PYTHON_EXECUTABLE}" DIRECTORY)
  get_filename_component(CONDA_PREFIX "${_python_bin_dir}" DIRECTORY)
endif()

# Build search paths list for pip-installed nvidia packages in conda env
set(NCCL_SEARCH_PATHS)

# Add conda/pip site-packages path if available (check multiple Python versions)
if(CONDA_PREFIX)
  list(APPEND NCCL_SEARCH_PATHS "${CONDA_PREFIX}/lib/python3.12/site-packages/nvidia/nccl")
  list(APPEND NCCL_SEARCH_PATHS "${CONDA_PREFIX}/lib/python3.11/site-packages/nvidia/nccl")
  list(APPEND NCCL_SEARCH_PATHS "${CONDA_PREFIX}/lib/python3.10/site-packages/nvidia/nccl")
endif()

# Add standard search paths
list(APPEND NCCL_SEARCH_PATHS
    /usr/include
    /usr/local/include
    /usr/local/cuda/include
    /usr/lib
    /usr/local/lib
    /usr/lib64
    /usr/local/cuda/lib64)

find_path(NCCL_INCLUDE_DIR
          NAMES nccl.h
          PATHS ${NCCL_SEARCH_PATHS}
          HINTS $ENV{NCCL_ROOT} $ENV{CUDA_HOME} $ENV{CUDA_PATH}
          PATH_SUFFIXES include)

find_library(NCCL_LIBRARY
             NAMES nccl libnccl nccl.so.2 libnccl.so.2
             PATHS ${NCCL_SEARCH_PATHS}
             HINTS $ENV{NCCL_ROOT} $ENV{CUDA_HOME} $ENV{CUDA_PATH}
             PATH_SUFFIXES lib lib64)

if(NCCL_INCLUDE_DIR AND NCCL_LIBRARY)
  message(STATUS "Found NCCL include: ${NCCL_INCLUDE_DIR}, lib: ${NCCL_LIBRARY}")
  set(NCCL_FOUND TRUE CACHE BOOL "NCCL library found")
  set(NCCL_INCLUDE_DIRS ${NCCL_INCLUDE_DIR} CACHE PATH "NCCL include dir")
  set(NCCL_LIBRARIES ${NCCL_LIBRARY} CACHE FILEPATH "NCCL library")
else()
  message(WARNING "NCCL not found; GPU communication features will be disabled. "
                  "Set NCCL_ROOT, CUDA_HOME, or CUDA_PATH environment variable to locate NCCL.")
  set(NCCL_FOUND FALSE CACHE BOOL "NCCL library found")
endif()

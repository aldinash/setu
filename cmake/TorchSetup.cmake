# Update cmake's `CMAKE_PREFIX_PATH` with torch location. This allows us to pick torch directly from
# the environment.
append_cmake_prefix_path("torch" "torch.utils.cmake_prefix_path")
# Import torch cmake configuration. Torch also imports CUDA languages with some customizations, so
# there is no need to do this explicitly with check_language/enable_language, etc.
find_package(Torch REQUIRED)
# ##################################################################################################
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

execute_process(COMMAND python3 -c "import torch; print(torch._C._PYBIND11_COMPILER_TYPE, end='')"
                OUTPUT_VARIABLE _PYBIND11_COMPILER_TYPE)
execute_process(COMMAND python3 -c "import torch; print(torch._C._PYBIND11_STDLIB, end='')"
                OUTPUT_VARIABLE _PYBIND11_STDLIB)
execute_process(COMMAND python3 -c "import torch; print(torch._C._PYBIND11_BUILD_ABI, end='')"
                OUTPUT_VARIABLE _PYBIND11_BUILD_ABI)

message(STATUS "PYBIND11_COMPILER_TYPE:" ${_PYBIND11_COMPILER_TYPE})
message(STATUS "PYBIND11_STDLIB:" ${_PYBIND11_STDLIB})
message(STATUS "PYBIND11_BUILD_ABI:" ${_PYBIND11_BUILD_ABI})

add_compile_definitions(
  PYBIND11_COMPILER_TYPE="${_PYBIND11_COMPILER_TYPE}" PYBIND11_STDLIB="${_PYBIND11_STDLIB}"
  PYBIND11_BUILD_ABI="${_PYBIND11_BUILD_ABI}")
# ##################################################################################################
# Normally `torch.utils.cpp_extension.CUDAExtension` would add `libtorch_python.so` for linking
# against an extension. Torch's cmake configuration does not include this library (presumably since
# the cmake config is used for standalone C++ binaries that link against torch). The
# `libtorch_python.so` library defines some of the glue code between torch/python via pybind and is
# required by SETU extensions for this reason. So, add it by manually using
# `append_torchlib_if_found` from torch's cmake setup.
append_torchlib_if_found(torch_python)
# ##################################################################################################
# Set up GPU language and check the torch version and warn if it isn't what is expected.
if(CUDA_FOUND)
  set(SETU_GPU_LANG "CUDA")

  if(NOT Torch_VERSION VERSION_EQUAL ${TORCH_SUPPORTED_VERSION_CUDA})
    message(WARNING "Pytorch version ${TORCH_SUPPORTED_VERSION_CUDA} "
                    "expected for CUDA build, saw ${Torch_VERSION} instead.")
  endif()
else()
  message(FATAL_ERROR "Can't find CUDA installation.")
endif()
# ##################################################################################################
# Override the GPU architectures detected by cmake/torch and filter them by the supported versions
# for the current language. The final set of arches is stored in `SETU_GPU_ARCHES`.
override_gpu_arches(SETU_GPU_ARCHES ${SETU_GPU_LANG} "${${SETU_GPU_LANG}_SUPPORTED_ARCHS}")

# Query torch for additional GPU compilation flags for the given `SETU_GPU_LANG`. The final set of
# arches is stored in `SETU_GPU_FLAGS`.
get_torch_gpu_compiler_flags(SETU_GPU_FLAGS ${SETU_GPU_LANG})
# ##################################################################################################

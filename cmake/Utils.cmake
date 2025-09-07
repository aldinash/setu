# ##################################################################################################
include(CMakeParseArguments)
# ##################################################################################################
# Find Python from executable path
macro(FIND_PYTHON_FROM_EXECUTABLE executable supported_versions)
  get_filename_component(executable ${executable} REALPATH)
  # cmake-lint: disable=C0103
  set(Python_EXECUTABLE ${executable})
  find_package(Python COMPONENTS Interpreter Development.Module)
  if(NOT Python_FOUND)
    message(FATAL_ERROR "Unable to find Python matching: ${executable}.")
  endif()
  set(ver "${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}")

  # Convert the parameter to a proper CMake list variable
  set(supported_versions_list ${supported_versions})

  if(NOT ver IN_LIST supported_versions_list)
    message(FATAL_ERROR "Python version (${ver}) is not supported: ${supported_versions}.")
  endif()
  message(STATUS "Found Python matching: ${executable}.")
endmacro()
# ##################################################################################################
# Run Python command and capture output
function(run_python out expr err_msg)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" "-c" "${expr}"
    OUTPUT_VARIABLE python_out
    RESULT_VARIABLE python_error_code
    ERROR_VARIABLE python_stderr
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(NOT python_error_code EQUAL 0)
    message(FATAL_ERROR "${err_msg}: ${python_stderr}")
  endif()
  set(${out}
      ${python_out}
      PARENT_SCOPE)
endfunction()

# Append path to CMAKE_PREFIX_PATH for given package
macro(APPEND_CMAKE_PREFIX_PATH pkg expr)
  run_python(prefix_path "import ${pkg}; print(${expr})" "Failed to locate ${pkg} path")
  list(APPEND CMAKE_PREFIX_PATH ${prefix_path})
endmacro()

# Get GPU compiler flags for Torch
function(get_torch_gpu_compiler_flags out_gpu_flags gpu_lang)
  if(${gpu_lang} STREQUAL "CUDA")
    run_python(
      gpu_flags
      "from torch.utils.cpp_extension import COMMON_NVCC_FLAGS; print(';'.join(COMMON_NVCC_FLAGS))"
      "Failed to determine Torch NVCC compiler flags")
    if(CUDA_VERSION VERSION_GREATER_EQUAL 11.8)
      list(APPEND gpu_flags "-DENABLE_FP8_E5M2")
    endif()
  endif()
  set(${out_gpu_flags}
      ${gpu_flags}
      PARENT_SCOPE)
endfunction()
# ##################################################################################################
# Override GPU architectures detected by cmake/torch
macro(OVERRIDE_GPU_ARCHES gpu_arches gpu_lang gpu_supported_archs)
  set(gpu_supported_archs_list ${gpu_supported_archs})
  message(STATUS "${gpu_lang} supported arches: ${gpu_supported_archs_list}")
  if(${gpu_lang} STREQUAL "CUDA")
    string(REGEX MATCHALL "-gencode arch=[^ ]+" cuda_arch_flags ${CMAKE_CUDA_FLAGS})
    string(REGEX REPLACE "-gencode arch=[^ ]+ *" "" CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS})
    if(NOT cuda_arch_flags)
      message(FATAL_ERROR "No architecture flags found in CMAKE_CUDA_FLAGS.")
    endif()
    set(${gpu_arches})
    foreach(arch ${cuda_arch_flags})
      string(REGEX MATCH "arch=compute_\([0-9]+a?\)" compute ${arch})
      if(compute)
        set(compute ${CMAKE_MATCH_1})
      endif()
      string(REGEX MATCH "code=sm_\([0-9]+a?\)" sm ${arch})
      if(sm)
        set(sm ${CMAKE_MATCH_1})
      endif()
      string(REGEX MATCH "code=compute_\([0-9]+a?\)" code ${arch})
      if(code)
        set(code ${CMAKE_MATCH_1})
      endif()
      if(NOT compute)
        message(FATAL_ERROR "Could not determine virtual architecture from: ${arch}.")
      endif()
      if(sm)
        set(virt "")
        set(code_arch ${sm})
      else()
        set(virt "-virtual")
        set(code_arch ${code})
      endif()
      string(REGEX REPLACE "\([0-9]+\)\([0-9]\)" "\\1.\\2" code_ver ${code_arch})
      if(code_ver IN_LIST gpu_supported_archs_list)
        list(APPEND ${gpu_arches} "${code_arch}${virt}")
      else()
        message(STATUS "Discarding unsupported CUDA arch ${code_ver}.")
      endif()
    endforeach()
  endif()
  message(STATUS "${gpu_lang} target arches: ${${gpu_arches}}")
endmacro()
# ##################################################################################################

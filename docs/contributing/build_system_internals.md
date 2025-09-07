# Build System Internals

## Overview

Setu uses a highly optimized CMake + Ninja build system designed for fast incremental builds during development. The build system has been specifically tuned for high-core-count systems (64+ cores) and includes several advanced optimizations.

## Build Performance

- **Incremental build time**: ~3.2 seconds for core file changes
- **70% faster** than baseline through optimizations
- **Target**: Sub-1-second builds for critical development paths

## Core Optimizations

### 1. Object Libraries for Single Compilation

**Problem**: Common source files (like `Logging.cpp`) were compiled multiple times for different targets.

**Solution**: OBJECT libraries compile sources once and reuse across targets.

```cmake
# Create object library for common sources (compiled only once!)
add_library(setu_common_objects OBJECT ${COMMON_SRC})
target_link_libraries(setu_common_objects PRIVATE setu_common)

# Reuse in multiple targets
target_sources(${TARGET} PRIVATE $<TARGET_OBJECTS:setu_common_objects>)
```

**Impact**: Eliminated redundant compilation of common files.

### 2. Module-Specific Precompiled Headers (PCH)

**Heavy headers precompiled once per module:**

- **Commons PCH**: `StdCommon.h` + JSON (applied to `setu_common_objects`)
- **Native PCH**: `StdCommon.h` + `TorchCommon.h` + `BoostCommon.h` + PyBind11 + DDSketch (applied to `_native` targets)
- **Kernels PCH**: `StdCommon.h` + `TorchCommon.h` + PyBind11 (applied to `setu_kernels_cuda_objects` and `_kernels` targets)

**Note**: The `_kernels` and `_kernels_static` targets use `PrecompiledKernelsHeaders.h`, not `PrecompiledNativeHeaders.h`

```cmake
# Commons PCH - basic headers for common object library
target_precompile_headers(setu_common_objects PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/csrc/include/setu/commons/PrecompiledCommonHeaders.h"
)

# Kernels PCH - CUDA-specific headers for kernel object library and targets
target_precompile_headers(setu_kernels_cuda_objects PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/csrc/include/setu/kernels/PrecompiledKernelHeaders.h"
)
target_precompile_headers(_kernels PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/csrc/include/setu/kernels/PrecompiledKernelHeaders.h")
target_precompile_headers(_kernels_static PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/csrc/include/setu/kernels/PrecompiledKernelHeaders.h")

# Native PCH - comprehensive headers for native C++ targets
target_precompile_headers(_native PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/csrc/include/setu/native/PrecompiledNativeHeaders.h")
target_precompile_headers(_native_static PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/csrc/include/setu/native/PrecompiledNativeHeaders.h")
```

**Impact**: Eliminates parsing of heavy dependencies like PyTorch, Boost, PyBind11.

### 3. Enhanced Compiler Flags

```cmake
set(CMAKE_CXX_FLAGS "-g -Wall -Wextra -fvisibility=hidden -pipe -fdiagnostics-color=always")
set(CMAKE_CXX_FLAGS_DEBUG "-g1 -O0")  # Faster debug info
```

**Key flags:**
- `-pipe`: Use pipes instead of temp files
- `-g1`: Minimal debug info for faster compilation
- `-fdiagnostics-color=always`: Colored output

### 4. ccache Optimization

**Dynamic cache sizing with compression:**

```bash
# Memory-based ccache sizing (5-10% of system RAM)
# Current system: 755GB RAM → 20GB cache (capped at max)
ccache --set-config max_size=$(CCACHE_SIZE_GB)G
ccache --set-config compression=true
ccache --set-config compression_level=6
ccache --set-config sloppiness=pch_defines,time_macros
```

**Dynamic sizing:**
- **Small machines (<128GB RAM)**: 5% of total memory, minimum 2GB
- **Large machines (≥128GB RAM)**: 10% of total memory, maximum 20GB
- **Automatic scaling**: Adapts to available system resources

**Features:**
- RAM disk cache location for fastest I/O (on machines ≥128GB RAM)
- PCH-friendly settings
- Automatic cache management

### 5. NVCC Thread Auto-Tuning

**Optimized for high-core systems:**

```cmake
if(N GREATER 16)
  math(EXPR NVCC_THREADS "${N} * 3 / 4")  # Use 75% cores
elseif(N GREATER 8)  
  math(EXPR NVCC_THREADS "${N} * 2 / 3")  # Use 67% cores
endif()
```

**Impact**: Scales CUDA compilation to available cores.

### 6. Mold Linker

**Fast linking with mold:**

```cmake
set(CMAKE_LINKER_TYPE MOLD)
```

**Impact**: 50%+ faster linking than default `ld`.

### 7. Dynamic Build Parallelism

**Ninja job pools optimized for system:**

```cmake
if(N GREATER 16)
  set(CMAKE_JOB_POOLS "compile=${N};link=4")  # Max compile, limit linking
elseif(N GREATER 8)
  math(EXPR COMPILE_JOBS "${N} * 3 / 4")
  set(CMAKE_JOB_POOLS "compile=${COMPILE_JOBS};link=3")
endif()
```

**Strategy**: Maximize compilation parallelism, serialize linking to avoid memory pressure.

## Build Architecture

### Directory Structure

```
build/debug/          # Build artifacts (symlinked to RAM disk)
├── CMakeFiles/       # Target-specific build files
├── lib*.a           # Static libraries
└── *.so             # Python extensions
```

### Target Dependencies

```
setu_common_objects  ← Common sources (compiled once)
├── _kernels         ← Kernel Python module
├── _native          ← Native Python module  
├── _kernels_static  ← Kernel static library
└── _native_static   ← Native static library
```

### CMake Module Organization

- `cmake/Extensions.cmake`: Core build logic and optimizations
- `cmake/TorchSetup.cmake`: PyTorch integration and NVCC tuning
- `cmake/CcacheSetup.cmake`: Compiler cache configuration
- `cmake/Dependencies.cmake`: External dependencies
- `cmake/Testing.cmake`: Test configuration

## RAM Disk Integration

**Automatic RAM disk setup:**

```makefile
RAMDISK_DIR := /dev/shm/setu-$(PROJECT_HASH)
CCACHE_RAMDISK_DIR := /dev/shm/setu-ccache-$(PROJECT_HASH)
```

**Benefits:**
- Build artifacts stored in memory
- ccache on RAM disk for fastest access
- Automatic cleanup on `make clean`

## Incremental Build Flow

1. **File Change Detection**: Ninja tracks dependencies
2. **Minimal Recompilation**: Only changed files + dependents
3. **PCH Reuse**: Heavy headers already compiled
4. **Object Library Reuse**: Common code not recompiled
5. **ccache Hit**: Previously compiled objects retrieved instantly
6. **Fast Linking**: Mold linker for rapid final step

## Performance Monitoring

**ccache statistics:**
```bash
make ccache/stats  # View hit rates and cache usage
```

**Build timing:**
- Use `time make build/native_incremental` for benchmarks
- Monitor individual compilation steps in build logs

## Common Header Organization

**Existing common headers leveraged:**

- `commons/StdCommon.h`: All standard library headers
- `commons/TorchCommon.h`: PyTorch and CUDA headers  
- `commons/BoostCommon.h`: Boost threading and containers
- `commons/ZmqCommon.h`: ZeroMQ networking

**Strategy**: Include heavy external dependencies in common headers, then reference in PCH.

## Future Optimizations

**Next targets for sub-1-second builds:**

1. **Template explicit instantiation** for common patterns
2. **Header dependency elimination** (IWYU analysis)  
3. **Debug info elimination** (`-g0`) for dev builds
4. **Module-level shared libraries** to avoid full relinking

## Troubleshooting

**Clean builds:**
```bash
make clean           # Clean all artifacts + RAM disk
make build/native    # Full rebuild (required after clean)
```

**ccache issues:**
```bash
make ccache/clear    # Clear cache if corrupted
```

**Build debugging:**
```bash
# View full build log
cat logs/cmake_build_native_incr_*.log

# Check ninja build graph
cd build/debug && ninja -t graph | dot -Tpng > deps.png
```

This optimized build system provides industry-leading incremental build performance while maintaining code quality and development velocity.

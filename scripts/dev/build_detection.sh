#!/bin/bash
# Build detection utilities for intelligent build system
# Should be sourced from other scripts, not executed directly

set -e
set -o pipefail

_script_dir="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${_script_dir}/logging.sh"
source "${_script_dir}/build_env.sh"

# Global variable to store detailed reason
export BUILD_REASON=""

# Check if setu package is installed in editable mode (optimized)
is_package_installed() {
    pip show setu >/dev/null 2>&1 && return 0 || return 1
}

# Get the pip install marker file path
get_pip_install_marker() {
    echo ".pip_install_marker"
}

# Create pip install marker after successful pip install
create_pip_install_marker() {
    local marker_file
    marker_file=$(get_pip_install_marker)
    touch "$marker_file"
}

# Check if pip install marker exists and get its path
get_pip_install_time() {
    local marker_file
    marker_file=$(get_pip_install_marker)

    if [ -f "$marker_file" ]; then
        echo "$marker_file"
        return 0
    fi

    # No marker file exists
    return 1
}

# Check if native build exists and is up to date
is_native_build_valid() {
    local build_type build_subdir
    build_type=$(get_build_type)
    build_subdir="build/$(echo "$build_type" | tr '[:upper:]' '[:lower:]')"

    # Check if build.ninja exists
    [ -f "$build_subdir/build.ninja" ]
}

# Check if any C++ source files have been modified since last build
cpp_files_modified_since_build() {
    local build_type build_subdir
    build_type=$(get_build_type)
    build_subdir="build/$(echo "$build_type" | tr '[:upper:]' '[:lower:]')"

    # If no build exists, consider files modified
    if [ ! -f "$build_subdir/build.ninja" ]; then
        return 0 # true - files are "modified"
    fi

    # Fast check: use find with -newer and exit on first match
    find csrc -type f \( -name "*.cpp" -o -name "*.h" -o -name "*.cu" -o -name "*.cuh" \) \
        -newer "$build_subdir/build.ninja" -print -quit 2>/dev/null | grep -q .
}

# Check if any CMake files have been modified since last build
cmake_files_modified_since_build() {
    local build_type build_subdir
    build_type=$(get_build_type)
    build_subdir="build/$(echo "$build_type" | tr '[:upper:]' '[:lower:]')"

    # If no build exists, consider files modified
    if [ ! -f "$build_subdir/build.ninja" ]; then
        return 0 # true - files are "modified"
    fi

    # Fast check: use find with -newer and exit on first match
    find . -maxdepth 2 -type f \( -name "CMakeLists.txt" -o -name "*.cmake" \) \
        -not -path "./build/*" -not -path "./env/*" \
        -newer "$build_subdir/build.ninja" -print -quit 2>/dev/null | grep -q .
}

# Check if new C++ files have been added (not tracked by incremental build)
new_cpp_files_exist() {
    local build_type build_subdir
    build_type=$(get_build_type)
    build_subdir="build/$(echo "$build_type" | tr '[:upper:]' '[:lower:]')"

    # If no build exists, any files are "new"
    if [ ! -f "$build_subdir/build.ninja" ]; then
        find csrc -type f \( -name "*.cpp" -o -name "*.cu" \) -print -quit 2>/dev/null | grep -q .
        return $?
    fi

    # Fast check: find any source files newer than build.ninja
    find csrc -type f \( -name "*.cpp" -o -name "*.cu" \) \
        -newer "$build_subdir/build.ninja" -print -quit 2>/dev/null | grep -q .
}

# Check if environment has changed significantly
environment_changed() {
    local build_type build_subdir
    build_type=$(get_build_type)
    build_subdir="build/$(echo "$build_type" | tr '[:upper:]' '[:lower:]')"

    # If no build, environment is considered "changed"
    if [ ! -f "$build_subdir/build.ninja" ]; then
        return 0
    fi

    # Fast check: only check most common environment files
    [ -f "setup.py" ] && [ "setup.py" -nt "$build_subdir/build.ninja" ] && return 0
    [ -f "pyproject.toml" ] && [ "pyproject.toml" -nt "$build_subdir/build.ninja" ] && return 0

    return 1 # No significant environment changes
}

# Check if C++ files have been added/deleted (compare file lists)
file_list_changed() {
    local build_ninja="$1"
    local file_list_cache
    file_list_cache="$(dirname "$build_ninja")/.file_list_cache"

    # Get current file list (sorted for consistent comparison)
    local current_files_temp
    current_files_temp=$(mktemp)
    find csrc \( -name "*.cpp" -o -name "*.h" -o -name "*.cu" -o -name "*.cuh" \) 2>/dev/null | sort >"$current_files_temp"

    # If no cache file exists, assume files changed (need full rebuild)
    if [ ! -f "$file_list_cache" ]; then
        rm -f "$current_files_temp"
        return 0 # Treat as changed (will trigger full rebuild which creates cache)
    fi

    # Compare with cached file list
    if ! cmp -s "$current_files_temp" "$file_list_cache"; then
        rm -f "$current_files_temp"
        return 0 # File list changed (additions/deletions detected)
    fi

    rm -f "$current_files_temp"
    return 1 # No change
}

# Update file list cache (called after successful cmake configure)
update_file_list_cache() {
    local build_ninja="$1"
    local file_list_cache
    file_list_cache="$(dirname "$build_ninja")/.file_list_cache"

    # Get current file list and save it (sorted for consistency)
    find csrc \( -name "*.cpp" -o -name "*.h" -o -name "*.cu" -o -name "*.cuh" \) 2>/dev/null | sort >"$file_list_cache"
}

# Determine the optimal build strategy (fast and correct)
determine_build_strategy() {
    # Reset reason
    BUILD_REASON=""

    # Environment files check (compare against pip install marker)
    # If any of these files are modified after package installation, we need to rebuild the package
    local pip_install_marker
    if is_package_installed && pip_install_marker=$(get_pip_install_time); then
        if [ -f "setup.py" ] && [ "setup.py" -nt "$pip_install_marker" ]; then
            echo "full_install - setup.py modified after package installation"
            return
        fi
        if [ -f "pyproject.toml" ] && [ "pyproject.toml" -nt "$pip_install_marker" ]; then
            echo "full_install - pyproject.toml modified after package installation"
            return
        fi
        if [ -f "requirements.txt" ] && [ "requirements.txt" -nt "$pip_install_marker" ]; then
            echo "full_install - requirements.txt modified after package installation"
            return
        fi
    fi

    # Calculate build paths
    local build_type build_ninja
    build_type=$(get_build_type)
    build_ninja="build/$(echo "$build_type" | tr '[:upper:]' '[:lower:]')/build.ninja"

    # Fast exit: Check if native build exists
    if [ ! -f "$build_ninja" ]; then
        if is_package_installed; then
            echo "native_build - build.ninja not found but package installed"
        else
            echo "full_install - setu package not installed"
        fi
        return
    fi

    # CMake files check (main file + cmake directory)
    if [ -f "CMakeLists.txt" ] && [ "CMakeLists.txt" -nt "$build_ninja" ]; then
        echo "native_build - CMakeLists.txt modified"
        return
    fi
    if [ -d "cmake" ] && find cmake -name "*.cmake" -newer "$build_ninja" -print -quit 2>/dev/null | grep -q .; then
        echo "native_build - CMake files in cmake/ directory modified"
        return
    fi

    # C++ files check: handle new/modified/deleted files
    # Strategy: Count changes (add/delete) = full rebuild, source mods = full rebuild, header mods = incremental

    # Check if any files have changed at all
    if find csrc -newer "$build_ninja" \( -name "*.cpp" -o -name "*.h" -o -name "*.cu" -o -name "*.cuh" \) -print -quit 2>/dev/null | grep -q .; then
        # Files have changed - now check if file list changed (additions/deletions)
        if file_list_changed "$build_ninja"; then
            echo "native_build - C++ files added/deleted - file list changed"
            return
        fi
    fi

    # File list same, so only modifications - incremental build is sufficient
    echo "incremental_build - C++ files modified (no additions/deletions)"
    return
}

# Determine the optimal test build strategy
determine_test_build_strategy() {
    # Reset reason
    BUILD_REASON=""

    # Calculate build paths correctly
    local build_type build_ninja
    build_type=$(get_build_type)
    build_ninja="build/$(echo "$build_type" | tr '[:upper:]' '[:lower:]')/build.ninja"

    # Check if build exists first
    if [ ! -f "$build_ninja" ]; then
        echo "full_test_build - build.ninja not found"
        return
    fi

    # Check if test binaries exist (look for specific test executables)
    local build_subdir
    build_subdir="build/$(echo "$build_type" | tr '[:upper:]' '[:lower:]')"

    local test_binaries_exist=false
    # Look for common test binary names
    if [ -f "$build_subdir/native_tests" ] || [ -f "$build_subdir/kernel_tests" ] || find "$build_subdir" -name "*_test" -executable -type f -print -quit 2>/dev/null | grep -q .; then
        test_binaries_exist=true
    fi

    if [ "$test_binaries_exist" = false ]; then
        echo "full_test_build - no test binaries found"
        return
    fi

    # CMake files check
    if [ -f "CMakeLists.txt" ] && [ "CMakeLists.txt" -nt "$build_ninja" ]; then
        echo "full_test_build - CMakeLists.txt modified"
        return
    fi
    if [ -d "cmake" ] && find cmake -name "*.cmake" -newer "$build_ninja" -print -quit 2>/dev/null | grep -q .; then
        echo "full_test_build - CMake files in cmake/ directory modified"
        return
    fi

    # Check if any files have changed at all
    if find csrc -newer "$build_ninja" \( -name "*.cpp" -o -name "*.h" -o -name "*.cu" -o -name "*.cuh" \) -print -quit 2>/dev/null | grep -q .; then
        # Files have changed - now check if file list changed (additions/deletions)
        if file_list_changed "$build_ninja"; then
            echo "native_build - C++ files added/deleted - file list changed"
            return
        fi
    fi

    # File list same, so only modifications - incremental build is sufficient
    echo "incremental_test_build - C++ files modified (no additions/deletions)"
    return
}

# Get human-readable explanation of build strategy
explain_build_strategy() {
    local strategy="$1"

    case "$strategy" in
        full_install)
            echo "Package not installed - will install editable package"
            ;;
        native_build)
            echo "Build native extension with cmake reconfigure - ${BUILD_REASON:-changes detected}"
            ;;
        incremental_build)
            echo "Incremental build sufficient - ${BUILD_REASON:-header files modified}"
            ;;
        *)
            echo "Unknown build strategy: $strategy"
            ;;
    esac
}

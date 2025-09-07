#!/bin/bash
set -e
set -o pipefail

# File finding and command execution utilities
# Should be sourced from other scripts, not executed directly

_script_dir="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${_script_dir}/logging.sh"
source "${_script_dir}/build_env.sh"

# Setup directories
setup_dirs() {
    # For scripts that need full build environment, use create_build_dirs
    # For simple scripts, just create basic directories
    if declare -F create_build_dirs >/dev/null 2>&1; then
        create_build_dirs
    else
        mkdir -p test_reports logs
        log_info "Created directories: test_reports/, logs/"
    fi
}

# Find files functions
find_python_files() {
    # Target specific directories like original Makefile
    echo "setu test examples setup.py"
}

find_cpp_files() {
    find csrc -type f \( -name "*.cpp" -o -name "*.h" \) \
        -not -path "*/third_party/*" \
        2>/dev/null
}

find_shell_scripts() {
    find scripts -type f -name "*.sh" 2>/dev/null
}

find_cmake_files() {
    find . -type f \( -name "CMakeLists.txt" -o -name "*.cmake" \) \
        -not -path "./build/*" \
        -not -path "./env/*" \
        2>/dev/null
}

find_yaml_files() {
    find . -type f \( -name "*.yml" -o -name "*.yaml" \) \
        -not -path "./build/*" \
        -not -path "./env/*" \
        2>/dev/null
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Run command with file list
run_with_files() {
    local cmd="$1"
    local files="$2"
    local description="$3"
    local warn_message="${4:-No files found}"

    if [ -n "$files" ]; then
        log_info "$description..."
        eval "$cmd $files"
    else
        log_warning "$warn_message"
    fi
}

# Run command and check success
run_check() {
    local cmd="$1"
    local success_msg="$2"
    local error_msg="$3"

    if eval "$cmd"; then
        [ -n "$success_msg" ] && log_success "$success_msg"
        return 0
    else
        log_error "$error_msg"
        return 1
    fi
}

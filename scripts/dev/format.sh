#!/bin/bash
# Formatting scripts for Setu project

set -e          # Exit on any error
set -o pipefail # Exit on pipe failures

_script_dir="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${_script_dir}/utils.sh"

# Initialize CI environment if needed
init_ci_environment

# Python formatting functions
format_black() {
    log_format "Formatting (Black)..."
    black setu test examples setup.py
}

format_isort() {
    log_format "Formatting (isort)..."
    isort --profile black setu test examples setup.py
}

format_autoflake() {
    log_format "Formatting (autoflake)..."
    autoflake --in-place --recursive --remove-all-unused-imports setu test examples setup.py
}

# C++ formatting functions
format_clang() {
    log_format "Formatting (clang-format)..."
    local cpp_files
    cpp_files=$(find_cpp_files)
    if [ -n "$cpp_files" ]; then
        # shellcheck disable=SC2086
        clang-format -i $cpp_files
    else
        log_warning "No C++ files found"
    fi
}

# Shell script formatting
format_shfmt() {
    log_format "Formatting (shfmt)..."
    local shell_scripts
    shell_scripts=$(find_shell_scripts)
    if [ -n "$shell_scripts" ]; then
        if command_exists shfmt; then
            # shellcheck disable=SC2086
            shfmt -w -i 4 -bn -ci $shell_scripts
        else
            log_error "shfmt not found. Install with: go install mvdan.cc/sh/v3/cmd/shfmt@latest"
            return 1
        fi
    else
        log_warning "No shell scripts found"
    fi
}

# CMake formatting
format_cmake() {
    log_format "Formatting (cmake-format)..."
    local cmake_files
    cmake_files=$(find_cmake_files)
    if [ -n "$cmake_files" ]; then
        if command_exists cmake-format; then
            # shellcheck disable=SC2086
            cmake-format -i $cmake_files
        else
            log_error "cmake-format not found. Install with: pip install cmakelang"
            return 1
        fi
    else
        log_warning "No CMake files found"
    fi
}

# YAML formatting
format_yaml() {
    log_format "Formatting (yamlfmt)..."
    local yaml_files
    yaml_files=$(find_yaml_files)
    if [ -n "$yaml_files" ]; then
        if command_exists yamlfmt; then
            # shellcheck disable=SC2086
            yamlfmt $yaml_files
        else
            log_error "yamlfmt not found. Install with: go install github.com/google/yamlfmt/cmd/yamlfmt@latest"
            return 1
        fi
    else
        log_warning "No YAML files found"
    fi
}

# Main formatting function
format_all() {
    log_info "Running all formatting..."

    # Python formatting (order matters: isort -> autoflake -> black)
    format_isort
    format_autoflake
    format_black

    # Other formatting
    format_clang
    format_shfmt
    format_cmake
    format_yaml

    log_success "Code Formatting Complete ${SUCCESS_ICON}"
}

# Main function
main() {
    case "${1:-all}" in
        black)
            format_black
            ;;
        isort)
            format_isort
            ;;
        autoflake)
            format_autoflake
            ;;
        clang)
            format_clang
            ;;
        shfmt)
            format_shfmt
            ;;
        cmake-format)
            format_cmake
            ;;
        yaml)
            format_yaml
            ;;
        all)
            format_all
            ;;
        *)
            echo "Usage: $0 {black|isort|autoflake|clang|shfmt|cmake-format|yaml|all}"
            exit 1
            ;;
    esac
}

# Run main function if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi

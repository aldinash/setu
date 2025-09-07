#!/bin/bash
# Setup and environment scripts for Setu project

set -e          # Exit on any error
set -o pipefail # Exit on pipe failures

_script_dir="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${_script_dir}/utils.sh"

# Dependencies installation
setup_dependencies() {
    log_info "Installing Python Dependencies..."
    setup_dirs

    local cuda_version torch_version timestamp
    cuda_version=$(get_cuda_version)
    torch_version=$(get_torch_version)
    timestamp=$(get_timestamp)

    log_info "Using CUDA=${YELLOW}${cuda_version}${BLUE}, Torch=${YELLOW}${torch_version}${RESET}"
    log_info "Full log: logs/pip_install_${timestamp}.log"

    # Install requirements with extra index URL
    pip install -r requirements.txt \
        2>&1 | tee "logs/pip_install_${timestamp}.log"

    log_success "Dependencies installed successfully ${PARTY_ICON}"
}

# Environment setup
setup_environment() {
    log_info "Creating conda/mamba environment..."

    local env_file="environment-dev.yml"
    if [ ! -f "$env_file" ]; then
        log_error "Environment file $env_file not found"
        return 1
    fi

    # Try mamba first, fall back to conda
    if command_exists mamba; then
        log_info "Using mamba to create environment..."
        mamba env create -f "$env_file" -p ./env
    elif command_exists conda; then
        log_info "Using conda to create environment..."
        conda env create -f "$env_file" -p ./env
    else
        log_error "Neither mamba nor conda found. Please install conda/mamba first."
        return 1
    fi

    log_success "Environment created successfully"
    log_info "Activate with: conda activate ./env"
}

# Environment update
update_environment() {
    log_info "Updating conda/mamba environment..."

    local env_file="environment-dev.yml"
    if [ ! -f "$env_file" ]; then
        log_error "Environment file $env_file not found"
        return 1
    fi

    # Try mamba first, fall back to conda
    if command_exists mamba; then
        log_info "Using mamba to update environment..."
        mamba env update -f "$env_file" -p ./env --prune
    elif command_exists conda; then
        log_info "Using conda to update environment..."
        conda env update -f "$env_file" -p ./env --prune
    else
        log_error "Neither mamba nor conda found. Please install conda/mamba first."
        return 1
    fi

    log_success "Environment updated successfully"
}

# Check system dependencies
check_system_deps() {
    log_info "Checking system dependencies..."

    local missing_deps=()

    # Check for essential build tools
    if ! command_exists cmake; then
        missing_deps+=("cmake")
    fi

    if ! command_exists ninja; then
        if ! command_exists make; then
            missing_deps+=("make or ninja")
        fi
    fi

    if ! command_exists gcc && ! command_exists clang; then
        missing_deps+=("gcc or clang")
    fi

    if ! command_exists python3; then
        missing_deps+=("python3")
    fi

    if ! command_exists pip; then
        missing_deps+=("pip")
    fi

    # Check for optional but recommended tools
    local optional_deps=()
    if ! command_exists ccache; then
        optional_deps+=("ccache (for faster builds)")
    fi

    if ! command_exists git; then
        optional_deps+=("git")
    fi

    # Report results
    if [ ${#missing_deps[@]} -eq 0 ]; then
        log_success "All essential system dependencies found"
    else
        log_error "Missing essential dependencies: ${missing_deps[*]}"
        log_info "Please install them using your system package manager"
        return 1
    fi

    if [ ${#optional_deps[@]} -gt 0 ]; then
        log_warning "Optional dependencies not found: ${optional_deps[*]}"
    fi

    return 0
}

# Check Python environment
check_python_env() {
    log_info "Checking Python environment..."

    # Check Python version
    local python_version
    python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
    log_info "Python version: $python_version"

    # Check if we're in a virtual environment
    if [ -n "${VIRTUAL_ENV:-}" ] || [ -n "${CONDA_DEFAULT_ENV:-}" ]; then
        log_success "Virtual environment detected"
        if [ -n "${CONDA_DEFAULT_ENV:-}" ]; then
            log_info "Conda environment: $CONDA_DEFAULT_ENV"
        fi
        if [ -n "${VIRTUAL_ENV:-}" ]; then
            log_info "Virtual environment: $VIRTUAL_ENV"
        fi
    else
        log_warning "No virtual environment detected. Consider using conda or venv."
    fi

    # Check for essential Python packages
    local missing_packages=()

    if ! python3 -c "import pip" 2>/dev/null; then
        missing_packages+=("pip")
    fi

    if ! python3 -c "import setuptools" 2>/dev/null; then
        missing_packages+=("setuptools")
    fi

    if ! python3 -c "import wheel" 2>/dev/null; then
        missing_packages+=("wheel")
    fi

    if [ ${#missing_packages[@]} -eq 0 ]; then
        log_success "Essential Python packages found"
    else
        log_warning "Missing Python packages: ${missing_packages[*]}"
        log_info "Install with: pip install ${missing_packages[*]}"
    fi
}

# Full system check
check_system() {
    log_info "Running full system check..."

    local failed=0
    check_system_deps || failed=$((failed + 1))
    check_python_env || failed=$((failed + 1))

    if [ $failed -eq 0 ]; then
        log_success "System check passed ${SUCCESS_ICON}"
    else
        log_error "System check failed with $failed issue(s)"
        return 1
    fi
}

# Main function
main() {
    case "${1:-check}" in
        dependencies)
            setup_dependencies
            ;;
        environment)
            setup_environment
            ;;
        update-environment)
            update_environment
            ;;
        check-system)
            check_system_deps
            ;;
        check-python)
            check_python_env
            ;;
        check)
            check_system
            ;;
        *)
            echo "Usage: $0 {dependencies|environment|update-environment|check-system|check-python|check}"
            exit 1
            ;;
    esac
}

# Run main function if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi

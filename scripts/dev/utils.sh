#!/bin/bash
set -e
set -o pipefail

# Main utility loader for build scripts
# Should be sourced from other scripts, not executed directly

_dev_script_dir="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# Load all utility modules
source "${_dev_script_dir}/logging.sh"
source "${_dev_script_dir}/build_env.sh"
source "${_dev_script_dir}/file_utils.sh"

# Global variable to track CI environment initialization
_ci_env_initialized=false

# Initialize CI environment with execute-once guard
init_ci_environment() {
    # Only initialize once
    if [[ "$_ci_env_initialized" == "true" ]]; then
        return 0
    fi

    # Only initialize in CI context
    if [[ "${SETU_IS_CI_CONTEXT:-0}" != "true" ]]; then
        return 0
    fi

    log_info "Initializing CI environment..."

    # Source container utilities and activate conda environment
    source "${_dev_script_dir}/../containers/utils.sh"
    init_conda
    activate_setu_conda_env
    login_huggingface

    # Mark as initialized
    _ci_env_initialized=true

    log_info "CI environment initialized successfully"
}

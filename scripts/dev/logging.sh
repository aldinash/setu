#!/bin/bash
set -e
set -o pipefail

# Logging utilities for build scripts
# Should be sourced from other scripts, not executed directly

# Prevent multiple sourcing
if [[ "${SETU_LOGGING_LOADED:-}" == "true" ]]; then
    return 0
fi
readonly SETU_LOGGING_LOADED="true"

# TTY and color detection
if [[ -t 1 ]] && [[ -z "${NO_COLOR:-}" ]]; then
    readonly RED='\033[0;31m'
    readonly GREEN='\033[0;32m'
    readonly YELLOW='\033[1;33m'
    readonly BLUE='\033[0;34m'
    # shellcheck disable=SC2034  # CYAN may be used in future log messages
    readonly CYAN='\033[0;36m'
    readonly BOLD='\033[1m'
    readonly RESET='\033[0m'
else
    readonly RED=''
    readonly GREEN=''
    readonly YELLOW=''
    readonly BLUE=''
    # shellcheck disable=SC2034  # CYAN may be used in future log messages
    readonly CYAN=''
    readonly BOLD=''
    readonly RESET=''
fi

readonly SUCCESS_ICON="✅"
readonly ERROR_ICON="❌"
readonly WARN_ICON="⚠️"
readonly INFO_ICON="ℹ️"
readonly LINT_ICON="✨"
readonly FORMAT_ICON="💅"
readonly BUILD_ICON="🔨"
readonly TEST_ICON="🧪"
readonly CLEAN_ICON="🧹"
# shellcheck disable=SC2034
readonly PARTY_ICON="🎉"

# Echo with color support
echo_color() {
    echo -e "$@"
}

# Log functions
log_info() {
    echo_color "${BLUE}${INFO_ICON} $*${RESET}"
}

log_success() {
    echo_color "${GREEN}${SUCCESS_ICON} $*${RESET}"
}

log_warning() {
    echo_color "${YELLOW}${WARN_ICON} $*${RESET}"
}

log_error() {
    echo_color "${RED}${ERROR_ICON} $*${RESET}"
}

log_lint() {
    echo_color "${BLUE}${LINT_ICON} $*${RESET}"
}

log_format() {
    echo_color "${BLUE}${FORMAT_ICON} $*${RESET}"
}

log_build() {
    echo_color "${BLUE}${BUILD_ICON} $*${RESET}"
}

log_test() {
    echo_color "${BLUE}${TEST_ICON} $*${RESET}"
}

log_clean() {
    echo_color "${BOLD}${CLEAN_ICON} $*${RESET}"
}

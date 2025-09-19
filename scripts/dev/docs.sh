#!/bin/bash
# Documentation scripts for Setu project

set -e          # Exit on any error
set -o pipefail # Exit on pipe failures

_script_dir="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${_script_dir}/utils.sh"

# Initialize CI environment if needed
init_ci_environment

# Documentation build
docs_build() {
    log_info "Building documentation using sphinx..."

    if [ ! -d "docs" ]; then
        log_error "docs/ directory not found"
        return 1
    fi

    cd docs || exit 1

    # Check for sphinx
    if ! command_exists sphinx-build; then
        log_error "sphinx-build not found. Install with: pip install sphinx"
        cd - >/dev/null || exit 1
        return 1
    fi

    # Build documentation
    make html

    cd - >/dev/null || exit 1

    log_success "Documentation built successfully"
    log_info "Open docs/_build/html/index.html to view"
}

# Documentation clean
docs_clean() {
    log_info "Cleaning documentation build artifacts..."

    if [ ! -d "docs" ]; then
        log_warning "docs/ directory not found"
        return 0
    fi

    cd docs || exit 1

    # Clean build artifacts
    if [ -f "Makefile" ]; then
        make clean
    else
        # Fallback manual cleanup
        rm -rf _build/ _static/ _templates/
    fi

    cd - >/dev/null || exit 1

    log_success "Documentation cleaned"
}

# Documentation serve
docs_serve() {
    log_info "Serving documentation locally..."

    if [ ! -d "docs/_build/html" ]; then
        log_warning "Documentation not built. Building first..."
        docs_build
    fi

    local port="${1:-8000}"
    local docs_dir="docs/_build/html"

    if [ ! -d "$docs_dir" ]; then
        log_error "Built documentation not found at $docs_dir"
        return 1
    fi

    log_info "Starting HTTP server on port $port..."
    log_info "Open http://localhost:$port in your browser"
    log_info "Press Ctrl+C to stop the server"

    # Try python3 -m http.server first, fallback to python2
    cd "$docs_dir" || exit 1
    if command_exists python3; then
        python3 -m http.server "$port"
    elif command_exists python; then
        python -m SimpleHTTPServer "$port"
    else
        log_error "Python not found. Cannot start HTTP server."
        cd - >/dev/null || exit 1
        return 1
    fi
    cd - >/dev/null || exit 1
}

# Initialize documentation
docs_init() {
    log_info "Initializing documentation structure..."

    if [ -d "docs" ]; then
        log_warning "docs/ directory already exists"
        read -p "Overwrite existing documentation? [y/N] " -r confirm
        if [[ ! $confirm =~ ^[Yy]([Ee][Ss])?$ ]]; then
            log_info "Aborted."
            return 1
        fi
        rm -rf docs/
    fi

    # Check for sphinx
    if ! command_exists sphinx-quickstart; then
        log_error "sphinx-quickstart not found. Install with: pip install sphinx"
        return 1
    fi

    # Create docs directory and initialize
    mkdir -p docs
    cd docs || exit 1

    # Run sphinx-quickstart with defaults
    sphinx-quickstart --quiet --project="Setu" --author="Vajra Team" \
        --release="1.0" --language="en" --ext-autodoc \
        --ext-viewcode --makefile --batchfile

    cd - >/dev/null || exit 1

    log_success "Documentation initialized"
    log_info "Edit docs/source/conf.py and docs/source/index.rst to customize"
}

# All docs operations
docs_all() {
    log_info "Running all documentation operations..."

    docs_clean
    docs_build

    log_success "All documentation operations completed"
}

# Main function
main() {
    case "${1:-build}" in
        build)
            docs_build
            ;;
        clean)
            docs_clean
            ;;
        serve)
            docs_serve "${2:-8000}"
            ;;
        init)
            docs_init
            ;;
        all)
            docs_all
            ;;
        *)
            echo "Usage: $0 {build|clean|serve [port]|init|all}"
            exit 1
            ;;
    esac
}

# Run main function if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi

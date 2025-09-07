# Setu Build System

Use the Makefile targets for all build operations:

```bash
# Linting
make lint                    # Run all linting checks
make lint/black             # Run specific linting check

# Formatting
make format                  # Format all code
make format/python          # Format specific language

# Building
make build                   # Build project (editable install)
make build/native           # Build native extension
make build/wheel            # Build Python wheel

# Testing
make test                    # Run all tests
make test/unit              # Run unit tests
make test/integration       # Run integration tests

# Setup
make setup/dependencies     # Install dependencies
make setup/environment      # Create conda environment
make setup/activate         # Show activation command
make setup/check           # Check system dependencies

# Documentation
make docs                   # Build documentation
make docs/serve            # Serve documentation locally

# Utilities
make clean                  # Clean build artifacts
make ccache/stats          # Show ccache statistics

# Debugging
make debug/crash           # Analyze crashes using gdb
make debug/deadlock        # Analyze deadlocks using py-spy and gdb

# Convenient aliases
make check                  # Alias for 'make lint'
```

## Configuration

### Environment Variables

- **`USE_RAMDISK`**: Set to `1` to use RAM disk for builds (faster I/O)
- **`SETU_IS_CI_CONTEXT`**: Set to `true` in CI environments
- **`BUILD_TYPE`**: Set build type (Debug, Release, etc.)

### Help

Run `make help` to see all available targets with descriptions.
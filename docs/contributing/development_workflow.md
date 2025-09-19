# Development Workflow Guide

This guide covers the day-to-day development workflow for contributing to Setu, from making changes to submitting pull requests.

## Development Environment Setup

### Initial Setup

Refer to [quick start](../quickstart.md) for project setup process.

## Making Changes

### Python Development

For Python-only changes, no build step is required:

```bash
# Edit Python files
vim setu/config/model_config.py

# Test immediately
python -m pytest test/unit/test_config.py -v

# Format code
make format/black format/isort
```

### C++ Development

C++ changes require compilation:

```bash
# Edit C++ files
vim csrc/setu/native/controller/replica/request_batcher/base_request_batcher.cpp

# Intelligent build (automatically detects what's needed)
make build

# Test your changes
# TODO: Add example

# Build and run C++ tests
make build/test  # Intelligent test build
make test/ctest  # Run C++ tests
make test/unit   # Full unit test suite
```

### Mixed Python/C++ Development

When working on features that span both languages:

```bash
# Make your changes
vim csrc/setu/native/inference_engine.cpp
vim setu/engine/inference_engine.py

# Intelligent build (detects changes and builds optimally)
make build
python test_your_feature.py

# Build tests if needed and run comprehensive tests
make build/test      # Build tests intelligently
make test/unit test/integration
```

## Code Quality Workflow

### Formatting

Setu uses automated formatting. **Always format before committing:**

```bash
# Format all code (Python + C++)
make format

# Format specific languages
make format/black    # Python formatting
make format/clang    # C++ formatting
make format/isort    # Python import sorting
```

### Linting

Run linters to catch issues early:

```bash
# Check all code quality
make lint

# Specific linters
make lint/pyright    # Python type checking
make lint/cpplint    # C++ style checking
make lint/black      # Python format checking
make lint/codespell  # Spelling checks
```

## Testing Strategy

### Test Categories

| Type | Command | Purpose | Speed |
|------|---------|---------|-------|
| Unit | `make test/unit` | Fast, isolated tests | ~30s |
| Integration | `make test/integration` | Feature-level tests | ~2-5min |
| Performance | `make test/performance` | Benchmark tests | ~5-15min |
| Functional | `make test/functional` | End-to-end tests | ~10-30min |

### Failed-Only Testing

For faster iteration during development, Setu provides failed-only test targets that rerun only the tests that failed in the previous run:

| Target | Scope | Fallback Behavior |
|--------|-------|-------------------|
| `make test/unit-failed-only` | Both Python + C++ failed tests | Runs all unit tests if no failures cached |
| `make test/pyunit-failed-only` | Python unit tests only | Runs all Python unit tests if no pytest cache |
| `make test/ctest-failed-only` | C++ tests only | Runs all C++ tests if no CTest failures |

**How it works:**
- **Python Tests**: Uses pytest's `--lf` (last-failed) flag with `.pytest_cache` 
- **C++ Tests**: Uses CTest's `--rerun-failed` flag with `LastTestsFailed.log`
- **Smart Fallback**: Automatically runs all tests if no previous failures are found
- **Separate Reports**: Generates dedicated test reports for failed test reruns

### Testing Workflow

#### During Development
```bash
# Quick feedback loop
make test/unit

# Fix failing tests, then rerun only the failures
make test/unit-failed-only

# Test specific components
make test/pyunit-failed-only  # Python failures only
make test/ctest-failed-only   # C++ failures only

# Test specific modules
python -m pytest test/unit/test_batcher.py -v

# Test with coverage
python -m pytest --cov=setu --cov-report=html test/unit/
```

#### Before Committing
```bash
# Build tests and run comprehensive testing
make build/test      # Build tests intelligently first
make test/unit test/integration

# Full test suite (takes longer)
make test
```

#### Performance Testing
```bash
# Run performance benchmarks
make test/performance

# Custom performance tests
python -m setu.benchmark.main
```

### Test Reports

All tests generate detailed reports:

```bash
# Run tests with reports
make test/unit  # Creates test_reports/pytest-unit-results.xml

# View coverage reports
open test_reports/python_coverage_html/index.html
```

## Build System Deep Dive

### Intelligent Build System

Setu features an intelligent build system that automatically detects what needs to be built based on your changes:

```bash
# Simply run this for all scenarios
make build      # Intelligent main build
make build/test # Intelligent test build

# The system will automatically:
# - Install package if not installed
# - Run full rebuild if CMake files changed  
# - Run incremental build if only source files changed
# - Skip build if nothing changed
```

#### Build Strategy Detection

The intelligent build analyzes your project state:

**Main Build (`make build`):**
| Scenario | Strategy | What it runs |
|----------|----------|--------------|
| Package not installed | `full_install` | `make build/editable` |
| Package config changed | `full_install` | `make build/editable` |  
| CMake files changed | `native_build` | `make build/native` |
| New C++ files added | `native_build` | `make build/native` |
| C++ sources modified | `incremental_build` | `make build/native_incremental` |

**Test Build (`make build/test`):**
| Scenario | Strategy | What it runs |
|----------|----------|--------------|
| No build exists | `full_test_build` | `make build/native_test_full` |
| Test binaries missing | `full_test_build` | `make build/native_test_full` |
| CMake files changed | `full_test_build` | `make build/native_test_full` |
| Source files modified | `incremental_test_build` | `make build/native_test_incremental` |

#### Advanced Build Options

For specific needs, you can still use individual commands:

```bash
# Main build commands
make build/smart               # Intelligent build (same as make build)
make build/native              # Full native build (always rebuilds)
make build/native_incremental  # Incremental build (fastest, but may fail if new files)
make build/editable            # Install editable package
make build/wheel               # Build Python wheel

# Test build commands  
make build/test                # Intelligent test build (same as make build/native_test)
make build/native_test         # Intelligent test build (alias)
make build/native_test_full    # Full test build (always rebuilds)
make build/native_test_incremental # Incremental test build
```

### Build Customization

Control builds with environment variables:

```bash
# Debug vs Release builds
BUILD_TYPE=Debug make build/native      # Default, includes debug symbols
BUILD_TYPE=Release make build/native    # Optimized for performance

# Parallel build jobs
CMAKE_BUILD_PARALLEL_LEVEL=8 make build/native
```

### Incremental Development

For fast development cycles:

```bash
# Simply use intelligent build for all scenarios
make build      # For main package
make build/test # For test binaries

# The system automatically chooses the fastest appropriate build
# - Incremental when possible
# - Full rebuild when necessary
# - Test builds are tracked separately for efficiency
```

### Build Logs

All builds generate detailed logs:

```bash
# Build logs are saved with timestamps
ls logs/
# cmake_configure_20241201_143022.log
# cmake_build_native_20241201_143055.log

# View the latest build log
ls -t logs/cmake_build_* | head -1 | xargs cat
```

## Docker Development

### Using the Development Container

Build and run the development container:

```bash
cd docker/containers/dev

# Build container
make build USERNAME=$USER

# Run interactive session
make run USERNAME=$USER

# Or start persistent container
make start USERNAME=$USER
make attach USERNAME=$USER
```

### Container Development Workflow

Inside the container:

```bash
# Repository is mounted at /repo
cd /repo

# All development commands work the same
make build
make test/unit
make format
```

### Container Benefits

- **Consistent Environment**: Same CUDA/PyTorch versions as CI
- **Pre-installed Tools**: All development tools ready
- **Isolation**: Doesn't affect host system
- **GPU Access**: Full CUDA support
- **Productivity Tools**: ZSH, Oh My Zsh, FZF for better CLI experience

## Git Workflow

### Branch Management

```bash
# Create feature branch
git checkout -b users/your-name/your-feature-name

# Make changes and commit
git add .
git commit -m "feat: add your feature description"

# Keep up to date with main
git fetch origin
git rebase origin/main

# Push your branch
git push origin users/your-name/your-feature-name
```

### Pull Request Naming

Follow conventional commit/pull request naming format:

```bash
# Format: type(scope): description
git commit -m "feat(batcher): add priority-based batching"
git commit -m "fix(memory): resolve memory leak in cache manager"
git commit -m "docs(setup): update installation instructions"
git commit -m "test(integration): add end-to-end pipeline test"
```

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `style`, `chore`

## Best Practices Summary

### Daily Development

1. **Start Clean**: `make format && make lint`
2. **Simply Build**: Use `make build` for main changes, `make build/test` for test changes - both are intelligent and fast
3. **Test Early**: Run relevant tests after each change, use failed-only targets for faster iteration
4. **Format Before Commit**: Always run `make format`

### Code Quality

1. **Follow Style Guides**: Use the provided style guides
2. **Write Tests**: Add tests for new functionality
3. **Document Changes**: Update docs for user-facing changes
4. **Use Type Hints**: Comprehensive type annotations

### Performance

1. **Profile First**: Measure before optimizing
2. **Test Performance**: Run benchmarks for performance changes
3. **Consider Memory**: GPU memory is often the bottleneck
4. **Batch Operations**: Process multiple items together

### Collaboration

1. **Small PRs**: Keep changes focused and reviewable
2. **Clear Commits**: Use conventional commit messages
3. **Document Decisions**: Explain why, not just what
4. **Ask Questions**: Use GitHub Discussions for help

## Efficient Debugging and Iteration

### Failed Test Debugging Workflow

When tests fail, use the failed-only targets for efficient debugging:

```bash
# Initial test run reveals failures
make test/unit
# Output: 5 failed, 708 passed

# Focus only on the failures
make test/unit-failed-only
# Reruns only the 5 failed tests

# Fix issues in specific language
make test/pyunit-failed-only  # If Python tests failed
make test/ctest-failed-only   # If C++ tests failed

# Verify fixes work
make test/unit-failed-only
# Should show fewer or no failures

# Final verification - run all tests
make test/unit
```

### Language-Specific Debugging

**Python Test Failures:**
```bash
# Run failed Python tests with verbose output
make test/pyunit-failed-only

# Debug specific test with pdb
python -m pytest test/unit/test_batcher.py::test_function -v -s --pdb

# Check test coverage for your changes
python -m pytest --cov=setu --cov-report=html test/unit/ --lf
```

**C++ Test Failures:**
```bash
# Run failed C++ tests with detailed output
make test/ctest-failed-only

# Debug specific C++ test manually
cd build/debug
./native_tests --gtest_filter="*BatcherTest*" --gtest_output=xml

# Use GDB for debugging segfaults
gdb --args ./native_tests --gtest_filter="*FailingTest*"
```

### Performance-Focused Iteration

**Fast Development Cycle:**
```bash
# Make changes
vim your_file.py

# Quick test (Python only, no build needed)
make test/pyunit-failed-only

# Make C++ changes  
vim csrc/your_file.cpp

# Build and test (C++ requires build)
make build && make test/ctest-failed-only

# Combined changes - test everything that failed
make build && make test/unit-failed-only
```

**Smart Test Selection:**
- Use failed-only targets during active development
- Run full test suite before commits
- Use specific test filters for focused debugging
- Leverage separate Python/C++ targets when working on specific components

This development workflow ensures high code quality, fast iteration, and smooth collaboration across the Setu project.
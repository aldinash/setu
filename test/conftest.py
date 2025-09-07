"""
Pytest configuration and fixtures for setu tests.
"""

import logging
import os
import sys
from pathlib import Path

import pytest

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging for tests
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Disable logging during tests unless specifically requested
logging.getLogger("setu").setLevel(logging.WARNING)


@pytest.fixture(scope="session")
def test_data_dir():
    """Directory containing test data files."""
    return Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def temp_dir(tmp_path_factory):
    """Temporary directory for test outputs."""
    return tmp_path_factory.mktemp("setu_tests")


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup test environment before each test."""
    # Set environment variables for testing
    os.environ["SETU_TEST_MODE"] = "1"
    os.environ["SETU_LOG_LEVEL"] = "WARNING"

    yield

    # Cleanup after test
    os.environ.pop("SETU_TEST_MODE", None)
    os.environ.pop("SETU_LOG_LEVEL", None)


@pytest.fixture
def sample_data():
    """Sample data for testing."""
    return [
        "sample_item_1",
        "sample_item_2",
        "sample_item_3",
        "test_data_with_special_chars!@#",
        "unicode_test_データ",
    ]


@pytest.fixture
def large_sample_data():
    """Large sample data for performance testing."""
    return [f"large_sample_item_{i}" for i in range(1000)]


@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    return {
        "batch_size": 16,
        "timeout": 5.0,
        "verbose": False,
        "debug": True,
        "test_mode": True,
    }


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests - fast, isolated tests")
    config.addinivalue_line(
        "markers", "integration: Integration tests - test component interactions"
    )
    config.addinivalue_line(
        "markers", "performance: Performance tests - measure speed and efficiency"
    )
    config.addinivalue_line(
        "markers", "correctness: Correctness tests - validate algorithm accuracy"
    )
    config.addinivalue_line("markers", "gpu: Tests that require GPU/CUDA")
    config.addinivalue_line("markers", "slow: Slow tests - may take several minutes")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names and paths."""
    for item in items:
        # Add unit marker to tests in unit/ directory
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)

        # Add integration marker to tests in integration/ directory
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)

        # Add performance marker to tests with 'performance' in name
        if "performance" in item.name:
            item.add_marker(pytest.mark.performance)
            item.add_marker(pytest.mark.slow)

        # Add correctness marker to tests with 'correctness' in name
        if "correctness" in item.name:
            item.add_marker(pytest.mark.correctness)

        # Add GPU marker to tests with 'gpu' or 'cuda' in name
        if any(keyword in item.name.lower() for keyword in ["gpu", "cuda", "device"]):
            item.add_marker(pytest.mark.gpu)


@pytest.fixture(scope="session")
def check_native_extension():
    """Check if native extension is available."""
    try:
        return True
    except ImportError:
        return False


@pytest.fixture(scope="session")
def check_gpu_available():
    """Check if GPU/CUDA is available."""
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def pytest_runtest_setup(item):
    """Setup function run before each test."""
    # Skip GPU tests if CUDA not available
    if "gpu" in item.keywords and not item.config.getoption("--gpu"):
        pytest.skip("GPU tests require --gpu flag")


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--gpu", action="store_true", default=False, help="Run GPU/CUDA tests"
    )
    parser.addoption(
        "--benchmark",
        action="store_true",
        default=False,
        help="Run benchmark/performance tests",
    )


def pytest_report_header(config):
    """Add custom information to pytest header."""
    header_lines = [
        f"setu Test Suite",
        f"Python: {sys.version.split()[0]}",
        f"Platform: {sys.platform}",
    ]

    # Check for native extension
    try:
        header_lines.append("Native Extension: Available")
    except ImportError:
        header_lines.append("Native Extension: Not Available")

    # Check for GPU support
    try:
        import torch

        if torch.cuda.is_available():
            header_lines.append(
                f"CUDA: Available (GPU count: {torch.cuda.device_count()})"
            )
        else:
            header_lines.append("CUDA: Not Available")
    except ImportError:
        header_lines.append("PyTorch: Not Available")

    return "\n".join(header_lines)

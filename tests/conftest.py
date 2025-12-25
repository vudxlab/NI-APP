from unittest.mock import Mock, MagicMock
from pathlib import Path
"""
Pytest configuration and shared fixtures for all tests.

This module provides common fixtures and configuration for the test suite.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock

# Add src to path for imports

# Configure Qt backend for tests
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'  # Run headless


def pytest_configure(config):
    """Configure pytest with custom markers."""
    # Add src to path BEFORE any imports happen
    import sys
    from pathlib import Path
    src_path = str(Path(__file__).parent.parent / 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    config.addinivalue_line("markers", "unit: Unit tests (fast, isolated)")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "gui: GUI tests requiring Qt")
    config.addinivalue_line("markers", "slow: Slow tests (> 1 second)")
    config.addinivalue_line("markers", "hardware: Tests requiring NI DAQ hardware")
    config.addinivalue_line("markers", "simulation: Tests using simulation mode")


@pytest.fixture(scope="session")
def sample_rate():
    """Standard sample rate for tests."""
    return 51200.0


@pytest.fixture(scope="session")
def n_channels():
    """Standard number of channels for tests."""
    return 4


@pytest.fixture(scope="session")
def buffer_size():
    """Standard buffer size for tests."""
    return 10000


@pytest.fixture
def random_data(n_channels, buffer_size):
    """Generate random test data."""
    return np.random.randn(n_channels, buffer_size)


@pytest.fixture
def sine_data(sample_rate):
    """Generate sine wave test data."""
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    return np.sin(2 * np.pi * 1000 * t)  # 1000 Hz sine wave


@pytest.fixture
def multi_tone_data(sample_rate):
    """Generate multi-tone test data."""
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))

    # Multiple frequencies: 100 Hz, 1000 Hz, 5000 Hz
    signal = (
        np.sin(2 * np.pi * 100 * t) +
        0.5 * np.sin(2 * np.pi * 1000 * t) +
        0.3 * np.sin(2 * np.pi * 5000 * t)
    )

    return signal


@pytest.fixture
def mock_daq_config():
    """Create a mock DAQ configuration."""
    from src.daq.daq_config import ChannelConfig, DAQConfig

    channels = [
        ChannelConfig(
            physical_channel=f"cDAQ1Mod0/ai{i}",
            name=f"Channel {i}",
            sensitivity=100.0,
            units="g"
        )
        for i in range(4)
    ]

    return DAQConfig(
        device_name="cDAQ1",
        sample_rate=51200.0,
        channels=channels,
        samples_per_read=1000
    )


@pytest.fixture
def mock_filter_config():
    """Create a mock filter configuration."""
    from src.processing.filters import FilterConfig, FilterType, FilterMode

    return FilterConfig(
        filter_type=FilterType.BUTTERWORTH,
        filter_mode=FilterMode.LOWPASS,
        cutoff_freq=1000.0,
        sample_rate=51200.0,
        order=4
    )


@pytest.fixture
def temp_directory(tmp_path):
    """Provide a temporary directory for file operations."""
    return tmp_path


@pytest.fixture
def temp_csv_file(temp_directory):
    """Provide a temporary CSV file path."""
    return temp_directory / "test_data.csv"


@pytest.fixture
def temp_hdf5_file(temp_directory):
    """Provide a temporary HDF5 file path."""
    return temp_directory / "test_data.h5"


@pytest.fixture
def temp_tdms_file(temp_directory):
    """Provide a temporary TDMS file path."""
    return temp_directory / "test_data.tdms"


# Skip tests that require hardware
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add skips."""
    for item in items:
        # Skip hardware tests by default
        if "hardware" in item.keywords:
            item.add_marker(pytest.mark.skip(
                reason="Hardware tests require NI DAQ device"
            ))

        # Skip GUI tests if Qt is not available
        if "gui" in item.keywords:
            try:
                import PyQt5
            except ImportError:
                item.add_marker(pytest.mark.skip(
                    reason="GUI tests require PyQt5"
                ))


# Shared test utilities
class TestDataGenerator:
    """Utility class for generating test data."""

    @staticmethod
    def generate_sine_wave(frequency, duration, sample_rate, amplitude=1.0):
        """Generate a sine wave."""
        t = np.linspace(0, duration, int(sample_rate * duration))
        return amplitude * np.sin(2 * np.pi * frequency * t)

    @staticmethod
    def generate_noise(n_samples, mean=0.0, std=1.0):
        """Generate Gaussian noise."""
        return np.random.normal(mean, std, n_samples)

    @staticmethod
    def generate_chirp(start_freq, end_freq, duration, sample_rate):
        """Generate a chirp signal."""
        t = np.linspace(0, duration, int(sample_rate * duration))
        return np.sin(2 * np.pi * (start_freq + (end_freq - start_freq) * t / duration) * t)

    @staticmethod
    def generate_impulse_response(duration, sample_rate, decay=0.9):
        """Generate an impulse response."""
        n_samples = int(sample_rate * duration)
        return decay ** np.arange(n_samples)


@pytest.fixture
def data_generator():
    """Provide a data generator utility."""
    return TestDataGenerator()


# Assert utilities
def assert_array_shape(arr, expected_shape):
    """Assert array has expected shape."""
    assert arr.shape == expected_shape, f"Expected shape {expected_shape}, got {arr.shape}"


def assert_array_range(arr, min_val, max_val):
    """Assert all array values are within range."""
    assert np.all(arr >= min_val), f"Array contains values < {min_val}"
    assert np.all(arr <= max_val), f"Array contains values > {max_val}"


def assert_signal_power(signal, expected_power, tolerance=0.1):
    """Assert signal has approximately expected power."""
    actual_power = np.mean(signal ** 2)
    relative_error = abs(actual_power - expected_power) / expected_power
    assert relative_error < tolerance, \
        f"Signal power {actual_power} differs from expected {expected_power}"


# Make utilities available globally
pytest.assert_array_shape = assert_array_shape
pytest.assert_array_range = assert_array_range
pytest.assert_signal_power = assert_signal_power

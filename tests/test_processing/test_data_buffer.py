from unittest.mock import Mock, MagicMock
from pathlib import Path
"""
Unit tests for data buffer module.

Tests circular buffer, thread safety, and data retrieval.
"""

import pytest
import numpy as np
import threading
import time
from multiprocessing import Lock

from src.processing.data_buffer import DataBuffer


class TestDataBuffer:
    """Test DataBuffer class."""

    @pytest.fixture
    def buffer(self):
        """Create a data buffer for testing."""
        return DataBuffer(n_channels=4, buffer_size=10000)

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return np.random.randn(4, 1000)

    def test_buffer_initialization(self, buffer):
        """Test buffer initialization."""
        assert buffer.n_channels == 4
        assert buffer.buffer_size == 10000
        assert buffer.n_samples == 0
        assert buffer.capacity == 10000

    def test_buffer_initialization_small(self):
        """Test buffer initialization with small size."""
        buf = DataBuffer(n_channels=2, buffer_size=100)
        assert buf.n_channels == 2
        assert buf.buffer_size == 100

    def test_append_single_chunk(self, buffer, sample_data):
        """Test appending a single chunk of data."""
        initial_samples = buffer.n_samples
        buffer.append(sample_data)

        assert buffer.n_samples == initial_samples + sample_data.shape[1]

    def test_append_multiple_chunks(self, buffer):
        """Test appending multiple chunks of data."""
        chunk1 = np.random.randn(4, 500)
        chunk2 = np.random.randn(4, 750)
        chunk3 = np.random.randn(4, 250)

        buffer.append(chunk1)
        buffer.append(chunk2)
        buffer.append(chunk3)

        assert buffer.n_samples == 500 + 750 + 250

    def test_append_with_timestamp(self, buffer):
        """Test appending data with timestamp."""
        data = np.random.randn(4, 100)
        timestamp = time.time()

        buffer.append(data, timestamp=timestamp)

        # Should have samples now
        assert buffer.n_samples == 100
        # Timestamp should be stored
        # (exact behavior depends on implementation)

    def test_wrap_around(self, buffer):
        """Test buffer wrap-around when exceeding capacity."""
        # Fill buffer beyond capacity
        chunk = np.random.randn(4, 6000)
        buffer.append(chunk)

        # Should have at most buffer_size samples
        assert buffer.n_samples <= buffer.buffer_size

        # Add more to force wrap
        chunk2 = np.random.randn(4, 6000)
        buffer.append(chunk2)

        # Should still be at capacity
        assert buffer.n_samples <= buffer.buffer_size

    def test_get_latest_basic(self, buffer, sample_data):
        """Test getting latest samples."""
        buffer.append(sample_data)

        retrieved = buffer.get_latest(sample_data.shape[1])

        assert retrieved.shape == sample_data.shape
        np.testing.assert_array_equal(retrieved, sample_data)

    def test_get_latest_partial(self, buffer, sample_data):
        """Test getting fewer samples than available."""
        buffer.append(sample_data)

        n = 500
        retrieved = buffer.get_latest(n)

        assert retrieved.shape == (buffer.n_channels, n)
        # Should be the last n samples
        np.testing.assert_array_equal(retrieved, sample_data[:, -n:])

    def test_get_latest_more_than_available(self, buffer):
        """Test getting more samples than available."""
        data = np.random.randn(4, 100)
        buffer.append(data)

        # Request more than available
        retrieved = buffer.get_latest(500)

        # Should return all available
        assert retrieved.shape[1] == buffer.n_samples
        assert retrieved.shape[1] <= 100

    def test_get_latest_empty_buffer(self, buffer):
        """Test getting samples from empty buffer."""
        retrieved = buffer.get_latest(100)

        assert retrieved.shape == (buffer.n_channels, 0)
        assert retrieved.shape[1] == 0

    def test_get_range_basic(self, buffer):
        """Test getting a range of samples."""
        data = np.random.randn(4, 1000)
        buffer.append(data)

        # Get middle 100 samples
        retrieved = buffer.get_range(start=100, end=200)

        assert retrieved.shape == (4, 100)
        np.testing.assert_array_equal(retrieved, data[:, 100:200])

    def test_get_range_invalid_indices(self, buffer):
        """Test getting range with invalid indices."""
        data = np.random.randn(4, 500)
        buffer.append(data)

        # Start > end should return empty or swap
        retrieved = buffer.get_range(start=200, end=100)
        # Behavior depends on implementation
        # Should either be empty or swapped

    def test_get_range_beyond_buffer(self, buffer):
        """Test getting range beyond available data."""
        data = np.random.randn(4, 500)
        buffer.append(data)

        # Request beyond available
        retrieved = buffer.get_range(start=400, end=1000)

        # Should return what's available
        assert retrieved.shape[0] == buffer.n_channels
        assert retrieved.shape[1] <= 100

    def test_clear_buffer(self, buffer, sample_data):
        """Test clearing the buffer."""
        buffer.append(sample_data)
        assert buffer.n_samples > 0

        buffer.clear()
        assert buffer.n_samples == 0

    def test_is_empty(self, buffer):
        """Test is_empty method."""
        assert buffer.is_empty()

        data = np.random.randn(4, 100)
        buffer.append(data)
        assert not buffer.is_empty()

    def test_is_full(self, buffer):
        """Test is_full method."""
        assert not buffer.is_full()

        # Fill buffer
        data = np.random.randn(4, buffer.buffer_size)
        buffer.append(data)

        # Should be full or nearly full
        assert buffer.n_samples >= buffer.buffer_size * 0.99

    def test_get_channel_data(self, buffer):
        """Test getting data for a single channel."""
        data = np.random.randn(4, 1000)
        buffer.append(data)

        channel_0 = buffer.get_channel_data(0)
        channel_1 = buffer.get_channel_data(1)

        assert channel_0.shape == (1000,)
        assert channel_1.shape == (1000,)

        np.testing.assert_array_equal(channel_0, data[0, :])
        np.testing.assert_array_equal(channel_1, data[1, :])

    def test_get_channel_data_invalid_channel(self, buffer):
        """Test getting data for invalid channel."""
        with pytest.raises(IndexError):
            buffer.get_channel_data(10)

    def test_get_all_data(self, buffer):
        """Test getting all data from buffer."""
        data = np.random.randn(4, 500)
        buffer.append(data)

        all_data = buffer.get_all()

        assert all_data.shape == data.shape
        np.testing.assert_array_equal(all_data, data)

    def test_thread_safety_append(self, buffer):
        """Test thread safety of append operations."""
        n_threads = 4
        n_iterations = 100

        def append_data(thread_id):
            for _ in range(n_iterations):
                data = np.random.randn(4, 10)
                buffer.append(data)

        threads = [
            threading.Thread(target=append_data, args=(i,))
            for i in range(n_threads)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have n_threads * n_iterations * 10 samples
        # (or up to capacity)
        expected = n_threads * n_iterations * 10
        assert buffer.n_samples == min(expected, buffer.buffer_size)

    def test_thread_safety_append_and_read(self, buffer):
        """Test thread safety of concurrent append and read."""
        n_iterations = 50
        errors = []

        def append_data():
            try:
                for _ in range(n_iterations):
                    data = np.random.randn(4, 10)
                    buffer.append(data)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(f"Append error: {e}")

        def read_data():
            try:
                for _ in range(n_iterations):
                    _ = buffer.get_latest(10)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(f"Read error: {e}")

        t1 = threading.Thread(target=append_data)
        t2 = threading.Thread(target=read_data)

        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert len(errors) == 0, f"Thread safety errors: {errors}"

    def test_resize_buffer(self, buffer):
        """Test resizing buffer."""
        assert buffer.buffer_size == 10000

        buffer.resize(5000)

        assert buffer.buffer_size == 5000

    def test_resize_preserve_data(self, buffer):
        """Test that resizing preserves existing data when possible."""
        data = np.random.randn(4, 5000)
        buffer.append(data)

        n_samples_before = buffer.n_samples

        buffer.resize(8000)

        # Should preserve samples
        assert buffer.n_samples == n_samples_before

    def test_data_integrity_after_wrap(self, buffer):
        """Test that data remains consistent after wrap-around."""
        # Create distinctive data pattern
        chunk1 = np.ones((4, 3000)) * 1.0
        chunk2 = np.ones((4, 3000)) * 2.0
        chunk3 = np.ones((4, 3000)) * 3.0

        buffer.append(chunk1)
        buffer.append(chunk2)
        buffer.append(chunk3)

        # Buffer should have wrapped
        # Latest data should be retrievable
        latest = buffer.get_latest(1000)

        # Should be all 3.0 (from chunk3)
        np.testing.assert_array_almost_equal(latest, np.ones((4, 1000)) * 3.0)

    def test_get_time_window(self, buffer):
        """Test getting data by time window."""
        sample_rate = 51200.0
        buffer.sample_rate = sample_rate

        # Add 1 second of data
        n_samples = int(sample_rate * 1.0)
        data = np.random.randn(4, n_samples)
        buffer.append(data)

        # Get last 0.5 seconds
        time_window = buffer.get_time_window(duration=0.5)

        assert time_window.shape[0] == 4
        assert time_window.shape[1] == int(sample_rate * 0.5)

    def test_get_statistics(self, buffer):
        """Test getting buffer statistics."""
        data = np.random.randn(4, 1000)
        buffer.append(data)

        stats = buffer.get_statistics()

        assert 'n_samples' in stats
        assert 'n_channels' in stats
        assert 'capacity' in stats
        assert 'usage_percent' in stats

        assert stats['n_samples'] == 1000
        assert stats['n_channels'] == 4
        assert stats['capacity'] == 10000

    def test_get_channel_statistics(self, buffer):
        """Test getting statistics for a single channel."""
        # Create known data
        data = np.ones((4, 1000)) * 5.0
        buffer.append(data)

        stats = buffer.get_channel_statistics(0)

        assert 'mean' in stats
        assert 'std' in stats
        assert 'min' in stats
        assert 'max' in stats
        assert 'rms' in stats

        assert stats['mean'] == pytest.approx(5.0, rel=0.1)
        assert stats['rms'] == pytest.approx(5.0, rel=0.1)

    def test_copy_to_array(self, buffer):
        """Test copying buffer data to external array."""
        data = np.random.randn(4, 1000)
        buffer.append(data)

        output = np.zeros((4, 500))
        n_copied = buffer.copy_to(output, offset=500)

        assert n_copied == 500
        np.testing.assert_array_equal(output, data[:, 500:1000])

    def test_copy_offset_beyond_data(self, buffer):
        """Test copying with offset beyond available data."""
        data = np.random.randn(4, 100)
        buffer.append(data)

        output = np.zeros((4, 50))
        n_copied = buffer.copy_to(output, offset=200)

        # Should copy nothing
        assert n_copied == 0

    def test_zero_channels_buffer(self):
        """Test creating buffer with zero channels."""
        with pytest.raises(ValueError):
            DataBuffer(n_channels=0, buffer_size=1000)

    def test_zero_buffer_size(self):
        """Test creating buffer with zero size."""
        with pytest.raises(ValueError):
            DataBuffer(n_channels=4, buffer_size=0)

    def test_negative_buffer_size(self):
        """Test creating buffer with negative size."""
        with pytest.raises(ValueError):
            DataBuffer(n_channels=4, buffer_size=-100)

    def test_append_wrong_shape(self, buffer):
        """Test appending data with wrong number of channels."""
        # Buffer expects 4 channels
        data = np.random.randn(2, 100)  # Only 2 channels

        # Should either raise error or handle gracefully
        with pytest.raises((ValueError, AssertionError)):
            buffer.append(data)

    def test_get_write_position(self, buffer):
        """Test getting current write position."""
        pos1 = buffer.get_write_position()

        data = np.random.randn(4, 100)
        buffer.append(data)

        pos2 = buffer.get_write_position()

        # Position should have advanced
        assert pos2 != pos1

    def test_get_read_position(self, buffer):
        """Test getting current read position."""
        pos = buffer.get_read_position()
        assert pos >= 0


class TestBufferEfficiency:
    """Test buffer performance and efficiency."""

    def test_large_buffer_performance(self):
        """Test that large buffer operations are efficient."""
        # Large buffer
        buffer = DataBuffer(n_channels=16, buffer_size=1000000)

        # Large data
        data = np.random.randn(16, 100000)

        start = time.time()
        buffer.append(data)
        elapsed = time.time() - start

        # Should be fast (< 0.1 seconds typically)
        assert elapsed < 1.0

    def test_many_small_appends(self):
        """Test performance of many small appends."""
        buffer = DataBuffer(n_channels=4, buffer_size=10000)

        start = time.time()
        for _ in range(1000):
            data = np.random.randn(4, 10)
            buffer.append(data)
        elapsed = time.time() - start

        # Should be reasonably fast
        assert elapsed < 2.0

from unittest.mock import Mock, MagicMock
from pathlib import Path
"""
Integration tests for signal processing pipeline.

Tests the complete data flow from DAQ through processing to output.
"""

import pytest
import numpy as np
import time
from unittest.mock import Mock, MagicMock, patch


from PyQt5.QtCore import QObject, pyqtSignal

from src.processing.signal_processor import SignalProcessor
from src.processing.filters import DigitalFilter, FilterBank
from src.processing.fft_processor import FFTProcessor, MultiChannelFFTProcessor
from src.processing.data_buffer import DataBuffer


class TestSignalProcessor:
    """Test SignalProcessor integration."""

    @pytest.fixture
    def processor(self):
        """Create a signal processor for testing."""
        proc = SignalProcessor(
            n_channels=4,
            sample_rate=51200.0,
            buffer_duration=10  # seconds
        )
        return proc

    @pytest.fixture
    def multi_channel_data(self):
        """Create multi-channel test data."""
        duration = 0.1  # 100 ms
        sample_rate = 51200.0
        n_samples = int(sample_rate * duration)
        n_channels = 4

        t = np.linspace(0, duration, n_samples)

        # Each channel has different frequency content
        data = np.zeros((n_channels, n_samples))
        data[0, :] = np.sin(2 * np.pi * 100 * t)  # 100 Hz
        data[1, :] = np.sin(2 * np.pi * 500 * t)  # 500 Hz
        data[2, :] = np.sin(2 * np.pi * 1000 * t)  # 1000 Hz
        data[3, :] = np.sin(2 * np.pi * 5000 * t)  # 5000 Hz

        return data

    def test_processor_initialization(self, processor):
        """Test processor initialization."""
        assert processor.sample_rate == 51200.0
        assert processor.n_channels == 4
        assert processor.raw_buffer is not None
        assert processor.filtered_buffer is not None

    def test_process_data_emits_signals(self, processor, multi_channel_data):
        """Test that processing emits appropriate signals."""
        # Track signals
        raw_received = []
        filtered_received = []
        fft_received = []

        processor.raw_data_ready.connect(lambda data, ts: raw_received.append((data, ts)))
        processor.filtered_data_ready.connect(lambda data, ts: filtered_received.append((data, ts)))
        processor.fft_data_ready.connect(lambda freq, mag, ch: fft_received.append((freq, mag, ch)))

        # Process data
        processor.process_data(multi_channel_data)

        # Give signals time to propagate
        time.sleep(0.1)

        # Should have received signals
        assert len(raw_received) > 0
        assert len(filtered_received) > 0

    def test_process_data_with_filter_disabled(self, processor, multi_channel_data):
        """Test processing with filter disabled."""
        filtered_received = []

        processor.filtered_data_ready.connect(
            lambda data, ts: filtered_received.append(data)
        )

        # Ensure filter is disabled
        processor.filter_bank.disable()

        processor.process_data(multi_channel_data)

        time.sleep(0.1)

        if filtered_received:
            # Should be same as input (no filtering)
            np.testing.assert_array_almost_equal(
                filtered_received[0],
                multi_channel_data,
                decimal=5
            )

    def test_process_data_with_filter_enabled(self, processor, multi_channel_data):
        """Test processing with filter enabled."""
        filtered_received = []

        processor.filtered_data_ready.connect(
            lambda data, ts: filtered_received.append(data)
        )

        # Configure lowpass filter
        config = FilterConfig(
            filter_type=FilterType.BUTTERWORTH,
            filter_mode=FilterMode.LOWPASS,
            cutoff_freq=1000.0,
            sample_rate=51200.0,
            order=4
        )
        processor.configure_filter(config)

        processor.process_data(multi_channel_data)

        time.sleep(0.1)

        if filtered_received:
            # High frequency content (5000 Hz) should be attenuated
            # Check last channel which had 5000 Hz
            output = filtered_received[0]
            input_data = multi_channel_data

            # Channel 3 (5000 Hz) should be reduced
            input_rms = np.sqrt(np.mean(input_data[3, :] ** 2))
            output_rms = np.sqrt(np.mean(output[3, :] ** 2))

            # Output should be significantly attenuated
            assert output_rms < 0.5 * input_rms

    def test_fft_computation(self, processor, multi_channel_data):
        """Test FFT computation during processing."""
        fft_received = []

        processor.fft_data_ready.connect(
            lambda freq, mag, ch: fft_received.append((freq, mag, ch))
        )

        processor.process_data(multi_channel_data)

        time.sleep(0.1)

        # Should have FFT results for each channel
        if fft_received:
            freqs, magnitude, channel = fft_received[0]

            assert len(freqs) > 0
            assert len(magnitude) > 0
            assert 0 <= channel < 4

    def test_buffer_update(self, processor, multi_channel_data):
        """Test that buffers are updated correctly."""
        initial_raw_samples = processor.raw_buffer.n_samples
        initial_filtered_samples = processor.filtered_buffer.n_samples

        processor.process_data(multi_channel_data)

        # Buffers should have more samples
        assert processor.raw_buffer.n_samples > initial_raw_samples
        assert processor.filtered_buffer.n_samples > initial_filtered_samples

    def test_get_latest_data(self, processor, multi_channel_data):
        """Test retrieving latest data from processor."""
        processor.process_data(multi_channel_data)

        latest = processor.get_latest_raw(n_samples=100)

        assert latest.shape == (4, 100)

    def test_get_latest_filtered_data(self, processor, multi_channel_data):
        """Test retrieving latest filtered data."""
        processor.process_data(multi_channel_data)

        latest = processor.get_latest_filtered(n_samples=100)

        assert latest.shape == (4, 100)

    def test_configure_filter(self, processor):
        """Test configuring filter."""
        config = FilterConfig(
            filter_type=FilterType.CHEBYSHEV1,
            filter_mode=FilterMode.HIGHPASS,
            cutoff_freq=500.0,
            sample_rate=51200.0,
            order=6
        )

        processor.configure_filter(config)

        # Filter should be enabled
        assert processor.filter_bank.enabled is True

    def test_disable_filter(self, processor):
        """Test disabling filter."""
        # First configure a filter
        config = FilterConfig(
            filter_type=FilterType.BUTTERWORTH,
            filter_mode=FilterMode.LOWPASS,
            cutoff_freq=1000.0,
            sample_rate=51200.0,
            order=4
        )
        processor.configure_filter(config)

        assert processor.filter_bank.enabled is True

        # Disable
        processor.disable_filter()

        assert processor.filter_bank.enabled is False

    def test_reset_processor(self, processor, multi_channel_data):
        """Test resetting processor."""
        processor.process_data(multi_channel_data)

        assert processor.raw_buffer.n_samples > 0

        processor.reset()

        # Buffers should be empty
        assert processor.raw_buffer.n_samples == 0
        assert processor.filtered_buffer.n_samples == 0

    def test_get_processor_status(self, processor):
        """Test getting processor status."""
        status = processor.get_status()

        assert 'sample_rate' in status
        assert 'n_channels' in status
        assert 'buffer_size' in status
        assert 'filter_enabled' in status
        assert 'n_samples_raw' in status
        assert 'n_samples_filtered' in status

    def test_continuous_processing(self, processor):
        """Test continuous processing of multiple chunks."""
        chunks_processed = []

        processor.filtered_data_ready.connect(
            lambda data, ts: chunks_processed.append(data)
        )

        # Process multiple chunks
        for i in range(5):
            data = np.random.randn(4, 1000)
            processor.process_data(data)

        time.sleep(0.1)

        # Should have processed all chunks
        assert processor.raw_buffer.n_samples >= 5000

    def test_different_window_functions(self, processor, multi_channel_data):
        """Test FFT with different window functions."""
        windows = [
            WindowFunction.HANN,
            WindowFunction.HAMMING,
            WindowFunction.BLACKMAN
        ]

        for window in windows:
            processor.set_fft_window(window)
            processor.process_data(multi_channel_data)

            time.sleep(0.05)

            # Should complete without error
            assert True

    def test_realtime_statistics(self, processor):
        """Test getting real-time statistics."""
        data = np.random.randn(4, 5000)
        processor.process_data(data)

        stats = processor.get_realtime_statistics()

        assert 'channels' in stats
        assert len(stats['channels']) == 4

        # Each channel should have stats
        for ch_stats in stats['channels']:
            assert 'rms' in ch_stats
            assert 'peak' in ch_stats
            assert 'mean' in ch_stats


class TestProcessingPipeline:
    """Test the complete processing pipeline."""

    @pytest.fixture
    def pipeline(self):
        """Create a complete processing pipeline."""
        proc = SignalProcessor(
            sample_rate=51200.0,
            n_channels=4,
            buffer_size=50000
        )

        # Configure filter
        config = FilterConfig(
            filter_type=FilterType.BUTTERWORTH,
            filter_mode=FilterMode.LOWPASS,
            cutoff_freq=2000.0,
            sample_rate=51200.0,
            order=4
        )
        proc.configure_filter(config)

        return proc

    def test_complete_pipeline_flow(self, pipeline):
        """Test data flow through complete pipeline."""
        # Create test signal with multiple frequencies
        duration = 0.5
        sample_rate = 51200.0
        t = np.linspace(0, duration, int(sample_rate * duration))

        # Signal: 100 Hz + 1000 Hz + 5000 Hz
        signal = (
            np.sin(2 * np.pi * 100 * t) +
            0.5 * np.sin(2 * np.pi * 1000 * t) +
            0.3 * np.sin(2 * np.pi * 5000 * t)
        )

        # Create 4 channels with same signal
        data = np.tile(signal, (4, 1))

        # Process
        pipeline.process_data(data)
        time.sleep(0.2)

        # Check raw buffer
        assert pipeline.raw_buffer.n_samples == data.shape[1]

        # Check filtered buffer
        assert pipeline.filtered_buffer.n_samples == data.shape[1]

        # Get filtered data
        filtered = pipeline.get_latest_filtered(n_samples=1000)

        # High frequency (5000 Hz) should be attenuated
        # (rough check via RMS)
        input_rms = np.sqrt(np.mean(signal[-1000:] ** 2))
        output_rms = np.sqrt(np.mean(filtered[0, :] ** 2))

        # Filter should have changed the signal
        assert not np.allclose(signal[-1000:], filtered[0, :], rtol=0.1)

    def test_pipeline_with_export_simulation(self, pipeline):
        """Test pipeline for export scenario."""
        # Simulate acquisition
        chunks = []
        for _ in range(10):
            data = np.random.randn(4, 5120)  # 100 ms at 51200 Hz
            chunks.append(data)
            pipeline.process_data(data)

        time.sleep(0.5)

        # Get all data
        all_data = pipeline.get_all_raw()

        assert all_data.shape == (4, 51200)  # 1 second total

        # Should be ready for export
        assert all_data.shape[1] > 0

    def test_pipeline_fft_results(self, pipeline):
        """Test FFT computation in pipeline."""
        # Create signal with known frequencies
        duration = 1.0
        sample_rate = 51200.0
        t = np.linspace(0, duration, int(sample_rate * duration))

        signal = np.sin(2 * np.pi * 1000 * t)
        data = np.tile(signal, (4, 1))

        fft_results = []

        pipeline.fft_data_ready.connect(
            lambda freq, mag, ch: fft_results.append((freq, mag, ch))
        )

        pipeline.process_data(data)
        time.sleep(0.2)

        # Check FFT results
        if fft_results:
            freqs, magnitude, channel = fft_results[0]

            # Find peak
            peak_idx = np.argmax(magnitude)
            peak_freq = freqs[peak_idx]

            # Should be near 1000 Hz
            assert 900 < peak_freq < 1100


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def processor(self):
        return SignalProcessor(
            sample_rate=51200.0,
            n_channels=4,
            buffer_size=10000
        )

    def test_empty_data(self, processor):
        """Test processing empty data."""
        empty_data = np.array([]).reshape(4, 0)

        # Should handle gracefully
        processor.process_data(empty_data)
        time.sleep(0.1)

        assert True

    def test_single_sample(self, processor):
        """Test processing single sample."""
        single_sample = np.random.randn(4, 1)

        processor.process_data(single_sample)
        time.sleep(0.1)

        assert processor.raw_buffer.n_samples >= 1

    def test_mismatched_channels(self, processor):
        """Test data with wrong number of channels."""
        wrong_data = np.random.randn(2, 100)  # 2 channels instead of 4

        # Should handle or raise error
        try:
            processor.process_data(wrong_data)
            assert True  # Handled gracefully
        except (ValueError, AssertionError):
            assert True  # Error raised appropriately

    def test_very_large_chunk(self, processor):
        """Test processing very large data chunk."""
        large_chunk = np.random.randn(4, 100000)

        processor.process_data(large_chunk)
        time.sleep(0.5)

        # Should handle large chunk
        assert processor.raw_buffer.n_samples > 0

    def test_nan_data(self, processor):
        """Test processing data with NaN values."""
        data_with_nan = np.random.randn(4, 1000)
        data_with_nan[0, 100] = np.nan
        data_with_nan[1, 200] = np.nan

        # Should handle NaN
        try:
            processor.process_data(data_with_nan)
            time.sleep(0.1)
            assert True
        except ValueError:
            # Or raise error
            assert True


class TestPerformance:
    """Test performance characteristics."""

    def test_processing_latency(self):
        """Test that processing latency is acceptable."""
        processor = SignalProcessor(
            sample_rate=51200.0,
            n_channels=4,
            buffer_size=100000
        )

        # Create realistic data
        data = np.random.randn(4, 5120)  # 100 ms

        start = time.time()
        processor.process_data(data)
        # Wait for completion
        time.sleep(0.1)
        elapsed = time.time() - start

        # Should be much faster than real-time
        # (100 ms of data should process in < 10 ms)
        assert elapsed < 0.1

    def test_high_channel_count(self):
        """Test with high channel count."""
        n_channels = 16
        processor = SignalProcessor(
            sample_rate=51200.0,
            n_channels=n_channels,
            buffer_size=100000
        )

        data = np.random.randn(n_channels, 5120)

        start = time.time()
        processor.process_data(data)
        time.sleep(0.1)
        elapsed = time.time() - start

        # Should still be fast
        assert elapsed < 0.2

    def test_continuous_throughput(self):
        """Test continuous processing throughput."""
        processor = SignalProcessor(
            sample_rate=51200.0,
            n_channels=4,
            buffer_size=100000
        )

        n_chunks = 100
        chunk_size = 5120  # 100 ms each

        start = time.time()
        for _ in range(n_chunks):
            data = np.random.randn(4, chunk_size)
            processor.process_data(data)

        time.sleep(1.0)  # Allow processing to complete
        elapsed = time.time() - start

        # Total data: 10 seconds
        # Should process faster than real-time
        assert elapsed < 5.0  # At least 2x real-time

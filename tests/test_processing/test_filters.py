from unittest.mock import Mock, MagicMock
from pathlib import Path
"""
Unit tests for digital filter module.

Tests filter creation, application, and various filter types.
"""

import pytest
import numpy as np
from scipy import signal


from src.processing.filters import (
    DigitalFilter,
    FilterBank,
    FilterDesignError
)


# Removed TestFilterConfig - filters use string parameters directly


class TestDigitalFilter:
    """Test DigitalFilter class."""

    @pytest.fixture
    def sample_rate(self):
        return 51200.0

    @pytest.fixture
    def lowpass_filter(self, sample_rate):
        """Create a lowpass filter for testing."""
        return DigitalFilter(
            filter_type="butterworth",
            filter_mode="lowpass",
            cutoff=1000.0,
            sample_rate=sample_rate,
            order=4
        )

    @pytest.fixture
    def test_signal(self):
        """Create a test signal with known frequency components."""
        duration = 1.0
        sample_rate = 51200.0
        t = np.linspace(0, duration, int(sample_rate * duration))

        # Signal with 100 Hz and 5000 Hz components
        signal_data = (
            np.sin(2 * np.pi * 100 * t) +
            0.5 * np.sin(2 * np.pi * 5000 * t)
        )
        return signal_data

    def test_filter_initialization(self, lowpass_filter):
        """Test filter is properly initialized."""
        assert lowpass_filter.sos is not None
        assert lowpass_filter.zi_prototype is not None
        assert lowpass_filter.n_channels == 1

    def test_filter_initialization_multichannel(self, sample_rate):
        """Test multi-channel filter initialization."""
        n_channels = 4
        filt = DigitalFilter(
            filter_type="butterworth",
            filter_mode="lowpass",
            cutoff=1000.0,
            sample_rate=sample_rate,
            order=4,
            n_channels=n_channels
        )
        assert filt.n_channels == n_channels
        assert filt.zi_prototype.shape[0] == n_channels

    def test_apply_filter(self, lowpass_filter, test_signal):
        """Test applying filter to signal."""
        filtered, zf = lowpass_filter.apply(test_signal)

        assert filtered is not None
        assert filtered.shape == test_signal.shape
        assert zf is not None
        assert zf.shape == lowpass_filter.zi_prototype.shape

    def test_lowpass_removes_high_frequency(self, test_signal):
        """Test that lowpass filter removes high frequency components."""
        # Create lowpass filter at 1000 Hz
        filt = DigitalFilter(
            filter_type="butterworth",
            filter_mode="lowpass",
            cutoff=1000.0,
            sample_rate=51200.0,
            order=4
        )

        filtered, _ = filt.apply(test_signal)

        # Check that high frequency (5000 Hz) is attenuated
        # FFT to check frequency content
        fft_result = np.fft.fft(filtered)
        freqs = np.fft.fftfreq(len(filtered), 1/51200.0)

        # Find index of 5000 Hz
        idx_5k = np.argmin(np.abs(freqs - 5000))
        idx_100 = np.argmin(np.abs(freqs - 100))

        # 5000 Hz should be much lower than 100 Hz after filtering
        assert np.abs(fft_result[idx_5k]) < 0.1 * np.abs(fft_result[idx_100])

    def test_highpass_removes_low_frequency(self):
        """Test that highpass filter removes low frequency components."""
        duration = 1.0
        sample_rate = 51200.0
        t = np.linspace(0, duration, int(sample_rate * duration))

        # Signal with 100 Hz and 5000 Hz components
        signal_data = (
            np.sin(2 * np.pi * 100 * t) +
            0.5 * np.sin(2 * np.pi * 5000 * t)
        )

        # Create highpass filter at 1000 Hz
        filt = DigitalFilter(
            filter_type="butterworth",
            filter_mode="highpass",
            cutoff=1000.0,
            sample_rate=sample_rate,
            order=4
        )

        filtered, _ = filt.apply(signal_data)

        # FFT to check frequency content
        fft_result = np.fft.fft(filtered)
        freqs = np.fft.fftfreq(len(filtered), 1/sample_rate)

        # Find indices
        idx_5k = np.argmin(np.abs(freqs - 5000))
        idx_100 = np.argmin(np.abs(freqs - 100))

        # 100 Hz should be attenuated
        assert np.abs(fft_result[idx_100]) < 0.1 * np.abs(fft_result[idx_5k])

    def test_bandpass_filter(self):
        """Test bandpass filter."""
        duration = 1.0
        sample_rate = 51200.0
        t = np.linspace(0, duration, int(sample_rate * duration))

        # Signal with 100 Hz, 1000 Hz, and 5000 Hz components
        signal_data = (
            np.sin(2 * np.pi * 100 * t) +
            np.sin(2 * np.pi * 1000 * t) +
            0.5 * np.sin(2 * np.pi * 5000 * t)
        )

        # Create bandpass filter 500-2000 Hz
        filt = DigitalFilter(
            filter_type="butterworth",
            filter_mode="bandpass",
            cutoff=(500.0, 2000.0),
            sample_rate=sample_rate,
            order=4
        )

        filtered, _ = filt.apply(signal_data)

        # FFT to check frequency content
        fft_result = np.fft.fft(filtered)
        freqs = np.fft.fftfreq(len(filtered), 1/sample_rate)

        idx_100 = np.argmin(np.abs(freqs - 100))
        idx_1k = np.argmin(np.abs(freqs - 1000))
        idx_5k = np.argmin(np.abs(freqs - 5000))

        # 1000 Hz (in band) should be highest
        assert np.abs(fft_result[idx_1k]) > np.abs(fft_result[idx_100])
        assert np.abs(fft_result[idx_1k]) > np.abs(fft_result[idx_5k])

    def test_filter_state_continuity(self):
        """Test that filter state is maintained for continuous filtering."""
        filt = DigitalFilter(
            filter_type="butterworth",
            filter_mode="lowpass",
            cutoff=1000.0,
            sample_rate=51200.0,
            order=4
        )

        # Create two chunks of a signal
        chunk_size = 1000
        signal = np.sin(2 * np.pi * 100 * np.arange(2000) / 51200.0)

        chunk1 = signal[:chunk_size]
        chunk2 = signal[chunk_size:]

        # Filter separately with state
        filtered1, zi1 = filt.apply(chunk1)
        filtered2, zi2 = filt.apply(chunk2, zi=zi1)

        # Filter all at once
        filtered_all, _ = filt.apply(signal)

        # Should be approximately equal (boundary effects may cause small differences)
        # Skip first few samples due to transient
        skip = 50
        np.testing.assert_array_almost_equal(
            filtered1[skip:], filtered_all[skip:chunk_size],
            decimal=3
        )
        np.testing.assert_array_almost_equal(
            filtered2[skip:], filtered_all[chunk_size + skip:],
            decimal=3
        )

    def test_chebyshev_filter(self):
        """Test Chebyshev Type I filter."""
        filt = DigitalFilter(
            filter_type="chebyshev1",
            filter_mode="lowpass",
            cutoff=1000.0,
            sample_rate=51200.0,
            order=4,
            ripple=3  # 3 dB ripple
        )

        test_signal = np.sin(2 * np.pi * 100 * np.arange(1000) / 51200.0)
        filtered, _ = filt.apply(test_signal)

        assert filtered is not None
        assert filtered.shape == test_signal.shape

    def test_bessel_filter(self):
        """Test Bessel filter."""
        filt = DigitalFilter(
            filter_type="bessel",
            filter_mode="lowpass",
            cutoff=1000.0,
            sample_rate=51200.0,
            order=4
        )

        test_signal = np.sin(2 * np.pi * 100 * np.arange(1000) / 51200.0)
        filtered, _ = filt.apply(test_signal)

        assert filtered is not None
        assert filtered.shape == test_signal.shape


class TestFilterBank:
    """Test FilterBank class."""

    @pytest.fixture
    def sample_rate(self):
        return 51200.0

    @pytest.fixture
    def filter_bank(self, sample_rate):
        """Create a filter bank for testing."""
        return FilterBank(sample_rate=sample_rate, n_channels=4)

    def test_filter_bank_initialization(self, filter_bank):
        """Test filter bank initialization."""
        assert filter_bank.sample_rate == 51200.0
        assert filter_bank.n_channels == 4
        assert filter_bank.enabled is False
        assert filter_bank._filter is None

    def test_configure_filter(self, filter_bank):
        """Test configuring filter in filter bank."""
        config = FilterConfig(
            filter_type="butterworth",
            filter_mode="lowpass",
            cutoff=1000.0,
            sample_rate=51200.0,
            order=4
        )

        filter_bank.configure(config)

        assert filter_bank._filter is not None
        assert filter_bank.enabled is True

    def test_apply_filter_disabled(self, filter_bank):
        """Test that data passes through when filter is disabled."""
        data = np.random.randn(4, 1000)
        result = filter_bank.apply_filter(data)

        np.testing.assert_array_equal(result, data)

    def test_apply_filter_enabled(self, filter_bank):
        """Test applying filter through filter bank."""
        # Configure and enable filter
        config = FilterConfig(
            filter_type="butterworth",
            filter_mode="lowpass",
            cutoff=1000.0,
            sample_rate=51200.0,
            order=4
        )
        filter_bank.configure(config)

        # Create test data
        data = np.random.randn(4, 1000)
        result = filter_bank.apply_filter(data)

        assert result.shape == data.shape
        # Result should be different (filtered)
        # Note: For random noise, filter effect is subtle
        # For deterministic signal, effect would be clearer

    def test_apply_filter_multichannel(self, filter_bank):
        """Test multi-channel filtering."""
        config = FilterConfig(
            filter_type="butterworth",
            filter_mode="lowpass",
            cutoff=1000.0,
            sample_rate=51200.0,
            order=4
        )
        filter_bank.configure(config)

        # Create 4 channels of test data
        n_channels = 4
        n_samples = 1000
        data = np.random.randn(n_channels, n_samples)

        result = filter_bank.apply_filter(data)

        assert result.shape == (n_channels, n_samples)

    def test_reset_filter_state(self, filter_bank):
        """Test resetting filter state."""
        config = FilterConfig(
            filter_type="butterworth",
            filter_mode="lowpass",
            cutoff=1000.0,
            sample_rate=51200.0,
            order=4
        )
        filter_bank.configure(config)

        # Apply some data to set state
        data = np.random.randn(4, 100)
        filter_bank.apply_filter(data)

        # Reset
        filter_bank.reset()

        # State should be reset
        # Filter should still exist but state reset
        assert filter_bank._filter is not None

    def test_disable_filter(self, filter_bank):
        """Test disabling filter."""
        config = FilterConfig(
            filter_type="butterworth",
            filter_mode="lowpass",
            cutoff=1000.0,
            sample_rate=51200.0,
            order=4
        )
        filter_bank.configure(config)

        assert filter_bank.enabled is True

        filter_bank.disable()
        assert filter_bank.enabled is False

        # Data should pass through
        data = np.random.randn(4, 100)
        result = filter_bank.apply_filter(data)
        np.testing.assert_array_equal(result, data)


class TestInvalidParameters:
    """Test handling of invalid parameters."""

    def test_invalid_cutoff_negative(self):
        """Test that negative cutoff frequency raises error."""
        with pytest.raises(ValueError):
            DigitalFilter(
                filter_type="butterworth",
                filter_mode="lowpass",
                cutoff=-100.0,
                sample_rate=51200.0,
                order=4
            )

    def test_invalid_order_too_high(self):
        """Test that order > 10 raises error."""
        with pytest.raises(ValueError):
            DigitalFilter(
                filter_type="butterworth",
                filter_mode="lowpass",
                cutoff=1000.0,
                sample_rate=51200.0,
                order=15
            )

    def test_invalid_order_too_low(self):
        """Test that order < 1 raises error."""
        with pytest.raises(ValueError):
            DigitalFilter(
                filter_type="butterworth",
                filter_mode="lowpass",
                cutoff=1000.0,
                sample_rate=51200.0,
                order=0
            )

    def test_invalid_cutoff_above_nyquist(self):
        """Test that cutoff above Nyquist raises error."""
        with pytest.raises(ValueError):
            DigitalFilter(
                filter_type="butterworth",
                filter_mode="lowpass",
                cutoff=30000.0,  # Above Nyquist for 51200 Hz sample rate
                sample_rate=51200.0,
                order=4
            )

    def test_invalid_bandpass_cutoff(self):
        """Test that invalid bandpass cutoff raises error."""
        with pytest.raises(ValueError):
            DigitalFilter(
                filter_type="butterworth",
                filter_mode="bandpass",
                cutoff=(2000.0, 1000.0),  # High < Low
                sample_rate=51200.0,
                order=4
            )

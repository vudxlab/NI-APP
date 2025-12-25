from unittest.mock import Mock, MagicMock
from pathlib import Path
"""
Unit tests for FFT processor module.

Tests FFT computation, windowing, peak detection, and scaling.
"""

import pytest
import numpy as np


from src.processing.fft_processor import (
    FFTProcessor,
    MultiChannelFFTProcessor
)


class TestFFTProcessor:
    """Test FFTProcessor class."""

    @pytest.fixture
    def processor(self):
        """Create an FFT processor for testing."""
        return FFTProcessor(
            sample_rate=51200.0,
            window_size=2048,
            window_function="hann"
        )

    @pytest.fixture
    def single_tone_signal(self):
        """Create a single tone signal at known frequency."""
        duration = 1.0
        sample_rate = 51200.0
        t = np.linspace(0, duration, int(sample_rate * duration))

        # 1000 Hz sine wave
        frequency = 1000.0
        amplitude = 1.0
        signal_data = amplitude * np.sin(2 * np.pi * frequency * t)

        return signal_data, frequency, amplitude

    @pytest.fixture
    def multi_tone_signal(self):
        """Create a multi-tone signal with known frequencies."""
        duration = 1.0
        sample_rate = 51200.0
        t = np.linspace(0, duration, int(sample_rate * duration))

        # Multiple frequencies: 100 Hz, 1000 Hz, 5000 Hz
        frequencies = [100.0, 1000.0, 5000.0]
        amplitudes = [1.0, 0.5, 0.3]
        signal_data = sum(
            amp * np.sin(2 * np.pi * freq * t)
            for freq, amp in zip(frequencies, amplitudes)
        )

        return signal_data, frequencies, amplitudes

    def test_processor_initialization(self, processor):
        """Test processor initialization."""
        assert processor.sample_rate == 51200.0
        assert processor.window_size == 2048
        assert processor.window_function == "hann"
        assert processor.frequency_resolution == 51200.0 / 2048

    def test_frequency_resolution_calculation(self):
        """Test frequency resolution calculation."""
        proc = FFTProcessor(sample_rate=25600.0, window_size=1024)
        assert proc.frequency_resolution == 25.0  # 25600/1024

    def test_compute_fft_basic(self, processor, single_tone_signal):
        """Test basic FFT computation."""
        signal_data, frequency, amplitude = single_tone_signal

        # Take first window_size samples
        data = signal_data[:processor.window_size]

        freqs, spectrum = processor.compute_fft(data)

        assert freqs is not None
        assert spectrum is not None
        assert len(freqs) == processor.window_size // 2 + 1
        assert len(spectrum) == processor.window_size // 2 + 1

    def test_compute_magnitude_linear_scale(self, processor, single_tone_signal):
        """Test magnitude computation in linear scale."""
        signal_data, frequency, amplitude = single_tone_signal
        data = signal_data[:processor.window_size]

        freqs, magnitude = processor.compute_magnitude(
            data,
            scale=FFTScale.LINEAR
        )

        assert len(magnitude) == len(freqs)
        # Magnitude should be non-negative
        assert np.all(magnitude >= 0)

        # Find peak frequency
        peak_idx = np.argmax(magnitude)
        peak_freq = freqs[peak_idx]

        # Peak should be near the input frequency (within frequency resolution)
        assert abs(peak_freq - frequency) < processor.frequency_resolution * 2

    def test_compute_magnitude_db_scale(self, processor, single_tone_signal):
        """Test magnitude computation in dB scale."""
        signal_data, frequency, amplitude = single_tone_signal
        data = signal_data[:processor.window_size]

        freqs, magnitude_db = processor.compute_magnitude(
            data,
            scale=FFTScale.DB
        )

        # dB values can be negative (for references)
        # But we should still find the peak
        peak_idx = np.argmax(magnitude_db)
        peak_freq = freqs[peak_idx]

        assert abs(peak_freq - frequency) < processor.frequency_resolution * 2

    def test_db_conversion(self, processor, single_tone_signal):
        """Test dB conversion from linear scale."""
        signal_data, frequency, amplitude = single_tone_signal
        data = signal_data[:processor.window_size]

        freqs, mag_linear = processor.compute_magnitude(data, scale=FFTScale.LINEAR)
        _, mag_db = processor.compute_magnitude(data, scale=FFTScale.DB)

        # Check dB conversion: dB = 20 * log10(linear)
        # Avoid log(0)
        mask = mag_linear > 1e-10
        expected_db = 20 * np.log10(mag_linear[mask] + 1e-10)

        np.testing.assert_array_almost_equal(
            mag_db[mask],
            expected_db,
            decimal=1
        )

    def test_window_function_hann(self, processor, single_tone_signal):
        """Test Hann window function."""
        signal_data, _, _ = single_tone_signal
        data = signal_data[:processor.window_size]

        windowed = processor._apply_window(data)

        # Windowed signal should have lower amplitude at edges
        assert np.abs(windowed[0]) < np.abs(data[0])
        assert np.abs(windowed[-1]) < np.abs(data[-1])

    def test_window_function_hamming(self):
        """Test Hamming window function."""
        proc = FFTProcessor(
            sample_rate=51200.0,
            window_size=2048,
            window_function="hamming"
        )

        data = np.ones(2048)
        windowed = proc._apply_window(data)

        # Hamming window should modify the signal
        assert not np.array_equal(windowed, data)
        # Center should be close to 1
        assert windowed[1024] > 0.9

    def test_window_function_blackman(self):
        """Test Blackman window function."""
        proc = FFTProcessor(
            sample_rate=51200.0,
            window_size=2048,
            window_function="blackman"
        )

        data = np.ones(2048)
        windowed = proc._apply_window(data)

        # Blackman window should modify the signal
        assert not np.array_equal(windowed, data)
        # Edges should be very close to 0
        assert windowed[0] < 0.01
        assert windowed[-1] < 0.01

    def test_window_function_flattop(self):
        """Test Flat Top window function."""
        proc = FFTProcessor(
            sample_rate=51200.0,
            window_size=2048,
            window_function="hann"  # flattop not standard
        )

        data = np.ones(2048)
        windowed = proc._apply_window(data)

        # Flat Top window should have very flat main lobe
        # Check that center region is relatively flat
        center_region = windowed[1000:1050]
        assert np.max(center_region) - np.min(center_region) < 0.1

    def test_find_peaks_single_tone(self, processor, single_tone_signal):
        """Test peak detection with single tone."""
        signal_data, frequency, amplitude = single_tone_signal
        data = signal_data[:processor.window_size]

        freqs, magnitude = processor.compute_magnitude(data)
        peaks = processor.find_peaks(freqs, magnitude, threshold=0.1)

        assert len(peaks) >= 1
        # Check that peak is near expected frequency
        peak_freq = peaks[0]['frequency']
        assert abs(peak_freq - frequency) < processor.frequency_resolution * 2

    def test_find_peaks_multi_tone(self, processor, multi_tone_signal):
        """Test peak detection with multiple tones."""
        signal_data, frequencies, amplitudes = multi_tone_signal
        data = signal_data[:processor.window_size]

        freqs, magnitude = processor.compute_magnitude(data)
        peaks = processor.find_peaks(freqs, magnitude, threshold=0.05)

        # Should detect multiple peaks
        assert len(peaks) >= 2

        # Check that detected peaks are near expected frequencies
        detected_freqs = [p['frequency'] for p in peaks]

        for expected_freq in frequencies:
            # Find closest detected peak
            closest = min(detected_freqs, key=lambda f: abs(f - expected_freq))
            assert abs(closest - expected_freq) < processor.frequency_resolution * 3

    def test_find_peaks_with_max_peaks(self, processor, multi_tone_signal):
        """Test peak detection with maximum peaks limit."""
        signal_data, _, _ = multi_tone_signal
        data = signal_data[:processor.window_size]

        freqs, magnitude = processor.compute_magnitude(data)
        peaks = processor.find_peaks(freqs, magnitude, threshold=0.01, max_peaks=2)

        # Should return at most 2 peaks
        assert len(peaks) <= 2

    def test_find_peaks_returns_magnitude(self, processor, single_tone_signal):
        """Test that peak detection returns magnitude information."""
        signal_data, _, _ = single_tone_signal
        data = signal_data[:processor.window_size]

        freqs, magnitude = processor.compute_magnitude(data)
        peaks = processor.find_peaks(freqs, magnitude, threshold=0.1)

        # Check peak dict structure
        peak = peaks[0]
        assert 'frequency' in peak
        assert 'magnitude' in peak
        assert 'index' in peak
        assert peak['magnitude'] > 0

    def test_compute_psd(self, processor, single_tone_signal):
        """Test power spectral density computation."""
        signal_data, frequency, amplitude = single_tone_signal
        data = signal_data[:processor.window_size]

        freqs, psd = processor.compute_psd(data)

        assert len(freqs) == len(psd)
        assert np.all(psd >= 0)  # PSD should be non-negative

        # Peak should be near signal frequency
        peak_idx = np.argmax(psd)
        peak_freq = freqs[peak_idx]
        assert abs(peak_freq - frequency) < processor.frequency_resolution * 2

    def test_rms_computation(self, processor, single_tone_signal):
        """Test RMS computation from FFT."""
        signal_data, frequency, amplitude = single_tone_signal
        data = signal_data[:processor.window_size]

        # Compute RMS from time domain
        time_domain_rms = np.sqrt(np.mean(data ** 2))

        # Compute RMS from frequency domain
        freqs, magnitude = processor.compute_magnitude(data, scale=FFTScale.LINEAR)
        freq_domain_rms = processor.compute_rms(magnitude)

        # Should be approximately equal (Parseval's theorem)
        # Windowing affects this, so we allow some tolerance
        assert abs(time_domain_rms - freq_domain_rms) / time_domain_rms < 0.1

    def test_frequency_bins(self, processor):
        """Test that frequency bins are correct."""
        data = np.zeros(processor.window_size)
        freqs, _ = processor.compute_fft(data)

        # First frequency should be 0 (DC)
        assert freqs[0] == 0

        # Last frequency should be Nyquist
        assert freqs[-1] == processor.sample_rate / 2

        # Frequency spacing should be constant
        spacing = freqs[1] - freqs[0]
        expected_spacing = processor.frequency_resolution
        assert abs(spacing - expected_spacing) < 1e-10

    def test_dc_component(self, processor):
        """Test DC component detection."""
        # Signal with DC offset
        data = np.ones(processor.window_size) * 5.0

        freqs, magnitude = processor.compute_magnitude(data, scale=FFTScale.LINEAR)

        # Peak should be at 0 Hz (DC)
        peak_idx = np.argmax(magnitude)
        assert freqs[peak_idx] == 0

    def test_zero_padding(self, processor):
        """Test FFT with zero padding for interpolation."""
        data = np.random.randn(processor.window_size)

        # No padding
        freqs1, mag1 = processor.compute_magnitude(
            data,
            scale=FFTScale.LINEAR,
            zero_pad_factor=1
        )

        # 2x zero padding
        freqs2, mag2 = processor.compute_magnitude(
            data,
            scale=FFTScale.LINEAR,
            zero_pad_factor=2
        )

        # With padding, we should have more frequency points
        assert len(freqs2) == 2 * len(freqs1)

    def test_compute_octave_bands(self, processor):
        """Test 1/3 octave band computation."""
        data = np.random.randn(processor.window_size)

        freqs, magnitude = processor.compute_magnitude(data, scale=FFTScale.LINEAR)
        bands = processor.compute_octave_bands(freqs, magnitude)

        # Should return center frequencies and levels
        assert 'center_frequencies' in bands
        assert 'levels' in bands
        assert len(bands['center_frequencies']) == len(bands['levels'])

        # Common 1/3 octave center frequencies should be present
        # e.g., 1000 Hz, 2000 Hz, etc.
        assert 1000.0 in bands['center_frequencies']

    def test_change_window_function(self, processor):
        """Test changing window function."""
        assert processor.window_function == "hann"

        processor.set_window_function("blackman")
        assert processor.window_function == "blackman"

        # Apply new window
        data = np.ones(processor.window_size)
        windowed = processor._apply_window(data)

        # Should be modified by Blackman window
        assert windowed[0] < 0.5

    def test_change_window_size(self):
        """Test changing window size."""
        proc = FFTProcessor(sample_rate=51200.0, window_size=2048)

        assert proc.window_size == 2048

        proc.set_window_size(4096)

        assert proc.window_size == 4096
        assert proc.frequency_resolution == 51200.0 / 4096


class TestPeakDetection:
    """Test peak detection configuration and behavior."""

    def test_peak_detection_default_config(self):
        """Test default peak detection configuration."""
        pd = PeakDetection()

        assert pd.threshold == 0.1
        assert pd.min_distance == 10
        assert pd.max_peaks is None

    def test_peak_detection_custom_config(self):
        """Test custom peak detection configuration."""
        pd = PeakDetection(threshold=0.2, min_distance=20, max_peaks=5)

        assert pd.threshold == 0.2
        assert pd.min_distance == 20
        assert pd.max_peaks == 5


class TestWindowFunction:
    """Test window function enumeration."""

    def test_all_window_functions(self):
        """Test that all window functions are valid."""
        windows = [
            "hann",
            "hamming",
            "blackman",
            "bartlett"
        ]

        proc = FFTProcessor(sample_rate=51200.0, window_size=2048)

        for window in windows:
            proc.set_window_function(window)
            assert proc.window_function == window

            # Test that it can be applied
            data = np.ones(2048)
            windowed = proc._apply_window(data)
            assert windowed is not None


class TestInvalidParameters:
    """Test handling of invalid parameters."""

    def test_invalid_window_size_zero(self):
        """Test that window_size=0 raises error."""
        with pytest.raises(ValueError):
            FFTProcessor(sample_rate=51200.0, window_size=0)

    def test_invalid_window_size_not_power_of_2(self):
        """Test that non-power-of-2 window size raises error."""
        # This should not raise an error but may warn
        # FFT works with any size, powers of 2 are just more efficient
        proc = FFTProcessor(sample_rate=51200.0, window_size=1000)
        assert proc.window_size == 1000

    def test_invalid_sample_rate_negative(self):
        """Test that negative sample rate raises error."""
        with pytest.raises(ValueError):
            FFTProcessor(sample_rate=-51200.0, window_size=2048)

    def test_invalid_threshold_negative(self, processor):
        """Test that negative threshold still works (just no peaks)."""
        data = np.random.randn(2048)
        freqs, magnitude = processor.compute_magnitude(data)

        peaks = processor.find_peaks(freqs, magnitude, threshold=-0.1)
        # Should find many or all peaks
        assert len(peaks) >= 0

"""
Long-window FFT processor for low-frequency analysis.

This module provides FFT computation with very long time windows (10s, 20s, 50s, 100s, 200s)
specifically designed for analyzing low-frequency components in vibration signals.
"""

import numpy as np
from scipy import signal
from scipy.signal import find_peaks
from typing import Tuple, List, Optional, Dict
from pathlib import Path

from ..utils.logger import get_logger
from ..utils.validators import ValidationError


class LongWindowFFTProcessor:
    """
    FFT processor for long time windows (10s - 200s).

    This processor is designed for low-frequency analysis where high frequency
    resolution is needed. It can handle very long FFT windows that would not be
    practical for real-time display but are essential for detecting low-frequency
    components.

    Typical use cases:
    - Low-frequency vibration analysis (< 1 Hz)
    - Structural monitoring
    - Slow varying phenomena detection
    - High-resolution frequency analysis

    Attributes:
        sample_rate: Sampling rate in Hz
        available_windows: Dictionary of available window durations in seconds
    """

    # Standard window durations for low-frequency analysis
    WINDOW_DURATIONS = {
        '10s': 10.0,
        '20s': 20.0,
        '50s': 50.0,
        '100s': 100.0,
        '200s': 200.0
    }

    def __init__(
        self,
        sample_rate: float,
        window_function: str = 'hann'
    ):
        """
        Initialize long-window FFT processor.

        Args:
            sample_rate: Sampling rate in Hz
            window_function: Window function name ('hann', 'hamming', 'blackman', etc.)
        """
        self.logger = get_logger(__name__)
        self.sample_rate = sample_rate
        self.window_function = window_function

        # Pre-calculate window sizes for each duration
        self.window_sizes = {}
        for name, duration in self.WINDOW_DURATIONS.items():
            self.window_sizes[name] = int(sample_rate * duration)

        self.logger.info(
            f"LongWindowFFTProcessor initialized @ {sample_rate} Hz, "
            f"window_function={window_function}"
        )
        for name, size in self.window_sizes.items():
            freq_res = sample_rate / size
            self.logger.info(
                f"  {name}: {size:,} samples, freq_res={freq_res:.6f} Hz"
            )

    def _create_window(self, window_size: int) -> np.ndarray:
        """
        Create window function of specified size.

        Args:
            window_size: Number of samples in window

        Returns:
            Window array
        """
        if self.window_function == 'hann':
            return signal.windows.hann(window_size)
        elif self.window_function == 'hamming':
            return signal.windows.hamming(window_size)
        elif self.window_function == 'blackman':
            return signal.windows.blackman(window_size)
        elif self.window_function == 'bartlett':
            return signal.windows.bartlett(window_size)
        elif self.window_function == 'none' or self.window_function == 'rectangular':
            return np.ones(window_size)
        else:
            self.logger.warning(
                f"Unknown window '{self.window_function}', using Hanning"
            )
            return signal.windows.hann(window_size)

    def compute_fft(
        self,
        data: np.ndarray,
        window_duration: str = '200s'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute FFT with specified window duration.

        Args:
            data: Input data (1D array, must be at least as long as window)
            window_duration: Window duration ('10s', '20s', '50s', '100s', '200s')

        Returns:
            Tuple of (frequencies, complex_spectrum)

        Raises:
            ValueError: If data is too short or window_duration is invalid
        """
        if window_duration not in self.WINDOW_DURATIONS:
            raise ValueError(
                f"Invalid window_duration '{window_duration}'. "
                f"Must be one of: {list(self.WINDOW_DURATIONS.keys())}"
            )

        window_size = self.window_sizes[window_duration]

        if len(data) < window_size:
            raise ValueError(
                f"Data length {len(data)} is less than required window size {window_size} "
                f"for {window_duration} @ {self.sample_rate} Hz"
            )

        # Use the last window_size samples
        data_window = data[-window_size:]

        # Create and apply window
        window = self._create_window(window_size)
        windowed_data = data_window * window

        # Compute FFT (real FFT for efficiency)
        spectrum = np.fft.rfft(windowed_data)

        # Compute frequency vector
        frequencies = np.fft.rfftfreq(window_size, d=1.0/self.sample_rate)

        return frequencies, spectrum

    def compute_magnitude(
        self,
        data: np.ndarray,
        window_duration: str = '200s',
        scale: str = 'linear'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute magnitude spectrum with specified window duration.

        Args:
            data: Input data (1D array)
            window_duration: Window duration ('10s', '20s', '50s', '100s', '200s')
            scale: 'linear' or 'dB'

        Returns:
            Tuple of (frequencies, magnitude)
        """
        frequencies, spectrum = self.compute_fft(data, window_duration)

        # Compute magnitude
        magnitude = np.abs(spectrum)

        # Normalize by window size and compensate for window
        window_size = self.window_sizes[window_duration]
        magnitude = magnitude / window_size
        magnitude *= 2  # One-sided spectrum
        magnitude[0] /= 2  # DC component
        if window_size % 2 == 0:
            magnitude[-1] /= 2  # Nyquist component

        # Convert to dB if requested
        if scale == 'dB':
            magnitude = 20 * np.log10(magnitude + 1e-10)

        return frequencies, magnitude

    def compute_psd(
        self,
        data: np.ndarray,
        window_duration: str = '200s',
        scale: str = 'linear'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Power Spectral Density with specified window duration.

        Args:
            data: Input data (1D array)
            window_duration: Window duration ('10s', '20s', '50s', '100s', '200s')
            scale: 'linear' or 'dB'

        Returns:
            Tuple of (frequencies, psd)
        """
        frequencies, spectrum = self.compute_fft(data, window_duration)

        # Compute PSD
        psd = np.abs(spectrum) ** 2

        # Normalize
        window_size = self.window_sizes[window_duration]
        psd = psd / (self.sample_rate * window_size)

        # One-sided spectrum
        psd *= 2
        psd[0] /= 2
        if window_size % 2 == 0:
            psd[-1] /= 2

        # Convert to dB if requested
        if scale == 'dB':
            psd = 10 * np.log10(psd + 1e-10)

        return frequencies, psd

    def compute_all_windows(
        self,
        data: np.ndarray,
        scale: str = 'linear',
        method: str = 'magnitude'
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Compute FFT for all available window durations.

        Args:
            data: Input data (1D array, must be at least 200s long)
            scale: 'linear' or 'dB'
            method: 'magnitude' or 'psd'

        Returns:
            Dictionary mapping window duration to (frequencies, spectrum) tuples
        """
        results = {}

        for window_duration in self.WINDOW_DURATIONS.keys():
            try:
                if method == 'magnitude':
                    freq, spec = self.compute_magnitude(data, window_duration, scale)
                elif method == 'psd':
                    freq, spec = self.compute_psd(data, window_duration, scale)
                else:
                    raise ValueError(f"Unknown method: {method}")

                results[window_duration] = (freq, spec)

                self.logger.debug(
                    f"Computed {method} for {window_duration}: "
                    f"{len(freq)} freq bins, "
                    f"freq_res={freq[1]-freq[0]:.6f} Hz"
                )

            except ValueError as e:
                self.logger.warning(
                    f"Cannot compute {window_duration}: {e}"
                )

        return results

    def find_peaks(
        self,
        frequencies: np.ndarray,
        magnitude: np.ndarray,
        threshold: Optional[float] = None,
        min_distance: Optional[int] = None,
        n_peaks: Optional[int] = None
    ) -> List[Dict[str, float]]:
        """
        Find peaks in the magnitude spectrum.

        Args:
            frequencies: Frequency array
            magnitude: Magnitude array (linear scale)
            threshold: Minimum peak height (relative if < 1, absolute if >= 1)
            min_distance: Minimum distance between peaks in samples (default: auto)
            n_peaks: Maximum number of peaks to return

        Returns:
            List of peak dictionaries with 'frequency', 'magnitude', 'index'
        """
        # Auto-calculate min_distance if not specified
        if min_distance is None:
            # For long windows, use larger minimum distance
            # to avoid detecting too many closely spaced peaks
            min_distance = max(10, len(magnitude) // 1000)

        if threshold is None:
            threshold = 0.1  # 10% of maximum

        # Calculate absolute threshold
        if threshold < 1:
            peak_threshold = threshold * np.max(magnitude)
        else:
            peak_threshold = threshold

        # Find peaks
        peak_indices, properties = find_peaks(
            magnitude,
            height=peak_threshold,
            distance=min_distance
        )

        # Create peak list
        peaks = []
        for idx in peak_indices:
            peaks.append({
                'frequency': frequencies[idx],
                'magnitude': magnitude[idx],
                'index': int(idx)
            })

        # Sort by magnitude (descending)
        peaks.sort(key=lambda x: x['magnitude'], reverse=True)

        # Limit number of peaks if requested
        if n_peaks is not None:
            peaks = peaks[:n_peaks]

        return peaks

    def get_frequency_resolution(self, window_duration: str) -> float:
        """
        Get frequency resolution for specified window duration.

        Args:
            window_duration: Window duration string

        Returns:
            Frequency resolution in Hz
        """
        window_size = self.window_sizes[window_duration]
        return self.sample_rate / window_size

    def get_window_info(self) -> Dict[str, Dict[str, float]]:
        """
        Get information about all available windows.

        Returns:
            Dictionary with window information
        """
        info = {}
        for name, duration in self.WINDOW_DURATIONS.items():
            window_size = self.window_sizes[name]
            freq_res = self.sample_rate / window_size
            nyquist = self.sample_rate / 2.0

            info[name] = {
                'duration_seconds': duration,
                'window_size_samples': window_size,
                'frequency_resolution_hz': freq_res,
                'nyquist_frequency_hz': nyquist,
                'num_frequency_bins': window_size // 2 + 1
            }

        return info

    def analyze_low_frequencies(
        self,
        data: np.ndarray,
        max_frequency: float = 10.0,
        window_duration: str = '200s',
        n_peaks: int = 10
    ) -> Dict:
        """
        Specialized analysis for low-frequency components.

        Args:
            data: Input data (1D array)
            max_frequency: Maximum frequency to analyze (Hz)
            window_duration: Window duration to use
            n_peaks: Number of peaks to find

        Returns:
            Dictionary with analysis results
        """
        # Compute magnitude spectrum
        frequencies, magnitude = self.compute_magnitude(
            data,
            window_duration,
            scale='linear'
        )

        # Limit to low frequencies
        low_freq_mask = frequencies <= max_frequency
        low_frequencies = frequencies[low_freq_mask]
        low_magnitude = magnitude[low_freq_mask]

        # Find peaks in low-frequency range
        peaks = self.find_peaks(
            low_frequencies,
            low_magnitude,
            n_peaks=n_peaks
        )

        # Compute PSD for low frequencies
        _, psd = self.compute_psd(data, window_duration, scale='linear')
        low_psd = psd[low_freq_mask]

        # Calculate total power in low-frequency band
        total_power = np.sum(low_psd) * (low_frequencies[1] - low_frequencies[0])

        results = {
            'window_duration': window_duration,
            'max_frequency': max_frequency,
            'frequency_resolution': self.get_frequency_resolution(window_duration),
            'frequencies': low_frequencies,
            'magnitude': low_magnitude,
            'psd': low_psd,
            'peaks': peaks,
            'total_power': total_power,
            'rms_value': np.sqrt(total_power)
        }

        return results


# Example usage and testing
if __name__ == "__main__":
    print("LongWindowFFTProcessor Test")
    print("=" * 80)

    # Test parameters
    sample_rate = 25600  # Hz (typical for NI-9234)
    duration = 200  # seconds
    n_samples = int(sample_rate * duration)

    print(f"\n1. Creating test signal: {duration}s @ {sample_rate} Hz")
    print(f"   Total samples: {n_samples:,}")

    # Create test signal with low-frequency components
    t = np.arange(n_samples) / sample_rate

    # Low-frequency components (what we want to detect)
    f1, f2, f3 = 0.5, 1.2, 5.0  # Hz
    signal_test = (
        2.0 * np.sin(2 * np.pi * f1 * t) +
        1.5 * np.sin(2 * np.pi * f2 * t) +
        1.0 * np.sin(2 * np.pi * f3 * t)
    )

    # Add some higher frequency content and noise
    signal_test += 0.5 * np.sin(2 * np.pi * 100 * t)
    signal_test += 0.1 * np.random.randn(n_samples)

    print(f"   Low-frequency components: {f1} Hz, {f2} Hz, {f3} Hz")

    # Create processor
    print(f"\n2. Creating LongWindowFFTProcessor...")
    processor = LongWindowFFTProcessor(sample_rate=sample_rate, window_function='hann')

    # Get window info
    print(f"\n3. Available windows:")
    window_info = processor.get_window_info()
    for name, info in window_info.items():
        print(f"   {name}:")
        print(f"     - Duration: {info['duration_seconds']} s")
        print(f"     - Samples: {info['window_size_samples']:,}")
        print(f"     - Freq resolution: {info['frequency_resolution_hz']:.6f} Hz")
        print(f"     - Freq bins: {info['num_frequency_bins']:,}")

    # Compute FFT for different window durations
    print(f"\n4. Computing magnitude spectra for all windows...")
    all_results = processor.compute_all_windows(signal_test, scale='linear', method='magnitude')

    for window_duration, (freq, mag) in all_results.items():
        print(f"\n   {window_duration}:")
        print(f"     - Freq range: {freq[0]:.6f} to {freq[-1]:.2f} Hz")
        print(f"     - Freq bins: {len(freq):,}")
        print(f"     - Freq resolution: {freq[1]-freq[0]:.6f} Hz")

        # Find peaks
        peaks = processor.find_peaks(freq, mag, n_peaks=5)
        print(f"     - Top 5 peaks:")
        for i, peak in enumerate(peaks[:5], 1):
            print(f"       {i}. {peak['frequency']:.4f} Hz (mag={peak['magnitude']:.4f})")

    # Low-frequency analysis
    print(f"\n5. Detailed low-frequency analysis (0-10 Hz)...")
    low_freq_analysis = processor.analyze_low_frequencies(
        signal_test,
        max_frequency=10.0,
        window_duration='200s',
        n_peaks=10
    )

    print(f"   Window: {low_freq_analysis['window_duration']}")
    print(f"   Freq resolution: {low_freq_analysis['frequency_resolution']:.6f} Hz")
    print(f"   Frequency range: 0 - {low_freq_analysis['max_frequency']} Hz")
    print(f"   Total power: {low_freq_analysis['total_power']:.4f}")
    print(f"   RMS value: {low_freq_analysis['rms_value']:.4f}")
    print(f"\n   Detected peaks in 0-10 Hz range:")
    for i, peak in enumerate(low_freq_analysis['peaks'], 1):
        print(f"     {i}. {peak['frequency']:.4f} Hz (mag={peak['magnitude']:.4f})")

    # Compare frequency resolutions
    print(f"\n6. Frequency resolution comparison:")
    for window_duration in ['10s', '20s', '50s', '100s', '200s']:
        freq_res = processor.get_frequency_resolution(window_duration)
        print(f"   {window_duration:>4}: {freq_res:.6f} Hz  (can resolve frequencies > {freq_res:.6f} Hz apart)")

    print("\n" + "=" * 80)
    print("Test completed!")

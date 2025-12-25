"""
FFT processor for frequency domain analysis.

This module provides FFT computation, power spectral density (PSD) calculation,
and peak detection for vibration analysis.
"""

import numpy as np
from scipy import signal
from scipy.signal import find_peaks
from typing import Tuple, List, Optional, Dict
import warnings

from ..utils.logger import get_logger
from ..utils.constants import ProcessingDefaults
from ..utils.validators import validate_fft_window_size, validate_window_function, ValidationError


class FFTProcessor:
    """
    FFT processor for frequency domain analysis.

    This class handles FFT computation with windowing, averaging,
    and peak detection for vibration analysis.
    """

    def __init__(
        self,
        sample_rate: float,
        window_size: int = ProcessingDefaults.DEFAULT_FFT_WINDOW_SIZE,
        window_function: str = ProcessingDefaults.DEFAULT_WINDOW_FUNCTION,
        overlap: float = ProcessingDefaults.DEFAULT_FFT_OVERLAP
    ):
        """
        Initialize FFT processor.

        Args:
            sample_rate: Sampling rate in Hz
            window_size: FFT window size (number of samples, should be power of 2)
            window_function: Window function name ("hann", "hamming", "blackman", etc.)
            overlap: Overlap fraction (0.0 to 1.0)

        Raises:
            ValidationError: If parameters are invalid
        """
        self.logger = get_logger(__name__)

        validate_fft_window_size(window_size, sample_rate)
        validate_window_function(window_function)

        if not 0.0 <= overlap < 1.0:
            raise ValidationError(f"Overlap must be in [0, 1), got {overlap}")

        self.sample_rate = sample_rate
        self.window_size = window_size
        self.window_function = window_function
        self.overlap = overlap

        # Create window
        self.window = self._create_window()

        # Pre-compute frequency vector
        self.frequencies = np.fft.rfftfreq(window_size, d=1.0/sample_rate)

        # Averaging
        self._averaged_psd = None
        self._n_averages = 0

        self.logger.info(
            f"FFTProcessor initialized: window_size={window_size}, "
            f"window={window_function}, overlap={overlap}"
        )

    def _create_window(self) -> np.ndarray:
        """
        Create window function.

        Returns:
            Window array of length window_size
        """
        if self.window_function == ProcessingDefaults.WINDOW_HANNING:
            return signal.windows.hann(self.window_size)
        elif self.window_function == ProcessingDefaults.WINDOW_HAMMING:
            return signal.windows.hamming(self.window_size)
        elif self.window_function == ProcessingDefaults.WINDOW_BLACKMAN:
            return signal.windows.blackman(self.window_size)
        elif self.window_function == ProcessingDefaults.WINDOW_BARTLETT:
            return signal.windows.bartlett(self.window_size)
        elif self.window_function == ProcessingDefaults.WINDOW_NONE:
            return np.ones(self.window_size)
        else:
            self.logger.warning(
                f"Unknown window '{self.window_function}', using Hanning"
            )
            return signal.windows.hann(self.window_size)

    def compute_fft(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute FFT of input data.

        Args:
            data: Input data (1D array of length >= window_size)

        Returns:
            Tuple of (frequencies, complex_spectrum)
        """
        if len(data) < self.window_size:
            raise ValueError(
                f"Data length {len(data)} < window size {self.window_size}"
            )

        # Use only the last window_size samples
        data_window = data[-self.window_size:]

        # Apply window
        windowed_data = data_window * self.window

        # Compute FFT (real FFT for efficiency)
        spectrum = np.fft.rfft(windowed_data)

        return self.frequencies, spectrum

    def compute_magnitude(
        self,
        data: np.ndarray,
        scale: str = 'linear'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute magnitude spectrum.

        Args:
            data: Input data (1D array)
            scale: 'linear' or 'dB'

        Returns:
            Tuple of (frequencies, magnitude)
        """
        frequencies, spectrum = self.compute_fft(data)

        # Compute magnitude
        magnitude = np.abs(spectrum)

        # Normalize by window size and compensate for window
        magnitude = magnitude / self.window_size
        magnitude *= 2  # Account for one-sided spectrum (except DC and Nyquist)
        magnitude[0] /= 2  # DC component
        if len(magnitude) > 1 and self.window_size % 2 == 0:
            magnitude[-1] /= 2  # Nyquist component

        # Convert to dB if requested
        if scale == 'dB':
            magnitude = 20 * np.log10(magnitude + 1e-10)

        return frequencies, magnitude

    def compute_psd(
        self,
        data: np.ndarray,
        scale: str = 'linear'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Power Spectral Density (PSD).

        Args:
            data: Input data (1D array)
            scale: 'linear' or 'dB'

        Returns:
            Tuple of (frequencies, psd)
        """
        frequencies, spectrum = self.compute_fft(data)

        # Compute PSD
        psd = np.abs(spectrum) ** 2

        # Normalize
        psd = psd / (self.sample_rate * self.window_size)

        # One-sided spectrum
        psd *= 2
        psd[0] /= 2
        if len(psd) > 1 and self.window_size % 2 == 0:
            psd[-1] /= 2

        # Convert to dB if requested
        if scale == 'dB':
            psd = 10 * np.log10(psd + 1e-10)

        return frequencies, psd

    def compute_welch_psd(
        self,
        data: np.ndarray,
        scale: str = 'linear'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute PSD using Welch's method (averaged periodogram).

        This provides better noise reduction than a single FFT.

        Args:
            data: Input data (1D array, should be longer than window_size)
            scale: 'linear' or 'dB'

        Returns:
            Tuple of (frequencies, psd)
        """
        # Use scipy's welch function
        frequencies, psd = signal.welch(
            data,
            fs=self.sample_rate,
            window=self.window_function,
            nperseg=self.window_size,
            noverlap=int(self.window_size * self.overlap),
            scaling='density',
            return_onesided=True
        )

        # Convert to dB if requested
        if scale == 'dB':
            psd = 10 * np.log10(psd + 1e-10)

        return frequencies, psd

    def find_peaks(
        self,
        frequencies: np.ndarray,
        magnitude: np.ndarray,
        threshold: Optional[float] = None,
        min_distance: int = 5,
        n_peaks: Optional[int] = None
    ) -> List[Dict[str, float]]:
        """
        Find peaks in the magnitude spectrum.

        Args:
            frequencies: Frequency array
            magnitude: Magnitude array (linear scale)
            threshold: Minimum peak height (relative to maximum if < 1, absolute if >= 1)
            min_distance: Minimum distance between peaks (in samples)
            n_peaks: Maximum number of peaks to return (sorted by magnitude)

        Returns:
            List of dictionaries with peak information:
                - frequency: Peak frequency in Hz
                - magnitude: Peak magnitude
                - index: Peak index in arrays
        """
        if threshold is None:
            threshold = ProcessingDefaults.PEAK_DETECTION_THRESHOLD

        # Calculate absolute threshold
        if threshold < 1:
            # Relative threshold
            peak_threshold = threshold * np.max(magnitude)
        else:
            # Absolute threshold
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

    def update_averaged_psd(
        self,
        data: np.ndarray,
        reset: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update exponentially averaged PSD.

        This maintains a running average of the PSD for smoother display.

        Args:
            data: Input data (1D array)
            reset: If True, reset the average

        Returns:
            Tuple of (frequencies, averaged_psd)
        """
        frequencies, psd = self.compute_psd(data, scale='linear')

        if reset or self._averaged_psd is None:
            self._averaged_psd = psd.copy()
            self._n_averages = 1
        else:
            # Exponential averaging
            alpha = 0.2  # Averaging factor (0 = keep old, 1 = use new)
            self._averaged_psd = (1 - alpha) * self._averaged_psd + alpha * psd
            self._n_averages += 1

        return frequencies, self._averaged_psd

    def reset_averaging(self) -> None:
        """Reset PSD averaging."""
        self._averaged_psd = None
        self._n_averages = 0
        self.logger.debug("PSD averaging reset")

    def get_frequency_resolution(self) -> float:
        """
        Get frequency resolution (bin width).

        Returns:
            Frequency resolution in Hz
        """
        return self.sample_rate / self.window_size

    def get_nyquist_frequency(self) -> float:
        """
        Get Nyquist frequency.

        Returns:
            Nyquist frequency in Hz
        """
        return self.sample_rate / 2.0

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"FFTProcessor(fs={self.sample_rate} Hz, "
            f"window_size={self.window_size}, "
            f"window={self.window_function}, "
            f"freq_res={self.get_frequency_resolution():.2f} Hz)"
        )


class MultiChannelFFTProcessor:
    """
    FFT processor for multiple channels.

    This maintains separate FFT processors for each channel with
    independent averaging.
    """

    def __init__(
        self,
        n_channels: int,
        sample_rate: float,
        window_size: int = ProcessingDefaults.DEFAULT_FFT_WINDOW_SIZE,
        window_function: str = ProcessingDefaults.DEFAULT_WINDOW_FUNCTION,
        overlap: float = ProcessingDefaults.DEFAULT_FFT_OVERLAP
    ):
        """
        Initialize multi-channel FFT processor.

        Args:
            n_channels: Number of channels
            sample_rate: Sampling rate in Hz
            window_size: FFT window size
            window_function: Window function name
            overlap: Overlap fraction
        """
        self.n_channels = n_channels
        self.sample_rate = sample_rate

        # Create FFT processor (shared for all channels)
        self.fft_processor = FFTProcessor(
            sample_rate,
            window_size,
            window_function,
            overlap
        )

        self.logger = get_logger(__name__)
        self.logger.info(
            f"MultiChannelFFTProcessor created for {n_channels} channels"
        )
    
    @property
    def window_size(self):
        """Get FFT window size."""
        return self.fft_processor.window_size
    
    @property
    def window_function(self):
        """Get window function name."""
        return self.fft_processor.window_function
    
    @property
    def overlap(self):
        """Get overlap fraction."""
        return self.fft_processor.overlap

    def compute_magnitude_multi(
        self,
        data: np.ndarray,
        scale: str = 'linear'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute magnitude spectrum for all channels.

        Args:
            data: Input data of shape (n_channels, n_samples)
            scale: 'linear' or 'dB'

        Returns:
            Tuple of (frequencies, magnitudes)
                frequencies: 1D array
                magnitudes: 2D array of shape (n_channels, n_frequencies)
        """
        if data.shape[0] != self.n_channels:
            raise ValueError(
                f"Expected {self.n_channels} channels, got {data.shape[0]}"
            )

        # Compute for first channel to get frequency array
        frequencies, mag0 = self.fft_processor.compute_magnitude(data[0, :], scale)

        # Allocate output
        magnitudes = np.zeros((self.n_channels, len(frequencies)))
        magnitudes[0, :] = mag0

        # Compute for remaining channels
        for ch in range(1, self.n_channels):
            _, magnitudes[ch, :] = self.fft_processor.compute_magnitude(
                data[ch, :],
                scale
            )

        return frequencies, magnitudes

    def compute_psd_multi(
        self,
        data: np.ndarray,
        scale: str = 'linear'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute PSD for all channels.

        Args:
            data: Input data of shape (n_channels, n_samples)
            scale: 'linear' or 'dB'

        Returns:
            Tuple of (frequencies, psds)
                frequencies: 1D array
                psds: 2D array of shape (n_channels, n_frequencies)
        """
        if data.shape[0] != self.n_channels:
            raise ValueError(
                f"Expected {self.n_channels} channels, got {data.shape[0]}"
            )

        # Compute for first channel to get frequency array
        frequencies, psd0 = self.fft_processor.compute_psd(data[0, :], scale)

        # Allocate output
        psds = np.zeros((self.n_channels, len(frequencies)))
        psds[0, :] = psd0

        # Compute for remaining channels
        for ch in range(1, self.n_channels):
            _, psds[ch, :] = self.fft_processor.compute_psd(data[ch, :], scale)

        return frequencies, psds


# Example usage and tests
if __name__ == "__main__":
    print("FFT Processor Test")
    print("=" * 60)

    # Test parameters
    sample_rate = 25600  # Hz
    duration = 1.0  # seconds
    n_samples = int(sample_rate * duration)
    t = np.arange(n_samples) / sample_rate

    # Create test signal with known frequency components
    freq1, freq2, freq3 = 100, 500, 3000  # Hz
    signal_test = (
        1.0 * np.sin(2 * np.pi * freq1 * t) +
        0.5 * np.sin(2 * np.pi * freq2 * t) +
        0.3 * np.sin(2 * np.pi * freq3 * t) +
        0.05 * np.random.randn(n_samples)
    )

    print(f"\n1. Test signal created: {n_samples} samples @ {sample_rate} Hz")
    print(f"   Frequency components: {freq1} Hz, {freq2} Hz, {freq3} Hz + noise")

    # Create FFT processor
    print("\n2. Creating FFT processor...")
    fft_proc = FFTProcessor(
        sample_rate=sample_rate,
        window_size=2048,
        window_function="hann"
    )
    print(f"   {fft_proc}")
    print(f"   Frequency resolution: {fft_proc.get_frequency_resolution():.2f} Hz")
    print(f"   Nyquist frequency: {fft_proc.get_nyquist_frequency():.1f} Hz")

    # Compute magnitude spectrum
    print("\n3. Computing magnitude spectrum...")
    frequencies, magnitude = fft_proc.compute_magnitude(signal_test, scale='linear')
    print(f"   Frequencies shape: {frequencies.shape}")
    print(f"   Magnitude shape: {magnitude.shape}")
    print(f"   Frequency range: {frequencies[0]:.1f} to {frequencies[-1]:.1f} Hz")

    # Find peaks
    print("\n4. Finding peaks...")
    peaks = fft_proc.find_peaks(frequencies, magnitude, threshold=0.1, n_peaks=5)
    print(f"   Found {len(peaks)} peaks:")
    for i, peak in enumerate(peaks[:5]):
        print(f"     Peak {i+1}: {peak['frequency']:.1f} Hz, "
              f"magnitude={peak['magnitude']:.4f}")

    # Compute PSD
    print("\n5. Computing PSD...")
    frequencies_psd, psd_linear = fft_proc.compute_psd(signal_test, scale='linear')
    frequencies_psd, psd_db = fft_proc.compute_psd(signal_test, scale='dB')
    print(f"   PSD shape: {psd_linear.shape}")
    print(f"   PSD range (linear): {psd_linear.min():.2e} to {psd_linear.max():.2e}")
    print(f"   PSD range (dB): {psd_db.min():.1f} to {psd_db.max():.1f} dB")

    # Compute Welch PSD
    print("\n6. Computing Welch PSD...")
    frequencies_welch, psd_welch = fft_proc.compute_welch_psd(signal_test, scale='linear')
    print(f"   Welch PSD shape: {psd_welch.shape}")
    print(f"   Welch provides better noise reduction through averaging")

    # Test averaging
    print("\n7. Testing PSD averaging...")
    fft_proc.reset_averaging()
    for i in range(5):
        # Simulate new data chunks
        chunk = signal_test[i*1000:(i+1)*1000 + 2048]
        if len(chunk) >= 2048:
            freq_avg, psd_avg = fft_proc.update_averaged_psd(chunk)
            print(f"   Average {i+1}: averaged over {fft_proc._n_averages} frames")

    # Test multi-channel FFT
    print("\n8. Testing multi-channel FFT...")
    n_channels = 4
    data_multi = np.array([signal_test[:10000]] * n_channels)
    print(f"   Multi-channel data shape: {data_multi.shape}")

    multi_fft = MultiChannelFFTProcessor(
        n_channels=n_channels,
        sample_rate=sample_rate,
        window_size=2048
    )

    frequencies_multi, magnitudes_multi = multi_fft.compute_magnitude_multi(
        data_multi,
        scale='linear'
    )
    print(f"   Frequencies shape: {frequencies_multi.shape}")
    print(f"   Magnitudes shape: {magnitudes_multi.shape}")

    print("\n" + "=" * 60)
    print("All FFT tests completed!")

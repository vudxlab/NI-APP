"""
Digital filter implementations for signal processing.

This module provides digital filter design and application using SciPy,
with support for multiple filter types and real-time filtering with state.
"""

import numpy as np
from scipy import signal
from typing import Union, Tuple, Optional, Dict
import warnings

from ..utils.logger import get_logger
from ..utils.constants import ProcessingDefaults
from ..utils.validators import (
    validate_filter_type,
    validate_filter_mode,
    validate_cutoff_frequency,
    validate_filter_order,
    ValidationError
)


class FilterDesignError(Exception):
    """Exception raised for filter design errors."""
    pass


class DigitalFilter:
    """
    Digital filter with state management for continuous filtering.

    This class encapsulates a digital filter designed using SciPy,
    storing the filter coefficients and state for real-time application.
    """

    def __init__(
        self,
        filter_type: str,
        filter_mode: str,
        cutoff: Union[float, Tuple[float, float]],
        sample_rate: float,
        order: int = 4
    ):
        """
        Initialize and design a digital filter.

        Args:
            filter_type: Filter type ("butterworth", "chebyshev1", "chebyshev2", "bessel")
            filter_mode: Filter mode ("lowpass", "highpass", "bandpass", "bandstop")
            cutoff: Cutoff frequency in Hz, or tuple (low, high) for bandpass/bandstop
            sample_rate: Sampling rate in Hz
            order: Filter order

        Raises:
            ValidationError: If parameters are invalid
            FilterDesignError: If filter design fails
        """
        self.logger = get_logger(__name__)

        # Validate parameters
        validate_filter_type(filter_type)
        validate_filter_mode(filter_mode)
        validate_filter_order(order)
        validate_cutoff_frequency(cutoff, sample_rate, filter_mode)

        self.filter_type = filter_type
        self.filter_mode = filter_mode
        self.cutoff = cutoff
        self.sample_rate = sample_rate
        self.order = order

        # Design the filter
        self.sos = None  # Second-order sections
        self.zi_prototype = None  # Initial state prototype

        self._design_filter()

        self.logger.info(
            f"Designed {filter_type} {filter_mode} filter, "
            f"order={order}, cutoff={cutoff} Hz"
        )

    def _design_filter(self) -> None:
        """
        Design the digital filter using SciPy.

        Raises:
            FilterDesignError: If filter design fails
        """
        try:
            # Normalize cutoff to Nyquist frequency
            nyq = self.sample_rate / 2.0

            if isinstance(self.cutoff, (list, tuple)):
                Wn = [f / nyq for f in self.cutoff]
            else:
                Wn = self.cutoff / nyq

            # Select filter design function
            if self.filter_type == ProcessingDefaults.FILTER_TYPE_BUTTERWORTH:
                self.sos = signal.butter(
                    self.order,
                    Wn,
                    btype=self.filter_mode,
                    output='sos'
                )
            elif self.filter_type == ProcessingDefaults.FILTER_TYPE_CHEBYSHEV1:
                # Chebyshev Type I: ripple in passband
                self.sos = signal.cheby1(
                    self.order,
                    0.5,  # 0.5 dB ripple
                    Wn,
                    btype=self.filter_mode,
                    output='sos'
                )
            elif self.filter_type == ProcessingDefaults.FILTER_TYPE_CHEBYSHEV2:
                # Chebyshev Type II: ripple in stopband
                self.sos = signal.cheby2(
                    self.order,
                    40,  # 40 dB attenuation in stopband
                    Wn,
                    btype=self.filter_mode,
                    output='sos'
                )
            elif self.filter_type == ProcessingDefaults.FILTER_TYPE_BESSEL:
                self.sos = signal.bessel(
                    self.order,
                    Wn,
                    btype=self.filter_mode,
                    output='sos'
                )

            # Initialize state for the filter
            # State shape: (n_sections, 2) for each second-order section
            self.zi_prototype = signal.sosfilt_zi(self.sos)

        except Exception as e:
            self.logger.error(f"Filter design failed: {e}")
            raise FilterDesignError(f"Failed to design filter: {e}")

    def apply(self, data: np.ndarray, zi: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply filter to data.

        Args:
            data: Input data (1D array)
            zi: Filter state from previous call (optional)

        Returns:
            Tuple of (filtered_data, final_state)
        """
        if zi is None:
            # Initialize state with appropriate initial conditions
            zi = self.zi_prototype * data[0]

        filtered, zf = signal.sosfilt(self.sos, data, zi=zi)

        return filtered, zf

    def get_frequency_response(self, n_points: int = 1024) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute frequency response of the filter.

        Args:
            n_points: Number of frequency points

        Returns:
            Tuple of (frequencies, magnitude_response_dB)
        """
        w, h = signal.sosfreqz(self.sos, worN=n_points, fs=self.sample_rate)
        magnitude_db = 20 * np.log10(np.abs(h) + 1e-10)

        return w, magnitude_db

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"DigitalFilter({self.filter_type}, {self.filter_mode}, "
            f"order={self.order}, cutoff={self.cutoff} Hz)"
        )


class FilterBank:
    """
    Manages filters for multiple channels.

    This class maintains separate filter states for each channel,
    allowing real-time continuous filtering of multi-channel data.
    """

    def __init__(self, n_channels: int):
        """
        Initialize filter bank.

        Args:
            n_channels: Number of channels
        """
        self.n_channels = n_channels
        self.filter: Optional[DigitalFilter] = None
        self.zi: Optional[np.ndarray] = None  # State for each channel
        self.enabled = False

        self.logger = get_logger(__name__)

    def design_filter(
        self,
        filter_type: str,
        filter_mode: str,
        cutoff: Union[float, Tuple[float, float]],
        sample_rate: float,
        order: int = 4
    ) -> None:
        """
        Design filter for the filter bank.

        Args:
            filter_type: Filter type
            filter_mode: Filter mode
            cutoff: Cutoff frequency (Hz)
            sample_rate: Sampling rate (Hz)
            order: Filter order

        Raises:
            FilterDesignError: If filter design fails
        """
        try:
            # Design the filter
            self.filter = DigitalFilter(
                filter_type,
                filter_mode,
                cutoff,
                sample_rate,
                order
            )

            # Initialize state for each channel
            # Shape: (n_channels, n_sections, 2)
            n_sections = self.filter.sos.shape[0]
            self.zi = np.zeros((self.n_channels, n_sections, 2))

            self.enabled = True

            self.logger.info(
                f"FilterBank configured for {self.n_channels} channels: "
                f"{filter_type} {filter_mode}, order={order}"
            )

        except Exception as e:
            self.logger.error(f"FilterBank design failed: {e}")
            raise

    def apply_filter(self, data: np.ndarray) -> np.ndarray:
        """
        Apply filter to multi-channel data with state preservation.

        Args:
            data: Input data of shape (n_channels, n_samples)

        Returns:
            Filtered data with same shape

        Raises:
            ValueError: If data shape is incompatible
        """
        if not self.enabled or self.filter is None:
            # No filter configured, return data unchanged
            return data

        if data.shape[0] != self.n_channels:
            raise ValueError(
                f"Expected {self.n_channels} channels, got {data.shape[0]}"
            )

        filtered = np.zeros_like(data)

        for ch in range(self.n_channels):
            # Initialize state on first use
            if np.all(self.zi[ch] == 0):
                self.zi[ch] = self.filter.zi_prototype * data[ch, 0]

            # Apply filter and update state
            filtered[ch, :], self.zi[ch] = self.filter.apply(
                data[ch, :],
                self.zi[ch]
            )

        return filtered

    def reset_state(self) -> None:
        """Reset filter state for all channels."""
        if self.zi is not None:
            self.zi.fill(0.0)
            self.logger.debug("Filter state reset")

    def disable(self) -> None:
        """Disable filtering (data passes through unchanged)."""
        self.enabled = False
        self.logger.info("FilterBank disabled")

    def enable(self) -> None:
        """Enable filtering."""
        if self.filter is not None:
            self.enabled = True
            self.logger.info("FilterBank enabled")
        else:
            self.logger.warning("Cannot enable FilterBank: no filter designed")

    def get_frequency_response(self, n_points: int = 1024) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Get frequency response of the filter.

        Args:
            n_points: Number of frequency points

        Returns:
            Tuple of (frequencies, magnitude_dB) or None if no filter
        """
        if self.filter is None:
            return None

        return self.filter.get_frequency_response(n_points)

    def __repr__(self) -> str:
        """String representation."""
        if self.filter:
            return f"FilterBank({self.n_channels} channels, filter={self.filter}, enabled={self.enabled})"
        else:
            return f"FilterBank({self.n_channels} channels, no filter)"


def apply_zero_phase_filter(
    data: np.ndarray,
    filter_type: str,
    filter_mode: str,
    cutoff: Union[float, Tuple[float, float]],
    sample_rate: float,
    order: int = 4
) -> np.ndarray:
    """
    Apply zero-phase filter to data (offline processing).

    This uses filtfilt which applies the filter forward and backward,
    resulting in zero phase distortion but doubling the filter order.

    Args:
        data: Input data of shape (n_channels, n_samples) or (n_samples,)
        filter_type: Filter type
        filter_mode: Filter mode
        cutoff: Cutoff frequency (Hz)
        sample_rate: Sampling rate (Hz)
        order: Filter order

    Returns:
        Filtered data with same shape

    Raises:
        FilterDesignError: If filter design fails
    """
    # Design filter
    filt = DigitalFilter(filter_type, filter_mode, cutoff, sample_rate, order)

    # Apply zero-phase filtering
    if data.ndim == 1:
        return signal.sosfiltfilt(filt.sos, data)
    else:
        # Multi-channel
        filtered = np.zeros_like(data)
        for ch in range(data.shape[0]):
            filtered[ch, :] = signal.sosfiltfilt(filt.sos, data[ch, :])
        return filtered


# Example usage and tests
if __name__ == "__main__":
    print("Digital Filter Test")
    print("=" * 60)

    # Test parameters
    sample_rate = 25600  # Hz
    duration = 1.0  # seconds
    n_samples = int(sample_rate * duration)
    t = np.arange(n_samples) / sample_rate

    # Create test signal: 100 Hz + 500 Hz + 5000 Hz + noise
    signal_test = (
        np.sin(2 * np.pi * 100 * t) +
        0.5 * np.sin(2 * np.pi * 500 * t) +
        0.3 * np.sin(2 * np.pi * 5000 * t) +
        0.1 * np.random.randn(n_samples)
    )

    print(f"\n1. Test signal created: {n_samples} samples @ {sample_rate} Hz")
    print(f"   Signal contains: 100 Hz, 500 Hz, 5000 Hz components + noise")

    # Test lowpass filter
    print("\n2. Testing lowpass filter (1000 Hz cutoff)...")
    try:
        lp_filter = DigitalFilter(
            "butterworth",
            "lowpass",
            cutoff=1000,
            sample_rate=sample_rate,
            order=4
        )
        print(f"   Filter designed: {lp_filter}")

        filtered_lp, _ = lp_filter.apply(signal_test)
        print(f"   Filtered signal shape: {filtered_lp.shape}")

        # Get frequency response
        freq, mag = lp_filter.get_frequency_response()
        cutoff_idx = np.argmin(np.abs(freq - 1000))
        print(f"   Magnitude at cutoff (1000 Hz): {mag[cutoff_idx]:.2f} dB")

    except Exception as e:
        print(f"   Error: {e}")

    # Test bandpass filter
    print("\n3. Testing bandpass filter (200-2000 Hz)...")
    try:
        bp_filter = DigitalFilter(
            "butterworth",
            "bandpass",
            cutoff=(200, 2000),
            sample_rate=sample_rate,
            order=4
        )
        print(f"   Filter designed: {bp_filter}")

        filtered_bp, _ = bp_filter.apply(signal_test)
        print(f"   Filtered signal shape: {filtered_bp.shape}")

    except Exception as e:
        print(f"   Error: {e}")

    # Test FilterBank with multiple channels
    print("\n4. Testing FilterBank with 4 channels...")
    try:
        # Create multi-channel signal
        data_multi = np.array([signal_test] * 4)
        print(f"   Multi-channel data shape: {data_multi.shape}")

        # Create and configure filter bank
        filter_bank = FilterBank(n_channels=4)
        filter_bank.design_filter(
            "butterworth",
            "highpass",
            cutoff=50,
            sample_rate=sample_rate,
            order=4
        )
        print(f"   {filter_bank}")

        # Apply filter
        filtered_multi = filter_bank.apply_filter(data_multi)
        print(f"   Filtered data shape: {filtered_multi.shape}")

        # Test continuous filtering (simulate streaming)
        print("\n5. Testing continuous filtering with state...")
        filter_bank.reset_state()

        chunk_size = 1000
        n_chunks = 5

        for i in range(n_chunks):
            start = i * chunk_size
            end = start + chunk_size
            chunk = data_multi[:, start:end]

            filtered_chunk = filter_bank.apply_filter(chunk)
            print(f"   Chunk {i+1}: input shape {chunk.shape}, "
                  f"output shape {filtered_chunk.shape}")

    except Exception as e:
        print(f"   Error: {e}")

    # Test zero-phase filtering
    print("\n6. Testing zero-phase filter...")
    try:
        filtered_zp = apply_zero_phase_filter(
            signal_test,
            "butterworth",
            "lowpass",
            cutoff=1000,
            sample_rate=sample_rate,
            order=4
        )
        print(f"   Zero-phase filtered shape: {filtered_zp.shape}")
        print(f"   Zero-phase filtering introduces no phase delay")

    except Exception as e:
        print(f"   Error: {e}")

    print("\n" + "=" * 60)
    print("All filter tests completed!")

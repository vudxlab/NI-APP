"""
Signal Processor - Coordinator for data processing pipeline.

This module coordinates the signal processing pipeline, managing the
data buffer, filters, and FFT processor. It provides a high-level
interface for real-time signal processing.
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any
from PyQt5.QtCore import QObject, pyqtSignal
import time

from .data_buffer import DataBuffer
from .filters import FilterBank
from .fft_processor import MultiChannelFFTProcessor
from ..utils.logger import get_logger
from ..utils.constants import DAQDefaults, ProcessingDefaults


class SignalProcessor(QObject):
    """
    Signal processor coordinator.

    This class manages the complete signal processing pipeline:
    - Raw data buffering
    - Optional digital filtering
    - FFT computation
    - Data distribution to GUI

    Signals:
        raw_data_ready: Emitted when new raw data is available
        filtered_data_ready: Emitted when new filtered data is available
        fft_data_ready: Emitted when new FFT data is available
    """

    # Qt signals for thread-safe communication
    raw_data_ready = pyqtSignal(np.ndarray, float)  # data, timestamp
    filtered_data_ready = pyqtSignal(np.ndarray, float)  # data, timestamp
    fft_data_ready = pyqtSignal(np.ndarray, np.ndarray, int)  # frequencies, magnitudes, channel

    def __init__(
        self,
        n_channels: int,
        sample_rate: float,
        buffer_duration: int = DAQDefaults.BUFFER_DURATION_SECONDS
    ):
        """
        Initialize signal processor.

        Args:
            n_channels: Number of channels
            sample_rate: Sampling rate in Hz
            buffer_duration: Buffer duration in seconds
        """
        super().__init__()

        self.logger = get_logger(__name__)

        self.n_channels = n_channels
        self.sample_rate = sample_rate
        self.buffer_duration = buffer_duration

        # Calculate buffer size
        buffer_size = int(sample_rate * buffer_duration)

        # Create data buffers
        self.raw_buffer = DataBuffer(n_channels, buffer_size)
        self.filtered_buffer = DataBuffer(n_channels, buffer_size)

        # Create filter bank
        self.filter_bank = FilterBank(n_channels)

        # Create FFT processor
        self.fft_processor = MultiChannelFFTProcessor(
            n_channels=n_channels,
            sample_rate=sample_rate,
            window_size=ProcessingDefaults.DEFAULT_FFT_WINDOW_SIZE,
            window_function=ProcessingDefaults.DEFAULT_WINDOW_FUNCTION,
            overlap=ProcessingDefaults.DEFAULT_FFT_OVERLAP
        )

        # Processing flags
        self.filtering_enabled = False
        self.fft_enabled = True

        # Statistics
        self._total_processed = 0
        self._last_process_time = 0

        self.logger.info(
            f"SignalProcessor initialized: {n_channels} channels @ {sample_rate} Hz, "
            f"buffer={buffer_duration}s ({buffer_size} samples)"
        )

    def process_data(self, data: np.ndarray, timestamp: Optional[float] = None) -> None:
        """
        Process incoming data through the pipeline.

        This is the main entry point for data processing. It:
        1. Stores raw data in buffer
        2. Applies filtering if enabled
        3. Stores filtered data in buffer
        4. Computes FFT if enabled
        5. Emits signals for GUI updates

        Args:
            data: Raw data array of shape (n_channels, n_samples)
            timestamp: Data timestamp (seconds since epoch)
        """
        start_time = time.time()

        if timestamp is None:
            timestamp = start_time

        # Validate data shape
        if data.shape[0] != self.n_channels:
            self.logger.error(
                f"Data shape mismatch: expected {self.n_channels} channels, "
                f"got {data.shape[0]}"
            )
            return

        # Store raw data
        self.raw_buffer.append(data, timestamp)

        # Emit raw data signal
        self.raw_data_ready.emit(data, timestamp)

        # Apply filtering if enabled
        if self.filtering_enabled:
            filtered_data = self.filter_bank.apply_filter(data)
        else:
            filtered_data = data

        # Store filtered data
        self.filtered_buffer.append(filtered_data, timestamp)

        # Emit filtered data signal
        self.filtered_data_ready.emit(filtered_data, timestamp)

        # Compute FFT if enabled
        if self.fft_enabled:
            self._compute_fft(filtered_data)

        # Update statistics
        self._total_processed += data.shape[1]
        self._last_process_time = time.time() - start_time

    def _compute_fft(self, data: np.ndarray) -> None:
        """
        Compute FFT for all channels.

        Args:
            data: Data array of shape (n_channels, n_samples)
        """
        try:
            # Check if we have enough data
            if data.shape[1] < self.fft_processor.window_size:
                self.logger.debug(
                    f"Not enough data for FFT: {data.shape[1]} < {self.fft_processor.window_size}"
                )
                return

            # Compute FFT for all channels
            frequencies, magnitudes = self.fft_processor.compute_magnitude_multi(
                data,
                scale='dB'  # Use dB scale for better visualization
            )

            # Emit FFT data for each channel
            # (GUI can choose which channels to display)
            for ch in range(self.n_channels):
                self.fft_data_ready.emit(frequencies, magnitudes[ch, :], ch)
            
            self.logger.debug(f"FFT emitted: {len(frequencies)} bins, {self.n_channels} channels")

        except Exception as e:
            self.logger.error(f"FFT computation failed: {e}")

    def configure_filter(
        self,
        filter_type: str,
        filter_mode: str,
        cutoff: float or Tuple[float, float],
        order: int = 4,
        enabled: bool = True
    ) -> None:
        """
        Configure digital filter.

        Args:
            filter_type: Filter type ("butterworth", "chebyshev1", etc.)
            filter_mode: Filter mode ("lowpass", "highpass", "bandpass", "bandstop")
            cutoff: Cutoff frequency in Hz (or tuple for bandpass/bandstop)
            order: Filter order
            enabled: Enable filtering immediately

        Raises:
            Exception: If filter configuration fails
        """
        try:
            self.filter_bank.design_filter(
                filter_type=filter_type,
                filter_mode=filter_mode,
                cutoff=cutoff,
                sample_rate=self.sample_rate,
                order=order
            )

            if enabled:
                self.filter_bank.enable()
                self.filtering_enabled = True
            else:
                self.filter_bank.disable()
                self.filtering_enabled = False

            self.logger.info(
                f"Filter configured: {filter_type} {filter_mode}, "
                f"cutoff={cutoff} Hz, order={order}, enabled={enabled}"
            )

        except Exception as e:
            self.logger.error(f"Filter configuration failed: {e}")
            raise

    def enable_filtering(self, enabled: bool = True) -> None:
        """
        Enable or disable filtering.

        Args:
            enabled: True to enable filtering
        """
        if enabled:
            self.filter_bank.enable()
            self.filtering_enabled = True
            self.logger.info("Filtering enabled")
        else:
            self.filter_bank.disable()
            self.filtering_enabled = False
            self.logger.info("Filtering disabled")

    def reset_filter_state(self) -> None:
        """Reset filter state (useful when changing configurations)."""
        self.filter_bank.reset_state()
        self.logger.debug("Filter state reset")

    def configure_fft(
        self,
        window_size: Optional[int] = None,
        window_function: Optional[str] = None,
        overlap: Optional[float] = None
    ) -> None:
        """
        Reconfigure FFT processor.

        Args:
            window_size: FFT window size (None = keep current)
            window_function: Window function name (None = keep current)
            overlap: Overlap fraction (None = keep current)
        """
        # Get current settings
        current_window_size = self.fft_processor.window_size
        current_window_func = self.fft_processor.window_function
        current_overlap = self.fft_processor.overlap

        # Use new values or keep current
        new_window_size = window_size if window_size is not None else current_window_size
        new_window_func = window_function if window_function is not None else current_window_func
        new_overlap = overlap if overlap is not None else current_overlap

        # Recreate FFT processor
        self.fft_processor = MultiChannelFFTProcessor(
            n_channels=self.n_channels,
            sample_rate=self.sample_rate,
            window_size=new_window_size,
            window_function=new_window_func,
            overlap=new_overlap
        )

        self.logger.info(
            f"FFT reconfigured: window_size={new_window_size}, "
            f"window={new_window_func}, overlap={new_overlap}"
        )

    def enable_fft(self, enabled: bool = True) -> None:
        """
        Enable or disable FFT computation.

        Args:
            enabled: True to enable FFT
        """
        self.fft_enabled = enabled
        status = "enabled" if enabled else "disabled"
        self.logger.info(f"FFT {status}")

    def get_raw_data(self, n_samples: Optional[int] = None) -> np.ndarray:
        """
        Get raw data from buffer.

        Args:
            n_samples: Number of samples to retrieve (None = all available)

        Returns:
            Raw data array of shape (n_channels, n_samples)
        """
        if n_samples is None:
            return self.raw_buffer.get_all()
        else:
            return self.raw_buffer.get_latest(n_samples)

    def get_filtered_data(self, n_samples: Optional[int] = None) -> np.ndarray:
        """
        Get filtered data from buffer.

        Args:
            n_samples: Number of samples to retrieve (None = all available)

        Returns:
            Filtered data array of shape (n_channels, n_samples)
        """
        if n_samples is None:
            return self.filtered_buffer.get_all()
        else:
            return self.filtered_buffer.get_latest(n_samples)

    def get_channel_data(
        self,
        channel_idx: int,
        n_samples: Optional[int] = None,
        filtered: bool = False
    ) -> np.ndarray:
        """
        Get data for a specific channel.

        Args:
            channel_idx: Channel index
            n_samples: Number of samples (None = all available)
            filtered: If True, return filtered data; otherwise raw data

        Returns:
            Data array of shape (n_samples,)
        """
        if filtered:
            return self.filtered_buffer.get_channel(channel_idx, n_samples)
        else:
            return self.raw_buffer.get_channel(channel_idx, n_samples)

    def clear_buffers(self) -> None:
        """Clear all data buffers."""
        self.raw_buffer.clear()
        self.filtered_buffer.clear()
        self.reset_filter_state()
        self._total_processed = 0
        self.logger.info("Buffers cleared")

    def get_status(self) -> Dict[str, Any]:
        """
        Get current processing status.

        Returns:
            Dictionary with status information
        """
        raw_stats = self.raw_buffer.get_stats()
        filtered_stats = self.filtered_buffer.get_stats()

        status = {
            'n_channels': self.n_channels,
            'sample_rate': self.sample_rate,
            'buffer_duration': self.buffer_duration,
            'filtering_enabled': self.filtering_enabled,
            'fft_enabled': self.fft_enabled,
            'total_processed': self._total_processed,
            'last_process_time_ms': self._last_process_time * 1000,
            'raw_buffer': {
                'samples_available': raw_stats['samples_available'],
                'fill_percentage': raw_stats['fill_percentage'],
            },
            'filtered_buffer': {
                'samples_available': filtered_stats['samples_available'],
                'fill_percentage': filtered_stats['fill_percentage'],
            }
        }

        if self.filtering_enabled and self.filter_bank.filter:
            status['filter'] = {
                'type': self.filter_bank.filter.filter_type,
                'mode': self.filter_bank.filter.filter_mode,
                'cutoff': self.filter_bank.filter.cutoff,
                'order': self.filter_bank.filter.order
            }

        status['fft'] = {
            'window_size': self.fft_processor.fft_processor.window_size,
            'window_function': self.fft_processor.fft_processor.window_function,
            'frequency_resolution': self.fft_processor.fft_processor.get_frequency_resolution()
        }

        return status

    def get_memory_usage(self) -> int:
        """
        Get total memory usage in bytes.

        Returns:
            Memory usage in bytes
        """
        return (
            self.raw_buffer.get_memory_usage() +
            self.filtered_buffer.get_memory_usage()
        )

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"SignalProcessor(channels={self.n_channels}, "
            f"fs={self.sample_rate} Hz, "
            f"filtering={self.filtering_enabled}, "
            f"fft={self.fft_enabled})"
        )


# Example usage and tests
if __name__ == "__main__":
    from PyQt5.QtCore import QCoreApplication
    import sys

    print("SignalProcessor Test")
    print("=" * 60)

    # Create Qt application (needed for signals)
    app = QCoreApplication(sys.argv)

    # Test parameters
    n_channels = 4
    sample_rate = 25600  # Hz
    buffer_duration = 10  # seconds

    # Create signal processor
    print(f"\n1. Creating signal processor...")
    processor = SignalProcessor(
        n_channels=n_channels,
        sample_rate=sample_rate,
        buffer_duration=buffer_duration
    )
    print(f"   {processor}")
    print(f"   Memory usage: {processor.get_memory_usage() / 1024 / 1024:.2f} MB")

    # Configure filter
    print(f"\n2. Configuring lowpass filter...")
    processor.configure_filter(
        filter_type="butterworth",
        filter_mode="lowpass",
        cutoff=1000,
        order=4,
        enabled=True
    )

    # Configure FFT
    print(f"\n3. Configuring FFT...")
    processor.configure_fft(
        window_size=2048,
        window_function="hann",
        overlap=0.5
    )

    # Connect signals
    def on_raw_data(data, timestamp):
        print(f"   Raw data received: shape={data.shape}, timestamp={timestamp:.3f}")

    def on_filtered_data(data, timestamp):
        print(f"   Filtered data received: shape={data.shape}")

    def on_fft_data(frequencies, magnitude, channel):
        print(f"   FFT data for channel {channel}: {len(frequencies)} frequencies")

    processor.raw_data_ready.connect(on_raw_data)
    processor.filtered_data_ready.connect(on_filtered_data)
    processor.fft_data_ready.connect(on_fft_data)

    # Process some test data
    print(f"\n4. Processing test data...")
    for i in range(3):
        # Generate test signal
        n_samples = 1000
        t = np.arange(n_samples) / sample_rate
        data = np.zeros((n_channels, n_samples))

        for ch in range(n_channels):
            freq = 100 + ch * 200  # Different frequency for each channel
            data[ch, :] = np.sin(2 * np.pi * freq * t) + 0.1 * np.random.randn(n_samples)

        # Process data
        processor.process_data(data)
        print(f"\n   Iteration {i+1} completed")

    # Get status
    print(f"\n5. Processor status:")
    status = processor.get_status()
    for key, value in status.items():
        if isinstance(value, dict):
            print(f"   {key}:")
            for k, v in value.items():
                print(f"     {k}: {v}")
        elif isinstance(value, float):
            print(f"   {key}: {value:.4f}")
        else:
            print(f"   {key}: {value}")

    # Test data retrieval
    print(f"\n6. Testing data retrieval...")
    raw_data = processor.get_raw_data(500)
    print(f"   Raw data (500 samples): {raw_data.shape}")

    filtered_data = processor.get_filtered_data(500)
    print(f"   Filtered data (500 samples): {filtered_data.shape}")

    channel_0_data = processor.get_channel_data(0, 500, filtered=True)
    print(f"   Channel 0 data (filtered): {channel_0_data.shape}")

    print("\n" + "=" * 60)
    print("All signal processor tests completed!")

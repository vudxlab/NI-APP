"""
Thread-safe circular buffer for real-time data storage.

This module implements a circular (ring) buffer for storing continuous
acquisition data with thread-safe read/write operations.
"""

import numpy as np
from threading import Lock
from typing import Optional, Tuple
import time

from ..utils.logger import get_logger


class DataBuffer:
    """
    Thread-safe circular buffer for multi-channel data.

    This buffer stores the most recent N samples of data in a circular
    fashion, overwriting old data when full. All operations are thread-safe.

    Attributes:
        n_channels: Number of data channels
        buffer_size: Maximum number of samples per channel
    """

    def __init__(self, n_channels: int, buffer_size: int):
        """
        Initialize the circular buffer.

        Args:
            n_channels: Number of channels
            buffer_size: Maximum number of samples to store per channel

        Raises:
            ValueError: If parameters are invalid
        """
        if n_channels <= 0:
            raise ValueError(f"n_channels must be positive, got {n_channels}")
        if buffer_size <= 0:
            raise ValueError(f"buffer_size must be positive, got {buffer_size}")

        self.n_channels = n_channels
        self.buffer_size = buffer_size

        # Circular buffer storage (channels × samples)
        self._buffer = np.zeros((n_channels, buffer_size), dtype=np.float64)

        # Write position (index of next write)
        self._write_idx = 0

        # Total number of samples written (for tracking)
        self._total_written = 0

        # Thread safety
        self._lock = Lock()

        # Timestamps (optional, for tracking data timing)
        self._timestamps = np.zeros(buffer_size, dtype=np.float64)
        self._last_timestamp = 0.0

        self.logger = get_logger(__name__)
        self.logger.info(
            f"DataBuffer created: {n_channels} channels, "
            f"{buffer_size} samples ({buffer_size * n_channels * 8 / 1024 / 1024:.2f} MB)"
        )

    def append(self, data: np.ndarray, timestamp: Optional[float] = None) -> None:
        """
        Append new data to the buffer.

        Data wraps around when buffer is full, overwriting oldest data.

        Args:
            data: Data array of shape (n_channels, n_samples) or (n_samples,) for single channel
            timestamp: Optional timestamp for the first sample (seconds since epoch)

        Raises:
            ValueError: If data shape is incompatible
        """
        # Handle single channel case
        if data.ndim == 1:
            if self.n_channels != 1:
                raise ValueError(
                    f"Expected {self.n_channels} channels, got single channel data"
                )
            data = data.reshape(1, -1)

        if data.shape[0] != self.n_channels:
            raise ValueError(
                f"Expected {self.n_channels} channels, got {data.shape[0]}"
            )

        n_samples = data.shape[1]

        if timestamp is None:
            timestamp = time.time()

        with self._lock:
            # Handle wrap-around
            if self._write_idx + n_samples <= self.buffer_size:
                # No wrap-around needed
                self._buffer[:, self._write_idx:self._write_idx + n_samples] = data
                self._timestamps[self._write_idx:self._write_idx + n_samples] = (
                    timestamp + np.arange(n_samples) / self.buffer_size  # Approximate
                )
            else:
                # Split write at boundary
                first_part = self.buffer_size - self._write_idx
                second_part = n_samples - first_part

                # Write first part (to end of buffer)
                self._buffer[:, self._write_idx:] = data[:, :first_part]
                self._timestamps[self._write_idx:] = (
                    timestamp + np.arange(first_part) / self.buffer_size
                )

                # Write second part (from start of buffer)
                self._buffer[:, :second_part] = data[:, first_part:]
                self._timestamps[:second_part] = (
                    timestamp + (first_part + np.arange(second_part)) / self.buffer_size
                )

            # Update write index (circular)
            self._write_idx = (self._write_idx + n_samples) % self.buffer_size
            self._total_written += n_samples
            self._last_timestamp = timestamp

    def get_latest(self, n_samples: int) -> np.ndarray:
        """
        Get the latest n_samples from the buffer.

        Args:
            n_samples: Number of samples to retrieve

        Returns:
            Data array of shape (n_channels, n_samples)

        Raises:
            ValueError: If n_samples is invalid
        """
        if n_samples <= 0:
            raise ValueError(f"n_samples must be positive, got {n_samples}")

        # Limit to buffer size
        n_samples = min(n_samples, self.buffer_size)

        # Also limit to total written
        n_samples = min(n_samples, self._total_written)

        if n_samples == 0:
            return np.zeros((self.n_channels, 0))

        with self._lock:
            # Calculate start index (going backwards from write position)
            start_idx = (self._write_idx - n_samples) % self.buffer_size

            if start_idx < self._write_idx:
                # No wrap-around
                return self._buffer[:, start_idx:self._write_idx].copy()
            else:
                # Wrap-around case: need to concatenate
                return np.concatenate([
                    self._buffer[:, start_idx:],
                    self._buffer[:, :self._write_idx]
                ], axis=1)

    def get_range(self, start_sample: int, end_sample: int) -> np.ndarray:
        """
        Get a specific range of samples from the buffer.

        Indices are relative to the oldest available data (0 = oldest).

        Args:
            start_sample: Starting sample index (inclusive)
            end_sample: Ending sample index (exclusive)

        Returns:
            Data array of shape (n_channels, end_sample - start_sample)

        Raises:
            ValueError: If indices are invalid
        """
        if start_sample < 0 or end_sample <= start_sample:
            raise ValueError(f"Invalid range: [{start_sample}, {end_sample})")

        n_available = min(self._total_written, self.buffer_size)

        if start_sample >= n_available:
            raise ValueError(
                f"start_sample {start_sample} exceeds available data {n_available}"
            )

        # Clip end_sample to available data
        end_sample = min(end_sample, n_available)
        n_samples = end_sample - start_sample

        with self._lock:
            # Calculate actual indices in circular buffer
            if self._total_written < self.buffer_size:
                # Buffer not full yet, data is at beginning
                return self._buffer[:, start_sample:end_sample].copy()
            else:
                # Buffer is full, calculate position relative to write index
                # Oldest data is at write_idx, newest is at write_idx - 1
                actual_start = (self._write_idx + start_sample) % self.buffer_size
                actual_end = (self._write_idx + end_sample) % self.buffer_size

                if actual_start < actual_end:
                    # No wrap-around
                    return self._buffer[:, actual_start:actual_end].copy()
                else:
                    # Wrap-around
                    return np.concatenate([
                        self._buffer[:, actual_start:],
                        self._buffer[:, :actual_end]
                    ], axis=1)

    def get_all(self) -> np.ndarray:
        """
        Get all available data from the buffer.

        Returns:
            Data array of shape (n_channels, n_available_samples)
        """
        n_available = min(self._total_written, self.buffer_size)
        return self.get_latest(n_available)

    def get_channel(self, channel_idx: int, n_samples: Optional[int] = None) -> np.ndarray:
        """
        Get data for a specific channel.

        Args:
            channel_idx: Channel index (0-based)
            n_samples: Number of samples to retrieve (None = all available)

        Returns:
            Data array of shape (n_samples,)

        Raises:
            ValueError: If channel_idx is invalid
        """
        if channel_idx < 0 or channel_idx >= self.n_channels:
            raise ValueError(
                f"channel_idx {channel_idx} out of range [0, {self.n_channels})"
            )

        if n_samples is None:
            data = self.get_all()
        else:
            data = self.get_latest(n_samples)

        return data[channel_idx, :]

    def clear(self) -> None:
        """Clear the buffer and reset counters."""
        with self._lock:
            self._buffer.fill(0.0)
            self._timestamps.fill(0.0)
            self._write_idx = 0
            self._total_written = 0
            self._last_timestamp = 0.0
            self.logger.debug("Buffer cleared")

    def get_stats(self) -> dict:
        """
        Get buffer statistics.

        Returns:
            Dictionary with buffer statistics
        """
        with self._lock:
            n_available = min(self._total_written, self.buffer_size)
            fill_percentage = (n_available / self.buffer_size) * 100

            stats = {
                'n_channels': self.n_channels,
                'buffer_size': self.buffer_size,
                'samples_written': self._total_written,
                'samples_available': n_available,
                'fill_percentage': fill_percentage,
                'is_full': self._total_written >= self.buffer_size,
                'write_index': self._write_idx,
                'last_timestamp': self._last_timestamp
            }

            if n_available > 0:
                all_data = self.get_all()
                stats.update({
                    'min_value': float(np.min(all_data)),
                    'max_value': float(np.max(all_data)),
                    'mean_value': float(np.mean(all_data)),
                    'std_value': float(np.std(all_data))
                })

            return stats

    def get_memory_usage(self) -> int:
        """
        Get buffer memory usage in bytes.

        Returns:
            Memory usage in bytes
        """
        # Buffer + timestamps
        return self._buffer.nbytes + self._timestamps.nbytes

    def __len__(self) -> int:
        """
        Get number of available samples.

        Returns:
            Number of available samples
        """
        return min(self._total_written, self.buffer_size)

    def __repr__(self) -> str:
        """String representation."""
        n_available = min(self._total_written, self.buffer_size)
        fill_pct = (n_available / self.buffer_size) * 100
        return (
            f"DataBuffer(channels={self.n_channels}, "
            f"size={self.buffer_size}, "
            f"available={n_available} ({fill_pct:.1f}%))"
        )


class MultiChannelBuffer:
    """
    Collection of separate buffers for multiple channels.

    This can be more memory-efficient than a single DataBuffer
    if channels are accessed independently frequently.
    """

    def __init__(self, n_channels: int, buffer_size: int):
        """
        Initialize multi-channel buffer.

        Args:
            n_channels: Number of channels
            buffer_size: Buffer size per channel
        """
        self.n_channels = n_channels
        self.buffer_size = buffer_size
        self._buffers = [
            DataBuffer(1, buffer_size) for _ in range(n_channels)
        ]
        self.logger = get_logger(__name__)

    def append(self, data: np.ndarray, timestamp: Optional[float] = None) -> None:
        """
        Append data to all channel buffers.

        Args:
            data: Data array of shape (n_channels, n_samples)
            timestamp: Optional timestamp
        """
        for i, buffer in enumerate(self._buffers):
            buffer.append(data[i:i+1, :], timestamp)

    def get_channel(self, channel_idx: int, n_samples: Optional[int] = None) -> np.ndarray:
        """
        Get data for a specific channel.

        Args:
            channel_idx: Channel index
            n_samples: Number of samples (None = all)

        Returns:
            Data array of shape (n_samples,)
        """
        return self._buffers[channel_idx].get_channel(0, n_samples)

    def get_all_channels(self, n_samples: Optional[int] = None) -> np.ndarray:
        """
        Get data for all channels.

        Args:
            n_samples: Number of samples (None = all)

        Returns:
            Data array of shape (n_channels, n_samples)
        """
        if n_samples is None:
            n_samples = len(self._buffers[0])

        data = np.zeros((self.n_channels, n_samples))
        for i, buffer in enumerate(self._buffers):
            data[i, :] = buffer.get_latest(n_samples)[0, :]

        return data

    def clear(self) -> None:
        """Clear all channel buffers."""
        for buffer in self._buffers:
            buffer.clear()


# Example usage and tests
if __name__ == "__main__":
    print("DataBuffer Test")
    print("=" * 60)

    # Create buffer for 4 channels, 1000 samples
    print("\n1. Creating buffer...")
    buffer = DataBuffer(n_channels=4, buffer_size=1000)
    print(f"Buffer: {buffer}")
    print(f"Memory usage: {buffer.get_memory_usage() / 1024:.2f} KB")

    # Add some data
    print("\n2. Adding data...")
    for i in range(5):
        data = np.random.randn(4, 100)
        buffer.append(data)
        print(f"  Iteration {i+1}: {buffer}")

    # Get statistics
    print("\n3. Buffer statistics:")
    stats = buffer.get_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    # Get latest data
    print("\n4. Retrieving data...")
    latest_100 = buffer.get_latest(100)
    print(f"  Latest 100 samples shape: {latest_100.shape}")

    latest_all = buffer.get_all()
    print(f"  All available data shape: {latest_all.shape}")

    # Get single channel
    print("\n5. Single channel access...")
    channel_0 = buffer.get_channel(0, 200)
    print(f"  Channel 0 (200 samples) shape: {channel_0.shape}")

    # Test wrap-around
    print("\n6. Testing wrap-around...")
    for i in range(15):
        data = np.random.randn(4, 100)
        buffer.append(data)

    print(f"  After wrap-around: {buffer}")
    print(f"  Total written: {buffer.get_stats()['samples_written']}")
    print(f"  Buffer full: {buffer.get_stats()['is_full']}")

    # Clear buffer
    print("\n7. Clearing buffer...")
    buffer.clear()
    print(f"  After clear: {buffer}")

    print("\n" + "=" * 60)
    print("All tests completed successfully!")

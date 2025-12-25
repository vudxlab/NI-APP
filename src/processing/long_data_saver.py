"""
Module for saving long-duration data buffers to temporary files.

This module provides functionality to save the most recent N seconds of data
from a DataBuffer to temporary files for long-window FFT analysis and
low-frequency analysis.
"""

import numpy as np
import h5py
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Tuple
from threading import Lock

from ..utils.logger import get_logger


class LongDataSaver:
    """
    Saves long-duration data buffers to temporary files.

    This class is designed for saving recent data (e.g., 200 seconds) to disk
    for subsequent long-window FFT analysis, particularly useful for low-frequency
    analysis where long time windows are needed.

    Attributes:
        buffer: Reference to the DataBuffer to save from
        sample_rate: Sampling rate in Hz
        max_duration_seconds: Maximum duration to save (default 200s)
        temp_dir: Directory for temporary files
    """

    def __init__(
        self,
        sample_rate: float,
        max_duration_seconds: float = 200.0,
        temp_dir: Optional[str] = None
    ):
        """
        Initialize the long data saver.

        Args:
            sample_rate: Sampling rate in Hz
            max_duration_seconds: Maximum duration to save in seconds
            temp_dir: Directory for temporary files (None = system temp)
        """
        self.logger = get_logger(__name__)
        self.sample_rate = sample_rate
        self.max_duration_seconds = max_duration_seconds
        self.max_samples = int(sample_rate * max_duration_seconds)

        # Setup temp directory
        if temp_dir is None:
            self.temp_dir = Path(tempfile.gettempdir()) / "ni_app_long_data"
        else:
            self.temp_dir = Path(temp_dir)

        self.temp_dir.mkdir(parents=True, exist_ok=True)

        # Current temp file info
        self._current_file_path: Optional[Path] = None
        self._last_save_time: Optional[datetime] = None
        self._lock = Lock()

        self.logger.info(
            f"LongDataSaver initialized: {max_duration_seconds}s "
            f"({self.max_samples} samples @ {sample_rate} Hz), "
            f"temp_dir={self.temp_dir}"
        )

    def save_to_temp_file(
        self,
        data: np.ndarray,
        metadata: Optional[Dict] = None,
        format: str = 'hdf5'
    ) -> Path:
        """
        Save data to a temporary file.

        Args:
            data: Data array of shape (n_channels, n_samples)
            metadata: Optional metadata dictionary to save with the data
            format: File format ('hdf5', 'npy', or 'npz')

        Returns:
            Path to the saved temporary file
        """
        with self._lock:
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

            if format == 'hdf5':
                filename = f"long_data_{timestamp}.h5"
                file_path = self.temp_dir / filename
                self._save_hdf5(file_path, data, metadata)
            elif format == 'npy':
                filename = f"long_data_{timestamp}.npy"
                file_path = self.temp_dir / filename
                np.save(file_path, data)
            elif format == 'npz':
                filename = f"long_data_{timestamp}.npz"
                file_path = self.temp_dir / filename
                if metadata is not None:
                    np.savez(file_path, data=data, **metadata)
                else:
                    np.savez(file_path, data=data)
            else:
                raise ValueError(f"Unsupported format: {format}")

            self._current_file_path = file_path
            self._last_save_time = datetime.now()

            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            self.logger.info(
                f"Saved {data.shape[1]} samples ({data.shape[1]/self.sample_rate:.1f}s) "
                f"to {file_path.name} ({file_size_mb:.2f} MB)"
            )

            return file_path

    def _save_hdf5(
        self,
        file_path: Path,
        data: np.ndarray,
        metadata: Optional[Dict]
    ) -> None:
        """
        Save data to HDF5 format with metadata.

        Args:
            file_path: Path to save file
            data: Data array
            metadata: Metadata dictionary
        """
        with h5py.File(file_path, 'w') as f:
            # Save data
            f.create_dataset(
                'data',
                data=data,
                compression='gzip',
                compression_opts=4
            )

            # Save metadata
            if metadata is None:
                metadata = {}

            metadata['sample_rate'] = self.sample_rate
            metadata['n_channels'] = data.shape[0]
            metadata['n_samples'] = data.shape[1]
            metadata['duration_seconds'] = data.shape[1] / self.sample_rate
            metadata['save_timestamp'] = datetime.now().isoformat()

            for key, value in metadata.items():
                f.attrs[key] = value

    def save_from_buffer(
        self,
        buffer,
        duration_seconds: Optional[float] = None,
        format: str = 'hdf5'
    ) -> Path:
        """
        Save the most recent N seconds from a DataBuffer to a temp file.

        Args:
            buffer: DataBuffer instance to read from
            duration_seconds: Duration to save (None = use max_duration_seconds)
            format: File format ('hdf5', 'npy', or 'npz')

        Returns:
            Path to the saved temporary file
        """
        if duration_seconds is None:
            duration_seconds = self.max_duration_seconds

        # Calculate number of samples to retrieve
        n_samples = int(self.sample_rate * duration_seconds)
        n_samples = min(n_samples, self.max_samples)

        # Get data from buffer
        data = buffer.get_latest(n_samples)

        # Get buffer stats for metadata
        stats = buffer.get_stats()
        metadata = {
            'buffer_size': stats['buffer_size'],
            'samples_retrieved': n_samples,
            'duration_requested_seconds': duration_seconds,
            'duration_actual_seconds': data.shape[1] / self.sample_rate,
        }

        return self.save_to_temp_file(data, metadata, format)

    def load_temp_file(
        self,
        file_path: Optional[Path] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Load data from a temporary file.

        Args:
            file_path: Path to load (None = load most recent saved file)

        Returns:
            Tuple of (data, metadata)
                data: numpy array of shape (n_channels, n_samples)
                metadata: dictionary of metadata
        """
        if file_path is None:
            file_path = self._current_file_path

        if file_path is None:
            raise ValueError("No file path specified and no file has been saved yet")

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Determine format from extension
        if file_path.suffix == '.h5':
            return self._load_hdf5(file_path)
        elif file_path.suffix == '.npy':
            data = np.load(file_path)
            metadata = {}
            return data, metadata
        elif file_path.suffix == '.npz':
            npz_data = np.load(file_path)
            data = npz_data['data']
            metadata = {key: npz_data[key] for key in npz_data.keys() if key != 'data'}
            return data, metadata
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

    def _load_hdf5(self, file_path: Path) -> Tuple[np.ndarray, Dict]:
        """
        Load data from HDF5 file.

        Args:
            file_path: Path to HDF5 file

        Returns:
            Tuple of (data, metadata)
        """
        with h5py.File(file_path, 'r') as f:
            data = f['data'][:]
            metadata = dict(f.attrs)

        return data, metadata

    def get_current_file_path(self) -> Optional[Path]:
        """Get the path to the most recently saved file."""
        return self._current_file_path

    def get_last_save_time(self) -> Optional[datetime]:
        """Get the timestamp of the last save operation."""
        return self._last_save_time

    def cleanup_old_files(self, max_age_hours: float = 24.0) -> int:
        """
        Clean up old temporary files.

        Args:
            max_age_hours: Maximum age of files to keep (in hours)

        Returns:
            Number of files deleted
        """
        import time

        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        deleted_count = 0

        for file_path in self.temp_dir.glob("long_data_*"):
            if file_path.is_file():
                file_age = current_time - file_path.stat().st_mtime
                if file_age > max_age_seconds:
                    try:
                        file_path.unlink()
                        deleted_count += 1
                        self.logger.debug(f"Deleted old temp file: {file_path.name}")
                    except Exception as e:
                        self.logger.warning(f"Failed to delete {file_path.name}: {e}")

        if deleted_count > 0:
            self.logger.info(f"Cleaned up {deleted_count} old temporary files")

        return deleted_count

    def get_temp_dir_size(self) -> float:
        """
        Get total size of temporary directory in MB.

        Returns:
            Total size in MB
        """
        total_size = 0
        for file_path in self.temp_dir.glob("long_data_*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size

        return total_size / (1024 * 1024)

    def list_temp_files(self) -> list:
        """
        List all temporary files in the temp directory.

        Returns:
            List of file paths
        """
        return sorted(self.temp_dir.glob("long_data_*"), key=lambda x: x.stat().st_mtime, reverse=True)


# Example usage
if __name__ == "__main__":
    from data_buffer import DataBuffer

    print("LongDataSaver Test")
    print("=" * 60)

    # Test parameters
    sample_rate = 25600  # Hz
    n_channels = 4
    duration = 200  # seconds

    print(f"\n1. Creating test buffer with {duration}s of data...")
    buffer_size = int(sample_rate * duration)
    buffer = DataBuffer(n_channels=n_channels, buffer_size=buffer_size)

    # Fill buffer with test data
    print("   Filling buffer with test data...")
    for i in range(200):
        data_chunk = np.random.randn(n_channels, int(sample_rate))  # 1 second chunks
        buffer.append(data_chunk)

    print(f"   Buffer: {buffer}")

    # Create saver
    print(f"\n2. Creating LongDataSaver...")
    saver = LongDataSaver(sample_rate=sample_rate, max_duration_seconds=200.0)

    # Save 200s of data
    print(f"\n3. Saving 200s of data to temp file...")
    file_path = saver.save_from_buffer(buffer, duration_seconds=200.0, format='hdf5')
    print(f"   Saved to: {file_path}")
    print(f"   File size: {file_path.stat().st_size / (1024*1024):.2f} MB")

    # Load data back
    print(f"\n4. Loading data back from file...")
    loaded_data, metadata = saver.load_temp_file()
    print(f"   Loaded data shape: {loaded_data.shape}")
    print(f"   Metadata:")
    for key, value in metadata.items():
        if isinstance(value, float):
            print(f"     {key}: {value:.2f}")
        else:
            print(f"     {key}: {value}")

    # Verify data integrity
    print(f"\n5. Verifying data integrity...")
    original_data = buffer.get_all()
    if np.allclose(original_data, loaded_data):
        print("   ✓ Data integrity verified!")
    else:
        print("   ✗ Data mismatch!")

    # Test different durations
    print(f"\n6. Testing different durations...")
    for duration in [10, 20, 50, 100]:
        file_path = saver.save_from_buffer(buffer, duration_seconds=duration)
        data, meta = saver.load_temp_file(file_path)
        print(f"   {duration}s: {data.shape[1]} samples, {data.shape[1]/sample_rate:.1f}s")

    # List temp files
    print(f"\n7. Listing temp files...")
    temp_files = saver.list_temp_files()
    print(f"   Found {len(temp_files)} temp files:")
    for f in temp_files[:5]:
        print(f"     {f.name} ({f.stat().st_size/(1024*1024):.2f} MB)")

    print(f"\n   Total temp dir size: {saver.get_temp_dir_size():.2f} MB")

    # Cleanup
    print(f"\n8. Cleaning up old files...")
    deleted = saver.cleanup_old_files(max_age_hours=0.0)  # Delete all
    print(f"   Deleted {deleted} files")

    print("\n" + "=" * 60)
    print("LongDataSaver test completed!")

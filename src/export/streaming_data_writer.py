"""
Streaming Data Writers for continuous data acquisition.

This module provides streaming writers for HDF5, CSV, and TDMS formats
that support continuous append operations during data acquisition.
"""

import numpy as np
import csv
import threading
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from pathlib import Path
from datetime import datetime

try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False

try:
    from nptdms import TdmsWriter, ChannelObject, RootObject, GroupObject
    NPTDMS_AVAILABLE = True
except ImportError:
    NPTDMS_AVAILABLE = False

try:
    import scipy.io
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from ..utils.logger import get_logger


class StreamingWriter(ABC):
    """
    Abstract base class for streaming data writers.

    All streaming writers must implement this interface for consistent behavior
    across different file formats.
    """

    def __init__(
        self,
        filepath: str,
        n_channels: int,
        sample_rate: float,
        channel_names: List[str],
        channel_units: List[str]
    ):
        """
        Initialize streaming writer.

        Args:
            filepath: Output file path
            n_channels: Number of channels
            sample_rate: Sampling rate in Hz
            channel_names: List of channel names
            channel_units: List of units for each channel
        """
        self.filepath = Path(filepath)
        self.n_channels = n_channels
        self.sample_rate = sample_rate
        self.channel_names = channel_names
        self.channel_units = channel_units

        self.logger = get_logger(__name__)
        self._lock = threading.Lock()
        self._is_open = False
        self._total_samples = 0
        self._start_time = datetime.now()

    @abstractmethod
    def open(self) -> None:
        """Open the file and prepare for writing."""
        pass

    @abstractmethod
    def append(self, data: np.ndarray, timestamp: float) -> None:
        """
        Append new data to the file.

        Args:
            data: Data array of shape (n_channels, n_samples)
            timestamp: Timestamp of first sample
        """
        pass

    @abstractmethod
    def flush(self) -> None:
        """Force write buffered data to disk."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the file properly."""
        pass

    def get_total_samples(self) -> int:
        """Return total number of samples written."""
        return self._total_samples

    def get_duration_seconds(self) -> float:
        """Return total duration of data written."""
        return self._total_samples / self.sample_rate if self.sample_rate > 0 else 0.0

    def is_open(self) -> bool:
        """Check if file is currently open."""
        return self._is_open


class StreamingHDF5Writer(StreamingWriter):
    """
    Streaming HDF5 writer with SWMR (Single-Writer-Multiple-Reader) support.

    This writer uses HDF5's SWMR mode to allow concurrent reading while writing,
    making it ideal for real-time analysis of data being acquired.
    """

    def __init__(
        self,
        filepath: str,
        n_channels: int,
        sample_rate: float,
        channel_names: List[str],
        channel_units: List[str],
        compression_level: int = 4
    ):
        """
        Initialize HDF5 streaming writer.

        Args:
            filepath: Output HDF5 file path
            n_channels: Number of channels
            sample_rate: Sampling rate in Hz
            channel_names: List of channel names
            channel_units: List of units for each channel
            compression_level: gzip compression level (0-9, default 4)
        """
        super().__init__(filepath, n_channels, sample_rate, channel_names, channel_units)

        if not H5PY_AVAILABLE:
            raise ImportError("h5py not available. Install with: pip install h5py")

        self.compression_level = compression_level
        self.file: Optional[h5py.File] = None
        self.datasets: List[h5py.Dataset] = []

        # Chunk size: 1 second of data
        self.chunk_size = int(sample_rate)

    def open(self) -> None:
        """Open HDF5 file and create datasets with SWMR mode."""
        with self._lock:
            if self._is_open:
                self.logger.warning("File already open")
                return

            try:
                # Create parent directory if needed
                self.filepath.parent.mkdir(parents=True, exist_ok=True)

                # Create HDF5 file with latest library version (required for SWMR)
                self.file = h5py.File(str(self.filepath), 'w', libver='latest')

                # Create data group
                data_group = self.file.create_group('data')

                # Create dataset for each channel
                self.datasets = []
                for i, (name, units) in enumerate(zip(self.channel_names, self.channel_units)):
                    # Sanitize channel name
                    safe_name = name.replace(' ', '_').replace('/', '_')
                    dataset_name = f'channel_{i:02d}_{safe_name}'

                    # Create dataset with unlimited growth
                    dataset = data_group.create_dataset(
                        dataset_name,
                        shape=(0,),
                        maxshape=(None,),
                        chunks=(self.chunk_size,),
                        dtype='float64',
                        compression='gzip',
                        compression_opts=self.compression_level
                    )

                    # Add attributes
                    dataset.attrs['channel_name'] = name
                    dataset.attrs['units'] = units
                    dataset.attrs['sample_rate'] = self.sample_rate
                    dataset.attrs['channel_index'] = i

                    self.datasets.append(dataset)

                # Create metadata group
                meta_group = self.file.create_group('metadata')
                meta_group.attrs['start_time'] = self._start_time.isoformat()
                meta_group.attrs['sample_rate'] = self.sample_rate
                meta_group.attrs['n_channels'] = self.n_channels
                meta_group.attrs['format_version'] = '1.0'

                # Enable SWMR mode (must be done after creating all datasets)
                self.file.swmr_mode = True

                self._is_open = True
                self.logger.info(f"HDF5 file opened: {self.filepath}")

            except Exception as e:
                self.logger.error(f"Failed to open HDF5 file: {e}")
                if self.file:
                    self.file.close()
                raise

    def append(self, data: np.ndarray, timestamp: float) -> None:
        """
        Append data to HDF5 file.

        Args:
            data: Data array of shape (n_channels, n_samples)
            timestamp: Timestamp of first sample (not used in HDF5, but kept for interface)
        """
        if not self._is_open:
            raise RuntimeError("File not open. Call open() first.")

        with self._lock:
            try:
                n_new_samples = data.shape[1]

                # Append each channel's data
                for i, dataset in enumerate(self.datasets):
                    current_size = dataset.shape[0]
                    new_size = current_size + n_new_samples

                    # Resize dataset
                    dataset.resize(new_size, axis=0)

                    # Write new data
                    dataset[current_size:new_size] = data[i, :]

                self._total_samples += n_new_samples

                # Flush to disk (critical for SWMR)
                self.file.flush()

            except Exception as e:
                self.logger.error(f"Failed to append data: {e}")
                raise

    def flush(self) -> None:
        """Force write to disk."""
        if self._is_open and self.file:
            with self._lock:
                self.file.flush()

    def close(self) -> None:
        """Close HDF5 file."""
        with self._lock:
            if not self._is_open:
                return

            try:
                # Update metadata with final statistics
                if self.file and 'metadata' in self.file:
                    self.file['metadata'].attrs['end_time'] = datetime.now().isoformat()
                    self.file['metadata'].attrs['total_samples'] = self._total_samples
                    self.file['metadata'].attrs['duration_seconds'] = self.get_duration_seconds()

                # Close file
                if self.file:
                    self.file.close()
                    self.file = None

                self._is_open = False
                self.logger.info(f"HDF5 file closed: {self.filepath} ({self._total_samples} samples)")

            except Exception as e:
                self.logger.error(f"Error closing HDF5 file: {e}")
                raise


class StreamingCSVWriter(StreamingWriter):
    """
    Streaming CSV writer with periodic flush.

    Uses simple append mode with periodic close/reopen for data safety.
    """

    def __init__(
        self,
        filepath: str,
        n_channels: int,
        sample_rate: float,
        channel_names: List[str],
        channel_units: List[str]
    ):
        """Initialize CSV streaming writer."""
        super().__init__(filepath, n_channels, sample_rate, channel_names, channel_units)

        self.file_handle = None
        self.csv_writer = None
        self._header_written = False
        self._time_offset = 0.0

    def open(self) -> None:
        """Open CSV file and write header."""
        with self._lock:
            if self._is_open:
                self.logger.warning("File already open")
                return

            try:
                # Create parent directory if needed
                self.filepath.parent.mkdir(parents=True, exist_ok=True)

                # Open file in write mode
                self.file_handle = open(self.filepath, 'w', newline='')

                # Write metadata as comments
                self.file_handle.write(f"# NI DAQ Data Acquisition\n")
                self.file_handle.write(f"# Start Time: {self._start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                self.file_handle.write(f"# Sample Rate: {self.sample_rate} Hz\n")
                self.file_handle.write(f"# Channels: {self.n_channels}\n")
                self.file_handle.write("#\n")

                # Create CSV writer
                self.csv_writer = csv.writer(self.file_handle)

                # Write header row
                header = ['Time (s)'] + [
                    f"{name} ({units})"
                    for name, units in zip(self.channel_names, self.channel_units)
                ]
                self.csv_writer.writerow(header)

                self._header_written = True
                self._is_open = True
                self._time_offset = 0.0

                self.logger.info(f"CSV file opened: {self.filepath}")

            except Exception as e:
                self.logger.error(f"Failed to open CSV file: {e}")
                if self.file_handle:
                    self.file_handle.close()
                raise

    def append(self, data: np.ndarray, timestamp: float) -> None:
        """
        Append data to CSV file.

        Args:
            data: Data array of shape (n_channels, n_samples)
            timestamp: Timestamp of first sample (not used, time is computed from sample count)
        """
        if not self._is_open:
            raise RuntimeError("File not open. Call open() first.")

        with self._lock:
            try:
                n_new_samples = data.shape[1]

                # Transpose data for row-major output
                data_transposed = data.T  # Shape: (n_samples, n_channels)

                # Write data rows
                for i in range(n_new_samples):
                    time_value = (self._total_samples + i) / self.sample_rate
                    row = [time_value] + data_transposed[i, :].tolist()
                    self.csv_writer.writerow(row)

                self._total_samples += n_new_samples

            except Exception as e:
                self.logger.error(f"Failed to append CSV data: {e}")
                raise

    def flush(self) -> None:
        """
        Flush CSV file and reopen for safety.

        This close/reopen strategy ensures data is written even if acquisition
        is interrupted unexpectedly.
        """
        if not self._is_open or not self.file_handle:
            return

        with self._lock:
            try:
                # Flush and close
                self.file_handle.flush()
                self.file_handle.close()

                # Reopen in append mode
                self.file_handle = open(self.filepath, 'a', newline='')
                self.csv_writer = csv.writer(self.file_handle)

            except Exception as e:
                self.logger.error(f"Error during CSV flush: {e}")
                # Try to reopen anyway
                try:
                    self.file_handle = open(self.filepath, 'a', newline='')
                    self.csv_writer = csv.writer(self.file_handle)
                except:
                    self._is_open = False
                    raise

    def close(self) -> None:
        """Close CSV file."""
        with self._lock:
            if not self._is_open:
                return

            try:
                if self.file_handle:
                    self.file_handle.flush()
                    self.file_handle.close()
                    self.file_handle = None

                self._is_open = False
                self.logger.info(f"CSV file closed: {self.filepath} ({self._total_samples} samples)")

            except Exception as e:
                self.logger.error(f"Error closing CSV file: {e}")
                raise


class StreamingTDMSWriter(StreamingWriter):
    """
    Streaming TDMS writer with periodic flush.

    Uses TDMS append mode with periodic close/reopen for data safety.
    """

    def __init__(
        self,
        filepath: str,
        n_channels: int,
        sample_rate: float,
        channel_names: List[str],
        channel_units: List[str]
    ):
        """Initialize TDMS streaming writer."""
        super().__init__(filepath, n_channels, sample_rate, channel_names, channel_units)

        if not NPTDMS_AVAILABLE:
            raise ImportError("nptdms not available. Install with: pip install nptdms")

        self.group_name = "Data"
        self._first_write = True

    def open(self) -> None:
        """Create TDMS file with root metadata."""
        with self._lock:
            if self._is_open:
                self.logger.warning("File already open")
                return

            try:
                # Create parent directory if needed
                self.filepath.parent.mkdir(parents=True, exist_ok=True)

                # Create root object with metadata
                root_metadata = {
                    'description': 'NI DAQ Data Acquisition',
                    'start_time': self._start_time.isoformat(),
                    'sample_rate': self.sample_rate,
                    'n_channels': self.n_channels
                }

                root_object = RootObject(properties=root_metadata)

                # Create group object
                group_metadata = {
                    'description': 'Acceleration data',
                    'sample_rate': self.sample_rate
                }
                group_object = GroupObject(self.group_name, properties=group_metadata)

                # Write initial file with metadata only
                with TdmsWriter(str(self.filepath)) as writer:
                    writer.write_segment([root_object, group_object])

                self._is_open = True
                self._first_write = True

                self.logger.info(f"TDMS file opened: {self.filepath}")

            except Exception as e:
                self.logger.error(f"Failed to open TDMS file: {e}")
                raise

    def append(self, data: np.ndarray, timestamp: float) -> None:
        """
        Append data to TDMS file.

        Args:
            data: Data array of shape (n_channels, n_samples)
            timestamp: Timestamp of first sample (not used in TDMS)
        """
        if not self._is_open:
            raise RuntimeError("File not open. Call open() first.")

        with self._lock:
            try:
                n_new_samples = data.shape[1]

                # Create channel objects for this segment
                channels = []
                for i, (name, units) in enumerate(zip(self.channel_names, self.channel_units)):
                    safe_name = name.replace(' ', '_').replace('/', '_')

                    # Create channel properties
                    channel_props = {
                        'unit_string': units,
                        'channel_index': i,
                        'description': f'Channel {i}: {name}'
                    }

                    # Create channel object with data
                    channel = ChannelObject(
                        self.group_name,
                        safe_name,
                        data[i, :],
                        properties=channel_props
                    )
                    channels.append(channel)

                # Append to file
                with TdmsWriter(str(self.filepath), mode='a') as writer:
                    writer.write_segment(channels)

                self._total_samples += n_new_samples
                self._first_write = False

            except Exception as e:
                self.logger.error(f"Failed to append TDMS data: {e}")
                raise

    def flush(self) -> None:
        """
        Flush TDMS file.

        For TDMS, each append operation closes the file, so flush is a no-op.
        """
        pass  # TDMS append mode handles this automatically

    def close(self) -> None:
        """Close TDMS file."""
        with self._lock:
            if not self._is_open:
                return

            try:
                # TDMS files are automatically closed after each write
                # Just mark as closed and log
                self._is_open = False
                self.logger.info(f"TDMS file closed: {self.filepath} ({self._total_samples} samples)")

            except Exception as e:
                self.logger.error(f"Error closing TDMS file: {e}")
                raise


class StreamingMATWriter(StreamingWriter):
    """
    Streaming MATLAB .mat writer with buffered approach.

    Unlike HDF5, MATLAB .mat files don't support true streaming/append mode.
    Strategy: Buffer data in memory and rewrite entire file on flush.
    This is acceptable for typical acquisition durations (< 10 minutes).

    For very long acquisitions, consider using HDF5 and converting to MAT later.
    """

    def __init__(
        self,
        filepath: str,
        n_channels: int,
        sample_rate: float,
        channel_names: List[str],
        channel_units: List[str],
        array_name: Optional[str] = None
    ):
        """
        Initialize MATLAB .mat streaming writer.

        Args:
            filepath: Output .mat file path
            n_channels: Number of channels
            sample_rate: Sampling rate in Hz
            channel_names: List of channel names
            channel_units: List of units for each channel
            array_name: Name for the data array in .mat file (defaults to file prefix)
        """
        super().__init__(filepath, n_channels, sample_rate, channel_names, channel_units)

        if not SCIPY_AVAILABLE:
            raise ImportError("scipy not available. Install with: pip install scipy")

        # Extract array name from filepath if not provided
        if array_name is None:
            # Use filename without extension as array name
            self.array_name = self.filepath.stem
        else:
            self.array_name = array_name

        # In-memory buffer for accumulated data
        self.data_buffer: List[np.ndarray] = []
        self.buffer_samples = 0

        # Memory limit: ~500MB of float64 data
        # At 6 channels, ~10.4M samples = ~69 seconds @ 51.2 kHz
        self.max_buffer_samples = 10_000_000

    def open(self) -> None:
        """Initialize the writer (create empty buffer)."""
        with self._lock:
            if self._is_open:
                self.logger.warning("File already open")
                return

            try:
                # Create parent directory if needed
                self.filepath.parent.mkdir(parents=True, exist_ok=True)

                # Initialize empty buffer
                self.data_buffer = []
                self.buffer_samples = 0
                self._total_samples = 0

                self._is_open = True
                self.logger.info(f"MAT writer initialized: {self.filepath}")

            except Exception as e:
                self.logger.error(f"Failed to initialize MAT writer: {e}")
                raise

    def append(self, data: np.ndarray, timestamp: float) -> None:
        """
        Append data to in-memory buffer.

        Args:
            data: Data array of shape (n_channels, n_samples)
            timestamp: Timestamp of first sample (stored but not used in MAT format)
        """
        if not self._is_open:
            raise RuntimeError("File not open. Call open() first.")

        with self._lock:
            try:
                n_new_samples = data.shape[1]

                # Check buffer size limit
                if self.buffer_samples + n_new_samples > self.max_buffer_samples:
                    self.logger.warning(
                        f"Buffer size limit reached ({self.max_buffer_samples} samples). "
                        "Flushing to disk..."
                    )
                    self._flush_internal()

                # Add to buffer (keep internal format for now)
                self.data_buffer.append(data.copy())
                self.buffer_samples += n_new_samples
                self._total_samples += n_new_samples

            except Exception as e:
                self.logger.error(f"Failed to append MAT data: {e}")
                raise

    def flush(self) -> None:
        """Write buffered data to .mat file."""
        if not self._is_open:
            return

        with self._lock:
            self._flush_internal()

    def _flush_internal(self) -> None:
        """Internal flush implementation (assumes lock is held)."""
        if not self.data_buffer:
            return  # Nothing to write

        try:
            # Concatenate all buffered data
            # Shape: (n_channels, total_samples)
            data_internal = np.concatenate(self.data_buffer, axis=1)

            # Transpose to MATLAB convention: (n_samples, n_channels)
            data_matlab = data_internal.T

            # Prepare .mat file dictionary
            mat_dict = {
                self.array_name: data_matlab,
                'channel_names': self.channel_names,
                'channel_units': self.channel_units,
                'sample_rate': self.sample_rate,
                'start_time': self._start_time.isoformat()
            }

            # Write to file
            scipy.io.savemat(
                str(self.filepath),
                mat_dict,
                do_compression=True,
                oned_as='column'
            )

            self.logger.info(
                f"MAT file flushed: {self.filepath} "
                f"({data_matlab.shape[0]} samples, {data_matlab.shape[1]} channels)"
            )

            # Clear buffer after successful write
            self.data_buffer = []
            self.buffer_samples = 0

        except Exception as e:
            self.logger.error(f"Failed to flush MAT file: {e}")
            raise

    def close(self) -> None:
        """Close MAT writer and write final data."""
        with self._lock:
            if not self._is_open:
                return

            try:
                # Final flush
                self._flush_internal()

                self._is_open = False
                self.logger.info(
                    f"MAT file closed: {self.filepath} ({self._total_samples} samples)"
                )

            except Exception as e:
                self.logger.error(f"Error closing MAT file: {e}")
                raise


def create_streaming_writer(
    file_format: str,
    filepath: str,
    n_channels: int,
    sample_rate: float,
    channel_names: List[str],
    channel_units: List[str],
    **kwargs
) -> StreamingWriter:
    """
    Factory function to create appropriate streaming writer.

    Args:
        file_format: Format type ('hdf5', 'csv', 'tdms', or 'mat')
        filepath: Output file path
        n_channels: Number of channels
        sample_rate: Sampling rate in Hz
        channel_names: List of channel names
        channel_units: List of units for each channel
        **kwargs: Additional format-specific arguments (e.g., compression_level for HDF5, array_name for MAT)

    Returns:
        StreamingWriter instance

    Raises:
        ValueError: If file_format is not supported
    """
    format_lower = file_format.lower()

    if format_lower == 'hdf5':
        compression_level = kwargs.get('compression_level', 4)
        return StreamingHDF5Writer(
            filepath, n_channels, sample_rate, channel_names, channel_units,
            compression_level=compression_level
        )
    elif format_lower == 'csv':
        return StreamingCSVWriter(
            filepath, n_channels, sample_rate, channel_names, channel_units
        )
    elif format_lower == 'tdms':
        return StreamingTDMSWriter(
            filepath, n_channels, sample_rate, channel_names, channel_units
        )
    elif format_lower == 'mat':
        array_name = kwargs.get('array_name', None)
        return StreamingMATWriter(
            filepath, n_channels, sample_rate, channel_names, channel_units,
            array_name=array_name
        )
    else:
        raise ValueError(f"Unsupported file format: {file_format}. Supported: hdf5, csv, tdms, mat")

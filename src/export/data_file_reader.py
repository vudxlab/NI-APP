"""
Data File Reader for reading acquisition files in multiple formats.

This module provides unified reading capabilities for HDF5, CSV, and TDMS files,
with support for reading the most recent N seconds of data from growing files.
"""

import numpy as np
import re
from typing import Tuple, Dict, Any, Optional
from pathlib import Path

try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    from nptdms import TdmsFile
    NPTDMS_AVAILABLE = True
except ImportError:
    NPTDMS_AVAILABLE = False

try:
    import scipy.io
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from ..utils.logger import get_logger


class DataFileReader:
    """
    Unified reader for acquisition data files.

    Supports reading from HDF5, CSV, and TDMS formats with automatic format detection
    and efficient reading of recent data from growing files.
    """

    def __init__(self):
        """Initialize the data file reader."""
        self.logger = get_logger(__name__)

    @staticmethod
    def detect_format(filepath: str) -> str:
        """
        Detect file format from extension.

        Args:
            filepath: Path to data file

        Returns:
            Format string ('hdf5', 'csv', 'tdms', or 'mat')

        Raises:
            ValueError: If format cannot be determined
        """
        path = Path(filepath)
        suffix = path.suffix.lower()

        if suffix in ['.h5', '.hdf5']:
            return 'hdf5'
        elif suffix == '.csv':
            return 'csv'
        elif suffix == '.tdms':
            return 'tdms'
        elif suffix == '.mat':
            return 'mat'
        else:
            raise ValueError(f"Unknown file format: {suffix}")

    def read_recent_seconds(
        self,
        filepath: str,
        duration_seconds: float,
        sample_rate: float
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Read the most recent N seconds from a data file.

        This method automatically detects the file format and reads the appropriate
        amount of data from the end of the file.

        Args:
            filepath: Path to data file
            duration_seconds: Duration to read in seconds (e.g., 10, 20, 50, 100, 200)
            sample_rate: Expected sampling rate in Hz

        Returns:
            Tuple of (data, metadata) where:
                - data: numpy array of shape (n_channels, n_samples)
                - metadata: dict with file information

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is not supported or file is empty
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        # Detect format
        file_format = self.detect_format(str(filepath))

        # Read based on format
        if file_format == 'hdf5':
            return self._read_hdf5_recent(filepath, duration_seconds, sample_rate)
        elif file_format == 'csv':
            return self._read_csv_recent(filepath, duration_seconds, sample_rate)
        elif file_format == 'tdms':
            return self._read_tdms_recent(filepath, duration_seconds, sample_rate)
        elif file_format == 'mat':
            return self._read_mat_recent(filepath, duration_seconds, sample_rate)
        else:
            raise ValueError(f"Unsupported format: {file_format}")

    def _read_hdf5_recent(
        self,
        filepath: Path,
        duration_seconds: float,
        sample_rate: float
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Read recent data from HDF5 file using SWMR mode.

        Args:
            filepath: Path to HDF5 file
            duration_seconds: Duration to read
            sample_rate: Sampling rate

        Returns:
            (data, metadata) tuple
        """
        if not H5PY_AVAILABLE:
            raise ImportError("h5py not available. Install with: pip install h5py")

        try:
            # Open in SWMR read mode for concurrent access
            with h5py.File(filepath, 'r', swmr=True) as f:
                # Refresh to see latest data
                f.id.refresh()

                # Get channel datasets
                data_group = f['data']
                channel_keys = sorted([k for k in data_group.keys() if k.startswith('channel_')])

                if not channel_keys:
                    raise ValueError("No channel data found in HDF5 file")

                # Calculate number of samples to read
                n_samples_requested = int(duration_seconds * sample_rate)

                # Read data from each channel
                channel_data = []
                for key in channel_keys:
                    dataset = data_group[key]
                    total_samples = dataset.shape[0]

                    if total_samples == 0:
                        raise ValueError(f"Channel {key} has no data")

                    # Read last N samples (or all if file has less)
                    n_samples_to_read = min(n_samples_requested, total_samples)
                    start_idx = total_samples - n_samples_to_read

                    data = dataset[start_idx:total_samples]
                    channel_data.append(data)

                # Stack into array (n_channels, n_samples)
                data_array = np.array(channel_data)

                # Get metadata
                metadata = {}
                if 'metadata' in f:
                    meta_group = f['metadata']
                    for key in meta_group.attrs:
                        metadata[key] = meta_group.attrs[key]

                # Add actual samples read
                metadata['samples_read'] = data_array.shape[1]
                metadata['duration_read'] = data_array.shape[1] / sample_rate
                metadata['n_channels'] = data_array.shape[0]

                self.logger.info(
                    f"Read {data_array.shape[1]} samples from HDF5 "
                    f"({metadata['duration_read']:.2f}s)"
                )

                return data_array, metadata

        except Exception as e:
            self.logger.error(f"Failed to read HDF5 file: {e}")
            raise

    def _read_csv_recent(
        self,
        filepath: Path,
        duration_seconds: float,
        sample_rate: float
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Read recent data from CSV file.

        Args:
            filepath: Path to CSV file
            duration_seconds: Duration to read
            sample_rate: Sampling rate

        Returns:
            (data, metadata) tuple
        """
        if not PANDAS_AVAILABLE:
            # Fallback to slower numpy method if pandas not available
            return self._read_csv_recent_numpy(filepath, duration_seconds, sample_rate)

        try:
            # Read CSV file
            df = pd.read_csv(filepath, comment='#')

            # Calculate number of rows to read
            n_samples_requested = int(duration_seconds * sample_rate)

            # Get last N rows
            df_recent = df.tail(n_samples_requested)

            # Extract data (skip first column which is time)
            data = df_recent.iloc[:, 1:].values.T  # Shape: (n_channels, n_samples)

            # Extract metadata from comments
            metadata = self._parse_csv_metadata(filepath)
            metadata['samples_read'] = data.shape[1]
            metadata['duration_read'] = data.shape[1] / sample_rate
            metadata['n_channels'] = data.shape[0]

            self.logger.info(
                f"Read {data.shape[1]} samples from CSV "
                f"({metadata['duration_read']:.2f}s)"
            )

            return data, metadata

        except Exception as e:
            self.logger.error(f"Failed to read CSV file: {e}")
            raise

    def _read_csv_recent_numpy(
        self,
        filepath: Path,
        duration_seconds: float,
        sample_rate: float
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Read recent data from CSV using numpy (slower fallback).

        Args:
            filepath: Path to CSV file
            duration_seconds: Duration to read
            sample_rate: Sampling rate

        Returns:
            (data, metadata) tuple
        """
        try:
            # Load entire file
            data_full = np.loadtxt(filepath, delimiter=',', skiprows=self._count_header_rows(filepath))

            # Calculate samples to read
            n_samples_requested = int(duration_seconds * sample_rate)

            # Get last N rows
            if data_full.shape[0] > n_samples_requested:
                data_recent = data_full[-n_samples_requested:, :]
            else:
                data_recent = data_full

            # Extract data (skip first column which is time)
            data = data_recent[:, 1:].T  # Shape: (n_channels, n_samples)

            # Parse metadata
            metadata = self._parse_csv_metadata(filepath)
            metadata['samples_read'] = data.shape[1]
            metadata['duration_read'] = data.shape[1] / sample_rate
            metadata['n_channels'] = data.shape[0]

            return data, metadata

        except Exception as e:
            self.logger.error(f"Failed to read CSV with numpy: {e}")
            raise

    def _read_tdms_recent(
        self,
        filepath: Path,
        duration_seconds: float,
        sample_rate: float
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Read recent data from TDMS file.

        Args:
            filepath: Path to TDMS file
            duration_seconds: Duration to read
            sample_rate: Sampling rate

        Returns:
            (data, metadata) tuple
        """
        if not NPTDMS_AVAILABLE:
            raise ImportError("nptdms not available. Install with: pip install nptdms")

        try:
            # Read TDMS file
            tdms_file = TdmsFile.read(filepath)

            # Get data group (typically named "Data")
            group_name = "Data"
            if group_name not in tdms_file.groups():
                # Try to find first available group
                groups = tdms_file.groups()
                if not groups:
                    raise ValueError("No groups found in TDMS file")
                group_name = groups[0].name

            group = tdms_file[group_name]

            # Calculate samples to read
            n_samples_requested = int(duration_seconds * sample_rate)

            # Read data from each channel
            channel_data = []
            for channel in group.channels():
                full_data = channel[:]

                if len(full_data) == 0:
                    raise ValueError(f"Channel {channel.name} has no data")

                # Get last N samples
                n_samples_to_read = min(n_samples_requested, len(full_data))
                recent_data = full_data[-n_samples_to_read:]

                channel_data.append(recent_data)

            # Stack into array (n_channels, n_samples)
            data = np.array(channel_data)

            # Get metadata
            metadata = {}
            if tdms_file.properties:
                for key, value in tdms_file.properties.items():
                    metadata[key] = value

            if group.properties:
                for key, value in group.properties.items():
                    metadata[key] = value

            # Add read statistics
            metadata['samples_read'] = data.shape[1]
            metadata['duration_read'] = data.shape[1] / sample_rate
            metadata['n_channels'] = data.shape[0]

            self.logger.info(
                f"Read {data.shape[1]} samples from TDMS "
                f"({metadata['duration_read']:.2f}s)"
            )

            return data, metadata

        except Exception as e:
            self.logger.error(f"Failed to read TDMS file: {e}")
            raise

    def _read_mat_recent(
        self,
        filepath: Path,
        duration_seconds: float,
        sample_rate: float
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Read recent data from MATLAB .mat file.

        Args:
            filepath: Path to .mat file
            duration_seconds: Duration to read
            sample_rate: Sampling rate

        Returns:
            (data, metadata) tuple
        """
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy not available. Install with: pip install scipy")

        try:
            # Load .mat file
            mat_contents = scipy.io.loadmat(str(filepath))

            # Filter out metadata keys
            metadata_keys = {'__header__', '__version__', '__globals__'}
            data_keys = [k for k in mat_contents.keys() if k not in metadata_keys]

            if not data_keys:
                raise ValueError("No data arrays found in .mat file")

            # Use first data array (or look for specific names)
            # Priority: look for common names first
            priority_names = ['data', 'acquisition', 'Setup9', 'Setup']
            main_key = None

            for name in priority_names:
                if name in data_keys:
                    main_key = name
                    break

            if main_key is None:
                # Use first available data key
                main_key = data_keys[0]

            # Get the data array
            data_matlab = mat_contents[main_key]  # Shape: (n_samples, n_channels)

            if data_matlab.ndim != 2:
                raise ValueError(
                    f"Expected 2D array, got {data_matlab.ndim}D array"
                )

            n_samples_total, n_channels = data_matlab.shape

            # Calculate samples to read
            n_samples_requested = int(duration_seconds * sample_rate)

            # Get last N samples
            if n_samples_total > n_samples_requested:
                data_recent = data_matlab[-n_samples_requested:, :]
            else:
                data_recent = data_matlab

            # Transpose to internal format: (n_channels, n_samples)
            data = data_recent.T

            # Extract metadata
            metadata = {
                'array_name': main_key,
                'samples_read': data.shape[1],
                'duration_read': data.shape[1] / sample_rate,
                'n_channels': data.shape[0],
                'total_samples_in_file': n_samples_total
            }

            # Try to extract additional metadata
            if 'sample_rate' in mat_contents:
                metadata['file_sample_rate'] = float(mat_contents['sample_rate'].flat[0])

            if 'channel_names' in mat_contents:
                # Handle cell array of strings from MATLAB
                try:
                    ch_names = mat_contents['channel_names']
                    if isinstance(ch_names, np.ndarray):
                        # Flatten and convert to list of strings
                        metadata['channel_names'] = [str(n).strip() for n in ch_names.flatten()]
                except:
                    metadata['has_channel_names'] = True

            if 'channel_units' in mat_contents:
                try:
                    ch_units = mat_contents['channel_units']
                    if isinstance(ch_units, np.ndarray):
                        metadata['channel_units'] = [str(u).strip() for u in ch_units.flatten()]
                except:
                    metadata['has_channel_units'] = True

            if 'start_time' in mat_contents:
                try:
                    metadata['start_time'] = str(mat_contents['start_time'].flat[0])
                except:
                    pass

            self.logger.info(
                f"Read {data.shape[1]} samples from MAT file "
                f"({metadata['duration_read']:.2f}s, array: {main_key})"
            )

            return data, metadata

        except Exception as e:
            self.logger.error(f"Failed to read MAT file: {e}")
            raise

    def get_file_info(self, filepath: str) -> Dict[str, Any]:
        """
        Get information about a data file without reading all data.

        Args:
            filepath: Path to data file

        Returns:
            Dictionary with file information (format, channels, duration, etc.)

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        # Detect format
        file_format = self.detect_format(str(filepath))

        info = {
            'filepath': str(filepath),
            'format': file_format,
            'file_size_bytes': filepath.stat().st_size
        }

        # Get format-specific info
        if file_format == 'hdf5':
            info.update(self._get_hdf5_info(filepath))
        elif file_format == 'csv':
            info.update(self._get_csv_info(filepath))
        elif file_format == 'tdms':
            info.update(self._get_tdms_info(filepath))
        elif file_format == 'mat':
            info.update(self._get_mat_info(filepath))

        return info

    def _get_hdf5_info(self, filepath: Path) -> Dict[str, Any]:
        """Get HDF5 file information."""
        if not H5PY_AVAILABLE:
            return {}

        try:
            with h5py.File(filepath, 'r', swmr=True) as f:
                f.id.refresh()

                info = {}

                # Get metadata
                if 'metadata' in f:
                    for key in f['metadata'].attrs:
                        info[key] = f['metadata'].attrs[key]

                # Get channel info
                if 'data' in f:
                    channel_keys = [k for k in f['data'].keys() if k.startswith('channel_')]
                    info['n_channels'] = len(channel_keys)

                    if channel_keys:
                        # Get total samples from first channel
                        first_channel = f['data'][channel_keys[0]]
                        info['total_samples'] = first_channel.shape[0]

                        if 'sample_rate' in info:
                            info['duration_seconds'] = info['total_samples'] / info['sample_rate']

                return info

        except Exception as e:
            self.logger.warning(f"Could not read HDF5 info: {e}")
            return {}

    def _get_csv_info(self, filepath: Path) -> Dict[str, Any]:
        """Get CSV file information."""
        info = self._parse_csv_metadata(filepath)

        # Try to count rows efficiently
        try:
            if PANDAS_AVAILABLE:
                df = pd.read_csv(filepath, comment='#', usecols=[0])
                info['total_samples'] = len(df)
            else:
                # Count lines manually
                with open(filepath, 'r') as f:
                    header_rows = self._count_header_rows(filepath)
                    total_lines = sum(1 for _ in f)
                    info['total_samples'] = total_lines - header_rows

            if 'sample_rate' in info and info['sample_rate'] > 0:
                info['duration_seconds'] = info['total_samples'] / info['sample_rate']

        except Exception as e:
            self.logger.warning(f"Could not count CSV rows: {e}")

        return info

    def _get_tdms_info(self, filepath: Path) -> Dict[str, Any]:
        """Get TDMS file information."""
        if not NPTDMS_AVAILABLE:
            return {}

        try:
            tdms_file = TdmsFile.read(filepath)

            info = {}

            # Get root properties
            if tdms_file.properties:
                for key, value in tdms_file.properties.items():
                    info[key] = value

            # Get group info
            groups = tdms_file.groups()
            if groups:
                group = groups[0]  # Use first group
                info['n_channels'] = len(group.channels())

                if group.properties:
                    for key, value in group.properties.items():
                        info[key] = value

                # Get total samples from first channel
                if group.channels():
                    first_channel = group.channels()[0]
                    info['total_samples'] = len(first_channel[:])

                    if 'sample_rate' in info and info['sample_rate'] > 0:
                        info['duration_seconds'] = info['total_samples'] / info['sample_rate']

            return info

        except Exception as e:
            self.logger.warning(f"Could not read TDMS info: {e}")
            return {}

    def _get_mat_info(self, filepath: Path) -> Dict[str, Any]:
        """Get MAT file information."""
        if not SCIPY_AVAILABLE:
            return {}

        try:
            # Load .mat file
            mat_contents = scipy.io.loadmat(str(filepath))

            info = {}

            # Find main data array (ignore metadata keys)
            metadata_keys = {'__header__', '__version__', '__globals__'}
            data_keys = [k for k in mat_contents.keys() if k not in metadata_keys]

            if data_keys:
                # Use first data array or priority name
                priority_names = ['data', 'acquisition', 'Setup9', 'Setup']
                main_key = None
                for name in priority_names:
                    if name in data_keys:
                        main_key = name
                        break
                if main_key is None:
                    main_key = data_keys[0]

                data_array = mat_contents[main_key]

                if data_array.ndim == 2:
                    n_samples, n_channels = data_array.shape
                    info['total_samples'] = n_samples
                    info['n_channels'] = n_channels
                    info['array_name'] = main_key

            # Extract metadata
            if 'sample_rate' in mat_contents:
                try:
                    info['sample_rate'] = float(mat_contents['sample_rate'].flat[0])
                    if 'total_samples' in info:
                        info['duration_seconds'] = info['total_samples'] / info['sample_rate']
                except:
                    pass

            if 'channel_names' in mat_contents:
                info['has_channel_names'] = True

            if 'channel_units' in mat_contents:
                info['has_channel_units'] = True

            return info

        except Exception as e:
            self.logger.warning(f"Could not read MAT info: {e}")
            return {}

    @staticmethod
    def _parse_csv_metadata(filepath: Path) -> Dict[str, Any]:
        """Parse metadata from CSV comment lines."""
        metadata = {}

        try:
            with open(filepath, 'r') as f:
                for line in f:
                    if not line.startswith('#'):
                        break

                    # Parse metadata from comment lines
                    line = line[1:].strip()  # Remove # and whitespace

                    if ':' in line:
                        key, value = line.split(':', 1)
                        key = key.strip().lower().replace(' ', '_')
                        value = value.strip()

                        numeric_keys = {
                            'sample_rate',
                            'duration',
                            'channels',
                            'samples',
                            'n_channels',
                            'total_samples'
                        }

                        # Try to convert to appropriate type
                        try:
                            if '.' in value:
                                value = float(value)
                            else:
                                value = int(value)
                        except ValueError:
                            if key in numeric_keys:
                                num_match = re.search(
                                    r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?",
                                    value
                                )
                                if num_match:
                                    value = float(num_match.group(0))

                        metadata[key] = value

        except Exception as e:
            # If we can't parse metadata, return empty dict
            pass

        return metadata

    @staticmethod
    def _count_header_rows(filepath: Path) -> int:
        """Count number of header rows in CSV file."""
        count = 0
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    if line.startswith('#') or line.strip() == '':
                        count += 1
                    else:
                        # First non-comment line is header
                        count += 1
                        break
        except:
            count = 1  # Assume at least one header row

        return count

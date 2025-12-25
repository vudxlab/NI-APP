"""
HDF5 Exporter for saving data to HDF5 format.

This module handles exporting acceleration data to HDF5 files
with hierarchical structure and metadata.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime

try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False
    print("Warning: h5py not available. Install with: pip install h5py")

from ..utils.logger import get_logger
from ..utils.constants import ExportDefaults
from ..daq.daq_config import DAQConfig


class HDF5Exporter:
    """Export data to HDF5 format."""

    def __init__(self):
        """Initialize the HDF5 exporter."""
        self.logger = get_logger(__name__)

        if not H5PY_AVAILABLE:
            self.logger.error("h5py not available")

    def export(
        self,
        filepath: str,
        data: np.ndarray,
        sample_rate: float,
        channel_names: List[str],
        channel_units: List[str],
        config: Optional[DAQConfig] = None,
        metadata: Optional[Dict[str, Any]] = None,
        compression: Optional[str] = ExportDefaults.HDF5_COMPRESSION,
        compression_level: int = ExportDefaults.HDF5_COMPRESSION_LEVEL
    ) -> int:
        """
        Export data to HDF5 file.

        Args:
            filepath: Path to output HDF5 file
            data: Data array of shape (n_channels, n_samples)
            sample_rate: Sampling rate in Hz
            channel_names: List of channel names
            channel_units: List of channel units
            config: Optional DAQ configuration
            metadata: Optional additional metadata
            compression: Compression algorithm ('gzip', 'lzf', etc.)
            compression_level: Compression level (0-9 for gzip)

        Returns:
            Number of samples written

        Raises:
            IOError: If file cannot be written
        """
        if not H5PY_AVAILABLE:
            raise ImportError("h5py not available. Install with: pip install h5py")

        self.logger.info(f"Exporting to HDF5: {filepath}")

        try:
            n_channels, n_samples = data.shape

            with h5py.File(filepath, 'w') as f:
                # Create main group
                main_group = f.create_group('data')
                main_group.attrs['description'] = 'NI DAQ Acceleration Data'

                # Store each channel as a dataset
                for i, (name, units) in enumerate(zip(channel_names, channel_units)):
                    channel_name = name.replace(' ', '_').replace('/', '_')
                    dataset_path = f'data/channel_{i:02d}_{channel_name}'

                    # Create dataset
                    if compression:
                        dataset = f.create_dataset(
                            dataset_path,
                            data=data[i, :],
                            compression=compression,
                            compression_opts=compression_level
                        )
                    else:
                        dataset = f.create_dataset(
                            dataset_path,
                            data=data[i, :]
                        )

                    # Add attributes
                    dataset.attrs['channel_name'] = name
                    dataset.attrs['units'] = units
                    dataset.attrs['sample_rate'] = sample_rate
                    dataset.attrs['n_samples'] = n_samples

                # Store metadata
                meta_group = f.create_group('metadata')

                # Export info
                export_attrs = {
                    'export_date': datetime.now().isoformat(),
                    'sample_rate': sample_rate,
                    'n_channels': n_channels,
                    'n_samples': n_samples,
                    'duration_seconds': n_samples / sample_rate
                }

                for key, value in export_attrs.items():
                    meta_group.attrs[key] = value

                # DAQ configuration
                if config:
                    config_group = f.create_group('metadata/daq_config')
                    config_group.attrs['device_name'] = config.device_name
                    config_group.attrs['sample_rate'] = config.sample_rate
                    config_group.attrs['acquisition_mode'] = config.acquisition_mode
                    config_group.attrs['buffer_size_seconds'] = config.buffer_size_seconds

                    # Channel configurations
                    for i, ch in enumerate(config.channels):
                        ch_group = f.create_group(f'metadata/daq_config/channel_{i:02d}')
                        ch_group.attrs['name'] = ch.name
                        ch_group.attrs['physical_channel'] = ch.physical_channel
                        ch_group.attrs['units'] = ch.units
                        ch_group.attrs['sensitivity'] = ch.sensitivity
                        ch_group.attrs['iepe_enabled'] = ch.iepe_enabled
                        ch_group.attrs['coupling'] = ch.coupling

                # Additional metadata
                if metadata:
                    extra_group = f.create_group('metadata/extra')
                    for key, value in metadata.items():
                        if isinstance(value, (str, int, float, bool)):
                            extra_group.attrs[key] = value
                        else:
                            # Store as dataset if not a scalar
                            extra_group.create_dataset(key, data=value)

            self.logger.info(f"HDF5 export complete: {n_samples} samples written")
            return n_samples

        except Exception as e:
            self.logger.error(f"HDF5 export failed: {e}")
            raise

    def read_channel(
        self,
        filepath: str,
        channel_idx: int = 0
    ) -> tuple:
        """
        Read a specific channel from HDF5 file.

        Args:
            filepath: Path to HDF5 file
            channel_idx: Channel index to read

        Returns:
            Tuple of (data, attributes)
        """
        if not H5PY_AVAILABLE:
            raise ImportError("h5py not available")

        try:
            with h5py.File(filepath, 'r') as f:
                # Find the channel dataset
                channel_keys = [k for k in f['data'].keys() if k.startswith('channel_')]

                if channel_idx >= len(channel_keys):
                    raise ValueError(f"Channel {channel_idx} not found")

                dataset_path = f'data/{channel_keys[channel_idx]}'
                dataset = f[dataset_path]

                data = dataset[:]
                attrs = dict(dataset.attrs)

                return data, attrs

        except Exception as e:
            self.logger.error(f"HDF5 read failed: {e}")
            raise

    def get_info(self, filepath: str) -> Dict[str, Any]:
        """
        Get information about an HDF5 file.

        Args:
            filepath: Path to HDF5 file

        Returns:
            Dictionary with file information
        """
        if not H5PY_AVAILABLE:
            raise ImportError("h5py not available")

        try:
            with h5py.File(filepath, 'r') as f:
                info = {
                    'n_channels': len([k for k in f['data'].keys() if k.startswith('channel_')]),
                    'groups': list(f.keys()),
                }

                # Get metadata if available
                if 'metadata' in f:
                    for key in f['metadata'].attrs:
                        info[key] = f['metadata'].attrs[key]

                return info

        except Exception as e:
            self.logger.error(f"Failed to read HDF5 info: {e}")
            raise

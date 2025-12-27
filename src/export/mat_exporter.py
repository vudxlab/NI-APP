"""
MATLAB .mat Exporter for saving data to .mat format.

This module handles exporting acceleration data to MATLAB .mat files
with simple structure compatible with MATLAB/Octave.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

try:
    import scipy.io
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available. Install with: pip install scipy")

from ..utils.logger import get_logger
from ..daq.daq_config import DAQConfig


class MATExporter:
    """Export data to MATLAB .mat format."""

    def __init__(self):
        """Initialize the MAT exporter."""
        self.logger = get_logger(__name__)

        if not SCIPY_AVAILABLE:
            self.logger.error("scipy not available")

    def export(
        self,
        filepath: str,
        data: np.ndarray,
        sample_rate: float,
        channel_names: List[str],
        channel_units: List[str],
        config: Optional[DAQConfig] = None,
        metadata: Optional[Dict[str, Any]] = None,
        array_name: Optional[str] = None
    ) -> int:
        """
        Export data to MATLAB .mat file.

        Args:
            filepath: Path to output .mat file
            data: Data array of shape (n_channels, n_samples)
            sample_rate: Sampling rate in Hz
            channel_names: List of channel names
            channel_units: List of channel units
            config: Optional DAQ configuration
            metadata: Optional additional metadata
            array_name: Name for main data array (defaults to filename stem)

        Returns:
            Number of samples written

        Raises:
            IOError: If file cannot be written
        """
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy not available. Install with: pip install scipy")

        self.logger.info(f"Exporting to MAT: {filepath}")

        try:
            n_channels, n_samples = data.shape

            # Determine array name from filename if not provided
            if array_name is None:
                array_name = Path(filepath).stem

            # Transpose to MATLAB convention: (n_samples, n_channels)
            data_matlab = data.T

            # Build .mat dictionary
            mat_dict = {
                array_name: data_matlab,
                'channel_names': channel_names,
                'channel_units': channel_units,
                'sample_rate': sample_rate,
                'export_time': datetime.now().isoformat()
            }

            # Add optional metadata
            if metadata:
                # Filter out non-serializable metadata
                for key, value in metadata.items():
                    if isinstance(value, (int, float, str, bool, list, tuple)):
                        mat_dict[f'meta_{key}'] = value

            # Add DAQ config if provided
            if config:
                mat_dict['daq_sample_rate'] = config.sample_rate
                mat_dict['daq_n_channels'] = config.get_num_enabled_channels()
                mat_dict['daq_device'] = config.device_name

            # Write .mat file
            scipy.io.savemat(
                filepath,
                mat_dict,
                do_compression=True,
                oned_as='column'
            )

            self.logger.info(
                f"MAT export complete: {filepath} "
                f"({n_samples} samples, {n_channels} channels)"
            )

            return n_samples

        except Exception as e:
            error_msg = f"Failed to export MAT file: {e}"
            self.logger.error(error_msg)
            raise IOError(error_msg) from e

    def get_info(self, filepath: str) -> Dict[str, Any]:
        """
        Get information about a MAT file.

        Args:
            filepath: Path to .mat file

        Returns:
            Dictionary with file information
        """
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy not available")

        try:
            # Load .mat file
            mat_contents = scipy.io.loadmat(str(filepath))

            info = {}

            # Find main data array (ignore metadata keys)
            metadata_keys = {'__header__', '__version__', '__globals__'}
            data_keys = [k for k in mat_contents.keys() if k not in metadata_keys]

            if data_keys:
                # Use first data array
                main_key = data_keys[0]
                data_array = mat_contents[main_key]

                if data_array.ndim == 2:
                    n_samples, n_channels = data_array.shape
                    info['total_samples'] = n_samples
                    info['n_channels'] = n_channels
                    info['array_name'] = main_key

            # Extract metadata
            if 'sample_rate' in mat_contents:
                info['sample_rate'] = float(mat_contents['sample_rate'].flat[0])
                if 'total_samples' in info:
                    info['duration_seconds'] = info['total_samples'] / info['sample_rate']

            if 'channel_names' in mat_contents:
                info['has_channel_names'] = True

            if 'channel_units' in mat_contents:
                info['has_channel_units'] = True

            return info

        except Exception as e:
            self.logger.error(f"Failed to read MAT info: {e}")
            raise

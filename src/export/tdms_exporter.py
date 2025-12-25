"""
TDMS Exporter for saving data to TDMS format.

This module handles exporting acceleration data to TDMS files
(NI's native file format compatible with LabVIEW and DIAdem).
"""

import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime

try:
    from nptdms import TdmsFile, RootObject, Group, Channel
    from nptdms.types import Int32Channel, Int64Channel, DoubleChannel
    NPTDMS_AVAILABLE = True
except ImportError:
    NPTDMS_AVAILABLE = False
    print("Warning: nptdms not available. Install with: pip install nptdms")

from ..utils.logger import get_logger
from ..daq.daq_config import DAQConfig


class TDMSExporter:
    """Export data to TDMS format."""

    def __init__(self):
        """Initialize the TDMS exporter."""
        self.logger = get_logger(__name__)

        if not NPTDMS_AVAILABLE:
            self.logger.error("nptdms not available")

    def export(
        self,
        filepath: str,
        data: np.ndarray,
        sample_rate: float,
        channel_names: List[str],
        channel_units: List[str],
        config: Optional[DAQConfig] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Export data to TDMS file.

        Args:
            filepath: Path to output TDMS file
            data: Data array of shape (n_channels, n_samples)
            sample_rate: Sampling rate in Hz
            channel_names: List of channel names
            channel_units: List of channel units
            config: Optional DAQ configuration
            metadata: Optional additional metadata

        Returns:
            Number of samples written

        Raises:
            IOError: If file cannot be written
        """
        if not NPTDMS_AVAILABLE:
            raise ImportError("nptdms not available. Install with: pip install nptdms")

        self.logger.info(f"Exporting to TDMS: {filepath}")

        try:
            n_channels, n_samples = data.shape

            # Create TDMS file
            tdms_file = TdmsFile(filepath)

            # Create root properties
            root = tdms_file.root_object
            root.properties = {
                'description': 'NI DAQ Acceleration Data',
                'export_date': datetime.now().isoformat(),
                'format_version': '1.0'
            }

            # Add metadata to root
            if metadata:
                for key, value in metadata.items():
                    root.properties[key] = value

            # Create a group for the data
            group_name = "Acceleration_Data"
            group = Group(group_name)
            group.properties = {
                'description': 'Acceleration measurements',
                'sample_rate': sample_rate,
                'n_channels': n_channels
            }
            tdms_file.groups.append(group)

            # Add each channel
            for i, (name, units) in enumerate(zip(channel_names, channel_units)):
                # Sanitize channel name for TDMS
                channel_name = name.replace(' ', '_').replace('/', '_')
                full_path = f"/{group_name}/{channel_name}"

                # Create channel
                channel = Channel(
                    group_name,
                    channel_name,
                    data[i, :]
                )
                channel.properties = {
                    'unit_string': units,
                    'channel_index': i,
                    'description': f'Channel {i+1}: {name}'
                }
                tdms_file.channels.append(channel)

            # Add DAQ configuration as a group
            if config:
                config_group = Group("DAQ_Configuration")
                config_group.properties = {
                    'device_name': config.device_name,
                    'sample_rate': config.sample_rate,
                    'acquisition_mode': config.acquisition_mode,
                    'buffer_size_seconds': config.buffer_size_seconds
                }
                tdms_file.groups.append(config_group)

                # Add channel configurations
                for i, ch in enumerate(config.channels):
                    ch_name = ch.name.replace(' ', '_').replace('/', '_')
                    config_channel = Channel(
                        "DAQ_Configuration",
                        f"Channel_{i:02d}_{ch_name}"
                    )
                    config_channel.properties = {
                        'name': ch.name,
                        'physical_channel': ch.physical_channel,
                        'units': ch.units,
                        'sensitivity_mV_g': ch.sensitivity,
                        'iepe_enabled': ch.iepe_enabled,
                        'coupling': ch.coupling,
                        'input_range': ch.input_range
                    }
                    tdms_file.channels.append(config_channel)

            # Write file
            tdms_file.write()

            self.logger.info(f"TDMS export complete: {n_samples} samples written")
            return n_samples

        except Exception as e:
            self.logger.error(f"TDMS export failed: {e}")
            raise

    def read_channel(
        self,
        filepath: str,
        channel_name: str,
        group_name: str = "Acceleration_Data"
    ) -> tuple:
        """
        Read a specific channel from TDMS file.

        Args:
            filepath: Path to TDMS file
            channel_name: Name of the channel to read
            group_name: Name of the group containing the channel

        Returns:
            Tuple of (data, properties)
        """
        if not NPTDMS_AVAILABLE:
            raise ImportError("nptdms not available")

        try:
            tdms_file = TdmsFile(filepath)
            tdms_file.read()

            channel = tdms_file.object(group_name, channel_name)

            if channel is None:
                raise ValueError(f"Channel {group_name}/{channel_name} not found")

            return channel.data, channel.properties

        except Exception as e:
            self.logger.error(f"TDMS read failed: {e}")
            raise

    def get_info(self, filepath: str) -> Dict[str, Any]:
        """
        Get information about a TDMS file.

        Args:
            filepath: Path to TDMS file

        Returns:
            Dictionary with file information
        """
        if not NPTDMS_AVAILABLE:
            raise ImportError("nptdms not available")

        try:
            tdms_file = TdmsFile(filepath)
            tdms_file.read()

            info = {
                'groups': tdms_file.groups,
                'n_channels': len(tdms_file.channels)
            }

            # Get root properties
            info['properties'] = dict(tdms_file.root_object.properties)

            return info

        except Exception as e:
            self.logger.error(f"Failed to read TDMS info: {e}")
            raise

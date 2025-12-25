"""
CSV Exporter for saving data to CSV format.

This module handles exporting acceleration data to CSV files
with proper headers and metadata.
"""

import numpy as np
import csv
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from ..utils.logger import get_logger
from ..daq.daq_config import DAQConfig


class CSVExporter:
    """Export data to CSV format."""

    def __init__(self):
        """Initialize the CSV exporter."""
        self.logger = get_logger(__name__)

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
        Export data to CSV file.

        Args:
            filepath: Path to output CSV file
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
        self.logger.info(f"Exporting to CSV: {filepath}")

        try:
            n_channels, n_samples = data.shape

            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)

                # Write metadata as comments
                f.write(f"# NI DAQ Vibration Analysis Export\n")
                f.write(f"# Export Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"# Sample Rate: {sample_rate} Hz\n")
                f.write(f"# Channels: {n_channels}\n")
                f.write(f"# Samples: {n_samples}\n")
                f.write(f"# Duration: {n_samples / sample_rate:.3f} seconds\n")

                if config:
                    f.write(f"# Device: {config.device_name}\n")
                    f.write(f"# Acquisition Mode: {config.acquisition_mode}\n")

                if metadata:
                    for key, value in metadata.items():
                        f.write(f"# {key}: {value}\n")

                f.write("#\n")

                # Write header
                header = ['Time (s)'] + [f"{name} ({units})" for name, units in zip(channel_names, channel_units)]
                writer.writerow(header)

                # Write data
                time_axis = np.arange(n_samples) / sample_rate

                # Transpose data for row-major output (time, channels)
                data_transposed = data.T

                for i in range(n_samples):
                    row = [time_axis[i]] + data_transposed[i, :].tolist()
                    writer.writerow(row)

            self.logger.info(f"CSV export complete: {n_samples} samples written")
            return n_samples

        except Exception as e:
            self.logger.error(f"CSV export failed: {e}")
            raise

    def export_chunked(
        self,
        filepath: str,
        data_generator,
        sample_rate: float,
        channel_names: List[str],
        channel_units: List[str],
        chunk_size: int = 10000,
        config: Optional[DAQConfig] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Export large datasets in chunks to manage memory.

        Args:
            filepath: Path to output CSV file
            data_generator: Generator yielding (n_channels, n_samples) arrays
            sample_rate: Sampling rate in Hz
            channel_names: List of channel names
            channel_units: List of channel units
            chunk_size: Number of samples per chunk
            config: Optional DAQ configuration
            metadata: Optional additional metadata

        Returns:
            Total number of samples written
        """
        self.logger.info(f"Exporting to CSV (chunked): {filepath}")

        try:
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)

                # Write metadata header
                f.write(f"# NI DAQ Vibration Analysis Export (Chunked)\n")
                f.write(f"# Export Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"# Sample Rate: {sample_rate} Hz\n")

                if config:
                    f.write(f"# Device: {config.device_name}\n")

                f.write("#\n")

                # Write header
                header = ['Time (s)'] + [f"{name} ({units})" for name, units in zip(channel_names, channel_units)]
                writer.writerow(header)

                # Write data in chunks
                total_samples = 0
                sample_offset = 0

                for chunk_data in data_generator:
                    n_channels, n_samples = chunk_data.shape
                    time_axis = (sample_offset + np.arange(n_samples)) / sample_rate
                    data_transposed = chunk_data.T

                    for i in range(n_samples):
                        row = [time_axis[i]] + data_transposed[i, :].tolist()
                        writer.writerow(row)

                    total_samples += n_samples
                    sample_offset += n_samples
                    self.logger.debug(f"Written {total_samples} samples...")

            self.logger.info(f"CSV export complete: {total_samples} samples written")
            return total_samples

        except Exception as e:
            self.logger.error(f"CSV export failed: {e}")
            raise

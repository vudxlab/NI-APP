"""
Export Manager for coordinating data exports.

This module manages the export process, supporting multiple formats
and running exports in background threads.
"""

from typing import Optional, Callable
from pathlib import Path
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal, QObject

from .csv_exporter import CSVExporter
from .hdf5_exporter import HDF5Exporter
from .tdms_exporter import TDMSExporter
from ..utils.logger import get_logger
from ..utils.constants import ExportDefaults
from ..daq.daq_config import DAQConfig


class ExportWorker(QThread):
    """
    Background thread for exporting data.

    Runs export operations in a background thread to prevent
    GUI blocking during large file writes.
    """

    # Signals
    progress = pyqtSignal(int)  # Samples written
    finished = pyqtSignal(bool, str)  # Success/failure, message
    error = pyqtSignal(str)  # Error message

    def __init__(self, exporter, filepath: str, data: np.ndarray,
                 sample_rate: float, channel_names: list, channel_units: list,
                 config: Optional[DAQConfig] = None, metadata: Optional[dict] = None):
        """
        Initialize export worker.

        Args:
            exporter: Exporter instance (CSV, HDF5, or TDMS)
            filepath: Output file path
            data: Data array
            sample_rate: Sampling rate
            channel_names: Channel names
            channel_units: Channel units
            config: DAQ configuration
            metadata: Additional metadata
        """
        super().__init__()
        self.exporter = exporter
        self.filepath = filepath
        self.data = data
        self.sample_rate = sample_rate
        self.channel_names = channel_names
        self.channel_units = channel_units
        self.config = config
        self.metadata = metadata

    def run(self):
        """Execute export in background thread."""
        try:
            samples_written = self.exporter.export(
                self.filepath,
                self.data,
                self.sample_rate,
                self.channel_names,
                self.channel_units,
                self.config,
                self.metadata
            )

            self.progress.emit(samples_written)
            self.finished.emit(True, f"Export complete: {samples_written:,} samples written")

        except Exception as e:
            self.error.emit(str(e))
            self.finished.emit(False, f"Export failed: {e}")


class ExportManager(QObject):
    """
    Manager for data export operations.

    Coordinates export operations across multiple formats,
    runs exports in background threads, and handles progress reporting.
    """

    # Signals
    export_started = pyqtSignal(str)  # File path
    export_progress = pyqtSignal(int, int)  # Current, total
    export_finished = pyqtSignal(bool, str)  # Success, message
    export_error = pyqtSignal(str)  # Error message

    def __init__(self):
        """Initialize the export manager."""
        super().__init__()

        self.logger = get_logger(__name__)
        self.current_worker: Optional[ExportWorker] = None

        # Create exporters
        self.csv_exporter = CSVExporter()
        self.hdf5_exporter = HDF5Exporter()
        self.tdms_exporter = TDMSExporter()

        self.logger.info("ExportManager initialized")

    def export(
        self,
        filepath: str,
        format: str,
        data: np.ndarray,
        sample_rate: float,
        channel_names: list,
        channel_units: list,
        config: Optional[DAQConfig] = None,
        metadata: Optional[dict] = None,
        asynchronous: bool = True
    ) -> int:
        """
        Export data to file.

        Args:
            filepath: Output file path
            format: Export format ('csv', 'hdf5', 'tdms')
            data: Data array of shape (n_channels, n_samples)
            sample_rate: Sampling rate in Hz
            channel_names: List of channel names
            channel_units: List of channel units
            config: Optional DAQ configuration
            metadata: Optional additional metadata
            asynchronous: Run in background thread if True

        Returns:
            Number of samples to write (0 if asynchronous)

        Raises:
            ValueError: If format is invalid
            IOError: If export fails
        """
        # Select exporter
        if format == ExportDefaults.FORMAT_CSV:
            exporter = self.csv_exporter
        elif format == ExportDefaults.FORMAT_HDF5:
            exporter = self.hdf5_exporter
        elif format == ExportDefaults.FORMAT_TDMS:
            exporter = self.tdms_exporter
        else:
            raise ValueError(f"Unsupported format: {format}")

        n_samples = data.shape[1]

        if asynchronous:
            # Run in background thread
            self.current_worker = ExportWorker(
                exporter, filepath, data, sample_rate,
                channel_names, channel_units, config, metadata
            )

            # Connect signals
            self.current_worker.progress.connect(lambda n: self.export_progress.emit(n, n_samples))
            self.current_worker.finished.connect(self.export_finished)
            self.current_worker.error.connect(self.export_error)

            # Start thread
            self.current_worker.start()
            self.export_started.emit(filepath)

            self.logger.info(f"Export started (async): {filepath}")
            return 0

        else:
            # Run synchronously
            try:
                samples_written = exporter.export(
                    filepath, data, sample_rate,
                    channel_names, channel_units,
                    config, metadata
                )
                self.export_finished.emit(True, f"Export complete: {samples_written:,} samples")
                return samples_written

            except Exception as e:
                self.export_error.emit(str(e))
                self.export_finished.emit(False, f"Export failed: {e}")
                raise

    def cancel_export(self) -> None:
        """Cancel the current export operation."""
        if self.current_worker and self.current_worker.isRunning():
            self.current_worker.terminate()
            self.current_worker.wait()
            self.logger.info("Export cancelled")

    def is_exporting(self) -> bool:
        """
        Check if an export is currently in progress.

        Returns:
            True if export is running
        """
        return self.current_worker and self.current_worker.isRunning()

    def get_supported_formats(self) -> list:
        """
        Get list of supported export formats.

        Returns:
            List of format identifiers
        """
        return ExportDefaults.SUPPORTED_FORMATS

    def get_format_extension(self, format: str) -> str:
        """
        Get file extension for a format.

        Args:
            format: Format identifier

        Returns:
            File extension (with dot)
        """
        return ExportDefaults.FILE_EXTENSIONS.get(format, '.dat')

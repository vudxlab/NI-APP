"""
Acquisition Thread for continuous DAQ data collection.

This module implements a QThread that runs in the background to continuously
acquire data from the DAQ hardware without blocking the GUI.
"""

import time
import errno
import numpy as np
from typing import Optional
from pathlib import Path
from datetime import datetime
from PyQt5.QtCore import QThread, pyqtSignal

from .daq_manager import DAQManager, DAQError
from .daq_config import DAQConfig
from ..utils.logger import get_logger
from ..utils.constants import DAQDefaults, ErrorMessages, LogMessages
from ..export.streaming_data_writer import create_streaming_writer, StreamingWriter


class AcquisitionThread(QThread):
    """
    Background thread for continuous DAQ acquisition.

    This thread runs the acquisition loop in the background, continuously
    reading data from the DAQ and emitting signals with the data.

    Signals:
        data_ready: Emitted when new data is available (timestamp, data, scaled_data)
        error_occurred: Emitted when an error occurs (error_message)
        acquisition_started: Emitted when acquisition starts successfully
        acquisition_stopped: Emitted when acquisition stops
        status_update: Emitted periodically with acquisition statistics
    """

    # Qt signals for thread-safe communication
    data_ready = pyqtSignal(float, np.ndarray, np.ndarray)  # timestamp, raw_data, scaled_data
    error_occurred = pyqtSignal(str)  # error_message
    acquisition_started = pyqtSignal()
    acquisition_stopped = pyqtSignal()
    status_update = pyqtSignal(dict)  # statistics
    save_file_created = pyqtSignal(str)  # filepath
    save_file_closed = pyqtSignal(str)  # filepath
    save_error = pyqtSignal(str)  # error_message

    def __init__(self, daq_manager: DAQManager, config: DAQConfig, autosave_config=None, parent=None):
        """
        Initialize the acquisition thread.

        Args:
            daq_manager: DAQManager instance
            config: DAQ configuration
            autosave_config: Auto-save configuration dictionary (optional)
            parent: Parent QObject
        """
        super().__init__(parent)

        self.logger = get_logger(__name__)

        self.daq_manager = daq_manager
        self.config = config
        self.autosave_config = autosave_config

        # Control flags
        self._stop_requested = False
        self._pause_requested = False
        self._running = False

        # Statistics
        self._samples_acquired = 0
        self._errors_count = 0
        self._start_time = 0
        self._last_status_update = 0
        self._status_update_interval = 1.0  # seconds

        # Error handling
        self._max_consecutive_errors = 10
        self._consecutive_errors = 0
        self._retry_delay = 0.1  # seconds

        # Auto-save attributes
        self.streaming_writer: Optional[StreamingWriter] = None
        self.current_save_file: Optional[Path] = None
        self.samples_since_flush = 0
        self.flush_interval_samples = 256000  # ~10 seconds @ 25.6 kHz

        self.logger.info("AcquisitionThread created")

    def run(self):
        """
        Main acquisition loop (runs in background thread).

        This method is called when the thread starts. It continuously
        reads data from the DAQ until stopped.
        """
        self.logger.info("Acquisition thread starting...")

        try:
            # Initialize
            self._stop_requested = False
            self._pause_requested = False
            self._running = True
            self._samples_acquired = 0
            self._errors_count = 0
            self._consecutive_errors = 0
            self._start_time = time.time()
            self._last_status_update = self._start_time

            # Start DAQ acquisition
            try:
                self.daq_manager.start_acquisition()
                self.acquisition_started.emit()
                self.logger.info(LogMessages.ACQUISITION_STARTED)
            except DAQError as e:
                error_msg = f"Failed to start acquisition: {e}"
                self.logger.error(error_msg)
                self.error_occurred.emit(error_msg)
                self._running = False
                return

            # Initialize streaming writer if auto-save enabled
            if self.autosave_config and self.autosave_config.get('enabled', False):
                try:
                    self._init_streaming_writer()
                except Exception as e:
                    error_msg = f"Failed to initialize streaming writer: {e}"
                    self.logger.error(error_msg)
                    self.error_occurred.emit(error_msg)
                    # Continue without auto-save

            # Main acquisition loop
            while not self._stop_requested:
                # Handle pause
                if self._pause_requested:
                    time.sleep(0.1)
                    continue

                try:
                    # Read data from DAQ
                    timestamp = time.time()

                    # Read raw data (voltage)
                    raw_data = self.daq_manager.read_samples(
                        num_samples=self.config.samples_per_channel,
                        timeout=DAQDefaults.READ_TIMEOUT
                    )

                    # Read scaled data (engineering units)
                    scaled_data = self.daq_manager.channel_manager.scale_data(raw_data)

                    # Emit data signal
                    self.data_ready.emit(timestamp, raw_data, scaled_data)

                    # Auto-save data if enabled
                    if self.streaming_writer:
                        try:
                            self.streaming_writer.append(scaled_data, timestamp)
                            self.samples_since_flush += scaled_data.shape[1]

                            # Periodic flush for CSV/TDMS
                            if self.samples_since_flush >= self.flush_interval_samples:
                                self.streaming_writer.flush()
                                self.samples_since_flush = 0

                        except OSError as e:
                            if e.errno == errno.ENOSPC:  # Disk full
                                error_msg = "Disk full - stopping acquisition"
                                self.logger.critical(error_msg)
                                self.error_occurred.emit(error_msg)
                                self._stop_requested = True
                            else:
                                error_msg = f"Save error: {e}"
                                self.logger.error(error_msg)
                                self.save_error.emit(error_msg)
                        except Exception as e:
                            error_msg = f"Save error: {e}"
                            self.logger.error(error_msg)
                            self.save_error.emit(error_msg)

                    # Update statistics
                    self._samples_acquired += raw_data.shape[1]
                    self._consecutive_errors = 0  # Reset error counter on success

                    # Emit status update periodically
                    if time.time() - self._last_status_update >= self._status_update_interval:
                        self._emit_status_update()

                except DAQError as e:
                    self._handle_acquisition_error(e)

                except Exception as e:
                    # Unexpected error
                    error_msg = f"Unexpected error in acquisition loop: {e}"
                    self.logger.error(error_msg)
                    self.error_occurred.emit(error_msg)
                    self._errors_count += 1
                    self._consecutive_errors += 1

                    # Check if too many consecutive errors
                    if self._consecutive_errors >= self._max_consecutive_errors:
                        error_msg = (
                            f"Too many consecutive errors ({self._consecutive_errors}), "
                            "stopping acquisition"
                        )
                        self.logger.critical(error_msg)
                        self.error_occurred.emit(error_msg)
                        break

                    # Brief delay before retry
                    time.sleep(self._retry_delay)

        except Exception as e:
            # Fatal error
            error_msg = f"Fatal error in acquisition thread: {e}"
            self.logger.critical(error_msg)
            self.error_occurred.emit(error_msg)

        finally:
            # Cleanup
            try:
                self.daq_manager.stop_acquisition()
                self.logger.info(LogMessages.ACQUISITION_STOPPED)
            except Exception as e:
                self.logger.error(f"Error stopping acquisition: {e}")

            # Close streaming writer if active
            if self.streaming_writer:
                self._close_streaming_writer()

            self._running = False
            self.acquisition_stopped.emit()

            # Final status update
            self._emit_status_update()

    def _handle_acquisition_error(self, error: DAQError):
        """
        Handle DAQ acquisition errors.

        Args:
            error: DAQError exception
        """
        error_msg = ErrorMessages.DAQ_ACQUISITION_ERROR.format(error)
        self.logger.error(error_msg)
        self.error_occurred.emit(error_msg)

        self._errors_count += 1
        self._consecutive_errors += 1

        # Check if too many consecutive errors
        if self._consecutive_errors >= self._max_consecutive_errors:
            error_msg = (
                f"Too many consecutive errors ({self._consecutive_errors}), "
                "stopping acquisition"
            )
            self.logger.critical(error_msg)
            self.error_occurred.emit(error_msg)
            self._stop_requested = True
        else:
            # Brief delay before retry
            time.sleep(self._retry_delay)

    def _emit_status_update(self):
        """Emit status update signal with current statistics."""
        current_time = time.time()
        elapsed_time = current_time - self._start_time

        # Calculate statistics
        if elapsed_time > 0:
            samples_per_second = self._samples_acquired / elapsed_time
        else:
            samples_per_second = 0

        expected_rate = self.config.sample_rate * self.config.get_num_enabled_channels()
        if expected_rate > 0:
            actual_rate_percentage = (samples_per_second / expected_rate) * 100
        else:
            actual_rate_percentage = 0

        status = {
            'running': self._running,
            'paused': self._pause_requested,
            'elapsed_time': elapsed_time,
            'samples_acquired': self._samples_acquired,
            'samples_per_second': samples_per_second,
            'expected_rate': expected_rate,
            'actual_rate_percentage': actual_rate_percentage,
            'errors_count': self._errors_count,
            'consecutive_errors': self._consecutive_errors,
            'channels': self.config.get_num_enabled_channels()
        }

        self.status_update.emit(status)
        self._last_status_update = current_time

    def _init_streaming_writer(self):
        """Initialize streaming writer with timestamped filename."""
        if not self.autosave_config:
            return

        # Generate timestamped filename
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_prefix = self.autosave_config.get('file_prefix', 'acquisition')
        file_format = self.autosave_config.get('file_format', 'hdf5')

        # Determine file extension
        ext_map = {'hdf5': '.h5', 'tdms': '.tdms', 'csv': '.csv'}
        file_ext = ext_map.get(file_format, '.h5')

        filename = f"{file_prefix}_{timestamp_str}{file_ext}"

        # Get save location
        save_location = Path(self.autosave_config.get('save_location', Path.home() / 'NI_DAQ_Data'))

        # Create directory if it doesn't exist
        try:
            save_location.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise RuntimeError(f"Cannot create save directory: {save_location}. Error: {e}")

        # Full file path
        self.current_save_file = save_location / filename

        # Get channel information
        channel_names = [ch.name for ch in self.config.channels if ch.enabled]
        channel_units = [ch.units for ch in self.config.channels if ch.enabled]
        n_channels = len(channel_names)

        # Create streaming writer
        compression_level = self.autosave_config.get('compression_level', 4)

        self.streaming_writer = create_streaming_writer(
            file_format=file_format,
            filepath=str(self.current_save_file),
            n_channels=n_channels,
            sample_rate=self.config.sample_rate,
            channel_names=channel_names,
            channel_units=channel_units,
            compression_level=compression_level
        )

        # Open the file
        self.streaming_writer.open()

        # Reset flush counter
        self.samples_since_flush = 0

        # Emit signal
        self.save_file_created.emit(str(self.current_save_file))

        self.logger.info(f"Streaming writer initialized: {self.current_save_file}")

    def _close_streaming_writer(self):
        """Close streaming writer and emit file path."""
        if not self.streaming_writer:
            return

        try:
            self.streaming_writer.close()

            # Emit signal with file path
            if self.current_save_file:
                self.save_file_closed.emit(str(self.current_save_file))

            self.logger.info(f"Streaming writer closed: {self.current_save_file}")

        except Exception as e:
            self.logger.error(f"Error closing streaming writer: {e}")

        finally:
            self.streaming_writer = None
            self.current_save_file = None

    def stop(self):
        """
        Request the thread to stop gracefully.

        This sets a flag that will cause the acquisition loop to exit.
        The thread should be joined (wait()) after calling this method.
        """
        self.logger.info("Stop requested for acquisition thread")
        self._stop_requested = True

    def pause(self):
        """Pause data acquisition (without stopping the thread)."""
        self.logger.info("Pause requested for acquisition thread")
        self._pause_requested = True

    def resume(self):
        """Resume data acquisition after pause."""
        self.logger.info("Resume requested for acquisition thread")
        self._pause_requested = False

    def is_running(self) -> bool:
        """
        Check if acquisition is currently running.

        Returns:
            True if acquisition is running
        """
        return self._running and not self._pause_requested

    def is_paused(self) -> bool:
        """
        Check if acquisition is paused.

        Returns:
            True if acquisition is paused
        """
        return self._pause_requested

    def get_statistics(self) -> dict:
        """
        Get current acquisition statistics.

        Returns:
            Dictionary with statistics
        """
        current_time = time.time()
        elapsed_time = current_time - self._start_time if self._running else 0

        if elapsed_time > 0:
            samples_per_second = self._samples_acquired / elapsed_time
        else:
            samples_per_second = 0

        return {
            'running': self._running,
            'paused': self._pause_requested,
            'elapsed_time': elapsed_time,
            'samples_acquired': self._samples_acquired,
            'samples_per_second': samples_per_second,
            'errors_count': self._errors_count,
            'consecutive_errors': self._consecutive_errors
        }

    def set_status_update_interval(self, interval: float):
        """
        Set the interval for status updates.

        Args:
            interval: Update interval in seconds
        """
        if interval > 0:
            self._status_update_interval = interval
            self.logger.debug(f"Status update interval set to {interval}s")

    def set_max_consecutive_errors(self, max_errors: int):
        """
        Set the maximum number of consecutive errors before stopping.

        Args:
            max_errors: Maximum consecutive errors
        """
        if max_errors > 0:
            self._max_consecutive_errors = max_errors
            self.logger.debug(f"Max consecutive errors set to {max_errors}")

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"AcquisitionThread(running={self._running}, "
            f"paused={self._pause_requested}, "
            f"samples={self._samples_acquired})"
        )


# Example usage and tests
if __name__ == "__main__":
    from PyQt5.QtCore import QCoreApplication
    from .daq_config import create_default_config
    import sys

    print("AcquisitionThread Test")
    print("=" * 60)

    # Create Qt application
    app = QCoreApplication(sys.argv)

    # Create configuration
    print("\n1. Creating DAQ configuration...")
    config = create_default_config(device_name="cDAQ1", num_modules=3)
    print(f"   Config: {config}")

    # Create DAQ manager
    print("\n2. Creating DAQ manager...")
    daq_manager = DAQManager()

    try:
        daq_manager.configure(config)
        daq_manager.create_task()
        print(f"   DAQ Manager: {daq_manager}")
    except Exception as e:
        print(f"   Configuration/Task creation: {e}")
        print("   (Running in simulation mode)")

    # Create acquisition thread
    print("\n3. Creating acquisition thread...")
    acquisition_thread = AcquisitionThread(daq_manager, config)

    # Connect signals
    data_count = [0]  # Use list to allow modification in nested function

    def on_data_ready(timestamp, raw_data, scaled_data):
        data_count[0] += 1
        if data_count[0] <= 3:  # Only print first 3
            print(f"   Data received #{data_count[0]}: "
                  f"raw shape={raw_data.shape}, "
                  f"scaled shape={scaled_data.shape}, "
                  f"timestamp={timestamp:.3f}")

    def on_error(error_msg):
        print(f"   Error: {error_msg}")

    def on_started():
        print("   Acquisition started!")

    def on_stopped():
        print("   Acquisition stopped!")

    def on_status(status):
        if status['samples_acquired'] > 0:
            print(f"   Status: {status['samples_acquired']} samples, "
                  f"{status['samples_per_second']:.1f} samples/s, "
                  f"errors={status['errors_count']}")

    acquisition_thread.data_ready.connect(on_data_ready)
    acquisition_thread.error_occurred.connect(on_error)
    acquisition_thread.acquisition_started.connect(on_started)
    acquisition_thread.acquisition_stopped.connect(on_stopped)
    acquisition_thread.status_update.connect(on_status)

    # Start acquisition
    print("\n4. Starting acquisition...")
    acquisition_thread.start()

    # Let it run for a bit
    print("\n5. Acquiring data for 2 seconds...")

    # Use a timer to stop after 2 seconds
    from PyQt5.QtCore import QTimer

    def stop_acquisition():
        print("\n6. Stopping acquisition...")
        acquisition_thread.stop()
        acquisition_thread.wait()  # Wait for thread to finish

        # Print final statistics
        print("\n7. Final statistics:")
        stats = acquisition_thread.get_statistics()
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.2f}")
            else:
                print(f"   {key}: {value}")

        # Cleanup and exit
        daq_manager.close_task()
        app.quit()

    # Stop after 2 seconds
    QTimer.singleShot(2000, stop_acquisition)

    # Run the Qt event loop
    print("\n   Running event loop...")
    sys.exit(app.exec_())

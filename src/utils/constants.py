"""
Constants and configuration values for the NI DAQ application.

This module contains hardware specifications, default values, and application constants.
"""

from enum import Enum
from typing import Dict, List


# ============================================================================
# NI-9234 Hardware Specifications
# ============================================================================

class NI9234Specs:
    """NI-9234 module specifications."""

    # Analog Input
    NUM_CHANNELS_PER_MODULE = 4
    RESOLUTION_BITS = 24
    INPUT_RANGE_VOLTS = 5.0  # ±5V
    MAX_SAMPLE_RATE = 51200  # Hz (51.2 kS/s per channel)

    # IEPE Configuration
    IEPE_EXCITATION_VOLTAGE = 24.0  # Volts
    IEPE_EXCITATION_CURRENT = 0.002  # Amps (2 mA)

    # Anti-aliasing Filter
    # Filter cutoff at 0.48 × sample rate
    ANTI_ALIAS_FILTER_RATIO = 0.48

    # Coupling
    COUPLING_MODES = ["AC", "DC"]
    DEFAULT_COUPLING = "AC"


# ============================================================================
# DAQ Configuration Defaults
# ============================================================================

class DAQDefaults:
    """Default DAQ configuration values."""

    # Sample rate (Hz) - 25.6 kHz gives usable bandwidth to ~10 kHz
    SAMPLE_RATE = 25600

    # Common sample rates for NI-9234
    COMMON_SAMPLE_RATES = [1024, 2048, 4096, 8192, 12800, 25600, 51200]

    # Buffer configuration
    SAMPLES_PER_CHANNEL = 4096  # Number of samples per read (must be >= FFT window size)
    READ_TIMEOUT = 10.0  # Seconds

    # Buffer size for circular buffer (300 seconds of data for long FFT windows)
    BUFFER_DURATION_SECONDS = 300

    # Acquisition mode
    ACQUISITION_MODE_CONTINUOUS = "continuous"
    ACQUISITION_MODE_FINITE = "finite"
    DEFAULT_ACQUISITION_MODE = ACQUISITION_MODE_CONTINUOUS


# ============================================================================
# Channel Configuration Defaults
# ============================================================================

class ChannelDefaults:
    """Default channel configuration values."""

    # IEPE
    IEPE_ENABLED = True

    # Accelerometer sensitivity (mV/g)
    ACCELEROMETER_SENSITIVITY = 100.0  # Common value for many accelerometers

    # Engineering units
    UNITS_G = "g"
    UNITS_MS2 = "m/s²"
    UNITS_MMS2 = "mm/s²"
    DEFAULT_UNITS = UNITS_G

    # Supported units
    SUPPORTED_UNITS = [UNITS_G, UNITS_MS2, UNITS_MMS2]


# ============================================================================
# Signal Processing Defaults
# ============================================================================

class ProcessingDefaults:
    """Default signal processing parameters."""

    # Filter types
    FILTER_TYPE_BUTTERWORTH = "butterworth"
    FILTER_TYPE_CHEBYSHEV1 = "chebyshev1"
    FILTER_TYPE_CHEBYSHEV2 = "chebyshev2"
    FILTER_TYPE_BESSEL = "bessel"

    FILTER_TYPES = [
        FILTER_TYPE_BUTTERWORTH,
        FILTER_TYPE_CHEBYSHEV1,
        FILTER_TYPE_CHEBYSHEV2,
        FILTER_TYPE_BESSEL
    ]

    # Filter modes
    FILTER_MODE_LOWPASS = "lowpass"
    FILTER_MODE_HIGHPASS = "highpass"
    FILTER_MODE_BANDPASS = "bandpass"
    FILTER_MODE_BANDSTOP = "bandstop"

    FILTER_MODES = [
        FILTER_MODE_LOWPASS,
        FILTER_MODE_HIGHPASS,
        FILTER_MODE_BANDPASS,
        FILTER_MODE_BANDSTOP
    ]

    # Default filter parameters
    DEFAULT_FILTER_TYPE = FILTER_TYPE_BUTTERWORTH
    DEFAULT_FILTER_ORDER = 4
    DEFAULT_LOWPASS_CUTOFF = 1000.0  # Hz
    DEFAULT_HIGHPASS_CUTOFF = 10.0  # Hz

    # FFT parameters
    # Standard window sizes for real-time display
    FFT_WINDOW_SIZES = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]

    # Long window sizes for low-frequency analysis (powers of 2)
    # At 25.6 kHz: 2^18=10.2s, 2^19=20.5s, 2^20=40.9s, 2^21=81.9s, 2^22=163.8s
    LONG_FFT_WINDOW_SIZES = [262144, 524288, 1048576, 2097152, 4194304]

    DEFAULT_FFT_WINDOW_SIZE = 2048
    DEFAULT_FFT_OVERLAP = 0.5  # 50% overlap

    # Window functions
    WINDOW_HANNING = "hann"
    WINDOW_HAMMING = "hamming"
    WINDOW_BLACKMAN = "blackman"
    WINDOW_BARTLETT = "bartlett"
    WINDOW_NONE = "boxcar"

    WINDOW_FUNCTIONS = [
        WINDOW_HANNING,
        WINDOW_HAMMING,
        WINDOW_BLACKMAN,
        WINDOW_BARTLETT,
        WINDOW_NONE
    ]

    DEFAULT_WINDOW_FUNCTION = WINDOW_HANNING

    # Peak detection
    PEAK_DETECTION_THRESHOLD = 0.1  # Relative to maximum


# ============================================================================
# GUI Configuration
# ============================================================================

class GUIDefaults:
    """Default GUI configuration values."""

    # Plot update rate
    PLOT_UPDATE_RATE_HZ = 30  # Hz
    PLOT_UPDATE_INTERVAL_MS = int(1000 / PLOT_UPDATE_RATE_HZ)

    # Time window options (seconds)
    TIME_WINDOWS = [0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
    DEFAULT_TIME_WINDOW = 5.0

    # Plot downsampling threshold (number of points)
    PLOT_DOWNSAMPLE_THRESHOLD = 1000

    # Color palette for multi-channel plots
    PLOT_COLORS = [
        '#1f77b4',  # Blue
        '#ff7f0e',  # Orange
        '#2ca02c',  # Green
        '#d62728',  # Red
        '#9467bd',  # Purple
        '#8c564b',  # Brown
        '#e377c2',  # Pink
        '#7f7f7f',  # Gray
        '#bcbd22',  # Olive
        '#17becf',  # Cyan
        '#aec7e8',  # Light blue
        '#ffbb78',  # Light orange
    ]

    # Window geometry (optimized for Full HD 1920x1080)
    DEFAULT_WINDOW_WIDTH = 1600
    DEFAULT_WINDOW_HEIGHT = 900

    # Status update interval
    STATUS_UPDATE_INTERVAL_MS = 100


# ============================================================================
# Export Configuration
# ============================================================================

class ExportDefaults:
    """Default export configuration values."""

    # File formats
    FORMAT_CSV = "csv"
    FORMAT_HDF5 = "hdf5"
    FORMAT_TDMS = "tdms"

    SUPPORTED_FORMATS = [FORMAT_CSV, FORMAT_HDF5, FORMAT_TDMS]
    DEFAULT_FORMAT = FORMAT_TDMS

    # CSV settings
    CSV_DELIMITER = ","
    CSV_DECIMAL = "."

    # HDF5 settings
    HDF5_COMPRESSION = "gzip"
    HDF5_COMPRESSION_LEVEL = 4

    # File extensions
    FILE_EXTENSIONS = {
        FORMAT_CSV: ".csv",
        FORMAT_HDF5: ".hdf5",
        FORMAT_TDMS: ".tdms"
    }


# ============================================================================
# Application Configuration
# ============================================================================

class AppConfig:
    """Application-level configuration."""

    # Application name and version
    APP_NAME = "NI DAQ Vibration Analysis"
    APP_VERSION = "1.0.0"
    ORGANIZATION_NAME = "NI-DAQ-APP"

    # Configuration file paths
    # Use project's config directory instead of user home directory
    import os
    from pathlib import Path
    
    # Get project root (4 levels up from this file: src/utils/constants.py -> project root)
    _PROJECT_ROOT = Path(__file__).parent.parent.parent
    CONFIG_DIR = str(_PROJECT_ROOT / "config")
    CONFIG_FILE = "default_config.json"

    # Thread priorities
    ACQUISITION_THREAD_PRIORITY = "high"

    # Error recovery
    MAX_ACQUISITION_RETRIES = 3
    RETRY_DELAY_SECONDS = 1.0


# ============================================================================
# Unit Conversions
# ============================================================================

class UnitConversions:
    """Unit conversion factors."""

    # Standard gravity
    STANDARD_GRAVITY = 9.80665  # m/s²

    # Conversions to m/s²
    G_TO_MS2 = STANDARD_GRAVITY
    MMS2_TO_MS2 = 0.001

    @staticmethod
    def convert_to_ms2(value: float, from_unit: str) -> float:
        """
        Convert acceleration value to m/s².

        Args:
            value: Acceleration value
            from_unit: Source unit ("g", "m/s²", "mm/s²")

        Returns:
            Acceleration in m/s²
        """
        if from_unit == ChannelDefaults.UNITS_G:
            return value * UnitConversions.G_TO_MS2
        elif from_unit == ChannelDefaults.UNITS_MMS2:
            return value * UnitConversions.MMS2_TO_MS2
        else:  # Already in m/s²
            return value

    @staticmethod
    def convert_from_ms2(value: float, to_unit: str) -> float:
        """
        Convert acceleration value from m/s².

        Args:
            value: Acceleration value in m/s²
            to_unit: Target unit ("g", "m/s²", "mm/s²")

        Returns:
            Acceleration in target unit
        """
        if to_unit == ChannelDefaults.UNITS_G:
            return value / UnitConversions.G_TO_MS2
        elif to_unit == ChannelDefaults.UNITS_MMS2:
            return value / UnitConversions.MMS2_TO_MS2
        else:  # Already in m/s²
            return value


# ============================================================================
# Error Messages
# ============================================================================

class ErrorMessages:
    """Common error messages."""

    # DAQ errors
    DAQ_NOT_FOUND = "No NI DAQ devices found. Please check connections and drivers."
    DAQ_CONFIG_ERROR = "Error configuring DAQ: {}"
    DAQ_ACQUISITION_ERROR = "Error during data acquisition: {}"
    DAQ_TIMEOUT = "DAQ read timeout. No data received."

    # Channel errors
    CHANNEL_CONFIG_ERROR = "Invalid channel configuration: {}"
    CHANNEL_RANGE_ERROR = "Channel {} exceeds input range"

    # Processing errors
    FILTER_DESIGN_ERROR = "Error designing filter: {}"
    FFT_ERROR = "Error computing FFT: {}"

    # Export errors
    EXPORT_ERROR = "Error exporting data: {}"
    FILE_WRITE_ERROR = "Error writing file {}: {}"

    # General errors
    INVALID_PARAMETER = "Invalid parameter: {}"
    NOT_IMPLEMENTED = "Feature not yet implemented: {}"


# ============================================================================
# Log Messages
# ============================================================================

class LogMessages:
    """Common log messages."""

    # Application lifecycle
    APP_STARTED = "Application started"
    APP_STOPPED = "Application stopped"

    # DAQ operations
    DAQ_CONNECTED = "Connected to DAQ device: {}"
    DAQ_DISCONNECTED = "Disconnected from DAQ device"
    ACQUISITION_STARTED = "Data acquisition started"
    ACQUISITION_STOPPED = "Data acquisition stopped"

    # Data processing
    FILTER_APPLIED = "Filter applied: {} {} order {}"
    FFT_COMPUTED = "FFT computed with window size {}"

    # Export
    EXPORT_STARTED = "Export started to {}"
    EXPORT_COMPLETED = "Export completed: {}"


if __name__ == "__main__":
    # Print some key constants for verification
    print(f"NI-9234 Specifications:")
    print(f"  Channels per module: {NI9234Specs.NUM_CHANNELS_PER_MODULE}")
    print(f"  Resolution: {NI9234Specs.RESOLUTION_BITS} bits")
    print(f"  Input range: ±{NI9234Specs.INPUT_RANGE_VOLTS}V")
    print(f"  Max sample rate: {NI9234Specs.MAX_SAMPLE_RATE} Hz")
    print(f"\nDefault sample rate: {DAQDefaults.SAMPLE_RATE} Hz")
    print(f"Common sample rates: {DAQDefaults.COMMON_SAMPLE_RATES}")
    print(f"\nPlot update rate: {GUIDefaults.PLOT_UPDATE_RATE_HZ} Hz")
    print(f"Default FFT window: {ProcessingDefaults.DEFAULT_FFT_WINDOW_SIZE}")

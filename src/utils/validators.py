"""
Input validation utilities for the NI DAQ application.

This module provides functions to validate user inputs, DAQ configurations,
and processing parameters to prevent errors and ensure data integrity.
"""

from typing import Union, List, Optional, Tuple
import os
from pathlib import Path

from .constants import (
    NI9234Specs,
    DAQDefaults,
    ChannelDefaults,
    ProcessingDefaults,
    ExportDefaults
)


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


# ============================================================================
# DAQ Configuration Validation
# ============================================================================

def validate_sample_rate(sample_rate: float) -> bool:
    """
    Validate sample rate for NI-9234.

    Args:
        sample_rate: Sample rate in Hz

    Returns:
        True if valid

    Raises:
        ValidationError: If sample rate is invalid
    """
    if not isinstance(sample_rate, (int, float)):
        raise ValidationError(f"Sample rate must be a number, got {type(sample_rate)}")

    if sample_rate <= 0:
        raise ValidationError(f"Sample rate must be positive, got {sample_rate}")

    if sample_rate > NI9234Specs.MAX_SAMPLE_RATE:
        raise ValidationError(
            f"Sample rate {sample_rate} Hz exceeds maximum "
            f"{NI9234Specs.MAX_SAMPLE_RATE} Hz for NI-9234"
        )

    return True


def validate_samples_per_channel(samples: int) -> bool:
    """
    Validate samples per channel buffer size.

    Args:
        samples: Number of samples

    Returns:
        True if valid

    Raises:
        ValidationError: If samples value is invalid
    """
    if not isinstance(samples, int):
        raise ValidationError(f"Samples must be an integer, got {type(samples)}")

    if samples <= 0:
        raise ValidationError(f"Samples must be positive, got {samples}")

    if samples > 1000000:  # Arbitrary upper limit for sanity
        raise ValidationError(f"Samples {samples} is unreasonably large (> 1M)")

    return True


def validate_channel_name(name: str) -> bool:
    """
    Validate channel name.

    Args:
        name: Channel name

    Returns:
        True if valid

    Raises:
        ValidationError: If name is invalid
    """
    if not isinstance(name, str):
        raise ValidationError(f"Channel name must be a string, got {type(name)}")

    if not name or not name.strip():
        raise ValidationError("Channel name cannot be empty")

    if len(name) > 100:
        raise ValidationError(f"Channel name too long (max 100 characters): {name}")

    return True


def validate_physical_channel(channel: str) -> bool:
    """
    Validate physical channel string format.

    Expected format: "deviceName/ai0" or "deviceNameMod1/ai0"

    Args:
        channel: Physical channel string

    Returns:
        True if valid

    Raises:
        ValidationError: If channel format is invalid
    """
    if not isinstance(channel, str):
        raise ValidationError(f"Physical channel must be a string, got {type(channel)}")

    if not channel or not channel.strip():
        raise ValidationError("Physical channel cannot be empty")

    # Check for basic format: something/ai[0-3]
    if '/' not in channel:
        raise ValidationError(f"Physical channel must contain '/': {channel}")

    parts = channel.split('/')
    if len(parts) != 2:
        raise ValidationError(f"Invalid physical channel format: {channel}")

    device, ai_channel = parts

    if not ai_channel.startswith('ai'):
        raise ValidationError(f"Channel must be analog input (ai): {channel}")

    try:
        channel_num = int(ai_channel[2:])
        if channel_num < 0:
            raise ValidationError(f"Channel number must be non-negative: {channel}")
    except ValueError:
        raise ValidationError(f"Invalid channel number in: {channel}")

    return True


def validate_coupling(coupling: str) -> bool:
    """
    Validate coupling mode.

    Args:
        coupling: Coupling mode ("AC" or "DC")

    Returns:
        True if valid

    Raises:
        ValidationError: If coupling is invalid
    """
    if not isinstance(coupling, str):
        raise ValidationError(f"Coupling must be a string, got {type(coupling)}")

    if coupling not in NI9234Specs.COUPLING_MODES:
        raise ValidationError(
            f"Invalid coupling '{coupling}', must be one of {NI9234Specs.COUPLING_MODES}"
        )

    return True


def validate_sensitivity(sensitivity: float) -> bool:
    """
    Validate accelerometer sensitivity (mV/g).

    Args:
        sensitivity: Sensitivity in mV/g

    Returns:
        True if valid

    Raises:
        ValidationError: If sensitivity is invalid
    """
    if not isinstance(sensitivity, (int, float)):
        raise ValidationError(f"Sensitivity must be a number, got {type(sensitivity)}")

    if sensitivity <= 0:
        raise ValidationError(f"Sensitivity must be positive, got {sensitivity}")

    # Reasonable range: 1 mV/g to 10000 mV/g (10 V/g)
    if sensitivity < 1.0 or sensitivity > 10000.0:
        raise ValidationError(
            f"Sensitivity {sensitivity} mV/g is outside reasonable range (1-10000 mV/g)"
        )

    return True


def validate_units(units: str) -> bool:
    """
    Validate engineering units.

    Args:
        units: Engineering units

    Returns:
        True if valid

    Raises:
        ValidationError: If units are invalid
    """
    if not isinstance(units, str):
        raise ValidationError(f"Units must be a string, got {type(units)}")

    if units not in ChannelDefaults.SUPPORTED_UNITS:
        raise ValidationError(
            f"Invalid units '{units}', must be one of {ChannelDefaults.SUPPORTED_UNITS}"
        )

    return True


# ============================================================================
# Signal Processing Validation
# ============================================================================

def validate_filter_type(filter_type: str) -> bool:
    """
    Validate filter type.

    Args:
        filter_type: Filter type

    Returns:
        True if valid

    Raises:
        ValidationError: If filter type is invalid
    """
    if not isinstance(filter_type, str):
        raise ValidationError(f"Filter type must be a string, got {type(filter_type)}")

    if filter_type not in ProcessingDefaults.FILTER_TYPES:
        raise ValidationError(
            f"Invalid filter type '{filter_type}', "
            f"must be one of {ProcessingDefaults.FILTER_TYPES}"
        )

    return True


def validate_filter_mode(filter_mode: str) -> bool:
    """
    Validate filter mode.

    Args:
        filter_mode: Filter mode

    Returns:
        True if valid

    Raises:
        ValidationError: If filter mode is invalid
    """
    if not isinstance(filter_mode, str):
        raise ValidationError(f"Filter mode must be a string, got {type(filter_mode)}")

    if filter_mode not in ProcessingDefaults.FILTER_MODES:
        raise ValidationError(
            f"Invalid filter mode '{filter_mode}', "
            f"must be one of {ProcessingDefaults.FILTER_MODES}"
        )

    return True


def validate_cutoff_frequency(
    cutoff: Union[float, Tuple[float, float]],
    sample_rate: float,
    filter_mode: str
) -> bool:
    """
    Validate filter cutoff frequency.

    Args:
        cutoff: Cutoff frequency (Hz) or tuple of (low, high) for bandpass/stop
        sample_rate: Sample rate in Hz
        filter_mode: Filter mode

    Returns:
        True if valid

    Raises:
        ValidationError: If cutoff is invalid
    """
    nyquist = sample_rate / 2.0

    # Bandpass and bandstop require two cutoff frequencies
    if filter_mode in [ProcessingDefaults.FILTER_MODE_BANDPASS,
                       ProcessingDefaults.FILTER_MODE_BANDSTOP]:
        if not isinstance(cutoff, (list, tuple)) or len(cutoff) != 2:
            raise ValidationError(
                f"Filter mode '{filter_mode}' requires two cutoff frequencies"
            )

        low, high = cutoff

        if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
            raise ValidationError("Cutoff frequencies must be numbers")

        if low <= 0 or high <= 0:
            raise ValidationError("Cutoff frequencies must be positive")

        if low >= high:
            raise ValidationError(
                f"Low cutoff ({low}) must be less than high cutoff ({high})"
            )

        if high >= nyquist:
            raise ValidationError(
                f"High cutoff {high} Hz must be less than Nyquist frequency {nyquist} Hz"
            )

    else:
        # Lowpass and highpass require single cutoff
        if not isinstance(cutoff, (int, float)):
            raise ValidationError("Cutoff frequency must be a number")

        if cutoff <= 0:
            raise ValidationError(f"Cutoff frequency must be positive, got {cutoff}")

        if cutoff >= nyquist:
            raise ValidationError(
                f"Cutoff {cutoff} Hz must be less than Nyquist frequency {nyquist} Hz"
            )

    return True


def validate_filter_order(order: int) -> bool:
    """
    Validate filter order.

    Args:
        order: Filter order

    Returns:
        True if valid

    Raises:
        ValidationError: If order is invalid
    """
    if not isinstance(order, int):
        raise ValidationError(f"Filter order must be an integer, got {type(order)}")

    if order < 1:
        raise ValidationError(f"Filter order must be at least 1, got {order}")

    if order > 20:
        raise ValidationError(
            f"Filter order {order} is too high (max 20 for numerical stability)"
        )

    return True


def validate_fft_window_size(window_size: int, sample_rate: float) -> bool:
    """
    Validate FFT window size.

    Args:
        window_size: Number of samples in FFT window
        sample_rate: Sample rate in Hz

    Returns:
        True if valid

    Raises:
        ValidationError: If window size is invalid
    """
    if not isinstance(window_size, int):
        raise ValidationError(f"Window size must be an integer, got {type(window_size)}")

    if window_size <= 0:
        raise ValidationError(f"Window size must be positive, got {window_size}")

    # Check if power of 2 for efficient FFT
    if window_size & (window_size - 1) != 0:
        raise ValidationError(
            f"Window size {window_size} should be a power of 2 for efficient FFT"
        )

    if window_size < 64:
        raise ValidationError(f"Window size {window_size} is too small (min 64)")

    # Increased limit for long-window FFT analysis (low-frequency analysis)
    # Max: 2^24 = 16,777,216 samples (~655 seconds @ 25.6 kHz)
    if window_size > 16777216:
        raise ValidationError(f"Window size {window_size} is too large (max 16,777,216)")

    return True


def validate_window_function(window: str) -> bool:
    """
    Validate window function name.

    Args:
        window: Window function name

    Returns:
        True if valid

    Raises:
        ValidationError: If window function is invalid
    """
    if not isinstance(window, str):
        raise ValidationError(f"Window function must be a string, got {type(window)}")

    if window not in ProcessingDefaults.WINDOW_FUNCTIONS:
        raise ValidationError(
            f"Invalid window function '{window}', "
            f"must be one of {ProcessingDefaults.WINDOW_FUNCTIONS}"
        )

    return True


# ============================================================================
# Export Validation
# ============================================================================

def validate_export_format(format_: str) -> bool:
    """
    Validate export file format.

    Args:
        format_: File format

    Returns:
        True if valid

    Raises:
        ValidationError: If format is invalid
    """
    if not isinstance(format_, str):
        raise ValidationError(f"Format must be a string, got {type(format_)}")

    if format_ not in ExportDefaults.SUPPORTED_FORMATS:
        raise ValidationError(
            f"Invalid format '{format_}', "
            f"must be one of {ExportDefaults.SUPPORTED_FORMATS}"
        )

    return True


def validate_file_path(file_path: str, must_exist: bool = False) -> bool:
    """
    Validate file path.

    Args:
        file_path: Path to file
        must_exist: If True, file must already exist

    Returns:
        True if valid

    Raises:
        ValidationError: If file path is invalid
    """
    if not isinstance(file_path, str):
        raise ValidationError(f"File path must be a string, got {type(file_path)}")

    if not file_path or not file_path.strip():
        raise ValidationError("File path cannot be empty")

    path = Path(file_path)

    if must_exist and not path.exists():
        raise ValidationError(f"File does not exist: {file_path}")

    # Check if parent directory exists (for new files)
    if not must_exist:
        parent = path.parent
        if not parent.exists():
            raise ValidationError(f"Parent directory does not exist: {parent}")

        # Check write permissions
        if not os.access(parent, os.W_OK):
            raise ValidationError(f"No write permission for directory: {parent}")

    return True


# ============================================================================
# Numeric Range Validation
# ============================================================================

def validate_range(
    value: Union[int, float],
    min_value: Optional[Union[int, float]] = None,
    max_value: Optional[Union[int, float]] = None,
    param_name: str = "Value"
) -> bool:
    """
    Validate that a numeric value is within a specified range.

    Args:
        value: Value to validate
        min_value: Minimum allowed value (inclusive)
        max_value: Maximum allowed value (inclusive)
        param_name: Parameter name for error messages

    Returns:
        True if valid

    Raises:
        ValidationError: If value is out of range
    """
    if not isinstance(value, (int, float)):
        raise ValidationError(f"{param_name} must be a number, got {type(value)}")

    if min_value is not None and value < min_value:
        raise ValidationError(
            f"{param_name} {value} is less than minimum {min_value}"
        )

    if max_value is not None and value > max_value:
        raise ValidationError(
            f"{param_name} {value} exceeds maximum {max_value}"
        )

    return True


# Example usage and tests
if __name__ == "__main__":
    # Test sample rate validation
    try:
        validate_sample_rate(25600)
        print("✓ Valid sample rate: 25600 Hz")
    except ValidationError as e:
        print(f"✗ {e}")

    try:
        validate_sample_rate(100000)
        print("✗ Should have failed for 100 kHz")
    except ValidationError as e:
        print(f"✓ Caught invalid sample rate: {e}")

    # Test physical channel validation
    try:
        validate_physical_channel("cDAQ1Mod1/ai0")
        print("✓ Valid physical channel: cDAQ1Mod1/ai0")
    except ValidationError as e:
        print(f"✗ {e}")

    # Test cutoff frequency validation
    try:
        validate_cutoff_frequency(1000.0, 25600, "lowpass")
        print("✓ Valid lowpass cutoff: 1000 Hz @ 25.6 kHz")
    except ValidationError as e:
        print(f"✗ {e}")

    try:
        validate_cutoff_frequency((100, 5000), 25600, "bandpass")
        print("✓ Valid bandpass cutoff: 100-5000 Hz @ 25.6 kHz")
    except ValidationError as e:
        print(f"✗ {e}")

    print("\nAll validation tests completed!")

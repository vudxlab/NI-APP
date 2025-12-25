"""
DAQ configuration dataclasses.

This module defines the data structures for DAQ and channel configurations,
including serialization/deserialization for saving and loading settings.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
import json

from ..utils.constants import (
    NI9234Specs,
    DAQDefaults,
    ChannelDefaults
)
from ..utils.validators import (
    validate_sample_rate,
    validate_samples_per_channel,
    validate_channel_name,
    validate_physical_channel,
    validate_coupling,
    validate_sensitivity,
    validate_units,
    ValidationError
)


@dataclass
class ChannelConfig:
    """
    Configuration for a single DAQ channel.

    Attributes:
        physical_channel: Physical channel identifier (e.g., "cDAQ1Mod1/ai0")
        name: User-friendly channel name
        enabled: Whether this channel is enabled for acquisition
        input_range: Input voltage range (±V)
        coupling: AC or DC coupling mode
        iepe_enabled: Enable IEPE excitation for accelerometers
        sensitivity: Accelerometer sensitivity in mV/g
        units: Engineering units ("g", "m/s²", "mm/s²")
    """

    physical_channel: str
    name: str = "Channel"
    enabled: bool = True
    input_range: float = NI9234Specs.INPUT_RANGE_VOLTS
    coupling: str = NI9234Specs.DEFAULT_COUPLING
    iepe_enabled: bool = ChannelDefaults.IEPE_ENABLED
    sensitivity: float = ChannelDefaults.ACCELEROMETER_SENSITIVITY
    units: str = ChannelDefaults.DEFAULT_UNITS

    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()

    def validate(self) -> None:
        """
        Validate all channel configuration parameters.

        Raises:
            ValidationError: If any parameter is invalid
        """
        validate_physical_channel(self.physical_channel)
        validate_channel_name(self.name)
        validate_coupling(self.coupling)
        validate_sensitivity(self.sensitivity)
        validate_units(self.units)

        # Validate input range
        if self.input_range <= 0:
            raise ValidationError(f"Input range must be positive: {self.input_range}")

        if self.input_range > NI9234Specs.INPUT_RANGE_VOLTS:
            raise ValidationError(
                f"Input range {self.input_range}V exceeds maximum "
                f"{NI9234Specs.INPUT_RANGE_VOLTS}V for NI-9234"
            )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.

        Returns:
            Dictionary representation
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChannelConfig':
        """
        Create ChannelConfig from dictionary.

        Args:
            data: Dictionary with channel configuration

        Returns:
            ChannelConfig instance
        """
        return cls(**data)

    def get_scale_factor(self) -> float:
        """
        Calculate scaling factor from voltage to engineering units.

        For IEPE accelerometers:
            acceleration = voltage / (sensitivity / 1000)

        Returns:
            Scale factor (units per volt)
        """
        if self.iepe_enabled and self.sensitivity > 0:
            # Convert mV/g to V/g
            sensitivity_v_per_g = self.sensitivity / 1000.0
            # Scale factor in g per volt
            scale_factor = 1.0 / sensitivity_v_per_g

            # Convert to target units if needed
            if self.units == ChannelDefaults.UNITS_MS2:
                # Convert g to m/s²
                from ..utils.constants import UnitConversions
                scale_factor *= UnitConversions.G_TO_MS2
            elif self.units == ChannelDefaults.UNITS_MMS2:
                # Convert g to mm/s²
                from ..utils.constants import UnitConversions
                scale_factor *= UnitConversions.G_TO_MS2 * 1000

            return scale_factor
        else:
            # No scaling, return raw voltage
            return 1.0

    def get_display_range(self) -> tuple:
        """
        Get expected display range in engineering units.

        Returns:
            Tuple of (min, max) values in engineering units
        """
        scale_factor = self.get_scale_factor()
        min_val = -self.input_range * scale_factor
        max_val = self.input_range * scale_factor
        return (min_val, max_val)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ChannelConfig(physical_channel='{self.physical_channel}', "
            f"name='{self.name}', enabled={self.enabled}, "
            f"coupling={self.coupling}, iepe={self.iepe_enabled}, "
            f"sensitivity={self.sensitivity} mV/g, units={self.units})"
        )


@dataclass
class DAQConfig:
    """
    Overall DAQ configuration.

    Attributes:
        device_name: DAQ device identifier (e.g., "cDAQ1")
        sample_rate: Sampling rate in Hz
        samples_per_channel: Number of samples per channel to read in each iteration
        channels: List of channel configurations
        acquisition_mode: "continuous" or "finite"
        buffer_size_seconds: Size of circular buffer in seconds
    """

    device_name: str = ""
    sample_rate: float = DAQDefaults.SAMPLE_RATE
    samples_per_channel: int = DAQDefaults.SAMPLES_PER_CHANNEL
    channels: List[ChannelConfig] = field(default_factory=list)
    acquisition_mode: str = DAQDefaults.DEFAULT_ACQUISITION_MODE
    buffer_size_seconds: int = DAQDefaults.BUFFER_DURATION_SECONDS

    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()

    def validate(self) -> None:
        """
        Validate all DAQ configuration parameters.

        Raises:
            ValidationError: If any parameter is invalid
        """
        validate_sample_rate(self.sample_rate)
        validate_samples_per_channel(self.samples_per_channel)

        if self.acquisition_mode not in [
            DAQDefaults.ACQUISITION_MODE_CONTINUOUS,
            DAQDefaults.ACQUISITION_MODE_FINITE
        ]:
            raise ValidationError(
                f"Invalid acquisition mode '{self.acquisition_mode}', "
                f"must be 'continuous' or 'finite'"
            )

        # Validate each channel
        for i, channel in enumerate(self.channels):
            try:
                channel.validate()
            except ValidationError as e:
                raise ValidationError(f"Channel {i} invalid: {e}")

        # Check for duplicate physical channels
        physical_channels = [ch.physical_channel for ch in self.channels if ch.enabled]
        if len(physical_channels) != len(set(physical_channels)):
            raise ValidationError("Duplicate physical channels detected")

    def get_enabled_channels(self) -> List[ChannelConfig]:
        """
        Get list of enabled channels.

        Returns:
            List of enabled ChannelConfig objects
        """
        return [ch for ch in self.channels if ch.enabled]

    def get_num_enabled_channels(self) -> int:
        """
        Get number of enabled channels.

        Returns:
            Number of enabled channels
        """
        return len(self.get_enabled_channels())

    def get_buffer_size_samples(self) -> int:
        """
        Calculate buffer size in samples.

        Returns:
            Number of samples for circular buffer
        """
        return int(self.sample_rate * self.buffer_size_seconds)

    def get_physical_channel_list(self) -> List[str]:
        """
        Get list of physical channel names for enabled channels.

        Returns:
            List of physical channel strings
        """
        return [ch.physical_channel for ch in self.get_enabled_channels()]

    def get_nyquist_frequency(self) -> float:
        """
        Get Nyquist frequency (half the sample rate).

        Returns:
            Nyquist frequency in Hz
        """
        return self.sample_rate / 2.0

    def get_anti_alias_cutoff(self) -> float:
        """
        Get anti-aliasing filter cutoff frequency.

        NI-9234 has hardware anti-aliasing filter at 0.48 × sample_rate.

        Returns:
            Anti-aliasing filter cutoff frequency in Hz
        """
        return self.sample_rate * NI9234Specs.ANTI_ALIAS_FILTER_RATIO

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.

        Returns:
            Dictionary representation
        """
        return {
            "device_name": self.device_name,
            "sample_rate": self.sample_rate,
            "samples_per_channel": self.samples_per_channel,
            "channels": [ch.to_dict() for ch in self.channels],
            "acquisition_mode": self.acquisition_mode,
            "buffer_size_seconds": self.buffer_size_seconds
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DAQConfig':
        """
        Create DAQConfig from dictionary.

        Args:
            data: Dictionary with DAQ configuration

        Returns:
            DAQConfig instance
        """
        # Convert channel dictionaries to ChannelConfig objects
        channels_data = data.pop('channels', [])
        channels = [ChannelConfig.from_dict(ch) for ch in channels_data]

        return cls(channels=channels, **data)

    def to_json(self, filepath: str) -> None:
        """
        Save configuration to JSON file.

        Args:
            filepath: Path to JSON file
        """
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, filepath: str) -> 'DAQConfig':
        """
        Load configuration from JSON file.

        Args:
            filepath: Path to JSON file

        Returns:
            DAQConfig instance
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"DAQConfig(device='{self.device_name}', "
            f"sample_rate={self.sample_rate} Hz, "
            f"channels={len(self.channels)} ({self.get_num_enabled_channels()} enabled), "
            f"mode={self.acquisition_mode})"
        )


def create_default_config(
    device_name: str = "cDAQ1",
    num_modules: int = 1,
    module_start_idx: int = 1
) -> DAQConfig:
    """
    Create a default DAQ configuration.

    Args:
        device_name: DAQ device name
        num_modules: Number of NI-9234 modules
        module_start_idx: Starting module index (default 1 for Mod1)

    Returns:
        DAQConfig with default settings
    """
    channels = []

    for mod_idx in range(module_start_idx, module_start_idx + num_modules):
        for ai_idx in range(NI9234Specs.NUM_CHANNELS_PER_MODULE):
            physical_channel = f"{device_name}Mod{mod_idx}/ai{ai_idx}"
            channel_num = (mod_idx - module_start_idx) * NI9234Specs.NUM_CHANNELS_PER_MODULE + ai_idx
            channel_name = f"Channel {channel_num + 1}"

            channel = ChannelConfig(
                physical_channel=physical_channel,
                name=channel_name,
                enabled=True,
                input_range=NI9234Specs.INPUT_RANGE_VOLTS,
                coupling=NI9234Specs.DEFAULT_COUPLING,
                iepe_enabled=ChannelDefaults.IEPE_ENABLED,
                sensitivity=ChannelDefaults.ACCELEROMETER_SENSITIVITY,
                units=ChannelDefaults.DEFAULT_UNITS
            )
            channels.append(channel)

    config = DAQConfig(
        device_name=device_name,
        sample_rate=DAQDefaults.SAMPLE_RATE,
        samples_per_channel=DAQDefaults.SAMPLES_PER_CHANNEL,
        channels=channels,
        acquisition_mode=DAQDefaults.DEFAULT_ACQUISITION_MODE
    )

    return config


# Example usage
if __name__ == "__main__":
    # Create a default configuration for 3 modules (12 channels)
    config = create_default_config(device_name="cDAQ1", num_modules=3)

    print(f"Created configuration: {config}")
    print(f"Number of channels: {len(config.channels)}")
    print(f"Enabled channels: {config.get_num_enabled_channels()}")
    print(f"Nyquist frequency: {config.get_nyquist_frequency()} Hz")
    print(f"Anti-alias cutoff: {config.get_anti_alias_cutoff()} Hz")
    print(f"Buffer size: {config.get_buffer_size_samples()} samples")

    print("\nChannel details:")
    for i, ch in enumerate(config.channels[:4]):  # Show first 4 channels
        scale = ch.get_scale_factor()
        range_vals = ch.get_display_range()
        print(f"  {ch.name}: {ch.physical_channel}, "
              f"scale={scale:.2f} {ch.units}/V, "
              f"range={range_vals[0]:.1f} to {range_vals[1]:.1f} {ch.units}")

    # Test serialization
    print("\nTesting JSON serialization...")
    test_file = "/tmp/test_daq_config.json"
    config.to_json(test_file)
    print(f"Saved to {test_file}")

    loaded_config = DAQConfig.from_json(test_file)
    print(f"Loaded configuration: {loaded_config}")
    print(f"Configurations match: {config.to_dict() == loaded_config.to_dict()}")

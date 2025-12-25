"""
Channel Manager for NI DAQ.

This module handles channel configuration, validation, and application
to nidaqmx tasks. It manages IEPE settings, input ranges, coupling modes,
and scaling for accelerometer measurements.
"""

from typing import List, Optional, Dict
import logging

try:
    import nidaqmx
    from nidaqmx.constants import (
        TerminalConfiguration,
        Coupling,  # Fixed: was CouplingTypes
        CurrentShuntResistorLocation,
        ExcitationSource
    )
    NIDAQMX_AVAILABLE = True
except ImportError as e:
    NIDAQMX_AVAILABLE = False
    Coupling = None
    nidaqmx = None
    TerminalConfiguration = None
    CurrentShuntResistorLocation = None
    ExcitationSource = None
    logging.warning(f"nidaqmx not available - running in simulation mode: {e}")

from .daq_config import ChannelConfig, DAQConfig
from ..utils.logger import get_logger
from ..utils.constants import NI9234Specs, ChannelDefaults
from ..utils.validators import ValidationError


class ChannelManager:
    """
    Manages DAQ channel configuration and application to nidaqmx tasks.

    This class handles:
    - Parsing and validating channel configurations
    - Applying IEPE excitation settings
    - Configuring input ranges and coupling
    - Setting up channel scaling for accelerometers
    """

    def __init__(self):
        """Initialize the ChannelManager."""
        self.logger = get_logger(__name__)
        self.channels: List[ChannelConfig] = []
        self._scale_factors: Dict[str, float] = {}

    def set_channels(self, channels: List[ChannelConfig]) -> None:
        """
        Set the list of channel configurations.

        Args:
            channels: List of ChannelConfig objects

        Raises:
            ValidationError: If channel configuration is invalid
        """
        # Validate all channels
        for i, channel in enumerate(channels):
            try:
                channel.validate()
            except ValidationError as e:
                raise ValidationError(f"Channel {i} ({channel.name}) invalid: {e}")

        self.channels = channels
        self._update_scale_factors()
        self.logger.info(f"Configured {len(channels)} channels")

    def _update_scale_factors(self) -> None:
        """Update internal scale factor cache."""
        self._scale_factors = {}
        for channel in self.channels:
            if channel.enabled:
                self._scale_factors[channel.physical_channel] = channel.get_scale_factor()

    def get_enabled_channels(self) -> List[ChannelConfig]:
        """
        Get list of enabled channels.

        Returns:
            List of enabled ChannelConfig objects
        """
        return [ch for ch in self.channels if ch.enabled]

    def get_channel_by_physical_name(self, physical_channel: str) -> Optional[ChannelConfig]:
        """
        Get channel configuration by physical channel name.

        Args:
            physical_channel: Physical channel identifier (e.g., "cDAQ1Mod1/ai0")

        Returns:
            ChannelConfig if found, None otherwise
        """
        for channel in self.channels:
            if channel.physical_channel == physical_channel:
                return channel
        return None

    def apply_to_task(self, task: 'nidaqmx.Task', daq_config: DAQConfig) -> None:
        """
        Apply channel configurations to an nidaqmx task.

        This method adds analog input channels to the task with appropriate
        settings for IEPE accelerometers.

        Args:
            task: nidaqmx Task object
            daq_config: DAQ configuration

        Raises:
            RuntimeError: If nidaqmx is not available
            ValidationError: If configuration is invalid
        """
        if not NIDAQMX_AVAILABLE:
            raise RuntimeError("nidaqmx library not available")

        enabled_channels = self.get_enabled_channels()

        if not enabled_channels:
            raise ValidationError("No channels enabled")

        self.logger.info(f"Adding {len(enabled_channels)} channels to task")

        for channel_config in enabled_channels:
            self._add_channel_to_task(task, channel_config)

        self.logger.info(f"Successfully configured {len(enabled_channels)} channels")

    def _add_channel_to_task(
        self,
        task: 'nidaqmx.Task',
        channel_config: ChannelConfig
    ) -> None:
        """
        Add a single channel to the nidaqmx task.

        Args:
            task: nidaqmx Task object
            channel_config: Channel configuration

        Raises:
            Exception: If channel cannot be added
        """
        if not NIDAQMX_AVAILABLE:
            return

        try:
            # Add analog input voltage channel
            # NI-9234 measures voltage, we'll apply scaling afterwards
            ai_channel = task.ai_channels.add_ai_voltage_chan(
                physical_channel=channel_config.physical_channel,
                name_to_assign_to_channel=channel_config.name,
                terminal_config=TerminalConfiguration.DEFAULT,  # Use hardware default (module-specific)
                min_val=-channel_config.input_range,
                max_val=channel_config.input_range,
                units=nidaqmx.constants.VoltageUnits.VOLTS
            )

            # Configure IEPE excitation if enabled
            if channel_config.iepe_enabled:
                self._configure_iepe(ai_channel, channel_config)
                # IEPE requires AC coupling - force it
                self._configure_coupling(ai_channel, channel_config, force_ac_for_iepe=True)
            else:
                # Non-IEPE channels can use configured coupling
                self._configure_coupling(ai_channel, channel_config, force_ac_for_iepe=False)

            self.logger.debug(
                f"Added channel {channel_config.name} "
                f"({channel_config.physical_channel}) "
                f"IEPE={channel_config.iepe_enabled}, "
                f"Coupling={channel_config.coupling}"
            )

        except Exception as e:
            self.logger.error(
                f"Failed to add channel {channel_config.physical_channel}: {e}"
            )
            raise

    def _configure_iepe(
        self,
        ai_channel: 'nidaqmx._task_modules.channels.ai_channel.AIChannel',
        channel_config: ChannelConfig
    ) -> None:
        """
        Configure IEPE excitation for accelerometer channel.

        Args:
            ai_channel: nidaqmx AI channel object
            channel_config: Channel configuration
        """
        if not NIDAQMX_AVAILABLE:
            return

        try:
            # Enable IEPE excitation
            # NI-9234 has built-in IEPE current excitation at 2mA (NOT 24V!)
            ai_channel.ai_excit_src = ExcitationSource.INTERNAL
            ai_channel.ai_excit_val = NI9234Specs.IEPE_EXCITATION_CURRENT  # 0.002A = 2mA

            # Some NI-9234 modules support current excitation setting
            try:
                ai_channel.ai_excit_use_for_scaling = True
            except AttributeError:
                # Older nidaqmx versions may not have this attribute
                pass

            self.logger.debug(
                f"Configured IEPE excitation for {channel_config.name}: "
                f"{NI9234Specs.IEPE_EXCITATION_CURRENT*1000}mA"  # Show in mA
            )

        except Exception as e:
            self.logger.warning(
                f"Could not configure IEPE for {channel_config.name}: {e}"
            )

    def _configure_coupling(
        self,
        ai_channel: 'nidaqmx._task_modules.channels.ai_channel.AIChannel',
        channel_config: ChannelConfig,
        force_ac_for_iepe: bool = False
    ) -> None:
        """
        Configure AC/DC coupling for channel.

        Args:
            ai_channel: nidaqmx AI channel object
            channel_config: Channel configuration
            force_ac_for_iepe: If True, force AC coupling (required for IEPE)
        """
        if not NIDAQMX_AVAILABLE:
            return

        try:
            # IEPE channels MUST use AC coupling
            if force_ac_for_iepe:
                ai_channel.ai_coupling = Coupling.AC
                self.logger.debug(
                    f"Set coupling to AC for {channel_config.name} (IEPE required)"
                )
            elif channel_config.coupling.upper() == "AC":
                ai_channel.ai_coupling = Coupling.AC  # Fixed: was CouplingTypes
                self.logger.debug(
                    f"Set coupling to AC for {channel_config.name}"
                )
            elif channel_config.coupling.upper() == "DC":
                ai_channel.ai_coupling = Coupling.DC  # Fixed: was CouplingTypes
                self.logger.debug(
                    f"Set coupling to DC for {channel_config.name}"
                )
            else:
                self.logger.warning(
                    f"Unknown coupling mode '{channel_config.coupling}', "
                    f"defaulting to AC"
                )
                ai_channel.ai_coupling = Coupling.AC  # Fixed: was CouplingTypes

        except Exception as e:
            self.logger.warning(
                f"Could not configure coupling for {channel_config.name}: {e}"
            )

    def scale_data(self, raw_data, channel_indices: Optional[List[int]] = None) -> 'numpy.ndarray':
        """
        Apply scaling factors to convert voltage to engineering units.

        Args:
            raw_data: Raw voltage data (numpy array), shape (n_channels, n_samples)
            channel_indices: Optional list of channel indices to scale.
                           If None, scales all enabled channels.

        Returns:
            Scaled data in engineering units
        """
        import numpy as np

        # Make a copy to avoid modifying original data
        scaled_data = np.array(raw_data, dtype=np.float64)

        enabled_channels = self.get_enabled_channels()

        if channel_indices is None:
            channel_indices = range(len(enabled_channels))

        for i in channel_indices:
            if i < len(enabled_channels):
                channel = enabled_channels[i]
                scale_factor = self._scale_factors.get(
                    channel.physical_channel,
                    channel.get_scale_factor()
                )

                # Apply scaling: acceleration = voltage * scale_factor
                scaled_data[i, :] *= scale_factor

        return scaled_data

    def get_channel_names(self) -> List[str]:
        """
        Get list of enabled channel names.

        Returns:
            List of channel names
        """
        return [ch.name for ch in self.get_enabled_channels()]

    def get_channel_units(self) -> List[str]:
        """
        Get list of engineering units for enabled channels.

        Returns:
            List of units strings
        """
        return [ch.units for ch in self.get_enabled_channels()]

    def get_channel_info(self) -> List[Dict[str, any]]:
        """
        Get detailed information about all enabled channels.

        Returns:
            List of dictionaries with channel information
        """
        info = []
        for channel in self.get_enabled_channels():
            info.append({
                'name': channel.name,
                'physical_channel': channel.physical_channel,
                'units': channel.units,
                'sensitivity': channel.sensitivity,
                'iepe_enabled': channel.iepe_enabled,
                'coupling': channel.coupling,
                'input_range': channel.input_range,
                'scale_factor': channel.get_scale_factor(),
                'display_range': channel.get_display_range()
            })
        return info

    def validate_configuration(self) -> bool:
        """
        Validate the current channel configuration.

        Returns:
            True if configuration is valid

        Raises:
            ValidationError: If configuration is invalid
        """
        if not self.channels:
            raise ValidationError("No channels configured")

        enabled = self.get_enabled_channels()
        if not enabled:
            raise ValidationError("No channels enabled")

        # Check for duplicate physical channels
        physical_channels = [ch.physical_channel for ch in enabled]
        if len(physical_channels) != len(set(physical_channels)):
            raise ValidationError("Duplicate physical channels detected")

        # Validate each channel
        for channel in self.channels:
            channel.validate()

        return True

    def __repr__(self) -> str:
        """String representation."""
        enabled = len(self.get_enabled_channels())
        total = len(self.channels)
        return f"ChannelManager(channels={total}, enabled={enabled})"


# Mock classes for when nidaqmx is not available
class MockTask:
    """Mock nidaqmx Task for testing without hardware."""

    class AIChannels:
        def add_ai_voltage_chan(self, **kwargs):
            return MockAIChannel(kwargs.get('physical_channel', 'mock'))

    def __init__(self):
        self.ai_channels = self.AIChannels()


class MockAIChannel:
    """Mock AI channel for testing without hardware."""

    def __init__(self, physical_channel):
        self.physical_channel = physical_channel
        self.ai_excit_src = None
        self.ai_excit_val = None
        self.ai_excit_use_for_scaling = None
        self.ai_coupling = None


# Example usage
if __name__ == "__main__":
    import numpy as np
    from .daq_config import create_default_config

    # Create a default configuration
    config = create_default_config(device_name="cDAQ1", num_modules=3)

    # Create channel manager
    manager = ChannelManager()
    manager.set_channels(config.channels)

    print(f"Channel Manager: {manager}")
    print(f"\nEnabled channels: {manager.get_channel_names()}")
    print(f"Channel units: {manager.get_channel_units()}")

    # Show channel info
    print("\nChannel Information:")
    for info in manager.get_channel_info()[:4]:  # Show first 4
        print(f"  {info['name']}: {info['physical_channel']}")
        print(f"    Units: {info['units']}, Sensitivity: {info['sensitivity']} mV/g")
        print(f"    Scale factor: {info['scale_factor']:.2f} {info['units']}/V")
        print(f"    Display range: {info['display_range'][0]:.1f} to "
              f"{info['display_range'][1]:.1f} {info['units']}")

    # Test scaling
    print("\nTesting data scaling:")
    # Simulate raw voltage data (12 channels, 100 samples)
    raw_data = np.random.randn(12, 100) * 0.5  # Random voltages ±0.5V
    print(f"Raw data shape: {raw_data.shape}")
    print(f"Raw data range: {raw_data.min():.3f} to {raw_data.max():.3f} V")

    scaled_data = manager.scale_data(raw_data)
    print(f"Scaled data range: {scaled_data.min():.3f} to {scaled_data.max():.3f} g")

    # Test with mock task (no hardware required)
    if not NIDAQMX_AVAILABLE:
        print("\nTesting with mock task (nidaqmx not available):")
        mock_task = MockTask()
        try:
            manager.apply_to_task(mock_task, config)
        except RuntimeError as e:
            print(f"Expected error: {e}")

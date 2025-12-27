"""
Application Settings Data Classes.

This module defines data structures for application settings,
including GUI preferences, processing defaults, and export settings.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any
import json


@dataclass
class GUISettings:
    """GUI-related settings."""

    # Window geometry and state
    window_width: int = 1280
    window_height: int = 800
    window_x: int = 100
    window_y: int = 100
    window_maximized: bool = False

    # Dock widget states
    daq_config_visible: bool = True
    channel_config_visible: bool = True
    filter_config_visible: bool = True
    export_visible: bool = False

    # Tab selection
    current_tab: int = 0  # 0 = Time Domain, 1 = Frequency Domain

    # Plot settings
    realtime_time_window: float = 5.0  # seconds
    realtime_update_rate: int = 30  # Hz
    realtime_autoscale: bool = True
    realtime_downsample_threshold: int = 1000  # Maximum points to display before downsampling

    # FFT plot settings
    fft_scale: str = "dB"  # "linear" or "dB"
    fft_show_peaks: bool = True
    fft_peak_threshold: float = 0.1

    # Theme
    use_dark_theme: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GUISettings':
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ProcessingSettings:
    """Signal processing settings."""

    # Filter settings
    filter_type: str = "butterworth"
    filter_mode: str = "lowpass"
    filter_cutoff: float = 1000.0
    filter_order: int = 4
    filter_enabled: bool = False

    # FFT settings
    fft_window_size: int = 2048
    fft_window_function: str = "hann"
    fft_overlap: float = 0.5
    fft_enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProcessingSettings':
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ExportSettings:
    """Export-related settings."""

    # Default export format
    default_format: str = "tdms"

    # Default export directory
    default_directory: str = ""

    # Include metadata by default
    include_metadata: bool = True

    # HDF5 compression
    hdf5_compression: bool = True
    hdf5_compression_level: int = 4

    # Export time range
    time_range_selection: str = "all"  # "all", "10", "30", "60", "custom"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExportSettings':
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ApplicationSettings:
    """
    Main application settings container.

    Contains all application settings that can be persisted
    between sessions.
    """

    version: str = "1.0"
    gui: GUISettings = field(default_factory=GUISettings)
    processing: ProcessingSettings = field(default_factory=ProcessingSettings)
    export: ExportSettings = field(default_factory=ExportSettings)

    # Last used DAQ settings
    last_device_name: str = "cDAQ1"
    last_sample_rate: float = 25600.0
    last_num_modules: int = 3

    # Misc
    auto_save_last_config: bool = True
    check_for_updates: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'version': self.version,
            'gui': self.gui.to_dict(),
            'processing': self.processing.to_dict(),
            'export': self.export.to_dict(),
            'last_device_name': self.last_device_name,
            'last_sample_rate': self.last_sample_rate,
            'last_num_modules': self.last_num_modules,
            'auto_save_last_config': self.auto_save_last_config,
            'check_for_updates': self.check_for_updates
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ApplicationSettings':
        """Create ApplicationSettings from dictionary."""
        settings = cls()

        # Version check
        if 'version' in data:
            settings.version = data['version']

        # GUI settings
        if 'gui' in data:
            settings.gui = GUISettings.from_dict(data['gui'])

        # Processing settings
        if 'processing' in data:
            settings.processing = ProcessingSettings.from_dict(data['processing'])

        # Export settings
        if 'export' in data:
            settings.export = ExportSettings.from_dict(data['export'])

        # Other settings
        for key in ['last_device_name', 'last_sample_rate', 'last_num_modules',
                     'auto_save_last_config', 'check_for_updates']:
            if key in data:
                setattr(settings, key, data[key])

        return settings

    def save(self, filepath: str) -> None:
        """
        Save settings to JSON file.

        Args:
            filepath: Path to JSON file
        """
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> 'ApplicationSettings':
        """
        Load settings from JSON file.

        Args:
            filepath: Path to JSON file

        Returns:
            ApplicationSettings instance
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ApplicationSettings(version={self.version}, "
            f"filter_enabled={self.processing.filter_enabled})"
        )


# Example usage
if __name__ == "__main__":
    # Create default settings
    settings = ApplicationSettings()

    print("Default Application Settings:")
    print(f"  Window: {settings.gui.window_width}x{settings.gui.window_height}")
    print(f"  Filter: {settings.processing.filter_type} {settings.processing.filter_mode}")
    print(f"  Export: {settings.export.default_format} format")

    # Save to file
    test_file = "/tmp/test_app_settings.json"
    settings.save(test_file)
    print(f"\nSaved to {test_file}")

    # Load back
    loaded_settings = ApplicationSettings.load(test_file)
    print(f"\nLoaded settings: {loaded_settings}")

    # Verify
    print(f"\nSettings match: {settings.to_dict() == loaded_settings.to_dict()}")

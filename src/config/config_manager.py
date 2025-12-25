"""
Configuration Manager for application settings.

This module handles loading, saving, and managing application settings
with automatic directory creation and error handling.
"""

import json
import os
from pathlib import Path
from typing import Optional

from .app_settings import ApplicationSettings
from ..utils.logger import get_logger
from ..utils.constants import AppConfig


class ConfigManager:
    """
    Manager for application configuration.

    Handles:
    - Loading application settings from JSON
    - Saving application settings to JSON
    - Creating config directory if needed
    - Providing default settings
    """

    def __init__(self):
        """Initialize the configuration manager."""
        self.logger = get_logger(__name__)

        # Settings instance
        self._settings: Optional[ApplicationSettings] = None

        # Config directory and file
        self.config_dir = Path(AppConfig.CONFIG_DIR)
        self.config_file = self.config_dir / AppConfig.CONFIG_FILE

        self.logger.debug(f"Config manager initialized: {self.config_file}")

    def get_config_dir(self) -> Path:
        """
        Get the configuration directory.

        Returns:
            Path to configuration directory
        """
        return self.config_dir

    def ensure_config_dir(self) -> None:
        """
        Ensure configuration directory exists.

        Creates the directory and any necessary parent directories.
        """
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.logger.debug(f"Config directory ensured: {self.config_dir}")

    def load_settings(self) -> ApplicationSettings:
        """
        Load application settings.

        Returns:
            ApplicationSettings instance (default if file doesn't exist)

        Raises:
            IOError: If file exists but cannot be read
            ValueError: If file contains invalid JSON
        """
        if self.config_file.exists():
            try:
                self._settings = ApplicationSettings.load(str(self.config_file))
                self.logger.info(f"Settings loaded from {self.config_file}")
                return self._settings

            except json.JSONDecodeError as e:
                self.logger.error(f"Invalid JSON in config file: {e}")
                # Backup the corrupted file
                self._backup_corrupted_config()
                # Return default settings
                self._settings = ApplicationSettings()
                return self._settings

            except Exception as e:
                self.logger.error(f"Failed to load settings: {e}")
                self._settings = ApplicationSettings()
                return self._settings

        else:
            # File doesn't exist, return defaults and save them
            self.logger.info("Config file not found, creating defaults")
            self._settings = ApplicationSettings()
            self.save_settings()
            return self._settings

    def save_settings(self, settings: Optional[ApplicationSettings] = None) -> None:
        """
        Save application settings to JSON file.

        Args:
            settings: ApplicationSettings to save (uses current if None)

        Raises:
            IOError: If file cannot be written
        """
        if settings is not None:
            self._settings = settings

        if self._settings is None:
            self.logger.warning("No settings to save")
            return

        # Ensure directory exists
        self.ensure_config_dir()

        try:
            self._settings.save(str(self.config_file))
            self.logger.info(f"Settings saved to {self.config_file}")

        except Exception as e:
            self.logger.error(f"Failed to save settings: {e}")
            raise

    def get_settings(self) -> ApplicationSettings:
        """
        Get current application settings.

        Loads from file if not already loaded.

        Returns:
            Current ApplicationSettings instance
        """
        if self._settings is None:
            self._settings = self.load_settings()

        return self._settings

    def update_setting(self, category: str, key: str, value: any) -> None:
        """
        Update a specific setting value.

        Args:
            category: Settings category ('gui', 'processing', 'export')
            key: Setting key
            value: New value
        """
        settings = self.get_settings()

        if category == 'gui':
            setattr(settings.gui, key, value)
        elif category == 'processing':
            setattr(settings.processing, key, value)
        elif category == 'export':
            setattr(settings.export, key, value)
        else:
            self.logger.warning(f"Unknown settings category: {category}")
            return

        # Save updated settings
        self.save_settings()

    def get_setting(self, category: str, key: str, default=None):
        """
        Get a specific setting value.

        Args:
            category: Settings category ('gui', 'processing', 'export')
            key: Setting key
            default: Default value if not found

        Returns:
            Setting value or default
        """
        settings = self.get_settings()

        if category == 'gui':
            return getattr(settings.gui, key, default)
        elif category == 'processing':
            return getattr(settings.processing, key, default)
        elif category == 'export':
            return getattr(settings.export, key, default)
        else:
            self.logger.warning(f"Unknown settings category: {category}")
            return default

    def reset_to_defaults(self) -> None:
        """Reset all settings to default values."""
        self.logger.info("Resetting settings to defaults")
        self._settings = ApplicationSettings()
        self.save_settings()

    def export_settings(self, filepath: str) -> None:
        """
        Export current settings to a specific file.

        Args:
            filepath: Path to export settings
        """
        settings = self.get_settings()
        settings.save(filepath)
        self.logger.info(f"Settings exported to {filepath}")

    def import_settings(self, filepath: str) -> None:
        """
        Import settings from a specific file.

        Args:
            filepath: Path to import settings from

        Raises:
            IOError: If file cannot be read
            ValueError: If file contains invalid data
        """
        self._settings = ApplicationSettings.load(filepath)
        self.logger.info(f"Settings imported from {filepath}")

    def _backup_corrupted_config(self) -> None:
        """
        Backup corrupted configuration file.

        Renames the corrupted file with a .bak extension.
        """
        if self.config_file.exists():
            backup_path = self.config_file.with_suffix('.json.bak')
            try:
                self.config_file.rename(backup_path)
                self.logger.info(f"Corrupted config backed up to {backup_path}")
            except Exception as e:
                self.logger.error(f"Failed to backup corrupted config: {e}")

    def get_config_info(self) -> dict:
        """
        Get information about the configuration.

        Returns:
            Dictionary with configuration info
        """
        return {
            'config_dir': str(self.config_dir),
            'config_file': str(self.config_file),
            'exists': self.config_file.exists(),
            'settings_loaded': self._settings is not None
        }


# Global config manager instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """
    Get the global configuration manager instance.

    Returns:
        ConfigManager instance (singleton)
    """
    global _config_manager

    if _config_manager is None:
        _config_manager = ConfigManager()

    return _config_manager


# Example usage
if __name__ == "__main__":
    import tempfile

    print("Config Manager Test")
    print("=" * 60)

    # Create a temporary config directory for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        # Override config directory for testing
        test_config_dir = Path(tmpdir)
        original_config_dir = AppConfig.CONFIG_DIR

        # Patch AppConfig
        AppConfig.CONFIG_DIR = str(test_config_dir)

        try:
            # Create config manager
            manager = ConfigManager()

            print(f"\n1. Config directory: {manager.get_config_dir()}")
            print(f"   Config file: {manager.config_file}")

            # Load settings (will create defaults)
            print("\n2. Loading settings...")
            settings = manager.load_settings()
            print(f"   Settings loaded: {settings}")

            # Modify and save
            print("\n3. Modifying settings...")
            manager.update_setting('gui', 'window_width', 1920)
            manager.update_setting('processing', 'filter_enabled', True)
            print(f"   Settings updated")

            # Reload
            print("\n4. Reloading settings...")
            settings2 = manager.load_settings()
            print(f"   Window width: {settings2.gui.window_width}")
            print(f"   Filter enabled: {settings2.processing.filter_enabled}")

            # Export info
            print("\n5. Config info:")
            info = manager.get_config_info()
            for key, value in info.items():
                print(f"   {key}: {value}")

        finally:
            # Restore original config dir
            AppConfig.CONFIG_DIR = original_config_dir

    print("\n" + "=" * 60)
    print("Config manager test complete!")

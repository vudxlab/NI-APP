#!/usr/bin/env python3
"""
NI DAQ Vibration Analysis - Main Entry Point.

This is the main entry point for the application.
It initializes the Qt application, sets up logging,
loads configuration, and launches the main window.
"""

import sys
import os
import logging
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from PyQt5.QtWidgets import QApplication, QSplashScreen
from PyQt5.QtCore import Qt, QCoreApplication
from PyQt5.QtGui import QFont, QPixmap, QSurfaceFormat

# Import application components
from src.gui.main_window import MainWindow
from src.utils.logger import setup_logger
from src.utils.constants import AppConfig
from src.config.config_manager import get_config_manager


def setup_high_dpi():
    """
    Configure high DPI scaling for better display on modern screens.

    Enables:
    - Automatic scaling on high DPI displays
    - Fractional scaling (1.5x, 2x, etc.)
    - Proper font and UI element sizing
    """
    # Enable high DPI scaling
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    # Set DPI awareness for Windows
    if sys.platform == 'win32':
        try:
            import ctypes
            ctypes.windll.shcoreinfo.SetProcessDpiAwareness(1)
        except (AttributeError, OSError):
            pass


def initialize_application():
    """
    Initialize and configure the Qt application.

    Returns:
        QApplication instance
    """
    # Set attributes required by QtWebEngine before QApplication
    QCoreApplication.setAttribute(Qt.AA_ShareOpenGLContexts, True)
    setup_high_dpi()

    # Create QApplication
    app = QApplication(sys.argv)

    # Set application metadata
    app.setApplicationName(AppConfig.APP_NAME)
    app.setApplicationVersion(AppConfig.APP_VERSION)
    app.setOrganizationName(AppConfig.ORGANIZATION_NAME)
    app.setOrganizationDomain("ni-daq-app")

    # Set default font size for better readability
    font = app.font()
    font.setPointSize(10)
    app.setFont(font)

    # Set application style
    app.setStyle("Fusion")

    logger = setup_logger("ni_daq_app", level=logging.INFO)

    logger.info("=" * 60)
    logger.info(f"{AppConfig.APP_NAME} v{AppConfig.APP_VERSION}")
    logger.info("Application starting...")
    logger.info("=" * 60)

    return app


def create_splash_screen():
    """
    Create and show splash screen during loading.

    Returns:
        QSplashScreen instance (or None if disabled)
    """
    # Disable splash screen for now (requires QPixmap setup)
    # TODO: Implement proper splash screen with QPixmap
    return None


def run_application():
    """
    Main application entry point.

    Initializes and runs the application with proper error handling
    and logging.
    """
    # Initialize Qt application
    app = initialize_application()
    logger = logging.getLogger("ni_daq_app")

    # Show splash screen (optional)
    splash = create_splash_screen()

    if splash:
        splash.showMessage("Loading configuration...")
        QApplication.processEvents()

    # Load configuration
    try:
        config_manager = get_config_manager()
        settings = config_manager.get_settings()
        logger.info("Configuration loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        logger.info("Using default configuration")

    if splash:
        splash.showMessage("Creating main window...")
        QApplication.processEvents()

    # Create and show main window
    try:
        window = MainWindow()

        logger.info("Main window created")

        if splash:
            splash.showMessage("Ready!")
            QApplication.processEvents()

        window.show()

        if splash:
            splash.finish()
            QApplication.processEvents()

        logger.info("Application started successfully")

    except Exception as e:
        logger.error(f"Failed to create main window: {e}")

        if splash:
            splash.close()

        from PyQt5.QtWidgets import QMessageBox
        QMessageBox.critical(
            None,
            "Initialization Error",
            f"Failed to create application window:\n\n{e}"
        )
        sys.exit(1)

    # Run event loop
    try:
        exit_code = app.exec_()

    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        exit_code = 0

    except Exception as e:
        logger.exception(f"Unexpected error during execution: {e}")
        exit_code = 1

    # Cleanup
    logger.info("=" * 60)
    logger.info(f"Application exiting with code {exit_code}")
    logger.info("=" * 60)

    sys.exit(exit_code)


def main():
    """
    Entry point for the application.

    This function is called when the script is run directly.
    It wraps the application startup with error handling.
    """
    try:
        run_application()

    except Exception as e:
        # Last resort error handling
        print(f"Fatal error: {e}", file=sys.stderr)
        sys.exit(1)


# Script execution
if __name__ == "__main__":
    main()

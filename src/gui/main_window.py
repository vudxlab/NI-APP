"""
Main application window for NI DAQ Vibration Analysis.

This module implements the main GUI window that coordinates all components
including DAQ configuration, real-time plots, FFT display, and controls.
"""

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QStatusBar, QMenuBar, QMenu, QAction, QMessageBox, QFileDialog,
    QDockWidget, QTabWidget
)
from PyQt5.QtCore import Qt, QTimer, pyqtSlot
from PyQt5.QtGui import QIcon, QKeySequence
from typing import Optional
from pathlib import Path
import time

from ..daq.daq_manager import DAQManager
from ..daq.daq_config import DAQConfig, create_default_config
from ..daq.acquisition_thread import AcquisitionThread
from ..processing.signal_processor import SignalProcessor
from ..utils.logger import get_logger
from ..utils.constants import AppConfig, GUIDefaults, DAQDefaults
from ..config.config_manager import get_config_manager


class MainWindow(QMainWindow):
    """
    Main application window.

    This is the central hub of the application, coordinating:
    - DAQ configuration and control
    - Real-time data acquisition
    - Signal processing
    - Plotting and visualization
    - User controls
    """

    def __init__(self):
        """Initialize the main window."""
        super().__init__()

        self.logger = get_logger(__name__)

        # Components
        self.daq_manager: Optional[DAQManager] = None
        self.acquisition_thread: Optional[AcquisitionThread] = None
        self.signal_processor: Optional[SignalProcessor] = None
        self.config: Optional[DAQConfig] = None

        # State
        self.is_acquiring = False

        # UI Components (will be set by _create_widgets)
        self.daq_config_panel = None
        self.channel_config_widget = None
        self.realtime_plot_widget = None
        self.fft_plot_widget = None
        self.filter_config_panel = None
        self.export_panel = None

        # Status update timer
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self._update_status_bar)
        self.status_timer.setInterval(GUIDefaults.STATUS_UPDATE_INTERVAL_MS)

        # Configuration manager
        self.config_manager = get_config_manager()

        # Initialize UI
        self._init_ui()

        # Load application settings and restore window state
        self._load_app_settings()

        # Initialize DAQ
        self._init_daq()

        self.logger.info("MainWindow initialized")

    def _init_ui(self):
        """Initialize the user interface."""
        # Set window properties
        self.setWindowTitle(f"{AppConfig.APP_NAME} v{AppConfig.APP_VERSION}")
        self.resize(GUIDefaults.DEFAULT_WINDOW_WIDTH, GUIDefaults.DEFAULT_WINDOW_HEIGHT)

        # Create menu bar
        self._create_menu_bar()

        # Create central widget with layout
        self._create_central_widget()

        # Create status bar
        self._create_status_bar()

        # Create dock widgets
        self._create_dock_widgets()

    def _create_menu_bar(self):
        """Create the menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        # New configuration
        new_action = QAction("&New Configuration", self)
        new_action.setShortcut(QKeySequence.New)
        new_action.triggered.connect(self._on_new_configuration)
        file_menu.addAction(new_action)

        # Open configuration
        open_action = QAction("&Open Configuration...", self)
        open_action.setShortcut(QKeySequence.Open)
        open_action.triggered.connect(self._on_open_configuration)
        file_menu.addAction(open_action)

        # Save configuration
        save_action = QAction("&Save Configuration...", self)
        save_action.setShortcut(QKeySequence.Save)
        save_action.triggered.connect(self._on_save_configuration)
        file_menu.addAction(save_action)

        file_menu.addSeparator()

        # Export data
        export_action = QAction("&Export Data...", self)
        export_action.setShortcut("Ctrl+E")
        export_action.triggered.connect(self._on_export_data)
        file_menu.addAction(export_action)

        file_menu.addSeparator()

        # Exit
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut(QKeySequence.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # View menu
        view_menu = menubar.addMenu("&View")

        # Toggle dock widgets (will be populated when docks are created)
        self.view_menu = view_menu

        # Acquisition menu
        acq_menu = menubar.addMenu("&Acquisition")

        # Start acquisition
        self.start_action = QAction("&Start", self)
        self.start_action.setShortcut("F5")
        self.start_action.triggered.connect(self._on_start_acquisition)
        acq_menu.addAction(self.start_action)

        # Stop acquisition
        self.stop_action = QAction("S&top", self)
        self.stop_action.setShortcut("F6")
        self.stop_action.setEnabled(False)
        self.stop_action.triggered.connect(self._on_stop_acquisition)
        acq_menu.addAction(self.stop_action)

        acq_menu.addSeparator()

        # Clear buffers
        clear_action = QAction("&Clear Buffers", self)
        clear_action.setShortcut("Ctrl+L")
        clear_action.triggered.connect(self._on_clear_buffers)
        acq_menu.addAction(clear_action)

        # Tools menu
        tools_menu = menubar.addMenu("&Tools")

        # Device info
        device_info_action = QAction("&Device Information", self)
        device_info_action.triggered.connect(self._on_device_info)
        tools_menu.addAction(device_info_action)

        # Settings
        settings_action = QAction("&Settings...", self)
        settings_action.setShortcut("Ctrl+,")
        settings_action.triggered.connect(self._on_settings)
        tools_menu.addAction(settings_action)

        # Help menu
        help_menu = menubar.addMenu("&Help")

        # About
        about_action = QAction("&About", self)
        about_action.triggered.connect(self._on_about)
        help_menu.addAction(about_action)

    def _create_central_widget(self):
        """Create the central widget with main layout."""
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)

        # Create horizontal splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)  # Prevent collapsing panels

        # Create configuration tabs (left side)
        self.config_tabs = QTabWidget()
        self.config_tabs.setMinimumWidth(300)  # Minimum width
        
        # DAQ Configuration tab
        from .widgets.daq_config_panel import DAQConfigPanel
        self.daq_config_panel = DAQConfigPanel()
        self.daq_config_panel.start_requested.connect(self._on_start_acquisition)
        self.daq_config_panel.stop_requested.connect(self._on_stop_acquisition)
        self.config_tabs.addTab(self.daq_config_panel, "DAQ")
        
        # Channel Configuration tab
        from .widgets.channel_config_widget import ChannelConfigWidget
        self.channel_config_widget = ChannelConfigWidget()
        self.config_tabs.addTab(self.channel_config_widget, "Channels")
        
        # Filter Configuration tab
        from .widgets.filter_config_panel import FilterConfigPanel
        self.filter_config_panel = FilterConfigPanel()
        self.filter_config_panel.filter_changed.connect(self._on_filter_changed)
        self.filter_config_panel.filter_enabled.connect(self._on_filter_enabled)
        self.config_tabs.addTab(self.filter_config_panel, "Filter")
        
        # Export tab
        from .widgets.export_panel import ExportPanel
        self.export_panel = ExportPanel()
        self.config_tabs.addTab(self.export_panel, "Export")

        # Create plot tabs (right side)
        self.plot_tabs = QTabWidget()
        self.plot_tabs.setMinimumWidth(400)  # Minimum width for plots

        # Real-time plot tab
        from .widgets.realtime_plot_widget import RealtimePlotWidget

        realtime_tab = QWidget()
        realtime_layout = QVBoxLayout(realtime_tab)
        realtime_layout.setContentsMargins(0, 0, 0, 0)

        self.realtime_plot_widget = RealtimePlotWidget()
        realtime_layout.addWidget(self.realtime_plot_widget)

        # FFT plot tab
        from .widgets.fft_plot_widget import FFTPlotWidget

        fft_tab = QWidget()
        fft_layout = QVBoxLayout(fft_tab)
        fft_layout.setContentsMargins(0, 0, 0, 0)

        self.fft_plot_widget = FFTPlotWidget()
        self.fft_plot_widget.fft_size_changed.connect(self._on_fft_size_changed)
        fft_layout.addWidget(self.fft_plot_widget)

        self.plot_tabs.addTab(realtime_tab, "Time Domain")
        self.plot_tabs.addTab(fft_tab, "Frequency Domain")

        # Add both tab widgets to splitter
        splitter.addWidget(self.config_tabs)  # Left: Config tabs
        splitter.addWidget(self.plot_tabs)    # Right: Plot tabs
        
        # Set initial splitter sizes (25% config, 75% plots)
        splitter.setSizes([400, 1200])
        
        # Add splitter to main layout
        main_layout.addWidget(splitter)

    def _create_status_bar(self):
        """Create the status bar."""
        self.statusBar().showMessage("Ready")

    def _create_dock_widgets(self):
        """Create dock widgets for configuration panels."""
        # NOTE: Dock widgets replaced by tabs in _create_central_widget()
        # All configuration panels are now in config_tabs (left side)
        # This method kept for compatibility but does nothing
        pass

    def _init_daq(self):
        """Initialize DAQ manager and default configuration."""
        try:
            # Create DAQ manager
            self.daq_manager = DAQManager()

            # Create default configuration
            self.config = create_default_config(
                device_name=self.config_manager.get_setting(
                    '', 'last_device_name', 'cDAQ1'
                ),
                num_modules=self.config_manager.get_setting(
                    '', 'last_num_modules', 3
                )
            )

            # Update UI with configuration
            if self.daq_config_panel:
                self.daq_config_panel.set_config(self.config)
            if self.channel_config_widget:
                self.channel_config_widget.set_channels(self.config.channels)

            self.logger.info("DAQ initialized with default configuration")

        except Exception as e:
            self.logger.error(f"Failed to initialize DAQ: {e}")
            QMessageBox.warning(
                self,
                "DAQ Initialization",
                f"Failed to initialize DAQ:\n{e}\n\nRunning in simulation mode."
            )

    def _load_app_settings(self):
        """Load application settings and apply to UI."""
        try:
            settings = self.config_manager.get_settings()

            # Restore window geometry
            # Use setGeometry() instead of resize() to force size
            from PyQt5.QtCore import QRect
            self.setGeometry(QRect(
                settings.gui.window_x,
                settings.gui.window_y,
                settings.gui.window_width,
                settings.gui.window_height
            ))
            
            # Set size constraints to prevent auto-resize by dock widgets
            self.setMinimumSize(800, 600)  # Minimum reasonable size
            
            if settings.gui.window_maximized:
                self.showMaximized()
            else:
                # Force the size one more time after a short delay to override dock widget sizing
                from PyQt5.QtCore import QTimer
                QTimer.singleShot(100, lambda: self.resize(
                    settings.gui.window_width,
                    settings.gui.window_height
                ))

            # Restore dock widget visibility
            # (Dock widgets will be created after this method is called,
            #  so we'll handle this in closeEvent)

            # Restore plot settings
            if self.realtime_plot_widget:
                # Will be configured when acquisition starts
                pass

            # Restore filter settings
            if self.filter_config_panel:
                filter_config = {
                    'type': settings.processing.filter_type,
                    'mode': settings.processing.filter_mode,
                    'cutoff': settings.processing.filter_cutoff,
                    'order': settings.processing.filter_order,
                    'enabled': settings.processing.filter_enabled
                }
                self.filter_config_panel.set_filter_config(filter_config)

            self.logger.info("Application settings loaded")

        except Exception as e:
            self.logger.error(f"Failed to load app settings: {e}")

    def _save_app_settings(self):
        """Save current application settings."""
        try:
            settings = self.config_manager.get_settings()

            # Save window geometry
            settings.gui.window_width = self.width()
            settings.gui.window_height = self.height()
            settings.gui.window_x = self.x()
            settings.gui.window_y = self.y()
            settings.gui.window_maximized = self.isMaximized()

            # Save current tab selection
            if hasattr(self, 'plot_tabs'):
                settings.gui.current_tab = self.plot_tabs.currentIndex()

            # Save last DAQ settings
            if self.config:
                settings.last_device_name = self.config.device_name
                settings.last_sample_rate = self.config.sample_rate
                # Count modules from channels
                unique_modules = set()
                for ch in self.config.channels:
                    if 'Mod' in ch.physical_channel:
                        mod_num = int(ch.physical_channel.split('Mod')[1].split('/')[0])
                        unique_modules.add(mod_num)
                settings.last_num_modules = len(unique_modules)

            # Save filter settings
            if self.filter_config_panel:
                filter_config = self.filter_config_panel.get_filter_config()
                settings.processing.filter_type = filter_config.get('type', 'butterworth')
                settings.processing.filter_mode = filter_config.get('mode', 'lowpass')
                settings.processing.filter_cutoff = filter_config.get('cutoff_low', 1000.0)
                settings.processing.filter_order = filter_config.get('order', 4)
                settings.processing.filter_enabled = filter_config.get('enabled', False)

            # Save to file
            self.config_manager.save_settings(settings)

            self.logger.info("Application settings saved")

        except Exception as e:
            self.logger.error(f"Failed to save app settings: {e}")

    def _on_start_acquisition(self):
        """Handle start acquisition request."""
        if self.is_acquiring:
            self.logger.warning("Acquisition already running")
            return

        try:
            # Get current configuration from UI
            if self.daq_config_panel:
                self.config = self.daq_config_panel.get_config()

            if self.channel_config_widget:
                self.config.channels = self.channel_config_widget.get_channels()

            # Get auto-save configuration
            autosave_config = self.daq_config_panel.get_autosave_config()

            # Configure DAQ
            self.daq_manager.configure(self.config)
            self.daq_manager.create_task()

            # Create signal processor
            self.signal_processor = SignalProcessor(
                n_channels=self.config.get_num_enabled_channels(),
                sample_rate=self.config.sample_rate,
                buffer_duration=self.config.buffer_size_seconds
            )

            # Configure real-time plot widget
            channel_names = [ch.name for ch in self.config.get_enabled_channels()]
            channel_units = self.config.get_enabled_channels()[0].units if self.config.get_enabled_channels() else "g"

            self.realtime_plot_widget.configure(
                n_channels=self.config.get_num_enabled_channels(),
                sample_rate=self.config.sample_rate,
                channel_names=channel_names,
                channel_units=channel_units
            )

            # Configure FFT plot widget
            self.fft_plot_widget.configure(
                n_channels=self.config.get_num_enabled_channels(),
                sample_rate=self.config.sample_rate,
                channel_names=channel_names,
                channel_units=channel_units
            )

            # Connect signal processor to realtime plot
            self.signal_processor.filtered_data_ready.connect(self.realtime_plot_widget.update_plot)

            # Start realtime plot updates
            self.realtime_plot_widget.start()

            # Create and start acquisition thread with auto-save config
            self.acquisition_thread = AcquisitionThread(
                self.daq_manager,
                self.config,
                autosave_config=autosave_config
            )

            # Connect signals
            self.acquisition_thread.data_ready.connect(self._on_data_ready)
            self.acquisition_thread.error_occurred.connect(self._on_acquisition_error)
            self.acquisition_thread.acquisition_started.connect(self._on_acquisition_started)
            self.acquisition_thread.acquisition_stopped.connect(self._on_acquisition_stopped)
            self.acquisition_thread.status_update.connect(self._on_status_update)

            # Connect auto-save signals
            self.acquisition_thread.save_file_created.connect(self._on_save_file_created)
            self.acquisition_thread.save_file_closed.connect(self._on_save_file_closed)
            self.acquisition_thread.save_error.connect(self._on_save_error)

            # Start thread
            self.acquisition_thread.start()

            self.logger.info("Starting acquisition...")

        except Exception as e:
            self.logger.error(f"Failed to start acquisition: {e}")
            QMessageBox.critical(
                self,
                "Acquisition Error",
                f"Failed to start acquisition:\n{e}"
            )
            self._cleanup_acquisition()

    def _on_stop_acquisition(self):
        """Handle stop acquisition request."""
        if not self.is_acquiring:
            return

        self.logger.info("Stopping acquisition...")

        if self.acquisition_thread:
            self.acquisition_thread.stop()
            self.acquisition_thread.wait(5000)  # Wait up to 5 seconds

        self._cleanup_acquisition()

    def _cleanup_acquisition(self):
        """Clean up acquisition resources."""
        if self.acquisition_thread:
            self.acquisition_thread = None

        if self.daq_manager:
            self.daq_manager.close_task()

        self.is_acquiring = False

    @pyqtSlot(float, object, object)
    def _on_data_ready(self, timestamp, raw_data, scaled_data):
        """Handle new data from acquisition thread."""
        if self.signal_processor:
            self.signal_processor.process_data(scaled_data, timestamp)

    @pyqtSlot(str)
    def _on_acquisition_error(self, error_msg):
        """Handle acquisition error."""
        self.logger.error(f"Acquisition error: {error_msg}")
        self.statusBar().showMessage(f"Error: {error_msg}", 5000)

    @pyqtSlot()
    def _on_acquisition_started(self):
        """Handle acquisition started signal."""
        self.is_acquiring = True
        self.start_action.setEnabled(False)
        self.stop_action.setEnabled(True)

        if self.daq_config_panel:
            self.daq_config_panel.set_acquisition_state(True)

        self.status_timer.start()
        self.statusBar().showMessage("Acquisition running")
        self.logger.info("Acquisition started")

    @pyqtSlot()
    def _on_acquisition_stopped(self):
        """Handle acquisition stopped signal."""
        self.is_acquiring = False
        self.start_action.setEnabled(True)
        self.stop_action.setEnabled(False)

        if self.daq_config_panel:
            self.daq_config_panel.set_acquisition_state(False)

        # Stop plot updates
        if self.realtime_plot_widget:
            self.realtime_plot_widget.stop()
        if self.fft_plot_widget:
            self.fft_plot_widget.stop()

        self.status_timer.stop()
        self.statusBar().showMessage("Acquisition stopped")
        self.logger.info("Acquisition stopped")

    @pyqtSlot(dict)
    def _on_status_update(self, status):
        """Handle status update from acquisition thread."""
        # Update status bar with acquisition statistics
        samples = status.get('samples_acquired', 0)
        rate = status.get('samples_per_second', 0)
        errors = status.get('errors_count', 0)

        status_msg = f"Samples: {samples:,} | Rate: {rate:.1f} S/s | Errors: {errors}"
        self.statusBar().showMessage(status_msg)

    @pyqtSlot(str)
    def _on_save_file_created(self, filepath: str):
        """
        Handle save file created signal from acquisition thread.

        Args:
            filepath: Path to the created save file
        """
        self.logger.info(f"Save file created: {filepath}")

        # Update FFT widget with file path
        if self.fft_plot_widget:
            self.fft_plot_widget.set_data_file(filepath)

        # Update status bar
        filename = Path(filepath).name
        self.statusBar().showMessage(f"Saving to: {filename}", 5000)

    @pyqtSlot(str)
    def _on_save_file_closed(self, filepath: str):
        """
        Handle save file closed signal from acquisition thread.

        Args:
            filepath: Path to the closed save file
        """
        self.logger.info(f"Save file closed: {filepath}")

        # Show file size in status bar
        try:
            size_mb = Path(filepath).stat().st_size / (1024 * 1024)
            filename = Path(filepath).name
            self.statusBar().showMessage(f"Saved: {filename} ({size_mb:.1f} MB)", 10000)
        except Exception as e:
            self.logger.error(f"Failed to get file size: {e}")
            self.statusBar().showMessage(f"File saved: {Path(filepath).name}", 10000)

    @pyqtSlot(str)
    def _on_save_error(self, error_msg: str):
        """
        Handle save error signal from acquisition thread.

        Args:
            error_msg: Error message
        """
        self.logger.error(f"Save error: {error_msg}")
        QMessageBox.warning(self, "Save Error", f"Data save error:\n{error_msg}")

    def _update_status_bar(self):
        """Update status bar periodically."""
        # This can be used for additional status updates if needed
        pass

    @pyqtSlot(int)
    def _on_fft_size_changed(self, fft_size: int):
        """
        Handle FFT window size change.
        
        Args:
            fft_size: New FFT window size
        """
        if self.signal_processor is None:
            self.logger.warning("Cannot change FFT size: signal processor not initialized")
            return
        
        try:
            # Update FFT window size in signal processor
            self.signal_processor.configure_fft(window_size=fft_size)
            
            self.statusBar().showMessage(f"FFT window size changed to {fft_size}", 3000)
            self.logger.info(f"FFT window size changed to {fft_size}")
            
        except Exception as e:
            self.logger.error(f"Failed to change FFT size: {e}")
            QMessageBox.warning(
                self,
                "FFT Size Change Error",
                f"Failed to change FFT window size:\n{e}"
            )
    
    def _on_filter_changed(self, config: dict):
        """
        Handle filter configuration change.

        Args:
            config: Filter configuration dictionary
        """
        if self.signal_processor is None:
            return

        try:
            # Apply filter to signal processor
            self.signal_processor.configure_filter(
                filter_type=config['type'],
                filter_mode=config['mode'],
                cutoff=config['cutoff'],
                order=config['order'],
                enabled=config['enabled']
            )

            self.statusBar().showMessage(
                f"Filter configured: {config['type']} {config['mode']}, "
                f"cutoff={config['cutoff']}, order={config['order']}",
                3000
            )

        except Exception as e:
            self.logger.error(f"Failed to configure filter: {e}")
            QMessageBox.warning(
                self,
                "Filter Error",
                f"Failed to configure filter:\n{e}"
            )

    @pyqtSlot(bool)
    def _on_filter_enabled(self, enabled: bool):
        """
        Handle filter enable/disable.

        Args:
            enabled: True to enable filtering
        """
        if self.signal_processor is None:
            return

        self.signal_processor.enable_filtering(enabled)

        status = "enabled" if enabled else "disabled"
        self.statusBar().showMessage(f"Filtering {status}", 2000)
        self.logger.info(f"Filtering {status}")

    def _on_new_configuration(self):
        """Create new configuration."""
        # TODO: Implement configuration wizard
        pass

    def _on_open_configuration(self):
        """Open configuration from file."""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Open Configuration",
            "",
            "JSON Files (*.json);;All Files (*)"
        )

        if filename:
            try:
                self.config = DAQConfig.from_json(filename)
                if self.daq_config_panel:
                    self.daq_config_panel.set_config(self.config)
                if self.channel_config_widget:
                    self.channel_config_widget.set_channels(self.config.channels)

                self.statusBar().showMessage(f"Configuration loaded from {filename}", 3000)
                self.logger.info(f"Configuration loaded from {filename}")

            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Load Error",
                    f"Failed to load configuration:\n{e}"
                )

    def _on_save_configuration(self):
        """Save configuration to file."""
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save Configuration",
            "",
            "JSON Files (*.json);;All Files (*)"
        )

        if filename:
            try:
                # Get current configuration from UI
                if self.daq_config_panel:
                    self.config = self.daq_config_panel.get_config()
                if self.channel_config_widget:
                    self.config.channels = self.channel_config_widget.get_channels()

                self.config.to_json(filename)
                self.statusBar().showMessage(f"Configuration saved to {filename}", 3000)
                self.logger.info(f"Configuration saved to {filename}")

            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Save Error",
                    f"Failed to save configuration:\n{e}"
                )

    def _on_export_data(self):
        """Export acquired data using the export panel."""
        if self.signal_processor is None:
            QMessageBox.warning(
                self,
                "Export Data",
                "No data available. Start acquisition first."
            )
            return

        # Get data from signal processor
        try:
            # Get all available filtered data
            data = self.signal_processor.get_filtered_data()

            if data is None or data.shape[1] == 0:
                QMessageBox.warning(
                    self,
                    "Export Data",
                    "No data available in buffer."
                )
                return

            # Get channel info
            channel_names = self.signal_processor.get_channel_names()
            channel_units = self.signal_processor.get_channel_units()

            # Set data in export panel
            self.export_panel.set_data(
                data=data,
                sample_rate=self.config.sample_rate,
                channel_names=channel_names,
                channel_units=channel_units
            )

            # Show export panel
            self.export_dock = self.findChild(QDockWidget, "Export")
            if self.export_dock:
                self.export_dock.show()
                self.export_dock.raise_()

            self.statusBar().showMessage(
                f"Ready to export: {data.shape[1]:,} samples from {len(channel_names)} channels",
                5000
            )

        except Exception as e:
            self.logger.error(f"Failed to prepare export: {e}")
            QMessageBox.critical(
                self,
                "Export Error",
                f"Failed to prepare data for export:\n{e}"
            )

    def _on_clear_buffers(self):
        """Clear all data buffers."""
        if self.signal_processor:
            self.signal_processor.clear_buffers()
            self.statusBar().showMessage("Buffers cleared", 2000)
            self.logger.info("Buffers cleared")

    def _on_device_info(self):
        """Show device information."""
        try:
            devices = DAQManager.enumerate_devices()
            info_text = "Available DAQ Devices:\n\n"

            for dev in devices:
                info_text += f"Name: {dev['name']}\n"
                info_text += f"Type: {dev['product_type']}\n"
                info_text += f"Serial: {dev['serial_number']}\n"
                info_text += f"Channels: {dev['num_channels']}\n\n"

            QMessageBox.information(self, "Device Information", info_text)

        except Exception as e:
            QMessageBox.warning(
                self,
                "Device Information",
                f"Failed to enumerate devices:\n{e}"
            )

    def _on_settings(self):
        """Open settings dialog."""
        # TODO: Implement settings dialog
        QMessageBox.information(
            self,
            "Settings",
            "Settings dialog will be implemented later"
        )

    def _on_about(self):
        """Show about dialog."""
        about_text = f"""
        <h2>{AppConfig.APP_NAME}</h2>
        <p>Version {AppConfig.APP_VERSION}</p>
        <p>Real-time vibration data acquisition and analysis application
        for NI-9178 DAQ chassis with NI-9234 modules.</p>
        <p><b>Features:</b></p>
        <ul>
        <li>Multi-channel data acquisition (12+ channels)</li>
        <li>Real-time filtering and FFT analysis</li>
        <li>Live plotting and visualization</li>
        <li>Data export (CSV, HDF5, TDMS)</li>
        </ul>
        <p>Built with Python, PyQt5, and NI-DAQmx</p>
        """

        QMessageBox.about(self, "About", about_text)

    def closeEvent(self, event):
        """Handle window close event."""
        if self.is_acquiring:
            reply = QMessageBox.question(
                self,
                "Acquisition Running",
                "Acquisition is still running. Stop and exit?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )

            if reply == QMessageBox.Yes:
                self._on_stop_acquisition()
                # Auto-save disabled - edit config/default_config.json manually
                # self._save_app_settings()
                event.accept()
            else:
                event.ignore()
        else:
            # Auto-save disabled - edit config/default_config.json manually
            # self._save_app_settings()
            self.logger.info("Application closing")
            event.accept()


# Example usage
if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    app.setApplicationName(AppConfig.APP_NAME)
    app.setOrganizationName(AppConfig.ORGANIZATION_NAME)

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())

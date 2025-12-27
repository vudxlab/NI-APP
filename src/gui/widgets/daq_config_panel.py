"""
DAQ Configuration Panel.

This widget provides controls for configuring the DAQ device,
sample rate, acquisition mode, and start/stop controls.
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QLabel, QComboBox, QSpinBox, QPushButton, QLineEdit, QFileDialog,
    QCheckBox, QScrollArea
)
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QFont, QPalette, QColor
from pathlib import Path

from ...daq.daq_manager import DAQManager
from ...daq.daq_config import DAQConfig
from ...utils.logger import get_logger
from ...utils.constants import DAQDefaults, NI9234Specs


class DAQConfigPanel(QWidget):
    """
    DAQ configuration panel widget.

    Provides controls for:
    - Device selection
    - Sample rate configuration
    - Buffer size settings
    - Acquisition mode
    - Start/Stop buttons
    """

    # Signals
    start_requested = pyqtSignal()
    stop_requested = pyqtSignal()
    save_data_requested = pyqtSignal()
    config_changed = pyqtSignal(DAQConfig)
    downsample_threshold_changed = pyqtSignal(int)

    def __init__(self, parent=None):
        """
        Initialize the DAQ configuration panel.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)

        self.logger = get_logger(__name__)
        self.config: DAQConfig = None

        self._init_ui()
        self._connect_signals()

    def _init_ui(self):
        """Initialize the user interface."""
        # Create main layout for the scroll area
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Create scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # Create content widget to hold all groups
        content_widget = QWidget()
        layout = QVBoxLayout(content_widget)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)

        # Device selection group
        device_group = self._create_device_group()
        layout.addWidget(device_group)

        # Timing configuration group
        timing_group = self._create_timing_group()
        layout.addWidget(timing_group)

        # Auto-save configuration group
        autosave_group = self._create_autosave_group()
        layout.addWidget(autosave_group)

        # Acquisition control group
        control_group = self._create_control_group()
        layout.addWidget(control_group)

        # Status display
        status_group = self._create_status_group()
        layout.addWidget(status_group)

        # Add stretch to push everything to top
        layout.addStretch()

        # Set content widget to scroll area
        scroll_area.setWidget(content_widget)

        # Add scroll area to main layout
        main_layout.addWidget(scroll_area)

        # Now refresh devices after all widgets are created
        self._refresh_devices()

    def _create_device_group(self) -> QGroupBox:
        """Create device selection group."""
        group = QGroupBox("Device")
        layout = QFormLayout()

        # Device combo box
        self.device_combo = QComboBox()
        self.device_combo.setToolTip("Select DAQ device")
        layout.addRow("Device:", self.device_combo)

        # Refresh button
        refresh_layout = QHBoxLayout()
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.setToolTip("Refresh device list")
        self.refresh_button.clicked.connect(self._refresh_devices)
        refresh_layout.addWidget(self.refresh_button)
        refresh_layout.addStretch()
        layout.addRow("", refresh_layout)

        group.setLayout(layout)

        # Note: _refresh_devices() will be called after status_label is created

        return group

    def _create_timing_group(self) -> QGroupBox:
        """Create timing configuration group."""
        group = QGroupBox("Timing")
        layout = QFormLayout()

        # Sample rate combo box
        self.sample_rate_combo = QComboBox()
        self.sample_rate_combo.setToolTip("Sampling rate (Hz)")

        # Populate with common sample rates
        for rate in DAQDefaults.COMMON_SAMPLE_RATES:
            if rate <= NI9234Specs.MAX_SAMPLE_RATE:
                self.sample_rate_combo.addItem(f"{rate:,} Hz", rate)

        # Set default to 25.6 kHz
        default_idx = self.sample_rate_combo.findData(DAQDefaults.SAMPLE_RATE)
        if default_idx >= 0:
            self.sample_rate_combo.setCurrentIndex(default_idx)

        layout.addRow("Sample Rate:", self.sample_rate_combo)

        # Samples per channel
        self.samples_spin = QSpinBox()
        self.samples_spin.setRange(100, 10000)
        self.samples_spin.setValue(DAQDefaults.SAMPLES_PER_CHANNEL)
        self.samples_spin.setSingleStep(100)
        self.samples_spin.setToolTip("Number of samples to read per iteration")
        layout.addRow("Samples/Read:", self.samples_spin)

        # Downsample threshold
        self.downsample_spin = QSpinBox()
        self.downsample_spin.setRange(100, 10000)
        self.downsample_spin.setValue(1000)  # Will be updated from config
        self.downsample_spin.setSingleStep(100)
        self.downsample_spin.setToolTip("Maximum points to display before downsampling (plot optimization)")
        self.downsample_spin.valueChanged.connect(self.downsample_threshold_changed.emit)
        layout.addRow("Downsample:", self.downsample_spin)

        # Buffer duration
        self.buffer_duration_spin = QSpinBox()
        self.buffer_duration_spin.setRange(5, 300)
        self.buffer_duration_spin.setValue(DAQDefaults.BUFFER_DURATION_SECONDS)
        self.buffer_duration_spin.setSuffix(" s")
        self.buffer_duration_spin.setToolTip("Circular buffer duration")
        layout.addRow("Buffer Duration:", self.buffer_duration_spin)

        # Acquisition mode
        self.acq_mode_combo = QComboBox()
        self.acq_mode_combo.addItem("Continuous", DAQDefaults.ACQUISITION_MODE_CONTINUOUS)
        self.acq_mode_combo.addItem("Finite", DAQDefaults.ACQUISITION_MODE_FINITE)
        self.acq_mode_combo.setToolTip("Acquisition mode")
        layout.addRow("Mode:", self.acq_mode_combo)

        group.setLayout(layout)
        return group

    def _create_autosave_group(self) -> QGroupBox:
        """Create manual save configuration group."""
        group = QGroupBox("Data Save Settings")
        layout = QFormLayout()

        # Save Data Now button
        self.save_data_button = QPushButton("Save Data Now")
        self.save_data_button.setMinimumHeight(35)
        self.save_data_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                font-weight: bold;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #0D47A1;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        self.save_data_button.setToolTip("Save current stable data to file")
        self.save_data_button.setEnabled(False)  # Disabled until acquisition starts
        layout.addRow("", self.save_data_button)

        # Save location
        self.save_location_edit = QLineEdit()
        self.save_location_edit.setText(str(Path.home() / "NI_DAQ_Data"))
        self.save_location_edit.setToolTip("Directory where data files will be saved")

        browse_button = QPushButton("Browse...")
        browse_button.setToolTip("Select save directory")
        browse_button.clicked.connect(self._on_browse_save_location)

        location_layout = QHBoxLayout()
        location_layout.addWidget(self.save_location_edit)
        location_layout.addWidget(browse_button)
        layout.addRow("Save Location:", location_layout)

        # File format
        self.file_format_combo = QComboBox()
        self.file_format_combo.addItem("HDF5 (Recommended)", "hdf5")
        self.file_format_combo.addItem("TDMS", "tdms")
        self.file_format_combo.addItem("CSV", "csv")
        self.file_format_combo.setToolTip("File format for saved data")
        default_format_idx = self.file_format_combo.findData("csv")
        if default_format_idx >= 0:
            self.file_format_combo.setCurrentIndex(default_format_idx)
        layout.addRow("Format:", self.file_format_combo)

        # File naming prefix
        self.file_prefix_edit = QLineEdit()
        self.file_prefix_edit.setText("acquisition")
        self.file_prefix_edit.setToolTip("Prefix for data filenames (timestamp will be added)")
        layout.addRow("File Prefix:", self.file_prefix_edit)

        # Compression level (for HDF5)
        self.compression_level_spin = QSpinBox()
        self.compression_level_spin.setRange(0, 9)
        self.compression_level_spin.setValue(4)
        self.compression_level_spin.setToolTip("HDF5 compression level (0=none, 9=maximum)")
        layout.addRow("Compression:", self.compression_level_spin)

        group.setLayout(layout)
        return group

    def _create_control_group(self) -> QGroupBox:
        """Create acquisition control group."""
        group = QGroupBox("Control")
        layout = QVBoxLayout()

        # Start button
        self.start_button = QPushButton("Start Acquisition")
        self.start_button.setMinimumHeight(40)
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        self.start_button.clicked.connect(self._on_start_clicked)
        layout.addWidget(self.start_button)

        # Stop button
        self.stop_button = QPushButton("Stop Acquisition")
        self.stop_button.setMinimumHeight(40)
        self.stop_button.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                font-weight: bold;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
            QPushButton:pressed {
                background-color: #c41408;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        self.stop_button.clicked.connect(self._on_stop_clicked)
        self.stop_button.setEnabled(False)
        layout.addWidget(self.stop_button)

        group.setLayout(layout)
        return group

    def _create_status_group(self) -> QGroupBox:
        """Create status display group."""
        group = QGroupBox("Status")
        layout = QFormLayout()

        # Status indicator
        self.status_label = QLabel("Not Configured")
        self.status_label.setAlignment(Qt.AlignCenter)
        font = QFont()
        font.setBold(True)
        self.status_label.setFont(font)
        self._set_status_color("gray")
        layout.addRow("", self.status_label)
        
        # Add separator
        separator = QLabel()
        separator.setFrameStyle(QLabel.HLine | QLabel.Sunken)
        layout.addRow(separator)
        
        # Device info section
        device_info_label = QLabel("Connected Device:")
        device_font = QFont()
        device_font.setBold(True)
        device_info_label.setFont(device_font)
        layout.addRow(device_info_label)
        
        # Device name
        self.device_name_label = QLabel("No device")
        self.device_name_label.setStyleSheet("color: #666; margin-left: 10px;")
        layout.addRow("  Device:", self.device_name_label)
        
        # Product type
        self.product_type_label = QLabel("-")
        self.product_type_label.setStyleSheet("color: #666; margin-left: 10px;")
        layout.addRow("  Type:", self.product_type_label)
        
        # Serial number
        self.serial_number_label = QLabel("-")
        self.serial_number_label.setStyleSheet("color: #666; margin-left: 10px;")
        layout.addRow("  Serial:", self.serial_number_label)
        
        # Modules section
        modules_label = QLabel("Modules:")
        modules_label.setFont(device_font)
        layout.addRow(modules_label)
        
        # Module list (will be populated dynamically)
        self.modules_label = QLabel("No modules")
        self.modules_label.setStyleSheet("color: #666; margin-left: 10px; font-size: 10px;")
        self.modules_label.setWordWrap(True)
        layout.addRow("", self.modules_label)
        
        # Add another separator
        separator2 = QLabel()
        separator2.setFrameStyle(QLabel.HLine | QLabel.Sunken)
        layout.addRow(separator2)

        # Acquisition info
        acq_info_label = QLabel("Acquisition Info:")
        acq_info_label.setFont(device_font)
        layout.addRow(acq_info_label)

        # Enabled channels
        self.channels_label = QLabel("0")
        self.channels_label.setStyleSheet("color: #666; margin-left: 10px;")
        layout.addRow("  Channels:", self.channels_label)

        # Nyquist frequency
        self.nyquist_label = QLabel("0 Hz")
        self.nyquist_label.setStyleSheet("color: #666; margin-left: 10px;")
        layout.addRow("  Nyquist:", self.nyquist_label)

        group.setLayout(layout)
        return group

    def _connect_signals(self):
        """Connect internal signals."""
        self.sample_rate_combo.currentIndexChanged.connect(self._on_config_changed)
        self.samples_spin.valueChanged.connect(self._on_config_changed)
        self.buffer_duration_spin.valueChanged.connect(self._on_config_changed)
        self.acq_mode_combo.currentIndexChanged.connect(self._on_config_changed)
        self.device_combo.currentIndexChanged.connect(self._on_device_changed)
        self.save_data_button.clicked.connect(self._on_save_data_clicked)

    def _refresh_devices(self):
        """Refresh the list of available devices."""
        self.device_combo.clear()

        try:
            devices = DAQManager.enumerate_devices()

            if devices:
                for device in devices:
                    display_name = f"{device['name']} ({device['product_type']})"
                    self.device_combo.addItem(display_name, device['name'])

                self.status_label.setText("Ready")
                self._set_status_color("green")
                
                # Update device info for first device
                if len(devices) > 0:
                    self._update_device_info(devices[0])
            else:
                self.device_combo.addItem("No devices found", None)
                self.status_label.setText("No Devices")
                self._set_status_color("orange")
                self._clear_device_info()

        except Exception as e:
            self.logger.error(f"Failed to enumerate devices: {e}")
            self.device_combo.addItem("Error enumerating devices", None)
            self.status_label.setText("Error")
            self._set_status_color("red")
            self._clear_device_info()

    def _set_status_color(self, color: str):
        """
        Set status label color.

        Args:
            color: Color name ("green", "red", "orange", "gray")
        """
        colors = {
            "green": "#4CAF50",
            "red": "#f44336",
            "orange": "#FF9800",
            "gray": "#9E9E9E"
        }

        color_code = colors.get(color, "#9E9E9E")
        self.status_label.setStyleSheet(f"""
            QLabel {{
                background-color: {color_code};
                color: white;
                padding: 8px;
                border-radius: 4px;
            }}
        """)
    
    def _update_device_info(self, device_info: dict):
        """
        Update device information display.
        
        Args:
            device_info: Device information dictionary
        """
        device_name = device_info.get('name', 'Unknown')
        
        self.device_name_label.setText(device_name)
        self.device_name_label.setStyleSheet("color: #2196F3; font-weight: bold; margin-left: 10px;")
        
        self.product_type_label.setText(device_info.get('product_type', 'Unknown'))
        self.product_type_label.setStyleSheet("color: #4CAF50; margin-left: 10px;")
        
        serial = device_info.get('serial_number', 'N/A')
        if serial and serial != '0':
            self.serial_number_label.setText(serial)
        else:
            self.serial_number_label.setText('N/A')
        self.serial_number_label.setStyleSheet("color: #666; margin-left: 10px;")
        
        # Get and display modules
        try:
            modules = DAQManager.get_device_modules(device_name)
            if modules:
                module_text = ""
                for i, mod in enumerate(modules):
                    module_text += f"  • {mod['name']}: {mod['product_type']} ({mod['num_channels']} ch)\n"
                self.modules_label.setText(module_text.strip())
                self.modules_label.setStyleSheet("color: #666; margin-left: 10px; font-size: 10px;")
            else:
                self.modules_label.setText("  No modules detected")
                self.modules_label.setStyleSheet("color: #999; margin-left: 10px; font-size: 10px;")
        except Exception as e:
            self.logger.error(f"Failed to get modules: {e}")
            self.modules_label.setText("  Error getting modules")
            self.modules_label.setStyleSheet("color: #f44336; margin-left: 10px; font-size: 10px;")
        
        self.logger.info(f"Device info updated: {device_name}")
    
    def _clear_device_info(self):
        """Clear device information display."""
        self.device_name_label.setText("No device")
        self.device_name_label.setStyleSheet("color: #999; margin-left: 10px;")
        
        self.product_type_label.setText("-")
        self.product_type_label.setStyleSheet("color: #999; margin-left: 10px;")
        
        self.serial_number_label.setText("-")
        self.serial_number_label.setStyleSheet("color: #999; margin-left: 10px;")
        
        self.modules_label.setText("No modules")
        self.modules_label.setStyleSheet("color: #999; margin-left: 10px; font-size: 10px;")
    
    def _on_device_changed(self):
        """Handle device selection change."""
        # Update device info when device changes
        device_name = self.device_combo.currentData()
        if device_name:
            try:
                devices = DAQManager.enumerate_devices()
                for device in devices:
                    if device['name'] == device_name:
                        self._update_device_info(device)
                        break
            except Exception as e:
                self.logger.error(f"Failed to get device info: {e}")
                self._clear_device_info()
        else:
            self._clear_device_info()
        
        # Also trigger config changed
        self._on_config_changed()

    def _on_start_clicked(self):
        """Handle start button click."""
        self.start_requested.emit()

    def _on_stop_clicked(self):
        """Handle stop button click."""
        self.stop_requested.emit()

    def _on_save_data_clicked(self):
        """Handle save data button click."""
        self.save_data_requested.emit()

    def _on_browse_save_location(self):
        """Open directory picker for save location."""
        current_path = self.save_location_edit.text()
        if not current_path:
            current_path = str(Path.home())

        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Save Location",
            current_path,
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )

        if directory:
            self.save_location_edit.setText(directory)
            self.logger.info(f"Save location set to: {directory}")

    def _on_config_changed(self):
        """Handle configuration change."""
        # Update Nyquist frequency display
        sample_rate = self.sample_rate_combo.currentData()
        if sample_rate:
            nyquist = sample_rate / 2.0
            self.nyquist_label.setText(f"{nyquist:,.0f} Hz")

    def set_config(self, config: DAQConfig):
        """
        Set configuration from DAQConfig object.

        Args:
            config: DAQ configuration
        """
        self.config = config

        # Block signals during update
        self.sample_rate_combo.blockSignals(True)
        self.samples_spin.blockSignals(True)
        self.buffer_duration_spin.blockSignals(True)
        self.acq_mode_combo.blockSignals(True)
        self.device_combo.blockSignals(True)

        # Set values
        sample_rate_idx = self.sample_rate_combo.findData(config.sample_rate)
        if sample_rate_idx >= 0:
            self.sample_rate_combo.setCurrentIndex(sample_rate_idx)

        self.samples_spin.setValue(config.samples_per_channel)
        self.buffer_duration_spin.setValue(config.buffer_size_seconds)

        acq_mode_idx = self.acq_mode_combo.findData(config.acquisition_mode)
        if acq_mode_idx >= 0:
            self.acq_mode_combo.setCurrentIndex(acq_mode_idx)

        device_idx = self.device_combo.findData(config.device_name)
        if device_idx >= 0:
            self.device_combo.setCurrentIndex(device_idx)

        # Update channels label
        self.channels_label.setText(str(config.get_num_enabled_channels()))

        # Unblock signals
        self.sample_rate_combo.blockSignals(False)
        self.samples_spin.blockSignals(False)
        self.buffer_duration_spin.blockSignals(False)
        self.acq_mode_combo.blockSignals(False)
        self.device_combo.blockSignals(False)

        # Trigger update
        self._on_config_changed()

        self.logger.debug(f"Configuration set: {config}")

    def get_config(self) -> DAQConfig:
        """
        Get current configuration as DAQConfig object.

        Returns:
            Current DAQ configuration
        """
        if self.config is None:
            # Create new config with current values
            from ...daq.daq_config import create_default_config
            self.config = create_default_config()

        # Update with current UI values
        self.config.device_name = self.device_combo.currentData() or ""
        self.config.sample_rate = self.sample_rate_combo.currentData()
        self.config.samples_per_channel = self.samples_spin.value()
        self.config.buffer_size_seconds = self.buffer_duration_spin.value()
        self.config.acquisition_mode = self.acq_mode_combo.currentData()

        return self.config

    def get_save_config(self) -> dict:
        """
        Get data save configuration.

        Returns:
            Dictionary with save settings:
                - save_location: str
                - file_format: str ('hdf5', 'tdms', or 'csv')
                - file_prefix: str
                - compression_level: int
        """
        return {
            'save_location': self.save_location_edit.text(),
            'file_format': self.file_format_combo.currentData(),
            'file_prefix': self.file_prefix_edit.text(),
            'compression_level': self.compression_level_spin.value()
        }

    def get_autosave_config(self) -> dict:
        """
        Get auto-save configuration (deprecated, use get_save_config).

        Returns:
            Dictionary with save settings
        """
        config = self.get_save_config()
        config['enabled'] = False  # No longer auto-saving
        return config

    def set_acquisition_state(self, running: bool):
        """
        Update UI based on acquisition state.

        Args:
            running: True if acquisition is running
        """
        # Disable/enable controls
        self.device_combo.setEnabled(not running)
        self.sample_rate_combo.setEnabled(not running)
        self.samples_spin.setEnabled(not running)
        self.buffer_duration_spin.setEnabled(not running)
        self.acq_mode_combo.setEnabled(not running)
        self.refresh_button.setEnabled(not running)

        # Disable/enable save controls (keep them enabled during acquisition for flexibility)
        self.save_location_edit.setEnabled(not running)
        self.file_format_combo.setEnabled(not running)
        self.file_prefix_edit.setEnabled(not running)
        self.compression_level_spin.setEnabled(not running)

        # Update buttons
        self.start_button.setEnabled(not running)
        self.stop_button.setEnabled(running)
        self.save_data_button.setEnabled(running)  # Enable save button during acquisition

        # Reset save button text when stopping acquisition
        if not running:
            self.save_data_button.setText("Save Data Now")

        # Update status
        if running:
            self.status_label.setText("Acquiring")
            self._set_status_color("green")
        else:
            self.status_label.setText("Ready")
            self._set_status_color("gray")

    def set_saving_state(self, saving: bool):
        """
        Update save button based on saving state.

        Args:
            saving: True if currently saving data
        """
        if saving:
            self.save_data_button.setText("Stop Saving")
            self.save_data_button.setStyleSheet("""
                QPushButton {
                    background-color: #FF9800;
                    color: white;
                    font-weight: bold;
                    border: none;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #F57C00;
                }
                QPushButton:pressed {
                    background-color: #E65100;
                }
                QPushButton:disabled {
                    background-color: #cccccc;
                    color: #666666;
                }
            """)
        else:
            self.save_data_button.setText("Save Data Now")
            self.save_data_button.setStyleSheet("""
                QPushButton {
                    background-color: #2196F3;
                    color: white;
                    font-weight: bold;
                    border: none;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #1976D2;
                }
                QPushButton:pressed {
                    background-color: #0D47A1;
                }
                QPushButton:disabled {
                    background-color: #cccccc;
                    color: #666666;
                }
            """)

    def get_downsample_threshold(self) -> int:
        """
        Get current downsample threshold.

        Returns:
            Current downsample threshold value
        """
        return self.downsample_spin.value()

    def set_downsample_threshold(self, threshold: int):
        """
        Set downsample threshold.

        Args:
            threshold: Downsample threshold value (100-10000)
        """
        self.downsample_spin.setValue(threshold)


# Example usage
if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication
    from ...daq.daq_config import create_default_config

    app = QApplication(sys.argv)

    # Create panel
    panel = DAQConfigPanel()

    # Set default config
    config = create_default_config(device_name="cDAQ1", num_modules=3)
    panel.set_config(config)

    # Connect signals
    panel.start_requested.connect(lambda: print("Start requested"))
    panel.stop_requested.connect(lambda: print("Stop requested"))

    panel.show()

    sys.exit(app.exec_())

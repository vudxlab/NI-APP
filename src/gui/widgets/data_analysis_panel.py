"""
Data Analysis Panel.

Provides controls for analyzing saved data using the existing
Time Domain and Frequency Domain tabs.
"""

import re
from pathlib import Path

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QLabel, QComboBox, QPushButton, QFileDialog, QLineEdit,
    QDoubleSpinBox, QCheckBox, QSpinBox, QScrollArea, QFrame
)
from PyQt5.QtCore import QTimer, Qt

from ...export.data_file_reader import DataFileReader
from ...utils.logger import get_logger


class DataAnalysisPanel(QWidget):
    """Left-side panel for saved-data analysis with modular frame structure."""

    def __init__(self, realtime_plot_widget, fft_plot_widget, plot_tabs=None, parent=None):
        super().__init__(parent)

        self.logger = get_logger(__name__)

        self.realtime_plot_widget = realtime_plot_widget
        self.fft_plot_widget = fft_plot_widget
        self.plot_tabs = plot_tabs

        self.current_data_file: Path = None
        self.data_file_info = None
        self.sample_rate = 25600.0
        self.channel_units = "g"

        self._init_ui()
        self._connect_signals()

    def _init_ui(self):
        """Initialize the user interface with modular frames."""
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

        # Frame 1: Data Source
        data_source_group = self._create_data_source_group()
        layout.addWidget(data_source_group)

        # Frame 2: Signal Configuration
        signal_config_group = self._create_signal_config_group()
        layout.addWidget(signal_config_group)

        # Frame 3: Analysis Options
        analysis_options_group = self._create_analysis_options_group()
        layout.addWidget(analysis_options_group)

        # Frame 4: Analysis Control
        control_group = self._create_control_group()
        layout.addWidget(control_group)

        # Frame 5: Export Result
        export_result_group = self._create_export_result_group()
        layout.addWidget(export_result_group)

        # Add stretch to push everything to top
        layout.addStretch()

        # Set content widget to scroll area
        scroll_area.setWidget(content_widget)

        # Add scroll area to main layout
        main_layout.addWidget(scroll_area)

    def _create_data_source_group(self) -> QGroupBox:
        """Create data source frame."""
        group = QGroupBox("Data Source")
        layout = QFormLayout()

        # File path display
        file_layout = QHBoxLayout()
        self.file_path_edit = QLineEdit()
        self.file_path_edit.setReadOnly(True)
        self.file_path_edit.setPlaceholderText("No file selected")
        self.file_path_edit.setStyleSheet("color: #666;")
        file_layout.addWidget(self.file_path_edit)

        self.browse_button = QPushButton("Browse...")
        self.browse_button.setToolTip("Select data file (HDF5, CSV, TDMS, MAT)")
        self.browse_button.clicked.connect(self._on_browse_file)
        file_layout.addWidget(self.browse_button)

        layout.addRow("File:", file_layout)

        # File information labels
        self.file_format_label = QLabel("-")
        self.file_format_label.setStyleSheet("color: #666;")
        layout.addRow("Format:", self.file_format_label)

        self.file_channels_label = QLabel("-")
        self.file_channels_label.setStyleSheet("color: #666;")
        layout.addRow("Channels:", self.file_channels_label)

        self.file_duration_label = QLabel("-")
        self.file_duration_label.setStyleSheet("color: #666;")
        layout.addRow("Duration:", self.file_duration_label)

        self.file_size_label = QLabel("-")
        self.file_size_label.setStyleSheet("color: #666;")
        layout.addRow("Size:", self.file_size_label)

        group.setLayout(layout)
        return group

    def _create_signal_config_group(self) -> QGroupBox:
        """Create signal configuration frame."""
        group = QGroupBox("Signal Configuration")
        layout = QFormLayout()

        # Sample Rate
        self.sample_rate_spin = QDoubleSpinBox()
        self.sample_rate_spin.setRange(1.0, 1000000.0)
        self.sample_rate_spin.setDecimals(1)
        self.sample_rate_spin.setValue(25600.0)
        self.sample_rate_spin.setSuffix(" Hz")
        self.sample_rate_spin.setToolTip(
            "Sample rate (Hz) - Auto-filled from file metadata if available.\n"
            "Manually enter if file does not contain sample rate information."
        )
        self.sample_rate_spin.valueChanged.connect(self._on_sample_rate_changed)
        layout.addRow("Sample Rate:", self.sample_rate_spin)

        # Sample Rate Status
        self.sample_rate_status_label = QLabel("")
        self.sample_rate_status_label.setStyleSheet("color: #666; font-style: italic; font-size: 10px;")
        layout.addRow("", self.sample_rate_status_label)

        # Time Window
        self.time_window_combo = QComboBox()
        for seconds in [1, 2, 5, 10, 20, 50, 100, 200]:
            self.time_window_combo.addItem(f"{seconds}s", seconds)
        self.time_window_combo.setCurrentText("10s")
        self.time_window_combo.setToolTip("Duration of data to load from saved file")
        layout.addRow("Time Window:", self.time_window_combo)

        # Channel Units
        self.channel_units_combo = QComboBox()
        self.channel_units_combo.addItem("g (acceleration)", "g")
        self.channel_units_combo.addItem("m/s² (acceleration)", "m/s²")
        self.channel_units_combo.addItem("V (voltage)", "V")
        self.channel_units_combo.addItem("Pa (pressure)", "Pa")
        self.channel_units_combo.setCurrentIndex(0)
        self.channel_units_combo.setToolTip("Units for channel data display")
        self.channel_units_combo.currentIndexChanged.connect(self._on_units_changed)
        layout.addRow("Units:", self.channel_units_combo)

        group.setLayout(layout)
        return group

    def _create_analysis_options_group(self) -> QGroupBox:
        """Create analysis options frame."""
        group = QGroupBox("Analysis Options")
        layout = QFormLayout()

        # Time Domain section label
        time_domain_label = QLabel("Time Domain:")
        time_domain_label.setStyleSheet("font-weight: bold; color: #2196F3;")
        layout.addRow(time_domain_label)

        # Plot Mode
        self.plot_mode_combo = QComboBox()
        self.plot_mode_combo.addItem("Overlay", "overlay")
        self.plot_mode_combo.addItem("Stack", "stack")
        self.plot_mode_combo.setToolTip("Plot display mode for time domain")
        self.plot_mode_combo.currentIndexChanged.connect(self._on_plot_mode_changed)
        layout.addRow("  Plot Mode:", self.plot_mode_combo)

        # Auto-scale
        self.auto_scale_checkbox = QCheckBox("Enable auto-scaling")
        self.auto_scale_checkbox.setChecked(True)
        self.auto_scale_checkbox.setToolTip("Automatically scale Y-axis to fit data")
        layout.addRow("  Auto-scale:", self.auto_scale_checkbox)

        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        layout.addRow(separator)

        # FFT section label
        fft_label = QLabel("FFT:")
        fft_label.setStyleSheet("font-weight: bold; color: #FF9800;")
        layout.addRow(fft_label)

        # FFT Window
        self.fft_window_combo = QComboBox()
        for seconds in [10, 20, 50, 100, 200]:
            self.fft_window_combo.addItem(f"{seconds}s", seconds)
        self.fft_window_combo.setCurrentText("10s")
        self.fft_window_combo.setToolTip("FFT window duration for frequency analysis")
        layout.addRow("  FFT Window:", self.fft_window_combo)

        # Magnitude Scale
        self.magnitude_scale_combo = QComboBox()
        self.magnitude_scale_combo.addItem("dB (logarithmic)", "db")
        self.magnitude_scale_combo.addItem("Linear", "linear")
        self.magnitude_scale_combo.setCurrentIndex(0)
        self.magnitude_scale_combo.setToolTip("FFT magnitude scale")
        layout.addRow("  Scale:", self.magnitude_scale_combo)

        # Frequency Range
        self.freq_range_combo = QComboBox()
        self.freq_range_combo.addItem("0-20 Hz", (0, 20))
        self.freq_range_combo.addItem("0-100 Hz", (0, 100))
        self.freq_range_combo.addItem("0-1 kHz", (0, 1000))
        self.freq_range_combo.addItem("0-5 kHz", (0, 5000))
        self.freq_range_combo.addItem("Full Range", None)
        self.freq_range_combo.setCurrentIndex(3)  # Default to 0-5 kHz
        self.freq_range_combo.setToolTip("Frequency range for FFT display")
        layout.addRow("  Range:", self.freq_range_combo)

        # Peak Detection
        self.peak_detection_checkbox = QCheckBox("Enable peak detection")
        self.peak_detection_checkbox.setChecked(True)
        self.peak_detection_checkbox.setToolTip("Detect and mark frequency peaks")
        self.peak_detection_checkbox.toggled.connect(self._on_peak_detection_toggled)
        layout.addRow("  Peak Detection:", self.peak_detection_checkbox)

        # Peak Threshold
        self.peak_threshold_spin = QDoubleSpinBox()
        self.peak_threshold_spin.setRange(0.01, 1.0)
        self.peak_threshold_spin.setDecimals(2)
        self.peak_threshold_spin.setSingleStep(0.01)
        self.peak_threshold_spin.setValue(0.1)
        self.peak_threshold_spin.setToolTip("Minimum threshold for peak detection")
        layout.addRow("    Threshold:", self.peak_threshold_spin)

        # Max Peaks
        self.max_peaks_spin = QSpinBox()
        self.max_peaks_spin.setRange(1, 20)
        self.max_peaks_spin.setValue(10)
        self.max_peaks_spin.setToolTip("Maximum number of peaks to display")
        layout.addRow("    Max Peaks:", self.max_peaks_spin)

        group.setLayout(layout)
        return group

    def _create_control_group(self) -> QGroupBox:
        """Create analysis control frame."""
        group = QGroupBox("Control")
        layout = QVBoxLayout()

        # Load Time Domain button
        self.load_time_button = QPushButton("Load Time Domain")
        self.load_time_button.setMinimumHeight(40)
        self.load_time_button.setStyleSheet("""
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
        self.load_time_button.setToolTip("Load time-domain data into Time Domain tab")
        self.load_time_button.setEnabled(False)
        self.load_time_button.clicked.connect(self._on_load_time_domain)
        layout.addWidget(self.load_time_button)

        # Analyze FFT button
        self.analyze_fft_button = QPushButton("Analyze FFT")
        self.analyze_fft_button.setMinimumHeight(40)
        self.analyze_fft_button.setStyleSheet("""
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
        self.analyze_fft_button.setToolTip("Perform FFT analysis and display in Frequency Domain tab")
        self.analyze_fft_button.setEnabled(False)
        self.analyze_fft_button.clicked.connect(self._on_analyze_fft)
        layout.addWidget(self.analyze_fft_button)

        # Clear/Reset button
        self.clear_button = QPushButton("Clear/Reset")
        self.clear_button.setMinimumHeight(30)
        self.clear_button.setStyleSheet("""
            QPushButton {
                background-color: #9E9E9E;
                color: white;
                font-weight: bold;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #757575;
            }
            QPushButton:pressed {
                background-color: #616161;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        self.clear_button.setToolTip("Clear loaded data and reset analysis")
        self.clear_button.setEnabled(False)
        self.clear_button.clicked.connect(self._on_clear_reset)
        layout.addWidget(self.clear_button)

        group.setLayout(layout)
        return group

    def _create_export_result_group(self) -> QGroupBox:
        """Create export result frame (placeholder)."""
        group = QGroupBox("Export Result")
        layout = QVBoxLayout()

        layout.addStretch()

        # Placeholder label
        placeholder_label = QLabel("Export functionality - Coming soon")
        placeholder_label.setAlignment(Qt.AlignCenter)
        placeholder_label.setStyleSheet("color: #999; font-style: italic;")
        layout.addWidget(placeholder_label)

        layout.addStretch()

        group.setLayout(layout)
        group.setMinimumHeight(60)
        return group

    def _connect_signals(self):
        """Connect internal signals."""
        # Signals are already connected in widget creation
        pass

    # ========== Event Handlers ==========

    def _on_browse_file(self):
        """Handle browse for saved data file."""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Open Saved Data",
            "",
            "Data Files (*.h5 *.hdf5 *.tdms *.csv *.mat);;All Files (*)"
        )

        if filename:
            self.set_data_file(filename)

    def _on_load_time_domain(self):
        """Load saved data into the Time Domain tab."""
        if not self.current_data_file or not self.current_data_file.exists():
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(self, "No Data", "No data file available for analysis.")
            return

        duration = float(self.time_window_combo.currentData())

        try:
            reader = DataFileReader()
            data, metadata = reader.read_recent_seconds(
                str(self.current_data_file),
                duration,
                self.sample_rate
            )

            samples_read = metadata.get('samples_read', 0)
            actual_duration = metadata.get('duration_read', 0)

            if samples_read < 2:
                from PyQt5.QtWidgets import QMessageBox
                QMessageBox.warning(
                    self,
                    "Insufficient Data",
                    "Not enough data for time-domain display."
                )
                return

            sample_rate = self._parse_sample_rate(metadata.get('sample_rate'))
            if sample_rate:
                self.sample_rate = sample_rate

            n_channels = data.shape[0]
            channel_names = [f"Channel {i + 1}" for i in range(n_channels)]

            if self.realtime_plot_widget:
                self.realtime_plot_widget.configure(
                    n_channels=n_channels,
                    sample_rate=self.sample_rate,
                    channel_names=channel_names,
                    channel_units=self.channel_units
                )
                self.realtime_plot_widget.update_plot(data, 0.0)
                QTimer.singleShot(
                    self.realtime_plot_widget.update_interval_ms + 10,
                    self.realtime_plot_widget.stop
                )
                if self.plot_tabs is not None:
                    self.plot_tabs.setCurrentIndex(0)

            self.logger.info(
                f"Loaded {samples_read} samples ({actual_duration:.2f}s) for time-domain display"
            )

        except Exception as e:
            self.logger.error(f"Time-domain analysis failed: {e}")
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Analysis Error", f"Failed to load data:\n{str(e)}")

    def _on_analyze_fft(self):
        """Trigger FFT analysis using FFT widget."""
        if not self.current_data_file or not self.current_data_file.exists():
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(self, "No Data", "No data file available for FFT analysis.")
            return

        if self.fft_plot_widget:
            # Get FFT options from panel
            fft_window = self.fft_window_combo.currentData()
            magnitude_scale = self.magnitude_scale_combo.currentData()
            freq_range = self.freq_range_combo.currentData()
            peak_detection = self.peak_detection_checkbox.isChecked()
            peak_threshold = self.peak_threshold_spin.value()
            max_peaks = self.max_peaks_spin.value()

            # TODO: Apply these options to FFT widget before analysis
            # For now, just trigger the existing analyze saved data functionality

            # Switch to FFT tab
            if self.plot_tabs is not None:
                self.plot_tabs.setCurrentIndex(1)

            self.logger.info(f"FFT analysis requested with window={fft_window}s, scale={magnitude_scale}")
            self.logger.info(f"Peak detection: {peak_detection}, threshold: {peak_threshold}, max_peaks: {max_peaks}")

    def _on_clear_reset(self):
        """Clear loaded data and reset analysis."""
        # Clear file info
        self.current_data_file = None
        self.data_file_info = None

        # Reset file display
        self.file_path_edit.clear()
        self.file_path_edit.setPlaceholderText("No file selected")
        self.file_format_label.setText("-")
        self.file_channels_label.setText("-")
        self.file_duration_label.setText("-")
        self.file_size_label.setText("-")

        # Disable buttons
        self.load_time_button.setEnabled(False)
        self.analyze_fft_button.setEnabled(False)
        self.clear_button.setEnabled(False)

        # Clear plots
        if self.realtime_plot_widget:
            self.realtime_plot_widget._on_clear()
        if self.fft_plot_widget:
            self.fft_plot_widget._on_clear()

        self.logger.info("Data cleared and analysis reset")

    def _on_sample_rate_changed(self, value):
        """Handle sample rate change."""
        self.sample_rate = value
        # Sync to FFT widget
        if self.fft_plot_widget:
            self.fft_plot_widget.sample_rate = value
        # Update status to show manual entry
        if self.sample_rate_status_label.text() == "(from file)":
            self.sample_rate_status_label.setText("(manual)")
            self.sample_rate_status_label.setStyleSheet("color: #2196F3; font-style: italic; font-size: 10px;")
        self.logger.debug(f"Sample rate changed to: {value} Hz")

    def _on_units_changed(self):
        """Handle units change."""
        self.channel_units = self.channel_units_combo.currentData()
        self.logger.debug(f"Channel units changed to: {self.channel_units}")

    def _on_plot_mode_changed(self):
        """Handle plot mode change."""
        plot_mode = self.plot_mode_combo.currentData()
        # TODO: Apply plot mode to realtime plot widget if data is loaded
        self.logger.debug(f"Plot mode changed to: {plot_mode}")

    def _on_peak_detection_toggled(self, checked):
        """Handle peak detection toggle."""
        # Enable/disable peak-related controls
        self.peak_threshold_spin.setEnabled(checked)
        self.max_peaks_spin.setEnabled(checked)
        self.logger.debug(f"Peak detection: {checked}")

    # ========== Public Methods ==========

    def set_data_file(self, filepath: str):
        """
        Set the current data file for analysis.

        Args:
            filepath: Path to the data file created during acquisition
        """
        self.current_data_file = Path(filepath)

        # Update file path display
        self.file_path_edit.setText(str(self.current_data_file))
        self.file_path_edit.setStyleSheet("color: #2196F3;")

        try:
            reader = DataFileReader()
            self.data_file_info = reader.get_file_info(filepath)

            # Update file format
            file_ext = self.current_data_file.suffix.upper()
            format_map = {'.H5': 'HDF5', '.HDF5': 'HDF5', '.TDMS': 'TDMS', '.CSV': 'CSV', '.MAT': 'MATLAB'}
            file_format = format_map.get(file_ext, file_ext[1:] if file_ext else 'Unknown')
            self.file_format_label.setText(file_format)
            self.file_format_label.setStyleSheet("color: #4CAF50; font-weight: bold;")

            # Update channels
            n_channels = self.data_file_info.get('num_channels', 0)
            self.file_channels_label.setText(f"{n_channels} channels")
            self.file_channels_label.setStyleSheet("color: #2196F3;")

            # Update duration
            duration = self.data_file_info.get('duration_seconds', 0)
            if duration > 0:
                self.file_duration_label.setText(f"{duration:.1f} seconds")
                self.file_duration_label.setStyleSheet("color: #2196F3;")
            else:
                self.file_duration_label.setText("Unknown")
                self.file_duration_label.setStyleSheet("color: #999;")

            # Update file size
            if self.current_data_file.exists():
                file_size_bytes = self.current_data_file.stat().st_size
                file_size_mb = file_size_bytes / (1024 * 1024)
                self.file_size_label.setText(f"{file_size_mb:.2f} MB")
                self.file_size_label.setStyleSheet("color: #666;")

            # Try to get sample rate from file
            file_sample_rate = self._parse_sample_rate(self.data_file_info.get('sample_rate'))

            if file_sample_rate and file_sample_rate > 0:
                # Auto-fill from file metadata
                self.sample_rate = file_sample_rate
                self.sample_rate_spin.blockSignals(True)
                self.sample_rate_spin.setValue(file_sample_rate)
                self.sample_rate_spin.blockSignals(False)
                self.sample_rate_status_label.setText("(from file)")
                self.sample_rate_status_label.setStyleSheet("color: #4CAF50; font-style: italic; font-size: 10px;")
            else:
                # No sample rate in file - keep current value
                self.sample_rate = self.sample_rate_spin.value()
                self.sample_rate_status_label.setText("(manual - file has no sample rate)")
                self.sample_rate_status_label.setStyleSheet("color: #FF9800; font-style: italic; font-size: 10px;")

            # Sync sample rate to FFT widget
            if self.fft_plot_widget:
                self.fft_plot_widget.set_data_file(filepath)
                self.fft_plot_widget.sample_rate = self.sample_rate

            # Enable control buttons
            self.load_time_button.setEnabled(True)
            self.analyze_fft_button.setEnabled(True)
            self.clear_button.setEnabled(True)

            self.logger.info(f"Saved analysis file set: {filepath}")

        except Exception as e:
            self.logger.error(f"Failed to read file info: {e}")
            self.file_format_label.setText("Error")
            self.file_format_label.setStyleSheet("color: #f44336;")
            self.file_channels_label.setText("Error reading file")
            self.file_channels_label.setStyleSheet("color: #f44336;")

    def _parse_sample_rate(self, value):
        """Parse sample rate value that may include units."""
        if value is None:
            return None

        if isinstance(value, (int, float)):
            return float(value) if value > 0 else None

        if not isinstance(value, str):
            return None

        match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", value)
        if match:
            parsed = float(match.group(0))
            return parsed if parsed > 0 else None

        return None

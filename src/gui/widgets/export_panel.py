"""
Export Panel widget for data export controls.

This widget provides controls for exporting acquired data
to various file formats.
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QLabel, QComboBox, QPushButton, QProgressBar, QFileDialog,
    QSpinBox, QCheckBox, QLineEdit, QRadioButton, QButtonGroup,
    QScrollArea
)
from PyQt5.QtCore import pyqtSignal, Qt
from pathlib import Path
from typing import Optional
import numpy as np

from ...utils.logger import get_logger
from ...utils.constants import ExportDefaults
from ...export.export_manager import ExportManager


class ExportPanel(QWidget):
    """
    Export panel widget.

    Provides controls for:
    - File format selection
    - File path selection
    - Export options
    - Progress display
    """

    # Signals
    export_requested = pyqtSignal(str, str)  # filepath, format

    def __init__(self, parent=None):
        """
        Initialize the export panel.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)

        self.logger = get_logger(__name__)

        # Data storage
        self.data: Optional[np.ndarray] = None
        self.sample_rate: float = 25600.0
        self.channel_names: list = []
        self.channel_units: list = []

        # Export manager
        self.export_manager = ExportManager()

        # Connect export manager signals
        self.export_manager.export_started.connect(self._on_export_started)
        self.export_manager.export_progress.connect(self._on_export_progress)
        self.export_manager.export_finished.connect(self._on_export_finished)
        self.export_manager.export_error.connect(self._on_export_error)

        self._init_ui()

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

        # File selection group
        file_group = self._create_file_group()
        layout.addWidget(file_group)

        # Format selection group
        format_group = self._create_format_group()
        layout.addWidget(format_group)

        # Options group
        options_group = self._create_options_group()
        layout.addWidget(options_group)

        # Progress group
        progress_group = self._create_progress_group()
        layout.addWidget(progress_group)

        # Export button
        export_layout = QHBoxLayout()
        self.export_button = QPushButton("Export Data")
        self.export_button.setMinimumHeight(40)
        self.export_button.setStyleSheet("""
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
        self.export_button.clicked.connect(self._on_export_clicked)
        self.export_button.setEnabled(False)  # Disabled until data available
        export_layout.addWidget(self.export_button)
        layout.addLayout(export_layout)

        # Add stretch to push everything to top
        layout.addStretch()

        # Set content widget to scroll area
        scroll_area.setWidget(content_widget)

        # Add scroll area to main layout
        main_layout.addWidget(scroll_area)

    def _create_file_group(self) -> QGroupBox:
        """Create file selection group."""
        group = QGroupBox("Output File")
        layout = QFormLayout()

        # File path
        path_layout = QHBoxLayout()
        self.filepath_edit = QLineEdit()
        self.filepath_edit.setPlaceholderText("No file selected...")
        self.filepath_edit.setReadOnly(True)
        path_layout.addWidget(self.filepath_edit)

        self.browse_button = QPushButton("Browse...")
        self.browse_button.clicked.connect(self._on_browse_clicked)
        path_layout.addWidget(self.browse_button)

        layout.addRow("Path:", path_layout)

        # Filename
        self.filename_edit = QLineEdit()
        self.filename_edit.setPlaceholderText("auto-generated")
        layout.addRow("Name:", self.filename_edit)

        group.setLayout(layout)
        return group

    def _create_format_group(self) -> QGroupBox:
        """Create format selection group."""
        group = QGroupBox("Format")
        layout = QVBoxLayout()

        # Format radio buttons
        format_layout = QVBoxLayout()
        self.format_group = QButtonGroup()

        # CSV
        self.csv_radio = QRadioButton("CSV")
        self.csv_radio.setToolTip("Comma-separated values, universal format")
        format_layout.addWidget(self.csv_radio)
        self.format_group.addButton(self.csv_radio, 0)

        # HDF5
        self.hdf5_radio = QRadioButton("HDF5")
        self.hdf5_radio.setToolTip("Hierarchical Data Format, efficient for large datasets")
        self.hdf5_radio.setChecked(True)  # Default
        format_layout.addWidget(self.hdf5_radio)
        self.format_group.addButton(self.hdf5_radio, 1)

        # TDMS
        self.tdms_radio = QRadioButton("TDMS")
        self.tdms_radio.setToolTip("NI's native format, compatible with LabVIEW/DIAdem")
        format_layout.addWidget(self.tdms_radio)
        self.format_group.addButton(self.tdms_radio, 2)

        layout.addLayout(format_layout)

        # Format description
        self.format_description = QLabel()
        self.format_description.setWordWrap(True)
        self.format_description.setStyleSheet("QLabel { color: #666666; font-size: 10px; }")
        layout.addWidget(self.format_description)

        self._update_format_description()

        group.setLayout(layout)
        return group

    def _create_options_group(self) -> QGroupBox:
        """Create export options group."""
        group = QGroupBox("Options")
        layout = QFormLayout()

        # Time range selection
        self.time_range_combo = QComboBox()
        self.time_range_combo.addItem("All Available Data", "all")
        self.time_range_combo.addItem("Last 10 seconds", 10)
        self.time_range_combo.addItem("Last 30 seconds", 30)
        self.time_range_combo.addItem("Last 60 seconds", 60)
        self.time_range_combo.addItem("Custom (seconds)", "custom")
        layout.addRow("Time Range:", self.time_range_combo)

        # Custom time (only shown when custom selected)
        self.custom_time_spin = QSpinBox()
        self.custom_time_spin.setRange(1, 3600)
        self.custom_time_spin.setValue(60)
        self.custom_time_spin.setSuffix(" s")
        self.custom_time_spin.setEnabled(False)
        layout.addRow("Custom:", self.custom_time_spin)

        # Include metadata
        self.metadata_checkbox = QCheckBox("Include metadata")
        self.metadata_checkbox.setChecked(True)
        layout.addRow("", self.metadata_checkbox)

        # Compression (for HDF5)
        self.compression_checkbox = QCheckBox("Enable compression")
        self.compression_checkbox.setChecked(True)
        layout.addRow("", self.compression_checkbox)

        group.setLayout(layout)
        return group

    def _create_progress_group(self) -> QGroupBox:
        """Create progress display group."""
        group = QGroupBox("Export Progress")
        layout = QVBoxLayout()

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        # Status label
        self.status_label = QLabel("Ready to export")
        self.status_label.setWordWrap(True)
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)

        group.setLayout(layout)
        return group

    def _update_format_description(self):
        """Update format description label."""
        descriptions = {
            0: "CSV: Universal text format, readable by any application",
            1: "HDF5: Efficient binary format with hierarchical structure",
            2: "TDMS: NI's native format, compatible with LabVIEW and DIAdem"
        }

        format_id = self.format_group.checkedId()
        self.format_description.setText(descriptions.get(format_id, ""))

    def set_data(
        self,
        data: np.ndarray,
        sample_rate: float,
        channel_names: list,
        channel_units: list
    ):
        """
        Set data to be exported.

        Args:
            data: Data array of shape (n_channels, n_samples)
            sample_rate: Sampling rate in Hz
            channel_names: List of channel names
            channel_units: List of channel units
        """
        self.data = data
        self.sample_rate = sample_rate
        self.channel_names = channel_names
        self.channel_units = channel_units

        # Enable export button
        self.export_button.setEnabled(True)

        # Update status
        n_samples = data.shape[1]
        duration = n_samples / sample_rate
        self.status_label.setText(f"Ready: {n_samples:,} samples ({duration:.1f} s)")

        self.logger.debug(f"Export data set: {data.shape}, {sample_rate} Hz")

    def _get_selected_format(self) -> str:
        """Get selected export format."""
        format_id = self.format_group.checkedId()

        if format_id == 0:
            return ExportDefaults.FORMAT_CSV
        elif format_id == 1:
            return ExportDefaults.FORMAT_HDF5
        else:
            return ExportDefaults.FORMAT_TDMS

    def _get_output_path(self) -> Optional[str]:
        """Get complete output file path."""
        # Get format
        format = self._get_selected_format()
        extension = ExportDefaults.FILE_EXTENSIONS.get(format, '.dat')

        # Get filename
        filename = self.filename_edit.text().strip()
        if not filename:
            # Auto-generate filename
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ni_daq_export_{timestamp}{extension}"

        # Ensure extension matches format
        if not filename.endswith(extension):
            filename = filename + Path(filename).suffix + extension

        # Get directory
        filepath = self.filepath_edit.text().strip()
        if not filepath:
            return None

        # Combine
        import os
        return os.path.join(filepath, filename)

    def _on_browse_clicked(self):
        """Handle browse button click."""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Export Directory",
            ""
        )

        if directory:
            self.filepath_edit.setText(directory)

    def _on_export_clicked(self):
        """Handle export button click."""
        # Validate
        if self.data is None:
            self.status_label.setText("No data available")
            return

        filepath = self._get_output_path()
        if not filepath:
            self.status_label.setText("Please select output directory")
            return

        format = self._get_selected_format()

        # Reset progress
        self.progress_bar.setValue(0)
        self.export_button.setEnabled(False)

        # Start export
        try:
            self.export_manager.export(
                filepath=filepath,
                format=format,
                data=self.data,
                sample_rate=self.sample_rate,
                channel_names=self.channel_names,
                channel_units=self.channel_units,
                asynchronous=True
            )
        except Exception as e:
            self.status_label.setText(f"Export failed: {e}")
            self.export_button.setEnabled(True)

    def _on_export_started(self, filepath: str):
        """Handle export started signal."""
        self.status_label.setText(f"Exporting to {Path(filepath).name}...")
        self.logger.info(f"Export started: {filepath}")

    def _on_export_progress(self, current: int, total: int):
        """Handle export progress signal."""
        if total > 0:
            progress = int((current / total) * 100)
            self.progress_bar.setValue(progress)
            self.status_label.setText(f"Exporting: {current:,} / {total:,} samples")

    def _on_export_finished(self, success: bool, message: str):
        """Handle export finished signal."""
        self.progress_bar.setValue(100 if success else 0)
        self.status_label.setText(message)
        self.export_button.setEnabled(True)

        if success:
            self.logger.info(f"Export finished: {message}")
        else:
            self.logger.error(f"Export failed: {message}")

    def _on_export_error(self, error: str):
        """Handle export error signal."""
        self.status_label.setText(f"Error: {error}")
        self.export_button.setEnabled(True)
        self.logger.error(f"Export error: {error}")


# Example usage
if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication
    import numpy as np

    app = QApplication(sys.argv)

    # Create panel
    panel = ExportPanel()

    # Set some test data
    sample_rate = 25600
    n_samples = 50000
    n_channels = 4

    data = np.random.randn(n_channels, n_samples) * 0.1
    channel_names = [f"Channel {i+1}" for i in range(n_channels)]
    channel_units = ["g"] * n_channels

    panel.set_data(data, sample_rate, channel_names, channel_units)

    panel.show()

    sys.exit(app.exec_())

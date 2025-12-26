"""
Data Analysis Panel.

Provides controls for analyzing saved data using the existing
Time Domain and Frequency Domain tabs.
"""

import re
from pathlib import Path

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QPushButton, QMessageBox, QFileDialog
)
from PyQt5.QtCore import QTimer

from ...export.data_file_reader import DataFileReader
from ...utils.logger import get_logger


class DataAnalysisPanel(QWidget):
    """Left-side panel for saved-data analysis."""

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

    def _init_ui(self):
        """Initialize UI layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        file_layout = QHBoxLayout()
        file_layout.addWidget(QLabel("Data File:"))

        self.file_status_label = QLabel("No data file")
        self.file_status_label.setStyleSheet("color: #666; font-style: italic;")
        file_layout.addWidget(self.file_status_label)

        file_layout.addStretch()

        self.browse_button = QPushButton("Browse...")
        self.browse_button.clicked.connect(self._on_browse_file)
        file_layout.addWidget(self.browse_button)

        layout.addLayout(file_layout)

        controls_layout = QHBoxLayout()
        controls_layout.addWidget(QLabel("Time Duration:"))

        self.duration_combo = QComboBox()
        for seconds in [1, 2, 5, 10, 20, 50, 100, 200]:
            self.duration_combo.addItem(f"{seconds}s", seconds)
        self.duration_combo.setCurrentText("10s")
        self.duration_combo.setToolTip("Duration of data to load from saved file")
        controls_layout.addWidget(self.duration_combo)

        controls_layout.addSpacing(20)

        self.load_button = QPushButton("Load Time Data")
        self.load_button.setEnabled(False)
        self.load_button.clicked.connect(self._on_load_time_domain)
        self.load_button.setToolTip("Load time-domain data into the Time Domain tab")
        controls_layout.addWidget(self.load_button)

        controls_layout.addStretch()
        layout.addLayout(controls_layout)

        hint_label = QLabel(
            "Use the Frequency Domain tab's Analyze Saved Data button for FFT analysis "
            "(filter settings apply to saved data too)."
        )
        hint_label.setStyleSheet("color: #666; font-style: italic;")
        layout.addWidget(hint_label)
        layout.addStretch()

    def set_data_file(self, filepath: str):
        """
        Set the current data file for analysis.

        Args:
            filepath: Path to the data file created during acquisition
        """
        self.current_data_file = Path(filepath)
        self.load_button.setEnabled(True)

        try:
            reader = DataFileReader()
            self.data_file_info = reader.get_file_info(filepath)

            duration = self.data_file_info.get('duration_seconds', 0)
            if duration > 0:
                self.file_status_label.setText(
                    f"{self.current_data_file.name} ({duration:.1f}s)"
                )
                self.file_status_label.setStyleSheet("color: #4CAF50; font-style: italic;")
            else:
                self.file_status_label.setText(self.current_data_file.name)
                self.file_status_label.setStyleSheet("color: #2196F3; font-style: italic;")

            self.sample_rate = self._parse_sample_rate(self.data_file_info.get('sample_rate'))

            if self.fft_plot_widget:
                self.fft_plot_widget.set_data_file(filepath)

            self.logger.info(f"Saved analysis file set: {filepath}")

        except Exception as e:
            self.logger.error(f"Failed to read file info: {e}")
            self.file_status_label.setText("File: Error reading info")
            self.file_status_label.setStyleSheet("color: #f44336; font-style: italic;")

    def _on_browse_file(self):
        """Handle browse for saved data file."""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Open Saved Data",
            "",
            "Data Files (*.h5 *.hdf5 *.tdms *.csv);;All Files (*)"
        )

        if filename:
            self.set_data_file(filename)

    def _on_load_time_domain(self):
        """Load saved data into the Time Domain tab."""
        if not self.current_data_file or not self.current_data_file.exists():
            QMessageBox.warning(self, "No Data", "No data file available for analysis.")
            return

        duration = float(self.duration_combo.currentData())

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
            QMessageBox.critical(self, "Analysis Error", f"Failed to load data:\n{str(e)}")

    def _parse_sample_rate(self, value):
        """Parse sample rate value that may include units."""
        if value is None:
            return self.sample_rate

        if isinstance(value, (int, float)):
            return float(value)

        if not isinstance(value, str):
            return self.sample_rate

        match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", value)
        if match:
            return float(match.group(0))

        return self.sample_rate

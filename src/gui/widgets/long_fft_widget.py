"""
Long-Window FFT Analysis Widget.

This widget provides GUI interface for long-window FFT analysis
with window sizes from 10s to 200s for low-frequency analysis.
"""

import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QComboBox,
    QCheckBox, QPushButton, QLabel, QTableWidget, QTableWidgetItem,
    QProgressBar, QMessageBox, QSpinBox, QDoubleSpinBox, QSplitter
)
from PyQt5.QtCore import pyqtSlot, pyqtSignal, Qt, QThread, QObject
from PyQt5.QtGui import QFont
from typing import Optional, List, Dict
from pathlib import Path

try:
    import pyqtgraph as pg
    PYQTGRAPH_AVAILABLE = True
except ImportError:
    PYQTGRAPH_AVAILABLE = False

from ...processing.long_data_saver import LongDataSaver
from ...processing.long_window_fft import LongWindowFFTProcessor
from ...utils.logger import get_logger


class LongFFTWorker(QObject):
    """Worker thread for long-window FFT computation."""

    finished = pyqtSignal()
    progress = pyqtSignal(int)
    result_ready = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.data = None
        self.sample_rate = None
        self.window_duration = None
        self.max_frequency = None
        self.n_peaks = 10

    def set_params(self, data, sample_rate, window_duration, max_frequency, n_peaks):
        """Set computation parameters."""
        self.data = data
        self.sample_rate = sample_rate
        self.window_duration = window_duration
        self.max_frequency = max_frequency
        self.n_peaks = n_peaks

    def run(self):
        """Run FFT analysis in background thread."""
        try:
            self.progress.emit(10)

            # Create processor
            processor = LongWindowFFTProcessor(
                sample_rate=self.sample_rate,
                window_function='hann'
            )

            self.progress.emit(30)

            # Compute magnitude spectrum
            frequencies, magnitude = processor.compute_magnitude(
                self.data,
                window_duration=self.window_duration,
                scale='linear'
            )

            self.progress.emit(60)

            # Find peaks
            peaks = processor.find_peaks(
                frequencies,
                magnitude,
                threshold=0.05,
                n_peaks=self.n_peaks
            )

            self.progress.emit(80)

            # Analyze low frequencies
            low_freq_analysis = processor.analyze_low_frequencies(
                self.data,
                max_frequency=self.max_frequency,
                window_duration=self.window_duration,
                n_peaks=self.n_peaks
            )

            self.progress.emit(100)

            # Prepare results
            results = {
                'frequencies': frequencies,
                'magnitude': magnitude,
                'peaks': peaks,
                'low_freq_analysis': low_freq_analysis,
                'window_duration': self.window_duration,
                'freq_resolution': processor.get_frequency_resolution(self.window_duration)
            }

            self.result_ready.emit(results)

        except Exception as e:
            self.error.emit(str(e))

        finally:
            self.finished.emit()


class LongFFTAnalysisWidget(QWidget):
    """
    Widget for long-window FFT analysis.

    Features:
    - Save buffer data to temporary file
    - Analyze with windows: 10s, 20s, 50s, 100s, 200s
    - High-resolution frequency analysis for low frequencies
    - Peak detection and display
    - Export results
    """

    # Signals
    save_buffer_requested = pyqtSignal(float)  # duration in seconds

    def __init__(self, parent=None):
        """Initialize the widget."""
        super().__init__(parent)

        self.logger = get_logger(__name__)

        # Data
        self.buffer = None  # Reference to data buffer
        self.sample_rate = 25600.0  # Default sample rate
        self.n_channels = 0
        self.channel_names: List[str] = []

        # Saved data
        self.saved_data: Optional[np.ndarray] = None
        self.saved_metadata: Optional[Dict] = None
        self.current_file: Optional[Path] = None

        # Analysis results
        self.analysis_results: Optional[Dict] = None

        # Processors
        self.saver: Optional[LongDataSaver] = None
        self.processor: Optional[LongWindowFFTProcessor] = None

        # Worker thread
        self.worker_thread: Optional[QThread] = None
        self.worker: Optional[LongFFTWorker] = None

        # Plot items
        self.plot_curve = None
        self.peak_markers = None

        self._init_ui()

    def _init_ui(self):
        """Initialize user interface."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)

        # Title
        title = QLabel("Long-Window FFT Analysis (Low-Frequency Analysis)")
        title_font = QFont()
        title_font.setPointSize(12)
        title_font.setBold(True)
        title.setFont(title_font)
        main_layout.addWidget(title)

        # Create splitter for controls and plot
        splitter = QSplitter(Qt.Vertical)

        # === Top Section: Controls ===
        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)

        # Control panel 1: Buffer management
        buffer_group = QGroupBox("Step 1: Save Buffer Data")
        buffer_layout = QHBoxLayout()

        buffer_layout.addWidget(QLabel("Duration:"))
        self.duration_combo = QComboBox()
        self.duration_combo.addItems(["50s", "100s", "200s"])
        self.duration_combo.setCurrentText("200s")
        buffer_layout.addWidget(self.duration_combo)

        buffer_layout.addSpacing(20)

        self.save_button = QPushButton("Save Current Buffer")
        self.save_button.clicked.connect(self._on_save_buffer)
        buffer_layout.addWidget(self.save_button)

        self.buffer_status_label = QLabel("Status: No data saved")
        buffer_layout.addWidget(self.buffer_status_label)

        buffer_layout.addStretch()
        buffer_group.setLayout(buffer_layout)
        controls_layout.addWidget(buffer_group)

        # Control panel 2: Analysis settings
        analysis_group = QGroupBox("Step 2: Configure Analysis")
        analysis_layout = QHBoxLayout()

        analysis_layout.addWidget(QLabel("Channel:"))
        self.channel_combo = QComboBox()
        analysis_layout.addWidget(self.channel_combo)

        analysis_layout.addSpacing(20)

        analysis_layout.addWidget(QLabel("Window:"))
        self.window_combo = QComboBox()
        self.window_combo.addItems(["10s", "20s", "50s", "100s", "200s"])
        self.window_combo.setCurrentText("200s")
        self.window_combo.currentTextChanged.connect(self._on_window_changed)
        analysis_layout.addWidget(self.window_combo)

        analysis_layout.addSpacing(20)

        analysis_layout.addWidget(QLabel("Max Freq (Hz):"))
        self.max_freq_spin = QDoubleSpinBox()
        self.max_freq_spin.setRange(0.1, 1000.0)
        self.max_freq_spin.setValue(10.0)
        self.max_freq_spin.setDecimals(1)
        analysis_layout.addWidget(self.max_freq_spin)

        analysis_layout.addSpacing(20)

        self.analyze_button = QPushButton("Analyze")
        self.analyze_button.clicked.connect(self._on_analyze)
        self.analyze_button.setEnabled(False)
        analysis_layout.addWidget(self.analyze_button)

        analysis_layout.addStretch()
        analysis_group.setLayout(analysis_layout)
        controls_layout.addWidget(analysis_group)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        controls_layout.addWidget(self.progress_bar)

        # Info panel
        info_group = QGroupBox("Analysis Information")
        info_layout = QVBoxLayout()

        self.info_label = QLabel("No analysis performed yet")
        self.info_label.setWordWrap(True)
        info_layout.addWidget(self.info_label)

        info_group.setLayout(info_layout)
        controls_layout.addWidget(info_group)

        splitter.addWidget(controls_widget)

        # === Bottom Section: Results ===
        results_widget = QWidget()
        results_layout = QHBoxLayout(results_widget)

        # Left: Plot
        plot_group = QGroupBox("Frequency Spectrum")
        plot_layout = QVBoxLayout()

        if PYQTGRAPH_AVAILABLE:
            self.plot_widget = pg.PlotWidget()
            self.plot_widget.setBackground('w')
            self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
            self.plot_widget.setLabel('left', 'Magnitude')
            self.plot_widget.setLabel('bottom', 'Frequency', units='Hz')
            self.plot_widget.addLegend()
            self.plot_widget.setAntialiasing(True)
            plot_layout.addWidget(self.plot_widget)
        else:
            placeholder = QLabel("PyQtGraph not available")
            placeholder.setAlignment(Qt.AlignCenter)
            plot_layout.addWidget(placeholder)

        plot_group.setLayout(plot_layout)
        results_layout.addWidget(plot_group, stretch=2)

        # Right: Peaks table
        table_group = QGroupBox("Detected Peaks")
        table_layout = QVBoxLayout()

        self.peaks_table = QTableWidget()
        self.peaks_table.setColumnCount(3)
        self.peaks_table.setHorizontalHeaderLabels(['Rank', 'Frequency (Hz)', 'Magnitude'])
        self.peaks_table.horizontalHeader().setStretchLastSection(True)
        table_layout.addWidget(self.peaks_table)

        # Export button
        self.export_button = QPushButton("Export Results")
        self.export_button.clicked.connect(self._on_export)
        self.export_button.setEnabled(False)
        table_layout.addWidget(self.export_button)

        table_group.setLayout(table_layout)
        results_layout.addWidget(table_group, stretch=1)

        splitter.addWidget(results_widget)

        # Set splitter sizes
        splitter.setSizes([300, 400])

        main_layout.addWidget(splitter)

    def set_buffer(self, buffer):
        """Set reference to data buffer."""
        self.buffer = buffer
        self.logger.info("Buffer reference set")

    def set_sample_rate(self, sample_rate: float):
        """Set sample rate."""
        self.sample_rate = sample_rate

        # Reinitialize processors
        self.saver = LongDataSaver(
            sample_rate=sample_rate,
            max_duration_seconds=200.0
        )

        self.processor = LongWindowFFTProcessor(
            sample_rate=sample_rate,
            window_function='hann'
        )

        self.logger.info(f"Sample rate set to {sample_rate} Hz")

    def set_channels(self, n_channels: int, channel_names: List[str]):
        """Set channel information."""
        self.n_channels = n_channels
        self.channel_names = channel_names

        # Update channel combo
        self.channel_combo.clear()
        for i, name in enumerate(channel_names):
            self.channel_combo.addItem(f"Ch{i}: {name}", i)

        self.logger.info(f"Channels configured: {n_channels} channels")

    @pyqtSlot()
    def _on_save_buffer(self):
        """Handle save buffer button click."""
        if self.buffer is None:
            QMessageBox.warning(
                self,
                "No Buffer",
                "No data buffer available. Please start acquisition first."
            )
            return

        if self.saver is None:
            QMessageBox.warning(
                self,
                "Not Initialized",
                "Sample rate not set. Please initialize properly."
            )
            return

        try:
            # Get duration
            duration_str = self.duration_combo.currentText()
            duration = float(duration_str.replace('s', ''))

            self.save_button.setEnabled(False)
            self.buffer_status_label.setText("Status: Saving...")

            # Save buffer
            file_path = self.saver.save_from_buffer(
                self.buffer,
                duration_seconds=duration,
                format='hdf5'
            )

            # Load it back
            self.saved_data, self.saved_metadata = self.saver.load_temp_file(file_path)
            self.current_file = file_path

            # Update UI
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            status_text = (
                f"Status: Saved {self.saved_data.shape[1]} samples "
                f"({self.saved_data.shape[1]/self.sample_rate:.1f}s, "
                f"{file_size_mb:.1f} MB)"
            )
            self.buffer_status_label.setText(status_text)

            # Enable analysis
            self.analyze_button.setEnabled(True)

            self.logger.info(f"Buffer saved: {file_path}")

        except Exception as e:
            self.logger.error(f"Failed to save buffer: {e}")
            QMessageBox.critical(
                self,
                "Save Error",
                f"Failed to save buffer:\n{e}"
            )

        finally:
            self.save_button.setEnabled(True)

    @pyqtSlot()
    def _on_analyze(self):
        """Handle analyze button click."""
        if self.saved_data is None:
            QMessageBox.warning(
                self,
                "No Data",
                "No saved data available. Please save buffer first."
            )
            return

        try:
            # Get parameters
            channel_idx = self.channel_combo.currentData()
            window_duration = self.window_combo.currentText()
            max_frequency = self.max_freq_spin.value()

            # Get channel data
            channel_data = self.saved_data[channel_idx, :]

            # Check if data is long enough
            required_samples = self.processor.window_sizes[window_duration]
            if len(channel_data) < required_samples:
                QMessageBox.warning(
                    self,
                    "Insufficient Data",
                    f"Data length ({len(channel_data)} samples) is less than "
                    f"required for {window_duration} window ({required_samples} samples).\n\n"
                    f"Please save more data or use shorter window."
                )
                return

            # Disable buttons
            self.analyze_button.setEnabled(False)
            self.save_button.setEnabled(False)

            # Show progress
            self.progress_bar.setValue(0)
            self.progress_bar.setVisible(True)

            # Create worker thread
            self.worker_thread = QThread()
            self.worker = LongFFTWorker()
            self.worker.moveToThread(self.worker_thread)

            # Set parameters
            self.worker.set_params(
                channel_data,
                self.sample_rate,
                window_duration,
                max_frequency,
                n_peaks=20
            )

            # Connect signals
            self.worker_thread.started.connect(self.worker.run)
            self.worker.finished.connect(self.worker_thread.quit)
            self.worker.finished.connect(self._on_analysis_finished)
            self.worker.progress.connect(self.progress_bar.setValue)
            self.worker.result_ready.connect(self._on_results_ready)
            self.worker.error.connect(self._on_analysis_error)

            # Start thread
            self.worker_thread.start()

            self.logger.info(f"Started analysis: {window_duration}, channel {channel_idx}")

        except Exception as e:
            self.logger.error(f"Failed to start analysis: {e}")
            QMessageBox.critical(
                self,
                "Analysis Error",
                f"Failed to start analysis:\n{e}"
            )
            self.analyze_button.setEnabled(True)
            self.save_button.setEnabled(True)

    @pyqtSlot(dict)
    def _on_results_ready(self, results):
        """Handle analysis results."""
        self.analysis_results = results

        # Update plot
        self._update_plot(results)

        # Update peaks table
        self._update_peaks_table(results['low_freq_analysis']['peaks'])

        # Update info
        info_text = (
            f"Window Duration: {results['window_duration']}\n"
            f"Frequency Resolution: {results['freq_resolution']:.6f} Hz\n"
            f"Total Power (0-{self.max_freq_spin.value()} Hz): "
            f"{results['low_freq_analysis']['total_power']:.4f}\n"
            f"RMS Value: {results['low_freq_analysis']['rms_value']:.4f}\n"
            f"Peaks Detected: {len(results['low_freq_analysis']['peaks'])}"
        )
        self.info_label.setText(info_text)

        # Enable export
        self.export_button.setEnabled(True)

        self.logger.info("Analysis results displayed")

    @pyqtSlot()
    def _on_analysis_finished(self):
        """Handle analysis completion."""
        self.progress_bar.setVisible(False)
        self.analyze_button.setEnabled(True)
        self.save_button.setEnabled(True)

    @pyqtSlot(str)
    def _on_analysis_error(self, error_msg):
        """Handle analysis error."""
        self.logger.error(f"Analysis error: {error_msg}")
        QMessageBox.critical(
            self,
            "Analysis Error",
            f"Analysis failed:\n{error_msg}"
        )

    def _update_plot(self, results):
        """Update the plot with results."""
        if not PYQTGRAPH_AVAILABLE:
            return

        # Clear previous plot
        self.plot_widget.clear()

        # Get data limited to max frequency
        max_freq = self.max_freq_spin.value()
        frequencies = results['low_freq_analysis']['frequencies']
        magnitude = results['low_freq_analysis']['magnitude']

        # Plot spectrum
        self.plot_curve = self.plot_widget.plot(
            frequencies,
            magnitude,
            pen=pg.mkPen(color='b', width=2),
            name='Magnitude'
        )

        # Plot peaks
        peaks = results['low_freq_analysis']['peaks']
        if len(peaks) > 0:
            peak_freqs = [p['frequency'] for p in peaks]
            peak_mags = [p['magnitude'] for p in peaks]

            self.peak_markers = self.plot_widget.plot(
                peak_freqs,
                peak_mags,
                pen=None,
                symbol='o',
                symbolBrush='r',
                symbolSize=8,
                name='Peaks'
            )

    def _update_peaks_table(self, peaks):
        """Update the peaks table."""
        self.peaks_table.setRowCount(len(peaks))

        for i, peak in enumerate(peaks):
            # Rank
            rank_item = QTableWidgetItem(str(i + 1))
            rank_item.setTextAlignment(Qt.AlignCenter)
            self.peaks_table.setItem(i, 0, rank_item)

            # Frequency
            freq_item = QTableWidgetItem(f"{peak['frequency']:.6f}")
            freq_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.peaks_table.setItem(i, 1, freq_item)

            # Magnitude
            mag_item = QTableWidgetItem(f"{peak['magnitude']:.6f}")
            mag_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.peaks_table.setItem(i, 2, mag_item)

        self.peaks_table.resizeColumnsToContents()

    @pyqtSlot()
    def _on_window_changed(self):
        """Handle window size change."""
        if self.processor:
            window_duration = self.window_combo.currentText()
            freq_res = self.processor.get_frequency_resolution(window_duration)
            self.logger.debug(f"Window changed to {window_duration}, freq_res={freq_res:.6f} Hz")

    @pyqtSlot()
    def _on_export(self):
        """Handle export button click."""
        if self.analysis_results is None:
            return

        try:
            # Export to CSV
            import csv
            from datetime import datetime

            filename = f"long_fft_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)

                # Header
                writer.writerow(['Long-Window FFT Analysis Results'])
                writer.writerow([f"Window Duration: {self.analysis_results['window_duration']}"])
                writer.writerow([f"Frequency Resolution: {self.analysis_results['freq_resolution']:.6f} Hz"])
                writer.writerow([])

                # Peaks
                writer.writerow(['Detected Peaks'])
                writer.writerow(['Rank', 'Frequency (Hz)', 'Magnitude'])

                peaks = self.analysis_results['low_freq_analysis']['peaks']
                for i, peak in enumerate(peaks, 1):
                    writer.writerow([i, f"{peak['frequency']:.6f}", f"{peak['magnitude']:.6f}"])

            QMessageBox.information(
                self,
                "Export Successful",
                f"Results exported to:\n{filename}"
            )

            self.logger.info(f"Results exported to {filename}")

        except Exception as e:
            self.logger.error(f"Export failed: {e}")
            QMessageBox.critical(
                self,
                "Export Error",
                f"Failed to export results:\n{e}"
            )


# Test widget standalone
if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)

    widget = LongFFTAnalysisWidget()
    widget.set_sample_rate(25600.0)
    widget.set_channels(4, ["X-axis", "Y-axis", "Z-axis", "Reference"])
    widget.resize(1200, 800)
    widget.show()

    sys.exit(app.exec_())

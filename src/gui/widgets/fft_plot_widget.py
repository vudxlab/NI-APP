"""
FFT Plot Widget for frequency-domain visualization.

This widget provides real-time FFT/spectrum plotting using Plotly.
"""

import numpy as np
from pathlib import Path
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QComboBox,
    QCheckBox, QPushButton, QLabel, QSpinBox, QDoubleSpinBox, QMenu, QAction,
    QMessageBox, QDialog, QDialogButtonBox, QScrollArea
)
from PyQt5.QtCore import QTimer, pyqtSlot, pyqtSignal, Qt, QThread, QObject
from typing import List, Dict, Optional, Tuple, Union

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from .plotly_view import PlotlyView
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: Plotly or QtWebEngine not available. Install plotly and PyQtWebEngine.")

from ...utils.logger import get_logger
from ...utils.constants import GUIDefaults, ProcessingDefaults
from ...processing.long_window_fft import LongWindowFFTProcessor
from ...processing.filters import apply_zero_phase_filter
from ...export.data_file_reader import DataFileReader


class FFTPlotWidget(QWidget):
    """
    Real-time FFT/frequency-domain plot widget.

    Features:
    - Multi-channel spectrum display
    - Linear or logarithmic magnitude scale (dB)
    - Configurable frequency range
    - Peak detection and annotation
    - Channel visibility control
    - Grid and legend
    """
    
    # Signals
    fft_size_changed = pyqtSignal(int)  # Emits new FFT size
    filter_config_updated = pyqtSignal(dict)
    filter_enabled_updated = pyqtSignal(bool)

    def __init__(self, parent=None):
        """
        Initialize the FFT plot widget.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)

        self.logger = get_logger(__name__)

        # Data storage
        self.n_channels = 0
        self.channel_names: List[str] = []
        self.channel_units: str = "g"
        self.sample_rate: float = 25600.0  # Default sample rate

        # FFT data
        self.frequencies: Optional[np.ndarray] = None
        self.magnitudes: Optional[np.ndarray] = None
        self.peaks: List[Dict] = []

        # Plot view (Plotly) - support both overlay and stack modes
        self.plot_view_single: Optional['PlotlyView'] = None  # For overlay mode
        self.plot_views_stack: List['PlotlyView'] = []  # For stack mode (1 per channel)
        self.plot_container_stack: Optional[QWidget] = None  # Container for stack plots
        self.scroll_area: Optional['QScrollArea'] = None
        self.channel_visible: List[bool] = []

        # Display settings
        self.magnitude_scale = "dB"  # "linear" or "dB"
        self.auto_scale = True
        self.show_peaks = True
        self.peak_threshold = ProcessingDefaults.PEAK_DETECTION_THRESHOLD
        self.frequency_limit = "Full"  # "Full" or specific Hz value
        self.display_mode = "overlay"  # "overlay" or "stack"
        self.stack_offset = 20.0  # dB offset between stacked channels

        # Stack plot settings
        self.max_visible_channels = 4  # Max channels to show without scrolling
        self.subplot_height = 200  # Fixed height per individual plot in pixels
        self.min_subplot_height = 160  # Minimum height for stack plots
        self._last_layout: Optional[Dict] = None  # Cache last layout (deprecated for stack mode)
        self._current_plot_height: Optional[int] = None  # Track current height (deprecated for stack mode)
        self._layout_initialized: bool = False  # Track if layout is set (deprecated for stack mode)

        # File-based FFT settings
        self.current_data_file: Optional[Path] = None
        self.data_file_info: Optional[dict] = None
        self.long_fft_processor: Optional[LongWindowFFTProcessor] = None
        self.long_window_duration = "200s"  # Default long window
        self.filter_config: Dict = {}
        self.filter_enabled = False

        if not PLOTLY_AVAILABLE:
            self.logger.error("Plotly or QtWebEngine not available")

        self._init_ui()

    def _init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Controls
        controls_layout = self._create_controls()
        layout.addLayout(controls_layout)

        if PLOTLY_AVAILABLE:
            # Create scroll area for plot view
            self.scroll_area = QScrollArea()
            self.scroll_area.setWidgetResizable(True)
            self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

            # Initialize with overlay mode (single plot view)
            self.plot_view_single = PlotlyView()
            self.scroll_area.setWidget(self.plot_view_single)
            layout.addWidget(self.scroll_area)

            # Initial empty plot will be set after widget is shown
            # to ensure proper rendering. Increase delay to 500ms for Plotly.js to load
            print("FFTPlotWidget: Scheduling _init_empty_plot in 500ms")
            QTimer.singleShot(500, self._init_empty_plot)
        else:
            # Placeholder if PyQtGraph not available
            placeholder = QLabel("Plotly/QtWebEngine not installed.\nInstall with: pip install plotly PyQtWebEngine")
            placeholder.setAlignment(Qt.AlignCenter)
            placeholder.setStyleSheet("QLabel { color: red; font-size: 14px; }")
            layout.addWidget(placeholder)

    def _create_controls(self) -> QVBoxLayout:
        """Create control panel."""
        controls_main_layout = QVBoxLayout()

        # === Row 1: FFT Analysis Controls ===
        analysis_row = QHBoxLayout()

        analysis_row.addWidget(QLabel("FFT Duration:"))

        self.fft_duration_combo = QComboBox()
        self.fft_duration_combo.addItems(["10s", "20s", "50s", "100s", "200s"])
        self.fft_duration_combo.setCurrentText("200s")
        self.fft_duration_combo.setToolTip("Duration of data to use for FFT analysis")
        analysis_row.addWidget(self.fft_duration_combo)

        analysis_row.addSpacing(20)

        self.filtfilt_button = QPushButton("Filtfilt")
        self.filtfilt_button.clicked.connect(self._open_filtfilt_dialog)
        self.filtfilt_button.setToolTip("Configure filter settings for analysis")
        analysis_row.addWidget(self.filtfilt_button)

        analysis_row.addSpacing(10)

        self.analyze_button = QPushButton("Analyze Saved Data")
        self.analyze_button.clicked.connect(self._on_analyze)
        self.analyze_button.setEnabled(False)
        self.analyze_button.setToolTip("Analyze most recent data from saved file")
        self.analyze_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                font-weight: bold;
                padding: 5px 15px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:disabled {
                background-color: #CCCCCC;
                color: #666666;
            }
        """)
        analysis_row.addWidget(self.analyze_button)

        analysis_row.addSpacing(10)

        self.file_status_label = QLabel("No data file")
        self.file_status_label.setStyleSheet("color: #666; font-style: italic;")
        analysis_row.addWidget(self.file_status_label)

        analysis_row.addStretch()
        controls_main_layout.addLayout(analysis_row)

        # === Row 2: Standard Controls ===
        controls_layout = QHBoxLayout()

        # Magnitude scale selection
        scale_label = QLabel("Scale:")
        controls_layout.addWidget(scale_label)

        self.scale_combo = QComboBox()
        self.scale_combo.addItem("Linear", "linear")
        self.scale_combo.addItem("dB", "dB")
        self.scale_combo.setCurrentIndex(1)  # Default to dB
        self.scale_combo.currentIndexChanged.connect(self._on_scale_changed)
        controls_layout.addWidget(self.scale_combo)

        controls_layout.addSpacing(20)

        # Frequency range
        freq_label = QLabel("Freq Range:")
        controls_layout.addWidget(freq_label)

        self.freq_combo = QComboBox()
        self.freq_combo.addItem("Full", "Full")
        self.freq_combo.addItem("0-20 Hz", 20)
        self.freq_combo.addItem("0-100 Hz", 100)
        self.freq_combo.addItem("0-200 Hz", 200)
        self.freq_combo.addItem("0-500 Hz", 500)
        self.freq_combo.addItem("0-1 kHz", 1000)
        self.freq_combo.addItem("0-2 kHz", 2000)
        self.freq_combo.addItem("0-5 kHz", 5000)
        self.freq_combo.currentIndexChanged.connect(self._on_freq_range_changed)
        controls_layout.addWidget(self.freq_combo)

        controls_layout.addSpacing(20)

        # Show peaks checkbox
        self.peaks_checkbox = QCheckBox("Show Peaks")
        self.peaks_checkbox.setChecked(self.show_peaks)
        self.peaks_checkbox.stateChanged.connect(self._on_peaks_changed)
        controls_layout.addWidget(self.peaks_checkbox)

        controls_layout.addSpacing(20)

        # Peak threshold
        threshold_label = QLabel("Threshold:")
        controls_layout.addWidget(threshold_label)

        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.01, 1.0)
        self.threshold_spin.setSingleStep(0.05)
        self.threshold_spin.setValue(self.peak_threshold)
        self.threshold_spin.valueChanged.connect(self._on_threshold_changed)
        controls_layout.addWidget(self.threshold_spin)

        controls_layout.addSpacing(20)

        # Display mode: Stack/Overlay
        mode_label = QLabel("Display:")
        controls_layout.addWidget(mode_label)
        
        self.display_mode_combo = QComboBox()
        self.display_mode_combo.addItem("Overlay", "overlay")
        self.display_mode_combo.addItem("Stack", "stack")
        self.display_mode_combo.currentIndexChanged.connect(self._on_display_mode_changed)
        controls_layout.addWidget(self.display_mode_combo)

        controls_layout.addSpacing(20)

        # Clear button
        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self._on_clear)
        controls_layout.addWidget(self.clear_button)

        controls_layout.addStretch()
        controls_main_layout.addLayout(controls_layout)

        return controls_main_layout

    def _get_magnitude_unit(self) -> str:
        """Get magnitude unit string based on scale."""
        if self.magnitude_scale == "dB":
            return "dB"
        else:
            return self.channel_units + "/√Hz"

    def configure(
        self,
        n_channels: int,
        sample_rate: float,
        channel_names: Optional[List[str]] = None,
        channel_units: str = "g"
    ):
        """
        Configure the FFT plot widget.

        Args:
            n_channels: Number of channels
            sample_rate: Sampling rate in Hz
            channel_names: Optional list of channel names
            channel_units: Engineering units for display
        """
        self.n_channels = n_channels
        self.sample_rate = sample_rate
        self.channel_units = channel_units

        if channel_names is None:
            self.channel_names = [f"Channel {i+1}" for i in range(n_channels)]
        else:
            self.channel_names = channel_names

        # Initialize visibility
        self.channel_visible = [True] * n_channels

        # Rebuild stack container if in stack mode
        if PLOTLY_AVAILABLE:
            if self.display_mode == "stack":
                # Recreate stack container with new number of channels
                self._create_stack_plot_container()
            else:
                # Just trigger a redraw for overlay mode
                self._rebuild_plots()

        # Initialize long-window FFT processor
        self.long_fft_processor = LongWindowFFTProcessor(
            sample_rate=sample_rate,
            window_function='hann'
        )

        self.logger.info(
            f"FFT plot configured: {n_channels} channels @ {sample_rate} Hz, units={channel_units}"
        )

    def set_data_file(self, filepath: str):
        """
        Set the current data file for FFT analysis.

        Args:
            filepath: Path to the data file created during acquisition
        """
        self.current_data_file = Path(filepath)
        self.analyze_button.setEnabled(True)

        # Get file info
        try:
            reader = DataFileReader()
            self.data_file_info = reader.get_file_info(filepath)

            duration = self.data_file_info.get('duration_seconds', 0)
            if duration > 0:
                self.file_status_label.setText(f"File: {duration:.1f}s available")
                self.file_status_label.setStyleSheet("color: #4CAF50; font-style: italic;")
            else:
                self.file_status_label.setText(f"File: {Path(filepath).name}")
                self.file_status_label.setStyleSheet("color: #2196F3; font-style: italic;")

            self.logger.info(f"Data file set: {filepath}")

        except Exception as e:
            self.logger.error(f"Failed to get file info: {e}")
            self.file_status_label.setText("File: Error reading info")
            self.file_status_label.setStyleSheet("color: #f44336; font-style: italic;")

    def set_filter_config(self, config: dict) -> None:
        """
        Set the filter configuration used for offline FFT analysis.

        Args:
            config: Filter configuration dictionary
        """
        if config is None:
            self.filter_config = {}
        else:
            self.filter_config = config.copy()
            self.filter_enabled = bool(config.get('enabled', self.filter_enabled))

    def set_filter_enabled(self, enabled: bool) -> None:
        """Enable or disable offline filtering for FFT analysis."""
        self.filter_enabled = enabled

    def get_filter_config(self) -> dict:
        """Get current filter configuration for offline FFT analysis."""
        config = self.filter_config.copy() if self.filter_config else {}
        config['enabled'] = self.filter_enabled
        return config

    def _on_analyze(self):
        """Analyze saved data when button clicked."""
        if not self.current_data_file or not self.current_data_file.exists():
            QMessageBox.warning(self, "No Data", "No data file available for analysis.")
            return

        # Get selected duration (this is the MAXIMUM duration)
        duration_str = self.fft_duration_combo.currentText()
        max_duration = float(duration_str.replace('s', ''))

        try:
            # Read recent data (up to max_duration)
            reader = DataFileReader()
            data, metadata = reader.read_recent_seconds(
                str(self.current_data_file),
                max_duration,
                self.sample_rate
            )

            actual_duration = metadata.get('duration_read', 0)
            samples_read = metadata.get('samples_read', 0)

            # Check if we have enough data for meaningful FFT
            if samples_read < 256:  # Minimum FFT size
                QMessageBox.warning(
                    self,
                    "Insufficient Data",
                    f"Not enough data for FFT analysis.\n"
                    f"Available: {samples_read} samples ({actual_duration:.2f}s)\n"
                    f"Minimum required: 256 samples"
                )
                return

            # Ensure plot is configured for saved data analysis
            if self.n_channels != data.shape[0]:
                sample_rate = self.sample_rate
                if self.data_file_info and 'sample_rate' in self.data_file_info:
                    sample_rate = float(self.data_file_info['sample_rate'])

                channel_names = [f"Channel {i + 1}" for i in range(data.shape[0])]
                self.configure(
                    n_channels=data.shape[0],
                    sample_rate=sample_rate,
                    channel_names=channel_names,
                    channel_units=self.channel_units
                )

            self.logger.info(
                f"Loaded {samples_read} samples ({actual_duration:.2f}s) for FFT analysis "
                f"(requested: {max_duration}s)"
            )

            # Apply filter before FFT if enabled
            data_to_analyze = data
            if self.filter_enabled:
                data_to_analyze = self._apply_filter_for_analysis(data)

            # Perform FFT using available data
            self._compute_and_display_fft(data_to_analyze, f"{actual_duration:.1f}s")

        except Exception as e:
            self.logger.error(f"FFT analysis failed: {e}")
            QMessageBox.critical(self, "Analysis Error", f"Failed to analyze data:\n{str(e)}")

    def _open_filtfilt_dialog(self):
        """Open filter configuration dialog for saved-data analysis."""
        from .filter_config_panel import FilterConfigPanel

        dialog = QDialog(self)
        dialog.setWindowTitle("Filter Status")

        layout = QVBoxLayout(dialog)
        panel = FilterConfigPanel()

        panel.set_filter_config(self.get_filter_config())

        layout.addWidget(panel)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec_() == QDialog.Accepted:
            config = panel.get_current_ui_config()
            self.filter_config = config.copy()
            self.filter_enabled = bool(config.get('enabled', False))
            self.filter_config_updated.emit(config)
            self.filter_enabled_updated.emit(self.filter_enabled)

    def _apply_filter_for_analysis(self, data: np.ndarray) -> np.ndarray:
        """
        Apply configured filter to offline data before FFT analysis.

        Args:
            data: Input data of shape (n_channels, n_samples)

        Returns:
            Filtered data with same shape
        """
        if not self.filter_config:
            return data

        filter_type = self.filter_config.get('type', ProcessingDefaults.DEFAULT_FILTER_TYPE)
        filter_mode = self.filter_config.get('mode', ProcessingDefaults.FILTER_MODE_LOWPASS)
        order = int(self.filter_config.get('order', ProcessingDefaults.DEFAULT_FILTER_ORDER))
        cutoff = self._resolve_filter_cutoff(filter_mode)

        if cutoff is None:
            raise ValueError("Filter cutoff is not configured")

        self.logger.info(
            f"Applying filter before FFT: {filter_type} {filter_mode}, "
            f"cutoff={cutoff}, order={order}"
        )

        return apply_zero_phase_filter(
            data,
            filter_type,
            filter_mode,
            cutoff,
            self.sample_rate,
            order
        )

    def _resolve_filter_cutoff(
        self,
        filter_mode: str
    ) -> Optional[Union[float, Tuple[float, float]]]:
        """Resolve cutoff frequency based on filter mode and config."""
        cutoff = self.filter_config.get('cutoff')

        if filter_mode in [ProcessingDefaults.FILTER_MODE_BANDPASS,
                           ProcessingDefaults.FILTER_MODE_BANDSTOP]:
            if isinstance(cutoff, (list, tuple)) and len(cutoff) == 2:
                return float(cutoff[0]), float(cutoff[1])

            cutoff_low = self.filter_config.get('cutoff_low')
            cutoff_high = self.filter_config.get('cutoff_high')
            if cutoff_low is None or cutoff_high is None:
                return None
            return float(cutoff_low), float(cutoff_high)

        if cutoff is None:
            cutoff = self.filter_config.get('cutoff_low')

        if cutoff is None:
            return None

        return float(cutoff)

    def _compute_and_display_fft(self, data: np.ndarray, window_duration: str):
        """
        Compute FFT on loaded data and display results.

        Args:
            data: Data array of shape (n_channels, n_samples)
            window_duration: Duration string (e.g., "10.5s")
        """
        try:
            # Compute FFT for each channel
            all_frequencies = None
            all_magnitudes = []

            n_samples = data.shape[1]

            for ch_idx in range(data.shape[0]):
                channel_data = data[ch_idx, :]

                # Compute FFT directly using numpy (more flexible than LongWindowFFTProcessor)
                # Apply Hann window
                window = np.hanning(n_samples)
                windowed_data = channel_data * window

                # Compute FFT
                fft_result = np.fft.rfft(windowed_data)
                frequencies = np.fft.rfftfreq(n_samples, 1.0 / self.sample_rate)

                # Compute magnitude
                magnitude = np.abs(fft_result)

                # Apply scaling based on selected scale
                if self.magnitude_scale.lower() == "db":
                    # Convert to dB (with floor to avoid log(0))
                    magnitude = 20 * np.log10(magnitude + 1e-10)
                else:
                    # Normalize by window size for linear scale
                    magnitude = magnitude * 2.0 / n_samples

                if all_frequencies is None:
                    all_frequencies = frequencies

                all_magnitudes.append(magnitude)

            # Store FFT results
            self.frequencies = all_frequencies
            self.magnitudes = np.array(all_magnitudes)

            # Find peaks if enabled
            if self.show_peaks:
                self.peaks = self._find_peaks_all_channels()
            else:
                self.peaks = []

            # Update plots
            if self.display_mode == "overlay":
                self._update_overlay_plots()
            else:
                self._update_stack_plots()

            freq_resolution = frequencies[1] - frequencies[0] if len(frequencies) > 1 else 0
            self.logger.info(
                f"FFT complete: {window_duration} window, "
                f"{n_samples} samples, "
                f"freq_resolution={freq_resolution:.4f} Hz"
            )

        except Exception as e:
            self.logger.error(f"FFT computation failed: {e}")
            raise

    def _find_peaks_all_channels(self) -> List[Dict]:
        """Find peaks for all channels."""
        from scipy.signal import find_peaks

        peaks_list = []

        for ch_idx in range(self.n_channels):
            if ch_idx >= len(self.magnitudes):
                continue

            magnitude = self.magnitudes[ch_idx]

            # Find peaks using scipy
            try:
                # Calculate threshold
                max_magnitude = np.max(magnitude)
                threshold = max_magnitude * self.peak_threshold

                # Find peaks with minimum distance between them
                peak_indices, properties = find_peaks(
                    magnitude,
                    height=threshold,
                    distance=10  # Minimum 10 samples between peaks
                )

                # Limit to top 10 peaks
                if len(peak_indices) > 10:
                    # Sort by magnitude and take top 10
                    sorted_indices = np.argsort(magnitude[peak_indices])[::-1]
                    peak_indices = peak_indices[sorted_indices[:10]]

                # Store peak information
                channel_peaks = {
                    'channel': ch_idx,
                    'frequencies': self.frequencies[peak_indices],
                    'magnitudes': magnitude[peak_indices],
                    'indices': peak_indices
                }
                peaks_list.append(channel_peaks)

            except Exception as e:
                self.logger.warning(f"Peak detection failed for channel {ch_idx}: {e}")

        return peaks_list

    def _populate_fft_size_combo(self):
        """Populate FFT size combo box with time-based options."""
        # Block signals during population
        self.fft_size_combo.blockSignals(True)
        self.fft_size_combo.clear()

        # IMPORTANT: Real-time mode uses SMALL windows, Long-Window mode uses LARGE windows
        # This is because real-time FFT needs fast updates, long-window FFT needs high frequency resolution

        # Define desired time windows based on mode
        if self.fft_mode == "realtime":
            # Real-time mode: Small windows (10ms - 640ms)
            time_windows_sec = [0.01, 0.02, 0.05, 0.08, 0.1, 0.2, 0.32, 0.64]
        else:  # long mode
            # Long-window mode: Large windows (10s - 200s)
            # Note: These are NOT used for real-time display!
            # User must click "Analyze" button to compute
            time_windows_sec = [10, 20, 50, 100, 200]

        for time_sec in time_windows_sec:
            # Calculate number of samples needed
            samples_needed = int(time_sec * self.sample_rate)

            # Find nearest power of 2
            fft_size = self._nearest_power_of_2(samples_needed)

            # Calculate actual time duration and frequency resolution
            actual_time_sec = fft_size / self.sample_rate
            freq_resolution = self.sample_rate / fft_size

            # Format label - show in seconds or milliseconds
            if time_sec >= 1.0:
                time_str = f"{time_sec:.1f} s"
                actual_str = f"{actual_time_sec:.2f} s"
            else:
                time_ms = time_sec * 1000
                actual_ms = actual_time_sec * 1000
                time_str = f"{time_ms:.0f} ms"
                actual_str = f"{actual_ms:.1f} ms"

            # Show frequency resolution in appropriate unit
            if freq_resolution >= 1.0:
                freq_str = f"{freq_resolution:.2f} Hz"
            else:
                freq_str = f"{freq_resolution*1000:.2f} mHz"

            label = f"{time_str} → {fft_size} pts ({actual_str}, Δf={freq_str})"
            self.fft_size_combo.addItem(label, fft_size)

        # Set default based on mode
        if self.fft_mode == "realtime":
            # Set default to ~80ms (index 3)
            if self.fft_size_combo.count() > 3:
                self.fft_size_combo.setCurrentIndex(3)
        else:
            # Set default to 200s (last item)
            if self.fft_size_combo.count() > 0:
                self.fft_size_combo.setCurrentIndex(self.fft_size_combo.count() - 1)

        # Re-enable signals
        self.fft_size_combo.blockSignals(False)

        self.logger.debug(f"Populated FFT size combo with {len(time_windows_sec)} time-based options for {self.fft_mode} mode")
    
    def _nearest_power_of_2(self, n: int) -> int:
        """
        Find nearest power of 2 to n.
        
        Args:
            n: Input number
            
        Returns:
            Nearest power of 2
        """
        import math
        
        if n <= 0:
            return 512  # Minimum FFT size
        
        # Calculate log base 2
        power = round(math.log2(n))
        
        # Ensure minimum and maximum bounds
        # Extended range for long FFT windows: 2^9 (512) to 2^24 (16,777,216)
        power = max(9, min(24, power))
        
        return 2 ** power

    def _create_curves(self):
        """No-op for Plotly backend (plots are rebuilt per update)."""
        return
    
    def _create_overlay_plots(self):
        """No-op for Plotly backend (plots are rebuilt per update)."""
        return
    
    def _create_stack_plots(self):
        """No-op for Plotly backend (plots are rebuilt per update)."""
        return

    def _plotly_background(self) -> str:
        return "#ffffff" if GUIDefaults.PLOT_BACKGROUND == "w" else GUIDefaults.PLOT_BACKGROUND

    def _base_layout(self, y_title: str, x_title: str, show_legend: bool) -> Dict:
        return {
            "autosize": True,
            "margin": {"l": 60, "r": 20, "t": 20, "b": 45},
            "plot_bgcolor": self._plotly_background(),
            "paper_bgcolor": self._plotly_background(),
            "showlegend": show_legend,
            "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
            "xaxis": {
                "title": x_title,
                "showgrid": True,
                "gridcolor": "#e0e0e0",
                "zeroline": False,
                "automargin": True
            },
            "yaxis": {
                "title": y_title,
                "showgrid": True,
                "gridcolor": "#e0e0e0",
                "zeroline": False,
                "automargin": True
            }
        }

    def _init_empty_plot(self):
        """Initialize empty plot after widget is visible."""
        print("FFTPlotWidget._init_empty_plot: Called")
        if PLOTLY_AVAILABLE and self.plot_view_single is not None:
            print(f"FFTPlotWidget._init_empty_plot: Calling update_plot, view size={self.plot_view_single.size()}")
            self.plot_view_single.update_plot(
                [],
                self._base_layout(
                    y_title=f"Magnitude ({self._get_magnitude_unit()})",
                    x_title="Frequency (Hz)",
                    show_legend=True
                )
            )
            print("FFTPlotWidget._init_empty_plot: update_plot called")
        else:
            print(f"FFTPlotWidget._init_empty_plot: Skipped, PLOTLY_AVAILABLE={PLOTLY_AVAILABLE}, plot_view_single={self.plot_view_single}")
    
    def _rebuild_plots(self):
        """Rebuild plots when switching between overlay and stack modes."""
        if not PLOTLY_AVAILABLE:
            return

        if self.frequencies is not None and self.magnitudes is not None:
            if self.display_mode == "overlay":
                self._update_overlay_plots()
            else:
                self._update_stack_plots()

        self.logger.info(f"Rebuilt FFT plots in {self.display_mode} mode")

    def _create_stack_plot_container(self):
        """Create container with individual PlotlyView widgets for stack mode."""
        if not PLOTLY_AVAILABLE:
            return

        # Destroy any existing stack container
        self._destroy_stack_plot_container()

        # Create container widget
        self.plot_container_stack = QWidget()
        container_layout = QVBoxLayout(self.plot_container_stack)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(2)  # Minimal spacing between plots

        # Create PlotlyView for each channel
        self.plot_views_stack = []
        for i in range(self.n_channels):
            plot_view = PlotlyView()
            plot_view.setFixedHeight(self.subplot_height)
            plot_view.setMinimumWidth(400)

            # Add to container
            container_layout.addWidget(plot_view)
            self.plot_views_stack.append(plot_view)

            # Initialize with empty plot
            channel_name = self.channel_names[i] if i < len(self.channel_names) else f"Channel {i+1}"
            QTimer.singleShot(
                500 + i * 100,  # Stagger initialization to avoid overload
                lambda view=plot_view, name=channel_name: view.update_plot(
                    [],
                    self._base_layout(
                        y_title=f"{self._get_magnitude_unit()}",
                        x_title="Frequency (Hz)",
                        show_legend=False
                    )
                )
            )

        # Add stretch at the end to push plots to top
        container_layout.addStretch()

        # Set container to scroll area
        self.scroll_area.setWidget(self.plot_container_stack)
        self._update_stack_plot_heights()

        self.logger.info(f"Created FFT stack plot container with {self.n_channels} PlotlyView widgets")

    def _destroy_stack_plot_container(self):
        """Destroy stack plot container and clean up PlotlyView widgets."""
        if self.plot_views_stack:
            # Clear and delete all PlotlyView widgets
            for plot_view in self.plot_views_stack:
                try:
                    plot_view.deleteLater()
                except Exception as e:
                    self.logger.warning(f"Error deleting FFT PlotlyView: {e}")
            self.plot_views_stack = []
            self.logger.debug("Destroyed all FFT stack PlotlyView widgets")

        if self.plot_container_stack is not None:
            try:
                self.plot_container_stack.deleteLater()
            except Exception as e:
                self.logger.warning(f"Error deleting FFT container: {e}")
            self.plot_container_stack = None
            self.logger.debug("Destroyed FFT stack plot container")

    def _update_stack_plot_heights(self):
        """Resize stack plots to fit available space when few channels are visible."""
        if not self.plot_views_stack or not self.scroll_area:
            return

        visible_count = sum(1 for visible in self.channel_visible if visible)
        if visible_count <= 0:
            visible_count = self.n_channels

        target_height = self.subplot_height
        viewport_height = self.scroll_area.viewport().height()
        if visible_count <= self.max_visible_channels and viewport_height > 0:
            target_height = max(self.min_subplot_height, int(viewport_height / max(1, visible_count)))

        for plot_view in self.plot_views_stack:
            plot_view.setFixedHeight(target_height)

    @pyqtSlot(np.ndarray, np.ndarray, int)
    def update_plot(self, frequencies: np.ndarray, magnitude: np.ndarray, channel: int):
        """
        Update plot with new FFT data (called from signal) - DEPRECATED.

        This method is no longer used as FFT is now file-based.
        Kept for backwards compatibility.

        Args:
            frequencies: Frequency array (Hz)
            magnitude: Magnitude array (linear or dB)
            channel: Channel index
        """
        # No longer used - FFT is file-based now
        pass

    def _update_plots(self):
        """
        Update plots with pending data (called by timer) - DEPRECATED.

        This method is no longer used as FFT is now file-based.
        """
        # No longer used - FFT is file-based now
        pass
    
    def _update_overlay_plots(self):
        """Update plots in overlay mode (single plot with all channels)."""
        if self.frequencies is None or self.magnitudes is None:
            return
        if not PLOTLY_AVAILABLE or self.plot_view_single is None:
            return

        colors = GUIDefaults.PLOT_COLORS
        traces = []
        for channel in range(self.n_channels):
            if channel >= len(self.magnitudes):
                continue
            if channel < len(self.channel_visible) and not self.channel_visible[channel]:
                continue

            freq = self.frequencies
            mag = self.magnitudes[channel]

            if self.frequency_limit != "Full":
                max_freq = float(self.frequency_limit)
                mask = freq <= max_freq
                freq_plot = freq[mask]
                mag_plot = mag[mask]
            else:
                freq_plot = freq
                mag_plot = mag.copy()

            traces.append(
                go.Scatter(
                    x=freq_plot,
                    y=mag_plot,
                    mode="lines",
                    name=self.channel_names[channel],
                    line={"color": colors[channel % len(colors)], "width": GUIDefaults.PLOT_LINE_WIDTH}
                )
            )

            if self.show_peaks and self.peaks:
                peak_info = next((p for p in self.peaks if p['channel'] == channel), None)
                if peak_info is not None:
                    peak_freqs = peak_info['frequencies']
                    peak_mags = peak_info['magnitudes']
                    if self.frequency_limit != "Full":
                        max_freq = float(self.frequency_limit)
                        mask = peak_freqs <= max_freq
                        peak_freqs = peak_freqs[mask]
                        peak_mags = peak_mags[mask]
                    if len(peak_freqs) > 0:
                        traces.append(
                            go.Scatter(
                                x=peak_freqs,
                                y=peak_mags,
                                mode="markers",
                                name=f"{self.channel_names[channel]} peaks",
                                marker={"color": colors[channel % len(colors)], "size": 8},
                                showlegend=False
                            )
                        )

        layout = self._base_layout(
            y_title=f"Magnitude ({self._get_magnitude_unit()})",
            x_title="Frequency (Hz)",
            show_legend=True
        )
        if self.frequency_limit != "Full":
            layout["xaxis"]["range"] = [0, float(self.frequency_limit)]
        self.plot_view_single.update_plot(traces, layout)

    def _update_stack_plots(self):
        """Update plots in stack mode (separate PlotlyView for each channel)."""
        if self.frequencies is None or self.magnitudes is None:
            return
        if not PLOTLY_AVAILABLE or not self.plot_views_stack:
            return

        # Ensure we have the correct number of PlotlyView widgets
        if len(self.plot_views_stack) != self.n_channels:
            self.logger.warning(f"Mismatch: {len(self.plot_views_stack)} views vs {self.n_channels} channels. Recreating container.")
            self._create_stack_plot_container()
            return

        # Update each PlotlyView individually
        colors = GUIDefaults.PLOT_COLORS
        for channel in range(self.n_channels):
            # Skip hidden channels
            if channel < len(self.channel_visible) and not self.channel_visible[channel]:
                # Clear the plot for hidden channels
                self.plot_views_stack[channel].update_plot([], self._base_layout(
                    y_title=f"{self._get_magnitude_unit()}",
                    x_title="Frequency (Hz)" if channel == self.n_channels - 1 else "",
                    show_legend=False
                ))
                continue

            if channel >= len(self.magnitudes):
                continue

            freq = self.frequencies
            mag = self.magnitudes[channel]

            # Apply frequency limit if set
            if self.frequency_limit != "Full":
                max_freq = float(self.frequency_limit)
                mask = freq <= max_freq
                freq_plot = freq[mask]
                mag_plot = mag[mask]
            else:
                freq_plot = freq
                mag_plot = mag.copy()

            # Create trace for this channel
            trace = go.Scatter(
                x=freq_plot,
                y=mag_plot,
                mode="lines",
                name=self.channel_names[channel] if channel < len(self.channel_names) else f"Ch{channel+1}",
                line={"color": colors[channel % len(colors)], "width": GUIDefaults.PLOT_LINE_WIDTH}
            )

            traces = [trace]

            # Add peaks if enabled
            if self.show_peaks and self.peaks:
                peak_info = next((p for p in self.peaks if p['channel'] == channel), None)
                if peak_info is not None:
                    peak_freqs = peak_info['frequencies']
                    peak_mags = peak_info['magnitudes']
                    if self.frequency_limit != "Full":
                        max_freq = float(self.frequency_limit)
                        mask = peak_freqs <= max_freq
                        peak_freqs = peak_freqs[mask]
                        peak_mags = peak_mags[mask]
                    if len(peak_freqs) > 0:
                        traces.append(
                            go.Scatter(
                                x=peak_freqs,
                                y=peak_mags,
                                mode="markers",
                                marker={"color": colors[channel % len(colors)], "size": 8},
                                showlegend=False
                            )
                        )

            # Create layout for this individual plot
            layout = {
                "autosize": True,
                "margin": {"l": 60, "r": 20, "t": 30, "b": 40 if channel == self.n_channels - 1 else 20},
                "plot_bgcolor": self._plotly_background(),
                "paper_bgcolor": self._plotly_background(),
                "showlegend": False,
                "title": {
                    "text": f"<b>{self.channel_names[channel] if channel < len(self.channel_names) else f'Ch{channel+1}'}</b>",
                    "font": {"size": 14, "color": "#111"},
                    "x": 0.01,
                    "xanchor": "left"
                },
                "xaxis": {
                    "title": "Frequency (Hz)" if channel == self.n_channels - 1 else "",
                    "showgrid": True,
                    "gridcolor": "#e0e0e0",
                    "zeroline": False,
                    "automargin": True
                },
                "yaxis": {
                    "title": f"{self._get_magnitude_unit()}",
                    "showgrid": True,
                    "gridcolor": "#e0e0e0",
                    "zeroline": False,
                    "automargin": True
                },
                "uirevision": f"channel_{channel}"  # Keep zoom/pan state per channel
            }

            # Apply frequency limit to x-axis if set
            if self.frequency_limit != "Full":
                layout["xaxis"]["range"] = [0, float(self.frequency_limit)]

            # Update this specific PlotlyView
            self.plot_views_stack[channel].update_plot(traces, layout)

    def _find_peaks(self, frequencies: np.ndarray, magnitude: np.ndarray) -> List[Tuple[float, float, int]]:
        """
        Find peaks in the magnitude spectrum.

        Args:
            frequencies: Frequency array
            magnitude: Magnitude array

        Returns:
            List of (frequency, magnitude, index) tuples
        """
        try:
            from scipy.signal import find_peaks

            # Calculate threshold
            if self.magnitude_scale == "dB":
                # For dB, use relative threshold
                max_mag = np.max(magnitude)
                threshold = max_mag * self.peak_threshold
            else:
                # For linear, use relative threshold
                max_mag = np.max(magnitude)
                threshold = max_mag * self.peak_threshold

            # Find peaks
            peak_indices, properties = find_peaks(
                magnitude,
                height=threshold,
                distance=10  # Minimum distance between peaks
            )

            peaks = []
            for idx in peak_indices:
                peaks.append((frequencies[idx], magnitude[idx], idx))

            # Sort by magnitude (descending)
            peaks.sort(key=lambda x: x[1], reverse=True)

            # Limit to top 10 peaks
            return peaks[:10]

        except Exception as e:
            self.logger.error(f"Error finding peaks: {e}")
            return []

    def _on_scale_changed(self, index: int):
        """Handle magnitude scale change."""
        self.magnitude_scale = self.scale_combo.currentData()
        if self.frequencies is not None and self.magnitudes is not None:
            if self.display_mode == "overlay":
                self._update_overlay_plots()
            else:
                self._update_stack_plots()

        self.logger.debug(f"Magnitude scale changed to {self.magnitude_scale}")

    def _on_freq_range_changed(self, index: int):
        """Handle frequency range change."""
        self.frequency_limit = self.freq_combo.currentData()
        if self.frequencies is not None and self.magnitudes is not None:
            if self.display_mode == "overlay":
                self._update_overlay_plots()
            else:
                self._update_stack_plots()
        self.logger.debug(f"Frequency range changed to {self.frequency_limit}")

    def _on_peaks_changed(self, state: int):
        """Handle show peaks checkbox change."""
        self.show_peaks = (state == Qt.Checked)

        if self.show_peaks and self.magnitudes is not None and not self.peaks:
            self.peaks = self._find_peaks_all_channels()
        if self.frequencies is not None and self.magnitudes is not None:
            if self.display_mode == "overlay":
                self._update_overlay_plots()
            else:
                self._update_stack_plots()

        self.logger.debug(f"Show peaks: {self.show_peaks}")

    def _on_threshold_changed(self, value: float):
        """Handle peak threshold change."""
        self.peak_threshold = value
        self.logger.debug(f"Peak threshold changed to {value}")
    
    def _on_fft_size_changed(self, index: int):
        """Handle FFT window size change."""
        fft_size = self.fft_size_combo.currentData()
        self.logger.info(f"FFT window size changed to {fft_size}")
        
        # Emit signal to notify main window
        self.fft_size_changed.emit(fft_size)
    
    def _on_display_mode_changed(self, index: int):
        """Handle display mode change (overlay/stack)."""
        old_mode = self.display_mode
        self.display_mode = self.display_mode_combo.currentData()
        self.logger.debug(f"FFT display mode changing from {old_mode} to {self.display_mode}")

        # Switch between overlay and stack modes
        if old_mode != self.display_mode and self.n_channels > 0:
            try:
                if self.display_mode == "stack":
                    # Switching to stack mode: create individual plot views
                    self.logger.info("FFT: Switching to stack mode: creating individual PlotlyView widgets")

                    # Destroy single plot view if it exists
                    if self.plot_view_single is not None:
                        self.plot_view_single.deleteLater()
                        self.plot_view_single = None

                    # Create stack plot container
                    self._create_stack_plot_container()

                else:  # overlay mode
                    # Switching to overlay mode: use single plot view
                    self.logger.info("FFT: Switching to overlay mode: creating single PlotlyView widget")

                    # Destroy stack plot container
                    self._destroy_stack_plot_container()

                    # Create single plot view
                    self.plot_view_single = PlotlyView()
                    self.plot_view_single.setMinimumHeight(300)
                    self.scroll_area.setWidget(self.plot_view_single)

                    # Initialize with empty plot
                    QTimer.singleShot(500, self._init_empty_plot)

                # Trigger redraw with current data if available
                self._rebuild_plots()

            except Exception as e:
                self.logger.error(f"Error during FFT display mode change: {e}")

        self.logger.debug(f"FFT display mode changed to {self.display_mode}")

    def _on_clear(self):
        """Clear the plot."""
        if not PLOTLY_AVAILABLE:
            return

        if self.display_mode == "overlay" and self.plot_view_single is not None:
            # Clear single overlay plot
            self.plot_view_single.update_plot([], self._base_layout(
                y_title=f"Magnitude ({self._get_magnitude_unit()})",
                x_title="Frequency (Hz)",
                show_legend=True
            ))
        elif self.display_mode == "stack" and self.plot_views_stack:
            # Clear all stack plots
            for i, plot_view in enumerate(self.plot_views_stack):
                channel_name = self.channel_names[i] if i < len(self.channel_names) else f"Ch{i+1}"
                layout = self._base_layout(
                    y_title=f"{self._get_magnitude_unit()}",
                    x_title="Frequency (Hz)" if i == len(self.plot_views_stack) - 1 else "",
                    show_legend=False
                )
                layout["title"] = {
                    "text": f"<b>{channel_name}</b>",
                    "font": {"size": 14, "color": "#111"},
                    "x": 0.01,
                    "xanchor": "left"
                }
                plot_view.update_plot([], layout)

        self.logger.debug("FFT plot cleared")

    def set_channel_visibility(self, channel_idx: int, visible: bool):
        """
        Set visibility for a specific channel.

        Args:
            channel_idx: Channel index
            visible: True to show, False to hide
        """
        if 0 <= channel_idx < len(self.channel_visible):
            self.channel_visible[channel_idx] = visible
            if self.frequencies is not None and self.magnitudes is not None:
                if self.display_mode == "overlay":
                    self._update_overlay_plots()
                else:
                    self._update_stack_plots()
            if self.display_mode == "stack":
                self._update_stack_plot_heights()

    def show_all_channels(self):
        """Show all channels."""
        for i in range(len(self.channel_visible)):
            self.set_channel_visibility(i, True)

    def hide_all_channels(self):
        """Hide all channels."""
        for i in range(len(self.channel_visible)):
            self.set_channel_visibility(i, False)

    def start(self):
        """Start plot updates - No-op for FFT (file-based analysis only)."""
        # FFT widget is file-based, no real-time updates
        self.logger.debug("FFT plot start called (no-op for file-based analysis)")

    def stop(self):
        """Stop plot updates - No-op for FFT (file-based analysis only)."""
        # FFT widget is file-based, no real-time updates
        self.logger.debug("FFT plot stop called (no-op for file-based analysis)")

    def closeEvent(self, event):
        """Handle widget close event."""
        # No cleanup needed for FFT widget (file-based only)
        event.accept()

    def resizeEvent(self, event):
        """Handle resize events to keep stack plots sized to the window."""
        super().resizeEvent(event)
        if self.display_mode == "stack":
            self._update_stack_plot_heights()
    
    def contextMenuEvent(self, event):
        """Handle right-click context menu."""
        menu = QMenu(self)
        
        # Display mode submenu
        display_menu = menu.addMenu("Display Mode")
        
        overlay_action = QAction("Overlay", self)
        overlay_action.setCheckable(True)
        overlay_action.setChecked(self.display_mode == "overlay")
        overlay_action.triggered.connect(lambda: self._set_display_mode("overlay"))
        display_menu.addAction(overlay_action)
        
        stack_action = QAction("Stack", self)
        stack_action.setCheckable(True)
        stack_action.setChecked(self.display_mode == "stack")
        stack_action.triggered.connect(lambda: self._set_display_mode("stack"))
        display_menu.addAction(stack_action)
        
        menu.addSeparator()
        
        # Channel visibility submenu
        channels_menu = menu.addMenu("Channels")
        
        show_all_action = QAction("Show All", self)
        show_all_action.triggered.connect(self.show_all_channels)
        channels_menu.addAction(show_all_action)
        
        hide_all_action = QAction("Hide All", self)
        hide_all_action.triggered.connect(self.hide_all_channels)
        channels_menu.addAction(hide_all_action)
        
        channels_menu.addSeparator()
        
        # Individual channel toggles
        for i in range(min(self.n_channels, len(self.channel_names))):
            action = QAction(self.channel_names[i], self)
            action.setCheckable(True)
            action.setChecked(self.channel_visible[i] if i < len(self.channel_visible) else False)
            action.triggered.connect(lambda checked, ch=i: self.set_channel_visibility(ch, checked))
            channels_menu.addAction(action)
        
        menu.addSeparator()
        
        # Other options
        clear_action = QAction("Clear Plot", self)
        clear_action.triggered.connect(self._on_clear)
        menu.addAction(clear_action)
        
        # Show menu at cursor position
        menu.exec_(event.globalPos())
    
    def _set_display_mode(self, mode: str):
        """Set display mode and update UI."""
        self.display_mode = mode
        # Update combo box
        idx = self.display_mode_combo.findData(mode)
        if idx >= 0:
            self.display_mode_combo.setCurrentIndex(idx)
        self.logger.debug(f"Display mode set to {mode}")

    # ========================================================================
    # Long-Window FFT Methods
    # ========================================================================

    def set_buffer(self, buffer):
        """Set reference to data buffer for long-window FFT."""
        self.buffer = buffer
        self.logger.info("Buffer reference set for long-window FFT")

    @pyqtSlot()
    def _on_fft_mode_changed(self):
        """Handle FFT mode change (realtime vs long-window)."""
        old_mode = self.fft_mode
        self.fft_mode = self.fft_mode_combo.currentData()

        # Show/hide controls based on mode
        is_long_mode = (self.fft_mode == "long")

        # Show/hide long window controls
        self.long_controls_widget.setVisible(is_long_mode)

        # Show/hide FFT Size controls (only for real-time mode)
        # In long mode, window size is selected via "Window" dropdown
        self.fft_size_label.setVisible(not is_long_mode)
        self.fft_size_combo.setVisible(not is_long_mode)

        # Re-populate FFT size combo with appropriate window sizes
        if old_mode != self.fft_mode:
            self._populate_fft_size_combo()

            # If switching to long mode, stop real-time updates
            if is_long_mode:
                self.logger.info("Switched to Long-Window mode - real-time FFT updates paused")
                # Real-time FFT will not update until user clicks "Analyze"
            else:
                self.logger.info("Switched to Real-time mode - resuming FFT updates")

        self.logger.info(f"FFT mode changed to: {self.fft_mode}")

    @pyqtSlot()
    def _on_save_buffer(self):
        """Save buffer data to temporary file."""
        if self.buffer is None:
            QMessageBox.warning(
                self,
                "No Buffer",
                "No data buffer available. Please start acquisition first."
            )
            self.logger.error("Save buffer failed: buffer is None")
            return

        if self.long_data_saver is None:
            QMessageBox.warning(
                self,
                "Not Initialized",
                "Sample rate not set. Please initialize properly."
            )
            self.logger.error("Save buffer failed: long_data_saver is None")
            return

        try:
            # Get duration
            duration_str = self.save_duration_combo.currentText()
            duration = float(duration_str.replace('s', ''))

            self.logger.info(f"Saving buffer: duration={duration}s, buffer type={type(self.buffer)}")

            self.save_buffer_button.setEnabled(False)
            self.buffer_status_label.setText("Saving...")

            # Save buffer
            self.logger.debug(f"Calling save_from_buffer with duration={duration}")
            file_path = self.long_data_saver.save_from_buffer(
                self.buffer,
                duration_seconds=duration,
                format='hdf5'
            )
            self.logger.debug(f"Buffer saved to: {file_path}")

            # Load it back
            self.logger.debug("Loading saved data...")
            self.saved_data, metadata = self.long_data_saver.load_temp_file(file_path)
            self.logger.debug(f"Loaded data shape: {self.saved_data.shape}")

            # Update UI
            n_samples = self.saved_data.shape[1]
            duration_actual = n_samples / self.sample_rate
            status_text = f"✓ {n_samples} samples ({duration_actual:.1f}s)"
            self.buffer_status_label.setText(status_text)

            # Enable analysis
            self.analyze_long_button.setEnabled(True)

            self.logger.info(f"Buffer saved successfully: {file_path}, shape={self.saved_data.shape}")

        except Exception as e:
            import traceback
            self.logger.error(f"Failed to save buffer: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            QMessageBox.critical(
                self,
                "Save Error",
                f"Failed to save buffer:\n{e}"
            )
            self.buffer_status_label.setText("✗ Error")

        finally:
            self.save_buffer_button.setEnabled(True)

    @pyqtSlot()
    def _on_long_window_changed(self):
        """Handle long window duration change."""
        self.long_window_duration = self.long_window_combo.currentText()
        if self.long_fft_processor:
            freq_res = self.long_fft_processor.get_frequency_resolution(self.long_window_duration)
            self.logger.debug(f"Long window changed to {self.long_window_duration}, freq_res={freq_res:.6f} Hz")

    @pyqtSlot()
    def _on_analyze_long(self):
        """Analyze with long-window FFT."""
        if self.saved_data is None:
            QMessageBox.warning(
                self,
                "No Data",
                "No saved data available. Please save buffer first."
            )
            return

        try:
            window_duration = self.long_window_combo.currentText()

            # Analyze all visible channels
            for ch_idx in range(self.n_channels):
                if not self.channel_visible[ch_idx]:
                    continue

                channel_data = self.saved_data[ch_idx, :]

                # Check if data is long enough
                required_samples = self.long_fft_processor.window_sizes[window_duration]
                if len(channel_data) < required_samples:
                    QMessageBox.warning(
                        self,
                        "Insufficient Data",
                        f"Data length ({len(channel_data)} samples) is less than "
                        f"required for {window_duration} window ({required_samples} samples).\n\n"
                        f"Please save more data or use shorter window."
                    )
                    return

            # Disable buttons during analysis
            self.analyze_long_button.setEnabled(False)

            # Perform FFT for all visible channels
            self._perform_long_fft_analysis()

        except Exception as e:
            self.logger.error(f"Failed to analyze: {e}")
            QMessageBox.critical(
                self,
                "Analysis Error",
                f"Failed to analyze:\n{e}"
            )

        finally:
            self.analyze_long_button.setEnabled(True)

    def _perform_long_fft_analysis(self):
        """
        Perform long-window FFT analysis on saved data - DEPRECATED.

        This method is no longer used. Use _on_analyze() instead.
        """
        self.logger.warning("_perform_long_fft_analysis() is deprecated - use _on_analyze() instead")
        return

        window_duration = self.long_window_combo.currentText()
        self.logger.info(f"Window duration: {window_duration}")

        # Get frequency limit for display
        freq_limit = self.freq_combo.currentData()
        max_freq = freq_limit if isinstance(freq_limit, (int, float)) else None
        self.logger.info(f"Frequency limit: {max_freq}")

        # Determine scale - handle both "dB" and "linear"
        # Note: magnitude_scale is "dB" or "linear", need lowercase for processor
        scale = "linear" if self.magnitude_scale == "linear" else "dB"
        self.logger.info(f"Scale: {scale}")

        # Compute for each visible channel
        n_computed = 0
        for ch_idx in range(self.n_channels):
            if not self.channel_visible[ch_idx]:
                self.logger.debug(f"Skipping channel {ch_idx} (not visible)")
                continue

            channel_data = self.saved_data[ch_idx, :]
            self.logger.info(f"Computing FFT for channel {ch_idx}, data shape: {channel_data.shape}")

            try:
                # Compute magnitude spectrum
                frequencies, magnitude = self.long_fft_processor.compute_magnitude(
                    channel_data,
                    window_duration=window_duration,
                    scale=scale
                )
                self.logger.info(f"FFT computed: freq shape={frequencies.shape}, mag shape={magnitude.shape}, "
                               f"freq range=[{frequencies[0]:.6f}, {frequencies[-1]:.2f}] Hz, "
                               f"mag range=[{magnitude.min():.2f}, {magnitude.max():.2f}]")

                # Limit to max frequency if specified
                if max_freq is not None:
                    mask = frequencies <= max_freq
                    frequencies_filtered = frequencies[mask]
                    magnitude_filtered = magnitude[mask]
                    self.logger.info(f"Filtered to {max_freq} Hz: {len(frequencies_filtered)} points")
                else:
                    frequencies_filtered = frequencies
                    magnitude_filtered = magnitude

                # Update plot (similar to real-time)
                self._pending_data[ch_idx] = (frequencies_filtered, magnitude_filtered)
                n_computed += 1
                self.logger.info(f"Channel {ch_idx} data added to pending_data")

            except Exception as e:
                import traceback
                self.logger.error(f"Failed to compute FFT for channel {ch_idx}: {e}")
                self.logger.error(f"Traceback: {traceback.format_exc()}")

        self.logger.info(f"Computed FFT for {n_computed} channels, pending_data keys: {list(self._pending_data.keys())}")

        # Force plot update
        self.logger.info("Calling _update_plots()...")
        self._update_plots()
        self.logger.info("Plot update completed")

        # Log info
        freq_res = self.long_fft_processor.get_frequency_resolution(window_duration)
        self.logger.info(
            f"Long-window FFT analysis completed: {window_duration}, "
            f"freq_res={freq_res:.6f} Hz, {n_computed} channels processed"
        )


# Example usage and testing
if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtCore import QTimer

    app = QApplication(sys.argv)

    # Create widget
    widget = FFTPlotWidget()

    # Configure for 4 channels
    widget.configure(
        n_channels=4,
        sample_rate=25600,
        channel_names=["Ch1", "Ch2", "Ch3", "Ch4"],
        channel_units="g"
    )

    # Generate and display test FFT data
    def generate_test_fft():
        """Generate simulated FFT data."""
        sample_rate = 25600
        n_samples = 2048
        frequencies = np.fft.rfftfreq(n_samples, 1/sample_rate)

        # Create spectrum with peaks at specific frequencies
        channel = 0  # Current channel being updated
        magnitude = np.random.rand(len(frequencies)) * 0.01  # Noise floor

        # Add some peaks
        peak_freqs = [100, 500, 1000, 2000, 5000]
        for freq in peak_freqs:
            idx = np.argmin(np.abs(frequencies - freq))
            magnitude[idx] = 1.0 / (1 + 0.1 * np.abs(frequencies - freq))
            magnitude[idx] += 0.5  # Peak magnitude

        # Convert to dB
        magnitude_db = 20 * np.log10(magnitude + 1e-10)

        return frequencies, magnitude_db, channel

    # Set up timer to simulate real-time data
    test_timer = QTimer()
    test_channel = [0]  # Use list to allow modification in closure

    def update_test():
        freq, mag, ch = generate_test_fft()
        test_channel[0] = (test_channel[0] + 1) % 4  # Cycle through channels
        widget.update_plot(freq, mag, test_channel[0])

    test_timer.timeout.connect(update_test)
    test_timer.start(200)  # Update every 200ms

    widget.show()

    sys.exit(app.exec_())

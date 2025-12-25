"""
Real-time Plot Widget for time-domain visualization.

This widget provides high-performance real-time plotting of multi-channel
acceleration data using Plotly.
"""

import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QComboBox,
    QCheckBox, QPushButton, QLabel, QSpinBox, QMenu, QAction
)
from PyQt5.QtCore import QTimer, pyqtSlot, Qt
from typing import List, Dict, Optional

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from .plotly_view import PlotlyView
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: Plotly or QtWebEngine not available. Install plotly and PyQtWebEngine.")

from ...utils.logger import get_logger
from ...utils.constants import GUIDefaults


class RealtimePlotWidget(QWidget):
    """
    Real-time time-domain plot widget.

    Features:
    - Multi-channel overlay or stacked plots
    - Configurable time window
    - Auto-scaling and manual scaling
    - Channel visibility control
    - Efficient downsampling for display
    - Grid and legend
    """

    def __init__(self, parent=None):
        """
        Initialize the real-time plot widget.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)

        self.logger = get_logger(__name__)

        # Data storage
        self.n_channels = 0
        self.channel_names: List[str] = []
        self.channel_units: str = "g"
        self.sample_rate: float = 25600.0

        # Plot data buffers
        self.time_data: Optional[np.ndarray] = None
        self.plot_data: Optional[np.ndarray] = None

        # Plot data cache
        self.channel_visible: List[bool] = []
        
        # Plot view (Plotly)
        self.plot_view: Optional['PlotlyView'] = None

        # Update rate limiting
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_plots)
        self.update_interval_ms = GUIDefaults.PLOT_UPDATE_INTERVAL_MS
        self.update_timer.setInterval(self.update_interval_ms)

        # Pending data
        self._pending_data: Optional[np.ndarray] = None
        self._pending_timestamp: Optional[float] = None

        # Display settings
        self.time_window = GUIDefaults.DEFAULT_TIME_WINDOW  # seconds
        self.auto_scale = True
        self.downsample_threshold = GUIDefaults.PLOT_DOWNSAMPLE_THRESHOLD
        self.display_mode = "overlay"  # "overlay" or "stack"
        self.stack_offset = 10.0  # Offset between stacked channels (in display units)

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
            self.plot_view = PlotlyView()
            layout.addWidget(self.plot_view)
            # Initial empty plot will be set after widget is shown
            # to ensure proper rendering. Increase delay to 500ms for Plotly.js to load
            print("RealtimePlotWidget: Scheduling _init_empty_plot in 500ms")
            QTimer.singleShot(500, self._init_empty_plot)
        else:
            # Placeholder if PyQtGraph not available
            placeholder = QLabel("Plotly/QtWebEngine not installed.\nInstall with: pip install plotly PyQtWebEngine")
            placeholder.setAlignment(Qt.AlignCenter)
            placeholder.setStyleSheet("QLabel { color: red; font-size: 14px; }")
            layout.addWidget(placeholder)

    def _create_controls(self) -> QHBoxLayout:
        """Create control panel."""
        controls_layout = QHBoxLayout()

        # Time window selection
        window_label = QLabel("Time Window:")
        controls_layout.addWidget(window_label)

        self.window_combo = QComboBox()
        for window in GUIDefaults.TIME_WINDOWS:
            self.window_combo.addItem(f"{window} s", window)

        # Set default
        default_idx = self.window_combo.findData(self.time_window)
        if default_idx >= 0:
            self.window_combo.setCurrentIndex(default_idx)

        self.window_combo.currentIndexChanged.connect(self._on_window_changed)
        controls_layout.addWidget(self.window_combo)

        controls_layout.addSpacing(20)

        # Auto-scale checkbox
        self.autoscale_checkbox = QCheckBox("Auto Scale")
        self.autoscale_checkbox.setChecked(self.auto_scale)
        self.autoscale_checkbox.stateChanged.connect(self._on_autoscale_changed)
        controls_layout.addWidget(self.autoscale_checkbox)

        controls_layout.addSpacing(20)

        # Update rate control
        rate_label = QLabel("Update Rate:")
        controls_layout.addWidget(rate_label)

        self.rate_spin = QSpinBox()
        self.rate_spin.setRange(1, 60)
        self.rate_spin.setValue(GUIDefaults.PLOT_UPDATE_RATE_HZ)
        self.rate_spin.setSuffix(" Hz")
        self.rate_spin.valueChanged.connect(self._on_rate_changed)
        controls_layout.addWidget(self.rate_spin)

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

        return controls_layout

    def configure(
        self,
        n_channels: int,
        sample_rate: float,
        channel_names: Optional[List[str]] = None,
        channel_units: str = "g"
    ):
        """
        Configure the plot widget.

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

        # Trigger a redraw on next update
        if PLOTLY_AVAILABLE:
            self._rebuild_plots()

        self.logger.info(
            f"Plot configured: {n_channels} channels @ {sample_rate} Hz, units={channel_units}"
        )

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
        print("RealtimePlotWidget._init_empty_plot: Called")
        if PLOTLY_AVAILABLE and self.plot_view is not None:
            print(f"RealtimePlotWidget._init_empty_plot: Calling update_plot, view size={self.plot_view.size()}")
            self.plot_view.update_plot(
                [],
                self._base_layout(
                    y_title=f"Acceleration ({self.channel_units})",
                    x_title="Time (s)",
                    show_legend=True
                )
            )
            print("RealtimePlotWidget._init_empty_plot: update_plot called")
        else:
            print(f"RealtimePlotWidget._init_empty_plot: Skipped, PLOTLY_AVAILABLE={PLOTLY_AVAILABLE}, plot_view={self.plot_view}")
    
    def _rebuild_plots(self):
        """Rebuild plots when switching between overlay and stack modes."""
        if not PLOTLY_AVAILABLE:
            return
        if self._pending_data is not None:
            self._update_plots()

        self.logger.info(f"Rebuilt plots in {self.display_mode} mode")

    @pyqtSlot(np.ndarray, float)
    def update_plot(self, data: np.ndarray, timestamp: float):
        """
        Update plot with new data (called from signal).

        Args:
            data: Data array of shape (n_channels, n_samples)
            timestamp: Data timestamp
        """
        # Store pending data (will be plotted in timer callback)
        self._pending_data = data
        self._pending_timestamp = timestamp

        # Start update timer if not running
        if not self.update_timer.isActive():
            self.update_timer.start()

    def _update_plots(self):
        """Update plots with pending data (called by timer)."""
        if self._pending_data is None or not PLOTLY_AVAILABLE:
            return

        try:
            data = self._pending_data

            # Calculate number of samples to display
            n_samples_window = int(self.time_window * self.sample_rate)

            # Get latest data from buffer
            if data.shape[1] > n_samples_window:
                data = data[:, -n_samples_window:]

            # Create time axis
            n_samples = data.shape[1]
            time_axis = np.arange(n_samples) / self.sample_rate
            time_axis = time_axis - time_axis[-1]  # Make latest sample = 0

            # Update curves based on display mode
            if self.display_mode == "overlay":
                self._update_overlay_plots(time_axis, data, n_samples)
            else:  # stack
                self._update_stack_plots(time_axis, data, n_samples)

            # Clear pending data
            self._pending_data = None

        except Exception as e:
            self.logger.error(f"Error updating plot: {e}")
    
    def _update_overlay_plots(self, time_axis: np.ndarray, data: np.ndarray, n_samples: int):
        """Update plots in overlay mode (single plot with all channels)."""
        if not PLOTLY_AVAILABLE or self.plot_view is None:
            return

        colors = GUIDefaults.PLOT_COLORS
        traces = []
        for i in range(self.n_channels):
            if i < len(self.channel_visible) and not self.channel_visible[i]:
                continue
            if i >= data.shape[0]:
                continue

            if n_samples > self.downsample_threshold:
                step = max(1, n_samples // self.downsample_threshold)
                x_vals = time_axis[::step]
                y_vals = data[i, ::step]
            else:
                x_vals = time_axis
                y_vals = data[i, :]

            traces.append(
                go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode="lines",
                    name=self.channel_names[i],
                    line={"color": colors[i % len(colors)], "width": GUIDefaults.PLOT_LINE_WIDTH}
                )
            )

        layout = self._base_layout(
            y_title=f"Acceleration ({self.channel_units})",
            x_title="Time (s)",
            show_legend=True
        )
        self.plot_view.update_plot(traces, layout)
    
    def _update_stack_plots(self, time_axis: np.ndarray, data: np.ndarray, n_samples: int):
        """Update plots in stack mode (separate subplot for each channel)."""
        print(f"_update_stack_plots: Called with n_channels={self.n_channels}, data.shape={data.shape}, n_samples={n_samples}")

        if not PLOTLY_AVAILABLE or self.plot_view is None:
            print(f"_update_stack_plots: Skipping - PLOTLY_AVAILABLE={PLOTLY_AVAILABLE}, plot_view={self.plot_view}")
            return

        # Count visible channels
        visible_count = sum(1 for i in range(min(self.n_channels, len(self.channel_visible)))
                           if self.channel_visible[i] and i < data.shape[0])

        print(f"_update_stack_plots: visible_count={visible_count}, channel_visible={self.channel_visible}")

        if visible_count == 0:
            print("_update_stack_plots: No visible channels, returning")
            return

        # Create subplots with proper spacing
        fig = make_subplots(
            rows=self.n_channels,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=[self.channel_names[i] if i < len(self.channel_names) else f"Ch{i+1}"
                          for i in range(self.n_channels)]
        )

        colors = GUIDefaults.PLOT_COLORS

        for i in range(self.n_channels):
            if i < len(self.channel_visible) and not self.channel_visible[i]:
                continue
            if i >= data.shape[0]:
                continue

            if n_samples > self.downsample_threshold:
                step = max(1, n_samples // self.downsample_threshold)
                x_vals = time_axis[::step]
                y_vals = data[i, ::step]
            else:
                x_vals = time_axis
                y_vals = data[i, :]

            print(f"_update_stack_plots: Adding trace {i}: x_vals.shape={x_vals.shape}, y_vals.shape={y_vals.shape}, time_axis.shape={time_axis.shape}")

            # CRITICAL FIX: Convert numpy arrays to Python lists
            # fig.to_dict() has a bug with numpy arrays in subplots - it loses data
            x_list = x_vals.tolist() if hasattr(x_vals, 'tolist') else list(x_vals)
            y_list = y_vals.tolist() if hasattr(y_vals, 'tolist') else list(y_vals)

            fig.add_trace(
                go.Scatter(
                    x=x_list,  # ← Use Python list instead of numpy array
                    y=y_list,  # ← Use Python list instead of numpy array
                    mode="lines",
                    name=self.channel_names[i] if i < len(self.channel_names) else f"Ch{i+1}",
                    line={"color": colors[i % len(colors)], "width": GUIDefaults.PLOT_LINE_WIDTH},
                    showlegend=False
                ),
                row=i + 1,
                col=1
            )

            # Update y-axis for this subplot
            fig.update_yaxes(
                title_text=f"{self.channel_units}",
                showgrid=True,
                gridcolor="#e0e0e0",
                zeroline=False,
                automargin=True,
                row=i + 1,
                col=1
            )

        # Update x-axis only for bottom subplot
        fig.update_xaxes(
            title_text="Time (s)",
            showgrid=True,
            gridcolor="#e0e0e0",
            zeroline=False,
            automargin=True,
            row=self.n_channels,
            col=1
        )

        # Calculate appropriate height (min 150px per subplot)
        plot_height = max(400, self.n_channels * 150)

        print(f"_update_stack_plots: Creating layout with height={plot_height}")

        fig.update_layout(
            height=plot_height,
            margin={"l": 60, "r": 20, "t": 40, "b": 45},
            plot_bgcolor=self._plotly_background(),
            paper_bgcolor=self._plotly_background(),
            showlegend=False
        )

        fig_dict = fig.to_dict()

        print(f"_update_stack_plots: fig_dict has {len(fig_dict['data'])} traces")
        print(f"_update_stack_plots: Layout height={fig_dict['layout'].get('height', 'NOT SET')}")

        # Debug: Check axes in layout
        layout_axes = [key for key in fig_dict['layout'].keys() if key.startswith('xaxis') or key.startswith('yaxis')]
        print(f"_update_stack_plots: Layout has axes: {layout_axes[:10]}...")  # Print first 10

        # Debug: Print trace axis assignments
        for i, trace in enumerate(fig_dict['data'][:3]):  # First 3 traces
            print(f"_update_stack_plots: Trace {i} - xaxis={trace.get('xaxis', 'NOT SET')}, yaxis={trace.get('yaxis', 'NOT SET')}, x_len={len(trace.get('x', []))}, y_len={len(trace.get('y', []))}")

        self.plot_view.update_plot(fig_dict["data"], fig_dict["layout"])

    def _on_window_changed(self, index: int):
        """Handle time window change."""
        self.time_window = self.window_combo.currentData()
        self.logger.debug(f"Time window changed to {self.time_window}s")

    def _on_autoscale_changed(self, state: int):
        """Handle auto-scale checkbox change."""
        self.auto_scale = (state == Qt.Checked)

        self.logger.debug(f"Auto-scale: {self.auto_scale}")

    def _on_rate_changed(self, value: int):
        """Handle update rate change."""
        self.update_interval_ms = int(1000 / value)
        self.update_timer.setInterval(self.update_interval_ms)
        self.logger.debug(f"Update rate: {value} Hz ({self.update_interval_ms} ms)")
    
    def _on_display_mode_changed(self, index: int):
        """Handle display mode change (overlay/stack)."""
        old_mode = self.display_mode
        self.display_mode = self.display_mode_combo.currentData()
        self.logger.debug(f"Display mode changed to: {self.display_mode}")
        
        # Rebuild plots if mode changed
        if old_mode != self.display_mode and self.n_channels > 0:
            self._rebuild_plots()
        self.logger.debug(f"Display mode changed to {self.display_mode}")

    def _on_clear(self):
        """Clear the plot."""
        if not PLOTLY_AVAILABLE or self.plot_view is None:
            return

        self.plot_view.update_plot([], self._base_layout(
            y_title=f"Acceleration ({self.channel_units})",
            x_title="Time (s)",
            show_legend=True
        ))

        self.logger.debug("Plot cleared")

    def set_channel_visibility(self, channel_idx: int, visible: bool):
        """
        Set visibility for a specific channel.

        Args:
            channel_idx: Channel index
            visible: True to show, False to hide
        """
        if 0 <= channel_idx < len(self.channel_visible):
            self.channel_visible[channel_idx] = visible

    def show_all_channels(self):
        """Show all channels."""
        for i in range(len(self.channel_visible)):
            self.set_channel_visibility(i, True)

    def hide_all_channels(self):
        """Hide all channels."""
        for i in range(len(self.channel_visible)):
            self.set_channel_visibility(i, False)

    def start(self):
        """Start plot updates."""
        if not self.update_timer.isActive():
            self.update_timer.start()
            self.logger.debug("Plot updates started")

    def stop(self):
        """Stop plot updates."""
        if self.update_timer.isActive():
            self.update_timer.stop()
            self.logger.debug("Plot updates stopped")

    def closeEvent(self, event):
        """Handle widget close event."""
        self.stop()
        event.accept()
    
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


# Example usage and testing
if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtCore import QTimer

    app = QApplication(sys.argv)

    # Create widget
    widget = RealtimePlotWidget()

    # Configure for 4 channels
    widget.configure(
        n_channels=4,
        sample_rate=25600,
        channel_names=["Ch1", "Ch2", "Ch3", "Ch4"],
        channel_units="g"
    )

    # Generate and display test data
    def generate_test_data():
        """Generate simulated test data."""
        n_samples = 1000
        t = np.arange(n_samples) / 25600

        data = np.zeros((4, n_samples))
        for i in range(4):
            freq = 100 * (i + 1)  # Different frequency for each channel
            data[i, :] = np.sin(2 * np.pi * freq * t) + 0.1 * np.random.randn(n_samples)

        return data

    # Set up timer to simulate real-time data
    test_timer = QTimer()

    def update_test():
        data = generate_test_data()
        widget.update_plot(data, 0.0)

    test_timer.timeout.connect(update_test)
    test_timer.start(100)  # Update every 100ms

    widget.show()

    sys.exit(app.exec_())

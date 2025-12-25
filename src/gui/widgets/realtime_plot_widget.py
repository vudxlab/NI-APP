"""
Real-time Plot Widget for time-domain visualization.

This widget provides high-performance real-time plotting of multi-channel
acceleration data using PyQtGraph.
"""

import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QComboBox,
    QCheckBox, QPushButton, QLabel, QSpinBox, QMenu, QAction
)
from PyQt5.QtCore import QTimer, pyqtSlot, Qt
from typing import List, Dict, Optional

try:
    import pyqtgraph as pg
    PYQTGRAPH_AVAILABLE = True
except ImportError:
    PYQTGRAPH_AVAILABLE = False
    print("Warning: PyQtGraph not available. Install with: pip install pyqtgraph")

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

        # Plot curves
        self.curves: List['pg.PlotDataItem'] = []
        self.channel_visible: List[bool] = []
        
        # Plot widgets for stack mode (one per channel)
        self.plot_widgets: List['pg.PlotWidget'] = []
        self.graphics_layout = None  # For multi-subplot layout

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

        if not PYQTGRAPH_AVAILABLE:
            self.logger.error("PyQtGraph not available")

        self._init_ui()

    def _init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Controls
        controls_layout = self._create_controls()
        layout.addLayout(controls_layout)

        if PYQTGRAPH_AVAILABLE:
            pg.setConfigOptions(antialias=True)

            # Create container for plots (will switch between single plot and GraphicsLayoutWidget)
            self.plot_container = QVBoxLayout()
            
            # Create single plot widget (for overlay mode)
            self.plot_widget = pg.PlotWidget()
            self.plot_widget.setBackground(GUIDefaults.PLOT_BACKGROUND)
            plot_item = self.plot_widget.getPlotItem()
            self._apply_plot_style(plot_item, show_bottom_axis=True)
            plot_item.setLabel('left', 'Acceleration', units=self.channel_units, color=GUIDefaults.PLOT_AXIS_COLOR)
            plot_item.setLabel('bottom', 'Time', units='s', color=GUIDefaults.PLOT_AXIS_COLOR)
            self.plot_widget.addLegend()

            self.plot_container.addWidget(self.plot_widget)
            layout.addLayout(self.plot_container)
        else:
            # Placeholder if PyQtGraph not available
            placeholder = QLabel("PyQtGraph not installed.\nInstall with: pip install pyqtgraph")
            placeholder.setAlignment(Qt.AlignCenter)
            placeholder.setStyleSheet("QLabel { color: red; font-size: 14px; }")
            layout.addWidget(placeholder)

    def _apply_plot_style(self, plot_item: 'pg.PlotItem', show_bottom_axis: bool):
        plot_item.showGrid(x=True, y=True, alpha=GUIDefaults.PLOT_GRID_ALPHA)
        plot_item.getViewBox().setBorder(
            pg.mkPen(GUIDefaults.PLOT_FRAME_COLOR, width=GUIDefaults.PLOT_FRAME_WIDTH)
        )
        axis_pen = pg.mkPen(GUIDefaults.PLOT_AXIS_COLOR, width=GUIDefaults.PLOT_AXIS_WIDTH)
        for axis_name in ("left", "bottom"):
            axis = plot_item.getAxis(axis_name)
            axis.setPen(axis_pen)
            axis.setTextPen(GUIDefaults.PLOT_AXIS_COLOR)
        bottom_axis = plot_item.getAxis("bottom")
        bottom_axis.setStyle(showValues=show_bottom_axis)
        plot_item.showAxis('bottom')

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

        # Create plot curves
        if PYQTGRAPH_AVAILABLE:
            self._create_curves()

        # Update plot labels
        if PYQTGRAPH_AVAILABLE:
            self.plot_widget.setLabel(
                'left',
                'Acceleration',
                units=channel_units,
                color=GUIDefaults.PLOT_AXIS_COLOR
            )

        self.logger.info(
            f"Plot configured: {n_channels} channels @ {sample_rate} Hz, units={channel_units}"
        )

    def _create_curves(self):
        """Create plot curves for each channel."""
        if not PYQTGRAPH_AVAILABLE:
            return

        if self.display_mode == "overlay":
            self._create_overlay_plots()
        else:  # stack mode
            self._create_stack_plots()
    
    def _create_overlay_plots(self):
        """Create single plot with all channels overlaid."""
        # Clear existing curves
        self.plot_widget.clear()
        self.curves = []

        # Create a curve for each channel with different colors
        colors = GUIDefaults.PLOT_COLORS

        for i in range(self.n_channels):
            color = colors[i % len(colors)]
            pen = pg.mkPen(color=color, width=GUIDefaults.PLOT_LINE_WIDTH)

            curve = self.plot_widget.plot(
                pen=pen,
                name=self.channel_names[i]
            )

            # Enable downsampling for performance
            curve.setDownsampling(auto=True)
            curve.setClipToView(True)

            self.curves.append(curve)

        self.logger.debug(f"Created {len(self.curves)} overlay curves")
    
    def _create_stack_plots(self):
        """Create separate subplot for each channel."""
        self.curves = []
        self.plot_widgets = []
        
        # Clear single plot widget
        self.plot_widget.clear()
        
        # Create GraphicsLayoutWidget for multiple subplots
        if self.graphics_layout is None:
            self.graphics_layout = pg.GraphicsLayoutWidget()
            self.graphics_layout.setBackground(GUIDefaults.PLOT_BACKGROUND)
        else:
            self.graphics_layout.clear()
        
        colors = GUIDefaults.PLOT_COLORS
        
        first_plot = None
        for i in range(self.n_channels):
            # Create subplot
            plot = self.graphics_layout.addPlot(row=i, col=0)
            self._apply_plot_style(plot, show_bottom_axis=(i == self.n_channels - 1))
            plot.setLabel(
                'left',
                self.channel_names[i],
                units=self.channel_units,
                color=GUIDefaults.PLOT_AXIS_COLOR
            )
            
            # Only show x-axis label on bottom plot
            if i == self.n_channels - 1:
                plot.setLabel('bottom', 'Time', units='s', color=GUIDefaults.PLOT_AXIS_COLOR)

            if first_plot is None:
                first_plot = plot
            else:
                plot.setXLink(first_plot)
            
            # Create curve for this channel
            color = colors[i % len(colors)]
            pen = pg.mkPen(color=color, width=GUIDefaults.PLOT_LINE_WIDTH)
            curve = plot.plot(pen=pen)
            curve.setDownsampling(auto=True)
            curve.setClipToView(True)
            
            self.curves.append(curve)
            self.plot_widgets.append(plot)
        
        self.logger.debug(f"Created {len(self.curves)} stacked subplots")
    
    def _rebuild_plots(self):
        """Rebuild plots when switching between overlay and stack modes."""
        if not PYQTGRAPH_AVAILABLE:
            return
        
        # Clear container
        while self.plot_container.count():
            item = self.plot_container.takeAt(0)
            if item.widget():
                item.widget().setParent(None)
        
        if self.display_mode == "overlay":
            # Show single plot widget
            self.plot_container.addWidget(self.plot_widget)
            self._create_overlay_plots()
        else:  # stack
            # Show graphics layout with subplots
            if self.graphics_layout is None:
                self.graphics_layout = pg.GraphicsLayoutWidget()
                self.graphics_layout.setBackground(GUIDefaults.PLOT_BACKGROUND)
            self.plot_container.addWidget(self.graphics_layout)
            self._create_stack_plots()
        
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
        if self._pending_data is None or not PYQTGRAPH_AVAILABLE:
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
        for i, curve in enumerate(self.curves):
            if i < len(self.channel_visible) and self.channel_visible[i]:
                # Downsample if needed
                if n_samples > self.downsample_threshold:
                    step = n_samples // self.downsample_threshold
                    time_ds = time_axis[::step]
                    data_ds = data[i, ::step]
                    curve.setData(time_ds, data_ds)
                else:
                    curve.setData(time_axis, data[i, :])
                curve.show()
            else:
                curve.hide()
        
        # Auto-scale if enabled
        if self.auto_scale:
            self.plot_widget.enableAutoRange()
    
    def _update_stack_plots(self, time_axis: np.ndarray, data: np.ndarray, n_samples: int):
        """Update plots in stack mode (separate subplot for each channel)."""
        for i, (curve, plot_widget) in enumerate(zip(self.curves, self.plot_widgets)):
            if i < len(self.channel_visible) and self.channel_visible[i]:
                # Downsample if needed
                if n_samples > self.downsample_threshold:
                    step = n_samples // self.downsample_threshold
                    time_ds = time_axis[::step]
                    data_ds = data[i, ::step]
                    curve.setData(time_ds, data_ds)
                else:
                    curve.setData(time_axis, data[i, :])
                curve.show()
                
                # Auto-scale individual subplot if enabled
                if self.auto_scale:
                    plot_widget.enableAutoRange()
            else:
                curve.hide()

    def _on_window_changed(self, index: int):
        """Handle time window change."""
        self.time_window = self.window_combo.currentData()
        self.logger.debug(f"Time window changed to {self.time_window}s")

    def _on_autoscale_changed(self, state: int):
        """Handle auto-scale checkbox change."""
        self.auto_scale = (state == Qt.Checked)

        if not PYQTGRAPH_AVAILABLE:
            return

        if self.auto_scale:
            self.plot_widget.enableAutoRange()
        else:
            self.plot_widget.disableAutoRange()

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
        if not PYQTGRAPH_AVAILABLE:
            return

        for curve in self.curves:
            curve.clear()

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

            if PYQTGRAPH_AVAILABLE and channel_idx < len(self.curves):
                if visible:
                    self.curves[channel_idx].show()
                else:
                    self.curves[channel_idx].hide()

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

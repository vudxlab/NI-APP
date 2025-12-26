"""
Real-time Plot Widget for time-domain visualization.

This widget provides high-performance real-time plotting of multi-channel
acceleration data using Plotly.
"""

import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QComboBox,
    QCheckBox, QPushButton, QLabel, QSpinBox, QMenu, QAction, QScrollArea
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

        # Plot view (Plotly) - support both overlay and stack modes
        self.plot_view_single: Optional['PlotlyView'] = None  # For overlay mode
        self.plot_views_stack: List['PlotlyView'] = []  # For stack mode (1 per channel)
        self.plot_container_stack: Optional[QWidget] = None  # Container for stack plots
        self.scroll_area: Optional[QScrollArea] = None

        # Stack plot settings
        self.max_visible_channels = 4  # Max channels to show without scrolling
        self.subplot_height = 200  # Fixed height per individual plot in pixels
        self.min_subplot_height = 160  # Minimum height for stack plots
        self._last_layout: Optional[Dict] = None  # Cache last layout (deprecated for stack mode)
        self._current_plot_height: Optional[int] = None  # Track current height (deprecated for stack mode)
        self._layout_initialized: bool = False  # Track if layout is set (deprecated for stack mode)

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

        # Reset cached plot data
        self.time_data = None
        self.plot_data = None

        # Reset layout cache (deprecated for stack mode)
        self._last_layout = None
        self._layout_initialized = False
        self._current_plot_height = None

        # Rebuild stack container if in stack mode
        if PLOTLY_AVAILABLE:
            if self.display_mode == "stack":
                # Recreate stack container with new number of channels
                self._create_stack_plot_container()
            else:
                # Just trigger a redraw for overlay mode
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
        if PLOTLY_AVAILABLE and self.plot_view_single is not None:
            print(f"RealtimePlotWidget._init_empty_plot: Calling update_plot, view size={self.plot_view_single.size()}")
            self.plot_view_single.update_plot(
                [],
                self._base_layout(
                    y_title=f"Acceleration ({self.channel_units})",
                    x_title="Time (s)",
                    show_legend=True
                )
            )
            print("RealtimePlotWidget._init_empty_plot: update_plot called")
        else:
            print(f"RealtimePlotWidget._init_empty_plot: Skipped, PLOTLY_AVAILABLE={PLOTLY_AVAILABLE}, plot_view_single={self.plot_view_single}")
    
    def _rebuild_plots(self):
        """Rebuild plots when switching between overlay and stack modes."""
        if not PLOTLY_AVAILABLE:
            return
        if self._pending_data is not None:
            self._update_plots()
            return

        if self.plot_data is not None:
            data = self.plot_data
            n_samples = data.shape[1]
            if self.time_data is not None and len(self.time_data) == n_samples:
                time_axis = self.time_data
            else:
                time_axis = np.arange(n_samples) / self.sample_rate
                time_axis = time_axis - time_axis[-1]

            if self.display_mode == "overlay":
                self._update_overlay_plots(time_axis, data, n_samples)
            else:
                self._update_stack_plots(time_axis, data, n_samples)

        self.logger.info(f"Rebuilt plots in {self.display_mode} mode")

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
                lambda view=plot_view, name=channel_name: self._init_stack_plot_view(view, name)
            )

        # Add stretch at the end to push plots to top
        container_layout.addStretch()

        # Set container to scroll area
        self.scroll_area.setWidget(self.plot_container_stack)
        self._update_stack_plot_heights()

        self.logger.info(f"Created stack plot container with {self.n_channels} PlotlyView widgets")

    def _destroy_stack_plot_container(self):
        """Destroy stack plot container and clean up PlotlyView widgets."""
        # Stop updates first to prevent crashes
        was_active = self.update_timer.isActive()
        if was_active:
            self.update_timer.stop()

        if self.plot_views_stack:
            # Clear and delete all PlotlyView widgets
            for plot_view in self.plot_views_stack:
                try:
                    plot_view.deleteLater()
                except Exception as e:
                    self.logger.warning(f"Error deleting PlotlyView: {e}")
            self.plot_views_stack = []
            self.logger.debug("Destroyed all stack PlotlyView widgets")

        if self.plot_container_stack is not None:
            try:
                self.plot_container_stack.deleteLater()
            except Exception as e:
                self.logger.warning(f"Error deleting container: {e}")
            self.plot_container_stack = None
            self.logger.debug("Destroyed stack plot container")

        # Restart timer if it was active
        if was_active:
            self.update_timer.start()

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
        self.plot_data = data

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

            # Cache data for redraws (mode switch, refresh)
            self.plot_data = data
            self.time_data = time_axis

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
        if not PLOTLY_AVAILABLE or self.plot_view_single is None:
            return

        # Reset plot view size for overlay mode
        if self.plot_view_single:
            self.plot_view_single.setMinimumHeight(300)
            self.plot_view_single.setMaximumHeight(16777215)  # Qt default max

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
        self.plot_view_single.update_plot(traces, layout)
    
    def _update_stack_plots(self, time_axis: np.ndarray, data: np.ndarray, n_samples: int):
        """Update plots in stack mode (separate PlotlyView for each channel)."""
        if not PLOTLY_AVAILABLE:
            return

        # If stack views not created yet, skip this update
        if not self.plot_views_stack:
            self.logger.debug("Stack plot views not ready yet, skipping update")
            return

        # Ensure we have the correct number of PlotlyView widgets
        if len(self.plot_views_stack) != self.n_channels:
            self.logger.warning(f"Mismatch: {len(self.plot_views_stack)} views vs {self.n_channels} channels.")
            # Don't recreate during update - just skip this update
            # Container will be recreated by configure() or mode change
            return

        # Update each PlotlyView individually
        colors = GUIDefaults.PLOT_COLORS
        for i in range(self.n_channels):
            # Skip hidden channels
            if i < len(self.channel_visible) and not self.channel_visible[i]:
                # Clear the plot for hidden channels
                layout = self._base_layout(
                    y_title=f"{self.channel_units}",
                    x_title="Time (s)",
                    show_legend=False
                )
                channel_name = self.channel_names[i] if i < len(self.channel_names) else f"Ch{i+1}"
                layout["title"] = {
                    "text": f"<b>{channel_name}</b>",
                    "font": {"size": 14, "color": "#111"},
                    "x": 0.01,
                    "xanchor": "left"
                }
                self.plot_views_stack[i].update_plot([], layout)
                continue

            if i >= data.shape[0]:
                continue

            # Downsample if needed
            if n_samples > self.downsample_threshold:
                step = max(1, n_samples // self.downsample_threshold)
                x_vals = time_axis[::step]
                y_vals = data[i, ::step]
            else:
                x_vals = time_axis
                y_vals = data[i, :]

            # Create single trace for this channel
            trace = go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="lines",
                name=self.channel_names[i] if i < len(self.channel_names) else f"Ch{i+1}",
                line={"color": colors[i % len(colors)], "width": GUIDefaults.PLOT_LINE_WIDTH}
            )

            # Create layout for this individual plot
            # Show x-axis label on all plots for better clarity
            layout = {
                "autosize": True,
                "margin": {"l": 60, "r": 20, "t": 30, "b": 40},
                "plot_bgcolor": self._plotly_background(),
                "paper_bgcolor": self._plotly_background(),
                "showlegend": False,
                "title": {
                    "text": f"<b>{self.channel_names[i] if i < len(self.channel_names) else f'Ch{i+1}'}</b>",
                    "font": {"size": 14, "color": "#111"},
                    "x": 0.01,
                    "xanchor": "left"
                },
                "xaxis": {
                    "title": "Time (s)",  # Show on all plots
                    "showgrid": True,
                    "gridcolor": "#e0e0e0",
                    "zeroline": False,
                    "automargin": True,
                    "showticklabels": True  # Always show tick labels
                },
                "yaxis": {
                    "title": f"{self.channel_units}",
                    "showgrid": True,
                    "gridcolor": "#e0e0e0",
                    "zeroline": False,
                    "automargin": True
                },
                "uirevision": f"channel_{i}"  # Keep zoom/pan state per channel
            }

            # Update this specific PlotlyView
            self.plot_views_stack[i].update_plot([trace], layout)

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
        self.logger.debug(f"Display mode changing from {old_mode} to {self.display_mode}")

        # Switch between overlay and stack modes
        if old_mode != self.display_mode and self.n_channels > 0:
            # Stop timer during mode switch to prevent crashes
            was_active = self.update_timer.isActive()
            if was_active:
                self.update_timer.stop()

            try:
                if self.display_mode == "stack":
                    # Switching to stack mode: create individual plot views
                    self.logger.info("Switching to stack mode: creating individual PlotlyView widgets")

                    # Destroy single plot view if it exists
                    if self.plot_view_single is not None:
                        self.plot_view_single.deleteLater()
                        self.plot_view_single = None

                    # Create stack plot container
                    self._create_stack_plot_container()
                    self._update_stack_plot_heights()

                else:  # overlay mode
                    # Switching to overlay mode: use single plot view
                    self.logger.info("Switching to overlay mode: creating single PlotlyView widget")

                    # Destroy stack plot container (timer already stopped above)
                    self._destroy_stack_plot_container()

                    # Create single plot view
                    self.plot_view_single = PlotlyView()
                    self.plot_view_single.setMinimumHeight(300)
                    self.scroll_area.setWidget(self.plot_view_single)

                    # Initialize with empty plot
                    QTimer.singleShot(500, self._init_empty_plot)

                # Trigger redraw with pending data if available
                self._rebuild_plots()

            finally:
                # Restart timer if it was active
                if was_active:
                    self.update_timer.start()

        self.logger.debug(f"Display mode changed to {self.display_mode}")

    def _on_clear(self):
        """Clear the plot."""
        if not PLOTLY_AVAILABLE:
            return

        if self.display_mode == "overlay" and self.plot_view_single is not None:
            # Clear single overlay plot
            self.plot_view_single.update_plot([], self._base_layout(
                y_title=f"Acceleration ({self.channel_units})",
                x_title="Time (s)",
                show_legend=True
            ))
        elif self.display_mode == "stack" and self.plot_views_stack:
            # Clear all stack plots
            for i, plot_view in enumerate(self.plot_views_stack):
                channel_name = self.channel_names[i] if i < len(self.channel_names) else f"Ch{i+1}"
                layout = self._base_layout(
                    y_title=f"{self.channel_units}",
                    x_title="Time (s)",
                    show_legend=False
                )
                layout["title"] = {
                    "text": f"<b>{channel_name}</b>",
                    "font": {"size": 14, "color": "#111"},
                    "x": 0.01,
                    "xanchor": "left"
                }
                plot_view.update_plot([], layout)

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

    def _init_stack_plot_view(self, view: 'PlotlyView', channel_name: str):
        """Initialize a stack plot view with empty content if no data is ready."""
        if self.plot_data is not None or self._pending_data is not None:
            return
        layout = self._base_layout(
            y_title=f"{self.channel_units}",
            x_title="Time (s)",
            show_legend=False
        )
        layout["title"] = {
            "text": f"<b>{channel_name}</b>",
            "font": {"size": 14, "color": "#111"},
            "x": 0.01,
            "xanchor": "left"
        }
        view.update_plot([], layout)

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

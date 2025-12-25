from unittest.mock import Mock, MagicMock
from pathlib import Path
"""
GUI tests for NI DAQ application widgets.

Tests GUI components using pytest-qt.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock


from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QApplication
from PyQt5.QtTest import QTest

# Import widgets
try:
    from src.gui.widgets.daq_config_panel import DAQConfigPanel
    from src.gui.widgets.channel_config_widget import ChannelConfigWidget
    from src.gui.widgets.filter_config_panel import FilterConfigPanel
    from src.gui.widgets.realtime_plot_widget import RealtimePlotWidget
    from src.gui.widgets.fft_plot_widget import FFTPlotWidget
    from src.gui.widgets.export_panel import ExportPanel
    WIDGETS_AVAILABLE = True
except ImportError:
    WIDGETS_AVAILABLE = False

from src.daq.daq_config import ChannelConfig, DAQConfig


@pytest.mark.skipif(not WIDGETS_AVAILABLE, reason="GUI widgets not available")
class TestDAQConfigPanel:
    """Test DAQConfigPanel widget."""

    @pytest.fixture
    def panel(self, qtbot):
        widget = DAQConfigPanel()
        qtbot.addWidget(widget)
        widget.show()
        return widget

    def test_panel_initialization(self, panel):
        """Test panel is initialized correctly."""
        assert panel.device_combo is not None
        assert panel.sample_rate_combo is not None
        assert panel.start_button is not None
        assert panel.stop_button is not None

    def test_device_selection(self, panel, qtbot):
        """Test device selection combo box."""
        # Add test devices
        panel.update_devices(["cDAQ1", "cDAQ2", "Dev1"])

        assert panel.device_combo.count() >= 3

        # Select device
        panel.device_combo.setCurrentText("cDAQ1")
        assert panel.device_combo.currentText() == "cDAQ1"

    def test_sample_rate_selection(self, panel):
        """Test sample rate selection."""
        rates = panel.get_available_sample_rates()
        assert len(rates) > 0
        assert 51200.0 in rates

        panel.set_sample_rate(25600.0)
        assert panel.get_sample_rate() == 25600.0

    def test_start_stop_buttons(self, panel):
        """Test start and stop buttons."""
        # Initially, start should be enabled, stop disabled
        assert panel.start_button.isEnabled()
        assert not panel.stop_button.isEnabled()

        # Simulate acquisition started
        panel.set_acquisition_running(True)

        assert not panel.start_button.isEnabled()
        assert panel.stop_button.isEnabled()

        # Simulate acquisition stopped
        panel.set_acquisition_running(False)

        assert panel.start_button.isEnabled()
        assert not panel.stop_button.isEnabled()

    def test_signal_emission(self, panel, qtbot):
        """Test that signals are emitted correctly."""
        start_clicked = []
        stop_clicked = []

        panel.start_requested.connect(lambda: start_clicked.append(True))
        panel.stop_requested.connect(lambda: stop_clicked.append(True))

        QTest.mouseClick(panel.start_button, Qt.LeftButton)
        QTest.mouseClick(panel.stop_button, Qt.LeftButton)

        assert len(start_clicked) == 1
        assert len(stop_clicked) == 1


@pytest.mark.skipif(not WIDGETS_AVAILABLE, reason="GUI widgets not available")
class TestChannelConfigWidget:
    """Test ChannelConfigWidget widget."""

    @pytest.fixture
    def widget(self, qtbot):
        w = ChannelConfigWidget()
        qtbot.addWidget(w)
        w.show()
        return w

    @pytest.fixture
    def sample_channels(self):
        """Create sample channels."""
        return [
            ChannelConfig(f"cDAQ1Mod0/ai{i}", name=f"Channel {i}")
            for i in range(4)
        ]

    def test_widget_initialization(self, widget):
        """Test widget is initialized correctly."""
        assert widget.table.rowCount() == 0

    def test_load_channels(self, widget, sample_channels):
        """Test loading channels into widget."""
        widget.load_channels(sample_channels)

        assert widget.table.rowCount() == 4

    def test_enable_disable_all(self, widget, sample_channels):
        """Test enable/disable all buttons."""
        widget.load_channels(sample_channels)

        # Click disable all
        QTest.mouseClick(widget.disable_all_button, Qt.LeftButton)

        # All should be disabled
        for row in range(widget.table.rowCount()):
            checkbox = widget.table.cellWidget(row, 0)
            assert not checkbox.isChecked()

        # Click enable all
        QTest.mouseClick(widget.enable_all_button, Qt.LeftButton)

        # All should be enabled
        for row in range(widget.table.rowCount()):
            checkbox = widget.table.cellWidget(row, 0)
            assert checkbox.isChecked()

    def test_get_channel_configs(self, widget, sample_channels):
        """Test getting channel configurations."""
        widget.load_channels(sample_channels)

        configs = widget.get_channel_configs()

        assert len(configs) == 4
        assert all(isinstance(c, ChannelConfig) for c in configs)


@pytest.mark.skipif(not WIDGETS_AVAILABLE, reason="GUI widgets not available")
class TestFilterConfigPanel:
    """Test FilterConfigPanel widget."""

    @pytest.fixture
    def panel(self, qtbot):
        widget = FilterConfigPanel()
        qtbot.addWidget(widget)
        widget.show()
        return widget

    def test_panel_initialization(self, panel):
        """Test panel is initialized correctly."""
        assert panel.enabled_checkbox is not None
        assert panel.type_combo is not None
        assert panel.mode_combo is not None
        assert panel.cutoff_spin is not None

    def test_filter_types(self, panel):
        """Test available filter types."""
        types = []
        for i in range(panel.type_combo.count()):
            types.append(panel.type_combo.itemText(i))

        assert "Butterworth" in types
        assert "Chebyshev" in types or "Chebyshev Type I" in types
        assert "Bessel" in types

    def test_filter_modes(self, panel):
        """Test available filter modes."""
        modes = []
        for i in range(panel.mode_combo.count()):
            modes.append(panel.mode_combo.itemText(i))

        assert "Lowpass" in modes or "Low Pass" in modes
        assert "Highpass" in modes or "High Pass" in modes

    def test_cutoff_frequency(self, panel):
        """Test cutoff frequency setting."""
        panel.set_cutoff_frequency(1000.0)
        assert panel.get_cutoff_frequency() == 1000.0

    def test_filter_order(self, panel):
        """Test filter order setting."""
        panel.set_filter_order(6)
        assert panel.get_filter_order() == 6

    def test_enable_filter(self, panel, qtbot):
        """Test enabling/disabling filter."""
        assert not panel.is_enabled()

        QTest.mouseClick(panel.enabled_checkbox, Qt.LeftButton)
        assert panel.is_enabled()

        QTest.mouseClick(panel.enabled_checkbox, Qt.LeftButton)
        assert not panel.is_enabled()

    def test_get_filter_config(self, panel):
        """Test getting filter configuration."""
        config = panel.get_filter_config()

        assert 'type' in config
        assert 'mode' in config
        assert 'cutoff' in config
        assert 'order' in config


@pytest.mark.skipif(not WIDGETS_AVAILABLE, reason="GUI widgets not available")
class TestRealtimePlotWidget:
    """Test RealtimePlotWidget widget."""

    @pytest.fixture
    def widget(self, qtbot):
        w = RealtimePlotWidget()
        qtbot.addWidget(w)
        w.show()
        return widget

    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        return np.random.randn(4, 1000)

    def test_widget_initialization(self, widget):
        """Test widget is initialized correctly."""
        assert widget.plot_widget is not None

    def test_configure_plot(self, widget):
        """Test configuring the plot."""
        widget.configure(
            n_channels=4,
            sample_rate=51200.0,
            channel_names=['CH1', 'CH2', 'CH3', 'CH4'],
            channel_units=['g', 'g', 'g', 'g']
        )

        # Should create curves
        assert widget.n_channels == 4

    def test_update_plot(self, widget, sample_data, qtbot):
        """Test updating the plot with data."""
        widget.configure(
            n_channels=4,
            sample_rate=51200.0,
            channel_names=['CH1', 'CH2', 'CH3', 'CH4'],
            channel_units=['g', 'g', 'g', 'g']
        )

        # Update plot
        widget.update_plot(sample_data, timestamp=0.0)

        # Wait for update
        qtbot.wait(100)

        assert True  # If no exception, test passes

    def test_autoscale(self, widget, sample_data):
        """Test autoscale functionality."""
        widget.configure(
            n_channels=4,
            sample_rate=51200.0,
            channel_names=['CH1', 'CH2', 'CH3', 'CH4'],
            channel_units=['g', 'g', 'g', 'g']
        )

        widget.set_autoscale(True)
        assert widget.get_autoscale() is True

        widget.set_autoscale(False)
        assert widget.get_autoscale() is False

    def test_time_window(self, widget):
        """Test time window setting."""
        widget.set_time_window(5.0)
        assert widget.get_time_window() == 5.0

        widget.set_time_window(10.0)
        assert widget.get_time_window() == 10.0

    def test_clear_plot(self, widget, sample_data, qtbot):
        """Test clearing the plot."""
        widget.configure(
            n_channels=4,
            sample_rate=51200.0,
            channel_names=['CH1', 'CH2', 'CH3', 'CH4'],
            channel_units=['g', 'g', 'g', 'g']
        )

        widget.update_plot(sample_data, timestamp=0.0)
        qtbot.wait(100)

        widget.clear()
        qtbot.wait(100)

        # Plot should be cleared
        assert True


@pytest.mark.skipif(not WIDGETS_AVAILABLE, reason="GUI widgets not available")
class TestFFTPlotWidget:
    """Test FFTPlotWidget widget."""

    @pytest.fixture
    def widget(self, qtbot):
        w = FFTPlotWidget()
        qtbot.addWidget(w)
        w.show()
        return widget

    @pytest.fixture
    def sample_fft_data(self):
        """Create sample FFT data."""
        freqs = np.linspace(0, 25600, 1025)
        magnitude = np.random.rand(1025) * 10
        return freqs, magnitude

    def test_widget_initialization(self, widget):
        """Test widget is initialized correctly."""
        assert widget.plot_widget is not None

    def test_configure_plot(self, widget):
        """Test configuring the plot."""
        widget.configure(
            n_channels=4,
            sample_rate=51200.0,
            channel_names=['CH1', 'CH2', 'CH3', 'CH4']
        )

        assert widget.n_channels == 4

    def test_update_plot(self, widget, sample_fft_data, qtbot):
        """Test updating the plot with FFT data."""
        widget.configure(
            n_channels=4,
            sample_rate=51200.0,
            channel_names=['CH1', 'CH2', 'CH3', 'CH4']
        )

        freqs, magnitude = sample_fft_data
        widget.update_plot(freqs, magnitude, channel=0)

        qtbot.wait(100)
        assert True

    def test_scale_mode(self, widget):
        """Test scale mode switching."""
        widget.configure(
            n_channels=4,
            sample_rate=51200.0,
            channel_names=['CH1', 'CH2', 'CH3', 'CH4']
        )

        widget.set_scale('dB')
        assert widget.get_scale() == 'dB'

        widget.set_scale('linear')
        assert widget.get_scale() == 'linear'

    def test_peak_detection(self, widget):
        """Test peak detection display."""
        widget.configure(
            n_channels=4,
            sample_rate=51200.0,
            channel_names=['CH1', 'CH2', 'CH3', 'CH4']
        )

        widget.set_show_peaks(True)
        assert widget.get_show_peaks() is True

        widget.set_peak_threshold(0.2)
        assert widget.get_peak_threshold() == 0.2


@pytest.mark.skipif(not WIDGETS_AVAILABLE, reason="GUI widgets not available")
class TestExportPanel:
    """Test ExportPanel widget."""

    @pytest.fixture
    def panel(self, qtbot):
        widget = ExportPanel()
        qtbot.addWidget(widget)
        widget.show()
        return widget

    def test_panel_initialization(self, panel):
        """Test panel is initialized correctly."""
        assert panel.path_edit is not None
        assert panel.browse_button is not None
        assert panel.export_button is not None

    def test_format_selection(self, panel):
        """Test format selection."""
        # Test each format
        formats = ['csv', 'hdf5', 'tdms']

        for fmt in formats:
            panel.set_format(fmt)
            assert panel.get_format() == fmt

    def test_file_path(self, panel):
        """Test file path setting."""
        test_path = "/tmp/test_data.csv"

        panel.set_file_path(test_path)
        assert panel.get_file_path() == test_path

    def test_export_button_state(self, panel):
        """Test export button state."""
        # Initially should be disabled (no data)
        assert not panel.export_button.isEnabled()

        # Set data available
        panel.set_data_available(True)
        assert panel.export_button.isEnabled()

    def test_progress_update(self, panel):
        """Test progress bar update."""
        panel.set_progress(50)
        assert panel.progress_bar.value() == 50

        panel.set_progress(100)
        assert panel.progress_bar.value() == 100


@pytest.mark.skipif(not WIDGETS_AVAILABLE, reason="GUI widgets not available")
class TestWidgetIntegration:
    """Test widget interactions."""

    def test_daq_panel_to_channel_config(self, qtbot):
        """Test interaction between DAQ panel and channel config."""
        daq_panel = DAQConfigPanel()
        channel_widget = ChannelConfigWidget()

        qtbot.addWidget(daq_panel)
        qtbot.addWidget(channel_widget)
        daq_panel.show()
        channel_widget.show()

        # Set device
        daq_panel.update_devices(["cDAQ1"])
        daq_panel.device_combo.setCurrentText("cDAQ1")

        # Should be able to configure channels
        channels = [
            ChannelConfig("cDAQ1Mod0/ai0", name="CH0"),
            ChannelConfig("cDAQ1Mod0/ai1", name="CH1")
        ]
        channel_widget.load_channels(channels)

        assert channel_widget.table.rowCount() == 2


@pytest.mark.skipif(not WIDGETS_AVAILABLE, reason="GUI widgets not available")
class TestWidgetThreading:
    """Test widget behavior with threading."""

    def test_plot_updates_from_timer(self, qtbot):
        """Test that plot updates work with timer-based updates."""
        widget = RealtimePlotWidget()
        qtbot.addWidget(widget)
        widget.show()

        widget.configure(
            n_channels=4,
            sample_rate=51200.0,
            channel_names=['CH1', 'CH2', 'CH3', 'CH4'],
            channel_units=['g', 'g', 'g', 'g']
        )

        # Simulate multiple updates
        for i in range(10):
            data = np.random.randn(4, 100)
            widget.update_plot(data, timestamp=i * 0.01)

        qtbot.wait(500)
        assert True


class TestWidgetStyling:
    """Test widget styling and appearance."""

    @pytest.mark.skipif(not WIDGETS_AVAILABLE, reason="GUI widgets not available")
    def test_dark_mode_support(self, qtbot):
        """Test dark mode styling."""
        widget = FilterConfigPanel()
        qtbot.addWidget(widget)
        widget.show()

        # Apply dark theme
        widget.setProperty("darkMode", True)
        widget.style().polish(widget)

        assert True  # Verify no crash

    @pytest.mark.skipif(not WIDGETS_AVAILABLE, reason="GUI widgets not available")
    def test_widget_size_constraints(self, qtbot):
        """Test widget size constraints."""
        widget = DAQConfigPanel()
        qtbot.addWidget(widget)
        widget.show()

        # Should have reasonable size
        assert widget.width() > 0
        assert widget.height() > 0

        # Test minimum size
        min_size = widget.minimumSizeHint()
        assert min_size.width() > 0
        assert min_size.height() > 0

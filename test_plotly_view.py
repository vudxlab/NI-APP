#!/usr/bin/env python3
"""
Simple test script to verify PlotlyView fixes.
This script creates a minimal window with both time-domain and frequency-domain plots.

Run this script to test if the Plotly plots are working correctly:
    source venv/bin/activate
    python test_plotly_view.py

Expected output:
- Window should open with 2 tabs
- After ~500ms, both tabs should show empty plot grids with axes
- After ~1 second, time-domain plot should show sine waves
- Console should show debug messages from PlotlyView
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout
from PyQt5.QtCore import Qt, QCoreApplication, QTimer
import numpy as np

# Set QtWebEngine attribute before QApplication
QCoreApplication.setAttribute(Qt.AA_ShareOpenGLContexts, True)

print("=" * 80)
print("PlotlyView Test Script")
print("=" * 80)
print()

from src.gui.widgets.realtime_plot_widget import RealtimePlotWidget
from src.gui.widgets.fft_plot_widget import FFTPlotWidget


class TestWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Plotly View Test")
        self.resize(1200, 800)

        # Create central widget with tabs
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Create tab widget
        tabs = QTabWidget()

        # Time domain tab
        self.realtime_widget = RealtimePlotWidget()
        self.realtime_widget.configure(
            n_channels=4,
            sample_rate=25600,
            channel_names=["Ch1", "Ch2", "Ch3", "Ch4"],
            channel_units="g"
        )
        tabs.addTab(self.realtime_widget, "Time Domain")

        # Frequency domain tab
        self.fft_widget = FFTPlotWidget()
        self.fft_widget.configure(
            n_channels=4,
            sample_rate=25600,
            channel_names=["Ch1", "Ch2", "Ch3", "Ch4"],
            channel_units="g"
        )
        tabs.addTab(self.fft_widget, "Frequency Domain")

        layout.addWidget(tabs)

        # Timer to send test data after widget is shown
        QTimer.singleShot(500, self.send_test_data)

    def send_test_data(self):
        """Send test data to plots after a delay."""
        print()
        print("=" * 80)
        print("SENDING TEST DATA TO PLOTS")
        print("=" * 80)

        # Generate test time-domain data
        n_samples = 1000
        t = np.arange(n_samples) / 25600
        data = np.zeros((4, n_samples))
        for i in range(4):
            freq = 100 * (i + 1)
            data[i, :] = np.sin(2 * np.pi * freq * t) + 0.1 * np.random.randn(n_samples)

        # Send to time-domain plot
        print(f"Sending data to realtime_widget: shape={data.shape}")
        self.realtime_widget.update_plot(data, 0.0)
        print("Time-domain data sent")
        print("=" * 80)
        print()


def main():
    print("Creating QApplication...")
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    print("Creating test window...")
    window = TestWindow()

    print("Showing window...")
    window.show()

    print()
    print("=" * 80)
    print("TEST WINDOW DISPLAYED")
    print("=" * 80)
    print("Watch for debug messages below...")
    print("Expected: PlotlyView init messages, HTML loading, then plot updates")
    print("=" * 80)
    print()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

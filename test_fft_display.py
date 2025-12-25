#!/usr/bin/env python3
"""
Test FFT display to debug why frequency domain is not showing data.
"""

import sys
import numpy as np
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer
import logging

# Set debug logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')

from src.gui.widgets.fft_plot_widget import FFTPlotWidget

def main():
    app = QApplication(sys.argv)
    
    # Create FFT widget
    print("\n1. Creating FFT widget...")
    widget = FFTPlotWidget()
    
    # Configure for 4 channels
    print("\n2. Configuring FFT widget...")
    widget.configure(
        n_channels=4,
        sample_rate=25600,
        channel_names=["Ch1", "Ch2", "Ch3", "Ch4"],
        channel_units="g"
    )
    
    # Start widget
    print("\n3. Starting FFT widget...")
    widget.start()
    
    # Generate test FFT data
    print("\n4. Generating test FFT data...")
    sample_rate = 25600
    n_samples = 2048
    frequencies = np.fft.rfftfreq(n_samples, 1/sample_rate)
    
    # Create spectrum with peaks
    magnitude = np.random.rand(len(frequencies)) * 0.01  # Noise floor
    
    # Add peaks at specific frequencies
    for freq in [100, 500, 1000, 2000, 5000]:
        idx = np.argmin(np.abs(frequencies - freq))
        magnitude[idx] = 1.0
    
    # Convert to dB
    magnitude_db = 20 * np.log10(magnitude + 1e-10)
    
    print(f"   Frequency range: {frequencies[0]:.1f} - {frequencies[-1]:.1f} Hz")
    print(f"   Magnitude range: {magnitude_db.min():.1f} - {magnitude_db.max():.1f} dB")
    
    # Send FFT data to widget
    print("\n5. Sending FFT data to widget...")
    for ch in range(4):
        widget.update_plot(frequencies, magnitude_db, ch)
        print(f"   Sent data for channel {ch}")
    
    # Show widget
    print("\n6. Showing widget...")
    widget.show()
    
    print("\n✅ If you see FFT plots, it works!")
    print("❌ If plots are empty, there's a bug in the widget")
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()



# NI DAQ Vibration Analysis GUI

A professional Python GUI application for real-time acceleration data acquisition and analysis using NI-9178 DAQ chassis with NI-9234 modules.

## Features

- **Real-time Data Acquisition**: Continuous acquisition from 12+ channels at up to 51.2 kS/s
- **Live Visualization**: Real-time time-domain plotting with PyQtGraph for high performance
- **Frequency Analysis**: FFT/PSD computation with configurable parameters
- **Digital Filtering**: Butterworth, Chebyshev, and Bessel filters (lowpass, highpass, bandpass, bandstop)
- **Data Export**: Save to CSV, HDF5, or TDMS formats
- **Configuration Management**: Save and load acquisition settings

## Hardware Requirements

- **DAQ Chassis**: NI-9178 CompactDAQ chassis
- **Modules**: NI-9234 (4-channel, 24-bit, ±5V, IEPE-enabled analog input modules)
- **Sensors**: IEPE accelerometers with configurable sensitivity (mV/g)

## Software Requirements

- Python 3.8 or higher
- NI-DAQmx driver (download from ni.com)
- See `requirements.txt` for Python package dependencies

## Installation

1. **Install NI-DAQmx driver**:
   - Download from [ni.com](https://www.ni.com/en-us/support/downloads/drivers/download.ni-daqmx.html)
   - Follow NI's installation instructions

2. **Clone the repository**:
   ```bash
   cd /home/nm0610/Code/NI-APP
   ```

3. **Create virtual environment** (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Linux/Mac
   # or
   venv\Scripts\activate  # On Windows
   ```

4. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Connect your hardware**:
   - Connect NI-9178 chassis to your computer via USB
   - Install NI-9234 modules in the chassis
   - Connect IEPE accelerometers to the input channels

2. **Run the application**:
   ```bash
   python src/main.py
   ```

3. **Configure acquisition**:
   - Select your DAQ device from the dropdown
   - Configure channels (enable/disable, sensitivity, units)
   - Set sample rate and buffer size
   - Click "Start" to begin acquisition

4. **View and analyze data**:
   - Real-time plots show time-domain data
   - FFT plots display frequency spectrum
   - Apply filters for signal conditioning
   - Export data for further analysis

## Project Structure

```
NI-APP/
├── config/                 # Configuration files
├── src/                   # Source code
│   ├── main.py           # Application entry point
│   ├── gui/              # GUI components
│   ├── daq/              # DAQ interface
│   ├── processing/       # Signal processing
│   ├── export/           # Data export
│   ├── config/           # Configuration management
│   └── utils/            # Utilities
├── tests/                # Unit and integration tests
└── resources/            # Icons and stylesheets
```

## Technical Details

### Data Flow Architecture

```
Hardware → DAQManager → AcquisitionThread (QThread)
              ↓
          DataBuffer (circular buffer, 60s)
              ↓
          SignalProcessor
              ├→ Filters → filtered data
              └→ FFT → frequency data
              ↓
          Plot Widgets (real-time display)
```

### Threading

- **Main Thread**: GUI updates and user interaction
- **Acquisition Thread**: Background continuous DAQ reading (QThread)
- **Export Thread**: Background file writing (QThread)

Communication between threads uses Qt signals/slots for thread safety.

### Performance

- GUI responsiveness: < 100 ms
- Plot update rate: 30 Hz minimum
- Continuous acquisition: 51.2 kS/s × 12+ channels with no sample drops
- Memory usage: < 1 GB for 60-second buffer

## Configuration

Application settings are stored in:
- Linux: `~/.config/ni-daq-app/config.json`
- Windows: `%APPDATA%/NIDAQApp/config.json`

Configuration includes:
- Last used DAQ device and settings
- Channel configurations
- Filter parameters
- Window geometry and preferences

## Testing

Run unit tests:
```bash
pytest tests/
```

Run with coverage:
```bash
pytest --cov=src tests/
```

## Troubleshooting

### DAQ device not detected
- Ensure NI-DAQmx driver is installed
- Check USB connection
- Run NI MAX (Measurement & Automation Explorer) to verify device

### Performance issues with many channels
- Reduce plot update rate
- Enable downsampling in plot settings
- Reduce number of displayed channels
- Lower sample rate if appropriate for your application

### Import errors
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Activate virtual environment if using one

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]

## Authors

[Add author information here]

## Acknowledgments

- Built with PyQt5 for the GUI framework
- Uses PyQtGraph for high-performance real-time plotting
- NI-DAQmx for hardware interfacing
- SciPy for signal processing algorithms

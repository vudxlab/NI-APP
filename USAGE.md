# NI DAQ Vibration Analysis - Usage Guide

## Quick Start

### 1. Run the Application

```bash
cd /home/nm0610/Code/NI-APP
source venv/bin/activate
python run_app.py
```

### 2. Configure DAQ

1. **Select Device**: Choose your cDAQ device from the dropdown (e.g., cDAQ1)
2. **Set Sample Rate**: Select from 1024, 2048, 4096, 8192, 12800, 25600, or 51200 Hz
3. **Configure Buffer**: Set buffer size (default: 60 seconds)

### 3. Configure Channels

In the "Channel Configuration" panel:
- Enable/disable channels
- Set accelerometer sensitivity (mV/g)
- Choose units: g, m/s², or mm/s²
- Set coupling: AC or DC

### 4. Start Acquisition

1. Click **"Start"** button or press **F5**
2. Real-time data will appear in:
   - **Time Domain** tab: Time-series plots
   - **Frequency Domain** tab: FFT plots

### 5. Apply Filters (Optional)

In the "Filter Configuration" panel:
- **Filter Type**: Butterworth, Chebyshev I/II, Bessel
- **Filter Mode**: Lowpass, Highpass, Bandpass, Bandstop
- **Cutoff Frequency**: Set in Hz
- **Order**: Filter order (typically 4-8)

### 6. Export Data

1. Stop acquisition (F6)
2. Go to **File > Export Data** or press **Ctrl+E**
3. Choose format:
   - **CSV**: Simple text format
   - **HDF5**: Compressed binary format
   - **TDMS**: NI LabVIEW format
4. Select channels and time range
5. Click **Export**

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| F5 | Start acquisition |
| F6 | Stop acquisition |
| Ctrl+E | Export data |
| Ctrl+O | Open configuration |
| Ctrl+S | Save configuration |
| Ctrl+L | Clear buffers |
| Ctrl+Q | Quit application |

## Tips

### Hardware Detection

- App automatically detects NI DAQ hardware on startup
- If no hardware found, app runs in simulation mode
- Check NI MAX if devices not detected

### Performance

- For 12+ channels @ 51.2 kHz, reduce plot update rate if GUI lags
- Enable downsampling for very long recordings
- Use HDF5 format for large datasets (best compression)

### Troubleshooting

**Warning: "nidaqmx not available"**
- This is a harmless warning from import time
- If devices are detected, nidaqmx is working correctly
- Ignore this warning

**"Failed to start acquisition"**
- Ensure at least one channel is enabled
- Check device is not in use by another application
- Verify sample rate is valid for your hardware

**"AttributeError" on startup**
- Clear Python cache: `find . -name __pycache__ -exec rm -rf {} +`
- Restart application

## Hardware Setup

### Supported Hardware

- **Chassis**: NI-9178 CompactDAQ
- **Modules**: 
  - NI-9234: 4-ch, 24-bit, IEPE, ±5V (primary)
  - NI-9215: 4-ch, 16-bit, ±10V
  - NI-9237: 4-ch bridge/strain

### Sensor Connection

1. Connect IEPE accelerometers to NI-9234 channels
2. Enable IEPE excitation in channel settings
3. Set correct sensitivity (check sensor datasheet)
4. Use AC coupling for vibration measurements

### Sample Rate Selection

| Application | Recommended Rate |
|-------------|------------------|
| General vibration | 25.6 kHz |
| Low frequency (<500 Hz) | 2-8 kHz |
| High frequency analysis | 51.2 kHz |
| Modal analysis | 12.8-25.6 kHz |

**Note**: Nyquist theorem - measure up to 1/2 of sample rate

## File Locations

- **Config**: `~/.config/ni-daq-app/config.json`
- **Logs**: Console output (redirect to file if needed)
- **Exports**: User-selected directory

## Advanced Usage

### Command Line Options

```bash
# Disable splash screen
NI_APP_NO_SPLASH=1 python run_app.py

# Run tests
pytest tests/ -v

# Generate coverage report
pytest --cov=src tests/
```

### Configuration Files

Save/load DAQ configurations:
- **File > Save Configuration**: Save current setup
- **File > Open Configuration**: Load saved setup
- Format: JSON with all channel/timing parameters

## Support

For issues or questions:
1. Check logs for error messages
2. Verify hardware in NI MAX
3. Test with NI's example programs first
4. Check NI-DAQmx driver version

## Version

Current version: 1.0.0


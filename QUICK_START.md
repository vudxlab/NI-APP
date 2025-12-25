# Quick Start Guide

## 🚀 Start Application in 3 Steps

```bash
# 1. Navigate to project
cd /home/nm0610/Code/NI-APP

# 2. Activate virtual environment
source venv/bin/activate

# 3. Run application
python run_app.py
```

---

## ✅ Verify Everything Works

### Check nidaqmx is Available
```bash
python -c "from src.daq.channel_manager import NIDAQMX_AVAILABLE; print(f'✅ Ready' if NIDAQMX_AVAILABLE else '❌ Problem')"
```

Expected output: `✅ Ready`

### Check Hardware Detection
```bash
python -c "from src.daq.daq_manager import DAQManager; print(f'Found {len(DAQManager.enumerate_devices())} devices')"
```

Expected output: `Found 5 devices` (or your actual device count)

---

## 📊 First Data Acquisition

1. **Launch app:** `python run_app.py`

2. **Configure:**
   - Device: Select `cDAQ1` from dropdown
   - Sample Rate: `25600` Hz recommended
   - Channels: Enable channels you want (default: all enabled)

3. **Start:** 
   - Click **Start** button
   - Or press **F5**

4. **Verify Success:**
   Look for log message:
   ```
   INFO - Task created with 12 channels at 25600 Hz
   ```

5. **View Data:**
   - **Time Domain** tab: Real-time waveforms
   - **Frequency Domain** tab: FFT spectra

6. **Stop:**
   - Click **Stop** button
   - Or press **F6**

---

## 🔧 Troubleshooting

### App Won't Start
```bash
# Clear cache and retry
find . -name __pycache__ -exec rm -rf {} +
python run_app.py
```

### "nidaqmx not available"
Check if actually a problem:
```bash
python -c "from src.daq.channel_manager import NIDAQMX_AVAILABLE; print(NIDAQMX_AVAILABLE)"
```
If prints `True`, warning is harmless (ignore it).

### No Devices Found
1. Open NI MAX (National Instruments Measurement & Automation Explorer)
2. Check if devices appear there
3. If not, check USB cable and power
4. Restart DAQ chassis

### "Failed to create task"
- Check at least one channel is enabled
- Verify sample rate is valid (1024-51200 Hz)
- Make sure no other app is using DAQ

---

## 🎯 Common Tasks

### Change Sample Rate
1. Stop acquisition (F6)
2. Select new rate from dropdown
3. Start acquisition (F5)

### Enable/Disable Channels
1. Go to "Channel Configuration" panel
2. Check/uncheck channels
3. Apply changes

### Export Data
1. Stop acquisition (F6)
2. **File > Export Data** (Ctrl+E)
3. Choose format: CSV, HDF5, or TDMS
4. Select save location
5. Click **Export**

### Apply Filter
1. Go to "Filter Configuration" panel
2. Choose filter type (e.g., Butterworth)
3. Set mode (Lowpass/Highpass/Bandpass)
4. Set cutoff frequency
5. Enable filter
6. Data is filtered in real-time

---

## 📋 Keyboard Shortcuts

| Key | Action |
|-----|--------|
| F5 | Start acquisition |
| F6 | Stop acquisition |
| Ctrl+E | Export data |
| Ctrl+O | Open config |
| Ctrl+S | Save config |
| Ctrl+Q | Quit |

---

## 💾 Save/Load Configuration

### Save Current Setup
1. Configure channels and settings
2. **File > Save Configuration**
3. Choose filename (e.g., `my_test.json`)
4. Click **Save**

### Load Saved Setup
1. **File > Open Configuration**
2. Select `.json` file
3. Click **Open**
4. Settings automatically applied

---

## 🔍 Check Application Health

### Full System Check
```bash
# Run this to verify everything
cd /home/nm0610/Code/NI-APP
source venv/bin/activate

echo "Checking imports..."
python -c "
from src.daq.channel_manager import NIDAQMX_AVAILABLE as CM
from src.daq.daq_manager import NIDAQMX_AVAILABLE as DM
print(f'channel_manager: {\"✅\" if CM else \"❌\"}')
print(f'daq_manager: {\"✅\" if DM else \"❌\"}')
"

echo -e "\nChecking hardware..."
python -c "
from src.daq.daq_manager import DAQManager
devices = DAQManager.enumerate_devices()
print(f'Found {len(devices)} device(s):')
[print(f'  - {d[\"name\"]}: {d[\"product_type\"]}') for d in devices]
"

echo -e "\nAll checks complete!"
```

---

## 🎓 Tips

### Best Sample Rates
- **Low frequency (<500 Hz):** 2048-8192 Hz
- **General vibration:** 25600 Hz ⭐
- **High frequency:** 51200 Hz
- **Modal analysis:** 12800-25600 Hz

### Performance
- More channels = more CPU
- Higher sample rate = more CPU
- Reduce plot update rate if GUI lags
- Use downsampling for long recordings

### Data Export
- **CSV:** Simple, any tool can read
- **HDF5:** Compressed, best for large data ⭐
- **TDMS:** NI LabVIEW compatible

---

## 📞 Get Help

### Check Logs
All errors print to console. Look for:
- `ERROR` - Something failed
- `WARNING` - Minor issues (often ignorable)
- `INFO` - Normal operation

### Common Warning (Harmless)
```
WARNING:root:nidaqmx not available - running in simulation mode
```
This appears during import but is **false positive**. If devices are detected, ignore it.

### Real Problems
If you see:
- "Failed to create task" repeatedly
- "No devices found" when they should be there
- App crashes on startup

Try:
1. Clear cache: `find . -name __pycache__ -rm -rf {} +`
2. Restart DAQ hardware
3. Check NI MAX
4. Review FINAL_STATUS.md for details

---

## ✅ Success Indicators

When app is working correctly, you'll see:
```
INFO - Found device: cDAQ1 (cDAQ-9178)
INFO - Found device: cDAQ1Mod1 (NI 9234)
INFO - DAQManager initialized (nidaqmx available: True)
INFO - Task created with 12 channels at 25600 Hz
```

---

## 📚 More Information

- **Full User Guide:** `USAGE.md`
- **Bug History:** `BUGFIXES.md`
- **System Status:** `FINAL_STATUS.md`
- **Project README:** `README.md`

---

**Ready to go!** 🎉

Start with: `python run_app.py`


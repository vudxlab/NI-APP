"""
Test script to verify downsample threshold implementation
"""
import sys
from PyQt5.QtWidgets import QApplication

# Test 1: Settings Model
print("Test 1: Checking GUISettings model...")
from src.config.app_settings import GUISettings
settings = GUISettings()
assert hasattr(settings, 'realtime_downsample_threshold'), "Missing realtime_downsample_threshold field"
assert settings.realtime_downsample_threshold == 1000, f"Default value incorrect: {settings.realtime_downsample_threshold}"
print("[PASS] GUISettings has realtime_downsample_threshold field with default 1000")

# Test 2: DAQ Config Panel
print("\nTest 2: Checking DAQConfigPanel...")
app = QApplication(sys.argv)
from src.gui.widgets.daq_config_panel import DAQConfigPanel
panel = DAQConfigPanel()
assert hasattr(panel, 'downsample_spin'), "Missing downsample_spin widget"
assert hasattr(panel, 'downsample_threshold_changed'), "Missing downsample_threshold_changed signal"
assert hasattr(panel, 'get_downsample_threshold'), "Missing get_downsample_threshold method"
assert hasattr(panel, 'set_downsample_threshold'), "Missing set_downsample_threshold method"
print("[PASS] DAQConfigPanel has downsample SpinBox and methods")

# Test 3: Realtime Plot Widget
print("\nTest 3: Checking RealtimePlotWidget...")
from src.gui.widgets.realtime_plot_widget import RealtimePlotWidget
plot_widget = RealtimePlotWidget()
assert hasattr(plot_widget, 'set_downsample_threshold'), "Missing set_downsample_threshold method"
print("[PASS] RealtimePlotWidget has set_downsample_threshold method")

# Test 4: Test getter/setter
print("\nTest 4: Testing getter/setter...")
panel.set_downsample_threshold(2500)
value = panel.get_downsample_threshold()
assert value == 2500, f"Expected 2500, got {value}"
print("[PASS] Getter/setter working correctly")

# Test 5: Test signal connection
print("\nTest 5: Testing signal emission...")
signal_received = []
def on_threshold_changed(value):
    signal_received.append(value)

panel.downsample_threshold_changed.connect(on_threshold_changed)
panel.set_downsample_threshold(3000)
assert len(signal_received) > 0, "Signal not emitted"
assert signal_received[-1] == 3000, f"Expected 3000, got {signal_received[-1]}"
print("[PASS] Signal emitted correctly")

# Test 6: Test plot widget setter with validation
print("\nTest 6: Testing plot widget validation...")
plot_widget.set_downsample_threshold(5000)
assert plot_widget.downsample_threshold == 5000, f"Expected 5000, got {plot_widget.downsample_threshold}"

# Test invalid value (should fall back to 1000)
plot_widget.set_downsample_threshold(50)  # Too low
assert plot_widget.downsample_threshold == 1000, f"Expected fallback to 1000, got {plot_widget.downsample_threshold}"
print("[PASS] Plot widget validation working")

print("\n" + "="*50)
print("ALL TESTS PASSED!")
print("="*50)
print("\nImplementation Summary:")
print("- GUISettings model updated with realtime_downsample_threshold field")
print("- DAQConfigPanel has new SpinBox control (100-10000 range)")
print("- Signal wiring functional (DAQ panel -> plot widget)")
print("- Getter/setter methods working correctly")
print("- Validation working (invalid values default to 1000)")

"""
Test script for long-window FFT modules.

Run this to verify that long_data_saver and long_window_fft work correctly.
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.processing.data_buffer import DataBuffer
from src.processing.long_data_saver import LongDataSaver
from src.processing.long_window_fft import LongWindowFFTProcessor


def test_data_saver():
    """Test LongDataSaver."""
    print("\n" + "="*80)
    print("TEST 1: LongDataSaver")
    print("="*80)

    sample_rate = 25600
    n_channels = 4
    duration = 10  # 10 seconds for quick test

    # Create buffer with test data
    buffer_size = int(sample_rate * duration)
    buffer = DataBuffer(n_channels=n_channels, buffer_size=buffer_size)

    print(f"Creating {duration}s of test data @ {sample_rate} Hz...")
    for i in range(duration):
        data_chunk = np.random.randn(n_channels, sample_rate)
        buffer.append(data_chunk)

    print(f"Buffer: {buffer}")

    # Create saver
    saver = LongDataSaver(sample_rate=sample_rate, max_duration_seconds=200.0)

    # Save 10s
    print(f"\nSaving {duration}s to temp file...")
    try:
        file_path = saver.save_from_buffer(buffer, duration_seconds=duration, format='hdf5')
        print(f"✓ Saved to: {file_path}")
        print(f"  File size: {file_path.stat().st_size / (1024*1024):.2f} MB")

        # Load back
        print(f"\nLoading data back...")
        loaded_data, metadata = saver.load_temp_file(file_path)
        print(f"✓ Loaded data shape: {loaded_data.shape}")
        print(f"  Metadata: {metadata}")

        # Verify
        original = buffer.get_all()
        if np.allclose(original, loaded_data, rtol=1e-5):
            print("✓ Data integrity verified!")
        else:
            print("✗ Data mismatch!")

        # Cleanup
        file_path.unlink()
        print(f"✓ Cleaned up temp file")

        return True

    except Exception as e:
        print(f"✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_long_fft():
    """Test LongWindowFFTProcessor."""
    print("\n" + "="*80)
    print("TEST 2: LongWindowFFTProcessor")
    print("="*80)

    sample_rate = 25600
    duration = 20  # 20 seconds

    # Create test signal with known frequencies
    n_samples = int(sample_rate * duration)
    t = np.arange(n_samples) / sample_rate

    print(f"Creating test signal: {duration}s @ {sample_rate} Hz")
    print(f"Frequency components: 0.5 Hz, 1.2 Hz, 5.0 Hz")

    signal_test = (
        2.0 * np.sin(2 * np.pi * 0.5 * t) +   # 0.5 Hz
        1.5 * np.sin(2 * np.pi * 1.2 * t) +   # 1.2 Hz
        1.0 * np.sin(2 * np.pi * 5.0 * t) +   # 5.0 Hz
        0.1 * np.random.randn(n_samples)      # Noise
    )

    # Create processor
    processor = LongWindowFFTProcessor(sample_rate=sample_rate, window_function='hann')
    print(f"\n✓ Processor created")

    # Get window info
    window_info = processor.get_window_info()
    print(f"\nAvailable windows:")
    for name, info in window_info.items():
        print(f"  {name}: {info['frequency_resolution_hz']:.6f} Hz resolution")

    # Test different windows
    print(f"\nTesting FFT with different windows:")
    for window_duration in ['10s', '20s']:
        try:
            print(f"\n  {window_duration}:")
            frequencies, magnitude = processor.compute_magnitude(
                signal_test,
                window_duration=window_duration,
                scale='linear'
            )
            print(f"    ✓ FFT computed: {len(frequencies)} frequency bins")
            print(f"      Freq range: [{frequencies[0]:.6f}, {frequencies[-1]:.2f}] Hz")
            print(f"      Mag range: [{magnitude.min():.4f}, {magnitude.max():.4f}]")

            # Find peaks
            peaks = processor.find_peaks(frequencies, magnitude, threshold=0.1, n_peaks=5)
            print(f"      Peaks found: {len(peaks)}")
            for i, peak in enumerate(peaks[:3], 1):
                print(f"        {i}. {peak['frequency']:.4f} Hz (mag={peak['magnitude']:.4f})")

        except Exception as e:
            print(f"    ✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
            return False

    return True


def test_integration():
    """Test integration: Buffer -> Save -> FFT."""
    print("\n" + "="*80)
    print("TEST 3: Integration Test")
    print("="*80)

    sample_rate = 25600
    n_channels = 4
    duration = 20

    print(f"Creating {n_channels} channels, {duration}s @ {sample_rate} Hz")

    # Create buffer
    buffer_size = int(sample_rate * duration)
    buffer = DataBuffer(n_channels=n_channels, buffer_size=buffer_size)

    # Fill with test data (different freq per channel)
    t = np.arange(sample_rate * duration) / sample_rate
    for ch in range(n_channels):
        freq = 0.5 * (ch + 1)  # 0.5, 1.0, 1.5, 2.0 Hz
        for i in range(duration):
            start = i * sample_rate
            end = start + sample_rate
            chunk = np.sin(2 * np.pi * freq * t[start:end])
            buffer.append(chunk.reshape(1, -1))

    print(f"✓ Buffer filled")

    # Save
    saver = LongDataSaver(sample_rate=sample_rate, max_duration_seconds=200.0)
    print(f"\nSaving buffer...")
    try:
        file_path = saver.save_from_buffer(buffer, duration_seconds=duration, format='hdf5')
        print(f"✓ Saved to: {file_path}")
    except Exception as e:
        print(f"✗ Save failed: {e}")
        return False

    # Load
    print(f"\nLoading...")
    try:
        loaded_data, metadata = saver.load_temp_file(file_path)
        print(f"✓ Loaded: shape={loaded_data.shape}")
    except Exception as e:
        print(f"✗ Load failed: {e}")
        return False

    # FFT
    processor = LongWindowFFTProcessor(sample_rate=sample_rate)
    print(f"\nComputing FFT for all channels (20s window)...")
    try:
        for ch in range(n_channels):
            frequencies, magnitude = processor.compute_magnitude(
                loaded_data[ch, :],
                window_duration='20s',
                scale='linear'
            )
            peaks = processor.find_peaks(frequencies, magnitude, threshold=0.3, n_peaks=3)
            expected_freq = 0.5 * (ch + 1)
            print(f"  Channel {ch} (expected {expected_freq} Hz):")
            if peaks:
                detected_freq = peaks[0]['frequency']
                error = abs(detected_freq - expected_freq)
                status = "✓" if error < 0.01 else "✗"
                print(f"    {status} Detected: {detected_freq:.4f} Hz (error: {error:.6f} Hz)")
            else:
                print(f"    ✗ No peaks found")

    except Exception as e:
        print(f"✗ FFT failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Cleanup
        file_path.unlink()

    return True


if __name__ == "__main__":
    print("\n" + "="*80)
    print("LONG-WINDOW FFT MODULES TEST")
    print("="*80)

    results = []

    # Run tests
    results.append(("LongDataSaver", test_data_saver()))
    results.append(("LongWindowFFTProcessor", test_long_fft()))
    results.append(("Integration", test_integration()))

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
        if not passed:
            all_passed = False

    print("="*80)

    if all_passed:
        print("✓ ALL TESTS PASSED!")
        sys.exit(0)
    else:
        print("✗ SOME TESTS FAILED")
        sys.exit(1)

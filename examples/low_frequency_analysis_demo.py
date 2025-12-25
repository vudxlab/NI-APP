"""
Demo: Low-frequency Analysis with Long-window FFT

This example demonstrates how to:
1. Save 200 seconds of data from a buffer to a temporary file
2. Load the saved data
3. Perform FFT analysis with different window sizes (10s, 20s, 50s, 100s, 200s)
4. Analyze low-frequency components

This is particularly useful for detecting low-frequency vibrations and
slow-varying phenomena that require high frequency resolution.
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from processing.data_buffer import DataBuffer
from processing.long_data_saver import LongDataSaver
from processing.long_window_fft import LongWindowFFTProcessor


def create_test_data(sample_rate, duration, n_channels=4):
    """
    Create realistic test data with low-frequency components.

    Simulates vibration data with:
    - Low-frequency structural modes (0.5 Hz, 1.2 Hz, 3.5 Hz)
    - Mid-frequency machine components (50 Hz, 120 Hz)
    - High-frequency content (500 Hz)
    - Noise
    """
    n_samples = int(sample_rate * duration)
    t = np.arange(n_samples) / sample_rate

    print(f"Creating test data: {n_channels} channels, {duration}s @ {sample_rate} Hz")
    print(f"Total samples: {n_samples:,}")

    # Initialize data array
    data = np.zeros((n_channels, n_samples))

    # Channel 0: Low-frequency structural modes
    data[0] = (
        3.0 * np.sin(2 * np.pi * 0.5 * t) +   # 0.5 Hz - structural mode
        2.0 * np.sin(2 * np.pi * 1.2 * t) +   # 1.2 Hz - structural mode
        1.5 * np.sin(2 * np.pi * 3.5 * t)     # 3.5 Hz - structural mode
    )

    # Channel 1: Machine vibration (rotating equipment)
    data[1] = (
        2.5 * np.sin(2 * np.pi * 0.8 * t) +   # 0.8 Hz - slow oscillation
        2.0 * np.sin(2 * np.pi * 50 * t) +    # 50 Hz - machine fundamental
        1.0 * np.sin(2 * np.pi * 100 * t)     # 100 Hz - 2nd harmonic
    )

    # Channel 2: Mixed frequency content
    data[2] = (
        1.8 * np.sin(2 * np.pi * 0.3 * t) +   # 0.3 Hz - very low frequency
        1.5 * np.sin(2 * np.pi * 2.1 * t) +   # 2.1 Hz
        1.2 * np.sin(2 * np.pi * 120 * t)     # 120 Hz
    )

    # Channel 3: High-frequency with low-frequency modulation
    carrier = np.sin(2 * np.pi * 500 * t)
    modulation = 0.5 * np.sin(2 * np.pi * 1.5 * t) + 0.5
    data[3] = 2.0 * modulation * carrier + 1.0 * np.sin(2 * np.pi * 0.6 * t)

    # Add noise to all channels
    noise_level = 0.1
    for ch in range(n_channels):
        data[ch] += noise_level * np.random.randn(n_samples)

    print("Frequency components:")
    print("  Channel 0: 0.5, 1.2, 3.5 Hz (structural modes)")
    print("  Channel 1: 0.8, 50, 100 Hz (machine vibration)")
    print("  Channel 2: 0.3, 2.1, 120 Hz (mixed)")
    print("  Channel 3: 0.6, 1.5 Hz (modulation), 500 Hz (carrier)")

    return data


def main():
    print("=" * 80)
    print("LOW-FREQUENCY ANALYSIS DEMO")
    print("=" * 80)

    # Configuration
    SAMPLE_RATE = 25600  # Hz (NI-9234 default)
    N_CHANNELS = 4
    BUFFER_DURATION = 200  # seconds

    # ========================================================================
    # STEP 1: Create and populate data buffer
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 1: Creating and populating data buffer")
    print("=" * 80)

    buffer_size = int(SAMPLE_RATE * BUFFER_DURATION)
    buffer = DataBuffer(n_channels=N_CHANNELS, buffer_size=buffer_size)

    # Create test data
    test_data = create_test_data(SAMPLE_RATE, BUFFER_DURATION, N_CHANNELS)

    # Add to buffer (simulate real-time acquisition in 1-second chunks)
    print(f"\nFilling buffer with {BUFFER_DURATION}s of data...")
    chunk_size = int(SAMPLE_RATE)  # 1 second chunks
    for i in range(0, test_data.shape[1], chunk_size):
        chunk = test_data[:, i:i+chunk_size]
        buffer.append(chunk)

    print(f"Buffer status: {buffer}")
    stats = buffer.get_stats()
    print(f"Buffer fill: {stats['fill_percentage']:.1f}%")
    print(f"Memory usage: {buffer.get_memory_usage() / (1024*1024):.2f} MB")

    # ========================================================================
    # STEP 2: Save buffer data to temporary file
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 2: Saving data to temporary file")
    print("=" * 80)

    saver = LongDataSaver(sample_rate=SAMPLE_RATE, max_duration_seconds=200.0)

    print(f"Saving 200s of data to HDF5 file...")
    file_path = saver.save_from_buffer(buffer, duration_seconds=200.0, format='hdf5')

    print(f"\nFile saved: {file_path}")
    print(f"File size: {file_path.stat().st_size / (1024*1024):.2f} MB")

    # ========================================================================
    # STEP 3: Load data from file
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 3: Loading data from temporary file")
    print("=" * 80)

    print(f"Loading data from: {file_path.name}")
    loaded_data, metadata = saver.load_temp_file(file_path)

    print(f"\nLoaded data shape: {loaded_data.shape}")
    print(f"Duration: {loaded_data.shape[1] / SAMPLE_RATE:.1f} seconds")
    print(f"\nMetadata:")
    for key, value in metadata.items():
        if isinstance(value, (int, float)):
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")
        else:
            print(f"  {key}: {value}")

    # ========================================================================
    # STEP 4: Perform long-window FFT analysis
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 4: Long-window FFT analysis")
    print("=" * 80)

    processor = LongWindowFFTProcessor(sample_rate=SAMPLE_RATE, window_function='hann')

    # Display window information
    print("\nAvailable FFT windows:")
    window_info = processor.get_window_info()
    print(f"\n{'Window':<8} {'Duration':<12} {'Samples':<12} {'Freq Res (Hz)':<15} {'Freq Bins':<12}")
    print("-" * 70)
    for name, info in window_info.items():
        print(f"{name:<8} {info['duration_seconds']:<12.0f} "
              f"{info['window_size_samples']:<12,} "
              f"{info['frequency_resolution_hz']:<15.6f} "
              f"{info['num_frequency_bins']:<12,}")

    # Analyze Channel 0 (structural modes: 0.5, 1.2, 3.5 Hz)
    print("\n" + "-" * 80)
    print("Analyzing Channel 0 (Structural Modes)")
    print("-" * 80)

    channel_data = loaded_data[0, :]

    # Compute for all window durations
    print("\nComputing FFT for all window durations...")
    all_results = processor.compute_all_windows(channel_data, scale='linear', method='magnitude')

    print(f"\n{'Window':<10} {'Freq Range (Hz)':<20} {'Top 3 Peaks (Hz)':<50}")
    print("-" * 80)

    for window_duration in ['10s', '20s', '50s', '100s', '200s']:
        if window_duration in all_results:
            freq, mag = all_results[window_duration]

            # Find peaks
            peaks = processor.find_peaks(freq, mag, threshold=0.1, n_peaks=3)

            freq_range = f"{freq[0]:.6f} - {freq[-1]:.1f}"
            peak_str = ", ".join([f"{p['frequency']:.4f}" for p in peaks])

            print(f"{window_duration:<10} {freq_range:<20} {peak_str:<50}")

    # ========================================================================
    # STEP 5: Detailed low-frequency analysis (0-10 Hz)
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 5: Detailed low-frequency analysis (0-10 Hz)")
    print("=" * 80)

    for ch in range(N_CHANNELS):
        print(f"\n--- Channel {ch} ---")

        channel_data = loaded_data[ch, :]

        # Analyze low frequencies with 200s window for best resolution
        low_freq_results = processor.analyze_low_frequencies(
            channel_data,
            max_frequency=10.0,
            window_duration='200s',
            n_peaks=10
        )

        print(f"Window: {low_freq_results['window_duration']}")
        print(f"Frequency resolution: {low_freq_results['frequency_resolution']:.6f} Hz")
        print(f"Total power (0-10 Hz): {low_freq_results['total_power']:.4f}")
        print(f"RMS value: {low_freq_results['rms_value']:.4f}")

        print(f"\nTop 5 peaks in 0-10 Hz range:")
        for i, peak in enumerate(low_freq_results['peaks'][:5], 1):
            print(f"  {i}. {peak['frequency']:>8.4f} Hz  (magnitude: {peak['magnitude']:>8.4f})")

    # ========================================================================
    # STEP 6: Comparison of frequency resolutions
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 6: Frequency resolution comparison")
    print("=" * 80)

    print("\nAbility to resolve closely-spaced frequencies:")
    print(f"\n{'Window':<8} {'Freq Resolution':<18} {'Example: Can Resolve':<40}")
    print("-" * 70)

    for window_duration in ['10s', '20s', '50s', '100s', '200s']:
        freq_res = processor.get_frequency_resolution(window_duration)

        # Example: can we resolve 0.5 Hz and 0.6 Hz?
        f1, f2 = 0.5, 0.6
        can_resolve = "YES" if (f2 - f1) > freq_res else "NO"

        print(f"{window_duration:<8} {freq_res:<18.6f} "
              f"{f1} Hz and {f2} Hz? {can_resolve} (need > {freq_res:.6f} Hz apart)")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"""
This demo showed how to:

1. ✓ Create a 200-second data buffer with multi-channel vibration data
2. ✓ Save the buffer to a temporary HDF5 file ({file_path.stat().st_size / (1024*1024):.2f} MB)
3. ✓ Load the data back for offline analysis
4. ✓ Perform FFT with 5 different window sizes (10s, 20s, 50s, 100s, 200s)
5. ✓ Detect low-frequency components with high resolution

Key findings:
- 200s window provides {processor.get_frequency_resolution('200s'):.6f} Hz resolution
- This allows detection of very closely-spaced low-frequency components
- Ideal for structural monitoring, machinery diagnostics, and slow phenomena

Files saved in: {saver.temp_dir}
Total temp dir size: {saver.get_temp_dir_size():.2f} MB

For actual use:
- Integrate with real-time DAQ buffer
- Set up periodic saves (e.g., every minute)
- Configure automatic cleanup of old files
- Add GUI controls for window selection
""")

    print("=" * 80)
    print("Demo completed successfully!")
    print("=" * 80)

    # Cleanup
    print(f"\nCleaning up temporary files...")
    deleted = saver.cleanup_old_files(max_age_hours=0.0)
    print(f"Deleted {deleted} temporary file(s)")


if __name__ == "__main__":
    main()

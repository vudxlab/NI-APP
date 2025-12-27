"""
Test script for MAT format support.
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from export.data_file_reader import DataFileReader

def test_mat_reading():
    """Test reading S9.mat file."""
    print("="*60)
    print("Testing MAT Format Implementation")
    print("="*60)

    # Create reader
    reader = DataFileReader()

    # Test 1: Detect format
    print('\n[Test 1] Detect format')
    try:
        format_detected = reader.detect_format('S9.mat')
        print(f'  Format detected: {format_detected}')
        print(f'  ✓ PASS' if format_detected == 'mat' else f'  ✗ FAIL')
    except Exception as e:
        print(f'  ✗ FAIL: {e}')
        return

    # Test 2: Get file info
    print('\n[Test 2] Get file info')
    try:
        info = reader.get_file_info('S9.mat')
        print(f'  File info:')
        for key, value in sorted(info.items()):
            print(f'    {key}: {value}')
        print(f'  ✓ PASS')
    except Exception as e:
        print(f'  ✗ FAIL: {e}')
        import traceback
        traceback.print_exc()
        return

    # Test 3: Read recent 10 seconds
    print('\n[Test 3] Read last 10 seconds (assuming 51.2 kHz sample rate)')
    try:
        # Try with estimated sample rate
        sample_rate = 51200.0  # 51.2 kHz
        data, metadata = reader.read_recent_seconds(
            'S9.mat',
            duration_seconds=10.0,
            sample_rate=sample_rate
        )
        print(f'  Data shape: {data.shape}')
        print(f'  Expected: (6, ~512000) or less')
        print(f'  Metadata:')
        for key, value in sorted(metadata.items()):
            if key not in ['channel_names', 'channel_units']:  # Skip long lists
                print(f'    {key}: {value}')

        # Check shape
        n_channels, n_samples = data.shape
        success = n_channels == 6
        print(f'  ✓ PASS' if success else f'  ✗ FAIL: Expected 6 channels, got {n_channels}')

    except Exception as e:
        print(f'  ✗ FAIL: {e}')
        import traceback
        traceback.print_exc()
        return

    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)

if __name__ == '__main__':
    test_mat_reading()

from unittest.mock import Mock, MagicMock
from pathlib import Path
"""
Unit tests for export modules.

Tests CSV, HDF5, and TDMS exporters.
"""

import pytest
import numpy as np
import tempfile
import h5py


try:
    from src.export.csv_exporter import CSVExporter
    from src.export.hdf5_exporter import HDF5Exporter
    from src.export.tdms_exporter import TDMSExporter
    EXPORTERS_AVAILABLE = True
except ImportError:
    EXPORTERS_AVAILABLE = False

from src.daq.daq_config import ChannelConfig


@pytest.mark.skipif(not EXPORTERS_AVAILABLE, reason="Export modules not available")
class TestCSVExporter:
    """Test CSVExporter class."""

    @pytest.fixture
    def csv_exporter(self):
        return CSVExporter()

    @pytest.fixture
    def sample_data(self):
        """Create sample data for export."""
        return np.random.randn(4, 51200)  # 4 channels, 1 second at 51200 Hz

    @pytest.fixture
    def sample_metadata(self):
        """Create sample metadata."""
        return {
            'device_name': 'cDAQ1',
            'sample_rate': 51200.0,
            'duration': 1.0,
            'export_date': '2024-01-01 12:00:00'
        }

    def test_export_basic(self, csv_exporter, sample_data, sample_metadata):
        """Test basic CSV export."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            filepath = f.name

        try:
            channel_names = ['CH1', 'CH2', 'CH3', 'CH4']
            channel_units = ['g', 'g', 'g', 'g']

            csv_exporter.export(
                filepath=filepath,
                data=sample_data,
                sample_rate=51200.0,
                channel_names=channel_names,
                channel_units=channel_units,
                config={},
                metadata=sample_metadata
            )

            # Verify file was created
            assert Path(filepath).exists()

            # Read and verify content
            with open(filepath, 'r') as f:
                lines = f.readlines()

            # Should have header + data rows
            assert len(lines) > sample_data.shape[1]

        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_export_with_metadata(self, csv_exporter, sample_data, sample_metadata):
        """Test CSV export with metadata."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            filepath = f.name

        try:
            channel_names = ['CH1', 'CH2', 'CH3', 'CH4']
            channel_units = ['g', 'g', 'g', 'g']

            csv_exporter.export(
                filepath=filepath,
                data=sample_data,
                sample_rate=51200.0,
                channel_names=channel_names,
                channel_units=channel_units,
                config={},
                metadata=sample_metadata,
                include_metadata=True
            )

            # Verify metadata is in file
            with open(filepath, 'r') as f:
                content = f.read()

            assert 'device_name' in content or 'cDAQ1' in content

        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_export_without_metadata(self, csv_exporter, sample_data):
        """Test CSV export without metadata."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            filepath = f.name

        try:
            channel_names = ['CH1', 'CH2', 'CH3', 'CH4']
            channel_units = ['g', 'g', 'g', 'g']

            csv_exporter.export(
                filepath=filepath,
                data=sample_data,
                sample_rate=51200.0,
                channel_names=channel_names,
                channel_units=channel_units,
                config={},
                metadata={},
                include_metadata=False
            )

            # File should still exist
            assert Path(filepath).exists()

        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_export_time_selection(self, csv_exporter):
        """Test CSV export with time range selection."""
        data = np.random.randn(4, 51200)  # 1 second

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            filepath = f.name

        try:
            channel_names = ['CH1', 'CH2', 'CH3', 'CH4']
            channel_units = ['g', 'g', 'g', 'g']

            # Export first 0.5 seconds only
            csv_exporter.export(
                filepath=filepath,
                data=data,
                sample_rate=51200.0,
                channel_names=channel_names,
                channel_units=channel_units,
                config={},
                metadata={},
                time_range=(0, 0.5)
            )

            # Verify file has fewer rows
            with open(filepath, 'r') as f:
                lines = f.readlines()

            # Header + ~25600 data rows
            assert 25600 < len(lines) < 26000

        finally:
            Path(filepath).unlink(missing_ok=True)


@pytest.mark.skipif(not EXPORTERS_AVAILABLE, reason="Export modules not available")
class TestHDF5Exporter:
    """Test HDF5Exporter class."""

    @pytest.fixture
    def hdf5_exporter(self):
        return HDF5Exporter()

    @pytest.fixture
    def sample_data(self):
        """Create sample data for export."""
        return np.random.randn(4, 51200)

    @pytest.fixture
    def sample_metadata(self):
        """Create sample metadata."""
        return {
            'device_name': 'cDAQ1',
            'sample_rate': 51200.0,
            'duration': 1.0
        }

    def test_export_basic(self, hdf5_exporter, sample_data, sample_metadata):
        """Test basic HDF5 export."""
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.h5', delete=False) as f:
            filepath = f.name

        try:
            channel_names = ['CH1', 'CH2', 'CH3', 'CH4']
            channel_units = ['g', 'g', 'g', 'g']

            hdf5_exporter.export(
                filepath=filepath,
                data=sample_data,
                sample_rate=51200.0,
                channel_names=channel_names,
                channel_units=channel_units,
                config={},
                metadata=sample_metadata
            )

            # Verify file was created
            assert Path(filepath).exists()

            # Verify structure
            with h5py.File(filepath, 'r') as f:
                assert 'data' in f
                assert 'metadata' in f

        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_export_with_compression(self, hdf5_exporter, sample_data):
        """Test HDF5 export with compression."""
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.h5', delete=False) as f:
            filepath = f.name

        try:
            channel_names = ['CH1', 'CH2', 'CH3', 'CH4']
            channel_units = ['g', 'g', 'g', 'g']

            hdf5_exporter.export(
                filepath=filepath,
                data=sample_data,
                sample_rate=51200.0,
                channel_names=channel_names,
                channel_units=channel_units,
                config={},
                metadata={},
                compression=True,
                compression_level=4
            )

            # Verify compression was applied
            with h5py.File(filepath, 'r') as f:
                dataset = f['data/channel_00_CH1']
                assert dataset.compression is not None

        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_export_data_attributes(self, hdf5_exporter, sample_data):
        """Test that HDF5 datasets have proper attributes."""
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.h5', delete=False) as f:
            filepath = f.name

        try:
            channel_names = ['CH1', 'CH2', 'CH3', 'CH4']
            channel_units = ['g', 'g', 'g', 'g']

            hdf5_exporter.export(
                filepath=filepath,
                data=sample_data,
                sample_rate=51200.0,
                channel_names=channel_names,
                channel_units=channel_units,
                config={},
                metadata={}
            )

            # Verify dataset attributes
            with h5py.File(filepath, 'r') as f:
                dataset = f['data/channel_00_CH1']
                assert 'units' in dataset.attrs
                assert dataset.attrs['units'] == 'g'
                assert 'sample_rate' in dataset.attrs

        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_export_metadata_structure(self, hdf5_exporter, sample_data):
        """Test metadata structure in HDF5 file."""
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.h5', delete=False) as f:
            filepath = f.name

        try:
            channel_names = ['CH1', 'CH2', 'CH3', 'CH4']
            channel_units = ['g', 'g', 'g', 'g']

            metadata = {
                'device_name': 'cDAQ1',
                'sample_rate': 51200.0,
                'duration': 1.0,
                'export_date': '2024-01-01'
            }

            hdf5_exporter.export(
                filepath=filepath,
                data=sample_data,
                sample_rate=51200.0,
                channel_names=channel_names,
                channel_units=channel_units,
                config={},
                metadata=metadata
            )

            # Verify metadata
            with h5py.File(filepath, 'r') as f:
                meta_group = f['metadata']
                assert 'device_name' in meta_group.attrs
                assert 'sample_rate' in meta_group.attrs

        finally:
            Path(filepath).unlink(missing_ok=True)


@pytest.mark.skipif(not EXPORTERS_AVAILABLE, reason="Export modules not available")
class TestTDMSExporter:
    """Test TDMSExporter class."""

    @pytest.fixture
    def tdms_exporter(self):
        return TDMSExporter()

    @pytest.fixture
    def sample_data(self):
        """Create sample data for export."""
        return np.random.randn(4, 51200)

    def test_export_basic(self, tdms_exporter, sample_data):
        """Test basic TDMS export."""
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.tdms', delete=False) as f:
            filepath = f.name

        try:
            channel_names = ['CH1', 'CH2', 'CH3', 'CH4']
            channel_units = ['g', 'g', 'g', 'g']

            tdms_exporter.export(
                filepath=filepath,
                data=sample_data,
                sample_rate=51200.0,
                channel_names=channel_names,
                channel_units=channel_units,
                config={},
                metadata={}
            )

            # Verify file was created
            assert Path(filepath).exists()

        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_export_with_properties(self, tdms_exporter, sample_data):
        """Test TDMS export with channel properties."""
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.tdms', delete=False) as f:
            filepath = f.name

        try:
            channel_names = ['CH1', 'CH2', 'CH3', 'CH4']
            channel_units = ['g', 'g', 'g', 'g']

            tdms_exporter.export(
                filepath=filepath,
                data=sample_data,
                sample_rate=51200.0,
                channel_names=channel_names,
                channel_units=channel_units,
                config={},
                metadata={},
                include_properties=True
            )

            # Verify file was created
            assert Path(filepath).exists()

        finally:
            Path(filepath).unlink(missing_ok=True)


@pytest.mark.skipif(not EXPORTERS_AVAILABLE, reason="Export modules not available")
class TestExportEdgeCases:
    """Test edge cases for all exporters."""

    @pytest.fixture
    def sample_data(self):
        return np.random.randn(4, 51200)

    def test_export_empty_data(self, sample_data):
        """Test exporting empty data."""
        empty_data = np.array([]).reshape(4, 0)

        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            filepath = f.name

        try:
            exporter = CSVExporter()
            exporter.export(
                filepath=filepath,
                data=empty_data,
                sample_rate=51200.0,
                channel_names=['CH1'],
                channel_units=['g'],
                config={},
                metadata={}
            )

            # Should create file even if empty
            assert Path(filepath).exists()

        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_export_single_channel(self, sample_data):
        """Test exporting single channel."""
        single_channel = sample_data[0:1, :]

        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            filepath = f.name

        try:
            exporter = CSVExporter()
            exporter.export(
                filepath=filepath,
                data=single_channel,
                sample_rate=51200.0,
                channel_names=['CH1'],
                channel_units=['g'],
                config={},
                metadata={}
            )

            assert Path(filepath).exists()

        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_export_many_channels(self):
        """Test exporting many channels."""
        n_channels = 16
        data = np.random.randn(n_channels, 51200)
        channel_names = [f'CH{i}' for i in range(n_channels)]
        channel_units = ['g'] * n_channels

        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            filepath = f.name

        try:
            exporter = CSVExporter()
            exporter.export(
                filepath=filepath,
                data=data,
                sample_rate=51200.0,
                channel_names=channel_names,
                channel_units=channel_units,
                config={},
                metadata={}
            )

            assert Path(filepath).exists()

        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_export_large_data(self):
        """Test exporting large dataset."""
        # 10 seconds of data at 51200 Hz
        data = np.random.randn(4, 512000)

        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            filepath = f.name

        try:
            exporter = HDF5Exporter()
            exporter.export(
                filepath=filepath,
                data=data,
                sample_rate=51200.0,
                channel_names=['CH1', 'CH2', 'CH3', 'CH4'],
                channel_units=['g', 'g', 'g', 'g'],
                config={},
                metadata={}
            )

            assert Path(filepath).exists()

        finally:
            Path(filepath).unlink(missing_ok=True)


@pytest.mark.skipif(not EXPORTERS_AVAILABLE, reason="Export modules not available")
class TestExportFormats:
    """Test different export format support."""

    def test_csv_format_support(self):
        """Test CSV format is supported."""
        exporter = CSVExporter()
        assert exporter.supports_format('csv')

    def test_hdf5_format_support(self):
        """Test HDF5 format is supported."""
        exporter = HDF5Exporter()
        assert exporter.supports_format('hdf5') or exporter.supports_format('h5')

    def test_tdms_format_support(self):
        """Test TDMS format is supported."""
        exporter = TDMSExporter()
        assert exporter.supports_format('tdms')

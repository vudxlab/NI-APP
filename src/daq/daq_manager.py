"""
DAQ Manager for NI Hardware.

This module provides the main interface to NI DAQ hardware for data acquisition.
It handles device enumeration, task creation, configuration, and data reading.
"""

from typing import Optional, List, Dict, Tuple
import time
import numpy as np

try:
    import nidaqmx
    from nidaqmx.constants import AcquisitionType, READ_ALL_AVAILABLE
    from nidaqmx.stream_readers import AnalogMultiChannelReader
    NIDAQMX_AVAILABLE = True
except ImportError:
    NIDAQMX_AVAILABLE = False
    nidaqmx = None
    AcquisitionType = None
    AnalogMultiChannelReader = None
    import warnings
    warnings.warn("nidaqmx not available - running in simulation mode")

from .daq_config import DAQConfig
from .channel_manager import ChannelManager
from ..utils.logger import get_logger
from ..utils.constants import DAQDefaults, ErrorMessages
from ..utils.validators import ValidationError


class DAQError(Exception):
    """Custom exception for DAQ-related errors."""
    pass


class DAQManager:
    """
    Main interface to NI DAQ hardware.

    This class manages:
    - Device enumeration and discovery
    - Task creation and configuration
    - Data acquisition
    - Error handling and recovery
    """

    def __init__(self):
        """Initialize the DAQManager."""
        self.logger = get_logger(__name__)
        self.task: Optional['nidaqmx.Task'] = None
        self.channel_manager = ChannelManager()
        self.config: Optional[DAQConfig] = None
        self._is_running = False
        self._reader: Optional['AnalogMultiChannelReader'] = None
        self._simulation_mode = not NIDAQMX_AVAILABLE

        if self._simulation_mode:
            self.logger.warning("DAQManager initialized in simulation mode (no hardware)")
        else:
            self.logger.info(f"DAQManager initialized (nidaqmx available: {NIDAQMX_AVAILABLE})")

    @staticmethod
    def enumerate_devices() -> List[Dict[str, any]]:
        """
        Enumerate available NI DAQ devices.

        Returns:
            List of dictionaries with device information:
                - name: Device name (e.g., "cDAQ1")
                - product_type: Product type (e.g., "NI-9178")
                - serial_number: Serial number
                - num_channels: Number of analog input channels

        Raises:
            DAQError: If enumeration fails
        """
        logger = get_logger(__name__)

        if not NIDAQMX_AVAILABLE:
            logger.warning("nidaqmx not available, returning simulated device")
            return [{
                'name': 'SimulatedDAQ',
                'product_type': 'Simulated',
                'serial_number': 'SIM-0000',
                'num_channels': 12
            }]

        try:
            system = nidaqmx.system.System.local()
            devices = []

            for device in system.devices:
                device_info = {
                    'name': device.name,
                    'product_type': device.product_type,
                    'serial_number': str(device.dev_serial_num),
                    'num_channels': len(device.ai_physical_chans)
                }
                devices.append(device_info)
                logger.info(f"Found device: {device.name} ({device.product_type})")

            if not devices:
                logger.warning("No NI DAQ devices found")

            return devices

        except Exception as e:
            logger.error(f"Error enumerating devices: {e}")
            raise DAQError(f"Failed to enumerate devices: {e}")

    @staticmethod
    def get_device_modules(device_name: str) -> List[Dict[str, any]]:
        """
        Get list of modules in a cDAQ chassis.
        
        Args:
            device_name: Device name (e.g., "cDAQ1")
        
        Returns:
            List of dictionaries with module information:
                - name: Module name (e.g., "cDAQ1Mod1")
                - product_type: Module type (e.g., "NI-9234")
                - num_channels: Number of channels
        """
        logger = get_logger(__name__)
        
        if not NIDAQMX_AVAILABLE:
            logger.warning("nidaqmx not available, returning simulated modules")
            return [
                {'name': f'{device_name}Mod1', 'product_type': 'NI-9234', 'num_channels': 4},
                {'name': f'{device_name}Mod2', 'product_type': 'NI-9234', 'num_channels': 4},
                {'name': f'{device_name}Mod3', 'product_type': 'NI-9215', 'num_channels': 4}
            ]
        
        try:
            system = nidaqmx.system.System.local()
            device = system.devices[device_name]
            modules = []
            
            # Get chassis modules (if it's a cDAQ chassis)
            if hasattr(device, 'chassis_module_devices'):
                for mod_name in device.chassis_module_devices:
                    try:
                        mod_device = system.devices[mod_name]
                        module_info = {
                            'name': mod_name,
                            'product_type': mod_device.product_type,
                            'num_channels': len(mod_device.ai_physical_chans) if hasattr(mod_device, 'ai_physical_chans') else 0
                        }
                        modules.append(module_info)
                        logger.debug(f"Found module: {mod_name} ({mod_device.product_type})")
                    except Exception as e:
                        logger.warning(f"Could not get info for module {mod_name}: {e}")
            else:
                # Not a chassis, treat as single device
                logger.debug(f"{device_name} is not a chassis device")
            
            return modules
            
        except Exception as e:
            logger.error(f"Error getting modules for {device_name}: {e}")
            return []
    
    @staticmethod
    def get_device_channels(device_name: str) -> List[str]:
        """
        Get list of analog input channels for a device.

        Args:
            device_name: Device name (e.g., "cDAQ1")

        Returns:
            List of physical channel names

        Raises:
            DAQError: If device not found or error occurs
        """
        logger = get_logger(__name__)

        if not NIDAQMX_AVAILABLE:
            # Return simulated channels
            return [f"{device_name}Mod1/ai{i}" for i in range(12)]

        try:
            system = nidaqmx.system.System.local()
            device = system.devices[device_name]

            channels = [chan.name for chan in device.ai_physical_chans]
            logger.info(f"Device {device_name} has {len(channels)} AI channels")

            return channels

        except KeyError:
            raise DAQError(f"Device '{device_name}' not found")
        except Exception as e:
            logger.error(f"Error getting channels for {device_name}: {e}")
            raise DAQError(f"Failed to get device channels: {e}")

    def configure(self, config: DAQConfig) -> None:
        """
        Configure the DAQ with given configuration.

        Args:
            config: DAQ configuration

        Raises:
            DAQError: If configuration fails
            ValidationError: If configuration is invalid
        """
        try:
            # Validate configuration
            config.validate()

            # Store configuration
            self.config = config

            # Configure channel manager
            self.channel_manager.set_channels(config.channels)

            self.logger.info(f"DAQ configured: {config}")

        except ValidationError as e:
            self.logger.error(f"Invalid configuration: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Configuration failed: {e}")
            raise DAQError(f"Failed to configure DAQ: {e}")

    def create_task(self) -> None:
        """
        Create and configure the nidaqmx task.

        Raises:
            DAQError: If task creation fails
        """
        if self.config is None:
            raise DAQError("DAQ not configured. Call configure() first.")

        # Debug: log NIDAQMX_AVAILABLE status
        self.logger.info(f"create_task: NIDAQMX_AVAILABLE={NIDAQMX_AVAILABLE}, _simulation_mode={self._simulation_mode}")
        
        if self._simulation_mode:
            self.logger.info("Simulated task created")
            self._is_running = False
            return

        try:
            # Close existing task if any
            if self.task is not None:
                self.close_task()

            # Debug: Check if nidaqmx is available
            if nidaqmx is None:
                raise DAQError("nidaqmx library not available")
            
            # Create new task
            self.task = nidaqmx.Task()
            self.logger.debug("Created nidaqmx Task")

            # Add channels using channel manager
            self.channel_manager.apply_to_task(self.task, self.config)

            # Configure timing
            self._configure_timing()

            # Create reader for efficient data reading
            self._reader = AnalogMultiChannelReader(self.task.in_stream)

            self.logger.info(
                f"Task created with {self.config.get_num_enabled_channels()} channels "
                f"at {self.config.sample_rate} Hz"
            )

        except Exception as e:
            self.logger.error(f"Failed to create task: {e}")
            if self.task is not None:
                try:
                    self.task.close()
                except:
                    pass
                self.task = None
            raise DAQError(f"Task creation failed: {e}")

    def _configure_timing(self) -> None:
        """
        Configure timing for the DAQ task.

        Raises:
            DAQError: If timing configuration fails
        """
        if self.task is None or self.config is None:
            return

        try:
            # Set up sample clock timing
            if self.config.acquisition_mode == DAQDefaults.ACQUISITION_MODE_CONTINUOUS:
                acquisition_type = AcquisitionType.CONTINUOUS
                samples_per_channel = self.config.samples_per_channel
            else:
                acquisition_type = AcquisitionType.FINITE
                samples_per_channel = self.config.samples_per_channel

            self.task.timing.cfg_samp_clk_timing(
                rate=self.config.sample_rate,
                sample_mode=acquisition_type,
                samps_per_chan=samples_per_channel
            )

            self.logger.debug(
                f"Configured timing: {self.config.sample_rate} Hz, "
                f"mode={self.config.acquisition_mode}"
            )

        except Exception as e:
            self.logger.error(f"Failed to configure timing: {e}")
            raise DAQError(f"Timing configuration failed: {e}")

    def start_acquisition(self) -> None:
        """
        Start data acquisition.

        Raises:
            DAQError: If acquisition cannot be started
        """
        if self.config is None:
            raise DAQError("DAQ not configured")

        if self._simulation_mode:
            self._is_running = True
            self.logger.info("Simulated acquisition started")
            return

        if self.task is None:
            raise DAQError("Task not created. Call create_task() first.")

        try:
            self.task.start()
            self._is_running = True
            self.logger.info("Acquisition started")

        except Exception as e:
            self._is_running = False
            self.logger.error(f"Failed to start acquisition: {e}")
            raise DAQError(f"Failed to start acquisition: {e}")

    def stop_acquisition(self) -> None:
        """Stop data acquisition."""
        if self._simulation_mode:
            self._is_running = False
            self.logger.info("Simulated acquisition stopped")
            return

        if self.task is not None and self._is_running:
            try:
                self.task.stop()
                self._is_running = False
                self.logger.info("Acquisition stopped")
            except Exception as e:
                self.logger.error(f"Error stopping acquisition: {e}")

    def read_samples(
        self,
        num_samples: Optional[int] = None,
        timeout: float = DAQDefaults.READ_TIMEOUT
    ) -> np.ndarray:
        """
        Read samples from the DAQ.

        Args:
            num_samples: Number of samples to read per channel.
                        If None, uses config.samples_per_channel
            timeout: Read timeout in seconds

        Returns:
            NumPy array of shape (n_channels, n_samples) with voltage data

        Raises:
            DAQError: If read fails or timeout occurs
        """
        if self.config is None:
            raise DAQError("DAQ not configured")

        if self._simulation_mode:
            return self._read_simulated_samples(num_samples)

        if self.task is None or not self._is_running:
            raise DAQError("Acquisition not running")

        if num_samples is None:
            num_samples = self.config.samples_per_channel

        try:
            # Create buffer for reading
            n_channels = self.config.get_num_enabled_channels()
            data = np.zeros((n_channels, num_samples), dtype=np.float64)

            # Read data using the reader
            samples_read = self._reader.read_many_sample(
                data,
                number_of_samples_per_channel=num_samples,
                timeout=timeout
            )

            if samples_read == 0:
                raise DAQError("No samples read (timeout or no data available)")

            # Return only the samples that were actually read
            if samples_read < num_samples:
                data = data[:, :samples_read]

            return data

        except Exception as e:
            self.logger.error(f"Error reading samples: {e}")
            raise DAQError(f"Failed to read samples: {e}")

    def _read_simulated_samples(self, num_samples: Optional[int] = None) -> np.ndarray:
        """
        Generate simulated data for testing without hardware.

        Args:
            num_samples: Number of samples per channel

        Returns:
            Simulated data array
        """
        if num_samples is None:
            num_samples = self.config.samples_per_channel

        n_channels = self.config.get_num_enabled_channels()

        # Generate simulated sinusoidal data with some noise
        t = np.arange(num_samples) / self.config.sample_rate
        data = np.zeros((n_channels, num_samples))

        for i in range(n_channels):
            # Each channel has different frequency content
            freq1 = 100 + i * 50  # Base frequency
            freq2 = 500 + i * 100  # Harmonic

            signal = (
                0.5 * np.sin(2 * np.pi * freq1 * t) +
                0.2 * np.sin(2 * np.pi * freq2 * t) +
                0.05 * np.random.randn(num_samples)
            )

            data[i, :] = signal

        # Simulate acquisition delay
        time.sleep(0.001)

        return data

    def read_scaled_samples(
        self,
        num_samples: Optional[int] = None,
        timeout: float = DAQDefaults.READ_TIMEOUT
    ) -> np.ndarray:
        """
        Read samples and apply scaling to engineering units.

        Args:
            num_samples: Number of samples to read per channel
            timeout: Read timeout in seconds

        Returns:
            NumPy array with data in engineering units (g, m/s², etc.)

        Raises:
            DAQError: If read fails
        """
        # Read raw voltage data
        raw_data = self.read_samples(num_samples, timeout)

        # Apply scaling
        scaled_data = self.channel_manager.scale_data(raw_data)

        return scaled_data

    def close_task(self) -> None:
        """Close and cleanup the DAQ task."""
        if self.task is not None:
            try:
                if self._is_running:
                    self.stop_acquisition()

                self.task.close()
                self.logger.info("Task closed")

            except Exception as e:
                self.logger.error(f"Error closing task: {e}")

            finally:
                self.task = None
                self._reader = None
                self._is_running = False

    def is_running(self) -> bool:
        """
        Check if acquisition is currently running.

        Returns:
            True if acquisition is running
        """
        return self._is_running

    def get_status(self) -> Dict[str, any]:
        """
        Get current DAQ status.

        Returns:
            Dictionary with status information
        """
        status = {
            'configured': self.config is not None,
            'task_created': self.task is not None,
            'running': self._is_running,
            'simulation_mode': self._simulation_mode,
        }

        if self.config is not None:
            status.update({
                'device_name': self.config.device_name,
                'sample_rate': self.config.sample_rate,
                'num_channels': self.config.get_num_enabled_channels(),
                'acquisition_mode': self.config.acquisition_mode
            })

        return status

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup."""
        self.close_task()

    def __repr__(self) -> str:
        """String representation."""
        if self.config:
            return (
                f"DAQManager(device='{self.config.device_name}', "
                f"running={self._is_running}, "
                f"channels={self.config.get_num_enabled_channels()})"
            )
        else:
            return "DAQManager(not configured)"


# Example usage
if __name__ == "__main__":
    from .daq_config import create_default_config

    print("NI DAQ Manager Test")
    print("=" * 60)

    # Enumerate devices
    print("\n1. Enumerating devices...")
    try:
        devices = DAQManager.enumerate_devices()
        print(f"Found {len(devices)} device(s):")
        for dev in devices:
            print(f"  - {dev['name']}: {dev['product_type']} "
                  f"({dev['num_channels']} channels)")
    except DAQError as e:
        print(f"Error: {e}")

    # Create configuration
    print("\n2. Creating configuration...")
    config = create_default_config(device_name="cDAQ1", num_modules=3)
    print(f"Config: {config}")

    # Create DAQ manager
    print("\n3. Creating DAQ manager...")
    with DAQManager() as daq:
        print(f"DAQ Manager: {daq}")

        # Configure
        print("\n4. Configuring DAQ...")
        try:
            daq.configure(config)
            print("Configuration successful")
        except (DAQError, ValidationError) as e:
            print(f"Configuration error: {e}")

        # Create task
        print("\n5. Creating task...")
        try:
            daq.create_task()
            print("Task created successfully")
        except DAQError as e:
            print(f"Task creation error: {e}")

        # Get status
        print("\n6. DAQ Status:")
        status = daq.get_status()
        for key, value in status.items():
            print(f"  {key}: {value}")

        # Start acquisition
        print("\n7. Starting acquisition...")
        try:
            daq.start_acquisition()
            print("Acquisition started")

            # Read some samples
            print("\n8. Reading samples...")
            for i in range(3):
                data = daq.read_scaled_samples(num_samples=100)
                print(f"  Read {i+1}: shape={data.shape}, "
                      f"range=[{data.min():.3f}, {data.max():.3f}]")

        except DAQError as e:
            print(f"Acquisition error: {e}")
        finally:
            # Stop acquisition
            print("\n9. Stopping acquisition...")
            daq.stop_acquisition()
            print("Acquisition stopped")

    print("\n10. Cleanup complete (context manager)")
    print("=" * 60)

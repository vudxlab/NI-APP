"""
Channel Configuration Widget.

This widget provides a table for configuring individual channel settings
including enable/disable, name, coupling, IEPE, sensitivity, and units.
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QPushButton, QHeaderView, QComboBox, QDoubleSpinBox, QCheckBox,
    QAbstractItemView
)
from PyQt5.QtCore import Qt, pyqtSignal
from typing import List

from ...daq.daq_config import ChannelConfig
from ...utils.logger import get_logger
from ...utils.constants import NI9234Specs, ChannelDefaults


class ChannelConfigWidget(QWidget):
    """
    Channel configuration table widget.

    Provides a table with rows for each channel and columns for:
    - Enable/Disable checkbox
    - Channel name
    - Physical channel (read-only)
    - Coupling (AC/DC)
    - IEPE enable
    - Sensitivity (mV/g)
    - Units (g, m/s², mm/s²)
    """

    # Signals
    channels_changed = pyqtSignal(list)  # List of ChannelConfig

    # Column indices
    COL_ENABLED = 0
    COL_NAME = 1
    COL_PHYSICAL = 2
    COL_COUPLING = 3
    COL_IEPE = 4
    COL_SENSITIVITY = 5
    COL_UNITS = 6

    def __init__(self, parent=None):
        """
        Initialize the channel configuration widget.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)

        self.logger = get_logger(__name__)
        self.channels: List[ChannelConfig] = []

        self._init_ui()

    def _init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Create table
        self.table = QTableWidget()
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels([
            "Enabled",
            "Name",
            "Physical Channel",
            "Coupling",
            "IEPE",
            "Sensitivity\n(mV/g)",
            "Units"
        ])

        # Table properties
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)

        # Column resize modes
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(self.COL_ENABLED, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(self.COL_NAME, QHeaderView.Stretch)
        header.setSectionResizeMode(self.COL_PHYSICAL, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(self.COL_COUPLING, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(self.COL_IEPE, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(self.COL_SENSITIVITY, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(self.COL_UNITS, QHeaderView.ResizeToContents)

        layout.addWidget(self.table)

        # Button row
        button_layout = QHBoxLayout()

        self.enable_all_button = QPushButton("Enable All")
        self.enable_all_button.clicked.connect(self._on_enable_all)
        button_layout.addWidget(self.enable_all_button)

        self.disable_all_button = QPushButton("Disable All")
        self.disable_all_button.clicked.connect(self._on_disable_all)
        button_layout.addWidget(self.disable_all_button)

        button_layout.addStretch()

        self.apply_button = QPushButton("Apply Changes")
        self.apply_button.clicked.connect(self._on_apply_changes)
        button_layout.addWidget(self.apply_button)

        layout.addLayout(button_layout)

    def set_channels(self, channels: List[ChannelConfig]):
        """
        Set channel configurations.

        Args:
            channels: List of ChannelConfig objects
        """
        self.channels = channels
        self._populate_table()

    def _populate_table(self):
        """Populate table with channel configurations."""
        self.table.setRowCount(len(self.channels))

        for row, channel in enumerate(self.channels):
            # Enabled checkbox
            enabled_widget = QWidget()
            enabled_layout = QHBoxLayout(enabled_widget)
            enabled_layout.setContentsMargins(0, 0, 0, 0)
            enabled_layout.setAlignment(Qt.AlignCenter)

            enabled_checkbox = QCheckBox()
            enabled_checkbox.setChecked(channel.enabled)
            enabled_checkbox.stateChanged.connect(
                lambda state, r=row: self._on_enabled_changed(r, state)
            )
            enabled_layout.addWidget(enabled_checkbox)

            self.table.setCellWidget(row, self.COL_ENABLED, enabled_widget)

            # Name (editable)
            name_item = QTableWidgetItem(channel.name)
            self.table.setItem(row, self.COL_NAME, name_item)

            # Physical channel (read-only)
            physical_item = QTableWidgetItem(channel.physical_channel)
            physical_item.setFlags(physical_item.flags() & ~Qt.ItemIsEditable)
            physical_item.setForeground(Qt.gray)
            self.table.setItem(row, self.COL_PHYSICAL, physical_item)

            # Coupling combo box
            coupling_combo = QComboBox()
            coupling_combo.addItems(NI9234Specs.COUPLING_MODES)
            coupling_combo.setCurrentText(channel.coupling)
            coupling_combo.currentTextChanged.connect(
                lambda text, r=row: self._on_coupling_changed(r, text)
            )
            self.table.setCellWidget(row, self.COL_COUPLING, coupling_combo)

            # IEPE checkbox
            iepe_widget = QWidget()
            iepe_layout = QHBoxLayout(iepe_widget)
            iepe_layout.setContentsMargins(0, 0, 0, 0)
            iepe_layout.setAlignment(Qt.AlignCenter)

            iepe_checkbox = QCheckBox()
            iepe_checkbox.setChecked(channel.iepe_enabled)
            iepe_checkbox.stateChanged.connect(
                lambda state, r=row: self._on_iepe_changed(r, state)
            )
            iepe_layout.addWidget(iepe_checkbox)

            self.table.setCellWidget(row, self.COL_IEPE, iepe_widget)

            # Sensitivity spin box
            sensitivity_spin = QDoubleSpinBox()
            sensitivity_spin.setRange(1.0, 10000.0)
            sensitivity_spin.setDecimals(2)
            sensitivity_spin.setValue(channel.sensitivity)
            sensitivity_spin.setSuffix(" mV/g")
            sensitivity_spin.valueChanged.connect(
                lambda value, r=row: self._on_sensitivity_changed(r, value)
            )
            self.table.setCellWidget(row, self.COL_SENSITIVITY, sensitivity_spin)

            # Units combo box
            units_combo = QComboBox()
            units_combo.addItems(ChannelDefaults.SUPPORTED_UNITS)
            units_combo.setCurrentText(channel.units)
            units_combo.currentTextChanged.connect(
                lambda text, r=row: self._on_units_changed(r, text)
            )
            self.table.setCellWidget(row, self.COL_UNITS, units_combo)

        self.logger.debug(f"Populated table with {len(self.channels)} channels")

    def _on_enabled_changed(self, row: int, state: int):
        """Handle enabled checkbox change."""
        if row < len(self.channels):
            self.channels[row].enabled = (state == Qt.Checked)

    def _on_coupling_changed(self, row: int, text: str):
        """Handle coupling combo box change."""
        if row < len(self.channels):
            self.channels[row].coupling = text

    def _on_iepe_changed(self, row: int, state: int):
        """Handle IEPE checkbox change."""
        if row < len(self.channels):
            self.channels[row].iepe_enabled = (state == Qt.Checked)

    def _on_sensitivity_changed(self, row: int, value: float):
        """Handle sensitivity spin box change."""
        if row < len(self.channels):
            self.channels[row].sensitivity = value

    def _on_units_changed(self, row: int, text: str):
        """Handle units combo box change."""
        if row < len(self.channels):
            self.channels[row].units = text

    def _on_enable_all(self):
        """Enable all channels."""
        for row in range(self.table.rowCount()):
            widget = self.table.cellWidget(row, self.COL_ENABLED)
            if widget:
                checkbox = widget.findChild(QCheckBox)
                if checkbox:
                    checkbox.setChecked(True)

        self.logger.debug("All channels enabled")

    def _on_disable_all(self):
        """Disable all channels."""
        for row in range(self.table.rowCount()):
            widget = self.table.cellWidget(row, self.COL_ENABLED)
            if widget:
                checkbox = widget.findChild(QCheckBox)
                if checkbox:
                    checkbox.setChecked(False)

        self.logger.debug("All channels disabled")

    def _on_apply_changes(self):
        """Apply changes and emit signal."""
        # Update channel names from table
        for row in range(self.table.rowCount()):
            if row < len(self.channels):
                name_item = self.table.item(row, self.COL_NAME)
                if name_item:
                    self.channels[row].name = name_item.text()

        # Emit signal
        self.channels_changed.emit(self.channels)
        self.logger.info("Channel configuration changes applied")

    def get_channels(self) -> List[ChannelConfig]:
        """
        Get current channel configurations.

        Returns:
            List of ChannelConfig objects
        """
        # Update channel names from table
        for row in range(self.table.rowCount()):
            if row < len(self.channels):
                name_item = self.table.item(row, self.COL_NAME)
                if name_item:
                    self.channels[row].name = name_item.text()

        return self.channels

    def get_enabled_channel_count(self) -> int:
        """
        Get number of enabled channels.

        Returns:
            Number of enabled channels
        """
        return sum(1 for ch in self.channels if ch.enabled)


# Example usage
if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication
    from ...daq.daq_config import create_default_config

    app = QApplication(sys.argv)

    # Create widget
    widget = ChannelConfigWidget()

    # Create some test channels
    config = create_default_config(device_name="cDAQ1", num_modules=3)
    widget.set_channels(config.channels)

    # Connect signal
    widget.channels_changed.connect(
        lambda channels: print(f"Channels changed: {len(channels)} channels, "
                              f"{sum(1 for ch in channels if ch.enabled)} enabled")
    )

    widget.show()

    sys.exit(app.exec_())

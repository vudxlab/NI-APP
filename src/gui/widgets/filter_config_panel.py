"""
Filter Configuration Panel.

This widget provides controls for configuring digital filters
including filter type, mode, cutoff frequencies, and order.
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QLabel, QComboBox, QSpinBox, QDoubleSpinBox, QPushButton,
    QCheckBox, QTabWidget, QScrollArea, QGridLayout, QSpacerItem,
    QSizePolicy
)
from PyQt5.QtCore import pyqtSignal, Qt

try:
    import pyqtgraph as pg
    PYQTGRAPH_AVAILABLE = True
except ImportError:
    PYQTGRAPH_AVAILABLE = False

from ...utils.logger import get_logger
from ...utils.constants import ProcessingDefaults


class FilterConfigPanel(QWidget):
    """
    Filter configuration panel widget.

    Provides controls for:
    - Filter type (Butterworth, Chebyshev I/II, Bessel)
    - Filter mode (lowpass, highpass, bandpass, bandstop)
    - Cutoff frequency/ies
    - Filter order
    - Enable/disable filtering
    """

    # Signals
    filter_changed = pyqtSignal(dict)  # Emit filter configuration
    filter_enabled = pyqtSignal(bool)  # Emit enable/disable state

    def __init__(self, parent=None):
        """
        Initialize the filter configuration panel.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)

        self.logger = get_logger(__name__)

        # Current filter configuration
        self.filter_config = {
            'type': ProcessingDefaults.DEFAULT_FILTER_TYPE,
            'mode': ProcessingDefaults.FILTER_MODE_HIGHPASS,
            'cutoff_low': 0.1,
            'cutoff_high': 100.0,
            'order': ProcessingDefaults.DEFAULT_FILTER_ORDER,
            'enabled': False
        }

        self._init_ui()

    def _init_ui(self):
        """Initialize the user interface."""
        # Create main layout for the scroll area
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Create scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # Create content widget to hold all groups
        content_widget = QWidget()
        layout = QGridLayout(content_widget)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setHorizontalSpacing(12)
        layout.setVerticalSpacing(8)

        # Filter type selection
        type_group = self._create_type_group()
        layout.addWidget(type_group, 0, 0)

        # Filter mode selection
        mode_group = self._create_mode_group()
        layout.addWidget(mode_group, 1, 0)

        # Cutoff frequency group
        cutoff_group = self._create_cutoff_group()
        layout.addWidget(cutoff_group, 0, 1)

        # Filter order group
        order_group = self._create_order_group()
        layout.addWidget(order_group, 1, 1)

        # Set equal column stretches for both columns
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 1)

        # Add stretch to push everything to top
        layout.addItem(
            QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding),
            2,
            0,
            1,
            2
        )

        # Set content widget to scroll area
        scroll_area.setWidget(content_widget)

        # Add scroll area to main layout
        main_layout.addWidget(scroll_area)

        # Initialize UI state with defaults
        self._update_cutoff_ui()
        self._update_type_description()
        self._update_mode_description()


    def _create_type_group(self) -> QGroupBox:
        """Create filter type selection group."""
        group = QGroupBox("Filter Type")
        layout = QVBoxLayout()

        self.type_combo = QComboBox()
        self.type_combo.addItem("Butterworth", ProcessingDefaults.FILTER_TYPE_BUTTERWORTH)
        self.type_combo.addItem("Chebyshev Type I", ProcessingDefaults.FILTER_TYPE_CHEBYSHEV1)
        self.type_combo.addItem("Chebyshev Type II", ProcessingDefaults.FILTER_TYPE_CHEBYSHEV2)
        self.type_combo.addItem("Bessel", ProcessingDefaults.FILTER_TYPE_BESSEL)

        # Add tooltips
        self.type_combo.setItemData(0, "Flat passband, good all-around filter", Qt.ToolTipRole)
        self.type_combo.setItemData(1, "Ripple in passband, steep rolloff", Qt.ToolTipRole)
        self.type_combo.setItemData(2, "Ripple in stopband, steep rolloff", Qt.ToolTipRole)
        self.type_combo.setItemData(3, "Linear phase, good for pulses", Qt.ToolTipRole)

        self.type_combo.currentIndexChanged.connect(self._on_parameter_changed)
        layout.addWidget(self.type_combo)

        # Description label
        self.type_description = QLabel()
        self.type_description.setWordWrap(True)
        self.type_description.setStyleSheet("QLabel { color: #666666; font-size: 10px; }")
        layout.addWidget(self.type_description)

        # Update description
        self._update_type_description()

        group.setLayout(layout)
        return group

    def _create_mode_group(self) -> QGroupBox:
        """Create filter mode selection group."""
        group = QGroupBox("Filter Mode")
        layout = QVBoxLayout()

        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Lowpass", ProcessingDefaults.FILTER_MODE_LOWPASS)
        self.mode_combo.addItem("Highpass", ProcessingDefaults.FILTER_MODE_HIGHPASS)
        self.mode_combo.addItem("Bandpass", ProcessingDefaults.FILTER_MODE_BANDPASS)
        self.mode_combo.addItem("Bandstop (Notch)", ProcessingDefaults.FILTER_MODE_BANDSTOP)

        # Set default to Highpass
        self.mode_combo.setCurrentIndex(1)

        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        layout.addWidget(self.mode_combo)

        # Mode description
        self.mode_description = QLabel()
        self.mode_description.setWordWrap(True)
        self.mode_description.setStyleSheet("QLabel { color: #666666; font-size: 10px; }")
        layout.addWidget(self.mode_description)

        # Update description
        self._update_mode_description()

        group.setLayout(layout)
        return group

    def _create_cutoff_group(self) -> QGroupBox:
        """Create cutoff frequency group."""
        group = QGroupBox("Cutoff Frequency")
        layout = QFormLayout()

        # Low/Single cutoff
        self.cutoff_low_spin = QDoubleSpinBox()
        self.cutoff_low_spin.setRange(0.001, 25000.0)
        self.cutoff_low_spin.setDecimals(3)
        self.cutoff_low_spin.setValue(0.1)
        self.cutoff_low_spin.setSuffix(" Hz")
        self.cutoff_low_spin.setSingleStep(0.001)
        self.cutoff_low_spin.valueChanged.connect(self._on_parameter_changed)

        self.cutoff_low_label = QLabel("Cutoff:")
        layout.addRow(self.cutoff_low_label, self.cutoff_low_spin)

        # High cutoff (for bandpass/bandstop)
        self.cutoff_high_spin = QDoubleSpinBox()
        self.cutoff_high_spin.setRange(0.001, 25000.0)
        self.cutoff_high_spin.setDecimals(3)
        self.cutoff_high_spin.setValue(100.0)
        self.cutoff_high_spin.setSuffix(" Hz")
        self.cutoff_high_spin.setSingleStep(0.001)
        self.cutoff_high_spin.valueChanged.connect(self._on_parameter_changed)
        self.cutoff_high_spin.setEnabled(False)  # Only for bandpass/bandstop

        self.cutoff_high_label = QLabel("Upper:")
        layout.addRow(self.cutoff_high_label, self.cutoff_high_spin)

        # Info label
        self.cutoff_info = QLabel()
        self.cutoff_info.setWordWrap(True)
        self.cutoff_info.setStyleSheet("QLabel { color: #999999; font-size: 9px; }")
        layout.addRow(self.cutoff_info)

        group.setLayout(layout)
        return group

    def _create_order_group(self) -> QGroupBox:
        """Create filter order group."""
        group = QGroupBox("Filter Order")
        layout = QVBoxLayout()

        # Order spin box
        order_layout = QHBoxLayout()
        order_layout.addWidget(QLabel("Order:"))

        self.order_spin = QSpinBox()
        self.order_spin.setRange(1, 20)
        self.order_spin.setValue(ProcessingDefaults.DEFAULT_FILTER_ORDER)
        self.order_spin.valueChanged.connect(self._on_parameter_changed)
        order_layout.addWidget(self.order_spin)
        order_layout.addStretch()

        layout.addLayout(order_layout)

        # Order description
        order_desc = QLabel()
        order_desc.setWordWrap(True)
        order_desc.setText(
            "Higher orders = steeper rolloff but may cause instability. "
            "Recommended: 4-8 for most applications."
        )
        order_desc.setStyleSheet("QLabel { color: #666666; font-size: 9px; }")
        layout.addWidget(order_desc)

        group.setLayout(layout)
        return group

    def _update_type_description(self):
        """Update filter type description label."""
        descriptions = {
            ProcessingDefaults.FILTER_TYPE_BUTTERWORTH:
                "Flat passband with no ripple. Good all-around filter.",
            ProcessingDefaults.FILTER_TYPE_CHEBYSHEV1:
                "Ripple in passband, steeper rolloff than Butterworth.",
            ProcessingDefaults.FILTER_TYPE_CHEBYSHEV2:
                "Ripple in stopband, steeper rolloff than Butterworth.",
            ProcessingDefaults.FILTER_TYPE_BESSEL:
                "Linear phase response, good for preserving wave shapes."
        }

        filter_type = self.type_combo.currentData()
        self.type_description.setText(descriptions.get(filter_type, ""))

    def _update_mode_description(self):
        """Update filter mode description label."""
        descriptions = {
            ProcessingDefaults.FILTER_MODE_LOWPASS:
                "Passes frequencies below cutoff. Use to remove high-frequency noise.",
            ProcessingDefaults.FILTER_MODE_HIGHPASS:
                "Passes frequencies above cutoff. Use to remove DC offset and low-frequency drift.",
            ProcessingDefaults.FILTER_MODE_BANDPASS:
                "Passes frequencies between cutoffs. Use to isolate specific frequency band.",
            ProcessingDefaults.FILTER_MODE_BANDSTOP:
                "Attenuates frequencies between cutoffs. Use to remove specific frequency (notch filter)."
        }

        filter_mode = self.mode_combo.currentData()
        self.mode_description.setText(descriptions.get(filter_mode, ""))

    def _update_cutoff_ui(self):
        """Update cutoff frequency UI based on filter mode."""
        filter_mode = self.mode_combo.currentData()

        if filter_mode in [ProcessingDefaults.FILTER_MODE_BANDPASS,
                           ProcessingDefaults.FILTER_MODE_BANDSTOP]:
            # Need two cutoffs
            self.cutoff_low_label.setText("Lower Cutoff:")
            self.cutoff_high_spin.setEnabled(True)
            self.cutoff_info.setText("Set lower and upper frequency limits")
        else:
            # Need single cutoff
            if filter_mode == ProcessingDefaults.FILTER_MODE_LOWPASS:
                self.cutoff_low_label.setText("Cutoff:")
                self.cutoff_info.setText("Frequencies below this value will be passed")
            else:  # highpass
                self.cutoff_low_label.setText("Cutoff:")
                self.cutoff_info.setText("Frequencies above this value will be passed")

            self.cutoff_high_spin.setEnabled(False)

    def _on_parameter_changed(self):
        """Handle parameter change."""
        # Update descriptions
        self._update_type_description()

    def _on_mode_changed(self):
        """Handle filter mode change."""
        self._update_cutoff_ui()
        self._update_mode_description()
        self._on_parameter_changed()

    def get_filter_config(self) -> dict:
        """
        Get current filter configuration.

        Returns:
            Dictionary with filter configuration
        """
        return self.filter_config.copy()

    def get_current_ui_config(self) -> dict:
        """
        Build filter configuration from current UI values.

        Returns:
            Dictionary with filter configuration
        """
        config = {
            'type': self.type_combo.currentData(),
            'mode': self.mode_combo.currentData(),
            'cutoff_low': self.cutoff_low_spin.value(),
            'cutoff_high': self.cutoff_high_spin.value(),
            'order': self.order_spin.value(),
            'enabled': True  # Always enabled when OK is clicked
        }

        if config['mode'] in [ProcessingDefaults.FILTER_MODE_BANDPASS,
                              ProcessingDefaults.FILTER_MODE_BANDSTOP]:
            config['cutoff'] = (config['cutoff_low'], config['cutoff_high'])
        else:
            config['cutoff'] = config['cutoff_low']

        return config

    def set_filter_config(self, config: dict):
        """
        Set filter configuration.

        Args:
            config: Filter configuration dictionary
        """
        current_enabled = self.filter_config.get('enabled', False)
        self.filter_config = config.copy()

        # Block signals during update
        self.type_combo.blockSignals(True)
        self.mode_combo.blockSignals(True)
        self.cutoff_low_spin.blockSignals(True)
        self.cutoff_high_spin.blockSignals(True)
        self.order_spin.blockSignals(True)

        try:
            # Set values
            type_idx = self.type_combo.findData(config.get('type'))
            if type_idx >= 0:
                self.type_combo.setCurrentIndex(type_idx)

            mode_idx = self.mode_combo.findData(config.get('mode'))
            if mode_idx >= 0:
                self.mode_combo.setCurrentIndex(mode_idx)

            self.cutoff_low_spin.setValue(config.get('cutoff_low', 0.1))
            self.cutoff_high_spin.setValue(config.get('cutoff_high', 100.0))
            self.order_spin.setValue(config.get('order', ProcessingDefaults.DEFAULT_FILTER_ORDER))

        finally:
            # Unblock signals
            self.type_combo.blockSignals(False)
            self.mode_combo.blockSignals(False)
            self.cutoff_low_spin.blockSignals(False)
            self.cutoff_high_spin.blockSignals(False)
            self.order_spin.blockSignals(False)

        # Update UI state
        self._update_cutoff_ui()
        self._update_type_description()
        self._update_mode_description()

        self.logger.debug(f"Filter config set: {config}")


# Example usage
if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)

    # Create panel
    panel = FilterConfigPanel()

    # Connect signals
    panel.filter_changed.connect(
        lambda config: print(f"Filter changed: {config}")
    )
    panel.filter_enabled.connect(
        lambda enabled: print(f"Filter {'enabled' if enabled else 'disabled'}")
    )

    panel.show()

    sys.exit(app.exec_())

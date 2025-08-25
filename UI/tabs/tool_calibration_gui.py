from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
                             QGridLayout, QLabel, QPushButton, QSpinBox,
                             QTextEdit)
from PyQt5.QtCore import QTimer
from UI.utils.ui_helpers import show_message_box


class ToolCalibrationTab(QWidget):
    def __init__(self, server_client, config_manager):
        super().__init__()
        self.server_client = server_client
        self.config_manager = config_manager

        self.init_ui()
        self.setup_timers()

    def init_ui(self):
        layout = QVBoxLayout()

        # Tool calibration controls
        tool_group = self.create_tool_calibration_group()
        layout.addWidget(tool_group)

        # Tool status
        self.tool_status_text = QTextEdit()
        self.tool_status_text.setMaximumHeight(150)
        self.tool_status_text.setReadOnly(True)
        layout.addWidget(QLabel("Tool Calibration Status:"))
        layout.addWidget(self.tool_status_text)

        self.setLayout(layout)

    def create_tool_calibration_group(self):
        group = QGroupBox("Tool Calibration")
        layout = QGridLayout()

        # Device selection (based on config tool types)
        layout.addWidget(QLabel("Device:"), 0, 0)
        self.device_spinbox = QSpinBox()
        self.device_spinbox.setRange(0, 5)

        # Set default from config
        tool_types = self.config_manager.get_tool_types()
        if tool_types:
            probe_idx = tool_types.get('probe', 0)
            self.device_spinbox.setValue(probe_idx)
        layout.addWidget(self.device_spinbox, 0, 1)

        # Start/Stop calibration
        self.start_calib_btn = QPushButton("Start Tool Calibration")
        self.start_calib_btn.clicked.connect(self.start_tool_calibration)
        layout.addWidget(self.start_calib_btn, 1, 0, 1, 2)

        self.stop_calib_btn = QPushButton("Stop Tool Calibration")
        self.stop_calib_btn.clicked.connect(self.stop_tool_calibration)
        self.stop_calib_btn.setEnabled(False)
        layout.addWidget(self.stop_calib_btn, 2, 0, 1, 2)

        # Calibrate tool
        self.calibrate_btn = QPushButton("Calibrate Tool")
        self.calibrate_btn.clicked.connect(self.calibrate_tool)
        layout.addWidget(self.calibrate_btn, 3, 0, 1, 2)

        group.setLayout(layout)
        return group

    def setup_timers(self):
        # Status update timer (inactive by default)
        self.status_timer = None

    def start_tool_calibration(self):
        device = self.device_spinbox.value()
        result = self.server_client.start_tool_calibration(device)

        if result.get('status') == 'success':
            self.start_calib_btn.setEnabled(False)
            self.stop_calib_btn.setEnabled(True)
            show_message_box(self, "Success", "Tool calibration started")

            # Start status update timer
            self.status_timer = QTimer()
            self.status_timer.timeout.connect(self.update_tool_status)
            self.status_timer.start(1000)  # Update every second
        else:
            show_message_box(self, "Error", result.get('details', 'Failed to start calibration'), "error")

    def stop_tool_calibration(self):
        result = self.server_client.stop_tool_calibration()

        if result.get('status') == 'success':
            self.start_calib_btn.setEnabled(True)
            self.stop_calib_btn.setEnabled(False)
            if self.status_timer:
                self.status_timer.stop()
            show_message_box(self, "Success", "Tool calibration stopped")
            self.update_tool_status()
        else:
            show_message_box(self, "Error", result.get('details', 'Failed to stop calibration'), "error")

    def calibrate_tool(self):
        result = self.server_client.calibrate_tool()

        if result.get('status') == 'success':
            show_message_box(self, "Success", "Tool calibration completed successfully")
            self.update_tool_status()
        else:
            show_message_box(self, "Error", result.get('details', 'Calibration failed'), "error")

    def update_tool_status(self):
        status = self.server_client.get_tool_calibration_status()

        if status:
            status_text = f"""Calibration Active: {status.get('calibration_active', False)}
Matrices Collected: {status.get('matrices_collected', 0)}
Tool Tip Vector: {status.get('tool_tip_vector', 'Not calibrated')}"""
            self.tool_status_text.setText(status_text.strip())
        else:
            self.tool_status_text.setText("Error getting tool calibration status")
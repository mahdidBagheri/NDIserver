from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
                             QPushButton, QLabel, QSpinBox, QTextEdit,
                             QCheckBox, QProgressBar, QLCDNumber)
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QPalette
import json


class StreamingWidget(QWidget):
    position_updated = pyqtSignal(dict)

    def __init__(self, api_client):
        super().__init__()
        self.api_client = api_client
        self.streaming_active = False
        self.setup_ui()
        self.setup_connections()

    def setup_ui(self):
        layout = QVBoxLayout(self)

        # Streaming configuration group
        config_group = QGroupBox("Streaming Configuration")
        config_layout = QVBoxLayout(config_group)

        # Port and frequency settings
        settings_layout = QHBoxLayout()

        settings_layout.addWidget(QLabel("UDP Port:"))
        self.port_spin = QSpinBox()
        self.port_spin.setRange(1024, 65535)
        self.port_spin.setValue(11111)
        settings_layout.addWidget(self.port_spin)

        settings_layout.addWidget(QLabel("Streaming Frequency:"))
        self.frequency_spin = QSpinBox()
        self.frequency_spin.setRange(1, 100)
        self.frequency_spin.setValue(30)
        self.frequency_spin.setSuffix(" Hz")
        settings_layout.addWidget(self.frequency_spin)

        settings_layout.addWidget(QLabel("Raw Frequency:"))
        self.raw_frequency_spin = QSpinBox()
        self.raw_frequency_spin.setRange(5, 60)
        self.raw_frequency_spin.setValue(10)
        self.raw_frequency_spin.setSuffix(" Hz")
        settings_layout.addWidget(self.raw_frequency_spin)

        settings_layout.addStretch()
        config_layout.addLayout(settings_layout)

        # Force stop option
        force_layout = QHBoxLayout()
        self.force_stop_calibration_cb = QCheckBox("Force Stop Tool Calibration")
        self.force_stop_calibration_cb.setChecked(True)
        force_layout.addWidget(self.force_stop_calibration_cb)
        force_layout.addStretch()
        config_layout.addLayout(force_layout)

        layout.addWidget(config_group)

        # Streaming controls group
        controls_group = QGroupBox("Streaming Controls")
        controls_layout = QVBoxLayout(controls_group)

        # Main control buttons
        button_layout = QHBoxLayout()
        self.start_streaming_btn = QPushButton("Start Streaming")
        self.start_streaming_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }")

        self.stop_streaming_btn = QPushButton("Stop Streaming")
        self.stop_streaming_btn.setStyleSheet("QPushButton { background-color: #f44336; color: white; }")
        self.stop_streaming_btn.setEnabled(False)

        button_layout.addWidget(self.start_streaming_btn)
        button_layout.addWidget(self.stop_streaming_btn)
        button_layout.addStretch()

        controls_layout.addLayout(button_layout)

        # Raw streaming controls
        raw_controls_layout = QHBoxLayout()
        raw_controls_layout.addWidget(QLabel("Raw Streaming:"))

        self.start_raw_btn = QPushButton("Start Raw")
        self.stop_raw_btn = QPushButton("Stop Raw")
        self.set_raw_freq_btn = QPushButton("Set Raw Frequency")

        raw_controls_layout.addWidget(self.start_raw_btn)
        raw_controls_layout.addWidget(self.stop_raw_btn)
        raw_controls_layout.addWidget(self.set_raw_freq_btn)
        raw_controls_layout.addStretch()

        controls_layout.addLayout(raw_controls_layout)

        layout.addWidget(controls_group)

        # Status display group
        status_group = QGroupBox("Streaming Status")
        status_layout = QVBoxLayout(status_group)

        # Status labels
        status_info_layout = QHBoxLayout()

        self.status_label = QLabel("Status: Not streaming")
        self.status_label.setStyleSheet("font-weight: bold;")
        status_info_layout.addWidget(self.status_label)

        self.client_ip_label = QLabel("Client IP: Not set")
        status_info_layout.addWidget(self.client_ip_label)

        status_info_layout.addStretch()
        status_layout.addLayout(status_info_layout)

        # FPS display
        fps_layout = QHBoxLayout()
        fps_layout.addWidget(QLabel("Current FPS:"))

        self.fps_display = QLCDNumber(5)
        self.fps_display.setSegmentStyle(QLCDNumber.Flat)
        self.fps_display.setMaximumHeight(40)
        fps_layout.addWidget(self.fps_display)

        fps_layout.addStretch()
        status_layout.addLayout(fps_layout)

        layout.addWidget(status_group)

        # Position data group
        position_group = QGroupBox("Current Position Data")
        position_layout = QVBoxLayout(position_group)

        # Get latest position button
        self.get_position_btn = QPushButton("Get Latest Position")
        position_layout.addWidget(self.get_position_btn)

        # Position display
        self.position_text = QTextEdit()
        self.position_text.setReadOnly(True)
        self.position_text.setMaximumHeight(150)
        position_layout.addWidget(self.position_text)

        layout.addWidget(position_group)

        # Results/Log display
        results_group = QGroupBox("Streaming Log")
        results_layout = QVBoxLayout(results_group)

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMaximumHeight(150)
        results_layout.addWidget(self.results_text)

        layout.addWidget(results_group)

        # Timer for position updates
        self.position_timer = QTimer()
        self.position_timer.timeout.connect(self.update_position_data)

        layout.addStretch()

    def setup_connections(self):
        self.start_streaming_btn.clicked.connect(self.start_streaming)
        self.stop_streaming_btn.clicked.connect(self.stop_streaming)
        self.start_raw_btn.clicked.connect(self.start_raw_streaming)
        self.stop_raw_btn.clicked.connect(self.stop_raw_streaming)
        self.set_raw_freq_btn.clicked.connect(self.set_raw_frequency)
        self.get_position_btn.clicked.connect(self.get_latest_position)

    def start_streaming(self):
        port = self.port_spin.value()
        frequency = self.frequency_spin.value()
        raw_frequency = self.raw_frequency_spin.value()
        force_stop_calibration = self.force_stop_calibration_cb.isChecked()

        try:
            response = self.api_client.start_streaming(
                port=port,
                frequency=frequency,
                force_stop_calibration=force_stop_calibration,
                streaming_raw_frequency=raw_frequency
            )

            if response.get('status') == 'started':
                self.streaming_active = True
                self.start_streaming_btn.setEnabled(False)
                self.stop_streaming_btn.setEnabled(True)

                target_ip = response.get('target_ip', 'Unknown')
                actual_port = response.get('port', port)
                actual_freq = response.get('frequency', frequency)

                self.status_label.setText(f"Status: Streaming at {actual_freq} Hz")
                self.status_label.setStyleSheet("font-weight: bold; color: green;")
                self.client_ip_label.setText(f"Client IP: {target_ip}:{actual_port}")

                self.results_text.append(f"✓ Started streaming to {target_ip}:{actual_port} at {actual_freq} Hz")

                # Start position updates
                self.position_timer.start(1000)  # Update every second

            elif response.get('status') == 'already_running':
                self.results_text.append("⚠ Streaming is already active")

            else:
                error_msg = response.get('details', 'Unknown error')
                self.results_text.append(f"✗ Failed to start streaming: {error_msg}")

        except Exception as e:
            self.results_text.append(f"✗ Error starting streaming: {str(e)}")

    def stop_streaming(self):
        raw_frequency = self.raw_frequency_spin.value()

        try:
            response = self.api_client.stop_streaming(raw_frequency)

            if response.get('status') == 'stopped':
                self.streaming_active = False
                self.start_streaming_btn.setEnabled(True)
                self.stop_streaming_btn.setEnabled(False)

                self.status_label.setText("Status: Not streaming")
                self.status_label.setStyleSheet("font-weight: bold; color: red;")
                self.fps_display.display(0)

                self.results_text.append("✓ Streaming stopped")

                # Stop position updates
                self.position_timer.stop()

            elif response.get('status') == 'not_active':
                self.results_text.append("⚠ Streaming was not active")

            else:
                self.results_text.append("✗ Error stopping streaming")

        except Exception as e:
            self.results_text.append(f"✗ Error stopping streaming: {str(e)}")

    def start_raw_streaming(self):
        try:
            response = self.api_client._make_request('POST', '/start_raw_streaming')
            self.results_text.append("✓ Started raw streaming")
        except Exception as e:
            self.results_text.append(f"✗ Error starting raw streaming: {str(e)}")

    def stop_raw_streaming(self):
        try:
            response = self.api_client._make_request('POST', '/stop_raw_streaming')
            self.results_text.append("✓ Stopped raw streaming")
        except Exception as e:
            self.results_text.append(f"✗ Error stopping raw streaming: {str(e)}")

    def set_raw_frequency(self):
        frequency = self.raw_frequency_spin.value()
        try:
            response = self.api_client._make_request('POST', '/set_raw_streaming_frequency',
                                                     params={'frequency': frequency})
            self.results_text.append(f"✓ Set raw streaming frequency to {frequency} Hz")
        except Exception as e:
            self.results_text.append(f"✗ Error setting raw frequency: {str(e)}")

    def get_latest_position(self):
        try:
            response = self.api_client.get_latest_position()

            if response.get('status') == 'success':
                data = response.get('data', {})

                # Format and display position data
                formatted_data = json.dumps(data, indent=2)
                self.position_text.setPlainText(formatted_data)

                # Emit signal for visualization
                self.position_updated.emit(data)

                # Update FPS if available
                if 'frame' in data:
                    # This is a simple frame counter, not actual FPS
                    # Real FPS calculation would need timestamp comparison
                    pass

            elif response.get('status') == 'streaming_inactive':
                self.position_text.setPlainText("Streaming is not active")

            elif response.get('status') == 'no_data':
                self.position_text.setPlainText("No position data available yet")

        except Exception as e:
            self.results_text.append(f"✗ Error getting position: {str(e)}")

    def update_position_data(self):
        """Automatically update position data while streaming"""
        if self.streaming_active:
            self.get_latest_position()

    def update_status(self, status):
        # Update status based on server response
        connected = status.get('connected', False)
        ndi_initialized = status.get('ndi_tracker_status') == 'initialized'
        has_fine_registration = status.get('has_fine_registration', False)
        tool_calibration_active = status.get('tool_calibration_active', False)
        streaming_active = status.get('streaming_active', False)
        client_ip = status.get('client_ip', 'Not set')

        # Update client IP display
        self.client_ip_label.setText(f"Client IP: {client_ip}")

        # Update streaming status
        if streaming_active != self.streaming_active:
            self.streaming_active = streaming_active
            self.start_streaming_btn.setEnabled(not streaming_active)
            self.stop_streaming_btn.setEnabled(streaming_active)

            if streaming_active:
                self.status_label.setText("Status: Streaming")
                self.status_label.setStyleSheet("font-weight: bold; color: green;")
                self.position_timer.start(1000)
            else:
                self.status_label.setText("Status: Not streaming")
                self.status_label.setStyleSheet("font-weight: bold; color: red;")
                self.position_timer.stop()
                self.fps_display.display(0)

        # Enable/disable controls based on requirements
        can_stream = connected and ndi_initialized
        self.start_streaming_btn.setEnabled(can_stream and not streaming_active)

        # Show warnings
        if tool_calibration_active and not self.force_stop_calibration_cb.isChecked():
            self.results_text.append("⚠ Tool calibration is active. Enable 'Force Stop' or stop calibration first.")

        if not has_fine_registration:
            self.results_text.append("⚠ Fine registration not performed. Streaming without registration.")
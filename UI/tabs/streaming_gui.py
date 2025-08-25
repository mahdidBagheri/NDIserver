from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
                             QGridLayout, QLabel, QPushButton, QSpinBox,
                             QCheckBox, QTextEdit)
from PyQt5.QtCore import QTimer
from utils.ui_helpers import show_message_box


class StreamingTab(QWidget):
    def __init__(self, server_client, visualization_3d):
        super().__init__()
        self.server_client = server_client
        self.visualization_3d = visualization_3d
        self.streaming_active = False

        self.init_ui()
        self.setup_timers()

    def init_ui(self):
        layout = QVBoxLayout()

        # Streaming controls
        stream_group = self.create_streaming_controls_group()
        layout.addWidget(stream_group)

        # Visualization options
        viz_group = self.create_visualization_group()
        layout.addWidget(viz_group)

        # Current position display
        self.position_text = QTextEdit()
        self.position_text.setMaximumHeight(100)
        self.position_text.setReadOnly(True)
        layout.addWidget(QLabel("Current Position:"))
        layout.addWidget(self.position_text)

        self.setLayout(layout)

    def create_streaming_controls_group(self):
        group = QGroupBox("Streaming Controls")
        layout = QGridLayout()

        # Port and frequency
        layout.addWidget(QLabel("Port:"), 0, 0)
        self.stream_port = QSpinBox()
        self.stream_port.setRange(1024, 65535)
        self.stream_port.setValue(11111)
        layout.addWidget(self.stream_port, 0, 1)

        layout.addWidget(QLabel("Frequency (Hz):"), 1, 0)
        self.stream_frequency = QSpinBox()
        self.stream_frequency.setRange(1, 100)
        self.stream_frequency.setValue(30)
        layout.addWidget(self.stream_frequency, 1, 1)

        # Start/Stop streaming
        self.start_stream_btn = QPushButton("Start Streaming")
        self.start_stream_btn.clicked.connect(self.start_streaming)
        layout.addWidget(self.start_stream_btn, 2, 0, 1, 2)

        self.stop_stream_btn = QPushButton("Stop Streaming")
        self.stop_stream_btn.clicked.connect(self.stop_streaming)
        self.stop_stream_btn.setEnabled(False)
        layout.addWidget(self.stop_stream_btn, 3, 0, 1, 2)

        group.setLayout(layout)
        return group

    def create_visualization_group(self):
        group = QGroupBox("Real-time Visualization")
        layout = QVBoxLayout()

        self.show_realtime_position = QCheckBox("Show Real-time Position")
        self.show_realtime_position.setChecked(True)
        layout.addWidget(self.show_realtime_position)

        self.show_probe_trail = QCheckBox("Show Probe Trail")
        self.show_probe_trail.setChecked(False)
        layout.addWidget(self.show_probe_trail)

        # Clear trail button
        self.clear_trail_btn = QPushButton("Clear Trail")
        self.clear_trail_btn.clicked.connect(self.clear_probe_trail)
        layout.addWidget(self.clear_trail_btn)

        group.setLayout(layout)
        return group

    def setup_timers(self):
        # Streaming update timer (inactive by default)
        self.streaming_timer = None

    def start_streaming(self):
        port = self.stream_port.value()
        frequency = self.stream_frequency.value()

        result = self.server_client.start_streaming(port, frequency)

        if result.get('status') in ['started', 'already_running']:
            self.streaming_active = True
            self.start_stream_btn.setEnabled(False)
            self.stop_stream_btn.setEnabled(True)

            # Start streaming update timer
            self.streaming_timer = QTimer()
            self.streaming_timer.timeout.connect(self.update_streaming_visualization)
            self.streaming_timer.start(200)  # Update every 200ms

            show_message_box(self, "Success", "Streaming started")
        else:
            show_message_box(self, "Error", result.get('details', 'Failed to start streaming'), "error")

    def stop_streaming(self):
        result = self.server_client.stop_streaming()

        if result.get('status') == 'success':
            self.streaming_active = False
            self.start_stream_btn.setEnabled(True)
            self.stop_stream_btn.setEnabled(False)
            if self.streaming_timer:
                self.streaming_timer.stop()
            show_message_box(self, "Success", "Streaming stopped")
        else:
            show_message_box(self, "Error", result.get('details', 'Failed to stop streaming'), "error")

    def update_streaming_visualization(self):
        if not self.streaming_active:
            return

        result = self.server_client.get_latest_position()

        if result.get('status') == 'success':
            data = result.get('data', {})
            position = data.get('position')

            if position and self.show_realtime_position.isChecked():
                # Update probe position visualization
                self.visualization_3d.update_probe_position(position)

                # Update trail if enabled
                if self.show_probe_trail.isChecked():
                    self.visualization_3d.add_to_probe_trail(position)

            # Update position display
            pos_text = f"""Position: {position if position else 'No data'}
Original: {data.get('original', 'No data')}
Timestamp: {data.get('timestamp', 'Unknown')}
Frame: {data.get('frame', 'Unknown')}"""
            self.position_text.setText(pos_text.strip())
        else:
            self.position_text.setText(f"Error getting position: {result.get('details', 'Unknown error')}")

    def clear_probe_trail(self):
        self.visualization_3d.clear_probe_trail()
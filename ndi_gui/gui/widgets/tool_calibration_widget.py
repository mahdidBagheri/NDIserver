from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
                             QPushButton, QLabel, QSpinBox, QProgressBar,
                             QTextEdit, QCheckBox, QTableWidget, QTableWidgetItem,
                             QHeaderView, QComboBox)
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont


class ToolCalibrationWidget(QWidget):
    def __init__(self, api_client):
        super().__init__()
        self.api_client = api_client
        self.calibration_active = False
        self.setup_ui()
        self.setup_connections()

    def setup_ui(self):
        layout = QVBoxLayout(self)

        # Calibration data collection group
        collection_group = QGroupBox("Tool Tip Calibration Data Collection")
        collection_layout = QVBoxLayout(collection_group)

        # Collection parameters
        params_layout = QHBoxLayout()

        params_layout.addWidget(QLabel("Device Index:"))
        self.device_spin = QSpinBox()
        self.device_spin.setRange(0, 5)
        self.device_spin.setValue(0)
        params_layout.addWidget(self.device_spin)

        params_layout.addWidget(QLabel("Force Stop Streaming:"))
        self.force_stop_cb = QCheckBox()
        self.force_stop_cb.setChecked(True)
        params_layout.addWidget(self.force_stop_cb)

        params_layout.addStretch()
        collection_layout.addLayout(params_layout)

        # Collection controls
        collection_controls = QHBoxLayout()
        self.start_calibration_btn = QPushButton("Start Calibration")
        self.start_calibration_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }")

        self.stop_calibration_btn = QPushButton("Stop Collection")
        self.stop_calibration_btn.setStyleSheet("QPushButton { background-color: #f44336; color: white; }")
        self.stop_calibration_btn.setEnabled(False)

        collection_controls.addWidget(self.start_calibration_btn)
        collection_controls.addWidget(self.stop_calibration_btn)
        collection_controls.addStretch()

        collection_layout.addLayout(collection_controls)

        # Collection status
        self.collection_status_label = QLabel("Status: Ready for calibration")
        collection_layout.addWidget(self.collection_status_label)

        # Progress bar
        self.collection_progress = QProgressBar()
        self.collection_progress.setVisible(False)
        collection_layout.addWidget(self.collection_progress)

        layout.addWidget(collection_group)

        # Manual data input group
        manual_group = QGroupBox("Manual Transformation Input")
        manual_layout = QVBoxLayout(manual_group)

        # Matrix table
        self.matrix_table = QTableWidget(4, 4)
        self.matrix_table.setHorizontalHeaderLabels(['X', 'Y', 'Z', 'W'])
        self.matrix_table.setVerticalHeaderLabels(['X', 'Y', 'Z', 'W'])

        # Set default identity matrix
        for i in range(4):
            for j in range(4):
                value = "1.0" if i == j else "0.0"
                self.matrix_table.setItem(i, j, QTableWidgetItem(value))

        manual_layout.addWidget(self.matrix_table)

        # Manual input controls
        manual_controls = QHBoxLayout()
        self.add_matrix_btn = QPushButton("Add Matrix")
        self.load_file_btn = QPushButton("Load from File")

        manual_controls.addWidget(self.add_matrix_btn)
        manual_controls.addWidget(self.load_file_btn)
        manual_controls.addStretch()

        manual_layout.addLayout(manual_controls)
        layout.addWidget(manual_group)

        # Calibration computation group
        computation_group = QGroupBox("Tool Tip Computation")
        computation_layout = QVBoxLayout(computation_group)

        # Computation controls
        comp_controls = QHBoxLayout()
        self.visualize_tool_cb = QCheckBox("Visualize Calibration")
        self.calibrate_btn = QPushButton("Calibrate Tool Tip")
        self.calibrate_btn.setStyleSheet("QPushButton { background-color: #2196F3; color: white; font-weight: bold; }")

        comp_controls.addWidget(self.visualize_tool_cb)
        comp_controls.addStretch()
        comp_controls.addWidget(self.calibrate_btn)

        computation_layout.addLayout(comp_controls)

        # Calibration progress
        self.calibration_progress = QProgressBar()
        self.calibration_progress.setVisible(False)
        computation_layout.addWidget(self.calibration_progress)

        layout.addWidget(computation_group)

        # Touch point calculation group
        touch_group = QGroupBox("Touch Point Calculation")
        touch_layout = QVBoxLayout(touch_group)

        # Touch point parameters
        touch_params = QHBoxLayout()

        touch_params.addWidget(QLabel("Probe Index:"))
        self.probe_idx_spin = QSpinBox()
        self.probe_idx_spin.setRange(0, 5)
        self.probe_idx_spin.setValue(0)
        touch_params.addWidget(self.probe_idx_spin)

        touch_params.addWidget(QLabel("Endoscope Index:"))
        self.endoscope_idx_spin = QSpinBox()
        self.endoscope_idx_spin.setRange(0, 5)
        self.endoscope_idx_spin.setValue(2)
        touch_params.addWidget(self.endoscope_idx_spin)

        touch_params.addStretch()
        touch_layout.addLayout(touch_params)

        # Touch point button
        self.get_touch_point_btn = QPushButton("Get Probe Touch Point")
        touch_layout.addWidget(self.get_touch_point_btn)

        layout.addWidget(touch_group)

        # Results display
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout(results_group)

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMaximumHeight(200)
        results_layout.addWidget(self.results_text)

        layout.addWidget(results_group)

        # Status timer
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_calibration_status)

        layout.addStretch()

    def setup_connections(self):
        self.start_calibration_btn.clicked.connect(self.start_calibration)
        self.stop_calibration_btn.clicked.connect(self.stop_calibration)
        self.add_matrix_btn.clicked.connect(self.add_manual_matrix)
        self.load_file_btn.clicked.connect(self.load_from_file)
        self.calibrate_btn.clicked.connect(self.calibrate_tool)
        self.get_touch_point_btn.clicked.connect(self.get_touch_point)

    def start_calibration(self):
        device = self.device_spin.value()
        force_stop = self.force_stop_cb.isChecked()

        try:
            response = self.api_client.start_tool_calibration(force_stop, device)

            if response.get('status') == 'success':
                self.calibration_active = True
                self.start_calibration_btn.setEnabled(False)
                self.stop_calibration_btn.setEnabled(True)
                self.collection_status_label.setText("Status: Collecting calibration data...")
                self.collection_progress.setVisible(True)
                self.collection_progress.setRange(0, 0)  # Indeterminate

                # Start status updates
                self.status_timer.start(1000)

                self.results_text.append("✓ Started tool calibration data collection")
                self.results_text.append("Move the tool to different orientations while keeping the tip fixed")

            else:
                error_msg = response.get('details', 'Unknown error')
                self.results_text.append(f"✗ Failed to start calibration: {error_msg}")

        except Exception as e:
            self.results_text.append(f"✗ Error starting calibration: {str(e)}")

    def stop_calibration(self):
        try:
            response = self.api_client.end_tool_calibration()

            self.calibration_active = False
            self.start_calibration_btn.setEnabled(True)
            self.stop_calibration_btn.setEnabled(False)
            self.collection_progress.setVisible(False)
            self.status_timer.stop()

            if response.get('status') == 'success':
                matrices_count = response.get('matrices_collected', 0)
                self.collection_status_label.setText(f"Status: Collected {matrices_count} matrices")
                self.results_text.append(f"✓ Stopped collection. Collected {matrices_count} transformation matrices")

                # Enable calibration button if we have enough data
                self.calibrate_btn.setEnabled(matrices_count >= 4)

            else:
                self.collection_status_label.setText("Status: Error stopping collection")
                self.results_text.append("✗ Error stopping calibration collection")

        except Exception as e:
            self.results_text.append(f"✗ Error stopping calibration: {str(e)}")
            self.calibration_active = False
            self.start_calibration_btn.setEnabled(True)
            self.stop_calibration_btn.setEnabled(False)
            self.collection_progress.setVisible(False)
            self.status_timer.stop()

    def add_manual_matrix(self):
        # Extract matrix from table
        matrix = []
        try:
            for i in range(4):
                row = []
                for j in range(4):
                    item = self.matrix_table.item(i, j)
                    if item:
                        row.append(float(item.text()))
                    else:
                        row.append(0.0)
                matrix.append(row)

            response = self.api_client._make_request('POST', '/add_tool_transformation',
                                                     json={'matrix': matrix})

            if response.get('status') == 'success':
                self.results_text.append("✓ Added transformation matrix manually")
            else:
                self.results_text.append("✗ Failed to add matrix")

        except ValueError:
            self.results_text.append("✗ Invalid matrix values. Please enter numeric values only.")
        except Exception as e:
            self.results_text.append(f"✗ Error adding matrix: {str(e)}")

    def load_from_file(self):
        try:
            response = self.api_client._make_request('POST', '/load_tool_transformations_from_file',
                                                     params={'filename': 'tool_tip.txt'})

            if response.get('status') == 'success':
                count = response.get('matrices_loaded', 0)
                self.results_text.append(f"✓ Loaded {count} matrices from file")
                self.calibrate_btn.setEnabled(count >= 4)
            else:
                error_msg = response.get('details', 'Unknown error')
                self.results_text.append(f"✗ Failed to load from file: {error_msg}")

        except Exception as e:
            self.results_text.append(f"✗ Error loading from file: {str(e)}")

    def calibrate_tool(self):
        # Show progress
        self.calibration_progress.setVisible(True)
        self.calibration_progress.setRange(0, 0)  # Indeterminate
        self.calibrate_btn.setEnabled(False)

        visualize = self.visualize_tool_cb.isChecked()

        try:
            response = self.api_client.calibrate_tool(visualize)

            if response.get('status') == 'success':
                self.results_text.append("✓ Tool tip calibration completed successfully")

                # Display results
                if 'tool_tip_vector' in response:
                    tip_vector = response['tool_tip_vector']
                    self.results_text.append(
                        f"Tool tip vector: [{tip_vector[0]:.3f}, {tip_vector[1]:.3f}, {tip_vector[2]:.3f}]")

                if 'calibration_error' in response:
                    error = response['calibration_error']
                    self.results_text.append(f"Calibration error: {error:.3f} mm")

            else:
                error_msg = response.get('details', 'Unknown error')
                self.results_text.append(f"✗ Tool calibration failed: {error_msg}")

        except Exception as e:
            self.results_text.append(f"✗ Tool calibration error: {str(e)}")

        finally:
            self.calibration_progress.setVisible(False)
            self.calibrate_btn.setEnabled(True)

    def get_touch_point(self):
        probe_idx = self.probe_idx_spin.value()
        endoscope_idx = self.endoscope_idx_spin.value()

        try:
            response = self.api_client._make_request('POST', '/get_probe_touchpoint',
                                                     params={'probe_idx': probe_idx, 'endoscope_idx': endoscope_idx})

            if 'touch_point' in response:
                touch_point = response['touch_point']
                self.results_text.append(
                    f"✓ Touch point: [{touch_point[0]:.3f}, {touch_point[1]:.3f}, {touch_point[2]:.3f}]")
            else:
                self.results_text.append("✗ Failed to get touch point")

        except Exception as e:
            self.results_text.append(f"✗ Error getting touch point: {str(e)}")

    def update_calibration_status(self):
        if not self.calibration_active:
            return

        try:
            response = self.api_client.get_tool_calibration_status()

            if 'matrices_collected' in response:
                count = response['matrices_collected']
                self.collection_status_label.setText(f"Status: Collected {count} matrices")

        except Exception as e:
            pass  # Silently fail status updates

    def update_status(self, status):
        # Enable/disable controls based on connection and NDI status
        connected = status.get('connected', False)
        ndi_initialized = status.get('ndi_tracker_status') == 'initialized'
        streaming_active = status.get('streaming_active', False)

        # Can start calibration if connected, NDI initialized, and not currently calibrating
        can_start = connected and ndi_initialized and not self.calibration_active
        self.start_calibration_btn.setEnabled(can_start)

        # Touch point calculation requires NDI to be initialized
        self.get_touch_point_btn.setEnabled(connected and ndi_initialized)

        # Show warning if streaming is active
        if streaming_active and not self.force_stop_cb.isChecked():
            self.collection_status_label.setText("Status: Stop streaming first or enable 'Force Stop Streaming'")
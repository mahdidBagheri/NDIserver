from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
                             QPushButton, QLabel, QSpinBox, QProgressBar,
                             QTextEdit, QCheckBox, QDoubleSpinBox, QSlider,
                             QComboBox, QListWidget)
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont


class FineRegistrationWidget(QWidget):
    points_updated = pyqtSignal(list)
    transformation_updated = pyqtSignal(list)  # New signal for combined transformation matrix

    def __init__(self, api_client):
        super().__init__()
        self.api_client = api_client
        self.gathering_active = False
        self.fine_points = []
        self.setup_ui()
        self.setup_connections()

    def perform_fine_registration(self):
        # Show progress
        self.registration_progress.setVisible(True)
        self.registration_progress.setRange(0, 0)  # Indeterminate
        self.fine_register_btn.setEnabled(False)

        cloud_id = self.cloud_id_spin.value()
        downsample_factor = self.downsample_spin.value()
        visualize = self.visualize_fine_cb.isChecked()

        try:
            response = self.api_client.fine_register(cloud_id, downsample_factor, visualize)

            if response.get('status') == 'success':
                self.results_text.append("✓ Fine registration completed successfully")

                # Display results
                if 'rmse' in response:
                    self.results_text.append(f"RMSE: {response['rmse']:.3f} mm")

                if 'iterations' in response:
                    self.results_text.append(f"ICP Iterations: {response['iterations']}")

                if 'fitness' in response:
                    self.results_text.append(f"Fitness: {response['fitness']:.6f}")

                # Emit combined transformation matrix for visualization
                if 'combined_transformation' in response:
                    combined_transformation = response['combined_transformation']
                    self.transformation_updated.emit(combined_transformation)

            else:
                error_msg = response.get('details', 'Unknown error')
                self.results_text.append(f"✗ Fine registration failed: {error_msg}")

        except Exception as e:
            self.results_text.append(f"✗ Fine registration error: {str(e)}")

        finally:
            self.registration_progress.setVisible(False)
            self.fine_register_btn.setEnabled(True)

    def setup_ui(self):
        layout = QVBoxLayout(self)

        # Point gathering group
        gather_group = QGroupBox("Fine Point Gathering")
        gather_layout = QVBoxLayout(gather_group)

        # Gathering parameters
        params_layout = QHBoxLayout()

        params_layout.addWidget(QLabel("Gathering Frequency:"))
        self.gather_freq_spin = QSpinBox()
        self.gather_freq_spin.setRange(10, 120)
        self.gather_freq_spin.setValue(60)
        self.gather_freq_spin.setSuffix(" Hz")
        params_layout.addWidget(self.gather_freq_spin)

        params_layout.addWidget(QLabel("Raw Streaming Freq:"))
        self.raw_freq_spin = QSpinBox()
        self.raw_freq_spin.setRange(5, 60)
        self.raw_freq_spin.setValue(10)
        self.raw_freq_spin.setSuffix(" Hz")
        params_layout.addWidget(self.raw_freq_spin)

        params_layout.addStretch()
        gather_layout.addLayout(params_layout)

        # Gathering controls
        gather_controls = QHBoxLayout()
        self.start_gather_btn = QPushButton("Start Gathering")
        self.start_gather_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }")

        self.stop_gather_btn = QPushButton("Stop Gathering")
        self.stop_gather_btn.setStyleSheet("QPushButton { background-color: #f44336; color: white; }")
        self.stop_gather_btn.setEnabled(False)

        self.reset_gather_btn = QPushButton("Reset Points")

        gather_controls.addWidget(self.start_gather_btn)
        gather_controls.addWidget(self.stop_gather_btn)
        gather_controls.addWidget(self.reset_gather_btn)
        gather_controls.addStretch()

        gather_layout.addLayout(gather_controls)

        # Gathering status
        self.gather_status_label = QLabel("Status: Ready to gather")
        gather_layout.addWidget(self.gather_status_label)

        # Progress bar for gathering
        self.gather_progress = QProgressBar()
        self.gather_progress.setVisible(False)
        gather_layout.addWidget(self.gather_progress)

        layout.addWidget(gather_group)

        # Registration group
        registration_group = QGroupBox("Fine Registration (ICP)")
        registration_layout = QVBoxLayout(registration_group)

        # Registration parameters
        reg_params_layout = QHBoxLayout()

        reg_params_layout.addWidget(QLabel("Point Cloud ID:"))
        self.cloud_id_spin = QSpinBox()
        self.cloud_id_spin.setRange(0, 10)
        self.cloud_id_spin.setValue(0)
        reg_params_layout.addWidget(self.cloud_id_spin)

        reg_params_layout.addWidget(QLabel("Downsample Factor:"))
        self.downsample_spin = QDoubleSpinBox()
        self.downsample_spin.setRange(0.1, 1.0)
        self.downsample_spin.setValue(1.0)
        self.downsample_spin.setSingleStep(0.1)
        reg_params_layout.addWidget(self.downsample_spin)

        reg_params_layout.addStretch()
        registration_layout.addLayout(reg_params_layout)

        # Registration controls
        reg_controls = QHBoxLayout()
        self.visualize_fine_cb = QCheckBox("Visualize Registration")
        self.fine_register_btn = QPushButton("Perform Fine Registration")
        self.fine_register_btn.setStyleSheet(
            "QPushButton { background-color: #2196F3; color: white; font-weight: bold; }")

        reg_controls.addWidget(self.visualize_fine_cb)
        reg_controls.addStretch()
        reg_controls.addWidget(self.fine_register_btn)

        registration_layout.addLayout(reg_controls)

        # Progress bar for registration
        self.registration_progress = QProgressBar()
        self.registration_progress.setVisible(False)
        registration_layout.addWidget(self.registration_progress)

        layout.addWidget(registration_group)

        # Results display
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout(results_group)

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMaximumHeight(200)
        results_layout.addWidget(self.results_text)

        layout.addWidget(results_group)

        # Load/Save group
        file_group = QGroupBox("Load/Save")
        file_layout = QHBoxLayout(file_group)

        self.load_fine_btn = QPushButton("Load Last Fine Transform")
        file_layout.addWidget(self.load_fine_btn)

        layout.addWidget(file_group)

        # Status timer
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_gathering_status)

        layout.addStretch()

    def setup_connections(self):
        self.start_gather_btn.clicked.connect(self.start_gathering)
        self.stop_gather_btn.clicked.connect(self.stop_gathering)
        self.reset_gather_btn.clicked.connect(self.reset_gathering)
        self.fine_register_btn.clicked.connect(self.perform_fine_registration)
        self.load_fine_btn.clicked.connect(self.load_fine_transform)

    def start_gathering(self):
        frequency = self.gather_freq_spin.value()
        raw_freq = self.raw_freq_spin.value()

        try:
            response = self.api_client.start_fine_gather(frequency, raw_freq)

            if response.get('status') == 'success':
                self.gathering_active = True
                self.start_gather_btn.setEnabled(False)
                self.stop_gather_btn.setEnabled(True)
                self.gather_status_label.setText("Status: Gathering points...")
                self.gather_progress.setVisible(True)
                self.gather_progress.setRange(0, 0)  # Indeterminate

                # Start status updates
                self.status_timer.start(1000)  # Update every second

                self.results_text.append("✓ Started fine point gathering")
            else:
                error_msg = response.get('details', 'Unknown error')
                self.results_text.append(f"✗ Failed to start gathering: {error_msg}")

        except Exception as e:
            self.results_text.append(f"✗ Error starting gathering: {str(e)}")

    def stop_gathering(self):
        raw_freq = self.raw_freq_spin.value()

        try:
            response = self.api_client.end_fine_gather(raw_freq)

            self.gathering_active = False
            self.start_gather_btn.setEnabled(True)
            self.stop_gather_btn.setEnabled(False)
            self.gather_progress.setVisible(False)
            self.status_timer.stop()

            if response.get('status') == 'success':
                points_count = response.get('points_collected', 0)
                self.gather_status_label.setText(f"Status: Collected {points_count} points")
                self.results_text.append(f"✓ Stopped gathering. Collected {points_count} points")

                # Update fine registration button availability
                self.fine_register_btn.setEnabled(points_count > 0)

            else:
                self.gather_status_label.setText("Status: Error stopping gathering")
                self.results_text.append("✗ Error stopping gathering")

        except Exception as e:
            self.results_text.append(f"✗ Error stopping gathering: {str(e)}")
            self.gathering_active = False
            self.start_gather_btn.setEnabled(True)
            self.stop_gather_btn.setEnabled(False)
            self.gather_progress.setVisible(False)
            self.status_timer.stop()

    def reset_gathering(self):
        try:
            response = self.api_client.reset_fine_gather()

            if response.get('results') == 'OK':
                self.results_text.append("✓ Reset fine gathering data")
                self.gather_status_label.setText("Status: Ready to gather")
                self.fine_register_btn.setEnabled(False)
                self.fine_points = []
                self.points_updated.emit([])
            else:
                self.results_text.append("✗ Failed to reset gathering data")

        except Exception as e:
            self.results_text.append(f"✗ Error resetting gathering: {str(e)}")

    def update_gathering_status(self):
        if not self.gathering_active:
            return

        try:
            response = self.api_client.get_fine_points_status()

            if 'points_collected' in response:
                points_count = response['points_collected']
                self.gather_status_label.setText(f"Status: Collected {points_count} points")

        except Exception as e:
            pass  # Silently fail status updates

    def perform_fine_registration(self):
        # Show progress
        self.registration_progress.setVisible(True)
        self.registration_progress.setRange(0, 0)  # Indeterminate
        self.fine_register_btn.setEnabled(False)

        cloud_id = self.cloud_id_spin.value()
        downsample_factor = self.downsample_spin.value()
        visualize = self.visualize_fine_cb.isChecked()

        try:
            response = self.api_client.fine_register(cloud_id, downsample_factor, visualize)

            if response.get('status') == 'success':
                self.results_text.append("✓ Fine registration completed successfully")

                # Display results
                if 'rmse' in response:
                    self.results_text.append(f"RMSE: {response['rmse']:.3f} mm")

                if 'iterations' in response:
                    self.results_text.append(f"ICP Iterations: {response['iterations']}")

                if 'fitness' in response:
                    self.results_text.append(f"Fitness: {response['fitness']:.6f}")

            else:
                error_msg = response.get('details', 'Unknown error')
                self.results_text.append(f"✗ Fine registration failed: {error_msg}")

        except Exception as e:
            self.results_text.append(f"✗ Fine registration error: {str(e)}")

        finally:
            self.registration_progress.setVisible(False)
            self.fine_register_btn.setEnabled(True)

    def load_fine_transform(self):
        try:
            response = self.api_client._make_request('POST', '/load_last_fine_transform')

            if response.get('status') == 'success':
                self.results_text.append("✓ Loaded last fine transformation")
            else:
                self.results_text.append("✗ Failed to load fine transformation")

        except Exception as e:
            self.results_text.append(f"✗ Error loading transformation: {str(e)}")

    def update_status(self, status):
        # Enable/disable controls based on connection and coarse registration status
        connected = status.get('connected', False)
        has_coarse = status.get('has_coarse_registration', False)
        ndi_initialized = status.get('ndi_tracker_status') == 'initialized'

        # Can only start gathering if connected, NDI initialized, and coarse registration done
        can_gather = connected and ndi_initialized and has_coarse
        self.start_gather_btn.setEnabled(can_gather and not self.gathering_active)

        # Registration button enabled if we have points and coarse registration
        self.fine_register_btn.setEnabled(connected and has_coarse)

        if not has_coarse:
            self.gather_status_label.setText("Status: Coarse registration required first")
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
                             QPushButton, QLineEdit, QLabel, QSpinBox,
                             QTableWidget, QTableWidgetItem, QHeaderView,
                             QProgressBar, QTextEdit, QCheckBox, QDoubleSpinBox)
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont
import json


class CoarseRegistrationWidget(QWidget):
    points_updated = pyqtSignal(list)
    transformation_updated = pyqtSignal(list)  # New signal for transformation matrix

    def __init__(self, api_client):
        super().__init__()
        self.api_client = api_client
        self.coarse_points = []
        self.setup_ui()
        self.setup_connections()

    def perform_registration(self):
        if self.points_table.rowCount() < 3:
            self.results_text.append("✗ Need at least 3 points for registration")
            return

        # Show progress
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        self.register_btn.setEnabled(False)

        try:
            visualize = self.visualize_cb.isChecked()
            response = self.api_client.coarse_register(visualize)

            if response.get('status') == 'success':
                self.results_text.append("✓ Coarse registration completed successfully")

                # Display registration results
                if 'rmse' in response:
                    self.results_text.append(f"RMSE: {response['rmse']:.3f} mm")

                # Emit transformation matrix for visualization
                if 'transformation_matrix' in response:
                    self.results_text.append("Transformation matrix computed")
                    transformation_matrix = response['transformation_matrix']
                    self.transformation_updated.emit(transformation_matrix)

            else:
                error_msg = response.get('details', 'Unknown error')
                self.results_text.append(f"✗ Registration failed: {error_msg}")

        except Exception as e:
            self.results_text.append(f"✗ Registration error: {str(e)}")

        finally:
            self.progress_bar.setVisible(False)
            self.register_btn.setEnabled(True)

    def setup_ui(self):
        layout = QVBoxLayout(self)

        # Point input group
        input_group = QGroupBox("Add Coarse Points")
        input_layout = QVBoxLayout(input_group)

        # Unity point coordinates
        coords_layout = QHBoxLayout()
        coords_layout.addWidget(QLabel("Unity Point (X, Y, Z):"))

        self.x_spin = QDoubleSpinBox()
        self.x_spin.setRange(-1000, 1000)
        self.x_spin.setDecimals(3)
        self.x_spin.setSuffix(" mm")

        self.y_spin = QDoubleSpinBox()
        self.y_spin.setRange(-1000, 1000)
        self.y_spin.setDecimals(3)
        self.y_spin.setSuffix(" mm")

        self.z_spin = QDoubleSpinBox()
        self.z_spin.setRange(-1000, 1000)
        self.z_spin.setDecimals(3)
        self.z_spin.setSuffix(" mm")

        coords_layout.addWidget(self.x_spin)
        coords_layout.addWidget(self.y_spin)
        coords_layout.addWidget(self.z_spin)

        input_layout.addLayout(coords_layout)

        # Point number
        point_layout = QHBoxLayout()
        point_layout.addWidget(QLabel("Point Number:"))
        self.point_number_spin = QSpinBox()
        self.point_number_spin.setRange(1, 100)
        self.point_number_spin.setValue(1)
        point_layout.addWidget(self.point_number_spin)
        point_layout.addStretch()
        input_layout.addLayout(point_layout)

        # Add point button
        self.add_point_btn = QPushButton("Add Coarse Point")
        self.add_point_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }")
        input_layout.addWidget(self.add_point_btn)

        layout.addWidget(input_group)

        # Points table
        table_group = QGroupBox("Coarse Points")
        table_layout = QVBoxLayout(table_group)

        self.points_table = QTableWidget()
        self.points_table.setColumnCount(7)
        self.points_table.setHorizontalHeaderLabels([
            "Point #", "Unity X", "Unity Y", "Unity Z",
            "NDI X", "NDI Y", "NDI Z"
        ])

        # Make table stretch to fill width
        header = self.points_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)

        table_layout.addWidget(self.points_table)

        # Table controls
        table_controls = QHBoxLayout()
        self.refresh_table_btn = QPushButton("Refresh Table")
        self.clear_points_btn = QPushButton("Clear All Points")
        self.clear_points_btn.setStyleSheet("QPushButton { background-color: #f44336; color: white; }")

        table_controls.addWidget(self.refresh_table_btn)
        table_controls.addWidget(self.clear_points_btn)
        table_controls.addStretch()

        table_layout.addLayout(table_controls)
        layout.addWidget(table_group)

        # Registration group
        registration_group = QGroupBox("Coarse Registration")
        registration_layout = QVBoxLayout(registration_group)

        # Registration controls
        reg_controls = QHBoxLayout()
        self.visualize_cb = QCheckBox("Visualize Registration")
        self.register_btn = QPushButton("Perform Coarse Registration")
        self.register_btn.setStyleSheet("QPushButton { background-color: #2196F3; color: white; font-weight: bold; }")

        reg_controls.addWidget(self.visualize_cb)
        reg_controls.addStretch()
        reg_controls.addWidget(self.register_btn)

        registration_layout.addLayout(reg_controls)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        registration_layout.addWidget(self.progress_bar)

        # Results display
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMaximumHeight(150)
        registration_layout.addWidget(self.results_text)

        layout.addWidget(registration_group)

        # Load/Save group
        file_group = QGroupBox("Load/Save")
        file_layout = QHBoxLayout(file_group)

        self.load_coarse_btn = QPushButton("Load Last Coarse Transform")
        file_layout.addWidget(self.load_coarse_btn)

        layout.addWidget(file_group)

        layout.addStretch()

    def setup_connections(self):
        self.add_point_btn.clicked.connect(self.add_coarse_point)
        self.refresh_table_btn.clicked.connect(self.refresh_points_table)
        self.clear_points_btn.clicked.connect(self.clear_all_points)
        self.register_btn.clicked.connect(self.perform_registration)
        self.load_coarse_btn.clicked.connect(self.load_coarse_transform)

        # Auto-increment point number after adding point
        self.add_point_btn.clicked.connect(lambda: self.point_number_spin.setValue(
            self.point_number_spin.value() + 1))

    def add_coarse_point(self):
        unity_point = [
            self.x_spin.value(),
            self.y_spin.value(),
            self.z_spin.value()
        ]
        point_number = self.point_number_spin.value()

        try:
            response = self.api_client.set_coarse_point(unity_point, point_number)

            if response.get('status') == 'success':
                self.results_text.append(f"✓ Point {point_number} added successfully")
                self.refresh_points_table()
            else:
                error_msg = response.get('message', response.get('details', 'Unknown error'))
                self.results_text.append(f"✗ Failed to add point {point_number}: {error_msg}")

        except Exception as e:
            self.results_text.append(f"✗ Error adding point: {str(e)}")

    def refresh_points_table(self):
        try:
            response = self.api_client.get_coarse_points()

            if 'coarse_points' in response:
                points = response['coarse_points']
                self.coarse_points = []

                self.points_table.setRowCount(len(points))

                for i, point_data in enumerate(points):
                    # Extract data
                    point_num = point_data.get('point_number', i + 1)
                    unity_point = point_data.get('unity_point', [0, 0, 0])
                    ndi_point = point_data.get('ndi_point', [0, 0, 0])

                    # Add to visualization list
                    self.coarse_points.append(unity_point)

                    # Fill table
                    self.points_table.setItem(i, 0, QTableWidgetItem(str(point_num)))
                    self.points_table.setItem(i, 1, QTableWidgetItem(f"{unity_point[0]:.3f}"))
                    self.points_table.setItem(i, 2, QTableWidgetItem(f"{unity_point[1]:.3f}"))
                    self.points_table.setItem(i, 3, QTableWidgetItem(f"{unity_point[2]:.3f}"))
                    self.points_table.setItem(i, 4, QTableWidgetItem(f"{ndi_point[0]:.3f}"))
                    self.points_table.setItem(i, 5, QTableWidgetItem(f"{ndi_point[1]:.3f}"))
                    self.points_table.setItem(i, 6, QTableWidgetItem(f"{ndi_point[2]:.3f}"))

                # Emit signal for visualization update
                self.points_updated.emit(self.coarse_points)

        except Exception as e:
            self.results_text.append(f"✗ Error refreshing table: {str(e)}")

    def clear_all_points(self):
        try:
            response = self.api_client.reset_coarse_points()

            if response.get('status') == 'success':
                self.results_text.append("✓ All coarse points cleared")
                self.points_table.setRowCount(0)
                self.coarse_points = []
                self.points_updated.emit([])
                self.point_number_spin.setValue(1)
            else:
                self.results_text.append("✗ Failed to clear points")

        except Exception as e:
            self.results_text.append(f"✗ Error clearing points: {str(e)}")

    def perform_registration(self):
        if self.points_table.rowCount() < 3:
            self.results_text.append("✗ Need at least 3 points for registration")
            return

        # Show progress
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        self.register_btn.setEnabled(False)

        try:
            visualize = self.visualize_cb.isChecked()
            response = self.api_client.coarse_register(visualize)

            if response.get('status') == 'success':
                self.results_text.append("✓ Coarse registration completed successfully")

                # Display registration results
                if 'rmse' in response:
                    self.results_text.append(f"RMSE: {response['rmse']:.3f} mm")

                if 'transformation_matrix' in response:
                    self.results_text.append("Transformation matrix computed")

            else:
                error_msg = response.get('details', 'Unknown error')
                self.results_text.append(f"✗ Registration failed: {error_msg}")

        except Exception as e:
            self.results_text.append(f"✗ Registration error: {str(e)}")

        finally:
            self.progress_bar.setVisible(False)
            self.register_btn.setEnabled(True)

    def load_coarse_transform(self):
        try:
            response = self.api_client._make_request('POST', '/load_last_coarse_transform')

            if response.get('status') == 'success':
                self.results_text.append("✓ Loaded last coarse transformation")
            else:
                self.results_text.append("✗ Failed to load coarse transformation")

        except Exception as e:
            self.results_text.append(f"✗ Error loading transformation: {str(e)}")

    def update_status(self, status):
        # Enable/disable controls based on connection and NDI status
        connected = status.get('connected', False)
        ndi_initialized = status.get('ndi_tracker_status') == 'initialized'

        self.add_point_btn.setEnabled(connected and ndi_initialized)
        self.register_btn.setEnabled(connected and self.points_table.rowCount() >= 3)

        # Auto-refresh table when connected
        if connected:
            self.refresh_points_table()
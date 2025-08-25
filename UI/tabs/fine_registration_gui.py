from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
                             QGridLayout, QLabel, QPushButton, QSpinBox,
                             QDoubleSpinBox, QCheckBox, QTextEdit, QFileDialog)
from PyQt5.QtCore import QTimer
from utils.ui_helpers import show_message_box
import numpy as np
import pyvista as pv


class FineRegistrationTab(QWidget):
    def __init__(self, server_client, visualization_3d, config_manager):
        super().__init__()
        self.server_client = server_client
        self.visualization_3d = visualization_3d
        self.config_manager = config_manager

        self.fine_points = []

        self.init_ui()
        self.setup_timers()

    def init_ui(self):
        layout = QVBoxLayout()

        # Fine registration controls
        fine_group = self.create_fine_controls_group()
        layout.addWidget(fine_group)

        # Visualization options
        viz_group = self.create_visualization_group()
        layout.addWidget(viz_group)

        # Status display
        self.status_text = QTextEdit()
        self.status_text.setMaximumHeight(100)
        self.status_text.setReadOnly(True)
        layout.addWidget(QLabel("Fine Registration Status:"))
        layout.addWidget(self.status_text)

        self.setLayout(layout)

    def create_fine_controls_group(self):
        group = QGroupBox("Fine Registration")
        layout = QGridLayout()

        # Gathering frequency
        layout.addWidget(QLabel("Frequency (Hz):"), 0, 0)
        self.fine_frequency = QSpinBox()
        self.fine_frequency.setRange(1, 100)
        self.fine_frequency.setValue(60)
        layout.addWidget(self.fine_frequency, 0, 1)

        # Start/Stop gathering
        self.start_fine_btn = QPushButton("Start Fine Gathering")
        self.start_fine_btn.clicked.connect(self.start_fine_gathering)
        layout.addWidget(self.start_fine_btn, 1, 0, 1, 2)

        self.stop_fine_btn = QPushButton("Stop Fine Gathering")
        self.stop_fine_btn.clicked.connect(self.stop_fine_gathering)
        self.stop_fine_btn.setEnabled(False)
        layout.addWidget(self.stop_fine_btn, 2, 0, 1, 2)

        # Registration parameters
        layout.addWidget(QLabel("Model ID:"), 3, 0)
        self.model_id = QSpinBox()
        self.model_id.setRange(0, 10)
        layout.addWidget(self.model_id, 3, 1)

        layout.addWidget(QLabel("Downsample:"), 4, 0)
        self.downsample_factor = QDoubleSpinBox()
        self.downsample_factor.setRange(0.1, 1.0)
        self.downsample_factor.setValue(1.0)
        self.downsample_factor.setSingleStep(0.1)
        layout.addWidget(self.downsample_factor, 4, 1)

        # Perform fine registration
        self.fine_register_btn = QPushButton("Perform Fine Registration")
        self.fine_register_btn.clicked.connect(self.perform_fine_registration)
        layout.addWidget(self.fine_register_btn, 5, 0, 1, 2)

        self.reset_fine_btn = QPushButton("Reset Fine Gathering")
        self.reset_fine_btn.clicked.connect(self.reset_fine_gathering)
        layout.addWidget(self.reset_fine_btn, 6, 0, 1, 2)

        group.setLayout(layout)
        return group

    def create_visualization_group(self):
        group = QGroupBox("Visualization")
        layout = QVBoxLayout()

        self.show_fine_points = QCheckBox("Show Fine Points")
        self.show_fine_points.setChecked(True)
        self.show_fine_points.toggled.connect(self.update_fine_visualization)
        layout.addWidget(self.show_fine_points)

        self.show_ct_cloud = QCheckBox("Show CT Point Cloud")
        self.show_ct_cloud.setChecked(False)
        self.show_ct_cloud.toggled.connect(self.update_ct_visualization)
        layout.addWidget(self.show_ct_cloud)

        self.load_ct_btn = QPushButton("Load Different CT Point Cloud")
        self.load_ct_btn.clicked.connect(self.load_ct_pointcloud)
        layout.addWidget(self.load_ct_btn)

        # Add debug button
        self.debug_ct_btn = QPushButton("Debug CT Point Cloud")
        self.debug_ct_btn.clicked.connect(self.debug_ct_pointcloud)
        layout.addWidget(self.debug_ct_btn)

        group.setLayout(layout)
        return group

    def setup_timers(self):
        # Fine points status update timer (inactive by default)
        self.fine_timer = None

    def start_fine_gathering(self):
        frequency = self.fine_frequency.value()
        result = self.server_client.start_fine_gathering(frequency)

        if result.get('status') == 'success':
            self.start_fine_btn.setEnabled(False)
            self.stop_fine_btn.setEnabled(True)
            show_message_box(self, "Success", "Fine gathering started")

            # Start timer to update fine points status
            self.fine_timer = QTimer()
            self.fine_timer.timeout.connect(self.update_fine_points_status)
            self.fine_timer.start(1000)  # Update every 1 second
        else:
            show_message_box(self, "Error", result.get('details', 'Failed to start gathering'), "error")

    def stop_fine_gathering(self):
        result = self.server_client.stop_fine_gathering()

        if result.get('status') == 'success':
            self.start_fine_btn.setEnabled(True)
            self.stop_fine_btn.setEnabled(False)
            if self.fine_timer:
                self.fine_timer.stop()
            show_message_box(self, "Success", "Fine gathering stopped")
            self.update_fine_points_status()
        else:
            show_message_box(self, "Error", result.get('details', 'Failed to stop gathering'), "error")

    def perform_fine_registration(self):
        model_id = self.model_id.value()
        downsample = self.downsample_factor.value()

        result = self.server_client.perform_fine_registration(model_id, downsample)

        if result.get('status') == 'success':
            show_message_box(self, "Success", "Fine registration completed successfully")
            self.update_fine_visualization()
        else:
            show_message_box(self, "Error", result.get('details', 'Registration failed'), "error")

    def reset_fine_gathering(self):
        result = self.server_client.reset_fine_gathering()

        if result.get('status') == 'success':
            self.update_fine_points_status()
            show_message_box(self, "Success", "Fine gathering reset")
        else:
            show_message_box(self, "Error", result.get('details', 'Reset failed'), "error")

    def update_fine_points_status(self):
        status = self.server_client.get_fine_points_status()

        if status:
            status_text = f"""Gathering Active: {status.get('gathering_active', False)}
Points Collected: {status.get('points_collected', 0)}
Collection Rate: {status.get('collection_rate', 0):.1f} Hz"""
            self.status_text.setText(status_text.strip())

            # Update visualization if gathering is active
            if status.get('gathering_active', False):
                self.update_fine_visualization()
        else:
            self.status_text.setText("Error getting fine points status")

    def update_fine_visualization(self):
        # This would need to get fine points from server
        # For now, just update with empty list
        self.visualization_3d.update_fine_points(
            self.fine_points, self.show_fine_points.isChecked()
        )

    def update_ct_visualization(self):
        self.visualization_3d.show_ct_pointcloud(self.show_ct_cloud.isChecked())

    def load_ct_pointcloud(self):
        """Load a different CT point cloud file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load CT Point Cloud", "",
            "NumPy Files (*.npy);;Point Cloud Files (*.ply *.stl *.obj *.vtk);;All Files (*)"
        )

        if file_path:
            try:
                print(f"Loading CT point cloud from: {file_path}")

                if file_path.endswith('.npy'):
                    ct_points = np.load(file_path)
                    if ct_points.shape[1] >= 3:
                        self.visualization_3d.load_ct_pointcloud(ct_points[:, :3])
                    else:
                        raise ValueError("Invalid point cloud format - need at least 3 columns")
                else:
                    # Load other formats using PyVista
                    ct_pointcloud = pv.read(file_path)
                    self.visualization_3d.load_ct_pointcloud(ct_pointcloud.points)

                # Auto-check the checkbox and update visualization
                self.show_ct_cloud.setChecked(True)
                self.update_ct_visualization()

                show_message_box(self, "Success", f"Loaded CT point cloud: {len(ct_points)} points")

            except Exception as e:
                show_message_box(self, "Error", f"Error loading CT point cloud: {str(e)}", "error")

    def debug_ct_pointcloud(self):
        """Debug method to check CT point cloud status"""
        if hasattr(self.visualization_3d, 'ct_pointcloud') and self.visualization_3d.ct_pointcloud is not None:
            ct_pc = self.visualization_3d.ct_pointcloud
            bounds = ct_pc.bounds
            debug_info = f"""CT Point Cloud Debug Info:
Points: {ct_pc.n_points}
Bounds: X[{bounds[0]:.1f}, {bounds[1]:.1f}] Y[{bounds[2]:.1f}, {bounds[3]:.1f}] Z[{bounds[4]:.1f}, {bounds[5]:.1f}]
Center: [{ct_pc.center[0]:.1f}, {ct_pc.center[1]:.1f}, {ct_pc.center[2]:.1f}]
Show checkbox: {self.show_ct_cloud.isChecked()}
Actor exists: {self.visualization_3d.ct_actor is not None}"""
        else:
            debug_info = "No CT point cloud loaded"

        show_message_box(self, "CT Point Cloud Debug", debug_info)
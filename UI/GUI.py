import sys
import json
import numpy as np
import requests
import time
from threading import Thread, Event
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                             QWidget, QPushButton, QLabel, QLineEdit, QSpinBox,
                             QCheckBox, QGroupBox, QGridLayout, QTextEdit, QTabWidget,
                             QComboBox, QDoubleSpinBox, QProgressBar, QMessageBox,
                             QFileDialog, QSplitter)
from PyQt5.QtCore import QTimer, QThread, pyqtSignal, Qt
from PyQt5.QtGui import QFont, QColor, QPalette

import pyvista as pv
from pyvistaqt import QtInteractor
import vtk


class NDITrackingGUI(QMainWindow):
    def __init__(self, server_url="http://localhost:8000"):
        super().__init__()
        self.server_url = server_url
        self.setWindowTitle("NDI Tracking Visualization System")
        self.setGeometry(100, 100, 1600, 1000)

        # Initialize data storage
        self.coarse_points = []
        self.fine_points = []
        self.ct_pointcloud = None
        self.streaming_active = False
        self.current_position = None

        # Initialize UI
        self.init_ui()

        # Start status update timer
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_status)
        self.status_timer.start(1000)  # Update every second

        # Streaming update timer
        self.streaming_timer = QTimer()
        self.streaming_timer.timeout.connect(self.update_streaming_visualization)

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Create main splitter
        main_splitter = QSplitter(Qt.Horizontal)

        # Left panel for controls
        control_panel = self.create_control_panel()
        main_splitter.addWidget(control_panel)

        # Right panel for 3D visualization
        viz_panel = self.create_visualization_panel()
        main_splitter.addWidget(viz_panel)

        # Set splitter proportions
        main_splitter.setSizes([400, 1200])

        layout = QHBoxLayout()
        layout.addWidget(main_splitter)
        central_widget.setLayout(layout)

    def create_control_panel(self):
        control_widget = QWidget()
        control_widget.setMaximumWidth(400)
        control_widget.setMinimumWidth(350)

        layout = QVBoxLayout()

        # Server connection group
        server_group = self.create_server_group()
        layout.addWidget(server_group)

        # Create tabs for different functions
        tab_widget = QTabWidget()

        # Coarse registration tab
        coarse_tab = self.create_coarse_tab()
        tab_widget.addTab(coarse_tab, "Coarse Registration")

        # Fine registration tab
        fine_tab = self.create_fine_tab()
        tab_widget.addTab(fine_tab, "Fine Registration")

        # Tool calibration tab
        tool_tab = self.create_tool_tab()
        tab_widget.addTab(tool_tab, "Tool Calibration")

        # Streaming tab
        streaming_tab = self.create_streaming_tab()
        tab_widget.addTab(streaming_tab, "Streaming")

        layout.addWidget(tab_widget)

        # Status display
        status_group = self.create_status_group()
        layout.addWidget(status_group)

        control_widget.setLayout(layout)
        return control_widget

    def create_server_group(self):
        group = QGroupBox("Server Connection")
        layout = QGridLayout()

        # Server URL
        layout.addWidget(QLabel("Server URL:"), 0, 0)
        self.server_url_input = QLineEdit(self.server_url)
        layout.addWidget(self.server_url_input, 0, 1)

        # Connect button
        self.connect_btn = QPushButton("Test Connection")
        self.connect_btn.clicked.connect(self.test_connection)
        layout.addWidget(self.connect_btn, 1, 0, 1, 2)

        # Initialize NDI button
        self.init_ndi_btn = QPushButton("Initialize NDI")
        self.init_ndi_btn.clicked.connect(self.initialize_ndi)
        layout.addWidget(self.init_ndi_btn, 2, 0, 1, 2)

        # Connection status
        self.connection_status = QLabel("Not connected")
        layout.addWidget(self.connection_status, 3, 0, 1, 2)

        group.setLayout(layout)
        return group

    def create_coarse_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()

        # Point input group
        point_group = QGroupBox("Point Input")
        point_layout = QGridLayout()

        # Unity point coordinates
        point_layout.addWidget(QLabel("Unity X:"), 0, 0)
        self.unity_x = QDoubleSpinBox()
        self.unity_x.setRange(-1000, 1000)
        self.unity_x.setDecimals(3)
        point_layout.addWidget(self.unity_x, 0, 1)

        point_layout.addWidget(QLabel("Unity Y:"), 1, 0)
        self.unity_y = QDoubleSpinBox()
        self.unity_y.setRange(-1000, 1000)
        self.unity_y.setDecimals(3)
        point_layout.addWidget(self.unity_y, 1, 1)

        point_layout.addWidget(QLabel("Unity Z:"), 2, 0)
        self.unity_z = QDoubleSpinBox()
        self.unity_z.setRange(-1000, 1000)
        self.unity_z.setDecimals(3)
        point_layout.addWidget(self.unity_z, 2, 1)

        # Point number
        point_layout.addWidget(QLabel("Point #:"), 3, 0)
        self.point_number = QSpinBox()
        self.point_number.setRange(0, 100)
        point_layout.addWidget(self.point_number, 3, 1)

        # Set point button
        self.set_point_btn = QPushButton("Set Coarse Point")
        self.set_point_btn.clicked.connect(self.set_coarse_point)
        point_layout.addWidget(self.set_point_btn, 4, 0, 1, 2)

        point_group.setLayout(point_layout)
        layout.addWidget(point_group)

        # Registration group
        reg_group = QGroupBox("Registration")
        reg_layout = QVBoxLayout()

        self.coarse_register_btn = QPushButton("Perform Coarse Registration")
        self.coarse_register_btn.clicked.connect(self.perform_coarse_registration)
        reg_layout.addWidget(self.coarse_register_btn)

        self.reset_coarse_btn = QPushButton("Reset Coarse Points")
        self.reset_coarse_btn.clicked.connect(self.reset_coarse_points)
        reg_layout.addWidget(self.reset_coarse_btn)

        # Visualization options
        self.show_coarse_points = QCheckBox("Show Coarse Points")
        self.show_coarse_points.setChecked(True)
        self.show_coarse_points.toggled.connect(self.update_coarse_visualization)
        reg_layout.addWidget(self.show_coarse_points)

        reg_group.setLayout(reg_layout)
        layout.addWidget(reg_group)

        # Coarse points list
        self.coarse_points_text = QTextEdit()
        self.coarse_points_text.setMaximumHeight(150)
        self.coarse_points_text.setReadOnly(True)
        layout.addWidget(QLabel("Coarse Points:"))
        layout.addWidget(self.coarse_points_text)

        widget.setLayout(layout)
        return widget

    def create_fine_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()

        # Fine registration controls
        fine_group = QGroupBox("Fine Registration")
        fine_layout = QGridLayout()

        # Gathering frequency
        fine_layout.addWidget(QLabel("Frequency (Hz):"), 0, 0)
        self.fine_frequency = QSpinBox()
        self.fine_frequency.setRange(1, 100)
        self.fine_frequency.setValue(60)
        fine_layout.addWidget(self.fine_frequency, 0, 1)

        # Start/Stop gathering
        self.start_fine_btn = QPushButton("Start Fine Gathering")
        self.start_fine_btn.clicked.connect(self.start_fine_gathering)
        fine_layout.addWidget(self.start_fine_btn, 1, 0, 1, 2)

        self.stop_fine_btn = QPushButton("Stop Fine Gathering")
        self.stop_fine_btn.clicked.connect(self.stop_fine_gathering)
        self.stop_fine_btn.setEnabled(False)
        fine_layout.addWidget(self.stop_fine_btn, 2, 0, 1, 2)

        # Registration parameters
        fine_layout.addWidget(QLabel("Model ID:"), 3, 0)
        self.model_id = QSpinBox()
        self.model_id.setRange(0, 10)
        fine_layout.addWidget(self.model_id, 3, 1)

        fine_layout.addWidget(QLabel("Downsample:"), 4, 0)
        self.downsample_factor = QDoubleSpinBox()
        self.downsample_factor.setRange(0.1, 1.0)
        self.downsample_factor.setValue(1.0)
        self.downsample_factor.setSingleStep(0.1)
        fine_layout.addWidget(self.downsample_factor, 4, 1)

        # Perform fine registration
        self.fine_register_btn = QPushButton("Perform Fine Registration")
        self.fine_register_btn.clicked.connect(self.perform_fine_registration)
        fine_layout.addWidget(self.fine_register_btn, 5, 0, 1, 2)

        self.reset_fine_btn = QPushButton("Reset Fine Gathering")
        self.reset_fine_btn.clicked.connect(self.reset_fine_gathering)
        fine_layout.addWidget(self.reset_fine_btn, 6, 0, 1, 2)

        fine_group.setLayout(fine_layout)
        layout.addWidget(fine_group)

        # Visualization options
        viz_group = QGroupBox("Visualization")
        viz_layout = QVBoxLayout()

        self.show_fine_points = QCheckBox("Show Fine Points")
        self.show_fine_points.setChecked(True)
        self.show_fine_points.toggled.connect(self.update_fine_visualization)
        viz_layout.addWidget(self.show_fine_points)

        self.show_ct_cloud = QCheckBox("Show CT Point Cloud")
        self.show_ct_cloud.setChecked(False)
        self.show_ct_cloud.toggled.connect(self.update_ct_visualization)
        viz_layout.addWidget(self.show_ct_cloud)

        self.load_ct_btn = QPushButton("Load CT Point Cloud")
        self.load_ct_btn.clicked.connect(self.load_ct_pointcloud)
        viz_layout.addWidget(self.load_ct_btn)

        viz_group.setLayout(viz_layout)
        layout.addWidget(viz_group)

        # Fine points status
        self.fine_status_text = QTextEdit()
        self.fine_status_text.setMaximumHeight(100)
        self.fine_status_text.setReadOnly(True)
        layout.addWidget(QLabel("Fine Registration Status:"))
        layout.addWidget(self.fine_status_text)

        widget.setLayout(layout)
        return widget

    def create_tool_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()

        # Tool calibration group
        tool_group = QGroupBox("Tool Calibration")
        tool_layout = QGridLayout()

        # Device selection
        tool_layout.addWidget(QLabel("Device:"), 0, 0)
        self.device_spinbox = QSpinBox()
        self.device_spinbox.setRange(0, 5)
        tool_layout.addWidget(self.device_spinbox, 0, 1)

        # Start/Stop calibration
        self.start_calib_btn = QPushButton("Start Tool Calibration")
        self.start_calib_btn.clicked.connect(self.start_tool_calibration)
        tool_layout.addWidget(self.start_calib_btn, 1, 0, 1, 2)

        self.stop_calib_btn = QPushButton("Stop Tool Calibration")
        self.stop_calib_btn.clicked.connect(self.stop_tool_calibration)
        self.stop_calib_btn.setEnabled(False)
        tool_layout.addWidget(self.stop_calib_btn, 2, 0, 1, 2)

        # Calibrate tool
        self.calibrate_btn = QPushButton("Calibrate Tool")
        self.calibrate_btn.clicked.connect(self.calibrate_tool)
        tool_layout.addWidget(self.calibrate_btn, 3, 0, 1, 2)

        tool_group.setLayout(tool_layout)
        layout.addWidget(tool_group)

        # Tool status
        self.tool_status_text = QTextEdit()
        self.tool_status_text.setMaximumHeight(150)
        self.tool_status_text.setReadOnly(True)
        layout.addWidget(QLabel("Tool Calibration Status:"))
        layout.addWidget(self.tool_status_text)

        widget.setLayout(layout)
        return widget

    def create_streaming_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()

        # Streaming controls
        stream_group = QGroupBox("Streaming Controls")
        stream_layout = QGridLayout()

        # Port and frequency
        stream_layout.addWidget(QLabel("Port:"), 0, 0)
        self.stream_port = QSpinBox()
        self.stream_port.setRange(1024, 65535)
        self.stream_port.setValue(11111)
        stream_layout.addWidget(self.stream_port, 0, 1)

        stream_layout.addWidget(QLabel("Frequency (Hz):"), 1, 0)
        self.stream_frequency = QSpinBox()
        self.stream_frequency.setRange(1, 100)
        self.stream_frequency.setValue(30)
        stream_layout.addWidget(self.stream_frequency, 1, 1)

        # Start/Stop streaming
        self.start_stream_btn = QPushButton("Start Streaming")
        self.start_stream_btn.clicked.connect(self.start_streaming)
        stream_layout.addWidget(self.start_stream_btn, 2, 0, 1, 2)

        self.stop_stream_btn = QPushButton("Stop Streaming")
        self.stop_stream_btn.clicked.connect(self.stop_streaming)
        self.stop_stream_btn.setEnabled(False)
        stream_layout.addWidget(self.stop_stream_btn, 3, 0, 1, 2)

        stream_group.setLayout(stream_layout)
        layout.addWidget(stream_group)

        # Visualization options
        stream_viz_group = QGroupBox("Real-time Visualization")
        stream_viz_layout = QVBoxLayout()

        self.show_realtime_position = QCheckBox("Show Real-time Position")
        self.show_realtime_position.setChecked(True)
        self.show_realtime_position.toggled.connect(self.update_streaming_visualization)
        stream_viz_layout.addWidget(self.show_realtime_position)

        self.show_probe_trail = QCheckBox("Show Probe Trail")
        self.show_probe_trail.setChecked(False)
        self.show_probe_trail.toggled.connect(self.update_streaming_visualization)
        stream_viz_layout.addWidget(self.show_probe_trail)

        # Clear trail button
        self.clear_trail_btn = QPushButton("Clear Trail")
        self.clear_trail_btn.clicked.connect(self.clear_probe_trail)
        stream_viz_layout.addWidget(self.clear_trail_btn)

        stream_viz_group.setLayout(stream_viz_layout)
        layout.addWidget(stream_viz_group)

        # Current position display
        self.position_text = QTextEdit()
        self.position_text.setMaximumHeight(100)
        self.position_text.setReadOnly(True)
        layout.addWidget(QLabel("Current Position:"))
        layout.addWidget(self.position_text)

        widget.setLayout(layout)
        return widget

    def create_status_group(self):
        group = QGroupBox("System Status")
        layout = QVBoxLayout()

        self.status_text = QTextEdit()
        self.status_text.setMaximumHeight(150)
        self.status_text.setReadOnly(True)
        layout.addWidget(self.status_text)

        group.setLayout(layout)
        return group

    def create_visualization_panel(self):
        # Create PyVista plotter widget
        self.plotter = QtInteractor(self)
        self.plotter.background_color = 'white'

        # Initialize visualization elements
        self.coarse_actors = []
        self.fine_actors = []
        self.ct_actor = None
        self.probe_actor = None
        self.probe_trail = []

        # Add coordinate axes
        self.plotter.add_axes()

        return self.plotter.interactor

    # Server communication methods
    def test_connection(self):
        try:
            self.server_url = self.server_url_input.text()
            response = requests.get(f"{self.server_url}/", timeout=5)
            if response.status_code == 200:
                self.connection_status.setText("Connected ✓")
                self.connection_status.setStyleSheet("color: green")
                self.update_status()
            else:
                self.connection_status.setText("Connection failed")
                self.connection_status.setStyleSheet("color: red")
        except Exception as e:
            self.connection_status.setText(f"Error: {str(e)}")
            self.connection_status.setStyleSheet("color: red")

    def initialize_ndi(self):
        try:
            response = requests.post(f"{self.server_url}/initialize_ndi")
            if response.status_code == 200:
                result = response.json()
                self.show_message("NDI Initialization", f"Status: {result.get('status', 'Unknown')}")
            else:
                self.show_message("Error", f"Failed to initialize NDI: {response.status_code}")
        except Exception as e:
            self.show_message("Error", f"Error initializing NDI: {str(e)}")

    def update_status(self):
        try:
            response = requests.get(f"{self.server_url}/")
            if response.status_code == 200:
                status = response.json()
                status_text = f"""
Application: {status.get('application', 'Unknown')}
Status: {status.get('status', 'Unknown')}
Data Source: {status.get('data_source', 'Unknown')}
NDI Tracker: {status.get('ndi_tracker_status', 'Unknown')}
Coarse Points: {status.get('coarse_points_loaded', 0)}
Fine Points: {status.get('fine_points_loaded', 0)}
Streaming: {status.get('streaming_active', False)}
Coarse Registration: {status.get('has_coarse_registration', False)}
Fine Registration: {status.get('has_fine_registration', False)}
Tool Calibration: {status.get('has_tool_calibration', False)}
                """
                self.status_text.setText(status_text.strip())
        except Exception as e:
            self.status_text.setText(f"Error getting status: {str(e)}")

    # Coarse registration methods
    def set_coarse_point(self):
        try:
            unity_point = [self.unity_x.value(), self.unity_y.value(), self.unity_z.value()]
            point_number = self.point_number.value()

            data = {
                "unity_point": unity_point,
                "point_number": point_number
            }

            response = requests.post(f"{self.server_url}/set_coarse_point", json=data)
            if response.status_code == 200:
                result = response.json()
                if result.get('status') == 'success':
                    self.update_coarse_points_display()
                    self.update_coarse_visualization()
                    self.show_message("Success", f"Point {point_number} set successfully")
                else:
                    self.show_message("Error", result.get('details', 'Unknown error'))
            else:
                self.show_message("Error", f"Failed to set point: {response.status_code}")
        except Exception as e:
            self.show_message("Error", f"Error setting coarse point: {str(e)}")

    def perform_coarse_registration(self):
        try:
            response = requests.post(f"{self.server_url}/coarse_register", params={"visualize": False})
            if response.status_code == 200:
                result = response.json()
                if result.get('status') == 'success':
                    self.show_message("Success", "Coarse registration completed successfully")
                    self.update_coarse_visualization()
                else:
                    self.show_message("Error", result.get('details', 'Registration failed'))
            else:
                self.show_message("Error", f"Failed to perform registration: {response.status_code}")
        except Exception as e:
            self.show_message("Error", f"Error performing coarse registration: {str(e)}")

    def reset_coarse_points(self):
        try:
            response = requests.post(f"{self.server_url}/reset_coarse_points")
            if response.status_code == 200:
                self.update_coarse_points_display()
                self.update_coarse_visualization()
                self.show_message("Success", "Coarse points reset")
        except Exception as e:
            self.show_message("Error", f"Error resetting coarse points: {str(e)}")

    def update_coarse_points_display(self):
        # This would need to be implemented based on your server's coarse points endpoint
        # For now, just update from the status
        pass

    def update_coarse_visualization(self):
        # Remove existing coarse actors
        for actor in self.coarse_actors:
            self.plotter.remove_actor(actor)
        self.coarse_actors.clear()

        if self.show_coarse_points.isChecked():
            # Get coarse points from server and visualize
            # This would need to be implemented based on your server's API
            pass

    # Fine registration methods
    def start_fine_gathering(self):
        try:
            frequency = self.fine_frequency.value()
            response = requests.post(f"{self.server_url}/start_fine_gather", params={"frequency": frequency})
            if response.status_code == 200:
                result = response.json()
                if result.get('status') == 'success':
                    self.start_fine_btn.setEnabled(False)
                    self.stop_fine_btn.setEnabled(True)
                    self.show_message("Success", "Fine gathering started")
                    # Start timer to update fine points visualization
                    self.fine_timer = QTimer()
                    self.fine_timer.timeout.connect(self.update_fine_points_status)
                    self.fine_timer.start(500)  # Update every 500ms
                else:
                    self.show_message("Error", result.get('details', 'Failed to start gathering'))
        except Exception as e:
            self.show_message("Error", f"Error starting fine gathering: {str(e)}")

    def stop_fine_gathering(self):
        try:
            response = requests.post(f"{self.server_url}/end_fine_gather")
            if response.status_code == 200:
                self.start_fine_btn.setEnabled(True)
                self.stop_fine_btn.setEnabled(False)
                if hasattr(self, 'fine_timer'):
                    self.fine_timer.stop()
                self.show_message("Success", "Fine gathering stopped")
                self.update_fine_points_status()
        except Exception as e:
            self.show_message("Error", f"Error stopping fine gathering: {str(e)}")

    def perform_fine_registration(self):
        try:
            model_id = self.model_id.value()
            downsample = self.downsample_factor.value()

            params = {
                "id": model_id,
                "downsample_factor": downsample,
                "visualize": False
            }

            response = requests.post(f"{self.server_url}/fine_register", params=params)
            if response.status_code == 200:
                result = response.json()
                if result.get('status') == 'success':
                    self.show_message("Success", "Fine registration completed successfully")
                    self.update_fine_visualization()
                else:
                    self.show_message("Error", result.get('details', 'Registration failed'))
            else:
                self.show_message("Error", f"Failed to perform fine registration: {response.status_code}")
        except Exception as e:
            self.show_message("Error", f"Error performing fine registration: {str(e)}")

    def reset_fine_gathering(self):
        try:
            response = requests.post(f"{self.server_url}/reset_fine_gather")
            if response.status_code == 200:
                self.update_fine_points_status()
                self.show_message("Success", "Fine gathering reset")
        except Exception as e:
            self.show_message("Error", f"Error resetting fine gathering: {str(e)}")

    def update_fine_points_status(self):
        try:
            response = requests.get(f"{self.server_url}/get_fine_points_status")
            if response.status_code == 200:
                status = response.json()
                status_text = f"""
Gathering Active: {status.get('gathering_active', False)}
Points Collected: {status.get('points_collected', 0)}
Collection Rate: {status.get('collection_rate', 0):.1f} Hz
                """
                self.fine_status_text.setText(status_text.strip())

                # Update visualization if gathering is active
                if status.get('gathering_active', False):
                    self.update_fine_visualization()
        except Exception as e:
            self.fine_status_text.setText(f"Error getting status: {str(e)}")

    def update_fine_visualization(self):
        # Remove existing fine actors
        for actor in self.fine_actors:
            self.plotter.remove_actor(actor)
        self.fine_actors.clear()

        if self.show_fine_points.isChecked():
            # Get fine points from server and visualize
            # This would need access to the fine points data
            pass

    # Tool calibration methods
    def start_tool_calibration(self):
        try:
            device = self.device_spinbox.value()
            response = requests.post(f"{self.server_url}/start_tool_calibration", params={"device": device})
            if response.status_code == 200:
                result = response.json()
                if result.get('status') == 'success':
                    self.start_calib_btn.setEnabled(False)
                    self.stop_calib_btn.setEnabled(True)
                    self.show_message("Success", "Tool calibration started")
                else:
                    self.show_message("Error", result.get('details', 'Failed to start calibration'))
        except Exception as e:
            self.show_message("Error", f"Error starting tool calibration: {str(e)}")

    def stop_tool_calibration(self):
        try:
            response = requests.post(f"{self.server_url}/end_tool_calibration")
            if response.status_code == 200:
                self.start_calib_btn.setEnabled(True)
                self.stop_calib_btn.setEnabled(False)
                self.show_message("Success", "Tool calibration stopped")
                self.update_tool_status()
        except Exception as e:
            self.show_message("Error", f"Error stopping tool calibration: {str(e)}")

    def calibrate_tool(self):
        try:
            response = requests.post(f"{self.server_url}/calibrate_tool", params={"visualize": False})
            if response.status_code == 200:
                result = response.json()
                if result.get('status') == 'success':
                    self.show_message("Success", "Tool calibration completed successfully")
                    self.update_tool_status()
                else:
                    self.show_message("Error", result.get('details', 'Calibration failed'))
        except Exception as e:
            self.show_message("Error", f"Error calibrating tool: {str(e)}")

    def update_tool_status(self):
        try:
            response = requests.get(f"{self.server_url}/get_tool_calibration_status")
            if response.status_code == 200:
                status = response.json()
                status_text = f"""
Calibration Active: {status.get('calibration_active', False)}
Matrices Collected: {status.get('matrices_collected', 0)}
Tool Tip Vector: {status.get('tool_tip_vector', 'Not calibrated')}
                """
                self.tool_status_text.setText(status_text.strip())
        except Exception as e:
            self.tool_status_text.setText(f"Error getting status: {str(e)}")

    # Streaming methods
    def start_streaming(self):
        try:
            port = self.stream_port.value()
            frequency = self.stream_frequency.value()

            params = {
                "port": port,
                "frequency": frequency
            }

            response = requests.post(f"{self.server_url}/start_streaming", params=params)
            if response.status_code == 200:
                result = response.json()
                if result.get('status') in ['started', 'already_running']:
                    self.streaming_active = True
                    self.start_stream_btn.setEnabled(False)
                    self.stop_stream_btn.setEnabled(True)
                    self.streaming_timer.start(100)  # Update every 100ms
                    self.show_message("Success", "Streaming started")
                else:
                    self.show_message("Error", result.get('details', 'Failed to start streaming'))
        except Exception as e:
            self.show_message("Error", f"Error starting streaming: {str(e)}")

    def stop_streaming(self):
        try:
            response = requests.post(f"{self.server_url}/stop_streaming")
            if response.status_code == 200:
                self.streaming_active = False
                self.start_stream_btn.setEnabled(True)
                self.stop_stream_btn.setEnabled(False)
                self.streaming_timer.stop()
                self.show_message("Success", "Streaming stopped")
        except Exception as e:
            self.show_message("Error", f"Error stopping streaming: {str(e)}")

    def update_streaming_visualization(self):
        if not self.streaming_active:
            return

        try:
            response = requests.get(f"{self.server_url}/get_latest_position")
            if response.status_code == 200:
                result = response.json()
                if result.get('status') == 'success':
                    data = result.get('data', {})
                    position = data.get('position')

                    if position and self.show_realtime_position.isChecked():
                        # Update probe position visualization
                        self.update_probe_position(position)

                        # Update trail if enabled
                        if self.show_probe_trail.isChecked():
                            self.add_to_probe_trail(position)

                    # Update position display
                    pos_text = f"""
Position: {position if position else 'No data'}
Timestamp: {data.get('timestamp', 'Unknown')}
Frame: {data.get('frame', 'Unknown')}
                    """
                    self.position_text.setText(pos_text.strip())

        except Exception as e:
            self.position_text.setText(f"Error getting position: {str(e)}")

    def update_probe_position(self, position):
        # Remove existing probe actor
        if self.probe_actor:
            self.plotter.remove_actor(self.probe_actor)

        # Create sphere at probe position
        sphere = pv.Sphere(radius=2.0, center=position)
        self.probe_actor = self.plotter.add_mesh(sphere, color='red', name='probe_position')

    def add_to_probe_trail(self, position):
        self.probe_trail.append(position)

        # Limit trail length
        if len(self.probe_trail) > 1000:
            self.probe_trail.pop(0)

        # Update trail visualization
        if len(self.probe_trail) > 1:
            trail_points = np.array(self.probe_trail)
            trail_line = pv.PolyData(trail_points)

            # Remove existing trail
            if hasattr(self, 'trail_actor'):
                self.plotter.remove_actor(self.trail_actor)

            self.trail_actor = self.plotter.add_mesh(trail_line, color='blue',
                                                     line_width=2, name='probe_trail')

    def clear_probe_trail(self):
        self.probe_trail.clear()
        if hasattr(self, 'trail_actor'):
            self.plotter.remove_actor(self.trail_actor)

    # CT point cloud methods
    def load_ct_pointcloud(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load CT Point Cloud", "",
            "Point Cloud Files (*.ply *.stl *.obj *.vtk);;All Files (*)"
        )

        if file_path:
            try:
                self.ct_pointcloud = pv.read(file_path)
                self.show_message("Success", f"Loaded CT point cloud: {file_path}")
                self.update_ct_visualization()
            except Exception as e:
                self.show_message("Error", f"Error loading CT point cloud: {str(e)}")

    def update_ct_visualization(self):
        # Remove existing CT actor
        if self.ct_actor:
            self.plotter.remove_actor(self.ct_actor)
            self.ct_actor = None

        if self.show_ct_cloud.isChecked() and self.ct_pointcloud is not None:
            self.ct_actor = self.plotter.add_mesh(
                self.ct_pointcloud,
                color='lightgray',
                opacity=0.5,
                name='ct_pointcloud'
            )

    # Utility methods
    def show_message(self, title, message):
        msg = QMessageBox()
        msg.setWindowTitle(title)
        msg.setText(message)
        msg.exec_()

    def closeEvent(self, event):
        # Stop streaming if active
        if self.streaming_active:
            try:
                requests.post(f"{self.server_url}/stop_streaming")
            except:
                pass

        event.accept()


def main():
    app = QApplication(sys.argv)

    # Set application style
    app.setStyle('Fusion')

    # Create and show the GUI
    gui = NDITrackingGUI()
    gui.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
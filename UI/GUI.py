import sys
import time
import threading
import numpy as np
import requests
import json
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget, QPushButton,
                             QVBoxLayout, QHBoxLayout, QLabel, QTextEdit, QTableWidget,
                             QTableWidgetItem, QHeaderView, QSplitter, QCheckBox, QSpinBox,
                             QDoubleSpinBox, QFormLayout, QGroupBox, QMessageBox)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
import matplotlib

matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Server configuration
SERVER_URL = "http://127.0.0.1:8000"


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111, projection='3d')
        super(MplCanvas, self).__init__(fig)


class ServerStatusThread(QThread):
    update_signal = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.running = True

    def run(self):
        while self.running:
            try:
                response = requests.get(f"{SERVER_URL}/")
                if response.status_code == 200:
                    data = response.json()
                    self.update_signal.emit(data)
                time.sleep(1)  # Update every second
            except Exception as e:
                print(f"Error fetching server status: {e}")
                time.sleep(3)  # Wait longer on error

    def stop(self):
        self.running = False
        self.wait()


class CoarsePointsThread(QThread):
    update_signal = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.running = True

    def run(self):
        while self.running:
            try:
                # Signal the main thread to refresh coarse points
                self.update_signal.emit()
                time.sleep(0.5)  # Update every 0.5 seconds
            except Exception as e:
                print(f"Error in coarse points thread: {e}")
                time.sleep(2)  # Wait longer on error

    def stop(self):
        self.running = False
        self.wait()


class NDITrackingUI(QMainWindow):
    def __init__(self, ndi_server=None, config=None, args=None):
        super().__init__()

        # Store reference to the NDI server and config
        self.ndi_server = ndi_server
        self.config = config or {}
        self.args = args or {}

        # Set window properties
        self.setWindowTitle("NDI Tracking Server Interface")
        self.setGeometry(100, 100, 1200, 800)

        # Store coarse points data
        self.coarse_unity_points = []
        self.coarse_ndi_points = []
        self.last_coarse_points_count = 0

        # CT Point Cloud data - initialize first
        self.ct_point_cloud = None
        self.show_ct_point_cloud = False

        # Create tab widget
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Create tabs
        self.init_tab = QWidget()
        self.coarse_tab = QWidget()
        self.fine_tab = QWidget()
        self.streaming_tab = QWidget()
        self.tool_calibration_tab = QWidget()

        # Add tabs to widget
        self.tabs.addTab(self.init_tab, "Initialization")
        self.tabs.addTab(self.coarse_tab, "Coarse Registration")
        self.tabs.addTab(self.fine_tab, "Fine Registration")
        self.tabs.addTab(self.streaming_tab, "Streaming")
        self.tabs.addTab(self.tool_calibration_tab, "Tool Calibration")

        # Connect tab changed signal to update flags
        self.tabs.currentChanged.connect(self.on_tab_changed)

        # Setup each tab
        self.setup_init_tab()
        self.setup_coarse_tab()
        self.setup_fine_tab()
        self.setup_streaming_tab()
        self.setup_tool_calibration_tab()

        # Load CT point cloud AFTER UI is set up
        self.load_ct_point_cloud()

        # Start background threads
        self.server_status_thread = ServerStatusThread()
        self.server_status_thread.update_signal.connect(self.update_server_status)
        self.server_status_thread.start()

        # Start coarse points thread for more frequent updates
        self.coarse_points_thread = CoarsePointsThread()
        self.coarse_points_thread.update_signal.connect(self.refresh_coarse_points)
        self.coarse_points_thread.start()

        # Use timer for fine points visualization to avoid blocking the UI
        self.fine_points_timer = QTimer()
        self.fine_points_timer.timeout.connect(self.update_fine_points)
        self.fine_points_timer.start(100)  # Update at 10 Hz

        # Variable to keep track of the last number of fine points for efficient updates
        self.last_fine_points_count = 0
        # Store a downsampled copy of fine points for visualization
        self.fine_points_to_plot = []
        # Flag to indicate if full replot is needed (e.g. on tab switch)
        self.fine_plot_needs_full_update = True

    def load_ct_point_cloud(self):
        """Load CT point cloud data from the specified file in config"""
        try:
            if self.config and "CT_PC_address" in self.config:
                file_path = self.config["CT_PC_address"]
                print(f"Loading CT point cloud from: {file_path}")  # Debug print
                self.log_message(f"Loading CT point cloud from: {file_path}")

                # Check if file exists
                import os
                if not os.path.exists(file_path):
                    print(f"CT point cloud file not found: {file_path}")
                    self.log_message(f"CT point cloud file not found: {file_path}")
                    return

                # Load point cloud data
                self.ct_point_cloud = np.load(file_path)
                print(f"CT point cloud shape: {self.ct_point_cloud.shape}")  # Debug print

                # Ensure it's 3D points
                if len(self.ct_point_cloud.shape) != 2 or self.ct_point_cloud.shape[1] != 3:
                    print(f"Invalid CT point cloud shape: {self.ct_point_cloud.shape}. Expected (N, 3)")
                    self.log_message(f"Invalid CT point cloud shape: {self.ct_point_cloud.shape}. Expected (N, 3)")
                    self.ct_point_cloud = None
                    return

                # If the point cloud is too large, downsample it for better performance
                if len(self.ct_point_cloud) > 10000:
                    # Randomly sample 10000 points
                    indices = np.random.choice(len(self.ct_point_cloud), size=10000, replace=False)
                    self.ct_point_cloud = self.ct_point_cloud[indices]
                    print(f"Downsampled CT point cloud to 10000 points")

                print(f"CT point cloud loaded successfully: {len(self.ct_point_cloud)} points")
                self.log_message(f"CT point cloud loaded: {len(self.ct_point_cloud)} points")

                # Enable the toggle buttons now that we have data
                if hasattr(self, 'toggle_ct_pc_btn'):
                    self.toggle_ct_pc_btn.setEnabled(True)
                if hasattr(self, 'fine_toggle_ct_pc_btn'):
                    self.fine_toggle_ct_pc_btn.setEnabled(True)

            else:
                print("No CT_PC_address found in config")
                self.log_message("No CT_PC_address found in config")

        except Exception as e:
            self.ct_point_cloud = None
            print(f"Error loading CT point cloud: {e}")
            self.log_message(f"Error loading CT point cloud: {e}")

    def transform_ct_point_cloud(self):
        """Transform CT point cloud using the current coarse registration transformation"""
        if self.ct_point_cloud is None or not hasattr(self.ndi_server, 'coarse_registration'):
            return None

        # Check if transformation matrix exists
        if not hasattr(self.ndi_server.coarse_registration,
                       'transformation_matrix') or self.ndi_server.coarse_registration.transformation_matrix is None:
            print("No transformation matrix available")
            return None

        # Get transformation matrix
        transform = self.ndi_server.coarse_registration.transformation_matrix
        print(f"Using transformation matrix:\n{transform}")

        # Apply transformation to each point in the CT point cloud
        transformed_points = np.zeros_like(self.ct_point_cloud)
        for i, point in enumerate(self.ct_point_cloud):
            # Convert to homogeneous coordinates
            homog_point = np.append(point, 1)
            # Apply transformation
            transformed_homog = np.dot(transform, homog_point)
            # Convert back from homogeneous coordinates
            transformed_points[i] = transformed_homog[:3]

        print(f"Transformed {len(transformed_points)} CT points")
        return transformed_points


    def closeEvent(self, event):
        # Stop background threads when window is closed
        self.server_status_thread.stop()
        self.coarse_points_thread.stop()
        self.fine_points_timer.stop()
        event.accept()

    def setup_init_tab(self):
        layout = QVBoxLayout()

        # Status group
        status_group = QGroupBox("Server Status")
        status_layout = QFormLayout()

        self.status_label = QLabel("Checking server status...")
        self.ndi_status_label = QLabel("Unknown")
        self.data_source_label = QLabel("Unknown")
        self.coarse_points_label = QLabel("0")
        self.fine_points_label = QLabel("0")

        status_layout.addRow("Server Status:", self.status_label)
        status_layout.addRow("NDI Tracker:", self.ndi_status_label)
        status_layout.addRow("Data Source:", self.data_source_label)
        status_layout.addRow("Coarse Points:", self.coarse_points_label)
        status_layout.addRow("Fine Points:", self.fine_points_label)

        status_group.setLayout(status_layout)

        # NDI Controls group
        ndi_group = QGroupBox("NDI Tracker Controls")
        ndi_layout = QVBoxLayout()

        self.initialize_ndi_btn = QPushButton("Initialize NDI Tracker")
        self.initialize_ndi_btn.clicked.connect(self.initialize_ndi)

        self.force_restart_checkbox = QCheckBox("Force Restart")

        self.check_tools_btn = QPushButton("Check Tools")
        self.check_tools_btn.clicked.connect(self.check_tools)

        self.tool_status_label = QLabel("No tool information available")

        ndi_layout.addWidget(self.initialize_ndi_btn)
        ndi_layout.addWidget(self.force_restart_checkbox)
        ndi_layout.addWidget(self.check_tools_btn)
        ndi_layout.addWidget(self.tool_status_label)

        ndi_group.setLayout(ndi_layout)

        # Log output
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout()

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)

        log_layout.addWidget(self.log_output)
        log_group.setLayout(log_layout)

        # Add all groups to main layout
        layout.addWidget(status_group)
        layout.addWidget(ndi_group)
        layout.addWidget(log_group)

        self.init_tab.setLayout(layout)

    def setup_coarse_tab(self):
        layout = QHBoxLayout()

        # Left panel - Display
        left_panel = QWidget()
        left_layout = QVBoxLayout()

        # Points information
        info_group = QGroupBox("Coarse Registration Points")
        info_layout = QVBoxLayout()

        self.coarse_info_label = QLabel("Waiting for points from another client...")
        self.coarse_count_label = QLabel("0 points available")

        info_layout.addWidget(self.coarse_info_label)
        info_layout.addWidget(self.coarse_count_label)
        info_group.setLayout(info_layout)

        # Point table
        table_group = QGroupBox("Points Data")
        table_layout = QVBoxLayout()

        self.points_table = QTableWidget(0, 7)  # Rows will be added dynamically
        self.points_table.setHorizontalHeaderLabels(["#", "Unity X", "Unity Y", "Unity Z",
                                                     "NDI X", "NDI Y", "NDI Z"])
        self.points_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        table_layout.addWidget(self.points_table)
        table_group.setLayout(table_layout)

        # Registration controls
        reg_group = QGroupBox("Registration Controls")
        reg_layout = QVBoxLayout()

        self.register_btn = QPushButton("Perform Registration")
        self.register_btn.clicked.connect(self.perform_coarse_registration)

        self.visualize_checkbox = QCheckBox("Visualize Registration")

        self.reset_points_btn = QPushButton("Reset All Points")
        self.reset_points_btn.clicked.connect(self.reset_coarse_points)

        self.refresh_btn = QPushButton("Refresh Points Data")
        self.refresh_btn.clicked.connect(self.refresh_coarse_points_manual)

        # CT Point Cloud toggle
        self.toggle_ct_pc_btn = QPushButton("Show CT Point Cloud")
        self.toggle_ct_pc_btn.clicked.connect(self.toggle_ct_point_cloud)
        self.toggle_ct_pc_btn.setEnabled(False)  # Will be enabled if CT data is loaded

        reg_layout.addWidget(self.register_btn)
        reg_layout.addWidget(self.visualize_checkbox)
        reg_layout.addWidget(self.reset_points_btn)
        reg_layout.addWidget(self.refresh_btn)
        reg_layout.addWidget(self.toggle_ct_pc_btn)
        reg_group.setLayout(reg_layout)

        # Results group
        results_group = QGroupBox("Registration Results")
        results_layout = QVBoxLayout()

        self.coarse_results = QTextEdit()
        self.coarse_results.setReadOnly(True)

        results_layout.addWidget(self.coarse_results)
        results_group.setLayout(results_layout)

        # Add groups to left layout
        left_layout.addWidget(info_group)
        left_layout.addWidget(table_group)
        left_layout.addWidget(reg_group)
        left_layout.addWidget(results_group)
        left_panel.setLayout(left_layout)

        # Right panel - Visualization
        right_panel = QWidget()
        right_layout = QVBoxLayout()

        # 3D plot for point cloud visualization
        self.coarse_canvas = MplCanvas(self, width=5, height=4, dpi=100)
        self.coarse_canvas.axes.set_xlabel('X')
        self.coarse_canvas.axes.set_ylabel('Y')
        self.coarse_canvas.axes.set_zlabel('Z')
        self.coarse_canvas.axes.set_title('Coarse Registration Points')

        right_layout.addWidget(self.coarse_canvas)
        right_panel.setLayout(right_layout)

        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([400, 800])

        layout.addWidget(splitter)
        self.coarse_tab.setLayout(layout)

    def setup_fine_tab(self):
        layout = QHBoxLayout()

        # Left panel - Controls
        left_panel = QWidget()
        left_layout = QVBoxLayout()

        # Fine registration controls
        control_group = QGroupBox("Fine Registration Controls")
        control_layout = QVBoxLayout()

        # Start/stop gathering buttons
        gather_layout = QHBoxLayout()
        self.start_fine_gather_btn = QPushButton("Start Fine Point Gathering")
        self.start_fine_gather_btn.clicked.connect(self.start_fine_gather)

        self.end_fine_gather_btn = QPushButton("Stop Fine Point Gathering")
        self.end_fine_gather_btn.clicked.connect(self.end_fine_gather)

        self.reset_fine_gather_btn = QPushButton("Reset Fine Points")
        self.reset_fine_gather_btn.clicked.connect(self.reset_fine_gather)

        gather_layout.addWidget(self.start_fine_gather_btn)
        gather_layout.addWidget(self.end_fine_gather_btn)

        # Frequency input
        freq_layout = QHBoxLayout()
        self.frequency_spin = QSpinBox()
        self.frequency_spin.setRange(1, 100)
        self.frequency_spin.setValue(60)

        freq_layout.addWidget(QLabel("Frequency (Hz):"))
        freq_layout.addWidget(self.frequency_spin)

        # Fine registration parameters
        params_layout = QFormLayout()

        self.id_spin = QSpinBox()
        self.id_spin.setRange(0, 99)

        self.downsample_spin = QDoubleSpinBox()
        self.downsample_spin.setRange(0.1, 1.0)
        self.downsample_spin.setValue(1.0)
        self.downsample_spin.setSingleStep(0.1)

        self.fine_visualize_checkbox = QCheckBox()

        params_layout.addRow("Surface ID:", self.id_spin)
        params_layout.addRow("Downsample Factor:", self.downsample_spin)
        params_layout.addRow("Visualize:", self.fine_visualize_checkbox)

        # Register button
        self.fine_register_btn = QPushButton("Perform Fine Registration")
        self.fine_register_btn.clicked.connect(self.perform_fine_registration)

        # CT Point Cloud toggle - IMPORTANT: This is the toggle button for fine tab
        self.fine_toggle_ct_pc_btn = QPushButton("Show CT Point Cloud")
        self.fine_toggle_ct_pc_btn.clicked.connect(self.toggle_ct_point_cloud)
        self.fine_toggle_ct_pc_btn.setEnabled(False)  # Will be enabled if CT data is loaded
        self.fine_toggle_ct_pc_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }")

        # Status display
        self.fine_status_label = QLabel("Not gathering")
        self.fine_points_count_label = QLabel("0 points")

        # Add all to control layout
        control_layout.addLayout(gather_layout)
        control_layout.addWidget(self.reset_fine_gather_btn)
        control_layout.addLayout(freq_layout)
        control_layout.addLayout(params_layout)
        control_layout.addWidget(self.fine_register_btn)
        control_layout.addWidget(self.fine_toggle_ct_pc_btn)  # Make sure this is added
        control_layout.addWidget(self.fine_status_label)
        control_layout.addWidget(self.fine_points_count_label)

        control_group.setLayout(control_layout)

        # Results group
        results_group = QGroupBox("Fine Registration Results")
        results_layout = QVBoxLayout()

        self.fine_results = QTextEdit()
        self.fine_results.setReadOnly(True)

        results_layout.addWidget(self.fine_results)
        results_group.setLayout(results_layout)

        # Add groups to left layout
        left_layout.addWidget(control_group)
        left_layout.addWidget(results_group)
        left_panel.setLayout(left_layout)

        # Right panel - Visualization
        right_panel = QWidget()
        right_layout = QVBoxLayout()

        # 3D plot for point cloud visualization
        self.fine_canvas = MplCanvas(self, width=5, height=4, dpi=100)
        self.fine_canvas.axes.set_xlabel('X')
        self.fine_canvas.axes.set_ylabel('Y')
        self.fine_canvas.axes.set_zlabel('Z')
        self.fine_canvas.axes.set_title('Fine Registration Points')

        right_layout.addWidget(self.fine_canvas)
        right_panel.setLayout(right_layout)

        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([400, 800])

        layout.addWidget(splitter)
        self.fine_tab.setLayout(layout)

    def setup_streaming_tab(self):
        layout = QVBoxLayout()

        # Streaming controls
        control_group = QGroupBox("Streaming Controls")
        control_layout = QFormLayout()

        # IP and port inputs
        self.client_ip_input = QTextEdit()
        self.client_ip_input.setMaximumHeight(30)
        self.client_ip_input.setText("127.0.0.1")

        self.port_spin = QSpinBox()
        self.port_spin.setRange(1024, 65535)
        self.port_spin.setValue(11111)

        self.stream_freq_spin = QSpinBox()
        self.stream_freq_spin.setRange(1, 100)
        self.stream_freq_spin.setValue(30)

        control_layout.addRow("Client IP:", self.client_ip_input)
        control_layout.addRow("Port:", self.port_spin)
        control_layout.addRow("Frequency (Hz):", self.stream_freq_spin)

        # Buttons
        button_layout = QHBoxLayout()

        self.set_ip_btn = QPushButton("Set Client IP")
        self.set_ip_btn.clicked.connect(self.set_client_ip)

        self.start_streaming_btn = QPushButton("Start Streaming")
        self.start_streaming_btn.clicked.connect(self.start_streaming)

        self.stop_streaming_btn = QPushButton("Stop Streaming")
        self.stop_streaming_btn.clicked.connect(self.stop_streaming)

        button_layout.addWidget(self.set_ip_btn)
        button_layout.addWidget(self.start_streaming_btn)
        button_layout.addWidget(self.stop_streaming_btn)

        # Status display
        self.streaming_status_label = QLabel("Not streaming")

        # Add all to control layout
        control_group_widget = QWidget()
        control_group_layout = QVBoxLayout()
        control_group_layout.addLayout(control_layout)
        control_group_layout.addLayout(button_layout)
        control_group_layout.addWidget(self.streaming_status_label)
        control_group_widget.setLayout(control_group_layout)
        control_group.setLayout(control_group_layout)

        # Latest position data
        position_group = QGroupBox("Latest Position Data")
        position_layout = QVBoxLayout()

        self.position_data = QTextEdit()
        self.position_data.setReadOnly(True)

        self.refresh_position_btn = QPushButton("Refresh Position Data")
        self.refresh_position_btn.clicked.connect(self.refresh_position_data)

        position_layout.addWidget(self.position_data)
        position_layout.addWidget(self.refresh_position_btn)
        position_group.setLayout(position_layout)

        # Add all to main layout
        layout.addWidget(control_group)
        layout.addWidget(position_group)

        self.streaming_tab.setLayout(layout)

    def setup_tool_calibration_tab(self):
        layout = QVBoxLayout()

        # Tool calibration controls
        control_group = QGroupBox("Tool Calibration Controls")
        control_layout = QVBoxLayout()

        # Start/stop calibration buttons
        button_layout = QHBoxLayout()

        self.start_calibration_btn = QPushButton("Start Tool Calibration")
        self.start_calibration_btn.clicked.connect(self.start_tool_calibration)

        self.end_calibration_btn = QPushButton("End Tool Calibration")
        self.end_calibration_btn.clicked.connect(self.end_tool_calibration)

        self.force_stop_streaming_checkbox = QCheckBox("Force Stop Streaming")

        button_layout.addWidget(self.start_calibration_btn)
        button_layout.addWidget(self.end_calibration_btn)
        button_layout.addWidget(self.force_stop_streaming_checkbox)

        # Device selection
        device_layout = QHBoxLayout()
        self.device_spin = QSpinBox()
        self.device_spin.setRange(0, 10)

        device_layout.addWidget(QLabel("Device Index:"))
        device_layout.addWidget(self.device_spin)

        # Calibration button and visualization option
        calibrate_layout = QHBoxLayout()
        self.calibrate_tool_btn = QPushButton("Calibrate Tool")
        self.calibrate_tool_btn.clicked.connect(self.calibrate_tool)

        self.tool_visualize_checkbox = QCheckBox("Visualize")

        calibrate_layout.addWidget(self.calibrate_tool_btn)
        calibrate_layout.addWidget(self.tool_visualize_checkbox)

        # Status display
        self.tool_status_label = QLabel("Not calibrating")
        self.tool_points_label = QLabel("0 points")

        # Add all to control layout
        control_layout.addLayout(button_layout)
        control_layout.addLayout(device_layout)
        control_layout.addLayout(calibrate_layout)
        control_layout.addWidget(self.tool_status_label)
        control_layout.addWidget(self.tool_points_label)

        control_group.setLayout(control_layout)

        # Results group
        results_group = QGroupBox("Calibration Results")
        results_layout = QVBoxLayout()

        self.calibration_results = QTextEdit()
        self.calibration_results.setReadOnly(True)

        results_layout.addWidget(self.calibration_results)
        results_group.setLayout(results_layout)

        # Load from file
        file_group = QGroupBox("Load from File")
        file_layout = QHBoxLayout()

        self.file_input = QTextEdit()
        self.file_input.setMaximumHeight(30)
        self.file_input.setText("tool_tip.txt")

        self.load_file_btn = QPushButton("Load from File")
        self.load_file_btn.clicked.connect(self.load_tool_transformations)

        file_layout.addWidget(QLabel("Filename:"))
        file_layout.addWidget(self.file_input)
        file_layout.addWidget(self.load_file_btn)

        file_group.setLayout(file_layout)

        # Add all to main layout
        layout.addWidget(control_group)
        layout.addWidget(results_group)
        layout.addWidget(file_group)

        self.tool_calibration_tab.setLayout(layout)

    def toggle_ct_point_cloud(self):
        """Toggle the visibility of the CT point cloud"""
        if self.ct_point_cloud is None:
            self.log_message("CT point cloud data is not available")
            return

        self.show_ct_point_cloud = not self.show_ct_point_cloud

        # Update button text and style based on current state
        if self.show_ct_point_cloud:
            button_text = "Hide CT Point Cloud"
            button_style = "QPushButton { background-color: #f44336; color: white; font-weight: bold; }"
        else:
            button_text = "Show CT Point Cloud"
            button_style = "QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }"

        self.toggle_ct_pc_btn.setText(button_text)
        self.toggle_ct_pc_btn.setStyleSheet(button_style)
        self.fine_toggle_ct_pc_btn.setText(button_text)
        self.fine_toggle_ct_pc_btn.setStyleSheet(button_style)

        # Force immediate update of visualizations
        current_tab = self.tabs.currentWidget()
        if current_tab == self.coarse_tab:
            self.update_coarse_visualization()
        elif current_tab == self.fine_tab:
            # Force immediate update of fine visualization
            self.fine_plot_needs_full_update = True
            self.update_fine_visualization_immediate()

        self.log_message(f"CT point cloud {'shown' if self.show_ct_point_cloud else 'hidden'}")

    def update_fine_visualization_immediate(self):
        """Immediately update fine visualization (called when CT toggle is pressed)"""
        try:
            print(f"Updating fine visualization immediately. Show CT: {self.show_ct_point_cloud}")

            # Clear existing plot
            self.fine_canvas.axes.clear()
            self.fine_canvas.axes.set_xlabel('X')
            self.fine_canvas.axes.set_ylabel('Y')
            self.fine_canvas.axes.set_zlabel('Z')

            # Get current fine points count
            current_count = 0
            if self.ndi_server and hasattr(self.ndi_server, 'fine_registration'):
                fine_points = self.ndi_server.fine_registration.fine_points
                current_count = len(fine_points)

                # Update points to plot if we have fine points
                if current_count > 0:
                    if current_count > 2000:
                        indices = np.linspace(0, current_count - 1, 2000, dtype=int)
                        self.fine_points_to_plot = [fine_points[i] for i in indices]
                    else:
                        self.fine_points_to_plot = fine_points.copy()

            self.fine_canvas.axes.set_title(f'Fine Registration Points ({current_count} total)')

            # Plot CT point cloud FIRST if enabled (so it appears behind fine points)
            if self.show_ct_point_cloud and self.ct_point_cloud is not None:
                # Try to get transformed points
                transformed_points = self.transform_ct_point_cloud()
                points_to_plot = transformed_points if transformed_points is not None else self.ct_point_cloud

                print(
                    f"Plotting {'transformed' if transformed_points is not None else 'original'} CT point cloud with {len(points_to_plot)} points")
                self.fine_canvas.axes.scatter(
                    points_to_plot[:, 0],
                    points_to_plot[:, 1],
                    points_to_plot[:, 2],
                    c='red', marker='.', s=1, alpha=0.6, label='CT Point Cloud'
                )

            # Plot fine points if we have them
            if self.fine_points_to_plot:
                points_array = np.array(self.fine_points_to_plot)
                print(f"Plotting {len(points_array)} fine points")
                self.fine_canvas.axes.scatter(
                    points_array[:, 0],
                    points_array[:, 1],
                    points_array[:, 2],
                    c='blue', marker='.', s=3, alpha=0.8, label='Fine Points'
                )

            # Add legend if we have any points
            if self.show_ct_point_cloud or len(self.fine_points_to_plot) > 0:
                self.fine_canvas.axes.legend()

            # Auto-adjust axis limits to fit all points
            all_points = []

            if self.fine_points_to_plot:
                all_points.append(np.array(self.fine_points_to_plot))

            if self.show_ct_point_cloud and self.ct_point_cloud is not None:
                all_points.append(self.ct_point_cloud)

            if all_points:
                all_points = np.vstack(all_points)
                max_range = np.max(np.max(all_points, axis=0) - np.min(all_points, axis=0))
                mid_x = np.mean([np.min(all_points[:, 0]), np.max(all_points[:, 0])])
                mid_y = np.mean([np.min(all_points[:, 1]), np.max(all_points[:, 1])])
                mid_z = np.mean([np.min(all_points[:, 2]), np.max(all_points[:, 2])])

                self.fine_canvas.axes.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
                self.fine_canvas.axes.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
                self.fine_canvas.axes.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)

            # Refresh canvas
            self.fine_canvas.draw()
            print("Fine visualization update completed")

        except Exception as e:
            print(f"Error updating fine visualization immediately: {e}")
            self.log_message(f"Error updating fine visualization: {e}")

    def log_message(self, message):
        """Add message to log output with timestamp"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        if hasattr(self, 'log_output'):
            self.log_output.append(f"[{timestamp}] {message}")
        else:
            print(f"[{timestamp}] {message}")  # Fallback if log_output not yet initialized

    def update_server_status(self, status_data):
        """Update UI with server status information"""
        try:
            self.status_label.setText("Running")
            self.ndi_status_label.setText(
                "Initialized" if status_data.get("ndi_tracker_status") == "initialized" else "Not Initialized")
            self.data_source_label.setText(status_data.get("data_source", "Unknown"))
            self.coarse_points_label.setText(str(status_data.get("coarse_points_loaded", 0)))
            self.fine_points_label.setText(str(status_data.get("fine_points_loaded", 0)))

            # Update streaming status if we're on that tab
            if self.tabs.currentWidget() == self.streaming_tab:
                streaming_active = status_data.get("streaming_active", False)
                self.streaming_status_label.setText(f"{'Streaming' if streaming_active else 'Not streaming'}")

            # If coarse points count changed, log it (actual refresh is handled by the coarse_points_thread)
            new_coarse_count = status_data.get("coarse_points_loaded", 0)
            if new_coarse_count != self.last_coarse_points_count:
                self.last_coarse_points_count = new_coarse_count
                self.log_message(f"Coarse points count changed to {new_coarse_count}")

        except Exception as e:
            self.log_message(f"Error updating status: {e}")

    def initialize_ndi(self):
        """Initialize the NDI tracker"""
        try:
            force_restart = self.force_restart_checkbox.isChecked()

            self.log_message(f"Initializing NDI tracker (force_restart={force_restart})...")

            response = requests.post(f"{SERVER_URL}/initialize_ndi", params={"force_restart": force_restart})
            data = response.json()

            if response.status_code == 200:
                status = data.get("status")
                message = data.get("message")

                if status == "success":
                    self.log_message(f"Success: {message}")
                    tools_detected = data.get("details", {}).get("tools_detected", 0)
                    self.log_message(f"Detected {tools_detected} tool(s)")

                elif status == "already_initialized":
                    self.log_message(f"Note: {message}")

                else:
                    self.log_message(f"Error: {message}")

            else:
                self.log_message(f"Error: {response.status_code} - {data}")

        except Exception as e:
            self.log_message(f"Error initializing NDI tracker: {e}")

    def check_tools(self):
        """Check which tools are currently visible"""
        try:
            self.log_message("Checking tool visibility...")

            response = requests.post(f"{SERVER_URL}/check_tools")

            if response.status_code == 200:
                tool_status = response.json()

                status_text = "Tool Status:\n"
                for tool, visible in tool_status.items():
                    status_text += f"- {tool}: {'Visible ✓' if visible else 'Not visible ✗'}\n"

                self.tool_status_label.setText(status_text)
                self.log_message("Tool check completed")

            else:
                self.log_message(f"Error checking tools: {response.status_code}")

        except Exception as e:
            self.log_message(f"Error checking tools: {e}")

    def refresh_coarse_points(self):
        """Refresh the coarse points table and visualization"""
        try:
            # Only update if we're on the coarse tab to save resources
            if self.tabs.currentWidget() != self.coarse_tab:
                return

            # Make request to get coarse points status
            response = requests.get(f"{SERVER_URL}/")
            if response.status_code != 200:
                return

            data = response.json()
            coarse_points_count = data.get("coarse_points_loaded", 0)

            # Update UI labels
            if coarse_points_count > 0:
                self.coarse_count_label.setText(f"{coarse_points_count} points available")
                self.coarse_info_label.setText("Points have been set by another client")
            else:
                self.coarse_count_label.setText("0 points available")
                self.coarse_info_label.setText("Waiting for points from another client...")
                self.points_table.setRowCount(0)
                self.coarse_unity_points = []
                self.coarse_ndi_points = []
                self.update_coarse_visualization()
                return

            # Try to get all coarse points at once first (more efficient)
            try:
                points_response = requests.get(f"{SERVER_URL}/get_coarse_points")
                if points_response.status_code == 200:
                    points_data = points_response.json()
                    unity_points = points_data.get("unity_points", [])
                    ndi_points = points_data.get("ndi_points", [])

                    if unity_points and ndi_points and len(unity_points) == len(ndi_points):
                        # Clear table and repopulate
                        self.points_table.setRowCount(0)
                        for i, (unity_point, ndi_point) in enumerate(zip(unity_points, ndi_points)):
                            # Add to table
                            row = self.points_table.rowCount()
                            self.points_table.insertRow(row)
                            self.points_table.setItem(row, 0, QTableWidgetItem(str(i)))

                            for j, val in enumerate(unity_point):
                                self.points_table.setItem(row, j + 1, QTableWidgetItem(f"{val:.3f}"))

                            for j, val in enumerate(ndi_point):
                                self.points_table.setItem(row, j + 4, QTableWidgetItem(f"{val:.3f}"))

                        # Store points for visualization
                        self.coarse_unity_points = np.array(unity_points)
                        self.coarse_ndi_points = np.array(ndi_points)

                        # Update visualization
                        self.update_coarse_visualization()
                        return
            except Exception as e:
                self.log_message(f"Error getting all points at once: {e}")

            # If batch endpoint failed, try getting points one by one
            try:
                unity_points = []
                ndi_points = []

                # Clear table
                self.points_table.setRowCount(0)

                for i in range(coarse_points_count):
                    # Try to get individual point pairs
                    point_response = requests.get(f"{SERVER_URL}/get_coarse_point/{i}")
                    if point_response.status_code == 200:
                        point_data = point_response.json()
                        unity_point = point_data.get("unity_point")
                        ndi_point = point_data.get("ndi_point")

                        if unity_point and ndi_point:
                            unity_points.append(unity_point)
                            ndi_points.append(ndi_point)

                            # Add to table
                            row = self.points_table.rowCount()
                            self.points_table.insertRow(row)
                            self.points_table.setItem(row, 0, QTableWidgetItem(str(i)))

                            for j, val in enumerate(unity_point):
                                self.points_table.setItem(row, j + 1, QTableWidgetItem(f"{val:.3f}"))

                            for j, val in enumerate(ndi_point):
                                self.points_table.setItem(row, j + 4, QTableWidgetItem(f"{val:.3f}"))

                # Store points for visualization
                if unity_points and ndi_points:
                    self.coarse_unity_points = np.array(unity_points)
                    self.coarse_ndi_points = np.array(ndi_points)

                # Update visualization
                self.update_coarse_visualization()
            except Exception as e:
                self.log_message(f"Error getting points individually: {e}")

            # If we still couldn't get points data, show a message
            if not len(self.coarse_unity_points) or not len(self.coarse_ndi_points):
                self.points_table.setRowCount(1)
                self.points_table.setItem(0, 0, QTableWidgetItem(""))
                self.points_table.setItem(0, 1, QTableWidgetItem("Points exist but details are not accessible"))

        except Exception as e:
            self.log_message(f"Error refreshing coarse points: {e}")

    def refresh_coarse_points_manual(self):
        """Manually triggered refresh of coarse points"""
        self.log_message("Manually refreshing coarse points...")
        self.refresh_coarse_points()

    def update_coarse_visualization(self):
        """Update the 3D visualization of coarse points"""
        try:
            # Clear existing plot
            self.coarse_canvas.axes.clear()
            self.coarse_canvas.axes.set_xlabel('X')
            self.coarse_canvas.axes.set_ylabel('Y')
            self.coarse_canvas.axes.set_zlabel('Z')
            self.coarse_canvas.axes.set_title('Coarse Registration Points')

            # Plot CT point cloud if enabled
            if self.show_ct_point_cloud and self.ct_point_cloud is not None:
                # Try to get transformed points
                transformed_points = self.transform_ct_point_cloud()
                points_to_plot = transformed_points if transformed_points is not None else self.ct_point_cloud

                print(
                    f"Plotting {'transformed' if transformed_points is not None else 'original'} CT point cloud with {len(points_to_plot)} points")
                self.coarse_canvas.axes.scatter(
                    points_to_plot[:, 0],
                    points_to_plot[:, 1],
                    points_to_plot[:, 2],
                    c='r', marker='.', s=1, alpha=0.5, label='CT Point Cloud'
                )

            # Plot unity points if we have them
            if len(self.coarse_unity_points) > 0:
                self.coarse_canvas.axes.scatter(
                    self.coarse_unity_points[:, 0],
                    self.coarse_unity_points[:, 1],
                    self.coarse_unity_points[:, 2],
                    c='g', marker='o', label='Unity', s=50  # Increase size for better visibility
                )

            # Plot NDI points if we have them
            if len(self.coarse_ndi_points) > 0:
                self.coarse_canvas.axes.scatter(
                    self.coarse_ndi_points[:, 0],
                    self.coarse_ndi_points[:, 1],
                    self.coarse_ndi_points[:, 2],
                    c='b', marker='^', label='NDI', s=50  # Increase size for better visibility
                )

            # Draw lines connecting corresponding points if we have both sets
            if len(self.coarse_unity_points) > 0 and len(self.coarse_ndi_points) > 0 and len(
                    self.coarse_unity_points) == len(self.coarse_ndi_points):
                for i in range(len(self.coarse_unity_points)):
                    self.coarse_canvas.axes.plot(
                        [self.coarse_unity_points[i, 0], self.coarse_ndi_points[i, 0]],
                        [self.coarse_unity_points[i, 1], self.coarse_ndi_points[i, 1]],
                        [self.coarse_unity_points[i, 2], self.coarse_ndi_points[i, 2]],
                        'k--', alpha=0.3  # Black dashed line with transparency
                    )

            # Add legend if we have points
            if self.show_ct_point_cloud or len(self.coarse_unity_points) > 0 or len(self.coarse_ndi_points) > 0:
                self.coarse_canvas.axes.legend()

            # Auto-adjust axis limits to fit all points
            all_points = []

            if len(self.coarse_unity_points) > 0:
                all_points.append(self.coarse_unity_points)

            if len(self.coarse_ndi_points) > 0:
                all_points.append(self.coarse_ndi_points)

            if self.show_ct_point_cloud and self.ct_point_cloud is not None:
                all_points.append(self.ct_point_cloud)

            if all_points:
                all_points = np.vstack(all_points)
                max_range = np.max(np.max(all_points, axis=0) - np.min(all_points, axis=0))
                mid_x = np.mean([np.min(all_points[:, 0]), np.max(all_points[:, 0])])
                mid_y = np.mean([np.min(all_points[:, 1]), np.max(all_points[:, 1])])
                mid_z = np.mean([np.min(all_points[:, 2]), np.max(all_points[:, 2])])

                self.coarse_canvas.axes.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
                self.coarse_canvas.axes.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
                self.coarse_canvas.axes.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)

            # Refresh canvas
            self.coarse_canvas.draw()

        except Exception as e:
            self.log_message(f"Error updating coarse visualization: {e}")

    def on_tab_changed(self, index):
        """Handle tab change events"""
        # When switching to fine tab, force a full update of the plot
        if self.tabs.widget(index) == self.fine_tab:
            self.fine_plot_needs_full_update = True
            # Force immediate update if CT point cloud should be shown
            if self.show_ct_point_cloud:
                self.update_fine_visualization_immediate()

    def update_fine_points(self):
        """Update the fine points visualization using direct access to ndi_server"""
        try:
            # Only update if we're on the fine tab to save resources
            if self.tabs.currentWidget() != self.fine_tab or not self.ndi_server:
                return

            # Get fine points directly from the server instance
            fine_points = self.ndi_server.fine_registration.fine_points
            current_count = len(fine_points)

            # Update the points count label
            self.fine_points_count_label.setText(f"{current_count} points collected")

            # Check if we need to update the visualization
            if current_count == 0 and not self.show_ct_point_cloud:
                if self.last_fine_points_count > 0 or self.fine_plot_needs_full_update:
                    # Clear plot
                    self.fine_canvas.axes.clear()
                    self.fine_canvas.axes.set_xlabel('X')
                    self.fine_canvas.axes.set_ylabel('Y')
                    self.fine_canvas.axes.set_zlabel('Z')
                    self.fine_canvas.axes.set_title('Fine Registration Points')
                    self.fine_canvas.draw()
                    self.last_fine_points_count = 0
                    self.fine_points_to_plot = []
                    self.fine_plot_needs_full_update = False
                return

            # Only do expensive resampling and plotting if:
            # 1. The number of points has changed significantly (added 50+ new points)
            # 2. We haven't plotted anything yet
            # 3. We've explicitly requested a full update (e.g. after tab switch)
            # 4. CT point cloud visibility has changed
            if (abs(current_count - self.last_fine_points_count) > 50 or
                    self.last_fine_points_count == 0 or
                    self.fine_plot_needs_full_update):

                # If there are too many points, sample them for performance
                if current_count > 2000:
                    indices = np.linspace(0, current_count - 1, 2000, dtype=int)
                    self.fine_points_to_plot = [fine_points[i] for i in indices]
                else:
                    self.fine_points_to_plot = fine_points.copy() if fine_points else []

                # Clear existing plot
                self.fine_canvas.axes.clear()
                self.fine_canvas.axes.set_xlabel('X')
                self.fine_canvas.axes.set_ylabel('Y')
                self.fine_canvas.axes.set_zlabel('Z')
                self.fine_canvas.axes.set_title(f'Fine Registration Points ({current_count} total)')

                # Plot CT point cloud FIRST if enabled (so it appears behind fine points)
                if self.show_ct_point_cloud and self.ct_point_cloud is not None:
                    # Try to get transformed points
                    transformed_points = self.transform_ct_point_cloud()
                    points_to_plot = transformed_points if transformed_points is not None else self.ct_point_cloud

                    print(
                        f"Plotting {'transformed' if transformed_points is not None else 'original'} CT point cloud in fine tab with {len(points_to_plot)} points")
                    self.fine_canvas.axes.scatter(
                        points_to_plot[:, 0],
                        points_to_plot[:, 1],
                        points_to_plot[:, 2],
                        c='red', marker='.', s=1, alpha=0.6, label='CT Point Cloud'
                    )

                # Convert to numpy array for plotting
                if self.fine_points_to_plot:
                    points_array = np.array(self.fine_points_to_plot)

                    # Plot points
                    self.fine_canvas.axes.scatter(
                        points_array[:, 0],
                        points_array[:, 1],
                        points_array[:, 2],
                        c='blue', marker='.', s=3, alpha=0.8, label='Fine Points'
                    )

                # Add legend
                if self.show_ct_point_cloud or len(self.fine_points_to_plot) > 0:
                    self.fine_canvas.axes.legend()

                # Auto-adjust axis limits to fit all points
                all_points = []

                if self.fine_points_to_plot:
                    all_points.append(np.array(self.fine_points_to_plot))

                if self.show_ct_point_cloud and self.ct_point_cloud is not None:
                    all_points.append(self.ct_point_cloud)

                if all_points:
                    all_points = np.vstack(all_points)
                    max_range = np.max(np.max(all_points, axis=0) - np.min(all_points, axis=0))
                    mid_x = np.mean([np.min(all_points[:, 0]), np.max(all_points[:, 0])])
                    mid_y = np.mean([np.min(all_points[:, 1]), np.max(all_points[:, 1])])
                    mid_z = np.mean([np.min(all_points[:, 2]), np.max(all_points[:, 2])])

                    self.fine_canvas.axes.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
                    self.fine_canvas.axes.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
                    self.fine_canvas.axes.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)

                # Refresh canvas
                self.fine_canvas.draw()

                # Update the last count
                self.last_fine_points_count = current_count
                self.fine_plot_needs_full_update = False

        except Exception as e:
            print(f"Error updating fine points visualization: {e}")

    # [Rest of the methods remain the same - reset_coarse_points, perform_coarse_registration, etc.]
    # I'll include a few key ones to show they're unchanged:

    def reset_coarse_points(self):
        """Reset all coarse registration points"""
        try:
            if QMessageBox.question(self, "Confirm Reset",
                                    "Are you sure you want to reset all coarse points?",
                                    QMessageBox.Yes | QMessageBox.No) == QMessageBox.No:
                return

            self.log_message("Resetting all coarse points...")

            response = requests.post(f"{SERVER_URL}/reset_coarse_points")

            if response.status_code == 200:
                # Clear table
                self.points_table.setRowCount(0)

                # Clear points data
                self.coarse_unity_points = []
                self.coarse_ndi_points = []

                # Clear plot
                self.update_coarse_visualization()

                self.coarse_count_label.setText("0 points available")
                self.coarse_info_label.setText("Waiting for points from another client...")

                self.log_message("All coarse points reset successfully")

            else:
                error_message = response.json().get("detail", "Unknown error")
                self.log_message(f"Error resetting coarse points: {error_message}")
                QMessageBox.warning(self, "Error", f"Failed to reset coarse points: {error_message}")

        except Exception as e:
            self.log_message(f"Error resetting coarse points: {e}")
            QMessageBox.critical(self, "Error", f"An error occurred: {e}")

    def perform_coarse_registration(self):
        """Perform coarse registration"""
        try:
            visualize = self.visualize_checkbox.isChecked()

            self.log_message(f"Performing coarse registration (visualize={visualize})...")

            response = requests.post(f"{SERVER_URL}/coarse_register", params={"visualize": visualize})

            if response.status_code == 200:
                result = response.json()

                # Display results
                self.coarse_results.clear()
                self.coarse_results.append(json.dumps(result, indent=2))

                status = result.get("status")
                if status == "success":
                    self.log_message("Coarse registration completed successfully")
                    QMessageBox.information(self, "Registration Complete",
                                            f"Coarse registration completed successfully.\nRMS Error: {result.get('rms_error', 'N/A')}")
                else:
                    self.log_message(f"Coarse registration completed with status: {status}")

            else:
                error_message = response.json().get("detail", "Unknown error")
                self.log_message(f"Error performing coarse registration: {error_message}")
                QMessageBox.warning(self, "Error", f"Failed to perform coarse registration: {error_message}")

        except Exception as e:
            self.log_message(f"Error performing coarse registration: {e}")
            QMessageBox.critical(self, "Error", f"An error occurred: {e}")

    def start_fine_gather(self):
        """Start gathering fine registration points"""
        try:
            frequency = self.frequency_spin.value()

            self.log_message(f"Starting fine point gathering at {frequency} Hz...")

            response = requests.post(f"{SERVER_URL}/start_fine_gather", params={"frequency": frequency})

            if response.status_code == 200:
                result = response.json()

                status = result.get("status")
                message = result.get("message")

                if status == "started":
                    self.log_message(f"Fine point gathering started: {message}")
                    self.fine_status_label.setText("Gathering")

                else:
                    self.log_message(f"Note: {message}")

            else:
                error_message = response.json().get("detail", "Unknown error")
                self.log_message(f"Error starting fine point gathering: {error_message}")
                QMessageBox.warning(self, "Error", f"Failed to start fine point gathering: {error_message}")

        except Exception as e:
            self.log_message(f"Error starting fine point gathering: {e}")
            QMessageBox.critical(self, "Error", f"An error occurred: {e}")

    def end_fine_gather(self):
        """Stop gathering fine registration points"""
        try:
            self.log_message("Stopping fine point gathering...")

            response = requests.post(f"{SERVER_URL}/end_fine_gather")

            if response.status_code == 200:
                result = response.json()

                status = result.get("status")
                message = result.get("message")

                if status == "success":
                    points_collected = result.get("total_points", 0)
                    self.log_message(f"Fine point gathering stopped: {points_collected} points collected")
                    self.fine_status_label.setText("Not gathering")

                else:
                    self.log_message(f"Note: {message}")

            else:
                error_message = response.json().get("detail", "Unknown error")
                self.log_message(f"Error stopping fine point gathering: {error_message}")
                QMessageBox.warning(self, "Error", f"Failed to stop fine point gathering: {error_message}")

        except Exception as e:
            self.log_message(f"Error stopping fine point gathering: {e}")
            QMessageBox.critical(self, "Error", f"An error occurred: {e}")

    def reset_fine_gather(self):
        """Reset fine registration points"""
        try:
            if QMessageBox.question(self, "Confirm Reset",
                                    "Are you sure you want to reset all fine points?",
                                    QMessageBox.Yes | QMessageBox.No) == QMessageBox.No:
                return

            self.log_message("Resetting all fine points...")

            response = requests.post(f"{SERVER_URL}/reset_fine_gather")

            if response.status_code == 200:
                # Reset tracking variables
                self.last_fine_points_count = 0
                self.fine_points_to_plot = []
                self.fine_plot_needs_full_update = True

                # Force update of visualization
                self.update_fine_visualization_immediate()

                # Reset point count
                self.fine_points_count_label.setText("0 points")

                self.log_message("All fine points reset successfully")

            else:
                error_message = response.json().get("detail", "Unknown error")
                self.log_message(f"Error resetting fine points: {error_message}")
                QMessageBox.warning(self, "Error", f"Failed to reset fine points: {error_message}")

        except Exception as e:
            self.log_message(f"Error resetting fine points: {e}")
            QMessageBox.critical(self, "Error", f"An error occurred: {e}")

    def perform_fine_registration(self):
        """Perform fine registration using ICP"""
        try:
            id_value = self.id_spin.value()
            downsample = self.downsample_spin.value()
            visualize = self.fine_visualize_checkbox.isChecked()

            self.log_message(
                f"Performing fine registration (id={id_value}, downsample={downsample}, visualize={visualize})...")

            response = requests.post(
                f"{SERVER_URL}/fine_register",
                params={"id": id_value, "downsample_factor": downsample, "visualize": visualize}
            )

            if response.status_code == 200:
                result = response.json()

                # Display results
                self.fine_results.clear()
                self.fine_results.append(json.dumps(result, indent=2))

                status = result.get("status")
                if status == "success":
                    self.log_message("Fine registration completed successfully")
                    QMessageBox.information(self, "Registration Complete",
                                            f"Fine registration completed successfully.\nRMSE: {result.get('rmse', 'N/A')}")
                else:
                    self.log_message(f"Fine registration completed with status: {status}")

            else:
                error_message = response.json().get("detail", "Unknown error")
                self.log_message(f"Error performing fine registration: {error_message}")
                QMessageBox.warning(self, "Error", f"Failed to perform fine registration: {error_message}")

        except Exception as e:
            self.log_message(f"Error performing fine registration: {e}")
            QMessageBox.critical(self, "Error", f"An error occurred: {e}")

    def set_client_ip(self):
        """Set the client IP for streaming"""
        try:
            ip = self.client_ip_input.toPlainText().strip()

            self.log_message(f"Setting client IP to {ip}...")

            response = requests.post(f"{SERVER_URL}/set_client_ip", params={"ip": ip})

            if response.status_code == 200:
                result = response.json()

                self.log_message(f"Client IP set: {result.get('message')}")

            else:
                error_message = response.json().get("detail", "Unknown error")
                self.log_message(f"Error setting client IP: {error_message}")
                QMessageBox.warning(self, "Error", f"Failed to set client IP: {error_message}")

        except Exception as e:
            self.log_message(f"Error setting client IP: {e}")
            QMessageBox.critical(self, "Error", f"An error occurred: {e}")

    def start_streaming(self):
        """Start streaming NDI tracking data"""
        try:
            port = self.port_spin.value()
            frequency = self.stream_freq_spin.value()

            self.log_message(f"Starting streaming on port {port} at {frequency} Hz...")

            response = requests.post(
                f"{SERVER_URL}/start_streaming",
                params={"port": port, "frequency": frequency, "force_stop_calibration": False}
            )

            if response.status_code == 200:
                result = response.json()

                status = result.get("status")
                message = result.get("message")

                if status == "started":
                    self.log_message(f"Streaming started: {message}")
                    self.streaming_status_label.setText("Streaming")

                elif status == "already_running":
                    self.log_message(f"Note: {message}")

                else:
                    self.log_message(f"Unexpected status: {status} - {message}")

            else:
                error_message = response.json().get("detail", "Unknown error")
                self.log_message(f"Error starting streaming: {error_message}")
                QMessageBox.warning(self, "Error", f"Failed to start streaming: {error_message}")

        except Exception as e:
            self.log_message(f"Error starting streaming: {e}")
            QMessageBox.critical(self, "Error", f"An error occurred: {e}")

    def stop_streaming(self):
        """Stop streaming NDI tracking data"""
        try:
            self.log_message("Stopping streaming...")

            response = requests.post(f"{SERVER_URL}/stop_streaming")

            if response.status_code == 200:
                result = response.json()

                status = result.get("status")
                message = result.get("message")

                if status == "stopped":
                    self.log_message(f"Streaming stopped: {message}")
                    self.streaming_status_label.setText("Not streaming")

                elif status == "not_active":
                    self.log_message(f"Note: {message}")

                else:
                    self.log_message(f"Unexpected status: {status} - {message}")

            else:
                error_message = response.json().get("detail", "Unknown error")
                self.log_message(f"Error stopping streaming: {error_message}")
                QMessageBox.warning(self, "Error", f"Failed to stop streaming: {error_message}")

        except Exception as e:
            self.log_message(f"Error stopping streaming: {e}")
            QMessageBox.critical(self, "Error", f"An error occurred: {e}")

    def refresh_position_data(self):
        """Get the latest position data from streaming"""
        try:
            response = requests.get(f"{SERVER_URL}/get_latest_position")

            if response.status_code == 200:
                result = response.json()

                # Display results
                self.position_data.clear()
                self.position_data.append(json.dumps(result, indent=2))

            else:
                error_message = response.json().get("detail", "Unknown error")
                self.log_message(f"Error getting position data: {error_message}")

        except Exception as e:
            self.log_message(f"Error getting position data: {e}")

    def start_tool_calibration(self):
        """Start tool calibration"""
        try:
            force_stop = self.force_stop_streaming_checkbox.isChecked()
            device = self.device_spin.value()

            self.log_message(f"Starting tool calibration (device={device}, force_stop_streaming={force_stop})...")

            response = requests.post(
                f"{SERVER_URL}/start_tool_calibration",
                params={"force_stop_streaming": force_stop, "device": device}
            )

            if response.status_code == 200:
                result = response.json()

                status = result.get("status")
                message = result.get("message")

                if status == "started":
                    self.log_message(f"Tool calibration started: {message}")
                    self.tool_status_label.setText("Calibrating")

                else:
                    self.log_message(f"Note: {message}")

            else:
                error_message = response.json().get("detail", "Unknown error")
                self.log_message(f"Error starting tool calibration: {error_message}")
                QMessageBox.warning(self, "Error", f"Failed to start tool calibration: {error_message}")

        except Exception as e:
            self.log_message(f"Error starting tool calibration: {e}")
            QMessageBox.critical(self, "Error", f"An error occurred: {e}")

    def end_tool_calibration(self):
        """Stop tool calibration"""
        try:
            self.log_message("Stopping tool calibration...")

            response = requests.post(f"{SERVER_URL}/end_tool_calibration")

            if response.status_code == 200:
                result = response.json()

                status = result.get("status")
                message = result.get("message")

                if status == "success":
                    points_collected = result.get("transformations_collected", 0)
                    self.log_message(f"Tool calibration stopped: {points_collected} transformations collected")
                    self.tool_status_label.setText("Not calibrating")
                    self.tool_points_label.setText(f"{points_collected} transformations")

                else:
                    self.log_message(f"Note: {message}")

            else:
                error_message = response.json().get("detail", "Unknown error")
                self.log_message(f"Error stopping tool calibration: {error_message}")
                QMessageBox.warning(self, "Error", f"Failed to stop tool calibration: {error_message}")

        except Exception as e:
            self.log_message(f"Error stopping tool calibration: {e}")
            QMessageBox.critical(self, "Error", f"An error occurred: {e}")

    def calibrate_tool(self):
        """Process collected data to find tool tip vector"""
        try:
            visualize = self.tool_visualize_checkbox.isChecked()

            self.log_message(f"Calibrating tool (visualize={visualize})...")

            response = requests.post(f"{SERVER_URL}/calibrate_tool", params={"visualize": visualize})

            if response.status_code == 200:
                result = response.json()

                # Display results
                self.calibration_results.clear()
                self.calibration_results.append(json.dumps(result, indent=2))

                status = result.get("status")

                if status == "success":
                    tool_tip = result.get("tool_tip_vector")
                    self.log_message(f"Tool calibrated successfully: tip vector = {tool_tip}")

                else:
                    error_message = result.get("message", "Unknown error")
                    self.log_message(f"Calibration error: {error_message}")
                    QMessageBox.warning(self, "Calibration Error", error_message)

            else:
                error_message = response.json().get("detail", "Unknown error")
                self.log_message(f"Error calibrating tool: {error_message}")
                QMessageBox.warning(self, "Error", f"Failed to calibrate tool: {error_message}")

        except Exception as e:
            self.log_message(f"Error calibrating tool: {e}")
            QMessageBox.critical(self, "Error", f"An error occurred: {e}")

    def load_tool_transformations(self):
        """Load tool transformations from a file"""
        try:
            filename = self.file_input.toPlainText().strip()

            self.log_message(f"Loading tool transformations from {filename}...")

            response = requests.post(f"{SERVER_URL}/load_tool_transformations_from_file", params={"filename": filename})

            if response.status_code == 200:
                result = response.json()

                status = result.get("status")

                if status == "success":
                    count = result.get("transformations_loaded", 0)
                    self.log_message(f"Loaded {count} transformations from {filename}")
                    self.tool_points_label.setText(f"{count} transformations")

                else:
                    error_message = result.get("message", "Unknown error")
                    self.log_message(f"Loading error: {error_message}")
                    QMessageBox.warning(self, "Loading Error", error_message)

            else:
                error_message = response.json().get("detail", "Unknown error")
                self.log_message(f"Error loading transformations: {error_message}")
                QMessageBox.warning(self, "Error", f"Failed to load transformations: {error_message}")

        except Exception as e:
            self.log_message(f"Error loading transformations: {e}")
            QMessageBox.critical(self, "Error", f"An error occurred: {e}")


def launch_ui(ndi_server=None, config=None, args=None):
    app = QApplication(sys.argv)
    window = NDITrackingUI(ndi_server, config, args)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    # Test with sample data
    import numpy as np

    # Create sample CT point cloud data for testing
    sample_points = np.random.rand(5000, 3) * 100  # 5000 random 3D points
    np.save('sample_ct_pointcloud.npy', sample_points)

    config = {"CT_PC_address": "sample_ct_pointcloud.npy"}
    launch_ui(None, config)
import sys
import json
import numpy as np
import requests
import time
import os
import argparse
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
    def __init__(self, config_path, server_url="http://localhost:8000"):
        super().__init__()
        self.server_url = server_url
        self.config_path = config_path
        self.config = None
        self.setWindowTitle("NDI Tracking Visualization System")
        self.setGeometry(100, 100, 1600, 1000)

        # Load configuration
        self.load_configuration()

        # Initialize data storage
        self.coarse_points = {}  # Store coarse point pairs
        self.fine_points = []
        self.ct_pointcloud = None
        self.streaming_active = False
        self.current_position = None

        # Performance optimization flags - IMPROVED
        self.render_pending = False
        self.last_render_time = 0
        self.min_render_interval = 0.020  # 50 FPS max (was 30 FPS)
        self.interaction_render_interval = 0.050  # 20 FPS during interaction
        self.is_interacting = False

        # Initialize UI
        self.init_ui()

        # Load CT point cloud if available
        self.load_ct_from_config()

        # Start status update timer
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_status)
        self.status_timer.start(2000)  # Update every 2 seconds

        # Start coarse points monitoring timer
        self.coarse_monitor_timer = QTimer()
        self.coarse_monitor_timer.timeout.connect(self.check_coarse_points_updates)
        self.coarse_monitor_timer.start(1000)  # Check every 1 second

        # Streaming update timer
        self.streaming_timer = QTimer()
        self.streaming_timer.timeout.connect(self.update_streaming_visualization)

        # Render timer for throttling
        self.render_timer = QTimer()
        self.render_timer.timeout.connect(self.throttled_render)
        self.render_timer.setSingleShot(True)

    def throttled_render(self):
        """Improved throttled rendering"""
        current_time = time.time()

        # Use different intervals for interaction vs static
        interval = self.interaction_render_interval if self.is_interacting else self.min_render_interval

        if current_time - self.last_render_time >= interval:
            try:
                self.plotter.render()
                self.last_render_time = current_time
                self.render_pending = False
            except Exception as e:
                print(f"Render error: {str(e)}")
                self.render_pending = False
        else:
            # Schedule another render attempt
            remaining_time = int((interval - (current_time - self.last_render_time)) * 1000)
            self.render_timer.start(max(remaining_time, 5))  # Minimum 5ms delay

    def request_render(self):
        """Request a render with improved throttling"""
        if not self.render_pending:
            self.render_pending = True
            # Shorter delay for more responsive updates
            self.render_timer.start(5)

    def setup_interaction_detection(self):
        """Setup interaction detection for better performance during camera movement"""
        try:
            # Get the interactor style
            interactor = self.plotter.interactor
            if interactor:
                # Add observers for interaction events
                interactor.AddObserver('StartInteractionEvent', self.on_interaction_start)
                interactor.AddObserver('EndInteractionEvent', self.on_interaction_end)
                print("Interaction detection setup successful")
        except Exception as e:
            print(f"Failed to setup interaction detection: {str(e)}")

    def on_interaction_start(self, obj, event):
        """Called when user starts interacting with the scene"""
        self.is_interacting = True
        # Optionally reduce quality during interaction
        if self.ct_actor:
            try:
                # Reduce opacity during interaction for better performance
                self.ct_actor.GetProperty().SetOpacity(0.3)
            except:
                pass

    def on_interaction_end(self, obj, event):
        """Called when user stops interacting with the scene"""
        self.is_interacting = False
        # Restore quality after interaction
        if self.ct_actor:
            try:
                # Restore opacity
                self.ct_actor.GetProperty().SetOpacity(0.6)
            except:
                pass
        # Force a high-quality render
        self.request_render()

    def load_configuration(self):
        """Load configuration from JSON file"""
        try:
            with open(self.config_path, 'r') as file:
                self.config = json.load(file)
            print(f"Loaded configuration from: {self.config_path}")
            print(f"Tool types: {self.config.get('tool_types', {})}")
            print(f"CT point cloud path: {self.config.get('CT_pc', 'Not specified')}")
        except Exception as e:
            self.show_message("Configuration Error", f"Failed to load configuration: {str(e)}")
            # Use default configuration
            self.config = {
                "tool_types": {"probe": 0, "reference": 1, "endoscope": 2},
                "CT_pc": "",
                "probe_tip_vector": [0.0, 0.0, -161.148433, 1.0]
            }

    def load_ct_from_config(self):
        """Load CT point cloud from configuration file path"""
        if not self.config or not self.config.get('CT_pc'):
            print("No CT point cloud specified in config")
            return

        ct_path = self.config['CT_pc']

        # Handle relative paths
        if not os.path.isabs(ct_path):
            # Make path relative to config file directory
            config_dir = os.path.dirname(self.config_path)
            ct_path = os.path.join(config_dir, ct_path)

        if os.path.exists(ct_path):
            try:
                print(f"Loading CT point cloud from config: {ct_path}")
                # Load numpy array
                ct_points = np.load(ct_path)

                # Convert numpy array to PyVista point cloud
                if ct_points.shape[1] >= 3:
                    self.ct_pointcloud = pv.PolyData(ct_points[:, :3])
                    print(f"Loaded CT point cloud with {self.ct_pointcloud.n_points} points")

                    # Show bounds information
                    bounds = self.ct_pointcloud.bounds
                    print(
                        f"CT Point Cloud Bounds: X[{bounds[0]:.1f}, {bounds[1]:.1f}] Y[{bounds[2]:.1f}, {bounds[3]:.1f}] Z[{bounds[4]:.1f}, {bounds[5]:.1f}]")

                    # Auto-show CT cloud with longer delay for stability
                    QTimer.singleShot(2000, self.delayed_ct_visualization)  # Increased delay

                    # Just print success, no popup
                    print(f"Successfully loaded CT point cloud: {os.path.basename(ct_path)} ({len(ct_points)} points)")
                else:
                    print(f"Invalid CT point cloud format in {ct_path}")
                    self.show_message("Error", f"Invalid CT point cloud format in {ct_path}")
            except Exception as e:
                print(f"Error loading CT point cloud: {str(e)}")
                self.show_message("Error", f"Failed to load CT point cloud from {ct_path}: {str(e)}")
        else:
            print(f"CT point cloud file not found: {ct_path}")
            self.show_message("Warning", f"CT point cloud file not found: {ct_path}")

    def delayed_ct_visualization(self):
        """Delayed CT visualization to ensure UI is ready - with error handling"""
        try:
            if hasattr(self, 'show_ct_cloud'):
                self.show_ct_cloud.setChecked(True)
                # Add a small delay before updating visualization
                QTimer.singleShot(500, self.safe_update_ct_visualization)
                print("CT point cloud visualization will be enabled shortly")
        except Exception as e:
            print(f"Error in delayed CT visualization: {str(e)}")

    def safe_update_ct_visualization(self):
        """Safely update CT visualization with error handling"""
        try:
            self.update_ct_visualization()
        except Exception as e:
            print(f"Error in safe CT visualization update: {str(e)}")
            # Try to show a smaller subset
            try:
                self.fallback_minimal_ct_visualization()
            except:
                print("All CT visualization attempts failed")

    def fallback_minimal_ct_visualization(self):
        """Minimal fallback CT visualization with only 1000 points"""
        if self.ct_pointcloud is not None:
            try:
                print("Attempting minimal CT visualization with 1000 points")
                points = self.ct_pointcloud.points

                # Use systematic sampling for maximum stability
                step = max(1, self.ct_pointcloud.n_points // 1000)
                minimal_points = points[::step][:1000].copy()

                point_cloud = pv.PolyData(minimal_points)

                self.ct_actor = self.plotter.add_mesh(
                    point_cloud,
                    style='points',
                    color='lightblue',
                    point_size=4.0,
                    opacity=0.8,
                    name='ct_pointcloud_minimal',
                    render=False
                )

                if self.ct_actor:
                    print(f"Minimal CT visualization successful with {len(minimal_points)} points")
                    self.plotter.render()

            except Exception as e:
                print(f"Minimal CT visualization failed: {str(e)}")

    def check_coarse_points_updates(self):
        """Check for new coarse points from server - skip during interaction"""
        # Skip updates during interaction for better performance
        if self.is_interacting:
            return

        try:
            response = requests.get(f"{self.server_url}/get_coarse_points", timeout=2)
            if response.status_code == 200:
                server_points = response.json()

                # DEBUG: Print what we get from server
                print(f"DEBUG: Server returned {len(server_points) if server_points else 0} coarse points")
                if server_points:
                    print(f"DEBUG: First server point: {server_points[0]}")
                    print(f"DEBUG: Local points count: {len(self.coarse_points)}")

                # Check if we have new points
                if self.has_new_coarse_points(server_points):
                    print("DEBUG: New coarse points detected from server - updating visualization")
                    self.update_coarse_points_from_server(server_points)
                    self.update_coarse_points_display()
                    self.update_coarse_visualization()
                else:
                    print("DEBUG: No new coarse points detected")

        except Exception as e:
            print(f"DEBUG: Error checking coarse points: {str(e)}")

    def has_new_coarse_points(self, server_points):
        """Check if server has different coarse points than local storage"""
        if not server_points:
            has_new = len(self.coarse_points) > 0
            print(f"DEBUG: Server has no points, local has {len(self.coarse_points)}, has_new: {has_new}")
            return has_new

        # Check if number of points is different
        if len(server_points) != len(self.coarse_points):
            print(f"DEBUG: Point count differs - Server: {len(server_points)}, Local: {len(self.coarse_points)}")
            return True

        # If we have no local points but server has points
        if not self.coarse_points and server_points:
            print("DEBUG: No local points but server has points")
            return True

        # Check if any point data is different
        for i, point_data in enumerate(server_points):
            # Try different possible field names
            point_num = (point_data.get('point_number') or
                         point_data.get('point_id') or
                         point_data.get('id') or
                         point_data.get('index') or i)

            if point_num not in self.coarse_points:
                print(f"DEBUG: Point {point_num} not in local storage")
                return True

            # Compare unity points (use small tolerance for floating point comparison)
            server_unity = (point_data.get('unity_point') or
                            point_data.get('unity_coordinate') or
                            point_data.get('unity') or
                            point_data.get('target_point') or [0, 0, 0])

            local_unity = self.coarse_points[point_num]['unity_point']

            if len(local_unity) != len(server_unity):
                print(f"DEBUG: Unity point length differs for point {point_num}")
                return True

            for j in range(len(local_unity)):
                if abs(local_unity[j] - server_unity[j]) > 0.001:  # 1mm tolerance
                    print(
                        f"DEBUG: Unity point {point_num} coordinate {j} differs: {local_unity[j]} vs {server_unity[j]}")
                    return True

        print("DEBUG: No differences found")
        return False

    def update_coarse_points_from_server(self, server_points):
        """Update local coarse points storage from server data"""
        print(f"DEBUG: Updating from server data: {server_points}")

        # Clear existing points
        old_count = len(self.coarse_points)
        self.coarse_points.clear()

        # Add server points to local storage
        for i, point_data in enumerate(server_points):
            print(f"DEBUG: Processing server point {i}: {point_data}")

            # Try different possible field names that the server might use
            point_num = (point_data.get('point_number') or
                         point_data.get('point_id') or
                         point_data.get('id') or
                         point_data.get('index') or i)

            unity_point = (point_data.get('unity_point') or
                           point_data.get('unity_coordinate') or
                           point_data.get('unity') or
                           point_data.get('target_point') or [0, 0, 0])

            ndi_point = (point_data.get('ndi_point') or
                         point_data.get('ndi_coordinate') or
                         point_data.get('ndi') or
                         point_data.get('source_point') or [0, 0, 0])

            data_source = (point_data.get('data_source') or
                           point_data.get('source') or 'server')

            if point_num is not None:
                self.coarse_points[point_num] = {
                    'unity_point': unity_point,
                    'ndi_point': ndi_point,
                    'data_source': data_source
                }
                print(f"DEBUG: Added point {point_num}: Unity{unity_point}, NDI{ndi_point}")

        print(f"DEBUG: Updated local coarse points from server: {old_count} -> {len(self.coarse_points)} points")

        # Update the point number spinner to next available number
        if self.coarse_points:
            max_point_num = max(self.coarse_points.keys())
            self.point_number.setValue(max_point_num + 1)
            print(f"DEBUG: Set next point number to {max_point_num + 1}")
        else:
            self.point_number.setValue(0)
            print("DEBUG: Reset point number to 0")

    def refresh_coarse_points_from_server(self):
        """Manually refresh coarse points from server"""
        try:
            response = requests.get(f"{self.server_url}/get_coarse_points", timeout=5)
            if response.status_code == 200:
                server_points = response.json()
                self.update_coarse_points_from_server(server_points)
                self.update_coarse_points_display()
                self.update_coarse_visualization()
                self.show_message("Success", f"Refreshed {len(self.coarse_points)} coarse points from server")
            else:
                self.show_message("Error", f"Failed to get coarse points: {response.status_code}")
        except Exception as e:
            self.show_message("Error", f"Error refreshing coarse points: {str(e)}")

    def debug_server_coarse_points(self):
        """Debug method to check what the server returns for coarse points"""
        try:
            print("DEBUG: Calling /get_coarse_points endpoint...")
            response = requests.get(f"{self.server_url}/get_coarse_points", timeout=5)
            print(f"DEBUG: Response status: {response.status_code}")

            if response.status_code == 200:
                server_points = response.json()
                print(f"DEBUG: Raw server response: {server_points}")
                print(f"DEBUG: Response type: {type(server_points)}")

                if server_points:
                    print(f"DEBUG: Number of points: {len(server_points)}")
                    for i, point in enumerate(server_points):
                        print(f"DEBUG: Point {i}: {point}")
                        print(
                            f"DEBUG: Point {i} keys: {list(point.keys()) if isinstance(point, dict) else 'Not a dict'}")
                else:
                    print("DEBUG: Server returned empty or null response")

                # Show in message box too
                self.show_message("Debug Server Response",
                                  f"Status: {response.status_code}\n"
                                  f"Points count: {len(server_points) if server_points else 0}\n"
                                  f"Raw response: {str(server_points)[:500]}...")
            else:
                error_msg = f"Server error: {response.status_code}\nResponse: {response.text}"
                print(f"DEBUG: {error_msg}")
                self.show_message("Debug Server Error", error_msg)

        except Exception as e:
            error_msg = f"Error calling server: {str(e)}"
            print(f"DEBUG: {error_msg}")
            self.show_message("Debug Error", error_msg)

    def force_update_coarse_visualization(self):
        """Force update coarse visualization for testing"""
        print("DEBUG: Force updating coarse visualization...")
        print(f"DEBUG: Local coarse points: {self.coarse_points}")
        self.update_coarse_points_display()
        self.update_coarse_visualization()
        self.show_message("Debug", f"Forced update with {len(self.coarse_points)} points")

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

        # Setup interaction detection for better performance
        QTimer.singleShot(1000, self.setup_interaction_detection)  # Setup after UI is ready

    def create_control_panel(self):
        control_widget = QWidget()
        control_widget.setMaximumWidth(400)
        control_widget.setMinimumWidth(350)

        layout = QVBoxLayout()

        # Configuration info group
        config_group = self.create_config_group()
        layout.addWidget(config_group)

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

    def create_config_group(self):
        """Create configuration information group"""
        group = QGroupBox("Configuration")
        layout = QGridLayout()

        # Config file path
        layout.addWidget(QLabel("Config File:"), 0, 0)
        config_label = QLabel(os.path.basename(self.config_path))
        config_label.setToolTip(self.config_path)
        layout.addWidget(config_label, 0, 1)

        # Tool types
        if self.config:
            tool_types = self.config.get('tool_types', {})
            layout.addWidget(QLabel("Tools:"), 1, 0)
            tools_text = ", ".join([f"{name}({idx})" for name, idx in tool_types.items()])
            tools_label = QLabel(tools_text)
            tools_label.setWordWrap(True)
            layout.addWidget(tools_label, 1, 1)

            # CT point cloud info
            ct_path = self.config.get('CT_pc', 'Not specified')
            layout.addWidget(QLabel("CT PC:"), 2, 0)
            ct_label = QLabel(os.path.basename(ct_path) if ct_path else "Not specified")
            ct_label.setToolTip(ct_path)
            layout.addWidget(ct_label, 2, 1)

        group.setLayout(layout)
        return group

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

        # Check tools button
        self.check_tools_btn = QPushButton("Check Tool Visibility")
        self.check_tools_btn.clicked.connect(self.check_tools)
        layout.addWidget(self.check_tools_btn, 3, 0, 1, 2)

        # Connection status
        self.connection_status = QLabel("Not connected")
        layout.addWidget(self.connection_status, 4, 0, 1, 2)

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

        # Add refresh button
        self.refresh_coarse_btn = QPushButton("Refresh from Server")
        self.refresh_coarse_btn.clicked.connect(self.refresh_coarse_points_from_server)
        reg_layout.addWidget(self.refresh_coarse_btn)

        # Add debug button to test server response
        self.debug_server_btn = QPushButton("Debug Server Response")
        self.debug_server_btn.clicked.connect(self.debug_server_coarse_points)
        reg_layout.addWidget(self.debug_server_btn)

        # Add force update button for debugging
        self.force_update_btn = QPushButton("Force Update Visualization")
        self.force_update_btn.clicked.connect(self.force_update_coarse_visualization)
        reg_layout.addWidget(self.force_update_btn)

        # Visualization options
        self.show_coarse_points = QCheckBox("Show Coarse Points")
        self.show_coarse_points.setChecked(True)
        self.show_coarse_points.toggled.connect(self.update_coarse_visualization)
        reg_layout.addWidget(self.show_coarse_points)

        self.show_coarse_matches = QCheckBox("Show Point Matches")
        self.show_coarse_matches.setChecked(True)
        self.show_coarse_matches.toggled.connect(self.update_coarse_visualization)
        reg_layout.addWidget(self.show_coarse_matches)

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

        self.load_ct_btn = QPushButton("Load Different CT Point Cloud")
        self.load_ct_btn.clicked.connect(self.load_ct_pointcloud)
        viz_layout.addWidget(self.load_ct_btn)

        # Add debug button
        self.debug_ct_btn = QPushButton("Debug CT Point Cloud")
        self.debug_ct_btn.clicked.connect(self.debug_ct_pointcloud)
        viz_layout.addWidget(self.debug_ct_btn)

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

        # Device selection (based on config tool types)
        tool_layout.addWidget(QLabel("Device:"), 0, 0)
        self.device_spinbox = QSpinBox()
        self.device_spinbox.setRange(0, 5)
        if self.config and 'tool_types' in self.config:
            probe_idx = self.config['tool_types'].get('probe', 0)
            self.device_spinbox.setValue(probe_idx)
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
        # Create PyVista plotter widget with maximum performance optimizations
        self.plotter = QtInteractor(self)
        self.plotter.background_color = 'white'

        # CRITICAL PERFORMANCE OPTIMIZATIONS
        self.plotter.enable_anti_aliasing = False
        self.plotter.enable_depth_peeling = False

        # Get render window and set performance properties
        render_window = self.plotter.render_window
        render_window.SetMultiSamples(0)  # Disable multisampling
        render_window.SetLineSmoothing(0)  # Disable line smoothing
        render_window.SetPolygonSmoothing(0)  # Disable polygon smoothing
        render_window.SetPointSmoothing(0)  # Disable point smoothing

        # Get renderer and optimize
        renderer = self.plotter.renderer
        renderer.SetUseFXAA(False)  # Disable FXAA
        renderer.SetUseDepthPeeling(False)  # Disable depth peeling
        renderer.SetMaximumNumberOfPeels(0)  # No peeling layers

        # Optimize render window interactor
        interactor = self.plotter.interactor
        render_window_interactor = interactor.GetRenderWindow().GetInteractor()
        if render_window_interactor:
            render_window_interactor.SetDesiredUpdateRate(30)  # 30 FPS during interaction
            render_window_interactor.SetStillUpdateRate(5)  # 5 FPS when still

        # Initialize visualization elements
        self.coarse_actors = []
        self.coarse_line_actors = []
        self.fine_actors = []
        self.ct_actor = None
        self.probe_actor = None
        self.probe_trail = []

        # Add smaller coordinate axes for better performance
        self.plotter.add_axes(viewport=(0, 0, 0.15, 0.15))

        # Set better camera properties for smoother interaction
        camera = self.plotter.camera
        camera.SetParallelProjection(False)  # Use perspective projection (usually faster)

        # Initial render
        try:
            self.plotter.render()
            print("Initial render completed successfully")
        except Exception as e:
            print(f"Initial render failed: {str(e)}")

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

    def check_tools(self):
        """Check tool visibility"""
        try:
            response = requests.post(f"{self.server_url}/check_tools")
            if response.status_code == 200:
                result = response.json()
                tool_status = []
                for tool, visible in result.items():
                    status = "✓" if visible else "✗"
                    tool_status.append(f"{tool}: {status}")
                self.show_message("Tool Visibility", "\n".join(tool_status))
            else:
                self.show_message("Error", f"Failed to check tools: {response.status_code}")
        except Exception as e:
            self.show_message("Error", f"Error checking tools: {str(e)}")

    def update_status(self):
        try:
            response = requests.get(f"{self.server_url}/")
            if response.status_code == 200:
                status = response.json()
                status_text = f"""Application: {status.get('application', 'Unknown')}
Status: {status.get('status', 'Unknown')}
Data Source: {status.get('data_source', 'Unknown')}
NDI Tracker: {status.get('ndi_tracker_status', 'Unknown')}
Coarse Points: {status.get('coarse_points_loaded', 0)}
Fine Points: {status.get('fine_points_loaded', 0)}
Streaming: {status.get('streaming_active', False)}
Coarse Registration: {status.get('has_coarse_registration', False)}
Fine Registration: {status.get('has_fine_registration', False)}
Tool Calibration: {status.get('has_tool_calibration', False)}"""
                self.status_text.setText(status_text.strip())
        except Exception as e:
            self.status_text.setText(f"Error getting status: {str(e)}")

    # Enhanced Coarse registration methods
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
                    # Store the point pair locally for immediate visualization
                    ndi_point = result.get('ndi_point', [0, 0, 0])
                    self.coarse_points[point_number] = {
                        'unity_point': unity_point,
                        'ndi_point': ndi_point,
                        'data_source': result.get('data_source', 'unknown')
                    }

                    self.update_coarse_points_display()
                    self.update_coarse_visualization()
                    self.show_message("Success", f"Point {point_number} set successfully")

                    # Auto-increment point number
                    self.point_number.setValue(point_number + 1)

                    # Force immediate check for server updates (in case other clients also added points)
                    QTimer.singleShot(500, self.check_coarse_points_updates)
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
                # Clear local storage
                self.coarse_points.clear()
                self.update_coarse_points_display()
                self.update_coarse_visualization()
                self.show_message("Success", "Coarse points reset")
                # Reset point number
                self.point_number.setValue(0)
            else:
                self.show_message("Error", f"Failed to reset coarse points: {response.status_code}")
        except Exception as e:
            self.show_message("Error", f"Error resetting coarse points: {str(e)}")

    def update_coarse_points_display(self):
        """Update the coarse points text display"""
        if not self.coarse_points:
            self.coarse_points_text.setText("No coarse points set")
            return

        display_text = []
        for point_num, point_data in sorted(self.coarse_points.items()):
            unity_pt = point_data['unity_point']
            ndi_pt = point_data['ndi_point']
            source = point_data['data_source']

            text = f"Point {point_num}:\n"
            text += f"  Unity: [{unity_pt[0]:.2f}, {unity_pt[1]:.2f}, {unity_pt[2]:.2f}]\n"
            text += f"  NDI: [{ndi_pt[0]:.2f}, {ndi_pt[1]:.2f}, {ndi_pt[2]:.2f}]\n"
            text += f"  Source: {source}\n"
            display_text.append(text)

        self.coarse_points_text.setText("\n".join(display_text))

    def update_coarse_visualization(self):
        """Update coarse point visualization - PERFORMANCE OPTIMIZED"""
        print(f"DEBUG: update_coarse_visualization called with {len(self.coarse_points)} points")

        # Batch remove all actors without rendering
        all_actors_to_remove = self.coarse_actors + self.coarse_line_actors
        for actor in all_actors_to_remove:
            try:
                self.plotter.remove_actor(actor, render=False)
            except:
                pass

        self.coarse_actors.clear()
        self.coarse_line_actors.clear()

        if not self.show_coarse_points.isChecked() or not self.coarse_points:
            print("DEBUG: Not showing points or no points to show")
            self.request_render()
            return

        # Collect points for visualization
        unity_points = []
        ndi_points = []
        point_numbers = []

        for point_num, point_data in self.coarse_points.items():
            unity_points.append(point_data['unity_point'])
            ndi_points.append(point_data['ndi_point'])
            point_numbers.append(point_num)

        unity_points = np.array(unity_points)
        ndi_points = np.array(ndi_points)

        try:
            # Create Unity points - use simple spheres instead of glyphs
            if len(unity_points) > 0:
                for i, (point, point_num) in enumerate(zip(unity_points, point_numbers)):
                    # Use simple sphere geometry (much faster than glyphs)
                    sphere = pv.Sphere(radius=2.0, center=point, phi_resolution=8, theta_resolution=8)  # Low resolution
                    actor = self.plotter.add_mesh(
                        sphere,
                        color='blue',
                        name=f'unity_point_{point_num}',
                        opacity=0.8,
                        render=False,
                        pickable=False  # Non-pickable for better performance
                    )
                    self.coarse_actors.append(actor)

                    # Add simple text labels (no point labels which are slow)
                    try:
                        text_actor = self.plotter.add_text(
                            f'U{point_num}',
                            position=point + np.array([2, 2, 2]),  # Offset slightly
                            font_size=8,
                            color='blue',
                            name=f'unity_text_{point_num}',
                            render=False
                        )
                        self.coarse_actors.append(text_actor)
                    except:
                        pass  # Skip labels if they fail

            # Create NDI points
            if len(ndi_points) > 0:
                for i, (point, point_num) in enumerate(zip(ndi_points, point_numbers)):
                    sphere = pv.Sphere(radius=2.0, center=point, phi_resolution=8, theta_resolution=8)  # Low resolution
                    actor = self.plotter.add_mesh(
                        sphere,
                        color='red',
                        name=f'ndi_point_{point_num}',
                        opacity=0.8,
                        render=False,
                        pickable=False
                    )
                    self.coarse_actors.append(actor)

                    # Add text labels
                    try:
                        text_actor = self.plotter.add_text(
                            f'N{point_num}',
                            position=point + np.array([2, 2, 2]),
                            font_size=8,
                            color='red',
                            name=f'ndi_text_{point_num}',
                            render=False
                        )
                        self.coarse_actors.append(text_actor)
                    except:
                        pass

            # Create connection lines (simplified)
            if self.show_coarse_matches.isChecked() and len(unity_points) > 0:
                for i, point_num in enumerate(point_numbers):
                    # Create individual line segments (simpler than complex polydata)
                    line = pv.Line(unity_points[i], ndi_points[i])

                    actor = self.plotter.add_mesh(
                        line,
                        color='green',
                        line_width=2,
                        name=f'match_line_{point_num}',
                        opacity=0.7,
                        render=False,
                        pickable=False
                    )
                    self.coarse_line_actors.append(actor)

            print(
                f"DEBUG: Created {len(self.coarse_actors)} coarse actors and {len(self.coarse_line_actors)} line actors")

        except Exception as e:
            print(f"DEBUG: Error in coarse visualization: {str(e)}")

        # Single render at the end
        self.request_render()

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
                    self.fine_timer.start(1000)  # Update every 1 second
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
                status_text = f"""Gathering Active: {status.get('gathering_active', False)}
Points Collected: {status.get('points_collected', 0)}
Collection Rate: {status.get('collection_rate', 0):.1f} Hz"""
                self.fine_status_text.setText(status_text.strip())

                # Update visualization if gathering is active
                if status.get('gathering_active', False):
                    self.update_fine_visualization()
        except Exception as e:
            self.fine_status_text.setText(f"Error getting status: {str(e)}")

    def update_fine_visualization(self):
        # Remove existing fine actors
        for actor in self.fine_actors:
            try:
                self.plotter.remove_actor(actor, render=False)
            except:
                pass
        self.fine_actors.clear()

        if self.show_fine_points.isChecked():
            # Get fine points from server and visualize
            # This would need access to the fine points data
            pass

        self.request_render()

    # Tool calibration methods
    def start_tool_calibration(self):
        try:
            device = self.device_spinbox.value()
            response = requests.post(f"{self.server_url}/start_tool_calibration",
                                     params={"device": device, "force_stop_streaming": True})
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
                status_text = f"""Calibration Active: {status.get('calibration_active', False)}
Matrices Collected: {status.get('matrices_collected', 0)}
Tool Tip Vector: {status.get('tool_tip_vector', 'Not calibrated')}"""
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
                "frequency": frequency,
                "force_stop_calibration": True
            }

            response = requests.post(f"{self.server_url}/start_streaming", params=params)
            if response.status_code == 200:
                result = response.json()
                if result.get('status') in ['started', 'already_running']:
                    self.streaming_active = True
                    self.start_stream_btn.setEnabled(False)
                    self.stop_stream_btn.setEnabled(True)
                    self.streaming_timer.start(200)  # Update every 200ms
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
                    pos_text = f"""Position: {position if position else 'No data'}
Original: {data.get('original', 'No data')}
Timestamp: {data.get('timestamp', 'Unknown')}
Frame: {data.get('frame', 'Unknown')}"""
                    self.position_text.setText(pos_text.strip())

        except Exception as e:
            self.position_text.setText(f"Error getting position: {str(e)}")

    def update_probe_position(self, position):
        # Remove existing probe actor
        if self.probe_actor:
            try:
                self.plotter.remove_actor(self.probe_actor, render=False)
            except:
                pass

        # Create sphere at probe position
        try:
            sphere = pv.Sphere(radius=2.0, center=position)
            self.probe_actor = self.plotter.add_mesh(sphere, color='red', name='probe_position', render=False)
            self.request_render()
        except Exception as e:
            print(f"Error updating probe position: {str(e)}")

    def add_to_probe_trail(self, position):
        self.probe_trail.append(position)

        # Limit trail length for performance
        if len(self.probe_trail) > 500:  # Reduced from 1000
            self.probe_trail.pop(0)

        # Update trail visualization less frequently
        if len(self.probe_trail) > 1 and len(self.probe_trail) % 5 == 0:  # Update every 5 points
            try:
                trail_points = np.array(self.probe_trail)
                trail_line = pv.PolyData(trail_points)

                # Remove existing trail
                if hasattr(self, 'trail_actor'):
                    try:
                        self.plotter.remove_actor(self.trail_actor, render=False)
                    except:
                        pass

                self.trail_actor = self.plotter.add_mesh(trail_line, color='blue',
                                                         line_width=2, name='probe_trail', render=False)
                self.request_render()
            except Exception as e:
                print(f"Error updating probe trail: {str(e)}")

    def clear_probe_trail(self):
        self.probe_trail.clear()
        if hasattr(self, 'trail_actor'):
            try:
                self.plotter.remove_actor(self.trail_actor, render=False)
            except:
                pass
        self.request_render()

    # CT point cloud methods (PERFORMANCE OPTIMIZED)
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
                    # Load numpy array
                    ct_points = np.load(file_path)
                    print(f"Loaded numpy array with shape: {ct_points.shape}")

                    if ct_points.shape[1] >= 3:
                        self.ct_pointcloud = pv.PolyData(ct_points[:, :3])
                        print(f"Created PyVista point cloud with {self.ct_pointcloud.n_points} points")
                    else:
                        raise ValueError("Invalid point cloud format - need at least 3 columns")
                else:
                    # Load other formats using PyVista
                    self.ct_pointcloud = pv.read(file_path)
                    print(f"Loaded point cloud with {self.ct_pointcloud.n_points} points")

                # Auto-check the checkbox and update visualization
                self.show_ct_cloud.setChecked(True)
                self.update_ct_visualization()

                # Show bounds information in console only
                bounds = self.ct_pointcloud.bounds
                print(
                    f"CT Point Cloud Bounds: X[{bounds[0]:.1f}, {bounds[1]:.1f}] Y[{bounds[2]:.1f}, {bounds[3]:.1f}] Z[{bounds[4]:.1f}, {bounds[5]:.1f}]")
                print(f"Successfully loaded: {os.path.basename(file_path)} ({self.ct_pointcloud.n_points} points)")

            except Exception as e:
                print(f"Error loading CT point cloud: {str(e)}")
                self.show_message("Error", f"Error loading CT point cloud: {str(e)}")

    def update_ct_visualization(self):
        """Update CT point cloud visualization - MAXIMUM PERFORMANCE VERSION"""
        # Remove existing CT actor
        if self.ct_actor:
            try:
                self.plotter.remove_actor(self.ct_actor, render=False)
            except:
                pass
            self.ct_actor = None

        if self.show_ct_cloud.isChecked() and self.ct_pointcloud is not None:
            print(f"Displaying CT point cloud with {self.ct_pointcloud.n_points} points")

            try:
                points = self.ct_pointcloud.points

                # AGGRESSIVE downsampling for smooth interaction - MAX 2000 points
                max_points = 2000  # Reduced from 5000
                if self.ct_pointcloud.n_points > max_points:
                    print(f"Downsampling from {self.ct_pointcloud.n_points} to {max_points} points")
                    # Use systematic sampling (faster than random)
                    step = self.ct_pointcloud.n_points // max_points
                    indices = np.arange(0, self.ct_pointcloud.n_points, step)[:max_points]
                    downsampled_points = points[indices].copy()
                else:
                    print(f"Using all {self.ct_pointcloud.n_points} points")
                    downsampled_points = points.copy()

                print(f"Selected {len(downsampled_points)} points for visualization")

                # Create optimized point cloud
                point_cloud = pv.PolyData(downsampled_points)

                # Use vertices instead of spheres for much better performance
                point_cloud.verts = np.hstack([[1, i] for i in range(len(downsampled_points))])

                # Add with performance optimizations
                self.ct_actor = self.plotter.add_mesh(
                    point_cloud,
                    style='points',
                    color='lightgray',
                    point_size=2.0,  # Smaller points
                    opacity=0.6,  # Slightly more transparent
                    name='ct_pointcloud',
                    render=False,
                    pickable=False  # Make non-pickable for better performance
                )

                if self.ct_actor:
                    print(f"CT point cloud actor created successfully with {len(downsampled_points)} points")
                    # Set level of detail for the actor
                    if hasattr(self.ct_actor, 'GetMapper'):
                        mapper = self.ct_actor.GetMapper()
                        if mapper:
                            # Enable immediate mode rendering for better interaction
                            mapper.SetImmediateModeRendering(True)

            except Exception as e:
                print(f"Error creating CT visualization: {str(e)}")

        # Request render
        self.request_render()

    def debug_ct_pointcloud(self):
        """Debug method to check CT point cloud status"""
        if self.ct_pointcloud is None:
            print("No CT point cloud loaded")
            self.show_message("Debug Info", "No CT point cloud loaded")
            return

        print(f"CT Point Cloud Info:")
        print(f"  Points: {self.ct_pointcloud.n_points}")
        print(f"  Bounds: {self.ct_pointcloud.bounds}")
        print(f"  Center: {self.ct_pointcloud.center}")
        print(f"  Show checkbox checked: {self.show_ct_cloud.isChecked()}")
        print(f"  CT actor exists: {self.ct_actor is not None}")

        if self.ct_actor:
            print(f"  CT actor visible: {self.ct_actor.GetVisibility()}")

        # Print some sample points
        if self.ct_pointcloud.n_points > 0:
            sample_points = self.ct_pointcloud.points[:min(5, self.ct_pointcloud.n_points)]
            print(f"  Sample points: {sample_points}")

        # Show debug info in message box
        bounds = self.ct_pointcloud.bounds
        debug_info = f"""CT Point Cloud Debug Info:
Points: {self.ct_pointcloud.n_points}
Bounds: X[{bounds[0]:.1f}, {bounds[1]:.1f}] Y[{bounds[2]:.1f}, {bounds[3]:.1f}] Z[{bounds[4]:.1f}, {bounds[5]:.1f}]
Center: [{self.ct_pointcloud.center[0]:.1f}, {self.ct_pointcloud.center[1]:.1f}, {self.ct_pointcloud.center[2]:.1f}]
Show checkbox: {self.show_ct_cloud.isChecked()}
Actor exists: {self.ct_actor is not None}
Actor visible: {self.ct_actor.GetVisibility() if self.ct_actor else 'N/A'}"""

        self.show_message("CT Point Cloud Debug", debug_info)

    # Utility methods
    def show_message(self, title, message):
        try:
            msg = QMessageBox()
            msg.setWindowTitle(title)
            msg.setText(str(message))
            msg.exec_()
        except Exception as e:
            print(f"Error showing message: {str(e)}")

    def closeEvent(self, event):
        # Stop streaming if active
        if self.streaming_active:
            try:
                requests.post(f"{self.server_url}/stop_streaming")
            except:
                pass

        event.accept()


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="NDI Tracking GUI")
    parser.add_argument("--ndi_config_path", type=str, required=True,
                        help="Path to the NDI configuration JSON file")
    parser.add_argument("--server_url", type=str, default="http://localhost:8000",
                        help="NDI server URL (default: http://localhost:8000)")

    args = parser.parse_args()

    # Validate config file exists
    if not os.path.exists(args.ndi_config_path):
        print(f"Error: Configuration file not found: {args.ndi_config_path}")
        sys.exit(1)

    app = QApplication(sys.argv)

    # Set application style
    app.setStyle('Fusion')

    # Create and show the GUI
    gui = NDITrackingGUI(args.ndi_config_path, args.server_url)
    gui.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
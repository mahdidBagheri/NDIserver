import sys
import os
import argparse
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                             QWidget, QTabWidget, QSplitter, QGroupBox, QGridLayout,
                             QLabel, QPushButton, QTextEdit, QLineEdit)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QFont

from config_manager_gui import ConfigManager
from server_communication_gui import ServerClient
from visualization_3d_gui import Visualization3D
from tabs.coarse_registration_gui import CoarseRegistrationTab
from tabs.fine_registration_gui import FineRegistrationTab
from tabs.tool_calibration_gui import ToolCalibrationTab
from tabs.streaming_gui import StreamingTab


class NDITrackingGUI(QMainWindow):
    def __init__(self, config_path, server_url="http://localhost:8000"):
        super().__init__()
        self.setWindowTitle("NDI Tracking Visualization System")
        self.setGeometry(100, 100, 1600, 1000)

        # Initialize core components
        self.config_manager = ConfigManager(config_path)
        self.server_client = ServerClient(server_url)
        self.visualization_3d = Visualization3D()

        # Initialize UI
        self.init_ui()

        # Setup timers
        self.setup_timers()

        # Connect signals
        self.connect_signals()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Create main splitter
        main_splitter = QSplitter(Qt.Horizontal)

        # Left panel for controls
        control_panel = self.create_control_panel()
        main_splitter.addWidget(control_panel)

        # Right panel for 3D visualization
        viz_widget = self.visualization_3d.get_widget()
        main_splitter.addWidget(viz_widget)

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

        # Configuration info
        config_group = self.create_config_group()
        layout.addWidget(config_group)

        # Server connection
        server_group = self.create_server_group()
        layout.addWidget(server_group)

        # Create tabs
        tab_widget = QTabWidget()

        # Initialize tabs with shared components
        self.coarse_tab = CoarseRegistrationTab(self.server_client, self.visualization_3d)
        self.fine_tab = FineRegistrationTab(self.server_client, self.visualization_3d, self.config_manager)
        self.tool_tab = ToolCalibrationTab(self.server_client, self.config_manager)
        self.streaming_tab = StreamingTab(self.server_client, self.visualization_3d)

        tab_widget.addTab(self.coarse_tab, "Coarse Registration")
        tab_widget.addTab(self.fine_tab, "Fine Registration")
        tab_widget.addTab(self.tool_tab, "Tool Calibration")
        tab_widget.addTab(self.streaming_tab, "Streaming")

        layout.addWidget(tab_widget)

        # Status display
        status_group = self.create_status_group()
        layout.addWidget(status_group)

        control_widget.setLayout(layout)
        return control_widget

    def create_config_group(self):
        group = QGroupBox("Configuration")
        layout = QGridLayout()

        config = self.config_manager.get_config()

        # Config file path
        layout.addWidget(QLabel("Config File:"), 0, 0)
        config_label = QLabel(os.path.basename(self.config_manager.config_path))
        config_label.setToolTip(self.config_manager.config_path)
        layout.addWidget(config_label, 0, 1)

        # Tool types
        if config:
            tool_types = config.get('tool_types', {})
            layout.addWidget(QLabel("Tools:"), 1, 0)
            tools_text = ", ".join([f"{name}({idx})" for name, idx in tool_types.items()])
            tools_label = QLabel(tools_text)
            tools_label.setWordWrap(True)
            layout.addWidget(tools_label, 1, 1)

        # CT point cloud info
        ct_path = config.get('CT_pc', 'Not specified') if config else 'Not specified'
        layout.addWidget(QLabel("CT PC:"), 2, 0)
        ct_label = QLabel(os.path.basename(ct_path) if ct_path != 'Not specified' else "Not specified")
        ct_label.setToolTip(ct_path)
        layout.addWidget(ct_label, 2, 1)

        group.setLayout(layout)
        return group

    def create_server_group(self):
        group = QGroupBox("Server Connection")
        layout = QGridLayout()

        # Server URL
        layout.addWidget(QLabel("Server URL:"), 0, 0)
        self.server_url_input = QLineEdit(self.server_client.server_url)
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

    def create_status_group(self):
        group = QGroupBox("System Status")
        layout = QVBoxLayout()

        self.status_text = QTextEdit()
        self.status_text.setMaximumHeight(150)
        self.status_text.setReadOnly(True)
        layout.addWidget(self.status_text)

        group.setLayout(layout)
        return group

    def setup_timers(self):
        # Status update timer
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_status)
        self.status_timer.start(2000)

    def connect_signals(self):
        # Connect server client signals
        self.server_client.connection_status_changed.connect(self.on_connection_status_changed)

        # Load CT point cloud from config
        self.config_manager.load_ct_pointcloud(self.visualization_3d)

    def test_connection(self):
        new_url = self.server_url_input.text()
        self.server_client.set_server_url(new_url)
        self.server_client.test_connection()

    def initialize_ndi(self):
        """Initialize NDI system"""
        result = self.server_client.initialize_ndi()
        if result:
            self.show_message("NDI Initialization",
                              f"Status: {result.get('status', 'Unknown')}")
        else:
            self.show_message("Error", "Failed to initialize NDI")

    def check_tools(self):
        """Check tool visibility"""
        result = self.server_client.check_tools()
        if result:
            tool_status = []
            for tool, visible in result.items():
                status = "✓" if visible else "✗"
                tool_status.append(f"{tool}: {status}")
            self.show_message("Tool Visibility", "\n".join(tool_status))
        else:
            self.show_message("Error", "Failed to check tools")

    def on_connection_status_changed(self, connected, message):
        if connected:
            self.connection_status.setText("Connected ✓")
            self.connection_status.setStyleSheet("color: green")
        else:
            self.connection_status.setText(f"Error: {message}")
            self.connection_status.setStyleSheet("color: red")

    def update_status(self):
        status = self.server_client.get_status()
        if status:
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
        else:
            self.status_text.setText("Error getting server status")

    def show_message(self, title, message):
        """Show message dialog"""
        from PyQt5.QtWidgets import QMessageBox
        try:
            msg = QMessageBox()
            msg.setWindowTitle(title)
            msg.setText(str(message))
            msg.exec_()
        except Exception as e:
            print(f"Error showing message: {e}")

    def closeEvent(self, event):
        # Cleanup
        self.server_client.cleanup()
        event.accept()


def main():
    """Main entry point for the NDI Tracking GUI application"""
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

    # Create QApplication
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    # Set application properties
    app.setApplicationName("NDI Tracking GUI")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("NDI Tracking System")

    try:
        # Create and show the main GUI
        gui = NDITrackingGUI(args.ndi_config_path, args.server_url)
        gui.show()

        print(f"NDI Tracking GUI started successfully")
        print(f"Config file: {args.ndi_config_path}")
        print(f"Server URL: {args.server_url}")

        # Run the application
        sys.exit(app.exec_())

    except Exception as e:
        print(f"Error starting GUI: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
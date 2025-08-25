from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
                             QPushButton, QLineEdit, QLabel, QSpinBox,
                             QCheckBox, QTextEdit, QProgressBar)
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont


class ConnectionWidget(QWidget):
    def __init__(self, api_client):
        super().__init__()
        self.api_client = api_client
        self.setup_ui()
        self.setup_connections()

    def setup_ui(self):
        layout = QVBoxLayout(self)

        # Server connection group
        connection_group = QGroupBox("Server Connection")
        connection_layout = QVBoxLayout(connection_group)

        # Server address
        addr_layout = QHBoxLayout()
        addr_layout.addWidget(QLabel("Server Address:"))
        self.server_addr_edit = QLineEdit("http://localhost:8000")
        addr_layout.addWidget(self.server_addr_edit)
        connection_layout.addLayout(addr_layout)

        # Connection buttons
        btn_layout = QHBoxLayout()
        self.connect_btn = QPushButton("Connect")
        self.disconnect_btn = QPushButton("Disconnect")
        self.disconnect_btn.setEnabled(False)
        btn_layout.addWidget(self.connect_btn)
        btn_layout.addWidget(self.disconnect_btn)
        connection_layout.addLayout(btn_layout)

        layout.addWidget(connection_group)

        # NDI Tracker group
        ndi_group = QGroupBox("NDI Tracker")
        ndi_layout = QVBoxLayout(ndi_group)

        # Initialize button
        init_layout = QHBoxLayout()
        self.init_btn = QPushButton("Initialize NDI")
        self.force_restart_cb = QCheckBox("Force Restart")
        init_layout.addWidget(self.init_btn)
        init_layout.addWidget(self.force_restart_cb)
        ndi_layout.addLayout(init_layout)

        # Find reference button
        self.find_ref_btn = QPushButton("Find Reference")
        ndi_layout.addWidget(self.find_ref_btn)

        # Check tools button
        self.check_tools_btn = QPushButton("Check Tools")
        ndi_layout.addWidget(self.check_tools_btn)

        layout.addWidget(ndi_group)

        # Client IP group
        ip_group = QGroupBox("Client IP Configuration")
        ip_layout = QVBoxLayout(ip_group)

        # Current IP display
        self.current_ip_label = QLabel("Current IP: Not set")
        ip_layout.addWidget(self.current_ip_label)

        # Set IP button
        self.set_ip_btn = QPushButton("Set Client IP (Auto-detect)")
        ip_layout.addWidget(self.set_ip_btn)

        layout.addWidget(ip_group)

        # Status display
        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout(status_group)

        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMaximumHeight(150)
        status_layout.addWidget(self.status_text)

        layout.addWidget(status_group)

        layout.addStretch()

    def setup_connections(self):
        self.connect_btn.clicked.connect(self.connect_to_server)
        self.disconnect_btn.clicked.connect(self.disconnect_from_server)
        self.init_btn.clicked.connect(self.initialize_ndi)
        self.find_ref_btn.clicked.connect(self.find_reference)
        self.check_tools_btn.clicked.connect(self.check_tools)
        self.set_ip_btn.clicked.connect(self.set_client_ip)

    def connect_to_server(self):
        address = self.server_addr_edit.text()
        self.api_client.set_base_url(address)

        # Test connection
        try:
            response = self.api_client.get_status()
            if response:
                self.connect_btn.setEnabled(False)
                self.disconnect_btn.setEnabled(True)
                self.status_text.append("✓ Connected to server successfully")
                self.update_status_display(response)
            else:
                self.status_text.append("✗ Failed to connect to server")
        except Exception as e:
            self.status_text.append(f"✗ Connection error: {str(e)}")

    def disconnect_from_server(self):
        self.api_client.set_base_url(None)
        self.connect_btn.setEnabled(True)
        self.disconnect_btn.setEnabled(False)
        self.status_text.append("Disconnected from server")

    def initialize_ndi(self):
        force_restart = self.force_restart_cb.isChecked()
        try:
            response = self.api_client.initialize_ndi(force_restart)
            if response.get('status') == 'success':
                self.status_text.append("✓ NDI tracker initialized successfully")
            else:
                self.status_text.append(f"✗ NDI initialization failed: {response.get('message', 'Unknown error')}")
        except Exception as e:
            self.status_text.append(f"✗ NDI initialization error: {str(e)}")

    def find_reference(self):
        try:
            response = self.api_client.find_reference()
            if response.get('status') == 'success':
                self.status_text.append("✓ Reference found successfully")
            else:
                self.status_text.append(f"✗ Reference search failed: {response.get('message', 'Unknown error')}")
        except Exception as e:
            self.status_text.append(f"✗ Reference search error: {str(e)}")

    def check_tools(self):
        try:
            response = self.api_client.check_tools()
            self.status_text.append("Tool visibility status:")
            for tool, visible in response.items():
                status = "✓ Visible" if visible else "✗ Not visible"
                self.status_text.append(f"  {tool}: {status}")
        except Exception as e:
            self.status_text.append(f"✗ Tool check error: {str(e)}")

    def set_client_ip(self):
        try:
            response = self.api_client.set_client_ip()
            if response.get('status') == 'success':
                ip = response.get('client_ip')
                self.current_ip_label.setText(f"Current IP: {ip}")
                self.status_text.append(f"✓ Client IP set to: {ip}")
            else:
                self.status_text.append(f"✗ Failed to set client IP: {response.get('details', 'Unknown error')}")
        except Exception as e:
            self.status_text.append(f"✗ Set client IP error: {str(e)}")

    def update_status(self, status):
        if status.get('connected'):
            self.update_status_display(status)

    def update_status_display(self, status):
        # Update current IP
        client_ip = status.get('client_ip', 'Not set')
        self.current_ip_label.setText(f"Current IP: {client_ip}")

        # Enable/disable buttons based on status
        ndi_initialized = status.get('ndi_tracker_status') == 'initialized'
        self.find_ref_btn.setEnabled(ndi_initialized)
        self.check_tools_btn.setEnabled(ndi_initialized)
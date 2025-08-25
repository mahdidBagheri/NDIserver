from PyQt5.QtWidgets import (QMainWindow, QTabWidget, QVBoxLayout, QHBoxLayout,
                             QWidget, QStatusBar, QMenuBar, QAction, QSplitter,
                             QDockWidget, QTextEdit, QLabel)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QIcon, QFont

from .widgets.connection_widget import ConnectionWidget
from .widgets.coarse_registration_widget import CoarseRegistrationWidget
from .widgets.fine_registration_widget import FineRegistrationWidget
from .widgets.tool_calibration_widget import ToolCalibrationWidget
from .widgets.streaming_widget import StreamingWidget
from .widgets.visualization_widget import VisualizationWidget
from .utils.api_client import APIClient
from .utils.status_manager import StatusManager


class NDIMainWindow(QMainWindow):
    status_updated = pyqtSignal(dict)

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.api_client = APIClient()
        self.status_manager = StatusManager(self.api_client)

        self.setWindowTitle("NDI Tracking Control Center")
        self.setGeometry(100, 100, 1400, 900)

        self.setup_ui()
        self.setup_connections()
        self.setup_status_updates()

    def setup_ui(self):
        # Create central widget with splitter
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QHBoxLayout(central_widget)

        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # Left panel - Control tabs
        self.create_control_panel(splitter)

        # Right panel - Visualization
        self.create_visualization_panel(splitter)

        # Set splitter proportions
        splitter.setSizes([600, 800])

        # Create menu bar
        self.create_menu_bar()

        # Create status bar
        self.create_status_bar()

        # Create log dock
        self.create_log_dock()

    def create_control_panel(self, parent):
        # Tab widget for different control categories
        self.tab_widget = QTabWidget()
        parent.addWidget(self.tab_widget)

        # Connection tab
        self.connection_widget = ConnectionWidget(self.api_client)
        self.tab_widget.addTab(self.connection_widget, "Connection")

        # Coarse Registration tab
        self.coarse_widget = CoarseRegistrationWidget(self.api_client)
        self.tab_widget.addTab(self.coarse_widget, "Coarse Registration")

        # Fine Registration tab
        self.fine_widget = FineRegistrationWidget(self.api_client)
        self.tab_widget.addTab(self.fine_widget, "Fine Registration")

        # Tool Calibration tab
        self.tool_widget = ToolCalibrationWidget(self.api_client)
        self.tab_widget.addTab(self.tool_widget, "Tool Calibration")

        # Streaming tab
        self.streaming_widget = StreamingWidget(self.api_client)
        self.tab_widget.addTab(self.streaming_widget, "Streaming")

    def create_visualization_panel(self, parent):
        self.visualization_widget = VisualizationWidget(self.api_client, self.config)
        parent.addWidget(self.visualization_widget)

    def create_menu_bar(self):
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu('File')

        save_state_action = QAction('Save State', self)
        save_state_action.triggered.connect(self.save_state)
        file_menu.addAction(save_state_action)

        load_state_action = QAction('Load State', self)
        load_state_action.triggered.connect(self.load_state)
        file_menu.addAction(load_state_action)

        file_menu.addSeparator()

        exit_action = QAction('Exit', self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # View menu
        view_menu = menubar.addMenu('View')

        self.log_dock_action = QAction('Show Log', self, checkable=True)
        self.log_dock_action.setChecked(True)
        self.log_dock_action.triggered.connect(self.toggle_log_dock)
        view_menu.addAction(self.log_dock_action)

        # Tools menu
        tools_menu = menubar.addMenu('Tools')

        settings_action = QAction('Settings', self)
        settings_action.triggered.connect(self.show_settings)
        tools_menu.addAction(settings_action)

    def create_status_bar(self):
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # Connection status
        self.connection_label = QLabel("Disconnected")
        self.connection_label.setStyleSheet("color: red; font-weight: bold;")
        self.status_bar.addWidget(self.connection_label)

        self.status_bar.addPermanentWidget(QLabel("Ready"))

    def create_log_dock(self):
        self.log_dock = QDockWidget("Log", self)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(200)
        self.log_dock.setWidget(self.log_text)

        self.addDockWidget(Qt.BottomDockWidgetArea, self.log_dock)

    def setup_connections(self):
        # Connect status updates
        self.status_manager.status_updated.connect(self.update_status)

        # Connect visualization updates
        self.coarse_widget.points_updated.connect(self.visualization_widget.update_coarse_points)
        self.coarse_widget.transformation_updated.connect(self.visualization_widget.update_coarse_transformation)
        self.fine_widget.points_updated.connect(self.visualization_widget.update_fine_points)
        self.fine_widget.transformation_updated.connect(self.visualization_widget.update_fine_transformation)
        self.streaming_widget.position_updated.connect(self.visualization_widget.update_tool_position)

    def setup_status_updates(self):
        # Timer for periodic status updates
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.status_manager.refresh_status)
        self.status_timer.start(2000)  # Update every 2 seconds

    def update_status(self, status):
        # Update connection status
        if status.get('connected', False):
            self.connection_label.setText("Connected")
            self.connection_label.setStyleSheet("color: green; font-weight: bold;")
        else:
            self.connection_label.setText("Disconnected")
            self.connection_label.setStyleSheet("color: red; font-weight: bold;")

        # Update widgets with status
        self.connection_widget.update_status(status)
        self.coarse_widget.update_status(status)
        self.fine_widget.update_status(status)
        self.tool_widget.update_status(status)
        self.streaming_widget.update_status(status)

    def toggle_log_dock(self, visible):
        self.log_dock.setVisible(visible)

    def save_state(self):
        # Implement state saving
        pass

    def load_state(self):
        # Implement state loading
        pass

    def show_settings(self):
        # Show settings dialog
        pass

    def log_message(self, message):
        self.log_text.append(message)
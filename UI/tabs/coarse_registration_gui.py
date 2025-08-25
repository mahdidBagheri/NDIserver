from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
                             QGridLayout, QLabel, QPushButton, QDoubleSpinBox,
                             QSpinBox, QCheckBox, QTextEdit, QMessageBox)
from PyQt5.QtCore import QTimer
from utils.ui_helpers import show_message_box


class CoarseRegistrationTab(QWidget):
    def __init__(self, server_client, visualization_3d):
        super().__init__()
        self.server_client = server_client
        self.visualization_3d = visualization_3d
        self.coarse_points = {}

        self.init_ui()
        self.connect_signals()
        self.setup_monitoring()

    def init_ui(self):
        layout = QVBoxLayout()

        # Point input group
        point_group = self.create_point_input_group()
        layout.addWidget(point_group)

        # Registration group
        reg_group = self.create_registration_group()
        layout.addWidget(reg_group)

        # Points display
        self.points_text = QTextEdit()
        self.points_text.setMaximumHeight(150)
        self.points_text.setReadOnly(True)
        layout.addWidget(QLabel("Coarse Points:"))
        layout.addWidget(self.points_text)

        self.setLayout(layout)

    def create_point_input_group(self):
        group = QGroupBox("Point Input")
        layout = QGridLayout()

        # Unity coordinates
        layout.addWidget(QLabel("Unity X:"), 0, 0)
        self.unity_x = QDoubleSpinBox()
        self.unity_x.setRange(-1000, 1000)
        self.unity_x.setDecimals(3)
        layout.addWidget(self.unity_x, 0, 1)

        layout.addWidget(QLabel("Unity Y:"), 1, 0)
        self.unity_y = QDoubleSpinBox()
        self.unity_y.setRange(-1000, 1000)
        self.unity_y.setDecimals(3)
        layout.addWidget(self.unity_y, 1, 1)

        layout.addWidget(QLabel("Unity Z:"), 2, 0)
        self.unity_z = QDoubleSpinBox()
        self.unity_z.setRange(-1000, 1000)
        self.unity_z.setDecimals(3)
        layout.addWidget(self.unity_z, 2, 1)

        # Point number
        layout.addWidget(QLabel("Point #:"), 3, 0)
        self.point_number = QSpinBox()
        self.point_number.setRange(0, 100)
        layout.addWidget(self.point_number, 3, 1)

        # Set point button
        self.set_point_btn = QPushButton("Set Coarse Point")
        self.set_point_btn.clicked.connect(self.set_coarse_point)
        layout.addWidget(self.set_point_btn, 4, 0, 1, 2)

        group.setLayout(layout)
        return group

    def create_registration_group(self):
        group = QGroupBox("Registration")
        layout = QVBoxLayout()

        # Registration controls
        self.register_btn = QPushButton("Perform Coarse Registration")
        self.register_btn.clicked.connect(self.perform_registration)
        layout.addWidget(self.register_btn)

        self.reset_btn = QPushButton("Reset Coarse Points")
        self.reset_btn.clicked.connect(self.reset_points)
        layout.addWidget(self.reset_btn)

        self.refresh_btn = QPushButton("Refresh from Server")
        self.refresh_btn.clicked.connect(self.refresh_from_server)
        layout.addWidget(self.refresh_btn)

        # Debug buttons
        self.debug_btn = QPushButton("Debug Server Response")
        self.debug_btn.clicked.connect(self.debug_server_response)
        layout.addWidget(self.debug_btn)

        # Visualization options
        self.show_points_cb = QCheckBox("Show Coarse Points")
        self.show_points_cb.setChecked(True)
        self.show_points_cb.toggled.connect(self.update_visualization)
        layout.addWidget(self.show_points_cb)

        self.show_matches_cb = QCheckBox("Show Point Matches")
        self.show_matches_cb.setChecked(True)
        self.show_matches_cb.toggled.connect(self.update_visualization)
        layout.addWidget(self.show_matches_cb)

        group.setLayout(layout)
        return group

    def connect_signals(self):
        # Connect to server signals
        self.server_client.coarse_points_updated.connect(self.on_server_points_updated)

    def setup_monitoring(self):
        # Monitor server for coarse points updates
        self.monitor_timer = QTimer()
        self.monitor_timer.timeout.connect(self.check_server_updates)
        self.monitor_timer.start(1000)  # Check every second

    def check_server_updates(self):
        """Check server for coarse points updates"""
        self.server_client.get_coarse_points()

    def on_server_points_updated(self, server_points):
        """Handle server points update signal"""
        self.update_points_from_server(server_points)
        self.update_points_display()
        self.update_visualization()

    def update_points_from_server(self, server_points):
        """Update local points from server data"""
        self.coarse_points.clear()

        for i, point_data in enumerate(server_points):
            # Handle different possible field names
            point_num = (point_data.get('point_number') or
                         point_data.get('point_id') or
                         point_data.get('id') or i)

            unity_point = (point_data.get('unity_point') or
                           point_data.get('unity_coordinate') or
                           point_data.get('unity') or [0, 0, 0])

            ndi_point = (point_data.get('ndi_point') or
                         point_data.get('ndi_coordinate') or
                         point_data.get('ndi') or [0, 0, 0])

            data_source = (point_data.get('data_source') or
                           point_data.get('source') or 'server')

            self.coarse_points[point_num] = {
                'unity_point': unity_point,
                'ndi_point': ndi_point,
                'data_source': data_source
            }

        # Update next point number
        if self.coarse_points:
            max_num = max(self.coarse_points.keys())
            self.point_number.setValue(max_num + 1)
        else:
            self.point_number.setValue(0)

    def set_coarse_point(self):
        unity_point = [self.unity_x.value(), self.unity_y.value(), self.unity_z.value()]
        point_number = self.point_number.value()

        result = self.server_client.set_coarse_point(unity_point, point_number)

        if result.get('status') == 'success':
            # Store locally for immediate visualization
            ndi_point = result.get('ndi_point', [0, 0, 0])
            self.coarse_points[point_number] = {
                'unity_point': unity_point,
                'ndi_point': ndi_point,
                'data_source': result.get('data_source', 'local')
            }

            self.update_points_display()
            self.update_visualization()
            show_message_box(self, "Success", f"Point {point_number} set successfully")
            self.point_number.setValue(point_number + 1)
        else:
            show_message_box(self, "Error", result.get('details', 'Unknown error'), "error")

    def perform_registration(self):
        result = self.server_client.perform_coarse_registration()

        if result.get('status') == 'success':
            show_message_box(self, "Success", "Coarse registration completed")
        else:
            show_message_box(self, "Error", result.get('details', 'Registration failed'), "error")

    def reset_points(self):
        result = self.server_client.reset_coarse_points()

        if result.get('status') == 'success':
            self.coarse_points.clear()
            self.update_points_display()
            self.update_visualization()
            self.point_number.setValue(0)
            show_message_box(self, "Success", "Coarse points reset")
        else:
            show_message_box(self, "Error", result.get('details', 'Reset failed'), "error")

    def refresh_from_server(self):
        points = self.server_client.get_coarse_points()
        if points is not None:
            show_message_box(self, "Success", f"Refreshed {len(points)} points from server")

    def debug_server_response(self):
        points = self.server_client.get_coarse_points()
        debug_info = f"""Server Response Debug:
Points count: {len(points) if points else 0}
Raw response: {str(points)[:500]}..."""
        show_message_box(self, "Debug Info", debug_info)

    def update_points_display(self):
        """Update the points text display"""
        if not self.coarse_points:
            self.points_text.setText("No coarse points set")
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

        self.points_text.setText("\n".join(display_text))

    def update_visualization(self):
        """Update 3D visualization"""
        show_points = self.show_points_cb.isChecked()
        show_matches = self.show_matches_cb.isChecked()

        self.visualization_3d.update_coarse_points(
            self.coarse_points, show_points, show_matches
        )
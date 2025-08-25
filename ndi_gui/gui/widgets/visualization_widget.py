import numpy as np
import os
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QCheckBox, QGroupBox, QLabel, QSlider, QSpinBox)
from PyQt5.QtCore import QTimer, pyqtSignal
import pyvista as pv
from pyvistaqt import QtInteractor


class VisualizationWidget(QWidget):
    def __init__(self, api_client, config):
        super().__init__()
        self.api_client = api_client
        self.config = config
        self.setup_ui()
        self.setup_scene()

        # Data storage
        self.coarse_points = []
        self.fine_points = []
        self.tool_position = None
        self.endoscope_position = None

        # CT point cloud data
        self.ct_point_cloud = None
        self.ct_actor = None
        self.transformed_ct_actor = None

        # Transformation matrices
        self.coarse_transformation = None
        self.fine_transformation = None

        # Load CT point cloud if specified
        self.load_ct_point_cloud()

    def setup_ui(self):
        layout = QVBoxLayout(self)

        # Control panel
        control_group = QGroupBox("Visualization Controls")
        control_layout = QVBoxLayout(control_group)

        # First row - basic visibility controls
        visibility_layout = QHBoxLayout()

        self.show_coarse_cb = QCheckBox("Show Coarse Points")
        self.show_coarse_cb.setChecked(True)
        self.show_coarse_cb.toggled.connect(self.update_visibility)

        self.show_fine_cb = QCheckBox("Show Fine Points")
        self.show_fine_cb.setChecked(True)
        self.show_fine_cb.toggled.connect(self.update_visibility)

        self.show_tools_cb = QCheckBox("Show Tools")
        self.show_tools_cb.setChecked(True)
        self.show_tools_cb.toggled.connect(self.update_visibility)

        visibility_layout.addWidget(self.show_coarse_cb)
        visibility_layout.addWidget(self.show_fine_cb)
        visibility_layout.addWidget(self.show_tools_cb)
        visibility_layout.addStretch()

        control_layout.addLayout(visibility_layout)

        # Second row - CT point cloud controls
        ct_layout = QHBoxLayout()

        self.show_original_ct_cb = QCheckBox("Show Original CT")
        self.show_original_ct_cb.setChecked(True)
        self.show_original_ct_cb.toggled.connect(self.update_ct_visibility)

        self.show_transformed_ct_cb = QCheckBox("Show Transformed CT")
        self.show_transformed_ct_cb.setChecked(True)
        self.show_transformed_ct_cb.toggled.connect(self.update_ct_visibility)

        # CT point cloud opacity
        ct_layout.addWidget(QLabel("CT Opacity:"))
        self.ct_opacity_slider = QSlider()
        self.ct_opacity_slider.setOrientation(1)  # Horizontal
        self.ct_opacity_slider.setRange(0, 100)
        self.ct_opacity_slider.setValue(80)
        self.ct_opacity_slider.valueChanged.connect(self.update_ct_opacity)

        # CT point size
        ct_layout.addWidget(QLabel("Point Size:"))
        self.ct_point_size_spin = QSpinBox()
        self.ct_point_size_spin.setRange(1, 10)
        self.ct_point_size_spin.setValue(2)
        self.ct_point_size_spin.valueChanged.connect(self.update_ct_point_size)

        ct_layout.addWidget(self.show_original_ct_cb)
        ct_layout.addWidget(self.show_transformed_ct_cb)
        ct_layout.addWidget(self.ct_opacity_slider)
        ct_layout.addWidget(self.ct_point_size_spin)
        ct_layout.addStretch()

        control_layout.addLayout(ct_layout)

        # Third row - utility buttons
        button_layout = QHBoxLayout()

        reset_btn = QPushButton("Reset View")
        reset_btn.clicked.connect(self.reset_camera)

        clear_btn = QPushButton("Clear Scene")
        clear_btn.clicked.connect(self.clear_scene)

        reload_ct_btn = QPushButton("Reload CT")
        reload_ct_btn.clicked.connect(self.load_ct_point_cloud)

        button_layout.addWidget(reset_btn)
        button_layout.addWidget(clear_btn)
        button_layout.addWidget(reload_ct_btn)
        button_layout.addStretch()

        control_layout.addLayout(button_layout)

        layout.addWidget(control_group)

        # PyVista widget
        self.plotter = QtInteractor(self)
        layout.addWidget(self.plotter.interactor)

    def setup_scene(self):
        # Set up the 3D scene
        self.plotter.set_background('black')

        # Add coordinate axes
        self.plotter.show_axes()

        # Add grid
        self.plotter.show_grid()

        # Initialize empty actors
        self.coarse_actor = None
        self.fine_actor = None
        self.tool_actor = None
        self.endoscope_actor = None

    def load_ct_point_cloud(self):
        """Load CT point cloud from configuration"""
        try:
            ct_path = self.config.get("CT_pc")
            if ct_path and os.path.exists(ct_path):
                self.ct_point_cloud = np.load(ct_path)
                print(f"Loaded CT point cloud with {len(self.ct_point_cloud)} points from {ct_path}")
                self.render_original_ct()
            else:
                print(f"CT point cloud file not found: {ct_path}")
                self.ct_point_cloud = None
        except Exception as e:
            print(f"Error loading CT point cloud: {str(e)}")
            self.ct_point_cloud = None

    def render_original_ct(self):
        """Render the original CT point cloud"""
        if self.ct_actor:
            self.plotter.remove_actor(self.ct_actor)

        if (self.ct_point_cloud is not None and
                self.show_original_ct_cb.isChecked()):
            # Create point cloud
            point_cloud = pv.PolyData(self.ct_point_cloud)

            opacity = self.ct_opacity_slider.value() / 100.0
            point_size = self.ct_point_size_spin.value()

            self.ct_actor = self.plotter.add_mesh(
                point_cloud,
                color='cyan',
                name='original_ct',
                point_size=point_size,
                render_points_as_spheres=True,
                opacity=opacity
            )

    def render_transformed_ct(self):
        """Render the transformed CT point cloud"""
        if self.transformed_ct_actor:
            self.plotter.remove_actor(self.transformed_ct_actor)

        if (self.ct_point_cloud is not None and
                self.show_transformed_ct_cb.isChecked() and
                (self.coarse_transformation is not None or self.fine_transformation is not None)):

            # Use fine transformation if available, otherwise coarse
            transformation = (self.fine_transformation if self.fine_transformation is not None
                              else self.coarse_transformation)

            if transformation is not None:
                # Transform CT point cloud
                transformed_points = self.transform_point_cloud(self.ct_point_cloud, transformation)

                # Create transformed point cloud
                transformed_cloud = pv.PolyData(transformed_points)

                opacity = self.ct_opacity_slider.value() / 100.0
                point_size = self.ct_point_size_spin.value()

                # Use different color for transformed CT
                color = 'yellow' if self.fine_transformation is not None else 'orange'

                self.transformed_ct_actor = self.plotter.add_mesh(
                    transformed_cloud,
                    color=color,
                    name='transformed_ct',
                    point_size=point_size,
                    render_points_as_spheres=True,
                    opacity=opacity
                )

    def transform_point_cloud(self, points, transformation_matrix):
        """Transform point cloud using transformation matrix"""
        # Convert to homogeneous coordinates
        ones = np.ones((points.shape[0], 1))
        homogeneous_points = np.hstack([points, ones])

        # Apply transformation
        transformed_homogeneous = (transformation_matrix @ homogeneous_points.T).T

        # Convert back to 3D coordinates
        transformed_points = transformed_homogeneous[:, :3]

        return transformed_points

    def update_coarse_points(self, points):
        """Update coarse registration points"""
        self.coarse_points = points
        self.render_coarse_points()

    def update_fine_points(self, points):
        """Update fine registration points"""
        self.fine_points = points
        self.render_fine_points()

    def update_coarse_transformation(self, transformation_matrix):
        """Update coarse transformation matrix"""
        self.coarse_transformation = np.array(transformation_matrix)
        print("Updated coarse transformation matrix")
        self.render_transformed_ct()

    def update_fine_transformation(self, transformation_matrix):
        """Update fine transformation matrix (combined transformation)"""
        self.fine_transformation = np.array(transformation_matrix)
        print("Updated fine transformation matrix")
        self.render_transformed_ct()

    def update_tool_position(self, tool_data):
        """Update tool positions from streaming data"""
        if 'transformed_matrix' in tool_data:
            matrix = np.array(tool_data['transformed_matrix'])
            self.tool_position = matrix[:3, 3]  # Extract position

        if 'endoscope_transformed_matrix' in tool_data:
            matrix = np.array(tool_data['endoscope_transformed_matrix'])
            self.endoscope_position = matrix[:3, 3]  # Extract position

        self.render_tools()

    def render_coarse_points(self):
        """Render coarse registration points"""
        if self.coarse_actor:
            self.plotter.remove_actor(self.coarse_actor)

        if self.coarse_points and self.show_coarse_cb.isChecked():
            # Convert points to numpy array
            points_array = np.array(self.coarse_points)

            # Create point cloud
            point_cloud = pv.PolyData(points_array)

            # Add spheres for better visibility
            spheres = point_cloud.glyph(scale=False, geom=pv.Sphere(radius=2))

            self.coarse_actor = self.plotter.add_mesh(
                spheres,
                color='red',
                name='coarse_points',
                render_points_as_spheres=True
            )

    def render_fine_points(self):
        """Render fine registration points"""
        if self.fine_actor:
            self.plotter.remove_actor(self.fine_actor)

        if self.fine_points and self.show_fine_cb.isChecked():
            # Convert points to numpy array
            points_array = np.array(self.fine_points)

            # Create point cloud
            point_cloud = pv.PolyData(points_array)

            self.fine_actor = self.plotter.add_mesh(
                point_cloud,
                color='blue',
                name='fine_points',
                point_size=3,
                render_points_as_spheres=True
            )

    def render_tools(self):
        """Render tool positions"""
        # Remove existing tool actors
        if self.tool_actor:
            self.plotter.remove_actor(self.tool_actor)
        if self.endoscope_actor:
            self.plotter.remove_actor(self.endoscope_actor)

        if not self.show_tools_cb.isChecked():
            return

        # Render probe tool
        if self.tool_position is not None:
            tool_sphere = pv.Sphere(radius=3, center=self.tool_position)
            self.tool_actor = self.plotter.add_mesh(
                tool_sphere,
                color='green',
                name='probe_tool'
            )

        # Render endoscope tool
        if self.endoscope_position is not None:
            endoscope_sphere = pv.Sphere(radius=3, center=self.endoscope_position)
            self.endoscope_actor = self.plotter.add_mesh(
                endoscope_sphere,
                color='yellow',
                name='endoscope_tool'
            )

    def update_visibility(self):
        """Update visibility of different elements"""
        self.render_coarse_points()
        self.render_fine_points()
        self.render_tools()

    def update_ct_visibility(self):
        """Update CT point cloud visibility"""
        self.render_original_ct()
        self.render_transformed_ct()

    def update_ct_opacity(self):
        """Update CT point cloud opacity"""
        self.render_original_ct()
        self.render_transformed_ct()

    def update_ct_point_size(self):
        """Update CT point cloud point size"""
        self.render_original_ct()
        self.render_transformed_ct()

    def reset_camera(self):
        """Reset camera to default view"""
        self.plotter.reset_camera()

    def clear_scene(self):
        """Clear all objects from scene"""
        self.plotter.clear()
        self.setup_scene()

        # Re-render everything
        self.load_ct_point_cloud()
        self.render_coarse_points()
        self.render_fine_points()
        self.render_tools()
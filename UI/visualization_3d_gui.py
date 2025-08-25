import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from PyQt5.QtCore import QObject, pyqtSignal
import time


class Visualization3D(QObject):
    def __init__(self):
        super().__init__()
        self.setup_plotter()
        self.setup_actors()
        self.setup_performance_optimization()

    def setup_plotter(self):
        self.plotter = QtInteractor()
        self.plotter.background_color = 'white'
        self.plotter.enable_anti_aliasing = False

        # Set render window properties
        render_window = self.plotter.render_window
        render_window.SetMultiSamples(0)

        # Add coordinate axes
        self.plotter.add_axes(viewport=(0, 0, 0.2, 0.2))

    def setup_actors(self):
        # Storage for different types of actors
        self.coarse_actors = []
        self.coarse_line_actors = []
        self.fine_actors = []
        self.ct_actor = None
        self.probe_actor = None
        self.trail_actor = None
        self.probe_trail = []

    def setup_performance_optimization(self):
        self.render_pending = False
        self.last_render_time = 0
        self.min_render_interval = 0.033  # ~30 FPS max

    def get_widget(self):
        return self.plotter.interactor

    def request_render(self):
        """Request a render with throttling"""
        current_time = time.time()
        if current_time - self.last_render_time >= self.min_render_interval:
            try:
                self.plotter.render()
                self.last_render_time = current_time
            except Exception as e:
                print(f"Render error: {e}")

    # Coarse Registration Visualization
    def update_coarse_points(self, coarse_points_dict, show_points=True, show_matches=True):
        """Update coarse point visualization"""
        # Clear existing coarse actors
        self.clear_coarse_actors()

        if not show_points or not coarse_points_dict:
            self.request_render()
            return

        # Collect points for visualization
        unity_points = []
        ndi_points = []
        point_numbers = []

        for point_num, point_data in coarse_points_dict.items():
            unity_points.append(point_data['unity_point'])
            ndi_points.append(point_data['ndi_point'])
            point_numbers.append(point_num)

        unity_points = np.array(unity_points)
        ndi_points = np.array(ndi_points)

        try:
            # Create Unity spheres
            if len(unity_points) > 0:
                unity_cloud = pv.PolyData(unity_points)
                unity_spheres = unity_cloud.glyph(scale=False, geom=pv.Sphere(radius=3.0))
                actor = self.plotter.add_mesh(
                    unity_spheres, color='blue', name='unity_points',
                    opacity=0.8, render=False
                )
                self.coarse_actors.append(actor)

                # Add labels
                for point, point_num in zip(unity_points, point_numbers):
                    try:
                        self.plotter.add_point_labels(
                            [point], [f'U{point_num}'], point_size=0,
                            font_size=10, text_color='blue',
                            name=f'unity_label_{point_num}', render=False
                        )
                    except Exception as e:
                        print(f"Error adding Unity label {point_num}: {e}")

            # Create NDI spheres
            if len(ndi_points) > 0:
                ndi_cloud = pv.PolyData(ndi_points)
                ndi_spheres = ndi_cloud.glyph(scale=False, geom=pv.Sphere(radius=3.0))
                actor = self.plotter.add_mesh(
                    ndi_spheres, color='red', name='ndi_points',
                    opacity=0.8, render=False
                )
                self.coarse_actors.append(actor)

                # Add labels
                for point, point_num in zip(ndi_points, point_numbers):
                    try:
                        self.plotter.add_point_labels(
                            [point], [f'N{point_num}'], point_size=0,
                            font_size=10, text_color='red',
                            name=f'ndi_label_{point_num}', render=False
                        )
                    except Exception as e:
                        print(f"Error adding NDI label {point_num}: {e}")

            # Create connection lines
            if show_matches and len(unity_points) > 0:
                all_line_points = []
                all_lines = []

                for i in range(len(unity_points)):
                    all_line_points.extend([unity_points[i], ndi_points[i]])
                    all_lines.extend([2, len(all_line_points) - 2, len(all_line_points) - 1])

                if all_line_points:
                    lines_polydata = pv.PolyData(np.array(all_line_points))
                    lines_polydata.lines = np.array(all_lines)

                    actor = self.plotter.add_mesh(
                        lines_polydata, color='green', line_width=2,
                        name='match_lines', opacity=0.7, render=False
                    )
                    self.coarse_line_actors.append(actor)

        except Exception as e:
            print(f"Error in coarse visualization: {e}")

        self.request_render()

    def clear_coarse_actors(self):
        """Clear all coarse registration actors"""
        all_actors = self.coarse_actors + self.coarse_line_actors
        for actor in all_actors:
            try:
                self.plotter.remove_actor(actor, render=False)
            except:
                pass
        self.coarse_actors.clear()
        self.coarse_line_actors.clear()

    # CT Point Cloud Visualization
    def load_ct_pointcloud(self, ct_points, max_points=5000):
        """Load and display CT point cloud"""
        try:
            # Always downsample for performance
            if len(ct_points) > max_points:
                np.random.seed(42)  # Reproducible sampling
                indices = np.random.choice(len(ct_points), max_points, replace=False)
                downsampled_points = ct_points[indices].copy()
            else:
                downsampled_points = ct_points.copy()

            # Store the point cloud
            self.ct_pointcloud = pv.PolyData(downsampled_points)
            print(f"CT point cloud loaded: {len(downsampled_points)} points")

        except Exception as e:
            print(f"Error loading CT point cloud: {e}")

    def show_ct_pointcloud(self, show=True):
        """Show/hide CT point cloud"""
        # Remove existing CT actor
        if self.ct_actor:
            try:
                self.plotter.remove_actor(self.ct_actor, render=False)
            except:
                pass
            self.ct_actor = None

        if show and hasattr(self, 'ct_pointcloud') and self.ct_pointcloud is not None:
            try:
                self.ct_actor = self.plotter.add_mesh(
                    self.ct_pointcloud, style='points', color='lightgray',
                    point_size=3.0, opacity=0.7, name='ct_pointcloud', render=False
                )
                print(f"CT point cloud displayed: {self.ct_pointcloud.n_points} points")
            except Exception as e:
                print(f"Error displaying CT point cloud: {e}")

        self.request_render()

    # Fine Registration Visualization
    def update_fine_points(self, fine_points, show=True):
        """Update fine points visualization"""
        # Clear existing fine actors
        for actor in self.fine_actors:
            try:
                self.plotter.remove_actor(actor, render=False)
            except:
                pass
        self.fine_actors.clear()

        if show and fine_points and len(fine_points) > 0:
            try:
                points_array = np.array(fine_points)
                point_cloud = pv.PolyData(points_array)
                actor = self.plotter.add_mesh(
                    point_cloud, style='points', color='orange',
                    point_size=2.0, opacity=0.8, name='fine_points', render=False
                )
                self.fine_actors.append(actor)
            except Exception as e:
                print(f"Error updating fine points: {e}")

        self.request_render()

    # Streaming Visualization
    def update_probe_position(self, position):
        """Update probe position"""
        if self.probe_actor:
            try:
                self.plotter.remove_actor(self.probe_actor, render=False)
            except:
                pass

        try:
            sphere = pv.Sphere(radius=2.0, center=position)
            self.probe_actor = self.plotter.add_mesh(
                sphere, color='red', name='probe_position', render=False
            )
            self.request_render()
        except Exception as e:
            print(f"Error updating probe position: {e}")

    def add_to_probe_trail(self, position):
        """Add position to probe trail"""
        self.probe_trail.append(position)

        # Limit trail length
        if len(self.probe_trail) > 500:
            self.probe_trail.pop(0)

        # Update trail visualization less frequently
        if len(self.probe_trail) > 1 and len(self.probe_trail) % 5 == 0:
            try:
                trail_points = np.array(self.probe_trail)
                trail_line = pv.PolyData(trail_points)

                if self.trail_actor:
                    try:
                        self.plotter.remove_actor(self.trail_actor, render=False)
                    except:
                        pass

                self.trail_actor = self.plotter.add_mesh(
                    trail_line, color='blue', line_width=2,
                    name='probe_trail', render=False
                )
                self.request_render()
            except Exception as e:
                print(f"Error updating probe trail: {e}")

    def clear_probe_trail(self):
        """Clear probe trail"""
        self.probe_trail.clear()
        if self.trail_actor:
            try:
                self.plotter.remove_actor(self.trail_actor, render=False)
            except:
                pass
        self.request_render()

    def reset_camera(self):
        """Reset camera view"""
        try:
            self.plotter.reset_camera()
            self.request_render()
        except Exception as e:
            print(f"Error resetting camera: {e}")
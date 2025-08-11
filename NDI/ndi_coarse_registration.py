import numpy as np
import os
import re
import logging
from typing import Dict, List, Tuple, Optional, Any
import open3d as o3d
import threading

# Setup logging
from Utils.Utils import save_state

logger = logging.getLogger(__name__)


class CoarseRegistration:
    def __init__(self, tip_vector):
        # Coarse points storage
        self.coarse_points = {
            "unity_points": [],
            "ndi_points": [],
            "point_numbers": []
        }

        # Original to zero-indexed mapping
        self.original_to_zero_indexed = {}

        # Transformation matrix from coarse registration
        self.transformation_matrix = None

        # Parsed matrices from coarse.txt
        self.matrices = {}

        # Probe tip position in probe's local coordinates
        self.tip_vector = tip_vector

        # Parse coarse file on initialization
        self.parse_coarse_file()

    def parse_coarse_file(self) -> Dict[int, np.ndarray]:
        """
        Parse the coarse.txt file to extract matrices for each point

        Returns:
            Dict mapping point numbers (zero-indexed) to their transformation matrices
        """
        matrices_by_point = {}
        original_point_numbers = []

        try:
            if not os.path.exists("coarse.txt"):
                logger.error("coarse.txt file not found!")
                return {}

            with open("coarse.txt", "r") as file:
                content = file.read()

            logger.info(f"Loaded coarse.txt, size: {len(content)} bytes")

            # Extract data for each point (d1 through d7)
            point_pattern = r'd(\d+):\n\(\[.+?\], \[.+?\], \[.+?\], \[array\(\[\[(.*?)\],\s*\[(.*?)\],\s*\[(.*?)\],\s*\[(.*?)\]\]\)'
            points = re.findall(point_pattern, content, re.DOTALL)

            logger.info(f"Found {len(points)} point entries in coarse.txt")

            # First collect all original point numbers
            for point_match in points:
                original_point_num = int(point_match[0])
                original_point_numbers.append(original_point_num)

            # Sort original point numbers to ensure consistent zero-indexing
            original_point_numbers.sort()

            # Create mapping from original to zero-indexed
            for i, orig_num in enumerate(original_point_numbers):
                self.original_to_zero_indexed[orig_num] = i

            logger.info(f"Mapped original point numbers to zero-indexed: {self.original_to_zero_indexed}")

            # Now process the matrices with zero-indexed point numbers
            for point_match in points:
                original_point_num = int(point_match[0])
                point_num = self.original_to_zero_indexed[original_point_num]  # Zero-indexed point number

                # Extract the matrix (tool transformation)
                matrix = np.zeros((4, 4))

                try:
                    for row_idx in range(4):
                        row_str = point_match[row_idx + 1]
                        # Clean and parse the row values
                        row_values = re.findall(r'[-+]?(?:\d*\.\d+|\d+)e?[-+]?\d*', row_str)
                        if len(row_values) >= 4:
                            matrix[row_idx, :] = [float(v) for v in row_values[:4]]
                except Exception as e:
                    logger.error(
                        f"Error parsing matrix for point {point_num} (original: {original_point_num}): {str(e)}")
                    continue

                matrices_by_point[point_num] = matrix

            logger.info(f"Successfully loaded {len(matrices_by_point)} points with matrices")
            logger.info(f"Available zero-indexed point numbers: {sorted(matrices_by_point.keys())}")

            # Store matrices for later use
            self.matrices = matrices_by_point
            return matrices_by_point
        except Exception as e:
            logger.error(f"Error parsing coarse file: {str(e)}")
            return {}

    def reset_coarse_points(self) -> Dict[str, str]:
        """
        Reset all stored coarse points

        Returns:
            Status information
        """
        self.coarse_points = {
            "unity_points": [],
            "ndi_points": [],
            "point_numbers": []
        }
        self.transformation_matrix = None
        return {"status": "success", "message": "All coarse points have been reset"}

    def get_available_points(self) -> Dict[str, Any]:
        """
        Get available point numbers from coarse data

        Returns:
            Information about available points
        """
        return {
            "available_point_numbers": sorted(list(self.matrices.keys())),
            "point_count": len(self.matrices),
            "note": "Point numbers are zero-indexed"
        }

    def debug_coarse_data(self) -> Dict[str, Any]:
        """
        Debug information about coarse data loading

        Returns:
            Detailed debug information
        """
        return {
            "coarse_matrices_loaded": len(self.matrices),
            "available_point_numbers": sorted(list(self.matrices.keys())),
            "original_to_zero_mapping": self.original_to_zero_indexed,
            "coarse_file_exists": os.path.exists("../coarse.txt"),
            "coarse_file_size": os.path.getsize("../coarse.txt") if os.path.exists("../coarse.txt") else None,
            "original_point_numbers_found": list(self.original_to_zero_indexed.keys()),
            "probe_tip_vector": self.tip_vector.tolist()
        }

    def set_coarse_point(self, unity_point: List[float], point_number: int, tool_matrix: np.ndarray) -> Dict[str, Any]:
        """
        Set a coarse registration point

        Args:
            unity_point: 3D point in Unity coordinates
            point_number: Zero-indexed point number
            tool_matrix: 4x4 transformation matrix for the tool

        Returns:
            Information about the set point
        """

        if tool_matrix is None:
            return {
                "status": "error",
                "message": "could not detect probe!"
            }
        # Convert to numpy array
        unity_point_np = np.array(unity_point)

        # Calculate the probe tip position in world coordinates using fixed tip vector
        ndi_point = np.dot(tool_matrix, self.tip_vector)[:3]

        # Store the points
        self.coarse_points["unity_points"].append(unity_point_np)
        self.coarse_points["ndi_points"].append(ndi_point)
        self.coarse_points["point_numbers"].append(point_number)

        return {
            "status": "success",
            "tool_matrix": tool_matrix.tolist(),
            "ndi_point": ndi_point.tolist(),
            "unity_point": unity_point_np.tolist(),
            "point_number": point_number,
            "tip_vector_used": self.tip_vector,
            "message": "success"
        }

    def get_coarse_points(self) -> Dict[str, Any]:
        """
        Get all stored coarse registration points

        Returns:
            All stored coarse points
        """
        return {
            "unity_points": [p.tolist() for p in self.coarse_points["unity_points"]],
            "ndi_points": [p.tolist() for p in self.coarse_points["ndi_points"]],
            "point_numbers": self.coarse_points["point_numbers"],
            "num_points": len(self.coarse_points["point_numbers"])
        }

    def perform_coarse_registration(self, visualize: bool = False) -> Dict[str, Any]:
        """
        Perform coarse registration using SVD

        Args:
            visualize: Whether to visualize the registration result

        Returns:
            Registration results
        """
        if len(self.coarse_points["unity_points"]) < 3:
            return {
                "status": "error",
                "message": f"At least 3 points needed for registration. Only {len(self.coarse_points['unity_points'])} points available."
            }

        # Convert to numpy arrays
        unity_points = np.array(self.coarse_points["unity_points"])
        ndi_points = np.array(self.coarse_points["ndi_points"])

        # Calculate centroids
        unity_centroid = np.mean(unity_points, axis=0)
        ndi_centroid = np.mean(ndi_points, axis=0)

        # Center the points
        unity_centered = unity_points - unity_centroid
        ndi_centered = ndi_points - ndi_centroid

        # Calculate the correlation matrix
        H = np.dot(unity_centered.T, ndi_centered)

        try :
            # SVD decomposition
            U, S, Vt = np.linalg.svd(H)
        except Exception as e:
            return {
                "status": "error",
                "message": f"SVD failed"
            }
        # Calculate rotation matrix
        R = np.dot(Vt.T, U.T)

        # Ensure proper rotation (no reflection)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = np.dot(Vt.T, U.T)

        # Calculate translation
        t = ndi_centroid - np.dot(R, unity_centroid)

        # Create 4x4 transformation matrix
        transform = np.eye(4)
        transform[:3, :3] = R
        transform[:3, 3] = t

        # Calculate registration error
        transformed_unity = np.zeros_like(unity_points)
        for i in range(len(unity_points)):
            transformed_unity[i] = np.dot(R, unity_points[i]) + t

        mse = np.mean(np.sum((transformed_unity - ndi_points) ** 2, axis=1))
        rmse = np.sqrt(mse)

        # Store the transformation matrix
        self.transformation_matrix = transform
        save_state("saved_state.json", {"coarse_transform":transform})
        logger.info(f"Coarse registration completed with RMSE: {rmse}")

        # Visualization info for response
        vis_info = {
            "status": "not_requested",
            "message": "No visualization requested"
        }

        # Visualize if requested
        if visualize:
            self._visualize_registration(unity_points, ndi_points, transform)
            vis_info = {
                "status": "launched",
                "message": "Open3D visualization opened on server"
            }

        return {
            "status":"success",
            "transformation_matrix": transform.tolist(),
            "rmse": float(rmse),
            "num_points_used": len(unity_points),
            "point_errors": [float(np.linalg.norm(transformed_unity[i] - ndi_points[i])) for i in
                             range(len(unity_points))],
            "visualization": vis_info
        }

    def _visualize_registration(self, unity_points: np.ndarray, ndi_points: np.ndarray, transform: np.ndarray) -> None:
        """
        Visualize the registration result using Open3D

        Args:
            unity_points: Points in Unity coordinates
            ndi_points: Points in NDI coordinates
            transform: Transformation matrix from Unity to NDI
        """
        # Create Open3D point clouds for visualization
        unity_pcd = o3d.geometry.PointCloud()
        unity_pcd.points = o3d.utility.Vector3dVector(unity_points)
        unity_pcd.paint_uniform_color([1, 0, 0])  # Red for Unity points

        # Apply transformation to Unity points
        unity_pcd.transform(transform)

        ndi_pcd = o3d.geometry.PointCloud()
        ndi_pcd.points = o3d.utility.Vector3dVector(ndi_points)
        ndi_pcd.paint_uniform_color([0, 1, 0])  # Green for NDI points

        # Create a coordinate frame
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=20)

        # Launch visualization in a non-blocking way
        def open_visualization():
            logger.info("Opening coarse registration visualization window with Open3D...")
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name="Coarse Registration Visualization")

            # Add geometries to the visualizer
            vis.add_geometry(unity_pcd)
            vis.add_geometry(ndi_pcd)
            vis.add_geometry(coord_frame)

            # Make points larger for better visibility (coarse points are few in number)
            opt = vis.get_render_option()
            opt.point_size = 5.0  # Larger point size
            opt.background_color = np.array([0.1, 0.1, 0.1])  # Dark gray background

            # Set view
            vis.get_view_control().set_lookat([0, 0, 0])
            vis.get_view_control().set_front([0, -1, 0])
            vis.get_view_control().set_up([0, 0, 1])
            vis.get_view_control().set_zoom(0.8)

            vis.run()
            vis.destroy_window()

        # Start visualization in a separate thread
        vis_thread = threading.Thread(target=open_visualization)
        vis_thread.daemon = True
        vis_thread.start()

        logger.info("Visualization thread started")

    def get_transformation_matrix(self) -> Optional[np.ndarray]:
        """
        Get the current coarse transformation matrix

        Returns:
            The coarse transformation matrix or None if not available
        """
        return self.transformation_matrix
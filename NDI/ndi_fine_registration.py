import copy
import json

import numpy as np
import os
import logging
import threading
import time
from typing import List, Dict, Any, Optional, Tuple
import open3d as o3d
from threading import Event

from Utils.Utils import save_state

logger = logging.getLogger(__name__)



class FineRegistration:
    def __init__(self, config):
        # Fine points storage
        self.fine_points = []
        self.config = config

        # State variables
        self.gathering_active = False
        self.gathering_thread = None
        self.gathering_stop_event = Event()
        self.gathering_frequency = 10  # Hz

        # Fine matrices from file (for simulation)
        self.fine_matrices = []
        self.streaming_position = 0  # For cycling through fine.txt data

        # Transformation matrices
        self.fine_transformation_matrix = None

        # Load matrices from fine.txt
        self.parse_fine_file()

        self.combined_transformation = None

    def parse_fine_file(self) -> List[np.ndarray]:
        """
        Parse the fine.txt file to extract matrices for simulation

        Returns:
            List of 4x4 transformation matrices
        """
        matrices = []

        try:
            if not os.path.exists("NDI\\fine.txt"):
                logger.error("fine.txt file not found!")
                return []

            with open("NDI\\fine.txt", "r") as file:
                content = file.read()

            logger.info(f"Loaded fine.txt, size: {len(content)} bytes")

            # Look for 4x4 matrices in the file
            import re
            pattern = r'array\(\[\[(.*?)\],\s*\[(.*?)\],\s*\[(.*?)\],\s*\[(.*?)\]\]\)'
            matrix_matches = re.findall(pattern, content)

            logger.info(f"Found {len(matrix_matches)} potential matrices in fine.txt")

            for match in matrix_matches:
                matrix = np.zeros((4, 4))
                try:
                    for row_idx, row_str in enumerate(match):
                        # Parse row values
                        values = [float(x) for x in re.findall(r'[-+]?(?:\d*\.\d+|\d+)e?[-+]?\d*', row_str)]
                        if len(values) >= 4:
                            matrix[row_idx, :4] = values[:4]

                    # Only include valid matrices (no NaN values)
                    if not np.isnan(matrix).any() and not np.all(matrix == 0):
                        matrices.append(matrix)
                except Exception as e:
                    logger.error(f"Error parsing fine matrix: {str(e)}")
                    continue

            logger.info(f"Successfully loaded {len(matrices)} valid matrices from fine.txt")
            self.fine_matrices = matrices
            return matrices
        except Exception as e:
            logger.error(f"Error parsing fine file: {str(e)}")
            return []

    def start_fine_gather(self, frequency, tip_vector, ndi_tracking, is_local) -> Dict[str, Any]:
        """
        Start continuously gathering fine registration points

        Args:
            frequency: How many points to collect per second

        Returns:
            Status information about the gathering process
        """
        # Check if already active
        if self.gathering_active:
            return {
                "status": "already_active",
                "message": "Fine point gathering is already active",
                "total_points": len(self.fine_points)
            }

        # Reset points and prepare for gathering
        # self.fine_points = []
        self.gathering_frequency = frequency
        self.gathering_stop_event.clear()

        # Start gathering thread
        self.gathering_thread = threading.Thread(
            target=self._gathering_thread_function,
            args=(self.gathering_stop_event, self.gathering_frequency, tip_vector, is_local, ndi_tracking)
        )
        self.gathering_thread.daemon = True
        self.gathering_thread.start()

        self.gathering_active = True

        logger.info(f"Started fine point gathering at {frequency} Hz")

        return {
            "status": "success",
            "message": f"Started gathering fine registration points at {frequency} Hz"
        }

    def reset_fine_gather(self):
        self.fine_points=[]

    def end_fine_gather(self) -> Dict[str, Any]:
        """
        Stop gathering fine registration points

        Returns:
            Information about the gathered points
        """
        if not self.gathering_active:
            return {"status": "warning", "message": "Fine gathering was not active"}

        # Signal the thread to stop
        self.gathering_stop_event.set()

        # Wait for thread to finish (with timeout)
        if self.gathering_thread and self.gathering_thread.is_alive():
            self.gathering_thread.join(timeout=2.0)

        self.gathering_active = False

        logger.info(f"Ended fine point gathering, collected {len(self.fine_points)} points")

        return {
            "status": "success",
            "message": f"Stopped gathering fine registration points",
            "total_points_collected": len(self.fine_points)
        }

    def get_fine_points_status(self) -> Dict[str, Any]:
        """
        Get the current status of fine point gathering

        Returns:
            Status information
        """
        return {
            "active": self.gathering_active,
            "num_points_collected": len(self.fine_points)
        }

    def simulate_fine_gather(self, num_points: int = 100, replace_existing: bool = False,
                             downsample_factor: float = 1.0, tip_vector: np.ndarray = None,
                             is_local: bool = True, ndi_tracker=None) -> Dict[str, Any]:
        """
        Gather points from either local files or the NDI tracker

        Args:
            num_points: Maximum number of points to gather (for local mode only)
            replace_existing: If True, replace existing points; if False, append
            downsample_factor: Factor to downsample points (for local mode only)
            tip_vector: Tool tip vector for transformations
            is_local: Whether to use local files or NDI tracker
            ndi_tracker: Reference to NDI tracker object (for real tracking)

        Returns:
            Information about the gathered points
        """
        # if not self.gathering_active:
        #     return {
        #         "status": "error",
        #         "message": "Fine gathering is not active"
        #     }

        # Reset points if requested
        if replace_existing:
            self.fine_points = []
            logger.info("Cleared existing points")

        new_points = []

        # Make sure we have a tip vector
        if tip_vector is None:
            tip_vector = np.array([0, 0, 0, 1])  # Default, will be replaced by actual value

        if is_local:
            # Simulation mode using local files
            if len(self.fine_matrices) == 0:
                return {
                    "status": "error",
                    "message": "No fine matrices available for simulation"
                }

            # Calculate how many points to gather
            available_points = len(self.fine_matrices)
            requested_points = min(num_points, available_points)

            logger.info(f"Simulating gathering of {requested_points} points (out of {available_points} available)")

            # For large datasets, implement downsampling
            if downsample_factor < 1.0 and downsample_factor > 0:
                step = int(1 / downsample_factor)
                selected_matrices = self.fine_matrices[:requested_points:step]
                logger.info(f"Downsampling by factor {downsample_factor} (step={step})")
            else:
                selected_matrices = self.fine_matrices[:requested_points]

            # Calculate probe tip positions
            total = len(selected_matrices)
            batch_size = 1000
            for i in range(0, total, batch_size):
                batch = selected_matrices[i:i + batch_size]
                for matrix in batch:
                    tip_position = np.dot(matrix, tip_vector)[:3]
                    if not np.isnan(tip_position).any():
                        new_points.append(tip_position)

                if i % 5000 == 0 and i > 0:
                    logger.info(f"Processed {i}/{total} matrices...")
        else:
            # Real NDI tracking mode
            if ndi_tracker is None:
                return {
                    "status": "error",
                    "message": "NDI tracker not provided"
                }

            try:
                # Get a frame from the NDI tracker
                port_handles, timestamps, framenumbers, tracking, quality = ndi_tracker.GetPosition()

                # The first matrix is for the probe
                if tracking and len(tracking) > 0:
                    probe_matrix = tracking[self.config["tool_types"]["probe"]]

                    # Calculate the probe tip position
                    tip_position = np.dot(probe_matrix, tip_vector)[:3]

                    if not np.isnan(tip_position).any():
                        new_points.append(tip_position)
                        logger.info(f"Gathered 1 real point from NDI tracker")
                    else:
                        logger.warning("Got invalid position from NDI tracker (NaN values)")
                else:
                    logger.warning("No tracking data received from NDI tracker")

            except Exception as e:
                logger.exception("Error getting data from NDI tracker")
                return {
                    "status": "error",
                    "message": f"Error getting data from NDI tracker: {str(e)}"
                }

        self.fine_points.extend(new_points)

        logger.info(f"Added {len(new_points)} points, total fine points: {len(self.fine_points)}")

        # Warning for very large point clouds
        point_count_warning = ""
        if len(self.fine_points) > 10000:
            point_count_warning = "Large point count may affect performance. Consider downsampling for visualization."

        return {
            "status": "success",
            "num_points_added": len(new_points),
            "total_points": len(self.fine_points),
            "warning": point_count_warning
        }

    def perform_fine_registration(self, id: int, coarse_transformation_matrix: np.ndarray,
                                  downsample_factor: float = 1.0, visualize: bool = False) -> Dict[str, Any]:
        """
        Perform fine registration using ICP algorithm

        Args:
            id: ID of the CT point cloud to register against
            coarse_transformation_matrix: Initial transformation from coarse registration
            downsample_factor: Factor to downsample points for visualization
            visualize: Whether to visualize the results

        Returns:
            Registration results
        """
        if len(self.fine_points) < 10:
            return {
                "status": "error",
                "message": f"At least 10 points needed for fine registration. Only {len(self.fine_points)} points available."
            }

        if coarse_transformation_matrix is None:
            return {
                "status": "error",
                "message": "Coarse registration must be performed before fine registration."
            }

        # Use the path for CT point cloud
        ct_pc_path = self.config["CT_PC_address"]

        # Check if CT point cloud exists
        if not os.path.exists(ct_pc_path):
            logger.error(f"CT point cloud file not found: {ct_pc_path}")
            return {
                "status": "error",
                "message": f"CT point cloud not found at {ct_pc_path}."
            }

        try:
            # Load CT point cloud
            ct_points = np.load(ct_pc_path)
            logger.info(f"Loaded CT point cloud with {len(ct_points)} points")

            # Create Open3D point clouds
            # CT points are the SOURCE (they need to be moved to align with NDI points)
            source = o3d.geometry.PointCloud()
            source.points = o3d.utility.Vector3dVector(ct_points)

            # NDI points are the TARGET (they represent where we want the CT to align)
            target = o3d.geometry.PointCloud()
            target.points = o3d.utility.Vector3dVector(np.array(self.fine_points))

            # We use the coarse transformation to initially align CT points with NDI
            initial_transform = coarse_transformation_matrix

            # Apply initial transformation to CT points
            logger.info("Applying coarse transformation to CT points before ICP refinement")
            source.transform(initial_transform)

            # Run ICP to refine the alignment
            threshold = 25.0  # Distance threshold for ICP

            # Initial guess is identity since we already applied the coarse transformation
            trans_init = np.eye(4)
            source_copy = copy.deepcopy(source)
            target_copy = copy.deepcopy(target)
            target_copy.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=13))
            source_copy.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=13))
            reg_p2p = o3d.pipelines.registration.registration_icp(
                target_copy,source_copy , threshold, trans_init,
                o3d.pipelines.registration.TransformationEstimationPointToPlane()
            )

            # Store the fine registration transformation
            self.fine_transformation_matrix = np.linalg.inv(reg_p2p.transformation)
            # self.fine_transformation_matrix = reg_p2p.transformation

            # Calculate the combined transformation (fine × coarse)
            self.combined_transformation = self.fine_transformation_matrix @ initial_transform

            save_state("saved_state.json", {"combined_transform":self.combined_transformation})

            logger.info(
                f"Fine registration (ICP) completed with fitness: {reg_p2p.fitness}, RMSE: {reg_p2p.inlier_rmse}")

            # Visualization info for response
            vis_info = {
                "status": "not_requested",
                "message": "No visualization requested"
            }

            # Only do visualization if requested
            if visualize:
                self._visualize_registration(ct_points, np.array(self.fine_points),
                                             self.combined_transformation, id, downsample_factor)

                vis_info = {
                    "status": "launched",
                    "message": "Open3D visualization opened on server",
                }

            # Prepare result
            result = {
                "status": "success",
                "coarse_transformation": initial_transform.tolist(),
                "fine_transformation": self.fine_transformation_matrix.tolist(),
                "combined_transformation": self.combined_transformation.tolist(),
                "fitness": reg_p2p.fitness,
                "inlier_rmse": reg_p2p.inlier_rmse,
                "num_points_used": len(self.fine_points),
                "visualization": vis_info
            }

            return result

        except Exception as e:
            logger.exception(f"Error during fine registration or visualization")
            return {
                "status": "error",
                "message": f"Error during fine registration: {str(e)}"
            }

    def _gathering_thread_function(self, stop_event: Event, frequency: float = 20,
                                   tip_vector: np.ndarray = None, is_local: bool = False,
                                   ndi_tracker=None) -> None:
        """
        Thread function to continuously gather fine registration points

        Args:
            stop_event: Event to signal thread to stop
            frequency: How many points to collect per second
            tip_vector: Tool tip vector for transformations
            is_local: Whether to use local files or NDI tracker
            ndi_tracker: Reference to NDI tracker object
        """
        logger.info(f"Starting fine point gathering at {frequency} Hz")

        # Make sure we have a tip vector
        if tip_vector is None:
            tip_vector = np.array([0, 0, 0, 1])  # Default, will be replaced by actual value

        # Calculate sleep time based on frequency
        sleep_time = 1.0 / frequency

        # Counter for statistics
        points_gathered = 0
        last_log_time = time.time()

        while not stop_event.is_set():
            start_time = time.time()

            # Use real NDI tracker
            if ndi_tracker is None:
                logger.error("NDI tracker not provided")
                break

            try:
                # Get tracking data
                tracking = ndi_tracker.GetPosition()

                if tracking and len(tracking) > 0:
                    # Get probe matrix
                    probe_matrix = tracking[self.config["tool_types"]["probe"]]

                    # Calculate probe tip position
                    tip_position = np.dot(probe_matrix, tip_vector)[:3]

                    # Add to fine points if valid
                    if not np.isnan(tip_position).any():
                        self.fine_points.append(tip_position)
                        points_gathered += 1

            except Exception as e:
                logger.warning(f"Error getting tracking data: {str(e)}")

            # Log statistics periodically
            current_time = time.time()
            if current_time - last_log_time >= 5.0:  # Log every 5 seconds
                pts_per_sec = points_gathered / (current_time - last_log_time)
                logger.info(f"Fine gathering at {pts_per_sec:.2f} points/sec, total: {len(self.fine_points)}")
                points_gathered = 0
                last_log_time = current_time

            # Sleep to maintain desired frequency
            elapsed = time.time() - start_time
            if elapsed < sleep_time:
                time.sleep(sleep_time - elapsed)

        logger.info(f"Fine gathering thread stopped. Total points collected: {len(self.fine_points)}")

    def _visualize_registration(self, ct_points: np.ndarray, ndi_points: np.ndarray,
                                combined_transformation: np.ndarray, id: int,
                                downsample_factor: float = 1.0) -> None:
        """
        Visualize the registration results using Open3D

        Args:
            ct_points: CT point cloud data
            ndi_points: NDI point cloud data
            combined_transformation: Combined transformation matrix
            id: ID for the visualization window
            downsample_factor: Factor to downsample points for visualization
        """
        # Prepare visualization
        source_pcd = o3d.geometry.PointCloud()
        source_pcd.points = o3d.utility.Vector3dVector(ct_points)
        source_pcd.paint_uniform_color([0, 0, 1])  # Blue for CT points (source)

        # Apply combined transformation to CT points to align with NDI
        source_pcd.transform(combined_transformation)

        target_pcd = o3d.geometry.PointCloud()
        target_pcd.points = o3d.utility.Vector3dVector(ndi_points)
        target_pcd.paint_uniform_color([1, 0, 0])  # Red for NDI points (target)

        # Downsample if necessary for visualization
        if downsample_factor < 1.0 and downsample_factor > 0:
            logger.info(f"Downsampling point clouds by factor {downsample_factor} for visualization")
            voxel_size = max(0.1, 0.5 / downsample_factor)
            source_vis = source_pcd.voxel_down_sample(voxel_size=voxel_size)
            target_vis = target_pcd.voxel_down_sample(voxel_size=voxel_size)
            logger.info(f"Downsampled source (CT): {len(source_pcd.points)} → {len(source_vis.points)}")
            logger.info(f"Downsampled target (NDI): {len(target_pcd.points)} → {len(target_vis.points)}")
        else:
            source_vis = source_pcd
            target_vis = target_pcd

        # Add coordinate frame for reference
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=20)

        # Launch visualization in a non-blocking way
        def open_visualization():
            logger.info("Opening visualization window with Open3D...")
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name=f"Registration Visualization - ID: {id}")
            vis.add_geometry(source_vis)
            vis.add_geometry(target_vis)
            vis.add_geometry(coord_frame)

            # Set rendering options
            opt = vis.get_render_option()
            opt.point_size = 2.0
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

    def get_fine_transformation_matrix(self) -> Optional[np.ndarray]:
        """
        Get the fine transformation matrix

        Returns:
            The fine transformation matrix or None if not available
        """
        return self.fine_transformation_matrix

    def get_fine_points(self) -> List[np.ndarray]:
        """
        Get the collected fine points

        Returns:
            List of fine points as 3D vectors
        """
        return self.fine_points


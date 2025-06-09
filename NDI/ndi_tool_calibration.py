import numpy as np
import logging
import os
import time
import threading
from typing import List, Dict, Any, Optional, Tuple
from io import BytesIO
import base64
import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend for server
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Setup logging
logger = logging.getLogger(__name__)


class ToolCalibration:
    def __init__(self):
        # Tool calibration variables
        self.calibration_active = False
        self.calibration_matrices = []
        self.calibration_result = None  # Store calibration results

        # Threading variables for automatic data collection
        self.calibration_thread = None
        self.stop_calibration_event = threading.Event()
        self.calibration_frequency = 10  # Hz - data collection frequency

        # Thread lock for safe matrix list access
        self.matrix_lock = threading.Lock()

        # For tracking reference transformations
        self.use_reference = True  # Whether to use reference coordinate system

        # Create visualization directory if it doesn't exist
        self.vis_dir = "tool_calibration_visualizations"
        if not os.path.exists(self.vis_dir):
            try:
                os.makedirs(self.vis_dir)
                logger.info(f"Created visualization directory: {self.vis_dir}")
            except Exception as e:
                logger.error(f"Failed to create visualization directory: {str(e)}")
                self.vis_dir = ".."  # Fallback to current directory

    def start_calibration(self, ndi_tracker=None, frequency: int = 100, device = 0) -> Dict[str, Any]:
        """
        Start collecting tool transformation matrices for calibration

        Args:
            ndi_tracker: The NDI tracker object to collect data from
            frequency: Frequency in Hz to collect data

        Returns:
            Status information about the tool calibration process
        """
        if self.calibration_active:
            return {
                "status": "already_active",
                "message": "Tool calibration is already active"
            }

        # Reset matrices and parameters
        with self.matrix_lock:
            self.calibration_matrices = []

        self.calibration_frequency = frequency
        self.stop_calibration_event.clear()

        # Start the calibration thread if NDI tracker is provided
        if ndi_tracker is not None:
            try:
                self.calibration_thread = threading.Thread(
                    target=self._calibration_thread_function,
                    args=(ndi_tracker, self.stop_calibration_event, self.calibration_frequency,device),
                    daemon=True
                )
                self.calibration_thread.start()
                thread_started = True
                logger.info(f"Started calibration data collection thread at {frequency} Hz")
            except Exception as e:
                logger.error(f"Failed to start calibration thread: {str(e)}")
                thread_started = False
        else:
            thread_started = False
            logger.info("No NDI tracker provided, starting in manual mode")

        self.calibration_active = True

        logger.info(f"Started tool calibration")

        return {
            "status": "success",
            "message": "Started collecting tool transformations for calibration",
            "note": "Move the probe around while keeping the tip at a fixed point in space (pivot point)",
            "automatic_collection": thread_started,
            "frequency": frequency if thread_started else None
        }

    def end_calibration(self) -> Dict[str, Any]:
        """
        Stop collecting tool transformation matrices for calibration

        Returns:
            Information about the collected data
        """
        if not self.calibration_active:
            return {
                "status": "not_active",
                "message": "Tool calibration was not active"
            }

        # Stop the calibration thread if it's running
        thread_stopped = False
        if self.calibration_thread and self.calibration_thread.is_alive():
            self.stop_calibration_event.set()
            self.calibration_thread.join(timeout=2.0)  # Wait up to 2 seconds
            thread_stopped = True
            logger.info("Stopped calibration data collection thread")

        self.calibration_active = False

        # Get the count of matrices with thread safety
        with self.matrix_lock:
            matrices_count = len(self.calibration_matrices)
            logger.info(f"Ended tool calibration, collected {matrices_count} matrices")

        # Save to file for backup
        file_saved = self._save_to_file()

        return {
            "status": "success",
            "message": "Stopped collecting tool transformations",
            "matrices_collected": matrices_count,
            "saved_to_file": file_saved,
            "thread_stopped": thread_stopped,
            "next_step": "Use /calibrate_tool endpoint to compute the tool vector"
        }

    def _calibration_thread_function(self, ndi_tracker, stop_event: threading.Event, frequency: int = 100, device=0):
        """
        Thread function to automatically collect probe transformations

        Args:
            ndi_tracker: NDI tracker object to get transformations from
            stop_event: Event to signal thread termination
            frequency: Data collection frequency in Hz
        """
        # Calculate sleep time based on frequency
        sleep_time = 1.0 / frequency

        # Counter for logging
        frame_counter = 0
        last_log_time = time.time()

        try:
            logger.info(f"Calibration thread starting at {frequency} Hz")

            while not stop_event.is_set():
                start_time = time.time()

                try:
                    # Get tracking data from NDI tracker
                    tracking = ndi_tracker.GetPosition()

                    self.calibration_matrices.append(tracking[device])

                    # Increment frame counter
                    frame_counter += 1

                except Exception as e:
                    logger.error(f"Error getting tracking data: {str(e)}")

                # Log statistics periodically
                current_time = time.time()
                if current_time - last_log_time >= 5.0:  # Log every 5 seconds
                    with self.matrix_lock:
                        matrices_count = len(self.calibration_matrices)

                    fps = frame_counter / (current_time - last_log_time)
                    logger.info(f"Calibration collecting at {fps:.2f} FPS, total matrices: {matrices_count}")
                    frame_counter = 0
                    last_log_time = current_time

                # Sleep to maintain desired frequency
                elapsed = time.time() - start_time
                if elapsed < sleep_time:
                    time.sleep(sleep_time - elapsed)

        except Exception as e:
            logger.exception(f"Error in calibration thread: {str(e)}")
        finally:
            logger.info("Calibration thread stopped")

    def add_transformation(self, matrix: List[float]) -> Dict[str, Any]:
        """
        Add a tool transformation matrix for calibration

        Args:
            matrix: Flattened 4x4 transformation matrix (16 values)

        Returns:
            Status information about the added matrix
        """
        if not self.calibration_active:
            return {
                "status": "error",
                "message": "Tool calibration is not active. Start calibration first."
            }

        if len(matrix) != 16:
            return {
                "status": "error",
                "message": f"Expected 16 values for a 4x4 matrix, got {len(matrix)}"
            }

        try:
            # Reshape into 4x4 matrix
            matrix_np = np.array(matrix).reshape(4, 4)

            # Add with thread safety
            with self.matrix_lock:
                self.calibration_matrices.append(matrix_np)
                matrices_count = len(self.calibration_matrices)

            return {
                "status": "success",
                "message": f"Added transformation matrix. Total matrices: {matrices_count}",
                "matrix_index": matrices_count - 1
            }

        except Exception as e:
            logger.error(f"Error processing transformation matrix: {str(e)}")
            return {
                "status": "error",
                "message": f"Error processing transformation matrix: {str(e)}"
            }

    def add_ndi_tracking_data(self, tracking_data: Tuple) -> Dict[str, Any]:
        """
        Add tracking data directly from NDI tracker

        Args:
            tracking_data: Tuple containing tracking matrices

        Returns:
            Status information about the added transformation
        """
        if not self.calibration_active:
            return {
                "status": "error",
                "message": "Tool calibration is not active. Start calibration first."
            }

        try:
            # Extract probe and reference transformations
            if len(tracking_data) < 2:
                return {
                    "status": "error",
                    "message": f"Expected at least two transformations (probe and reference), got {len(tracking_data)}"
                }

            probe_matrix = tracking_data[0]
            ref_matrix = tracking_data[1]

            # Calculate probe in reference coordinate system
            # probe_in_ref = inv(ref) * probe
            try:
                if self.use_reference:
                    ref_inv = np.linalg.inv(ref_matrix)
                    probe_in_ref = np.dot(ref_inv, probe_matrix)

                    # Add the transformed matrix to calibration matrices
                    with self.matrix_lock:
                        self.calibration_matrices.append(probe_in_ref)
                        matrices_count = len(self.calibration_matrices)
                else:
                    # Just use the probe matrix directly
                    with self.matrix_lock:
                        self.calibration_matrices.append(probe_matrix)
                        matrices_count = len(self.calibration_matrices)

                return {
                    "status": "success",
                    "message": f"Added transformation. Total matrices: {matrices_count}",
                    "matrix_index": matrices_count - 1
                }
            except np.linalg.LinAlgError:
                logger.error("Reference matrix is not invertible")
                return {
                    "status": "error",
                    "message": "Reference matrix is not invertible. Cannot calculate probe-in-reference transformation."
                }

        except Exception as e:
            logger.error(f"Error processing tracking data: {str(e)}")
            return {
                "status": "error",
                "message": f"Error processing tracking data: {str(e)}"
            }

    def process_ndi_tracking_tuple(self, ndi_tuple) -> Dict[str, Any]:
        """
        Process complete NDI tracking tuple

        Args:
            ndi_tuple: Complete tuple from NDI_Tracking.GetPosition() containing
                      (port_handles, timestamps, framenumbers, transformations, quality)

        Returns:
            Status information about the added transformation
        """
        if not self.calibration_active:
            return {
                "status": "error",
                "message": "Tool calibration is not active. Start calibration first."
            }

        try:
            # Extract the transformations based on the format we've seen
            transformations = None

            if isinstance(ndi_tuple, tuple) and len(ndi_tuple) >= 4:
                if isinstance(ndi_tuple[3], list) or isinstance(ndi_tuple[3], tuple):
                    transformations = ndi_tuple[3]  # Standard format
                else:
                    for item in ndi_tuple:
                        if isinstance(item, list) or isinstance(item, tuple):
                            if len(item) >= 2:
                                if isinstance(item[0], np.ndarray) and isinstance(item[1], np.ndarray):
                                    transformations = item
                                    break

            if transformations is None:
                return {
                    "status": "error",
                    "message": "Could not extract transformation matrices from NDI tuple"
                }

            # Check we have at least two transformations (probe and reference)
            if len(transformations) < 2:
                return {
                    "status": "error",
                    "message": f"Expected at least two transformations (probe and reference), got {len(transformations)}"
                }

            probe_matrix = transformations[0]
            ref_matrix = transformations[1]

            # Calculate probe in reference coordinate system
            # probe_in_ref = inv(ref) * probe
            if self.use_reference:
                try:
                    ref_inv = np.linalg.inv(ref_matrix)
                    probe_in_ref = np.dot(ref_inv, probe_matrix)

                    # Add the transformed matrix to calibration matrices with thread safety
                    with self.matrix_lock:
                        self.calibration_matrices.append(probe_in_ref)
                        matrices_count = len(self.calibration_matrices)

                    return {
                        "status": "success",
                        "message": f"Added probe-in-reference transformation. Total matrices: {matrices_count}",
                        "matrix_index": matrices_count - 1
                    }
                except np.linalg.LinAlgError:
                    logger.error("Reference matrix is not invertible")
                    return {
                        "status": "error",
                        "message": "Reference matrix is not invertible. Cannot calculate probe-in-reference transformation."
                    }
            else:
                # Just use the probe matrix directly
                with self.matrix_lock:
                    self.calibration_matrices.append(probe_matrix)
                    matrices_count = len(self.calibration_matrices)

                return {
                    "status": "success",
                    "message": f"Added probe transformation. Total matrices: {matrices_count}",
                    "matrix_index": matrices_count - 1
                }

        except Exception as e:
            logger.error(f"Error processing NDI tracking tuple: {str(e)}")
            return {
                "status": "error",
                "message": f"Error processing NDI tracking tuple: {str(e)}"
            }

    def load_transformations_from_file(self, filename: str = "tool_tip.txt") -> Dict[str, Any]:
        """
        Load tool transformation matrices from a file

        Args:
            filename: Path to the file containing transformation matrices

        Returns:
            Information about the loaded matrices
        """
        if self.calibration_active:
            return {
                "status": "error",
                "message": "Tool calibration is already active. End the current session before loading from file."
            }

        try:
            if not os.path.exists(filename):
                return {
                    "status": "error",
                    "message": f"File {filename} not found"
                }

            # Read transformations from file
            transformations = []
            with open(filename, 'r') as f:
                lines = f.readlines()

            for i, line in enumerate(lines):
                # Skip lines with NaN
                if "NaN" in line:
                    continue

                # Parse the line to get 16 numbers
                try:
                    values = [float(val) for val in line.strip().split()]

                    # Check if we have 16 values (4x4 matrix)
                    if len(values) == 16:
                        # Reshape into 4x4 matrix
                        matrix = np.array(values).reshape(4, 4)
                        transformations.append(matrix)
                except ValueError:
                    logger.warning(f"Skipping line {i + 1}: cannot convert to floats")
                    continue

            # Start tool calibration and update matrices
            if len(transformations) < 10:
                return {
                    "status": "error",
                    "message": f"Only {len(transformations)} valid matrices found in file. At least 10 needed."
                }

            # Update matrices with thread safety
            with self.matrix_lock:
                self.calibration_matrices = transformations

            self.calibration_active = True

            logger.info(f"Loaded {len(transformations)} transformation matrices from {filename}")

            return {
                "status": "success",
                "message": f"Loaded {len(transformations)} transformation matrices from {filename}",
                "matrices_loaded": len(transformations),
                "tool_calibration_active": True,
                "next_step": "Now run /calibrate_tool to compute the tool vector"
            }

        except Exception as e:
            logger.exception(f"Error loading transformations from file")
            return {
                "status": "error",
                "message": f"Error loading transformations from file: {str(e)}"
            }

    def calibrate_tool(self, visualize: bool = False) -> Dict[str, Any]:
        """
        Process collected transformation matrices to find the tool tip vector

        Args:
            visualize: Whether to create and save a visualization

        Returns:
            Tool tip vector and calibration statistics
        """
        # Get matrices with thread safety
        with self.matrix_lock:
            matrices = self.calibration_matrices.copy()

        if len(matrices) < 10:
            return {
                "status": "error",
                "message": f"At least 10 transformations needed for reliable calibration. Only {len(matrices)} available."
            }

        try:
            # Filter out matrices with NaN values
            valid_matrices = []
            for matrix in matrices:
                if not np.isnan(matrix).any():
                    valid_matrices.append(matrix)

            if len(valid_matrices) < 10:
                return {
                    "status": "error",
                    "message": f"Only {len(valid_matrices)} valid matrices after filtering NaNs. At least 10 needed."
                }

            logger.info(f"Calibrating tool using {len(valid_matrices)} valid matrices")

            # Compute the tool tip vector by determining the pivot point
            pivot_center, mean_x, std_x, errors = self._find_pivot_point(valid_matrices)

            # Store the result
            self.calibration_result = {
                "tool_tip_vector": mean_x.tolist(),
                "standard_deviation": std_x.tolist(),
                "mean_error": float(np.mean(errors)),
                "max_error": float(np.max(errors)),
                "min_error": float(np.min(errors)),
                "num_transformations_used": len(valid_matrices),
                "coordinate_system": "probe-in-reference" if self.use_reference else "probe-global",
                "pivot_center": pivot_center.tolist()
            }

            # Save the result to a file
            try:
                with open("../constant_vector.txt", "w") as f:
                    f.write(
                        f"Tool tip vector in local coordinates: [{mean_x[0]:.6f}, {mean_x[1]:.6f}, {mean_x[2]:.6f}]\n")
                    f.write(f"Standard deviation: [{std_x[0]:.6f}, {std_x[1]:.6f}, {std_x[2]:.6f}]\n")
                    f.write(f"Mean error: {np.mean(errors):.6f} units\n")
                    f.write(f"Max error: {np.max(errors):.6f} units\n")
                    f.write(f"Pivot center: [{pivot_center[0]:.6f}, {pivot_center[1]:.6f}, {pivot_center[2]:.6f}]\n")
                    f.write(f"Number of transformations: {len(valid_matrices)}\n")
                    f.write(f"Coordinate system: {'probe-in-reference' if self.use_reference else 'probe-global'}\n")

                file_saved = True
            except Exception as e:
                logger.error(f"Error saving tool tip vector: {str(e)}")
                file_saved = False

            # Create and save visualization if requested
            vis_path = None
            if visualize:
                try:
                    vis_path = self._create_visualization(
                        mean_x, pivot_center, errors, valid_matrices
                    )
                    logger.info(f"Created and saved visualization to {vis_path}")
                except Exception as e:
                    logger.error(f"Error creating visualization: {str(e)}")

            logger.info(f"Tool calibration successful. Tool tip vector: {mean_x}")

            # Create the response
            response = {
                "status": "success",
                "tool_tip_vector": mean_x.tolist(),
                "pivot_center": pivot_center.tolist(),
                "statistics": {
                    "standard_deviation": std_x.tolist(),
                    "mean_error": float(np.mean(errors)),
                    "max_error": float(np.max(errors)),
                    "min_error": float(np.min(errors))
                },
                "num_transformations_used": len(valid_matrices),
                "saved_to_file": file_saved,
                "coordinate_system": "probe-in-reference" if self.use_reference else "probe-global"
            }

            if visualize and vis_path:
                response["visualization_path"] = vis_path

            return response

        except Exception as e:
            logger.exception(f"Error during tool calibration")
            return {
                "status": "error",
                "message": f"Error during tool calibration: {str(e)}"
            }

    def get_calibration_status(self) -> Dict[str, Any]:
        """
        Get the current status of tool calibration

        Returns:
            Information about the tool calibration process
        """
        # Get matrix count with thread safety
        with self.matrix_lock:
            matrices_count = len(self.calibration_matrices)

        thread_active = self.calibration_thread is not None and self.calibration_thread.is_alive()

        return {
            "active": self.calibration_active,
            "thread_active": thread_active,
            "matrices_collected": matrices_count,
            "collection_frequency": self.calibration_frequency if thread_active else None,
            "has_result": self.calibration_result is not None,
            "result": self.calibration_result if self.calibration_result is not None else None,
            "using_reference": self.use_reference
        }

    def get_tool_tip_vector(self) -> Optional[np.ndarray]:
        """
        Get the calibrated tool tip vector if available

        Returns:
            Tool tip vector as numpy array or None if not calibrated
        """
        if self.calibration_result is None:
            return None

        return np.array(self.calibration_result["tool_tip_vector"])

    def _find_pivot_point(self, transformations: List[np.ndarray]) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, List[float]]:
        """
        Find the pivot point and tool tip vector from a set of transformation matrices

        The pivot point is the center of rotation in global coordinates, and the tool tip vector
        is the constant vector in the probe's local coordinate system.

        Args:
            transformations: List of 4x4 transformation matrices

        Returns:
            Tuple containing (pivot_center, mean_x, std_x, errors)
        """
        # Step 1: Initialize matrices for the least squares problem
        # We want to solve Ax = b
        num_transforms = len(transformations)

        # For each transformation, we create 3 equations (one for each coordinate)
        A = np.zeros((3 * num_transforms, 6))  # 6 unknowns: 3 for pivot point, 3 for local vector
        b = np.zeros(3 * num_transforms)

        # Fill the A matrix and b vector
        for i, T in enumerate(transformations):
            # Extract rotation matrix and translation vector
            R = T[:3, :3]
            t = T[:3, 3]

            # For each transform, add 3 equations
            row_idx = 3 * i

            # Identity for pivot point coordinates
            A[row_idx:row_idx + 3, 0:3] = np.eye(3)

            # Negative rotation matrix for local vector coordinates
            A[row_idx:row_idx + 3, 3:6] = -R

            # Right hand side is the translation vector
            b[row_idx:row_idx + 3] = t

        # Step 2: Solve the least squares problem
        # This gives us both the pivot point and the local vector
        try:
            solution, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)

            # Extract pivot point and local vector
            pivot_center = solution[0:3]
            local_vector = solution[3:6]

            # Step 3: Calculate errors
            errors = []
            x_vectors = []

            for T in transformations:
                # Extract rotation matrix and translation vector
                R = T[:3, :3]
                t = T[:3, 3]

                # For validation, compute where the local vector would map to
                transformed_point = R @ local_vector + t

                # Calculate error as distance to the computed pivot center
                error = np.linalg.norm(transformed_point - pivot_center)
                errors.append(error)

                # Calculate x vector for each transformation matrix (for alternate method)
                try:
                    T_inv = np.linalg.inv(T)
                    pivot_homog = np.append(pivot_center, 1)
                    x_homog = T_inv @ pivot_homog
                    x = x_homog[:3] / x_homog[3]
                    x_vectors.append(x)
                except np.linalg.LinAlgError:
                    # Skip matrices that can't be inverted
                    pass

            # Calculate mean and standard deviation of x vectors (alternate method for validation)
            if x_vectors:
                alt_mean_x = np.mean(x_vectors, axis=0)
                std_x = np.std(x_vectors, axis=0)

                # Choose which mean to use (they should be very similar)
                # The least squares solution typically has lower overall error
                mean_x = local_vector

                # Log both for comparison
                logger.info(f"Least squares local vector: {local_vector}")
                logger.info(f"Averaging local vector: {alt_mean_x}")
                logger.info(f"Std dev: {std_x}")
            else:
                mean_x = local_vector
                std_x = np.zeros(3)

            return pivot_center, mean_x, std_x, errors

        except Exception as e:
            logger.exception("Error in least squares solution")
            raise ValueError(f"Could not solve for pivot point: {str(e)}")

    def _create_visualization(self, mean_x: np.ndarray, pivot_center: np.ndarray,
                              errors: List[float], transformations: List[np.ndarray]) -> str:
        """
        Create visualization of tool calibration results and save it locally

        Args:
            mean_x: Tool tip vector in local coordinates
            pivot_center: Calculated pivot center in global coordinates
            errors: Error for each transformation
            transformations: List of transformation matrices

        Returns:
            Path to the saved visualization image
        """
        # Create a filename with timestamp
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"tool_calibration_{timestamp}.png"
        filepath = os.path.join(self.vis_dir, filename)

        # Create a figure
        fig = plt.figure(figsize=(15, 10))

        # Create first subplot for probe positions
        ax1 = fig.add_subplot(121, projection='3d')

        # Plot all probe positions
        probe_positions = []
        for T in transformations:
            # Get the probe origin (translation part of the matrix)
            probe_pos = T[:3, 3]
            probe_positions.append(probe_pos)

        probe_positions = np.array(probe_positions)
        ax1.scatter(probe_positions[:, 0], probe_positions[:, 1], probe_positions[:, 2],
                    c=errors, cmap='viridis', label='Probe positions')

        # Plot the pivot center
        ax1.scatter([pivot_center[0]], [pivot_center[1]], [pivot_center[2]],
                    color='red', s=200, marker='*', label='Pivot center')

        # Add colorbar to show error scale
        cbar = plt.colorbar(ax1.scatter(probe_positions[:, 0], probe_positions[:, 1], probe_positions[:, 2],
                                        c=errors, cmap='viridis', label='Probe positions'),
                            ax=ax1, shrink=0.6)
        cbar.set_label('Error (units)')

        # Set labels and title
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title('Probe Positions and Pivot Center')
        ax1.legend()

        # Create second subplot for verification
        ax2 = fig.add_subplot(122, projection='3d')

        # Plot all calculated tip positions
        tip_positions = []
        for i, T in enumerate(transformations):
            # Calculate where the tip vector would be in global coordinates
            tip_pos = T[:3, :3] @ mean_x + T[:3, 3]
            tip_positions.append(tip_pos)

        tip_positions = np.array(tip_positions)
        sc = ax2.scatter(tip_positions[:, 0], tip_positions[:, 1], tip_positions[:, 2],
                         c=errors, cmap='viridis', label='Calculated tip positions')

        # Plot the pivot center again for reference
        ax2.scatter([pivot_center[0]], [pivot_center[1]], [pivot_center[2]],
                    color='red', s=200, marker='*', label='Pivot center')

        # Add colorbar
        cbar2 = plt.colorbar(sc, ax=ax2, shrink=0.6)
        cbar2.set_label('Error (units)')

        # Set labels and title
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        ax2.set_title('Verification: Calculated Tip Positions')
        ax2.legend()

        plt.tight_layout()

        # Add information text at the bottom
        coord_system = "probe-in-reference" if self.use_reference else "probe-global"
        fig.text(0.5, 0.01,
                 f"Tool tip vector: [{mean_x[0]:.2f}, {mean_x[1]:.2f}, {mean_x[2]:.2f}] | "
                 f"Pivot center: [{pivot_center[0]:.2f}, {pivot_center[1]:.2f}, {pivot_center[2]:.2f}] | "
                 f"Mean error: {np.mean(errors):.2f} | "
                 f"Max error: {np.max(errors):.2f} | "
                 f"Coordinate system: {coord_system}",
                 ha='center', fontsize=10)

        # Save the figure to disk
        try:
            fig.savefig(filepath, format='png', dpi=150, bbox_inches='tight')
            plt.close(fig)
            logger.info(f"Visualization saved to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Failed to save visualization: {str(e)}")
            plt.close(fig)
            raise

    def _save_to_file(self, filename: str = "tool_tip.txt") -> bool:
        """
        Save collected transformation matrices to a file

        Args:
            filename: Path to save the file

        Returns:
            True if saved successfully, False otherwise
        """
        with self.matrix_lock:
            if not self.calibration_matrices:
                return False

            matrices_to_save = self.calibration_matrices.copy()

        try:
            with open(filename, "w") as f:
                for matrix in matrices_to_save:
                    # Flatten the matrix and save as space-separated numbers
                    flat_matrix = matrix.flatten()
                    f.write(" ".join([str(val) for val in flat_matrix]) + "\n")

            return True
        except Exception as e:
            logger.error(f"Error saving tool calibration data: {str(e)}")
            return False
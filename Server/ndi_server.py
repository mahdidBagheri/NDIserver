import argparse

from fastapi import FastAPI, Request
from pydantic import BaseModel
import numpy as np
import uvicorn
import logging
import time
import threading
from typing import List
import socket
from threading import Event
import json

# Import our custom modules
from NDI.ndi_coarse_registration import CoarseRegistration
from NDI.ndi_fine_registration import FineRegistration
from NDI.ndi_tool_calibration import ToolCalibration
from Server.Scheme import CoarsePointInput, MatrixInput


class NDI_Server():
    def __init__(self, config, args):
        self.logger = None
        self.config = config
        self.args = args
        self.app = FastAPI(title="NDI Tracking Server")
        self.client_ip = "127.0.0.1"  # Default to localhost
        self.ndi_tracker_initialized = False

        # Streaming variables
        self.streaming_active = False
        self.streaming_thread = None
        self.streaming_stop_event = Event()
        self.streaming_port = 11111  # Default UDP port
        self.streaming_frequency = 30  # Default frequency in Hz
        self.streaming_position = 0  # For cycling through fine.txt data
        self.latest_streaming_data = None  # Store the most recent streaming data packet
        self.combined_transformation = None

        self.coarse_registration = CoarseRegistration(tip_vector=config["probe_tip_vector"])
        self.fine_registration = FineRegistration(config, args)
        self.tool_calibration = ToolCalibration()

        if args.is_local:
            from NDI import NDI_Tracking_simulator as NDI_Tracking
            self.ndi_tracking = NDI_Tracking.NDI_Tracking_Simulator(config, args)
        else:
            from NDI import NDI_Tracking
            self.ndi_tracking = NDI_Tracking.NDI_Tracking(config, args)



        self.define_logger()
        self._setup_routes()
    def define_logger(self):
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def _setup_routes(self):

        @self.app.get("/")
        def read_root():
            """Root endpoint that provides basic information"""

            return {
                "application": "NDI Tracking Server",
                "status": "running",
                "coarse_points_loaded": len(self.coarse_registration.matrices),
                "fine_points_loaded": len(self.fine_registration.fine_matrices),
                "data_source": "local files" if self.args.is_local else "NDI tracker",
                "ndi_tracker_status": "initialized" if self.ndi_tracker_initialized else "not initialized",
                "streaming_active": self.streaming_active,
                "client_ip": self.client_ip,
                "has_coarse_registration": self.coarse_registration.transformation_matrix is not None,
                "has_fine_registration": self.fine_registration.get_fine_transformation_matrix() is not None,
                "has_combined_transformation": self.combined_transformation is not None,
                "has_tool_calibration": self.tool_calibration.get_tool_tip_vector() is not None
            }

        @self.app.middleware("http")
        async def capture_client_ip(request: Request, call_next):

            # Extract client IP from request
            forwarded_for = request.headers.get("X-Forwarded-For")
            if forwarded_for:
                # If behind a proxy, X-Forwarded-For contains the original client IP
                ip = forwarded_for.split(",")[0].strip()
            else:
                # Otherwise, use the direct client IP
                ip = request.client.host

            # Don't update if it's localhost or internal request
            if ip not in ["127.0.0.1", "localhost", "::1"] and not ip.startswith("192.168.") and not ip.startswith("10."):
                client_ip = ip
                self.logger.info(f"Set streaming target IP to client IP: {client_ip}")

            response = await call_next(request)
            return response

        @self.app.post("/set_client_ip")
        def set_client_ip(ip: str):
            """Manually set the client IP address for UDP streaming"""

            # Validate IP format
            try:
                socket.inet_aton(ip)
                client_ip = ip
                self.logger.info(f"Manually set streaming target IP to: {client_ip}")
                return {
                    "status": "success",
                    "message": f"Client IP set to {client_ip}",
                    "client_ip": client_ip
                }
            except socket.error:
                return {
                    "status":"error",
                    "details":"Invalid IP address format"
                }


        @self.app.get("/get_client_ip")
        def get_client_ip():
            """Get the current client IP used for UDP streaming"""
            return {
                "client_ip": self.client_ip,
                "streaming_active": self.streaming_active,
                "streaming_port": self.streaming_port if self.streaming_active else None
            }

        @self.app.post("/start_raw_streaming")
        def start_raw_streaming():
            if not self.ndi_tracker_initialized and self.args.initialization_required:
                self.logger.error("initialization required")
                return {
                    "status": "error",
                    "details": "initialization required"
                }

            self.ndi_tracking.start_streaming()



        @self.app.post("/set_raw_streaming_frequency")
        def set_raw_streaming_frequency(frequency:int):
            self.ndi_tracking.set_streaming_frequency(frequency)
        @self.app.post("/stop_raw_streaming")
        def stop_raw_streaming():
            self.ndi_tracking.stop_streaming()

        # Data source management endpoints
        @self.app.post("/initialize_ndi")
        def initialize_ndi(force_restart: bool = False):
            """Initialize the NDI tracker explicitly"""

            # Check if already initialized
            if self.ndi_tracker_initialized and not force_restart:
                return {
                    "status": "already_initialized",
                    "message": "NDI tracker is already initialized. Use force_restart=true to reinitialize.",
                    "ndi_tracker_status": "initialized"
                }

            # If already initialized and force_restart is True, stop it first
            if self.ndi_tracker_initialized and force_restart:
                try:
                    self.logger.info("Stopping NDI tracker for forced restart...")
                    self.ndi_tracking.stop()
                    self.ndi_tracker_initialized = False
                    self.logger.info("NDI tracker stopped successfully for restart")
                except Exception as e:
                    self.logger.error(f"Error stopping NDI tracker: {str(e)}")
                    return {
                        "status": "error",
                        "message": f"Failed to stop NDI tracker for restart: {str(e)}",
                        "ndi_tracker_status": "error_stopping"
                    }

            # Initialize the tracker
            try:
                self.logger.info("Initializing NDI tracker...")
                self.ndi_tracking.start()
                self.ndi_tracker_initialized = True
                status_details = {}
                # Check if initialization was successful by trying to get a position
                for i in range(5):
                    try:
                        tracking = self.ndi_tracking.GetPosition()
                        tools_detected = len(tracking)
                        self.logger.info(f"NDI tracker initialized successfully. Detected {tools_detected} tool(s).")
                        status_details = {
                            "tools_detected": tools_detected,
                        }
                    except Exception as e:
                        self.logger.warning(f"NDI tracker initialized but error when getting position: {str(e)}")
                        status_details = {"warning": f"Could not verify tracking data: {str(e)}"}

                return {
                    "status": "success",
                    "message": "NDI tracker successfully initialized",
                    "ndi_tracker_status": "initialized",
                    "details": status_details,
                    "force_restart_used": force_restart
                }

            except Exception as e:
                self.logger.error(f"Failed to initialize NDI tracker: {str(e)}")
                return {
                    "status": "error",
                    "message": f"Failed to initialize NDI tracker: {str(e)}",
                    "ndi_tracker_status": "initialization_failed"
                }

        @self.app.post("/reset_coarse_points")
        def reset_coarse_points():
            """Reset all stored coarse points"""
            return self.coarse_registration.reset_coarse_points()


        @self.app.post("/set_coarse_point")
        def set_coarse_point(point_data: CoarsePointInput):
            """Set a coarse registration point"""

            unity_point = point_data.unity_point
            point_number = point_data.point_number

            if len(unity_point) != 3:
                return {
                    "status":"error",
                    "details":"unity_point must have exactly 3 elements"
                }

            if not self.ndi_tracker_initialized and self.args.initialization_required:
                self.logger.error("initialization required")
                return {
                    "status": "error",
                    "details": "initialization required"
                }

            # Use real NDI tracker
            self.logger.info("Getting point from real NDI tracker")

            if not self.ndi_tracker_initialized and self.args.initialization_required:
                return {
                    "status": "error",
                    "message": "could not detect tracker not initialized!"
                }

            try:
                # Get tracking data - NOTE: GetPosition returns just tracking data
                tracking = self.ndi_tracking.GetPosition()
            except Exception as e:
                self.logger.exception(f"Error getting data from NDI tracker {e}")
                return {
                    "status": "error",
                    "message": "could not detect reference!"
                }

            # Check if we got valid tracking data
            if tracking[self.config["tool_types"]["probe"]] is not None:
                # The first matrix is for the probe
                tool_matrix = tracking[self.config["tool_types"]["probe"]]
                self.logger.info("Got real-time tool matrix from NDI tracker")
            else:
                return {
                    "status": "error",
                    "message": "could not detect probe!"
                }

            # Set the coarse point using the tool matrix and unity point
            result = self.coarse_registration.set_coarse_point(unity_point, point_number, tool_matrix)
            print(self.coarse_registration.coarse_points)
            # Use the current tip vector for calculating ndi_point
            # If tool is calibrated, use that instead
            tool_tip = self.config["probe_tip_vector"]
            if self.tool_calibration.get_tool_tip_vector() is not None:
                calibrated_tip = self.tool_calibration.get_tool_tip_vector()
                tool_tip = np.append(calibrated_tip, 1.0)  # Convert to homogeneous coordinates

            # Recalculate ndi_point with the current tip vector
            ndi_point = np.dot(tool_matrix, tool_tip)[:3]
            result["ndi_point"] = ndi_point.tolist()
            result["data_source"] = "local file" if self.args.is_local else "NDI tracker"

            return result

        @self.app.post("/coarse_register")
        def coarse_register(visualize: bool = False):
            """Perform coarse registration"""

            result = self.coarse_registration.perform_coarse_registration(visualize)

            # Update the combined transformation if successful
            if result.get("status") != "error" and "transformation_matrix" in result:
                # If we only have coarse, use that as combined
                self.combined_transformation = np.array(result["transformation_matrix"])
                self.logger.info("Updated combined transformation with coarse registration result")

            return result


        # Fine registration endpoints
        @self.app.post("/start_fine_gather")
        def start_fine_gather(frequency: int = 60, streaming_raw_frequncy: int = 10):
            """Start gathering fine registration points"""
            self.ndi_tracking.set_streaming_frequency(streaming_raw_frequncy)
            return self.fine_registration.start_fine_gather(frequency, self.config["probe_tip_vector"], self.ndi_tracking, self.args.is_local)


        @self.app.post("/end_fine_gather")
        def end_fine_gather(streaming_raw_frequncy: int = 30):
            """Stop gathering fine registration points"""
            self.ndi_tracking.set_streaming_frequency(streaming_raw_frequncy)
            result = self.fine_registration.end_fine_gather()
            result["data_source"] = "local files" if self.args.is_local else "NDI tracker"
            return result


        @self.app.get("/get_fine_points_status")
        def get_fine_points_status():
            """Get the current status of fine point gathering"""
            return self.fine_registration.get_fine_points_status()

        @self.app.post("/reset_fine_gather")
        def reset_fine_gather():
            self.fine_registration.reset_fine_gather()
            result = {"results":"OK"}
            return result

        @self.app.post("/fine_register")
        def fine_register(id: int, downsample_factor: float = 1.0, visualize: bool = False):
            """Perform fine registration using ICP"""

            if self.coarse_registration.transformation_matrix is None:
                return {
                    "status":"error",
                    "details":"Coarse registration must be performed first. Please call /coarse_register endpoint."
                }

            result = self.fine_registration.perform_fine_registration(
                id=id,
                coarse_transformation_matrix=self.coarse_registration.transformation_matrix,
                downsample_factor=downsample_factor,
                visualize=visualize
            )

            # Update the combined transformation if successful
            if result.get("status") != "error" and "combined_transformation" in result:
                self.combined_transformation = np.array(result["combined_transformation"])
                self.logger.info("Updated combined transformation with fine registration result")

            return result


        # Tool calibration endpoints
        @self.app.post("/start_tool_calibration")
        def start_tool_calibration(force_stop_streaming: bool = False, device: int = 0):
            """Start collecting tool transformation matrices for calibration"""

            # Check if streaming is active
            if self.streaming_active:
                if force_stop_streaming:
                    # Stop streaming first
                    stop_streaming()
                    self.logger.info("Forced stop of UDP streaming to start tool calibration")
                else:

                    return {
                        "status": "error",
                        "details": "UDP streaming is active. Stop streaming first or use force_stop_streaming=true."
                    }

            # Start tool calibration

            return self.tool_calibration.start_calibration(ndi_tracker=self.ndi_tracking,device = device)


        @self.app.post("/end_tool_calibration")
        def end_tool_calibration():
            """Stop collecting tool transformation matrices for calibration"""
            return self.tool_calibration.end_calibration()


        @self.app.post("/calibrate_tool")
        def calibrate_tool(visualize: bool = False):
            """Process collected data to find tool tip vector"""
            result = self.tool_calibration.calibrate_tool(visualize)

            # If calibration was successful, update the tip vector
            if result.get("status") == "success" and "tool_tip_vector" in result:
                self.logger.info(f"Tool tip vector calibrated: {result['tool_tip_vector']}")

            return result


        @self.app.get("/get_tool_calibration_status")
        def get_tool_calibration_status():
            """Get current tool calibration status"""
            return self.tool_calibration.get_calibration_status()


        @self.app.post("/add_tool_transformation")
        def add_tool_transformation(matrix_input: MatrixInput):
            """Manually add a transformation matrix for calibration"""
            return self.tool_calibration.add_transformation(matrix_input.matrix)


        @self.app.post("/load_tool_transformations_from_file")
        def load_tool_transformations_from_file(filename: str = "tool_tip.txt"):
            """Load tool transformations from a file"""
            return self.tool_calibration.load_transformations_from_file(filename)


        # Streaming endpoints
        @self.app.post("/start_streaming")
        def start_streaming(port: int = 11111, frequency: int = 30, force_stop_calibration: bool = False, streaming_raw_frequency:int=10):
            """Start streaming NDI tracking data over UDP"""
            # Check if tool calibration is active

            if not self.ndi_tracker_initialized and self.args.initialization_required:
                self.logger.error("initialization required")
                return {
                    "status": "error",
                    "details": "initialization required"
                }
            if self.tool_calibration.calibration_active:
                if force_stop_calibration:
                    # Stop tool calibration first
                    end_tool_calibration()
                    self.logger.info("Forced stop of tool calibration to start streaming")
                else:

                    return {
                        "status": "error",
                        "details": "Tool calibration is active. Stop calibration first or use force_stop_calibration=true."
                    }

            if self.streaming_active:
                return {
                    "status": "already_running",
                    "message": f"Streaming already active on port {self.streaming_port} at {self.streaming_frequency} Hz",
                    "port": self.streaming_port,
                    "frequency": self.streaming_frequency
                }

            # Validate parameters
            if port < 1024 or port > 65535:
                return {
                    "status": "error",
                    "details": "Port must be between 1024 and 65535"
                }

            if frequency < 1 or frequency > 100:
                return {
                    "status": "error",
                    "details": "Frequency must be between 1 and 100 Hz"
                }

            # Check if fine registration has been performed
            if self.combined_transformation is None:
                self.logger.warning("Starting streaming without fine registration")

            # Initialize streaming
            streaming_port = port
            streaming_frequency = frequency
            self.streaming_stop_event.clear()

            # Start streaming thread
            streaming_thread = threading.Thread(
                target=self.udp_streaming_thread,
                args=(streaming_port, self.streaming_stop_event, streaming_frequency)
            )
            streaming_thread.daemon = True
            streaming_thread.start()

            self.streaming_active = True
            self.ndi_tracking.set_streaming_frequency(streaming_raw_frequency)

            return {
                "status": "started",
                "message": f"Started streaming to {self.client_ip} on port {streaming_port} at {streaming_frequency} Hz",
                "target_ip": self.client_ip,
                "port": streaming_port,
                "frequency": streaming_frequency,
                "data_source": "local files (looping)" if self.args.is_local else "NDI tracker (real-time)"
            }


        @self.app.post("/stop_streaming")
        def stop_streaming(streaming_raw_frequncy:int = 30):
            """Stop the UDP streaming of NDI tracking data"""

            if not self.streaming_active:
                return {
                    "status": "not_active",
                    "message": "Streaming is not currently active"
                }

            # Signal the thread to stop
            self.streaming_stop_event.set()
            self.ndi_tracking.set_streaming_frequency(streaming_raw_frequncy)
            # Wait for thread to finish (with timeout)
            if self.streaming_thread and self.streaming_thread.is_alive():
                self.streaming_thread.join(timeout=2.0)

            self.streaming_active = False

            return {
                "status": "stopped",
                "message": "Streaming has been stopped"
            }


        @self.app.get("/streaming_status")
        def get_streaming_status():
            """Get the current status of UDP streaming"""

            return {
                "active": self.streaming_active,
                "target_ip": self.client_ip,
                "port": self.streaming_port if self.streaming_active else None,
                "frequency": self.streaming_frequency if self.streaming_active else None,
                "data_source": "local files (looping)" if self.args.is_local else "NDI tracker (real-time)",
                "fine_registration_performed": self.combined_transformation is not None,
                "tool_calibrated": self.tool_calibration.get_tool_tip_vector() is not None
            }


        @self.app.get("/get_latest_position")
        def get_latest_position():
            """Get the most recently streamed position data"""
            if not self.streaming_active:
                return {
                    "status": "streaming_inactive",
                    "message": "Streaming is not currently active. Start streaming to get position data."
                }

            if self.latest_streaming_data is None:
                return {
                    "status": "no_data",
                    "message": "No position data available yet"
                }

            # Return the latest data
            return {
                "status": "success",
                "data": self.latest_streaming_data,
                "timestamp": time.time()
            }

        @self.app.post("/get_probe_touchpoint")
        def get_probe_touchpoint(probe_idx: int = 0, endoscope_idx : int = 2 ):
            return self.tool_calibration.calculate_touch_point(self.ndi_tracking, self.config["probe_tip_vector"][0:3], probe_idx=probe_idx, endoscope_idx=endoscope_idx)

        @self.app.post("/find_reference")
        def find_reference(max_tries:int = 50, wait_time:float=1.0):
            if not self.ndi_tracker_initialized and self.args.initialization_required:
                return {
                    "status":"error",
                    "details":"initialization required"
                }
            r = self.ndi_tracking.find_reference(max_tries, wait_time)
            return r

        @self.app.post('/check_tools')
        def check_tools():
            tools_transformation = self.ndi_tracking.GetPosition()
            tool_visibility = {}
            for tool in self.config["tool_types"].keys():
                tool_transform = tools_transformation[self.config["tool_types"][tool]]
                print(f"tool: {tool_transform}")
                if tool_transform is None:
                    tool_visibility.update({tool:False})
                else:
                    tool_visibility.update({tool:True})
            return tool_visibility

        @self.app.post("/load_last_fine_transform")
        def load_last_transform():
            with open("saved_state.json") as f:
                last_state = json.load(f)
            self.fine_registration.combined_transformation = np.asarray(last_state["combined_transform"])
            return {
                "status":"success",
                "transformation":f"{self.fine_registration.combined_transformation}"
            }

        @self.app.post("/load_last_coarse_transform")
        def load_last_transform():
            with open("saved_state.json") as f:
                last_state = json.load(f)
            self.coarse_registration.transformation_matrix = np.asarray(last_state["coarse_transform"])
            return {
                "status":"success",
                "transformation":f"{self.coarse_registration.transformation_matrix}"
            }

        @self.app.on_event("shutdown")
        def shutdown_event():
            """Clean up resources when the application is shutting down"""

            # Stop streaming if active
            if self.streaming_active:
                self.streaming_stop_event.set()
                self.logger.info("Stopped streaming during shutdown")


            # Stop fine gathering if active
            if self.fine_registration.gathering_active:
                self.fine_registration.end_fine_gather()
                self.logger.info("Stopped fine gathering during shutdown")

            # Stop NDI tracker if initialized
            if self.ndi_tracker_initialized:
                try:
                    self.logger.info("Stopping NDI tracker during shutdown...")
                    self.ndi_tracking.stop()
                    ndi_tracker_initialized = False
                    self.logger.info("NDI tracker stopped")
                except Exception as e:
                    self.logger.error(f"Error stopping NDI tracker: {str(e)}")

    def udp_streaming_thread(self, port, stop_event, frequency=30):
        """Thread function to stream NDI transformations over UDP"""

        try:
            # Create UDP socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.logger.info(f"Starting UDP streaming to {self.client_ip}:{port} at {frequency} Hz")

            # Calculate sleep time based on frequency
            sleep_time = 1.0 / frequency

            # Counter for logging
            frame_counter = 0
            last_log_time = time.time()

            # Pre-calculate the inverse of fine transformation if available
            fine_inverse = None
            if self.fine_registration.get_fine_transformation_matrix() is not None:
                fine_inverse = np.linalg.inv(self.fine_registration.get_fine_transformation_matrix())
                self.logger.info("Using inverse of fine transformation for streaming")

            while not stop_event.is_set():
                start_time = time.time()

                # Use real NDI tracker
                try:
                    # Get tracking data - NOTE: GetPosition returns just tracking data
                    tracking = self.ndi_tracking.GetPosition()


                    probe_matrix = tracking[self.config["tool_types"]["probe"]]
                    endospcope_matrix= tracking[self.config["tool_types"]["endospcope"]]

                    # Calculate probe tip position
                    tool_tip = self.tip_vector
                    if self.tool_calibration.get_tool_tip_vector() is not None:
                        # Use calibrated tool tip if available
                        calibrated_tip = self.tool_calibration.get_tool_tip_vector()
                        tool_tip = np.append(calibrated_tip, 1.0)  # Convert to homogeneous coordinates

                    tip_position = np.dot(probe_matrix, tool_tip)[:3]

                    # Apply transformations
                    transformed_position = tip_position
                    transformed_matrix = []
                    endoscope_transformed_matrix = []

                    if self.combined_transformation is not None:
                        # Create homogeneous position for combined transform
                        homogeneous_pos = np.append(tip_position, 1.0)
                        transformed_position = np.dot(self.combined_transformation, homogeneous_pos)[:3]

                    # Apply inverse of fine transformation from the left to make matrix
                    # relative to the registered coordinate system

                    if fine_inverse is not None:
                        transformed_matrix = np.linalg.inv(self.combined_transformation) @ probe_matrix
                        endoscope_transformed_matrix = np.linalg.inv(self.combined_transformation) @ endospcope_matrix


                    # Create data packet
                    data = {
                        "position": transformed_position.tolist(),
                        "source": "ndi_tracker",
                        "original": tip_position.tolist(),
                        "timestamp": time.time(),
                        "frame": frame_counter,
                        # Quality information not available since GetPosition only returns tracking
                        "matrix": probe_matrix.tolist(),  # Original matrix
                        "transformed_matrix": transformed_matrix.tolist(),
                        "endoscope_transformed_matrix":endoscope_transformed_matrix.tolist()
                    }
                except Exception as e:
                    # Error getting tracking data
                    data = {
                        "error": str(e),
                        "timestamp": time.time(),
                        "frame": frame_counter
                    }

                # Store the latest data for the API endpoint
                self.latest_streaming_data = data

                # Send data packet via UDP to the client IP
                packet = json.dumps(data).encode('utf-8')
                try:
                    sock.sendto(packet, ("192.168.31.69", port))
                except Exception as e:
                    self.logger.error(f"Error sending UDP packet to {self.client_ip}:{port}: {str(e)}")

                # Log statistics periodically
                frame_counter += 1
                current_time = time.time()
                if current_time - last_log_time >= 5.0:  # Log every 5 seconds
                    fps = frame_counter / (current_time - last_log_time)
                    self.logger.info(f"Streaming at {fps:.2f} FPS to {self.client_ip}:{port}")
                    frame_counter = 0
                    last_log_time = current_time

                # Sleep to maintain desired frequency
                elapsed = time.time() - start_time
                if elapsed < sleep_time:
                    time.sleep(sleep_time - elapsed)

        except Exception as e:
            self.logger.exception(f"Error in UDP streaming thread: {str(e)}")
        finally:
            self.logger.info("UDP streaming stopped")



    def run(self):
        uvicorn.run(self.app, host="0.0.0.0", port=8000)


from fastapi import FastAPI, HTTPException, Request
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

# Import NDI tracking module
from NDI import NDI_Tracking

try:
    with open("../constant_vector.txt", "r") as f:
        lines = f.readlines()
        l = lines[0]
        tip_vector = np.array([float(l[0]), float(l[1]), float(l[2]), 1.0])
except:
    tip_vector = np.array([3.330330, 1.016458, -159.557461, 1.0])


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="NDI Tracking Server")

# Global configuration
IS_LOCAL = True  # Default to using local files
ndi_tracker_initialized = False

# Client IP tracking
client_ip = "127.0.0.1"  # Default to localhost

# Tip position in tool coordinates
tip_vector = np.array([3.330330, 1.016458, -159.557461, 1.0])

# Create module instances
coarse_registration = CoarseRegistration()
fine_registration = FineRegistration()
tool_calibration = ToolCalibration()

# Streaming variables
streaming_active = False
streaming_thread = None
streaming_stop_event = Event()
streaming_port = 11111  # Default UDP port
streaming_frequency = 30  # Default frequency in Hz
streaming_position = 0  # For cycling through fine.txt data
latest_streaming_data = None  # Store the most recent streaming data packet

# Combined transformation matrix - combination of coarse and fine
combined_transformation = None


# Pydantic models for request validation
class CoarsePointInput(BaseModel):
    unity_point: List[float]
    point_number: int


class CenterInput(BaseModel):
    center: List[float]


class MatrixInput(BaseModel):
    matrix: List[float]


# Middleware to capture client IP address
@app.middleware("http")
async def capture_client_ip(request: Request, call_next):
    global client_ip

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
        logger.info(f"Set streaming target IP to client IP: {client_ip}")

    response = await call_next(request)
    return response


def udp_streaming_thread(port, stop_event, frequency=30):
    """Thread function to stream NDI transformations over UDP"""
    global streaming_position, combined_transformation, latest_streaming_data, client_ip

    try:
        # Create UDP socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        logger.info(f"Starting UDP streaming to {client_ip}:{port} at {frequency} Hz")

        # Calculate sleep time based on frequency
        sleep_time = 1.0 / frequency

        # Counter for logging
        frame_counter = 0
        last_log_time = time.time()

        # Pre-calculate the inverse of fine transformation if available
        fine_inverse = None
        if fine_registration.get_fine_transformation_matrix() is not None:
            fine_inverse = np.linalg.inv(fine_registration.get_fine_transformation_matrix())
            logger.info("Using inverse of fine transformation for streaming")

        while not stop_event.is_set():
            start_time = time.time()

            if IS_LOCAL:
                # Use fine.txt data in a loop
                fine_matrices = fine_registration.fine_matrices
                if len(fine_matrices) == 0:
                    logger.error("No fine matrices available for streaming")
                    break

                # Get next matrix and cycle through
                current_matrix = fine_matrices[streaming_position]
                streaming_position = (streaming_position + 1) % len(fine_matrices)

                # Calculate probe tip position
                tool_tip = tip_vector
                if tool_calibration.get_tool_tip_vector() is not None:
                    # Use calibrated tool tip if available
                    calibrated_tip = tool_calibration.get_tool_tip_vector()
                    tool_tip = np.append(calibrated_tip, 1.0)  # Convert to homogeneous coordinates

                tip_position = np.dot(current_matrix, tool_tip)[:3]

                # Apply transformations
                transformed_position = tip_position
                transformed_matrix = current_matrix.copy()

                if combined_transformation is not None:
                    # Create homogeneous position for combined transform
                    homogeneous_pos = np.append(tip_position, 1.0)
                    transformed_position = np.dot(combined_transformation, homogeneous_pos)[:3]

                # Apply inverse of fine transformation from the left to make matrix
                # relative to the registered coordinate system
                if fine_inverse is not None:
                    transformed_matrix = np.linalg.inv(combined_transformation) @ current_matrix

                # Create data packet
                data = {
                    "position": transformed_position.tolist(),
                    "source": "simulated",
                    "original": tip_position.tolist(),
                    "timestamp": time.time(),
                    "frame": frame_counter,
                    "matrix": current_matrix.tolist(),  # Original matrix
                    "transformed_matrix": transformed_matrix.tolist(),  # Matrix with fine_inverse applied from left
                    "endoscope_transformed_matrix": np.eye(4).tolist()
                }
            else:
                # Use real NDI tracker
                try:
                    # Get tracking data - NOTE: GetPosition returns just tracking data
                    tracking = NDI_Tracking.ndi_tracking.GetPosition()

                    if tracking and len(tracking) > 0:
                        # Get probe matrix
                        probe_matrix = tracking[0]
                        endospcope_matrix= tracking[2]

                        # Calculate probe tip position
                        tool_tip = tip_vector
                        if tool_calibration.get_tool_tip_vector() is not None:
                            # Use calibrated tool tip if available
                            calibrated_tip = tool_calibration.get_tool_tip_vector()
                            tool_tip = np.append(calibrated_tip, 1.0)  # Convert to homogeneous coordinates

                        tip_position = np.dot(probe_matrix, tool_tip)[:3]

                        # Apply transformations
                        transformed_position = tip_position
                        transformed_matrix = probe_matrix.copy()

                        if combined_transformation is not None:
                            # Create homogeneous position for combined transform
                            homogeneous_pos = np.append(tip_position, 1.0)
                            transformed_position = np.dot(combined_transformation, homogeneous_pos)[:3]

                        # Apply inverse of fine transformation from the left to make matrix
                        # relative to the registered coordinate system
                        if fine_inverse is not None:
                            transformed_matrix = np.linalg.inv(combined_transformation) @ probe_matrix
                            endoscope_transformed_matrix = np.linalg.inv(combined_transformation) @ endospcope_matrix

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
                    else:
                        # No tracking data available
                        data = {
                            "error": "No tracking data",
                            "timestamp": time.time(),
                            "frame": frame_counter
                        }
                except Exception as e:
                    # Error getting tracking data
                    data = {
                        "error": str(e),
                        "timestamp": time.time(),
                        "frame": frame_counter
                    }

            # Store the latest data for the API endpoint
            latest_streaming_data = data

            # Send data packet via UDP to the client IP
            packet = json.dumps(data).encode('utf-8')
            try:
                sock.sendto(packet, ("192.168.31.69", port))
            except Exception as e:
                logger.error(f"Error sending UDP packet to {client_ip}:{port}: {str(e)}")

            # Log statistics periodically
            frame_counter += 1
            current_time = time.time()
            if current_time - last_log_time >= 5.0:  # Log every 5 seconds
                fps = frame_counter / (current_time - last_log_time)
                logger.info(f"Streaming at {fps:.2f} FPS to {client_ip}:{port}")
                frame_counter = 0
                last_log_time = current_time

            # Sleep to maintain desired frequency
            elapsed = time.time() - start_time
            if elapsed < sleep_time:
                time.sleep(sleep_time - elapsed)

    except Exception as e:
        logger.exception(f"Error in UDP streaming thread: {str(e)}")
    finally:
        logger.info("UDP streaming stopped")


# Root endpoint
@app.get("/")
def read_root():
    """Root endpoint that provides basic information"""
    global combined_transformation

    return {
        "application": "NDI Tracking Server",
        "status": "running",
        "coarse_points_loaded": len(coarse_registration.matrices),
        "fine_points_loaded": len(fine_registration.fine_matrices),
        "data_source": "local files" if IS_LOCAL else "NDI tracker",
        "ndi_tracker_status": "initialized" if ndi_tracker_initialized else "not initialized",
        "streaming_active": streaming_active,
        "client_ip": client_ip,
        "has_coarse_registration": coarse_registration.transformation_matrix is not None,
        "has_fine_registration": fine_registration.get_fine_transformation_matrix() is not None,
        "has_combined_transformation": combined_transformation is not None,
        "has_tool_calibration": tool_calibration.get_tool_tip_vector() is not None
    }


# Client IP management endpoints
@app.post("/set_client_ip")
def set_client_ip(ip: str):
    """Manually set the client IP address for UDP streaming"""
    global client_ip

    # Validate IP format
    try:
        socket.inet_aton(ip)
        client_ip = ip
        logger.info(f"Manually set streaming target IP to: {client_ip}")
        return {
            "status": "success",
            "message": f"Client IP set to {client_ip}",
            "client_ip": client_ip
        }
    except socket.error:
        raise HTTPException(status_code=400, detail="Invalid IP address format")


@app.get("/get_client_ip")
def get_client_ip():
    """Get the current client IP used for UDP streaming"""
    return {
        "client_ip": client_ip,
        "streaming_active": streaming_active,
        "streaming_port": streaming_port if streaming_active else None
    }


# Data source management endpoints
@app.post("/initialize_ndi")
def initialize_ndi(force_restart: bool = False):
    """Initialize the NDI tracker explicitly"""
    global ndi_tracker_initialized

    # Check if already initialized
    if ndi_tracker_initialized and not force_restart:
        return {
            "status": "already_initialized",
            "message": "NDI tracker is already initialized. Use force_restart=true to reinitialize.",
            "ndi_tracker_status": "initialized"
        }

    # If already initialized and force_restart is True, stop it first
    if ndi_tracker_initialized and force_restart:
        try:
            logger.info("Stopping NDI tracker for forced restart...")
            NDI_Tracking.ndi_tracking.stop()
            ndi_tracker_initialized = False
            logger.info("NDI tracker stopped successfully for restart")
        except Exception as e:
            logger.error(f"Error stopping NDI tracker: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to stop NDI tracker for restart: {str(e)}",
                "ndi_tracker_status": "error_stopping"
            }

    # Initialize the tracker
    try:
        logger.info("Initializing NDI tracker...")
        NDI_Tracking.ndi_tracking.start()
        ndi_tracker_initialized = True

        # Check if initialization was successful by trying to get a position
        try:
            tracking = NDI_Tracking.ndi_tracking.GetPosition()
            if tracking is not None:
                tools_detected = len(tracking)
                logger.info(f"NDI tracker initialized successfully. Detected {tools_detected} tool(s).")
                status_details = {
                    "tools_detected": tools_detected,
                }
            else:
                logger.warning("NDI tracker initialized but no tracking data available.")
                status_details = {"tools_detected": 0}
        except Exception as e:
            logger.warning(f"NDI tracker initialized but error when getting position: {str(e)}")
            status_details = {"warning": f"Could not verify tracking data: {str(e)}"}

        return {
            "status": "success",
            "message": "NDI tracker successfully initialized",
            "ndi_tracker_status": "initialized",
            "details": status_details,
            "force_restart_used": force_restart
        }

    except Exception as e:
        logger.error(f"Failed to initialize NDI tracker: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to initialize NDI tracker: {str(e)}",
            "ndi_tracker_status": "initialization_failed"
        }


@app.post("/set_data_source")
def set_data_source(is_local: bool):
    """Set the data source for NDI tracking data"""
    global IS_LOCAL, ndi_tracker_initialized

    # If switching from local to real NDI
    if IS_LOCAL and not is_local and not ndi_tracker_initialized:
        try:
            logger.info("Initializing NDI tracker...")
            NDI_Tracking.ndi_tracking.start()
            ndi_tracker_initialized = True
            ndi_status = "initialized"
        except Exception as e:
            logger.error(f"Failed to initialize NDI tracker: {str(e)}")
            ndi_status = f"initialization failed: {str(e)}"

    # If switching from real NDI to local
    elif not IS_LOCAL and is_local and ndi_tracker_initialized:
        try:
            logger.info("Stopping NDI tracker...")
            NDI_Tracking.ndi_tracking.stop()
            ndi_tracker_initialized = False
            ndi_status = "stopped"
        except Exception as e:
            logger.error(f"Error stopping NDI tracker: {str(e)}")
            ndi_status = f"error stopping: {str(e)}"
    else:
        ndi_status = "initialized" if ndi_tracker_initialized else "not initialized"

    IS_LOCAL = is_local

    return {
        "status": "success",
        "is_local": IS_LOCAL,
        "ndi_tracker_status": ndi_status,
        "data_source": "local files" if IS_LOCAL else "NDI tracker"
    }


# Coarse registration endpoints
@app.get("/available_points")
def available_points():
    """Get available point numbers from coarse data"""
    return coarse_registration.get_available_points()


@app.get("/debug_coarse_data")
def debug_coarse_data():
    """Debug endpoint to check coarse data loading"""
    return coarse_registration.debug_coarse_data()


@app.post("/reset_coarse_points")
def reset_coarse_points():
    """Reset all stored coarse points"""
    return coarse_registration.reset_coarse_points()


@app.post("/set_coarse_point")
def set_coarse_point(point_data: CoarsePointInput):
    """Set a coarse registration point"""
    global ndi_tracker_initialized

    unity_point = point_data.unity_point
    point_number = point_data.point_number

    if len(unity_point) != 3:
        raise HTTPException(status_code=400, detail="unity_point must have exactly 3 elements")

    # Get probe transformation matrix and calculate tip position
    if IS_LOCAL:
        # Use local files mode
        logger.info(
            f"Using local file for point_number={point_number}, available keys: {sorted(coarse_registration.matrices.keys())}")

        # Get transformation matrix from the preloaded matrices
        if point_number not in coarse_registration.matrices:
            # Check if there are any matrices loaded
            if not coarse_registration.matrices:
                raise HTTPException(
                    status_code=500,
                    detail="No matrices loaded from coarse.txt file. Please check if the file exists and is formatted correctly."
                )
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid point_number {point_number}. Available point numbers are: {sorted(coarse_registration.matrices.keys())}"
                )

        tool_matrix = coarse_registration.matrices[point_number]
    else:
        # Use real NDI tracker
        logger.info("Getting point from real NDI tracker")

        if not ndi_tracker_initialized:
            try:
                logger.info("Initializing NDI tracker (auto)...")
                NDI_Tracking.ndi_tracking.start()
                ndi_tracker_initialized = True
            except Exception as e:
                logger.error(f"Failed to initialize NDI tracker: {str(e)}")
                raise HTTPException(status_code=500,
                                    detail=f"NDI tracker not initialized. Use /set_data_source first or check tracker.")

        try:
            # Get tracking data - NOTE: GetPosition returns just tracking data
            tracking = NDI_Tracking.ndi_tracking.GetPosition()

            # Check if we got valid tracking data
            if tracking and len(tracking) > 0:
                # The first matrix is for the probe
                tool_matrix = tracking[0]
                logger.info("Got real-time tool matrix from NDI tracker")
            else:
                raise HTTPException(status_code=500, detail="No tracking data received from NDI tracker")

        except Exception as e:
            logger.exception("Error getting data from NDI tracker")
            raise HTTPException(status_code=500, detail=f"Error getting data from NDI tracker: {str(e)}")

    # Set the coarse point using the tool matrix and unity point
    result = coarse_registration.set_coarse_point(unity_point, point_number, tool_matrix)

    # Use the current tip vector for calculating ndi_point
    # If tool is calibrated, use that instead
    tool_tip = tip_vector
    if tool_calibration.get_tool_tip_vector() is not None:
        calibrated_tip = tool_calibration.get_tool_tip_vector()
        tool_tip = np.append(calibrated_tip, 1.0)  # Convert to homogeneous coordinates

    # Recalculate ndi_point with the current tip vector
    ndi_point = np.dot(tool_matrix, tool_tip)[:3]
    result["ndi_point"] = ndi_point.tolist()
    result["data_source"] = "local file" if IS_LOCAL else "NDI tracker"

    return result


@app.get("/get_coarse_points")
def get_coarse_points():
    """Get all stored coarse registration points"""
    return coarse_registration.get_coarse_points()


@app.post("/coarse_register")
def coarse_register(visualize: bool = False):
    """Perform coarse registration"""
    global combined_transformation

    result = coarse_registration.perform_coarse_registration(visualize)

    # Update the combined transformation if successful
    if result.get("status") != "error" and "transformation_matrix" in result:
        # If we only have coarse, use that as combined
        combined_transformation = np.array(result["transformation_matrix"])
        logger.info("Updated combined transformation with coarse registration result")

    return result


# Fine registration endpoints
@app.post("/start_fine_gather")
def start_fine_gather(frequency: int = 60):
    """Start gathering fine registration points"""
    return fine_registration.start_fine_gather(frequency, tip_vector, NDI_Tracking.ndi_tracking, IS_LOCAL)


@app.post("/end_fine_gather")
def end_fine_gather():
    """Stop gathering fine registration points"""
    result = fine_registration.end_fine_gather()
    result["data_source"] = "local files" if IS_LOCAL else "NDI tracker"
    return result


@app.get("/get_fine_points_status")
def get_fine_points_status():
    """Get the current status of fine point gathering"""
    return fine_registration.get_fine_points_status()


@app.post("/simulate_fine_gather")
def simulate_fine_gather(num_points: int = 100, replace_existing: bool = False, downsample_factor: float = 1.0):
    """Simulate gathering fine points"""
    # If tool is calibrated, use the calibrated tip vector
    current_tip_vector = tip_vector
    if tool_calibration.get_tool_tip_vector() is not None:
        calibrated_tip = tool_calibration.get_tool_tip_vector()
        current_tip_vector = np.append(calibrated_tip, 1.0)  # Convert to homogeneous coordinates
        logger.info("Using calibrated tool tip vector for fine gathering")

    result = fine_registration.simulate_fine_gather(
        num_points=num_points,
        replace_existing=replace_existing,
        downsample_factor=downsample_factor,
        tip_vector=current_tip_vector,
        is_local=IS_LOCAL,
        ndi_tracker=NDI_Tracking.ndi_tracking if ndi_tracker_initialized else None
    )

    if "status" in result and result["status"] == "success":
        result["data_source"] = "local files" if IS_LOCAL else "NDI tracker"

    return result


@app.post("/reset_fine_gather")
def reset_fine_gather():
    fine_registration.reset_fine_gather()
    result = {"results":"OK"}
    return result

@app.post("/fine_register")
def fine_register(id: int, downsample_factor: float = 1.0, visualize: bool = False):
    """Perform fine registration using ICP"""
    global combined_transformation

    if coarse_registration.transformation_matrix is None:
        raise HTTPException(
            status_code=400,
            detail="Coarse registration must be performed first. Please call /coarse_register endpoint."
        )

    result = fine_registration.perform_fine_registration(
        id=id,
        coarse_transformation_matrix=coarse_registration.transformation_matrix,
        downsample_factor=downsample_factor,
        visualize=visualize
    )

    # Update the combined transformation if successful
    if result.get("status") != "error" and "combined_transformation" in result:
        combined_transformation = np.array(result["combined_transformation"])
        logger.info("Updated combined transformation with fine registration result")

    return result


# Tool calibration endpoints
@app.post("/start_tool_calibration")
def start_tool_calibration(force_stop_streaming: bool = False, device: int = 0):
    """Start collecting tool transformation matrices for calibration"""
    global streaming_active, tool_calibration

    # Check if streaming is active
    if streaming_active:
        if force_stop_streaming:
            # Stop streaming first
            stop_streaming()
            logger.info("Forced stop of UDP streaming to start tool calibration")
        else:
            raise HTTPException(
                status_code=400,
                detail="UDP streaming is active. Stop streaming first or use force_stop_streaming=true."
            )

    # Start tool calibration

    return tool_calibration.start_calibration(ndi_tracker=NDI_Tracking.ndi_tracking,device = device)


@app.post("/end_tool_calibration")
def end_tool_calibration():
    """Stop collecting tool transformation matrices for calibration"""
    return tool_calibration.end_calibration()


@app.post("/calibrate_tool")
def calibrate_tool(visualize: bool = False):
    """Process collected data to find tool tip vector"""
    result = tool_calibration.calibrate_tool(visualize)

    # If calibration was successful, update the tip vector
    if result.get("status") == "success" and "tool_tip_vector" in result:
        logger.info(f"Tool tip vector calibrated: {result['tool_tip_vector']}")

    return result


@app.get("/get_tool_calibration_status")
def get_tool_calibration_status():
    """Get current tool calibration status"""
    return tool_calibration.get_calibration_status()


@app.post("/add_tool_transformation")
def add_tool_transformation(matrix_input: MatrixInput):
    """Manually add a transformation matrix for calibration"""
    return tool_calibration.add_transformation(matrix_input.matrix)


@app.post("/load_tool_transformations_from_file")
def load_tool_transformations_from_file(filename: str = "tool_tip.txt"):
    """Load tool transformations from a file"""
    return tool_calibration.load_transformations_from_file(filename)


# Streaming endpoints
@app.post("/start_streaming")
def start_streaming(port: int = 11111, frequency: int = 30, force_stop_calibration: bool = False):
    """Start streaming NDI tracking data over UDP"""
    global streaming_active, streaming_thread, streaming_stop_event, streaming_port, streaming_frequency
    global tool_calibration

    # Check if tool calibration is active
    if tool_calibration.calibration_active:
        if force_stop_calibration:
            # Stop tool calibration first
            end_tool_calibration()
            logger.info("Forced stop of tool calibration to start streaming")
        else:
            raise HTTPException(
                status_code=400,
                detail="Tool calibration is active. Stop calibration first or use force_stop_calibration=true."
            )

    if streaming_active:
        return {
            "status": "already_running",
            "message": f"Streaming already active on port {streaming_port} at {streaming_frequency} Hz",
            "port": streaming_port,
            "frequency": streaming_frequency
        }

    # Validate parameters
    if port < 1024 or port > 65535:
        raise HTTPException(status_code=400, detail="Port must be between 1024 and 65535")

    if frequency < 1 or frequency > 100:
        raise HTTPException(status_code=400, detail="Frequency must be between 1 and 100 Hz")

    # Check if fine registration has been performed
    if combined_transformation is None:
        logger.warning("Starting streaming without fine registration")

    # Initialize streaming
    streaming_port = port
    streaming_frequency = frequency
    streaming_stop_event.clear()

    # Start streaming thread
    streaming_thread = threading.Thread(
        target=udp_streaming_thread,
        args=(streaming_port, streaming_stop_event, streaming_frequency)
    )
    streaming_thread.daemon = True
    streaming_thread.start()

    streaming_active = True

    return {
        "status": "started",
        "message": f"Started streaming to {client_ip} on port {streaming_port} at {streaming_frequency} Hz",
        "target_ip": client_ip,
        "port": streaming_port,
        "frequency": streaming_frequency,
        "data_source": "local files (looping)" if IS_LOCAL else "NDI tracker (real-time)"
    }


@app.post("/stop_streaming")
def stop_streaming():
    """Stop the UDP streaming of NDI tracking data"""
    global streaming_active, streaming_thread, streaming_stop_event

    if not streaming_active:
        return {
            "status": "not_active",
            "message": "Streaming is not currently active"
        }

    # Signal the thread to stop
    streaming_stop_event.set()

    # Wait for thread to finish (with timeout)
    if streaming_thread and streaming_thread.is_alive():
        streaming_thread.join(timeout=2.0)

    streaming_active = False

    return {
        "status": "stopped",
        "message": "Streaming has been stopped"
    }


@app.get("/streaming_status")
def get_streaming_status():
    """Get the current status of UDP streaming"""
    global combined_transformation

    return {
        "active": streaming_active,
        "target_ip": client_ip,
        "port": streaming_port if streaming_active else None,
        "frequency": streaming_frequency if streaming_active else None,
        "data_source": "local files (looping)" if IS_LOCAL else "NDI tracker (real-time)",
        "fine_registration_performed": combined_transformation is not None,
        "tool_calibrated": tool_calibration.get_tool_tip_vector() is not None
    }


@app.get("/get_latest_position")
def get_latest_position():
    """Get the most recently streamed position data"""
    if not streaming_active:
        return {
            "status": "streaming_inactive",
            "message": "Streaming is not currently active. Start streaming to get position data."
        }

    if latest_streaming_data is None:
        return {
            "status": "no_data",
            "message": "No position data available yet"
        }

    # Return the latest data
    return {
        "status": "success",
        "data": latest_streaming_data,
        "timestamp": time.time()
    }

@app.get("/get_probe_touchpoint")
def get_probe_touchpoint(probe_idx: int = 0, endoscope_idx : int = 2 ):
    return tool_calibration.calculate_touch_point(NDI_Tracking.ndi_tracking, tip_vector, probe_idx=probe_idx, endoscope_idx=endoscope_idx)

@app.on_event("shutdown")
def shutdown_event():
    """Clean up resources when the application is shutting down"""
    global ndi_tracker_initialized, streaming_active, streaming_stop_event

    # Stop streaming if active
    if streaming_active:
        streaming_stop_event.set()
        logger.info("Stopped streaming during shutdown")


    # Stop fine gathering if active
    if fine_registration.gathering_active:
        fine_registration.end_fine_gather()
        logger.info("Stopped fine gathering during shutdown")

    # Stop NDI tracker if initialized
    if ndi_tracker_initialized:
        try:
            logger.info("Stopping NDI tracker during shutdown...")
            NDI_Tracking.ndi_tracking.stop()
            ndi_tracker_initialized = False
            logger.info("NDI tracker stopped")
        except Exception as e:
            logger.error(f"Error stopping NDI tracker: {str(e)}")


# Run the application if this script is executed directly
if __name__ == "__main__":
    uvicorn.run("ndi_server:app", host="0.0.0.0", port=8000, reload=True)
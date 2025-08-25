import requests
import json
from PyQt5.QtCore import QObject, pyqtSignal


class ServerClient(QObject):
    # Signals for communication with UI components
    connection_status_changed = pyqtSignal(bool, str)  # connected, message
    coarse_points_updated = pyqtSignal(list)  # server_points

    def __init__(self, server_url):
        super().__init__()
        self.server_url = server_url
        self._last_coarse_points = []

    def set_server_url(self, url):
        self.server_url = url

    def test_connection(self):
        try:
            response = requests.get(f"{self.server_url}/", timeout=5)
            if response.status_code == 200:
                self.connection_status_changed.emit(True, "Connected")
                return True
            else:
                self.connection_status_changed.emit(False, f"HTTP {response.status_code}")
                return False
        except Exception as e:
            self.connection_status_changed.emit(False, str(e))
            return False

    def get_status(self):
        try:
            response = requests.get(f"{self.server_url}/", timeout=3)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"Error getting status: {e}")
        return None

    # NDI System Control
    def initialize_ndi(self):
        try:
            response = requests.post(f"{self.server_url}/initialize_ndi", timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"Error initializing NDI: {e}")
        return None

    def check_tools(self):
        try:
            response = requests.post(f"{self.server_url}/check_tools", timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"Error checking tools: {e}")
        return None

    # Coarse Registration API
    def get_coarse_points(self):
        try:
            response = requests.get(f"{self.server_url}/get_coarse_points", timeout=3)
            if response.status_code == 200:
                points = response.json()
                # Emit signal if points changed
                if points != self._last_coarse_points:
                    self._last_coarse_points = points.copy() if points else []
                    self.coarse_points_updated.emit(points or [])
                return points
        except Exception as e:
            print(f"Error getting coarse points: {e}")
        return []

    def set_coarse_point(self, unity_point, point_number):
        try:
            data = {
                "unity_point": unity_point,
                "point_number": point_number
            }
            response = requests.post(f"{self.server_url}/set_coarse_point", json=data, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"Error setting coarse point: {e}")
        return {"status": "error", "details": str(e)}

    def perform_coarse_registration(self):
        try:
            response = requests.post(f"{self.server_url}/coarse_register",
                                     params={"visualize": False}, timeout=30)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"Error performing coarse registration: {e}")
        return {"status": "error", "details": str(e)}

    def reset_coarse_points(self):
        try:
            response = requests.post(f"{self.server_url}/reset_coarse_points", timeout=10)
            if response.status_code == 200:
                self._last_coarse_points = []
                self.coarse_points_updated.emit([])
                return response.json()
        except Exception as e:
            print(f"Error resetting coarse points: {e}")
        return {"status": "error", "details": str(e)}

    # Fine Registration API
    def start_fine_gathering(self, frequency):
        try:
            response = requests.post(f"{self.server_url}/start_fine_gather",
                                     params={"frequency": frequency}, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"Error starting fine gathering: {e}")
        return {"status": "error", "details": str(e)}

    def stop_fine_gathering(self):
        try:
            response = requests.post(f"{self.server_url}/end_fine_gather", timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"Error stopping fine gathering: {e}")
        return {"status": "error", "details": str(e)}

    def perform_fine_registration(self, model_id, downsample_factor):
        try:
            params = {
                "id": model_id,
                "downsample_factor": downsample_factor,
                "visualize": False
            }
            response = requests.post(f"{self.server_url}/fine_register",
                                     params=params, timeout=60)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"Error performing fine registration: {e}")
        return {"status": "error", "details": str(e)}

    def reset_fine_gathering(self):
        try:
            response = requests.post(f"{self.server_url}/reset_fine_gather", timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"Error resetting fine gathering: {e}")
        return {"status": "error", "details": str(e)}

    def get_fine_points_status(self):
        try:
            response = requests.get(f"{self.server_url}/get_fine_points_status", timeout=3)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"Error getting fine points status: {e}")
        return None

    # Tool Calibration API
    def start_tool_calibration(self, device):
        try:
            response = requests.post(f"{self.server_url}/start_tool_calibration",
                                     params={"device": device, "force_stop_streaming": True},
                                     timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"Error starting tool calibration: {e}")
        return {"status": "error", "details": str(e)}

    def stop_tool_calibration(self):
        try:
            response = requests.post(f"{self.server_url}/end_tool_calibration", timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"Error stopping tool calibration: {e}")
        return {"status": "error", "details": str(e)}

    def calibrate_tool(self):
        try:
            response = requests.post(f"{self.server_url}/calibrate_tool",
                                     params={"visualize": False}, timeout=30)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"Error calibrating tool: {e}")
        return {"status": "error", "details": str(e)}

    def get_tool_calibration_status(self):
        try:
            response = requests.get(f"{self.server_url}/get_tool_calibration_status", timeout=3)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"Error getting tool calibration status: {e}")
        return None

    # Streaming API
    def start_streaming(self, port, frequency):
        try:
            params = {
                "port": port,
                "frequency": frequency,
                "force_stop_calibration": True
            }
            response = requests.post(f"{self.server_url}/start_streaming",
                                     params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"Error starting streaming: {e}")
        return {"status": "error", "details": str(e)}

    def stop_streaming(self):
        try:
            response = requests.post(f"{self.server_url}/stop_streaming", timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"Error stopping streaming: {e}")
        return {"status": "error", "details": str(e)}

    def get_latest_position(self):
        try:
            response = requests.get(f"{self.server_url}/get_latest_position", timeout=3)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"Error getting latest position: {e}")
        return {"status": "error", "details": str(e)}

    def cleanup(self):
        # Stop any active streaming
        try:
            self.stop_streaming()
        except:
            pass
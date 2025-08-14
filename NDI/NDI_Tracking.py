import json
import time
import socket
import threading
import numpy as np
from sksurgerynditracker.nditracker import NDITracker


class NDI_Tracking():
    def __init__(self, config, args):
        self.SETTINGS = config["device_params"]
        self.config = config
        self.args = args
        self.last_reference = None

        # Streaming related attributes
        self.streaming = False
        self.streaming_thread = None
        self.streaming_frequency = 30  # Default 30Hz
        self.streaming_socket = None
        self.streaming_address = None
        self.streaming_port = None

    def start(self):
        self.TRACKER = NDITracker(self.SETTINGS)
        self.TRACKER.start_tracking()

    def get_tracking(self):
        port_handles, timestamps, framenumbers, tracking, quality = self.TRACKER.get_frame()
        for i, t in enumerate(tracking):
            if np.isnan(tracking[i]).any():
                tracking[i] = None
            print(t)
        return tracking

    def GetPosition(self):
        tracking = self.get_tracking()
        if self.args.reference_required:
            if tracking[self.config["tool_types"]["reference"]] is None and self.last_reference is None:
                raise Exception(
                    "could not detect reference! Reference is required in this mode, if you do not want the reference try reference_required = false")

            self.last_reference = tracking[self.config["tool_types"]["reference"]]

            transforms = [None, self.last_reference, None]
            if tracking[self.config["tool_types"]["probe"]] is None:
                print("Could not detect probe!")
            else:
                transforms[self.config["tool_types"]["probe"]] = np.linalg.inv(self.last_reference) @ tracking[
                    self.config["tool_types"]["probe"]]

            if tracking[self.config["tool_types"]["endoscope"]] is None:
                print("Could not detect endoscope!")
            else:
                transforms[self.config["tool_types"]["endoscope"]] = np.linalg.inv(self.last_reference) @ tracking[
                    self.config["tool_types"]["endoscope"]]

            return transforms

        else:
            return tracking

    def stop(self):
        if self.streaming:
            self.stop_streaming()
        self.TRACKER.stop_tracking()
        self.TRACKER.close()

    def start_streaming(self, address='127.0.0.1', port=5556):
        """
        Start streaming tracking data over UDP at the set frequency

        Args:
            address (str): Target IP address
            port (int): Target UDP port
        """
        if self.streaming:
            print("Streaming is already active")
            return

        self.streaming_address = address
        self.streaming_port = port
        self.streaming_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.streaming = True

        # Start streaming in a separate thread
        self.streaming_thread = threading.Thread(target=self._streaming_worker)
        self.streaming_thread.daemon = True
        self.streaming_thread.start()
        print(f"Started streaming to {address}:{port} at {self.streaming_frequency}Hz")

    def stop_streaming(self):
        """
        Stop streaming tracking data
        """
        if not self.streaming:
            print("Streaming is not active")
            return

        self.streaming = False
        if self.streaming_thread:
            self.streaming_thread.join(timeout=1.0)
        if self.streaming_socket:
            self.streaming_socket.close()
            self.streaming_socket = None
        print("Stopped streaming")

    def set_streaming_frequency(self, frequency):
        """
        Set the streaming frequency in Hz

        Args:
            frequency (float): Streaming frequency in Hz
        """
        if frequency <= 0:
            print("Frequency must be positive")
            return

        self.streaming_frequency = frequency
        print(f"Set streaming frequency to {frequency}Hz")

    def _streaming_worker(self):
        """
        Worker thread function that handles the streaming
        """
        period = 1.0 / self.streaming_frequency

        while self.streaming:
            start_time = time.time()

            try:
                # Get tracking data
                tracking_data = self.get_tracking()

                # Convert tracking data to a JSON string
                # We need to handle numpy arrays for JSON serialization
                tracking_json = {}
                for i, transform in enumerate(tracking_data):
                    if transform is not None:
                        tracking_json[next((k for k, v in self.config["tool_types"].items() if v == i), None)] = transform.tolist()
                    else:
                        tracking_json[next((k for k, v in self.config["tool_types"].items() if v == i), None)] =None

                # Add timestamp
                tracking_json["timestamp"] = time.time()

                # Serialize and send
                data = json.dumps(tracking_json).encode('utf-8')
                self.streaming_socket.sendto(data, (self.streaming_address, self.streaming_port))

                # Sleep to maintain frequency
                elapsed = time.time() - start_time
                sleep_time = period - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            except Exception as e:
                print(f"Streaming error: {e}")
                time.sleep(0.1)  # Avoid busy waiting in case of error
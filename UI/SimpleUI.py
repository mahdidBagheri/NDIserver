import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import requests
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D  # Import the 3D toolkit
import threading
import time
from datetime import datetime


class NDITrackerUI:
    def __init__(self, root):
        self.root = root
        self.root.title("NDI Tracker Control Panel")
        self.root.geometry("900x700")
        self.root.minsize(800, 600)

        self.base_url = "http://localhost:8000"
        self.streaming_active = False
        self.fine_gathering_active = False
        self.tool_calibration_active = False

        # Create main frame with tabs
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(expand=True, fill="both", padx=10, pady=10)

        # Create tabs
        self.setup_tab = ttk.Frame(self.notebook)
        self.registration_tab = ttk.Frame(self.notebook)
        self.calibration_tab = ttk.Frame(self.notebook)
        self.streaming_tab = ttk.Frame(self.notebook)
        self.diagnostic_tab = ttk.Frame(self.notebook)

        self.notebook.add(self.setup_tab, text="Setup")
        self.notebook.add(self.registration_tab, text="Registration")
        self.notebook.add(self.calibration_tab, text="Tool Calibration")
        self.notebook.add(self.streaming_tab, text="Streaming")
        self.notebook.add(self.diagnostic_tab, text="Diagnostics")

        # Initialize tabs
        self.init_setup_tab()
        self.init_registration_tab()
        self.init_calibration_tab()
        self.init_streaming_tab()
        self.init_diagnostic_tab()

        # Status bar
        self.status_frame = ttk.Frame(root)
        self.status_frame.pack(fill="x", padx=10, pady=5)

        self.status_label = ttk.Label(self.status_frame, text="Status: Not connected")
        self.status_label.pack(side="left")

        self.server_status_label = ttk.Label(self.status_frame, text="Server: Unknown")
        self.server_status_label.pack(side="right")

        # Start periodic status updates
        self.update_status()

    def init_setup_tab(self):
        # Server connection frame
        server_frame = ttk.LabelFrame(self.setup_tab, text="Server Connection")
        server_frame.pack(fill="x", padx=10, pady=10)

        ttk.Label(server_frame, text="Server URL:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.url_entry = ttk.Entry(server_frame, width=40)
        self.url_entry.insert(0, self.base_url)
        self.url_entry.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        ttk.Button(server_frame, text="Connect", command=self.connect_to_server).grid(row=0, column=2, padx=5, pady=5)

        # NDI Initialization frame
        ndi_frame = ttk.LabelFrame(self.setup_tab, text="NDI Tracker")
        ndi_frame.pack(fill="x", padx=10, pady=10)

        self.force_restart_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(ndi_frame, text="Force restart", variable=self.force_restart_var).grid(row=0, column=0, padx=5,
                                                                                               pady=5)

        ttk.Button(ndi_frame, text="Initialize NDI Tracker", command=self.initialize_ndi).grid(row=0, column=1, padx=5,
                                                                                               pady=5)

        # Tool status frame
        tool_frame = ttk.LabelFrame(self.setup_tab, text="Tool Status")
        tool_frame.pack(fill="x", padx=10, pady=10)

        ttk.Button(tool_frame, text="Check Tools", command=self.check_tools).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(tool_frame, text="Find Reference", command=self.find_reference).grid(row=0, column=1, padx=5, pady=5)

        # Log frame
        log_frame = ttk.LabelFrame(self.setup_tab, text="Logs")
        log_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.log_text = scrolledtext.ScrolledText(log_frame, height=10)
        self.log_text.pack(fill="both", expand=True, padx=5, pady=5)

        ttk.Button(log_frame, text="Clear Log", command=lambda: self.log_text.delete(1.0, tk.END)).pack(side="right",
                                                                                                        padx=5, pady=5)

    def init_registration_tab(self):
        # Create notebook for registration subtabs
        reg_notebook = ttk.Notebook(self.registration_tab)
        reg_notebook.pack(fill="both", expand=True)

        # Coarse registration tab
        coarse_tab = ttk.Frame(reg_notebook)
        reg_notebook.add(coarse_tab, text="Coarse Registration")

        # Coarse registration UI
        coarse_frame = ttk.LabelFrame(coarse_tab, text="Coarse Registration Points")
        coarse_frame.pack(fill="x", padx=10, pady=10)

        # Point selection
        point_frame = ttk.Frame(coarse_frame)
        point_frame.pack(fill="x", padx=5, pady=5)

        ttk.Label(point_frame, text="Point Number:").grid(row=0, column=0, padx=5, pady=5)
        self.point_number_var = tk.IntVar(value=1)
        self.point_spinbox = ttk.Spinbox(point_frame, from_=1, to=10, textvariable=self.point_number_var, width=5)
        self.point_spinbox.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(point_frame, text="Unity Point (X, Y, Z):").grid(row=0, column=2, padx=5, pady=5)
        self.unity_point_x = ttk.Entry(point_frame, width=8)
        self.unity_point_x.insert(0, "0.0")
        self.unity_point_x.grid(row=0, column=3, padx=2, pady=5)

        self.unity_point_y = ttk.Entry(point_frame, width=8)
        self.unity_point_y.insert(0, "0.0")
        self.unity_point_y.grid(row=0, column=4, padx=2, pady=5)

        self.unity_point_z = ttk.Entry(point_frame, width=8)
        self.unity_point_z.insert(0, "0.0")
        self.unity_point_z.grid(row=0, column=5, padx=2, pady=5)

        ttk.Button(point_frame, text="Set Point", command=self.set_coarse_point).grid(row=0, column=6, padx=5, pady=5)

        # Buttons for coarse registration
        button_frame = ttk.Frame(coarse_frame)
        button_frame.pack(fill="x", padx=5, pady=5)

        ttk.Button(button_frame, text="Reset Coarse Points", command=self.reset_coarse_points).pack(side="left", padx=5,
                                                                                                    pady=5)

        self.visualize_coarse_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(button_frame, text="Visualize", variable=self.visualize_coarse_var).pack(side="left", padx=5,
                                                                                                 pady=5)

        ttk.Button(button_frame, text="Perform Coarse Registration", command=self.perform_coarse_registration).pack(
            side="left", padx=5, pady=5)

        ttk.Button(button_frame, text="Load Last Coarse Transform", command=self.load_last_coarse_transform).pack(
            side="right", padx=5, pady=5)

        # Fine registration tab
        fine_tab = ttk.Frame(reg_notebook)
        reg_notebook.add(fine_tab, text="Fine Registration")

        # Fine registration UI
        fine_frame = ttk.LabelFrame(fine_tab, text="Fine Registration")
        fine_frame.pack(fill="x", padx=10, pady=10)

        # Fine gathering controls
        gather_frame = ttk.Frame(fine_frame)
        gather_frame.pack(fill="x", padx=5, pady=5)

        ttk.Label(gather_frame, text="Frequency (Hz):").grid(row=0, column=0, padx=5, pady=5)
        self.fine_freq_var = tk.IntVar(value=60)
        ttk.Spinbox(gather_frame, from_=10, to=100, textvariable=self.fine_freq_var, width=5).grid(row=0, column=1,
                                                                                                   padx=5, pady=5)

        # Changed to separate start/stop buttons
        ttk.Button(gather_frame, text="Start Fine Gathering", command=self.start_fine_gathering).grid(row=0, column=2,
                                                                                                      padx=5, pady=5)
        ttk.Button(gather_frame, text="Stop Fine Gathering", command=self.stop_fine_gathering).grid(row=0, column=3,
                                                                                                    padx=5, pady=5)
        ttk.Button(gather_frame, text="Reset Fine Gathering", command=self.reset_fine_gathering).grid(row=0, column=4,
                                                                                                      padx=5, pady=5)

        # Fine registration controls
        reg_frame = ttk.Frame(fine_frame)
        reg_frame.pack(fill="x", padx=5, pady=5)

        ttk.Label(reg_frame, text="ID:").grid(row=0, column=0, padx=5, pady=5)
        self.fine_id_var = tk.IntVar(value=0)
        ttk.Spinbox(reg_frame, from_=0, to=10, textvariable=self.fine_id_var, width=5).grid(row=0, column=1, padx=5,
                                                                                            pady=5)

        ttk.Label(reg_frame, text="Downsample:").grid(row=0, column=2, padx=5, pady=5)
        self.downsample_var = tk.DoubleVar(value=1.0)
        ttk.Spinbox(reg_frame, from_=0.1, to=1.0, increment=0.1, textvariable=self.downsample_var, width=5).grid(row=0,
                                                                                                                 column=3,
                                                                                                                 padx=5,
                                                                                                                 pady=5)

        self.visualize_fine_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(reg_frame, text="Visualize", variable=self.visualize_fine_var).grid(row=0, column=4, padx=5,
                                                                                            pady=5)

        ttk.Button(reg_frame, text="Perform Fine Registration", command=self.perform_fine_registration).grid(row=0,
                                                                                                             column=5,
                                                                                                             padx=5,
                                                                                                             pady=5)

        ttk.Button(reg_frame, text="Load Last Fine Transform", command=self.load_last_fine_transform).grid(row=0,
                                                                                                           column=6,
                                                                                                           padx=5,
                                                                                                           pady=5)

        # Status frame
        status_frame = ttk.LabelFrame(fine_tab, text="Fine Registration Status")
        status_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.fine_status_text = scrolledtext.ScrolledText(status_frame, height=8)
        self.fine_status_text.pack(fill="both", expand=True, padx=5, pady=5)

        ttk.Button(status_frame, text="Get Status", command=self.get_fine_points_status).pack(side="right", padx=5,
                                                                                              pady=5)

    def init_calibration_tab(self):
        # Tool calibration frame
        calib_frame = ttk.LabelFrame(self.calibration_tab, text="Tool Tip Calibration")
        calib_frame.pack(fill="x", padx=10, pady=10)

        # Device selection
        device_frame = ttk.Frame(calib_frame)
        device_frame.pack(fill="x", padx=5, pady=5)

        ttk.Label(device_frame, text="Device:").pack(side="left", padx=5, pady=5)
        self.device_var = tk.IntVar(value=0)
        ttk.Spinbox(device_frame, from_=0, to=10, textvariable=self.device_var, width=5).pack(side="left", padx=5,
                                                                                              pady=5)

        self.force_stop_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(device_frame, text="Force Stop Streaming", variable=self.force_stop_var).pack(side="left",
                                                                                                      padx=5, pady=5)

        # Calibration controls - Changed to separate buttons
        control_frame = ttk.Frame(calib_frame)
        control_frame.pack(fill="x", padx=5, pady=5)

        ttk.Button(control_frame, text="Start Tool Calibration", command=self.start_tool_calibration).pack(side="left",
                                                                                                           padx=5,
                                                                                                           pady=5)
        ttk.Button(control_frame, text="Stop Tool Calibration", command=self.stop_tool_calibration).pack(side="left",
                                                                                                         padx=5, pady=5)

        self.visualize_calib_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(control_frame, text="Visualize", variable=self.visualize_calib_var).pack(side="left", padx=5,
                                                                                                 pady=5)

        ttk.Button(control_frame, text="Calibrate Tool", command=self.calibrate_tool).pack(side="left", padx=5, pady=5)

        ttk.Button(control_frame, text="Load From File", command=self.load_tool_transformations).pack(side="right",
                                                                                                      padx=5, pady=5)

        # Probe touchpoint frame
        touch_frame = ttk.LabelFrame(self.calibration_tab, text="Probe Touchpoint")
        touch_frame.pack(fill="x", padx=10, pady=10)

        probe_frame = ttk.Frame(touch_frame)
        probe_frame.pack(fill="x", padx=5, pady=5)

        ttk.Label(probe_frame, text="Probe Index:").grid(row=0, column=0, padx=5, pady=5)
        self.probe_idx_var = tk.IntVar(value=0)
        ttk.Spinbox(probe_frame, from_=0, to=10, textvariable=self.probe_idx_var, width=5).grid(row=0, column=1, padx=5,
                                                                                                pady=5)

        ttk.Label(probe_frame, text="Endoscope Index:").grid(row=0, column=2, padx=5, pady=5)
        self.endoscope_idx_var = tk.IntVar(value=2)
        ttk.Spinbox(probe_frame, from_=0, to=10, textvariable=self.endoscope_idx_var, width=5).grid(row=0, column=3,
                                                                                                    padx=5, pady=5)

        ttk.Button(probe_frame, text="Get Probe Touchpoint", command=self.get_probe_touchpoint).grid(row=0, column=4,
                                                                                                     padx=5, pady=5)

        # Status frame
        status_frame = ttk.LabelFrame(self.calibration_tab, text="Calibration Status")
        status_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.calib_status_text = scrolledtext.ScrolledText(status_frame, height=10)
        self.calib_status_text.pack(fill="both", expand=True, padx=5, pady=5)

        ttk.Button(status_frame, text="Get Status", command=self.get_tool_calibration_status).pack(side="right", padx=5,
                                                                                                   pady=5)

    def init_streaming_tab(self):
        # Client IP frame
        ip_frame = ttk.LabelFrame(self.streaming_tab, text="Client IP Configuration")
        ip_frame.pack(fill="x", padx=10, pady=10)

        ttk.Label(ip_frame, text="Client IP:").grid(row=0, column=0, padx=5, pady=5)
        self.client_ip_entry = ttk.Entry(ip_frame, width=20)
        self.client_ip_entry.insert(0, "192.168.31.69")
        self.client_ip_entry.grid(row=0, column=1, padx=5, pady=5)

        ttk.Button(ip_frame, text="Set Client IP", command=self.set_client_ip).grid(row=0, column=2, padx=5, pady=5)
        ttk.Button(ip_frame, text="Get Client IP", command=self.get_client_ip).grid(row=0, column=3, padx=5, pady=5)

        # Streaming controls frame
        streaming_frame = ttk.LabelFrame(self.streaming_tab, text="Streaming Controls")
        streaming_frame.pack(fill="x", padx=10, pady=10)

        control_frame = ttk.Frame(streaming_frame)
        control_frame.pack(fill="x", padx=5, pady=5)

        ttk.Label(control_frame, text="Port:").grid(row=0, column=0, padx=5, pady=5)
        self.port_var = tk.IntVar(value=11111)
        ttk.Spinbox(control_frame, from_=1024, to=65535, textvariable=self.port_var, width=8).grid(row=0, column=1,
                                                                                                   padx=5, pady=5)

        ttk.Label(control_frame, text="Frequency (Hz):").grid(row=0, column=2, padx=5, pady=5)
        self.stream_freq_var = tk.IntVar(value=30)
        ttk.Spinbox(control_frame, from_=1, to=100, textvariable=self.stream_freq_var, width=5).grid(row=0, column=3,
                                                                                                     padx=5, pady=5)

        self.force_stop_calib_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(control_frame, text="Force Stop Calibration", variable=self.force_stop_calib_var).grid(row=0,
                                                                                                               column=4,
                                                                                                               padx=5,
                                                                                                               pady=5)

        ttk.Label(control_frame, text="Raw Frequency:").grid(row=0, column=5, padx=5, pady=5)
        self.raw_freq_var = tk.IntVar(value=10)
        ttk.Spinbox(control_frame, from_=1, to=100, textvariable=self.raw_freq_var, width=5).grid(row=0, column=6,
                                                                                                  padx=5, pady=5)

        button_frame = ttk.Frame(streaming_frame)
        button_frame.pack(fill="x", padx=5, pady=5)

        self.stream_button = ttk.Button(button_frame, text="Start Streaming", command=self.toggle_streaming)
        self.stream_button.pack(side="left", padx=5, pady=5)

        ttk.Button(button_frame, text="Get Streaming Status", command=self.get_streaming_status).pack(side="left",
                                                                                                      padx=5, pady=5)
        ttk.Button(button_frame, text="Get Latest Position", command=self.get_latest_position).pack(side="left", padx=5,
                                                                                                    pady=5)

        # Raw streaming frame
        raw_frame = ttk.LabelFrame(self.streaming_tab, text="Raw NDI Streaming")
        raw_frame.pack(fill="x", padx=10, pady=10)

        raw_control_frame = ttk.Frame(raw_frame)
        raw_control_frame.pack(fill="x", padx=5, pady=5)

        ttk.Label(raw_control_frame, text="Raw Frequency:").pack(side="left", padx=5, pady=5)
        self.raw_stream_freq_var = tk.IntVar(value=10)
        ttk.Spinbox(raw_control_frame, from_=1, to=100, textvariable=self.raw_stream_freq_var, width=5).pack(
            side="left", padx=5, pady=5)

        ttk.Button(raw_control_frame, text="Start Raw Streaming", command=self.start_raw_streaming).pack(side="left",
                                                                                                         padx=5, pady=5)
        ttk.Button(raw_control_frame, text="Set Raw Frequency", command=self.set_raw_streaming_frequency).pack(
            side="left", padx=5, pady=5)
        ttk.Button(raw_control_frame, text="Stop Raw Streaming", command=self.stop_raw_streaming).pack(side="left",
                                                                                                       padx=5, pady=5)

        # Streaming data display
        data_frame = ttk.LabelFrame(self.streaming_tab, text="Streaming Data")
        data_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.stream_data_text = scrolledtext.ScrolledText(data_frame, height=10)
        self.stream_data_text.pack(fill="both", expand=True, padx=5, pady=5)

    def init_diagnostic_tab(self):
        # Server info frame
        info_frame = ttk.LabelFrame(self.diagnostic_tab, text="Server Information")
        info_frame.pack(fill="x", padx=10, pady=10)

        self.server_info_text = scrolledtext.ScrolledText(info_frame, height=10)
        self.server_info_text.pack(fill="both", expand=True, padx=5, pady=5)

        ttk.Button(info_frame, text="Refresh Server Info", command=self.get_server_info).pack(side="right", padx=5,
                                                                                              pady=5)

        # Visualization frame
        vis_frame = ttk.LabelFrame(self.diagnostic_tab, text="Visualization")
        vis_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Create a 3D plot
        self.fig = plt.figure(figsize=(5, 4))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_title("Position Visualization")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.grid(True)

        self.canvas = FigureCanvasTkAgg(self.fig, master=vis_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        control_frame = ttk.Frame(vis_frame)
        control_frame.pack(fill="x", padx=5, pady=5)

        ttk.Button(control_frame, text="Start Live Visualization", command=self.toggle_visualization).pack(side="left",
                                                                                                           padx=5,
                                                                                                           pady=5)

    # Helper functions for API calls
    def api_call(self, endpoint, method="get", data=None, params=None):
        url = f"{self.base_url}{endpoint}"
        try:
            if method.lower() == "get":
                response = requests.get(url, params=params)
            elif method.lower() == "post":
                response = requests.post(url, json=data, params=params)

            if response.status_code >= 200 and response.status_code < 300:
                return response.json()
            else:
                self.log(f"API Error ({response.status_code}): {response.text}")
                return {"status": "error", "details": response.text}
        except Exception as e:
            self.log(f"Connection error: {str(e)}")
            return {"status": "error", "details": str(e)}

    def log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)

    # Server setup functions
    def connect_to_server(self):
        self.base_url = self.url_entry.get().rstrip("/")
        self.log(f"Connecting to server: {self.base_url}")

        response = self.api_call("/")
        if response.get("status") != "error":
            self.log(f"Connected to server: {response}")
            self.status_label.config(text="Status: Connected")
            self.server_status_label.config(text=f"Server: {response.get('application', 'Unknown')}")
        else:
            self.status_label.config(text="Status: Connection failed")

    def update_status(self):
        try:
            response = self.api_call("/")
            if response.get("status") != "error":
                self.server_status_label.config(text=f"Server: Running")

                # Update streaming status
                self.streaming_active = response.get("streaming_active", False)
                if self.streaming_active:
                    self.stream_button.config(text="Stop Streaming")
                else:
                    self.stream_button.config(text="Start Streaming")
            else:
                self.server_status_label.config(text="Server: Not responding")
        except:
            self.server_status_label.config(text="Server: Connection error")

        # Schedule next update
        self.root.after(5000, self.update_status)

    def initialize_ndi(self):
        force_restart = self.force_restart_var.get()
        self.log(f"Initializing NDI tracker (force_restart={force_restart})")

        response = self.api_call("/initialize_ndi", "post", {"force_restart": force_restart})

        if response.get("status") == "success" or response.get("status") == "already_initialized":
            self.log(f"NDI tracker initialized: {response.get('message')}")

            if "details" in response and "tools_detected" in response["details"]:
                self.log(f"Detected {response['details']['tools_detected']} tool(s)")
        else:
            self.log(f"Failed to initialize NDI tracker: {response.get('message')}")

    def check_tools(self):
        self.log("Checking tools visibility...")
        response = self.api_call("/check_tools", "post")

        if response.get("status") != "error":
            self.log(f"Tool visibility status: {json.dumps(response, indent=2)}")

            # Format nicely in the log
            for tool, visible in response.items():
                self.log(f" {tool}: {'Visible' if visible else 'Not visible'}")
        else:
            self.log(f"Failed to check tools: {response.get('details')}")

    def find_reference(self):
        self.log("Finding reference...")
        response = self.api_call("/find_reference", "post", {"max_tries": 50, "wait_time": 1.0})

        if response.get("status") == "success":
            self.log(f"Reference found: {response.get('message')}")
        else:
            self.log(f"Failed to find reference: {response.get('details')}")

    # Registration functions
    def set_coarse_point(self):
        try:
            point_number = int(self.point_number_var.get())
            unity_point = [
                float(self.unity_point_x.get()),
                float(self.unity_point_y.get()),
                float(self.unity_point_z.get())
            ]

            self.log(f"Setting coarse point {point_number} at Unity position {unity_point}")

            data = {
                "unity_point": unity_point,
                "point_number": point_number
            }

            response = self.api_call("/set_coarse_point", "post", data)

            if response.get("status") == "success":
                self.log(f"Coarse point {point_number} set successfully")
                self.log(f"NDI point: {response.get('ndi_point')}")

                # Increment the point number for convenience
                self.point_number_var.set(point_number + 1)
            else:
                self.log(f"Failed to set coarse point: {response.get('details')}")
        except ValueError as e:
            self.log(f"Invalid input: {str(e)}")

    def reset_coarse_points(self):
        self.log("Resetting all coarse points...")
        response = self.api_call("/reset_coarse_points", "post")

        if response.get("status") == "success":
            self.log("Coarse points reset successfully")
            self.point_number_var.set(1)
        else:
            self.log(f"Failed to reset coarse points: {response.get('details')}")

    def perform_coarse_registration(self):
        visualize = self.visualize_coarse_var.get()
        self.log(f"Performing coarse registration (visualize={visualize})...")

        response = self.api_call("/coarse_register", "post", None, {"visualize": visualize})

        if response.get("status") == "success":
            self.log("Coarse registration successful")
            self.log(f"RMS error: {response.get('rms_error')}")
            self.log(f"Transformation matrix: {json.dumps(response.get('transformation_matrix'), indent=2)}")
        else:
            self.log(f"Coarse registration failed: {response.get('details')}")

    def load_last_coarse_transform(self):
        self.log("Loading last saved coarse transformation...")
        response = self.api_call("/load_last_coarse_transform", "post")

        if response.get("status") == "success":
            self.log("Coarse transformation loaded successfully")
            self.log(f"Transformation: {response.get('transformation')}")
        else:
            self.log(f"Failed to load coarse transformation: {response.get('details')}")

    # Split the fine gathering functions into start and stop
    def start_fine_gathering(self):
        frequency = self.fine_freq_var.get()
        self.log(f"Starting fine point gathering at {frequency} Hz...")

        response = self.api_call("/start_fine_gather", "post",
                                 {"frequency": frequency, "streaming_raw_frequncy": self.raw_freq_var.get()})

        if response.get("status") == "started":
            self.fine_gathering_active = True
            self.log("Fine point gathering started")
        else:
            self.log(f"Failed to start fine gathering: {response.get('details')}")

    def stop_fine_gathering(self):
        self.log("Stopping fine point gathering...")

        response = self.api_call("/end_fine_gather", "post",
                                 {"streaming_raw_frequncy": self.raw_freq_var.get()})

        if response.get("status") == "success":
            self.fine_gathering_active = False
            self.log(f"Fine gathering stopped, collected {response.get('points_collected')} points")
        else:
            self.log(f"Failed to stop fine gathering: {response.get('details')}")

    def reset_fine_gathering(self):
        self.log("Resetting fine gathering data...")
        response = self.api_call("/reset_fine_gather", "post")

        if response.get("results") == "OK":
            self.log("Fine gathering data reset successfully")
        else:
            self.log(f"Failed to reset fine gathering: {response}")

    def get_fine_points_status(self):
        self.log("Getting fine points status...")
        response = self.api_call("/get_fine_points_status", "get")

        self.fine_status_text.delete(1.0, tk.END)
        self.fine_status_text.insert(tk.END, json.dumps(response, indent=2))

        if "gathering_active" in response:
            self.fine_gathering_active = response["gathering_active"]

    def perform_fine_registration(self):
        id_value = self.fine_id_var.get()
        downsample = self.downsample_var.get()
        visualize = self.visualize_fine_var.get()

        self.log(f"Performing fine registration (id={id_value}, downsample={downsample}, visualize={visualize})...")

        params = {
            "id": id_value,
            "downsample_factor": downsample,
            "visualize": visualize
        }

        response = self.api_call("/fine_register", "post", None, params)

        if response.get("status") == "success":
            self.log("Fine registration successful")
            self.log(f"RMS error: {response.get('rms_error')}")

            # Show in status text
            self.fine_status_text.delete(1.0, tk.END)
            self.fine_status_text.insert(tk.END, json.dumps(response, indent=2))
        else:
            self.log(f"Fine registration failed: {response.get('details')}")

    def load_last_fine_transform(self):
        self.log("Loading last saved fine transformation...")
        response = self.api_call("/load_last_fine_transform", "post")

        if response.get("status") == "success":
            self.log("Fine transformation loaded successfully")
            self.log(f"Transformation: {response.get('transformation')}")
        else:
            self.log(f"Failed to load fine transformation: {response.get('details')}")

    # Split the tool calibration functions into start and stop
    def start_tool_calibration(self):
        device = self.device_var.get()
        force_stop = self.force_stop_var.get()

        self.log(f"Starting tool calibration (device={device}, force_stop_streaming={force_stop})...")

        response = self.api_call("/start_tool_calibration", "post",
                                 {"force_stop_streaming": force_stop, "device": device})

        if response.get("status") == "started":
            self.tool_calibration_active = True
            self.log("Tool calibration started")
        else:
            self.log(f"Failed to start tool calibration: {response.get('details')}")

    def stop_tool_calibration(self):
        self.log("Stopping tool calibration...")

        response = self.api_call("/end_tool_calibration", "post")

        if response.get("status") == "success":
            self.tool_calibration_active = False
            self.log(f"Tool calibration stopped, collected {response.get('transformations_collected')} transformations")
        else:
            self.log(f"Failed to stop tool calibration: {response.get('details')}")

    def calibrate_tool(self):
        visualize = self.visualize_calib_var.get()
        self.log(f"Calibrating tool (visualize={visualize})...")

        response = self.api_call("/calibrate_tool", "post", {"visualize": visualize})

        if response.get("status") == "success":
            self.log("Tool calibration successful")
            self.log(f"Tool tip vector: {response.get('tool_tip_vector')}")
            self.log(f"RMS error: {response.get('rms_error')}")

            # Display in status text
            self.calib_status_text.delete(1.0, tk.END)
            self.calib_status_text.insert(tk.END, json.dumps(response, indent=2))
        else:
            self.log(f"Tool calibration failed: {response.get('details')}")

    def get_tool_calibration_status(self):
        self.log("Getting tool calibration status...")
        response = self.api_call("/get_tool_calibration_status", "get")

        self.calib_status_text.delete(1.0, tk.END)
        self.calib_status_text.insert(tk.END, json.dumps(response, indent=2))

        if "calibration_active" in response:
            self.tool_calibration_active = response["calibration_active"]

    def load_tool_transformations(self):
        filename = filedialog.askopenfilename(
            title="Select tool transformations file",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )

        if filename:
            self.log(f"Loading tool transformations from {filename}...")
            response = self.api_call("/load_tool_transformations_from_file", "post", {"filename": filename})

            if response.get("status") == "success":
                self.log(f"Loaded {response.get('transformations_loaded')} transformations successfully")
            else:
                self.log(f"Failed to load transformations: {response.get('details')}")

    def get_probe_touchpoint(self):
        probe_idx = self.probe_idx_var.get()
        endoscope_idx = self.endoscope_idx_var.get()

        self.log(f"Getting probe touchpoint (probe_idx={probe_idx}, endoscope_idx={endoscope_idx})...")

        response = self.api_call("/get_probe_touchpoint", "post",
                                 {"probe_idx": probe_idx, "endoscope_idx": endoscope_idx})

        if response.get("status") == "success":
            self.log("Probe touchpoint calculated successfully")
            self.log(f"Touch point: {response.get('touch_point')}")

            # Display in status text
            self.calib_status_text.delete(1.0, tk.END)
            self.calib_status_text.insert(tk.END, json.dumps(response, indent=2))
        else:
            self.log(f"Failed to get probe touchpoint: {response.get('details')}")

    # Streaming functions
    def set_client_ip(self):
        ip = self.client_ip_entry.get()
        self.log(f"Setting client IP to {ip}...")

        response = self.api_call("/set_client_ip", "post", ip)

        if response.get("status") == "success":
            self.log(f"Client IP set to {response.get('client_ip')}")
        else:
            self.log(f"Failed to set client IP: {response.get('details')}")

    def get_client_ip(self):
        self.log("Getting client IP...")
        response = self.api_call("/get_client_ip", "get")

        if "client_ip" in response:
            self.log(f"Current client IP: {response.get('client_ip')}")
            self.client_ip_entry.delete(0, tk.END)
            self.client_ip_entry.insert(0, response.get('client_ip'))

            if response.get("streaming_active"):
                self.log(f"Streaming is active on port {response.get('streaming_port')}")
        else:
            self.log("Failed to get client IP")

    def toggle_streaming(self):
        if not self.streaming_active:
            # Start streaming
            port = self.port_var.get()
            frequency = self.stream_freq_var.get()
            force_stop = self.force_stop_calib_var.get()
            raw_freq = self.raw_freq_var.get()

            self.log(f"Starting streaming on port {port} at {frequency} Hz (raw freq: {raw_freq})...")

            response = self.api_call("/start_streaming", "post",
                                     {"port": port, "frequency": frequency,
                                      "force_stop_calibration": force_stop,
                                      "streaming_raw_frequency": raw_freq})

            if response.get("status") == "started":
                self.streaming_active = True
                self.stream_button.config(text="Stop Streaming")
                self.log(f"Streaming started to {response.get('target_ip')} on port {response.get('port')}")
            else:
                self.log(f"Failed to start streaming: {response.get('details')}")
        else:
            # Stop streaming
            raw_freq = self.raw_freq_var.get()
            self.log("Stopping streaming...")

            response = self.api_call("/stop_streaming", "post", {"streaming_raw_frequncy": raw_freq})

            if response.get("status") == "stopped":
                self.streaming_active = False
                self.stream_button.config(text="Start Streaming")
                self.log("Streaming stopped")
            else:
                self.log(f"Failed to stop streaming: {response.get('details')}")

    def get_streaming_status(self):
        self.log("Getting streaming status...")
        response = self.api_call("/streaming_status", "get")

        self.stream_data_text.delete(1.0, tk.END)
        self.stream_data_text.insert(tk.END, json.dumps(response, indent=2))

        if "active" in response:
            self.streaming_active = response["active"]
            if self.streaming_active:
                self.stream_button.config(text="Stop Streaming")
                self.log(f"Streaming is active to {response.get('target_ip')} on port {response.get('port')}")
            else:
                self.stream_button.config(text="Start Streaming")
                self.log("Streaming is not active")

    def get_latest_position(self):
        self.log("Getting latest position data...")
        response = self.api_call("/get_latest_position", "get")

        if response.get("status") == "success":
            self.log("Got latest position data")
            self.stream_data_text.delete(1.0, tk.END)
            self.stream_data_text.insert(tk.END, json.dumps(response.get("data"), indent=2))

            # Update visualization
            if "data" in response and "position" in response["data"]:
                pos = response["data"]["position"]
                self.ax.clear()
                self.ax.scatter(pos[0], pos[1], pos[2], color='red', marker='o', s=100)
                self.ax.set_xlabel("X")
                self.ax.set_ylabel("Y")
                self.ax.set_zlabel("Z")
                self.ax.set_title("Latest Position")

                # Set reasonable axis limits
                self.ax.set_xlim([-100, 100])
                self.ax.set_ylim([-100, 100])
                self.ax.set_zlim([-100, 100])

                self.canvas.draw()
        else:
            self.log(f"Failed to get position data: {response.get('message')}")

    def start_raw_streaming(self):
        self.log("Starting raw streaming...")
        response = self.api_call("/start_raw_streaming", "post")

        if response.get("status") != "error":
            self.log("Raw streaming started")
        else:
            self.log(f"Failed to start raw streaming: {response.get('details')}")

    def set_raw_streaming_frequency(self):
        frequency = self.raw_stream_freq_var.get()
        self.log(f"Setting raw streaming frequency to {frequency} Hz...")

        response = self.api_call("/set_raw_streaming_frequency", "post", frequency)

        if response.get("status") != "error":
            self.log(f"Raw streaming frequency set to {frequency} Hz")
        else:
            self.log(f"Failed to set raw streaming frequency: {response.get('details')}")

    def stop_raw_streaming(self):
        self.log("Stopping raw streaming...")
        response = self.api_call("/stop_raw_streaming", "post")

        if response.get("status") != "error":
            self.log("Raw streaming stopped")
        else:
            self.log(f"Failed to stop raw streaming: {response.get('details')}")

    # Diagnostic functions
    def get_server_info(self):
        self.log("Getting server information...")
        response = self.api_call("/", "get")

        if response.get("status") != "error":
            self.server_info_text.delete(1.0, tk.END)
            self.server_info_text.insert(tk.END, json.dumps(response, indent=2))
            self.log("Server information retrieved")
        else:
            self.log(f"Failed to get server information: {response.get('details')}")

    def toggle_visualization(self):
        # This would start a thread that periodically fetches position data and updates the plot
        self.log("Live visualization not implemented yet")
        messagebox.showinfo("Not Implemented", "Live visualization feature is not implemented yet.")


if __name__ == "__main__":
    root = tk.Tk()
    app = NDITrackerUI(root)
    root.mainloop()
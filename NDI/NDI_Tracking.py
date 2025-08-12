import json

import numpy as np
from sksurgerynditracker.nditracker import NDITracker


class NDI_Tracking():
    def __init__(self, config, args):
        self.SETTINGS = config["device_params"]
        self.config = config
        self.args = args
        self.last_reference = None

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
                raise Exception("could not detect reference! Reference is required in this mode, if you do not want the reference try reference_required = false")
        if self.last_reference is not None:
            self.last_reference = tracking[self.config["tool_types"]["reference"]]
        try:
            probe_relative_to_reference = np.linalg.inv(self.last_reference) @ tracking[self.config["tool_types"]["probe"]]
            endoscope_relative_to_reference = np.linalg.inv(self.last_reference) @ tracking[self.config["tool_types"]["endoscope"]]
            return [probe_relative_to_reference, tracking[self.config["tool_types"]["reference"]],endoscope_relative_to_reference]
        except Exception as e:
            print("Could not detect tool!")
            return tracking


    def stop(self):
        self.TRACKER.stop_tracking()
        self.TRACKER.close()


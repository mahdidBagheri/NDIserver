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

            self.last_reference = tracking[self.config["tool_types"]["reference"]]

            transforms = [None,self.last_reference,None]
            if tracking[self.config["tool_types"]["probe"]] is None:
                print("Could not detect probe!")
            else:
                transforms[self.config["tool_types"]["probe"]] = np.linalg.inv(self.last_reference) @ tracking[self.config["tool_types"]["probe"]]

            if tracking[self.config["tool_types"]["endoscope"]] is None:
                print("Could not detect endoscope!")
            else:
                transforms[self.config["tool_types"]["endoscope"]] = np.linalg.inv(self.last_reference) @ tracking[self.config["tool_types"]["endoscope"]]

            return transforms

        else:
            return tracking


    def stop(self):
        self.TRACKER.stop_tracking()
        self.TRACKER.close()


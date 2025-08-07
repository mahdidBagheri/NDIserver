
import json

import numpy as np
from NDI.NDI_Tracking import NDI_Tracking
last_reference = None

class NDI_Tracking_Simulator(NDI_Tracking):
    def __init__(self, config):
        super().__init__(config)

    def start(self):
        print("[START]: This is the local mode")

    def GetPosition(self):
        # port_handles, timestamps, framenumbers, tracking, quality = self.TRACKER.get_frame()

        tracking = [np.random.rand(4, 4), np.random.rand(4, 4), np.random.rand(4, 4)]
        for t in tracking:
          print(t)

        try:
            global last_reference
            if last_reference is None :
                if np.isnan(tracking[1]).any():
                    print("could not detect reference!")
                    return tracking
                else:
                    last_reference = tracking[1]



            if not np.isnan(tracking[1]).any():
                last_reference = tracking[1]
                probe_relative_to_reference = np.linalg.inv(last_reference) @ tracking[0]
                endoscope_relative_to_reference = np.linalg.inv(last_reference) @ tracking[2]
            else:
                probe_relative_to_reference = np.linalg.inv(tracking[1]) @ tracking[0]
                endoscope_relative_to_reference = np.linalg.inv(tracking[1]) @ tracking[2]
            return [probe_relative_to_reference, tracking[1],endoscope_relative_to_reference]
        except Exception as e:
            print("Could not detect reference!")
            return tracking


    def stop(self):
        print("[STOP]: This is the local mode")


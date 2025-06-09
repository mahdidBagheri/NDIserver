import numpy as np
from sksurgerynditracker.nditracker import NDITracker

last_reference = None

class NDI_Tracking():
    def __init__(self):
        self.SETTINGS = {
    "tracker type": "polaris",
    "romfiles" : ["C:/Users/Parsiss99/PlusApp-2.8.0.20191105-Win64/config/NdiToolDefinitions/8700340.rom",
                  "C:/Users/Parsiss99/AppData/Local/Parsiss/FilesOfTrackers/PolarisTrackerFiles/Replacable-Mayfield-Reference/Replacable-Mayfield-Reference.rom",
                  ""]

        }


    def start(self):
        self.TRACKER = NDITracker(self.SETTINGS)
        self.TRACKER.start_tracking()

    def GetPosition(self):
        port_handles, timestamps, framenumbers, tracking, quality = self.TRACKER.get_frame()
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
        self.TRACKER.stop_tracking()
        self.TRACKER.close()

ndi_tracking = NDI_Tracking()
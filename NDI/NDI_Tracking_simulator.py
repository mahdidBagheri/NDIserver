import numpy as np
from NDI.NDI_Tracking import NDI_Tracking
from NDI.PathSimulators.PathGenerator import PathGenerator

last_reference = None

class NDI_Tracking_Simulator(NDI_Tracking):
    def __init__(self, config, args):
        super().__init__(config, args)
        self.path_gen = PathGenerator()
    def start(self):
        print("[START]: This is the local mode")

    def get_tracking(self):
        tracking = self.path_gen.get_transformation_matrices()
        for i, t in enumerate(tracking):
            if np.isnan(tracking[i]).any():
                tracking[i] = None
            print(t)
        return tracking

    def stop(self):
        print("[STOP]: This is the local mode")


import json
import os
import numpy as np


class ConfigManager:
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = None
        self.load_configuration()

    def load_configuration(self):
        """Load configuration from JSON file"""
        try:
            with open(self.config_path, 'r') as file:
                self.config = json.load(file)
            print(f"Loaded configuration from: {self.config_path}")
        except Exception as e:
            print(f"Failed to load configuration: {e}")
            # Use default configuration
            self.config = {
                "tool_types": {"probe": 0, "reference": 1, "endoscope": 2},
                "CT_pc": "",
                "probe_tip_vector": [0.0, 0.0, -161.148433, 1.0]
            }

    def get_config(self):
        """Get the configuration dictionary"""
        return self.config

    def get_tool_types(self):
        """Get tool types mapping"""
        return self.config.get('tool_types', {})

    def get_probe_tip_vector(self):
        """Get probe tip vector"""
        return self.config.get('probe_tip_vector', [0.0, 0.0, -161.148433, 1.0])

    def load_ct_pointcloud(self, visualization_3d):
        """Load CT point cloud from config and add to visualization"""
        if not self.config or not self.config.get('CT_pc'):
            print("No CT point cloud specified in config")
            return False

        ct_path = self.config['CT_pc']

        # Handle relative paths
        if not os.path.isabs(ct_path):
            config_dir = os.path.dirname(self.config_path)
            ct_path = os.path.join(config_dir, ct_path)

        if os.path.exists(ct_path):
            try:
                print(f"Loading CT point cloud from config: {ct_path}")
                ct_points = np.load(ct_path)

                if ct_points.shape[1] >= 3:
                    visualization_3d.load_ct_pointcloud(ct_points[:, :3])
                    print(f"Successfully loaded CT point cloud: {len(ct_points)} points")
                    return True
                else:
                    print("Invalid CT point cloud format")
                    return False
            except Exception as e:
                print(f"Error loading CT point cloud: {e}")
                return False
        else:
            print(f"CT point cloud file not found: {ct_path}")
            return False
import numpy as np
from scipy.spatial.transform import Rotation
import time


class PathGenerator:
    def __init__(self, time_scale=0.1, noise_level=0.001):
        """
        Initialize the class with parameters for controlling motion and noise.

        Args:
            time_scale (float): Controls the speed of movement
            noise_level (float): Amount of noise for the steady transformation
        """
        self.time_scale = time_scale
        self.noise_level = noise_level
        self.start_time = time.time()
        self.counter= 0

    def _create_transform_matrix(self, translation, rotation_euler):
        """
        Create a 4x4 transformation matrix from translation vector and Euler angles.

        Args:
            translation (np.ndarray): 3D translation vector [x, y, z]
            rotation_euler (np.ndarray): Euler angles [roll, pitch, yaw] in radians

        Returns:
            np.ndarray: 4x4 transformation matrix
        """
        # Create rotation matrix from Euler angles
        r = Rotation.from_euler('xyz', rotation_euler)
        rotation_matrix = r.as_matrix()

        # Create 4x4 transformation matrix
        transform = np.eye(4)
        transform[:3, :3] = rotation_matrix
        transform[:3, 3] = translation

        return transform

    def get_transformation_matrices(self):
        """
        Returns a list of three 4x4 transformation matrices:
        - First matrix follows a circular path in xy-plane
        - Second matrix is steady with small random noise
        - Third matrix follows a figure-8 path in 3D space

        Returns:
            list: Three 4x4 numpy arrays representing transformation matrices
        """
        elapsed_time = time.time() - self.start_time
        t = elapsed_time * self.time_scale

        # First matrix - circular path
        radius = 1.0
        x1 = radius * np.cos(t)
        y1 = radius * np.sin(t)
        z1 = 0.5 * np.sin(0.5 * t)
        translation1 = np.array([x1, y1, z1])

        # Rotation for first matrix - slowly rotating
        rotation1 = np.array([0.2 * np.sin(0.3 * t),
                              0.1 * np.cos(0.2 * t),
                              t * 0.1])

        # Second matrix - steady with small noise
        translation2 = np.array([0.5, 0.5, 0.5]) + np.random.normal(0, self.noise_level, 3)
        rotation2 = np.array([0.1, 0.1, 0.1]) + np.random.normal(0, self.noise_level, 3)

        # Third matrix - figure-8 path
        x3 = np.sin(t)
        y3 = np.sin(t) * np.cos(t)
        z3 = 0.5 * np.cos(t)
        translation3 = np.array([x3, y3, z3])

        # Rotation for third matrix - more complex rotation
        rotation3 = np.array([np.sin(0.5 * t),
                              np.cos(0.7 * t),
                              0.5 * np.sin(0.3 * t)])

        # Create transformation matrices
        transform1 = self._create_transform_matrix(translation1, rotation1)

        transform2 = self._create_transform_matrix(translation2, rotation2)

        self.counter += 1

        if self.counter in [100]:
            transform1 = np.array([[np.nan, np.nan, np.nan, np.nan],
                                   [np.nan, np.nan, np.nan, np.nan],
                                   [np.nan, np.nan, np.nan, np.nan],
                                   [0.0, 0.0, 0.0, 1.0]])
        transform3 = self._create_transform_matrix(translation3, rotation3)



        return [transform1, transform2, transform3]

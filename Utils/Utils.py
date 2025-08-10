import json
import os

import numpy as np


def save_state(filename, data_dict):
    """
    Save a dictionary with NumPy arrays to JSON.
    If the file exists, update it with new or modified fields.

    Args:
        filename (str): Path to the JSON file
        data_dict (dict): Dictionary with values that may include NumPy arrays
    """
    # Process all numpy arrays in the dictionary
    processed_dict = {}

    # Load existing data if file exists
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            try:
                processed_dict = json.load(f)
            except json.JSONDecodeError:
                # If file exists but is not valid JSON, start with empty dict
                processed_dict = {}

    # Process new data and update/add to the existing dictionary
    for key, value in data_dict.items():
        if isinstance(value, np.ndarray):
            processed_dict[key] = value.tolist()
        else:
            processed_dict[key] = value

    # Write the updated dictionary to file
    with open(filename, 'w') as f:
        json.dump(processed_dict, f)


# Run the application if this script is executed directly
import argparse
import json

import uvicorn

from Network.Broadcaster import IPBroadcaster
from Server.ndi_server import NDI_Server

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process NDI configuration.")
    parser.add_argument("--ndi_config_path", type=str, required=True, help="Path to the NDI configuration file (required)")
    parser.add_argument("--is_local", action='store_true', required=True, help="is local")
    parser.add_argument("--reference_required", action='store_true', required=True, help="need reference in the field")

    args = parser.parse_args()

    with open(args.ndi_config_path, 'r') as file:
        config = json.load(file)

    broadcaster = IPBroadcaster(interval=3)
    broadcaster.start()

    ndi_server = NDI_Server(config, args)
    ndi_server.run()

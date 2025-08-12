

# Run the application if this script is executed directly
import argparse
import json

from Server import ndi_server
from UI.GUI import launch_ui
import uvicorn
import threading

from Network.Broadcaster import IPBroadcaster
from Server.ndi_server import NDI_Server

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process NDI configuration.")
    parser.add_argument("--ndi_config_path", type=str, required=True, help="Path to the NDI configuration file (required)")
    parser.add_argument("--is_local", action=argparse.BooleanOptionalAction)
    parser.add_argument("--reference_required", action=argparse.BooleanOptionalAction, help="need reference in the field")

    args = parser.parse_args()

    with open(args.ndi_config_path, 'r') as file:
        config = json.load(file)

    broadcaster = IPBroadcaster(interval=3)
    broadcaster.start()
    ndiserver = NDI_Server(config, args)
    server_thread = threading.Thread(target=ndiserver.run, daemon=True)
    server_thread.start()

    launch_ui(ndiserver, config, args)

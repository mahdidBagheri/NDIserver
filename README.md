# NDI Tracking Server

A FastAPI-based server for NDI (Northern Digital Inc.) tracking system integration, providing coarse and fine registration, tool calibration, and real-time UDP streaming capabilities.

## Overview

This server provides a REST API interface for NDI tracking operations including:
- **Coarse Registration**: Initial alignment between coordinate systems
- **Fine Registration**: Precise alignment using ICP (Iterative Closest Point) algorithm
- **Tool Calibration**: Calibration of tracking tool tip positions
- **Real-time Streaming**: UDP streaming of tracking data to clients
- **Dual Mode Operation**: Support for both real NDI hardware and local file simulation

## Features

- 🎯 **Coarse & Fine Registration**: Multi-stage registration process for accurate coordinate alignment
- 🔧 **Tool Calibration**: Automatic tool tip vector calculation
- 📡 **UDP Streaming**: Real-time position streaming at configurable frequencies (1-100 Hz)
- 🔄 **Dual Mode**: Switch between real NDI hardware and local file simulation
- 🌐 **Client IP Detection**: Automatic client IP detection for streaming
- 📊 **Comprehensive Logging**: Detailed logging for debugging and monitoring
- 🔒 **Thread-safe**: Safe concurrent operations with proper resource management
- 📈 **3D Visualization**: Built-in support for 3D point cloud visualization and plotting

## Requirements

### Dependencies
```
fastapi
pydantic
numpy
uvicorn
open3d==0.16.0
matplotlib
```

### Custom Modules
The server requires the following custom NDI modules:
- `NDI.ndi_coarse_registration`
- `NDI.ndi_fine_registration` 
- `NDI.ndi_tool_calibration`
- `NDI.NDI_Tracking`

### Hardware (Optional)
- NDI tracking system hardware (for real-time tracking mode)

## Installation

1. Install required Python packages:
```bash
pip install fastapi pydantic numpy uvicorn open3d==0.16.0 matplotlib
```

2. Ensure the NDI modules are properly installed and accessible

3. Run the server:
```bash
python ndi_server.py
```

The server will start on `http://0.0.0.0:8000` by default.

## Configuration

### Global Settings
- `IS_LOCAL`: Boolean flag to switch between local file simulation and real NDI hardware
- `tip_vector`: Default tool tip vector in tool coordinates
- `client_ip`: Target IP address for UDP streaming (auto-detected from client requests)

### Streaming Configuration
- `streaming_port`: UDP port for streaming (default: 11111)
- `streaming_frequency`: Streaming frequency in Hz (default: 30)

## API Endpoints

### System Status
- `GET /` - Get server status and configuration
- `POST /set_data_source` - Switch between local files and real NDI hardware
- `POST /initialize_ndi` - Initialize NDI tracker hardware

### Client Management
- `POST /set_client_ip` - Manually set client IP for streaming
- `GET /get_client_ip` - Get current client IP

### Coarse Registration
- `GET /available_points` - Get available point numbers from coarse data
- `POST /set_coarse_point` - Set a coarse registration point
- `GET /get_coarse_points` - Get all stored coarse points
- `POST /coarse_register` - Perform coarse registration (supports visualization)
- `POST /reset_coarse_points` - Reset all coarse points

### Fine Registration
- `POST /start_fine_gather` - Start gathering fine registration points
- `POST /end_fine_gather` - Stop gathering fine registration points
- `POST /simulate_fine_gather` - Simulate fine point gathering
- `POST /fine_register` - Perform fine registration using ICP (supports visualization)
- `GET /get_fine_points_status` - Get fine point gathering status

### Tool Calibration
- `POST /start_tool_calibration` - Start tool calibration data collection
- `POST /end_tool_calibration` - End tool calibration
- `POST /calibrate_tool` - Process calibration data (supports visualization)
- `GET /get_tool_calibration_status` - Get calibration status
- `POST /add_tool_transformation` - Manually add transformation matrix

### Streaming
- `POST /start_streaming` - Start UDP streaming
- `POST /stop_streaming` - Stop UDP streaming
- `GET /streaming_status` - Get streaming status
- `GET /get_latest_position` - Get most recent position data

## Usage Examples

### Basic Registration Workflow

1. **Initialize the system**:
```bash
curl -X POST "http://localhost:8000/set_data_source?is_local=true"
```

2. **Set coarse registration points**:
```bash
curl -X POST "http://localhost:8000/set_coarse_point" \
  -H "Content-Type: application/json" \
  -d '{"unity_point": [0, 0, 0], "point_number": 1}'
```

3. **Perform coarse registration with visualization**:
```bash
curl -X POST "http://localhost:8000/coarse_register?visualize=true"
```

4. **Gather fine registration points**:
```bash
curl -X POST "http://localhost:8000/simulate_fine_gather?num_points=100"
```

5. **Perform fine registration with visualization**:
```bash
curl -X POST "http://localhost:8000/fine_register?id=1&visualize=true"
```

### Start Streaming

```bash
curl -X POST "http://localhost:8000/start_streaming?port=11111&frequency=30"
```

### Tool Calibration with Visualization

```bash
# Start calibration
curl -X POST "http://localhost:8000/start_tool_calibration"

# End calibration and process data with visualization
curl -X POST "http://localhost:8000/end_tool_calibration"
curl -X POST "http://localhost:8000/calibrate_tool?visualize=true"
```

## Data Formats

### UDP Streaming Data Format
```json
{
  "position": [x, y, z],
  "source": "simulated|ndi_tracker",
  "original": [x, y, z],
  "timestamp": 1234567890.123,
  "frame": 1234,
  "matrix": [[4x4 transformation matrix]],
  "transformed_matrix": [[4x4 transformed matrix]]
}
```

### Coarse Point Input
```json
{
  "unity_point": [x, y, z],
  "point_number": 1
}
```

## Visualization Features

The server supports 3D visualization for various operations:

- **Coarse Registration**: Visualize point correspondences and transformation results
- **Fine Registration**: Display ICP alignment process with before/after point clouds
- **Tool Calibration**: Show collected tool positions and calculated tip vector

Visualization is enabled by adding `?visualize=true` to relevant endpoints.

## Logging

The server provides comprehensive logging with:
- Initialization status
- Registration progress
- Streaming statistics
- Error tracking
- Performance metrics

## Error Handling

The server includes robust error handling for:
- Invalid input parameters
- Hardware connection issues
- Registration failures
- Streaming interruptions
- Resource cleanup

## Development Notes

- The server uses FastAPI's automatic documentation at `/docs`
- Thread-safe operations for concurrent streaming and calibration
- Automatic resource cleanup on shutdown
- Middleware for client IP detection
- Support for both development (local files) and production (real hardware) modes
- 3D visualization capabilities using Open3D for point clouds and matplotlib for plots

## License

[Add your license information here]

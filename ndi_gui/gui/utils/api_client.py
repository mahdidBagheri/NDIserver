import requests
import json
from typing import Optional, Dict, Any


class APIClient:
    def __init__(self, base_url: str = None):
        self.base_url = base_url
        self.session = requests.Session()

    def set_base_url(self, url: str):
        self.base_url = url

    def _make_request(self, method: str, endpoint: str, **kwargs) -> Optional[Dict[Any, Any]]:
        if not self.base_url:
            raise Exception("Base URL not set")

        url = f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"

        try:
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"API request failed: {str(e)}")

    def get_status(self) -> Dict[Any, Any]:
        return self._make_request('GET', '/')

    def initialize_ndi(self, force_restart: bool = False) -> Dict[Any, Any]:
        return self._make_request('POST', '/initialize_ndi',
                                  params={'force_restart': force_restart})

    def find_reference(self, max_tries: int = 50, wait_time: float = 1.0) -> Dict[Any, Any]:
        return self._make_request('POST', '/find_reference',
                                  json={'max_tries': max_tries, 'wait_time': wait_time})

    def check_tools(self) -> Dict[Any, Any]:
        return self._make_request('POST', '/check_tools')

    def set_client_ip(self) -> Dict[Any, Any]:
        return self._make_request('POST', '/set_client_ip')

    def get_client_ip(self) -> Dict[Any, Any]:
        return self._make_request('GET', '/get_client_ip')

    # Coarse registration methods
    def reset_coarse_points(self) -> Dict[Any, Any]:
        return self._make_request('POST', '/reset_coarse_points')

    def set_coarse_point(self, unity_point: list, point_number: int) -> Dict[Any, Any]:
        return self._make_request('POST', '/set_coarse_point',
                                  json={'unity_point': unity_point, 'point_number': point_number})

    def coarse_register(self, visualize: bool = False) -> Dict[Any, Any]:
        return self._make_request('POST', '/coarse_register',
                                  params={'visualize': visualize})

    def get_coarse_points(self) -> Dict[Any, Any]:
        return self._make_request('GET', '/get_coarse_points')

    # Fine registration methods
    def start_fine_gather(self, frequency: int = 60, streaming_raw_frequency: int = 10) -> Dict[Any, Any]:
        return self._make_request('POST', '/start_fine_gather',
                                  params={'frequency': frequency, 'streaming_raw_frequncy': streaming_raw_frequency})

    def end_fine_gather(self, streaming_raw_frequency: int = 30) -> Dict[Any, Any]:
        return self._make_request('POST', '/end_fine_gather',
                                  params={'streaming_raw_frequncy': streaming_raw_frequency})

    def get_fine_points_status(self) -> Dict[Any, Any]:
        return self._make_request('GET', '/get_fine_points_status')

    def reset_fine_gather(self) -> Dict[Any, Any]:
        return self._make_request('POST', '/reset_fine_gather')

    def fine_register(self, id: int, downsample_factor: float = 1.0, visualize: bool = False) -> Dict[Any, Any]:
        return self._make_request('POST', '/fine_register',
                                  params={'id': id, 'downsample_factor': downsample_factor, 'visualize': visualize})

    # Tool calibration methods
    def start_tool_calibration(self, force_stop_streaming: bool = False, device: int = 0) -> Dict[Any, Any]:
        return self._make_request('POST', '/start_tool_calibration',
                                  params={'force_stop_streaming': force_stop_streaming, 'device': device})

    def end_tool_calibration(self) -> Dict[Any, Any]:
        return self._make_request('POST', '/end_tool_calibration')

    def calibrate_tool(self, visualize: bool = False) -> Dict[Any, Any]:
        return self._make_request('POST', '/calibrate_tool',
                                  params={'visualize': visualize})

    def get_tool_calibration_status(self) -> Dict[Any, Any]:
        return self._make_request('GET', '/get_tool_calibration_status')

    # Streaming methods
    def start_streaming(self, port: int = 11111, frequency: int = 30,
                        force_stop_calibration: bool = False, streaming_raw_frequency: int = 10) -> Dict[Any, Any]:
        return self._make_request('POST', '/start_streaming',
                                  params={'port': port, 'frequency': frequency,
                                          'force_stop_calibration': force_stop_calibration,
                                          'streaming_raw_frequency': streaming_raw_frequency})

    def stop_streaming(self, streaming_raw_frequency: int = 30) -> Dict[Any, Any]:
        return self._make_request('POST', '/stop_streaming',
                                  params={'streaming_raw_frequncy': streaming_raw_frequency})

    def get_streaming_status(self) -> Dict[Any, Any]:
        return self._make_request('GET', '/streaming_status')

    def get_latest_position(self) -> Dict[Any, Any]:
        return self._make_request('GET', '/get_latest_position')
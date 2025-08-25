from PyQt5.QtCore import QObject, pyqtSignal, QTimer
import logging


class StatusManager(QObject):
    status_updated = pyqtSignal(dict)

    def __init__(self, api_client):
        super().__init__()
        self.api_client = api_client
        self.logger = logging.getLogger(__name__)
        self.last_status = {}

    def refresh_status(self):
        """Refresh status from the server"""
        try:
            if not self.api_client.base_url:
                # Not connected
                status = {'connected': False}
                self.emit_status_if_changed(status)
                return

            # Get server status
            response = self.api_client.get_status()

            if response:
                # Add connection status
                status = response.copy()
                status['connected'] = True

                # Get additional status information
                try:
                    # Get streaming status
                    streaming_status = self.api_client.get_streaming_status()
                    status.update(streaming_status)
                except:
                    pass

                try:
                    # Get client IP
                    ip_status = self.api_client.get_client_ip()
                    status.update(ip_status)
                except:
                    pass

                try:
                    # Get fine points status
                    fine_status = self.api_client.get_fine_points_status()
                    if 'gathering_active' in fine_status:
                        status['fine_gathering_active'] = fine_status['gathering_active']
                except:
                    pass

                try:
                    # Get tool calibration status
                    tool_status = self.api_client.get_tool_calibration_status()
                    if 'calibration_active' in tool_status:
                        status['tool_calibration_active'] = tool_status['calibration_active']
                except:
                    pass

                self.emit_status_if_changed(status)

            else:
                # Connection failed
                status = {'connected': False}
                self.emit_status_if_changed(status)

        except Exception as e:
            self.logger.warning(f"Error refreshing status: {str(e)}")
            status = {'connected': False, 'error': str(e)}
            self.emit_status_if_changed(status)

    def emit_status_if_changed(self, status):
        """Only emit status signal if status has changed"""
        if status != self.last_status:
            self.last_status = status.copy()
            self.status_updated.emit(status)
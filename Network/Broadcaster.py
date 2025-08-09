import socket
import time
import threading
import atexit
import signal
import sys


class IPBroadcaster:
    """
    A class that broadcasts the system's IP address on the local network at regular intervals.
    Designed to be run in a thread and automatically stop on program exit.
    """

    def __init__(self, interval=3, port=5000, broadcast_ip="255.255.255.255"):
        """
        Initialize the IP broadcaster.

        Parameters:
        - interval: Time in seconds between broadcasts (default: 3)
        - port: UDP port to broadcast on (default: 5000)
        - broadcast_ip: Broadcast address (default: 255.255.255.255)
        """
        self.interval = interval
        self.port = port
        self.broadcast_ip = broadcast_ip
        self.running = False
        self.thread = None
        self.sock = None

        # Register cleanup handlers to ensure proper shutdown
        atexit.register(self.stop)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, sig, frame):
        """Handle signals like SIGINT (Ctrl+C) and SIGTERM"""
        self.stop()
        sys.exit(0)

    def get_local_ip(self):
        """Get the system's local IP address"""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "Unable to get IP"

    def broadcast_loop(self):
        """Main broadcasting loop that runs in a thread"""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        while self.running:
            local_ip = self.get_local_ip()
            message = f"Device IP: {local_ip}".encode('utf-8')
            try:
                self.sock.sendto(message, (self.broadcast_ip, self.port))
                print(f"Broadcasting IP: {local_ip} on port {self.port}")
            except Exception as e:
                print(f"Broadcast error: {e}")
                # If there's an error, try to recreate the socket
                try:
                    self.sock.close()
                    self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
                    self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                except:
                    pass
            time.sleep(self.interval)

    def start(self):
        """Start broadcasting IP in a separate thread"""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self.broadcast_loop)
            self.thread.daemon = True  # Daemon threads exit when the program exits
            self.thread.start()
            print("IP broadcasting started")
            return True
        return False

    def stop(self):
        """Stop broadcasting IP"""
        if self.running:
            self.running = False
            if self.sock:
                try:
                    self.sock.close()
                except:
                    pass  # Ignore errors when closing socket
            print("IP broadcasting stopped")
            return True
        return False

    def is_running(self):
        """Check if the broadcaster is running"""
        return self.running

    def set_interval(self, interval):
        """Change the broadcast interval"""
        self.interval = interval
        return True


# Example usage:
if __name__ == "__main__":
    print("Starting IP broadcaster. Press Ctrl+C to stop.")

    # Create and start the broadcaster
    broadcaster = IPBroadcaster(interval=3)
    broadcaster.start()

    try:
        # Keep the main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        # This will be caught by the signal handler
        pass

    print("Program terminated.")
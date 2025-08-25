import sys
import logging
import argparse
import json
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from gui.main_window import NDIMainWindow


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('ndi_gui.log'),
            logging.StreamHandler()
        ]
    )


def parse_arguments():
    parser = argparse.ArgumentParser(description='NDI Tracking GUI')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration file (JSON)')
    return parser.parse_args()


def load_config(config_path):
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON in configuration file: {config_path}")


def main():
    setup_logging()

    # Parse command line arguments
    args = parse_arguments()
    config = load_config(args.config)

    app = QApplication(sys.argv)
    app.setApplicationName("NDI Tracking Control")
    app.setApplicationVersion("1.0")

    # Enable high DPI scaling
    app.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    # Load dark theme
    try:
        with open('resources/styles/dark_theme.qss', 'r') as f:
            app.setStyleSheet(f.read())
    except FileNotFoundError:
        pass

    window = NDIMainWindow(config)
    window.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
from PyQt5.QtWidgets import QMessageBox, QGroupBox


def show_message_box(parent, title, message, msg_type="information"):
    """Helper function to show message boxes"""
    msg = QMessageBox(parent)
    msg.setWindowTitle(title)
    msg.setText(str(message))

    if msg_type == "warning":
        msg.setIcon(QMessageBox.Warning)
    elif msg_type == "error":
        msg.setIcon(QMessageBox.Critical)
    elif msg_type == "question":
        msg.setIcon(QMessageBox.Question)
    else:
        msg.setIcon(QMessageBox.Information)

    return msg.exec_()


def create_group_box(title, layout):
    """Helper to create group boxes with consistent styling"""
    group = QGroupBox(title)
    group.setLayout(layout)
    return group
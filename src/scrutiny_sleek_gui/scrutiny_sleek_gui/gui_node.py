#!/usr/bin/env python3
import os
import sys
import time
import signal
import shutil
import subprocess
import re
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

def cv_to_pixmap(cv_img):
    if cv_img is None:
        return None
    if len(cv_img.shape) == 2:
        h, w = cv_img.shape
        qimg = QImage(cv_img.data, w, h, w, QImage.Format_Grayscale8)
        return QPixmap.fromImage(qimg.copy())
    if len(cv_img.shape) == 3:
        h, w, c = cv_img.shape
        if c == 3:
            rgb = cv_img[:, :, ::-1].copy()
            qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
            return QPixmap.fromImage(qimg.copy())
        if c == 4:
            qimg = QImage(cv_img.data, w, h, 4 * w, QImage.Format_ARGB32)
            return QPixmap.fromImage(qimg.copy())
    return None

def human_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if value < 1024.0 or unit == 'TB':
            return f'{value:.1f} {unit}'
        value /= 1024.0
    return f'{num_bytes} B'

def age_text(age_sec: float) -> str:
    if age_sec < 0:
        return '--'
    if age_sec < 1.0:
        return f'{age_sec:.2f}s'
    return f'{age_sec:.1f}s'

def colorize_depth_mono8(depth8: np.ndarray) -> np.ndarray:
    if depth8 is None:
        return None
    if len(depth8.shape) != 2:
        return depth8
    color = cv2.applyColorMap(depth8, cv2.COLORMAP_JET)
    color[depth8 == 0] = [0, 0, 0]
    return color

def near_emphasis_from_metric_depth(
    depth_img: np.ndarray,
    near_min_m: float,
    near_max_m: float,
    blur_ksize: int = 3,
    equalize_hist: bool = False,
) -> np.ndarray:
    if depth_img is None or depth_img.size == 0:
        return None
    out = np.zeros(depth_img.shape[:2], dtype=np.uint8)
    if near_max_m <= near_min_m:
        return out
    depth = np.array(depth_img, dtype=np.float32)
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    valid = np.isfinite(depth) & (depth >= near_min_m) & (depth <= near_max_m)
    if not np.any(valid):
        return out
    norm = (near_max_m - depth) / (near_max_m - near_min_m)
    norm = np.clip(norm, 0.0, 1.0)
    out[valid] = (norm[valid] * 255.0).astype(np.uint8)
    if blur_ksize and blur_ksize >= 3:
        if blur_ksize % 2 == 0:
            blur_ksize += 1
        out = cv2.medianBlur(out, blur_ksize)
    if equalize_hist:
        out = cv2.equalizeHist(out)
    return out

class StatusDot(QLabel):
    def __init__(self, color='#666666', size=14):
        super().__init__()
        self._size = size
        self.setFixedSize(size, size)
        self.set_color(color)
    def set_color(self, color: str):
        self.setStyleSheet(
            f"""
            QLabel {{
                background: {color};
                border-radius: {self._size // 2}px;
                border: 1px solid #202020;
            }}
            """
        )

class PanelWidget(QFrame):
    COLOR_LOADING = '#d59b28'
    COLOR_OK = '#1f9d45'
    COLOR_ERROR = '#cc3333'
    def __init__(self, title: str):
        super().__init__()
        self.setStyleSheet("""
            QFrame {
                background: #111111;
                border: 1px solid #2a2a2a;
                border-radius: 12px;
            }
        """)
        self._root = QVBoxLayout(self)
        self._root.setContentsMargins(8, 8, 8, 8)
        self._root.setSpacing(6)
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(6)
        self.title_label = QLabel(title)
        self.title_label.setFont(QFont('Sans Serif', 10, QFont.Bold))
        self.title_label.setStyleSheet('color: #e8e8e8;')
        self.status_dot = StatusDot(self.COLOR_LOADING, 12)
        self.status_label = QLabel('Loading')
        self.status_label.setStyleSheet('color: #cfcfcf; font-size: 11px;')
        self.age_label = QLabel('--')
        self.age_label.setStyleSheet('color: #8d8d8d; font-size: 11px;')
        header.addWidget(self.title_label)
        header.addStretch(1)
        header.addWidget(self.status_dot)
        header.addWidget(self.status_label)
        header.addSpacing(8)
        header.addWidget(self.age_label)
        self._root.addLayout(header)
    def set_status(self, state: str, age_sec: Optional[float] = None):
        if state == 'ok':
            self.status_dot.set_color(self.COLOR_OK)
            self.status_label.setText('Running')
        elif state == 'error':
            self.status_dot.set_color(self.COLOR_ERROR)
            self.status_label.setText('Error')
        else:
            self.status_dot.set_color(self.COLOR_LOADING)
            self.status_label.setText('Loading')
        self.age_label.setText(age_text(age_sec) if age_sec is not None else '--')

class VideoPanel(PanelWidget):
    def __init__(self, title: str):
        super().__init__(title)
        self.image_label = QLabel('Waiting for frames')
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(80, 60)
        self.image_label.setStyleSheet("""
            QLabel {
                background: #090909;
                color: #6f6f6f;
                border-radius: 8px;
            }
        """)
        self._root.addWidget(self.image_label, 1)
    def set_pixmap(self, pixmap: QPixmap):
        if not pixmap:
            return
        self.image_label.setPixmap(
            pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

class LogPanel(PanelWidget):
    def __init__(self, title: str):
        super().__init__(title)
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setStyleSheet("""
            QTextEdit {
                background: #090909;
                color: #d2d2d2;
                border: none;
                border-radius: 8px;
                padding: 6px;
                font-family: monospace;
                font-size: 12px;
            }
        """)
        controls = QHBoxLayout()
        controls.setSpacing(6)
        self.autoscroll_label = QLabel('Autoscroll: On')
        self.autoscroll_label.setStyleSheet('color: #bcbcbc; font-size: 11px;')
        self.clear_btn = QPushButton('Clear')
        self.clear_btn.setFixedHeight(28)
        self.clear_btn.setStyleSheet("""
            QPushButton {
                background: #202020;
                color: #f0f0f0;
                border: 1px solid #333333;
                border-radius: 6px;
                padding: 0 10px;
            }
            QPushButton:hover { background: #2a2a2a; }
        """)
        self.autoscroll_btn = QPushButton('Toggle')
        self.autoscroll_btn.setFixedHeight(28)
        self.autoscroll_btn.setStyleSheet(self.clear_btn.styleSheet())
        controls.addWidget(self.autoscroll_label)
        controls.addStretch(1)
        controls.addWidget(self.autoscroll_btn)
        controls.addWidget(self.clear_btn)
        self._root.addLayout(controls)
        self._root.addWidget(self.log_box, 1)
        self.autoscroll_enabled = True
        self.clear_btn.clicked.connect(self.log_box.clear)
        self.autoscroll_btn.clicked.connect(self._toggle_autoscroll)
    def _toggle_autoscroll(self):
        self.autoscroll_enabled = not self.autoscroll_enabled
        self.autoscroll_label.setText(f"Autoscroll: {'On' if self.autoscroll_enabled else 'Off'}")
    def append_line(self, line: str):
        self.log_box.append(line)
        if self.autoscroll_enabled:
            self.log_box.moveCursor(self.log_box.textCursor().End)

@dataclass
class Topics:
    zed_left: str
    zed_depth: str
    tof: str
    measurement: str

class ScrutinyGuiNode(Node):
    def __init__(self):
        super().__init__('scrutiny_sleek_gui')
        self.declare_parameter('zed_left_topic', '/zed/zed_node/left/color/rect/image')
        self.declare_parameter('zed_depth_topic', '/zed/zed_node/depth/depth_registered')
        self.declare_parameter('tof_topic', '/scrutiny/tof/depth8')
        self.declare_parameter('measurement_topic', '/scrutiny/measurement/fused+overlay')
        self.declare_parameter('bag_dir', os.path.expanduser('~/bags'))
        self.declare_parameter('poweroff_command', 'systemctl poweroff')
        self.declare_parameter('zed_near_min_m', 0.25)
        self.declare_parameter('zed_near_max_m', 0.90)
        self.declare_parameter('depth_median_blur_ksize', 3)
        self.declare_parameter('zed_equalize_hist', False)
        self.declare_parameter('tof_equalize_hist', False)
        self.bridge = CvBridge()
        self.topics = Topics(
            zed_left=self.get_parameter('zed_left_topic').value,
            zed_depth=self.get_parameter('zed_depth_topic').value,
            tof=self.get_parameter('tof_topic').value,
            measurement=self.get_parameter('measurement_topic').value,
        )
        self.bag_dir = os.path.expanduser(self.get_parameter('bag_dir').value)
        self.poweroff_command = self.get_parameter('poweroff_command').value
        self.zed_near_min_m = float(self.get_parameter('zed_near_min_m').value)
        self.zed_near_max_m = float(self.get_parameter('zed_near_max_m').value)
        self.depth_median_blur_ksize = int(self.get_parameter('depth_median_blur_ksize').value)
        self.zed_equalize_hist = bool(self.get_parameter('zed_equalize_hist').value)
        self.tof_equalize_hist = bool(self.get_parameter('tof_equalize_hist').value)
        self.qos_cam = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            durability=DurabilityPolicy.VOLATILE,
        )

class MainWindow(QMainWindow):
    STARTUP_TIMEOUT_SEC = 30.0
    STREAM_STALE_SEC = 1.5
    STREAM_ERROR_SEC = 30.0
    DISK_WARN_FREE_GB = 15.0
    DISK_ERROR_FREE_GB = 5.0
    LOG_HEARTBEAT_SEC = 5.0
    def __init__(self, node: ScrutinyGuiNode):
        super().__init__()
        self.node = node
        self.bridge = node.bridge
        self.setWindowTitle('Scrutiny Camera Dashboard')
        screen = QApplication.primaryScreen()
        geo = screen.availableGeometry()
        self.resize(geo.width(), geo.height())
        self._bag_proc: Optional[subprocess.Popen] = None
        self._left_frame = None
        self._depth_frame = None
        self._tof_frame = None
        self._measurement_frame = None
        self._last_left_ts = 0.0
        self._last_depth_ts = 0.0
        self._last_tof_ts = 0.0
        self._last_measurement_ts = 0.0
        self._left_seen = False
        self._depth_seen = False
        self._tof_seen = False
        self._measurement_seen = False
        self._operator_mode = True
        self._start_time = time.time()
        self._last_heartbeat_log = 0.0
        self._last_disk_state = 'unknown'
        self._last_header_state = ''
        self._last_record_state = False
        self._last_temp_update_time = 0.0
        self._cpu_temp_c: Optional[float] = None
        self._gpu_temp_c: Optional[float] = None
        self._temp_read_failed = False
        self._zed_state = 'loading'
        self._tof_state = 'loading'
        self._build_ui()
        self._connect_ros()
        self._log_startup()
        self._ui_timer = QTimer(self)
        self._ui_timer.timeout.connect(self._tick_ui)
        self._ui_timer.start(200)

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        root.setStyleSheet('background: #050505;')
        outer = QVBoxLayout(root)
        outer.setContentsMargins(5, 5, 5, 5)
        outer.setSpacing(5)

        screen_width = QApplication.primaryScreen().availableGeometry().width()
        compact = screen_width <= 1100

        header = QFrame()
        header.setFixedHeight(72 if compact else 92)
        header.setStyleSheet("""
            QFrame { background: #0d0d0d; border: 1px solid #232323; border-radius: 9px; }
            QLabel { color: #eaeaea; }
        """)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(9, 6, 9, 6)
        header_layout.setSpacing(8)

        title_wrap = QVBoxLayout()
        title_wrap.setSpacing(1)
        self.title_label = QLabel('Scrutiny')
        self.title_label.setFont(QFont('Sans Serif', 14 if compact else 16, QFont.Bold))
        self.system_state_label = QLabel('System: Starting')
        self.system_state_label.setStyleSheet('color: #bdbdbd; font-size: 11px;')
        title_wrap.addWidget(self.title_label)
        title_wrap.addWidget(self.system_state_label)

        info_wrap = QVBoxLayout()
        info_wrap.setSpacing(1)
        self.uptime_label = QLabel('Uptime: 00:00:00')
        self.disk_label = QLabel('Disk: --')
        self.record_target_label = QLabel('Bag: Idle')
        self.temp_label = QLabel('CPU: -- | GPU: --')
        self.error_label = QLabel('Last event: startup')
        for lbl in [self.uptime_label, self.disk_label, self.record_target_label, self.temp_label, self.error_label]:
            lbl.setStyleSheet('color: #bdbdbd; font-size: 10px;')
        info_wrap.addWidget(self.disk_label)
        info_wrap.addWidget(self.temp_label)
        if not compact:
            info_wrap.addWidget(self.uptime_label)
            info_wrap.addWidget(self.record_target_label)
            info_wrap.addWidget(self.error_label)

        controls_wrap = QHBoxLayout()
        controls_wrap.setSpacing(6)
        self.record_status_label = QLabel('Idle')
        self.record_status_label.setStyleSheet('color: #cfcfcf; font-size: 11px;')
        self.view_btn = QPushButton('Engineer')
        self.view_btn.setFixedHeight(38)
        self.view_btn.setMinimumWidth(92)
        self.view_btn.clicked.connect(self.toggle_view)
        self.view_btn.setStyleSheet("""
            QPushButton { background: #176a9a; color: white; font-weight: bold; border: none; border-radius: 7px; padding: 0 10px; }
            QPushButton:hover { background: #1c7caf; }
        """)
        self.record_btn = QPushButton('Start Recording')
        self.record_btn.setFixedHeight(38)
        self.record_btn.setMinimumWidth(125 if compact else 150)
        self.record_btn.clicked.connect(self.toggle_recording)
        self.power_btn = QPushButton('Power Off')
        self.power_btn.setFixedHeight(38)
        self.power_btn.setMinimumWidth(90 if compact else 110)
        self.power_btn.clicked.connect(self.request_poweroff)
        self._set_record_button_idle()
        self._set_power_button_style()
        if not compact:
            controls_wrap.addWidget(self.record_status_label)
        controls_wrap.addWidget(self.view_btn)
        controls_wrap.addWidget(self.record_btn)
        controls_wrap.addWidget(self.power_btn)

        header_layout.addLayout(title_wrap, 1)
        header_layout.addLayout(info_wrap, 1)
        header_layout.addStretch(1)
        header_layout.addLayout(controls_wrap, 0)
        outer.addWidget(header, 0)

        self.pages = QStackedWidget()
        self.pages.setStyleSheet('background: #050505; border: none;')

        operator_page = QWidget()
        operator_layout = QVBoxLayout(operator_page)
        operator_layout.setContentsMargins(0, 0, 0, 0)
        operator_layout.setSpacing(0)
        self.measurement_panel = VideoPanel('Operator View — RGB + Stencil + Measurement')
        self.measurement_panel.title_label.setFont(QFont('Sans Serif', 11, QFont.Bold))
        operator_layout.addWidget(self.measurement_panel, 1)

        engineer_page = QWidget()
        engineer_grid = QGridLayout(engineer_page)
        engineer_grid.setContentsMargins(0, 0, 0, 0)
        engineer_grid.setSpacing(5)
        self.tof_panel = VideoPanel('ToF')
        self.depth_panel = VideoPanel('ZED Depth')
        self.rgb_panel = VideoPanel('ZED RGB')
        self.log_panel = LogPanel('Runtime Log')
        engineer_grid.addWidget(self.tof_panel, 0, 0)
        engineer_grid.addWidget(self.depth_panel, 0, 1)
        engineer_grid.addWidget(self.rgb_panel, 1, 0)
        engineer_grid.addWidget(self.log_panel, 1, 1)
        engineer_grid.setRowStretch(0, 1)
        engineer_grid.setRowStretch(1, 1)
        engineer_grid.setColumnStretch(0, 1)
        engineer_grid.setColumnStretch(1, 1)

        self.pages.addWidget(operator_page)
        self.pages.addWidget(engineer_page)
        self.pages.setCurrentIndex(0)
        outer.addWidget(self.pages, 1)

    def toggle_view(self):
        self._operator_mode = not self._operator_mode
        self.pages.setCurrentIndex(0 if self._operator_mode else 1)
        self.view_btn.setText('Engineer' if self._operator_mode else 'Operator')
        self.log('INFO', f"View -> {'Operator' if self._operator_mode else 'Engineer'}")

    def _set_record_button_idle(self):
        self.record_btn.setText('Start Recording')
        self.record_btn.setEnabled(True)
        self.record_btn.setStyleSheet("""
            QPushButton { background: #1f9d45; color: white; font-weight: bold; border: none; border-radius: 8px; padding: 0 16px; }
            QPushButton:hover { background: #228c43; }
        """)
    def _set_record_button_recording(self):
        self.record_btn.setText('Stop Recording')
        self.record_btn.setEnabled(True)
        self.record_btn.setStyleSheet("""
            QPushButton { background: #cc3333; color: white; font-weight: bold; border: none; border-radius: 8px; padding: 0 16px; }
            QPushButton:hover { background: #b62b2b; }
        """)
    def _set_record_button_busy(self):
        self.record_btn.setText('Working...')
        self.record_btn.setEnabled(False)
        self.record_btn.setStyleSheet("""
            QPushButton { background: #777777; color: white; font-weight: bold; border: none; border-radius: 8px; padding: 0 16px; }
        """)
    def _set_power_button_style(self):
        self.power_btn.setStyleSheet("""
            QPushButton { background: #272727; color: white; font-weight: bold; border: 1px solid #404040; border-radius: 8px; padding: 0 14px; }
            QPushButton:hover { background: #303030; }
        """)

    def log(self, level: str, message: str):
        stamp = time.strftime('%H:%M:%S')
        line = f'[{stamp}] [{level}] {message}'
        self.log_panel.append_line(line)
        self.error_label.setText(f'Last event: {message}')

    def _log_startup(self):
        self.log('INFO', 'GUI started')
        self.log('INFO', f'Subscribed to ZED RGB   : {self.node.topics.zed_left}')
        self.log('INFO', f'Subscribed to ZED Depth : {self.node.topics.zed_depth}')
        self.log('INFO', f'Subscribed to ToF       : {self.node.topics.tof}')
        self.log('INFO', f'Subscribed to Measurement: {self.node.topics.measurement}')
        self.log('INFO', f'Bag directory           : {self.node.bag_dir}')
        self.log('INFO', f'ZED near emphasis range : {self.node.zed_near_min_m:.2f} m to {self.node.zed_near_max_m:.2f} m')

    def _connect_ros(self):
        self.sub_left = self.node.create_subscription(Image, self.node.topics.zed_left, self._cb_left, self.node.qos_cam)
        self.sub_depth = self.node.create_subscription(Image, self.node.topics.zed_depth, self._cb_depth, self.node.qos_cam)
        self.sub_tof = self.node.create_subscription(Image, self.node.topics.tof, self._cb_tof, self.node.qos_cam)
        self.sub_measurement = self.node.create_subscription(Image, self.node.topics.measurement, self._cb_measurement, self.node.qos_cam)

    def _cb_left(self, msg: Image):
        try:
            self._left_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            self._last_left_ts = time.time()
            if not self._left_seen:
                self._left_seen = True
                self.log('INFO', f'First ZED RGB frame received ({msg.encoding})')
        except Exception as e:
            self.log('ERROR', f'ZED RGB decode failed: {e}')

    def _cb_depth(self, msg: Image):
        try:
            depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            near8 = near_emphasis_from_metric_depth(
                depth_img=depth_img,
                near_min_m=self.node.zed_near_min_m,
                near_max_m=self.node.zed_near_max_m,
                blur_ksize=self.node.depth_median_blur_ksize,
                equalize_hist=self.node.zed_equalize_hist,
            )
            if near8 is None:
                return
            self._depth_frame = colorize_depth_mono8(near8)
            self._last_depth_ts = time.time()
            if not self._depth_seen:
                self._depth_seen = True
                self.log('INFO', f'First ZED depth frame received ({msg.encoding})')
        except Exception as e:
            self.log('ERROR', f'ZED depth decode failed: {e}')

    def _cb_tof(self, msg: Image):
        try:
            if msg.encoding in ('mono8', '8UC1'):
                tof_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
            else:
                tof_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            if tof_img is None or tof_img.size == 0:
                return
            if len(tof_img.shape) == 2:
                tof_mono = tof_img.copy()
            elif len(tof_img.shape) == 3:
                tof_mono = cv2.cvtColor(tof_img, cv2.COLOR_BGR2GRAY)
            else:
                return
            k = self.node.depth_median_blur_ksize
            if k >= 3:
                if k % 2 == 0:
                    k += 1
                tof_mono = cv2.medianBlur(tof_mono, k)
            if self.node.tof_equalize_hist:
                tof_mono = cv2.equalizeHist(tof_mono)
            self._tof_frame = colorize_depth_mono8(tof_mono)
            self._last_tof_ts = time.time()
            if not self._tof_seen:
                self._tof_seen = True
                self.log('INFO', f'First ToF frame received ({msg.encoding})')
        except Exception as e:
            self.log('ERROR', f'ToF decode failed: {e}')

    def _cb_measurement(self, msg: Image):
        try:
            self._measurement_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self._last_measurement_ts = time.time()
            if not self._measurement_seen:
                self._measurement_seen = True
                self.log('INFO', f'First measurement frame received ({msg.encoding})')
        except Exception as e:
            self.log('ERROR', f'Measurement image decode failed: {e}')

    def toggle_recording(self):
        if self._bag_proc and self._bag_proc.poll() is None:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self):
        os.makedirs(self.node.bag_dir, exist_ok=True)
        out_path = os.path.join(self.node.bag_dir, f"scrutiny_{time.strftime('%Y%m%d_%H%M%S')}")
        cmd = [
            'ros2',
            'bag',
            'record',
            '-o',
            out_path,

            # Raw sensor images
            self.node.topics.zed_left,
            self.node.topics.zed_depth,
            '/scrutiny/tof/depth_raw',
            '/scrutiny/tof/depth8',

            # Processed measurement output
            '/scrutiny/measurement/image',
            '/scrutiny/object_diameter',
            '/scrutiny/measurement_status',
            '/scrutiny/measurement/fused_overlay',
            '/scrutiny/measurement/diameter_mm',


            # Camera calibration
            '/zed/zed_node/left/color/rect/camera_info',
            '/zed/zed_node/depth/depth_registered/camera_info',
            '/scrutiny/tof/camera_info',

            # Coordinate transforms
            '/tf',
            '/tf_static',
        ]

        try:
            self._set_record_button_busy()
            self.log('ACTION', 'Starting rosbag recording')
            self._bag_proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=os.setsid)
            self.record_status_label.setText('Recording')
            self.record_target_label.setText(f'Bag: {os.path.basename(out_path)}')
            self._set_record_button_recording()
            self.log('INFO', f'Recording started: {out_path}')
        except Exception as e:
            self.record_status_label.setText('Idle')
            self.record_target_label.setText('Bag: Idle')
            self._set_record_button_idle()
            self.log('ERROR', f'Recording start failed: {e}')

    def stop_recording(self):
        if not self._bag_proc:
            return
        self._set_record_button_busy()
        self.log('ACTION', 'Stopping rosbag recording')
        try:
            if self._bag_proc.poll() is None:
                os.killpg(os.getpgid(self._bag_proc.pid), signal.SIGINT)
                time.sleep(1.0)
        except Exception as e:
            self.log('WARN', f'Recording stop signal issue: {e}')
        self._bag_proc = None
        self.record_status_label.setText('Idle')
        self.record_target_label.setText('Bag: Idle')
        self._set_record_button_idle()
        self.log('INFO', 'Recording stopped')

    def request_poweroff(self):
        reply = QMessageBox.question(self, 'Power Off', 'Stop recording, close the GUI, and power off the system?', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply != QMessageBox.Yes:
            self.log('INFO', 'Power off cancelled')
            return
        self.log('ACTION', 'Power off requested')
        if self._bag_proc and self._bag_proc.poll() is None:
            self.stop_recording()
        cmd = self.node.poweroff_command.strip()
        if not cmd:
            self.log('WARN', 'Power off command is empty; closing GUI only')
            self.close()
            return
        try:
            subprocess.Popen(cmd.split())
            self.log('INFO', f'Power off command launched: {cmd}')
        except Exception as e:
            self.log('ERROR', f'Power off command failed: {e}')
        self.close()

    def _stream_state(self, seen_once: bool, last_ts: float, now: float) -> str:
        if not seen_once:
            if (now - self._start_time) < self.STARTUP_TIMEOUT_SEC:
                return 'loading'
            return 'error'
        age = now - last_ts
        if age <= self.STREAM_STALE_SEC:
            return 'ok'
        if age <= self.STREAM_ERROR_SEC:
            return 'loading'
        return 'error'

    def _disk_state(self):
        try:
            usage = shutil.disk_usage(self.node.bag_dir if os.path.exists(self.node.bag_dir) else '/')
            free_gb = usage.free / (1024 ** 3)
            if free_gb <= self.DISK_ERROR_FREE_GB:
                return 'error', usage
            if free_gb <= self.DISK_WARN_FREE_GB:
                return 'warn', usage
            return 'ok', usage
        except Exception:
            return 'unknown', None

    def _set_panel_statuses(self, now: float):
        self._zed_state = self._stream_state(self._left_seen or self._depth_seen, max(self._last_left_ts, self._last_depth_ts), now)
        self._tof_state = self._stream_state(self._tof_seen, self._last_tof_ts, now)
        left_age = now - self._last_left_ts if self._last_left_ts else None
        depth_age = now - self._last_depth_ts if self._last_depth_ts else None
        tof_age = now - self._last_tof_ts if self._last_tof_ts else None
        self.rgb_panel.set_status(self._stream_state(self._left_seen, self._last_left_ts, now), left_age)
        self.depth_panel.set_status(self._stream_state(self._depth_seen, self._last_depth_ts, now), depth_age)
        self.tof_panel.set_status(self._tof_state, tof_age)
        self.log_panel.set_status('ok', 0.0)
        measurement_age = now - self._last_measurement_ts if self._last_measurement_ts else None
        self.measurement_panel.set_status(self._stream_state(self._measurement_seen, self._last_measurement_ts, now), measurement_age)

    def _refresh_images(self):
        operator_frame = self._measurement_frame if self._measurement_frame is not None else self._left_frame
        if operator_frame is not None:
            pix = cv_to_pixmap(operator_frame)
            if pix:
                self.measurement_panel.set_pixmap(pix)
        else:
            self.measurement_panel.image_label.setText('Waiting for measurement/RGB frames')
        if self._tof_frame is not None:
            pix = cv_to_pixmap(self._tof_frame)
            if pix:
                self.tof_panel.set_pixmap(pix)
        else:
            self.tof_panel.image_label.setText('Waiting for frames')
        if self._depth_frame is not None:
            pix = cv_to_pixmap(self._depth_frame)
            if pix:
                self.depth_panel.set_pixmap(pix)
        else:
            self.depth_panel.image_label.setText('Waiting for frames')
        if self._left_frame is not None:
            pix = cv_to_pixmap(self._left_frame)
            if pix:
                self.rgb_panel.set_pixmap(pix)
        else:
            self.rgb_panel.image_label.setText('Waiting for frames')


    def _temp_style(self, temp_c: Optional[float]) -> str:
        if temp_c is None:
            return 'color: #bdbdbd; font-size: 12px;'
        if temp_c < 70.0:
            return 'color: #78d27f; font-size: 12px; font-weight: bold;'
        if temp_c < 85.0:
            return 'color: #ffb347; font-size: 12px; font-weight: bold;'
        return 'color: #ff5c5c; font-size: 12px; font-weight: bold;'

    def _read_jetson_temps(self):
        try:
            output = subprocess.check_output(
                'tegrastats --interval 1000 --count 1',
                shell=True,
                stderr=subprocess.DEVNULL
            ).decode(errors='ignore')

            cpu_match = re.search(r'CPU@([0-9]+(?:\.[0-9]+)?)C', output)
            gpu_match = re.search(r'GPU@([0-9]+(?:\.[0-9]+)?)C', output)

            cpu_temp = float(cpu_match.group(1)) if cpu_match else None
            gpu_temp = float(gpu_match.group(1)) if gpu_match else None

            self._temp_read_failed = False
            return cpu_temp, gpu_temp
        except Exception:
            if not self._temp_read_failed:
                self.log('WARN', 'Failed to read Jetson temperature from tegrastats')
                self._temp_read_failed = True
            return None, None

    def _update_temp_header(self, now: float):
        if (now - self._last_temp_update_time) >= 2.0:
            self._last_temp_update_time = now
            self._cpu_temp_c, self._gpu_temp_c = self._read_jetson_temps()

        cpu_text = '--' if self._cpu_temp_c is None else f'{self._cpu_temp_c:.1f}°C'
        gpu_text = '--' if self._gpu_temp_c is None else f'{self._gpu_temp_c:.1f}°C'
        self.temp_label.setText(f'CPU: {cpu_text} | GPU: {gpu_text}')

        ref_temp = self._cpu_temp_c if self._cpu_temp_c is not None else self._gpu_temp_c
        self.temp_label.setStyleSheet(self._temp_style(ref_temp))

    def _refresh_header(self, now: float):
        uptime = int(now - self._start_time)
        h = uptime // 3600
        m = (uptime % 3600) // 60
        s = uptime % 60
        self.uptime_label.setText(f'Uptime: {h:02d}:{m:02d}:{s:02d}')
        disk_state, usage = self._disk_state()
        if usage:
            self.disk_label.setText(f'Disk: free {human_bytes(usage.free)} / total {human_bytes(usage.total)}')
        else:
            self.disk_label.setText('Disk: unavailable')
        if disk_state == 'error':
            self.disk_label.setStyleSheet('color: #ff5c5c; font-size: 12px;')
        elif disk_state == 'warn':
            self.disk_label.setStyleSheet('color: #ffb347; font-size: 12px;')
        else:
            self.disk_label.setStyleSheet('color: #bdbdbd; font-size: 12px;')
        if disk_state != self._last_disk_state:
            if disk_state == 'warn' and usage:
                self.log('WARN', f'Low disk space warning: {human_bytes(usage.free)} free')
            elif disk_state == 'error' and usage:
                self.log('ERROR', f'Critical disk space: {human_bytes(usage.free)} free')
            elif disk_state == 'ok' and self._last_disk_state in ('warn', 'error'):
                self.log('INFO', 'Disk space back to normal')
            self._last_disk_state = disk_state
        recording = self._bag_proc is not None and self._bag_proc.poll() is None
        if recording:
            header_state = 'Recording'
        elif self._zed_state == 'error' or self._tof_state == 'error':
            header_state = 'Error'
        elif self._zed_state == 'ok' and self._tof_state == 'ok':
            header_state = 'Ready'
        else:
            header_state = 'Starting'
        self.system_state_label.setText(f'System: {header_state}')
        if header_state == 'Recording':
            self.system_state_label.setStyleSheet('color: #ffb347; font-size: 12px;')
        elif header_state == 'Ready':
            self.system_state_label.setStyleSheet('color: #78d27f; font-size: 12px;')
        elif header_state == 'Error':
            self.system_state_label.setStyleSheet('color: #ff5c5c; font-size: 12px;')
        else:
            self.system_state_label.setStyleSheet('color: #bdbdbd; font-size: 12px;')
        if header_state != self._last_header_state:
            self.log('INFO', f'System state -> {header_state}')
            self._last_header_state = header_state
        if recording != self._last_record_state:
            self.record_status_label.setText('Recording' if recording else 'Idle')
            self._last_record_state = recording

    def _emit_periodic_logs(self, now: float):
        if (now - self._last_heartbeat_log) < self.LOG_HEARTBEAT_SEC:
            return
        self._last_heartbeat_log = now
        left_age = now - self._last_left_ts if self._last_left_ts else -1
        depth_age = now - self._last_depth_ts if self._last_depth_ts else -1
        tof_age = now - self._last_tof_ts if self._last_tof_ts else -1
        self.log('INFO', (
            'Heartbeat '
            f"| ZED RGB: {self._stream_state(self._left_seen, self._last_left_ts, now)} ({age_text(left_age) if left_age >= 0 else '--'}) "
            f"| ZED Depth: {self._stream_state(self._depth_seen, self._last_depth_ts, now)} ({age_text(depth_age) if depth_age >= 0 else '--'}) "
            f"| ToF: {self._stream_state(self._tof_seen, self._last_tof_ts, now)} ({age_text(tof_age) if tof_age >= 0 else '--'})"
        ))

    def _tick_ui(self):
        now = time.time()
        self._set_panel_statuses(now)
        self._refresh_images()
        self._update_temp_header(now)
        self._refresh_header(now)
        self._emit_periodic_logs(now)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.showNormal()
        super().keyPressEvent(event)

    def closeEvent(self, event):
        if self._bag_proc and self._bag_proc.poll() is None:
            self.stop_recording()
        event.accept()

def main():
    rclpy.init(args=None)
    node = ScrutinyGuiNode()
    app = QApplication(sys.argv)
    win = MainWindow(node)
    win.showFullScreen()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spin_timer = QTimer()
    spin_timer.timeout.connect(lambda: executor.spin_once(timeout_sec=0.0))
    spin_timer.start(10)
    try:
        ret = app.exec()
    finally:
        try:
            executor.remove_node(node)
        except Exception:
            pass
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()
    sys.exit(ret)

if __name__ == '__main__':
    main()

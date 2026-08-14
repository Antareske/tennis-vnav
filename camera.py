"""OpenCV 摄像头驱动"""
import threading
import time

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


class Camera:
    """
    OpenCV 摄像头驱动。

    使用单例模式 + 独立采集线程，只保留最新帧。
    """

    _instance: "Camera | None" = None
    _lock = threading.Lock()

    def __init__(self, device: int = 0, width: int = 320, height: int = 180, fps: int = 10):
        self._cap = None
        self._device = device
        self._width = width
        self._height = height
        self._fps = fps
        self._frame = None
        self._frame_ts = 0.0
        self._frame_lock = threading.Lock()
        self._frame_ready = threading.Event()
        self._running = False
        self._thread: threading.Thread | None = None

    @classmethod
    def get_instance(cls, device: int = 0, width: int = 320, height: int = 180, fps: int = 10) -> "Camera":
        """获取 Camera 单例"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = Camera(device, width, height, fps)
                    if not instance._open():
                        cls._instance = None  # 清理失败单例，允许后续重试
                        raise RuntimeError(f"Failed to open camera device {device}")
                    cls._instance = instance
                    cls._instance._start_capture()
        return cls._instance

    def _open(self) -> bool:
        """打开摄像头，自动扫描可用设备（应对 USB 断连导致的设备号漂移）。"""
        if not _HAS_CV2:
            return False

        # 先尝试请求的设备号，失败后扫描 0..9
        candidates = [self._device] + [i for i in range(10) if i != self._device]
        for dev in candidates:
            cap = cv2.VideoCapture(dev)
            if cap.isOpened():
                # 尝试读取一帧确认设备真实可用
                ret, _ = cap.read()
                if not ret:
                    cap.release()
                    continue
                self._cap = cap
                self._device = dev
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
                self._cap.set(cv2.CAP_PROP_FPS, self._fps)
                # 取原始 YUYV（摄像头原生格式）：省 BGR 转换，数采直接喂 VENC
                cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
                return True
            cap.release()

        self._cap = None
        return False
        return False

    def _start_capture(self):
        """启动采集线程。

        若旧线程仍在退出（重连场景），先等它结束——防止两个线程
        同时操作同一个 VideoCapture 实例。
        """
        if self._running:
            return
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2)
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self):
        """独立采集线程，只保留最新帧。

        cap 按线程本地绑定：重连替换 self._cap 后，旧线程在下一轮循环
        发现 cap 不再匹配即退出，不会与新的采集线程抢占同一设备。
        """
        interval = 1.0 / max(1, self._fps)
        cap = self._cap
        while self._running and cap is self._cap:
            loop_start = time.monotonic()
            if cap is None or not cap.isOpened():
                time.sleep(0.1)
                continue
            ret, frame = cap.read()
            if cap is not self._cap:
                # 已被重连替换，丢弃旧线程的帧
                continue
            if not ret or frame is None:
                time.sleep(0.01)
                continue
            # cv2.read() 每次返回新分配的数组，无需 copy（省 ~14ms/帧 GIL）
            with self._frame_lock:
                self._frame = frame
                self._frame_ts = time.monotonic()
            self._frame_ready.set()
            elapsed = time.monotonic() - loop_start
            if elapsed < interval:
                time.sleep(interval - elapsed)

    def is_available(self) -> bool:
        """检查摄像头是否可用"""
        return self._cap is not None and self._cap.isOpened()

    def reconnect(self) -> bool:
        """摄像头断连后自动重连（扫描所有可用设备）。"""
        self.release()
        time.sleep(0.5)
        # 重置设备号，让 _open 重新扫描
        self._device = 0
        result = self._open()
        if result:
            self._start_capture()
        return result

    def read(self):
        """读取最新帧（不拷贝，直接返回引用）"""
        with self._frame_lock:
            if self._frame is None:
                return False, None
            return True, self._frame

    @property
    def frame_ts(self) -> float:
        """当前最新帧的采集时刻（monotonic），用于同帧去重。"""
        with self._frame_lock:
            return self._frame_ts

    def read_bgr(self):
        """读取最新帧并转换为 BGR（YOLO 路径用）。

        read() 返回原始 YUYV（摄像头原生格式）；本方法转 BGR，
        cv2.cvtColor 释放 GIL，不阻塞其他线程。
        """
        ret, frame = self.read()
        if not ret or frame is None:
            return False, None
        try:
            import cv2
            bgr = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_YUYV)
            return True, bgr
        except Exception:
            return False, None

    def release(self):
        """释放摄像头资源"""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1)
            self._thread = None
        with self._frame_lock:
            self._frame = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @classmethod
    def reset(cls):
        """重置单例"""
        with cls._lock:
            if cls._instance is not None:
                cls._instance.release()
            cls._instance = None

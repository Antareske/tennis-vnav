"""数据采集模块。

以 10 FPS 独立线程采集训练数据（时间驱动，与导航主循环解耦）。
图像经硬件 VENC 编码为 JPEG 落盘，状态/动作累积在内存中，stop() 时写入 .npy。

存储格式（简化 LeRobot 兼容，可直接转换为 parquet）：
  data/episode_{id:03d}/
  ├── images/
  │   └── frame_{idx:06d}.jpg    # JPEG 图像帧 (640×480)
  ├── states.npy                  # [N, 2] float32 (left_rpm, right_rpm)
  ├── actions.npy                 # [N, 2] float32 (left_pwm, right_pwm)
  ├── timestamps.npy              # [N] float64 采样时刻（抖动分析）
  └── meta.json                   # episode 元信息 + 采样抖动统计

用法:
  collector = DataCollector(output_dir="data/episode_001", encoder=hw_enc)
  slot = SharedSlot()                       # 主循环每 tick 写入 action
  collector.start_async(camera, motor, slot)
  while navigating:
      lp, rp = sm.tick(frame)
      slot.set(lp, rp)
  collector.stop()
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── 默认参数 ──
DEFAULT_FPS = 10
JPEG_QUALITY = 60

# 全局 data 根目录（相对于 tennis-vnav 目录）
DEFAULT_DATA_ROOT = Path(__file__).parent / "data"


def _resolve_data_root(root: Optional[str | Path] = None) -> Path:
    if root is None:
        return DEFAULT_DATA_ROOT
    return Path(root)


def next_episode_id(data_root: Path) -> int:
    """扫描 data 目录，返回下一个可用的 episode ID。"""
    if not data_root.exists():
        return 1
    existing = sorted(data_root.glob("episode_*"))
    if not existing:
        return 1
    max_id = 0
    for d in existing:
        try:
            eid = int(d.name.split("_")[1])
            max_id = max(max_id, eid)
        except (IndexError, ValueError):
            pass
    return max_id + 1


class SharedSlot:
    """线程安全的 (left, right) 值槽。"""

    def __init__(self, default: tuple[int, int] = (0, 0)):
        self._value = default
        self._lock = threading.Lock()

    def set(self, left: int, right: int) -> None:
        with self._lock:
            self._value = (left, right)

    def get(self) -> tuple[int, int]:
        with self._lock:
            return self._value


class Cv2JpegEncoder:
    """软件 JPEG 编码器（回退用，VENC 不可用时）。"""

    def __init__(self, quality: int = JPEG_QUALITY):
        self.quality = quality

    def encode(self, frame: np.ndarray) -> bytes:
        import cv2
        success, encoded = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        if not success:
            raise RuntimeError("cv2 JPEG 编码失败")
        return encoded.tobytes()

    def close(self) -> None:
        pass


class DataCollector:
    """导航数据采集器（异步线程模式）。

    采样节奏由独立线程的时间基准驱动：固定 100ms 间隔，
    与导航主循环帧率无关。JPEG 编码交给硬件 VENC（ctypes 调用
    释放 GIL，编码期间主循环照常运行）。

    Args:
        output_dir: episode 输出目录
        fps: 采样帧率（默认 10）
        encoder: JPEG 编码器（encode(frame)->bytes 接口）
    """

    def __init__(
        self,
        output_dir: str | Path,
        fps: int = DEFAULT_FPS,
        encoder=None,
    ):
        self.output_dir = Path(output_dir)
        self.fps = fps
        self._interval = 1.0 / fps
        self._encoder = encoder if encoder is not None else Cv2JpegEncoder()
        self._frame_count: int = 0
        self._running: bool = False
        self._thread: Optional[threading.Thread] = None
        self._start_time: float = 0.0
        self._states: list[tuple[int, int]] = []
        self._actions: list[tuple[int, int]] = []
        self._timestamps: list[float] = []
        # 图像写入路径：优先 tmpfs（RAM，无 SD 写入停顿），
        # stop() 时整体搬移到 episode 目录。tmpfs 满时回退直接写 SD。
        self._img_dir: Path = output_dir / "images"
        self._stage_dir: Optional[Path] = None
        self._stage_full = False

    # ── 公共接口 ──

    def start_async(self, camera, motor, action_slot: SharedSlot) -> None:
        """启动独立采样线程。

        Args:
            camera: Camera 单例（read() 返回最新帧）
            motor: MotorController（get_speeds_cached 查询 RPM）
            action_slot: 主循环每 tick 更新的 action 槽
        """
        self._running = True
        self._frame_count = 0
        self._start_time = time.time()
        self._states.clear()
        self._actions.clear()
        self._timestamps.clear()

        (self.output_dir / "images").mkdir(parents=True, exist_ok=True)

        # tmpfs 暂存目录（RAM 写入，避免 SD 卡停顿干扰采样节奏）
        import tempfile
        try:
            self._stage_dir = Path(tempfile.mkdtemp(prefix="dc_stage_", dir="/tmp"))
            (self._stage_dir / "images").mkdir()
            self._stage_full = False
            logger.info("图像暂存: %s (tmpfs)", self._stage_dir)
        except OSError as e:
            logger.warning("tmpfs 暂存不可用 (%s)，直接写 SD", e)
            self._stage_dir = None

        self._thread = threading.Thread(
            target=self._sample_loop,
            args=(camera, motor, action_slot),
            daemon=True,
        )
        self._thread.start()
        logger.info("数据采集线程已启动: %s @ %d FPS", self.output_dir, self.fps)

    def _sample_loop(self, camera, motor, action_slot: SharedSlot) -> None:
        """时间驱动的采样循环。"""
        next_sample = time.time()
        while self._running:
            now = time.time()
            if now >= next_sample:
                # 采样时刻到
                ret, frame = camera.read()
                if ret and frame is not None:
                    state = motor.get_speeds_cached(max_age=0.15)
                    action = action_slot.get()
                    self._add_sample(frame, state, action)
                # 下一个采样点（若已落后则跳到未来，避免追赶）
                next_sample += self._interval
                if next_sample <= time.time():
                    next_sample = time.time() + self._interval
            else:
                # 细粒度睡眠，保证 stop() 响应速度
                time.sleep(min(next_sample - now, 0.02))

    def _add_sample(
        self,
        frame: np.ndarray,
        state: tuple[int, int],
        action: tuple[int, int],
    ) -> None:
        """编码并写入一帧采样数据。"""
        try:
            jpeg = self._encoder.encode(frame)
        except Exception as e:
            logger.error("JPEG 编码失败: %s", e)
            return

        img_name = f"frame_{self._frame_count:06d}.jpg"
        # 优先写 tmpfs 暂存；满或不可用时回退直接写 SD
        target_dir = self._stage_dir / "images" if (self._stage_dir and not self._stage_full) else self._img_dir
        try:
            with open(target_dir / img_name, "wb") as f:
                f.write(jpeg)
        except OSError:
            if not self._stage_full:
                self._stage_full = True
                logger.warning("tmpfs 已满，后续图像直接写 SD")
                target_dir = self._img_dir
                with open(target_dir / img_name, "wb") as f:
                    f.write(jpeg)

        self._states.append(state)
        self._actions.append(action)
        self._timestamps.append(time.time())
        self._frame_count += 1

        if self._frame_count % 50 == 0:
            elapsed = time.time() - self._start_time
            logger.info("数据采集中: %d 帧 (%.1fs)", self._frame_count, elapsed)

    def stop(self) -> int:
        """停止采样线程，写入 .npy 和 meta.json。返回总帧数。"""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3)
            self._thread = None

        if self._frame_count == 0:
            logger.warning("数据采集: 无帧，跳过")
            return 0

        duration = time.time() - self._start_time
        effective_fps = self._frame_count / duration if duration > 0 else 0

        states_arr = np.array(self._states, dtype=np.float32)
        actions_arr = np.array(self._actions, dtype=np.float32)
        ts_arr = np.array(self._timestamps, dtype=np.float64)
        np.save(str(self.output_dir / "states.npy"), states_arr)
        np.save(str(self.output_dir / "actions.npy"), actions_arr)
        np.save(str(self.output_dir / "timestamps.npy"), ts_arr)

        jitter = {}
        if len(ts_arr) >= 2:
            intervals = np.diff(ts_arr)
            jitter = {
                "mean_interval_ms": round(float(intervals.mean() * 1000), 1),
                "std_interval_ms": round(float(intervals.std() * 1000), 1),
                "max_interval_ms": round(float(intervals.max() * 1000), 1),
                "min_interval_ms": round(float(intervals.min() * 1000), 1),
                "target_interval_ms": round(1.0 / self.fps * 1000, 1),
            }

        meta = {
            "fps": self.fps,
            "total_frames": self._frame_count,
            "duration_s": round(duration, 2),
            "effective_fps": round(effective_fps, 2),
            "state_dim": 2,
            "action_dim": 2,
            "state_keys": ["left_rpm", "right_rpm"],
            "action_keys": ["left_pwm", "right_pwm"],
            "image_width": 640,
            "image_height": 480,
        }
        if jitter:
            meta["sampling_jitter"] = jitter
        with open(self.output_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        # 暂存图像搬移到 episode 目录（一次性顺序写，快）
        if self._stage_dir is not None and not self._stage_full:
            import shutil
            staged_images = self._stage_dir / "images"
            try:
                for f in staged_images.iterdir():
                    shutil.move(str(f), str(self._img_dir / f.name))
                logger.info("暂存图像已搬移到 %s", self._img_dir)
            except OSError as e:
                logger.warning("暂存搬移失败: %s", e)
            finally:
                shutil.rmtree(str(self._stage_dir), ignore_errors=True)
            self._stage_dir = None

        self._states.clear()
        self._actions.clear()
        self._timestamps.clear()

        # 关闭编码器（释放 VENC 硬件资源）
        if self._encoder is not None:
            try:
                self._encoder.close()
            except Exception as e:
                logger.warning("编码器关闭异常: %s", e)

        logger.info(
            "数据采集已停止: %d 帧, %.1fs, 有效 FPS=%.1f (抖动 %s)",
            self._frame_count, duration, effective_fps,
            json.dumps(jitter, ensure_ascii=False) if jitter else "N/A",
        )
        return self._frame_count

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def is_running(self) -> bool:
        return self._running

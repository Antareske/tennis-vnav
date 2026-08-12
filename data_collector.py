"""数据采集模块。

以 10 FPS 从导航主循环采集训练数据，每个 episode（一次完整导航）保存为独立目录。

存储格式（简化 LeRobot 兼容，可直接转换为 parquet）：
  data/episode_{id:03d}/
  ├── images/
  │   └── frame_{idx:06d}.jpg    # JPEG 图像帧 (640×480)
  ├── states.npy                  # [N, 2] float32 (left_rpm, right_rpm)
  ├── actions.npy                 # [N, 2] float32 (left_pwm, right_pwm)
  └── meta.json                   # episode 元信息

用法:
  collector = DataCollector(output_dir="data/episode_001")
  collector.start()
  while navigating:
      target_left, target_right = sm.tick(frame)  # 算法输出 RPM
      if collector.should_sample():
          actual_rpm = motor.get_speeds()
          collector.add(frame.copy(), actual_rpm, (target_left, target_right))
  collector.stop()
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── 默认参数 ──
DEFAULT_FPS = 10
JPEG_QUALITY = 85  # 0-100, 85 在质量和体积间平衡

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


class DataCollector:
    """导航数据采集器。

    采样策略：基于时间间隔（非帧数），确保稳定 10 FPS 输出，
    不受主循环帧率波动影响。

    图像立即写入磁盘（JPEG），状态和动作为轻量 float32 数组，
    累积在内存中，stop() 时一次性写入 .npy。

    Args:
        output_dir: episode 输出目录
        fps: 采样帧率（默认 10）
    """

    def __init__(self, output_dir: str | Path, fps: int = DEFAULT_FPS):
        self.output_dir = Path(output_dir)
        self.fps = fps
        self._interval = 1.0 / fps
        self._last_sample_time: float = 0.0
        self._frame_count: int = 0
        self._running: bool = False
        self._start_time: float = 0.0
        # 内存累积（少量 float32，一集通常 < 5 KB）
        self._states: list[tuple[int, int]] = []
        self._actions: list[tuple[int, int]] = []

    # ── 公共接口 ──

    def start(self) -> None:
        """开始采集，创建输出目录。"""
        self._running = True
        self._frame_count = 0
        self._start_time = time.time()
        self._last_sample_time = 0.0  # 首帧立即采样
        self._states.clear()
        self._actions.clear()

        (self.output_dir / "images").mkdir(parents=True, exist_ok=True)
        logger.info("数据采集已启动: %s @ %d FPS", self.output_dir, self.fps)

    def should_sample(self) -> bool:
        """检查是否到达采样时刻（基于时间间隔）。"""
        if not self._running:
            return False
        return time.time() - self._last_sample_time >= self._interval

    def add(
        self,
        frame: np.ndarray,
        state: tuple[int, int],
        action: tuple[int, int],
    ) -> None:
        """添加一帧采样数据。

        Args:
            frame: BGR 图像帧 (H, W, 3)，调用方需确保数据独立（已 copy）
            state: (left_rpm, right_rpm) 实际轮速
            action: (left_pwm, right_pwm) PWM 指令
        """
        if not self._running:
            return

        # JPEG 图像立即写入磁盘
        img_name = f"frame_{self._frame_count:06d}.jpg"
        img_path = self.output_dir / "images" / img_name
        _write_jpeg(str(img_path), frame)

        # 状态和动作累积在内存（
        self._states.append(state)
        self._actions.append(action)

        self._frame_count += 1
        self._last_sample_time = time.time()

        if self._frame_count % 50 == 0:
            elapsed = time.time() - self._start_time
            logger.info("数据采集中: %d 帧 (%.1fs)", self._frame_count, elapsed)

    def stop(self) -> int:
        """停止采集，写入 .npy 和 meta.json。返回总帧数。"""
        self._running = False

        if self._frame_count == 0:
            logger.warning("数据采集: 无帧，跳过")
            return 0

        duration = time.time() - self._start_time
        effective_fps = self._frame_count / duration if duration > 0 else 0

        # 写入 numpy 数组
        states_arr = np.array(self._states, dtype=np.float32)
        actions_arr = np.array(self._actions, dtype=np.float32)
        np.save(str(self.output_dir / "states.npy"), states_arr)
        np.save(str(self.output_dir / "actions.npy"), actions_arr)

        # 写入 meta.json
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
        with open(self.output_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        # 释放内存
        self._states.clear()
        self._actions.clear()

        logger.info(
            "数据采集已停止: %d 帧, %.1fs, 有效 FPS=%.1f",
            self._frame_count, duration, effective_fps,
        )
        return self._frame_count

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def is_running(self) -> bool:
        return self._running


# ── JPEG 写入 ──

def _write_jpeg(path: str, frame: np.ndarray, quality: int = JPEG_QUALITY) -> None:
    """写入 JPEG 文件。"""
    try:
        import cv2
        success, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if success:
            with open(path, "wb") as f:
                f.write(encoded.tobytes())
        else:
            logger.error("JPEG 编码失败: %s", path)
    except Exception as e:
        logger.error("写入 JPEG 失败 %s: %s", path, e)

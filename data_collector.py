"""数据采集模块。

以 10 FPS 独立线程采集训练数据（时间驱动，与导航主循环解耦）。
图像为相机 MJPEG 原始 JPEG 字节直写落盘（相机硬件编码，无 VENC/软件
编码环节），状态/动作累积在内存中，stop() 时写入 .npy。

数据真实性契约（每帧样本内 image/state/action/timestamps 严格一一对应）：
  - 图像写盘失败 → 该帧的三个数组一起跳过，不产生错位；
  - RPM 查询失败 → 该帧 state 记 NaN（不沿用陈旧值冒充真实反馈），
    并在 meta.json 的 uart_errors 中累计；
  - 动作槽由状态机在指令下发瞬间更新（记录 = 真实输出，见 state_machine）。

存储格式（简化 LeRobot 兼容，可直接转换为 parquet）：
  data/episode_{id:03d}/
  ├── images/
  │   └── frame_{idx:06d}.jpg    # JPEG 图像帧 (640×480)
  ├── states.npy                  # [N, 2] float32 (left_rpm, right_rpm)
  ├── actions.npy                 # [N, 2] float32 (left_pwm, right_pwm)
  ├── timestamps.npy              # [N] float64 采样时刻（抖动分析）
  └── meta.json                   # 元信息 + 采样抖动统计 + 数据一致性校验

用法:
  collector = DataCollector(output_dir="data/episode_001", encoder=hw_enc)
  slot = SharedSlot()                       # 状态机下发指令瞬间写入
  collector.start_async(camera, motor, slot)
  while navigating:
      ...
  collector.stop(completion="done")         # done / aborted / error
"""

from __future__ import annotations

import json
import logging
import shutil
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
    """软件 JPEG 编码器（仅非板端调试环境使用）。

    板端（RISC-V）不使用软件回退：VENC 不可用时 main.py 直接报错退出，
    避免软件编码（~300ms/帧）拖垮 10 FPS 且静默产出 0 帧数据。
    """

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
    与导航主循环帧率无关。图像为相机 MJPEG 原始 JPEG 字节直写
    （encoder=None），或经外部编码器编码后落盘。

    数据一致性设计：
      - 图像先写 tmpfs 暂存，episode 结束时搬移至 SD；
      - tmpfs 满时：本帧起直写 SD，已暂存帧按每采样 2 帧渐进搬移
        （不阻塞采样节奏），stop() 时无条件收尾搬移；
      - stop() 校验 images 数 == npy 行数，写入 meta 的 aligned 字段；
      - 所有失败（编码/写盘/RPM 查询/线程退出）显式计数并写入 meta，
        不允许静默产出残缺数据。

    Args:
        output_dir: episode 输出目录
        fps: 采样帧率（默认 10）
        encoder: JPEG 编码器（encode(frame)->bytes 接口）；
            None 时帧为相机 MJPEG 原始 JPEG 字节，直写落盘
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
        # encoder=None 表示帧为相机 MJPEG 原始字节，直写落盘（不再默认
        # 软件编码器：会把原始 JPEG 字节误当图像数组重新编码）
        self._encoder = encoder
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
        # ── 显式失败/异常计数（写入 meta，杜绝静默错误）──
        self._completion: str = "done"          # done / aborted / error
        self._fatal_error: Optional[str] = None  # 致命错误说明
        self._duplicate_frames: int = 0          # 同帧重复采样次数
        self._uart_errors: int = 0               # RPM 查询失败次数
        self._write_errors: int = 0              # 图像写盘失败次数
        self._encode_errors: int = 0             # JPEG 编码失败次数
        self._camera_failures: int = 0           # 摄像头读帧失败次数
        self._motor_connected: bool = False      # 采样开始时的电机连接状态
        self._last_frame_ts: float = -1.0        # 上一采样帧的摄像头时间戳
        self._last_effective_fps: float = 0.0    # stop() 定格的最终有效 FPS

    # ── 公共接口 ──

    def start_async(self, camera, motor, action_slot: SharedSlot) -> None:
        """启动独立采样线程。

        Args:
            camera: Camera 单例（read() 返回最新帧）
            motor: MotorController（get_speeds_cached_fresh 查询真实 RPM）
            action_slot: 状态机在指令下发瞬间更新的动作槽
        """
        self._running = True
        self._frame_count = 0
        self._start_time = time.time()
        self._states.clear()
        self._actions.clear()
        self._timestamps.clear()
        self._completion = "done"
        self._fatal_error = None
        self._duplicate_frames = 0
        self._uart_errors = 0
        self._write_errors = 0
        self._encode_errors = 0
        self._camera_failures = 0
        self._motor_connected = bool(getattr(motor, "is_connected", False))
        self._last_frame_ts = -1.0

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
            self._stage_full = False

        self._thread = threading.Thread(
            target=self._sample_loop,
            args=(camera, motor, action_slot),
            daemon=True,
        )
        self._thread.start()
        logger.info("数据采集线程已启动: %s @ %d FPS", self.output_dir, self.fps)

    def _sample_loop(self, camera, motor, action_slot: SharedSlot) -> None:
        """时间驱动的采样循环（异常兜底：线程不允许静默死亡）。"""
        try:
            self._run_sample_loop(camera, motor, action_slot)
        except Exception as e:
            logger.exception("采样线程异常退出: %s", e)
            self._fatal_error = f"采样线程异常: {e}"
            self._running = False

    def _run_sample_loop(self, camera, motor, action_slot: SharedSlot) -> None:
        """时间驱动的采样循环主体。"""
        next_sample = time.time()
        while self._running:
            now = time.time()
            if now >= next_sample:
                # 采样时刻到
                ret, frame = camera.read()
                if ret and frame is not None:
                    frame_ts = camera.frame_ts
                    state = self._read_state(motor)
                    action = action_slot.get()
                    self._add_sample(frame, state, action, frame_ts)
                else:
                    self._camera_failures += 1
                # 下一个采样点（若已落后则跳到未来，避免追赶）
                next_sample += self._interval
                if next_sample <= time.time():
                    next_sample = time.time() + self._interval
            else:
                # 细粒度睡眠，保证 stop() 响应速度
                time.sleep(min(next_sample - now, 0.02))

    def _read_state(self, motor) -> tuple[float, float]:
        """读取真实轮速反馈。

        RPM 查询失败（串口超时/校验失败）时返回 (NaN, NaN) 并计数，
        不沿用陈旧缓存值冒充真实反馈。
        """
        fresh = motor.get_speeds_cached_fresh(max_age=0.15)
        if fresh is None:
            self._uart_errors += 1
            return (float("nan"), float("nan"))
        return fresh

    def _add_sample(
        self,
        frame: np.ndarray,
        state: tuple[float, float],
        action: tuple[int, int],
        frame_ts: float,
    ) -> None:
        """编码并写入一帧采样数据（帧内原子：图像与数组同步跳过）。"""
        try:
            jpeg = self._to_jpeg(frame)
        except Exception as e:
            logger.error("JPEG 编码失败: %s", e)
            self._encode_errors += 1
            return

        img_name = f"frame_{self._frame_count:06d}.jpg"

        # 同帧重复采样检测（采样率 10Hz 高于摄像头 ~9.4FPS）
        if frame_ts == self._last_frame_ts:
            self._duplicate_frames += 1
        self._last_frame_ts = frame_ts

        # 写盘：优先 tmpfs 暂存；满或不可用时回退直写 SD
        written = False
        if self._stage_dir is not None and not self._stage_full:
            try:
                with open(self._stage_dir / "images" / img_name, "wb") as f:
                    f.write(jpeg)
                written = True
            except OSError:
                self._stage_full = True
                logger.warning("tmpfs 已满，本帧起直写 SD（已暂存帧渐进搬移）")
        if not written:
            try:
                with open(self._img_dir / img_name, "wb") as f:
                    f.write(jpeg)
            except OSError as e:
                # 本帧写盘失败：图像与数组一起跳过，保持对齐
                logger.error("图像写盘失败（tmpfs 与 SD 均失败）: %s（本帧跳过）", e)
                self._write_errors += 1
                return
            # tmpfs 满后渐进搬移已暂存帧（每采样 2 帧，不阻塞采样节奏）
            if self._stage_dir is not None:
                self._drain_stage(max_moves=2)

        self._states.append(state)
        self._actions.append(action)
        self._timestamps.append(time.time())
        self._frame_count += 1

        if self._frame_count % 50 == 0:
            elapsed = time.time() - self._start_time
            logger.info("数据采集中: %d 帧 (%.1fs)", self._frame_count, elapsed)

    def _to_jpeg(self, frame) -> bytes:
        """帧 → JPEG 字节。

        encoder 为 None 时帧是相机 MJPEG 原始字节（OpenCV 以 (1, N)
        数组返回，非 1-D）——以 JPEG 魔数 FFD8 判定；多维图像数组
        （非 MJPEG 相机回退）走 cv2 软件编码兜底。
        """
        if self._encoder is not None:
            return self._encoder.encode(frame)
        raw = frame.reshape(-1)
        if raw.size >= 2 and int(raw[0]) == 0xFF and int(raw[1]) == 0xD8:
            return frame.tobytes()
        import cv2
        success, encoded = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not success:
            raise RuntimeError("cv2 JPEG 编码失败")
        return encoded.tobytes()

    def _drain_stage(self, max_moves: int) -> None:
        """把 tmpfs 暂存图像渐进搬移到 episode 目录（按文件名顺序）。"""
        if self._stage_dir is None:
            return
        staged_dir = self._stage_dir / "images"
        try:
            remaining = sorted(staged_dir.iterdir())
        except OSError:
            return
        for f in remaining[:max_moves]:
            try:
                shutil.move(str(f), str(self._img_dir / f.name))
            except OSError as e:
                logger.error("暂存搬移失败: %s（收尾时再试）", e)
                return
        if max_moves > len(remaining) or not any(staged_dir.iterdir()):
            shutil.rmtree(str(self._stage_dir), ignore_errors=True)
            self._stage_dir = None
            logger.info("暂存图像已全部搬移到 %s", self._img_dir)

    def stop(self, completion: str = "done") -> int:
        """停止采样线程，写入 .npy 和 meta.json。返回总帧数。

        落盘全程容错：任何异常都不再向主循环传播，而是写入 meta 的
        error 字段（数据不允许静默残缺）。

        Args:
            completion: episode 完成方式（done / aborted / error）
        """
        self._completion = completion
        self._running = False
        # 采样时长在停止采样瞬间定格：收尾搬移/落盘不属采样活动，
        # 不计入 effective_fps 分母
        duration = time.time() - self._start_time
        if self._thread is not None:
            self._thread.join(timeout=3)
            if self._thread.is_alive():
                # 线程仍存活（编码/串口卡死）：跳过编码器关闭，避免
                # VENC use-after-close 崩溃；数据可能缺尾帧，显式标记
                logger.error("采样线程未在 3s 内退出，跳过编码器关闭")
                self._fatal_error = (self._fatal_error or "") + "采样线程未及时退出"
            self._thread = None

        # 收尾搬移（无论是否 tmpfs 满）：确保暂存图像全部落位
        self._drain_stage(max_moves=10**9)

        n_images = len(list(self._img_dir.glob("*.jpg"))) if self._img_dir.exists() else 0

        effective_fps = self._frame_count / duration if duration > 0 else 0
        self._last_effective_fps = effective_fps

        jitter = {}
        if len(self._timestamps) >= 2:
            intervals = np.diff(np.asarray(self._timestamps, dtype=np.float64))
            jitter = {
                "mean_interval_ms": round(float(intervals.mean() * 1000), 1),
                "std_interval_ms": round(float(intervals.std() * 1000), 1),
                "max_interval_ms": round(float(intervals.max() * 1000), 1),
                "min_interval_ms": round(float(intervals.min() * 1000), 1),
                "target_interval_ms": round(1.0 / self.fps * 1000, 1),
            }

        # 质量门禁（原口径：effective_fps ≥ 9.5 且 抖动 std ≤ 40ms）
        qualified = bool(jitter) and effective_fps >= 9.5 \
            and jitter["std_interval_ms"] <= 40

        # 数据对齐校验：images 数必须与 npy 行数一致
        aligned = (n_images == self._frame_count)

        if self._frame_count == 0:
            # 0 帧也写 meta（此前静默跳过，无从审计）
            self._write_meta(0, effective_fps, jitter, qualified, n_images, aligned)
            logger.warning("数据采集: 无帧（fatal=%s）", self._fatal_error)
            return 0

        # 对齐保险：采样线程末帧半写（罕见异常路径）时三数组长度可能
        # 不一致，统一截断到最短长度并显式标记——宁缺勿错位
        min_len = min(len(self._states), len(self._actions), len(self._timestamps))
        if min_len != len(self._states) or min_len != len(self._timestamps):
            self._fatal_error = (self._fatal_error or "") + \
                f"数组长度不一致（{len(self._states)}/{len(self._actions)}/{len(self._timestamps)}），已截断到 {min_len}"
            del self._states[min_len:]
            del self._actions[min_len:]
            del self._timestamps[min_len:]

        # 写 npy（NaN 表示 RPM 查询失败的帧，见 _read_state）
        try:
            states_arr = np.array(self._states, dtype=np.float32)
            actions_arr = np.array(self._actions, dtype=np.float32)
            ts_arr = np.array(self._timestamps, dtype=np.float64)
            np.save(str(self.output_dir / "states.npy"), states_arr)
            np.save(str(self.output_dir / "actions.npy"), actions_arr)
            np.save(str(self.output_dir / "timestamps.npy"), ts_arr)
        except OSError as e:
            logger.error("npy 落盘失败: %s", e)
            self._fatal_error = (self._fatal_error or "") + f"npy 落盘失败: {e}"

        self._write_meta(self._frame_count, effective_fps, jitter, qualified,
                         n_images, aligned)

        self._states.clear()
        self._actions.clear()
        self._timestamps.clear()

        # 关闭编码器（释放 VENC 硬件资源）；仅线程已退出时关闭
        if self._encoder is not None and self._fatal_error is None:
            try:
                self._encoder.close()
            except Exception as e:
                logger.warning("编码器关闭异常: %s", e)

        logger.info(
            "数据采集已停止: %d 帧, %.1fs, 有效 FPS=%.1f (抖动 %s), images=%d, aligned=%s",
            self._frame_count, duration, effective_fps,
            json.dumps(jitter, ensure_ascii=False) if jitter else "N/A",
            n_images, aligned,
        )
        return self._frame_count

    def _write_meta(self, total_frames: int, effective_fps: float,
                    jitter: dict, qualified: bool, n_images: int,
                    aligned: bool) -> None:
        """写入 meta.json（含一致性校验与异常标记，落盘失败也尽力写 error）。"""
        meta = {
            "fps": self.fps,
            "total_frames": total_frames,
            "duration_s": round(time.time() - self._start_time, 2),
            "effective_fps": round(effective_fps, 2),
            "state_dim": 2,
            "action_dim": 2,
            "state_keys": ["left_rpm", "right_rpm"],
            "action_keys": ["left_pwm", "right_pwm"],
            "image_width": 640,
            "image_height": 480,
            # ── 数据一致性校验（供质量门禁与下游对账）──
            "n_images": n_images,
            "aligned": aligned,
            "qualified": qualified,
            "completion": self._completion,
            "image_source": "camera_mjpeg" if self._encoder is None else "encoder",
            "motor_connected": self._motor_connected,
            "duplicate_frames": self._duplicate_frames,
            "uart_errors": self._uart_errors,
            "write_errors": self._write_errors,
            "encode_errors": self._encode_errors,
            "camera_failures": self._camera_failures,
        }
        if jitter:
            meta["sampling_jitter"] = jitter
        if self._fatal_error:
            meta["error"] = self._fatal_error
        try:
            with open(self.output_dir / "meta.json", "w") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
        except OSError as e:
            logger.error("meta.json 落盘失败: %s", e)
            try:
                with open(self.output_dir / "meta.json", "w") as f:
                    json.dump({"error": f"meta 落盘失败: {e}",
                               "total_frames": total_frames,
                               "completion": self._completion},
                              f, indent=2, ensure_ascii=False)
            except OSError:
                pass

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def effective_fps(self) -> float:
        """当前有效采样率：运行中为实时值，结束后为 stop() 定格值。"""
        if self._running and self._start_time > 0:
            elapsed = time.time() - self._start_time
            return self._frame_count / elapsed if elapsed > 0 else 0.0
        return self._last_effective_fps

    @property
    def is_running(self) -> bool:
        return self._running

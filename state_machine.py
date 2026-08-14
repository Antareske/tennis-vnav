"""视觉导航状态机。

状态流转：
  SEARCH → OBSERVE → APPROACH → DONE
    ↑                      │
    └── 丢球超时 ──────────┘
"""

from __future__ import annotations

import logging
import math
import time
from enum import Enum, auto
from typing import Optional

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

from config import NavConfig
from detector import yolo_infer, select_best_bbox, is_bbox_valid
from estimator import PositionEstimator
from motor import MotorController

logger = logging.getLogger(__name__)


class NavState(Enum):
    SEARCH = auto()
    OBSERVE = auto()
    APPROACH = auto()
    DONE = auto()
    ERROR = auto()


class TennisNavStateMachine:
    """视觉导航状态机。

    每个 tick() 接收当前帧和 YOLO 检测结果，返回 (left_pwm, right_pwm) 电机指令。
    """

    def __init__(
        self,
        config: NavConfig,
        motor: MotorController,
        session,
        model_path: str,
        action_slot=None,
    ):
        self.config = config
        self.motor = motor
        self.session = session
        self.model_path = model_path
        # 动作槽语义 = 实际下发的最后一条 PWM 指令。所有电机指令必须走
        # _set_raw/_brake 通道，在指令下发瞬间同步动作槽，保证数采记录的
        # 是真实输出（而非 tick 返回值，后者在"未发新指令"的 tick 与真实
        # 输出脱节）。
        self._action_slot = action_slot
        self._last_cmd: tuple[int, int] = (0, 0)

        # 位姿估计器
        self.estimator = PositionEstimator(
            img_width=config.img_width,
            img_height=config.img_height,
            hfov_deg=config.hfov_deg,
            ball_diameter_m=config.ball_diameter_m,
            max_range_m=config.max_range_m,
        )

        # 状态
        self.state = NavState.SEARCH
        self._state_start_time = time.time()

        # SEARCH 相关
        self._search_confirm_count = 0

        # OBSERVE 相关
        self._observe_settle_time: float = 0.0
        self._observe_skip_count: int = 0

        # APPROACH 相关
        self._ball_lost_count: int = 0
        self._calibrated_fwd_pwm: int = 0   # 标定后的前进 PWM
        self._calibrated_rot_pwm: int = 0   # 标定后的旋转 PWM
        self._lateral_error_accum: float = 0.0  # 横向偏差累积（差速不足时递增）
        self._prev_ball_x_sign: int = 0         # 上一帧球在左(-1)还是右(+1)
        self._last_adapt_query: float = 0.0     # 上次 RPM 查询时刻（限频用）

        # 当前 bbox（跨状态共享）
        self._current_bbox: Optional[dict] = None
        self._ball_position: Optional[tuple[float, float]] = None  # (X, Z) 米

        # 目标深度（从 bbox 宽度反算）
        fx = self.estimator.focal_length_px
        self._target_z = (fx * config.ball_diameter_m) / config.target_bbox_width

    # ── 公共接口 ──

    def tick(self, frame: np.ndarray) -> tuple[int, int]:
        """主循环 tick。

        Args:
            frame: 当前摄像头帧 (BGR)

        Returns:
            (left_pwm, right_pwm)
        """
        # 运行 YOLO（所有状态共用）
        bboxes = yolo_infer(frame, model_path=self.model_path)
        self._current_bbox = select_best_bbox(
            bboxes, self.config.img_width, self.config.img_height,
            edge_margin=self.config.bbox_edge_margin,
        )

        # 状态分发
        try:
            if self.state == NavState.SEARCH:
                return self._handle_search(frame, bboxes)
            elif self.state == NavState.OBSERVE:
                return self._handle_observe(frame)
            elif self.state == NavState.APPROACH:
                return self._handle_approach(frame)
            elif self.state == NavState.DONE:
                return 0, 0
            elif self.state == NavState.ERROR:
                return 0, 0
        except Exception as e:
            logger.exception("状态机异常: %s", e)
            # 异常必须刹车：此前直接置 ERROR 会让电机保持最后 PWM 无限运动，
            # 且 episode 永不结束（is_done 只认 DONE）
            self._brake()
            self.state = NavState.ERROR
            return 0, 0

    def set_calibrated_pwm(self, forward_pwm: int, rotation_pwm: int):
        """设置标定后的 PWM 值（由 main.py 在启动时调用）。"""
        self._calibrated_fwd_pwm = forward_pwm
        self._calibrated_rot_pwm = rotation_pwm
        logger.info("Calibrated PWM: forward=%d, rotation=%d", forward_pwm, rotation_pwm)

    # ── 电机指令通道（记录 = 真实输出）──

    def _set_raw(self, left: int, right: int) -> None:
        """下发原始 PWM，并同步更新动作槽与最后指令寄存器。"""
        self.motor.set_raw_speed(left, right)
        self._last_cmd = (left, right)
        self._publish_action()

    def _brake(self) -> None:
        """刹车（motor 层重发兜底），并同步动作槽为 (0,0)。"""
        self.motor.brake()
        self._last_cmd = (0, 0)
        self._publish_action()

    def _publish_action(self) -> None:
        if self._action_slot is not None:
            self._action_slot.set(*self._last_cmd)

    @property
    def last_cmd(self) -> tuple[int, int]:
        """实际下发的最后一条 PWM 指令（供主循环日志与数采使用）。"""
        return self._last_cmd

    def is_done(self) -> bool:
        # ERROR 同样终止 episode（收尾刹车保存数据），避免挂死
        return self.state in (NavState.DONE, NavState.ERROR)

    @property
    def status_text(self) -> str:
        if self._ball_position:
            return (f"[{self.state.name}] "
                    f"ball: ({self._ball_position[0]:.3f}, {self._ball_position[1]:.3f})m")
        return f"[{self.state.name}]"

    # ── SEARCH ──

    def _handle_search(self, frame: np.ndarray, bboxes: list) -> tuple[int, int]:
        """顺时针旋转搜索网球。"""
        img_w, img_h = self.config.img_width, self.config.img_height
        margin = self.config.bbox_edge_margin

        # 搜索阶段放宽边缘检查：球只需大部分在画面内即可
        search_margin = max(5, margin // 4)
        if self._current_bbox and is_bbox_valid(self._current_bbox, img_w, img_h, search_margin):
            self._search_confirm_count += 1
            logger.info("SEARCH: 检测到完整网球 (%d/%d)",
                        self._search_confirm_count, self.config.search_confirm_frames)

            if self._search_confirm_count >= self.config.search_confirm_frames:
                logger.info("SEARCH → OBSERVE")
                self._brake()
                self._search_confirm_count = 0
                self._transition(NavState.OBSERVE)
                self._observe_settle_time = time.time() + self.config.settle_time_ms / 1000.0
                self._observe_skip_count = 0
                return 0, 0
        else:
            self._search_confirm_count = 0

        # 继续顺时针旋转（使用标定后的旋转 PWM）
        pwm = self._calibrated_rot_pwm if self._calibrated_rot_pwm > 0 else self.config.min_effective_pwm + 3
        self._set_raw(pwm, -pwm)
        return pwm, -pwm

    # ── OBSERVE ──

    def _handle_observe(self, frame: np.ndarray) -> tuple[int, int]:
        """停车观测，获取清晰静态帧用于位姿估计。"""
        # 等待 settle
        now = time.time()
        if now < self._observe_settle_time:
            return 0, 0

        # 丢弃前几帧（清空 buffer 中的运动帧）
        if self._observe_skip_count < self.config.skip_frames_after_stop:
            self._observe_skip_count += 1
            return 0, 0

        # 取当前帧做位姿估计
        if self._current_bbox is None:
            logger.warning("OBSERVE: 未检测到网球，回到 SEARCH")
            self._transition(NavState.SEARCH)
            return 0, 0

        # 检查 bbox 完整性：球很近时（bbox 大）触碰边缘是正常的，跳过检查
        margin = self.config.bbox_edge_margin
        img_w, img_h = self.config.img_width, self.config.img_height
        bbox_too_big_for_frame = self._current_bbox["w"] > 250  # Z < ~0.2m
        if not bbox_too_big_for_frame and not is_bbox_valid(self._current_bbox, img_w, img_h, margin):
            logger.info("OBSERVE: bbox 触碰边缘，微调使其居中")
            self._nudge_to_center()
            self._observe_settle_time = time.time() + self.config.settle_time_ms / 1000.0
            self._observe_skip_count = 0
            return 0, 0

        # 位姿估计
        pos = self.estimator.estimate(self._current_bbox, reset_filter=True)
        if pos is None:
            # 估计不可靠（太远等），前进一段再搜索
            logger.info("OBSERVE: 位姿估计不可靠，前进探索")
            pwm = self.config.min_effective_pwm + 10
            self._set_raw(pwm, pwm)
            time.sleep(0.5)
            self._brake()
            self._transition(NavState.SEARCH)
            return 0, 0

        self._ball_position = pos
        # 重置 APPROACH 横向修正累积器
        self._lateral_error_accum = 0.0
        self._prev_ball_x_sign = 0
        logger.info("OBSERVE → APPROACH, 球位置: X=%.3f Z=%.3f", pos[0], pos[1])
        self._transition(NavState.APPROACH)
        return 0, 0

    def _nudge_to_center(self):
        """微调旋转使球居中。"""
        if self._current_bbox is None:
            return
        cx = self._current_bbox["cx"]
        img_center = self.config.img_width // 2
        error = cx - img_center
        if abs(error) < 30:
            return
        pwm = max(1, self._calibrated_rot_pwm // 2) if self._calibrated_rot_pwm > 0 else 3
        if error > 0:
            self._set_raw(pwm, -pwm)
        else:
            self._set_raw(-pwm, pwm)
        time.sleep(0.15)
        self._brake()

    # ── APPROACH ──

    def _handle_approach(self, frame: np.ndarray) -> tuple[int, int]:
        """简单可靠的接近策略：始终前进 + 纯差速转向。"""
        if self._current_bbox is None:
            self._ball_lost_count += 1
            if self._ball_lost_count == 1:
                self._brake()
            # 等待 ~1.5s (约 20 帧) 尝试重新捕获
            if self._ball_lost_count > 20:
                logger.warning("APPROACH: 丢失网球超时，回到 SEARCH")
                self._transition(NavState.SEARCH)
            return 0, 0
        self._ball_lost_count = 0

        # 更新位置估计
        pos = self.estimator.estimate(self._current_bbox)
        if pos is None:
            return 0, 0
        self._ball_position = pos
        ball_x, ball_z = pos

        # 到达目标距离 → 停车完成
        if ball_z <= self._target_z * 1.1:
            logger.info("APPROACH → DONE (Z=%.3f <= target=%.3f)", ball_z, self._target_z)
            self._brake()
            self._transition(NavState.DONE)
            return 0, 0

        fwd = self._calibrated_fwd_pwm

        # 横向死区：小偏移（噪声级别）直行，避免频繁换向扭动
        if abs(ball_x) < self.config.approach_deadband_m:
            self._lateral_error_accum = 0.0
            self._prev_ball_x_sign = 0
            self._set_raw(fwd, fwd)
            return fwd, fwd

        # 基础修正：横向偏差 → PWM 差异
        base_correction = int(abs(ball_x) * 80.0)

        # 偏差累积：如果车没有在纠正（偏差方向不变），逐步增大差速
        cur_sign = 1 if ball_x > 0 else -1
        if cur_sign == self._prev_ball_x_sign and self._prev_ball_x_sign != 0:
            self._lateral_error_accum += abs(ball_x) * 0.15
        else:
            self._lateral_error_accum = 0.0
        self._prev_ball_x_sign = cur_sign

        # 累积偏差转为额外差速 PWM
        escalation = int(self._lateral_error_accum * 60.0)
        lateral_pwm = min(base_correction + escalation, fwd // 2)
        if ball_x > 0:
            lp, rp = fwd, fwd - lateral_pwm
        else:
            lp, rp = fwd - lateral_pwm, fwd
        # 确保慢侧轮不低于死区，否则转不动
        deadzone = self.config.min_effective_pwm
        lp = max(deadzone, lp)
        rp = max(deadzone, rp)

        # ── PWM 自适应（掉电补偿 + 打滑补偿，严格上限）──
        # RPM 查询限频：每 300ms 查询一次（串口往返 28ms，避免占用主循环），
        # 中间帧复用缓存值
        abs_max = self._calibrated_fwd_pwm + 8  # 硬上限：标定值+8
        now = time.time()
        if now - self._last_adapt_query >= 0.30:
            actual_l, actual_r = self.motor.get_speeds()
            self._last_adapt_query = now
        else:
            actual_l, actual_r = self.motor.get_speeds_cached(max_age=1.0)
        lp = self.motor.adapt_pwm(lp, actual_l, max_pwm=abs_max)
        rp = self.motor.adapt_pwm(rp, actual_r, max_pwm=abs_max)

        # 打滑检测：每轮实际 RPM 与其 PWM 的期望 RPM 比较（左右互比会
        # 在有意图的差速转向时必然误触发，系统性削弱转向修正）。
        # 打滑轮转速显著高于其 PWM 期望值 → 微调对侧（单帧最多 +1，保持
        # 实测有效的补偿方向）。注意日志轮别：此处"左轮打滑"指左轮转速
        # 异常偏高，与补偿动作（右 PWM+1）方向一致。
        exp_l = self.motor.expected_rpm(lp)
        exp_r = self.motor.expected_rpm(rp)
        al, ar = abs(actual_l), abs(actual_r)
        if exp_l > 0 and al > exp_l * 2 + 5:
            rp = min(rp + 1, abs_max)
            logger.info("slip: 左轮打滑(L=%d exp=%.0f) → 右PWM+1=%d", actual_l, exp_l, rp)
        elif exp_r > 0 and ar > exp_r * 2 + 5:
            lp = min(lp + 1, abs_max)
            logger.info("slip: 右轮打滑(R=%d exp=%.0f) → 左PWM+1=%d", actual_r, exp_r, lp)

        self._set_raw(lp, rp)
        return lp, rp

    # ── 辅助 ──

    def _transition(self, new_state: NavState):
        """状态切换。"""
        logger.info("状态: %s → %s", self.state.name, new_state.name)
        self.state = new_state
        self._state_start_time = time.time()
        # 非运动状态下刹车
        if new_state != NavState.APPROACH:
            self._brake()

    def _update_odometry(self):
        """基于电机 RPM 更新里程计。"""
        now = time.time()
        dt = now - self._last_odom_time
        if dt <= 0:
            self._last_od_time = now
            return

        left_rpm, right_rpm = self.motor.get_speeds()
        self._last_odom_time = now

        # RPM → 轮速 (m/s)
        wheel_radius = self.config.wheel_diameter_m / 2.0
        left_vel = left_rpm * (2.0 * math.pi * wheel_radius) / 60.0
        right_vel = right_rpm * (2.0 * math.pi * wheel_radius) / 60.0

        # 差速运动学
        linear = (left_vel + right_vel) / 2.0
        angular = (right_vel - left_vel) / self.config.wheel_base_m

        x, z, heading = self._robot_pose
        heading += angular * dt
        x += linear * math.cos(heading) * dt
        z += linear * math.sin(heading) * dt

        self._robot_pose = (x, z, heading)
        self._prev_left_rpm = left_rpm
        self._prev_right_rpm = right_rpm

    def _update_waypoint_index(self):
        """根据当前里程计位置更新最近 waypoint 索引。"""
        if not self._waypoints:
            return
        rx, rz, _ = self._robot_pose
        min_dist = float("inf")
        best_idx = self._waypoint_index
        for i in range(self._waypoint_index, len(self._waypoints)):
            wx, wz = self._waypoints[i]
            d = math.sqrt((wx - rx)**2 + (wz - rz)**2)
            if d < min_dist:
                min_dist = d
                best_idx = i
        self._waypoint_index = best_idx + 1

    def _bbox_width_to_depth(self, target_width_px: int) -> float:
        """将目标 bbox 宽度转换为目标深度（使用估计器的焦距参数）。"""
        fx = self.estimator.focal_length_px
        return (fx * self.config.ball_diameter_m) / target_width_px

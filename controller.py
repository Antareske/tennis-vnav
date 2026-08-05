"""Pure Pursuit 路径跟踪 + 视觉伺服闭环修正。"""

import math
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class PurePursuitController:
    """Pure Pursuit 路径跟踪控制器。

    在给定 waypoints 列表中寻找 lookahead 圆与路径的交点作为跟踪目标，
    计算到达该目标所需的曲率，转换为线速度和角速度指令。

    Args:
        lookahead_distance_m: 前视距离 (m)
        wheel_base_m: 轮距 (m)
        max_linear_speed: 最大线速度 (m/s)
        max_angular_speed: 最大角速度 (rad/s)
    """

    def __init__(
        self,
        lookahead_distance_m: float = 0.10,
        wheel_base_m: float = 0.10,
        max_linear_speed: float = 0.25,
        max_angular_speed: float = 20.0,
    ):
        self.lookahead = lookahead_distance_m
        self.wheel_base = wheel_base_m
        self.max_linear = max_linear_speed
        self.max_angular = math.radians(max_angular_speed)

    def update(
        self,
        robot_pose: tuple[float, float, float],
        waypoints: list[tuple[float, float]],
    ) -> tuple[float, float, Optional[tuple[float, float]]]:
        """计算控制指令。

        Args:
            robot_pose: (x, z, heading_rad) 机器人当前位姿
            waypoints: [(x, z), ...] 路径点列表

        Returns:
            (linear_vel, angular_vel_rad, target_point)
        """
        if not waypoints:
            return 0.0, 0.0, None

        rx, rz, rheading = robot_pose

        # 1. 找 lookahead 圆与路径的交点
        target = self._find_lookahead_point(rx, rz, waypoints)
        if target is None:
            # 没有交点，取最后一个 waypoint
            target = waypoints[-1]

        tx, tz = target

        # 2. 目标点在机器人坐标系中的位置
        dx = tx - rx
        dz = tz - rz

        # 旋转到机器人坐标系
        cos_h = math.cos(-rheading)
        sin_h = math.sin(-rheading)
        local_x = dx * cos_h - dz * sin_h
        local_z = dx * sin_h + dz * cos_h

        # 3. Pure Pursuit 曲率: κ = 2 * sin(α) / L
        # α = 目标点相对机器人朝向的夹角
        alpha = math.atan2(local_x, local_z)
        curvature = 2.0 * math.sin(alpha) / max(self.lookahead, 0.01)

        # 4. 转换为速度指令
        # 线速度：距离终点越近越慢
        dist_to_end = math.sqrt(dx * dx + dz * dz)
        linear_vel = self.max_linear * min(1.0, dist_to_end / self.lookahead)
        linear_vel = max(0.02, linear_vel)  # 不低于最小线速度

        # 角速度
        angular_vel = curvature * linear_vel
        angular_vel = max(-self.max_angular, min(self.max_angular, angular_vel))

        return linear_vel, angular_vel, target

    def _find_lookahead_point(
        self,
        rx: float,
        rz: float,
        waypoints: list[tuple[float, float]],
    ) -> Optional[tuple[float, float]]:
        """寻找 lookahead 圆与路径的交点。

        从最近的 waypoint 开始向后搜索，返回第一个进入 lookahead 圆的点。
        """
        if not waypoints:
            return None

        # 找最近 waypoint 索引
        closest_idx = 0
        closest_dist = float("inf")
        for i, (wx, wz) in enumerate(waypoints):
            d = math.sqrt((wx - rx)**2 + (wz - rz)**2)
            if d < closest_dist:
                closest_dist = d
                closest_idx = i

        # 从最近点向后搜索交点
        for i in range(closest_idx, len(waypoints)):
            wx, wz = waypoints[i]
            d = math.sqrt((wx - rx)**2 + (wz - rz)**2)
            if d >= self.lookahead:
                return (wx, wz)

        # 所有点都在 lookahead 圆内，返回终点
        return waypoints[-1]


class VisualServoCorrector:
    """视觉伺服修正器。

    在路径跟踪的 feedforward 基础上叠加视觉反馈修正。

    Args:
        kp_lateral: 横向偏差比例增益
        kp_depth: 深度偏差比例增益
    """

    def __init__(
        self,
        kp_lateral: float = 0.10,
        kp_depth: float = 0.0,   # TRACK 阶段不修正深度——路径已规划终点
    ):
        self.kp_lateral = kp_lateral
        self.kp_depth = kp_depth

    def correct(
        self,
        ball_x: float,
        ball_z: float,
        target_x: float,
        target_z: float,
        ff_linear: float,
        ff_angular: float,
    ) -> tuple[float, float]:
        """计算修正后的速度。

        Args:
            ball_x: 当前估计的球横向位置 (m)
            ball_z: 当前估计的球深度 (m)
            target_x: 目标横向位置 (m)，通常为 0
            target_z: 目标深度 (m)
            ff_linear: feedforward 线速度
            ff_angular: feedforward 角速度

        Returns:
            (corrected_linear, corrected_angular)
        """
        error_x = ball_x - target_x    # 横向偏差
        error_z = ball_z - target_z    # 深度偏差

        # 视觉反馈修正
        v_correction = -self.kp_depth * error_z
        w_correction = self.kp_lateral * error_x

        linear = ff_linear + v_correction
        angular = ff_angular + w_correction

        return linear, angular

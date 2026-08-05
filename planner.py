"""Bezier 路径规划。

从当前机器人位姿到目标位姿（网球正前方指定距离）规划一条平滑的三阶 Bezier 曲线，
并离散化为等间距 waypoints 供 Pure Pursuit 跟踪。
"""

import math
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class BezierPath:
    """三阶 Bezier 路径。

    B(t) = (1-t)³P0 + 3(1-t)²t*P1 + 3(1-t)t²*P2 + t³P3,  t ∈ [0, 1]

    Args:
        p0, p1, p2, p3: 4 个控制点 [(x, z), ...]
    """

    def __init__(self, p0, p1, p2, p3):
        self.p0 = tuple(p0)
        self.p1 = tuple(p1)
        self.p2 = tuple(p2)
        self.p3 = tuple(p3)
        self._waypoints: Optional[list[tuple[float, float]]] = None
        self._total_length: Optional[float] = None

    def evaluate(self, t: float) -> tuple[float, float]:
        """计算曲线上 t 位置的坐标。"""
        mt = 1.0 - t
        mt2 = mt * mt
        mt3 = mt2 * mt
        t2 = t * t
        t3 = t2 * t

        x = (mt3 * self.p0[0] + 3.0 * mt2 * t * self.p1[0] +
             3.0 * mt * t2 * self.p2[0] + t3 * self.p3[0])
        z = (mt3 * self.p0[1] + 3.0 * mt2 * t * self.p1[1] +
             3.0 * mt * t2 * self.p2[1] + t3 * self.p3[1])
        return (x, z)

    def tangent(self, t: float) -> tuple[float, float]:
        """计算曲线上 t 位置的切线方向。"""
        mt = 1.0 - t
        dx = (3.0 * mt * mt * (self.p1[0] - self.p0[0]) +
              6.0 * mt * t * (self.p2[0] - self.p1[0]) +
              3.0 * t * t * (self.p3[0] - self.p2[0]))
        dz = (3.0 * mt * mt * (self.p1[1] - self.p0[1]) +
              6.0 * mt * t * (self.p2[1] - self.p1[1]) +
              3.0 * t * t * (self.p3[1] - self.p2[1]))
        length = math.sqrt(dx * dx + dz * dz)
        if length < 1e-10:
            return (0.0, 0.0)
        return (dx / length, dz / length)

    def curvature_radius(self, t: float) -> float:
        """计算曲率半径（解析公式：κ = |B'×B''| / |B'|³）。

        三阶 Bezier:
          B'(t)  = 3(1-t)²(P₁-P₀) + 6(1-t)t(P₂-P₁) + 3t²(P₃-P₂)
          B''(t) = 6(1-t)(P₂-2P₁+P₀) + 6t(P₃-2P₂+P₁)
        """
        mt = 1.0 - t

        # 一阶导数 B'(t)
        dx1 = (3.0 * mt * mt * (self.p1[0] - self.p0[0]) +
               6.0 * mt * t * (self.p2[0] - self.p1[0]) +
               3.0 * t * t * (self.p3[0] - self.p2[0]))
        dz1 = (3.0 * mt * mt * (self.p1[1] - self.p0[1]) +
               6.0 * mt * t * (self.p2[1] - self.p1[1]) +
               3.0 * t * t * (self.p3[1] - self.p2[1]))

        speed = math.sqrt(dx1 * dx1 + dz1 * dz1)
        if speed < 1e-10:
            return float("inf")

        # 二阶导数 B''(t)
        dx2 = (6.0 * mt * (self.p2[0] - 2.0 * self.p1[0] + self.p0[0]) +
               6.0 * t * (self.p3[0] - 2.0 * self.p2[0] + self.p1[0]))
        dz2 = (6.0 * mt * (self.p2[1] - 2.0 * self.p1[1] + self.p0[1]) +
               6.0 * t * (self.p3[1] - 2.0 * self.p2[1] + self.p1[1]))

        # |B' × B''| = |dx1*dz2 - dz1*dx2|
        cross = abs(dx1 * dz2 - dz1 * dx2)
        if cross < 1e-15:
            return float("inf")

        # κ = cross / speed³
        curvature = cross / (speed * speed * speed)
        return 1.0 / curvature

    def discretize(self, spacing_m: float = 0.02) -> list[tuple[float, float]]:
        """将曲线离散化为等间距 waypoints。"""
        if self._waypoints is not None:
            return self._waypoints

        # 先通过采样估计总弧长
        samples = 200
        points = [self.evaluate(i / samples) for i in range(samples + 1)]
        total_len = 0.0
        for i in range(1, len(points)):
            dx = points[i][0] - points[i - 1][0]
            dz = points[i][1] - points[i - 1][1]
            total_len += math.sqrt(dx * dx + dz * dz)
        self._total_length = total_len

        # 以等弧长间距重采样
        num_waypoints = max(2, int(total_len / spacing_m) + 1)
        waypoints = []

        # 弧长参数化重采样
        arc_len = 0.0
        prev = points[0]
        waypoints.append(prev)

        seg_idx = 1
        target_arc = spacing_m

        while seg_idx < len(points) and target_arc < total_len:
            curr = points[seg_idx]
            seg_dx = curr[0] - prev[0]
            seg_dz = curr[1] - prev[1]
            seg_len = math.sqrt(seg_dx * seg_dx + seg_dz * seg_dz)

            while arc_len + seg_len >= target_arc and target_arc < total_len:
                # 在当前段内插值
                remaining = target_arc - arc_len
                if seg_len > 0:
                    t_seg = remaining / seg_len
                    wp_x = prev[0] + seg_dx * t_seg
                    wp_z = prev[1] + seg_dz * t_seg
                else:
                    wp_x, wp_z = prev
                waypoints.append((wp_x, wp_z))
                target_arc += spacing_m

            arc_len += seg_len
            prev = curr
            seg_idx += 1

        # 确保终点在列表中
        end = points[-1]
        if math.sqrt((waypoints[-1][0] - end[0])**2 + (waypoints[-1][1] - end[1])**2) > spacing_m * 0.5:
            waypoints.append(end)

        self._waypoints = waypoints
        return waypoints

    @property
    def total_length(self) -> float:
        if self._total_length is None:
            self.discretize()
        return self._total_length or 0.0


def plan_curve(
    ball_x: float,
    ball_z: float,
    target_z: float = 0.3,
    k_forward: float = 0.4,
    min_turning_radius_m: float = 0.15,
    waypoint_spacing_m: float = 0.02,
) -> BezierPath:
    """规划一条从机器人当前位置到目标位姿的 Bezier 曲线。

    机器人位于原点 (0, 0) 面朝 +Z 方向。球位于 (ball_x, ball_z)。
    目标：机器人移动到球正前方 target_z 处，面朝球。

    Args:
        ball_x: 球横向偏移 (m)，正值 = 右侧
        ball_z: 球深度 (m)
        target_z: 目标距离（机器人末端距球多远）
        k_forward: Bezier P1 前向延伸系数
        min_turning_radius_m: 最小转弯半径
        waypoint_spacing_m: waypoint 间距

    Returns:
        BezierPath 对象
    """
    # P0: 当前位置
    p0 = (0.0, 0.0)

    # P3: 目标位置 = 球前方 target_z 处
    # 如果球在正前方 (ball_x ≈ 0)，则 P3 = (0, ball_z - target_z)
    # 如果球在侧面，机器人需要绕过去
    end_z = ball_z - target_z
    if end_z < 0.05:
        # 球太近，末端不能后退到球后面
        # 目标设为球的正前方但不越过球
        end_z = max(0.05, ball_z * 0.3)
    p3 = (ball_x, end_z)

    # P1: 沿当前方向延伸
    extend = k_forward * max(0.15, abs(ball_z - target_z))
    p1 = (0.0, extend)

    # P2: 牵引至目标方向，调整权重以平滑转弯
    # 如果横向偏移大，P2 需要更靠外以产生平缓曲线
    lateral_pull = 0.7
    depth_pull = ball_z - target_z * 0.5
    p2 = (ball_x * lateral_pull, max(extend + 0.05, depth_pull))

    logger.info("Bezier 控制点: P0%s P1%s P2%s P3%s", p0, p1, p2, p3)

    path = BezierPath(p0, p1, p2, p3)

    # 曲率检查：如果最小曲率半径小于阈值，调整 P2
    min_radius = _check_min_radius(path)
    if min_radius < min_turning_radius_m:
        logger.warning("路径最小曲率半径 %.3fm < 阈值 %.3fm，调整 P2",
                       min_radius, min_turning_radius_m)
        # 将 P2 向外推以增大转弯半径
        push_factor = min_turning_radius_m / max(min_radius, 0.01)
        p2 = (ball_x * lateral_pull * push_factor,
              max(extend + 0.05, depth_pull * min(1.0, push_factor)))
        path = BezierPath(p0, p1, p2, p3)
        logger.info("调整后: P2%s", p2)

    # 预离散化
    path.discretize(waypoint_spacing_m)
    return path


def _check_min_radius(path: BezierPath, num_samples: int = 50) -> float:
    """返回曲线上的最小曲率半径（只检查 t ∈ [0, 0.85]——跳过末端逼近段）。"""
    min_r = float("inf")
    for i in range(int(num_samples * 0.85) + 1):
        t = i / num_samples
        r = path.curvature_radius(t)
        if r < min_r:
            min_r = r
    return min_r

"""单目视觉三维位姿估计。

基于小孔成像模型，联合 YOLO bbox 尺寸和位置估计网球相对于摄像头的三维位置。

核心方法：
  - 方法 A（主）：bbox 像素宽度 + 网球物理尺寸 → 深度
  - 方法 B（辅）：bbox 底部垂直位置 + 相机安装几何 → 深度（需标定相机倾角）
  - 初版使用方法 A，方法 B 留作后续扩展

采用地面平面假设（网球在地面上），返回 (横向偏移 X, 深度 Z) 二维坐标。
"""

from __future__ import annotations

import math
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class PositionEstimator:
    """单目位姿估计器。

    Args:
        img_width: 图像宽度 (px)
        img_height: 图像高度 (px)
        hfov_deg: 水平视场角 (度)
        ball_diameter_m: 网球真实直径 (m)
        max_range_m: 最大可信距离 (m)
        ema_alpha: Z 方向 EMA 平滑系数 (0~1)
    """

    def __init__(
        self,
        img_width: int = 640,
        img_height: int = 480,
        hfov_deg: float = 62.0,
        ball_diameter_m: float = 0.067,
        max_range_m: float = 3.0,
        ema_alpha: float = 0.5,
    ):
        self.img_width = img_width
        self.img_height = img_height
        self.hfov_rad = math.radians(hfov_deg)

        # 水平焦距 (px)
        self.fx = (img_width / 2.0) / math.tan(self.hfov_rad / 2.0)
        # 垂直焦距 (px), 假设方形像素
        self.fy = self.fx

        # VFOV (从 fy 反算)
        self.vfov_rad = 2.0 * math.atan2(img_height / 2.0, self.fy)

        self.ball_diameter = ball_diameter_m
        self.max_range = max_range_m
        self.ema_alpha = ema_alpha

        # EMA 滤波状态
        self._ema_z: Optional[float] = None
        self._ema_x: Optional[float] = None

    # ── 公共接口 ──

    def estimate(
        self,
        bbox: dict,
        reset_filter: bool = False,
    ) -> Optional[tuple[float, float]]:
        """从 bbox 估计网球相对位置。

        Args:
            bbox: {"x", "y", "w", "h", "cx", "cy"} 检测框
            reset_filter: 重置 EMA 滤波器

        Returns:
            (X, Z) 相对位置 (m)，X 为横向偏移（正值 = 右侧），Z 为深度；
            如果估计不可靠返回 None
        """
        if reset_filter:
            self._ema_z = None
            self._ema_x = None

        w = bbox["w"]
        cx = bbox["cx"]
        cy = bbox["cy"]

        # 方法 A：基于 bbox 宽度估计深度
        if w <= 0:
            return None
        z_raw = (self.fx * self.ball_diameter) / w

        # 超出可信范围
        if z_raw > self.max_range:
            logger.debug("Z=%.2f > max_range=%.2f，丢弃", z_raw, self.max_range)
            return None

        # 横向偏移
        x_raw = (cx - self.img_width / 2.0) * z_raw / self.fx

        # EMA 滤波
        z = self._ema_filter("_ema_z", z_raw)
        x = self._ema_filter("_ema_x", x_raw)

        logger.debug("估计: X=%.3fm, Z=%.3fm (raw: X=%.3f, Z=%.3f)", x, z, x_raw, z_raw)
        return (x, z)

    def estimate_from_ground(
        self,
        bbox: dict,
        camera_height_m: float,
        camera_tilt_deg: float,
    ) -> Optional[tuple[float, float]]:
        """方法 B：基于 bbox 底部触地点 + 相机安装几何估计深度。

        Args:
            bbox: 检测框
            camera_height_m: 相机离地高度 (m)
            camera_tilt_deg: 光轴下倾角 (度)，正值 = 向下倾斜

        Returns:
            (X, Z) 或 None
        """
        tilt_rad = math.radians(camera_tilt_deg)
        w, cx, cy = bbox["w"], bbox["cx"], bbox["cy"]
        h = bbox["h"]

        # 球触地点在图像底部的近似位置
        y_bottom = cy + h / 2.0

        # 从图像中心计算的俯角
        alpha = math.atan2(y_bottom - self.img_height / 2.0, self.fy)

        # 总俯角 = 相机倾角 + 图像俯角
        total_depression = tilt_rad + alpha

        if total_depression <= 0:
            return None

        z = camera_height_m / math.tan(total_depression)
        if z <= 0 or z > self.max_range:
            return None

        x = (cx - self.img_width / 2.0) * z / self.fx

        return (x, z)

    def reset(self):
        """重置 EMA 滤波器。"""
        self._ema_z = None
        self._ema_x = None

    # ── 内部 ──

    def _ema_filter(self, attr: str, raw: float) -> float:
        prev = getattr(self, attr)
        if prev is None:
            setattr(self, attr, raw)
            return raw
        smoothed = self.ema_alpha * raw + (1.0 - self.ema_alpha) * prev
        setattr(self, attr, smoothed)
        return smoothed

    # ── 属性 ──

    @property
    def focal_length_px(self) -> float:
        return self.fx

    @property
    def hfov_deg(self) -> float:
        return math.degrees(self.hfov_rad)

    @property
    def vfov_deg(self) -> float:
        return math.degrees(self.vfov_rad)


def estimate_angle_to_target(x: float, z: float) -> float:
    """计算目标相对于机器人正前方的方位角。

    Args:
        x: 横向偏移 (m)
        z: 深度 (m)

    Returns:
        方位角 (度)，正值 = 右侧
    """
    return math.degrees(math.atan2(x, z))

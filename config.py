"""导航系统可标定参数，集中管理。"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class NavConfig:
    # ── 相机（USB UVC 1e45:8022, 640x480@30fps, 顶部俯视安装）──
    img_width: int = 640
    img_height: int = 480
    hfov_deg: float = 62.0          # 水平视场角（需实测标定）
    camera_device: int = 0
    camera_fps: int = 15            # 采集帧率

    # 相机安装几何（未来启用基于地面平面的深度估计）
    # camera_height_m: float = 0.15
    # camera_tilt_deg: float = 30.0

    # ── 网球 ──
    ball_diameter_m: float = 0.067  # 标准网球直径
    target_bbox_width: int = 350    # ALIGN 阶段目标 bbox 宽度 (px)
    target_bbox_tolerance: int = 20 # 宽度容差
    max_range_m: float = 3.0        # 最大可信距离（超过则估计不可靠）

    # ── 电机（TtPidChassis, PWM -100 ~ 100）──
    max_pwm: int = 100
    min_effective_pwm: int = 22     # 最小有效 PWM（实测 22-25）
    speed_scale: float = 1.0        # PWM → 速度比例系数
    max_linear_speed: float = 0.34  # 最大线速度 m/s（实测 PWM>60=104RPM=0.34m/s）
    max_angular_speed: float = 20.0 # 最大角速度 (deg/s)

    # 运动学参数
    wheel_base_m: float = 0.10      # 轮距 (m)
    wheel_diameter_m: float = 0.062 # 轮径 (m)

    # ── RPM 速度目标（替代未标定的 m/s，直接对标电机实测）──
    target_forward_rpm: float = 18.0   # 前进目标 RPM
    target_rotation_rpm: float = 14.0   # 旋转目标 RPM（调高以增强差速转向）

    # ── 搜索阶段 ──
    search_rotation_pwm: int = 22   # 搜索旋转起始 PWM（≈ 死区值，由自动标定覆盖）
    bbox_edge_margin: int = 20      # bbox 距图像边缘最小像素
    search_confirm_frames: int = 2  # 连续确认帧数（抗运动模糊）

    # ── 观测阶段 ──
    settle_time_ms: int = 200       # 停车后等待时间
    skip_frames_after_stop: int = 3 # 停车后丢弃帧数（清空 buffer）

    # ── 路径规划 ──
    bezier_k_forward: float = 0.4   # Bezier P1 前向延伸系数
    waypoint_spacing_m: float = 0.02 # 离散化间距
    min_turning_radius_m: float = 0.01  # 最小转弯半径（设很小=基本不调整曲率）

    # ── 跟踪阶段 ──
    approach_deadband_m: float = 0.05   # APPROACH 横向死区：|ball_x| 小于此值直行
    lookahead_distance_m: float = 0.10  # Pure Pursuit 前视距离
    replan_threshold_m: float = 0.15    # 位置偏差超过此值重新规划
    replan_threshold_deg: float = 10.0  # 角度偏差超过此值重新规划
    track_max_linear_speed: float = 0.025 # 跟踪时最大线速度
    track_max_angular_speed: float = 8.0  # 跟踪时最大角速度

    # ── 状态采集 ──
    state_collect_fps: int = 10          # 状态采集频率

    # ── 模型 ──
    model_path: str = "models/tennis.onnx"

    # ── 运行时标记 ──
    headless: bool = False               # True = 无 GUI，直接运行

    @classmethod
    def from_json(cls, path: str) -> "NavConfig":
        import json
        with open(path) as f:
            d = json.load(f)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_json(self, path: str) -> None:
        import json
        d = {k: getattr(self, k) for k in self.__dataclass_fields__}
        with open(path, "w") as f:
            json.dump(d, f, indent=2, ensure_ascii=False)


# 全局默认配置
default_config = NavConfig()

# 标定数据文件路径（实测后填入此文件）
CALIB_FILE = Path(__file__).parent / "calib.json"

"""电机控制模块。

封装 TtPidChassis（ESP32 UART 协议），提供：
- 死区补偿（参考 akars map_deadzone）
- 差速运动学解算（线速度/角速度 → 左右 PWM）
- 高级控制接口（前进、转向、刹车、怠速）
- PWM 自动标定（PWM→RPM 扫描）

不直接使用 tennis_hunter.py 已删除的 N20 类。
"""

import logging
import math
from typing import Optional

logger = logging.getLogger(__name__)


def _get_chassis_class():
    """延迟导入 TtPidChassis（避免在无 serial 模块的环境导入失败）。"""
    from motor_tt_pid import TtPidChassis
    return TtPidChassis


# ── 死区补偿 ──

def map_deadzone(value: int, min_speed: int, max_pwm: int = 100) -> int:
    """将 [1, max_pwm] 线性映射到 [min_speed, max_pwm]。

    参考 akars/src/motor.rs:208-216 的实现。
    效果：value=1 → min_speed, value=max_pwm → max_pwm, value=0 → 0。

    Args:
        value: 输入 PWM 值 (-max_pwm ~ max_pwm)
        min_speed: 最小有效 PWM（低于此值电机不转）
        max_pwm: 最大 PWM

    Returns:
        补偿后的 PWM 值
    """
    if value == 0:
        return 0
    sign = 1 if value > 0 else -1
    mag = abs(value)
    if mag > max_pwm:
        mag = max_pwm
    if mag < 1:
        mag = 1
    mapped = min_speed + (mag - 1) * (max_pwm - min_speed) // (max_pwm - 1)
    return sign * min(mapped, max_pwm)


# ── 差速运动学 ──

def diff_drive_to_pwm(
    linear_vel: float,
    angular_vel: float,
    wheel_base_m: float = 0.10,
    wheel_diameter_m: float = 0.062,
    max_linear_speed: float = 0.3,
    max_angular_speed: float = 30.0,
    min_effective_pwm: int = 15,
    max_pwm: int = 100,
) -> tuple[int, int]:
    """差速运动学解算 + 死区补偿。

    Args:
        linear_vel: 线速度 (m/s)
        angular_vel: 角速度 (rad/s)
        wheel_base_m: 轮距 (m)
        wheel_diameter_m: 轮径 (m)
        max_linear_speed: 最大线速度 (m/s)
        max_angular_speed: 最大角速度 (rad/s)
        min_effective_pwm: 最小有效 PWM
        max_pwm: 最大 PWM

    Returns:
        (left_pwm, right_pwm) 范围 -max_pwm ~ max_pwm
    """
    # 限幅
    linear_vel = max(-max_linear_speed, min(max_linear_speed, linear_vel))
    angular_vel = max(-max_angular_speed, min(max_angular_speed, angular_vel))

    # 差速解算：v_l = v + ω * L/2, v_r = v - ω * L/2
    half_base = wheel_base_m / 2.0
    wheel_radius = wheel_diameter_m / 2.0

    left_wheel_vel = linear_vel + angular_vel * half_base   # m/s
    right_wheel_vel = linear_vel - angular_vel * half_base  # m/s

    # 轮速 → 角速度 (rad/s) → PWM
    left_raw = int(left_wheel_vel / max_linear_speed * max_pwm)
    right_raw = int(right_wheel_vel / max_linear_speed * max_pwm)

    # 死区补偿
    left_pwm = map_deadzone(left_raw, min_effective_pwm, max_pwm)
    right_pwm = map_deadzone(right_raw, min_effective_pwm, max_pwm)

    # 钳位
    left_pwm = max(-max_pwm, min(max_pwm, left_pwm))
    right_pwm = max(-max_pwm, min(max_pwm, right_pwm))

    return left_pwm, right_pwm


# ── 高级控制接口 ──

class MotorController:
    """电机控制器，封装 TtPidChassis + 运动学解算。"""

    def __init__(
        self,
        port: str = "/dev/ttyS1",
        wheel_base_m: float = 0.10,
        wheel_diameter_m: float = 0.062,
        max_linear_speed: float = 0.3,
        max_angular_speed: float = 30.0,
        min_effective_pwm: int = 15,
        max_pwm: int = 100,
    ):
        self._chassis: Optional[TtPidChassis] = None
        self._port = port
        self.wheel_base_m = wheel_base_m
        self.wheel_diameter_m = wheel_diameter_m
        self.max_linear_speed = max_linear_speed
        self.max_angular_speed = math.radians(max_angular_speed)
        self.min_effective_pwm = min_effective_pwm
        self.max_pwm = max_pwm
        self._connected = False

        # 标定扫描数据 [(pwm, rpm), ...] 用于运行时 PWM 自适应
        self._calib_sweep: list[tuple[int, float]] = []

    def set_calib_sweep(self, sweep: list[tuple[int, float]]) -> None:
        """保存标定扫描数据，供 expected_rpm() / adapt_pwm() 使用。"""
        self._calib_sweep = sweep

    def expected_rpm(self, pwm: int) -> float:
        """根据标定数据查表返回给定 PWM 下的期望 RPM（线性插值）。"""
        if not self._calib_sweep:
            return 0.0
        pwm_abs = abs(pwm)
        sweep = self._calib_sweep
        # 边界
        if pwm_abs <= sweep[0][0]:
            frac = pwm_abs / max(1, sweep[0][0])
            return frac * sweep[0][1]
        if pwm_abs >= sweep[-1][0]:
            # 外推
            extra = pwm_abs - sweep[-1][0]
            slope = (sweep[-1][1] - sweep[-2][1]) / max(1, sweep[-1][0] - sweep[-2][0])
            return sweep[-1][1] + extra * slope
        # 插值
        for i in range(len(sweep) - 1):
            if sweep[i][0] <= pwm_abs <= sweep[i + 1][0]:
                frac = (pwm_abs - sweep[i][0]) / (sweep[i + 1][0] - sweep[i][0])
                return sweep[i][1] + frac * (sweep[i + 1][1] - sweep[i][1])
        return 0.0

    def adapt_pwm(self, target_pwm: int, actual_rpm: int,
                  max_pwm: int | None = None) -> int:
        """自适应 PWM：若实际 RPM 显著低于期望，上调 PWM 补偿掉电/负载。

        上限严格受控：最多 +2，且不超过 max_pwm 或标定最大值 +8。

        Args:
            target_pwm: 导航算法输出的原始 PWM
            actual_rpm: ESP32 实测 RPM（取绝对值）
            max_pwm: 绝对上限（默认取标定最大 PWM + 8）

        Returns:
            调整后的 PWM
        """
        if target_pwm == 0 or not self._calib_sweep:
            return target_pwm

        expected = self.expected_rpm(target_pwm)
        if expected <= 0:
            return target_pwm

        if max_pwm is None:
            max_pwm = min(self._calib_sweep[-1][0] + 8, self.max_pwm)

        rpm_abs = abs(actual_rpm)
        if rpm_abs == 0:
            # RPM=0 可能是 ESP32 查询超时（返回默认值），非真实停转，
            # 跳过本帧补偿，避免误触发
            return target_pwm
        ratio = rpm_abs / expected

        if ratio < 0.5:
            adjusted = min(target_pwm + 2, max_pwm)
            logger.info("adapt: PWM %d→%d (RPM %.0f vs exp %.0f, ratio=%.2f, 严重掉速)",
                       target_pwm, adjusted, rpm_abs, expected, ratio)
            return adjusted
        elif ratio < 0.7:
            adjusted = min(target_pwm + 1, max_pwm)
            logger.info("adapt: PWM %d→%d (RPM %.0f vs exp %.0f, ratio=%.2f, 轻微掉速)",
                       target_pwm, adjusted, rpm_abs, expected, ratio)
            return adjusted
        return target_pwm

    def connect(self) -> bool:
        """连接电机控制板。"""
        try:
            TtPidChassis = _get_chassis_class()
            self._chassis = TtPidChassis(port=self._port)
            self._connected = True
            logger.info("电机已连接: %s", self._port)
            return True
        except Exception as e:
            logger.error("电机连接失败: %s", e)
            self._connected = False
            return False

    @property
    def is_connected(self) -> bool:
        return self._connected and self._chassis is not None

    def set_speed(self, left: int, right: int) -> None:
        """直接设置左右轮 PWM（带死区补偿）。"""
        if not self.is_connected:
            logger.warning("电机未连接，忽略 set_speed")
            return
        lp = map_deadzone(left, self.min_effective_pwm, self.max_pwm)
        rp = map_deadzone(right, self.min_effective_pwm, self.max_pwm)
        self._chassis.set_speed(lp, rp)

    def set_raw_speed(self, left: int, right: int) -> None:
        """直接设置 PWM（无死区补偿，用于标定后的精确控制）。"""
        if not self.is_connected:
            return
        self._chassis.set_speed(left, right)

    def drive(self, linear_vel: float, angular_vel: float) -> None:
        """差速驱动（线速度 m/s, 角速度 rad/s）。"""
        left, right = diff_drive_to_pwm(
            linear_vel, angular_vel,
            wheel_base_m=self.wheel_base_m,
            wheel_diameter_m=self.wheel_diameter_m,
            max_linear_speed=self.max_linear_speed,
            max_angular_speed=self.max_angular_speed,
            min_effective_pwm=self.min_effective_pwm,
            max_pwm=self.max_pwm,
        )
        self.set_speed(left, right)

    def forward(self, speed: float) -> None:
        """前进。"""
        self.drive(speed, 0.0)

    def backward(self, speed: float) -> None:
        """后退。"""
        self.drive(-speed, 0.0)

    def turn_in_place(self, angular_vel: float) -> None:
        """原地旋转。"""
        self.drive(0.0, angular_vel)

    def rotate_right(self, pwm: int) -> None:
        """以指定 PWM 原地右转。"""
        pwm = abs(pwm)
        self.set_speed(pwm, -pwm)

    def rotate_left(self, pwm: int) -> None:
        """以指定 PWM 原地左转。"""
        pwm = abs(pwm)
        self.set_speed(-pwm, pwm)

    def brake(self) -> None:
        """刹车。"""
        if self.is_connected:
            self._chassis.brake()
        else:
            logger.warning("电机未连接，忽略 brake")

    def coast(self) -> None:
        """滑行停止。"""
        if self.is_connected:
            self._chassis.sleep()
        else:
            logger.warning("电机未连接，忽略 coast")

    def stop(self) -> None:
        """停止（PWM 清零）。"""
        self.set_speed(0, 0)

    def get_speeds(self) -> tuple[int, int]:
        """获取左右轮实时 RPM。"""
        if self.is_connected:
            try:
                return self._chassis.get_speeds()
            except Exception:
                return 0, 0
        return 0, 0

    def get_encoder(self) -> tuple[int, int]:
        """读取编码器累计脉冲。"""
        if self.is_connected:
            try:
                return self._chassis.get_encoder()
            except Exception:
                return 0, 0
        return 0, 0

    def calibrate_pwm_for_rpm(self, target_rpm: float, timeout_s: float = 1.0) -> int:
        """从死区开始逐步增大 PWM，找到达到 target_rpm 的 PWM 值。

        记录所有 (PWM, RPM) 扫描点到 self._calib_sweep，
        供运行时 expected_rpm() / adapt_pwm() 使用。

        Args:
            target_rpm: 目标转速 (RPM)
            timeout_s: 每个 PWM 步的等待时间（让转速稳定）

        Returns:
            达到目标 RPM 所需的最小 PWM（原始值，不经 deadzone 补偿）
        """
        if not self.is_connected:
            return self.min_effective_pwm + 5

        import time as _time
        sweep = []
        for pwm in range(self.min_effective_pwm, self.max_pwm + 1, 2):
            self._chassis.set_speed(pwm, pwm)
            _time.sleep(timeout_s)

            left_samples, right_samples = [], []
            t0 = _time.time()
            while _time.time() - t0 < 0.3:
                l, r = self._chassis.get_speeds()
                if l != 0 or r != 0:
                    left_samples.append(abs(l))
                    right_samples.append(abs(r))
                _time.sleep(0.05)

            self._chassis.brake()

            if not left_samples:
                continue

            avg_rpm = (sum(left_samples) + sum(right_samples)) / (len(left_samples) + len(right_samples))
            sweep.append((pwm, avg_rpm))
            logger.info("  calibrate: PWM=%d → RPM=%.1f", pwm, avg_rpm)

            if avg_rpm >= target_rpm:
                logger.info("Target RPM=%.0f reached at PWM=%d (measured %.1f RPM)",
                           target_rpm, pwm, avg_rpm)
                self._calib_sweep = sweep
                return pwm

        self._calib_sweep = sweep
        return self.max_pwm

    def calibrate_rotation_pwm(self, target_rpm: float, timeout_s: float = 1.0) -> int:
        """找到差速旋转时刚好达到目标角速度 RPM 的 PWM 值。"""
        if not self.is_connected:
            return self.min_effective_pwm + 3

        import time as _time
        for pwm in range(self.min_effective_pwm, self.max_pwm + 1, 2):
            self._chassis.set_speed(pwm, -pwm)
            _time.sleep(timeout_s)

            left_samples, right_samples = [], []
            t0 = _time.time()
            while _time.time() - t0 < 0.3:
                l, r = self._chassis.get_speeds()
                if l != 0 or r != 0:
                    left_samples.append(abs(l))
                    right_samples.append(abs(r))
                _time.sleep(0.05)

            self._chassis.brake()

            if not left_samples:
                continue

            avg_rpm = (sum(left_samples) + sum(right_samples)) / (len(left_samples) + len(right_samples))
            logger.info("  calibrate_rot: PWM=%d → RPM=%.1f", pwm, avg_rpm)

            if avg_rpm >= target_rpm:
                logger.info("Rotation RPM=%.0f reached at PWM=%d", target_rpm, pwm)
                return pwm

        return self.max_pwm

    def close(self) -> None:
        """释放资源。"""
        if self.is_connected:
            self._chassis.close()
            self._connected = False

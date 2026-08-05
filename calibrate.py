#!/usr/bin/env python3
"""一次性标定数据采集脚本。

在真实硬件上一次运行，采集所有需要的标定数据，减少烧录次数。

测试项目（按顺序执行）：
  1. 电机死区阈值 — 确定最小有效 PWM
  2. PWM-速度映射 — 建立 PWM 到实际 RPM 的对应关系
  3. 转弯速率标定 — 建立差速 PWM 到角速度的对应关系
  4. 相机 FOV 标定 — 借助已知距离的网球反算 HFOV
  5. 目标距离标定 — 记录夹球位置对应的 bbox 宽度
  6. 搜索旋转速度验证 — 确认搜索转速下图像是否可接受

输出:
  calib.json — 可直接被 config.py 加载的标定参数
  calib_raw.json — 完整的原始采集数据（供后续分析）

用法:
  在开发机上通过 SSH 运行（小车需连接摄像头和电机）：
    ssh root@<robot>
    cd /root/tennis-vnav
    python calibrate.py

  或本地模拟（跳过电机/摄像头测试）：
    python calibrate.py --mock
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

# 确保在 tennis-vnav 目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from config import NavConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [CALIB] %(levelname)s: %(message)s",
)
logger = logging.getLogger("calibrate")

# ── 输出文件 ──
OUTPUT_CALIB = Path(__file__).parent / "calib.json"
OUTPUT_RAW = Path(__file__).parent / "calib_raw.json"


class CalibrationRunner:
    """标定数据采集器。"""

    def __init__(self, mock: bool = False):
        self.mock = mock
        self.config = NavConfig()
        self.raw_data = {
            "timestamp": datetime.now().isoformat(),
            "mock": mock,
            "tests": {},
        }
        self.calib_result = {}

        self._camera = None
        self._motor = None
        self._session = None

    # ── 初始化 ──

    def setup(self) -> bool:
        """初始化所有硬件。"""
        if self.mock:
            logger.info("=== 模拟模式：跳过硬件初始化 ===")
            return True

        # 摄像头
        logger.info("正在打开摄像头...")
        try:
            from camera import Camera
            self._camera = Camera.get_instance(
                device=self.config.camera_device,
                width=self.config.img_width,
                height=self.config.img_height,
                fps=15,
            )
            logger.info("摄像头就绪")
        except Exception as e:
            logger.error("摄像头失败: %s", e)
            return False

        # 电机
        logger.info("正在连接电机...")
        try:
            from motor import MotorController
            self._motor = MotorController(port="/dev/ttyS1")
            if not self._motor.connect():
                logger.error("电机连接失败")
                return False
            logger.info("电机就绪")
        except Exception as e:
            logger.error("电机失败: %s", e)
            return False

        # YOLO
        logger.info("正在加载 YOLO 模型...")
        try:
            from detector import _get_session
            self._session, _ = _get_session(self.config.model_path)
            logger.info("YOLO 模型就绪")
        except Exception as e:
            logger.warning("YOLO 加载失败（FOV/距离测试将跳过）: %s", e)

        return True

    def teardown(self):
        """清理硬件。"""
        if self._motor and self._motor.is_connected:
            self._motor.brake()
            self._motor.close()
        if self._camera:
            self._camera.release()
        logger.info("硬件已释放")

    # ── 辅助 ──

    def _wait(self, seconds: float, reason: str = ""):
        if reason:
            logger.info("等待 %.1fs (%s)...", seconds, reason)
        else:
            logger.info("等待 %.1fs...", seconds)
        time.sleep(seconds)

    def _read_frame(self) -> "Optional[np.ndarray]":
        """读取一帧。"""
        if self._camera is None:
            return None
        ret, frame = self._camera.read()
        if ret and frame is not None:
            return frame
        return None

    def _get_rpm(self, duration: float = 1.0) -> tuple[float, float]:
        """在 duration 秒内测量平均 RPM。"""
        if self._motor is None:
            return 0.0, 0.0
        left_samples = []
        right_samples = []
        start = time.time()
        while time.time() - start < duration:
            try:
                l, r = self._motor.get_speeds()
                left_samples.append(l)
                right_samples.append(r)
            except Exception:
                pass
            time.sleep(0.05)

        avg_left = sum(left_samples) / len(left_samples) if left_samples else 0.0
        avg_right = sum(right_samples) / len(right_samples) if right_samples else 0.0
        return avg_left, avg_right

    def _get_encoder_delta(self, duration: float = 0.5) -> tuple[int, int]:
        """在 duration 秒内测量编码器增量。"""
        if self._motor is None:
            return 0, 0
        try:
            e1_start, e2_start = self._motor.get_encoder()
        except Exception:
            return 0, 0
        time.sleep(duration)
        try:
            e1_end, e2_end = self._motor.get_encoder()
        except Exception:
            return 0, 0
        return e1_end - e1_start, e2_end - e2_start

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 测试 1: 电机死区阈值
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def test_deadzone(self) -> dict:
        """逐步增大 PWM，检测轮子开始转动的最小 PWM。"""
        logger.info("=" * 60)
        logger.info("测试 1/6: 电机死区阈值")
        logger.info("=" * 60)

        if self.mock:
            result = {"min_effective_pwm": 15, "details": "mock"}
            self.raw_data["tests"]["deadzone"] = result
            return result

        # 安全提示
        logger.info("⚠ 确保小车底盘悬空或在地面上有足够空间！")
        logger.info("测试将从小到大测试 PWM，检测车轮何时开始转动。")
        self._wait(1.0, "准备开始")

        results = {"left": [], "right": []}

        for wheel_name, pwm_func in [
            ("left", lambda p: self._motor.set_raw_speed(p, 0)),
            ("right", lambda p: self._motor.set_raw_speed(0, p)),
        ]:
            logger.info("--- 测试 %s 轮 ---", wheel_name)

            for pwm in range(5, 40, 2):  # 5, 7, 9, ..., 39
                self._motor.stop()
                self._wait(0.3, f"PWM={pwm} 前复位")

                pwm_func(pwm)
                self._wait(0.3, "稳定")

                enc_delta, _ = self._get_encoder_delta(0.5)
                moved = abs(enc_delta) > 2  # 至少 2 个脉冲

                logger.info("  PWM=%3d: enc_delta=%4d, moved=%s", pwm, enc_delta, moved)
                results[wheel_name].append({
                    "pwm": pwm,
                    "encoder_delta": enc_delta,
                    "moved": moved,
                })

                if moved:
                    logger.info("  >>> %s 轮在 PWM=%d 时开始转动", wheel_name, pwm)
                    break

            self._motor.stop()
            self._wait(0.5, "换边")

        # 取两者的较大值作为 min_effective_pwm
        left_threshold = 15
        right_threshold = 15
        for entry in results["left"]:
            if entry["moved"]:
                left_threshold = entry["pwm"]
                break
        for entry in results["right"]:
            if entry["moved"]:
                right_threshold = entry["pwm"]
                break

        min_pwm = max(left_threshold, right_threshold)
        # 加 2 的安全余量
        min_pwm = min_pwm + 2

        result = {
            "min_effective_pwm": min_pwm,
            "left_threshold": left_threshold,
            "right_threshold": right_threshold,
            "raw": results,
        }
        self.raw_data["tests"]["deadzone"] = result
        self.calib_result["min_effective_pwm"] = min_pwm
        self._motor.stop()
        logger.info("死区阈值: min_effective_pwm=%d", min_pwm)
        return result

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 测试 2: PWM-速度映射
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def test_pwm_speed(self, min_pwm: int = 15) -> dict:
        """测量不同 PWM 对应的实际转速。"""
        logger.info("=" * 60)
        logger.info("测试 2/6: PWM-速度映射")
        logger.info("=" * 60)

        if self.mock:
            result = {"mapping": [], "details": "mock"}
            self.raw_data["tests"]["pwm_speed"] = result
            return result

        logger.info("⚠ 确保小车有足够直线行驶空间（~2m）！")
        self._wait(1.0, "准备开始")

        # 从小到大的 PWM 值
        pwm_values = [min_pwm, 20, 25, 30, 40, 50, 60, 80, 100]
        mapping = []

        for pwm in pwm_values:
            self._motor.stop()
            self._wait(0.5, "复位")

            logger.info("--- PWM=%d ---", pwm)
            self._motor.set_raw_speed(pwm, pwm)

            self._wait(0.5, "加速稳定")
            left_rpm, right_rpm = self._get_rpm(2.0)
            self._motor.stop()

            avg_rpm = (abs(left_rpm) + abs(right_rpm)) / 2.0
            # RPM → 线速度 (m/s)
            wheel_radius = self.config.wheel_diameter_m / 2.0
            linear_speed = avg_rpm * (2.0 * math.pi * wheel_radius) / 60.0

            entry = {
                "pwm": pwm,
                "left_rpm": left_rpm,
                "right_rpm": right_rpm,
                "avg_rpm": avg_rpm,
                "linear_speed_ms": linear_speed,
            }
            mapping.append(entry)
            logger.info("  PWM=%d: avg_rpm=%.1f, speed=%.3f m/s", pwm, avg_rpm, linear_speed)

            self._wait(0.5, "冷却")

        # 计算 PWM-速度比例系数
        if len(mapping) >= 2:
            pwms = [m["pwm"] for m in mapping if m["avg_rpm"] > 0]
            speeds = [m["linear_speed_ms"] for m in mapping if m["avg_rpm"] > 0]
            if len(pwms) >= 2:
                # 线性拟合: speed = scale * pwm / max_pwm
                import numpy as np
                scale = np.polyfit([p / 100.0 for p in pwms], speeds, 1)[0]
                self.calib_result["speed_scale"] = round(float(scale), 3)

        result = {"mapping": mapping}
        self.raw_data["tests"]["pwm_speed"] = result
        logger.info("PWM-速度映射完成，speed_scale=%.3f",
                    self.calib_result.get("speed_scale", 1.5))
        return result

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 测试 3: 转弯速率标定
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def test_turning_rate(self, min_pwm: int = 15) -> dict:
        """测量差速旋转时的角速度。"""
        logger.info("=" * 60)
        logger.info("测试 3/6: 转弯速率标定")
        logger.info("=" * 60)

        if self.mock:
            result = {"turning": [], "details": "mock"}
            self.raw_data["tests"]["turning_rate"] = result
            return result

        logger.info("⚠ 确保小车有旋转空间！")
        self._wait(1.0, "准备开始")

        pwm_values = [min_pwm, 20, 25, 30]
        turning = []

        for pwm in pwm_values:
            self._motor.stop()
            self._wait(0.5, "复位")

            logger.info("--- 差速 PWM=(%d, -%d) ---", pwm, pwm)
            self._motor.set_raw_speed(pwm, -pwm)

            self._wait(0.5, "加速稳定")
            left_rpm, right_rpm = self._get_rpm(2.0)
            self._motor.stop()

            # 角速度 = (right_rpm - left_rpm) * (π * D) / (L * 60) ... 近似
            # 实际用 RPM 差 ÷ 轮距
            avg_rpm_diff = (abs(left_rpm) + abs(right_rpm)) / 2.0
            wheel_radius = self.config.wheel_diameter_m / 2.0
            wheel_circumference = 2.0 * math.pi * wheel_radius
            # 一秒钟内每个轮走的距离
            linear_per_wheel = avg_rpm_diff * wheel_circumference / 60.0  # m/s
            # 角速度 (rad/s)
            angular_rads = (2.0 * linear_per_wheel) / self.config.wheel_base_m
            angular_degs = math.degrees(angular_rads)

            entry = {
                "pwm": pwm,
                "left_rpm": left_rpm,
                "right_rpm": right_rpm,
                "angular_rads": angular_rads,
                "angular_degs": angular_degs,
            }
            turning.append(entry)
            logger.info("  PWM=%d: angular=%.1f deg/s", pwm, angular_degs)

            self._wait(0.5, "冷却")

        result = {"turning": turning}
        self.raw_data["tests"]["turning_rate"] = result

        # 推荐搜索旋转 PWM：能产生 ~20-30 deg/s 的 PWM
        for entry in turning:
            if entry["angular_degs"] >= 20:
                self.calib_result["search_rotation_pwm"] = entry["pwm"]
                break

        logger.info("转弯速率标定完成")
        return result

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 测试 4: 相机 FOV 标定
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def test_fov(self) -> dict:
        """通过已知距离的网球反算水平视场角。"""
        logger.info("=" * 60)
        logger.info("测试 4/6: 相机 FOV 标定")
        logger.info("=" * 60)

        if self.mock or self._session is None:
            result = {"hfov_deg": 62.0, "details": "mock or no model"}
            self.raw_data["tests"]["fov"] = result
            self.calib_result["hfov_deg"] = 62.0
            return result

        from detector import yolo_infer, select_best_bbox

        logger.info("需要将网球放置在摄像头正前方已知距离处。")
        logger.info("请准备一把卷尺。")
        logger.info("")
        logger.info("操作步骤：")
        logger.info("  1. 将网球放在摄像头正前方 0.5m 处")
        logger.info("  2. 确保网球完整出现在画面中")
        logger.info("  3. 按 Enter 采集该距离的数据")
        logger.info("  4. 重复 1.0m, 1.5m, 2.0m")
        logger.info("")

        distances = []
        target_distances = [0.5, 1.0, 1.5, 2.0]

        for dist in target_distances:
            input(f"[按 Enter 采集 {dist}m 距离的数据]")

            # 采集多帧取平均
            bbox_widths = []
            for _ in range(10):
                frame = self._read_frame()
                if frame is None:
                    continue
                bboxes = yolo_infer(frame, model_path=self.config.model_path)
                bbox = select_best_bbox(
                    bboxes, self.config.img_width, self.config.img_height,
                    edge_margin=10,
                )
                if bbox:
                    bbox_widths.append(bbox["w"])
                time.sleep(0.1)

            if bbox_widths:
                avg_width = sum(bbox_widths) / len(bbox_widths)
                logger.info("  distance=%.1fm: avg_bbox_width=%.1f px (%d samples)",
                            dist, avg_width, len(bbox_widths))
                distances.append({"distance_m": dist, "bbox_width_px": avg_width})
            else:
                logger.warning("  距离 %.1fm: 未检测到网球！", dist)

        if len(distances) < 2:
            logger.warning("数据不足，无法标定 FOV，使用默认值 62°")
            self.calib_result["hfov_deg"] = 62.0
            result = {"hfov_deg": 62.0, "error": "insufficient data"}
            self.raw_data["tests"]["fov"] = result
            return result

        # 反算 FOV: Z = (fx * D_real) / w → fx = Z * w / D_real
        # HFOV = 2 * atan(W/2 / fx)
        ball_diam = self.config.ball_diameter_m
        img_w = self.config.img_width

        fx_values = []
        for d in distances:
            fx = d["distance_m"] * d["bbox_width_px"] / ball_diam
            fx_values.append(fx)

        fx_avg = sum(fx_values) / len(fx_values)
        hfov = 2.0 * math.degrees(math.atan2(img_w / 2.0, fx_avg))

        logger.info("推导结果: fx=%.1f px, HFOV=%.1f°", fx_avg, hfov)

        result = {
            "hfov_deg": round(hfov, 1),
            "fx_px": round(fx_avg, 1),
            "distances": distances,
        }
        self.raw_data["tests"]["fov"] = result
        self.calib_result["hfov_deg"] = round(hfov, 1)
        return result

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 测试 5: 目标距离标定
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def test_target_distance(self) -> dict:
        """记录夹球位置对应的 bbox 宽度。"""
        logger.info("=" * 60)
        logger.info("测试 5/6: 目标距离标定")
        logger.info("=" * 60)

        if self.mock or self._session is None:
            result = {"target_bbox_width": 350, "details": "mock or no model"}
            self.raw_data["tests"]["target_distance"] = result
            self.calib_result["target_bbox_width"] = 350
            return result

        from detector import yolo_infer, select_best_bbox

        logger.info("请将网球放在夹球理想位置（机械臂能抓取的位置）。")
        logger.info("即网球在车正前方，距离约为机械臂展开后抓手正下方。")
        input("[按 Enter 采集目标距离数据]")

        bbox_widths = []
        for _ in range(20):
            frame = self._read_frame()
            if frame is None:
                continue
            bboxes = yolo_infer(frame, model_path=self.config.model_path)
            bbox = select_best_bbox(bboxes, self.config.img_width, self.config.img_height)
            if bbox:
                bbox_widths.append(bbox["w"])
            time.sleep(0.1)

        if bbox_widths:
            avg_width = sum(bbox_widths) / len(bbox_widths)
            import numpy as np
            std_width = float(np.std(bbox_widths))
            logger.info("目标 bbox 宽度: %.1f ± %.1f px (%d samples)",
                        avg_width, std_width, len(bbox_widths))

            target_w = int(round(avg_width))
            result = {
                "target_bbox_width": target_w,
                "std_px": round(std_width, 1),
                "samples": len(bbox_widths),
            }
            self.raw_data["tests"]["target_distance"] = result
            self.calib_result["target_bbox_width"] = target_w
        else:
            logger.warning("未检测到网球，使用默认值 350")
            result = {"target_bbox_width": 350, "error": "no detection"}
            self.calib_result["target_bbox_width"] = 350

        self.raw_data["tests"]["target_distance"] = result
        return result

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 测试 6: 搜索旋转速度验证
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def test_search_rotation(self) -> dict:
        """以搜索 PWM 旋转，验证图像质量和 YOLO 检测可靠性。"""
        logger.info("=" * 60)
        logger.info("测试 6/6: 搜索旋转验证")
        logger.info("=" * 60)

        search_pwm = self.calib_result.get("search_rotation_pwm",
                                           self.config.search_rotation_pwm)

        if self.mock:
            result = {"search_rotation_pwm": search_pwm, "details": "mock"}
            self.raw_data["tests"]["search_rotation"] = result
            return result

        from detector import yolo_infer

        logger.info("将网球放在小车侧前方（不在视野中央），模拟搜索场景。")
        logger.info("小车将以 PWM=%d 顺时针旋转 5 秒。", search_pwm)
        logger.info("期间连续采集 YOLO 检测结果。")
        input("[按 Enter 开始搜索旋转测试]")

        detections = []
        timestamps = []

        self._motor.set_raw_speed(search_pwm, -search_pwm)
        start = time.time()
        duration = 5.0

        while time.time() - start < duration:
            frame = self._read_frame()
            if frame is None:
                continue
            ts = time.time()
            bboxes = yolo_infer(frame, model_path=self.config.model_path)
            detections.append({
                "t": ts - start,
                "n_detections": len(bboxes),
                "bboxes": [{"w": b["w"], "h": b["h"], "conf": b["conf"]}
                          for b in bboxes[:3]],
            })
            timestamps.append(ts)

        self._motor.stop()

        # 统计
        n_frames = len(detections)
        n_detected = sum(1 for d in detections if d["n_detections"] > 0)
        detection_rate = n_detected / n_frames if n_frames > 0 else 0
        fps = n_frames / duration

        logger.info("搜索旋转结果: %d 帧, %d 帧有检测 (%.0f%%), FPS=%.1f",
                    n_frames, n_detected, detection_rate * 100, fps)

        if fps < 5:
            logger.warning("⚠ FPS 偏低 (%.1f)，建议降低搜索旋转速度", fps)

        result = {
            "search_rotation_pwm": search_pwm,
            "duration_s": duration,
            "total_frames": n_frames,
            "frames_with_detection": n_detected,
            "detection_rate": detection_rate,
            "fps": fps,
            "detections": detections,
        }
        self.raw_data["tests"]["search_rotation"] = result

        if fps < 5 or detection_rate < 0.5:
            # 降低搜索 PWM
            self.calib_result["search_rotation_pwm"] = max(
                self.calib_result.get("min_effective_pwm", 15),
                search_pwm - 3,
            )
            logger.info("搜索 PWM 调整为 %d", self.calib_result["search_rotation_pwm"])

        return result

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 保存结果
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def save(self):
        """保存标定结果。"""
        # 合并默认值
        config_dict = asdict(self.config)
        config_dict.update(self.calib_result)

        with open(OUTPUT_CALIB, "w") as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        logger.info("标定参数已保存: %s", OUTPUT_CALIB)

        with open(OUTPUT_RAW, "w") as f:
            json.dump(self.raw_data, f, indent=2, ensure_ascii=False)
        logger.info("原始数据已保存: %s", OUTPUT_RAW)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 主流程
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def run_all(self):
        """按顺序执行所有标定测试。"""
        logger.info("╔══════════════════════════════════════════╗")
        logger.info("║    AKA-VNav 一次性标定数据采集           ║")
        logger.info("║    所有测试按顺序执行                    ║")
        logger.info("║    请按提示操作                          ║")
        logger.info("╚══════════════════════════════════════════╝")
        logger.info("")

        if not self.setup():
            logger.error("硬件初始化失败，无法继续")
            return

        try:
            # 测试 1-3: 电机相关（不需要 YOLO，不需要人工交互）
            deadzone = self.test_deadzone()
            min_pwm = deadzone.get("min_effective_pwm", 15)

            pwm_speed = self.test_pwm_speed(min_pwm)
            turning = self.test_turning_rate(min_pwm)

            # 更新 config 中的临时值
            self.config.min_effective_pwm = min_pwm
            if "search_rotation_pwm" in self.calib_result:
                self.config.search_rotation_pwm = self.calib_result["search_rotation_pwm"]

            # 测试 4-5: 视觉相关（需要 YOLO + 人工放置网球）
            if not self.mock and self._session:
                fov = self.test_fov()
                target = self.test_target_distance()
            else:
                logger.info("跳过测试 4-5（模拟模式或无 YOLO 模型）")

            # 测试 6: 验证（需要 YOLO + 人工放置网球）
            if not self.mock and self._session:
                search_rot = self.test_search_rotation()
            else:
                logger.info("跳过测试 6（模拟模式或无 YOLO 模型）")

        finally:
            self.teardown()

        # 保存结果
        self.save()
        logger.info("")
        logger.info("=== 标定完成 ===")
        logger.info("标定参数: %s", OUTPUT_CALIB)
        logger.info("原始数据: %s", OUTPUT_RAW)
        logger.info("")
        logger.info("下一步: 将 calib.json 中的参数填入 config.py，")
        logger.info("或直接使用: python main.py --config calib.json")


# ── 入口 ──

def main():
    parser = argparse.ArgumentParser(description="一次性标定数据采集")
    parser.add_argument("--mock", action="store_true",
                        help="模拟模式（跳过硬件操作，仅生成模板）")
    parser.add_argument("--test", type=str, default=None,
                        help="仅运行指定测试 (deadzone/pwm_speed/turning/fov/target/search)")
    args = parser.parse_args()

    runner = CalibrationRunner(mock=args.mock)

    if args.test:
        runner.setup()
        try:
            if args.test == "deadzone":
                runner.test_deadzone()
            elif args.test == "pwm_speed":
                min_pwm = runner.config.min_effective_pwm
                runner.test_pwm_speed(min_pwm)
            elif args.test == "turning":
                min_pwm = runner.config.min_effective_pwm
                runner.test_turning_rate(min_pwm)
            elif args.test == "fov":
                runner.test_fov()
            elif args.test == "target":
                runner.test_target_distance()
            elif args.test == "search":
                runner.test_search_rotation()
            runner.save()
        finally:
            runner.teardown()
    else:
        runner.run_all()


if __name__ == "__main__":
    main()

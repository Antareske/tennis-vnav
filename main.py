#!/usr/bin/env python3
"""单目纯视觉网球导航 — 入口。

基于 AKA-00 组件，实现：
  SEARCH（旋转搜索）→ OBSERVE（停车观测）→ PLAN（路径规划）
  → TRACK（连续跟踪+闭环修正）→ ALIGN（最终对准）

用法:
  python main.py                          # 使用默认配置
  python main.py --config calib.json      # 使用标定后配置
  python main.py --headless               # 无 GUI 模式
"""

import argparse
import logging
import os
import sys
import time

# 确保在 tennis-vnav 目录下运行
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from config import NavConfig, CALIB_FILE
from camera import Camera
from motor import MotorController
from state_machine import TennisNavStateMachine, NavState
from data_collector import DataCollector, next_episode_id, _resolve_data_root

# ── 日志 ──

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("tennis-nav")


# ── 状态采集线程 ──

from state_collector import get_state_collector


def start_state_collector(motor: MotorController, config: NavConfig):
    """启动独立状态采集线程（保留 AKA-00 功能，可扩展为指定频率采集）。"""
    collector = get_state_collector()
    collector.set_motor_pair(motor._chassis if motor.is_connected else None)
    collector.start()
    logger.info("状态采集线程已启动 (%d fps)", config.state_collect_fps)


def stop_state_collector():
    collector = get_state_collector()
    collector.stop()
    logger.info("状态采集线程已停止")


# ── 主循环 ──

def main():
    parser = argparse.ArgumentParser(description="单目视觉网球导航")
    parser.add_argument("--config", type=str, default=None,
                        help="JSON 标定配置文件路径")
    parser.add_argument("--headless", action="store_true",
                        help="无 GUI 模式")
    parser.add_argument("--camera", type=int, default=0,
                        help="摄像头设备编号")
    parser.add_argument("--motor-port", type=str, default="/dev/ttyS1",
                        help="电机串口路径")
    parser.add_argument("--model", type=str, default="models/tennis.onnx",
                        help="YOLO 模型路径")
    parser.add_argument("--no-collect", action="store_true",
                        help="禁用数据采集")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="数据采集输出根目录（默认 data/）")
    args = parser.parse_args()

    # 加载配置
    if args.config and os.path.exists(args.config):
        config = NavConfig.from_json(args.config)
        logger.info("从 %s 加载配置", args.config)
    elif CALIB_FILE.exists():
        config = NavConfig.from_json(str(CALIB_FILE))
        logger.info("从 %s 加载配置", CALIB_FILE)
    else:
        config = NavConfig()
        logger.info("使用默认配置")

    config.camera_device = args.camera
    config.model_path = args.model
    config.headless = args.headless

    # ── 初始化摄像头 ──
    logger.info("正在打开摄像头 (device=%d)...", config.camera_device)
    try:
        camera = Camera.get_instance(
            device=config.camera_device,
            width=config.img_width,
            height=config.img_height,
            fps=config.camera_fps,
        )
    except Exception as e:
        logger.error("摄像头初始化失败: %s", e)
        sys.exit(1)

    logger.info("摄像头已就绪: %dx%d @ %dfps",
                config.img_width, config.img_height, config.camera_fps)

    # ── 初始化电机 ──
    motor = MotorController(
        port=args.motor_port,
        wheel_base_m=config.wheel_base_m,
        wheel_diameter_m=config.wheel_diameter_m,
        max_linear_speed=config.max_linear_speed,
        max_angular_speed=config.max_angular_speed,
        min_effective_pwm=config.min_effective_pwm,
        max_pwm=config.max_pwm,
    )

    if not motor.connect():
        logger.warning("电机连接失败，将在模拟模式下运行")
    else:
        logger.info("电机已连接: %s", args.motor_port)

    # ── 启动状态采集 ──
    start_state_collector(motor, config)

    # ── 电机标定（使用经验 PWM 值，跳过动态扫描）──
    # 经验值：PWM=26 对应 ~18 RPM 前进，PWM=26 对应 ~14 RPM 旋转差速
    # 若需动态标定，取消注释下面代码：
    # if motor.is_connected:
    #     logger.info("正在标定电机（从死区搜索目标 RPM）...")
    #     fwd_pwm = motor.calibrate_pwm_for_rpm(config.target_forward_rpm)
    #     rot_pwm = motor.calibrate_rotation_pwm(config.target_rotation_rpm)
    #     logger.info("标定完成: 前进 PWM=%d, 旋转 PWM=%d", fwd_pwm, rot_pwm)
    # else:
    #     fwd_pwm = config.min_effective_pwm + 5
    #     rot_pwm = config.min_effective_pwm + 3
    fwd_pwm = 26
    rot_pwm = 26
    # 初始化经验标定 sweep，使 adapt_pwm 正常工作
    # 数据来源：多次板端实测 PWM→RPM
    motor.set_calib_sweep([(22, 6.0), (24, 16.0), (26, 22.0)])
    logger.info("使用经验 PWM: 前进=%d, 旋转=%d", fwd_pwm, rot_pwm)

    # ── 初始化 YOLO ──
    from detector import _get_session
    session, input_name = _get_session(config.model_path)

    # ── 初始化状态机 ──
    sm = TennisNavStateMachine(
        config=config,
        motor=motor,
        session=session,
        model_path=config.model_path,
    )
    sm.set_calibrated_pwm(fwd_pwm, rot_pwm)

    # ── 数据采集 ──
    collector = None
    if not args.no_collect:
        data_root = _resolve_data_root(args.data_dir)
        episode_id = next_episode_id(data_root)
        episode_dir = data_root / f"episode_{episode_id:03d}"
        collector = DataCollector(output_dir=episode_dir, fps=10)
        collector.start()
    else:
        logger.info("数据采集已禁用")

    # ── 主循环 ──
    logger.info("开始导航主循环...")
    frame_count = 0
    fps_update_time = time.time()
    fps_frame_count = 0

    try:
        camera_error_count = 0
        while not sm.is_done():
            loop_start = time.time()

            # 读取帧（带断连重试）
            ret, frame = camera.read()
            if not ret or frame is None:
                camera_error_count += 1
                if camera_error_count > 30:  # ~2s 无帧，尝试重连
                    logger.warning("摄像头无数据，尝试重连...")
                    if camera.reconnect():
                        logger.info("摄像头重连成功")
                        camera_error_count = 0
                        continue
                    else:
                        logger.error("摄像头重连失败，退出")
                        break
                time.sleep(0.01)
                continue
            camera_error_count = 0

            frame_count += 1
            fps_frame_count += 1

            # 状态机 tick
            left_pwm, right_pwm = sm.tick(frame)

            # 数据采集 (10 FPS)
            if collector is not None and collector.should_sample():
                left_rpm, right_rpm = motor.get_speeds()
                collector.add(frame.copy(), (left_rpm, right_rpm),
                             (left_pwm, right_pwm))

            # FPS 显示
            now = time.time()
            if now - fps_update_time >= 2.0:
                fps = fps_frame_count / (now - fps_update_time)
                logger.info("%s | FPS=%.1f | PWM=(%d, %d)",
                            sm.status_text, fps, left_pwm, right_pwm)
                fps_update_time = now
                fps_frame_count = 0

            # 帧率控制
            elapsed = time.time() - loop_start
            target_interval = 1.0 / config.camera_fps
            if elapsed < target_interval:
                time.sleep(target_interval - elapsed)

        logger.info("导航完成！小车已到达目标位置。")
    except KeyboardInterrupt:
        logger.info("用户中断")
    except Exception as e:
        logger.exception("主循环异常: %s", e)
    finally:
        # ── 停止数据采集 ──
        if collector is not None:
            count = collector.stop()
            if count > 0:
                logger.info("数据已保存到: %s (%d 帧)", collector.output_dir, count)

        # ── 清理 ──
        logger.info("正在停止...")
        motor.brake()
        motor.close()
        stop_state_collector()
        camera.release()
        logger.info("已停止，共处理 %d 帧", frame_count)


if __name__ == "__main__":
    main()

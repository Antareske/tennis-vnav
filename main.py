#!/usr/bin/env python3
"""单目纯视觉网球导航 — 入口（多轮数采模式）。

工作流程：
  启动 → IDLE（等网页"开始采集"命令）
       → 采集一轮（0.5s 静止 → SEARCH → OBSERVE → APPROACH → DONE）
       → 保存数据 → 回到 IDLE

网页控制（ctrl-serve，http://192.168.4.1）通过 /tmp/vnav/ 文件通信：
  cmd.txt:    start / abort / clear
  status.json: 当前组号、阶段、帧数、FPS、错误

用法:
  python main.py --no-collect          # 单轮模式（禁用数采，仅导航）
  python main.py --data-dir DIR        # 数据根目录
"""

import argparse
import logging
import os
import signal
import sys
import time
from pathlib import Path

# 确保在 tennis-vnav 目录下运行
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from config import NavConfig, CALIB_FILE
from camera import Camera
from motor import MotorController
from state_machine import TennisNavStateMachine, NavState
from data_collector import (DataCollector, SharedSlot,
                            next_episode_id, _resolve_data_root)
from vnav_control import (read_command, wait_command, write_status,
                          count_episodes, delete_last_episode)

# ── 日志 ──

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("tennis-nav")


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
                        help="禁用数据采集（仅导航调试）")
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
    logger.info("摄像头已就绪: %dx%d @ %dfps%s",
                config.img_width, config.img_height, config.camera_fps,
                " (MJPEG 原始流)" if camera.is_mjpeg else " (非 MJPEG)")

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

    # ── 电机经验标定（来自 NavConfig，板端实测值；无运行时标定环节）──
    fwd_pwm = config.calib_fwd_pwm
    rot_pwm = config.calib_rot_pwm
    motor.set_calib_sweep(list(config.calib_sweep))
    logger.info("使用经验 PWM（NavConfig）: 前进=%d, 旋转=%d", fwd_pwm, rot_pwm)

    # ── 信号处理：SIGTERM 时先刹车再退出（finally 会再清理；
    #     崩溃路径无 finally 时电机保持最后 PWM，故必须急停）──
    def _on_terminate(signum, _frame):
        logger.warning("收到信号 %d，刹车后退出", signum)
        try:
            motor.brake()
        except Exception:
            pass
        sys.exit(128 + signum)

    signal.signal(signal.SIGTERM, _on_terminate)

    # ── 初始化 YOLO ──
    from detector import _get_session
    session, input_name = _get_session(config.model_path)

    # ── 清理旧命令文件，准备控制接口 ──
    try:
        os.remove("/tmp/vnav/cmd.txt")
    except OSError:
        pass
    data_root = _resolve_data_root(args.data_dir)
    last_episode_id = next_episode_id(data_root) - 1 if data_root.exists() else 0

    def push_status(phase: str, **extra):
        """写入状态文件。"""
        st = {
            "phase": phase,
            "episode_id": extra.get("episode_id"),
            "last_episode": last_episode_id,
            "total_episodes": count_episodes(data_root),
            "nav_state": extra.get("nav_state"),
            "frames": extra.get("frames", 0),
            "effective_fps": extra.get("effective_fps"),
            "error": extra.get("error"),
        }
        write_status(st)

    logger.info("就绪，等待网页命令 (http://192.168.4.1)...")

    try:
        while True:
            # ══════════ IDLE：等待 start 命令 ══════════
            push_status("IDLE")
            logger.info("── 空闲：等待开始采集命令 ──")
            while True:
                cmd = wait_command(0.5)
                if cmd == "start":
                    break
                elif cmd == "clear":
                    deleted = delete_last_episode(data_root)
                    last_episode_id = next_episode_id(data_root) - 1 if data_root.exists() else 0
                    logger.info("clear 完成，最近组: %d", last_episode_id)
                    push_status("IDLE")
                elif cmd == "abort":
                    # 空闲阶段的中止无意义，忽略
                    pass

            # ══════════ COLLECTING：采集一轮 ══════════
            episode_id = next_episode_id(data_root)
            episode_dir = data_root / f"episode_{episode_id:03d}"
            logger.info("── 第 %d 组采集开始: %s ──", episode_id, episode_dir)

            action_slot = SharedSlot()

            # 数采编码：相机 MJPEG 原始 JPEG 字节直写落盘（相机硬件编码，
            # 无 VENC/软件编码环节；内核 UVC 负载远低于 YUYV 原始流）
            collector = None
            if not args.no_collect:
                try:
                    collector = DataCollector(output_dir=episode_dir, fps=10,
                                              encoder=None)
                    collector.start_async(camera, motor, action_slot)
                    time.sleep(0.5)  # 导航前静止采样
                except Exception as e:
                    logger.exception("数采初始化失败: %s", e)
                    motor.brake()
                    push_status("IDLE", episode_id=episode_id,
                                error=f"第 {episode_id} 组初始化失败: {e}")
                    continue

            # 每集新状态机（清空位姿估计与状态；动作槽 = 实际下发指令）
            sm = TennisNavStateMachine(
                config=config,
                motor=motor,
                session=session,
                model_path=config.model_path,
                action_slot=action_slot,
            )
            sm.set_calibrated_pwm(fwd_pwm, rot_pwm)

            aborted = False
            error_msg = None
            frame_count = 0
            last_tick_time = 0.0
            last_frame_ts = -1.0
            last_status_time = 0.0
            fps_update_time = time.time()
            fps_frame_count = 0

            try:
                while not sm.is_done():
                    # 命令检查（每帧，非阻塞）
                    cmd = read_command()
                    if cmd == "abort":
                        aborted = True
                        logger.info("收到中止命令")
                        break
                    elif cmd == "start" or cmd == "clear":
                        # 采集中忽略这两个命令
                        pass

                    # 摄像头帧停滞检测（断连/掉线）：刹车 + 重连，
                    # 重连失败中止本集（此前帧冻结会让电机保持最后 PWM 空转）
                    if time.monotonic() - camera.frame_ts > config.camera_stall_timeout_s:
                        motor.brake()
                        action_slot.set(0, 0)
                        logger.error("摄像头帧停滞 %.1fs，刹车并尝试重连",
                                     time.monotonic() - camera.frame_ts)
                        reconnected = False
                        for attempt in range(3):
                            logger.info("摄像头重连尝试 %d/3", attempt + 1)
                            if camera.reconnect():
                                reconnected = True
                                break
                            time.sleep(1.0)
                        if reconnected:
                            logger.info("摄像头重连成功")
                            last_frame_ts = camera.frame_ts
                            continue
                        error_msg = "摄像头重连失败，中止本集"
                        logger.error(error_msg)
                        break

                    # 限频 + 去重：先查帧时间戳（MJPEG 解码 ~80ms/帧，
                    # 必须先过滤重复帧，否则每轮空转都要白付解码成本）
                    now = time.time()
                    if camera.frame_ts == last_frame_ts or now - last_tick_time < 0.15:
                        time.sleep(0.01)
                        continue
                    last_frame_ts = camera.frame_ts
                    last_tick_time = now

                    # 读取并解码帧（仅新帧）
                    ret, frame = camera.read_bgr()
                    if not ret or frame is None:
                        time.sleep(0.01)
                        continue
                    frame_count += 1

                    # 状态机 tick（动作槽由状态机在指令下发瞬间同步，
                    # 记录真实输出而非 tick 返回值）
                    sm.tick(frame)
                    fps_frame_count += 1

                    # FPS 显示
                    if now - fps_update_time >= 2.0:
                        fps = fps_frame_count / (now - fps_update_time)
                        logger.info("%s | FPS=%.1f | PWM=(%d, %d)",
                                    sm.status_text, fps, *sm.last_cmd)
                        fps_update_time = now
                        fps_frame_count = 0

                    # 状态推送（~1Hz，含实时有效 FPS 供网页显示）
                    if now - last_status_time >= 1.0:
                        last_status_time = now
                        push_status("COLLECTING",
                                    episode_id=episode_id,
                                    nav_state=sm.state.name,
                                    frames=collector.frame_count if collector else 0,
                                    effective_fps=collector.effective_fps if collector else None)

                # 正常完成 → 导航后静止采样 1s
                if not aborted and sm.state == NavState.DONE:
                    action_slot.set(0, 0)
                    if collector is not None:
                        logger.info("导航完成，导航后静止采样 1s...")
                        time.sleep(1.0)
                elif not aborted and sm.state == NavState.ERROR:
                    error_msg = "状态机异常终止（已刹车）"
            except Exception as e:
                logger.exception("主循环异常: %s", e)
                error_msg = f"{e}"

            # ── 收尾 ──
            motor.brake()
            action_slot.set(0, 0)
            if collector is not None:
                if aborted:
                    completion = "aborted"
                elif error_msg or sm.state == NavState.ERROR:
                    completion = "error"
                else:
                    completion = "done"
                try:
                    count = collector.stop(completion=completion)
                    if count > 0:
                        logger.info("数据已保存到: %s (%d 帧)", collector.output_dir, count)
                except Exception as e:
                    logger.exception("数据收尾异常（已尽力保存）: %s", e)
                    push_status("IDLE", episode_id=episode_id,
                                error=f"第 {episode_id} 组收尾出错: {e}")
            last_episode_id = episode_id

            # 本轮结果推送到网页（含最终有效 FPS）
            push_status("IDLE", episode_id=episode_id,
                        frames=collector.frame_count if collector else 0,
                        effective_fps=collector.effective_fps if collector else None)

            if aborted:
                logger.info("第 %d 组已中止（数据已保存）", episode_id)
            elif error_msg:
                logger.error("第 %d 组异常结束: %s", episode_id, error_msg)
                push_status("IDLE", episode_id=episode_id, error=f"第 {episode_id} 组出错: {error_msg}")
            else:
                logger.info("第 %d 组完成", episode_id)

    except KeyboardInterrupt:
        logger.info("用户中断")
    finally:
        # ── 清理 ──
        logger.info("正在停止...")
        motor.brake()
        motor.close()
        camera.release()
        logger.info("已停止")


if __name__ == "__main__":
    main()

"""vnav 控制接口 — 与 ctrl-serve（网页控制服务）的文件通信。

通信协议（/tmp/vnav/ 目录，tmpfs）：
  cmd.txt       ctrl-serve 写命令（start/abort/clear），导航进程读后删除
  status.json   导航进程原子写状态，ctrl-serve 按需读取返回给手机

命令语义：
  start   开始新一组采集（导航立即开始）
  abort   中止当前导航，保存已采数据，回到空闲
  clear   删除最近一组数据（仅空闲阶段有效）
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

CTRL_DIR = "/tmp/vnav"
CMD_FILE = "/tmp/vnav/cmd.txt"
STATUS_FILE = "/tmp/vnav/status.json"


def read_command() -> Optional[str]:
    """读取并消费命令文件（rename 原子消费）。

    先 rename 再读内容：与 ctrl-serve 的 tmp+rename 原子写同构，
    消除「读到旧命令后删除时误删新命令」的竞态窗口。
    """
    consumed = CMD_FILE + ".consumed"
    try:
        os.replace(CMD_FILE, consumed)
    except OSError:
        return None
    try:
        cmd = Path(consumed).read_text().strip()
    except OSError:
        cmd = ""
    finally:
        try:
            os.remove(consumed)
        except OSError:
            pass
    if cmd:
        logger.info("收到命令: %s", cmd)
        return cmd
    return None


def wait_command(timeout: float = 0.5) -> Optional[str]:
    """阻塞等待命令（空闲阶段用），超时返回 None。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        cmd = read_command()
        if cmd:
            return cmd
        time.sleep(0.05)
    return None


def write_status(status: dict) -> None:
    """原子写状态文件（tmp+rename，避免 ctrl-serve 读到半截）。"""
    try:
        os.makedirs(CTRL_DIR, exist_ok=True)
        tmp = STATUS_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(status, f, ensure_ascii=False)
        os.replace(tmp, STATUS_FILE)
    except OSError as e:
        logger.warning("写状态文件失败: %s", e)


def _episode_id_from_name(name: str) -> Optional[int]:
    """从目录名解析 episode 编号（非数字返回 None，不抛异常）。"""
    try:
        return int(name.split("_", 1)[1])
    except (IndexError, ValueError):
        return None


def count_episodes(data_root: Path) -> int:
    """统计 data 目录中的组数（仅计数已落盘完成的 episode）。"""
    if not data_root.exists():
        return 0
    count = 0
    for d in data_root.iterdir():
        if not d.is_dir() or not d.name.startswith("episode_"):
            continue
        if _episode_id_from_name(d.name) is None:
            continue
        if (d / "states.npy").exists():
            count += 1
    return count


def delete_last_episode(data_root: Path) -> Optional[int]:
    """删除编号最大的一组，返回被删的组号（无数据返回 None）。

    按数值编号排序（非字典序）；非法目录名跳过不崩溃。
    """
    if not data_root.exists():
        return None
    best_eid: Optional[int] = None
    best_dir = None
    for d in data_root.iterdir():
        if not d.is_dir() or not d.name.startswith("episode_"):
            continue
        eid = _episode_id_from_name(d.name)
        if eid is None:
            logger.warning("忽略非法 episode 目录名: %s", d.name)
            continue
        if best_eid is None or eid > best_eid:
            best_eid = eid
            best_dir = d
    if best_dir is None:
        logger.warning("无合法 episode 目录可删除")
        return None
    shutil.rmtree(str(best_dir), ignore_errors=True)
    logger.info("已删除第 %d 组数据: %s", best_eid, best_dir)
    return best_eid

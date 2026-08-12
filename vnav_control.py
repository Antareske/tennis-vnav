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
    """读取并消费命令文件（读到即删，原子消费）。"""
    try:
        cmd = Path(CMD_FILE).read_text().strip()
    except OSError:
        return None
    if cmd:
        try:
            os.remove(CMD_FILE)
        except OSError:
            pass
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


def count_episodes(data_root: Path) -> int:
    """统计 data 目录中的组数。"""
    if not data_root.exists():
        return 0
    return len([d for d in data_root.iterdir() if d.is_dir() and d.name.startswith("episode_")])


def delete_last_episode(data_root: Path) -> Optional[int]:
    """删除编号最大的一组，返回被删的组号（无数据返回 None）。"""
    if not data_root.exists():
        return None
    episodes = sorted(
        (d for d in data_root.iterdir() if d.is_dir() and d.name.startswith("episode_")),
        key=lambda d: d.name,
    )
    if not episodes:
        return None
    last = episodes[-1]
    eid = int(last.name.split("_")[1])
    shutil.rmtree(str(last), ignore_errors=True)
    logger.info("已删除第 %d 组数据: %s", eid, last)
    return eid

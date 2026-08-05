#!/bin/bash
# tennis-vnav 启动脚本 (Linux)
#
# 基于 AKA-00 init.sh 适配。
# - 初始化串口
# - 生成 SSL 证书
# - 启动主程序（带自动重启）
#
# 用法:
#   ./scripts/start.sh                 # 前台运行
#   ./scripts/start.sh --daemon        # 后台运行（通过 systemd 管理）

set -e

APP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOCK_FILE="/tmp/tennis-vnav-ota.lock"
PID_FILE="/var/run/tennis-vnav.pid"

# 防止重复启动
if [ -f "$PID_FILE" ]; then
    _old=$(cat "$PID_FILE")
    if kill -0 "$_old" 2>/dev/null; then
        echo "[start] 已在运行 (pid $_old)，退出"
        exit 0
    fi
fi
echo $$ > "$PID_FILE"

# ── UART 初始化（如果设备存在）──
for dev in /dev/ttyS1 /dev/ttyS2 /dev/ttyACM0; do
    if [ -e "$dev" ]; then
        stty -F "$dev" raw -echo 115200 2>/dev/null || true
        echo "[start] 初始化 $dev"
    fi
done

# ── SSL 证书 ──
CERT_DIR="$APP_DIR"
if ! ( [ -f "$CERT_DIR/cert.pem" ] && [ -f "$CERT_DIR/key.pem" ] ); then
    echo "[start] 生成自签名 SSL 证书..."
    openssl req -x509 -newkey rsa:2048 \
        -keyout "$CERT_DIR/key.pem" \
        -out "$CERT_DIR/cert.pem" \
        -days 3650 -nodes \
        -subj "/CN=tennis-vnav" 2>/dev/null || true
fi

# ── 主程序路径 ──
MAIN_SCRIPT="$APP_DIR/main.py"

# ── Python 检查 ──
if ! command -v python3 &>/dev/null; then
    echo "[start] 错误: 未找到 python3"
    exit 1
fi

# ── 自动重启循环 ──
echo "[start] tennis-vnav 启动 (PID $$)"
echo "[start] 工作目录: $APP_DIR"

while true; do
    # 等待 OTA 锁释放
    while [ -f "$LOCK_FILE" ]; do
        echo "[start] OTA 进行中，等待..."
        sleep 0.5
    done

    echo "[start] 启动导航程序..."
    cd "$APP_DIR"
    python3 "$MAIN_SCRIPT" "$@" || true

    ec=$?
    echo "[start] Python 退出 (exit=$ec), 2s 后重启..."
    sleep 2
done

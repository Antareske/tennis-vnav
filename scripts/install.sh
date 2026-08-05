#!/bin/bash
# tennis-vnav 一键安装脚本
#
# 在小车上运行（通过 SSH），完成:
#   1. 安装系统依赖 (hostapd, dnsmasq, python3, onnxruntime, etc.)
#   2. 配置 WiFi AP 热点
#   3. 安装 systemd 服务
#   4. 生成 SSL 证书
#   5. 运行标定（可选）
#
# 用法:
#   # 基础安装（仅依赖 + 服务）
#   ssh root@<robot> 'bash -s' < scripts/install.sh
#
#   # 完整安装（含标定）
#   ssh root@<robot> 'bash -s -- --calibrate' < scripts/install.sh

set -e

APP_DIR="${VNAV_HOME:-/root/tennis-vnav}"
DO_CALIBRATE=false

for arg in "$@"; do
    case "$arg" in
        --calibrate) DO_CALIBRATE=true ;;
    esac
done

echo "╔══════════════════════════════════════════╗"
echo "║   tennis-vnav 一键安装                    ║"
echo "╚══════════════════════════════════════════╝"
echo ""
echo "目标目录: $APP_DIR"
echo ""

# ── 0. 停止旧服务 ──
echo "=== [0/6] 停止旧 AKA-00 服务 ==="
systemctl stop tennis-vnav 2>/dev/null || true
systemctl stop tennis-vnav-ap 2>/dev/null || true
systemctl stop aka-00 2>/dev/null || true
# 杀掉旧的 AKA-00 进程（可能由 init.sh 直接启动）
if [ -f /var/run/aka-init.pid ]; then
    kill $(cat /var/run/aka-init.pid) 2>/dev/null || true
    rm -f /var/run/aka-init.pid
fi
killall python3 2>/dev/null || true
sleep 1
echo "旧服务已停止"

# ── 1. 系统依赖 ──
echo "=== [1/6] 安装系统依赖 ==="

if command -v apt-get &>/dev/null; then
    apt-get update -qq
    apt-get install -y -qq \
        python3 python3-pip \
        hostapd dnsmasq \
        openssl \
        rsync \
        iw wireless-tools \
        usbutils \
        2>&1 | tail -2
elif command -v apk &>/dev/null; then
    # Alpine
    apk add --no-cache \
        python3 py3-pip \
        hostapd dnsmasq \
        openssl \
        rsync \
        iw wireless-tools
fi

# Python 依赖
echo "安装 Python 依赖..."
pip3 install -q \
    opencv-python-headless \
    onnxruntime \
    numpy \
    pyserial \
    2>&1 | tail -2 || echo "  (部分依赖可能已安装或安装失败，继续)"

echo "依赖安装完成"

# ── 2. WiFi AP ──
echo ""
echo "=== [2/6] 配置 WiFi AP ==="
if [ -f "$APP_DIR/scripts/wifi_ap.sh" ]; then
    bash "$APP_DIR/scripts/wifi_ap.sh" --no-start
    echo "WiFi AP 配置完成"
else
    echo "跳过（未找到 wifi_ap.sh）"
fi

# ── 3. SSL 证书 ──
echo ""
echo "=== [3/6] 生成 SSL 证书 ==="
if [ -f "$APP_DIR/scripts/init_cert.sh" ]; then
    bash "$APP_DIR/scripts/init_cert.sh" "$APP_DIR"
else
    echo "跳过（未找到 init_cert.sh）"
fi

# ── 4. udev 规则 ──
echo ""
echo "=== [4/6] 安装 udev 规则（摄像头持久化）==="
if [ -f "$APP_DIR/services/99-tennis-camera.rules" ]; then
    cp -f "$APP_DIR/services/99-tennis-camera.rules" /etc/udev/rules.d/
    udevadm control --reload-rules 2>/dev/null || true
    udevadm trigger 2>/dev/null || true
    echo "udev 规则已安装: /dev/tennis-camera -> USB UVC (1e45:8022)"
else
    echo "跳过"
fi

# ── 5. systemd 服务 ──
echo ""
echo "=== [5/6] 安装 systemd 服务 ==="
if [ -f "$APP_DIR/services/tennis-vnav.service" ]; then
    cp -f "$APP_DIR/services/tennis-vnav.service" /etc/systemd/system/
    systemctl daemon-reload
    systemctl enable tennis-vnav-ap 2>/dev/null || true
    systemctl enable tennis-vnav 2>/dev/null || true
    echo "systemd 服务已安装"
else
    echo "跳过（未找到 .service 文件）"
fi

# ── 5. 标定 ──
echo ""
echo "=== [6/6] 标定 ==="
if $DO_CALIBRATE; then
    echo "启动标定脚本..."
    cd "$APP_DIR"
    python3 calibrate.py
else
    echo "跳过标定（使用 --calibrate 启用）"
    echo "后续可手动运行: cd $APP_DIR && python3 calibrate.py"
fi

# ── 完成 ──
echo ""
echo "╔══════════════════════════════════════════╗"
echo "║   安装完成！                              ║"
echo "╚══════════════════════════════════════════╝"
echo ""
echo "启动导航:"
echo "  systemctl start tennis-vnav"
echo ""
echo "查看日志:"
echo "  journalctl -u tennis-vnav -f"
echo ""
echo "WiFi AP (如已配置):"
echo "  SSID: tennis-vnav-xxxxxxx"
echo "  IP:   192.168.4.1"
echo ""
echo "摄像头（已安装 udev 持久化设备名）:"
echo "  /dev/tennis-camera -> USB UVC (1e45:8022)"
echo ""
echo "首次使用建议运行标定:"
echo "  cd $APP_DIR && python3 calibrate.py"

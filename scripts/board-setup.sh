#!/bin/bash
# tennis-vnav 板端一次性初始化（纯净镜像 → vnav 可用）
#
# 内容:
#   1. 安装 AP 热点（S98apstart + hostapd/udhcpd 配置，无密码开放热点）
#   2. 禁用 AKA-00 开机自启（S99webstart 移出 init.d，备份至 /root/；
#      保留 AKA-00 的 WiFi AP 用法，其服务不再开机运行）
#   3. 验证
#
# 用法:
#   ./scripts/board-setup.sh [--host root@192.168.4.1]
#
# 注意: 本脚本为一次性初始化（幂等）；日常部署用 ./scripts/deploy.sh。
# 依赖: sshpass（密码 root，可按需修改 VNAV_PASS）

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
HOST="root@192.168.4.1"
VNAV_PASS="${VNAV_PASS:-root}"

for arg in "$@"; do
    case "$arg" in
        --host=*) HOST="${arg#*=}" ;;
        *) HOST="$arg" ;;
    esac
done

SSH="sshpass -p $VNAV_PASS ssh -o StrictHostKeyChecking=no"
SCP="sshpass -p $VNAV_PASS scp -o StrictHostKeyChecking=no"

echo "=== tennis-vnav 板端初始化 → $HOST ==="

# ── 1. AP 热点（vnav 手机控制面依赖，保持无密码）──
echo "[1/3] 安装 AP 热点（S98apstart + hostapd/udhcpd 配置）..."
$SCP "$SCRIPT_DIR/services/S98apstart" "$HOST:/etc/init.d/S98apstart"
$SSH "$HOST" "chmod +x /etc/init.d/S98apstart"
$SCP "$SCRIPT_DIR/services/hostapd.conf" "$HOST:/etc/hostapd.conf"
$SCP "$SCRIPT_DIR/services/udhcpd.conf" "$HOST:/etc/udhcpd.conf"

# ── 2. 禁用 AKA-00 自启（幂等；AKA-00 文件保留但不运行）──
echo "[2/3] 禁用 AKA-00 开机自启（S99webstart → /root/ 备份）..."
$SSH "$HOST" "[ ! -e /etc/init.d/S99webstart ] || mv /etc/init.d/S99webstart /root/S99webstart.disabled"
$SSH "$HOST" "grep -rl 'AKA-00' /etc/init.d/ 2>/dev/null || echo '  init.d 无 AKA-00 引用'"

# ── 3. 验证 ──
echo "[3/3] 验证..."
$SSH "$HOST" "ls /etc/init.d/ | grep -E 'S98apstart' || echo '  S98apstart 缺失';
    grep -q 'ssid' /etc/hostapd.conf && echo '  hostapd.conf OK';
    grep -q 'interface wlan0' /etc/udhcpd.conf && echo '  udhcpd.conf OK'"

echo "=== 初始化完成（重启板端后 AP → vnav 自动生效）==="

#!/bin/bash
# tennis-vnav WiFi AP 热点配置 (Linux / systemd)
#
# 基于 AKA-00 init_ap_web.sh 适配，使用 hostapd + dnsmasq（更通用的 Linux 方案）。
# 创建 wlan0 作为 AP 热点（192.168.4.1），wlan1 作为 STA 客户端（可选连接外网）。
#
# 用法:
#   sudo ./scripts/wifi_ap.sh              # 安装 + 立即启动
#   sudo ./scripts/wifi_ap.sh --no-start   # 仅安装服务，不立即启动
#   sudo ./scripts/wifi_ap.sh --uninstall  # 卸载

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
AP_IFACE="${AP_IFACE:-wlan0}"
STA_IFACE="${STA_IFACE:-wlan1}"
AP_IP="192.168.4.1"
AP_SUBNET="192.168.4.0/24"
DHCP_RANGE_START="192.168.4.100"
DHCP_RANGE_END="192.168.4.200"

NO_START=false
UNINSTALL=false
for arg in "$@"; do
    case "$arg" in
        --no-start) NO_START=true ;;
        --uninstall) UNINSTALL=true ;;
    esac
done

# ── 卸载 ──
if $UNINSTALL; then
    echo "=== 卸载 WiFi AP ==="
    systemctl stop tennis-vnav-ap 2>/dev/null || true
    systemctl disable tennis-vnav-ap 2>/dev/null || true
    rm -f /etc/systemd/system/tennis-vnav-ap.service
    rm -f /etc/hostapd/tennis-vnav.conf
    rm -f /etc/dnsmasq.d/tennis-vnav.conf
    systemctl daemon-reload
    echo "卸载完成"
    exit 0
fi

# ── 依赖检查 ──
if ! command -v hostapd &>/dev/null; then
    echo "[!] 需要安装 hostapd: apt install hostapd"
    exit 1
fi
if ! command -v dnsmasq &>/dev/null; then
    echo "[!] 需要安装 dnsmasq: apt install dnsmasq"
    exit 1
fi

# ── 生成基于 MAC 地址的唯一 SSID ──
MAC_SUFFIX=$(cat /sys/class/net/${AP_IFACE}/address 2>/dev/null | tr -d ':' | tail -c 6 || echo "000000")
RANDOM_ID=$((16#${MAC_SUFFIX} % 9999999 + 1))
SSID="tennis-vnav-${RANDOM_ID}"

echo "=== 配置 WiFi AP ==="
echo "  接口:   ${AP_IFACE}"
echo "  SSID:   ${SSID}"
echo "  IP:     ${AP_IP}"

# ── hostapd 配置 ──
mkdir -p /etc/hostapd
cat > /etc/hostapd/tennis-vnav.conf <<EOF
interface=${AP_IFACE}
driver=nl80211
ssid=${SSID}
hw_mode=g
channel=6
auth_algs=1
wmm_enabled=0
EOF

# ── dnsmasq 配置 ──
mkdir -p /etc/dnsmasq.d
cat > /etc/dnsmasq.d/tennis-vnav.conf <<EOF
interface=${AP_IFACE}
dhcp-range=${DHCP_RANGE_START},${DHCP_RANGE_END},255.255.255.0,24h
dhcp-option=option:router,${AP_IP}
dhcp-option=option:dns-server,${AP_IP}
bind-interfaces
EOF

# ── systemd 服务 ──
cat > /etc/systemd/system/tennis-vnav-ap.service <<EOF
[Unit]
Description=tennis-vnav WiFi AP Hotspot
After=network.target
Wants=network.target

[Service]
Type=forking
ExecStartPre=/bin/sh -c 'killall wpa_supplicant hostapd dnsmasq 2>/dev/null; sleep 1'
ExecStartPre=/bin/sh -c 'ip link set ${AP_IFACE} up'
ExecStartPre=/bin/sh -c 'ip addr add ${AP_IP}/24 dev ${AP_IFACE} 2>/dev/null || true'
ExecStart=/usr/sbin/hostapd -B /etc/hostapd/tennis-vnav.conf
ExecStartPost=/usr/sbin/dnsmasq -C /etc/dnsmasq.d/tennis-vnav.conf
ExecStop=/bin/sh -c 'killall hostapd dnsmasq 2>/dev/null || true'
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable tennis-vnav-ap

# ── 可选: 创建 STA 接口 ──
if iw dev | grep -q "${AP_IFACE}"; then
    if ! iw dev | grep -q "${STA_IFACE}"; then
        echo "创建 STA 接口 ${STA_IFACE}..."
        iw phy phy0 interface add ${STA_IFACE} type managed 2>/dev/null || true
        ip link set ${STA_IFACE} up 2>/dev/null || true
    fi
fi

if ! $NO_START; then
    echo "启动 AP 热点..."
    systemctl restart tennis-vnav-ap
    echo ""
    echo "=== WiFi AP 就绪 ==="
    echo "  SSID: ${SSID}"
    echo "  IP:   ${AP_IP}"
    echo "  连接后访问: http://${AP_IP}"
    echo "  SSH:  ssh root@${AP_IP}"
else
    echo ""
    echo "=== WiFi AP 已安装（未启动）==="
    echo "  启动: systemctl start tennis-vnav-ap"
fi

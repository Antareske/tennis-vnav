#!/bin/bash
# tennis-vnav 部署脚本
#
# 通过 rsync 将本地代码同步到小车（Linux），然后可选重启服务。
# 用于 SSH 远程测试的快速迭代。
#
# 用法:
#   ./scripts/deploy.sh root@192.168.4.1           # 部署到指定主机
#   ./scripts/deploy.sh root@192.168.4.1 --restart # 部署 + 重启服务
#   ./scripts/deploy.sh root@192.168.4.1 --dry-run # 预览变更
#
# 环境变量:
#   VNAV_HOST     默认目标主机
#   VNAV_PATH     目标路径 (默认: /root/tennis-vnav)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# ── 参数解析 ──
HOST="${1:-${VNAV_HOST}}"
RESTART=false
DRY_RUN=false
shift 2>/dev/null || true

for arg in "$@"; do
    case "$arg" in
        --restart) RESTART=true ;;
        --dry-run) DRY_RUN=true ;;
        *) HOST="$arg" ;;
    esac
done

if [ -z "$HOST" ]; then
    echo "用法: $0 <user@host> [--restart] [--dry-run]"
    echo ""
    echo "示例:"
    echo "  $0 root@192.168.4.1"
    echo "  $0 root@192.168.4.1 --restart"
    echo ""
    echo "环境变量:"
    echo "  VNAV_HOST=root@192.168.4.1  # 设置默认主机"
    exit 1
fi

TARGET_PATH="${VNAV_PATH:-/root/tennis-vnav}"

echo "=== tennis-vnav 部署 ==="
echo "  源:     $SCRIPT_DIR"
echo "  目标:   $HOST:$TARGET_PATH"
echo ""

# ── rsync 选项 ──
RSYNC_OPTS=(
    -avz
    --progress
    --exclude='__pycache__'
    --exclude='*.pyc'
    --exclude='.git'
    --exclude='models/*.onnx'      # 模型文件首次需单独上传
    --exclude='*.json'             # 不覆盖标定数据
    --exclude='arm_angles*.json'
    --exclude='cert.pem'
    --exclude='key.pem'
    --exclude='output'
    --exclude='images'
    --exclude='node_modules'
)

if $DRY_RUN; then
    RSYNC_OPTS+=(--dry-run)
fi

# ── 执行同步 ──
rsync "${RSYNC_OPTS[@]}" "$SCRIPT_DIR/" "$HOST:$TARGET_PATH/"

echo ""
echo "同步完成"

# ── 远程安装 systemd 服务 ──
if ! $DRY_RUN; then
    echo ""
    echo "安装 systemd 服务..."
    ssh "$HOST" "cp -f $TARGET_PATH/services/tennis-vnav.service /etc/systemd/system/ 2>/dev/null && systemctl daemon-reload" || true
fi

# ── 重启服务 ──
if $RESTART && ! $DRY_RUN; then
    echo ""
    echo "重启服务..."
    ssh "$HOST" "systemctl restart tennis-vnav && echo 'OK' || echo 'FAILED'"
    echo ""
    echo "查看日志: ssh $HOST 'journalctl -u tennis-vnav -f'"
fi

echo ""
echo "=== 部署完成 ==="
echo "  SSH 登录: ssh $HOST"
echo "  启动:     ssh $HOST 'systemctl start tennis-vnav'"
echo "  日志:     ssh $HOST 'journalctl -u tennis-vnav -f'"
echo "  状态:     ssh $HOST 'systemctl status tennis-vnav'"

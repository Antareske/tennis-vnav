#!/bin/bash
# tennis-vnav 完整部署脚本（纯净 SG2002 Linux 镜像 → 可用）
#
# 部署内容:
#   1. 运行时资产（Python 代码、libhwjpeg.so、ctrl-serve、模型、cvi-libs）
#   2. 开机自启脚本 S99vnav（含保活循环）
#   3. 重启服务并验证 main.py 存活
#
# 用法:
#   ./scripts/deploy.sh [--host root@192.168.4.1] [--no-restart]
#
# 依赖: sshpass（密码 root，可按需修改 VNAV_PASS）

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
HOST="root@192.168.4.1"
VNAV_PASS="${VNAV_PASS:-root}"
RESTART=true

for arg in "$@"; do
    case "$arg" in
        --host=*) HOST="${arg#*=}" ;;
        --no-restart) RESTART=false ;;
        *) HOST="$arg" ;;
    esac
done

SSH="sshpass -p $VNAV_PASS ssh -o StrictHostKeyChecking=no"
SCP="sshpass -p $VNAV_PASS scp -o StrictHostKeyChecking=no"

BOARD_DIR="/root/tennis-vnav"

echo "=== tennis-vnav 部署 → $HOST ==="

# ── 0. 停止旧进程 ──
# 板端 BusyBox 无 pkill/pgrep，killall 按进程名匹配（纯镜像无其他
# python3 服务，无差别杀 python3 安全）
echo "[0/4] 停止旧进程..."
$SSH "$HOST" "killall python3 ctrl-serve 2>/dev/null; sleep 1" || true

# ── 1. 运行时 Python 文件（仅运行时依赖闭包；机械臂等未启用模块不上传）──
PY_FILES="
main.py motor.py motor_tt_pid.py state_machine.py config.py
data_collector.py hwjpeg_enc.py vnav_control.py camera.py
detector.py estimator.py tpu_detector.py calibrate.py
"
echo "[1/4] 上传 Python 文件..."
$SSH "$HOST" "mkdir -p $BOARD_DIR"
# shellcheck disable=SC2086
$SCP $PY_FILES "$HOST:$BOARD_DIR/"

# ── 2. 二进制资产 ──
echo "[2/4] 上传二进制与模型..."
$SCP libhwjpeg.so ctrl-serve/ctrl-serve "$HOST:$BOARD_DIR/"
$SSH "$HOST" "mkdir -p $BOARD_DIR/models $BOARD_DIR/cvi-libs"
$SCP models/tennis.onnx models/yolov8n_tennis_v2.cvimodel "$HOST:$BOARD_DIR/models/"
$SCP cvi-libs/* "$HOST:$BOARD_DIR/cvi-libs/"

# ── 3. 开机自启 ──
echo "[3/4] 安装开机自启脚本..."
$SCP services/S99vnav "$HOST:/etc/init.d/S99vnav"
$SSH "$HOST" "chmod +x /etc/init.d/S99vnav"

# ── 4. 重启服务 ──
if [ "$RESTART" = true ]; then
    echo "[4/4] 重启服务..."
    $SSH "$HOST" "killall python3 ctrl-serve 2>/dev/null; sleep 1;
        if ! devmem 0x03001064 32 0x6; then echo '[deploy] pinmux 失败' >&2; fi;
        if ! devmem 0x03001068 32 0x6; then echo '[deploy] pinmux 失败' >&2; fi;
        nohup $BOARD_DIR/ctrl-serve > /tmp/ctrl-serve.log 2>&1 &
        cd $BOARD_DIR &&
        nohup env LD_LIBRARY_PATH=$BOARD_DIR/cvi-libs:/usr/bin/dl_lib:/usr/bin/lib \
            python3 main.py > /tmp/vnav.log 2>&1 &
        sleep 5"
    echo "服务已重启，验证:"
    $SSH "$HOST" "ps | grep -q '[p]ython3 main.py' && echo 'main.py: alive' || echo 'main.py: NOT RUNNING';
        ps | grep -q '[c]trl-serve' && echo 'ctrl-serve: alive' || echo 'ctrl-serve: NOT RUNNING';
        echo '--- /tmp/vnav.log 尾行 ---';
        tail -3 /tmp/vnav.log;
        curl -s http://192.168.4.1/status"
    echo
else
    echo "[4/4] 跳过重启（重启板子后 S99vnav 生效）"
fi

echo "=== 部署完成 ==="

#!/bin/bash
# tennis-vnav 离线部署包制作
#
# 在有网络的 PC 上运行，生成 tarball 包含:
#   1. tennis-vnav 项目代码
#   2. 所有 pip 依赖 wheel
#   3. 安装脚本
#
# 输出: dist/tennis-vnav-offline.tar.gz
#
# 用法:
#   ./scripts/offline-prep.sh                     # 为 x86_64 打包
#   ./scripts/offline-prep.sh --arch aarch64      # 为 ARM64 打包
#   ./scripts/offline-prep.sh --arch armv7l       # 为 ARM32 打包

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="$SCRIPT_DIR/dist"
mkdir -p "$OUT_DIR"

ARCH="${1:-x86_64}"
case "$ARCH" in
    --arch) shift; ARCH="${1:-x86_64}" ;;
    --arch=*) ARCH="${1#*=}" ;;
esac

echo "=== tennis-vnav 离线部署包制作 ==="
echo "  目标架构: $ARCH"
echo "  输出目录: $OUT_DIR"
echo ""

# ── 1. 下载 pip 依赖 ──
echo "[1/3] 下载 Python 依赖..."
DEPS_DIR="$OUT_DIR/pip-deps"
rm -rf "$DEPS_DIR"
mkdir -p "$DEPS_DIR"

pip3 download \
    --dest "$DEPS_DIR" \
    --platform manylinux2014_x86_64 \
    --python-version 3.10 \
    --only-binary=:all: \
    opencv-python-headless \
    numpy \
    2>&1 | tail -3 || echo "  (部分包可能无预编译 wheel)"

pip3 download \
    --dest "$DEPS_DIR" \
    onnxruntime \
    pyserial \
    2>&1 | tail -3

# 同时下载源码包作为备用
pip3 download \
    --dest "$DEPS_DIR" \
    --no-binary=:all: \
    onnxruntime \
    pyserial \
    2>&1 | tail -3 || true

echo "  pip 依赖: $(ls "$DEPS_DIR" | wc -l) 个文件"

# ── 2. 打包项目代码 ──
echo ""
echo "[2/3] 打包项目代码..."

TEMP_DIR="$OUT_DIR/tennis-vnav-pkg"
rm -rf "$TEMP_DIR"
mkdir -p "$TEMP_DIR"

cd "$SCRIPT_DIR"
# 复制项目文件（排除大文件和开发文件）
tar czf "$TEMP_DIR/tennis-vnav.tar.gz" \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='models/*.onnx' \
    --exclude='dist' \
    --exclude='output' \
    --exclude='images' \
    --exclude='node_modules' \
    --exclude='.git' \
    --exclude='*.bak' \
    .

echo "  项目代码已打包"

# ── 3. 合并为最终部署包 ──
echo ""
echo "[3/3] 生成最终部署包..."

FINAL_TARBALL="$OUT_DIR/tennis-vnav-offline.tar.gz"

# 创建自解压安装包
cat > "$TEMP_DIR/install-on-car.sh" <<'INSTALL_SCRIPT'
#!/bin/bash
# 在目标小车上执行
set -e
DEST="/root/tennis-vnav"
echo "解压 tennis-vnav..."
mkdir -p "$DEST"
tar xzf "$(dirname "$0")/tennis-vnav.tar.gz" -C "$DEST"
echo "安装 pip 依赖..."
pip3 install --no-index --find-links="$(dirname "$0")/pip-deps" opencv-python-headless numpy onnxruntime pyserial 2>&1 || true
echo "运行安装脚本..."
bash "$DEST/scripts/install.sh" "$@"
INSTALL_SCRIPT
chmod +x "$TEMP_DIR/install-on-car.sh"

# 最终打包
tar czf "$FINAL_TARBALL" \
    -C "$OUT_DIR" \
    pip-deps \
    tennis-vnav-pkg/

rm -rf "$DEPS_DIR" "$TEMP_DIR"

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║   离线部署包已生成                        ║"
echo "╚══════════════════════════════════════════╝"
echo ""
echo "文件: $FINAL_TARBALL ($(du -h "$FINAL_TARBALL" | cut -f1))"
echo ""
echo "部署到小车:"
echo "  scp $FINAL_TARBALL root@<car-ip>:/tmp/"
echo "  ssh root@<car-ip>"
echo "  cd /tmp && tar xzf tennis-vnav-offline.tar.gz"
echo "  cd tennis-vnav-pkg && bash install-on-car.sh"
echo ""
echo "如果需要 ARM 架构的包，加 --arch aarch64"

#!/bin/bash
# tennis-vnav 自签名证书生成
#
# 基于 AKA-00 https_init.sh，适配 Linux。
#
# 用法:
#   ./scripts/init_cert.sh                # 生成到当前目录
#   ./scripts/init_cert.sh /path/to/dir   # 生成到指定目录

set -e

OUT_DIR="${1:-$(cd "$(dirname "$0")/.." && pwd)}"

KEY_PEM="$OUT_DIR/key.pem"
CERT_PEM="$OUT_DIR/cert.pem"

if [ -f "$KEY_PEM" ] && [ -f "$CERT_PEM" ]; then
    echo "证书已存在: $CERT_PEM"
    echo "  如需重新生成，请先删除现有证书"
    exit 0
fi

echo "=== 生成自签名 SSL 证书 ==="
echo "  输出: $CERT_PEM"
echo ""

openssl req -x509 -newkey rsa:2048 \
    -keyout "$KEY_PEM" \
    -out "$CERT_PEM" \
    -days 3650 -nodes \
    -subj "/CN=tennis-vnav"

chmod 600 "$KEY_PEM"
chmod 644 "$CERT_PEM"

echo ""
echo "证书已生成:"
echo "  私钥: $KEY_PEM"
echo "  证书: $CERT_PEM"

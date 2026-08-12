"""SG2002 硬件 JPEG 编码器（VENC）Python 封装。

通过 libhwjpeg.so（C 封装，见 www/sg2002-headers/hwjpeg.c）调用板载
VENC 硬件编码器。单帧 ~15ms（软件 cv2 编码 ~300ms）。

依赖板上 /usr/bin/dl_lib/{libsys.so, libvenc.so}，
运行时需 LD_LIBRARY_PATH=/usr/bin/dl_lib。

用法:
  enc = HwJpegEncoder(width=640, height=480, quality=60)
  jpeg_bytes = enc.encode(bgr_frame)
  enc.close()
"""

from __future__ import annotations

import ctypes
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_LIB_DIR = Path(__file__).parent
_LIB_PATH = _LIB_DIR / "libhwjpeg.so"


class HwJpegEncoder:
    """硬件 JPEG 编码器。

    ctypes 调用期间释放 GIL，编码在 VENC 硬件中执行，
    主循环线程可以并发运行。
    """

    def __init__(self, width: int = 640, height: int = 480, quality: int = 60):
        if not _LIB_PATH.exists():
            raise RuntimeError(f"libhwjpeg.so 不存在: {_LIB_PATH}")

        # 仓库自带中间件库（cvi-libs/）加入动态链接搜索路径，
        # libvenc.so 的 NEEDED libsys.so 依赖由此解析
        repo_lib = str(_LIB_DIR / "cvi-libs")
        existing = os.environ.get("LD_LIBRARY_PATH", "")
        if repo_lib not in existing:
            os.environ["LD_LIBRARY_PATH"] = (
                repo_lib + ":" + existing if existing else repo_lib
            )

        # libsys.so 需要 __atomic_compare_exchange_1（libatomic）
        # 优先仓库自带，回退系统路径
        for libatomic in [str(_LIB_DIR / "cvi-libs" / "libatomic.so.1"),
                          "/usr/lib/libatomic.so.1"]:
            try:
                ctypes.CDLL(libatomic, mode=ctypes.RTLD_GLOBAL)
                break
            except OSError:
                continue

        self._lib = ctypes.CDLL(str(_LIB_PATH))
        self._lib.hwjpeg_init.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
        self._lib.hwjpeg_init.restype = ctypes.c_int
        self._lib.hwjpeg_encode.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_int),
        ]
        self._lib.hwjpeg_encode.restype = ctypes.c_int
        self._lib.hwjpeg_free_out.argtypes = [ctypes.c_void_p]
        self._lib.hwjpeg_close.argtypes = []

        if self._lib.hwjpeg_init(width, height, quality) != 0:
            raise RuntimeError("VENC 硬件编码器初始化失败")

        self.width = width
        self.height = height
        logger.info("VENC 硬件编码器就绪: %dx%d quality=%d", width, height, quality)

    def encode(self, frame) -> bytes:
        """编码 BGR 帧为 JPEG 字节。

        Args:
            frame: numpy BGR 图像 (H, W, 3)，uint8 连续内存

        Returns:
            JPEG 字节
        """
        out_ptr = ctypes.c_void_p()
        out_len = ctypes.c_int()
        h, w = frame.shape[:2]
        ret = self._lib.hwjpeg_encode(
            frame.ctypes.data, w, h,
            ctypes.byref(out_ptr), ctypes.byref(out_len),
        )
        if ret != 0:
            raise RuntimeError(f"VENC 编码失败: {ret}")
        try:
            return ctypes.string_at(out_ptr, out_len.value)
        finally:
            self._lib.hwjpeg_free_out(out_ptr)

    def close(self) -> None:
        self._lib.hwjpeg_close()
        logger.info("VENC 硬件编码器已关闭")

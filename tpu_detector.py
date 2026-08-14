"""SG2002 TPU YOLO 检测器 (ctypes 封装 libcviruntime.so C API)

完全对照 akars/src/tpu.rs (Rust 版本, 已验证在 SG2002 RISC-V 上正常运行)。

调用链:
  CVI_NN_RegisterModel → CVI_NN_GetInputOutputTensors
  → CVI_NN_TensorPtr(input) 拿 sys_mem → 直接写 RGB planar
  → CVI_NN_Forward
  → CVI_NN_TensorPtr(output) 拿 sys_mem → 按 fmt 反量化
  → parse_yolov8_output (channel-first 转置格式)

关键差异 vs 之前的版本:
  - 不需要 CVI_RT_Init (Rust 代码没用)
  - 输入直接写 sys_mem，不用 SetTensorWithAlignedFrames（那函数要物理地址）
  - CVI_TENSOR 定义为完整 ctypes.Structure（不是 opaque c_void_p）
  - 输出反量化（通常是 INT8 → qscale 反量化）
  - YOLO 输出格式是 [batch, channels, num_boxes] channel-first（不是 ONNX 的 box-first）

Reference:
  - akars/src/tpu.rs
  - akars/src/detector.rs (parse_yolov8_output, correct_yolo_boxes, nms)
  - akars-assets/include/cviruntime.h
"""

from __future__ import annotations

import ctypes
import logging
import os
import time
from typing import Optional

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False
    np = None  # type: ignore

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# C 结构体定义 (必须与 cviruntime.h 完全一致)
# ═══════════════════════════════════════════════════════════════════════════════

CVI_DIM_MAX = 6


class CVI_SHAPE(ctypes.Structure):
    """CVI_SHAPE: 张量形状描述。"""
    _fields_ = [
        ("dim", ctypes.c_int32 * CVI_DIM_MAX),  # 各维度大小
        ("dim_size", ctypes.c_size_t),           # 有效维度数
    ]


class CVI_TENSOR(ctypes.Structure):
    """CVI_TENSOR: TPU 张量，完整对应 cviruntime.h 定义。

    关键字段:
      sys_mem  — 系统内存指针 (CPU 可读写)
      paddr    — 设备物理地址 (DMA 用)
      fmt      — 数据类型 (FP32=0, INT8=6, UINT8=7 等)
      qscale   — 量化缩放因子 (INT8 → FP32: value * qscale)
      zero_point — 非对称量化零点 (UINT8 → FP32: (value - zero_point) * qscale)
      count    — 元素个数
    """
    _fields_ = [
        ("name", ctypes.c_void_p),                 # char *
        ("shape", CVI_SHAPE),
        ("fmt", ctypes.c_int32),                   # CVI_FMT enum
        ("count", ctypes.c_size_t),
        ("mem_size", ctypes.c_size_t),
        ("sys_mem", ctypes.c_void_p),              # uint8_t * (CPU 可访问内存)
        ("paddr", ctypes.c_uint64),                # 设备物理地址
        ("mem_type", ctypes.c_int32),              # CVI_MEM_TYPE_E
        ("qscale", ctypes.c_float),                # 量化缩放因子
        ("zero_point", ctypes.c_int),              # 量化零点
        ("pixel_format", ctypes.c_int32),          # CVI_NN_PIXEL_FORMAT_E
        ("aligned", ctypes.c_bool),
        ("mean", ctypes.c_float * 3),
        ("scale", ctypes.c_float * 3),
        ("owner", ctypes.c_void_p),
        ("reserved", ctypes.c_char * 32),
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# 常量
# ═══════════════════════════════════════════════════════════════════════════════

# CVI_FMT (数据类型)
CVI_FMT_FP32 = 0
CVI_FMT_INT32 = 1
CVI_FMT_UINT32 = 2
CVI_FMT_BF16 = 3
CVI_FMT_INT16 = 4
CVI_FMT_UINT16 = 5
CVI_FMT_INT8 = 6
CVI_FMT_UINT8 = 7

CVI_RC_SUCCESS = 0


# ═══════════════════════════════════════════════════════════════════════════════
# 动态库加载 (延迟加载，不阻塞 x86 环境 import)
# ═══════════════════════════════════════════════════════════════════════════════

_lib = None
_lib_loaded = False


def _load_library():
    """查找并加载 libcviruntime.so (及依赖 libcvikernel.so)。

    搜索顺序：仓库自带 cvi-libs/ → 系统路径（兼容旧部署）。
    在 RISC-V 小车上调用，x86 环境没有这些 .so。
    """
    repo_lib = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cvi-libs")
    lib_dirs = [repo_lib, "/usr/bin/lib"]

    # 更新 LD_LIBRARY_PATH
    for d in lib_dirs:
        if os.path.isdir(d):
            existing = os.environ.get("LD_LIBRARY_PATH", "")
            if d not in existing:
                os.environ["LD_LIBRARY_PATH"] = (
                    d + ":" + existing if existing else d
                )

    # Step 1: 预加载 libcvikernel.so (libcviruntime 的依赖)
    for d in lib_dirs:
        dep_path = os.path.join(d, "libcvikernel.so")
        if os.path.exists(dep_path):
            ctypes.CDLL(dep_path)
            logger.debug("Pre-loaded: %s", dep_path)
            break
    else:
        logger.warning("libcvikernel.so not found in %s", lib_dirs)

    # Step 2: 加载 libcviruntime.so
    candidates = [os.path.join(d, "libcviruntime.so") for d in lib_dirs]
    for path in candidates:
        if os.path.exists(path):
            logger.info("Loading: %s", path)
            return ctypes.CDLL(path)

    raise RuntimeError(f"libcviruntime.so not found in {candidates}")


def _get_lib():
    """获取 libcviruntime.so 句柄并设置所有函数签名。

    延迟加载：只在 initialize() 时首次调用。
    所有函数签名必须在使用前设置，否则会 segfault。
    """
    global _lib, _lib_loaded
    if _lib_loaded:
        return _lib

    _lib = _load_library()

    # ── 以下顺序无关，但必须全部在使用前设置 ──

    # CVI_RC CVI_NN_RegisterModel(const char *model_file, CVI_MODEL_HANDLE *model);
    _lib.CVI_NN_RegisterModel.restype = ctypes.c_int
    _lib.CVI_NN_RegisterModel.argtypes = [
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]

    # CVI_RC CVI_NN_GetInputOutputTensors(
    #     CVI_MODEL_HANDLE model,
    #     CVI_TENSOR **inputs, int32_t *input_num,
    #     CVI_TENSOR **outputs, int32_t *output_num);
    _lib.CVI_NN_GetInputOutputTensors.restype = ctypes.c_int
    _lib.CVI_NN_GetInputOutputTensors.argtypes = [
        ctypes.c_void_p,                      # CVI_MODEL_HANDLE
        ctypes.POINTER(ctypes.c_void_p),      # CVI_TENSOR **inputs
        ctypes.POINTER(ctypes.c_int32),       # int32_t *input_num
        ctypes.POINTER(ctypes.c_void_p),      # CVI_TENSOR **outputs
        ctypes.POINTER(ctypes.c_int32),       # int32_t *output_num
    ]

    # CVI_RC CVI_NN_Forward(
    #     CVI_MODEL_HANDLE model,
    #     CVI_TENSOR inputs[], int32_t input_num,
    #     CVI_TENSOR outputs[], int32_t output_num);
    _lib.CVI_NN_Forward.restype = ctypes.c_int
    _lib.CVI_NN_Forward.argtypes = [
        ctypes.c_void_p,     # CVI_MODEL_HANDLE
        ctypes.c_void_p,     # CVI_TENSOR *inputs
        ctypes.c_int32,
        ctypes.c_void_p,     # CVI_TENSOR *outputs
        ctypes.c_int32,
    ]

    # void *CVI_NN_TensorPtr(CVI_TENSOR *tensor);
    _lib.CVI_NN_TensorPtr.restype = ctypes.c_void_p
    _lib.CVI_NN_TensorPtr.argtypes = [ctypes.c_void_p]

    # size_t CVI_NN_TensorSize(CVI_TENSOR *tensor);
    _lib.CVI_NN_TensorSize.restype = ctypes.c_size_t
    _lib.CVI_NN_TensorSize.argtypes = [ctypes.c_void_p]

    # size_t CVI_NN_TensorCount(CVI_TENSOR *tensor);
    _lib.CVI_NN_TensorCount.restype = ctypes.c_size_t
    _lib.CVI_NN_TensorCount.argtypes = [ctypes.c_void_p]

    # CVI_TENSOR *CVI_NN_GetTensorByName(const char *name, CVI_TENSOR *tensors, int32_t num);
    _lib.CVI_NN_GetTensorByName.restype = ctypes.c_void_p
    _lib.CVI_NN_GetTensorByName.argtypes = [
        ctypes.c_char_p,
        ctypes.c_void_p,
        ctypes.c_int32,
    ]

    # CVI_SHAPE CVI_NN_TensorShape(CVI_TENSOR *tensor);
    _lib.CVI_NN_TensorShape.restype = CVI_SHAPE
    _lib.CVI_NN_TensorShape.argtypes = [ctypes.c_void_p]

    # CVI_RC CVI_NN_CleanupModel(CVI_MODEL_HANDLE model);
    _lib.CVI_NN_CleanupModel.restype = ctypes.c_int
    _lib.CVI_NN_CleanupModel.argtypes = [ctypes.c_void_p]

    _lib_loaded = True
    return _lib


# ═══════════════════════════════════════════════════════════════════════════════
# 辅助: 从 CVI_TENSOR 读取 float32 数据 (含反量化)
# ═══════════════════════════════════════════════════════════════════════════════

def _tensor_data_to_f32(tensor_addr: int, lib) -> Optional[np.ndarray]:
    """从 CVI_TENSOR 的 sys_mem 读取数据并转为 float32。

    对照 akars tpu.rs::tensor_to_f32()。
    支持格式: FP32, INT8, UINT8, BF16, INT16。
    """
    if not _HAS_NUMPY:
        raise RuntimeError("numpy is required for TPU tensor data reading")

    ptr = ctypes.cast(tensor_addr, ctypes.POINTER(CVI_TENSOR))
    tensor = ptr.contents  # 拷贝结构体 (168 bytes, 可忽略)

    fmt = tensor.fmt
    count = tensor.count
    sys_mem = tensor.sys_mem
    qscale = tensor.qscale
    zero_point = tensor.zero_point

    if not sys_mem or count == 0:
        logger.error("Tensor sys_mem is NULL or count=0 (fmt=%d, count=%d)", fmt, count)
        return None

    try:
        if fmt == CVI_FMT_FP32:
            data_ptr = ctypes.cast(sys_mem, ctypes.POINTER(ctypes.c_float))
            return np.ctypeslib.as_array(data_ptr, shape=(count,))  # 视图, 不拷贝

        elif fmt == CVI_FMT_INT8:
            data_ptr = ctypes.cast(sys_mem, ctypes.POINTER(ctypes.c_int8))
            raw = np.ctypeslib.as_array(data_ptr, shape=(count,))
            return raw.astype(np.float32) * qscale

        elif fmt == CVI_FMT_UINT8:
            data_ptr = ctypes.cast(sys_mem, ctypes.POINTER(ctypes.c_uint8))
            raw = np.ctypeslib.as_array(data_ptr, shape=(count,))
            return (raw.astype(np.int32) - zero_point).astype(np.float32) * qscale

        elif fmt == CVI_FMT_BF16:
            # bf16: 16-bit brain floating point → 左移 16 位转 f32
            data_ptr = ctypes.cast(sys_mem, ctypes.POINTER(ctypes.c_uint16))
            raw = np.ctypeslib.as_array(data_ptr, shape=(count,))
            raw_u32 = raw.astype(np.uint32) << 16
            return raw_u32.view(np.float32).copy()

        elif fmt == CVI_FMT_INT16:
            data_ptr = ctypes.cast(sys_mem, ctypes.POINTER(ctypes.c_int16))
            raw = np.ctypeslib.as_array(data_ptr, shape=(count,))
            return raw.astype(np.float32) * qscale

        else:
            logger.error("Unsupported tensor format: %d", fmt)
            return None

    except Exception as e:
        logger.exception("Failed to read tensor data: %s", e)
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# YOLOv8 输出解析 (对照 akars detector.rs)
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_yolov8_output(
    data: np.ndarray,
    shape: tuple[int, int, int, int],
    classes_num: int,
    confidence_threshold: float,
) -> list[dict]:
    """解析 YOLOv8 TPU 模型输出 (channel-first 转置格式，numpy 向量化)。

    对照 akars detector.rs::parse_yolov8_output()。

    输出形状: [batch, channels, num_boxes, 1]
      - channels = 4 + classes_num (cx, cy, w, h, cls0_score, ...)
      - 数据布局 (channel-first, num_boxes 为内层步长):
        data[0*N .. 1*N]: cx of all boxes
        data[1*N .. 2*N]: cy of all boxes
        data[2*N .. 3*N]: w of all boxes
        data[3*N .. 4*N]: h of all boxes
        data[4*N .. 5*N]: class 0 score of all boxes
        ...

    使用 numpy 向量化操作代替 Python 逐框循环 (8400 boxes → ~0 开销)。

    Returns:
        检测框列表 (坐标在模型输入空间，未做 letterbox 修正)
        [{"cx", "cy", "w", "h", "conf", "cls"}, ...]
    """
    batch = shape[0]
    channels = shape[1]
    num_boxes = shape[2]

    detections = []
    for b in range(batch):
        # 重塑为 (channels, num_boxes) 便于向量化索引
        batch_start = b * channels * num_boxes
        d = data[batch_start:batch_start + channels * num_boxes].reshape(channels, num_boxes)

        # cx, cy, w, h: rows 0-3
        cx = d[0]  # (num_boxes,)
        cy = d[1]
        w = d[2]
        h = d[3]

        # 类别分数: rows 4..4+classes_num, 取每列最大值
        if classes_num == 1:
            scores = d[4]
            best_cls = np.zeros(num_boxes, dtype=np.int32)
        else:
            class_scores = d[4:4 + classes_num]  # (classes_num, num_boxes)
            best_cls = class_scores.argmax(axis=0).astype(np.int32)
            scores = class_scores.max(axis=0)

        # 置信度过滤 (向量化)
        mask = scores > confidence_threshold
        indices = np.where(mask)[0]

        for idx in indices:
            detections.append({
                "cx": float(cx[idx]),
                "cy": float(cy[idx]),
                "w": float(w[idx]),
                "h": float(h[idx]),
                "conf": float(scores[idx]),
                "cls": int(best_cls[idx]),
            })

    return detections


def _correct_yolo_boxes(
    detections: list[dict],
    image_h: int,
    image_w: int,
    input_h: int,
    input_w: int,
) -> list[dict]:
    """将检测框从模型输入空间 (letterbox 坐标) 映射回原始图像空间。

    对照 akars detector.rs::correct_yolo_boxes()。

    模型输入做了 letterbox: scale = min(input_w/image_w, input_h/image_h)。
    需要去除 padding 并反归一化。
    """
    scale = min(input_w / image_w, input_h / image_h)
    new_h = int(image_h * scale)
    new_w = int(image_w * scale)
    pad_top = (input_h - new_h) // 2
    pad_left = (input_w - new_w) // 2

    result = []
    for d in detections:
        cx = d["cx"]
        cy = d["cy"]
        w = d["w"]
        h = d["h"]

        # cx,cy,w,h → x1,y1,x2,y2 (模型输入空间)
        x1 = cx - 0.5 * w
        y1 = cy - 0.5 * h
        x2 = cx + 0.5 * w
        y2 = cy + 0.5 * h

        # 去除 letterbox padding 并缩放回原图
        x1 = max(0.0, (x1 - pad_left) / scale)
        y1 = max(0.0, (y1 - pad_top) / scale)
        x2 = min(float(image_w), (x2 - pad_left) / scale)
        y2 = min(float(image_h), (y2 - pad_top) / scale)

        result.append({
            "cx": (x1 + x2) / 2.0,
            "cy": (y1 + y2) / 2.0,
            "w": x2 - x1,
            "h": y2 - y1,
            "conf": d["conf"],
            "cls": d["cls"],
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
        })

    return result


def _nms(detections: list[dict], iou_threshold: float) -> list[dict]:
    """Non-Maximum Suppression (使用 cv2.dnn.NMSBoxes, C++ 原生实现)。

    按 conf 降序排列，IoU > threshold 的低分框被抑制。
    """
    if len(detections) <= 1:
        return detections

    try:
        import cv2
        # cv2.dnn.NMSBoxes 期望 [x, y, w, h] 格式
        boxes_list = [[d["x1"], d["y1"], d["w"], d["h"]] for d in detections]
        confs = [d["conf"] for d in detections]

        # 需要 float32 避免 cv2 类型转换开销
        boxes_np = np.array(boxes_list, dtype=np.float32)
        confs_np = np.array(confs, dtype=np.float32)

        indices = cv2.dnn.NMSBoxes(
            boxes_np.tolist(), confs_np.tolist(),
            score_threshold=0.0,  # 已在 parse 阶段过滤
            nms_threshold=iou_threshold,
        )

        if indices is None or len(indices) == 0:
            return []

        result = []
        for idx in indices:
            i = int(idx) if np.isscalar(idx) else int(idx[0])
            if 0 <= i < len(detections):
                result.append(detections[i])
        return result

    except Exception:
        # 回退到纯 Python NMS (cv2 不可用时)
        return _nms_python(detections, iou_threshold)


def _nms_python(detections: list[dict], iou_threshold: float) -> list[dict]:
    """纯 Python NMS 实现 (cv2 不可用时的回退方案)。"""
    if not detections:
        return []

    detections.sort(key=lambda d: d["conf"], reverse=True)
    suppressed = [False] * len(detections)

    for i in range(len(detections)):
        if suppressed[i]:
            continue
        for j in range(i + 1, len(detections)):
            if suppressed[j]:
                continue
            if detections[i]["cls"] != detections[j]["cls"]:
                continue
            if _iou(detections[i], detections[j]) > iou_threshold:
                suppressed[j] = True

    return [d for i, d in enumerate(detections) if not suppressed[i]]


def _iou(a: dict, b: dict) -> float:
    """计算两个检测框的 IoU (Intersection over Union)。

    使用 x1,y1,x2,y2 坐标 (correct_yolo_boxes 后已设置)。
    """
    x1 = max(a["x1"], b["x1"])
    y1 = max(a["y1"], b["y1"])
    x2 = min(a["x2"], b["x2"])
    y2 = min(a["y2"], b["y2"])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area_a = a["w"] * a["h"]
    area_b = b["w"] * b["h"]
    denom = area_a + area_b - inter_area

    if denom <= 0.0:
        return 0.0
    return inter_area / denom


# ═══════════════════════════════════════════════════════════════════════════════
# 预处理: BGR frame → RGB planar (对照 akars image_bridge.rs)
# ═══════════════════════════════════════════════════════════════════════════════

def _preprocess_into_buffer(
    frame: np.ndarray,
    out_buf: np.ndarray,
    input_w: int,
    input_h: int,
) -> tuple[float, int, int]:
    """BGR frame → RGB planar, 直接写入 out_buf (零额外拷贝)。

    out_buf 是 shape (3, input_h, input_w) uint8 的 numpy 数组，
    通常映射到 TPU 输入 tensor 的 sys_mem。

    Args:
        frame: BGR 图像 (H, W, 3), uint8
        out_buf: 输出缓冲区 (3, input_h, input_w), uint8
        input_w: 模型输入宽度 (如 640)
        input_h: 模型输入高度 (如 640)

    Returns:
        (scale, pad_left, pad_top) — letterbox 参数，用于后续坐标修正
    """
    import cv2

    img_h, img_w = frame.shape[:2]

    # Letterbox: 保持宽高比缩放
    scale = min(input_w / img_w, input_h / img_h)
    new_h = int(round(img_h * scale))
    new_w = int(round(img_w * scale))

    pad_top = (input_h - new_h) // 2
    pad_left = (input_w - new_w) // 2

    # 只清空 padding 区域，避免全量 1.2MB fill(0)
    if pad_top > 0:
        out_buf[:, :pad_top, :] = 0
        out_buf[:, pad_top + new_h:, :] = 0
    if pad_left > 0:
        out_buf[:, pad_top:pad_top + new_h, :pad_left] = 0
        out_buf[:, pad_top:pad_top + new_h, pad_left + new_w:] = 0

    # BGR → RGB + resize
    if new_w != img_w or new_h != img_h:
        resized = cv2.resize(frame, (new_w, new_h))
    else:
        resized = frame
    rgb = resized[..., ::-1]  # BGR → RGB (view, no copy)

    # 直接写 RGB planar 到 TPU 内存
    roi = out_buf[:, pad_top:pad_top + new_h, pad_left:pad_left + new_w]
    roi[0] = rgb[..., 0]  # R plane
    roi[1] = rgb[..., 1]  # G plane
    roi[2] = rgb[..., 2]  # B plane

    return scale, pad_left, pad_top


# ═══════════════════════════════════════════════════════════════════════════════
# TpuDetector 主类
# ═══════════════════════════════════════════════════════════════════════════════

class TpuDetector:
    """SG2002 TPU YOLO 检测器。

    对照 akars tpu.rs::YoloModel。

    用法:
        det = TpuDetector(model_path="models/yolov8n_tennis_v2.cvimodel")
        det.initialize()
        boxes = det.detect(frame, img_width=640, img_height=480)
        # boxes: [{"x","y","w","h","cx","cy","conf"}, ...]
        det.close()
    """

    def __init__(
        self,
        model_path: str = "models/yolov8n_tennis_v2.cvimodel",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        classes_num: int = 1,
    ):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.classes_num = classes_num

        # TPU 资源 (延迟初始化)
        self._model_handle = None       # CVI_MODEL_HANDLE (c_void_p)
        self._inputs_ptr = None         # CVI_TENSOR * (指向第一个输入张量)
        self._outputs_ptr = None        # CVI_TENSOR * (指向第一个输出张量)
        self._n_inputs = 0
        self._n_outputs = 0
        self._input_buf = None          # numpy view of TPU input sys_mem (zero-copy)

        # 输入/输出张量的关键信息 (初始化时缓存)
        self._input_w = 0
        self._input_h = 0
        self._output_shape = None       # (batch, channels, num_boxes, 1)

        self._initialized = False

    # ── 初始化 ──

    def initialize(self) -> bool:
        """初始化 TPU 并加载模型。

        对照 akars tpu.rs::YoloModel::open()。
        注意: 不需要 CVI_RT_Init，Rust 代码没有这步。
        """
        if self._initialized:
            return True

        lib = _get_lib()

        # Step 1: CVI_NN_RegisterModel
        model_path_bytes = self.model_path.encode("utf-8")
        model_handle = ctypes.c_void_p()
        ret = lib.CVI_NN_RegisterModel(model_path_bytes, ctypes.byref(model_handle))
        if ret != CVI_RC_SUCCESS:
            logger.error("CVI_NN_RegisterModel failed: %d (path=%s)",
                        ret, self.model_path)
            return False
        self._model_handle = model_handle
        logger.info("Model registered: %s", self.model_path)

        # Step 2: CVI_NN_GetInputOutputTensors
        inputs = ctypes.c_void_p()
        outputs = ctypes.c_void_p()
        n_in = ctypes.c_int32()
        n_out = ctypes.c_int32()

        ret = lib.CVI_NN_GetInputOutputTensors(
            model_handle,
            ctypes.byref(inputs), ctypes.byref(n_in),
            ctypes.byref(outputs), ctypes.byref(n_out),
        )
        if ret != CVI_RC_SUCCESS:
            logger.error("CVI_NN_GetInputOutputTensors failed: %d", ret)
            self._cleanup_model()
            return False

        self._inputs_ptr = inputs.value   # CVI_TENSOR * (整数地址)
        self._outputs_ptr = outputs.value
        self._n_inputs = n_in.value
        self._n_outputs = n_out.value

        logger.info("Model has %d input(s), %d output(s)", self._n_inputs, self._n_outputs)

        # Step 3: 获取默认输入张量 (对照 Rust: CVI_NN_GetTensorByName(NULL, inputs, input_num))
        input_tensor_addr = lib.CVI_NN_GetTensorByName(None, inputs, n_in)
        if not input_tensor_addr:
            logger.error("Default input tensor not found")
            self._cleanup_model()
            return False

        input_shape = lib.CVI_NN_TensorShape(input_tensor_addr)
        # shape.dim = [batch, channels, height, width] for 4D input
        # 对 YOLO 通常是 [1, 3, 640, 640]
        self._input_h = input_shape.dim[2]
        self._input_w = input_shape.dim[3]

        # Step 4: 获取输入 tensor 的 sys_mem (用于直接写入预处理数据)
        input_sys_mem = lib.CVI_NN_TensorPtr(input_tensor_addr)
        if not input_sys_mem:
            logger.error("Input tensor sys_mem is NULL")
            self._cleanup_model()
            return False
        self._input_sys_mem = input_sys_mem
        self._input_tensor_addr = input_tensor_addr

        # 创建 numpy view 直接映射 TPU 输入内存 (零拷贝写入)
        self._input_buf = np.ctypeslib.as_array(
            ctypes.cast(input_sys_mem, ctypes.POINTER(ctypes.c_uint8)),
            shape=(3, self._input_h, self._input_w),
        )

        logger.info("Input tensor: %dx%d (sys_mem=%s, buf=%s)",
                    self._input_w, self._input_h,
                    hex(input_sys_mem) if input_sys_mem else "NULL",
                    self._input_buf.shape)

        # Step 5: 读取输出张量形状
        if self._n_outputs > 0 and self._outputs_ptr:
            out_shape = lib.CVI_NN_TensorShape(self._outputs_ptr)
            self._output_shape = (
                out_shape.dim[0],
                out_shape.dim[1],
                out_shape.dim[2],
                out_shape.dim[3],
            )
            logger.info("Output shape: %s", self._output_shape)

            # 也检查输出 tensor 信息
            out_ptr = ctypes.cast(self._outputs_ptr, ctypes.POINTER(CVI_TENSOR))
            out_tensor = out_ptr.contents
            logger.info("Output tensor: fmt=%d count=%d mem_size=%d qscale=%.6f zero_point=%d",
                        out_tensor.fmt, out_tensor.count, out_tensor.mem_size,
                        out_tensor.qscale, out_tensor.zero_point)
        else:
            logger.warning("No output tensors found!")

        self._initialized = True
        return True

    # ── 推理 ──

    def detect(
        self,
        frame: np.ndarray,
        img_width: int = 640,
        img_height: int = 480,
    ) -> list[dict]:
        """运行 YOLO 检测。

        对照 akars tpu.rs::YoloModel::infer()。

        Args:
            frame: BGR 图像 (H, W, 3), uint8 numpy array
            img_width: 原始图像宽度 (用于 letterbox 坐标修正)
            img_height: 原始图像高度

        Returns:
            检测框列表:
            [{"x": int, "y": int, "w": int, "h": int, "cx": int, "cy": int, "conf": float}, ...]
        """
        if not self._initialized and not self.initialize():
            return []

        lib = _get_lib()
        input_w = self._input_w
        input_h = self._input_h

        # ── 预处理: BGR → RGB planar → 直接写入 TPU sys_mem (零额外拷贝) ──
        preprocess_start = time.time()
        scale, pad_left, pad_top = _preprocess_into_buffer(
            frame, self._input_buf, input_w, input_h,
        )
        preprocess_ms = (time.time() - preprocess_start) * 1000

        # ── Forward ──
        forward_start = time.time()
        ret = lib.CVI_NN_Forward(
            self._model_handle,
            ctypes.c_void_p(self._inputs_ptr),
            ctypes.c_int32(self._n_inputs),
            ctypes.c_void_p(self._outputs_ptr),
            ctypes.c_int32(self._n_outputs),
        )
        forward_ms = (time.time() - forward_start) * 1000

        if ret != CVI_RC_SUCCESS:
            # 推理失败与"无目标"必须区分：返回空列表会被状态机当丢球
            # 处理（无限旋转无告警），抛异常则走主循环的 episode 错误
            # 路径（刹车 + 保存数据 + status 报错）
            raise RuntimeError(f"CVI_NN_Forward failed: {ret}")

        # ── 读取输出 ──
        postprocess_start = time.time()

        # 读取输出数据 (含反量化)
        data = _tensor_data_to_f32(self._outputs_ptr, lib)
        if data is None:
            raise RuntimeError("TPU 输出张量读取失败")

        read_ms = (time.time() - postprocess_start) * 1000

        # 解析 YOLOv8 输出
        raw_detections = _parse_yolov8_output(
            data, self._output_shape,
            self.classes_num, self.conf_threshold,
        )

        # Letterbox 坐标修正
        detections = _correct_yolo_boxes(
            raw_detections, img_height, img_width, input_h, input_w,
        )

        # NMS
        detections = _nms(detections, self.iou_threshold)

        postprocess_ms = (time.time() - postprocess_start) * 1000

        # 转换为统一 bbox 格式 (与 detector.py yolo_infer 一致)
        boxes = []
        for d in detections:
            x1 = int(d["x1"])
            y1 = int(d["y1"])
            x2 = int(d["x2"])
            y2 = int(d["y2"])
            boxes.append({
                "x": x1,
                "y": y1,
                "w": x2 - x1,
                "h": y2 - y1,
                "cx": int((x1 + x2) / 2),
                "cy": int((y1 + y2) / 2),
                "conf": d["conf"],
            })

        total_ms = preprocess_ms + forward_ms + read_ms + postprocess_ms
        logger.debug(
            "TPU infer: pre=%.1fms fwd=%.1fms read=%.1fms post=%.1fms total=%.1fms → %d detections",
            preprocess_ms, forward_ms, read_ms, postprocess_ms, total_ms, len(boxes),
        )

        return boxes

    # ── 清理 ──

    def _cleanup_model(self):
        """清理模型资源。"""
        global _lib_loaded
        if self._model_handle is None:
            return
        if not _lib_loaded:
            return  # 库未加载 (非 RISC-V 环境)
        lib = _get_lib()
        lib.CVI_NN_CleanupModel(self._model_handle)
        self._model_handle = None

    def close(self):
        """释放 TPU 资源。"""
        self._cleanup_model()
        self._initialized = False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass  # 析构时不抛异常


# ═══════════════════════════════════════════════════════════════════════════════
# 全局单例
# ═══════════════════════════════════════════════════════════════════════════════

_tpu_detector: Optional[TpuDetector] = None


def get_detector(
    model_path: str = "models/yolov8n_tennis_v2.cvimodel",
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
) -> TpuDetector:
    """获取全局 TpuDetector 单例 (延迟初始化)。"""
    global _tpu_detector
    if _tpu_detector is None:
        _tpu_detector = TpuDetector(
            model_path=model_path,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
        )
        _tpu_detector.initialize()
    return _tpu_detector

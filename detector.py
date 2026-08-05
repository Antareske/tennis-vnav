"""YOLO 网球检测模块。

从 AKA-00 tennis_hunter.py 提取 yolo_infer() 并增强：
- is_bbox_valid() 检查检测框是否完整在画面内
- select_best_bbox() 统一接口，自动选择最优检测框
- 自动检测平台：SG2002 (RISC-V) 使用 TPU ctypes 后端，其他使用 ONNX runtime
"""

from __future__ import annotations

import logging
import os
import platform
from typing import Optional

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

logger = logging.getLogger(__name__)

# ── 平台检测 ──
_IS_RISCV = platform.machine() in ("riscv64", "riscv32")
_TPU_AVAILABLE = False
_tpu_detector = None  # 延迟初始化

if _IS_RISCV:
    try:
        from tpu_detector import TpuDetector, get_detector as _get_tpu
        _TPU_AVAILABLE = True
        logger.info("平台: RISC-V, 使用 TPU ctypes 后端")
    except Exception as e:
        logger.warning("TPU 后端加载失败: %s", e)
else:
    logger.info("平台: %s, 尝试 ONNX runtime", platform.machine())

# cvimodel 默认路径 (TPU 后端使用)
_DEFAULT_CVIMODEL = "models/yolov8n_tennis_v2.cvimodel"


def _resolve_cvimodel_path(model_path: str) -> str:
    """将 ONNX 模型路径转换为 cvimodel 路径 (TPU 后端)。

    尝试顺序:
      1. model_path 本身 (如果以 .cvimodel 结尾)
      2. 将 .onnx 替换为 .cvimodel
      3. 默认 cvimodel 路径
    """
    if model_path.endswith(".cvimodel"):
        return model_path
    # 尝试替换扩展名
    if model_path.endswith(".onnx"):
        alt = model_path[:-5] + ".cvimodel"
        if os.path.exists(alt):
            return alt
    # 尝试默认路径
    if os.path.exists(_DEFAULT_CVIMODEL):
        return _DEFAULT_CVIMODEL
    # 回退到模型自身的路径 (可能不存在，由 TpuDetector 报错)
    return model_path


def _get_tpu_backend(model_path: str):
    """获取或创建 TPU 检测器单例。"""
    global _tpu_detector
    cvimodel = _resolve_cvimodel_path(model_path)
    if _tpu_detector is not None:
        # 如果模型路径变更，重建
        if _tpu_detector.model_path != cvimodel:
            _tpu_detector.close()
            _tpu_detector = None
    if _tpu_detector is None:
        _tpu_detector = TpuDetector(model_path=cvimodel)
        if not _tpu_detector.initialize():
            logger.error("TPU 初始化失败")
            _tpu_detector = None
    return _tpu_detector

# ── ONNX 模型加载 ──
_ort_session = None
_input_name = None


def _get_session(model_path: str):
    """延迟加载 ONNX session（单例）。

    在 RISC-V TPU 后端不可用时返回 (None, None) — 此时 yolo_infer() 会走 TPU 路径。
    """
    global _ort_session, _input_name
    if _ort_session is not None:
        return _ort_session, _input_name
    if _TPU_AVAILABLE:
        return None, None
    import onnxruntime as ort
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    _ort_session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    _input_name = _ort_session.get_inputs()[0].name
    logger.info("YOLO 模型已加载: %s", model_path)
    return _ort_session, _input_name


# ── 预处理 ──

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    """YOLOv8 官方预处理：保持宽高比 resize + center pad。"""
    shape = img.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img


# ── 推理 ──

def yolo_infer(
    frame: np.ndarray,
    model_path: str = "models/tennis.onnx",
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    img_size: int = 640,
) -> list[dict]:
    """YOLO 推理，返回检测到的网球 bbox 列表。

    自动选择后端: RISC-V → TPU ctypes, 其他 → ONNX runtime。

    Args:
        frame: BGR 图像 (H, W, 3)
        model_path: 模型路径 (.onnx 或 .cvimodel)
        conf_threshold: 置信度阈值
        iou_threshold: NMS IoU 阈值
        img_size: 模型输入尺寸

    Returns:
        [{"x": int, "y": int, "w": int, "h": int, "conf": float, "cx": int, "cy": int}, ...]
    """
    # ── TPU 后端 (RISC-V / SG2002) ──
    if _TPU_AVAILABLE:
        tpu = _get_tpu_backend(model_path)
        if tpu is None:
            return []
        H, W = frame.shape[:2]
        return tpu.detect(frame, img_width=W, img_height=H)

    # ── ONNX 后端 (x86 开发/测试) ──
    session, input_name = _get_session(model_path)
    H, W = frame.shape[:2]
    input_img = letterbox(frame, new_shape=(img_size, img_size))

    blob = cv2.dnn.blobFromImage(
        input_img, scalefactor=1 / 255.0, size=(img_size, img_size),
        swapRB=True, crop=False,
    )
    outputs = session.run(None, {input_name: blob})
    pred = outputs[0].squeeze().T  # [C, N] -> [N, C]

    boxes_xywh = pred[:, :4]
    conf_scores = pred[:, 4]
    mask = conf_scores > conf_threshold

    pred = pred[mask]
    boxes_xywh = boxes_xywh[mask]
    conf_scores = conf_scores[mask]

    raw_boxes = []
    raw_confs = []
    for i in range(len(boxes_xywh)):
        cx, cy, w, h = boxes_xywh[i]
        shape = frame.shape[:2]
        r = min(img_size / shape[0], img_size / shape[1])
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = img_size - new_unpad[0], img_size - new_unpad[1]
        dw /= 2
        dh /= 2

        x1 = (cx - w / 2 - dw) / r
        y1 = (cy - h / 2 - dh) / r
        x2 = (cx + w / 2 - dw) / r
        y2 = (cy + h / 2 - dh) / r

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(W, x2)
        y2 = min(H, y2)

        raw_boxes.append([x1, y1, x2, y2])
        raw_confs.append(float(conf_scores[i]))

    if not raw_boxes:
        return []

    # NMS
    raw_boxes_np = np.array(raw_boxes, dtype=np.float32)
    indices = cv2.dnn.NMSBoxes(raw_boxes_np.tolist(), raw_confs, conf_threshold, iou_threshold)

    boxes = []
    if indices is not None and len(indices) > 0:
        for idx in indices:
            i = int(idx) if np.isscalar(idx) else int(idx[0])
            x1, y1, x2, y2 = raw_boxes_np[i]
            box = {
                "x": int(x1),
                "y": int(y1),
                "w": int(x2 - x1),
                "h": int(y2 - y1),
                "conf": raw_confs[i],
                "cx": int((x1 + x2) / 2),
                "cy": int((y1 + y2) / 2),
            }
            boxes.append(box)

    return boxes


# ── 检测质量检查 ──

def is_bbox_valid(
    bbox: dict,
    img_width: int,
    img_height: int,
    edge_margin: int = 20,
) -> bool:
    """检查 bbox 是否完整在画面内（四边均不触碰边缘）。

    Args:
        bbox: {"x", "y", "w", "h"} 检测框
        img_width: 图像宽度
        img_height: 图像高度
        edge_margin: 边缘安全距离 (px)

    Returns:
        True 如果 bbox 完全在画面内
    """
    x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
    if w <= 0 or h <= 0:
        return False
    if x < edge_margin:
        return False
    if y < edge_margin:
        return False
    if x + w > img_width - edge_margin:
        return False
    if y + h > img_height - edge_margin:
        return False
    return True


def select_best_bbox(
    bboxes: list[dict],
    img_width: int,
    img_height: int,
    edge_margin: int = 20,
) -> Optional[dict]:
    """从检测结果中选择最佳 bbox（优先完整在画面内的最大框）。

    Returns:
        最佳 bbox dict，或 None
    """
    if not bboxes:
        return None

    valid = [b for b in bboxes if is_bbox_valid(b, img_width, img_height, edge_margin)]

    if valid:
        # 有效框中取最大的（面积或宽度）
        return max(valid, key=lambda b: b["w"] * b["h"])

    # 没有完整框时，取置信度最高的（至少部分可见）
    return max(bboxes, key=lambda b: b["conf"])

"""Scratch segmentation with ONNX Runtime + NumPy only (no PyTorch / Ultralytics),
so the serving container stays small. Mirrors Ultralytics' YOLOv8-seg pre/post-processing:
letterbox -> model -> confidence filter -> NMS -> mask = sigmoid(coefs @ prototypes),
cropped to the box, thresholded at 0.5, mapped back to the original image.
Checked against the PyTorch model by tools/check_onnx_parity.py.
"""
import time

import cv2
import numpy as np
import onnxruntime as ort

IMG_SIZE = 512       # model input (training size)
NMS_IOU = 0.7        # Ultralytics default
MASK_THRESHOLD = 0.5
PAD_VALUE = 114      # Ultralytics letterbox gray


class ScratchSegmenter:
    def __init__(self, model_path, threshold):
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.threshold = threshold

    def letterbox(self, img):
        """Resize keeping the aspect ratio, pad to IMG_SIZE x IMG_SIZE (centered)."""
        h, w = img.shape[:2]
        scale = IMG_SIZE / max(h, w)
        nh, nw = round(h * scale), round(w * scale)
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        top, left = (IMG_SIZE - nh) // 2, (IMG_SIZE - nw) // 2
        canvas = np.full((IMG_SIZE, IMG_SIZE, 3), PAD_VALUE, np.uint8)
        canvas[top:top + nh, left:left + nw] = resized
        return canvas, scale, top, left

    def predict(self, img_bgr, conf=None):
        """Returns dict: union mask (original size, bool), per-region list, best score."""
        conf = self.threshold if conf is None else conf
        t0 = time.perf_counter()
        h0, w0 = img_bgr.shape[:2]
        canvas, scale, top, left = self.letterbox(img_bgr)
        x = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0  # BGR->RGB, CHW
        out0, protos = self.session.run(None, {self.input_name: x})

        pred = out0[0].T                       # (5376, 37): cx cy w h, score, 32 coefs
        scores = pred[:, 4]
        keep = scores >= conf
        pred, scores = pred[keep], scores[keep]
        boxes = np.column_stack([pred[:, 0] - pred[:, 2] / 2, pred[:, 1] - pred[:, 3] / 2,
                                 pred[:, 0] + pred[:, 2] / 2, pred[:, 1] + pred[:, 3] / 2])
        idx = nms(boxes, scores, NMS_IOU)
        boxes, scores, coefs = boxes[idx], scores[idx], pred[idx, 5:]

        union = np.zeros((h0, w0), bool)
        regions = []
        if len(idx):
            p = protos[0]                                        # (32, 128, 128)
            m = sigmoid(coefs @ p.reshape(32, -1)).reshape(-1, *p.shape[1:])
            for mask_small, box, score in zip(m, boxes, scores):
                mask = cv2.resize(mask_small, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
                x1, y1, x2, y2 = np.clip(box, 0, IMG_SIZE).astype(int)
                cropped = np.zeros_like(mask)
                cropped[y1:y2, x1:x2] = mask[y1:y2, x1:x2]          # keep only inside the box
                mask = cropped[top:IMG_SIZE - top, left:IMG_SIZE - left]   # remove padding
                mask = cv2.resize(mask, (w0, h0), interpolation=cv2.INTER_LINEAR) > MASK_THRESHOLD
                if not mask.any():
                    continue
                union |= mask
                ox = (np.array([x1, x2]) - left) / scale             # box in original pixels
                oy = (np.array([y1, y2]) - top) / scale
                regions.append({"score": round(float(score), 3),
                                "box": [int(ox[0]), int(oy[0]), int(ox[1]), int(oy[1])],
                                "area_px": int(mask.sum())})
        return {"mask": union, "regions": regions,
                "best_score": round(float(scores.max()), 3) if len(scores) else 0.0,
                "inference_ms": round((time.perf_counter() - t0) * 1000, 1)}


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def nms(boxes, scores, iou_thr):
    """Greedy non-maximum suppression: keep the best box, drop boxes overlapping it."""
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        area = lambda b: (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])  # noqa: E731
        iou = inter / (area(boxes[[i]]) + area(boxes[order[1:]]) - inter + 1e-9)
        order = order[1:][iou < iou_thr]
    return np.array(keep, int)

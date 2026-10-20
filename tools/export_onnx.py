"""Export the final PyTorch model to ONNX for the serving container.

    python tools/export_onnx.py
Writes output/models/final_v2_ft_replay.onnx and copies it to serve/model.onnx.
Fixed input 1x3x512x512 (the size the model was trained on).
"""
import functools
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO

# torch >= 2.9 defaults to the new dynamo-based ONNX exporter, which pulls in extra
# packages (onnxscript, onnx_ir). Ultralytics 8.3.40 cannot choose the exporter, so we
# route its call to the classic TorchScript exporter, which is stable for YOLO.
torch.onnx.export = functools.partial(torch.onnx.export, dynamo=False)

SRC = Path("output/models/final_v2_ft_replay.pt")
onnx_path = Path(YOLO(str(SRC)).export(format="onnx", imgsz=512, opset=17, simplify=True,
                                       dynamic=False))
shutil.copy(onnx_path, "serve/model.onnx")
print(f"exported {onnx_path} -> serve/model.onnx")

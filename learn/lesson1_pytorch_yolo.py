"""Lesson 1: what is inside YOLO? (PyTorch tensors, conv filters, a prediction)

Run from the repo root (venv active):
    python learn/lesson1_pytorch_yolo.py
Read each printed section and the comment above it.
"""
import torch
from ultralytics import YOLO

print("\n=== 1) A tensor: PyTorch's basic object (like a NumPy array) ===")
# A color image is a tensor of shape (channels, height, width): 3 x 512 x 512 numbers.
image = torch.rand(3, 512, 512)
print("image tensor shape:", tuple(image.shape), "| numbers in it:", image.numel())

print("\n=== 2) Autograd: PyTorch computes derivatives for us (this is backprop) ===")
# y = w * x. If we want y to be 10 and it is 6, how should w change?
w = torch.tensor(2.0, requires_grad=True)   # a "weight" we want to learn
x = torch.tensor(3.0)
y = w * x                                    # prediction: 6
loss = (y - 10) ** 2                         # how wrong: (6 - 10)^2 = 16
loss.backward()                              # compute d(loss)/d(w) automatically
print(f"prediction {y.item()}, loss {loss.item()}, gradient d(loss)/dw = {w.grad.item()}")
# gradient = 2*(y-10)*x = 2*(-4)*3 = -24 -> negative: increasing w lowers the loss
w_new = w.item() - 0.01 * w.grad.item()      # one training step, learning rate 0.01
print(f"one step of gradient descent: w {w.item()} -> {w_new:.2f}, "
      f"new prediction {w_new * x.item():.2f} (closer to 10)")
# YOLO training does exactly this, for 3.2 million weights at once, thousands of times.

print("\n=== 3) The YOLO model is a PyTorch network ===")
model = YOLO("yolov8n-seg.pt")               # COCO-pretrained weights (what we start from)
net = model.model                            # the actual torch.nn.Module
n_params = sum(p.numel() for p in net.parameters())
print(f"type: {type(net).__name__}, layers: {len(list(net.modules()))}, weights: {n_params:,}")

print("\n=== 4) The first layer: 16 small filters that slide over the image (a CNN) ===")
first_conv = net.model[0].conv
print("first conv layer:", first_conv)
print("its weight tensor shape:", tuple(first_conv.weight.shape),
      "= 16 filters x 3 colors x 3 x 3 pixels")
print("filter 0, red channel (9 learned numbers):\n", first_conv.weight[0, 0].detach().numpy().round(2))
# Each filter is a 3x3 pattern detector (an edge, a color change...). Deeper layers
# combine them into textures, then shapes, then objects -> "convolutional neural network".

print("\n=== 5) A prediction on a real seen_pool image ===")
result = model.predict("data/real/mvtec/metal_nut/test/scratch/018.png", imgsz=512, verbose=False)[0]
print("boxes found:", len(result.boxes))
for cls, conf in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist()):
    print(f"  class '{model.names[int(cls)]}' with confidence {conf:.2f}")
# This is the COCO model (80 everyday classes, no 'scratch'): it either finds nothing or
# something silly. That is why we fine-tune it on scratches (transfer learning).

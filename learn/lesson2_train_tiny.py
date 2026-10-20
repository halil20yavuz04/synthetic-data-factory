"""Lesson 2: what "training" means, on a tiny version of our problem.

Task: 32x32 gray patches of a textured surface. Label 1 = a scratch (a thin darker line
at a random position and angle), label 0 = clean surface.
  Part A: kNN on raw pixels (what you saw in class)
  Part B: a tiny CNN trained from scratch with the SAME loop YOLO uses in Colab
Run from the repo root (venv active, ~1 minute on the Mac):
    python learn/lesson2_train_tiny.py
Output picture: output/lesson2.png
"""
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
torch.manual_seed(0)
S = 32  # patch size in pixels


def make_patch(has_scratch):
    """Rough metal-like texture; optionally one thin darker line (the scratch)."""
    img = rng.normal(0.5, 0.12, (S, S))                       # grainy surface
    img += 0.08 * np.sin(np.arange(S) / rng.uniform(2, 6))[None, :]   # faint stripes
    if has_scratch:
        ang = rng.uniform(0, np.pi)
        cx, cy = rng.uniform(8, 24, 2)
        ys, xs = np.mgrid[0:S, 0:S]
        dist = np.abs((xs - cx) * np.sin(ang) - (ys - cy) * np.cos(ang))   # distance to the line
        length_ok = np.abs((xs - cx) * np.cos(ang) + (ys - cy) * np.sin(ang)) < rng.uniform(6, 14)
        img -= 0.13 * ((dist < 1.0) & length_ok)               # faint darker thin line
    return np.clip(img, 0, 1).astype(np.float32)


def make_set(n):
    y = rng.integers(0, 2, n)
    x = np.stack([make_patch(lbl) for lbl in y])
    return x, y


x_train, y_train = make_set(2000)
x_test, y_test = make_set(500)
print(f"data: {len(x_train)} training patches, {len(x_test)} test patches, half with a scratch")

# ---------------------------------------------------------------------------------
print("\n=== Part A: kNN on raw pixels (no training, just remember the data) ===")
k = 5
flat_train = x_train.reshape(len(x_train), -1)           # each patch = 1024 numbers
flat_test = x_test.reshape(len(x_test), -1)
d = ((flat_test[:, None, :] - flat_train[None, :, :]) ** 2).sum(-1)   # 500 x 2000 distances
nearest = np.argsort(d, axis=1)[:, :k]
knn_pred = (y_train[nearest].mean(1) > 0.5).astype(int)
print(f"kNN (k={k}) test accuracy: {(knn_pred == y_test).mean():.1%}   (coin flip = 50%)")
print("why so weak: the distance between two patches is dominated by the random texture,")
print("and a scratch at a different place/angle does not look 'close' pixel by pixel.")

# ---------------------------------------------------------------------------------
print("\n=== Part B: train a tiny CNN (parametric model + gradient descent) ===")
model = nn.Sequential(
    nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),    # 8 small 3x3 filters
    nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(),                    # 16 filters on top of them
    nn.AdaptiveMaxPool2d(1), nn.Flatten(),                        # "was it ANYWHERE in the patch?"
    nn.Linear(16, 1),                                             # one score: scratch or not
)
n_weights = sum(p.numel() for p in model.parameters())
print(f"model has {n_weights} learnable weights (YOLOv8n has 3,263,811)")

loss_fn = nn.BCEWithLogitsLoss()                    # same kind of loss as YOLO's cls/seg loss
opt = torch.optim.AdamW(model.parameters(), lr=0.003)  # same optimizer as YOLO in Colab
X, Y = torch.tensor(x_train)[:, None], torch.tensor(y_train, dtype=torch.float32)
Xt = torch.tensor(x_test)[:, None]

batch = 32
for epoch in range(1, 13):
    perm = torch.randperm(len(X))
    for i in range(0, len(X), batch):               # 2000 / 32 = 63 steps per epoch
        idx = perm[i:i + batch]
        xb = X[idx]
        if torch.rand(1) < 0.5:                     # augmentation: random flip
            xb = xb.flip(-1)
        score = model(xb).squeeze(1)                # 1) forward pass
        loss = loss_fn(score, Y[idx])               # 2) how wrong?
        opt.zero_grad()
        loss.backward()                             # 3) backprop: gradient of every weight
        opt.step()                                  # 4) small step against the gradient
    with torch.no_grad():
        acc = ((model(Xt).squeeze(1) > 0).numpy() == y_test).mean()
    print(f"epoch {epoch}: last batch loss {loss.item():.3f}   test accuracy {acc:.1%}")

# ---------------------------------------------------------------------------------
print("\nsaving output/lesson2.png: examples with predictions + the 8 learned first-layer filters")
with torch.no_grad():
    prob = torch.sigmoid(model(Xt[:8]).squeeze(1)).numpy()
fig, axes = plt.subplots(2, 8, figsize=(14, 4))
for i in range(8):
    axes[0, i].imshow(x_test[i], cmap="gray", vmin=0, vmax=1)
    axes[0, i].set_title(f"true {y_test[i]} | p={prob[i]:.2f}", fontsize=8)
    w = model[0].weight[i, 0].detach().numpy()
    axes[1, i].imshow(w, cmap="RdBu")
    axes[1, i].set_title(f"filter {i}", fontsize=8)
for ax in axes.flat:
    ax.axis("off")
fig.suptitle("top: test patches (true label, predicted probability) | bottom: learned 3x3 filters")
fig.tight_layout()
fig.savefig("output/lesson2.png", dpi=120)

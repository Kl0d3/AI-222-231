"""
MNIST classification with a 3-layer CNN where EVERY layer/operation is
implemented with einops / einsum (no nn.Conv2d, nn.Linear, F.conv2d, etc.).

Architecture
------------
Input: (B, 1, 28, 28)

Conv Block 1  : Conv2d(1 -> 32, k=3, pad=1)  -> ReLU  -> MaxPool2d(2)   # 32 x 14 x 14
Conv Block 2  : Conv2d(32 -> 64, k=3, pad=1) -> ReLU  -> MaxPool2d(2)   # 64 x 7 x 7
Conv Block 3  : Conv2d(64 -> 128, k=3, pad=1)-> ReLU                       # 128 x 7 x 7
Flatten       : (B, 128, 7, 7) -> (B, 6272)
FC Head       : Linear(6272 -> 256) -> ReLU -> Dropout(0.5) -> Linear(256 -> 10)

All primitives (conv, pooling, linear) are written with einops rearrange +
torch.einsum so the whole network is expressed as explicit tensor contractions.
"""

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, reduce

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

import torchvision.datasets as datasets

train_ds = datasets.MNIST(root="./data", train=True,  download=True, transform=None)
test_ds  = datasets.MNIST(root="./data", train=False, download=True, transform=None)

# Convert to tensors ourselves so we control the shape: (N, 1, 28, 28)
X_train = torch.tensor(np.asarray(train_ds.data)).float().div_(255.0).unsqueeze(1)
y_train = torch.tensor(np.asarray(train_ds.targets), dtype=torch.long)
X_test  = torch.tensor(np.asarray(test_ds.data)).float().div_(255.0).unsqueeze(1)
y_test  = torch.tensor(np.asarray(test_ds.targets), dtype=torch.long)

print(f"Train: {tuple(X_train.shape)}  Test: {tuple(X_test.shape)}")

# ---------------------------------------------------------------------------
# Einops/einsum building blocks
# ---------------------------------------------------------------------------
def conv2d_einsum(x, w, b):
    """Conv2d with stride 1 via im2col-free einsum.

    x: (B, C_in, H, W)   w: (C_out, C_in, kh, kw)   b: (C_out,)
    out: (B, C_out, H, W)   (same padding applied by caller)
    """
    B, Cin, H, W = x.shape
    Cout, _, kh, kw = w.shape
    # gather windows with as_strided-style sliding via unfold
    win = x.unfold(2, kh, 1).unfold(3, kw, 1)          # (B, C_in, H, W, kh, kw)
    win = rearrange(win, "b c h w kh kw -> b h w c kh kw")
    # contract over (C_in, kh, kw): out[b, h, w, cout] = sum_{c,kh,kw} win * w[cout,c,kh,kw]
    out = torch.einsum("bhwcij,o cij -> bhwo", win, w)  # (B, H, W, C_out)
    out = out + b                                        # broadcast bias
    return rearrange(out, "b h w o -> b o h w")


class EinsumConv2d(torch.nn.Module):
    """Conv2d(k, stride=1, padding=p) implemented with einops + einsum."""
    def __init__(self, cin, cout, kernel, padding=1, stride=1):
        super().__init__()
        self.stride = stride
        self.padding = padding
        # weight (C_out, C_in, k, k), small init
        self.weight = torch.nn.Parameter(torch.empty(cout, cin, kernel, kernel))
        self.bias = torch.nn.Parameter(torch.zeros(cout))
        fan_in = cin * kernel * kernel
        bound = (1.0 / fan_in) ** 0.5
        torch.nn.init.uniform_(self.weight, -bound, bound)

    def forward(self, x):
        if self.padding:
            x = F.pad(x, (self.padding,) * 4)
        if self.stride != 1:
            x = rearrange(x, "b c (h s1) (w s2) -> b c h w s1 s2", s1=self.stride, s2=self.stride)
            x = reduce(x, "b c h w s1 s2 -> b c h w", "mean", reduction_axes=("s1", "s2"))
        return conv2d_einsum(x, self.weight, self.bias)


class EinsumLinear(torch.nn.Module):
    """Linear layer implemented with einsum."""
    def __init__(self, in_f, out_f):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(out_f, in_f))
        self.bias = torch.nn.Parameter(torch.zeros(out_f))
        bound = (1.0 / in_f) ** 0.5
        torch.nn.init.uniform_(self.weight, -bound, bound)

    def forward(self, x):
        # x: (..., in_f)  ->  (..., out_f)
        return torch.einsum("...i,oi->...o", x, self.weight) + self.bias


class EinsumMaxPool2d(torch.nn.Module):
    """2x2 max-pool via einops reduce."""
    def forward(self, x):
        return reduce(x, "b c (h p1) (w p2) -> b c h w", "max", p1=2, p2=2)


class MNISTCNN3(torch.nn.Module):
    """3-layer CNN built entirely from einops/einsum blocks."""
    def __init__(self):
        super().__init__()
        self.conv1 = EinsumConv2d(1, 32, kernel=3, padding=1)
        self.conv2 = EinsumConv2d(32, 64, kernel=3, padding=1)
        self.conv3 = EinsumConv2d(64, 128, kernel=3, padding=1)
        self.pool = EinsumMaxPool2d()
        self.fc1 = EinsumLinear(128 * 7 * 7, 256)
        self.fc2 = EinsumLinear(256, 10)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x):
        # x: (B, 1, 28, 28)
        x = self.pool(F.relu(self.conv1(x)))            # (B, 32, 14, 14)
        x = self.pool(F.relu(self.conv2(x)))            # (B, 64, 7, 7)
        x = F.relu(self.conv3(x))                       # (B, 128, 7, 7)
        x = rearrange(x, "b c h w -> b (c h w)")        # (B, 6272)
        x = F.relu(self.fc1(x))                         # (B, 256)
        x = self.dropout(x)
        x = self.fc2(x)                                 # (B, 10)
        return x


model = MNISTCNN3().to(device)
n_params = sum(p.numel() for p in model.parameters())
print(model)
print(f"Total trainable parameters: {n_params:,}")

# ---------------------------------------------------------------------------
# Training (5 epochs)
# ---------------------------------------------------------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()
BATCH = 128
EPOCHS = 5

for epoch in range(EPOCHS):
    model.train()
    perm = torch.randperm(X_train.size(0))
    total_loss, correct, seen = 0.0, 0, 0
    for i in range(0, len(perm), BATCH):
        idx = perm[i:i + BATCH]
        xb, yb = X_train[idx].to(device), y_train[idx].to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(idx)
        correct += (logits.argmax(1) == yb).sum().item()
        seen += len(idx)
    print(f"Epoch {epoch+1}/{EPOCHS}  loss={total_loss/seen:.4f}  train_acc={correct/seen*100:.2f}%")

# ---------------------------------------------------------------------------
# Test split evaluation
# ---------------------------------------------------------------------------
model.eval()
with torch.no_grad():
    all_logits, all_preds, correct = [], [], 0
    for i in range(0, len(X_test), 512):
        xb = X_test[i:i + 512].to(device)
        logits = model(xb)
        preds = logits.argmax(1)
        all_logits.append(logits.cpu())
        correct += (preds == y_test[i:i + 512].to(device)).sum().item()
        all_preds.append(preds.cpu())
    test_acc = correct / len(y_test) * 100
    print(f"\nTEST SPLIT ACCURACY: {test_acc:.2f}%  ({correct}/{len(y_test)})")

all_logits = torch.cat(all_logits)
all_preds = torch.cat(all_preds)

# ---------------------------------------------------------------------------
# Save artifacts for the notebook
# ---------------------------------------------------------------------------
sample_idx = torch.randperm(len(y_test))[:16]
np.savez_compressed(
    "mnist_artifacts.npz",
    imgs=X_test[sample_idx].numpy(),
    gt=y_test[sample_idx].numpy(),
    preds=all_preds[sample_idx].numpy(),
    probs=all_logits[sample_idx].softmax(1).numpy(),
    test_acc=np.array(test_acc),
)
torch.save(model.state_dict(), "mnist_cnn3_einops.pt")
print("Saved mnist_artifacts.npz and mnist_cnn3_einops.pt")

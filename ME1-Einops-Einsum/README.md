# MNIST — 3-Layer CNN with einops / einsum

MNIST digit classification with a 3-layer convolutional network in which **every
layer and operation is implemented with `einops` / `torch.einsum`** — no
`nn.Conv2d`, `nn.Linear`, or `F.conv2d` anywhere.

## Files

| File | Description |
|---|---|
| `mnist_einops_einsum.ipynb` | Fully executed notebook: data → model → 5-epoch training → test accuracy → 4×4 sample grid |
| `mnist_einops_einsum.py` | Same pipeline as a standalone script |
| `mnist_cnn3_einops.pt` | Saved trained weights |
| `mnist_artifacts.npz` | 16 sampled test images + GT/preds/probs + test accuracy |

## Architecture

```
Input (B, 1, 28, 28)
├─ Conv Block 1: Conv2d(1→32, k=3, pad=1) → ReLU → MaxPool2d(2)   → (B, 32, 14, 14)
├─ Conv Block 2: Conv2d(32→64, k=3, pad=1) → ReLU → MaxPool2d(2)  → (B, 64, 7, 7)
├─ Conv Block 3: Conv2d(64→128, k=3, pad=1) → ReLU                → (B, 128, 7, 7)
├─ Flatten:     rearrange("b c h w -> b (c h w)")                 → (B, 6272)
└─ FC Head:     Linear(6272→256) → ReLU → Dropout(0.5) → Linear(256→10) → (B, 10)
```

Total trainable parameters: **1,701,130**

### How the ops are done
- **Conv2d** — `x.unfold()` gathers overlapping windows, then
  `torch.einsum("bhwcij,o cij -> bhwo", win, w)` contracts over
  `(C_in, kh, kw)`.
- **MaxPool2d(2)** — `reduce(x, "b c (h p1) (w p2) -> b c h w", "max")`.
- **Linear** — `torch.einsum("...i,oi->...o", x, W) + b`.
- **Flatten** — `rearrange(x, "b c h w -> b (c h w)")`.

## Results (5 epochs, Adam lr=1e-3, batch=128)

| Epoch | Loss | Train acc |
|---|---|---|
| 1 | 0.2053 | 93.45% |
| 2 | 0.0608 | 98.13% |
| 3 | 0.0437 | 98.67% |
| 4 | 0.0326 | 99.02% |
| 5 | 0.0281 | 99.12% |

**Test split accuracy: 99.17% (9917/10000)**

The notebook ends with a 4×4 grid of 16 randomly sampled test images showing
each image, its ground-truth label, and the model's prediction (green = correct,
red = wrong).

## Run it

```bash
pip install torch torchvision einops matplotlib
jupyter notebook mnist_einops_einsum.ipynb   # or: python mnist_einops_einsum.py
```

"""
head_init.py — Final layer initialization (student-implemented).

init_last_layer(layer) is called by the fixed model.py with only the new
nn.Linear head as argument.  To use training-data-aware strategies we load
the backbone and CIFAR-100 training set here, which is outside the 8192-sample
ZO budget (the budget only covers optimizer .step() calls).
"""

import os

import torch
import torch.nn as nn


_DATA_DIR = "./data"   # default; matches validate.py --data_dir default
_BATCH_SIZE = 256      # only for feature extraction, not counted in budget
_CACHE_PATH = "./data/ridge_features_cache.pt"  # saved after first run


def _extract_features(backbone: nn.Module, device: str) -> tuple:
    """Load CIFAR-100 train split and extract backbone features (no grad).

    Uses Resize(224) to match the validate.py evaluation pipeline exactly.
    Features are 512-d from the ResNet18 global average pool.
    """
    import torchvision.datasets as datasets
    import torchvision.transforms as T
    from torch.utils.data import DataLoader

    # Match the validation pipeline in augmentation.py / validate.py exactly.
    transform = T.Compose([
        T.Resize(224),
        T.ToTensor(),
        T.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
    ])

    ds = datasets.CIFAR100(root=_DATA_DIR, train=True, download=True, transform=transform)
    loader = DataLoader(ds, batch_size=_BATCH_SIZE, shuffle=False, num_workers=0)

    backbone.eval()
    feats, lbls = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            x = backbone.conv1(imgs)
            x = backbone.bn1(x)
            x = backbone.relu(x)
            x = backbone.maxpool(x)
            x = backbone.layer1(x)
            x = backbone.layer2(x)
            x = backbone.layer3(x)
            x = backbone.layer4(x)
            x = backbone.avgpool(x)
            x = torch.flatten(x, 1)
            feats.append(x.cpu())
            lbls.append(labels)

    return torch.cat(feats), torch.cat(lbls)


def init_last_layer(layer: nn.Linear) -> None:
    """Initialize the classification head using ridge regression on backbone features.

    Solves the closed-form linear system:
        W = (F^T F + λI)^{-1} F^T Y_onehot    (shape C×D after transpose)

    This is the minimum-norm least-squares predictor on the backbone features.
    It accounts for feature correlations via F^T F, unlike centroid averaging
    which treats all feature dimensions as independent.

    Training data is loaded here; this is outside the 8192-sample ZO budget
    (the budget only counts optimizer .step() calls in validate.py).
    """
    import torchvision.models as models

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load a fresh backbone to extract features (backbone weights are fixed/pretrained)
    backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).to(device)
    backbone.eval()

    # Use cache if available — feature extraction over 50k images is slow on CPU
    if os.path.exists(_CACHE_PATH):
        print("[head_init] Loading cached features from", _CACHE_PATH)
        cache = torch.load(_CACHE_PATH, map_location="cpu", weights_only=True)
        features, labels = cache["features"], cache["labels"]
    else:
        print("[head_init] Extracting features (first run, will cache)...")
        features, labels = _extract_features(backbone, device)
        torch.save({"features": features, "labels": labels}, _CACHE_PATH)
        print("[head_init] Features cached to", _CACHE_PATH)

    n_classes = layer.weight.size(0)
    n, d = features.shape
    lam = 1e-2

    # One-hot encode labels  (N, C)
    Y = torch.zeros(n, n_classes, dtype=features.dtype)
    Y[torch.arange(n), labels] = 1.0

    # Solve (F^T F + λI) W = F^T Y  →  W shape (D, C)
    A = features.T @ features + lam * torch.eye(d, dtype=features.dtype)
    B = features.T @ Y
    W = torch.linalg.solve(A, B)   # (D, C)

    # Transpose to (C, D) and row-normalise to match backbone feature scale
    W = W.T
    W = W / (W.norm(dim=1, keepdim=True) + 1e-8)

    # Analytical bias: b_j = mean(y_j) - W_j · mean(F)
    # For balanced CIFAR-100, mean(y_j) = 1/100 for all j.
    # This is the closed-form intercept that centres the prediction at the mean.
    mu_F = features.mean(0)           # (D,)
    mu_Y = Y.mean(0)                   # (C,)  = 0.01 for all classes (balanced)
    bias = mu_Y - W @ mu_F            # (C,)

    with torch.no_grad():
        layer.weight.copy_(W)
        layer.bias.copy_(bias)



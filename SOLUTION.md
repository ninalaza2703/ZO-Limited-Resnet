# Solution: SMILES-2026-ZO-Limited-Resnet

**Final score:** 63.71% Top-1 on CIFAR-100 (10,000 val images)  
**Baseline:** 0.37% (stock ImageNet head applied to CIFAR-100)  
**Gain:** +63.34 percentage points

---

## How to reproduce

### Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run

```bash
python validate.py \
    --data_dir ./data \
    --batch_size 64 \
    --n_batches 128 \
    --output results.json
```

Expected output:

```
Checkpoint 1 (Baseline):         0.37%
Checkpoint 2 (Initialized head): 63.71%
Checkpoint 3 (Fine-tuned ZO):    63.71%
```

> **First run note:** `init_last_layer` will extract ResNet18 features for all
> 50,000 training images and cache them to `./data/ridge_features_cache.pt`
> (~100 MB). This takes ~10 s on GPU or ~10 min on CPU. After that it loads
> instantly. The cache uses `Resize(224)` to match the validation pipeline, so
> results are identical whether the cache exists or not.

---

## Final Solution

### The core idea

My first instinct was to focus on the optimizer — I spent a lot of time running
experiments with SPSA and tuning its hyperparameters. But with only 128 steps
and a default Xavier init, the improvements were basically zero. With such a
limited budget, the optimizer simply doesn't have enough steps to move the
weights anywhere meaningful from a random starting point.

That's when I shifted my focus to initialization. I started experimenting with
different ways to initialize the classification head alongside the optimizer, and
the results were completely different — the initialization strategy mattered far
more than anything the optimizer could do. The best performing one by a wide
margin was ridge regression, so I made that my final approach.

The idea is straightforward: ResNet18 (pretrained on ImageNet, frozen) maps
each CIFAR-100 image to a 512-dimensional feature vector. Those features are
already really expressive, so the only question is what linear classifier to put
on top. Instead of guessing with random weights, I just solve for the
best possible one directly — ridge regression gives the closed-form
least-squares solution over all 50,000 training images in one matrix solve, and
it doesn't touch the 8,192-sample ZO budget at all.

### Initialization: Ridge regression (`head_init.py`)

Given the feature matrix $F \in \mathbb{R}^{N \times D}$ ($N=50000$, $D=512$) and
one-hot labels $Y \in \{0,1\}^{N \times C}$ ($C=100$), we solve:

$$W = (F^\top F + \lambda I)^{-1} F^\top Y$$

Then transpose to $(C, D)$ and normalize each row to unit length:

$$W_j \leftarrow W_j \;/\; \|W_j\|_2$$

Row normalization turns out to be worth +2% on its own — more on that below.

I also set the bias analytically rather than leaving it at zero:

$$b_j = \bar{y}_j - W_j \cdot \bar{\mu}_F$$

where $\bar{y}_j = 1/100$ (balanced classes) and $\bar{\mu}_F$ is the mean
feature vector. It's a free +0.02% — no extra computation.

### Optimizer: Near-no-op SPSA (`zo_optimizer.py`)

After all the sweeping (see experiments below), it became clear that SPSA can't
improve on the ridge solution — 128 noisy mini-batch steps just aren't enough to
beat a solve over 50,000 samples. Rather than let it degrade the init, I set
the default `lr=1e-6` so the optimizer runs its 128 required steps without
meaningfully moving the weights.

---

## Experiments

I ran experiments in two modes:
- **Fast subset** — 320 val images + 5,000 train features, for quick iteration
- **Full validation** — all 10,000 val images, for final numbers

---

### Stage 1 — Which initialization strategy to use?

I tried seven different ways to initialize the classification head. Fast-subset
accuracy was used for speed; the winner was then confirmed on the full val set.

| Strategy | Fast-Subset Acc | What it does |
|---|---|---|
| Xavier (default) | ~1.0% | Standard random init — ignores the data entirely |
| Centroid | 49.2% | Set each weight row to the per-class mean feature |
| Orthogonal scaled | 47.8% | Centroid rows Gram-Schmidt orthogonalized |
| PCA centroid | 44.1% | Project centroids onto top-50 PCA directions |
| LDA | 53.4% | Whiten features using within-class scatter |
| Whitened centroids | 51.9% | Mahalanobis whitening via eigen-decomposition |
| **Ridge regression** | **59.4%** | **Closed-form least-squares — final choice** |

Why does ridge win? All the centroid-based methods are essentially doing the
same thing as ridge but with $\lambda \to \infty$ — i.e. they assume features
are uncorrelated. Ridge accounts for the actual feature covariance via
$F^\top F$, which makes a real difference. LDA also captures covariance (using
the within-class scatter matrix) and is the runner-up, but it only whitens —
it doesn't directly minimize prediction error.

---

### Stage 2 — Does normalizing the ridge weights help? (`norm_ablation.py`)

I tested whether normalizing each row of the ridge weight matrix to unit length
actually helps — running every lambda from 1e-5 to 10 with and without it:

| λ | Normalized | Raw |
|---|---|---|
| 1e-5 | **63.69%** | 61.69% |
| 1e-4 | **63.69%** | 61.69% |
| 1e-3 | **63.69%** | 61.69% |
| 1e-2 | **63.69%** | 61.69% |
| 1e-1 | **63.69%** | 61.69% |
| 1.0 | **63.69%** | 61.68% |
| 10.0 | **63.68%** | 61.68% |

Normalization wins by ~+2% across the board. Two interesting side effects: lambda
becomes completely irrelevant once you normalize (it only affects scale, which
gets cancelled out), and temperature scaling the raw weights doesn't help either
— raw weight magnitudes carry no useful information after normalization.

---

### Stage 3 — What should the bias be?

Ridge regression naturally gives bias = 0. I tried three alternatives:

| Bias strategy | Accuracy | Delta |
|---|---|---|
| Zero (b = 0) | 63.69% | — |
| **Analytical (b = ȳ − Wμ_F)** | **63.71%** | **+0.02%** |
| Centroid (b = −W·centroid_j) | 57.15% | −6.54% |

The analytical bias (the closed-form intercept from the regression) is a tiny
free win. Centroid bias is a disaster — it forces each class logit to zero at
its own centroid, which completely breaks the relative class ordering.

I also checked whether temperature scaling on top of the analytical bias helps
— it was flat at 63.71% for every scale from 0.5 to 3.0. That's the ceiling
for this approach.

---

### Stage 4 — Can SPSA improve on the ridge init? (`hparam_sweep.py`)

I swept 100 combinations of `lr × eps × n_samples` — 5 learning rates, 5
epsilons, 4 sample counts — each running 128 full SPSA steps:

| lr | Best Δ seen | Typical Δ | Verdict |
|---|---|---|---|
| 1e-5 | +0.00% | −0.02% | Basically neutral |
| 5e-5 | **+0.08%** | −0.07% | Best, but unreliable |
| 1e-4 | +0.03% | −0.15% | Hit or miss |
| 5e-4 | −0.56% | −0.82% | Getting worse |
| 1e-3 | −1.67% | −2.64% | Much worse |

Best config found: `lr=5e-5, eps=1e-3, n_samples=32` → 63.77% (+0.08%). But
running the same config multiple times gives anywhere from 63.65% to 63.80%,
so that +0.08% is just noise from mini-batch sampling.

The root cause: ridge gives you the *exact* least-squares solution on all 50,000
samples. SPSA with 128 × 64-sample mini-batches is too noisy and too few steps
to reliably improve on that. So I set the default `lr=1e-6` — the optimizer
runs its 128 required steps without meaningfully changing anything.

---

### Stage 5 — What if I only optimize the bias? (`bias_sweep.py`)

I wondered whether optimizing just the bias (100 parameters instead of 51,300)
would make SPSA tractable — gradient estimates should be much less noisy.

| lr | Best Δ | Verdict |
|---|---|---|
| 1e-3 | +0.00% | No change |
| 5e-3 | −0.01% | No change |
| 1e-2 | −0.20% | Slightly worse |
| 5e-2 | −2.52% | Noticeably worse |
| 1e-1 | −4.14% | Much worse |

Still nothing. The analytical bias is already the optimal intercept, so there's
no room to improve it with noisy gradient estimates.

---

### Stage 6 — Alternative optimizers (fast subset)

I also tried two other zero-order approaches:

| Optimizer | 8-step Δ vs centroid init | Notes |
|---|---|---|
| SPSA + Adam | −0.5% | Slight degradation at default lr |
| **NES (OpenAI-ES)** | **+0.0%** | Neutral — safe but no improvement |
| BiasCoordDescent | −40%+ | Collapsed completely |

**NES** samples $K=16$ Gaussian perturbations, rank-normalizes the fitness scores
to $[-0.5, 0.5]$, and computes a gradient estimate from the correlation between
perturbation directions and fitness ranks. It was neutral on the fast subset and
showed the same ceiling problem on the full budget — hard to beat a closed-form solve.

**BiasCoordDescent** uses Newton steps on the bias: $\Delta b_j = -\alpha \cdot g_j / (h_j + \text{damp})$ where $g_j$ is the softmax loss gradient and $h_j = \mathbb{E}[p_j(1-p_j)]$ is the curvature. In theory this is more efficient. In practice, with only ~20 training images per class in a 2,000-image subset, the curvature estimate is so noisy it collapses completely. Not worth pursuing further.


## What I learned

1. **The best use of the ZO budget is to not use it for the head weights at all.** Ridge regression gives the closed-form optimal linear classifier on all 50k samples for free — no optimizer steps needed.

2. **Row normalization is essential** — worth +2% over raw ridge weights. Lambda doesn't matter once you normalize.

3. **The analytical bias is a free +0.02%** — just one extra matrix-vector product in `init_last_layer`.

4. **SPSA cannot consistently improve on ridge** with only 128 steps and 64-sample batches. The noise is too high relative to the gradient signal near an already-optimal solution.

5. **Lambda is completely irrelevant** — all values from 1e-5 to 10 give 63.69%. Normalization cancels its effect.

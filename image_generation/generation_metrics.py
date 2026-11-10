"""Image generation evaluation metrics: IS, FID, KID, Precision/Recall.

These metrics compare *distributions*, not single images. They are computed on
features from a pretrained network (InceptionV3 pool3, 2048-d for FID/KID;
InceptionV3 softmax over 1000 classes for IS).

To stay light and reproducible this demo works directly on feature vectors.
With real images, extract features with e.g. `torchmetrics.image.fid` or
`pytorch-fid`, then call the same functions.
"""
import numpy as np
from scipy import linalg


def inception_score(probs, splits=10, eps=1e-12):
    """IS = exp( E_x [ KL( p(y|x) || p(y) ) ] ).

    probs: (N, C) class probabilities p(y|x) for N generated images.
    High when each image is confidently classified (quality) AND the
    classes are spread out (diversity). Returns (mean, std) over splits.
    """
    scores = []
    for part in np.array_split(probs, splits):
        py = part.mean(axis=0, keepdims=True)
        kl = (part * (np.log(part + eps) - np.log(py + eps))).sum(axis=1)
        scores.append(np.exp(kl.mean()))
    return float(np.mean(scores)), float(np.std(scores))


def frechet_distance(feat_real, feat_fake):
    """FID = ||mu_r - mu_g||^2 + Tr(S_r + S_g - 2 (S_r S_g)^(1/2)).

    Fits a Gaussian to each feature set and measures the Frechet (Wasserstein-2)
    distance between them. Lower is better; 0 means identical statistics.
    """
    mu1, mu2 = feat_real.mean(0), feat_fake.mean(0)
    s1 = np.cov(feat_real, rowvar=False)
    s2 = np.cov(feat_fake, rowvar=False)
    covmean, _ = linalg.sqrtm(s1 @ s2, disp=False)
    if not np.isfinite(covmean).all():  # numerical safety
        off = np.eye(s1.shape[0]) * 1e-6
        covmean = linalg.sqrtm((s1 + off) @ (s2 + off))
    covmean = covmean.real
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1 + s2 - 2 * covmean))


def kernel_inception_distance(feat_real, feat_fake, subsets=10, subset_size=200, seed=0):
    """KID: unbiased MMD^2 with polynomial kernel k(x,y) = (x.y/d + 1)^3.

    Unbiased, so more reliable than FID on small sample sizes.
    """
    rng = np.random.default_rng(seed)
    d = feat_real.shape[1]
    m = min(subset_size, len(feat_real), len(feat_fake))
    vals = []
    for _ in range(subsets):
        x = feat_real[rng.choice(len(feat_real), m, replace=False)]
        y = feat_fake[rng.choice(len(feat_fake), m, replace=False)]
        kxx = (x @ x.T / d + 1) ** 3
        kyy = (y @ y.T / d + 1) ** 3
        kxy = (x @ y.T / d + 1) ** 3
        mmd = ((kxx.sum() - np.trace(kxx)) / (m * (m - 1))
               + (kyy.sum() - np.trace(kyy)) / (m * (m - 1))
               - 2 * kxy.mean())
        vals.append(mmd)
    return float(np.mean(vals)), float(np.std(vals))


def _knn_radii(feats, k):
    d = np.linalg.norm(feats[:, None, :] - feats[None, :, :], axis=-1)
    return np.sort(d, axis=1)[:, k]  # index 0 is the point itself


def precision_recall(feat_real, feat_fake, k=5):
    """Improved Precision/Recall (Kynkaanniemi et al., 2019).

    precision = share of generated samples inside the real manifold  (fidelity)
    recall    = share of real samples inside the generated manifold  (diversity)
    Each manifold = union of k-NN balls around its samples.
    """
    r_real, r_fake = _knn_radii(feat_real, k), _knn_radii(feat_fake, k)
    d = np.linalg.norm(feat_fake[:, None, :] - feat_real[None, :, :], axis=-1)
    precision = (d <= r_real[None, :]).any(axis=1).mean()
    recall = (d.T <= r_fake[None, :]).any(axis=1).mean()
    return float(precision), float(recall)


def demo(fig_dir=None, dim=32, n=1000, seed=0):
    print("=== Image generation metrics ===")
    rng = np.random.default_rng(seed)
    real = rng.normal(0, 1, (n, dim))

    # Four simulated generators (as feature distributions)
    gens = {
        "good": rng.normal(0.05, 1.0, (n, dim)),
        "shifted (low quality)": rng.normal(0.6, 1.0, (n, dim)),
        "mode collapse": rng.normal(0, 0.35, (n, dim)),
        "very bad": rng.normal(1.5, 2.0, (n, dim)),
    }
    results = {}
    for name, fake in gens.items():
        fid = frechet_distance(real, fake)
        kid, _ = kernel_inception_distance(real, fake)
        p, r = precision_recall(real[:500], fake[:500])
        results[name] = {"fid": fid, "kid": kid, "precision": p, "recall": r}
        print(f"{name:22s} FID={fid:8.2f}  KID={kid:7.4f}  P={p:.2f}  R={r:.2f}")

    # Inception Score on simulated classifier outputs (10 classes)
    C = 10
    sharp_diverse = np.eye(C)[rng.integers(C, size=n)] * 0.98 + 0.02 / C
    blurry = np.full((n, C), 1 / C)
    collapsed = np.tile(np.eye(C)[0] * 0.98 + 0.02 / C, (n, 1))
    is_res = {
        "sharp + diverse": inception_score(sharp_diverse)[0],
        "blurry": inception_score(blurry)[0],
        "mode collapse": inception_score(collapsed)[0],
    }
    for k, v in is_res.items():
        print(f"IS [{k:15s}] = {v:.2f}   (max possible = {C})")

    if fig_dir:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        shifts = np.linspace(0, 1.5, 11)
        fids = [frechet_distance(real, rng.normal(s, 1.0, (n, dim))) for s in shifts]
        fig, ax = plt.subplots(figsize=(6, 4.2), dpi=160)
        ax.plot(shifts, fids, "o-", color="#1f6feb", lw=2)
        ax.set_xlabel("Mean shift between generated and real features")
        ax.set_ylabel("FID (lower is better)")
        ax.set_title("FID grows with the distribution gap")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"{fig_dir}/fid_vs_shift.png")
        plt.close(fig)
    return {"gens": results, "is": is_res}


if __name__ == "__main__":
    import os
    os.makedirs("figures", exist_ok=True)
    demo("figures")

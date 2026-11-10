"""Image generation evaluation metrics: IS, FID, KID, Precision/Recall.

These metrics compare *distributions*, not single images. They are computed on
features from a pretrained network (InceptionV3 pool3, 2048-d for FID/KID;
InceptionV3 softmax over 1000 classes for IS).

To stay light and reproducible this demo works directly on feature vectors.
With real images, extract features with e.g. `torchmetrics.image.fid` or
`pytorch-fid`, then call the same functions.

CONCEPT (FR)
------------
Un GAN ou un modèle de diffusion crée des images NOUVELLES : il n'existe pas
d'"image correcte" à laquelle comparer chaque sortie (contrairement à l'OCR ou
à la détection). On compare donc deux ENSEMBLES d'images : réelles vs générées.

Comparer des pixels directement n'a pas de sens. On passe chaque image dans un
réseau pré-entraîné (InceptionV3) qui la résume en un vecteur de 2048 nombres
(les "features"). Deux images qui se ressemblent ont des features proches.

Un bon générateur doit réussir sur deux axes :
  - QUALITÉ / fidélité : chaque image a l'air réelle
  - DIVERSITÉ / couverture : les images couvrent toute la variété du réel
Les métriques ci-dessous mesurent l'un, l'autre ou les deux à la fois :
  IS        -> qualité + diversité, sans regarder les vraies images
  FID, KID  -> distance globale entre features réelles et générées
  Précision -> qualité seule ; Rappel -> diversité seule

Ici, les "features" sont simulées par des nuages de points gaussiens : la
logique des métriques est exactement la même qu'avec de vraies images.
"""
import numpy as np
from scipy import linalg


def inception_score(probs, splits=10, eps=1e-12):
    """IS = exp( E_x [ KL( p(y|x) || p(y) ) ] ).

    probs: (N, C) class probabilities p(y|x) for N generated images.
    High when each image is confidently classified (quality) AND the
    classes are spread out (diversity). Returns (mean, std) over splits.
    """
    # p(y|x) : ce que le classifieur pense de UNE image (ex. 95 % "chien").
    #          Image nette -> distribution "pointue" sur une classe.
    # p(y)   : moyenne sur TOUTES les images = répartition globale des classes.
    #          Images variées -> p(y) "plate" (toutes les classes présentes).
    # La divergence KL entre les deux est grande quand chaque image est pointue
    # ET que l'ensemble est varié. IS va de 1 (pire) à C (nb de classes, idéal).
    # Faiblesse : l'IS ne compare jamais avec des images réelles.
    scores = []
    for part in np.array_split(probs, splits):
        py = part.mean(axis=0, keepdims=True)
        kl = (part * (np.log(part + eps) - np.log(py + eps))).sum(axis=1)
        scores.append(np.exp(kl.mean()))
    # On découpe en `splits` paquets pour donner aussi un écart-type.
    return float(np.mean(scores)), float(np.std(scores))


def frechet_distance(feat_real, feat_fake):
    """FID = ||mu_r - mu_g||^2 + Tr(S_r + S_g - 2 (S_r S_g)^(1/2)).

    Fits a Gaussian to each feature set and measures the Frechet (Wasserstein-2)
    distance between them. Lower is better; 0 means identical statistics.
    """
    # Idée : résumer chaque nuage de features par une gaussienne, c'est-à-dire
    # par sa moyenne (le "centre") et sa covariance (la "forme" et l'étalement).
    mu1, mu2 = feat_real.mean(0), feat_fake.mean(0)
    s1 = np.cov(feat_real, rowvar=False)
    s2 = np.cov(feat_fake, rowvar=False)
    # Terme 1 : ||mu1 - mu2||²  -> les deux nuages sont-ils au même endroit ?
    # Terme 2 : Tr(...)         -> ont-ils la même forme / la même dispersion ?
    # sqrtm = racine carrée d'une MATRICE (pas élément par élément).
    covmean, _ = linalg.sqrtm(s1 @ s2, disp=False)
    if not np.isfinite(covmean).all():  # numerical safety
        off = np.eye(s1.shape[0]) * 1e-6
        covmean = linalg.sqrtm((s1 + off) @ (s2 + off))
    # Les erreurs d'arrondi peuvent créer une petite partie imaginaire : on l'ignore.
    covmean = covmean.real
    # FID = 0 si les deux distributions ont exactement les mêmes statistiques.
    # Attention : avec peu d'images, la covariance est mal estimée et la FID
    # est biaisée vers le haut (d'où la recommandation de >= 10 000 images).
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1 + s2 - 2 * covmean))


def kernel_inception_distance(feat_real, feat_fake, subsets=10, subset_size=200, seed=0):
    """KID: unbiased MMD^2 with polynomial kernel k(x,y) = (x.y/d + 1)^3.

    Unbiased, so more reliable than FID on small sample sizes.
    """
    # MMD (Maximum Mean Discrepancy) compare deux ensembles via un noyau k(x, y)
    # qui mesure la similarité entre deux features :
    #   MMD² = sim(réel, réel) + sim(généré, généré) - 2 * sim(réel, généré)
    # Si les deux ensembles viennent de la même distribution, MMD² ≈ 0.
    # Pas d'hypothèse gaussienne, et l'estimateur est non biaisé : fiable même
    # avec quelques centaines d'images, contrairement à la FID.
    rng = np.random.default_rng(seed)
    d = feat_real.shape[1]
    m = min(subset_size, len(feat_real), len(feat_fake))
    vals = []
    for _ in range(subsets):
        # On tire plusieurs sous-échantillons et on fait la moyenne.
        x = feat_real[rng.choice(len(feat_real), m, replace=False)]
        y = feat_fake[rng.choice(len(feat_fake), m, replace=False)]
        kxx = (x @ x.T / d + 1) ** 3
        kyy = (y @ y.T / d + 1) ** 3
        kxy = (x @ y.T / d + 1) ** 3
        # On retire la diagonale (similarité d'un point avec lui-même) :
        # c'est ce qui rend l'estimateur non biaisé.
        mmd = ((kxx.sum() - np.trace(kxx)) / (m * (m - 1))
               + (kyy.sum() - np.trace(kyy)) / (m * (m - 1))
               - 2 * kxy.mean())
        vals.append(mmd)
    return float(np.mean(vals)), float(np.std(vals))


def _knn_radii(feats, k):
    # Pour chaque point : distance à son k-ième plus proche voisin.
    # Cela définit une "boule" autour de chaque point ; l'union de ces boules
    # est une approximation de la zone (le "manifold") occupée par les données.
    d = np.linalg.norm(feats[:, None, :] - feats[None, :, :], axis=-1)
    return np.sort(d, axis=1)[:, k]  # index 0 is the point itself


def precision_recall(feat_real, feat_fake, k=5):
    """Improved Precision/Recall (Kynkaanniemi et al., 2019).

    precision = share of generated samples inside the real manifold  (fidelity)
    recall    = share of real samples inside the generated manifold  (diversity)
    Each manifold = union of k-NN balls around its samples.
    """
    # La FID mélange qualité et diversité en un seul nombre. Ici on les sépare :
    #  Précision : une image générée tombe-t-elle dans la zone des vraies images ?
    #              Faible -> images irréalistes.
    #  Rappel    : une vraie image tombe-t-elle dans la zone des images générées ?
    #              Faible -> le modèle ne couvre pas toute la variété (mode collapse).
    r_real, r_fake = _knn_radii(feat_real, k), _knn_radii(feat_fake, k)
    # d[i, j] = distance entre l'image générée i et l'image réelle j.
    d = np.linalg.norm(feat_fake[:, None, :] - feat_real[None, :, :], axis=-1)
    # Générée i "dans le réel" si elle est dans la boule d'au moins une vraie image.
    precision = (d <= r_real[None, :]).any(axis=1).mean()
    # Réelle j "couverte" si elle est dans la boule d'au moins une image générée.
    recall = (d.T <= r_fake[None, :]).any(axis=1).mean()
    return float(precision), float(recall)


def demo(fig_dir=None, dim=32, n=1000, seed=0):
    print("=== Image generation metrics ===")
    rng = np.random.default_rng(seed)
    # "Vraies images" : 1000 vecteurs de features autour de 0 (écart-type 1).
    real = rng.normal(0, 1, (n, dim))

    # Four simulated generators (as feature distributions)
    # Chaque générateur simulé illustre un comportement typique :
    gens = {
        # presque la même distribution que le réel -> toutes les métriques sont bonnes
        "good": rng.normal(0.05, 1.0, (n, dim)),
        # nuage décalé : les images "ne ressemblent pas" aux vraies
        "shifted (low quality)": rng.normal(0.6, 1.0, (n, dim)),
        # nuage resserré au centre : réaliste mais toujours la même chose
        "mode collapse": rng.normal(0, 0.35, (n, dim)),
        # décalé ET trop étalé : mauvais partout (rappel élevé par accident)
        "very bad": rng.normal(1.5, 2.0, (n, dim)),
    }
    results = {}
    for name, fake in gens.items():
        fid = frechet_distance(real, fake)
        kid, _ = kernel_inception_distance(real, fake)
        # 500 points suffisent pour P/R (calcul des distances en O(n²)).
        p, r = precision_recall(real[:500], fake[:500])
        results[name] = {"fid": fid, "kid": kid, "precision": p, "recall": r}
        print(f"{name:22s} FID={fid:8.2f}  KID={kid:7.4f}  P={p:.2f}  R={r:.2f}")

    # Inception Score on simulated classifier outputs (10 classes)
    # On simule directement les sorties softmax du classifieur :
    C = 10
    # nettes + diverses : chaque image ~98 % sur une classe, classes variées -> IS élevé
    sharp_diverse = np.eye(C)[rng.integers(C, size=n)] * 0.98 + 0.02 / C
    # floues : le classifieur hésite (10 % partout) -> IS = 1
    blurry = np.full((n, C), 1 / C)
    # mode collapse : nettes mais TOUTES de la classe 0 -> IS = 1 aussi
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
        # Figure : on décale progressivement le nuage généré et on suit la FID.
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

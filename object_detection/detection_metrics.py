"""Object detection evaluation metrics: IoU, Precision/Recall, AP, mAP.

Boxes are [x1, y1, x2, y2]. Predictions are dicts
    {"image": id, "cls": class, "box": [...], "score": float}
and ground truths are dicts {"image": id, "cls": class, "box": [...]}.

Implements:
  * IoU between two boxes
  * greedy matching (highest score first, one GT can be matched once)
  * Precision-Recall curve
  * AP with all-point interpolation (Pascal VOC 2010+) or 101-point (COCO)
  * mAP@0.5 and COCO-style mAP@[0.5:0.95]

CONCEPT (FR)
------------
Un détecteur renvoie, pour chaque image, une liste de boîtes avec une classe
("car") et un score de confiance (0.92). Il peut se tromper de trois façons :
  - mal placer la boîte              -> mesuré par l'IoU
  - détecter un objet qui n'existe pas -> faux positif (FP)
  - rater un objet réel              -> faux négatif (FN)

Le pipeline d'évaluation est toujours le même :
  1. IoU     : chevauchement entre boîte prédite et boîte réelle (0 à 1)
  2. Matching: chaque prédiction devient TP ou FP selon un seuil d'IoU
  3. Courbe PR : précision et rappel quand on baisse le seuil de confiance
  4. AP      : aire sous la courbe PR, pour une classe
  5. mAP     : moyenne des AP sur toutes les classes
COCO refait tout ça pour 10 seuils d'IoU (0.50 à 0.95) et fait la moyenne.
"""
import numpy as np


def iou(a, b):
    """Intersection over Union of two boxes [x1, y1, x2, y2]."""
    # Le rectangle d'intersection commence au plus grand des coins haut-gauche
    # et finit au plus petit des coins bas-droite.
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    # max(0, ...) : si les boîtes ne se touchent pas, l'intersection vaut 0.
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    # Union = A + B - intersection (sinon on compterait l'intersection 2 fois).
    union = area_a + area_b - inter
    # 1.0 = boîtes identiques ; 0.0 = aucun chevauchement.
    return inter / union if union > 0 else 0.0


def match(preds, gts, cls, iou_thr):
    """Label each prediction of class `cls` as TP (1) or FP (0).

    Returns (scores, tp_flags, number_of_gt) sorted by decreasing score.
    """
    # On évalue une classe à la fois, en commençant par les prédictions les plus
    # sûres (score le plus élevé) : elles ont la priorité pour être appariées.
    preds = sorted((p for p in preds if p["cls"] == cls), key=lambda p: -p["score"])
    gt_by_img = {}
    for g in gts:
        if g["cls"] == cls:
            gt_by_img.setdefault(g["image"], []).append(g["box"])
    # used[img][j] = True quand l'objet réel j a déjà été trouvé.
    used = {img: [False] * len(b) for img, b in gt_by_img.items()}
    # n_gt = nombre total d'objets réels = dénominateur du rappel.
    n_gt = sum(len(b) for b in gt_by_img.values())

    scores, tps = [], []
    for p in preds:
        # Chercher l'objet réel (de la même image) qui chevauche le plus.
        best, best_j = 0.0, -1
        for j, g in enumerate(gt_by_img.get(p["image"], [])):
            o = iou(p["box"], g)
            if o > best:
                best, best_j = o, j
        # TP seulement si : chevauchement suffisant ET objet pas encore pris.
        # Une 2e boîte sur le même objet (doublon) devient donc un FP.
        is_tp = best >= iou_thr and best_j >= 0 and not used[p["image"]][best_j]
        if is_tp:
            used[p["image"]][best_j] = True
        scores.append(p["score"])
        tps.append(1 if is_tp else 0)
    # Les objets réels jamais "used" sont les faux négatifs (FN = n_gt - TP).
    return np.array(scores), np.array(tps), n_gt


def pr_curve(tps, n_gt):
    """Cumulative precision and recall when lowering the score threshold."""
    # Les prédictions sont triées par score. Garder les k premières revient à
    # choisir un seuil de confiance. Le cumsum donne TP et FP pour chaque k.
    #   précision = TP / (TP + FP)  -> parmi ce que je détecte, combien est juste ?
    #   rappel    = TP / n_gt       -> parmi les vrais objets, combien j'en trouve ?
    # Baisser le seuil fait monter le rappel, mais en général baisser la précision.
    tp_cum = np.cumsum(tps)
    fp_cum = np.cumsum(1 - tps)
    recall = tp_cum / max(n_gt, 1)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
    return precision, recall


def average_precision(precision, recall, method="coco"):
    """Area under the interpolated PR curve.

    method="voc"  : all-point interpolation (Pascal VOC 2010+)
    method="coco" : 101 recall points 0, 0.01, ..., 1 (MS COCO)
    """
    # L'AP résume toute la courbe PR en un seul nombre entre 0 et 1.
    # Avantage : on n'a pas besoin de choisir un seuil de confiance.
    if len(precision) == 0:
        return 0.0
    # On ajoute les points extrêmes (rappel 0 et 1) pour fermer la courbe.
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    # precision envelope: p_interp(r) = max_{r' >= r} p(r')
    # Interpolation : la courbe brute fait des "dents de scie". On remplace
    # chaque précision par la meilleure précision atteignable à un rappel
    # supérieur ou égal. La courbe devient décroissante, l'aire est plus stable.
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    if method == "voc":
        # VOC : somme exacte des rectangles là où le rappel change.
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    # COCO : on lit la précision à 101 niveaux de rappel fixes et on fait la moyenne.
    rec_points = np.linspace(0, 1, 101)
    out = []
    for r in rec_points:
        k = np.searchsorted(mrec, r, side="left")
        out.append(mpre[k] if k < len(mpre) else 0.0)
    return float(np.mean(out))


def mean_ap(preds, gts, iou_thr=0.5, method="coco"):
    """mAP over classes at one IoU threshold. Returns (mAP, {cls: AP})."""
    # Une AP par classe, puis la moyenne : chaque classe compte autant, même
    # une classe rare. C'est le "m" (mean) de mAP.
    classes = sorted({g["cls"] for g in gts})
    aps = {}
    for c in classes:
        _, tps, n_gt = match(preds, gts, c, iou_thr)
        p, r = pr_curve(tps, n_gt)
        aps[c] = average_precision(p, r, method)
    return float(np.mean(list(aps.values()))), aps


def coco_map(preds, gts):
    """COCO primary metric: mAP averaged over IoU thresholds 0.50:0.05:0.95."""
    # Pascal VOC n'utilise que le seuil 0.5 : une boîte approximative suffit.
    # COCO fait la moyenne sur 0.50, 0.55, ..., 0.95 : il récompense aussi une
    # localisation PRÉCISE. Le score COCO est donc toujours plus bas que mAP@0.5.
    thrs = np.arange(0.5, 0.96, 0.05)
    return float(np.mean([mean_ap(preds, gts, t)[0] for t in thrs]))


def make_synthetic_dataset(n_images=50, classes=("person", "car", "dog"), seed=0):
    """Ground truth + noisy detector output (localisation noise, misses, false alarms)."""
    # Pas besoin d'un vrai réseau (YOLO, Faster R-CNN...) pour comprendre les
    # métriques : on simule un détecteur imparfait à partir de la vérité terrain.
    # seed fixe -> mêmes résultats à chaque exécution (reproductibilité).
    rng = np.random.default_rng(seed)
    gts, preds = [], []
    quality = {"person": 0.04, "car": 0.07, "dog": 0.12}  # localisation noise per class
    for img in range(n_images):
        for _ in range(rng.integers(1, 5)):
            c = classes[rng.integers(len(classes))]
            x, y = rng.uniform(0, 400, 2)
            w, h = rng.uniform(40, 200, 2)
            box = [x, y, x + w, y + h]
            gts.append({"image": img, "cls": c, "box": box})
            if rng.random() < 0.88:  # detected (12% missed)
                # Boîte détectée = vraie boîte + bruit gaussien sur chaque bord.
                # Plus le bruit est grand, plus l'IoU baisse.
                s = quality[c]
                noisy = [box[0] + rng.normal(0, s * w), box[1] + rng.normal(0, s * h),
                         box[2] + rng.normal(0, s * w), box[3] + rng.normal(0, s * h)]
                preds.append({"image": img, "cls": c, "box": noisy,
                              "score": float(np.clip(rng.normal(0.75, 0.15), 0.05, 1))})
            # sinon : objet raté -> faux négatif
        for _ in range(rng.poisson(1.0)):  # false positives (background)
            # Fausses alarmes : boîtes au hasard, avec des scores plutôt bas.
            c = classes[rng.integers(len(classes))]
            x, y = rng.uniform(0, 400, 2)
            w, h = rng.uniform(40, 200, 2)
            preds.append({"image": img, "cls": c, "box": [x, y, x + w, y + h],
                          "score": float(np.clip(rng.normal(0.45, 0.2), 0.01, 1))})
    return preds, gts


def demo(fig_dir=None):
    print("=== Object detection metrics ===")
    # Les deux boîtes de la diapo IoU : intersection 50x70, chaque boîte 100x100.
    a, b = [50, 50, 150, 150], [100, 80, 200, 180]
    print(f"IoU example: {iou(a, b):.3f}")

    preds, gts = make_synthetic_dataset()
    print(f"{len(gts)} ground-truth boxes, {len(preds)} predictions")
    m50, aps50 = mean_ap(preds, gts, 0.5)
    m75, _ = mean_ap(preds, gts, 0.75)
    m50_voc, _ = mean_ap(preds, gts, 0.5, method="voc")
    mcoco = coco_map(preds, gts)
    for c, ap in aps50.items():
        print(f"  AP@0.5 [{c:6s}] = {ap:.3f}")
    print(f"mAP@0.5 (COCO 101-pt) : {m50:.3f}")
    print(f"mAP@0.5 (VOC all-pt)  : {m50_voc:.3f}")
    print(f"mAP@0.75              : {m75:.3f}")
    print(f"mAP@[0.5:0.95] (COCO) : {mcoco:.3f}")

    if fig_dir:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        # Figure 1 : une courbe précision-rappel par classe.
        fig, ax = plt.subplots(figsize=(6, 4.2), dpi=160)
        for c, col in zip(sorted(aps50), ["#1f6feb", "#d97706", "#16a34a"]):
            _, tps, n_gt = match(preds, gts, c, 0.5)
            p, r = pr_curve(tps, n_gt)
            ax.plot(r, p, color=col, lw=2, label=f"{c} (AP={aps50[c]:.2f})")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_title("Precision-Recall curves @ IoU 0.5")
        ax.grid(alpha=0.3)
        ax.legend(loc="lower left")
        fig.tight_layout()
        fig.savefig(f"{fig_dir}/pr_curves.png")
        plt.close(fig)

        # Figure 2 : la mAP chute quand on exige une localisation plus précise.
        thrs = np.arange(0.5, 0.96, 0.05)
        maps = [mean_ap(preds, gts, t)[0] for t in thrs]
        fig, ax = plt.subplots(figsize=(6, 4.2), dpi=160)
        ax.bar([f"{t:.2f}" for t in thrs], maps, color="#1f6feb")
        ax.axhline(mcoco, color="#d97706", ls="--", label=f"mAP@[.5:.95] = {mcoco:.2f}")
        ax.set_xlabel("IoU threshold")
        ax.set_ylabel("mAP")
        ax.set_title("mAP drops as localisation gets stricter")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"{fig_dir}/map_vs_iou.png")
        plt.close(fig)
    return {"iou_example": iou(a, b), "map50": m50, "map50_voc": m50_voc,
            "map75": m75, "map_coco": mcoco, "aps50": aps50,
            "n_gt": len(gts), "n_pred": len(preds)}


if __name__ == "__main__":
    import os
    os.makedirs("figures", exist_ok=True)
    demo("figures")

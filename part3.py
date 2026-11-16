"""Part 3 — Segmentation and dimensionality reduction."""
import csv
import math
import os
import numpy as np
from helpers import load_records


def run_kmeans(X_std, k):
    n = X_std.shape[0]
    step = n // k
    init_indices = [i * step for i in range(k)]
    centroids = X_std[init_indices].copy()
    labels = np.full(n, -1, dtype=np.int64)
    passes = 0
    while True:
        passes += 1
        # Squared Euclidean distance to each centroid: (n, k)
        dists = np.sum((X_std[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(dists, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for c in range(k):
            mask = labels == c
            if np.any(mask):
                centroids[c] = np.mean(X_std[mask], axis=0)
    wcss = float(np.sum((X_std - centroids[labels]) ** 2))
    return labels, centroids, wcss, passes


def hierarchical_clustering(dist_matrix, names, linkage):
    clusters = [[i] for i in range(len(names))]
    merges = []
    for merge_no in range(1, len(names)):
        best_dist = None
        best_pair = None
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                c1, c2 = clusters[i], clusters[j]
                pair_dists = [dist_matrix[a][b] for a in c1 for b in c2]
                if linkage == "single":
                    d = min(pair_dists)
                elif linkage == "complete":
                    d = max(pair_dists)
                elif linkage == "average":
                    d = sum(pair_dists) / len(pair_dists)
                else:
                    raise ValueError(f"Unknown linkage: {linkage}")
                if best_dist is None or d < best_dist:
                    best_dist = d
                    best_pair = (i, j)
        i, j = best_pair
        c_a, c_b = clusters[i], clusters[j]
        if min(c_a) > min(c_b):
            c_a, c_b = c_b, c_a
        name_a = "+".join(names[idx] for idx in sorted(c_a))
        name_b = "+".join(names[idx] for idx in sorted(c_b))
        merges.append((linkage, merge_no, name_a, name_b, best_dist))
        merged = sorted(c_a + c_b)
        clusters = [clusters[idx] for idx in range(len(clusters)) if idx not in (i, j)]
        clusters.append(merged)

    heights = [m[4] for m in merges]
    gaps = [heights[idx + 1] - heights[idx] for idx in range(len(heights) - 1)]
    cut_idx = int(np.argmax(gaps))
    merges_below_cut = cut_idx + 1
    clusters_at_cut = len(names) - merges_below_cut
    return merges, clusters_at_cut


def main():
    _, X, y, value, _ = load_records()
    X_cont = X[:, :5]
    n = X_cont.shape[0]

    # Standardise X1..X5 over all 800 rows (sample SD ddof=1)
    means = np.mean(X_cont, axis=0)
    sds = np.sqrt(np.sum((X_cont - means) ** 2, axis=0) / (n - 1))
    X_std = (X_cont - means) / sds

    # Task 3.1: K-means for K = 2..6
    wcss_by_k = {}
    labels_k3 = None
    for k in range(2, 7):
        labels, _, wcss, _ = run_kmeans(X_std, k)
        wcss_by_k[k] = wcss
        if k == 3:
            labels_k3 = labels

    # Task 3.2: Elbow K
    elbow_k = None
    for k in (3, 4, 5):
        if (wcss_by_k[k] - wcss_by_k[k + 1]) < 0.5 * (wcss_by_k[k - 1] - wcss_by_k[k]):
            elbow_k = k
            break

    # Task 3.3: Profile K=3 segments ordered by ascending mean of X3 (col index 2)
    cluster_x3_means = []
    for c in range(3):
        mask = labels_k3 == c
        cluster_x3_means.append((float(np.mean(X_cont[mask, 2])), c))
    cluster_x3_means.sort(key=lambda t: t[0])

    seg_names = ["A", "B", "C"]
    seg_map = {}  # original cluster id -> segment name
    seg_masks = {}
    os.makedirs("outputs", exist_ok=True)
    with open(os.path.join("outputs", "part3_segments.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "segment", "size", "mean_X1", "mean_X2", "mean_X3",
            "mean_X4", "mean_X5", "outcome_rate", "mean_value",
        ])
        for seg_name, (_, c_id) in zip(seg_names, cluster_x3_means):
            mask = labels_k3 == c_id
            seg_map[c_id] = seg_name
            seg_masks[seg_name] = mask
            size = int(np.sum(mask))
            m_x = np.mean(X_cont[mask], axis=0)
            out_rate = float(np.mean(y[mask]))
            m_val = float(np.mean(value[mask]))
            writer.writerow([
                seg_name,
                size,
                f"{m_x[0]:.4f}",
                f"{m_x[1]:.4f}",
                f"{m_x[2]:.4f}",
                f"{m_x[3]:.4f}",
                f"{m_x[4]:.4f}",
                f"{out_rate:.4f}",
                f"{m_val:.4f}",
            ])

    # Task 3.4: Hierarchical clustering on supplied 6x6 table
    cohort_names = ["E1", "E2", "E3", "E4", "E5", "E6"]
    dist_table = [
        [0.0, 7.9, 7.4, 2.9, 2.9, 6.1],
        [7.9, 0.0, 1.4, 5.9, 5.3, 2.0],
        [7.4, 1.4, 0.0, 5.8, 4.6, 1.3],
        [2.9, 5.9, 5.8, 0.0, 2.8, 4.5],
        [2.9, 5.3, 4.6, 2.8, 0.0, 3.4],
        [6.1, 2.0, 1.3, 4.5, 3.4, 0.0],
    ]
    all_merges = []
    cuts = {}
    for link in ("single", "complete", "average"):
        merges, cut_n = hierarchical_clustering(dist_table, cohort_names, link)
        all_merges.extend(merges)
        cuts[link] = cut_n

    with open(os.path.join("outputs", "part3_hierarchical.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["linkage", "merge_no", "cluster_a", "cluster_b", "height"])
        for link, mno, ca, cb, h in all_merges:
            writer.writerow([link, mno, ca, cb, f"{h:.4f}"])

    # Task 3.5: PCA by the seven steps
    # Step 1 & 2: mean vector & centre
    mean_vec = np.mean(X_std, axis=0)
    X_c = X_std - mean_vec
    # Step 3: covariance matrix S = (1/(n-1)) X^T X
    S = (X_c.T @ X_c) / (n - 1)
    # Step 4: eigenvalues and eigenvectors
    evals, evecs = np.linalg.eigh(S)
    # Step 5: sort largest first
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]
    # Sign convention: entry of largest abs value is positive
    for j in range(evecs.shape[1]):
        max_idx = int(np.argmax(np.abs(evecs[:, j])))
        if evecs[max_idx, j] < 0:
            evecs[:, j] = -evecs[:, j]
    # Step 6 & 7: keep top 2 and project Z = X W
    W = evecs[:, :2]
    Z = X_c @ W

    eval_sum = float(np.sum(evals))
    exp_ratios = evals / eval_sum
    cum_ratios = np.cumsum(exp_ratios)
    comp_80 = int(np.argmax(cum_ratios >= 0.8) + 1)

    lines = [f"wcss_k{k} = {wcss_by_k[k]:.4f}" for k in range(2, 7)]
    lines.append(f"elbow_k = {elbow_k}")
    for seg_name in seg_names:
        lines.append(f"size_segment_{seg_name} = {int(np.sum(seg_masks[seg_name]))}")
    lines.append(f"single_clusters_at_cut = {cuts['single']}")
    lines.append(f"complete_clusters_at_cut = {cuts['complete']}")
    lines.append(f"average_clusters_at_cut = {cuts['average']}")
    lines.append(f"eigenvalue_sum = {eval_sum:.4f}")
    for i in range(5):
        lines.append(f"eigenvalue_{i + 1} = {evals[i]:.4f}")
    for i in range(5):
        lines.append(f"explained_ratio_{i + 1} = {exp_ratios[i]:.4f}")
    for i in range(5):
        lines.append(f"cumulative_ratio_{i + 1} = {cum_ratios[i]:.4f}")
    lines.append(f"components_for_80pct = {comp_80}")
    for j in range(5):
        lines.append(f"pc1_loading_X{j + 1} = {W[j, 0]:.4f}")
    for j in range(5):
        lines.append(f"pc2_loading_X{j + 1} = {W[j, 1]:.4f}")
    for r in range(3):
        lines.append(f"row{r + 1}_pc1 = {Z[r, 0]:.4f}")
        lines.append(f"row{r + 1}_pc2 = {Z[r, 1]:.4f}")
    for seg_name in seg_names:
        m_pc1 = float(np.mean(Z[seg_masks[seg_name], 0]))
        lines.append(f"mean_pc1_segment_{seg_name} = {m_pc1:.4f}")

    with open(os.path.join("outputs", "part3_results.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(
        "Part 3 complete. Key checkpoints: wcss_k2 =",
        f"{wcss_by_k[2]:.4f}",
        "eigenvalue_sum =",
        f"{eval_sum:.4f}",
    )


if __name__ == "__main__":
    main()

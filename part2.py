"""Part 2 — Classification and evaluation."""
import csv
import math
import os
import numpy as np
from helpers import (
    bce_loss,
    compute_metrics,
    fit_logistic,
    knn_predict_proba,
    load_records,
    predict_logistic_proba,
    standardise_train_test,
    stratified_split,
)


def main():
    _, X, y, _, _ = load_records()

    # Task 2.1: Stratified 70/30 split & standardisation
    X_train_raw, y_train, X_test_raw, y_test, _, _ = stratified_split(X, y)
    X_train, X_test, _, _ = standardise_train_test(X_train_raw, X_test_raw)
    train_rows = len(y_train)
    train_positives = int(np.sum(y_train))
    test_rows = len(y_test)
    test_positives = int(np.sum(y_test))

    # Task 2.2: Logistic regression from scratch (1000 updates, lr=0.1)
    w, b, losses = fit_logistic(
        X_train, y_train, lr=0.1, epochs=1000, checkpoints=(0, 1, 10, 100, 1000)
    )
    p_train = predict_logistic_proba(X_train, w, b)
    p_test = predict_logistic_proba(X_test, w, b)
    test_bce = bce_loss(y_test, p_test)

    # Task 2.3: Learning-rate effect (200 updates)
    lr_losses = {}
    for eta in (0.01, 0.1, 1.0):
        _, _, l_dict = fit_logistic(
            X_train, y_train, lr=eta, epochs=200, checkpoints=(200,)
        )
        lr_losses[eta] = l_dict[200]

    # Task 2.4: Odds ratios
    odds_ratios = [math.exp(float(wj)) for wj in w]

    # Task 2.5: Evaluate on test rows at tau = 0.5 and tau = 0.3
    m_05 = compute_metrics(y_test, (p_test >= 0.5).astype(np.int64))
    m_03 = compute_metrics(y_test, (p_test >= 0.3).astype(np.int64))
    blind_accuracy = float(np.sum(y_test == 0)) / test_rows

    # Task 2.6: Threshold sweep -> outputs/part2_thresholds.csv
    os.makedirs("outputs", exist_ok=True)
    with open(os.path.join("outputs", "part2_thresholds.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tau", "tp", "fp", "tn", "fn", "tpr", "fpr"])
        for k in range(1, 10):
            tau = k / 10.0
            m_t = compute_metrics(y_test, (p_test >= tau).astype(np.int64))
            writer.writerow([
                f"{tau:.4f}",
                m_t["tp"],
                m_t["fp"],
                m_t["tn"],
                m_t["fn"],
                f"{m_t['tpr']:.4f}",
                f"{m_t['fpr']:.4f}",
            ])

    # Task 2.7: KNN with 5-fold CV
    folds = np.arange(train_rows) % 5
    cv_results = []
    best_k = None
    best_cv_f1 = -1.0
    for k in range(1, 22, 2):
        fold_f1s = []
        for f_id in range(5):
            val_mask = folds == f_id
            ref_mask = ~val_mask
            _, preds_val = knn_predict_proba(
                X_train[ref_mask], y_train[ref_mask], X_train[val_mask], k
            )
            m_val = compute_metrics(y_train[val_mask], preds_val)
            fold_f1s.append(m_val["f1"])
        mean_f1 = float(np.mean(fold_f1s))
        cv_results.append((k, mean_f1))
        if mean_f1 > best_cv_f1:
            best_cv_f1 = mean_f1
            best_k = k

    with open(os.path.join("outputs", "part2_knn_cv.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["K", "mean_cv_f1"])
        for k, mf1 in cv_results:
            writer.writerow([k, f"{mf1:.4f}"])

    _, knn_test_preds = knn_predict_proba(X_train, y_train, X_test, best_k)
    m_knn = compute_metrics(y_test, knn_test_preds)

    # Task 2.8: Cost-based threshold selection on training rows
    best_cost_tau = None
    best_train_cost = None
    for k in range(1, 20):
        tau = k / 20.0
        m_tr = compute_metrics(y_train, (p_train >= tau).astype(np.int64))
        cost = 50 * m_tr["fp"] + 400 * m_tr["fn"]
        if best_train_cost is None or cost < best_train_cost:
            best_train_cost = cost
            best_cost_tau = tau

    m_test_cost = compute_metrics(y_test, (p_test >= best_cost_tau).astype(np.int64))
    cost_test_at_tau = 50 * m_test_cost["fp"] + 400 * m_test_cost["fn"]
    cost_test_at_05 = 50 * m_05["fp"] + 400 * m_05["fn"]
    cost_test_flag_nobody = 400 * test_positives

    lines = [
        f"train_rows = {train_rows}",
        f"train_positives = {train_positives}",
        f"test_rows = {test_rows}",
        f"test_positives = {test_positives}",
        f"loss_epoch_0 = {losses[0]:.4f}",
        f"loss_epoch_1 = {losses[1]:.4f}",
        f"loss_epoch_10 = {losses[10]:.4f}",
        f"loss_epoch_100 = {losses[100]:.4f}",
        f"loss_epoch_1000 = {losses[1000]:.4f}",
    ]
    for j in range(7):
        lines.append(f"weight_X{j + 1} = {w[j]:.4f}")
    lines.append(f"bias = {b:.4f}")
    lines.append(f"test_bce = {test_bce:.4f}")
    lines.append(f"loss_epoch_200_eta_0.01 = {lr_losses[0.01]:.4f}")
    lines.append(f"loss_epoch_200_eta_0.1 = {lr_losses[0.1]:.4f}")
    lines.append(f"loss_epoch_200_eta_1.0 = {lr_losses[1.0]:.4f}")
    for j in range(7):
        lines.append(f"odds_ratio_X{j + 1} = {odds_ratios[j]:.4f}")
    for prefix, m in (("tau_0.5", m_05), ("tau_0.3", m_03)):
        lines.extend([
            f"{prefix}_tp = {m['tp']}",
            f"{prefix}_fp = {m['fp']}",
            f"{prefix}_tn = {m['tn']}",
            f"{prefix}_fn = {m['fn']}",
            f"{prefix}_accuracy = {m['accuracy']:.4f}",
            f"{prefix}_precision = {m['precision']:.4f}",
            f"{prefix}_recall = {m['recall']:.4f}",
            f"{prefix}_f1 = {m['f1']:.4f}",
            f"{prefix}_specificity = {m['specificity']:.4f}",
        ])
    lines.append(f"blind_accuracy = {blind_accuracy:.4f}")
    lines.append(f"knn_best_k = {best_k}")
    lines.extend([
        f"knn_tp = {m_knn['tp']}",
        f"knn_fp = {m_knn['fp']}",
        f"knn_tn = {m_knn['tn']}",
        f"knn_fn = {m_knn['fn']}",
        f"knn_accuracy = {m_knn['accuracy']:.4f}",
        f"knn_precision = {m_knn['precision']:.4f}",
        f"knn_recall = {m_knn['recall']:.4f}",
        f"knn_f1 = {m_knn['f1']:.4f}",
        f"cost_tau = {best_cost_tau:.4f}",
        f"cost_train_at_tau = {best_train_cost}",
        f"cost_test_at_tau = {cost_test_at_tau}",
        f"cost_test_at_0.5 = {cost_test_at_05}",
        f"cost_test_flag_nobody = {cost_test_flag_nobody}",
    ])

    with open(os.path.join("outputs", "part2_results.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(
        "Part 2 complete. Key checkpoints: train_rows =",
        train_rows,
        "loss_epoch_0 =",
        f"{losses[0]:.4f}",
    )


if __name__ == "__main__":
    main()

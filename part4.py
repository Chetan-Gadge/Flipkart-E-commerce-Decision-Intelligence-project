"""Part 4 — Model-choice and architecture memo."""
import csv
import math
import os
import numpy as np
from helpers import (
    compute_metrics,
    fit_logistic,
    knn_predict_proba,
    load_records,
    predict_logistic_proba,
    standardise_train_test,
    stratified_split,
)


def binary_entropy(pos, n):
    if n == 0 or pos == 0 or pos == n:
        return 0.0
    p1 = pos / n
    p0 = 1.0 - p1
    return -(p1 * math.log2(p1) + p0 * math.log2(p0))


def binary_gini(pos, n):
    if n == 0:
        return 0.0
    p1 = pos / n
    p0 = 1.0 - p1
    return 1.0 - (p1 * p1 + p0 * p0)


def main():
    _, X, y, _, _ = load_records()
    X_train_raw, y_train, X_test_raw, y_test, _, _ = stratified_split(X, y)
    X_train_std, X_test_std, _, _ = standardise_train_test(X_train_raw, X_test_raw)
    n_train = len(y_train)
    pos_train = int(np.sum(y_train))

    # Task 4.1: Candidate splits on 560 training rows (original units)
    root_entropy = binary_entropy(pos_train, n_train)
    root_gini = binary_gini(pos_train, n_train)

    splits_def = [
        ("S1", X_train_raw[:, 5] == 1.0, X_test_raw[:, 5] == 1.0),
        ("S2", X_train_raw[:, 6] == 1.0, X_test_raw[:, 6] == 1.0),
        ("S3", X_train_raw[:, 2] > 0.5, X_test_raw[:, 2] > 0.5),
        ("S4", X_train_raw[:, 0] <= 24.0, X_test_raw[:, 0] <= 24.0),
    ]

    os.makedirs("outputs", exist_ok=True)
    split_rows = []
    best_ig_name, best_ig_val = None, -1.0
    best_gg_name, best_gg_val = None, -1.0
    best_stump_info = None

    for name, tr_mask, te_mask in splits_def:
        n_yes = int(np.sum(tr_mask))
        pos_yes = int(np.sum(y_train[tr_mask]))
        n_no = n_train - n_yes
        pos_no = pos_train - pos_yes

        child_ent = (n_yes / n_train) * binary_entropy(pos_yes, n_yes) + (
            n_no / n_train
        ) * binary_entropy(pos_no, n_no)
        child_gini = (n_yes / n_train) * binary_gini(pos_yes, n_yes) + (
            n_no / n_train
        ) * binary_gini(pos_no, n_no)
        ig = root_entropy - child_ent
        gg = root_gini - child_gini
        split_rows.append((name, n_yes, pos_yes, n_no, pos_no, ig, gg))
        if ig > best_ig_val:
            best_ig_val = ig
            best_ig_name = name
            best_stump_info = (pos_yes / n_yes, pos_no / n_no, te_mask)
        if gg > best_gg_val:
            best_gg_val = gg
            best_gg_name = name

    with open(os.path.join("outputs", "part4_splits.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "split", "n_yes", "positives_yes", "n_no", "positives_no",
            "info_gain", "gini_gain",
        ])
        for name, ny, py, nn, pn, ig, gg in split_rows:
            writer.writerow([name, ny, py, nn, pn, f"{ig:.4f}", f"{gg:.4f}"])

    # Task 4.2: Stump and voting models on 240 test rows
    stump_rate_yes, stump_rate_no, stump_test_mask = best_stump_info
    p_stump = np.where(stump_test_mask, stump_rate_yes, stump_rate_no)
    pred_stump = (p_stump >= 0.3).astype(np.int64)

    # Logistic model from Task 2.2
    w, b, _ = fit_logistic(X_train_std, y_train, lr=0.1, epochs=1000, checkpoints=())
    p_log = predict_logistic_proba(X_test_std, w, b)
    pred_log = (p_log >= 0.3).astype(np.int64)

    # KNN model from Task 2.7 (select best K via 5-fold CV)
    folds = np.arange(n_train) % 5
    best_k, best_cv_f1 = None, -1.0
    for k in range(1, 22, 2):
        fold_f1s = []
        for f_id in range(5):
            val_mask = folds == f_id
            ref_mask = ~val_mask
            _, preds_val = knn_predict_proba(
                X_train_std[ref_mask], y_train[ref_mask], X_train_std[val_mask], k
            )
            fold_f1s.append(compute_metrics(y_train[val_mask], preds_val)["f1"])
        mean_f1 = float(np.mean(fold_f1s))
        if mean_f1 > best_cv_f1:
            best_cv_f1 = mean_f1
            best_k = k

    p_knn, pred_knn = knn_predict_proba(X_train_std, y_train, X_test_std, best_k)

    pred_hard = ((pred_log + pred_knn + pred_stump) >= 2).astype(np.int64)
    p_soft = (p_log + p_knn + p_stump) / 3.0
    pred_soft = (p_soft >= 0.3).astype(np.int64)

    voting_models = [
        ("logistic", pred_log),
        ("knn", pred_knn),
        ("stump", pred_stump),
        ("hard_vote", pred_hard),
        ("soft_vote", pred_soft),
    ]
    with open(os.path.join("outputs", "part4_voting.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "tp", "fp", "tn", "fn", "precision", "recall", "f1"])
        for mname, preds in voting_models:
            m = compute_metrics(y_test, preds)
            writer.writerow([
                mname, m["tp"], m["fp"], m["tn"], m["fn"],
                f"{m['precision']:.4f}", f"{m['recall']:.4f}", f"{m['f1']:.4f}",
            ])

    # Task 4.3: Two rounds of AdaBoost by hand
    y_ab = np.array([1, 1, 1, -1, -1, -1, 1, -1], dtype=np.float64)
    h1 = np.array([1, 1, -1, -1, -1, -1, -1, -1], dtype=np.float64)
    h2 = np.array([-1, 1, 1, -1, 1, -1, 1, -1], dtype=np.float64)

    w1 = np.full(8, 1.0 / 8.0, dtype=np.float64)
    err1 = float(np.sum(w1[h1 != y_ab]))
    alpha1 = 0.5 * math.log((1.0 - err1) / err1)
    w1_unnorm = w1 * np.exp(-alpha1 * y_ab * h1)
    w1_after = w1_unnorm / np.sum(w1_unnorm)

    w2 = w1_after.copy()
    err2 = float(np.sum(w2[h2 != y_ab]))
    alpha2 = 0.5 * math.log((1.0 - err2) / err2)
    w2_unnorm = w2 * np.exp(-alpha2 * y_ab * h2)
    w2_after = w2_unnorm / np.sum(w2_unnorm)

    ens_pred = np.sign(alpha1 * h1 + alpha2 * h2)
    ens_acc = float(np.mean(ens_pred == y_ab))

    with open(os.path.join("outputs", "part4_adaboost.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "point", "weight_before", "weight_unnormalised", "weight_after"])
        for pt in range(8):
            writer.writerow([1, pt + 1, f"{w1[pt]:.4f}", f"{w1_unnorm[pt]:.4f}", f"{w1_after[pt]:.4f}"])
        for pt in range(8):
            writer.writerow([2, pt + 1, f"{w2[pt]:.4f}", f"{w2_unnorm[pt]:.4f}", f"{w2_after[pt]:.4f}"])

    # Task 4.4: Out-of-bag share
    oob_fraction = (1.0 - 1.0 / n_train) ** n_train

    # Task 4.5: Neural network parameter counts
    mlp_hidden1 = 784 * 16 + 16
    mlp_hidden2 = 16 * 16 + 16
    mlp_output = 16 * 10 + 10
    mlp_total = mlp_hidden1 + mlp_hidden2 + mlp_output

    cnn_conv1 = (5 * 5 * 1) * 6 + 6
    cnn_conv2 = (5 * 5 * 6) * 16 + 16
    cnn_dense1 = (5 * 5 * 16) * 32 + 32
    cnn_dense2 = 32 * 10 + 10
    cnn_total = cnn_conv1 + cnn_conv2 + cnn_dense1 + cnn_dense2

    with open(os.path.join("outputs", "part4_architecture.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "parameters"])
        for lname, lparams in [
            ("mlp_hidden1", mlp_hidden1),
            ("mlp_hidden2", mlp_hidden2),
            ("mlp_output", mlp_output),
            ("cnn_conv1", cnn_conv1),
            ("cnn_conv2", cnn_conv2),
            ("cnn_dense1", cnn_dense1),
            ("cnn_dense2", cnn_dense2),
        ]:
            writer.writerow([lname, lparams])

    fc_layer_params = (128 * 128 * 3) * 1024 + 1024
    conv_layer_params = (3 * 3 * 3) * 64 + 64
    rnn_total = (128 * 64) + (128 * 128) + 128 + (16 * 128) + 16

    lines = [
        f"root_entropy = {root_entropy:.4f}",
        f"root_gini = {root_gini:.4f}",
        f"best_split_by_info_gain = {best_ig_name}",
        f"best_split_by_gini_gain = {best_gg_name}",
        f"stump_rate_yes = {stump_rate_yes:.4f}",
        f"stump_rate_no = {stump_rate_no:.4f}",
        f"round1_error = {err1:.4f}",
        f"round1_alpha = {alpha1:.4f}",
        f"round2_error = {err2:.4f}",
        f"round2_alpha = {alpha2:.4f}",
        f"ensemble_accuracy = {ens_acc:.4f}",
        f"oob_fraction = {oob_fraction:.4f}",
        f"mlp_total = {mlp_total}",
        f"cnn_total = {cnn_total}",
        f"fc_layer_params = {fc_layer_params}",
        f"conv_layer_params = {conv_layer_params}",
        f"rnn_total = {rnn_total}",
    ]

    with open(os.path.join("outputs", "part4_results.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(
        "Part 4 complete. Key checkpoints: root_entropy =",
        f"{root_entropy:.4f}",
        "round1_error =",
        f"{err1:.4f}",
    )


if __name__ == "__main__":
    main()

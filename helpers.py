"""Shared data-loading, splitting, standardisation, model, and metric helpers."""
import csv
import math
import os
import numpy as np


def load_records(path=os.path.join("data", "records.csv")):
    """Load data/records.csv into numpy arrays."""
    ids = []
    rows_x = []
    outcomes = []
    values = []
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        for r in reader:
            ids.append(r[0])
            rows_x.append([float(v) for v in r[1:8]])
            outcomes.append(int(r[8]))
            values.append(float(r[9]))
    return (
        ids,
        np.array(rows_x, dtype=np.float64),
        np.array(outcomes, dtype=np.int64),
        np.array(values, dtype=np.float64),
        header,
    )


def stratified_split(X, y):
    """Stratified 70/30 split preserving file order within train and test."""
    idx0 = [i for i, val in enumerate(y) if val == 0]
    idx1 = [i for i, val in enumerate(y) if val == 1]
    n_train0 = (7 * len(idx0)) // 10
    n_train1 = (7 * len(idx1)) // 10
    train_idx = sorted(idx0[:n_train0] + idx1[:n_train1])
    test_idx = sorted(idx0[n_train0:] + idx1[n_train1:])
    return (
        X[train_idx],
        y[train_idx],
        X[test_idx],
        y[test_idx],
        train_idx,
        test_idx,
    )


def standardise_train_test(X_train, X_test):
    """Standardise X_train and X_test using training mean and sample SD (ddof=1)."""
    mean = np.mean(X_train, axis=0)
    sd = np.sqrt(np.sum((X_train - mean) ** 2, axis=0) / (X_train.shape[0] - 1))
    return (X_train - mean) / sd, (X_test - mean) / sd, mean, sd


def sigmoid(z):
    z_clipped = np.clip(z, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-z_clipped))


def bce_loss(y_true, y_prob):
    eps = 1e-15
    p = np.clip(y_prob, eps, 1.0 - eps)
    return -float(np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))


def fit_logistic(X_train, y_train, lr=0.1, epochs=1000, checkpoints=(0, 1, 10, 100, 1000)):
    """Fit logistic regression from zero initialisation using batch gradient descent."""
    n, d = X_train.shape
    w = np.zeros(d, dtype=np.float64)
    b = 0.0
    losses = {}
    if 0 in checkpoints:
        p0 = sigmoid(X_train @ w + b)
        losses[0] = bce_loss(y_train, p0)
    for ep in range(1, epochs + 1):
        p = sigmoid(X_train @ w + b)
        err = p - y_train
        dw = (X_train.T @ err) / n
        db = float(np.mean(err))
        w = w - lr * dw
        b = b - lr * db
        if ep in checkpoints:
            p_new = sigmoid(X_train @ w + b)
            losses[ep] = bce_loss(y_train, p_new)
    return w, b, losses


def predict_logistic_proba(X, w, b):
    return sigmoid(X @ w + b)


def compute_metrics(y_true, y_pred):
    """Compute confusion-matrix cells and classification metrics."""
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "tpr": recall,
        "fpr": fpr,
    }


def knn_predict_proba(X_ref, y_ref, X_query, k):
    """Predict positive class probability and label using Euclidean KNN with stable tie-breaking."""
    probs = np.zeros(X_query.shape[0], dtype=np.float64)
    preds = np.zeros(X_query.shape[0], dtype=np.int64)
    threshold_count = (k + 1) / 2.0
    for i in range(X_query.shape[0]):
        dists = np.sqrt(np.sum((X_ref - X_query[i]) ** 2, axis=1))
        order = np.argsort(dists, kind="stable")
        top_k = y_ref[order[:k]]
        pos_count = int(np.sum(top_k))
        probs[i] = pos_count / k
        preds[i] = 1 if pos_count >= threshold_count else 0
    return probs, preds

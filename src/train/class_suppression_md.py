#!/usr/bin/env python3
"""
A2 — Class suppression on match-disjoint bag2.

Mechanism: sklearn macro-F1 default averages over union(y_true, y_pred).
For classes where:
  Pr(class k in y_true on held-out match) is low
  AND Pr(F1_k > 0 if predicted) is low
suppressing those predictions (bias_k = -inf) drops them from macro denominator.

Approach:
1. Load match-disjoint bag2 OOF + calibrated bias
2. For each class k: try setting bias[k] = -inf, recompute F1
3. Greedy: keep changes that improve OOF F1
4. Apply final bias to test predictions, generate new submission
"""
import os
import pathlib
os.chdir(pathlib.Path(__file__).resolve().parents[2])

import sys
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, "src/train")
from train_v5z import prepare_samples, train_df, N_FOLDS


def build_md_oof_reindex(seed):
    train_samples = prepare_samples(train_df, is_train=True, augment=True)
    uid2idx = {}
    for i, s in enumerate(train_samples):
        uid2idx.setdefault(s["uid"], []).append(i)
    uid_last_a = {s["uid"]: s["target_action"] for s in train_samples}
    uid_to_match = train_df.groupby("rally_uid")["match"].first().to_dict()
    uid_list = sorted(uid_last_a.keys())
    uid_arr = np.array([uid_last_a[u] for u in uid_list])
    match_arr = np.array([uid_to_match[u] for u in uid_list])
    sgkf = StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=seed)
    sample_idx_to_oof_idx = np.empty(len(train_samples), dtype=np.int64)
    cur = 0
    for _, (_, va_uidx) in enumerate(sgkf.split(uid_list, uid_arr, groups=match_arr)):
        va_uids = set(np.array(uid_list)[va_uidx])
        for u in va_uids:
            for i in uid2idx.get(u, []):
                sample_idx_to_oof_idx[i] = cur
                cur += 1
    return sample_idx_to_oof_idx


def f1_with_bias(probs, bias, y_true):
    log_p = np.log(probs + 1e-12) + bias
    pred = log_p.argmax(1)
    return f1_score(y_true, pred, average="macro", zero_division=0), pred


def greedy_suppress(probs, bias_init, y_true, label):
    """Greedy: try suppressing each class, keep if F1 improves."""
    bias = bias_init.copy()
    n_class = probs.shape[1]
    base_f1, _ = f1_with_bias(probs, bias, y_true)
    print(f"  {label} baseline F1 = {base_f1:.4f}")

    suppressed = []
    while True:
        best_k = -1
        best_f1 = base_f1
        for k in range(n_class):
            if k in suppressed: continue
            test_bias = bias.copy()
            test_bias[k] = -1e9
            f1, _ = f1_with_bias(probs, test_bias, y_true)
            if f1 > best_f1 + 1e-7:
                best_f1 = f1
                best_k = k
        if best_k < 0: break
        bias[best_k] = -1e9
        suppressed.append(best_k)
        print(f"    suppress class {best_k}: F1 {base_f1:.4f} → {best_f1:.4f} (+{best_f1-base_f1:.4f})")
        base_f1 = best_f1

    return bias, base_f1, suppressed


def main():
    print("="*78)
    print("A2: Class suppression on match-disjoint bag2")
    print("="*78)

    # --- Load match-disjoint bag2 calibration params + OOF ---
    print("\nLoading match-disjoint bag2 artifacts...")
    p = np.load("artifacts/v5z_md_advcal_bag2_params.npz")
    a_alpha, p_alpha = float(p["a_alpha"]), float(p["p_alpha"])
    bias_a, bias_p = p["bias_a"].copy(), p["bias_p"].copy()
    print(f"  α_action={a_alpha:.2f}  α_point={p_alpha:.2f}")

    # Load OOF for both seeds, bag, get ensemble
    seeds = ["42", "1337"]
    lstm_a_l, lstm_p_l, lgb_a_l, lgb_p_l = [], [], [], []
    la_ref, lp_ref = None, None
    for seed in seeds:
        d = np.load(f"artifacts/v5z_md_full_s{seed}_oof.npz")
        s2o = build_md_oof_reindex(int(seed))
        lstm_a_l.append(d["lstm_a"][s2o]); lstm_p_l.append(d["lstm_p"][s2o])
        lgb_a_l.append(d["lgb_a"][s2o]);   lgb_p_l.append(d["lgb_p"][s2o])
        la_ref = d["la"][s2o]; lp_ref = d["lp"][s2o]

    lstm_a = np.mean(lstm_a_l, 0); lstm_p = np.mean(lstm_p_l, 0)
    lgb_a = np.mean(lgb_a_l, 0);   lgb_p = np.mean(lgb_p_l, 0)
    ens_a = a_alpha * lstm_a + (1-a_alpha) * lgb_a
    ens_p = p_alpha * lstm_p + (1-p_alpha) * lgb_p

    # --- Greedy suppression ---
    print("\nGreedy class suppression (action):")
    new_bias_a, f1_a_new, suppressed_a = greedy_suppress(ens_a, bias_a, la_ref, "action")

    print("\nGreedy class suppression (point):")
    new_bias_p, f1_p_new, suppressed_p = greedy_suppress(ens_p, bias_p, lp_ref, "point")

    f1_a_old, _ = f1_with_bias(ens_a, bias_a, la_ref)
    f1_p_old, _ = f1_with_bias(ens_p, bias_p, lp_ref)
    cv_old = 0.4*f1_a_old + 0.4*f1_p_old + 0.2
    cv_new = 0.4*f1_a_new + 0.4*f1_p_new + 0.2

    print("\n" + "="*78)
    print("SUMMARY")
    print("="*78)
    print(f"  Action:  F1 {f1_a_old:.4f} → {f1_a_new:.4f}  (Δ={f1_a_new-f1_a_old:+.4f})  suppressed: {suppressed_a}")
    print(f"  Point:   F1 {f1_p_old:.4f} → {f1_p_new:.4f}  (Δ={f1_p_new-f1_p_old:+.4f})  suppressed: {suppressed_p}")
    print(f"  CV:      {cv_old:.4f} → {cv_new:.4f}  (Δ={cv_new-cv_old:+.4f})")

    if cv_new - cv_old < 0.001:
        print("\n  → Suppression gain < 0.001. Marginal. Skip ship.")
        return

    # --- Apply to test predictions ---
    print("\nApplying suppression to test predictions...")
    lstm_test_a_l, lstm_test_p_l = [], []
    lgb_test_a_l, lgb_test_p_l = [], []
    test_uids_ref, test_sgp_ref = None, None
    for seed in seeds:
        d = np.load(f"artifacts/v5z_md_full_s{seed}_test.npz")
        lstm_test_a_l.append(d["lstm_a"]); lstm_test_p_l.append(d["lstm_p"])
        lgb_test_a_l.append(d["lgb_a"]);   lgb_test_p_l.append(d["lgb_p"])
        if test_uids_ref is None:
            test_uids_ref = d["test_uids"]; test_sgp_ref = d["test_sgp"]

    lstm_test_a = np.mean(lstm_test_a_l, 0); lstm_test_p = np.mean(lstm_test_p_l, 0)
    lgb_test_a = np.mean(lgb_test_a_l, 0);   lgb_test_p = np.mean(lgb_test_p_l, 0)

    ens_test_a = a_alpha * lstm_test_a + (1-a_alpha) * lgb_test_a
    ens_test_p = p_alpha * lstm_test_p + (1-p_alpha) * lgb_test_p
    pred_action = (np.log(ens_test_a + 1e-12) + new_bias_a).argmax(1)
    pred_point = (np.log(ens_test_p + 1e-12) + new_bias_p).argmax(1)

    sub = pd.DataFrame({
        "rally_uid": test_uids_ref,
        "actionId": pred_action,
        "pointId": pred_point,
        "serverGetPoint": test_sgp_ref,
    }).sort_values("rally_uid").reset_index(drop=True)

    out = "submissions/submission_v5z_md_suppress_bag2.csv"
    sub.to_csv(out, index=False)
    print(f"\nSaved: {out}")
    print(f"  Test action pred distribution: {dict(sorted(pd.Series(pred_action).value_counts().items()))}")
    print(f"  Test point  pred distribution: {dict(sorted(pd.Series(pred_point).value_counts().items()))}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
A v2 Full — historical player stats + sex-conditional cold-start.
Per-fold leakage-safe player_stats computation. LSTM reused.

Usage:
    V5Z_AV2=1 V5Z_SEED=42  python3 src/train/lgb_only_av2.py
    V5Z_AV2=1 V5Z_SEED=1337 python3 src/train/lgb_only_av2.py

Output:
    artifacts/v5z_av2_s{SEED}_oof.npz
    artifacts/v5z_av2_s{SEED}_test.npz
    logs/lgb_av2_s{SEED}.log

Pipeline:
  1. Reproduce V5Z's StratifiedKFold split.
  2. num_leaves sweep on fold 1 only ([63, 127, 255]).
  3. Full 5-fold with best num_leaves:
     - Per-fold: compute player_stats from train fold uids only (leakage-safe)
     - Build LGB features with per-fold player_stats
     - Train LGB
  4. Test preds: build features with FULL player_stats (all train, no leakage).
  5. Save in V5Z npz format (LSTM probs reused from existing artifacts).

Must-fix incorporated from debate:
  - Per-fold player_stats: train-fold-only computation
  - num_leaves sweep: try [63, 127, 255]
  - Cold-start: sex-conditional marginals
  - srv_rcv_seen_combo: interaction feature for unseen handling
  - LSTM unchanged (acknowledged ensemble caps gain to ~77%)
"""
import os
import pathlib
os.chdir(pathlib.Path(__file__).resolve().parents[2])

# Force AV2 ON before importing train_v5z so build_lgb_features picks it up
os.environ["V5Z_AV2"] = "1"

import sys
import time
import numpy as np
import lightgbm as lgb
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, "src/train")
from train_v5z import (
    build_lgb_features, compute_player_stats,
    train_df, test_df, prepare_samples,
    N_FOLDS, SEED, N_ACTION, N_POINT,
)


def log(msg):
    print(msg, flush=True)


def train_lgb_with_leaves(X_tr, y_tr, X_va, y_va, n_class, task_name, num_leaves):
    """Train LGB with explicit num_leaves override."""
    params = {
        "objective": "multiclass",
        "num_class": n_class,
        "learning_rate": 0.05,
        "num_leaves": num_leaves,
        "max_depth": 9,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "class_weight": "balanced",
        "verbose": -1,
        "seed": SEED,
        "n_jobs": -1,
    }
    dtrain = lgb.Dataset(X_tr, y_tr)
    dval = lgb.Dataset(X_va, y_va, reference=dtrain)
    model = lgb.train(
        params, dtrain,
        num_boost_round=1000,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
    )
    va_probs = model.predict(X_va)
    f1 = f1_score(y_va, va_probs.argmax(1), average="macro", zero_division=0)
    return model, va_probs, f1


def main():
    out_tag = f"v5z_av2_s{SEED}"
    log("\n" + "="*78)
    log(f"A v2 Full — historical player stats, seed={SEED}")
    log("="*78)

    # --- Reuse LSTM artifacts ---
    src_oof = f"artifacts/v5z_s{SEED}_oof.npz"
    src_test = f"artifacts/v5z_s{SEED}_test.npz"
    assert os.path.exists(src_oof), f"missing {src_oof}"
    assert os.path.exists(src_test), f"missing {src_test}"
    log(f"\nReusing LSTM artifacts: {src_oof}, {src_test}")
    src_oof_d = np.load(src_oof)
    src_test_d = np.load(src_test)

    # --- Reproduce fold split ---
    log("\nReproducing fold split...")
    train_samples = prepare_samples(train_df, is_train=True, augment=True)
    uid_last_a = {s["uid"]: s["target_action"] for s in train_samples}
    uid_list = sorted(uid_last_a.keys())
    uid_arr = np.array([uid_last_a[u] for u in uid_list])
    kf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    fold_splits = list(kf.split(uid_list, uid_arr))

    # --- num_leaves sweep on fold 1 ---
    log("\n" + "─"*70)
    log("num_leaves sweep on fold 1 (action only, [63, 127, 255])")
    log("─"*70)

    fold = 0  # fold 1 (0-indexed)
    tr_uidx, va_uidx = fold_splits[fold]
    tr_uids = set(np.array(uid_list)[tr_uidx])
    va_uids_set = set(np.array(uid_list)[va_uidx])

    t0 = time.time()
    log(f"  Computing player_stats from {len(tr_uids)} train uids...")
    pstats_f1, cold_f1, seen_f1 = compute_player_stats(train_df, restrict_uids=tr_uids)
    log(f"  player_stats: {len(pstats_f1)} players, {time.time()-t0:.1f}s")

    log("  Building LGB features (V5Z_AV2=1) with per-fold player_stats...")
    t0 = time.time()
    Xall, ya_all, yp_all, uids_all, _ = build_lgb_features(
        train_df, is_train=True, augment=True,
        player_stats=pstats_f1, cold_marginals=cold_f1, seen_players=seen_f1)
    log(f"  features: {Xall.shape[1]} cols, {time.time()-t0:.1f}s")

    new_cols = [c for c in Xall.columns if c.startswith("srv_") or c.startswith("rcv_")]
    log(f"  new AV2 columns ({len(new_cols)}): {new_cols[:5]}...{new_cols[-3:]}")
    assert len(new_cols) >= 30, f"expected ~30+ AV2 cols, got {len(new_cols)}"

    # uid → index
    uid2idx = {}
    for i, u in enumerate(uids_all):
        uid2idx.setdefault(u, []).append(i)
    tr_idx = [i for u in tr_uids for i in uid2idx.get(u, [])]
    va_idx = [i for u in va_uids_set for i in uid2idx.get(u, [])]
    X_tr, X_va = Xall.iloc[tr_idx], Xall.iloc[va_idx]
    ya_tr = [ya_all[i] for i in tr_idx]; ya_va = [ya_all[i] for i in va_idx]

    sweep_results = {}
    for nl in [63, 127, 255]:
        t0 = time.time()
        _, _, f1a = train_lgb_with_leaves(X_tr, ya_tr, X_va, ya_va, N_ACTION,
                                           "action", nl)
        sweep_results[nl] = f1a
        log(f"  num_leaves={nl}: action F1={f1a:.4f} ({time.time()-t0:.1f}s)")

    best_nl = max(sweep_results, key=sweep_results.get)
    log(f"  → best num_leaves: {best_nl} (F1={sweep_results[best_nl]:.4f})")

    # --- Full 5-fold with best num_leaves ---
    log("\n" + "═"*70)
    log(f"Full 5-fold training with num_leaves={best_nl}")
    log("═"*70)

    all_oof_a, all_oof_p = [], []
    test_a_list, test_p_list = [], []
    f1_a_folds, f1_p_folds = [], []

    test_uids_seq = src_test_d["test_uids"]

    for fold, (tr_uidx, va_uidx) in enumerate(fold_splits):
        log(f"\n{'─'*60}")
        log(f"Fold {fold+1}/{N_FOLDS}")
        tr_uids = set(np.array(uid_list)[tr_uidx])
        va_uids_set = set(np.array(uid_list)[va_uidx])

        # Per-fold player_stats
        t0 = time.time()
        pstats_f, cold_f, seen_f = compute_player_stats(train_df, restrict_uids=tr_uids)
        log(f"  player_stats: {len(pstats_f)} players, {time.time()-t0:.1f}s")

        # Build features for both train + val of this fold
        t0 = time.time()
        Xall, ya_all, yp_all, uids_all, _ = build_lgb_features(
            train_df, is_train=True, augment=True,
            player_stats=pstats_f, cold_marginals=cold_f, seen_players=seen_f)
        log(f"  features built: {time.time()-t0:.1f}s")

        uid2idx = {}
        for i, u in enumerate(uids_all):
            uid2idx.setdefault(u, []).append(i)
        tr_idx = [i for u in tr_uids for i in uid2idx.get(u, [])]
        va_idx = [i for u in va_uids_set for i in uid2idx.get(u, [])]
        X_tr, X_va = Xall.iloc[tr_idx], Xall.iloc[va_idx]
        ya_tr = [ya_all[i] for i in tr_idx]; ya_va = [ya_all[i] for i in va_idx]
        yp_tr = [yp_all[i] for i in tr_idx]; yp_va = [yp_all[i] for i in va_idx]

        # Action
        ma_model, ma_probs, fa = train_lgb_with_leaves(
            X_tr, ya_tr, X_va, ya_va, N_ACTION, "action", best_nl)
        # Point
        mp_model, mp_probs, fp = train_lgb_with_leaves(
            X_tr, yp_tr, X_va, yp_va, N_POINT, "point", best_nl)

        log(f"  fold {fold+1}: action F1={fa:.4f}  point F1={fp:.4f}")
        all_oof_a.append(ma_probs)
        all_oof_p.append(mp_probs)
        f1_a_folds.append(fa); f1_p_folds.append(fp)

        # Test predict per fold (use FULL player_stats trained per fold for now;
        # to be averaged across folds — each fold's model has slightly different
        # training data but same feature set)
        # Build test features with the SAME per-fold player_stats so model expects
        # consistent feature distribution. Acceptable since we average across folds.
        Xte, _, _, te_uids, _ = build_lgb_features(
            test_df, is_train=False,
            player_stats=pstats_f, cold_marginals=cold_f, seen_players=seen_f)
        assert np.array_equal(np.array(te_uids), test_uids_seq), "test uid order mismatch"
        test_a_list.append(ma_model.predict(Xte))
        test_p_list.append(mp_model.predict(Xte))

    log(f"\n  Mean LGB-only F1: action={np.mean(f1_a_folds):.4f}  point={np.mean(f1_p_folds):.4f}")

    # --- Concat OOF + test average ---
    new_lgb_a = np.vstack(all_oof_a)
    new_lgb_p = np.vstack(all_oof_p)
    test_lgb_a = np.mean(test_a_list, axis=0)
    test_lgb_p = np.mean(test_p_list, axis=0)

    assert new_lgb_a.shape == src_oof_d["lgb_a"].shape, \
        f"shape mismatch: new {new_lgb_a.shape} vs old {src_oof_d['lgb_a'].shape}"

    # --- Save in V5Z format ---
    out_oof = f"artifacts/{out_tag}_oof.npz"
    out_test = f"artifacts/{out_tag}_test.npz"
    np.savez(
        out_oof,
        lstm_a=src_oof_d["lstm_a"], lstm_p=src_oof_d["lstm_p"],
        lgb_a=new_lgb_a, lgb_p=new_lgb_p,
        la=src_oof_d["la"], lp=src_oof_d["lp"],
        a_scales=src_oof_d.get("a_scales", np.ones(N_ACTION)),
        p_scales=src_oof_d.get("p_scales", np.ones(N_POINT)),
        best_w=src_oof_d.get("best_w", 0.5),
        cv_score=src_oof_d.get("cv_score", 0.0),
    )
    np.savez(
        out_test,
        lstm_a=src_test_d["lstm_a"], lstm_p=src_test_d["lstm_p"],
        lgb_a=test_lgb_a, lgb_p=test_lgb_p,
        test_uids=src_test_d["test_uids"], test_sgp=src_test_d["test_sgp"],
    )

    log(f"\nSaved: {out_oof}")
    log(f"Saved: {out_test}")
    log(f"Best num_leaves: {best_nl}")
    log(f"\nNext: V5Z_TAG_PREFIX=v5z_av2 V5Z_SEEDS={SEED} python3 src/train/advcal_v5z.py")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
A v2 Step 0 — cheap baseline: retrain LGB only with 6 in-prefix observed
features added. Reuses existing LSTM artifacts to save GPU time.

Usage:
    V5Z_AV2_S0=1 V5Z_SEED=42   python3 src/train/lgb_only_av2s0.py
    V5Z_AV2_S0=1 V5Z_SEED=1337 python3 src/train/lgb_only_av2s0.py

Output:
    artifacts/v5z_av2s0_s{SEED}_oof.npz   (lstm reused, lgb new)
    artifacts/v5z_av2s0_s{SEED}_test.npz  (lstm reused, lgb new)

Notes:
- Reproduces V5Z_SEED's StratifiedKFold split exactly.
- Saves OOF in fold-iteration order (matches V5Z's convention so
  advcal_v5z's build_oof_reindex still works).
- ZERO leakage: in-prefix features only use the prefix rows that the
  test-time inference would also see. Per-player aggregation is NOT
  used (saved for full A v2 if cheap baseline passes).
"""
import os
import pathlib
os.chdir(pathlib.Path(__file__).resolve().parents[2])

# Force AV2_S0 ON before importing train_v5z so build_lgb_features picks it up
os.environ["V5Z_AV2_S0"] = "1"

import sys
import time
import numpy as np
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, "src/train")
from train_v5z import (
    build_lgb_features, train_df, test_df,
    prepare_samples, N_FOLDS, SEED, train_lgb_fold,
    N_ACTION, N_POINT,
)


def log(msg):
    print(msg, flush=True)


def main():
    out_tag = f"v5z_av2s0_s{SEED}"
    log(f"\n{'='*70}")
    log(f"A v2 Step 0 (cheap baseline) — LGB only, seed={SEED}")
    log(f"{'='*70}")

    # --- Load existing LSTM artifacts ---
    src_oof = f"artifacts/v5z_s{SEED}_oof.npz"
    src_test = f"artifacts/v5z_s{SEED}_test.npz"
    assert os.path.exists(src_oof), f"missing {src_oof}"
    assert os.path.exists(src_test), f"missing {src_test}"
    log(f"\nReusing LSTM artifacts from: {src_oof}, {src_test}")
    src_oof_d = np.load(src_oof)
    src_test_d = np.load(src_test)

    # --- Build new LGB features (V5Z_AV2_S0=1 already set) ---
    t0 = time.time()
    log("\nBuilding LGB features (with V5Z_AV2_S0=1, +6 in-prefix features)...")
    lgb_X_train, lgb_y_a, lgb_y_p, lgb_uids, _ = build_lgb_features(
        train_df, is_train=True, augment=True)
    lgb_X_test, _, _, lgb_test_uids, _ = build_lgb_features(
        test_df, is_train=False)
    log(f"  LGB features: {lgb_X_train.shape[1]} (was 49 baseline)")
    log(f"  prep time: {time.time()-t0:.1f}s")

    new_cols = [c for c in lgb_X_train.columns if c.startswith("srv_") or c.startswith("rcv_")]
    log(f"  new columns added: {new_cols}")
    assert len(new_cols) == 6, f"expected 6 in-prefix features, got {len(new_cols)}: {new_cols}"
    assert np.array_equal(np.array(lgb_test_uids), src_test_d["test_uids"]), \
        "test uid order mismatch between LGB build and saved LSTM artifacts"

    # --- Reproduce V5Z fold split ---
    log(f"\nReproducing fold split (StratifiedKFold(N={N_FOLDS}, random_state={SEED}))...")
    train_samples = prepare_samples(train_df, is_train=True, augment=True)
    uid_last_a = {s["uid"]: s["target_action"] for s in train_samples}
    uid_list = sorted(uid_last_a.keys())
    uid_arr = np.array([uid_last_a[u] for u in uid_list])

    lgb_uid2idx = {}
    for i, u in enumerate(lgb_uids):
        lgb_uid2idx.setdefault(u, []).append(i)

    kf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)

    # --- Per-fold LGB train ---
    all_oof_lgb_a, all_oof_lgb_p = [], []
    test_lgb_a_list, test_lgb_p_list = [], []
    f1_a_folds, f1_p_folds = [], []

    for fold, (tr_uidx, va_uidx) in enumerate(kf.split(uid_list, uid_arr)):
        log(f"\n{'─'*60}")
        log(f"Fold {fold+1}/{N_FOLDS}")
        tr_uids = set(np.array(uid_list)[tr_uidx])
        va_uids = set(np.array(uid_list)[va_uidx])

        lgb_tr_idx = [i for u in tr_uids for i in lgb_uid2idx.get(u, [])]
        lgb_va_idx = [i for u in va_uids for i in lgb_uid2idx.get(u, [])]
        X_tr = lgb_X_train.iloc[lgb_tr_idx]
        X_va = lgb_X_train.iloc[lgb_va_idx]
        y_a_tr = [lgb_y_a[i] for i in lgb_tr_idx]
        y_a_va = [lgb_y_a[i] for i in lgb_va_idx]
        y_p_tr = [lgb_y_p[i] for i in lgb_tr_idx]
        y_p_va = [lgb_y_p[i] for i in lgb_va_idx]

        log(f"  LGB Train: {len(X_tr)}, Val: {len(X_va)}")

        lgb_a_model, lgb_a_va_probs, fa = train_lgb_fold(
            X_tr, y_a_tr, X_va, y_a_va, N_ACTION, "actionId")
        lgb_p_model, lgb_p_va_probs, fp = train_lgb_fold(
            X_tr, y_p_tr, X_va, y_p_va, N_POINT, "pointId")

        all_oof_lgb_a.append(lgb_a_va_probs)
        all_oof_lgb_p.append(lgb_p_va_probs)
        f1_a_folds.append(fa); f1_p_folds.append(fp)

        # Test predict
        test_lgb_a_list.append(lgb_a_model.predict(lgb_X_test))
        test_lgb_p_list.append(lgb_p_model.predict(lgb_X_test))

    log(f"\n  LGB-only mean F1: action={np.mean(f1_a_folds):.4f} point={np.mean(f1_p_folds):.4f}")

    # --- Concat OOF in fold-iteration order ---
    new_lgb_a = np.vstack(all_oof_lgb_a)
    new_lgb_p = np.vstack(all_oof_lgb_p)

    # Sanity: shapes must match existing LSTM OOF
    assert new_lgb_a.shape == src_oof_d["lgb_a"].shape, (
        f"shape mismatch: new {new_lgb_a.shape} vs old {src_oof_d['lgb_a'].shape}")

    # Test: average across folds
    test_lgb_a = np.mean(test_lgb_a_list, axis=0)
    test_lgb_p = np.mean(test_lgb_p_list, axis=0)

    # --- Save in V5Z format (reuse LSTM, replace LGB) ---
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
    log(f"\nNext: V5Z_TAG_PREFIX=v5z_av2s0 V5Z_SEEDS=42,1337 python3 src/train/advcal_v5z.py")


if __name__ == "__main__":
    main()

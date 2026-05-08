#!/usr/bin/env python3
"""
A3 — Match-disjoint LGB with Empirical-Bayes prefix-only player features.

Reuses match-disjoint LSTM artifacts (v5z_md_full_s{SEED}). Only retrains LGB
with new EB features computed from rally's own visible prefix + global Beta prior.

Key difference from A v2:
- A v2 used train-derived per-player aggregates → cross-event biased estimator
- EB uses VISIBLE PREFIX of THE RALLY ITSELF + strong shrinkage to global prior
  → in-distribution by construction, handles unseen players naturally

Usage:
    V5Z_EB=1 V5Z_SEED=42   python3 src/train/lgb_md_eb.py
    V5Z_EB=1 V5Z_SEED=1337 python3 src/train/lgb_md_eb.py

Output:
    artifacts/v5z_mdeb_full_s{SEED}_oof.npz   (LSTM reused, LGB new with EB features)
    artifacts/v5z_mdeb_full_s{SEED}_test.npz
"""
import os
import pathlib
os.chdir(pathlib.Path(__file__).resolve().parents[2])

# Force EB ON before importing train_v5z
os.environ["V5Z_EB"] = "1"

import sys
import time
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, "src/train")
from train_v5z import (
    build_lgb_features, train_df, test_df,
    prepare_samples, N_FOLDS, SEED, train_lgb_fold,
    N_ACTION, N_POINT, ACTION_GROUPS,
)


def log(msg):
    print(msg, flush=True)


def compute_global_priors(df):
    """Global rates over ALL train strokes (used as Beta prior for shrinkage)."""
    n = len(df)
    actions = df["actionId"].values
    groups = np.array([ACTION_GROUPS.get(int(a), 0) for a in actions])
    return {
        "attack_ratio":  float((groups == 1).mean()),
        "control_ratio": float((groups == 2).mean()),
        "defense_ratio": float((groups == 3).mean()),
        "forehand_ratio": float((df["handId"] == 1).mean()),
        "topspin_ratio": float((df["spinId"] == 1).mean()),
    }


def main():
    out_tag = f"v5z_mdeb_full_s{SEED}"
    log("\n" + "="*78)
    log(f"A3: Match-disjoint LGB with Empirical-Bayes prefix-only features, seed={SEED}")
    log("="*78)

    # --- Reuse match-disjoint LSTM artifacts ---
    src_oof = f"artifacts/v5z_md_full_s{SEED}_oof.npz"
    src_test = f"artifacts/v5z_md_full_s{SEED}_test.npz"
    assert os.path.exists(src_oof), f"missing {src_oof}"
    log(f"\nReusing match-disjoint LSTM artifacts: {src_oof}")
    src_oof_d = np.load(src_oof)
    src_test_d = np.load(src_test)

    # --- Compute global priors from train ---
    log("\nComputing global priors from train data...")
    global_priors = compute_global_priors(train_df)
    log(f"  global priors: {global_priors}")

    # --- Build LGB features with EB ---
    log("\nBuilding LGB features (V5Z_EB=1)...")
    t0 = time.time()
    lgb_X_train, lgb_y_a, lgb_y_p, lgb_uids, _ = build_lgb_features(
        train_df, is_train=True, augment=True, global_priors=global_priors)
    lgb_X_test, _, _, lgb_test_uids, _ = build_lgb_features(
        test_df, is_train=False, global_priors=global_priors)
    log(f"  features: {lgb_X_train.shape[1]} cols, {time.time()-t0:.1f}s")

    new_cols = [c for c in lgb_X_train.columns if c.startswith("srv_eb_") or c.startswith("rcv_eb_") or c.startswith("eb_")]
    log(f"  EB columns ({len(new_cols)}): {new_cols}")
    assert len(new_cols) >= 13, f"expected ~13+ EB cols, got {len(new_cols)}"

    # --- Match-disjoint fold split (same as v5z_md_full was trained with) ---
    log("\nReproducing match-disjoint fold split...")
    train_samples = prepare_samples(train_df, is_train=True, augment=True)
    uid_last_a = {s["uid"]: s["target_action"] for s in train_samples}
    uid_to_match = train_df.groupby("rally_uid")["match"].first().to_dict()
    uid_list = sorted(uid_last_a.keys())
    uid_arr = np.array([uid_last_a[u] for u in uid_list])
    match_arr = np.array([uid_to_match[u] for u in uid_list])

    sgkf = StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=SEED)
    md_splits = list(sgkf.split(uid_list, uid_arr, groups=match_arr))

    lgb_uid2idx = {}
    for i, u in enumerate(lgb_uids):
        lgb_uid2idx.setdefault(u, []).append(i)

    # --- 5-fold LGB train with EB features ---
    log("\n" + "─"*70)
    log("Training LGB (5 folds, match-disjoint, with EB features)")
    log("─"*70)

    all_oof_a, all_oof_p = [], []
    test_a_l, test_p_l = [], []
    f1a_l, f1p_l = [], []

    for fold, (tr_uidx, va_uidx) in enumerate(md_splits):
        tr_uids = set(np.array(uid_list)[tr_uidx])
        va_uids_s = set(np.array(uid_list)[va_uidx])

        lgb_tr_idx = [i for u in tr_uids for i in lgb_uid2idx.get(u, [])]
        lgb_va_idx = [i for u in va_uids_s for i in lgb_uid2idx.get(u, [])]
        X_tr = lgb_X_train.iloc[lgb_tr_idx]
        X_va = lgb_X_train.iloc[lgb_va_idx]
        y_a_tr = [lgb_y_a[i] for i in lgb_tr_idx]
        y_a_va = [lgb_y_a[i] for i in lgb_va_idx]
        y_p_tr = [lgb_y_p[i] for i in lgb_tr_idx]
        y_p_va = [lgb_y_p[i] for i in lgb_va_idx]

        log(f"  fold {fold+1}: train={len(X_tr)}, val={len(X_va)}")

        ma_model, ma_probs, fa = train_lgb_fold(
            X_tr, y_a_tr, X_va, y_a_va, N_ACTION, "action")
        mp_model, mp_probs, fp = train_lgb_fold(
            X_tr, y_p_tr, X_va, y_p_va, N_POINT, "point")

        log(f"  fold {fold+1}: action F1={fa:.4f}  point F1={fp:.4f}")
        all_oof_a.append(ma_probs); all_oof_p.append(mp_probs)
        f1a_l.append(fa); f1p_l.append(fp)
        test_a_l.append(ma_model.predict(lgb_X_test))
        test_p_l.append(mp_model.predict(lgb_X_test))

    log(f"\n  Mean LGB-only F1: action={np.mean(f1a_l):.4f}  point={np.mean(f1p_l):.4f}")

    # --- Save with NEW orderings (match-disjoint with this seed's split) ---
    new_lgb_a = np.vstack(all_oof_a)
    new_lgb_p = np.vstack(all_oof_p)
    test_lgb_a = np.mean(test_a_l, 0)
    test_lgb_p = np.mean(test_p_l, 0)

    # Sanity: shapes
    assert new_lgb_a.shape == src_oof_d["lgb_a"].shape, \
        f"shape mismatch: {new_lgb_a.shape} vs {src_oof_d['lgb_a'].shape}"

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
    log(f"\nNext: V5Z_SEEDS={SEED} python3 src/train/advcal_match_disjoint.py  (with mdeb tag prefix)")


if __name__ == "__main__":
    main()

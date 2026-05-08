#!/usr/bin/env python3
"""
A1 — Match-disjoint OOF for LGB only (LSTM reused from existing artifacts).

Hypothesis: rally-level KFold OOF inflates F1 because train/val folds share
~99% players. StratifiedGroupKFold(groups=match_id) gives ~43 unseen matches
per val fold, mirroring the 55-match test set. If hypothesis holds, LGB
match-disjoint OOF F1 << LGB rally-fold OOF F1.

Usage:
    V5Z_SEED=42   python3 src/train/match_disjoint_lgb.py
    V5Z_SEED=1337 python3 src/train/match_disjoint_lgb.py

Output:
    artifacts/v5z_md_s{SEED}_oof.npz   (lstm reused, lgb new with match-disjoint folds)
    artifacts/v5z_md_s{SEED}_test.npz  (lstm reused, lgb new test preds avg over folds)
    logs/match_disjoint_lgb_s{SEED}.log
"""
import os
import pathlib
os.chdir(pathlib.Path(__file__).resolve().parents[2])

import sys
import time
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold

sys.path.insert(0, "src/train")
from train_v5z import (
    build_lgb_features, train_df, test_df,
    prepare_samples, N_FOLDS, SEED, train_lgb_fold,
    N_ACTION, N_POINT,
)


def log(msg):
    print(msg, flush=True)


def main():
    out_tag = f"v5z_md_s{SEED}"
    log("\n" + "="*78)
    log(f"A1: Match-disjoint LGB-only retrain, seed={SEED}")
    log("="*78)

    # --- Reuse LSTM artifacts (rally-KFold trained — biased but unavoidable for now) ---
    src_oof = f"artifacts/v5z_s{SEED}_oof.npz"
    src_test = f"artifacts/v5z_s{SEED}_test.npz"
    assert os.path.exists(src_oof), f"missing {src_oof}"
    log(f"\nReusing LSTM artifacts: {src_oof}")
    src_oof_d = np.load(src_oof)
    src_test_d = np.load(src_test)

    # --- Build LGB features (no AV2, vanilla 49 features) ---
    log("\nBuilding LGB features (vanilla 49)...")
    t0 = time.time()
    lgb_X_train, lgb_y_a, lgb_y_p, lgb_uids, _ = build_lgb_features(
        train_df, is_train=True, augment=True)
    lgb_X_test, _, _, lgb_test_uids, _ = build_lgb_features(
        test_df, is_train=False)
    log(f"  features: {lgb_X_train.shape[1]} cols, {time.time()-t0:.1f}s")

    # --- Build match-disjoint fold split ---
    log("\nBuilding match-disjoint StratifiedGroupKFold split...")
    train_samples = prepare_samples(train_df, is_train=True, augment=True)
    uid_last_a = {s["uid"]: s["target_action"] for s in train_samples}
    uid_to_match = train_df.groupby("rally_uid")["match"].first().to_dict()

    uid_list = sorted(uid_last_a.keys())
    uid_arr = np.array([uid_last_a[u] for u in uid_list])
    match_arr = np.array([uid_to_match[u] for u in uid_list])

    n_unique_matches = len(set(match_arr))
    log(f"  uids={len(uid_list)}, unique matches={n_unique_matches}")

    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    md_splits = list(sgkf.split(uid_list, uid_arr, groups=match_arr))

    # Diagnostic: per-fold match counts
    log("  Per-fold matches: train | val")
    for fi, (tr, va) in enumerate(md_splits):
        tr_matches = set(match_arr[tr]); va_matches = set(match_arr[va])
        log(f"    fold {fi+1}: {len(tr_matches):3d} train matches | {len(va_matches):3d} val matches  (overlap: {len(tr_matches & va_matches)})")

    # --- Also reproduce rally-KFold for comparison ---
    log("\nReproducing rally-KFold (the existing protocol)...")
    rkf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    rk_splits = list(rkf.split(uid_list, uid_arr))

    # uid → LGB index mapping
    lgb_uid2idx = {}
    for i, u in enumerate(lgb_uids):
        lgb_uid2idx.setdefault(u, []).append(i)

    # ============ TRAIN LGB on both protocols ============
    def run_protocol(splits, label):
        log(f"\n{'─'*70}")
        log(f"Training LGB with {label}")
        log(f"{'─'*70}")
        all_oof_a, all_oof_p = [], []
        all_oof_la, all_oof_lp = [], []
        test_a_l, test_p_l = [], []
        f1a_l, f1p_l = [], []

        for fold, (tr_uidx, va_uidx) in enumerate(splits):
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

            all_oof_a.append(ma_probs); all_oof_p.append(mp_probs)
            all_oof_la.append(np.array(y_a_va)); all_oof_lp.append(np.array(y_p_va))
            f1a_l.append(fa); f1p_l.append(fp)
            test_a_l.append(ma_model.predict(lgb_X_test))
            test_p_l.append(mp_model.predict(lgb_X_test))

        merged_a = np.vstack(all_oof_a)
        merged_p = np.vstack(all_oof_p)
        merged_la = np.concatenate(all_oof_la)
        merged_lp = np.concatenate(all_oof_lp)

        # Full-OOF macro F1 (concatenated)
        f1a_full = f1_score(merged_la, merged_a.argmax(1), average="macro", zero_division=0)
        f1p_full = f1_score(merged_lp, merged_p.argmax(1), average="macro", zero_division=0)

        log(f"\n  {label} per-fold mean F1: action={np.mean(f1a_l):.4f}  point={np.mean(f1p_l):.4f}")
        log(f"  {label} FULL-OOF F1:        action={f1a_full:.4f}  point={f1p_full:.4f}")

        return {
            "oof_a": merged_a, "oof_p": merged_p,
            "la": merged_la, "lp": merged_lp,
            "test_a": np.mean(test_a_l, 0), "test_p": np.mean(test_p_l, 0),
            "f1a_per_fold": f1a_l, "f1p_per_fold": f1p_l,
            "f1a_full": f1a_full, "f1p_full": f1p_full,
        }

    rk_result = run_protocol(rk_splits, "rally-KFold")
    md_result = run_protocol(md_splits, "match-disjoint")

    log("\n" + "="*78)
    log("DIAGNOSTIC SUMMARY")
    log("="*78)
    log(f"  Rally-KFold     LGB OOF: action={rk_result['f1a_full']:.4f}  point={rk_result['f1p_full']:.4f}")
    log(f"  Match-disjoint  LGB OOF: action={md_result['f1a_full']:.4f}  point={md_result['f1p_full']:.4f}")
    log(f"  Δ_action = {md_result['f1a_full']-rk_result['f1a_full']:+.4f}")
    log(f"  Δ_point  = {md_result['f1p_full']-rk_result['f1p_full']:+.4f}")
    log(f"  Δ_CV     = {0.4*(md_result['f1a_full']-rk_result['f1a_full']) + 0.4*(md_result['f1p_full']-rk_result['f1p_full']):+.4f}")

    log("\n  Interpretation:")
    delta_cv = 0.4*(md_result['f1a_full']-rk_result['f1a_full']) + 0.4*(md_result['f1p_full']-rk_result['f1p_full'])
    if abs(delta_cv) < 0.005:
        log(f"  → Hypothesis WEAK: match-disjoint ≈ rally-KFold (Δ={delta_cv:+.4f})")
        log(f"     CV protocol may not be the main bottleneck. Re-investigate.")
    elif delta_cv < -0.010:
        log(f"  → Hypothesis STRONG: match-disjoint OOF much lower (Δ={delta_cv:+.4f})")
        log(f"     OOF was inflated by player overlap. Use match-disjoint for selection.")
    else:
        log(f"  → Hypothesis MODERATE: match-disjoint slightly lower (Δ={delta_cv:+.4f})")

    # --- Save match-disjoint LGB OOF (in fold-iteration order, NOT sample-idx canonical) ---
    # Note: match-disjoint folds give different sample order than rally-KFold.
    # We save raw and labels together.
    out_oof = f"artifacts/{out_tag}_oof.npz"
    out_test = f"artifacts/{out_tag}_test.npz"

    # For LSTM-reuse path: we cannot directly merge match-disjoint LGB with rally-KFold LSTM
    # because their sample orderings differ. Save match-disjoint LGB separately.
    np.savez(
        out_oof,
        lgb_a=md_result["oof_a"], lgb_p=md_result["oof_p"],
        la=md_result["la"], lp=md_result["lp"],
        # Also save rally-KFold LGB for direct comparison
        lgb_a_rk=rk_result["oof_a"], lgb_p_rk=rk_result["oof_p"],
        la_rk=rk_result["la"], lp_rk=rk_result["lp"],
    )
    np.savez(
        out_test,
        lgb_a=md_result["test_a"], lgb_p=md_result["test_p"],
        lgb_a_rk=rk_result["test_a"], lgb_p_rk=rk_result["test_p"],
        test_uids=src_test_d["test_uids"], test_sgp=src_test_d["test_sgp"],
        # Reuse LSTM test preds (rally-KFold trained, biased)
        lstm_a=src_test_d["lstm_a"], lstm_p=src_test_d["lstm_p"],
    )

    log(f"\nSaved: {out_oof}")
    log(f"Saved: {out_test}")


if __name__ == "__main__":
    main()

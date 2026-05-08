"""
Attack 4 diagnostic: existing global LGB on L=1 OOF vs slice-trained LGB on L=1 only.

If slice-LGB is not meaningfully better (Δ < +0.003 macro-F1) than existing
global LGB on the L=1 slice, then P3's Expert A premise is cargo-culting.

Cost: ~30-60 min. Uses matched hyperparameters to existing LGB (lr=0.03, num_leaves=127, etc.)
"""
import os, sys, time
os.environ["V5Z_MATCH_DISJOINT"] = "1"
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

from train_v5z import (prepare_samples, build_lgb_features, train_df,
                       N_ACTION, N_POINT, N_FOLDS, log)


# Match existing LGB hyperparameters (from train_v5z.py line 1037)
def lgb_params(seed, n_classes):
    return {
        "objective": "multiclass",
        "num_class": n_classes,
        "metric": "multi_logloss",
        "learning_rate": 0.03,
        "num_leaves": 127,
        "max_depth": 9,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "class_weight": "balanced",
        "verbose": -1,
        "seed": seed,
        "n_jobs": -1,
    }


def train_lgb_slice(X_tr, y_tr, X_va, y_va, n_class, seed):
    params = lgb_params(seed, n_class)
    dtr = lgb.Dataset(X_tr, label=y_tr)
    dva = lgb.Dataset(X_va, label=y_va, reference=dtr)
    m = lgb.train(params, dtr, num_boost_round=1000,
                  valid_sets=[dva],
                  callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])
    return m, m.predict(X_va)


def main():
    log("="*78)
    log("ATTACK 4 DIAGNOSTIC — slice-LGB on L=1 vs existing global LGB on L=1 OOF")
    log("="*78)

    # ============================================================
    # 1. Build features + samples
    # ============================================================
    log("\nBuilding features...")
    train_samples = prepare_samples(train_df, is_train=True, augment=True)
    Xall, ya_all, yp_all, uids_all, _ = build_lgb_features(
        train_df, is_train=True, augment=True)
    log(f"  Total samples: {len(Xall)}, features: {Xall.shape[1]}")

    # Compute prefix_len in sample order
    sample_prefix_len = np.empty(len(train_samples), dtype=np.int32)
    seen = {}
    for i, s in enumerate(train_samples):
        u = s["uid"]
        seen[u] = seen.get(u, 0) + 1
        sample_prefix_len[i] = seen[u]
    L1_mask_sample = sample_prefix_len == 1
    L1_n = L1_mask_sample.sum()
    log(f"  L=1 samples: {L1_n}")

    # ============================================================
    # 2. Existing global LGB OOF on L=1 (read from artifacts, bag2 average)
    # ============================================================
    log("\n" + "─"*78)
    log("PART 1: Existing global LGB on L=1 OOF (bag of s42 + s1337)")
    log("─"*78)

    oof_s42 = np.load('artifacts/v5z_md_full_s42_oof.npz', allow_pickle=True)
    oof_s1337 = np.load('artifacts/v5z_md_full_s1337_oof.npz', allow_pickle=True)

    # Reindex per seed to sample order (same as Stage A)
    uid_list = sorted({s["uid"] for s in train_samples})
    uid_arr = np.array([s["target_action"] for s in train_samples
                        if s["uid"] in set(uid_list)])  # placeholder, recomputed below
    uid_last_a = {s["uid"]: s["target_action"] for s in train_samples}
    uid_arr = np.array([uid_last_a[u] for u in uid_list])
    uid_to_match = train_df.groupby("rally_uid")["match"].first().to_dict()
    match_arr = np.array([uid_to_match[u] for u in uid_list])

    uid2idx = {}
    for i, s in enumerate(train_samples):
        uid2idx.setdefault(s["uid"], []).append(i)

    def reindex_for_seed(seed):
        sgkf = StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=seed)
        s2o = np.empty(len(train_samples), dtype=np.int64)
        cur = 0
        for _, va_uidx in sgkf.split(uid_list, uid_arr, groups=match_arr):
            va_uids = set(np.array(uid_list)[va_uidx])
            for u in va_uids:
                for i in uid2idx.get(u, []):
                    s2o[i] = cur
                    cur += 1
        return s2o

    s2o_42 = reindex_for_seed(42)
    s2o_1337 = reindex_for_seed(1337)

    # d['lgb_a'][s2o_seed[i]] = sample i's prediction
    lgb_a_42 = oof_s42['lgb_a'][s2o_42]   # (N, 19) sample-order
    lgb_p_42 = oof_s42['lgb_p'][s2o_42]
    lgb_a_13 = oof_s1337['lgb_a'][s2o_1337]
    lgb_p_13 = oof_s1337['lgb_p'][s2o_1337]
    la_sample = oof_s42['la'][s2o_42]
    lp_sample = oof_s42['lp'][s2o_42]
    assert (la_sample == oof_s1337['la'][s2o_1337]).all(), "label mismatch!"

    # Bag2 LGB = mean across seeds
    lgb_a_bag2 = (lgb_a_42 + lgb_a_13) / 2
    lgb_p_bag2 = (lgb_p_42 + lgb_p_13) / 2

    # L=1 slice predictions
    pred_a_existing = lgb_a_bag2[L1_mask_sample].argmax(axis=1)
    pred_p_existing = lgb_p_bag2[L1_mask_sample].argmax(axis=1)
    la_L1 = la_sample[L1_mask_sample]
    lp_L1 = lp_sample[L1_mask_sample]

    f1a_existing = f1_score(la_L1, pred_a_existing, average='macro', zero_division=0)
    f1p_existing = f1_score(lp_L1, pred_p_existing, average='macro', zero_division=0)
    log(f"\n  Existing global LGB on L=1 OOF:")
    log(f"    F1_action = {f1a_existing:.4f}")
    log(f"    F1_point  = {f1p_existing:.4f}")
    log(f"    avg       = {(f1a_existing + f1p_existing)/2:.4f}")

    # ============================================================
    # 3. Slice-trained LGB on L=1 only (5-fold match-disjoint, matched hparams)
    # ============================================================
    log("\n" + "─"*78)
    log("PART 2: Slice-trained LGB on L=1 only (matched hyperparameters)")
    log("─"*78)

    # Match-disjoint folds on full uid_list (same as base) — but train/eval only on L=1
    sgkf = StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=42)
    fold_splits = list(sgkf.split(uid_list, uid_arr, groups=match_arr))

    # OOF predictions for L=1 samples only (will be filled per-fold)
    oof_a_slice = np.full((L1_n, N_ACTION), np.nan, dtype=np.float64)
    oof_p_slice = np.full((L1_n, N_POINT), np.nan, dtype=np.float64)
    L1_sample_indices = np.where(L1_mask_sample)[0]
    sample_idx_to_L1_idx = {sample_idx: l1_idx for l1_idx, sample_idx in enumerate(L1_sample_indices)}

    for fold, (tr_uidx, va_uidx) in enumerate(fold_splits):
        log(f"\n  Fold {fold+1}/{N_FOLDS}")
        tr_uids = set(np.array(uid_list)[tr_uidx])
        va_uids = set(np.array(uid_list)[va_uidx])
        tr_idx_all = [i for u in tr_uids for i in uid2idx.get(u, [])]
        va_idx_all = [i for u in va_uids for i in uid2idx.get(u, [])]
        # Restrict to L=1 only
        tr_idx_L1 = [i for i in tr_idx_all if sample_prefix_len[i] == 1]
        va_idx_L1 = [i for i in va_idx_all if sample_prefix_len[i] == 1]

        log(f"    train L=1 samples: {len(tr_idx_L1)}, val L=1 samples: {len(va_idx_L1)}")
        if len(tr_idx_L1) < 100 or len(va_idx_L1) < 10:
            log(f"    SKIP fold {fold+1} (too few L=1 samples)")
            continue

        X_tr = Xall.iloc[tr_idx_L1]
        X_va = Xall.iloc[va_idx_L1]
        ya_tr = [ya_all[i] for i in tr_idx_L1]
        ya_va = [ya_all[i] for i in va_idx_L1]
        yp_tr = [yp_all[i] for i in tr_idx_L1]
        yp_va = [yp_all[i] for i in va_idx_L1]

        t0 = time.time()
        _, pa = train_lgb_slice(X_tr, ya_tr, X_va, ya_va, N_ACTION, seed=42)
        _, pp = train_lgb_slice(X_tr, yp_tr, X_va, yp_va, N_POINT, seed=42)
        log(f"    slice-LGB trained ({time.time()-t0:.1f}s)")

        # Store predictions
        for j, sample_idx in enumerate(va_idx_L1):
            l1_idx = sample_idx_to_L1_idx[sample_idx]
            oof_a_slice[l1_idx] = pa[j]
            oof_p_slice[l1_idx] = pp[j]

    # Also run with seed=1337 for fair bag2 comparison
    log("\n" + "─"*78)
    log("PART 2b: Slice-LGB second seed (1337) for bag2 comparison")
    log("─"*78)
    sgkf_b = StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=1337)
    fold_splits_b = list(sgkf_b.split(uid_list, uid_arr, groups=match_arr))

    oof_a_slice_b = np.full((L1_n, N_ACTION), np.nan, dtype=np.float64)
    oof_p_slice_b = np.full((L1_n, N_POINT), np.nan, dtype=np.float64)

    for fold, (tr_uidx, va_uidx) in enumerate(fold_splits_b):
        log(f"\n  Fold {fold+1}/{N_FOLDS} (seed 1337)")
        tr_uids = set(np.array(uid_list)[tr_uidx])
        va_uids = set(np.array(uid_list)[va_uidx])
        tr_idx_all = [i for u in tr_uids for i in uid2idx.get(u, [])]
        va_idx_all = [i for u in va_uids for i in uid2idx.get(u, [])]
        tr_idx_L1 = [i for i in tr_idx_all if sample_prefix_len[i] == 1]
        va_idx_L1 = [i for i in va_idx_all if sample_prefix_len[i] == 1]
        if len(tr_idx_L1) < 100 or len(va_idx_L1) < 10:
            continue
        X_tr = Xall.iloc[tr_idx_L1]
        X_va = Xall.iloc[va_idx_L1]
        ya_tr = [ya_all[i] for i in tr_idx_L1]
        ya_va = [ya_all[i] for i in va_idx_L1]
        yp_tr = [yp_all[i] for i in tr_idx_L1]
        yp_va = [yp_all[i] for i in va_idx_L1]
        _, pa = train_lgb_slice(X_tr, ya_tr, X_va, ya_va, N_ACTION, seed=1337)
        _, pp = train_lgb_slice(X_tr, yp_tr, X_va, yp_va, N_POINT, seed=1337)
        for j, sample_idx in enumerate(va_idx_L1):
            l1_idx = sample_idx_to_L1_idx[sample_idx]
            oof_a_slice_b[l1_idx] = pa[j]
            oof_p_slice_b[l1_idx] = pp[j]

    # Bag of slice predictions across seeds
    oof_a_slice_bag = (oof_a_slice + oof_a_slice_b) / 2
    oof_p_slice_bag = (oof_p_slice + oof_p_slice_b) / 2

    pred_a_slice = oof_a_slice_bag.argmax(axis=1)
    pred_p_slice = oof_p_slice_bag.argmax(axis=1)
    f1a_slice = f1_score(la_L1, pred_a_slice, average='macro', zero_division=0)
    f1p_slice = f1_score(lp_L1, pred_p_slice, average='macro', zero_division=0)
    log(f"\n  Slice-LGB bag2 on L=1 OOF:")
    log(f"    F1_action = {f1a_slice:.4f}")
    log(f"    F1_point  = {f1p_slice:.4f}")
    log(f"    avg       = {(f1a_slice + f1p_slice)/2:.4f}")

    # ============================================================
    # 4. Verdict
    # ============================================================
    delta_a = f1a_slice - f1a_existing
    delta_p = f1p_slice - f1p_existing
    delta_avg = (delta_a + delta_p) / 2

    log("\n" + "="*78)
    log("ATTACK 4 VERDICT")
    log("="*78)
    log(f"\n  Action: existing={f1a_existing:.4f} | slice={f1a_slice:.4f} | Δ={delta_a:+.4f}")
    log(f"  Point:  existing={f1p_existing:.4f} | slice={f1p_slice:.4f} | Δ={delta_p:+.4f}")
    log(f"  Avg Δ:  {delta_avg:+.4f}")

    log("\n" + "─"*78)
    log("DECISION")
    log("─"*78)
    if delta_a < 0.003 and delta_p < 0.003:
        log(f"\n  ❌ KILL P3 — Δ_action={delta_a:+.4f} < +0.003 AND Δ_point={delta_p:+.4f} < +0.003")
        log(f"  → Slice-LGB is not meaningfully better than existing global LGB on L=1")
        log(f"  → Expert A premise is cargo culting — same architecture/hyperparameters,")
        log(f"     just trained on smaller subset. The +0.0588 Stage A gap is NOT recoverable")
        log(f"     by slice training; it's intrinsic to the macro-F1 + L=1 regime.")
    elif delta_avg >= 0.005:
        log(f"\n  ✅ PROCEED with modified P3 (per Debate#1 Agent 2 recommendation)")
        log(f"  → Δ_avg={delta_avg:+.4f} ≥ +0.005 — slice training adds genuine signal")
        log(f"  → Move to: stacking instead of replacement (Attack 2 mod)")
        log(f"           : remove per-server prior (Attack 6 KILL component)")
        log(f"           : restore +0.010 ship gate (Attack 7 mod)")
    else:
        log(f"\n  ⚠️  MARGINAL — Δ_avg={delta_avg:+.4f} ∈ [0.003, 0.005)")
        log(f"  → Borderline. Test-weighted projection: 0.32 × {delta_avg:.3f} = +{0.32*delta_avg:.4f}")
        log(f"  → Below +0.010 ship gate even at full lift recovery")

    # Save artifacts
    np.savez('artifacts/probe_p3_attack4.npz',
             f1a_existing=f1a_existing, f1p_existing=f1p_existing,
             f1a_slice=f1a_slice, f1p_slice=f1p_slice,
             delta_a=delta_a, delta_p=delta_p,
             oof_a_slice_bag=oof_a_slice_bag, oof_p_slice_bag=oof_p_slice_bag,
             la_L1=la_L1, lp_L1=lp_L1)
    log(f"\n  Artifacts saved to artifacts/probe_p3_attack4.npz")


if __name__ == "__main__":
    main()

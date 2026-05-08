#!/usr/bin/env python3
"""
Phase + EFOS + AOEL LGB retrain on match-disjoint folds.

Adds 10 features grounded in table tennis tactical literature:
- Wu Huanqun three-phase (STAS/RTAS/Stalemate)
- Wu Lames EFOS (first offensive stroke index, strokes since FOS)
- Wu 2022 AOEL (half-long ball flag, count)
- Side counts (server vs receiver strokes in prefix)

Usage:
    V5Z_PHASE=1 V5Z_SEED=42   python3 src/train/lgb_md_phase.py
    V5Z_PHASE=1 V5Z_SEED=1337 python3 src/train/lgb_md_phase.py

Output:
    artifacts/v5z_mdph_full_s{SEED}_oof.npz   (LSTM reused, LGB new with phase features)
    artifacts/v5z_mdph_full_s{SEED}_test.npz
"""
import os
import pathlib
os.chdir(pathlib.Path(__file__).resolve().parents[2])

os.environ["V5Z_PHASE"] = "1"

import sys
import time
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, "src/train")
from train_v5z import (
    build_lgb_features, train_df, test_df,
    prepare_samples, N_FOLDS, SEED, train_lgb_fold,
    N_ACTION, N_POINT,
)


def log(msg):
    print(msg, flush=True)


def main():
    out_tag = f"v5z_mdph_full_s{SEED}"
    log("\n" + "="*78)
    log(f"Phase+EFOS+AOEL LGB retrain (match-disjoint), seed={SEED}")
    log("="*78)

    src_oof = f"artifacts/v5z_md_full_s{SEED}_oof.npz"
    src_test = f"artifacts/v5z_md_full_s{SEED}_test.npz"
    assert os.path.exists(src_oof), f"missing {src_oof}"
    log(f"\nReusing match-disjoint LSTM artifacts: {src_oof}")
    src_oof_d = np.load(src_oof)
    src_test_d = np.load(src_test)

    log("\nBuilding LGB features (V5Z_PHASE=1)...")
    t0 = time.time()
    lgb_X_train, lgb_y_a, lgb_y_p, lgb_uids, _ = build_lgb_features(
        train_df, is_train=True, augment=True)
    lgb_X_test, _, _, lgb_test_uids, _ = build_lgb_features(
        test_df, is_train=False)
    log(f"  features: {lgb_X_train.shape[1]} cols, {time.time()-t0:.1f}s")

    new_cols = [c for c in lgb_X_train.columns if c in {
        "next_phase","next_in_initial_offensive","fos_idx","strokes_since_fos",
        "prefix_has_fos","last_was_half_long","prefix_half_long_count",
        "next_side_is_server","server_strokes_count","receiver_strokes_count"
    }]
    log(f"  NEW phase columns ({len(new_cols)}): {new_cols}")
    assert len(new_cols) == 10, f"expected 10 new cols, got {len(new_cols)}: {new_cols}"

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

    log("\nTraining LGB (5 folds, match-disjoint, with phase+EFOS+AOEL features)")
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

    new_lgb_a = np.vstack(all_oof_a)
    new_lgb_p = np.vstack(all_oof_p)
    test_lgb_a = np.mean(test_a_l, 0)
    test_lgb_p = np.mean(test_p_l, 0)

    assert new_lgb_a.shape == src_oof_d["lgb_a"].shape

    out_oof = f"artifacts/{out_tag}_oof.npz"
    out_test = f"artifacts/{out_tag}_test.npz"
    np.savez(
        out_oof,
        lstm_a=src_oof_d["lstm_a"], lstm_p=src_oof_d["lstm_p"],
        lgb_a=new_lgb_a, lgb_p=new_lgb_p,
        la=src_oof_d["la"], lp=src_oof_d["lp"],
    )
    np.savez(
        out_test,
        lstm_a=src_test_d["lstm_a"], lstm_p=src_test_d["lstm_p"],
        lgb_a=test_lgb_a, lgb_p=test_lgb_p,
        test_uids=src_test_d["test_uids"], test_sgp=src_test_d["test_sgp"],
    )
    log(f"\nSaved: {out_oof}")
    log(f"Saved: {out_test}")


if __name__ == "__main__":
    main()

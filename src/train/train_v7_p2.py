#!/usr/bin/env python3
"""
V7 P2: LGB feature expansion
============================================================
Adds to base 49 LGB features:
  A) Cross-task stacking: LSTM OOF action probs (19) + point probs (10) as features.
     Fold-safe by construction (v7_oof.npz is OOF). Test uses 15-model LSTM avg.
  B) Momentum/streak features (6): consecutive same action-group run length,
     context action entropy, last-3 counts (attack/control/defense),
     point-side alternation rate.
  C) OOF target encoding on key=(last_action * 11 + last_point):
     29 columns per model (19 for action head, 10 for point head), α=20 smoothing,
     5-fold OOF (uses existing StratifiedKFold splits).

Outputs:
  artifacts/v7_p2_oof.npz  (new LGB OOF + existing LSTM OOF + labels)
  artifacts/v7_p2_test.npz (LSTM + LGB test probs)
  artifacts/v7_p2_params.npz (calibration params)
  submissions/submission_v7_p2.csv
"""
import os
import pathlib
os.chdir(pathlib.Path(__file__).resolve().parents[2])

import sys
import argparse
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from collections import Counter, defaultdict

sys.path.insert(0, "src/train")
from train_v5 import (
    SEED, BATCH_SIZE, N_FOLDS, N_ACTION, N_POINT, DEVICE,
    ACTION_GROUPS, POINT_SIDE,
    train_df, test_df, prepare_samples, build_lgb_features,
    RallyDataset, PingPongModel, train_lgb_fold, log,
)
from run_v7_p1 import macro_f1_fast, plugin_add

SEEDS = [42, 1337, 2024]
LSTM_PATHS = {
    42:   "models/v5/model_v5_lstm_fold{fold}.pt",
    1337: "models/v7/model_v7_seed1337_fold{fold}.pt",
    2024: "models/v7/model_v7_seed2024_fold{fold}.pt",
}
TE_ALPHA = 20.0  # smoothing strength for target encoding


# ============================================================
# Momentum features from raw train/test df
# ============================================================
def momentum_features(rows_ctx):
    """rows_ctx: list of strike dicts (the context)."""
    k = len(rows_ctx)
    # consecutive same action-group run length (ending at last shot)
    last_g = ACTION_GROUPS.get(rows_ctx[-1]["actionId"], 0)
    run = 1
    for r in reversed(rows_ctx[:-1]):
        if ACTION_GROUPS.get(r["actionId"], 0) == last_g:
            run += 1
        else:
            break

    # action entropy over ctx
    ag_counts = Counter(ACTION_GROUPS.get(r["actionId"], 0) for r in rows_ctx)
    total = sum(ag_counts.values())
    ent = 0.0
    for v in ag_counts.values():
        p = v / total
        if p > 0:
            ent -= p * np.log(p)

    # last-3 shot group counts
    last3 = rows_ctx[-3:]
    l3_attack = sum(1 for r in last3 if ACTION_GROUPS.get(r["actionId"], 0) == 1)
    l3_control = sum(1 for r in last3 if ACTION_GROUPS.get(r["actionId"], 0) == 2)
    l3_defense = sum(1 for r in last3 if ACTION_GROUPS.get(r["actionId"], 0) == 3)

    # point-side alternation rate
    if k < 2:
        alt = 0.0
    else:
        alts = 0
        for i in range(1, k):
            s1 = POINT_SIDE.get(rows_ctx[i]["pointId"], 0)
            s2 = POINT_SIDE.get(rows_ctx[i-1]["pointId"], 0)
            if s1 != 0 and s2 != 0 and s1 != s2:
                alts += 1
        alt = alts / (k - 1)

    return {
        "mom_same_group_run": run,
        "mom_action_entropy": ent,
        "mom_l3_attack": l3_attack,
        "mom_l3_control": l3_control,
        "mom_l3_defense": l3_defense,
        "mom_side_alternation": alt,
    }


def build_momentum_df(df, is_train=True, augment=True):
    feats = []
    for uid, grp in df.groupby("rally_uid"):
        grp = grp.sort_values("strikeNumber")
        rows = grp.to_dict("records")
        N = len(rows)
        if is_train:
            start_k = 1 if augment else max(1, N - 1)
            for k in range(start_k, N):
                feats.append(momentum_features(rows[:k]))
        else:
            feats.append(momentum_features(rows))
    return pd.DataFrame(feats)


# ============================================================
# Target encoding on (last_action, last_point)
# ============================================================
def compute_te_train(keys, y, n_class, sample_fold, n_folds, alpha=TE_ALPHA):
    """OOF TE: for each fold, compute lookup from OOF training portion and apply to val."""
    y = np.asarray(y, dtype=np.int64)
    keys = np.asarray(keys, dtype=np.int64)
    sample_fold = np.asarray(sample_fold, dtype=np.int64)
    N = len(y)
    te = np.zeros((N, n_class), dtype=np.float32)
    global_p = np.bincount(y, minlength=n_class).astype(np.float64) / N

    for f in range(n_folds):
        tr_mask = sample_fold != f
        va_mask = sample_fold == f
        sums = {}
        counts = {}
        for k, yi in zip(keys[tr_mask], y[tr_mask]):
            if k not in sums:
                sums[k] = np.zeros(n_class, dtype=np.float64)
                counts[k] = 0
            sums[k][yi] += 1
            counts[k] += 1
        for idx in np.where(va_mask)[0]:
            k = int(keys[idx])
            n = counts.get(k, 0)
            if n == 0:
                te[idx] = global_p
            else:
                te[idx] = (sums[k] + alpha * global_p) / (n + alpha)
    return te


def compute_te_test(keys_test, keys_train, y_train, n_class, alpha=TE_ALPHA):
    y_train = np.asarray(y_train, dtype=np.int64)
    keys_train = np.asarray(keys_train, dtype=np.int64)
    keys_test = np.asarray(keys_test, dtype=np.int64)
    global_p = np.bincount(y_train, minlength=n_class).astype(np.float64) / len(y_train)
    sums = {}
    counts = {}
    for k, yi in zip(keys_train, y_train):
        if k not in sums:
            sums[k] = np.zeros(n_class, dtype=np.float64)
            counts[k] = 0
        sums[k][yi] += 1
        counts[k] += 1
    N = len(keys_test)
    te = np.zeros((N, n_class), dtype=np.float32)
    for i in range(N):
        k = int(keys_test[i])
        n = counts.get(k, 0)
        if n == 0:
            te[i] = global_p
        else:
            te[i] = (sums[k] + alpha * global_p) / (n + alpha)
    return te


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="xtask,mom,te",
                    help="Comma-separated subset of {xtask,mom,te}")
    ap.add_argument("--tag", default="v7_p2",
                    help="Artifact name prefix (e.g. v7_p2, v7_p2b)")
    args = ap.parse_args()
    use_xt = "xtask" in args.features.split(",")
    use_mom = "mom" in args.features.split(",")
    use_te = "te" in args.features.split(",")
    tag = args.tag

    t_all = time.time()
    log("=" * 70)
    log(f"V7 P2 · feature ablation  tag={tag}  xtask={use_xt} mom={use_mom} te={use_te}")
    log("=" * 70)

    # ---- 1. Build samples in rally order (1:1 with build_lgb_features) ----
    log("\n[1] Preparing samples...")
    t0 = time.time()
    train_samples = prepare_samples(train_df, is_train=True, augment=True)
    test_samples = prepare_samples(test_df, is_train=False)
    log(f"    train={len(train_samples)}, test={len(test_samples)} ({time.time()-t0:.1f}s)")

    # ---- 2. Rebuild fold iteration to derive sample_fold + OOF permutation ----
    log("\n[2] CV split + OOF alignment...")
    uid2idx = {}
    for i, s in enumerate(train_samples):
        uid2idx.setdefault(s["uid"], []).append(i)
    uid_last_a = {s["uid"]: s["target_action"] for s in train_samples}
    uid_list = sorted(uid_last_a.keys())
    uid_arr = np.array([uid_last_a[u] for u in uid_list])
    kf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    kf_splits = list(kf.split(uid_list, uid_arr))

    sample_fold = np.empty(len(train_samples), dtype=np.int64)
    sample_idx_to_oof_idx = np.empty(len(train_samples), dtype=np.int64)
    oof_cursor = 0
    for fold, (_, va_uidx) in enumerate(kf_splits):
        va_uids = set(np.array(uid_list)[va_uidx])
        for u in va_uids:
            for i in uid2idx.get(u, []):
                sample_fold[i] = fold
                sample_idx_to_oof_idx[i] = oof_cursor
                oof_cursor += 1
    assert oof_cursor == len(train_samples)
    log(f"    sample_fold built; {oof_cursor} samples mapped to OOF order")

    # ---- 3. Load v7_oof (OOF order) + labels ----
    log("\n[3] Loading v7_oof.npz for cross-task + labels...")
    oof = np.load("artifacts/v7_oof.npz")
    lstm_a_oof = oof["lstm_a"]      # (N, 19) OOF order
    lstm_p_oof = oof["lstm_p"]      # (N, 10)
    la = oof["la"].astype(np.int64)  # (N,) action labels OOF order
    lp = oof["lp"].astype(np.int64)

    # Reindex OOF → sample order: result[i] = oof_array[sample_idx_to_oof_idx[i]]
    lstm_a_smp = lstm_a_oof[sample_idx_to_oof_idx]
    lstm_p_smp = lstm_p_oof[sample_idx_to_oof_idx]
    la_smp = la[sample_idx_to_oof_idx]
    lp_smp = lp[sample_idx_to_oof_idx]

    # Sanity: recover target labels from samples directly, should match
    tgt_a = np.array([s["target_action"] for s in train_samples], dtype=np.int64)
    tgt_p = np.array([s["target_point"] for s in train_samples], dtype=np.int64)
    assert np.array_equal(tgt_a, la_smp), "action labels mismatch"
    assert np.array_equal(tgt_p, lp_smp), "point labels mismatch"
    log("    OOF → sample reindex verified (labels match)")

    # ---- 4. Base LGB features (rally order) ----
    log("\n[4] Building base LGB features...")
    t0 = time.time()
    lgb_base_train, lgb_ya, lgb_yp, lgb_uids, _ = build_lgb_features(
        train_df, is_train=True, augment=True)
    lgb_base_test, _, _, test_uids, test_sgp = build_lgb_features(
        test_df, is_train=False)
    log(f"    base: train={lgb_base_train.shape} test={lgb_base_test.shape} ({time.time()-t0:.1f}s)")

    # ---- 5. Momentum features ----
    log("\n[5] Momentum features...")
    t0 = time.time()
    mom_train = build_momentum_df(train_df, is_train=True, augment=True)
    mom_test = build_momentum_df(test_df, is_train=False)
    log(f"    mom: train={mom_train.shape} test={mom_test.shape} ({time.time()-t0:.1f}s)")

    # ---- 6. Cross-task stacking features ----
    # Train: lstm_a_smp, lstm_p_smp (rally order, OOF-safe)
    # Test: needs 15-model LSTM test avg
    log("\n[6] LSTM test predictions (15 models)...")
    n_test = len(test_samples)
    lstm_test_a = np.zeros((n_test, N_ACTION), dtype=np.float64)
    lstm_test_p = np.zeros((n_test, N_POINT), dtype=np.float64)
    test_dl = DataLoader(RallyDataset(test_samples, is_train=False), BATCH_SIZE,
                         shuffle=False, num_workers=0, pin_memory=True)
    n_models = len(SEEDS) * N_FOLDS
    t0 = time.time()
    for seed in SEEDS:
        for fold in range(N_FOLDS):
            path = LSTM_PATHS[seed].format(fold=fold + 1)
            model = PingPongModel(player_drop_p=0).to(DEVICE)
            model.load_state_dict(torch.load(path, map_location=DEVICE))
            model.eval()
            offset = 0
            with torch.no_grad():
                for batch in test_dl:
                    sc = batch["seq_cat"].to(DEVICE)
                    sn = batch["seq_num"].to(DEVICE)
                    ps, pr, pn = batch["pid_server"], batch["pid_receiver"], batch["pid_next"]
                    st = batch["static"].to(DEVICE)
                    lens = batch["length"]
                    nsns = batch["next_sn"]
                    a_log, p_log = model(sc, sn, ps, pr, pn, st, lens,
                                         next_sns=nsns, apply_mask=True)
                    bs = sc.size(0)
                    lstm_test_a[offset:offset+bs] += F.softmax(a_log, dim=-1).cpu().numpy() / n_models
                    lstm_test_p[offset:offset+bs] += F.softmax(p_log, dim=-1).cpu().numpy() / n_models
                    offset += bs
    log(f"    LSTM test preds done ({time.time()-t0:.1f}s)")

    # ---- 7. TE on (last_action, last_point) ----
    log("\n[7] Target encoding on (last_action, last_point) ...")
    # key = last_action * 11 + last_point   (pointId has 10 values, pad to 11)
    key_train = (lgb_base_train["last_actionId"].values.astype(np.int64) * 11
                 + lgb_base_train["last_pointId"].values.astype(np.int64))
    key_test = (lgb_base_test["last_actionId"].values.astype(np.int64) * 11
                + lgb_base_test["last_pointId"].values.astype(np.int64))
    y_a = np.array(lgb_ya, dtype=np.int64)
    y_p = np.array(lgb_yp, dtype=np.int64)
    t0 = time.time()
    te_action_train = compute_te_train(key_train, y_a, N_ACTION, sample_fold, N_FOLDS)
    te_point_train = compute_te_train(key_train, y_p, N_POINT, sample_fold, N_FOLDS)
    te_action_test = compute_te_test(key_test, key_train, y_a, N_ACTION)
    te_point_test = compute_te_test(key_test, key_train, y_p, N_POINT)
    log(f"    TE shape: action={te_action_train.shape} point={te_point_train.shape} "
        f"({time.time()-t0:.1f}s)")

    # ---- 8. Assemble enhanced feature DataFrames ----
    log("\n[8] Assembling enhanced feature frames...")
    def _xtask(arr, prefix, n):
        return pd.DataFrame(arr, columns=[f"{prefix}_{c}" for c in range(n)])

    xt_la_tr = _xtask(lstm_a_smp, "xt_la", N_ACTION)
    xt_lp_tr = _xtask(lstm_p_smp, "xt_lp", N_POINT)
    xt_la_te = _xtask(lstm_test_a, "xt_la", N_ACTION)
    xt_lp_te = _xtask(lstm_test_p, "xt_lp", N_POINT)

    te_a_tr = _xtask(te_action_train, "te_a", N_ACTION)
    te_p_tr = _xtask(te_point_train, "te_p", N_POINT)
    te_a_te = _xtask(te_action_test, "te_a", N_ACTION)
    te_p_te = _xtask(te_point_test, "te_p", N_POINT)

    # Concatenate (ordering: base | momentum | xtask | te)
    # For action-target model, include te_a (action TE); for point-target, te_p.
    # Both models get full xtask (LSTM's view of both tasks).
    def _concat(*parts):
        return pd.concat([p.reset_index(drop=True) for p in parts], axis=1)

    a_train_parts = [lgb_base_train]
    a_test_parts = [lgb_base_test]
    p_train_parts = [lgb_base_train]
    p_test_parts = [lgb_base_test]
    if use_mom:
        a_train_parts.append(mom_train); a_test_parts.append(mom_test)
        p_train_parts.append(mom_train); p_test_parts.append(mom_test)
    if use_xt:
        a_train_parts.extend([xt_la_tr, xt_lp_tr]); a_test_parts.extend([xt_la_te, xt_lp_te])
        p_train_parts.extend([xt_la_tr, xt_lp_tr]); p_test_parts.extend([xt_la_te, xt_lp_te])
    if use_te:
        a_train_parts.append(te_a_tr); a_test_parts.append(te_a_te)
        p_train_parts.append(te_p_tr); p_test_parts.append(te_p_te)

    X_action_train = _concat(*a_train_parts)
    X_action_test = _concat(*a_test_parts)
    X_point_train = _concat(*p_train_parts)
    X_point_test = _concat(*p_test_parts)
    log(f"    action model: train={X_action_train.shape} test={X_action_test.shape}")
    log(f"    point  model: train={X_point_train.shape} test={X_point_test.shape}")

    # ---- 9. Train 5-fold LGB ----
    log("\n[9] Training 5-fold LGB (action + point) on enhanced features...")
    lgb_oof_a = np.zeros((len(train_samples), N_ACTION), dtype=np.float64)
    lgb_oof_p = np.zeros((len(train_samples), N_POINT), dtype=np.float64)
    lgb_test_a = np.zeros((n_test, N_ACTION), dtype=np.float64)
    lgb_test_p = np.zeros((n_test, N_POINT), dtype=np.float64)

    for fold in range(N_FOLDS):
        va_idx = np.where(sample_fold == fold)[0]
        tr_idx = np.where(sample_fold != fold)[0]
        log(f"\n  fold {fold+1}/{N_FOLDS}  train={len(tr_idx)}  val={len(va_idx)}")

        Xa_tr = X_action_train.iloc[tr_idx]
        Xa_va = X_action_train.iloc[va_idx]
        Xp_tr = X_point_train.iloc[tr_idx]
        Xp_va = X_point_train.iloc[va_idx]
        ya_tr = y_a[tr_idx].tolist()
        ya_va = y_a[va_idx].tolist()
        yp_tr = y_p[tr_idx].tolist()
        yp_va = y_p[va_idx].tolist()

        t0 = time.time()
        a_model, _, _ = train_lgb_fold(Xa_tr, ya_tr, Xa_va, ya_va, N_ACTION, "actionId")
        lgb_oof_a[va_idx] = a_model.predict(Xa_va)
        lgb_test_a += a_model.predict(X_action_test) / N_FOLDS
        log(f"    action LGB done ({time.time()-t0:.1f}s)")

        t0 = time.time()
        p_model, _, _ = train_lgb_fold(Xp_tr, yp_tr, Xp_va, yp_va, N_POINT, "pointId")
        lgb_oof_p[va_idx] = p_model.predict(Xp_va)
        lgb_test_p += p_model.predict(X_point_test) / N_FOLDS
        log(f"    point  LGB done ({time.time()-t0:.1f}s)")

    # ---- 10. Evaluate raw OOF ----
    log(f"\n[10] Raw OOF (no calibration) ...")
    fa_base = f1_score(y_a, lgb_oof_a.argmax(1), average="macro", zero_division=0)
    fp_base = f1_score(y_p, lgb_oof_p.argmax(1), average="macro", zero_division=0)
    log(f"    P2 LGB F1_a={fa_base:.4f}  F1_p={fp_base:.4f}")
    # V7 LGB baseline was 0.3903 / 0.2523
    log(f"    V7 LGB base F1_a=0.3903  F1_p=0.2523")

    # ---- 11. Calibration: search per-task w on (lstm_bag, new_lgb), then additive bias ----
    # Convert to OOF order for compatibility with run_v7_cal.macro_f1_fast
    # Actually plugin_add takes any ordering; we'll feed in sample-order arrays.
    log(f"\n[11] Calibration search on bagged-LSTM + P2-LGB ensemble...")
    def search_w(lstm, lgb, labels, n_class, step=0.05, n_rounds=5):
        best = (-1.0, 0.0, None)
        for w in np.arange(0.0, 1.0001, step):
            ens = w * lstm + (1 - w) * lgb
            bias, f1 = plugin_add(ens, labels, n_class, n_rounds=n_rounds)
            if f1 > best[0]:
                best = (f1, float(w), bias)
        return best

    t0 = time.time()
    f1a, wa, bia = search_w(lstm_a_smp, lgb_oof_a, y_a, N_ACTION)
    log(f"    action: w_a={wa:.2f} F1_a={f1a:.4f} ({time.time()-t0:.1f}s)")
    t0 = time.time()
    f1p, wp, bip = search_w(lstm_p_smp, lgb_oof_p, y_p, N_POINT)
    log(f"    point:  w_p={wp:.2f} F1_p={f1p:.4f} ({time.time()-t0:.1f}s)")

    score = 0.4 * f1a + 0.4 * f1p + 0.2
    log(f"\n{'='*70}")
    log(f"V7 P2 CV Score: {score:.4f}  (F1_a={f1a:.4f}, F1_p={f1p:.4f})")
    log(f"V7 baseline:    0.4883 (F1_a=0.4387, F1_p=0.2820)")
    log(f"Δ:              {score - 0.4883:+.4f}")
    log(f"{'='*70}")

    # ---- 12. Save artifacts + generate submission ----
    log("\n[12] Saving artifacts + submission...")
    np.savez(f"artifacts/{tag}_oof.npz",
             lstm_a=lstm_a_smp, lstm_p=lstm_p_smp,
             lgb_a=lgb_oof_a, lgb_p=lgb_oof_p,
             la=y_a, lp=y_p,
             sample_fold=sample_fold)
    np.savez(f"artifacts/{tag}_test.npz",
             lstm_a=lstm_test_a, lstm_p=lstm_test_p,
             lgb_a=lgb_test_a, lgb_p=lgb_test_p)
    np.savez(f"artifacts/{tag}_params.npz",
             method="additive",
             w_a=np.float64(wa), w_p=np.float64(wp),
             bias_a=bia, bias_p=bip)

    ens_a_test = wa * lstm_test_a + (1 - wa) * lgb_test_a
    ens_p_test = wp * lstm_test_p + (1 - wp) * lgb_test_p
    log_a = np.log(ens_a_test + 1e-12) + bia
    log_p = np.log(ens_p_test + 1e-12) + bip
    pred_action = log_a.argmax(1)
    pred_point = log_p.argmax(1)

    sub = pd.DataFrame({
        "rally_uid": test_uids,
        "actionId": pred_action,
        "pointId": pred_point,
        "serverGetPoint": test_sgp,
    })
    sub = sub.sort_values("rally_uid").reset_index(drop=True)
    sub_path = f"submissions/submission_{tag}.csv"
    sub.to_csv(sub_path, index=False)
    log(f"    {sub_path} ({len(sub)} 筆)")
    log(f"    action dist: {dict(sorted(Counter(pred_action).items()))}")
    log(f"    point  dist: {dict(sorted(Counter(pred_point).items()))}")

    log(f"\nTOTAL: {(time.time()-t_all)/60:.1f} min")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
V6 Phase 1: Plug-in Macro-F1 calibration (additive bias) + per-task ensemble.
不重訓模型，只改決策層。
"""
import os
import sys
import pathlib
_HERE = pathlib.Path(__file__).resolve().parent
os.chdir(_HERE.parent.parent)
sys.path.insert(0, str(_HERE))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import time

from train_v5 import (
    SEED, BATCH_SIZE, N_FOLDS, N_ACTION, N_POINT, DEVICE,
    train_df, prepare_samples, build_lgb_features,
    RallyDataset, PingPongModel, train_lgb_fold,
    log,
)

OOF_CACHE = "artifacts/v5_oof.npz"


def compute_oof():
    log("準備 LSTM samples...")
    t0 = time.time()
    train_samples = prepare_samples(train_df, is_train=True, augment=True)
    log(f"  {len(train_samples)} samples ({time.time()-t0:.1f}s)")

    log("準備 LGB features...")
    t0 = time.time()
    lgb_X, lgb_ya, lgb_yp, lgb_uids, _ = build_lgb_features(
        train_df, is_train=True, augment=True)
    log(f"  {lgb_X.shape} ({time.time()-t0:.1f}s)")

    uid2idx = {}
    for i, s in enumerate(train_samples):
        uid2idx.setdefault(s["uid"], []).append(i)
    lgb_uid2idx = {}
    for i, u in enumerate(lgb_uids):
        lgb_uid2idx.setdefault(u, []).append(i)
    uid_last_a = {}
    for s in train_samples:
        uid_last_a[s["uid"]] = s["target_action"]
    uid_list = sorted(uid_last_a.keys())
    uid_arr = np.array([uid_last_a[u] for u in uid_list])

    kf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)

    oof_lstm_a, oof_lstm_p = [], []
    oof_lgb_a, oof_lgb_p = [], []
    oof_la, oof_lp = [], []

    for fold, (tr_uidx, va_uidx) in enumerate(kf.split(uid_list, uid_arr)):
        log(f"\nFold {fold+1}/{N_FOLDS}")
        tr_uids = set(np.array(uid_list)[tr_uidx])
        va_uids = set(np.array(uid_list)[va_uidx])

        # LSTM
        va_s = [train_samples[i] for u in va_uids for i in uid2idx.get(u, [])]
        model = PingPongModel(player_drop_p=0).to(DEVICE)
        model.load_state_dict(torch.load(
            f"models/v5/model_v5_lstm_fold{fold+1}.pt", map_location=DEVICE))
        model.eval()
        dl = DataLoader(RallyDataset(va_s, is_train=True), BATCH_SIZE,
                        shuffle=False, num_workers=0, pin_memory=True)
        la_probs, lp_probs, la, lp = [], [], [], []
        with torch.no_grad():
            for batch in dl:
                sc = batch["seq_cat"].to(DEVICE)
                sn = batch["seq_num"].to(DEVICE)
                ps, pr, pn = batch["pid_server"], batch["pid_receiver"], batch["pid_next"]
                st = batch["static"].to(DEVICE)
                lens = batch["length"]
                nsns = batch["next_sn"]
                a_log, p_log = model(sc, sn, ps, pr, pn, st, lens,
                                     next_sns=nsns, apply_mask=True)
                la_probs.append(F.softmax(a_log, dim=-1).cpu().numpy())
                lp_probs.append(F.softmax(p_log, dim=-1).cpu().numpy())
                la.extend(batch["target_action"])
                lp.extend(batch["target_point"])
        la_probs = np.vstack(la_probs)
        lp_probs = np.vstack(lp_probs)
        la = np.array(la)
        lp = np.array(lp)

        # LGB
        lgb_tr = [i for u in tr_uids for i in lgb_uid2idx.get(u, [])]
        lgb_va = [i for u in va_uids for i in lgb_uid2idx.get(u, [])]
        X_tr, X_va = lgb_X.iloc[lgb_tr], lgb_X.iloc[lgb_va]
        y_a_tr = [lgb_ya[i] for i in lgb_tr]
        y_a_va = [lgb_ya[i] for i in lgb_va]
        y_p_tr = [lgb_yp[i] for i in lgb_tr]
        y_p_va = [lgb_yp[i] for i in lgb_va]
        _, ga_p, _ = train_lgb_fold(X_tr, y_a_tr, X_va, y_a_va, N_ACTION, "actionId")
        _, gp_p, _ = train_lgb_fold(X_tr, y_p_tr, X_va, y_p_va, N_POINT, "pointId")

        log(f"  fold{fold+1} shapes: lstm={la_probs.shape} lgb={ga_p.shape}")

        oof_lstm_a.append(la_probs)
        oof_lstm_p.append(lp_probs)
        oof_lgb_a.append(ga_p)
        oof_lgb_p.append(gp_p)
        oof_la.append(la)
        oof_lp.append(lp)

    out = dict(
        lstm_a=np.vstack(oof_lstm_a),
        lstm_p=np.vstack(oof_lstm_p),
        lgb_a=np.vstack(oof_lgb_a),
        lgb_p=np.vstack(oof_lgb_p),
        la=np.concatenate(oof_la),
        lp=np.concatenate(oof_lp),
    )
    np.savez(OOF_CACHE, **out)
    log(f"\nOOF 已存到 {OOF_CACHE}")
    return out


def load_or_compute_oof():
    if os.path.exists(OOF_CACHE):
        log(f"載入 OOF cache: {OOF_CACHE}")
        d = np.load(OOF_CACHE)
        return {k: d[k] for k in d.files}
    return compute_oof()


def plugin_calibrate(probs, labels, n_class, n_rounds=5,
                     bias_lo=-5.0, bias_hi=5.0, step=0.1, verbose=False):
    """
    Plug-in Macro-F1 calibration via additive log-bias + coordinate descent.
    Decision rule: y_hat = argmax_k (log p_k + b_k)
    """
    log_p = np.log(probs + 1e-12)
    bias = np.zeros(n_class, dtype=np.float64)
    grid = np.arange(bias_lo, bias_hi + step / 2, step)

    def f1_of(b):
        pred = (log_p + b).argmax(1)
        return f1_score(labels, pred, average="macro", zero_division=0)

    best_f1 = f1_of(bias)
    if verbose:
        log(f"    初始 F1: {best_f1:.4f}")

    for r in range(n_rounds):
        improved = False
        for k in range(n_class):
            b_orig = bias[k]
            best_b = b_orig
            best_local = best_f1
            for b in grid:
                bias[k] = b
                f1 = f1_of(bias)
                if f1 > best_local + 1e-6:
                    best_local = f1
                    best_b = b
            bias[k] = best_b
            if best_local > best_f1 + 1e-6:
                best_f1 = best_local
                improved = True
        if verbose:
            log(f"    round {r+1}: F1={best_f1:.4f}")
        if not improved:
            break
    return bias, best_f1


def score(fa, fp):
    return 0.4 * fa + 0.4 * fp + 0.2


def main():
    log("=" * 60)
    log("V6 Phase 1: Plug-in Macro-F1 calibration")
    log("=" * 60)

    d = load_or_compute_oof()
    la, lp = d["la"], d["lp"]
    lstm_a, lstm_p = d["lstm_a"], d["lstm_p"]
    lgb_a, lgb_p = d["lgb_a"], d["lgb_p"]

    log(f"\nOOF 樣本: {len(la)}")
    log(f"  action 分佈: {np.bincount(la, minlength=N_ACTION).tolist()}")
    log(f"  point  分佈: {np.bincount(lp, minlength=N_POINT).tolist()}")

    # ---- 1. 單模型 baseline (argmax, no calibration)
    log(f"\n{'─'*60}")
    log("Step 1: 無校準 baseline")
    f1_of = lambda y, p: f1_score(y, p.argmax(1), average="macro", zero_division=0)
    for name, pa, pp in [("LSTM", lstm_a, lstm_p), ("LGB", lgb_a, lgb_p)]:
        fa, fp = f1_of(la, pa), f1_of(lp, pp)
        log(f"  {name:>4}: F1_a={fa:.4f} F1_p={fp:.4f} Score={score(fa,fp):.4f}")

    # ---- 2. 單模型 + plug-in 校準
    log(f"\n{'─'*60}")
    log("Step 2: 單模型 + Plug-in 校準")
    for name, pa, pp in [("LSTM", lstm_a, lstm_p), ("LGB", lgb_a, lgb_p)]:
        log(f"\n  {name} actionId 校準:")
        _, fa = plugin_calibrate(pa, la, N_ACTION, verbose=True)
        log(f"\n  {name} pointId 校準:")
        _, fp = plugin_calibrate(pp, lp, N_POINT, verbose=True)
        log(f"\n  {name} 最終: F1_a={fa:.4f} F1_p={fp:.4f} Score={score(fa,fp):.4f}")

    # ---- 3. Shared-weight ensemble (V5 風格) + plug-in
    log(f"\n{'─'*60}")
    log("Step 3: Shared-weight ensemble + Plug-in (V5 對照)")
    best_shared = (-1, 0.0, 0.0, 0.0)
    for w in np.arange(0.0, 1.01, 0.1):
        ea = w * lstm_a + (1 - w) * lgb_a
        ep = w * lstm_p + (1 - w) * lgb_p
        _, fa = plugin_calibrate(ea, la, N_ACTION, n_rounds=3)
        _, fp = plugin_calibrate(ep, lp, N_POINT, n_rounds=3)
        sc = score(fa, fp)
        log(f"  w_lstm={w:.2f}: F1_a={fa:.4f} F1_p={fp:.4f} Score={sc:.4f}")
        if sc > best_shared[0]:
            best_shared = (sc, w, fa, fp)
    log(f"\n  最佳 shared w={best_shared[1]:.2f} Score={best_shared[0]:.4f}")

    # ---- 4. Per-task ensemble (Phase 2 bonus)
    log(f"\n{'─'*60}")
    log("Step 4: Per-task ensemble weights + Plug-in")
    log("\n  action 權重搜尋:")
    best_a = (-1.0, 0.0)
    for w_a in np.arange(0.0, 1.01, 0.05):
        ea = w_a * lstm_a + (1 - w_a) * lgb_a
        _, fa = plugin_calibrate(ea, la, N_ACTION, n_rounds=3)
        log(f"    w_a={w_a:.2f}: F1_a={fa:.4f}")
        if fa > best_a[0]:
            best_a = (fa, w_a)

    log("\n  point 權重搜尋:")
    best_p = (-1.0, 0.0)
    for w_p in np.arange(0.0, 1.01, 0.05):
        ep = w_p * lstm_p + (1 - w_p) * lgb_p
        _, fp = plugin_calibrate(ep, lp, N_POINT, n_rounds=3)
        log(f"    w_p={w_p:.2f}: F1_p={fp:.4f}")
        if fp > best_p[0]:
            best_p = (fp, w_p)

    # 最佳組合 - 重跑一次完整校準
    log(f"\n  最佳 w_a={best_a[1]:.2f} (F1_a={best_a[0]:.4f})")
    log(f"  最佳 w_p={best_p[1]:.2f} (F1_p={best_p[0]:.4f})")

    best_ea = best_a[1] * lstm_a + (1 - best_a[1]) * lgb_a
    best_ep = best_p[1] * lstm_p + (1 - best_p[1]) * lgb_p
    log(f"\n  完整校準 action (n_rounds=5):")
    bias_a, fa = plugin_calibrate(best_ea, la, N_ACTION, n_rounds=5, verbose=True)
    log(f"\n  完整校準 point (n_rounds=5):")
    bias_p, fp = plugin_calibrate(best_ep, lp, N_POINT, n_rounds=5, verbose=True)

    final = score(fa, fp)
    log(f"\n{'='*60}")
    log(f"V6 Phase 1 最終 CV Score: {final:.4f}")
    log(f"  F1_action: {fa:.4f}")
    log(f"  F1_point:  {fp:.4f}")
    log(f"  AUC_sgp:   1.0000")
    log(f"{'='*60}")

    log(f"\n對比:")
    log(f"  V5  (乘性 scale):     Score=0.4851  F1_a=0.4354  F1_p=0.2774")
    log(f"  V6 Phase 1 (加性):    Score={final:.4f}  F1_a={fa:.4f}  F1_p={fp:.4f}")
    log(f"  增益:                 +{final-0.4851:.4f}")

    # 存下 bias + 權重
    np.savez("artifacts/v6_phase1_params.npz",
             bias_a=bias_a, bias_p=bias_p,
             w_a=np.float64(best_a[1]),
             w_p=np.float64(best_p[1]))
    log(f"\n參數已存到 v6_phase1_params.npz")


if __name__ == "__main__":
    main()

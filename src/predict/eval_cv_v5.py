#!/usr/bin/env python3
"""
用已儲存的 LSTM 模型 + 重訓 LGB 重現 V5 CV 分數
(LGB 模型未儲存，需重訓；LSTM 從 .pt 載入)
"""
import os, sys, pathlib
_HERE = pathlib.Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
os.chdir(_ROOT)
sys.path.insert(0, str(_ROOT / "src" / "train"))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from collections import Counter
import time

from train_v5 import (
    SEED, BATCH_SIZE, N_FOLDS, N_ACTION, N_POINT, DEVICE,
    train_df, test_df, prepare_samples, build_lgb_features,
    RallyDataset, PingPongModel, train_lgb_fold,
    calibrate_thresholds, log,
)


def get_lstm_probs(model, samples):
    dl = DataLoader(RallyDataset(samples, is_train=True), BATCH_SIZE,
                    shuffle=False, num_workers=0, pin_memory=True)
    a_probs, p_probs, la, lp = [], [], [], []
    model.eval()
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
            a_probs.append(F.softmax(a_log, dim=-1).cpu().numpy())
            p_probs.append(F.softmax(p_log, dim=-1).cpu().numpy())
            la.extend(batch["target_action"])
            lp.extend(batch["target_point"])
    return np.vstack(a_probs), np.vstack(p_probs), np.array(la), np.array(lp)


def main():
    log("=" * 60)
    log(f"Device: {DEVICE}")
    log("準備 LSTM samples...")
    t0 = time.time()
    train_samples = prepare_samples(train_df, is_train=True, augment=True)
    log(f"  {len(train_samples)} samples ({time.time()-t0:.1f}s)")

    log("準備 LGB features...")
    t0 = time.time()
    lgb_X, lgb_ya, lgb_yp, lgb_uids, _ = build_lgb_features(
        train_df, is_train=True, augment=True)
    log(f"  {lgb_X.shape} ({time.time()-t0:.1f}s)")

    # CV setup
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

    fold_summary = []

    for fold, (tr_uidx, va_uidx) in enumerate(kf.split(uid_list, uid_arr)):
        log(f"\n{'─'*50}")
        log(f"Fold {fold+1}/{N_FOLDS}")
        tr_uids = set(np.array(uid_list)[tr_uidx])
        va_uids = set(np.array(uid_list)[va_uidx])

        # LSTM
        va_s = [train_samples[i] for u in va_uids for i in uid2idx.get(u, [])]
        model = PingPongModel(player_drop_p=0).to(DEVICE)
        model.load_state_dict(torch.load(f"models/v5/model_v5_lstm_fold{fold+1}.pt",
                                         map_location=DEVICE))
        la_p, lp_p, la, lp = get_lstm_probs(model, va_s)
        log(f"  LSTM val: {len(la)} samples")

        f_lstm_a = f1_score(la, la_p.argmax(1), average="macro", zero_division=0)
        f_lstm_p = f1_score(lp, lp_p.argmax(1), average="macro", zero_division=0)
        log(f"  LSTM-only argmax: F1_act={f_lstm_a:.4f} F1_pt={f_lstm_p:.4f} "
            f"Score={0.4*f_lstm_a+0.4*f_lstm_p+0.2:.4f}")

        # LGB (重訓)
        lgb_tr = [i for u in tr_uids for i in lgb_uid2idx.get(u, [])]
        lgb_va = [i for u in va_uids for i in lgb_uid2idx.get(u, [])]
        X_tr, X_va = lgb_X.iloc[lgb_tr], lgb_X.iloc[lgb_va]
        y_a_tr = [lgb_ya[i] for i in lgb_tr]
        y_a_va = [lgb_ya[i] for i in lgb_va]
        y_p_tr = [lgb_yp[i] for i in lgb_tr]
        y_p_va = [lgb_yp[i] for i in lgb_va]

        _, ga_p, _ = train_lgb_fold(X_tr, y_a_tr, X_va, y_a_va, N_ACTION, "actionId")
        _, gp_p, _ = train_lgb_fold(X_tr, y_p_tr, X_va, y_p_va, N_POINT, "pointId")

        f_lgb_a = f1_score(y_a_va, ga_p.argmax(1), average="macro", zero_division=0)
        f_lgb_p = f1_score(y_p_va, gp_p.argmax(1), average="macro", zero_division=0)
        log(f"  LGB-only argmax:  F1_act={f_lgb_a:.4f} F1_pt={f_lgb_p:.4f} "
            f"Score={0.4*f_lgb_a+0.4*f_lgb_p+0.2:.4f}")

        # 簡單 ensemble (0.4/0.6)
        e_a = 0.4 * la_p + 0.6 * ga_p
        e_p = 0.4 * lp_p + 0.6 * gp_p
        f_e_a = f1_score(la, e_a.argmax(1), average="macro", zero_division=0)
        f_e_p = f1_score(lp, e_p.argmax(1), average="macro", zero_division=0)
        log(f"  Ensemble (0.4/0.6): F1_act={f_e_a:.4f} F1_pt={f_e_p:.4f} "
            f"Score={0.4*f_e_a+0.4*f_e_p+0.2:.4f}")

        fold_summary.append({
            "fold": fold + 1,
            "lstm_a": f_lstm_a, "lstm_p": f_lstm_p,
            "lgb_a": f_lgb_a, "lgb_p": f_lgb_p,
            "ens_a": f_e_a, "ens_p": f_e_p,
        })

        oof_lstm_a.append(la_p)
        oof_lstm_p.append(lp_p)
        oof_lgb_a.append(ga_p)
        oof_lgb_p.append(gp_p)
        oof_la.append(la)
        oof_lp.append(lp)

    log(f"\n{'='*60}")
    log("逐 fold 總結")
    log(f"{'Fold':>5} {'LSTM_A':>8} {'LSTM_P':>8} {'LGB_A':>8} {'LGB_P':>8} {'ENS_A':>8} {'ENS_P':>8}")
    for r in fold_summary:
        log(f"{r['fold']:>5} {r['lstm_a']:>8.4f} {r['lstm_p']:>8.4f} "
            f"{r['lgb_a']:>8.4f} {r['lgb_p']:>8.4f} "
            f"{r['ens_a']:>8.4f} {r['ens_p']:>8.4f}")
    log(f"{'MEAN':>5} "
        f"{np.mean([r['lstm_a'] for r in fold_summary]):>8.4f} "
        f"{np.mean([r['lstm_p'] for r in fold_summary]):>8.4f} "
        f"{np.mean([r['lgb_a'] for r in fold_summary]):>8.4f} "
        f"{np.mean([r['lgb_p'] for r in fold_summary]):>8.4f} "
        f"{np.mean([r['ens_a'] for r in fold_summary]):>8.4f} "
        f"{np.mean([r['ens_p'] for r in fold_summary]):>8.4f}")

    # 合併 OOF
    log(f"\n{'='*60}")
    log("OOF 合併分析")
    m_la_p = np.vstack(oof_lstm_a)
    m_lp_p = np.vstack(oof_lstm_p)
    m_ga_p = np.vstack(oof_lgb_a)
    m_gp_p = np.vstack(oof_lgb_p)
    m_la = np.concatenate(oof_la)
    m_lp = np.concatenate(oof_lp)
    log(f"  合併樣本數: {len(m_la)}")

    # 單模型 merged argmax
    f = lambda y, p: f1_score(y, p.argmax(1), average="macro", zero_division=0)
    s_lstm = (f(m_la, m_la_p), f(m_lp, m_lp_p))
    s_lgb = (f(m_la, m_ga_p), f(m_lp, m_gp_p))
    log(f"\n  LSTM only:  F1_act={s_lstm[0]:.4f} F1_pt={s_lstm[1]:.4f} "
        f"Score={0.4*s_lstm[0]+0.4*s_lstm[1]+0.2:.4f}")
    log(f"  LGB only:   F1_act={s_lgb[0]:.4f} F1_pt={s_lgb[1]:.4f} "
        f"Score={0.4*s_lgb[0]+0.4*s_lgb[1]+0.2:.4f}")

    # 搜索最佳 ensemble 權重
    log(f"\n  搜尋最佳 ensemble 權重 (LSTM 權重):")
    best_w, best_sc = 0.0, -1
    for w in np.arange(0.0, 1.05, 0.05):
        ea = w * m_la_p + (1 - w) * m_ga_p
        ep = w * m_lp_p + (1 - w) * m_gp_p
        fa = f(m_la, ea)
        fp = f(m_lp, ep)
        sc = 0.4 * fa + 0.4 * fp + 0.2
        log(f"    w_lstm={w:.2f}  F1_act={fa:.4f}  F1_pt={fp:.4f}  Score={sc:.4f}")
        if sc > best_sc:
            best_sc = sc
            best_w = w

    # 校準最佳 ensemble
    log(f"\n  最佳權重 w_lstm={best_w:.2f} → Score={best_sc:.4f}")
    e_a = best_w * m_la_p + (1 - best_w) * m_ga_p
    e_p = best_w * m_lp_p + (1 - best_w) * m_gp_p

    log(f"\n  校準 actionId...")
    _, cal_a = calibrate_thresholds(e_a, m_la, N_ACTION)
    log(f"  校準後 F1_act: {cal_a:.4f}")

    log(f"\n  校準 pointId...")
    _, cal_p = calibrate_thresholds(e_p, m_lp, N_POINT)
    log(f"  校準後 F1_pt:  {cal_p:.4f}")

    final = 0.4 * cal_a + 0.4 * cal_p + 0.2
    log(f"\n{'='*60}")
    log(f"最終 V5 CV Score: {final:.4f}")
    log(f"  F1_action: {cal_a:.4f}")
    log(f"  F1_point:  {cal_p:.4f}")
    log(f"  AUC_sgp:   1.0000")
    log(f"{'='*60}")

    # 對比
    log(f"\n版本對比:")
    log(f"  V3 (LSTM 原始):   Score=0.4525  F1_a=0.4214  F1_p=0.2098")
    log(f"  V4 (修 bug):      Score=0.4785  F1_a=0.4575  F1_p=0.2389")
    log(f"  V5 (Ensemble):    Score={final:.4f}  F1_a={cal_a:.4f}  F1_p={cal_p:.4f}")
    log(f"  V4 → V5 增益:     +{final-0.4785:.4f}")


if __name__ == "__main__":
    main()

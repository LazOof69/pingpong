#!/usr/bin/env python3
"""
V7 P0 Diagnostic Triage
-----------------------
決定是否值得跑 seed bagging / LGB feat expansion。

輸出：
1. per-class train support
2. LSTM vs LGB 相關性（per-class）
3. LSTM fold-to-fold F1 variance
4. Ensemble 表面平坦性分析
"""
import os, sys, pathlib
_HERE = pathlib.Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
os.chdir(_ROOT)
sys.path.insert(0, str(_ROOT / "src" / "train"))

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

from train_v5 import SEED, N_FOLDS, N_ACTION, N_POINT, train_df, prepare_samples, log


def main():
    log("=" * 70)
    log("V7 P0 · Diagnostic Triage")
    log("=" * 70)

    d = np.load("artifacts/v5_oof.npz")
    lstm_a, lstm_p = d["lstm_a"], d["lstm_p"]
    lgb_a, lgb_p = d["lgb_a"], d["lgb_p"]
    la, lp = d["la"], d["lp"]

    log(f"\nOOF 樣本: {len(la)}")
    log(f"  LSTM shape: action={lstm_a.shape} point={lstm_p.shape}")
    log(f"  LGB  shape: action={lgb_a.shape} point={lgb_p.shape}")

    # =======================================================
    # 1. Per-class train support
    # =======================================================
    log(f"\n{'─'*70}")
    log("【1】Per-class train support")
    act_hist = np.bincount(la, minlength=N_ACTION)
    pt_hist = np.bincount(lp, minlength=N_POINT)
    log(f"\n  actionId support (index : count):")
    for k in range(N_ACTION):
        marker = ""
        if act_hist[k] < 10:
            marker = "  ← VERY SPARSE (< 10)"
        elif act_hist[k] < 30:
            marker = "  ← sparse"
        log(f"    {k:>2}: {act_hist[k]:>5}{marker}")
    log(f"\n  pointId support:")
    for k in range(N_POINT):
        marker = "  ← sparse" if pt_hist[k] < 30 else ""
        log(f"    {k:>2}: {pt_hist[k]:>5}{marker}")

    sparse_a = [k for k in range(N_ACTION) if act_hist[k] < 10]
    if sparse_a:
        log(f"\n  ⚠  稀有 action 類 (<10 samples): {sparse_a}")
        log(f"     這些類的 F1 會被 label sparsity 硬頂")
        n_effective = N_ACTION - len(sparse_a)
        ceiling = n_effective / N_ACTION
        log(f"     Macro-F1 稀疏上限 (假設稀有類 F1=0): ≤ {ceiling:.4f}")

    # =======================================================
    # 2. LSTM vs LGB correlation per-class
    # =======================================================
    log(f"\n{'─'*70}")
    log("【2】LSTM vs LGB 機率相關性（per-class）")
    log(f"\n  actionId:")
    hi_corr_a = 0
    for k in range(N_ACTION):
        c = np.corrcoef(lstm_a[:, k], lgb_a[:, k])[0, 1]
        flag = ""
        if c > 0.9:
            flag = "  ← HIGH (ensemble trick 無效)"
            hi_corr_a += 1
        elif c < 0.5:
            flag = "  ← diverse"
        log(f"    class {k:>2}: r={c:>6.3f}{flag}  (support={act_hist[k]})")
    log(f"\n  pointId:")
    hi_corr_p = 0
    for k in range(N_POINT):
        c = np.corrcoef(lstm_p[:, k], lgb_p[:, k])[0, 1]
        flag = ""
        if c > 0.9:
            flag = "  ← HIGH"
            hi_corr_p += 1
        elif c < 0.5:
            flag = "  ← diverse"
        log(f"    class {k:>2}: r={c:>6.3f}{flag}  (support={pt_hist[k]})")

    log(f"\n  高相關類別: action={hi_corr_a}/{N_ACTION}, point={hi_corr_p}/{N_POINT}")

    # =======================================================
    # 3. LSTM fold-to-fold variance
    # =======================================================
    log(f"\n{'─'*70}")
    log("【3】LSTM fold-to-fold F1 variance")

    # 重建 fold 邊界（跟 train_v5 一致的 CV）
    train_samples = prepare_samples(train_df, is_train=True, augment=True)
    uid_last_a = {s["uid"]: s["target_action"] for s in train_samples}
    uid_list = sorted(uid_last_a.keys())
    uid_arr = np.array([uid_last_a[u] for u in uid_list])
    uid2idx = {}
    for i, s in enumerate(train_samples):
        uid2idx.setdefault(s["uid"], []).append(i)
    kf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)

    fold_sizes = []
    for _, va_uidx in kf.split(uid_list, uid_arr):
        va_uids = set(np.array(uid_list)[va_uidx])
        size = sum(len(uid2idx.get(u, [])) for u in va_uids)
        fold_sizes.append(size)
    cum = np.cumsum([0] + fold_sizes)
    log(f"  fold sizes: {fold_sizes}  total: {cum[-1]}")

    assert cum[-1] == len(la), f"fold sum {cum[-1]} != OOF {len(la)}"

    lstm_fold_fa, lstm_fold_fp = [], []
    lgb_fold_fa, lgb_fold_fp = [], []
    for fi in range(N_FOLDS):
        s, e = cum[fi], cum[fi+1]
        fa_l = f1_score(la[s:e], lstm_a[s:e].argmax(1), average="macro", zero_division=0)
        fp_l = f1_score(lp[s:e], lstm_p[s:e].argmax(1), average="macro", zero_division=0)
        fa_g = f1_score(la[s:e], lgb_a[s:e].argmax(1), average="macro", zero_division=0)
        fp_g = f1_score(lp[s:e], lgb_p[s:e].argmax(1), average="macro", zero_division=0)
        lstm_fold_fa.append(fa_l); lstm_fold_fp.append(fp_l)
        lgb_fold_fa.append(fa_g); lgb_fold_fp.append(fp_g)
        log(f"  fold {fi+1}: LSTM F1_a={fa_l:.4f} F1_p={fp_l:.4f} | LGB F1_a={fa_g:.4f} F1_p={fp_g:.4f}")

    log(f"\n  LSTM σ (std across folds):")
    log(f"    F1_a: {np.std(lstm_fold_fa):.4f}  (mean={np.mean(lstm_fold_fa):.4f})")
    log(f"    F1_p: {np.std(lstm_fold_fp):.4f}  (mean={np.mean(lstm_fold_fp):.4f})")
    log(f"  LGB  σ:")
    log(f"    F1_a: {np.std(lgb_fold_fa):.4f}  (mean={np.mean(lgb_fold_fa):.4f})")
    log(f"    F1_p: {np.std(lgb_fold_fp):.4f}  (mean={np.mean(lgb_fold_fp):.4f})")

    sigma_lstm = max(np.std(lstm_fold_fa), np.std(lstm_fold_fp))
    log(f"\n  LSTM max σ = {sigma_lstm:.4f}")
    if sigma_lstm < 0.01:
        log(f"  → seed bagging 預期增益 ~0.002 (不划算)")
    elif sigma_lstm < 0.015:
        log(f"  → seed bagging 預期增益 ~0.003 (邊緣)")
    else:
        log(f"  → seed bagging 預期增益 ≥ 0.005 (值得)")

    # =======================================================
    # 4. Ensemble flatness — why w_a=w_p=0.35?
    # =======================================================
    log(f"\n{'─'*70}")
    log("【4】Ensemble 表面平坦性")

    log(f"\n  重跑 per-task w search 並檢查 F1 表面形狀:")
    log(f"  action:")
    a_curve = []
    for w in np.arange(0, 1.01, 0.05):
        ea = w * lstm_a + (1 - w) * lgb_a
        fa = f1_score(la, ea.argmax(1), average="macro", zero_division=0)
        a_curve.append((w, fa))
        log(f"    w_a={w:.2f}  raw F1_a={fa:.4f}")
    log(f"  point:")
    p_curve = []
    for w in np.arange(0, 1.01, 0.05):
        ep = w * lstm_p + (1 - w) * lgb_p
        fp = f1_score(lp, ep.argmax(1), average="macro", zero_division=0)
        p_curve.append((w, fp))
        log(f"    w_p={w:.2f}  raw F1_p={fp:.4f}")

    def flatness(curve):
        vals = [v for _, v in curve]
        return np.std(vals), np.max(vals) - np.min(vals)
    sa, ra = flatness(a_curve)
    sp, rp = flatness(p_curve)
    log(f"\n  action F1 surface: σ={sa:.4f} range={ra:.4f}")
    log(f"  point  F1 surface: σ={sp:.4f} range={rp:.4f}")
    if max(sa, sp) < 0.005:
        log(f"  → FLAT: 兩模型貢獻類似，bagging LSTM 不會大幅改變")
    else:
        log(f"  → 有顯著變化，ensemble 仍有槓桿")

    # =======================================================
    # 5. Recommendation
    # =======================================================
    log(f"\n{'='*70}")
    log("【結論】")
    go_seed_bag = sigma_lstm > 0.015 and hi_corr_a < N_ACTION * 0.6
    go_rank = max(sa, sp) > 0.002  # 任何非零槓桿
    log(f"  1. Seed bagging?       {'GO' if go_seed_bag else 'NO (σ or corr 不利)'}")
    log(f"  2. Rank averaging?     {'GO' if go_rank else 'marginal'}")
    log(f"  3. LGB feat expansion? GO (無論如何試 target encoding)")
    log(f"  4. Per-class threshold on sparse action? GO")

    # 粗略 ceiling 估計
    sparse_pct = len(sparse_a) / N_ACTION
    ceiling_a_est = 1.0 - sparse_pct  # 假設稀疏類 F1=0，非稀疏類 F1=1
    realistic_a = ceiling_a_est * 0.55  # 更實際：非稀疏類平均 F1~0.55
    realistic_p = 0.30
    score_ceiling = 0.4 * realistic_a + 0.4 * realistic_p + 0.2
    log(f"\n  粗估實務上限:")
    log(f"    action ceiling (non-sparse × 0.55): {realistic_a:.3f}")
    log(f"    point  ceiling (OOF oracle):        {realistic_p:.3f}")
    log(f"    Score  ceiling:                     {score_ceiling:.4f}")
    log(f"    V6 P1 目前:                         0.4871")
    log(f"    可挖空間:                           +{score_ceiling-0.4871:.4f}")
    log(f"{'='*70}")


if __name__ == "__main__":
    main()

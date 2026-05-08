"""
P2 stacking probe — does kNN-LGB ensemble lift OOF macro-F1 over current ship?

Tests Caruana 2004 stacking of:
  Component 1: existing bag2 ship (LSTM+LGB α-blend + advcal additive bias)
  Component 2: kNN-LGB OOF predictions (from probe_p2_stage_a.npz)

If optimal w_kNN > 0 AND ensemble F1 - ship F1 ≥ +0.005 → proceed to full
hidden-state P2.
If lift < +0.002 → cheap kNN doesn't help in ensemble; full version unlikely better.

Cost: ~5 min.
"""
import os, sys
os.environ["V5Z_MATCH_DISJOINT"] = "1"
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

from train_v5z import prepare_samples, train_df, N_FOLDS, N_ACTION, N_POINT, log


def main():
    log("="*78)
    log("P2 Stacking probe — kNN-LGB + bag2 ship Caruana stacking")
    log("="*78)

    # Load kNN probe
    knn = np.load('artifacts/probe_p2_stage_a.npz', allow_pickle=True)
    knn_a = knn['knn_a_best']
    knn_p = knn['knn_p_best']
    la = knn['la_sample']
    lp = knn['lp_sample']
    log(f"\nkNN OOF loaded: action {knn_a.shape}, point {knn_p.shape}")

    # Reconstruct bag2 ship calibrated probs in sample order
    oof_s42 = np.load('artifacts/v5z_md_full_s42_oof.npz', allow_pickle=True)
    oof_s1337 = np.load('artifacts/v5z_md_full_s1337_oof.npz', allow_pickle=True)
    advcal = np.load('artifacts/v5z_md_advcal_bag2_params.npz', allow_pickle=True)
    a_alpha = float(advcal['a_alpha']); p_alpha = float(advcal['p_alpha'])
    bias_a = advcal['bias_a']; bias_p = advcal['bias_p']

    train_samples = prepare_samples(train_df, is_train=True, augment=True)
    uid_last_a = {s["uid"]: s["target_action"] for s in train_samples}
    uid_list = sorted(uid_last_a.keys())
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
                    s2o[i] = cur; cur += 1
        return s2o

    s2o_42 = reindex_for_seed(42)
    s2o_1337 = reindex_for_seed(1337)

    def get_can(d, s2o):
        return {k: d[k][s2o] for k in ['lstm_a','lstm_p','lgb_a','lgb_p','la','lp']}
    c42 = get_can(oof_s42, s2o_42); c13 = get_can(oof_s1337, s2o_1337)

    oof_a_raw = a_alpha * (c42['lstm_a']+c13['lstm_a'])/2 + (1-a_alpha) * (c42['lgb_a']+c13['lgb_a'])/2
    oof_p_raw = p_alpha * (c42['lstm_p']+c13['lstm_p'])/2 + (1-p_alpha) * (c42['lgb_p']+c13['lgb_p'])/2

    # Apply existing advcal bias → final ship logits
    log_p_a_ship = np.log(oof_a_raw + 1e-12) + bias_a  # (N, 19)
    log_p_p_ship = np.log(oof_p_raw + 1e-12) + bias_p  # (N, 10)

    # Convert ship logits → softmax probs (for stacking)
    def softmax(z):
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    p_ship_a = softmax(log_p_a_ship)
    p_ship_p = softmax(log_p_p_ship)

    # Sanity: ship F1
    f1a_ship = f1_score(la, p_ship_a.argmax(1), average='macro', zero_division=0)
    f1p_ship = f1_score(lp, p_ship_p.argmax(1), average='macro', zero_division=0)
    log(f"\n  Bag2 ship OOF: F1_a={f1a_ship:.4f}, F1_p={f1p_ship:.4f}, avg={(f1a_ship+f1p_ship)/2:.4f}")
    log(f"  kNN alone OOF: F1_a={f1_score(la, knn_a.argmax(1), average='macro', zero_division=0):.4f}, "
        f"F1_p={f1_score(lp, knn_p.argmax(1), average='macro', zero_division=0):.4f}")

    # ============================================================
    # Search optimal stacking weight w_kNN ∈ [0, 1]
    # ============================================================
    log("\n" + "─"*78)
    log("Stacking weight search (w_kNN ∈ [0, 1], step 0.05)")
    log("─"*78)

    best_a = (0.0, f1a_ship); best_p = (0.0, f1p_ship)
    log(f"\n  Action stacking:")
    log(f"  {'w_kNN':<10}{'F1_action':<12}{'Δ_vs_ship':<12}")
    for w in np.arange(0.0, 1.001, 0.05):
        ens = (1-w) * p_ship_a + w * knn_a
        f1 = f1_score(la, ens.argmax(1), average='macro', zero_division=0)
        delta = f1 - f1a_ship
        flag = " ⭐" if f1 > best_a[1] else ""
        log(f"  {w:<10.2f}{f1:<12.4f}{delta:<+12.4f}{flag}")
        if f1 > best_a[1]: best_a = (w, f1)

    log(f"\n  Point stacking:")
    log(f"  {'w_kNN':<10}{'F1_point':<12}{'Δ_vs_ship':<12}")
    for w in np.arange(0.0, 1.001, 0.05):
        ens = (1-w) * p_ship_p + w * knn_p
        f1 = f1_score(lp, ens.argmax(1), average='macro', zero_division=0)
        delta = f1 - f1p_ship
        flag = " ⭐" if f1 > best_p[1] else ""
        log(f"  {w:<10.2f}{f1:<12.4f}{delta:<+12.4f}{flag}")
        if f1 > best_p[1]: best_p = (w, f1)

    # Combined optimal
    log("\n" + "─"*78)
    log("Optimal stacking results")
    log("─"*78)
    log(f"\n  Action: best w_kNN = {best_a[0]:.2f}, F1 = {best_a[1]:.4f} (ship {f1a_ship:.4f}, Δ={best_a[1]-f1a_ship:+.4f})")
    log(f"  Point:  best w_kNN = {best_p[0]:.2f}, F1 = {best_p[1]:.4f} (ship {f1p_ship:.4f}, Δ={best_p[1]-f1p_ship:+.4f})")

    score_ship = 0.4 * f1a_ship + 0.4 * f1p_ship + 0.2
    score_stacked = 0.4 * best_a[1] + 0.4 * best_p[1] + 0.2
    log(f"\n  Bag2 ship score: {score_ship:.4f}")
    log(f"  Stacked score:   {score_stacked:.4f}")
    log(f"  Δ score:         {score_stacked - score_ship:+.4f}")

    # ============================================================
    # Held-out verification
    # ============================================================
    log("\n" + "─"*78)
    log("Held-out verification (OOF half split)")
    log("─"*78)
    np.random.seed(42)
    perm = np.random.permutation(len(la))
    half = len(la) // 2
    fit_idx, val_idx = perm[:half], perm[half:]

    # Fit weight on first half
    best_w_a_fit = 0.0; best_f1_a_fit = f1_score(la[fit_idx], p_ship_a[fit_idx].argmax(1), average='macro', zero_division=0)
    for w in np.arange(0.0, 1.001, 0.05):
        ens = (1-w) * p_ship_a[fit_idx] + w * knn_a[fit_idx]
        f1 = f1_score(la[fit_idx], ens.argmax(1), average='macro', zero_division=0)
        if f1 > best_f1_a_fit: best_f1_a_fit = f1; best_w_a_fit = w
    best_w_p_fit = 0.0; best_f1_p_fit = f1_score(lp[fit_idx], p_ship_p[fit_idx].argmax(1), average='macro', zero_division=0)
    for w in np.arange(0.0, 1.001, 0.05):
        ens = (1-w) * p_ship_p[fit_idx] + w * knn_p[fit_idx]
        f1 = f1_score(lp[fit_idx], ens.argmax(1), average='macro', zero_division=0)
        if f1 > best_f1_p_fit: best_f1_p_fit = f1; best_w_p_fit = w

    # Apply on held-out half
    ens_a_val = (1-best_w_a_fit) * p_ship_a[val_idx] + best_w_a_fit * knn_a[val_idx]
    ens_p_val = (1-best_w_p_fit) * p_ship_p[val_idx] + best_w_p_fit * knn_p[val_idx]
    f1a_val_stacked = f1_score(la[val_idx], ens_a_val.argmax(1), average='macro', zero_division=0)
    f1p_val_stacked = f1_score(lp[val_idx], ens_p_val.argmax(1), average='macro', zero_division=0)
    f1a_val_ship = f1_score(la[val_idx], p_ship_a[val_idx].argmax(1), average='macro', zero_division=0)
    f1p_val_ship = f1_score(lp[val_idx], p_ship_p[val_idx].argmax(1), average='macro', zero_division=0)

    log(f"\n  Held-out (val half) — fit weights from first half:")
    log(f"    Action: w_kNN={best_w_a_fit:.2f}, stacked F1={f1a_val_stacked:.4f}, ship F1={f1a_val_ship:.4f}, Δ={f1a_val_stacked-f1a_val_ship:+.4f}")
    log(f"    Point:  w_kNN={best_w_p_fit:.2f}, stacked F1={f1p_val_stacked:.4f}, ship F1={f1p_val_ship:.4f}, Δ={f1p_val_stacked-f1p_val_ship:+.4f}")
    val_delta = ((f1a_val_stacked - f1a_val_ship) + (f1p_val_stacked - f1p_val_ship)) / 2
    log(f"    Avg Δ:  {val_delta:+.4f}")

    log("\n" + "="*78)
    log("DECISION")
    log("="*78)
    if val_delta >= 0.005:
        log(f"\n  ✅ STRONG PROCEED — held-out lift {val_delta:+.4f} ≥ +0.005")
        log(f"  → Hidden-state version likely gives more (192-dim > 49-dim feature space)")
        log(f"  → Worth 4-6h investment")
    elif val_delta >= 0.002:
        log(f"\n  ⚠️  WEAK PROCEED — held-out lift {val_delta:+.4f} ∈ [0.002, 0.005)")
        log(f"  → Cheap version barely helps; full version may or may not deliver +0.010")
        log(f"  → 4-6h investment is medium-risk")
    elif val_delta >= 0:
        log(f"\n  ⚠️  MARGINAL — held-out lift {val_delta:+.4f} ∈ [0, 0.002)")
        log(f"  → Cheap kNN provides almost no ensemble lift")
        log(f"  → Full hidden-state version: minimal expected lift")
    else:
        log(f"\n  ❌ KILL P2 — held-out regression {val_delta:+.4f} < 0")
        log(f"  → kNN component hurts ensemble; full version unlikely to help")


if __name__ == "__main__":
    main()

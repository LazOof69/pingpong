#!/usr/bin/env python3
"""
Phase 2 LGB-deep: serverGetPoint LGB on hand-crafted rally features + LSTM trunk_out.

Per-sample structure (matches prepare_samples augment=True from train_v5z):
- Train: every (rally, k) prefix where k=1..N-1 → 70k+ samples
- Test: one sample per rally with full visible context

Features:
- 39 hand-crafted prefix features (same as predict_sgp_lgb.py)
- 128-dim trunk_out from LSTM (averaged over seeds and folds)

Holdout: match-disjoint StratifiedGroupKFold(N=5, groups=match_id) — uses
SEED=42 split (same as trunk extraction's s42 fold split).

Usage:
    V5Z_SEEDS=42  python3 src/train/predict_sgp_lgb_deep.py        # s42 trunk only
    V5Z_SEEDS=42,1337  python3 src/train/predict_sgp_lgb_deep.py   # both seeds avg

Outputs:
    artifacts/sgp_pred_NEWTEST_lgb_deep.npz  (rally_uids, sgp_pred, ...)
"""
import os
import pathlib
os.chdir(pathlib.Path(__file__).resolve().parents[2])

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

import sys
sys.path.insert(0, "src/train")

# Set defaults so train_v5z imports cleanly
os.environ.setdefault("V5Z_TEST_CSV", "data/test_new.csv")
os.environ.setdefault("V5Z_PID_TEST_CSV", "data/test.csv")
os.environ.setdefault("V5Z_INFER_TEST_ONLY", "1")
os.environ.setdefault("V5Z_MATCH_DISJOINT", "1")
os.environ.setdefault("V5Z_TAG", "v5z_md_full_s42")
os.environ.setdefault("V5Z_LSTM_WEIGHTS_TAG", "v5z_md_full_s42")

import train_v5z as M
from train_v5z import prepare_samples, train_df, test_df, N_FOLDS

ACTION_GROUPS = M.ACTION_GROUPS
POINT_DEPTH = M.POINT_DEPTH
POINT_SIDE = M.POINT_SIDE


def bayesian_player_stats(rally_meta, kappa=50):
    prior_global = float(rally_meta["serverGetPoint"].mean())
    prior_sex = {
        s: float(rally_meta[rally_meta["sex"] == s]["serverGetPoint"].mean())
        for s in rally_meta["sex"].unique()
    }
    srv = rally_meta.groupby("gamePlayerId").agg(
        n=("serverGetPoint", "size"), w=("serverGetPoint", "sum"), sex=("sex", "first")
    )
    rcv = rally_meta.groupby("gamePlayerOtherId").agg(
        n=("serverGetPoint", "size"), w=("serverGetPoint", "sum"), sex=("sex", "first")
    )
    srv["winrate"] = (srv["w"] + kappa * srv["sex"].map(prior_sex)) / (srv["n"] + kappa)
    rcv["loserate"] = (rcv["w"] + kappa * rcv["sex"].map(prior_sex)) / (rcv["n"] + kappa)
    return srv["winrate"].to_dict(), srv["n"].to_dict(), \
           rcv["loserate"].to_dict(), rcv["n"].to_dict(), \
           prior_sex, prior_global


def build_sample_features(rows, k, srv_w, srv_n, rcv_l, rcv_n, prior_sex, prior_global, known_matches=None):
    """Build feature dict from rows[:k]. Same schema as predict_sgp_lgb.py."""
    first = rows[0]
    visible = rows[:k]
    last_visible = visible[-1]
    sex = first["sex"]
    sex_p = prior_sex.get(sex, prior_global)
    srv_pid = first["gamePlayerId"]
    rcv_pid = first["gamePlayerOtherId"]

    f = {
        "ctx_len": k,
        "sex": sex,
        "numberGame": first["numberGame"],
        "score_self_start": first["scoreSelf"],
        "score_other_start": first["scoreOther"],
        "score_diff_start": first["scoreSelf"] - first["scoreOther"],
        "score_sum_start": first["scoreSelf"] + first["scoreOther"],
        "is_deuce_start": int(first["scoreSelf"] >= 10 and first["scoreOther"] >= 10),
        "srv_winrate": srv_w.get(srv_pid, sex_p),
        "srv_n_obs": srv_n.get(srv_pid, 0),
        "rcv_loserate": rcv_l.get(rcv_pid, sex_p),
        "rcv_n_obs": rcv_n.get(rcv_pid, 0),
        "srv_seen": int(srv_pid in srv_w),
        "rcv_seen": int(rcv_pid in rcv_l),
        "match_seen": int(known_matches is not None and first["match"] in known_matches),
        "last_strikeId": last_visible["strikeId"],
        "last_handId": last_visible["handId"],
        "last_strengthId": last_visible["strengthId"],
        "last_spinId": last_visible["spinId"],
        "last_pointId": last_visible["pointId"],
        "last_actionId": last_visible["actionId"],
        "last_positionId": last_visible["positionId"],
        "last_action_group": ACTION_GROUPS.get(last_visible["actionId"], 0),
        "last_point_depth": POINT_DEPTH.get(last_visible["pointId"], 0),
        "last_point_side": POINT_SIDE.get(last_visible["pointId"], 0),
        "last_score_diff": last_visible["scoreSelf"] - last_visible["scoreOther"],
    }

    groups = [ACTION_GROUPS.get(r["actionId"], 0) for r in visible]
    f["n_attacks"] = sum(1 for g in groups if g == 1)
    f["n_controls"] = sum(1 for g in groups if g == 2)
    f["n_defenses"] = sum(1 for g in groups if g == 3)
    f["n_serves"] = sum(1 for g in groups if g == 4)
    srv_groups = [ACTION_GROUPS.get(r["actionId"], 0) for r in visible if r["strikeNumber"] % 2 == 1]
    rcv_groups = [ACTION_GROUPS.get(r["actionId"], 0) for r in visible if r["strikeNumber"] % 2 == 0]
    f["n_srv_attacks"] = sum(1 for g in srv_groups if g == 1)
    f["n_rcv_attacks"] = sum(1 for g in rcv_groups if g == 1)
    f["n_srv_strokes"] = len(srv_groups)
    f["n_rcv_strokes"] = len(rcv_groups)

    if k >= 2:
        prev = visible[-2]
        f["prev_actionId"] = prev["actionId"]
        f["prev_pointId"] = prev["pointId"]
        f["prev_action_group"] = ACTION_GROUPS.get(prev["actionId"], 0)
        f["prev_handId"] = prev["handId"]
        f["prev_spinId"] = prev["spinId"]
    else:
        f["prev_actionId"] = -1
        f["prev_pointId"] = -1
        f["prev_action_group"] = -1
        f["prev_handId"] = -1
        f["prev_spinId"] = -1

    return f


def build_train_features(train_df, train_samples, srv_w, srv_n, rcv_l, rcv_n, prior_sex, prior_global, known_matches=None):
    """Build features for ALL prepare_samples-augmented samples. Returns DataFrame in same order as samples."""
    rally_rows = {}  # rally_uid → list of stroke rows sorted
    for uid, grp in train_df.groupby("rally_uid"):
        rally_rows[uid] = grp.sort_values("strikeNumber").to_dict("records")

    feats = []
    sgp = []
    for s in train_samples:
        uid = s["uid"]
        k = s["length"]
        rows = rally_rows[uid]
        f = build_sample_features(rows, k, srv_w, srv_n, rcv_l, rcv_n, prior_sex, prior_global, known_matches)
        feats.append(f)
        sgp.append(int(rows[0]["serverGetPoint"]) if "serverGetPoint" in rows[0] else 0)
    return pd.DataFrame(feats), np.array(sgp)


def build_test_features(test_df, test_samples, srv_w, srv_n, rcv_l, rcv_n, prior_sex, prior_global, known_matches=None):
    rally_rows = {}
    for uid, grp in test_df.groupby("rally_uid"):
        rally_rows[uid] = grp.sort_values("strikeNumber").to_dict("records")
    feats = []
    for s in test_samples:
        uid = s["uid"]
        k = s["length"]  # full visible
        rows = rally_rows[uid]
        f = build_sample_features(rows, k, srv_w, srv_n, rcv_l, rcv_n, prior_sex, prior_global, known_matches)
        feats.append(f)
    return pd.DataFrame(feats)


def main():
    print("=" * 70)
    print("Phase 2 LGB-deep: serverGetPoint with LSTM trunk_out")
    print("=" * 70)

    seeds = os.environ.get("V5Z_SEEDS", "42").split(",")
    kappa = int(os.environ.get("V5Z_KAPPA", "50"))
    print(f"Seeds: {seeds}, κ={kappa}")

    # Load trunk OOF for each seed, average
    trunk_oof_seeds = []
    for seed in seeds:
        path = f"artifacts/trunk_train_oof_s{seed}.npz"
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing trunk OOF: {path}. Run extract_trunk.py first.")
        d = np.load(path)
        trunk_oof_seeds.append(d["trunk"])
        print(f"  Loaded {path} shape={d['trunk'].shape}")
    trunk_oof = np.mean(trunk_oof_seeds, axis=0).astype(np.float32)
    print(f"  Avg trunk OOF: {trunk_oof.shape}")

    # Load trunk test for each seed, average
    trunk_test_seeds = []
    for seed in seeds:
        path = f"artifacts/trunk_test_NEWTEST_s{seed}.npz"
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing trunk test: {path}.")
        d = np.load(path)
        trunk_test_seeds.append(d["trunk"])
    trunk_test = np.mean(trunk_test_seeds, axis=0).astype(np.float32)
    print(f"  Avg trunk test: {trunk_test.shape}")

    # Generate samples (same prepare_samples as train_v5z)
    train_samples = prepare_samples(train_df, is_train=True, augment=True)
    test_samples = prepare_samples(test_df, is_train=False)
    assert len(train_samples) == trunk_oof.shape[0], f"sample count mismatch: {len(train_samples)} vs {trunk_oof.shape[0]}"
    assert len(test_samples) == trunk_test.shape[0]
    print(f"  Train samples: {len(train_samples)}, test samples: {len(test_samples)}")

    rally_meta_full = train_df.groupby("rally_uid").first()[
        ["gamePlayerId", "gamePlayerOtherId", "serverGetPoint", "sex", "match"]
    ].reset_index()
    train_matches = set(rally_meta_full["match"].unique())

    # Match-disjoint fold split (use SEED=42, matching extract_trunk's split for seed 42)
    uid2idx = {}
    for i, s in enumerate(train_samples):
        uid2idx.setdefault(s["uid"], []).append(i)
    uid_last_a = {s["uid"]: s["target_action"] for s in train_samples}
    uid_to_match = train_df.groupby("rally_uid")["match"].first().to_dict()
    uid_list = sorted(uid_last_a.keys())
    uid_arr = np.array([uid_last_a[u] for u in uid_list])
    match_arr = np.array([uid_to_match[u] for u in uid_list])

    sgkf = StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=42)
    kf_splits = list(sgkf.split(uid_list, uid_arr, groups=match_arr))

    rally_to_idx = {u: i for i, u in enumerate(rally_meta_full["rally_uid"].values)}
    oof_pred_per_rally = np.zeros(len(rally_meta_full))
    oof_count_per_rally = np.zeros(len(rally_meta_full))
    oof_label_rally = rally_meta_full["serverGetPoint"].values.astype(np.float64)
    fold_models = []

    print("\n--- Per-fold training ---")
    for fold, (tr_uidx, va_uidx) in enumerate(kf_splits):
        tr_uids = set(np.array(uid_list)[tr_uidx])
        va_uids = set(np.array(uid_list)[va_uidx])

        # Per-fold winrates from train fold rallies
        tr_meta = rally_meta_full[rally_meta_full["rally_uid"].isin(tr_uids)]
        srv_w, srv_n_, rcv_l, rcv_n_, prior_sex, prior_global = bayesian_player_stats(tr_meta, kappa)

        tr_sample_idx = [i for u in tr_uids for i in uid2idx.get(u, [])]
        va_sample_idx = [i for u in va_uids for i in uid2idx.get(u, [])]

        tr_samples = [train_samples[i] for i in tr_sample_idx]
        va_samples = [train_samples[i] for i in va_sample_idx]

        X_tr_hand, y_tr = build_train_features(train_df, tr_samples, srv_w, srv_n_, rcv_l, rcv_n_, prior_sex, prior_global)
        X_va_hand, y_va = build_train_features(train_df, va_samples, srv_w, srv_n_, rcv_l, rcv_n_, prior_sex, prior_global)

        trunk_tr = trunk_oof[tr_sample_idx]
        trunk_va = trunk_oof[va_sample_idx]
        trunk_cols = [f"trunk_{i}" for i in range(trunk_tr.shape[1])]
        X_tr_trunk = pd.DataFrame(trunk_tr, columns=trunk_cols, index=X_tr_hand.index)
        X_va_trunk = pd.DataFrame(trunk_va, columns=trunk_cols, index=X_va_hand.index)
        X_tr = pd.concat([X_tr_hand, X_tr_trunk], axis=1)
        X_va = pd.concat([X_va_hand, X_va_trunk], axis=1)

        print(f"\nFold {fold+1}/{N_FOLDS}: Tr {len(X_tr)} samples, Va {len(X_va)}, features={X_tr.shape[1]} (39 hand + 128 trunk)")

        model = lgb.train(
            {
                "objective": "binary",
                "metric": "auc",
                "learning_rate": 0.03,
                "num_leaves": 31,
                "min_data_in_leaf": 50,
                "lambda_l2": 1.0,
                "feature_fraction": 0.7,  # lower since 167 features
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                "seed": 42,
                "verbose": -1,
            },
            lgb.Dataset(X_tr, label=y_tr),
            num_boost_round=600,
            valid_sets=[lgb.Dataset(X_va, label=y_va)],
            callbacks=[lgb.early_stopping(40), lgb.log_evaluation(0)],
        )

        va_pred = model.predict(X_va)
        prefix_auc = roc_auc_score(y_va, va_pred)
        for sample_i, p in zip(va_sample_idx, va_pred):
            uid = train_samples[sample_i]["uid"]
            r = rally_to_idx[uid]
            oof_pred_per_rally[r] += p
            oof_count_per_rally[r] += 1
        print(f"  Fold {fold+1} prefix-AUC: {prefix_auc:.4f}, best_iter: {model.best_iteration}, n_trunk_in_top20: ", end="")
        fi = model.feature_importance(importance_type="gain")
        cols = list(X_tr.columns)
        top20 = np.argsort(fi)[::-1][:20]
        n_trunk_top = sum(1 for i in top20 if cols[i].startswith("trunk_"))
        print(f"{n_trunk_top}/20")
        fold_models.append((model, srv_w, srv_n_, rcv_l, rcv_n_, prior_sex, prior_global))

    oof_pred_rally = oof_pred_per_rally / np.maximum(oof_count_per_rally, 1)
    rally_mask = oof_count_per_rally > 0
    overall_auc = roc_auc_score(oof_label_rally[rally_mask], oof_pred_rally[rally_mask])
    print(f"\n{'='*70}")
    print(f"OOF AUC (per-rally aggregated, match-disjoint): {overall_auc:.4f}")

    # Test prediction
    print("\n--- Test prediction ---")
    srv_w_full, srv_n_full, rcv_l_full, rcv_n_full, prior_sex_full, prior_global_full = bayesian_player_stats(rally_meta_full, kappa)
    test_pred = np.zeros(len(test_samples))
    test_uids_ref = np.array([s["uid"] for s in test_samples])

    for model, srv_w_f, srv_n_f, rcv_l_f, rcv_n_f, ps_f, pg_f in fold_models:
        X_test_hand = build_test_features(test_df, test_samples, srv_w_f, srv_n_f, rcv_l_f, rcv_n_f, ps_f, pg_f, known_matches=train_matches)
        trunk_cols = [f"trunk_{i}" for i in range(trunk_test.shape[1])]
        X_test_trunk = pd.DataFrame(trunk_test, columns=trunk_cols, index=X_test_hand.index)
        X_test = pd.concat([X_test_hand, X_test_trunk], axis=1)
        test_pred += model.predict(X_test) / len(fold_models)

    print(f"  Test rallies: {len(test_pred)}")
    print(f"  Pred mean: {test_pred.mean():.4f}, std: {test_pred.std():.4f}, range: [{test_pred.min():.4f}, {test_pred.max():.4f}]")

    out_path = "artifacts/sgp_pred_NEWTEST_lgb_deep.npz"
    np.savez(out_path,
             rally_uids=test_uids_ref,
             sgp_pred=test_pred,
             method=f"lgb_deep_seeds{','.join(seeds)}_kappa{kappa}",
             holdout_auc=overall_auc,
             oof_pred=oof_pred_rally,
             oof_label=oof_label_rally)
    print(f"\n→ Saved {out_path}")
    print(f"   Holdout AUC: {overall_auc:.4f}")


if __name__ == "__main__":
    main()

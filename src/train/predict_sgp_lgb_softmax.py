#!/usr/bin/env python3
"""
Phase 4-B: LGB + LSTM action/point softmax features (29-dim, smaller than trunk's 128).

Hypothesis: LSTM trunk (128) had too many noise dims for 70k samples. Softmax
output is a more compressed, task-relevant representation. May escape the
LGB-deep failure (Pearson 0.90 with hand-features).

Per (rally, k) sample features:
- 39 hand-crafted (same as predict_sgp_lgb)
- LSTM action softmax (19) + LSTM point softmax (10)

Sources:
- Train: artifacts/v5z_md_full_NEWTEST_s{SEED}_oof.npz (OOF order, reindex to sample order)
- Test: artifacts/v5z_md_full_NEWTEST_s{SEED}_test.npz (1 per rally)

Holdout: match-disjoint StratifiedGroupKFold(N=5, groups=match_id).

Usage:
    V5Z_SEEDS=42  python3 src/train/predict_sgp_lgb_softmax.py
    V5Z_SEEDS=42,1337 python3 src/train/predict_sgp_lgb_softmax.py

Outputs:
    artifacts/sgp_pred_NEWTEST_lgb_softmax.npz
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


def make_hand_feats(rows, k, srv_w, srv_n, rcv_l, rcv_n, prior_sex, prior_global, known_matches=None):
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


def build_md_oof_reindex(seed):
    """Same logic as advcal_match_disjoint.build_md_oof_reindex."""
    train_samples = prepare_samples(train_df, is_train=True, augment=True)
    uid2idx = {}
    for i, s in enumerate(train_samples):
        uid2idx.setdefault(s["uid"], []).append(i)
    uid_last_a = {s["uid"]: s["target_action"] for s in train_samples}
    uid_to_match = train_df.groupby("rally_uid")["match"].first().to_dict()
    uid_list = sorted(uid_last_a.keys())
    uid_arr = np.array([uid_last_a[u] for u in uid_list])
    match_arr = np.array([uid_to_match[u] for u in uid_list])
    sgkf = StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=seed)
    sample_idx_to_oof_idx = np.empty(len(train_samples), dtype=np.int64)
    cur = 0
    for _, (_, va_uidx) in enumerate(sgkf.split(uid_list, uid_arr, groups=match_arr)):
        va_uids = set(np.array(uid_list)[va_uidx])
        for u in va_uids:
            for i in uid2idx.get(u, []):
                sample_idx_to_oof_idx[i] = cur
                cur += 1
    return sample_idx_to_oof_idx


def main():
    print("=" * 70)
    print("Phase 4-B LGB + LSTM softmax")
    print("=" * 70)

    seeds = os.environ.get("V5Z_SEEDS", "42").split(",")
    kappa = int(os.environ.get("V5Z_KAPPA", "50"))
    print(f"Seeds: {seeds}, κ={kappa}")

    # Load LSTM softmax for each seed, reindex OOF to sample order, average across seeds
    lstm_a_oof_seeds = []
    lstm_p_oof_seeds = []
    lstm_a_test_seeds = []
    lstm_p_test_seeds = []
    test_uids_ref = None
    for seed in seeds:
        d_oof = np.load(f"artifacts/v5z_md_full_NEWTEST_s{seed}_oof.npz")
        d_test = np.load(f"artifacts/v5z_md_full_NEWTEST_s{seed}_test.npz")
        s2o = build_md_oof_reindex(int(seed))  # sample_idx → oof_idx
        # Reindex: lstm_a_in_sample_order[i] = lstm_a_oof[s2o[i]]
        lstm_a_oof_seeds.append(d_oof["lstm_a"][s2o])
        lstm_p_oof_seeds.append(d_oof["lstm_p"][s2o])
        lstm_a_test_seeds.append(d_test["lstm_a"])
        lstm_p_test_seeds.append(d_test["lstm_p"])
        if test_uids_ref is None:
            test_uids_ref = d_test["test_uids"]
        print(f"  Loaded seed {seed}: oof shape {d_oof['lstm_a'].shape}, test shape {d_test['lstm_a'].shape}")
    lstm_a_oof = np.mean(lstm_a_oof_seeds, axis=0)
    lstm_p_oof = np.mean(lstm_p_oof_seeds, axis=0)
    lstm_a_test = np.mean(lstm_a_test_seeds, axis=0)
    lstm_p_test = np.mean(lstm_p_test_seeds, axis=0)
    print(f"  Avg OOF: action {lstm_a_oof.shape}, point {lstm_p_oof.shape}")
    print(f"  Avg test: action {lstm_a_test.shape}, point {lstm_p_test.shape}")

    # prepare_samples (same order for both train OOF and test)
    train_samples = prepare_samples(train_df, is_train=True, augment=True)
    test_samples = prepare_samples(test_df, is_train=False)
    assert len(train_samples) == lstm_a_oof.shape[0]
    assert len(test_samples) == lstm_a_test.shape[0]

    rally_meta_full = train_df.groupby("rally_uid").first()[
        ["gamePlayerId", "gamePlayerOtherId", "serverGetPoint", "sex", "match"]
    ].reset_index()
    train_matches = set(rally_meta_full["match"].unique())

    rally_rows_train = {uid: grp.sort_values("strikeNumber").to_dict("records")
                        for uid, grp in train_df.groupby("rally_uid")}
    rally_rows_test = {uid: grp.sort_values("strikeNumber").to_dict("records")
                       for uid, grp in test_df.groupby("rally_uid")}

    uid_to_idx = {u: i for i, u in enumerate(rally_meta_full["rally_uid"].values)}
    uid2sampleidx = {}
    for i, s in enumerate(train_samples):
        uid2sampleidx.setdefault(s["uid"], []).append(i)

    uid_last_a = {s["uid"]: s["target_action"] for s in train_samples}
    uid_to_match = train_df.groupby("rally_uid")["match"].first().to_dict()
    uid_list = sorted(uid_last_a.keys())
    uid_arr = np.array([uid_last_a[u] for u in uid_list])
    match_arr = np.array([uid_to_match[u] for u in uid_list])

    sgkf = StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=42)
    kf_splits = list(sgkf.split(uid_list, uid_arr, groups=match_arr))

    oof_pred_per_rally = np.zeros(len(rally_meta_full))
    oof_count_per_rally = np.zeros(len(rally_meta_full))
    oof_label_rally = rally_meta_full["serverGetPoint"].values.astype(np.float64)
    fold_models = []

    for fold, (tr_uidx, va_uidx) in enumerate(kf_splits):
        tr_uids = set(np.array(uid_list)[tr_uidx])
        va_uids = set(np.array(uid_list)[va_uidx])
        tr_meta = rally_meta_full[rally_meta_full["rally_uid"].isin(tr_uids)]
        srv_w, srv_n_, rcv_l, rcv_n_, prior_sex, prior_global = bayesian_player_stats(tr_meta, kappa)

        tr_sample_idx = [i for u in tr_uids for i in uid2sampleidx.get(u, [])]
        va_sample_idx = [i for u in va_uids for i in uid2sampleidx.get(u, [])]

        # Build hand-crafted features for tr/va samples
        feats_tr, sgp_tr = [], []
        for i in tr_sample_idx:
            s = train_samples[i]
            f = make_hand_feats(rally_rows_train[s["uid"]], s["length"], srv_w, srv_n_, rcv_l, rcv_n_, prior_sex, prior_global)
            feats_tr.append(f)
            sgp_tr.append(int(rally_rows_train[s["uid"]][0]["serverGetPoint"]))
        feats_va, sgp_va = [], []
        for i in va_sample_idx:
            s = train_samples[i]
            f = make_hand_feats(rally_rows_train[s["uid"]], s["length"], srv_w, srv_n_, rcv_l, rcv_n_, prior_sex, prior_global)
            feats_va.append(f)
            sgp_va.append(int(rally_rows_train[s["uid"]][0]["serverGetPoint"]))
        X_tr_hand = pd.DataFrame(feats_tr)
        X_va_hand = pd.DataFrame(feats_va)
        y_tr = np.array(sgp_tr)
        y_va = np.array(sgp_va)

        # Append softmax features
        sm_cols = [f"action_p{i}" for i in range(19)] + [f"point_p{i}" for i in range(10)]
        X_tr_sm = pd.DataFrame(
            np.hstack([lstm_a_oof[tr_sample_idx], lstm_p_oof[tr_sample_idx]]),
            columns=sm_cols, index=X_tr_hand.index
        )
        X_va_sm = pd.DataFrame(
            np.hstack([lstm_a_oof[va_sample_idx], lstm_p_oof[va_sample_idx]]),
            columns=sm_cols, index=X_va_hand.index
        )
        X_tr = pd.concat([X_tr_hand, X_tr_sm], axis=1)
        X_va = pd.concat([X_va_hand, X_va_sm], axis=1)

        print(f"\nFold {fold+1}/{N_FOLDS}: Tr {len(X_tr)} samples, Va {len(X_va)}, features={X_tr.shape[1]}")

        model = lgb.train(
            {
                "objective": "binary", "metric": "auc",
                "learning_rate": 0.03, "num_leaves": 31,
                "min_data_in_leaf": 50, "lambda_l2": 1.0,
                "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5,
                "seed": 42, "verbose": -1,
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
            r = uid_to_idx[uid]
            oof_pred_per_rally[r] += p
            oof_count_per_rally[r] += 1
        fi = model.feature_importance(importance_type="gain")
        cols = list(X_tr.columns)
        top10_idx = np.argsort(fi)[::-1][:10]
        top10_names = [cols[i] for i in top10_idx]
        n_sm_top = sum(1 for n in top10_names if n.startswith(("action_p", "point_p")))
        print(f"  Fold {fold+1} prefix-AUC: {prefix_auc:.4f}, best_iter: {model.best_iteration}, softmax-in-top10: {n_sm_top}")
        fold_models.append((model, srv_w, srv_n_, rcv_l, rcv_n_, prior_sex, prior_global))

    oof_pred_rally = oof_pred_per_rally / np.maximum(oof_count_per_rally, 1)
    rally_mask = oof_count_per_rally > 0
    overall_auc = roc_auc_score(oof_label_rally[rally_mask], oof_pred_rally[rally_mask])
    print(f"\n{'='*70}")
    print(f"OOF AUC (per-rally aggregated): {overall_auc:.4f}")

    # Test prediction
    print("\n--- Test prediction ---")
    srv_w_full, srv_n_full, rcv_l_full, rcv_n_full, prior_sex_full, prior_global_full = bayesian_player_stats(rally_meta_full, kappa)
    test_pred = np.zeros(len(test_samples))

    for model, srv_w_f, srv_n_f, rcv_l_f, rcv_n_f, ps_f, pg_f in fold_models:
        feats_test = []
        for s in test_samples:
            f = make_hand_feats(rally_rows_test[s["uid"]], s["length"], srv_w_f, srv_n_f, rcv_l_f, rcv_n_f, ps_f, pg_f, known_matches=train_matches)
            feats_test.append(f)
        X_test_hand = pd.DataFrame(feats_test)
        sm_cols = [f"action_p{i}" for i in range(19)] + [f"point_p{i}" for i in range(10)]
        X_test_sm = pd.DataFrame(
            np.hstack([lstm_a_test, lstm_p_test]),
            columns=sm_cols, index=X_test_hand.index
        )
        X_test = pd.concat([X_test_hand, X_test_sm], axis=1)
        test_pred += model.predict(X_test) / len(fold_models)

    print(f"  Test rallies: {len(test_pred)}")
    print(f"  Pred mean: {test_pred.mean():.4f}, std: {test_pred.std():.4f}, range: [{test_pred.min():.4f}, {test_pred.max():.4f}]")

    out_path = "artifacts/sgp_pred_NEWTEST_lgb_softmax.npz"
    np.savez(out_path,
             rally_uids=test_uids_ref,
             sgp_pred=test_pred,
             method=f"lgb_softmax_seeds{','.join(seeds)}_kappa{kappa}",
             holdout_auc=overall_auc,
             oof_pred=oof_pred_rally,
             oof_label=oof_label_rally)
    print(f"\n→ Saved {out_path}")
    print(f"   Holdout AUC: {overall_auc:.4f}")


if __name__ == "__main__":
    main()

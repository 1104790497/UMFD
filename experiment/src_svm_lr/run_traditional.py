import os
import warnings
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import xgboost as xgb
import lightgbm as lgb

from config import (
    TRAIN_FILE, VAL_FILE, TEST_FILE, OUTPUT_DIR, RESULTS_FILE,
    TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE, TFIDF_MAX_DF, TFIDF_MIN_DF,
    SVM_C_VALUES, LR_C_VALUES,
    XGB_N_ESTIMATORS, XGB_MAX_DEPTH, XGB_LR_VALUES,
    LGB_N_ESTIMATORS, LGB_MAX_DEPTH, LGB_LR_VALUES,
    RANDOM_STATE, MAX_ITER,
    PMFT_COLS, CMFT_COLS, STANCE_MAP, STANCE_NAMES,
)

warnings.filterwarnings("ignore")


def load_data():
    train_df = pd.read_excel(TRAIN_FILE, engine="openpyxl")
    val_df = pd.read_excel(VAL_FILE, engine="openpyxl")
    test_df = pd.read_excel(TEST_FILE, engine="openpyxl")

    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        assert all(c in df.columns for c in PMFT_COLS + CMFT_COLS + ["post_text", "combined_text", "stance"]), \
            f"Missing columns in {name}"

    print(f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")
    for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        print(f"  {name} FAVOR={len(df[df['stance']=='FAVOR'])} AGAINST={len(df[df['stance']=='AGAINST'])}")

    train_texts = (train_df["post_text"].fillna("") + " " + train_df["combined_text"].fillna("")).tolist()
    val_texts = (val_df["post_text"].fillna("") + " " + val_df["combined_text"].fillna("")).tolist()
    test_texts = (test_df["post_text"].fillna("") + " " + test_df["combined_text"].fillna("")).tolist()

    y_train = train_df["stance"].map(STANCE_MAP).values.astype(np.int64)
    y_val = val_df["stance"].map(STANCE_MAP).values.astype(np.int64)
    y_test = test_df["stance"].map(STANCE_MAP).values.astype(np.int64)

    return train_texts, val_texts, test_texts, y_train, y_val, y_test, train_df, val_df, test_df


def extract_mft_features(df):
    feats = np.zeros((len(df), 8), dtype=np.float32)
    for idx, (_, row) in enumerate(df.iterrows()):
        for j in range(5):
            p = int(row[PMFT_COLS[j]])
            c = int(row[CMFT_COLS[j]])
            feats[idx, j] = 1.0 if (p != 0 and c != 0 and p != c) else 0.0

        p_neg_c0 = 0
        for j in range(5):
            if int(row[PMFT_COLS[j]]) == -1 and int(row[CMFT_COLS[j]]) == 0:
                p_neg_c0 += 1
        feats[idx, 5] = float(p_neg_c0)

        p0_c_pos = 0
        for j in range(5):
            if int(row[PMFT_COLS[j]]) == 0 and int(row[CMFT_COLS[j]]) == 1:
                p0_c_pos += 1
        feats[idx, 6] = float(p0_c_pos)

        post_active = sum(1 for j in range(5) if int(row[PMFT_COLS[j]]) != 0)
        comment_active = sum(1 for j in range(5) if int(row[CMFT_COLS[j]]) != 0)
        feats[idx, 7] = float(post_active - comment_active)

    return feats


def grid_search(X_train, y_train, X_val, y_val, model_type):
    if model_type in ("SVM", "LR"):
        return _grid_search_linear(X_train, y_train, X_val, y_val, model_type)
    elif model_type == "XGB":
        return _grid_search_tree(X_train, y_train, X_val, y_val, "XGB")
    elif model_type == "LGB":
        return _grid_search_tree(X_train, y_train, X_val, y_val, "LGB")
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def _grid_search_linear(X_train, y_train, X_val, y_val, model_type):
    c_values = SVM_C_VALUES if model_type == "SVM" else LR_C_VALUES
    best_params = None
    best_val_f1 = -1.0

    for c in c_values:
        if model_type == "SVM":
            model = LinearSVC(C=c, max_iter=MAX_ITER, random_state=RANDOM_STATE,
                              dual="auto", class_weight="balanced")
        else:
            model = LogisticRegression(C=c, max_iter=MAX_ITER, random_state=RANDOM_STATE,
                                       class_weight="balanced")

        model.fit(X_train, y_train)
        val_preds = model.predict(X_val)
        val_f1 = f1_score(y_val, val_preds, average="macro")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_params = {"C": c}

    return best_params, best_val_f1


def _grid_search_tree(X_train, y_train, X_val, y_val, model_type):
    if model_type == "XGB":
        n_estimators_list = XGB_N_ESTIMATORS
        max_depth_list = XGB_MAX_DEPTH
        lr_list = XGB_LR_VALUES
    else:
        n_estimators_list = LGB_N_ESTIMATORS
        max_depth_list = LGB_MAX_DEPTH
        lr_list = LGB_LR_VALUES

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    best_params = None
    best_val_f1 = -1.0

    for n_est in n_estimators_list:
        for md in max_depth_list:
            for lr in lr_list:
                if model_type == "XGB":
                    model = xgb.XGBClassifier(
                        n_estimators=n_est, max_depth=md, learning_rate=lr,
                        scale_pos_weight=scale_pos_weight,
                        eval_metric="logloss",
                        random_state=RANDOM_STATE, verbosity=0, n_jobs=-1,
                    )
                else:
                    model = lgb.LGBMClassifier(
                        n_estimators=n_est, max_depth=md, learning_rate=lr,
                        class_weight="balanced",
                        random_state=RANDOM_STATE, verbose=-1, n_jobs=-1,
                    )

                if model_type == "XGB":
                    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                else:
                    model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

                val_preds = model.predict(X_val)
                val_f1 = f1_score(y_val, val_preds, average="macro")

                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_params = {"n_estimators": n_est, "max_depth": md, "learning_rate": lr}

    return best_params, best_val_f1


def run_experiments():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_texts, val_texts, test_texts, y_train, y_val, y_test, train_df, val_df, test_df = load_data()

    print("\n=== TF-IDF Vectorization ===")
    tfidf = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
        max_df=TFIDF_MAX_DF,
        min_df=TFIDF_MIN_DF,
        sublinear_tf=True,
    )
    X_train_tfidf = tfidf.fit_transform(train_texts)
    X_val_tfidf = tfidf.transform(val_texts)
    X_test_tfidf = tfidf.transform(test_texts)
    print(f"TF-IDF vocab size: {len(tfidf.vocabulary_)}")

    print("\n=== MFT 8-dim Feature Extraction ===")
    mft_train = extract_mft_features(train_df)
    mft_val = extract_mft_features(val_df)
    mft_test = extract_mft_features(test_df)

    scaler = StandardScaler()
    mft_train_scaled = scaler.fit_transform(mft_train)
    mft_val_scaled = scaler.transform(mft_val)
    mft_test_scaled = scaler.transform(mft_test)

    X_train_b = hstack([X_train_tfidf, mft_train_scaled])
    X_val_b = hstack([X_val_tfidf, mft_val_scaled])
    X_test_b = hstack([X_test_tfidf, mft_test_scaled])

    experiments = {
        "A": {"X_train": X_train_tfidf, "X_val": X_val_tfidf, "X_test": X_test_tfidf,
              "label": "TF-IDF"},
        "B": {"X_train": X_train_b, "X_val": X_val_b, "X_test": X_test_b,
              "label": "TF-IDF + MFT"},
    }

    model_classes = {
        "SVM": ("linear",),
        "LR": ("linear",),
        "XGB": ("tree",),
        "LGB": ("tree",),
    }

    results = []

    for exp_name, exp_data in experiments.items():
        for model_name, (model_family,) in model_classes.items():
            print(f"\n{'=' * 60}")
            print(f"Experiment {exp_name} ({exp_data['label']})  |  Model: {model_name}")
            print(f"{'=' * 60}")

            best_params, best_val_f1 = grid_search(
                exp_data["X_train"], y_train,
                exp_data["X_val"], y_val,
                model_name,
            )
            print(f"Best params: {best_params}  |  Val Macro F1: {best_val_f1:.4f}")

            if model_name == "SVM":
                model = LinearSVC(C=best_params["C"], max_iter=MAX_ITER,
                                  random_state=RANDOM_STATE, dual="auto",
                                  class_weight="balanced")
            elif model_name == "LR":
                model = LogisticRegression(C=best_params["C"], max_iter=MAX_ITER,
                                           random_state=RANDOM_STATE,
                                           class_weight="balanced")
            elif model_name == "XGB":
                scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
                model = xgb.XGBClassifier(
                    n_estimators=best_params["n_estimators"],
                    max_depth=best_params["max_depth"],
                    learning_rate=best_params["learning_rate"],
                    scale_pos_weight=scale_pos_weight,
                    eval_metric="logloss",
                    random_state=RANDOM_STATE, verbosity=0, n_jobs=-1,
                )
            elif model_name == "LGB":
                model = lgb.LGBMClassifier(
                    n_estimators=best_params["n_estimators"],
                    max_depth=best_params["max_depth"],
                    learning_rate=best_params["learning_rate"],
                    class_weight="balanced",
                    random_state=RANDOM_STATE, verbose=-1, n_jobs=-1,
                )

            model.fit(exp_data["X_train"], y_train)

            train_preds = model.predict(exp_data["X_train"])
            val_preds = model.predict(exp_data["X_val"])
            test_preds = model.predict(exp_data["X_test"])

            train_acc = accuracy_score(y_train, train_preds)
            train_f1 = f1_score(y_train, train_preds, average="macro")
            val_acc = accuracy_score(y_val, val_preds)
            val_f1 = f1_score(y_val, val_preds, average="macro")
            test_acc = accuracy_score(y_test, test_preds)
            test_f1 = f1_score(y_test, test_preds, average="macro")

            p = precision_score(y_test, test_preds, average=None, zero_division=0)
            r = recall_score(y_test, test_preds, average=None, zero_division=0)
            f = f1_score(y_test, test_preds, average=None, zero_division=0)

            print(f"Train: Acc={train_acc:.4f}  F1={train_f1:.4f}")
            print(f"Val:   Acc={val_acc:.4f}  F1={val_f1:.4f}")
            print(f"Test:  Acc={test_acc:.4f}  F1={test_f1:.4f}")
            print("\nTest Classification Report:")
            print(classification_report(y_test, test_preds, target_names=STANCE_NAMES, digits=4))

            results.append({
                "exp": exp_name,
                "features": exp_data["label"],
                "model": model_name,
                "best_params": str(best_params),
                "val_f1": round(best_val_f1, 4),
                "train_acc": round(train_acc, 4),
                "train_f1": round(train_f1, 4),
                "val_acc": round(val_acc, 4),
                "val_f1": round(val_f1, 4),
                "test_acc": round(test_acc, 4),
                "test_f1": round(test_f1, 4),
                "p_favor": round(p[0], 4),
                "r_favor": round(r[0], 4),
                "f_favor": round(f[0], 4),
                "p_against": round(p[1], 4),
                "r_against": round(r[1], 4),
                "f_against": round(f[1], 4),
            })

    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")
    df_results = pd.DataFrame(results)
    df_results.to_csv(RESULTS_FILE, index=False)
    print(df_results.to_string(index=False))
    print(f"\nResults saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    run_experiments()

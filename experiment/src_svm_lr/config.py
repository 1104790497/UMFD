import os

BASE_DIR = r"F:\RoBERTa\RoBERTa-base"
DATA_DIR = os.path.join(BASE_DIR, "data")

TRAIN_FILE = os.path.join(DATA_DIR, "训练集.xlsx")
VAL_FILE = os.path.join(DATA_DIR, "验证集.xlsx")
TEST_FILE = os.path.join(DATA_DIR, "测试集.xlsx")

OUTPUT_DIR = os.path.join(BASE_DIR, "src6", "output")
RESULTS_FILE = os.path.join(OUTPUT_DIR, "results.csv")

TFIDF_MAX_FEATURES = 10000
TFIDF_NGRAM_RANGE = (1, 3)
TFIDF_MAX_DF = 0.9
TFIDF_MIN_DF = 2

SVM_C_VALUES = [0.01, 0.1, 0.5, 1, 5, 10, 50, 100]
LR_C_VALUES = [0.01, 0.1, 0.5, 1, 5, 10, 50, 100]

XGB_N_ESTIMATORS = [100, 200]
XGB_MAX_DEPTH = [3, 5, 7]
XGB_LR_VALUES = [0.1]

LGB_N_ESTIMATORS = [100, 200]
LGB_MAX_DEPTH = [3, 5, 7]
LGB_LR_VALUES = [0.1]

RANDOM_STATE = 42
MAX_ITER = 5000

PMFT_COLS = ["post_mft1", "post_mft2", "post_mft3", "post_mft4", "post_mft5"]
CMFT_COLS = ["comment_mft1", "comment_mft2", "comment_mft3", "comment_mft4", "comment_mft5"]

STANCE_MAP = {"FAVOR": 0, "AGAINST": 1}
STANCE_NAMES = ["FAVOR", "AGAINST"]

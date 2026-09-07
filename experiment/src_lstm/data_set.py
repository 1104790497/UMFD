import re
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from collections import Counter

STANCE_MAP = {"FAVOR": 0, "AGAINST": 1}

PMFT_COLS = ["post_mft1", "post_mft2", "post_mft3", "post_mft4", "post_mft5"]
CMFT_COLS = ["comment_mft1", "comment_mft2", "comment_mft3", "comment_mft4", "comment_mft5"]

PAD_IDX = 0
UNK_IDX = 1


def tokenize(text):
    text = str(text).lower()
    return re.findall(r"[a-z0-9]+", text)


def build_vocab(texts, max_size=30000, min_freq=2):
    counter = Counter()
    for t in texts:
        counter.update(tokenize(t))
    vocab = {"[PAD]": PAD_IDX, "[UNK]": UNK_IDX}
    idx = 2
    for word, freq in counter.most_common(max_size - 2):
        if freq < min_freq:
            break
        vocab[word] = idx
        idx += 1
    return vocab


def texts_to_ids(texts, vocab, max_len):
    ids = np.zeros((len(texts), max_len), dtype=np.int64)
    masks = np.zeros((len(texts), max_len), dtype=np.float32)
    for i, t in enumerate(texts):
        tokens = tokenize(t)
        seq = [vocab.get(w, UNK_IDX) for w in tokens[:max_len]]
        ids[i, :len(seq)] = seq
        masks[i, :len(seq)] = 1.0
    return ids, masks


def extract_mft_features(df):
    feats = np.zeros((len(df), 8), dtype=np.float32)
    for idx, (_, row) in enumerate(df.iterrows()):
        for j in range(5):
            p = int(row[PMFT_COLS[j]])
            c = int(row[CMFT_COLS[j]])
            feats[idx, j] = 1.0 if (p != 0 and c != 0 and p != c) else 0.0
        p_neg_c0 = sum(1 for j in range(5) if int(row[PMFT_COLS[j]]) == -1 and int(row[CMFT_COLS[j]]) == 0)
        feats[idx, 5] = float(p_neg_c0)
        p0_c_pos = sum(1 for j in range(5) if int(row[PMFT_COLS[j]]) == 0 and int(row[CMFT_COLS[j]]) == 1)
        feats[idx, 6] = float(p0_c_pos)
        post_active = sum(1 for j in range(5) if int(row[PMFT_COLS[j]]) != 0)
        comment_active = sum(1 for j in range(5) if int(row[CMFT_COLS[j]]) != 0)
        feats[idx, 7] = float(post_active - comment_active)
    return feats


class LSTMStanceDataset(Dataset):
    def __init__(self, ids, masks, mft_features, labels):
        self.ids = ids
        self.masks = masks
        self.labels = labels
        self.mft_features = mft_features

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        item = {
            "input_ids": torch.tensor(self.ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.masks[idx], dtype=torch.float),
            "stance": torch.tensor(self.labels[idx], dtype=torch.long),
        }
        if self.mft_features is not None:
            item["mft_features"] = torch.tensor(self.mft_features[idx], dtype=torch.float)
        return item

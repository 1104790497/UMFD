import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from transformers import RobertaTokenizer

STANCE_MAP = {"FAVOR": 0, "AGAINST": 1}

CMFT_COLS = ["comment_mft1", "comment_mft2", "comment_mft3", "comment_mft4", "comment_mft5"]
PMFT_COLS = ["post_mft1", "post_mft2", "post_mft3", "post_mft4", "post_mft5"]


class StanceDataset(Dataset):
    def __init__(self, data_path, model_path, max_seq_len=512, max_comment_len=150,
                 use_mft=False):
        self.df = pd.read_excel(data_path, engine="openpyxl")
        self.tokenizer = RobertaTokenizer.from_pretrained(model_path)
        self.max_seq_len = max_seq_len
        self.max_comment_len = max_comment_len
        self.use_mft = use_mft

        self.mft_vecs = None
        if use_mft:
            self.mft_vecs = []
            for _, row in self.df.iterrows():
                feats = []
                # 1-5: conflict per dimension
                for j in range(5):
                    p = int(row[PMFT_COLS[j]])
                    c = int(row[CMFT_COLS[j]])
                    feats.append(1.0 if (p != 0 and c != 0 and p != c) else 0.0)
                # 6: P- C0 count
                p_neg_c0 = 0
                for j in range(5):
                    if int(row[PMFT_COLS[j]]) == -1 and int(row[CMFT_COLS[j]]) == 0:
                        p_neg_c0 += 1
                feats.append(float(p_neg_c0))
                # 7: P0 C+ count
                p0_c_pos = 0
                for j in range(5):
                    if int(row[PMFT_COLS[j]]) == 0 and int(row[CMFT_COLS[j]]) == 1:
                        p0_c_pos += 1
                feats.append(float(p0_c_pos))
                # 8: activation gap
                post_active = sum(1 for j in range(5) if int(row[PMFT_COLS[j]]) != 0)
                comment_active = sum(1 for j in range(5) if int(row[CMFT_COLS[j]]) != 0)
                feats.append(float(post_active - comment_active))
                self.mft_vecs.append(np.array(feats, dtype=np.float32))
            self.mft_dim = 8
        else:
            self.mft_dim = 0

        self.labels = self.df['stance'].map(STANCE_MAP).values.astype(np.int64)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        comment_text = str(row['combined_text'])
        post_text = str(row['post_text'])

        comment_tokens = self.tokenizer.encode(comment_text, add_special_tokens=False)
        if len(comment_tokens) > self.max_comment_len:
            comment_tokens = comment_tokens[:self.max_comment_len]

        max_post_len = self.max_seq_len - 3 - len(comment_tokens)
        if max_post_len < 0:
            max_post_len = 0

        post_tokens = self.tokenizer.encode(post_text, add_special_tokens=False)
        if len(post_tokens) > max_post_len:
            post_tokens = post_tokens[:max_post_len]

        input_ids = [self.tokenizer.cls_token_id] + post_tokens + \
                    [self.tokenizer.sep_token_id] + comment_tokens + [self.tokenizer.sep_token_id]
        attention_mask = [1] * len(input_ids)

        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[:self.max_seq_len]
            attention_mask = attention_mask[:self.max_seq_len]

        item = {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'stance': torch.tensor(self.labels[idx], dtype=torch.long),
        }
        if self.use_mft and self.mft_vecs is not None:
            item['mft_features'] = torch.tensor(self.mft_vecs[idx], dtype=torch.float)
        return item


def collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    stance = torch.stack([item['stance'] for item in batch])
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=1)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)

    result = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'stance': stance,
    }
    if 'mft_features' in batch[0]:
        result['mft_features'] = torch.stack([item['mft_features'] for item in batch])
    return result

import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import pandas as pd

from data_set import (
    build_vocab, texts_to_ids, extract_mft_features,
    LSTMStanceDataset, STANCE_MAP,
)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=300, hidden_dim=300,
                 num_layers=2, dropout=0.5, num_classes=2, use_mft=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers,
                            bidirectional=True, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.use_mft = use_mft
        mft_dim = 8 if use_mft else 0
        self.attn = nn.Linear(hidden_dim * 2, 1)
        self.classifier = nn.Linear(hidden_dim * 2 + mft_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask, mft_features=None):
        embedded = self.embedding(input_ids.long())
        embedded = self.dropout(embedded)
        lstm_out, _ = self.lstm(embedded)

        attn_scores = self.attn(lstm_out).squeeze(-1)
        attn_scores = attn_scores.masked_fill(attention_mask == 0, -1e9)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        pooled = torch.sum(lstm_out * attn_weights.unsqueeze(-1), dim=1)
        pooled = self.dropout(pooled)

        if self.use_mft and mft_features is not None:
            combined = torch.cat([pooled, mft_features], dim=1)
        else:
            combined = pooled
        return self.classifier(combined)


def train_epoch(model, dataloader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        stance = batch["stance"].to(device)
        mft = batch.get("mft_features", None)
        if mft is not None:
            mft = mft.to(device)

        optimizer.zero_grad()
        logits = model(input_ids, mask, mft_features=mft)
        loss = criterion(logits, stance)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(stance.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, acc, f1


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        stance = batch["stance"].to(device)
        mft = batch.get("mft_features", None)
        if mft is not None:
            mft = mft.to(device)

        logits = model(input_ids, mask, mft_features=mft)
        loss = criterion(logits, stance)

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(stance.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, acc, f1, all_labels, all_preds


EXP_CONFIG = {
    "A": {"use_mft": False, "label": "baseline (no MFT)"},
    "B": {"use_mft": True,  "label": "conflict features"},
}


def run_experiment(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    cfg = EXP_CONFIG[args.exp]
    print(f"Experiment: {args.exp}  {cfg['label']}")

    train_df = pd.read_excel(args.train_file, engine="openpyxl")
    val_df = pd.read_excel(args.val_file, engine="openpyxl")
    test_df = pd.read_excel(args.test_file, engine="openpyxl")

    train_texts = (train_df["post_text"].fillna("") + " " + train_df["combined_text"].fillna("")).tolist()
    val_texts = (val_df["post_text"].fillna("") + " " + val_df["combined_text"].fillna("")).tolist()
    test_texts = (test_df["post_text"].fillna("") + " " + test_df["combined_text"].fillna("")).tolist()

    y_train = train_df["stance"].map(STANCE_MAP).values.astype(np.int64)
    y_val = val_df["stance"].map(STANCE_MAP).values.astype(np.int64)
    y_test = test_df["stance"].map(STANCE_MAP).values.astype(np.int64)

    print(f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")

    print("Building vocabulary ...")
    vocab = build_vocab(train_texts, max_size=args.vocab_size, min_freq=2)
    print(f"Vocab size: {len(vocab)}")

    X_train_ids, X_train_mask = texts_to_ids(train_texts, vocab, args.max_seq_len)
    X_val_ids, X_val_mask = texts_to_ids(val_texts, vocab, args.max_seq_len)
    X_test_ids, X_test_mask = texts_to_ids(test_texts, vocab, args.max_seq_len)

    mft_train = extract_mft_features(train_df) if cfg["use_mft"] else None
    mft_val = extract_mft_features(val_df) if cfg["use_mft"] else None
    mft_test = extract_mft_features(test_df) if cfg["use_mft"] else None

    train_dataset = LSTMStanceDataset(X_train_ids, X_train_mask, mft_train, y_train)
    val_dataset = LSTMStanceDataset(X_val_ids, X_val_mask, mft_val, y_val)
    test_dataset = LSTMStanceDataset(X_test_ids, X_test_mask, mft_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = LSTMModel(
        vocab_size=len(vocab),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        num_classes=2,
        use_mft=cfg["use_mft"],
    ).to(device)
    print(f"Model params (M): {sum(p.numel() for p in model.parameters()) / 1e6:.2f}")

    train_labels = y_train
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    print(f"Class weights: {class_weights.cpu().numpy()}")

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    best_val_f1 = 0.0
    best_epoch = -1
    patience_counter = 0
    best_model_state = None

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, device)
        val_loss, val_acc, val_f1, _, _ = evaluate(
            model, val_loader, criterion, device)

        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} F1: {train_f1:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"  -> New best (val F1={val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_model_state)
    test_loss, test_acc, test_f1, test_labels, test_preds = evaluate(
        model, test_loader, criterion, device)

    p = precision_score(test_labels, test_preds, average=None, zero_division=0)
    r = recall_score(test_labels, test_preds, average=None, zero_division=0)
    f = f1_score(test_labels, test_preds, average=None, zero_division=0)

    print("\n" + "=" * 50)
    print(f"Experiment: {args.exp} ({cfg['label']})")
    print(f"Best epoch: {best_epoch}, Val F1: {best_val_f1:.4f}")
    print(f"Test (best): Loss={test_loss:.4f} | Acc={test_acc:.4f} | Macro F1={test_f1:.4f}")
    print("\nTest Set Detailed Metrics:")
    print(classification_report(test_labels, test_preds, target_names=["FAVOR", "AGAINST"]))
    print("=" * 50)

    return {
        "seed": args.seed, "exp": args.exp,
        "test_acc": test_acc, "test_f1": test_f1,
        "best_val_f1": best_val_f1, "best_epoch": best_epoch,
        "p_favor": p[0], "r_favor": r[0], "f_favor": f[0],
        "p_against": p[1], "r_against": r[1], "f_against": f[1],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, choices=["A", "B"])
    parser.add_argument("--run_all", action="store_true", help="Run A->B")
    parser.add_argument("--train_file", type=str,
                        default="F:\\RoBERTa\\RoBERTa-base\\data\\训练集.xlsx")
    parser.add_argument("--val_file", type=str,
                        default="F:\\RoBERTa\\RoBERTa-base\\data\\验证集.xlsx")
    parser.add_argument("--test_file", type=str,
                        default="F:\\RoBERTa\\RoBERTa-base\\data\\测试集.xlsx")
    parser.add_argument("--max_seq_len", type=int, default=200)
    parser.add_argument("--vocab_size", type=int, default=30000)
    parser.add_argument("--embedding_dim", type=int, default=300)
    parser.add_argument("--hidden_dim", type=int, default=300)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)

    args = parser.parse_args()
    seeds = args.seeds if args.seeds else [args.seed]
    experiments = ["A", "B"] if args.run_all else ([args.exp] if args.exp else [])

    for seed in seeds:
        print(f"\n{'#' * 60}")
        print(f"Seed={seed}")
        print("#" * 60)

        seed_results = []
        for exp in experiments:
            exp_args = argparse.Namespace(**vars(args))
            exp_args.exp = exp
            exp_args.seed = seed
            exp_args.run_all = False
            try:
                res = run_experiment(exp_args)
                seed_results.append(res)
            except Exception as e:
                print(f"Experiment {exp} (seed={seed}) failed: {e}")
                continue

        if seed_results:
            df = pd.DataFrame(seed_results)
            filename = f"results_lstm_seed{seed}.csv"
            df.to_csv(filename, index=False)
            print(f"\nSeed {seed} results saved to {filename}")
            print(df.to_string(index=False))
        else:
            print(f"\nSeed {seed}: all experiments failed")

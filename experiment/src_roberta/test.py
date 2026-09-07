import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import RobertaModel, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import pandas as pd

from data_set import StanceDataset, collate_fn


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class StanceModel(nn.Module):
    def __init__(self, model_path, num_classes=2, freeze_layers=0, use_mft=False):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(model_path, torch_dtype=torch.float32)
        self.roberta.config.hidden_dropout_prob = 0.3
        self.roberta.config.attention_probs_dropout_prob = 0.3
        hidden_size = self.roberta.config.hidden_size
        self.use_mft = use_mft

        for param in self.roberta.embeddings.parameters():
            param.requires_grad = False
        print("Embedding layer frozen")

        if freeze_layers > 0:
            for i in range(min(freeze_layers, len(self.roberta.encoder.layer))):
                for param in self.roberta.encoder.layer[i].parameters():
                    param.requires_grad = False
            print(f"Frozen first {freeze_layers} encoder layers")

        mft_dim = 8 if use_mft else 0
        self.classifier = nn.Linear(hidden_size + mft_dim, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, input_ids, attention_mask, mft_features=None):
        input_ids = input_ids.long()
        attention_mask = attention_mask.long()
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        if self.use_mft and mft_features is not None:
            combined = torch.cat([cls_output, mft_features], dim=1)
        else:
            combined = cls_output
        logits = self.classifier(combined)
        return logits


def train_epoch(model, dataloader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        stance = batch['stance'].to(device)
        mft_feat = batch.get('mft_features', None)
        if mft_feat is not None:
            mft_feat = mft_feat.to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask, mft_features=mft_feat)
        loss = criterion(logits, stance)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=-1).cpu().numpy()
        labels = stance.cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels)

    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    return avg_loss, acc, f1


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            stance = batch['stance'].to(device)
            mft_feat = batch.get('mft_features', None)
            if mft_feat is not None:
                mft_feat = mft_feat.to(device)

            logits = model(input_ids, attention_mask, mft_features=mft_feat)
            loss = criterion(logits, stance)

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            labels = stance.cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)

    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    return avg_loss, acc, f1, all_labels, all_preds


EXP_CONFIG = {
    'A': {'use_mft': False, 'label': 'baseline (no MFT)'},
    'B': {'use_mft': True,  'label': 'conflict features'},
}


def run_experiment(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    cfg = EXP_CONFIG[args.exp]
    print(f"Experiment: {args.exp}  {cfg['label']}")

    dataset_kwargs = dict(
        data_path=None, model_path=args.model_path,
        max_seq_len=args.max_seq_len, max_comment_len=args.max_comment_len,
    )

    dataset_kwargs['data_path'] = args.train_file
    train_dataset = StanceDataset(use_mft=cfg['use_mft'], **dataset_kwargs)
    dataset_kwargs['data_path'] = args.val_file
    val_dataset = StanceDataset(use_mft=cfg['use_mft'], **dataset_kwargs)
    dataset_kwargs['data_path'] = args.test_file
    test_dataset = StanceDataset(use_mft=cfg['use_mft'], **dataset_kwargs)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=0, pin_memory=True)

    model = StanceModel(
        model_path=args.model_path, num_classes=2,
        freeze_layers=args.freeze_layers, use_mft=cfg['use_mft'],
    ).to(device)

    train_labels = train_dataset.labels
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    print(f"Class weights: {class_weights.cpu().numpy()}")

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    roberta_trainable = [p for p in model.roberta.parameters() if p.requires_grad]
    head_params = [p for n, p in model.named_parameters()
                   if not n.startswith('roberta.') and p.requires_grad]
    lr_roberta = args.learning_rate * 0.5
    lr_head = args.learning_rate * 50
    print(f"RoBERTa LR: {lr_roberta}, head LR: {lr_head}")
    optimizer = AdamW([
        {'params': roberta_trainable, 'lr': lr_roberta},
        {'params': head_params, 'lr': lr_head},
    ], weight_decay=args.weight_decay)

    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    best_val_f1 = 0.0
    best_epoch = -1
    patience_counter = 0
    best_model_state = None

    swa_start = int(args.epochs * 0.25)
    swa_state = None
    swa_n = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, device)
        val_loss, val_acc, val_f1, _, _ = evaluate(
            model, val_loader, criterion, device)

        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} F1: {train_f1:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f}")

        if epoch >= swa_start:
            swa_n += 1
            if swa_state is None:
                swa_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                for k in swa_state:
                    swa_state[k] = swa_state[k] + (model.state_dict()[k] - swa_state[k]) / swa_n

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

    final_preds = test_preds
    final_labels = test_labels

    print("\n" + "=" * 50)
    print(f"Experiment: {args.exp} ({cfg['label']})")
    print(f"Best epoch: {best_epoch}, Val F1: {best_val_f1:.4f}")
    print(f"Test (best): Loss={test_loss:.4f} | Acc={test_acc:.4f} | Macro F1={test_f1:.4f}")

    if swa_state is not None:
        model.load_state_dict(swa_state)
        _, _, swa_val_f1, _, _ = evaluate(model, val_loader, criterion, device)
        swa_test_loss, swa_test_acc, swa_test_f1, swa_test_labels, swa_test_preds = evaluate(
            model, test_loader, criterion, device)
        print(f"\nSWA (averaged {swa_n} models from epoch {swa_start}):")
        print(f"  Val F1={swa_val_f1:.4f}  Test Acc={swa_test_acc:.4f}  Macro F1={swa_test_f1:.4f}")
        print("\nTest Set Detailed Metrics (SWA):")
        print(classification_report(swa_test_labels, swa_test_preds, target_names=['FAVOR', 'AGAINST']))
        final_preds = swa_test_preds
        final_labels = swa_test_labels
    else:
        print("\nTest Set Detailed Metrics:")
        print(classification_report(test_labels, test_preds, target_names=['FAVOR', 'AGAINST']))
    print("=" * 50)

    p = precision_score(final_labels, final_preds, average=None, zero_division=0)
    r = recall_score(final_labels, final_preds, average=None, zero_division=0)
    f = f1_score(final_labels, final_preds, average=None, zero_division=0)

    return {
        'seed': args.seed, 'exp': args.exp, 'freeze_layers': args.freeze_layers,
        'test_acc': test_acc, 'test_f1': test_f1,
        'best_val_f1': best_val_f1, 'best_epoch': best_epoch,
        'swa_test_f1': swa_test_f1 if swa_state else test_f1,
        'swa_n': swa_n,
        'p_favor': p[0], 'r_favor': r[0], 'f_favor': f[0],
        'p_against': p[1], 'r_against': r[1], 'f_against': f[1],
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, choices=['A', 'B'])
    parser.add_argument('--run_all', action='store_true', help='Run A->B')
    parser.add_argument('--train_file', type=str,
                        default='F:\\RoBERTa\\RoBERTa-base\\data\\训练集.xlsx')
    parser.add_argument('--val_file', type=str,
                        default='F:\\RoBERTa\\RoBERTa-base\\data\\验证集.xlsx')
    parser.add_argument('--test_file', type=str,
                        default='F:\\RoBERTa\\RoBERTa-base\\data\\测试集.xlsx')
    parser.add_argument('--model_path', type=str,
                        default='F:\\RoBERTa\\RoBERTa-base\\model_files')
    parser.add_argument('--max_seq_len', type=int, default=512)
    parser.add_argument('--max_comment_len', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0.5)
    parser.add_argument('--patience', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--seeds', type=int, nargs='+', default=None)
    parser.add_argument('--freeze_layers', type=int, default=6)

    args = parser.parse_args()
    seeds = args.seeds if args.seeds else [args.seed]
    experiments = ['A', 'B'] if args.run_all else ([args.exp] if args.exp else [])

    for seed in seeds:
        print(f"\n{'#' * 60}")
        print(f"Seed={seed}")
        print('#' * 60)

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
            filename = f'results_seed{seed}.csv'
            df.to_csv(filename, index=False)
            print(f"\nSeed {seed} results saved to {filename}")
            print(df.to_string(index=False))
        else:
            print(f"\nSeed {seed}: all experiments failed")

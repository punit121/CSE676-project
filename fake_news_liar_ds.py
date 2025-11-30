import json
import math
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader

plt.switch_backend("Agg")

SEED = 101
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "LIAR"

COLS = [
    "label",
    "statement",
    "subject",
    "speaker",
    "job_title",
    "state_info",
    "party",
    "barely_true_counts",
    "false_counts",
    "half_true_counts",
    "mostly_true_counts",
    "pants_fire_counts",
    "context",
]
TRUE_LABELS = {"true", "mostly-true"}
MAX_LEN = 32
MIN_FREQ = 2
MAX_VOCAB = 20000
NUM_CLASSES = 2
LABELS = ["False", "True"]


def load_split(name):
    path = DATA_DIR / f"{name}.tsv"
    df = pd.read_csv(path, sep="\t", header=None, names=COLS)
    df = df.dropna(subset=["statement", "label"])
    df["label"] = df["label"].apply(lambda s: 1 if s in TRUE_LABELS else 0)
    return df[["statement", "label"]]


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_vocab(texts):
    counter = Counter()
    for text in texts:
        counter.update(clean_text(text).split())
    items = [w for w, f in counter.most_common(MAX_VOCAB) if f >= MIN_FREQ]
    itos = ["<pad>", "<unk>"] + items
    stoi = {w: i for i, w in enumerate(itos)}
    return itos, stoi


def encode(text, stoi, pad_id, unk_id):
    tokens = clean_text(text).split()
    ids = [stoi.get(tok, unk_id) for tok in tokens[:MAX_LEN]]
    if len(ids) < MAX_LEN:
        ids += [pad_id] * (MAX_LEN - len(ids))
    return ids


def df_to_tensors(df, stoi, pad_id, unk_id):
    x = [encode(text, stoi, pad_id, unk_id) for text in df["statement"].tolist()]
    y = df["label"].astype(int).tolist()
    return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def make_loaders(train_data, val_data, test_data, batch_size):
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)
    return train_loader, val_loader, test_loader


class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, pad_id, dropout=0.0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        emb = self.embedding(x)
        out, _ = self.rnn(emb)
        last = out[:, -1, :]
        last = self.dropout(last)
        return self.fc(last)


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, pad_id, dropout=0.0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        emb = self.embedding(x)
        out, _ = self.lstm(emb)
        last = out[:, -1, :]
        last = self.dropout(last)
        return self.fc(last)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, num_classes, pad_id, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.pos_encoder = PositionalEncoding(embed_dim, MAX_LEN)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, num_classes)
        self.pad_id = pad_id

    def forward(self, x):
        emb = self.embedding(x)
        emb = self.pos_encoder(emb)
        mask = x.eq(self.pad_id)
        enc_out = self.encoder(emb, src_key_padding_mask=mask)
        pooled = self.dropout(enc_out.mean(dim=1))
        return self.fc(pooled)


def build_model(name, vocab_size, pad_id, cfg):
    if name == "RNN":
        return RNNClassifier(vocab_size, cfg["embed_dim"], cfg["hidden_dim"], NUM_CLASSES, pad_id, cfg.get("dropout", 0.0))
    if name == "LSTM":
        return LSTMClassifier(vocab_size, cfg["embed_dim"], cfg["hidden_dim"], NUM_CLASSES, pad_id, cfg.get("dropout", 0.0))
    if name == "Transformer":
        return TransformerClassifier(vocab_size, cfg["embed_dim"], cfg.get("num_heads", 4), cfg.get("num_layers", 2), NUM_CLASSES, pad_id, cfg.get("dropout", 0.1))
    raise ValueError(f"Unknown model {name}")


def clone_state_dict(state_dict):
    return {k: v.detach().cpu().clone() for k, v in state_dict.items()}


def evaluate(model, loader):
    was_training = model.training
    model.eval()
    preds = []
    labels = []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            logits = model(x_batch)
            preds.append(torch.argmax(logits, dim=1).cpu().numpy())
            labels.append(y_batch.cpu().numpy())
    if was_training:
        model.train()
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    cm = confusion_matrix(labels, preds)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "cm": cm}


def train_model(model, train_loader, val_loader, epochs, lr, label):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_state = clone_state_dict(model.state_dict())
    best_metrics = None
    best_f1 = -1.0
    for epoch in range(epochs):
        model.train()
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
        val_metrics = evaluate(model, val_loader)
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_state = clone_state_dict(model.state_dict())
            best_metrics = val_metrics
        print(f"{label} epoch {epoch + 1}/{epochs} val_acc {val_metrics['accuracy']:.4f} val_f1 {val_metrics['f1']:.4f}")
    model.load_state_dict(best_state)
    model = model.to(device)
    return model, best_metrics


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def plot_bar(names, values, title, ylabel, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(names, values, color="#4a90e2")
    ax.set_ylim(0, 1)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    for idx, val in enumerate(values):
        ax.text(idx, min(0.98, val + 0.02), f"{val:.2f}", ha="center")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_confusion(cm, labels, title, path):
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)

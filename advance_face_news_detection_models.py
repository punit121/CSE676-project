from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import TensorDataset

from fake_news_liar_ds import (
    LABELS,
    build_model,
    build_vocab,
    clone_state_dict,
    df_to_tensors,
    evaluate,
    load_split,
    make_loaders,
    plot_bar,
    plot_confusion,
    save_json,
    device,
)

BASE_DIR = Path(__file__).resolve().parent
ADV_MODEL_DIR = BASE_DIR / "model" / "LIAR" / "advance"
ADV_OUTPUT_DIR = BASE_DIR / "output" / "LIAR" / "advance"

for path in [ADV_MODEL_DIR, ADV_OUTPUT_DIR]:
    path.mkdir(parents=True, exist_ok=True)

ADVANCED_GRID = {
    "RNN": [
        {
            "batch_size": 128,
            "embed_dim": 192,
            "hidden_dim": 384,
            "epochs": 16,
            "lr": 3e-4,
            "dropout": 0.4,
            "patience": 3,
            "weight_decay": 1e-4,
            "lr_decay": 0.6,
            "min_lr": 3e-5,
            "grad_clip": 1.0,
        },
        {
            "batch_size": 160,
            "embed_dim": 224,
            "hidden_dim": 448,
            "epochs": 18,
            "lr": 2.5e-4,
            "dropout": 0.45,
            "patience": 4,
            "weight_decay": 1.5e-4,
            "lr_decay": 0.65,
            "min_lr": 2e-5,
            "grad_clip": 1.0,
        },
        {
            "batch_size": 192,
            "embed_dim": 256,
            "hidden_dim": 512,
            "epochs": 20,
            "lr": 2e-4,
            "dropout": 0.5,
            "patience": 4,
            "weight_decay": 1e-4,
            "lr_decay": 0.7,
            "min_lr": 1e-5,
            "grad_clip": 1.2,
        },
    ],
    "LSTM": [
        {
            "batch_size": 96,
            "embed_dim": 192,
            "hidden_dim": 384,
            "epochs": 16,
            "lr": 3e-4,
            "dropout": 0.4,
            "patience": 3,
            "weight_decay": 1e-4,
            "lr_decay": 0.6,
            "min_lr": 3e-5,
            "grad_clip": 0.9,
        },
        {
            "batch_size": 128,
            "embed_dim": 224,
            "hidden_dim": 448,
            "epochs": 18,
            "lr": 2.5e-4,
            "dropout": 0.45,
            "patience": 4,
            "weight_decay": 1.5e-4,
            "lr_decay": 0.65,
            "min_lr": 2e-5,
            "grad_clip": 1.0,
        },
        {
            "batch_size": 160,
            "embed_dim": 256,
            "hidden_dim": 512,
            "epochs": 20,
            "lr": 2e-4,
            "dropout": 0.5,
            "patience": 5,
            "weight_decay": 1e-4,
            "lr_decay": 0.7,
            "min_lr": 1e-5,
            "grad_clip": 1.2,
        },
    ],
    "Transformer": [
        {
            "batch_size": 80,
            "embed_dim": 288,
            "hidden_dim": 320,
            "epochs": 18,
            "lr": 4e-4,
            "dropout": 0.3,
            "num_heads": 6,
            "num_layers": 3,
            "patience": 3,
            "weight_decay": 1e-4,
            "lr_decay": 0.65,
            "min_lr": 3e-5,
            "grad_clip": 0.8,
        },
        {
            "batch_size": 96,
            "embed_dim": 320,
            "hidden_dim": 384,
            "epochs": 20,
            "lr": 3e-4,
            "dropout": 0.35,
            "num_heads": 8,
            "num_layers": 4,
            "patience": 4,
            "weight_decay": 1.5e-4,
            "lr_decay": 0.7,
            "min_lr": 2e-5,
            "grad_clip": 0.9,
        },
        {
            "batch_size": 112,
            "embed_dim": 384,
            "hidden_dim": 448,
            "epochs": 22,
            "lr": 2.5e-4,
            "dropout": 0.4,
            "num_heads": 8,
            "num_layers": 5,
            "patience": 5,
            "weight_decay": 1e-4,
            "lr_decay": 0.75,
            "min_lr": 1e-5,
            "grad_clip": 1.0,
        },
    ],
}


def train_with_early_stopping(model, train_loader, val_loader, cfg, label):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.0))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=cfg.get("lr_decay", 0.5),
        patience=cfg.get("lr_patience", 1),
        min_lr=cfg.get("min_lr", 1e-5),
        verbose=False,
    )
    best_state = clone_state_dict(model.state_dict())
    best_metrics = None
    best_f1 = -1.0
    best_epoch = 0
    epochs_without_improve = 0
    patience = cfg.get("patience", 2)
    min_delta = cfg.get("min_delta", 0.0)
    grad_clip = cfg.get("grad_clip", 1.0)
    epochs_run = 0
    for epoch in range(cfg["epochs"]):
        model.train()
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            if grad_clip:
                clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        val_metrics = evaluate(model, val_loader)
        scheduler.step(val_metrics["f1"])
        epochs_run = epoch + 1
        improved = val_metrics["f1"] > best_f1 + min_delta
        if improved:
            best_f1 = val_metrics["f1"]
            best_state = clone_state_dict(model.state_dict())
            best_metrics = val_metrics
            best_epoch = epochs_run
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
        print(f"{label} epoch {epochs_run}/{cfg['epochs']} val_acc {val_metrics['accuracy']:.4f} val_f1 {val_metrics['f1']:.4f}")
        if epochs_without_improve >= patience:
            break
    model.load_state_dict(best_state)
    model = model.to(device)
    return model, best_metrics, best_epoch, epochs_run


def run_advanced_stage(train_data, val_data, test_data, vocab_size, pad_id):
    metrics = {}
    names = []
    acc_values = []
    f1_values = []
    best_name = None
    best_cm = None
    best_f1 = -1.0
    for name, configs in ADVANCED_GRID.items():
        best_model = None
        best_cfg = None
        best_test = None
        best_val_f1 = -1.0
        for idx, cfg in enumerate(configs):
            train_loader, val_loader, test_loader = make_loaders(train_data, val_data, test_data, cfg["batch_size"])
            model = build_model(name, vocab_size, pad_id, cfg)
            model, val_metrics, best_epoch, epochs_run = train_with_early_stopping(
                model,
                train_loader,
                val_loader,
                cfg,
                f"advance_{name}_{idx + 1}",
            )
            test_metrics = evaluate(model, test_loader)
            if val_metrics["f1"] > best_val_f1:
                best_val_f1 = val_metrics["f1"]
                best_model = model
                best_cfg = dict(cfg)
                best_cfg["best_epoch"] = best_epoch
                best_cfg["trained_epochs"] = epochs_run
                best_test = test_metrics
        if best_model is None:
            continue
        cm_array = best_test["cm"]
        metrics[name] = {
            "accuracy": float(best_test["accuracy"]),
            "precision": float(best_test["precision"]),
            "recall": float(best_test["recall"]),
            "f1": float(best_test["f1"]),
            "cm": cm_array.tolist(),
            "config": best_cfg,
            "val_f1": float(best_val_f1),
        }
        if best_test["f1"] > best_f1:
            best_f1 = best_test["f1"]
            best_name = name
            best_cm = cm_array
        names.append(name)
        acc_values.append(best_test["accuracy"])
        f1_values.append(best_test["f1"])
        best_model = best_model.to("cpu")
        torch.save(best_model.state_dict(), ADV_MODEL_DIR / f"{name.lower()}_best.pt")
        print(f"advance {name} test_acc {best_test['accuracy']:.4f} test_f1 {best_test['f1']:.4f}")
    if metrics:
        save_json(metrics, ADV_OUTPUT_DIR / "metrics.json")
        plot_bar(names, acc_values, "Accuracy by Model (Advance)", "Accuracy", ADV_OUTPUT_DIR / "accuracy.png")
        plot_bar(names, f1_values, "F1 by Model (Advance)", "F1", ADV_OUTPUT_DIR / "f1.png")
        if best_cm is not None:
            plot_confusion(best_cm, LABELS, f"Advance {best_name} Confusion", ADV_OUTPUT_DIR / "confusion.png")


def main():
    train_df = load_split("train")
    val_df = load_split("valid")
    test_df = load_split("test")
    _, stoi = build_vocab(train_df["statement"].tolist())
    pad_id = stoi["<pad>"]
    unk_id = stoi["<unk>"]
    vocab_size = len(stoi)
    x_train, y_train = df_to_tensors(train_df, stoi, pad_id, unk_id)
    x_val, y_val = df_to_tensors(val_df, stoi, pad_id, unk_id)
    x_test, y_test = df_to_tensors(test_df, stoi, pad_id, unk_id)
    train_data = TensorDataset(x_train, y_train)
    val_data = TensorDataset(x_val, y_val)
    test_data = TensorDataset(x_test, y_test)
    run_advanced_stage(train_data, val_data, test_data, vocab_size, pad_id)


if __name__ == "__main__":
    main()

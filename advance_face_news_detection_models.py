from pathlib import Path

import torch
from torch.utils.data import TensorDataset

from fake_news_liar_ds import (
    LABELS,
    build_model,
    build_vocab,
    df_to_tensors,
    evaluate,
    load_split,
    make_loaders,
    plot_bar,
    plot_confusion,
    save_json,
    train_model,
)

BASE_DIR = Path(__file__).resolve().parent
ADV_MODEL_DIR = BASE_DIR / "model" / "LIAR" / "advance"
ADV_OUTPUT_DIR = BASE_DIR / "output" / "LIAR" / "advance"

for path in [ADV_MODEL_DIR, ADV_OUTPUT_DIR]:
    path.mkdir(parents=True, exist_ok=True)

ADVANCED_GRID = {
    "RNN": [
        {"batch_size": 64, "embed_dim": 128, "hidden_dim": 256, "epochs": 8, "lr": 8e-4, "num_heads": 4, "num_layers": 2, "dropout": 0.35},
        {"batch_size": 96, "embed_dim": 192, "hidden_dim": 256, "epochs": 9, "lr": 6e-4, "num_heads": 4, "num_layers": 2, "dropout": 0.3},
        {"batch_size": 80, "embed_dim": 160, "hidden_dim": 320, "epochs": 9, "lr": 7e-4, "num_heads": 4, "num_layers": 2, "dropout": 0.4},
    ],
    "LSTM": [
        {"batch_size": 64, "embed_dim": 128, "hidden_dim": 256, "epochs": 8, "lr": 8e-4, "num_heads": 4, "num_layers": 2, "dropout": 0.3},
        {"batch_size": 96, "embed_dim": 192, "hidden_dim": 256, "epochs": 9, "lr": 6e-4, "num_heads": 4, "num_layers": 2, "dropout": 0.35},
        {"batch_size": 80, "embed_dim": 160, "hidden_dim": 320, "epochs": 10, "lr": 5e-4, "num_heads": 4, "num_layers": 2, "dropout": 0.4},
    ],
    "Transformer": [
        {"batch_size": 64, "embed_dim": 128, "hidden_dim": 256, "epochs": 8, "lr": 9e-4, "num_heads": 4, "num_layers": 3, "dropout": 0.2},
        {"batch_size": 80, "embed_dim": 192, "hidden_dim": 320, "epochs": 9, "lr": 7e-4, "num_heads": 6, "num_layers": 3, "dropout": 0.25},
        {"batch_size": 96, "embed_dim": 256, "hidden_dim": 320, "epochs": 10, "lr": 6e-4, "num_heads": 8, "num_layers": 4, "dropout": 0.3},
    ],
}


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
            model, val_metrics = train_model(model, train_loader, val_loader, cfg["epochs"], cfg["lr"], f"advance_{name}_{idx + 1}")
            test_metrics = evaluate(model, test_loader)
            if val_metrics["f1"] > best_val_f1:
                best_val_f1 = val_metrics["f1"]
                best_model = model
                best_cfg = dict(cfg)
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

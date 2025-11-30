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
BASIC_MODEL_DIR = BASE_DIR / "model" / "LIAR" / "baisc"
BASIC_OUTPUT_DIR = BASE_DIR / "output" / "LIAR" / "basic"

for path in [BASIC_MODEL_DIR, BASIC_OUTPUT_DIR]:
    path.mkdir(parents=True, exist_ok=True)

BASIC_CFG = {
    "batch_size": 64,
    "embed_dim": 128,
    "hidden_dim": 128,
    "epochs": 6,
    "lr": 1e-3,
    "num_heads": 4,
    "num_layers": 2,
    "dropout": 0.2,
}


def run_basic_stage(train_data, val_data, test_data, vocab_size, pad_id):
    names = ["RNN", "LSTM", "Transformer"]
    metrics = {}
    acc_values = []
    f1_values = []
    best_name = None
    best_cm = None
    best_f1 = -1.0
    for name in names:
        train_loader, val_loader, test_loader = make_loaders(train_data, val_data, test_data, BASIC_CFG["batch_size"])
        model = build_model(name, vocab_size, pad_id, BASIC_CFG)
        model, _ = train_model(model, train_loader, val_loader, BASIC_CFG["epochs"], BASIC_CFG["lr"], f"basic_{name}")
        test_metrics = evaluate(model, test_loader)
        cm_array = test_metrics["cm"]
        metrics[name] = {
            "accuracy": float(test_metrics["accuracy"]),
            "precision": float(test_metrics["precision"]),
            "recall": float(test_metrics["recall"]),
            "f1": float(test_metrics["f1"]),
            "cm": cm_array.tolist(),
            "config": dict(BASIC_CFG),
        }
        if test_metrics["f1"] > best_f1:
            best_f1 = test_metrics["f1"]
            best_name = name
            best_cm = cm_array
        acc_values.append(test_metrics["accuracy"])
        f1_values.append(test_metrics["f1"])
        model = model.to("cpu")
        torch.save(model.state_dict(), BASIC_MODEL_DIR / f"{name.lower()}.pt")
        print(f"basic {name} test_acc {test_metrics['accuracy']:.4f} test_f1 {test_metrics['f1']:.4f}")
    save_json(metrics, BASIC_OUTPUT_DIR / "metrics.json")
    plot_bar(names, acc_values, "Accuracy by Model (Basic)", "Accuracy", BASIC_OUTPUT_DIR / "accuracy.png")
    plot_bar(names, f1_values, "F1 by Model (Basic)", "F1", BASIC_OUTPUT_DIR / "f1.png")
    if best_cm is not None:
        plot_confusion(best_cm, LABELS, f"Basic {best_name} Confusion", BASIC_OUTPUT_DIR / "confusion.png")


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
    run_basic_stage(train_data, val_data, test_data, vocab_size, pad_id)


if __name__ == "__main__":
    main()

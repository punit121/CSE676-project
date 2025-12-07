# pip install torch torchvision transformers pillow pandas numpy scikit-learn pytesseract requests

import os
import pandas as pd
from PIL import Image
import numpy as np
from typing import Optional
import requests
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms, models

import pytesseract
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Text model
TEXT_MODEL_NAME = "xlm-roberta-base"

MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 5
LR = 2e-5
IMAGE_SIZE = 224
HIDDEN_DIM = 512

# Number of images to download (set to 500 as requested)
MAX_IMAGES_TO_DOWNLOAD = 500

DATASET_NAME = "fakeddit"

BASE_DIR = "/content"
DATA_DIR = os.path.join(BASE_DIR, "data")
FAKEDDIT_DIR = os.path.join(DATA_DIR, "Fakeddit")
FAKEDDIT_IMG_DIR = os.path.join(FAKEDDIT_DIR, "images")

FAKEDDIT_TEST_TSV = os.path.join(FAKEDDIT_DIR, "multimodal_test_public.tsv")

# Unified CSV paths
TRAIN_CSV = os.path.join(DATA_DIR, f"{DATASET_NAME}_train.csv")
VAL_CSV   = os.path.join(DATA_DIR, f"{DATASET_NAME}_val.csv")
TEST_CSV  = os.path.join(DATA_DIR, f"{DATASET_NAME}_test.csv")



def download_images_from_urls(df, img_dir, max_images=500):
    """
    Download images from image_url column

    Args:
        df: DataFrame with 'id' and 'image_url' columns
        img_dir: Directory to save images
        max_images: Maximum number of images to download
    """
    print("\n" + "="*60)
    print(f"DOWNLOADING IMAGES (max {max_images})")
    print("="*60)

    os.makedirs(img_dir, exist_ok=True)

    # Filter rows with valid image URLs
    df_with_urls = df[df['image_url'].notna()].head(max_images * 2)  # Get extra in case some fail

    downloaded = 0
    failed = 0

    print(f"\n📥 Attempting to download {min(max_images, len(df_with_urls))} images...")

    for idx, row in tqdm(df_with_urls.iterrows(), total=min(max_images, len(df_with_urls)), desc="Downloading"):
        if downloaded >= max_images:
            break

        post_id = row['id']
        image_url = row['image_url']

        # Skip if already exists
        image_paths = [
            os.path.join(img_dir, f"{post_id}.jpg"),
            os.path.join(img_dir, f"{post_id}.png"),
            os.path.join(img_dir, f"{post_id}.jpeg"),
        ]

        if any(os.path.exists(p) for p in image_paths):
            downloaded += 1
            continue

        # Determine extension from URL
        ext = '.jpg'
        if '.png' in str(image_url).lower():
            ext = '.png'
        elif '.jpeg' in str(image_url).lower():
            ext = '.jpeg'

        image_path = os.path.join(img_dir, f"{post_id}{ext}")

        # Download image
        try:
            response = requests.get(
                image_url,
                timeout=10,
                headers={'User-Agent': 'Mozilla/5.0'},
                stream=True
            )

            if response.status_code == 200:
                with open(image_path, 'wb') as f:
                    f.write(response.content)
                downloaded += 1
            else:
                failed += 1
        except Exception as e:
            failed += 1
            continue

    print(f"\n✅ Downloaded: {downloaded} images")
    print(f"❌ Failed: {failed} images")

    return downloaded




def preprocess_fakeddit_with_download(
    test_tsv: str,
    img_dir: str,
    out_train: str,
    out_val: str,
    out_test: str,
    use_2way: bool = False,
    max_samples: int = 500,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42
) -> None:
    """
    Download images and preprocess data
    """

    print("="*60)
    print("PREPROCESSING FAKEDDIT WITH IMAGE DOWNLOAD")
    print("="*60)

    # Read TSV
    print(f"\n📄 Reading: {test_tsv}")
    df = pd.read_csv(test_tsv, sep='\t', on_bad_lines='skip')

    print(f"   Total rows: {len(df)}")

    # Filter for posts with images
    df = df[df['hasImage'] == True].reset_index(drop=True)
    print(f"   Rows with images: {len(df)}")

    # Take first max_samples
    df = df.head(max_samples * 3)  # Get extra in case some downloads fail

    # Download images
    downloaded_count = download_images_from_urls(df, img_dir, max_images=max_samples)

    if downloaded_count == 0:
        raise ValueError("No images were downloaded successfully!")

    # Select label column
    label_col = '2_way_label' if use_2way else '6_way_label'

    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found")

    # Build text
    if 'clean_title' in df.columns:
        df['text'] = df['clean_title'].fillna('').astype(str)
    else:
        df['text'] = df['title'].fillna('').astype(str)

    # Build image path
    def build_image_path(row_id):
        for ext in ['.jpg', '.jpeg', '.png']:
            rel_path = os.path.join('Fakeddit', 'images', f"{row_id}{ext}")
            full_path = os.path.join(DATA_DIR, rel_path)
            if os.path.exists(full_path):
                return rel_path
        return None

    df['image_path'] = df['id'].apply(build_image_path)

    # Map labels
    if use_2way:
        label_map = {0: 'real', 1: 'fake'}
    else:
        label_map = {
            0: 'true',
            1: 'satire',
            2: 'fake_news',
            3: 'false_connection',
            4: 'misleading',
            5: 'manipulated'
        }

    df['label'] = df[label_col].map(label_map).fillna('unknown')

    # Keep only rows with downloaded images
    df_unified = df[['id', 'text', 'image_path', 'label']].copy()
    df_unified = df_unified[df_unified['image_path'].notna()]
    df_unified = df_unified[df_unified['label'] != 'unknown']

    print(f"\n✅ Final dataset size: {len(df_unified)} samples")

    if len(df_unified) == 0:
        raise ValueError("No valid samples after filtering!")

    # Shuffle
    df_unified = df_unified.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # Split
    n = len(df_unified)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    df_train = df_unified.iloc[:n_train]
    df_val = df_unified.iloc[n_train:n_train + n_val]
    df_test = df_unified.iloc[n_train + n_val:]

    # Save
    df_train.to_csv(out_train, index=False)
    df_val.to_csv(out_val, index=False)
    df_test.to_csv(out_test, index=False)

    print(f"\n📊 Split Summary:")
    print(f"   Train: {len(df_train)} samples")
    print(f"   Val:   {len(df_val)} samples")
    print(f"   Test:  {len(df_test)} samples")

    print(f"\n📊 Label distribution (all data):")
    print(df_unified['label'].value_counts().to_dict())



class MultimodalFakeNewsDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        data_root: str,
        tokenizer: AutoTokenizer,
        label_encoder: Optional[LabelEncoder] = None,
        max_length: int = 128,
        image_size: int = 224,
    ):
        self.df = pd.read_csv(csv_path)
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Label encoding
        if label_encoder is None:
            self.label_encoder = LabelEncoder()
            self.df["label_id"] = self.label_encoder.fit_transform(self.df["label"])
        else:
            self.label_encoder = label_encoder
            self.df["label_id"] = self.label_encoder.transform(self.df["label"])

        # Image transforms
        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __len__(self):
        return len(self.df)

    def _load_image(self, rel_path: str) -> Image.Image:
        img_path = os.path.join(self.data_root, rel_path)
        try:
            img = Image.open(img_path).convert("RGB")
            return img
        except Exception as e:
            return Image.new('RGB', (224, 224), color='white')

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        base_text = str(row["text"])
        label_id = int(row["label_id"])

        # Load image
        image = self._load_image(row["image_path"])

        # Tokenize text
        encoded = self.tokenizer(
            base_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        # Image tensor
        image_tensor = self.image_transform(image)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image": image_tensor,
            "label": torch.tensor(label_id, dtype=torch.long),
        }



class TextEncoder(nn.Module):
    def __init__(self, model_name: str = TEXT_MODEL_NAME, output_dim: int = 768):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.output_dim = output_dim

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]


class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.output_dim = 512

    def forward(self, images):
        x = self.backbone(images)
        return x.view(x.size(0), -1)


class MultimodalFakeNewsClassifier(nn.Module):
    def __init__(self, text_model_name: str, num_labels: int, hidden_dim: int):
        super().__init__()
        self.text_encoder = TextEncoder(text_model_name)
        self.image_encoder = ImageEncoder()

        text_dim = self.text_encoder.output_dim
        img_dim = self.image_encoder.output_dim

        self.classifier = nn.Sequential(
            nn.Linear(text_dim + img_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(self, input_ids, attention_mask, images):
        v_text = self.text_encoder(input_ids, attention_mask)
        v_img = self.image_encoder(images)
        fused = torch.cat([v_text, v_img], dim=1)
        return self.classifier(fused)



def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids=input_ids, attention_mask=attention_mask, images=images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def eval_model(model, dataloader, criterion, device, return_predictions=False):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in tqdm(dataloader, desc="Evaluating"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask, images=images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(all_labels)
    accuracy = accuracy_score(all_labels, all_preds)

    if return_predictions:
        return avg_loss, accuracy, all_preds, all_labels

    return avg_loss, accuracy


def print_evaluation_metrics(y_true, y_pred, label_encoder):
    """Print detailed evaluation metrics"""
    print("\n" + "="*60)
    print("EVALUATION METRICS")
    print("="*60)

    # Overall metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')

    print(f"\n📊 Overall Metrics:")
    print(f"   Accuracy:  {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1-Score:  {f1:.4f}")

    # Per-class metrics
    print(f"\n📋 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))




def main():
    print("="*60)
    print("FAKEDDIT MULTIMODAL FAKE NEWS DETECTION")
    print(f"Device: {DEVICE}")
    print("="*60)

    # Check if data needs preprocessing
    if not (os.path.exists(TRAIN_CSV) and os.path.exists(VAL_CSV) and os.path.exists(TEST_CSV)):
        print("\n📋 Preprocessing data and downloading images...")
        preprocess_fakeddit_with_download(
            test_tsv=FAKEDDIT_TEST_TSV,
            img_dir=FAKEDDIT_IMG_DIR,
            out_train=TRAIN_CSV,
            out_val=VAL_CSV,
            out_test=TEST_CSV,
            use_2way=False,  # 6-way classification
            max_samples=MAX_IMAGES_TO_DOWNLOAD,
        )
    else:
        print("\n✓ Using existing preprocessed data")

    # Load tokenizer
    print("\n📦 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)

    # Create datasets
    train_df = pd.read_csv(TRAIN_CSV)
    label_encoder = LabelEncoder()
    label_encoder.fit(train_df["label"])

    print(f"\n📊 Dataset Info:")
    print(f"   Classes: {label_encoder.classes_}")
    print(f"   Number of classes: {len(label_encoder.classes_)}")

    train_dataset = MultimodalFakeNewsDataset(
        csv_path=TRAIN_CSV,
        data_root=DATA_DIR,
        tokenizer=tokenizer,
        label_encoder=label_encoder,
        max_length=MAX_LENGTH,
        image_size=IMAGE_SIZE,
    )
    val_dataset = MultimodalFakeNewsDataset(
        csv_path=VAL_CSV,
        data_root=DATA_DIR,
        tokenizer=tokenizer,
        label_encoder=label_encoder,
        max_length=MAX_LENGTH,
        image_size=IMAGE_SIZE,
    )
    test_dataset = MultimodalFakeNewsDataset(
        csv_path=TEST_CSV,
        data_root=DATA_DIR,
        tokenizer=tokenizer,
        label_encoder=label_encoder,
        max_length=MAX_LENGTH,
        image_size=IMAGE_SIZE,
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"\n   Train: {len(train_dataset)} samples")
    print(f"   Val:   {len(val_dataset)} samples")
    print(f"   Test:  {len(test_dataset)} samples")

    # Initialize model
    print("\n🔧 Initializing model...")
    num_labels = len(label_encoder.classes_)
    model = MultimodalFakeNewsClassifier(
        text_model_name=TEXT_MODEL_NAME,
        num_labels=num_labels,
        hidden_dim=HIDDEN_DIM,
    ).to(DEVICE)

    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    # Training
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)

    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{EPOCHS}")
        print(f"{'='*60}")

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_acc = eval_model(model, val_loader, criterion, DEVICE)

        print(f"\n  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_dir = os.path.join(BASE_DIR, "checkpoints")
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(save_dir, f"{DATASET_NAME}_best_model.pt"))
            print(f"  ✓ Best model saved! (val_acc: {val_acc:.4f})")

    # Final evaluation on test set
    print("\n" + "="*60)
    print("FINAL EVALUATION ON TEST SET")
    print("="*60)

    test_loss, test_acc, test_preds, test_labels = eval_model(
        model, test_loader, criterion, DEVICE, return_predictions=True
    )

    print(f"\n📊 Test Results:")
    print(f"   Loss: {test_loss:.4f}")
    print(f"   Accuracy: {test_acc:.4f}")

    # Detailed metrics
    print_evaluation_metrics(test_labels, test_preds, label_encoder)

    # Save final model
    save_dir = os.path.join(BASE_DIR, "checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, f"{DATASET_NAME}_final_model.pt"))

    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print(f"   Best Val Accuracy: {best_val_acc:.4f}")
    print(f"   Test Accuracy: {test_acc:.4f}")
    print(f"   Models saved to: {save_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
# pip install torch torchvision transformers pillow pandas numpy scikit-learn pytesseract requests imbalanced-learn albumentations

import os
import pandas as pd
from PIL import Image
import numpy as np
from typing import Optional
import requests
from tqdm import tqdm
import random

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms, models

import pytesseract
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from imblearn.over_sampling import SMOTE, RandomOverSampler
from collections import Counter


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Text model
TEXT_MODEL_NAME = "xlm-roberta-base"

MAX_LENGTH = 128
BATCH_SIZE = 32  
EPOCHS = 25 
LR = 5e-6  
IMAGE_SIZE = 224
HIDDEN_DIM = 768  

MAX_IMAGES_TO_DOWNLOAD = 10000

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

# Balanced CSV paths (after oversampling)
TRAIN_BALANCED_CSV = os.path.join(DATA_DIR, f"{DATASET_NAME}_train_balanced.csv")



def download_images_from_urls(df, img_dir, max_images=10000):

    os.makedirs(img_dir, exist_ok=True)

    # Filter rows with valid image URLs
    df_with_urls = df[df['image_url'].notna()].head(max_images * 2)

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

    print(f"\n Downloaded: {downloaded} images")

    return downloaded




def aggressive_balance_dataset(df, label_col='label', target_samples_per_class=3000, random_state=42):

    # Print original distribution
    print("\n📊 Original class distribution:")
    original_dist = df[label_col].value_counts().sort_index()
    for label, count in original_dist.items():
        print(f"   {label}: {count}")
    
    # Find max class count
    max_count = original_dist.max()
    target_count = max(max_count, target_samples_per_class)

    print(f"Target samples per class: {target_count}")

    balanced_dfs = []
    
    for label in df[label_col].unique():
        class_df = df[df[label_col] == label]
        current_count = len(class_df)
        
        if current_count < target_count:
            # Oversample to target
            n_samples_needed = target_count - current_count
            oversample_indices = np.random.choice(
                class_df.index, 
                size=n_samples_needed, 
                replace=True
            )
            oversampled_df = df.loc[oversample_indices]
            balanced_dfs.append(class_df)
            balanced_dfs.append(oversampled_df)
        else:
            balanced_dfs.append(class_df)
    
    df_balanced = pd.concat(balanced_dfs, ignore_index=True)
    
    # Shuffle
    df_balanced = df_balanced.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    # Print new distribution
    balanced_dist = df_balanced[label_col].value_counts().sort_index()
    for label, count in balanced_dist.items():
        print(f"   {label}: {count}")
    
    print(f"\n✅ Dataset size: {len(df)} → {len(df_balanced)} samples")
    
    return df_balanced


def preprocess_fakeddit_with_download(
    test_tsv: str,
    img_dir: str,
    out_train: str,
    out_val: str,
    out_test: str,
    out_train_balanced: str,
    use_2way: bool = False,
    max_samples: int = 10000,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42
) -> None:
    
    # Read TSV
    df = pd.read_csv(test_tsv, sep='\t', on_bad_lines='skip')

    print(f"Total rows: {len(df)}")

    # Filter for posts with images
    df = df[df['hasImage'] == True].reset_index(drop=True)
    print(f"Rows with images: {len(df)}")

    # Take first max_samples * 2 to ensure we get enough after download failures
    df = df.head(max_samples * 2)

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

    print(f"\n Final dataset size: {len(df_unified)} samples")

    if len(df_unified) == 0:
        raise ValueError("No valid samples after filtering!")

    # Shuffle
    df_unified = df_unified.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # Split
    n = len(df_unified)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    df_train = df_unified.iloc[:n_train].reset_index(drop=True)
    df_val = df_unified.iloc[n_train:n_train + n_val].reset_index(drop=True)
    df_test = df_unified.iloc[n_train + n_val:].reset_index(drop=True)

    # Save original splits
    df_train.to_csv(out_train, index=False)
    df_val.to_csv(out_val, index=False)
    df_test.to_csv(out_test, index=False)

    print(f"\n📊 Original Split Summary:")
    print(f"   Train: {len(df_train)} samples")
    print(f"   Val:   {len(df_val)} samples")
    print(f"   Test:  {len(df_test)} samples")

    print(f"\n📊 Label distribution (all data):")
    print(df_unified['label'].value_counts().to_dict())

    # Balancing with target 3000 samples per class
    df_train_balanced = aggressive_balance_dataset(
        df_train, 
        label_col='label', 
        target_samples_per_class=3000,
        random_state=seed
    )
    df_train_balanced.to_csv(out_train_balanced, index=False)

    print(f"Balanced training set to: {out_train_balanced}")



class MultimodalFakeNewsDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        data_root: str,
        tokenizer: AutoTokenizer,
        label_encoder: Optional[LabelEncoder] = None,
        max_length: int = 128,
        image_size: int = 224,
        is_train: bool = False,
        use_mixup: bool = False,
        mixup_alpha: float = 0.2,
    ):
        self.df = pd.read_csv(csv_path)
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_train = is_train
        self.use_mixup = use_mixup and is_train
        self.mixup_alpha = mixup_alpha

        # Label encoding
        if label_encoder is None:
            self.label_encoder = LabelEncoder()
            self.df["label_id"] = self.label_encoder.fit_transform(self.df["label"])
        else:
            self.label_encoder = label_encoder
            self.df["label_id"] = self.label_encoder.transform(self.df["label"])

        # Very aggressive augmentation for training
        if is_train:
            self.image_transform = transforms.Compose([
                transforms.Resize((image_size + 64, image_size + 64)),
                transforms.RandomCrop((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.2),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
                transforms.RandomRotation(20),
                transforms.RandomGrayscale(p=0.15),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
                transforms.ToTensor(),
                transforms.RandomErasing(p=0.2),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
        else:
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
        # Use ResNet50 for better feature extraction
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.output_dim = 2048

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

        # Very deep classifier with residual connections
        self.fusion = nn.Linear(text_dim + img_dim, hidden_dim * 2)
        
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.6),  # Higher dropout
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 4, num_labels),
        )

    def forward(self, input_ids, attention_mask, images):
        v_text = self.text_encoder(input_ids, attention_mask)
        v_img = self.image_encoder(images)
        fused = torch.cat([v_text, v_img], dim=1)
        fused = self.fusion(fused)
        return self.classifier(fused)


# Focal Loss for hard examples
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()



def train_one_epoch(model, dataloader, optimizer, criterion, device, scheduler=None):
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
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        
        optimizer.step()
        
        if scheduler:
            scheduler.step()

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


    # Overall metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)

    print(f"Overall Metrics:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")

    # Per-class metrics
    print(f"Classification Report:")
    print(classification_report(y_true, y_pred, target_names=label_encoder.classes_, zero_division=0))
    





def main():
    print(f"Device: {DEVICE}")

    needs_reprocessing = True
    
    if os.path.exists(TRAIN_BALANCED_CSV):
        try:
            existing_df = pd.read_csv(TRAIN_BALANCED_CSV)
            if len(existing_df) < 15000:  
                for file in [TRAIN_CSV, VAL_CSV, TEST_CSV, TRAIN_BALANCED_CSV]:
                    if os.path.exists(file):
                        os.remove(file)
                        print(f"   Deleted: {file}")
            else:
                needs_reprocessing = False
                print("\n✓ Using existing 10K preprocessed data")
        except:
            needs_reprocessing = True
    
    if needs_reprocessing:
        print("Preprocessing data and downloading 10K images...")
        preprocess_fakeddit_with_download(
            test_tsv=FAKEDDIT_TEST_TSV,
            img_dir=FAKEDDIT_IMG_DIR,
            out_train=TRAIN_CSV,
            out_val=VAL_CSV,
            out_test=TEST_CSV,
            out_train_balanced=TRAIN_BALANCED_CSV,
            use_2way=False,
            max_samples=MAX_IMAGES_TO_DOWNLOAD,
        )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)

    # Create datasets
    train_df = pd.read_csv(TRAIN_BALANCED_CSV)
    label_encoder = LabelEncoder()
    label_encoder.fit(train_df["label"])

    print(f"Classes: {label_encoder.classes_}")
    print(f"Number of classes: {len(label_encoder.classes_)}")

    train_dataset = MultimodalFakeNewsDataset(
        csv_path=TRAIN_BALANCED_CSV,
        data_root=DATA_DIR,
        tokenizer=tokenizer,
        label_encoder=label_encoder,
        max_length=MAX_LENGTH,
        image_size=IMAGE_SIZE,
        is_train=True,
        use_mixup=False,
    )
    val_dataset = MultimodalFakeNewsDataset(
        csv_path=VAL_CSV,
        data_root=DATA_DIR,
        tokenizer=tokenizer,
        label_encoder=label_encoder,
        max_length=MAX_LENGTH,
        image_size=IMAGE_SIZE,
        is_train=False,
    )
    test_dataset = MultimodalFakeNewsDataset(
        csv_path=TEST_CSV,
        data_root=DATA_DIR,
        tokenizer=tokenizer,
        label_encoder=label_encoder,
        max_length=MAX_LENGTH,
        image_size=IMAGE_SIZE,
        is_train=False,
    )

    # Regular dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    print(f"Train: {len(train_dataset)} samples (balanced)")
    print(f"Val:   {len(val_dataset)} samples")
    print(f"Test:  {len(test_dataset)} samples")

    print(f"Training set class distribution:")
    print(train_df['label'].value_counts().sort_index().to_dict())

    # Initialize model
    num_labels = len(label_encoder.classes_)
    model = MultimodalFakeNewsClassifier(
        text_model_name=TEXT_MODEL_NAME,
        num_labels=num_labels,
        hidden_dim=HIDDEN_DIM,
    ).to(DEVICE)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
    
    # Use Focal Loss
    criterion = FocalLoss(alpha=1.0, gamma=2.0)
    
    # Learning rate scheduler with warmup
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = len(train_loader) * 2
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training loop with early stopping
    best_val_acc = 0.0
    patience = 7  
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        
        print(f"Epoch {epoch}/{EPOCHS}")
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE, scheduler)
        val_loss, val_acc = eval_model(model, val_loader, criterion, DEVICE)

        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            save_dir = os.path.join(BASE_DIR, "checkpoints")
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(save_dir, f"{DATASET_NAME}_best_model.pt"))
            print(f"Best model saved! (val_acc: {val_acc:.4f})")
        else:
            patience_counter += 1
            print(f"Patience: {patience_counter}/{patience}")

            if patience_counter >= patience:
                break

    # Load best model for final evaluation
    print("Loading best model for final evaluation...")
    model.load_state_dict(torch.load(os.path.join(BASE_DIR, "checkpoints", f"{DATASET_NAME}_best_model.pt")))

    # Final evaluation on test set
    test_loss, test_acc, test_preds, test_labels = eval_model(
        model, test_loader, criterion, DEVICE, return_predictions=True
    )

    print(f"Test Results:")
    print(f"Loss: {test_loss:.4f}")
    print(f"Accuracy: {test_acc:.4f}")

    # Detailed metrics
    print_evaluation_metrics(test_labels, test_preds, label_encoder)


    print(f"Best Val Accuracy: {best_val_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
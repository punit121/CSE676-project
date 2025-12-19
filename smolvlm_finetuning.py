import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
from PIL import Image
import numpy as np
import requests
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, AutoModelForVision2Seq
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import warnings
warnings.filterwarnings('ignore')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "HuggingFaceTB/SmolVLM-Instruct"
BATCH_SIZE = 1
EPOCHS = 3
LR = 2e-5
IMAGE_SIZE = 384
GRADIENT_ACCUMULATION_STEPS = 4

TOTAL_SAMPLES = 10000
SAMPLES_PER_CLASS = 1667
MAX_IMAGES_TO_DOWNLOAD = 10000

BASE_DIR = "/content"
DATA_DIR = os.path.join(BASE_DIR, "data")
FAKEDDIT_DIR = os.path.join(DATA_DIR, "Fakeddit")
FAKEDDIT_IMG_DIR = os.path.join(FAKEDDIT_DIR, "images")
FAKEDDIT_TEST_TSV = os.path.join(FAKEDDIT_DIR, "multimodal_test_public.tsv")

TRAIN_CSV = os.path.join(DATA_DIR, "fakeddit_train_balanced.csv")
VAL_CSV = os.path.join(DATA_DIR, "fakeddit_val_balanced.csv")
TEST_CSV = os.path.join(DATA_DIR, "fakeddit_test_balanced.csv")

def download_images_from_urls(df, img_dir, max_images=10000):
    
    
    os.makedirs(img_dir, exist_ok=True)
    
    df_with_urls = df[df['image_url'].notna()].head(max_images * 2)
    downloaded = 0
    failed = 0
    
    
    
    for idx, row in tqdm(df_with_urls.iterrows(), total=min(max_images, len(df_with_urls)), desc="Downloading"):
        if downloaded >= max_images:
            break
            
        post_id = row['id']
        image_url = row['image_url']
        
        image_paths = [
            os.path.join(img_dir, f"{post_id}.jpg"),
            os.path.join(img_dir, f"{post_id}.png"),
            os.path.join(img_dir, f"{post_id}.jpeg"),
        ]
        
        if any(os.path.exists(p) for p in image_paths):
            downloaded += 1
            continue
        
        ext = '.jpg'
        if '.png' in str(image_url).lower():
            ext = '.png'
        elif '.jpeg' in str(image_url).lower():
            ext = '.jpeg'
            
        image_path = os.path.join(img_dir, f"{post_id}{ext}")
        
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
        except:
            failed += 1
            continue
    
    
    return downloaded

def oversample_dataframe(df, target_per_class=1667, seed=42):
    
    
    class_counts = df['label'].value_counts()
    
    
    balanced_dfs = []
    for label in df['label'].unique():
        class_df = df[df['label'] == label]
        current_count = len(class_df)
        
        if current_count < target_per_class:
            oversampled = class_df.sample(n=target_per_class, replace=True, random_state=seed)
        else:
            oversampled = class_df.sample(n=target_per_class, replace=False, random_state=seed)
        
        balanced_dfs.append(oversampled)
    
    balanced_df = pd.concat(balanced_dfs, ignore_index=True)
    balanced_df = balanced_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    
    
    return balanced_df

def preprocess_fakeddit_with_oversampling(
    test_tsv: str,
    img_dir: str,
    out_train: str,
    out_val: str,
    out_test: str,
    samples_per_class: int = 1667,
    use_2way: bool = False,
    max_images: int = 10000,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42
):
    
    
    
    df = pd.read_csv(test_tsv, sep='\t', on_bad_lines='skip')
    
    
    df = df[df['hasImage'] == True].reset_index(drop=True)
    
    
    df = df.head(max_images)
    
    label_col = '2_way_label' if use_2way else '6_way_label'
    
    if 'clean_title' in df.columns:
        df['text'] = df['clean_title'].fillna('').astype(str)
    else:
        df['text'] = df['title'].fillna('').astype(str)
    
    def build_image_path(row_id):
        for ext in ['.jpg', '.jpeg', '.png']:
            rel_path = os.path.join('Fakeddit', 'images', f"{row_id}{ext}")
            full_path = os.path.join(DATA_DIR, rel_path)
            if os.path.exists(full_path):
                return rel_path
        return None
    
    df['image_path'] = df['id'].apply(build_image_path)
    
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
    
    df_unified = df[['id', 'text', 'image_path', 'label']].copy()
    df_unified = df_unified[df_unified['image_path'].notna()]
    df_unified = df_unified[df_unified['label'] != 'unknown']
    
    
    
    if len(df_unified) == 0:
        raise ValueError("No valid samples!")
    
    df_balanced = oversample_dataframe(df_unified, target_per_class=samples_per_class, seed=seed)
    
    df_balanced = df_balanced.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    
    n = len(df_balanced)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    df_train = df_balanced.iloc[:n_train]
    df_val = df_balanced.iloc[n_train:n_train + n_val]
    df_test = df_balanced.iloc[n_train + n_val:]
    
    df_train.to_csv(out_train, index=False)
    df_val.to_csv(out_val, index=False)
    df_test.to_csv(out_test, index=False)
    
    

class SmolVLMClassifier(nn.Module):
    def __init__(self, model_name: str, num_labels: int):
        super().__init__()
        self.vision_model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        
        for name, param in self.vision_model.named_parameters():
            if 'layer.23' not in name and 'layer.22' not in name and 'layer.21' not in name:
                param.requires_grad = False
        
        hidden_size = self.vision_model.config.text_config.hidden_size
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_labels)
        ).half()
        
        self.num_labels = num_labels
    
    def forward(self, input_ids, pixel_values, attention_mask=None, labels=None):
        outputs = self.vision_model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        hidden_states = outputs.hidden_states[-1]
        pooled = hidden_states.mean(dim=1)
        
        pooled = pooled.half()
        
        logits = self.classifier(pooled)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.float(), labels)
        
        return type('Output', (), {'loss': loss, 'logits': logits})()

class SmolVLMFakeNewsDataset(Dataset):
    def __init__(self, csv_path: str, data_root: str, processor, label_encoder=None):
        self.df = pd.read_csv(csv_path)
        self.data_root = data_root
        self.processor = processor
        
        if label_encoder is None:
            self.label_encoder = LabelEncoder()
            self.df["label_id"] = self.label_encoder.fit_transform(self.df["label"])
        else:
            self.label_encoder = label_encoder
            self.df["label_id"] = self.label_encoder.transform(self.df["label"])
    
    def __len__(self):
        return len(self.df)
    
    def _load_image(self, rel_path: str):
        img_path = os.path.join(self.data_root, rel_path)
        try:
            return Image.open(img_path).convert("RGB")
        except:
            return Image.new('RGB', (384, 384), color='white')
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row["text"])
        label_id = int(row["label_id"])
        
        image = self._load_image(row["image_path"])
        
        words = text.split()[:15]
        text = ' '.join(words)
        
        prompt = f"Classify: {text}"
        
        return {
            'text': prompt,
            'image': image,
            'label': label_id
        }

def custom_collate_fn(batch, processor):
    texts = [item['text'] for item in batch]
    images = [item['image'] for item in batch]
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
    
    texts_with_image = [f"<image>{text}" for text in texts]
    
    try:
        inputs = processor(
            text=texts_with_image,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=False
        )
        
        max_len = 512
        if inputs['input_ids'].shape[1] > max_len:
            inputs['input_ids'] = inputs['input_ids'][:, :max_len]
            if 'attention_mask' in inputs:
                inputs['attention_mask'] = inputs['attention_mask'][:, :max_len]
    
    except Exception as e:
        texts_short = [f"<image>News classification" for _ in texts]
        inputs = processor(
            text=texts_short,
            images=images,
            return_tensors="pt",
            padding=True,
        )
    
    inputs['labels'] = labels
    return inputs

def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        pixel_values = batch['pixel_values'].to(device)
        attention_mask = batch.get('attention_mask')
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss / GRADIENT_ACCUMULATION_STEPS
        loss.backward()
        
        if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()
            torch.cuda.empty_cache()
        
        total_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS * labels.size(0)
        
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        progress_bar.set_postfix({'loss': loss.item() * GRADIENT_ACCUMULATION_STEPS, 'acc': correct/total})
    
    if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS != 0:
        optimizer.step()
        optimizer.zero_grad()
    
    return total_loss / total, correct / total

@torch.no_grad()
def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    for batch in tqdm(dataloader, desc="Evaluating"):
        input_ids = batch['input_ids'].to(device)
        pixel_values = batch['pixel_values'].to(device)
        attention_mask = batch.get('attention_mask')
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask
        )
        
        preds = torch.argmax(outputs.logits, dim=-1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels)

def print_detailed_metrics(y_true, y_pred, label_encoder):
    
    
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted'
    )
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Overall Precision: {precision:.4f}")
    print(f"Overall Recall: {recall:.4f}")
    print(f"Overall F1-score: {f1:.4f}")
    
    
    for i, class_name in enumerate(label_encoder.classes_):
        class_mask = y_true == i
        if class_mask.sum() > 0:
            class_acc = (y_pred[class_mask] == y_true[class_mask]).sum() / class_mask.sum()
            class_total = class_mask.sum()
            class_correct = (y_pred[class_mask] == y_true[class_mask]).sum()
            print(f"Class: {class_name} - Acc: {class_acc:.4f} ({class_correct}/{class_total})")   

def main():
    
    
    if not all(os.path.exists(p) for p in [TRAIN_CSV, VAL_CSV, TEST_CSV]):
        
        preprocess_fakeddit_with_oversampling(
            test_tsv=FAKEDDIT_TEST_TSV,
            img_dir=FAKEDDIT_IMG_DIR,
            out_train=TRAIN_CSV,
            out_val=VAL_CSV,
            out_test=TEST_CSV,
            samples_per_class=SAMPLES_PER_CLASS,
            use_2way=False,
            max_images=MAX_IMAGES_TO_DOWNLOAD,
        )
    else:
        pass
    
    
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    
    
    train_df = pd.read_csv(TRAIN_CSV)
    label_encoder = LabelEncoder()
    label_encoder.fit(train_df["label"])
    num_labels = len(label_encoder.classes_)
    
    
    
    train_dataset = SmolVLMFakeNewsDataset(TRAIN_CSV, DATA_DIR, processor, label_encoder)
    val_dataset = SmolVLMFakeNewsDataset(VAL_CSV, DATA_DIR, processor, label_encoder)
    test_dataset = SmolVLMFakeNewsDataset(TEST_CSV, DATA_DIR, processor, label_encoder)
    
    
    
    from functools import partial
    collate_fn = partial(custom_collate_fn, processor=processor)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=0
    )
    
    
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    model = SmolVLMClassifier(MODEL_NAME, num_labels).to(DEVICE)
    
    
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    
    
    
    best_val_acc = 0.0
    output_dir = os.path.join(BASE_DIR, "smolvlm_checkpoints")
    os.makedirs(output_dir, exist_ok=True)
    
    for epoch in range(1, EPOCHS + 1):
        
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, DEVICE)
        val_preds, val_labels = evaluate_model(model, val_loader, DEVICE)
        val_acc = accuracy_score(val_labels, val_preds)
        
    
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))
            
    
    
    model.load_state_dict(torch.load(os.path.join(output_dir, "best_model.pt")))
    
    test_preds, test_labels = evaluate_model(model, test_loader, DEVICE)
    
    torch.save(model.state_dict(), os.path.join(output_dir, "final_model.pt"))
    

if __name__ == "__main__":
    main()
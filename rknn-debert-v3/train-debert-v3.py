import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from tqdm.auto import tqdm

# Check the environment
torch.cuda.is_available()
# Enable PyTorch 2.x编 (Optional)
torch.set_float32_matmul_precision('high')  # Improve the matrix compute efficience.
# Set the device. 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create custom dataset for sentiment task. 
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(
            texts, 
            padding='max_length', 
            truncation=True, 
            max_length=max_length,
            return_tensors='pt'
        )
        self.labels = torch.tensor(labels)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': self.labels[idx]
        }
    
    def __len__(self):
        return len(self.labels)
    

# Read CSV with header: text,label
df = pd.read_csv("/hw_data/dean-ws/emotion-bert/raw_dataset.csv")
# split train dataset/test dataset/valid dataset
train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['label'])
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'])

# Initialization tokenizer
model_name = "/hw_data/dean-ws/emotion-bert/microsoft/deberta-v3-base"  # base/small
tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)

# Create PyTorch Dataset
train_dataset = SentimentDataset(train_df['text'].tolist(), train_df['label'].tolist(), tokenizer)
val_dataset = SentimentDataset(val_df['text'].tolist(), val_df['label'].tolist(), tokenizer)
test_dataset = SentimentDataset(test_df['text'].tolist(), test_df['label'].tolist(), tokenizer)

# Load pretrained model
model = DebertaV2ForSequenceClassification.from_pretrained(
    model_name,
    num_labels=3
).to(device)

# Optimizer setting
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
total_steps = len(train_dataset) // 16 * 5  # batch_size=16, epochs=5
scheduler = get_linear_schedule_with_warmup(
    optimizer, 
    num_warmup_steps=int(total_steps * 0.1),
    num_training_steps=total_steps
)

# Mixed precision train.
scaler = torch.cuda.amp.GradScaler()

# Loss function (support class)
class_weights = torch.tensor([1.0, 1.0, 1.0]).to(device)  # 根据实际数据调整
loss_fn = CrossEntropyLoss(weight=class_weights)

# train epoch 
def train_epoch(model, dataloader):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch in progress_bar:
        inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        # 混合精度上下文
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(**inputs)
            loss = loss_fn(outputs.logits, labels)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
        
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())
    
    return total_loss / len(dataloader)


# valid/test loop
# --------------------------------------------------
def evaluate(model, dataloader):
    model.eval()
    preds, true_labels = [], []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].numpy()
            
            outputs = model(**inputs)
            batch_preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            
            preds.extend(batch_preds)
            true_labels.extend(labels)
    
    return {
        'accuracy': accuracy_score(true_labels, preds),
        'macro_f1': f1_score(true_labels, preds, average='macro')
    }

# --------------------------------------------------
# Main train process.
# --------------------------------------------------
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

best_f1 = 0
for epoch in range(5):
    print(f"\nEpoch {epoch+1}/5")
    train_loss = train_epoch(model, train_loader)
    val_metrics = evaluate(model, val_loader)
    
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
    print(f"Val Macro F1: {val_metrics['macro_f1']:.4f}")
    
    # Save the best model.
    if val_metrics['macro_f1'] > best_f1:
        best_f1 = val_metrics['macro_f1']
        torch.save(model.state_dict(), "best_model.pt")

# --------------------------------------------------
        
# Test and deploy
# --------------------------------------------------
model.load_state_dict(torch.load("best_model.pt"))
test_loader = DataLoader(test_dataset, batch_size=32)
test_metrics = evaluate(model, test_loader)

print("\nFinal Test Metrics:")
print(f"Accuracy: {test_metrics['accuracy']:.4f}")
print(f"Macro F1: {test_metrics['macro_f1']:.4f}")

# Example for inference.
def predict(text):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return probs.cpu().numpy()

for txt in [
    "交警队出盖章文件了！hw智驾连环车祸+行车记录仪内存卡被偷最新动态！",
    "多家企业捐款驰援灾区#西藏日喀则地震",
    "坦克500 Hi4-Z、Hi4-T到底啥区别？应该怎么选？一条视频就明白了！#新车抢先看"]:
    M = {0:'负向', 1: '中性',  2:'正向' }
    probs = predict(txt)
    idx = np.argmax(probs)
    print(probs, M[idx])


# save the tokenizer
save_path = "./tokenizer-v2"  # 替换为目标路径
tokenizer.save_pretrained(save_path)
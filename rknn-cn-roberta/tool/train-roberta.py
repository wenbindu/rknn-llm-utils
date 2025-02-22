import torch
import pyarrow
import datasets
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification,Trainer,TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

print(torch.cuda.is_available())
print(f"PyTorch Version: {torch.__version__}")
print(f"pyarrow: {pyarrow.__version__}")
print(f"datasets: {datasets.__version__}")


df = pd.read_csv("./samples.csv")
model_name = "hfl/chinese-roberta-wwm-ext" 
# split dataset
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
print(f"训练集: {len(train_df)} 条, 验证集: {len(val_df)} 条, 测试集: {len(test_df)} 条")

# 4. 数据编码
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_df)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"  # 返回PyTorch张量
    )

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# 5. 定义评估指标
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    return {"accuracy": acc, "f1": f1}

# 6. 配置训练参数
training_args = TrainingArguments(
    output_dir="./results_roberta",
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.001,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    learning_rate=1e-5,
    fp16=True if torch.cuda.is_available() else False,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

# 9. 测试
test_results = trainer.evaluate(test_dataset)
print("测试集结果:", test_results)

# 10. 保存模型
model.save_pretrained("./cn_roberta_model")
tokenizer.save_pretrained("./cn_roberta_model_token")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict(text):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return probs.cpu().numpy()


for txt in ["hi"]:
    M = {0:'负向', 1: '中性',  2:'正向' }
    probs = predict(txt)
    idx = np.argmax(probs)
    print(probs, M[idx])
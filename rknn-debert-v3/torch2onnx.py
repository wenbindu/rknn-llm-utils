
import torch
from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification


model_name = "/hw_data/dean-ws/emotion-bert/microsoft/deberta-v3-base"  # 根据显存可选 base/small

# 加载训练好的模型
model = DebertaV2ForSequenceClassification.from_pretrained(model_name, num_labels=3)
model.load_state_dict(torch.load("best_model.pt"))
model.eval()

# 生成样本输入
tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)
dummy_input = tokenizer("This is a test", return_tensors="pt")

torch.onnx.export(
    model,
    (dummy_input["input_ids"], dummy_input["attention_mask"]),
    "deberta.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size"}
    },
    opset_version=18  # 14-19 for ver2.3
)



import torch
from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification
import onnxruntime as ort
import numpy as np
import onnx


model_name = "microsoft/deberta-v3-base"  # 根据显存可选 base/small
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



# inference with onnx
ort_session = ort.InferenceSession("deberta.onnx")

for txt in [
    "交警队出盖章文件了！hw智驾连环车祸+行车记录仪内存卡被偷最新动态！",
    "坦克500 Hi4-Z、Hi4-T到底啥区别？应该怎么选？一条视频就明白了！#新车抢先看",
    "特斯拉皮卡，辅助泊车撞墙了！",
    "阿达西平时开什么车？那当然是内燃机了，奥迪c51.8t手动挡#马牌xc7#马牌轮胎#奥迪c5",
    "67万全款买奔驰 半年还没提到车（一）按合同延期1天赔1万 已超150天，车主：车商反要求我们违约赔偿100万#奔驰",
    "多家企业捐款驰援灾区#西藏日喀则地震"]:
    
    M = {0:'负向', 1: '中性',  2:'正向' }
    inputs = tokenizer(txt, return_tensors="np")
    outputs = ort_session.run(
        None,
        {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}
    )
    exp_values = np.exp(outputs[0] - np.max(outputs[0], axis=1, keepdims=True))
    softmax_array = exp_values / np.sum(exp_values, axis=1, keepdims=True)
    idx = np.argmax(softmax_array)
    print(idx, M[idx], outputs[0], softmax_array)


# Load the ONNX model and Analyse.
model = onnx.load("deberta.onnx")
for input in model.graph.input:
    print(f"Input: {input.name}, Shape: {input.type.tensor_type.shape}")
for output in model.graph.output:
    print(f"Output: {output.name}, Shape: {output.type.tensor_type.shape}")

save_path = "./tokenizer-debert"  # 替换为目标路径
tokenizer.save_pretrained(save_path)
from transformers import AutoTokenizer
import numpy as np
from rknn.api import RKNN


# --------------------------------------------------
# Simulator-inference.
# --------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained("./cn_roberta_model_v1/")
ONNX_MODEL = './cn_roberta_v1.onnx'
# 加载onnx
rknn = RKNN(verbose=True)
rknn.config(
     target_platform='rk3588',   # 根据芯片型号修改
     optimization_level=3,       # 最高优化级别
     quantized_dtype='w8a8',  # 量化类型
     quantized_algorithm='normal', 
     quantized_method='channel',
 )

rknn.load_onnx(
     model=ONNX_MODEL,
     inputs=['input_ids', 'attention_mask'],
     input_size_list=[[1, 128], [1, 128]],  # 固定输入尺寸
     outputs=['logits']
 )

rknn.build(
     do_quantization=False, 
     rknn_batch_size=1
 )

# Inference with Examples
rknn.init_runtime()

for txt in [
    "交警队出盖章文件了！hw智驾连环车祸+行车记录仪内存卡被偷最新动态！",
    "坦克500 Hi4-Z、Hi4-T到底啥区别？应该怎么选？一条视频就明白了！#新车抢先看",
]:
    M = {0:'负向', 1: '中性',  2:'正向' }
    inputs = tokenizer(
        txt,
        padding="max_length",    # 填充到最大长度
        truncation=True,         # 截断超长文本
        max_length=128,          # 与模型输入尺寸一致
        return_tensors="np"
    )
    input_ids = inputs["input_ids"].astype(np.int64)
    attention_mask = inputs["attention_mask"].astype(np.int64)
    outputs = rknn.inference(inputs=[input_ids, attention_mask])
    # softmax
    exp_values = np.exp(outputs[0] - np.max(outputs[0], axis=1, keepdims=True))
    softmax_array = exp_values / np.sum(exp_values, axis=1, keepdims=True)
    idx = np.argmax(softmax_array)
    print(idx, M[idx], outputs[0], softmax_array)


# onnx to openvino please refer to onnx2ov.sh
import numpy as np
from openvino.runtime import Core
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('./cn_roberta_model_v1/')
# --------------------------------------------------
# Fixed-shape-inference.
# --------------------------------------------------
ie = Core()
# Load model
model = ie.read_model(model='openvino_model/cn_roberta_v1.xml')
compiled_model = ie.compile_model(model=model, device_name='CPU')
# Receive outputs
input_ids_layer = compiled_model.input(0)
attention_mask_layer = compiled_model.input(1)
output_layer = compiled_model.output(0)

# Input data(EX)
txt = '小米SU7：你压着我头发了！'
inputs = tokenizer(
    txt,
    padding='max_length',  # 填充到最大长度
    truncation=True,  # 截断超长文本
    max_length=128,  # 与模型输入尺寸一致
    return_tensors='np',
)
# Inference
result = compiled_model([inputs['input_ids'], inputs['attention_mask']])[output_layer]
print('Result:', result)

# --------------------------------------------------
# Dynamic-shape-inference.
# --------------------------------------------------
ie = Core()
model = ie.read_model(model='openvino_model/cn_roberta_v1.xml')
compiled_model = ie.compile_model(model=model, device_name='CPU')
# Get output layers
input_ids_layer = compiled_model.input(0)
attention_mask_layer = compiled_model.input(1)
output_layer = compiled_model.output(0)
# Inference
M = {0: '负向', 1: '中性', 2: '正向'}
inputs = tokenizer(txt, return_tensors='np')
outputs = compiled_model([inputs['input_ids'], inputs['attention_mask']])[output_layer]
idx = np.argmax(outputs)
print(idx, M[idx])
